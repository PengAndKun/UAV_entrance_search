from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from obstacle_avoidance.collect_route_episodes import FORWARD_DANGER_CM, action_payload, as_float


LLM_DIRECT_ACTIONS = {
    "forward",
    "slow_forward",
    "backoff",
    "backward",
    "left",
    "right",
    "side_step_left",
    "side_step_right",
    "up",
    "down",
    "yaw_left",
    "yaw_right",
    "hold",
}
LLM_DIRECT_METHOD_ID = "llm_direct_control_v1"
LLM_STRATEGY_METHOD_ID = "llm_strategy_pointcloud_rule_v1"

DIRECT_DECISION_SCHEMA: Dict[str, Any] = {
    "action_name": "hold",
    "forward_cm": 0.0,
    "right_cm": 0.0,
    "up_cm": 0.0,
    "yaw_delta_deg": 0.0,
    "reason": "Short explanation.",
    "confidence": 0.0,
}

STRATEGY_DECISION_SCHEMA: Dict[str, Any] = {
    "environment_id": "default_unreal_scene",
    "obstacle_hint": "unknown",
    "recommended_method": "pointcloud_direction_rule",
    "flyover_z_cm": 0.0,
    "lateral_preference": "auto",
    "vertical_policy": "auto",
    "strategy_reason": "Short explanation.",
}

CANONICAL_OBSTACLE_HINTS = {
    "unknown",
    "tree_trunk_or_pole",
    "tree_canopy_or_cluster",
    "tree",
    "pole",
    "fence_or_rail",
    "building",
    "mixed",
}

POINTCLOUD_HEIGHT_STRATEGY_KEYS = (
    "pointcloud_height_estimate",
    "pointcloud_recommended_flyover_z_cm",
    "pointcloud_recommended_vertical_offset_cm",
    "pointcloud_obstacle_height_cm",
    "pointcloud_obstacle_top_z_cm",
    "pointcloud_height_source",
    "pointcloud_flyover_recommended",
    "pointcloud_clearance_z_cm",
)


def clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    number = as_float(value, default)
    return max(low, min(high, number))


def normalize_direct_decision(raw: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    data = raw if isinstance(raw, dict) else {}
    action = str(data.get("action_name", data.get("action", "hold")) or "hold").strip().lower()
    action = action.replace(" ", "_")
    if action == "backward":
        action = "backoff"
    if action not in LLM_DIRECT_ACTIONS:
        action = "hold"

    payload = action_payload(action)
    payload["forward_cm"] = clamp(data.get("forward_cm", payload.get("forward_cm", 0.0)), -80.0, 80.0)
    payload["right_cm"] = clamp(data.get("right_cm", payload.get("right_cm", 0.0)), -60.0, 60.0)
    payload["up_cm"] = clamp(data.get("up_cm", payload.get("up_cm", 0.0)), -60.0, 60.0)
    payload["yaw_delta_deg"] = clamp(data.get("yaw_delta_deg", payload.get("yaw_delta_deg", 0.0)), -35.0, 35.0)
    payload["action_name"] = action

    if action in {"left", "side_step_left"} and abs(payload["right_cm"]) < 1.0:
        payload["right_cm"] = -20.0
    elif action in {"right", "side_step_right"} and abs(payload["right_cm"]) < 1.0:
        payload["right_cm"] = 20.0
    elif action == "forward" and payload["forward_cm"] < 1.0:
        payload["forward_cm"] = 20.0
    elif action == "slow_forward" and payload["forward_cm"] < 1.0:
        payload["forward_cm"] = 10.0
    elif action == "backoff" and payload["forward_cm"] > -1.0:
        payload["forward_cm"] = -20.0
    elif action == "up" and payload["up_cm"] < 1.0:
        payload["up_cm"] = 20.0
    elif action == "down" and payload["up_cm"] > -1.0:
        payload["up_cm"] = -20.0
    elif action == "yaw_left" and abs(payload["yaw_delta_deg"]) < 1.0:
        payload["yaw_delta_deg"] = -20.0
    elif action == "yaw_right" and abs(payload["yaw_delta_deg"]) < 1.0:
        payload["yaw_delta_deg"] = 20.0
    elif action == "hold":
        payload = action_payload("hold")

    meta = {
        "action_name": action,
        "reason": str(data.get("reason", data.get("selected_action_reason", "")) or ""),
        "confidence": clamp(data.get("confidence", 0.0), 0.0, 1.0),
        "raw_decision": data,
    }
    return payload, meta


def shield_direct_payload(
    payload: Dict[str, Any],
    pointcloud_summary: Dict[str, Any],
    current_pose: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    summary = pointcloud_summary if isinstance(pointcloud_summary, dict) else {}
    pose = current_pose if isinstance(current_pose, dict) else {}
    action = str(payload.get("action_name", "") or "").lower()
    forward_cm = as_float(payload.get("forward_cm"))
    up_cm = as_float(payload.get("up_cm"))
    front_min = as_float(summary.get("front_min_depth_cm"))
    forward_blocked = (
        forward_cm > 1.0
        and (
            not bool(summary.get("forward_swept_clear", True))
            or (front_min > 0.0 and front_min < FORWARD_DANGER_CM)
        )
    )
    if forward_blocked:
        if bool(summary.get("backoff_swept_clear", True)):
            shielded = action_payload("backoff")
        else:
            shielded = action_payload("hold")
        return shielded, {
            "state": "BLOCKED_FORWARD",
            "original_action": action,
            "reason": f"LLM forward blocked by safety shield: front={front_min:.1f}cm forward_swept_clear={bool(summary.get('forward_swept_clear', True))}",
        }

    pose_z = as_float(pose.get("z"))
    down_blocked = up_cm < -1.0 and (not bool(summary.get("down_swept_clear", False)) or pose_z <= 100.0)
    if down_blocked:
        return action_payload("hold"), {
            "state": "BLOCKED_DOWN",
            "original_action": action,
            "reason": f"LLM down blocked by safety shield: down_swept_clear={bool(summary.get('down_swept_clear', False))} z={pose_z:.1f}cm",
        }

    return dict(payload), {"state": "PASSED", "original_action": action, "reason": "LLM payload passed safety shield"}


def normalize_strategy_decision(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    environment_id = str(data.get("environment_id", "default_unreal_scene") or "default_unreal_scene").strip()
    raw_obstacle_hint = str(data.get("raw_obstacle_hint", data.get("obstacle_hint", "unknown")) or "unknown").strip()
    obstacle_hint = canonical_obstacle_hint(raw_obstacle_hint, environment_id=environment_id)
    lateral = str(data.get("lateral_preference", "auto") or "auto").strip().lower()
    vertical = str(data.get("vertical_policy", "auto") or "auto").strip().lower()
    strategy_reason = str(data.get("strategy_reason", data.get("reason", "")) or "")
    existing_note = str(data.get("llm_obstacle_note", "") or "").strip()
    note_parts = [existing_note] if existing_note else []
    if raw_obstacle_hint and raw_obstacle_hint.lower() not in {"unknown", obstacle_hint.lower()}:
        note_parts.append(f"LLM hint: {raw_obstacle_hint}")
    if strategy_reason:
        note_parts.append(f"Reason: {strategy_reason}")
    result = {
        "environment_id": environment_id or "default_unreal_scene",
        "obstacle_hint": obstacle_hint or "unknown",
        "recommended_method": "pointcloud_direction_rule",
        "flyover_z_cm": max(0.0, as_float(data.get("flyover_z_cm"))),
        "lateral_preference": lateral if lateral in {"auto", "left", "right", "none"} else "auto",
        "vertical_policy": vertical or "auto",
        "strategy_reason": strategy_reason,
        "raw_obstacle_hint": raw_obstacle_hint,
        "llm_obstacle_note": "; ".join(note_parts),
        "raw_decision": data,
    }
    for key in POINTCLOUD_HEIGHT_STRATEGY_KEYS:
        if key in data:
            result[key] = data[key]
    return result


def canonical_obstacle_hint(value: Any, *, environment_id: str = "") -> str:
    raw = str(value or "").strip()
    env = str(environment_id or "").strip().lower()
    env_key = re.sub(r"[^a-z0-9]+", "_", env).strip("_")
    text = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if text in CANONICAL_OBSTACLE_HINTS:
        return text
    if text in {"fence", "rail", "railing", "fence_or_rails", "fence_rail"}:
        return "fence_or_rail"
    if text in {"tree_or_pole", "trunk", "tree_trunk", "vertical_pole"}:
        return "tree_trunk_or_pole"
    if text in {"canopy", "tree_canopy", "cluster", "tree_cluster"}:
        return "tree_canopy_or_cluster"
    phrase = f"{text} {env_key}"
    if any(token in phrase for token in ("fence", "rail", "railing")):
        return "fence_or_rail"
    if any(token in phrase for token in ("building", "roof", "house", "wall")):
        return "building"
    if any(token in phrase for token in ("canopy", "cluster", "branch")):
        return "tree_canopy_or_cluster"
    if any(token in phrase for token in ("trunk", "pole")):
        return "tree_trunk_or_pole"
    if "mixed" in phrase:
        return "mixed"
    if "tree" in phrase:
        return "tree"
    if env_key in CANONICAL_OBSTACLE_HINTS:
        return env_key
    return "unknown"


def apply_strategy_to_episode_metadata(episode: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(episode) if isinstance(episode, dict) else {}
    normalized = normalize_strategy_decision(strategy)
    result["environment_id"] = normalized["environment_id"]
    result["obstacle_hint"] = normalized["obstacle_hint"]
    result["method"] = LLM_STRATEGY_METHOD_ID
    note = str(result.get("operator_note", "") or "").strip()
    llm_note = str(normalized.get("llm_obstacle_note", "") or "").strip()
    if llm_note and llm_note not in note:
        result["operator_note"] = f"{note} | {llm_note}" if note else llm_note
    result["llm_strategy"] = {
        key: value
        for key, value in normalized.items()
        if key not in {"raw_decision", "llm_raw"}
    }
    return result


def refine_strategy_with_pointcloud_context(strategy: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    result = normalize_strategy_decision(strategy)
    source_strategy = strategy if isinstance(strategy, dict) else {}
    for key in POINTCLOUD_HEIGHT_STRATEGY_KEYS:
        if key in source_strategy:
            result[key] = source_strategy[key]

    summary = event.get("pointcloud_summary") if isinstance(event, dict) and isinstance(event.get("pointcloud_summary"), dict) else {}
    height_estimate = result.get("pointcloud_height_estimate") if isinstance(result.get("pointcloud_height_estimate"), dict) else {}
    geometry = str(summary.get("obstacle_geometry", "") or "").lower()
    width_cm = as_float(summary.get("obstacle_width_cm"))
    summary_height_cm = as_float(summary.get("obstacle_height_cm"))
    estimated_height_cm = as_float(height_estimate.get("obstacle_height_cm"))
    top_z_cm = as_float(height_estimate.get("obstacle_top_z_cm"))
    current_z_cm = as_float(height_estimate.get("current_z_cm"))
    top_offset_cm = top_z_cm - current_z_cm if top_z_cm > 0.0 and current_z_cm > 0.0 else 0.0
    raw_hint = str(result.get("raw_obstacle_hint", "") or "").lower()
    reason = str(result.get("strategy_reason", "") or "").lower()
    env = str(result.get("environment_id", "") or "").lower()
    hint = str(result.get("obstacle_hint", "") or "").lower()

    low_obstacle_hint = any(token in raw_hint for token in ("low_obstacle", "low barrier", "low_barrier", "fence", "rail"))
    building_candidate = env in {"building_or_roof", "building"} or hint == "building"
    low_wide_geometry = (
        geometry == "low_obstacle"
        and width_cm >= 220.0
        and (
            0.0 < estimated_height_cm <= 480.0
            or 0.0 < summary_height_cm <= 520.0
            or 0.0 < top_offset_cm <= 420.0
        )
        and bool(summary.get("up_swept_clear", True))
    )
    strong_building_phrase = any(token in f"{raw_hint} {reason}" for token in ("facade", "entrance", "door", "house wall", "tall building"))
    if building_candidate and low_wide_geometry and (low_obstacle_hint or not strong_building_phrase):
        previous_env = result.get("environment_id", "")
        previous_hint = result.get("obstacle_hint", "")
        result["environment_id"] = "fence_or_rail"
        result["obstacle_hint"] = "fence_or_rail"
        result["semantic_refinement_source"] = "pointcloud_low_wide_flyover"
        result["semantic_refinement_reason"] = (
            f"corrected {previous_env}/{previous_hint} to fence_or_rail because pointcloud shows "
            f"low wide obstacle: geometry={geometry}, width={width_cm:.1f}cm, "
            f"height={estimated_height_cm or summary_height_cm:.1f}cm, up_swept_clear={bool(summary.get('up_swept_clear', True))}"
        )
        note = str(result.get("llm_obstacle_note", "") or "").strip()
        refinement_note = f"Semantic refinement: {result['semantic_refinement_reason']}"
        result["llm_obstacle_note"] = f"{note}; {refinement_note}" if note else refinement_note
    return result


def strategy_from_episode_metadata(episode: Dict[str, Any]) -> Dict[str, Any]:
    data = episode if isinstance(episode, dict) else {}
    cached = data.get("llm_strategy")
    if isinstance(cached, dict) and cached:
        raw_strategy = dict(cached)
    else:
        raw_strategy = {
            "environment_id": data.get("environment_id", "default_unreal_scene"),
            "obstacle_hint": data.get("obstacle_hint", "unknown"),
            "recommended_method": "pointcloud_direction_rule",
            "flyover_z_cm": data.get("flyover_z_cm", 0.0),
            "lateral_preference": data.get("lateral_preference", "auto"),
            "vertical_policy": data.get("vertical_policy", "auto"),
            "strategy_reason": "Using episode Environment/Hint without an execution-time LLM call.",
        }
    normalized = normalize_strategy_decision(raw_strategy)
    normalized["strategy_source"] = "episode_metadata"
    normalized["llm_call_required"] = False
    normalized["llm_error"] = ""
    return normalized


def build_direct_prompts(event: Dict[str, Any], episode: Dict[str, Any], last_action: Dict[str, Any]) -> Tuple[str, str]:
    context = {
        "task": "Control a UAV from start_pose to goal_pose while avoiding obstacles.",
        "allowed_actions": sorted(LLM_DIRECT_ACTIONS),
        "safety": [
            "Do not move forward if front_min_depth_cm is below the danger distance.",
            "Do not descend unless down_swept_clear is true and the UAV is safely above ground.",
            "Prefer small stable actions over oscillation.",
        ],
        "episode": episode,
        "frame": {
            "current_pose": event.get("current_pose", {}),
            "goal_pose": event.get("goal_pose", {}),
            "relative_target": event.get("relative_target", {}),
            "distance_to_goal_cm": event.get("distance_to_goal_cm"),
            "pointcloud_summary": event.get("pointcloud_summary", {}),
            "last_action": last_action,
        },
        "output_schema": DIRECT_DECISION_SCHEMA,
    }
    system_prompt = (
        "You are a cautious UAV obstacle-avoidance controller. "
        "Return only one JSON object matching the requested schema."
    )
    user_prompt = json.dumps(context, indent=2, ensure_ascii=False)
    return system_prompt, user_prompt


def build_strategy_prompts(event: Dict[str, Any], episode: Dict[str, Any]) -> Tuple[str, str]:
    context = {
        "task": "Classify the obstacle environment and choose a high-level strategy for the rule controller.",
        "available_environments": [
            "default_unreal_scene",
            "tree_trunk_or_pole",
            "tree_canopy_or_cluster",
            "tree_or_pole",
            "fence_or_rail",
            "building_or_roof",
            "mixed_obstacles",
        ],
        "available_obstacle_hints": sorted(CANONICAL_OBSTACLE_HINTS),
        "classification_rules": [
            "Use fence_or_rail for fences, rails, railings, low/wide horizontal barriers, and boundary walls that are best handled by a moderate flyover.",
            "Use building_or_roof only for a real building, house facade, roof, entrance wall, or tall large structure that needs high vantage/overfly behavior.",
            "Do not classify generic low_obstacle or low horizontal barrier as building_or_roof unless RGB clearly shows a building facade/roof/house.",
            "For narrow trunks, poles, branches, or thin vertical objects, use tree_trunk_or_pole or tree_canopy_or_cluster and prefer lateral avoidance.",
            "obstacle_hint must be one canonical value from available_obstacle_hints; put free-text visual details into strategy_reason.",
        ],
        "fixed_recommended_method": "pointcloud_direction_rule",
        "episode": episode,
        "frame": {
            "current_pose": event.get("current_pose", {}),
            "goal_pose": event.get("goal_pose", {}),
            "relative_target": event.get("relative_target", {}),
            "distance_to_goal_cm": event.get("distance_to_goal_cm"),
            "pointcloud_summary": event.get("pointcloud_summary", {}),
        },
        "output_schema": STRATEGY_DECISION_SCHEMA,
    }
    system_prompt = (
        "You are a UAV obstacle strategy classifier. "
        "Return only one JSON object. recommended_method must be pointcloud_direction_rule."
    )
    user_prompt = json.dumps(context, indent=2, ensure_ascii=False)
    return system_prompt, user_prompt
