from __future__ import annotations

import json
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
    obstacle_hint = str(data.get("obstacle_hint", "unknown") or "unknown").strip()
    lateral = str(data.get("lateral_preference", "auto") or "auto").strip().lower()
    vertical = str(data.get("vertical_policy", "auto") or "auto").strip().lower()
    return {
        "environment_id": environment_id or "default_unreal_scene",
        "obstacle_hint": obstacle_hint or "unknown",
        "recommended_method": "pointcloud_direction_rule",
        "flyover_z_cm": max(0.0, as_float(data.get("flyover_z_cm"))),
        "lateral_preference": lateral if lateral in {"auto", "left", "right", "none"} else "auto",
        "vertical_policy": vertical or "auto",
        "strategy_reason": str(data.get("strategy_reason", data.get("reason", "")) or ""),
        "raw_decision": data,
    }


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
