from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import run_drone_flight as flight

from .collect import ACTION_PAYLOADS, append_jsonl, relative_target, sanitize, write_json
from .geometry_v0 import best_candidate_action, score_candidate_actions, selected_action_reason, summarize_geometry_v0


DEFAULT_ROUTE_EPISODES: List[Dict[str, Any]] = [
    {"episode_id": "E01", "start_pose": [325.5, 526.0, 211.2, 0.2], "goal_pose": [734.3, 552.0, 211.2, 0.2]},
    {"episode_id": "E02", "start_pose": [1298.8, 513.8, 258.2, 0.3], "goal_pose": [2048.9, 443.8, 259.7, 0.3]},
    {"episode_id": "E03", "start_pose": [2475.1, 1129.3, 175.3, 74.2], "goal_pose": [2883.4, 1852.3, 162.0, 65.4]},
    {"episode_id": "E04", "start_pose": [3546.2, 2361.3, 162.0, -55.8], "goal_pose": [4425.2, 1100.3, 200.0, -55.6]},
    {"episode_id": "E05", "start_pose": [4870.3, -655.2, 257.9, -105.3], "goal_pose": [3820.9, -2436.0, 267.9, 123.3]},
    {"episode_id": "E06", "start_pose": [1875.2, -1390.4, 223.5, -89.8], "goal_pose": [1877.1, 1911.1, 223.5, -89.8]},
    {"episode_id": "E07", "start_pose": [1916.6, -2605.6, 223.5, 178.0], "goal_pose": [1429.5, -2646.4, 223.5, 178.0]},
    {"episode_id": "E08", "start_pose": [-549.9, -2703.1, 262.5, 93.0], "goal_pose": [-589.8, -1948.0, 262.4, 93.0]},
    {"episode_id": "E09", "start_pose": [-497.1, 1640.1, 195.2, 85.6], "goal_pose": [-462.9, 2084.6, 201.2, 85.6]},
    {"episode_id": "E10", "start_pose": [1960.5, 1975.9, 153.1, -87.4], "goal_pose": [1983.1, 1474.2, 153.1, -87.4]},
]

FORWARD_DANGER_CM = 250.0
GOAL_YAW_TOL_DEG = 12.0
SLOW_FORWARD_CAUTION_CM = 450.0
Z_ALIGN_TOL_CM = 45.0
DEFAULT_ROUTE_STEP_CM = 60.0
DEFAULT_ROUTE_SIDE_CORRECTION_CM = 35.0
DEFAULT_ROUTE_VERTICAL_STEP_CM = 35.0
FENCE_OVERFLY_MIN_VERTICAL_OFFSET_CM = 360.0
FENCE_OVERFLY_MAX_VERTICAL_OFFSET_CM = 650.0
FENCE_OVERFLY_UP_DEPTH_SCALE = 0.65
FENCE_DESCENT_MIN_RAW_PROGRESS = 0.9
FENCE_DESCENT_MAX_DISTANCE_CM = 260.0
FENCE_FINAL_XY_TOL_CM = 50.0
FENCE_FINAL_XY_ADJUST_ZONE_CM = 260.0
FENCE_CONTROLLED_DESCENT_MIN_P05_CM = 220.0
FENCE_CONTROLLED_DESCENT_MAX_CLOSE_FRACTION = 0.16
TREE_POLE_NARROW_WIDTH_CM = 340.0
TREE_POLE_WIDE_WIDTH_CM = 520.0
TREE_POLE_MIN_SIDE_CLEAR_CM = 135.0
TREE_POLE_LATERAL_STEP_CM = 18.0
TREE_POLE_FORWARD_PROBE_CLEAR_CM = 300.0
TREE_POLE_UP_CLEAR_CM = 320.0
REPEATED_HOLD_STOP_COUNT = 3
BUILDING_VANTAGE_MIN_Z_CM = 1200.0
BUILDING_VANTAGE_MIN_VERTICAL_OFFSET_CM = 900.0
BUILDING_VANTAGE_MAX_Z_CM = 1200.0
BUILDING_VANTAGE_FRONT_MAX_CM = 650.0
BUILDING_FORWARD_CLEAR_CM = 700.0
BUILDING_CRUISE_MIN_FRONT_CM = 600.0
BUILDING_MAX_VANTAGE_CREEP_FRONT_CM = 500.0
BUILDING_CLEAR_MIN_RAW_PROGRESS = 0.65
BUILDING_CLEAR_MIN_PROGRESS_DELTA = 0.5
BUILDING_DESCENT_MIN_RAW_PROGRESS = 0.9
BUILDING_DESCENT_MAX_DISTANCE_CM = 420.0
BUILDING_DESCENT_MAX_XY_DISTANCE_CM = 260.0
GOAL_ALIGNMENT_DISTANCE_MARGIN_CM = 130.0
GOAL_ALIGNMENT_PROGRESS = 0.9
POINTCLOUD_FORWARD_PROBE_MIN_FRONT_CM = 90.0
POINTCLOUD_FORWARD_PROBE_CREEP_FRONT_CM = 110.0
POINTCLOUD_FORWARD_PROBE_MIN_OFFSET_CM = 55.0
POINTCLOUD_FORWARD_PROBE_STEP_CM = 25.0
POINTCLOUD_FORWARD_CREEP_STEP_CM = 12.0
POINTCLOUD_FINAL_ALIGNMENT_MARGIN_CM = 60.0


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "hit", "collision", "collided", "impact"}
    return False


def normalize_angle_deg(angle: float) -> float:
    value = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 else value


def pose_dict(raw: Iterable[Any]) -> Dict[str, float]:
    values = list(raw)
    if len(values) < 4:
        raise ValueError(f"Pose must have x,y,z,yaw values, got {values!r}")
    return {
        "x": as_float(values[0]),
        "y": as_float(values[1]),
        "z": as_float(values[2]),
        "yaw": normalize_angle_deg(as_float(values[3])),
    }


def pose_from_state(state: Dict[str, Any]) -> Dict[str, float]:
    pose = state.get("pose") if isinstance(state, dict) else {}
    if not isinstance(pose, dict):
        pose = {}
    return {
        "x": as_float(pose.get("x")),
        "y": as_float(pose.get("y")),
        "z": as_float(pose.get("z")),
        "yaw": normalize_angle_deg(as_float(pose.get("yaw", pose.get("task_yaw", pose.get("yaw_deg", 0.0))))),
    }


def distance_3d_cm(pose: Dict[str, Any], goal: Dict[str, Any]) -> float:
    dx = as_float(goal.get("x")) - as_float(pose.get("x"))
    dy = as_float(goal.get("y")) - as_float(pose.get("y"))
    dz = as_float(goal.get("z")) - as_float(pose.get("z"))
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def distance_xy_cm(pose: Dict[str, Any], goal: Dict[str, Any]) -> float:
    dx = as_float(goal.get("x")) - as_float(pose.get("x"))
    dy = as_float(goal.get("y")) - as_float(pose.get("y"))
    return float(math.sqrt(dx * dx + dy * dy))


def segment_length_cm(start: Dict[str, Any], goal: Dict[str, Any]) -> float:
    return distance_3d_cm(start, goal)


def route_progress_and_deviation(pose: Dict[str, Any], start: Dict[str, Any], goal: Dict[str, Any]) -> Tuple[float, float]:
    sx, sy, sz = as_float(start.get("x")), as_float(start.get("y")), as_float(start.get("z"))
    gx, gy, gz = as_float(goal.get("x")), as_float(goal.get("y")), as_float(goal.get("z"))
    px, py, pz = as_float(pose.get("x")), as_float(pose.get("y")), as_float(pose.get("z"))
    vx, vy, vz = gx - sx, gy - sy, gz - sz
    wx, wy, wz = px - sx, py - sy, pz - sz
    denom = vx * vx + vy * vy + vz * vz
    if denom <= 1e-9:
        return 1.0, distance_3d_cm(pose, goal)
    t = (wx * vx + wy * vy + wz * vz) / denom
    progress = max(0.0, min(1.0, t))
    cx, cy, cz = sx + progress * vx, sy + progress * vy, sz + progress * vz
    deviation = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
    return float(progress), float(deviation)


def route_progress_raw_and_deviation(pose: Dict[str, Any], start: Dict[str, Any], goal: Dict[str, Any]) -> Tuple[float, float]:
    sx, sy, sz = as_float(start.get("x")), as_float(start.get("y")), as_float(start.get("z"))
    gx, gy, gz = as_float(goal.get("x")), as_float(goal.get("y")), as_float(goal.get("z"))
    px, py, pz = as_float(pose.get("x")), as_float(pose.get("y")), as_float(pose.get("z"))
    vx, vy, vz = gx - sx, gy - sy, gz - sz
    wx, wy, wz = px - sx, py - sy, pz - sz
    denom = vx * vx + vy * vy + vz * vz
    if denom <= 1e-9:
        return 1.0, distance_3d_cm(pose, goal)
    progress = (wx * vx + wy * vy + wz * vz) / denom
    cx, cy, cz = sx + progress * vx, sy + progress * vy, sz + progress * vz
    deviation = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
    return float(progress), float(deviation)


def route_completion_state(
    pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    reach_tol_cm: float,
) -> Tuple[bool, str, float, float]:
    distance = distance_3d_cm(pose, goal)
    raw_progress, cross_track_cm = route_progress_raw_and_deviation(pose, start, goal)
    if distance <= float(reach_tol_cm):
        return True, "within_reach_tolerance", raw_progress, cross_track_cm
    goal_plane_cross_track_tol_cm = max(120.0, float(reach_tol_cm))
    if raw_progress >= 1.0 and cross_track_cm <= goal_plane_cross_track_tol_cm:
        return True, "passed_goal_plane", raw_progress, cross_track_cm
    return False, "running", raw_progress, cross_track_cm


def collision_signal_from_mapping(payload: Any) -> Tuple[bool, bool, str, Any]:
    if not isinstance(payload, dict):
        return False, False, "", None
    if payload.get("available") is False:
        return False, False, "", None
    for key in (
        "collision_state",
        "impact_state",
        "hit_state",
        "collision",
        "impact",
        "hit",
        "Collision",
        "Hit",
    ):
        if key in payload:
            raw = payload.get(key)
            if isinstance(raw, (int, float)):
                return True, bool(float(raw) > 0.0), key, raw
            return True, as_bool(raw), key, raw
    return False, False, "", None


def commanded_translation_cm(payload: Dict[str, Any]) -> float:
    return float(
        math.sqrt(
            as_float(payload.get("forward_cm")) ** 2
            + as_float(payload.get("right_cm")) ** 2
            + as_float(payload.get("up_cm")) ** 2
        )
    )


def annotate_collision_state(
    event: Dict[str, Any],
    *,
    pre_pose: Dict[str, Any],
    post_pose: Dict[str, Any],
    payload: Dict[str, Any],
    explicit_collision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    explicit_available, explicit_hit, explicit_key, explicit_raw = collision_signal_from_mapping(explicit_collision)
    command_cm = commanded_translation_cm(payload)
    forward_command_cm = as_float(payload.get("forward_cm"))
    actual_cm = distance_3d_cm(pre_pose, post_pose)
    front_min = as_float(summary.get("front_min_depth_cm"))
    forward_blocked = not bool(summary.get("forward_swept_clear", True))
    likely_stalled = forward_command_cm >= 10.0 and actual_cm <= max(3.0, min(12.0, abs(forward_command_cm) * 0.2))
    near_obstacle = (front_min > 0.0 and front_min < FORWARD_DANGER_CM) or forward_blocked
    heuristic_hit = bool(not explicit_available and likely_stalled and near_obstacle)

    collision = bool(explicit_hit or heuristic_hit)
    if explicit_available:
        source = "unreal_get_hit"
        confidence = 1.0
        reason = f"explicit {explicit_key}={explicit_raw}"
    elif heuristic_hit:
        source = "physics_motion_heuristic"
        confidence = 0.62
        reason = (
            f"commanded {command_cm:.1f}cm but actual displacement {actual_cm:.1f}cm "
            f"near obstacle front={front_min:.1f}cm"
        )
    else:
        source = "none"
        confidence = 0.0
        reason = "no collision signal"

    event["collision_state"] = collision
    event["impact_state"] = collision
    event["collision_source"] = source
    event["impact_source"] = source
    event["impact_confidence"] = confidence
    event["avoidance_failed"] = collision
    event["impact_detail"] = {
        "reason": reason,
        "explicit_available": explicit_available,
        "explicit_key": explicit_key,
        "explicit_raw": explicit_raw,
        "commanded_translation_cm": round(command_cm, 3),
        "commanded_forward_cm": round(forward_command_cm, 3),
        "actual_translation_cm": round(actual_cm, 3),
        "front_min_depth_cm": round(front_min, 3),
        "forward_swept_clear": bool(summary.get("forward_swept_clear", True)),
    }
    return event


def path_yaw_deg(start: Dict[str, Any], goal: Dict[str, Any]) -> float:
    dx = as_float(goal.get("x")) - as_float(start.get("x"))
    dy = as_float(goal.get("y")) - as_float(start.get("y"))
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return normalize_angle_deg(as_float(goal.get("yaw", start.get("yaw", 0.0))))
    return normalize_angle_deg(math.degrees(math.atan2(dy, dx)))


def point_on_route(start: Dict[str, Any], goal: Dict[str, Any], progress: float) -> Dict[str, float]:
    t = max(0.0, min(1.0, float(progress)))
    return {
        "x": as_float(start.get("x")) + (as_float(goal.get("x")) - as_float(start.get("x"))) * t,
        "y": as_float(start.get("y")) + (as_float(goal.get("y")) - as_float(start.get("y"))) * t,
        "z": as_float(start.get("z")) + (as_float(goal.get("z")) - as_float(start.get("z"))) * t,
        "yaw": path_yaw_deg(start, goal),
    }


def body_delta_to_payload(
    current: Dict[str, Any],
    target: Dict[str, Any],
    *,
    route_step_cm: float,
    side_correction_cm: float,
    vertical_step_cm: float,
) -> Dict[str, float]:
    yaw = math.radians(as_float(current.get("yaw", current.get("task_yaw", 0.0))))
    dx = as_float(target.get("x")) - as_float(current.get("x"))
    dy = as_float(target.get("y")) - as_float(current.get("y"))
    dz = as_float(target.get("z")) - as_float(current.get("z"))
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    up = dz
    forward = max(0.0, min(float(route_step_cm), forward))
    right = max(-float(side_correction_cm), min(float(side_correction_cm), right))
    up = max(-float(vertical_step_cm), min(float(vertical_step_cm), up))
    yaw_delta = normalize_angle_deg(path_yaw_deg(current, target) - as_float(current.get("yaw", current.get("task_yaw", 0.0))))
    yaw_delta = max(-20.0, min(20.0, yaw_delta))
    return {"forward_cm": forward, "right_cm": right, "up_cm": up, "yaw_delta_deg": yaw_delta}


def body_delta_to_goal(current: Dict[str, Any], goal: Dict[str, Any]) -> Dict[str, float]:
    yaw = math.radians(as_float(current.get("yaw", current.get("task_yaw", 0.0))))
    dx = as_float(goal.get("x")) - as_float(current.get("x"))
    dy = as_float(goal.get("y")) - as_float(current.get("y"))
    dz = as_float(goal.get("z")) - as_float(current.get("z"))
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    target_yaw = math.degrees(math.atan2(dy, dx)) if abs(dx) > 1e-9 or abs(dy) > 1e-9 else as_float(current.get("yaw", 0.0))
    return {
        "forward_cm": float(forward),
        "right_cm": float(right),
        "up_cm": float(dz),
        "yaw_delta_deg": normalize_angle_deg(target_yaw - as_float(current.get("yaw", current.get("task_yaw", 0.0)))),
    }


def label_payload(payload: Dict[str, float]) -> Dict[str, float]:
    forward = as_float(payload.get("forward_cm"))
    right = as_float(payload.get("right_cm"))
    up = as_float(payload.get("up_cm"))
    yaw_delta = as_float(payload.get("yaw_delta_deg"))
    if abs(yaw_delta) >= 12.0 and forward < 5.0 and abs(right) < 5.0:
        name = "yaw_right" if yaw_delta > 0 else "yaw_left"
    elif up >= 10.0:
        name = "up"
    elif up <= -10.0:
        name = "down"
    elif right >= 10.0 and forward < 10.0:
        name = "side_step_right"
    elif right <= -10.0 and forward < 10.0:
        name = "side_step_left"
    elif forward >= 35.0:
        name = "forward"
    elif forward >= 1.0:
        name = "slow_forward"
    else:
        name = "hold"
    result = dict(payload)
    result["action_name"] = name
    return result


def should_stop_for_repeated_hold(events: List[Dict[str, Any]], *, threshold: int = REPEATED_HOLD_STOP_COUNT) -> bool:
    if threshold <= 0 or len(events) < threshold:
        return False
    recent = events[-threshold:]
    for event in recent:
        if str(event.get("selected_action", "")).lower() != "hold":
            return False
        if str(event.get("mission_phase", "")).upper() == "REACHED":
            return False
        if bool(event.get("goal_reached", False)):
            return False
    return True


def straight_line_payload(
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    route_step_cm: float,
    side_correction_cm: float,
    vertical_step_cm: float,
) -> Tuple[Dict[str, float], Dict[str, float], str]:
    progress, deviation = route_progress_and_deviation(current_pose, start, goal)
    length = max(1.0, segment_length_cm(start, goal))
    target_progress = min(1.0, progress + max(1.0, float(route_step_cm)) / length)
    target = point_on_route(start, goal, target_progress)
    payload = label_payload(
        body_delta_to_payload(
            current_pose,
            target,
            route_step_cm=route_step_cm,
            side_correction_cm=side_correction_cm,
            vertical_step_cm=vertical_step_cm,
        )
    )
    reason = (
        f"straight-line follow progress {progress:.3f}->{target_progress:.3f}, "
        f"path_deviation_cm={deviation:.1f}"
    )
    return payload, target, reason


def load_episodes(path: str) -> List[Dict[str, Any]]:
    if not path:
        return [dict(item) for item in DEFAULT_ROUTE_EPISODES]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("episodes"), list):
            rows = payload.get("episodes", [])
        elif isinstance(payload.get("projects"), list):
            projects = payload.get("projects", [])
            active_id = str(payload.get("active_project_id", "") or "")
            project = next((item for item in projects if str(item.get("project_id", "")) == active_id), None)
            if project is None and projects:
                project = projects[0]
            rows = project.get("episodes", []) if isinstance(project, dict) else []
        else:
            rows = []
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("episodes JSON must be a list or an object with an episodes list")
    episodes: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"episode #{index} must be an object")
        start = row.get("start_pose") or row.get("start")
        goal = row.get("goal_pose") or row.get("goal")
        if not isinstance(start, list) or not isinstance(goal, list):
            raise ValueError(f"episode #{index} must include start_pose and goal_pose lists")
        episode = {"episode_id": str(row.get("episode_id") or f"E{index:02d}"), "start_pose": start, "goal_pose": goal}
        for key in ("scenario_id", "environment_id", "method", "obstacle_hint", "operator_note", "enabled"):
            if key in row:
                episode[key] = row.get(key)
        episodes.append(episode)
    return episodes


def action_payload(action: str, *, yaw_delta_deg: Optional[float] = None) -> Dict[str, float]:
    payload = dict(ACTION_PAYLOADS.get(action, ACTION_PAYLOADS["hold"]))
    if yaw_delta_deg is not None:
        payload["yaw_delta_deg"] = float(yaw_delta_deg)
        payload["action_name"] = "yaw_left" if yaw_delta_deg < 0 else "yaw_right" if yaw_delta_deg > 0 else "hold"
    return payload


def last_side_action(last_action: Dict[str, Any]) -> str:
    name = str(last_action.get("action_name", "") if isinstance(last_action, dict) else "").lower()
    right_cm = as_float(last_action.get("right_cm")) if isinstance(last_action, dict) else 0.0
    if "left" in name or right_cm < -1.0:
        return "left"
    if "right" in name or right_cm > 1.0:
        return "right"
    return ""


def choose_recovery_action(scores: Dict[str, float], rel: Dict[str, Any], summary: Dict[str, Any], last_action: Dict[str, Any]) -> str:
    allowed = ("left", "right", "up", "backoff", "hold", "yaw_left", "yaw_right")
    filtered = {name: as_float(scores.get(name)) for name in allowed}
    bearing = as_float(rel.get("bearing_deg_body"))
    dz = as_float(rel.get("dz_cm"))
    front_min = as_float(summary.get("front_min_depth_cm"))
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    previous_side = last_side_action(last_action)
    if geometry == "vertical_wall" or front_min >= 95.0:
        if previous_side and bool(summary.get(f"{previous_side}_swept_clear", True)):
            return previous_side
        left = filtered.get("left", 0.0) + (0.12 if bearing < 0.0 else 0.0)
        right = filtered.get("right", 0.0) + (0.12 if bearing > 0.0 else 0.0)
        if left > 0.0 or right > 0.0:
            return "left" if left >= right else "right"
    if dz < -Z_ALIGN_TOL_CM:
        filtered["up"] = 0.0
    if bearing > GOAL_YAW_TOL_DEG:
        filtered["right"] += 0.16
        filtered["yaw_right"] += 0.08
    elif bearing < -GOAL_YAW_TOL_DEG:
        filtered["left"] += 0.16
        filtered["yaw_left"] += 0.08
    action, _score = best_candidate_action(filtered)
    if action in {"backoff", "hold", "yaw_left", "yaw_right"} and previous_side and bool(summary.get(f"{previous_side}_swept_clear", True)):
        return previous_side
    if action in {"backoff", "hold"} and front_min >= 70.0:
        left = filtered.get("left", 0.0)
        right = filtered.get("right", 0.0)
        if left > 0.0 or right > 0.0:
            return "left" if left >= right else "right"
    return action if action in filtered else "hold"


def is_soft_low_obstacle(summary: Dict[str, Any]) -> bool:
    front_min = as_float(summary.get("front_min_depth_cm"))
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    return geometry in {"low_obstacle", "thin_structure"} and front_min >= 120.0


def is_hard_blocked(summary: Dict[str, Any]) -> bool:
    front_min = as_float(summary.get("front_min_depth_cm"))
    if front_min <= 0.0:
        return False
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    if front_min < 95.0:
        return True
    if geometry == "vertical_wall" and front_min < FORWARD_DANGER_CM:
        return True
    if not bool(summary.get("forward_swept_clear", True)) and not is_soft_low_obstacle(summary) and front_min < FORWARD_DANGER_CM:
        return True
    return False


def effective_route_step_cm(
    *,
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    route_step_cm: float,
    front_min_cm: float,
    respect_front_caution: bool = True,
) -> float:
    step_cm = max(1.0, float(route_step_cm))
    goal_margin_cm = max(0.0, float(distance_to_goal_cm) - float(reach_tol_cm))
    if goal_margin_cm < step_cm:
        step_cm = max(5.0, goal_margin_cm)
    if respect_front_caution and front_min_cm > 0.0 and front_min_cm < SLOW_FORWARD_CAUTION_CM:
        step_cm = min(step_cm, 25.0)
    return step_cm


def distance_rule_recovery_action(summary: Dict[str, Any], rel: Dict[str, Any], last_action: Dict[str, Any]) -> str:
    def clear_depth(key: str) -> float:
        value = as_float(summary.get(key))
        return 999.0 if value <= 0.0 else value

    candidates: Dict[str, float] = {
        "hold": 95.0,
    }
    if bool(summary.get("left_swept_clear", True)):
        candidates["left"] = clear_depth("left_min_depth_cm")
    if bool(summary.get("right_swept_clear", True)):
        candidates["right"] = clear_depth("right_min_depth_cm")
    if bool(summary.get("up_swept_clear", True)):
        candidates["up"] = clear_depth("up_min_depth_cm")
    if bool(summary.get("backoff_swept_clear", True)):
        candidates["backoff"] = max(180.0, 380.0 - max(0.0, as_float(summary.get("front_min_depth_cm"))))

    bearing = as_float(rel.get("bearing_deg_body"))
    dz = as_float(rel.get("dz_cm"))
    if bearing < -GOAL_YAW_TOL_DEG and "left" in candidates:
        candidates["left"] += 60.0
    elif bearing > GOAL_YAW_TOL_DEG and "right" in candidates:
        candidates["right"] += 60.0
    if dz > Z_ALIGN_TOL_CM and "up" in candidates:
        candidates["up"] += 50.0

    previous_side = last_side_action(last_action)
    if previous_side in candidates:
        candidates[previous_side] += 35.0
    action, score = max(candidates.items(), key=lambda item: (float(item[1]), item[0]))
    if score < FORWARD_DANGER_CM and "backoff" in candidates:
        return "backoff"
    return str(action)


def is_fence_or_rail_context(*, obstacle_hint: str = "", environment_id: str = "") -> bool:
    text = f"{obstacle_hint} {environment_id}".lower()
    return any(token in text for token in ("fence", "rail", "railing", "栏", "围栏"))


def is_tree_or_pole_context(*, obstacle_hint: str = "", environment_id: str = "") -> bool:
    text = f"{obstacle_hint} {environment_id}".lower()
    return any(token in text for token in ("tree", "pole", "trunk", "branch"))


def is_fence_or_rail_context(*, obstacle_hint: str = "", environment_id: str = "") -> bool:
    text = f"{obstacle_hint} {environment_id}".lower()
    return any(token in text for token in ("fence", "rail", "railing", "fence_or_rail"))


def is_tree_or_pole_context(*, obstacle_hint: str = "", environment_id: str = "") -> bool:
    text = f"{obstacle_hint} {environment_id}".lower()
    return any(
        token in text
        for token in (
            "tree",
            "pole",
            "trunk",
            "branch",
            "canopy",
            "tree_or_pole",
            "tree_trunk_or_pole",
            "tree_canopy_or_cluster",
        )
    )


def classify_tree_or_pole_subtype(summary: Dict[str, Any]) -> Tuple[str, str]:
    width_cm = as_float(summary.get("obstacle_width_cm"))
    height_cm = as_float(summary.get("obstacle_height_cm"))
    if width_cm <= 0.0:
        return "tree_unknown", "tree/pole width unavailable"
    if width_cm <= TREE_POLE_NARROW_WIDTH_CM:
        return (
            "tree_trunk_or_pole",
            f"tree/pole width={width_cm:.1f}cm <= {TREE_POLE_NARROW_WIDTH_CM:.1f}cm",
        )
    if width_cm <= TREE_POLE_WIDE_WIDTH_CM:
        return (
            "tree_canopy_edge",
            f"tree/pole width={width_cm:.1f}cm <= {TREE_POLE_WIDE_WIDTH_CM:.1f}cm height={height_cm:.1f}cm",
        )
    return (
        "tree_canopy_or_cluster",
        f"tree/pole width={width_cm:.1f}cm > {TREE_POLE_WIDE_WIDTH_CM:.1f}cm height={height_cm:.1f}cm",
    )


def annotate_semantic_obstacle_context(
    summary: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
) -> None:
    if is_fence_or_rail_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        summary["semantic_obstacle_class"] = "fence_or_rail"
        summary["semantic_obstacle_subtype"] = "fence_or_rail"
        return
    if is_tree_or_pole_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        subtype, detail = classify_tree_or_pole_subtype(summary)
        summary["semantic_obstacle_class"] = "tree_or_pole"
        summary["semantic_obstacle_subtype"] = subtype
        summary["semantic_obstacle_subtype_reason"] = detail


def depth_predicted_fence_vertical_offset_cm(summary: Dict[str, Any]) -> float:
    front_min = as_float(summary.get("front_min_depth_cm"))
    up_min = as_float(summary.get("up_min_depth_cm"))
    down_min = as_float(summary.get("down_min_depth_cm"))
    required = FENCE_OVERFLY_MIN_VERTICAL_OFFSET_CM
    if up_min > 0.0:
        required = max(required, up_min * FENCE_OVERFLY_UP_DEPTH_SCALE)
    if front_min > 0.0 and front_min < FORWARD_DANGER_CM:
        required += 60.0
    if down_min > 0.0 and down_min < FORWARD_DANGER_CM:
        required += 40.0
    return max(FENCE_OVERFLY_MIN_VERTICAL_OFFSET_CM, min(FENCE_OVERFLY_MAX_VERTICAL_OFFSET_CM, required))


def fence_overfly_requirement(
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
) -> Tuple[bool, float, float, str]:
    if not is_fence_or_rail_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        return False, 0.0, 0.0, "not fence/rail context"

    progress, _route_offset_cm = route_progress_and_deviation(current_pose, start, goal)
    route_pose = point_on_route(start, goal, progress)
    route_z = as_float(route_pose.get("z"))
    pose_z = as_float(current_pose.get("z"))
    vertical_offset_cm = max(0.0, pose_z - route_z)
    required_offset = depth_predicted_fence_vertical_offset_cm(summary)
    front_min = as_float(summary.get("front_min_depth_cm"))
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    obstacle_ahead = (
        (front_min > 0.0 and front_min < SLOW_FORWARD_CAUTION_CM)
        or geometry in {"vertical_wall", "thin_structure", "low_obstacle", "unknown"}
        or not bool(summary.get("forward_swept_clear", True))
    )
    needed = bool(obstacle_ahead and vertical_offset_cm < required_offset)
    reason = (
        f"fence/rail depth predicted overfly offset: current_offset={vertical_offset_cm:.1f}cm "
        f"required_offset={required_offset:.1f}cm front={front_min:.1f}cm "
        f"up_depth={as_float(summary.get('up_min_depth_cm')):.1f}cm "
        f"down_depth={as_float(summary.get('down_min_depth_cm')):.1f}cm "
        f"geometry={geometry} obstacle_ahead={obstacle_ahead}"
    )
    return needed, required_offset, vertical_offset_cm, reason


def is_building_context(*, obstacle_hint: str = "", environment_id: str = "") -> bool:
    text = f"{obstacle_hint} {environment_id}".lower()
    return any(token in text for token in ("building", "roof", "house", "wall", "door", "entrance"))


def make_building_obstacle_state() -> Dict[str, Any]:
    return {
        "status": "idle",
        "front_building_obstacle": False,
        "current_front_building_obstacle": False,
        "observed": False,
        "cleared": False,
        "observed_progress": None,
        "cleared_progress": None,
        "last_progress": 0.0,
        "last_front_min_depth_cm": 0.0,
        "last_geometry": "unknown",
        "state_reason": "not initialized",
    }


def detect_front_building_obstacle(summary: Dict[str, Any]) -> Tuple[bool, str]:
    front_min = as_float(summary.get("front_min_depth_cm"))
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    if geometry in {"vertical_wall", "overhang_beam", "low_obstacle", "thin_structure"}:
        return True, f"geometry={geometry}"
    if geometry == "unknown" and front_min > 0.0 and front_min < BUILDING_FORWARD_CLEAR_CM:
        return True, f"unknown building-like front depth {front_min:.1f}cm"
    if front_min > 0.0 and front_min < BUILDING_VANTAGE_FRONT_MAX_CM:
        return True, f"front depth {front_min:.1f}cm below building vantage threshold"
    return False, f"no building-like front obstacle: geometry={geometry} front={front_min:.1f}cm"


def update_building_obstacle_state(
    state: Dict[str, Any],
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
) -> Dict[str, Any]:
    if not isinstance(state, dict):
        state = make_building_obstacle_state()

    progress, _cross_track_cm = route_progress_raw_and_deviation(current_pose, start, goal)
    front_min = as_float(summary.get("front_min_depth_cm"))
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    detected, detect_reason = detect_front_building_obstacle(summary)
    state["current_front_building_obstacle"] = bool(detected)
    state["last_progress"] = round(float(progress), 5)
    state["last_front_min_depth_cm"] = round(front_min, 3)
    state["last_geometry"] = geometry

    if not is_building_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        state.update(
            {
                "status": "disabled",
                "front_building_obstacle": False,
                "cleared": False,
                "state_reason": "not building context",
            }
        )
        return state

    status = str(state.get("status", "idle") or "idle")
    if status not in {"active", "cleared"} and detected:
        state["status"] = "active"
        state["front_building_obstacle"] = True
        state["observed"] = True
        state["cleared"] = False
        state["observed_progress"] = round(float(progress), 5)
        state["state_reason"] = f"building obstacle observed: {detect_reason}"
        status = "active"

    if status == "active":
        observed_progress = as_float(state.get("observed_progress"), progress)
        progress_clear_at = max(
            BUILDING_CLEAR_MIN_RAW_PROGRESS,
            observed_progress + BUILDING_CLEAR_MIN_PROGRESS_DELTA,
        )
        needed, required_z, vertical_offset_cm, vantage_reason = building_vantage_requirement(
            summary,
            current_pose,
            start,
            goal,
            obstacle_hint=obstacle_hint,
            environment_id=environment_id,
        )
        pose_z = as_float(current_pose.get("z"))
        height_ready = pose_z >= required_z
        front_clear = front_min <= 0.0 or front_min >= BUILDING_CRUISE_MIN_FRONT_CM or geometry == "none"
        progress_ready = progress >= progress_clear_at
        if height_ready and front_clear and progress_ready:
            state["status"] = "cleared"
            state["front_building_obstacle"] = False
            state["cleared"] = True
            state["cleared_progress"] = round(float(progress), 5)
            state["state_reason"] = (
                f"building obstacle cleared after overfly: progress={progress:.3f} "
                f"required_progress={progress_clear_at:.3f} z={pose_z:.1f}cm "
                f"required_z={required_z:.1f}cm front={front_min:.1f}cm geometry={geometry}"
            )
        else:
            state["front_building_obstacle"] = True
            state["cleared"] = False
            state["state_reason"] = (
                f"building obstacle active: {vantage_reason}; "
                f"clear_check height={height_ready} front={front_clear} "
                f"progress={progress:.3f}/{progress_clear_at:.3f}"
            )
    elif status == "cleared":
        state["front_building_obstacle"] = False
        state["cleared"] = True
        state["state_reason"] = (
            f"building obstacle already cleared; current detection ignored to avoid repeated overfly: {detect_reason}"
        )
    else:
        state["front_building_obstacle"] = False
        state["cleared"] = False
        state["state_reason"] = f"waiting for building obstacle observation: {detect_reason}"

    return state


def building_vantage_requirement(
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
) -> Tuple[bool, float, float, str]:
    if not is_building_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        return False, 0.0, 0.0, "not building context"

    front_min = as_float(summary.get("front_min_depth_cm"))
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    progress, _route_offset_cm = route_progress_and_deviation(current_pose, start, goal)
    route_pose = point_on_route(start, goal, progress)
    route_z = as_float(route_pose.get("z"))
    pose_z = as_float(current_pose.get("z"))
    required_z = max(BUILDING_VANTAGE_MIN_Z_CM, route_z + BUILDING_VANTAGE_MIN_VERTICAL_OFFSET_CM)
    vertical_offset_cm = pose_z - route_z
    height_ready = pose_z >= required_z
    front_ready = front_min <= 0.0 or front_min >= BUILDING_FORWARD_CLEAR_CM
    cruise_ready = front_min <= 0.0 or front_min >= BUILDING_CRUISE_MIN_FRONT_CM
    obstacle_ahead = (
        (front_min > 0.0 and front_min < BUILDING_VANTAGE_FRONT_MAX_CM)
        or geometry in {"vertical_wall", "overhang_beam", "low_obstacle", "thin_structure", "unknown"}
    )
    if height_ready and cruise_ready:
        return False, required_z, vertical_offset_cm, (
            f"building vantage reached: z={pose_z:.1f}cm required_z={required_z:.1f}cm "
            f"vertical_offset={vertical_offset_cm:.1f}cm front={front_min:.1f}cm "
            f"cruise_front={BUILDING_CRUISE_MIN_FRONT_CM:.1f}cm "
            f"clear_front={BUILDING_FORWARD_CLEAR_CM:.1f}cm obstacle_ahead={obstacle_ahead}"
        )
    if height_ready and not cruise_ready:
        if pose_z < BUILDING_VANTAGE_MAX_Z_CM:
            climb_target_z = min(BUILDING_VANTAGE_MAX_Z_CM, max(required_z, pose_z + DEFAULT_ROUTE_VERTICAL_STEP_CM))
            return True, climb_target_z, vertical_offset_cm, (
                f"building front not clear after initial vantage: z={pose_z:.1f}cm "
                f"target_z={climb_target_z:.1f}cm front={front_min:.1f}cm "
                f"required_cruise_front={BUILDING_CRUISE_MIN_FRONT_CM:.1f}cm geometry={geometry}"
            )
        return True, pose_z, vertical_offset_cm, (
            f"building front still blocked at max vantage: z={pose_z:.1f}cm "
            f"front={front_min:.1f}cm required_cruise_front={BUILDING_CRUISE_MIN_FRONT_CM:.1f}cm "
            f"geometry={geometry}"
        )
    return True, required_z, vertical_offset_cm, (
        f"building needs higher vantage before forward probe: z={pose_z:.1f}cm "
        f"required_z={required_z:.1f}cm vertical_offset={vertical_offset_cm:.1f}cm "
        f"front={front_min:.1f}cm required_front={BUILDING_FORWARD_CLEAR_CM:.1f}cm geometry={geometry}"
    )


def building_vantage_action(
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
    vertical_step_cm: float,
) -> Tuple[Optional[str], Optional[Dict[str, float]], str]:
    needed, required_z, _vertical_offset_cm, reason = building_vantage_requirement(
        summary,
        current_pose,
        start,
        goal,
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    if not needed:
        return None, None, reason

    pose_z = as_float(current_pose.get("z"))
    climb_cm = required_z - pose_z
    front_min = as_float(summary.get("front_min_depth_cm"))
    if climb_cm > 1.0 and bool(summary.get("up_swept_clear", True)):
        payload = action_payload("up")
        payload["up_cm"] = min(max(12.0, climb_cm), float(vertical_step_cm))
        return "up", payload, reason
    if (
        climb_cm <= 1.0
        and bool(summary.get("forward_swept_clear", True))
        and front_min >= BUILDING_MAX_VANTAGE_CREEP_FRONT_CM
    ):
        payload = action_payload("forward")
        payload["forward_cm"] = 20.0
        payload["up_cm"] = 0.0
        return "forward", payload, (
            f"{reason}; max vantage reached and forward swept volume is safe enough "
            f"for creep ({front_min:.1f}/{BUILDING_MAX_VANTAGE_CREEP_FRONT_CM:.1f}cm)"
        )
    if bool(summary.get("backoff_swept_clear", True)):
        payload = action_payload("backoff")
        return "backoff", payload, f"{reason}; up blocked, backoff for building replan"
    return "hold", action_payload("hold"), f"{reason}; up/backoff blocked, hold"


def building_cruise_forward_action(
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    route_step_cm: float,
    front_min_cm: float,
    reason_prefix: str,
) -> Tuple[str, Dict[str, float], str, str]:
    step_cm = effective_route_step_cm(
        distance_to_goal_cm=distance_to_goal_cm,
        reach_tol_cm=reach_tol_cm,
        route_step_cm=route_step_cm,
        front_min_cm=front_min_cm,
        respect_front_caution=False,
    )
    payload, line_target, line_reason = straight_line_payload(
        current_pose,
        start,
        goal,
        route_step_cm=step_cm,
        side_correction_cm=0.0,
        vertical_step_cm=0.0,
    )
    payload["up_cm"] = 0.0
    payload["action_name"] = "forward"
    reason = (
        f"{reason_prefix}; keep altitude and move forward before descending; "
        f"{line_reason}; line_target={line_target}"
    )
    return "forward", payload, reason, "BUILDING_CRUISE"


def building_descent_allowed_after_clear(
    building_state: Optional[Dict[str, Any]],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    obstacle_hint: str = "",
    environment_id: str = "",
) -> Tuple[bool, str]:
    if not isinstance(building_state, dict):
        return True, "no building state"
    if str(building_state.get("status", "") or "").lower() != "cleared":
        return True, "building not cleared yet"
    if not is_building_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        return True, "not building context"

    raw_progress, cross_track_cm = route_progress_raw_and_deviation(current_pose, start, goal)
    xy_distance_cm = distance_xy_cm(current_pose, goal)
    xy_gate_cm = max(BUILDING_DESCENT_MAX_XY_DISTANCE_CM, float(reach_tol_cm) + 80.0)
    near_goal_distance_cm = max(BUILDING_DESCENT_MAX_DISTANCE_CM, float(reach_tol_cm) + GOAL_ALIGNMENT_DISTANCE_MARGIN_CM)
    xy_ready = xy_distance_cm <= xy_gate_cm
    progress_or_distance_ready = (
        raw_progress >= BUILDING_DESCENT_MIN_RAW_PROGRESS
        or float(distance_to_goal_cm) <= near_goal_distance_cm
    )
    allowed = bool(xy_ready and progress_or_distance_ready)
    reason = (
        f"post-building descent gate allowed={allowed}: "
        f"distance={float(distance_to_goal_cm):.1f}/{near_goal_distance_cm:.1f}cm "
        f"xy_distance={xy_distance_cm:.1f}/{xy_gate_cm:.1f}cm "
        f"raw_progress={raw_progress:.3f}/{BUILDING_DESCENT_MIN_RAW_PROGRESS:.3f} "
        f"cross_track={cross_track_cm:.1f}cm"
    )
    return allowed, reason


def post_building_horizontal_approach_action(
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    route_step_cm: float,
    front_min_cm: float,
    gate_reason: str,
) -> Tuple[str, Dict[str, float], str, str]:
    step_cm = effective_route_step_cm(
        distance_to_goal_cm=distance_to_goal_cm,
        reach_tol_cm=reach_tol_cm,
        route_step_cm=route_step_cm,
        front_min_cm=front_min_cm,
        respect_front_caution=True,
    )
    payload, line_target, line_reason = straight_line_payload(
        current_pose,
        start,
        goal,
        route_step_cm=step_cm,
        side_correction_cm=0.0,
        vertical_step_cm=0.0,
    )
    payload["up_cm"] = 0.0
    action = str(payload.get("action_name", "forward"))
    reason = (
        f"building cleared but descent delayed until near goal; keep altitude and approach XY first; "
        f"{gate_reason}; {line_reason}; line_target={line_target}"
    )
    return action, payload, reason, "POST_BUILDING_APPROACH"


def fence_overfly_action(
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
    vertical_step_cm: float,
) -> Tuple[Optional[str], Optional[Dict[str, float]], str]:
    needed, _required_offset, _vertical_offset, reason = fence_overfly_requirement(
        summary,
        current_pose,
        start,
        goal,
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    if not needed:
        return None, None, reason
    if bool(summary.get("up_swept_clear", True)):
        payload = action_payload("up")
        payload["up_cm"] = float(vertical_step_cm)
        return "up", payload, f"{reason}; climb until depth-predicted fence overfly height is reached"
    action, fallback_reason = pointcloud_direction_rule_action(
        summary,
        {"bearing_deg_body": 0.0, "dz_cm": 0.0},
        action_payload("hold"),
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    payload = action_payload(action)
    return action, payload, f"{reason}; up blocked, fallback={fallback_reason}"


def fence_descent_allowed(
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    obstacle_hint: str = "",
    environment_id: str = "",
) -> Tuple[bool, str]:
    if not is_fence_or_rail_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        return True, "not fence/rail context"
    needed, required_offset, vertical_offset_cm, overfly_reason = fence_overfly_requirement(
        summary,
        current_pose,
        start,
        goal,
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    raw_progress, cross_track_cm = route_progress_raw_and_deviation(current_pose, start, goal)
    near_goal_distance_cm = max(FENCE_DESCENT_MAX_DISTANCE_CM, float(reach_tol_cm) + 40.0)
    xy_distance_cm = distance_xy_cm(current_pose, goal)
    xy_ready = xy_distance_cm <= FENCE_FINAL_XY_TOL_CM
    front_min = as_float(summary.get("front_min_depth_cm"))
    controlled_down_clear = controlled_fence_descent_clear(summary)
    front_safe = front_min <= 0.0 or front_min >= FORWARD_DANGER_CM or controlled_down_clear
    overfly_approach_ready = (
        not needed
        and vertical_offset_cm >= required_offset
        and (raw_progress >= FENCE_DESCENT_MIN_RAW_PROGRESS or float(distance_to_goal_cm) <= near_goal_distance_cm)
    )
    allowed = bool(xy_ready and front_safe)
    reason = (
        f"fence descent gate allowed={allowed}: distance={float(distance_to_goal_cm):.1f}/{near_goal_distance_cm:.1f}cm "
        f"xy_distance={xy_distance_cm:.1f}/{FENCE_FINAL_XY_TOL_CM:.1f}cm "
        f"raw_progress={raw_progress:.3f}/{FENCE_DESCENT_MIN_RAW_PROGRESS:.3f} "
        f"xy_ready={xy_ready} overfly_ready={overfly_approach_ready} "
        f"controlled_down_clear={controlled_down_clear} "
        f"front_safe={front_safe} cross_track={cross_track_cm:.1f}cm; {overfly_reason}"
    )
    return allowed, reason


def controlled_fence_descent_clear(summary: Dict[str, Any]) -> bool:
    if bool(summary.get("down_swept_clear", False)):
        return True
    down_p05 = as_float(summary.get("down_p05_depth_cm"))
    down_fraction = as_float(summary.get("down_close_fraction"), 1.0)
    down_min = as_float(summary.get("down_min_depth_cm"))
    return bool(
        down_p05 >= FENCE_CONTROLLED_DESCENT_MIN_P05_CM
        and down_fraction <= FENCE_CONTROLLED_DESCENT_MAX_CLOSE_FRACTION
        and down_min >= POINTCLOUD_FORWARD_PROBE_MIN_FRONT_CM
    )


def post_fence_horizontal_approach_action(
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    route_step_cm: float,
    front_min_cm: float,
    gate_reason: str,
) -> Tuple[str, Dict[str, float], str, str]:
    step_cm = effective_route_step_cm(
        distance_to_goal_cm=distance_to_goal_cm,
        reach_tol_cm=reach_tol_cm,
        route_step_cm=route_step_cm,
        front_min_cm=front_min_cm,
        respect_front_caution=False,
    )
    payload, line_target, line_reason = straight_line_payload(
        current_pose,
        start,
        goal,
        route_step_cm=step_cm,
        side_correction_cm=0.0,
        vertical_step_cm=0.0,
    )
    payload["up_cm"] = 0.0
    payload["action_name"] = "forward"
    reason = (
        f"fence/rail overfly complete enough for forward approach but not descent; keep height and approach XY; "
        f"{gate_reason}; {line_reason}; line_target={line_target}"
    )
    return "forward", payload, reason, "POST_FENCE_APPROACH"


def fence_xy_alignment_needed(current_pose: Dict[str, Any], goal: Dict[str, Any]) -> Tuple[bool, float, float]:
    xy_distance_cm = distance_xy_cm(current_pose, goal)
    dz = as_float(goal.get("z")) - as_float(current_pose.get("z"))
    needed = bool(
        xy_distance_cm > FENCE_FINAL_XY_TOL_CM
        and xy_distance_cm <= FENCE_FINAL_XY_ADJUST_ZONE_CM
        and dz < -Z_ALIGN_TOL_CM
    )
    return needed, xy_distance_cm, dz


def fence_xy_alignment_action(
    current_pose: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    route_step_cm: float,
    side_correction_cm: float,
    reason_prefix: str,
) -> Tuple[str, Dict[str, float], str, str]:
    delta = body_delta_to_goal(current_pose, goal)
    forward = as_float(delta.get("forward_cm"))
    right = as_float(delta.get("right_cm"))
    xy_distance_cm = distance_xy_cm(current_pose, goal)
    dz = as_float(goal.get("z")) - as_float(current_pose.get("z"))

    payload = action_payload("hold")
    detail = "hold horizontal position"
    if xy_distance_cm > FENCE_FINAL_XY_TOL_CM and abs(forward) >= abs(right):
        if forward > 0.0:
            payload = action_payload("forward")
            payload["forward_cm"] = min(float(route_step_cm), max(8.0, abs(forward) * 0.45))
            detail = "move forward toward target XY"
        else:
            payload = action_payload("backoff")
            payload["forward_cm"] = -min(float(route_step_cm), max(8.0, abs(forward) * 0.45))
            detail = "backoff toward target XY after overshoot"
    elif xy_distance_cm > FENCE_FINAL_XY_TOL_CM:
        if right > 0.0:
            payload = action_payload("right")
            payload["right_cm"] = min(float(side_correction_cm), max(8.0, abs(right) * 0.45))
            detail = "move right toward target XY"
        else:
            payload = action_payload("left")
            payload["right_cm"] = -min(float(side_correction_cm), max(8.0, abs(right) * 0.45))
            detail = "move left toward target XY"

    payload["up_cm"] = 0.0
    payload["yaw_delta_deg"] = 0.0
    action = str(payload.get("action_name", "hold"))
    reason = (
        f"{reason_prefix}; fence final XY alignment before descent: {detail}; "
        f"xy_distance={xy_distance_cm:.1f}/{FENCE_FINAL_XY_TOL_CM:.1f}cm "
        f"body_goal_forward={forward:.1f}cm body_goal_right={right:.1f}cm "
        f"body_goal_up={dz:.1f}cm"
    )
    return action, payload, reason, "FENCE_XY_ALIGN"


def pointcloud_direction_rule_action(
    summary: Dict[str, Any],
    rel: Dict[str, Any],
    last_action: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
) -> Tuple[str, str]:
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    bearing = as_float(rel.get("bearing_deg_body"))
    previous_side = last_side_action(last_action)
    fence_context = is_fence_or_rail_context(obstacle_hint=obstacle_hint, environment_id=environment_id)

    depth_keys = {
        "left": "left_min_depth_cm",
        "right": "right_min_depth_cm",
        "up": "up_min_depth_cm",
        "down": "down_min_depth_cm",
    }
    swept_keys = {
        "left": "left_swept_clear",
        "right": "right_swept_clear",
        "up": "up_swept_clear",
        "down": "down_swept_clear",
        "backoff": "backoff_swept_clear",
    }

    def action_available(action: str) -> bool:
        if action == "down":
            return bool(summary.get("down_swept_clear", False))
        return bool(summary.get(swept_keys.get(action, ""), True))

    def clearance_score(action: str) -> float:
        if action == "backoff":
            return 260.0
        value = as_float(summary.get(depth_keys.get(action, "")))
        return 999.0 if value <= 0.0 else value

    def choose(actions: Tuple[str, ...], *, prefer_goal_side: bool = False) -> str:
        candidates: Dict[str, float] = {}
        for action in actions:
            if not action_available(action):
                continue
            score = clearance_score(action)
            if prefer_goal_side:
                if action == "left" and bearing < -GOAL_YAW_TOL_DEG:
                    score += 80.0
                elif action == "right" and bearing > GOAL_YAW_TOL_DEG:
                    score += 80.0
            if action == previous_side:
                score += 45.0
            candidates[action] = score
        if not candidates:
            return ""
        action, _score = max(candidates.items(), key=lambda item: (float(item[1]), item[0]))
        return action

    tree_context = is_tree_or_pole_context(obstacle_hint=obstacle_hint, environment_id=environment_id)
    if tree_context and geometry in {"vertical_wall", "thin_structure", "unknown", "low_obstacle"}:
        width_cm = as_float(summary.get("obstacle_width_cm"))
        narrow = width_cm <= 0.0 or width_cm <= TREE_POLE_NARROW_WIDTH_CM
        if narrow:
            action = choose(("left", "right"), prefer_goal_side=True)
            if action:
                return action, f"tree/pole narrow obstacle width={width_cm:.1f}cm: choose clearer lateral side"
        action = choose(("up", "left", "right"), prefer_goal_side=True)
        if action:
            return action, f"tree/pole wide/uncertain obstacle width={width_cm:.1f}cm: choose safest clearance"

    if fence_context and geometry in {"vertical_wall", "thin_structure", "unknown", "low_obstacle"}:
        action = choose(("up",))
        if action:
            return action, "fence/rail obstacle: climb over before lateral correction"
        action = choose(("left", "right"), prefer_goal_side=True)
        if action:
            return action, "fence/rail obstacle but up blocked: choose clearer lateral side"

    if geometry in {"vertical_wall", "thin_structure", "unknown"}:
        action = choose(("left", "right"), prefer_goal_side=True)
        if action:
            return action, "front vertical/thin/unknown obstacle: choose clearer lateral side"
        action = choose(("up", "down"))
        if action:
            return action, "lateral sides blocked: choose clearer vertical clearance"
    elif geometry == "low_obstacle":
        action = choose(("up", "left", "right"), prefer_goal_side=True)
        if action:
            return action, "low obstacle: prefer climb-over if top swept volume is clear"
    elif geometry == "overhang_beam":
        action = choose(("down", "left", "right"), prefer_goal_side=True)
        if action:
            return action, "overhang/beam: prefer lower or lateral clearance instead of blind climb"
    else:
        action = choose(("left", "right", "up", "down"), prefer_goal_side=True)
        if action:
            return action, "unclassified point-cloud obstacle: choose largest swept-volume clearance"

    if action_available("backoff"):
        return "backoff", "no left/right/up/down swept-volume clearance: backoff"
    return "hold", "no swept-volume clearance: hold"


def tree_or_pole_recovery_action(
    summary: Dict[str, Any],
    rel: Dict[str, Any],
    last_action: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
    vertical_step_cm: float,
) -> Tuple[Optional[str], Optional[Dict[str, float]], str]:
    if not is_tree_or_pole_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        return None, None, "not tree/pole context"

    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    front_min = as_float(summary.get("front_min_depth_cm"))
    if front_min <= 0.0 or front_min >= TREE_POLE_FORWARD_PROBE_CLEAR_CM:
        subtype, subtype_reason = classify_tree_or_pole_subtype(summary)
        return None, None, (
            f"tree/pole split {subtype}: front={front_min:.1f}cm is clear enough for Route6_entrance_search/probe; "
            f"{subtype_reason}"
        )

    subtype, subtype_reason = classify_tree_or_pole_subtype(summary)
    progress, route_offset_cm = route_progress_and_deviation(current_pose, start, goal)
    route_pose = point_on_route(start, goal, progress)
    vertical_offset_cm = abs(as_float(current_pose.get("z")) - as_float(route_pose.get("z")))
    bearing = as_float(rel.get("bearing_deg_body"))
    previous_side = last_side_action(last_action)

    side_scores: Dict[str, float] = {}
    side_depths: Dict[str, float] = {
        "left": as_float(summary.get("left_min_depth_cm")),
        "right": as_float(summary.get("right_min_depth_cm")),
    }
    for action, clear_cm in side_depths.items():
        if clear_cm < TREE_POLE_MIN_SIDE_CLEAR_CM:
            continue
        score = clear_cm
        if action == previous_side:
            score += 45.0
        if action == "left" and bearing < -GOAL_YAW_TOL_DEG:
            score += 20.0
        elif action == "right" and bearing > GOAL_YAW_TOL_DEG:
            score += 20.0
        if subtype == "tree_trunk_or_pole":
            score += 25.0
        side_scores[action] = score

    if side_scores:
        action, _score = max(side_scores.items(), key=lambda item: (float(item[1]), item[0]))
        payload = action_payload(action)
        step_cm = min(TREE_POLE_LATERAL_STEP_CM, max(10.0, side_depths[action] - TREE_POLE_MIN_SIDE_CLEAR_CM))
        payload["forward_cm"] = 0.0
        payload["right_cm"] = step_cm if action == "right" else -step_cm
        payload["up_cm"] = 0.0
        payload["action_name"] = "side_step_right" if action == "right" else "side_step_left"
        reason = (
            f"tree/pole split {subtype}: {subtype_reason}; "
            f"front={front_min:.1f}cm geometry={geometry} route_offset={route_offset_cm:.1f}cm "
            f"vertical_offset={vertical_offset_cm:.1f}cm; relaxed side clearance "
            f"left={side_depths['left']:.1f}cm right={side_depths['right']:.1f}cm "
            f">= {TREE_POLE_MIN_SIDE_CLEAR_CM:.1f}cm; selected={action}"
        )
        return action, payload, reason

    up_min = as_float(summary.get("up_min_depth_cm"))
    if subtype in {"tree_canopy_edge", "tree_canopy_or_cluster"} and up_min >= TREE_POLE_UP_CLEAR_CM and bool(
        summary.get("up_swept_clear", True)
    ):
        payload = action_payload("up")
        payload["up_cm"] = float(vertical_step_cm)
        return (
            "up",
            payload,
            f"tree/pole split {subtype}: lateral blocked, canopy-like obstacle has up clearance {up_min:.1f}cm",
        )

    if bool(summary.get("backoff_swept_clear", True)) and front_min < POINTCLOUD_FORWARD_PROBE_MIN_FRONT_CM:
        return (
            "backoff",
            action_payload("backoff"),
            f"tree/pole split {subtype}: front={front_min:.1f}cm and both sides below relaxed clearance",
        )
    return (
        "hold",
        action_payload("hold"),
        f"tree/pole split {subtype}: no relaxed lateral/up clearance, hold for operator or replan",
    )


def pointcloud_direction_rule_should_expand(summary: Dict[str, Any]) -> Tuple[bool, str]:
    available = bool(summary.get("available", False))
    if not available:
        return True, "pointcloud unavailable"
    front_min = as_float(summary.get("front_min_depth_cm"))
    forward_swept_clear = bool(summary.get("forward_swept_clear", True))
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    if not forward_swept_clear:
        return True, "forward swept volume blocked"
    if front_min > 0.0 and front_min < SLOW_FORWARD_CAUTION_CM:
        return True, f"front obstacle inside caution distance {front_min:.1f}cm"
    if geometry not in {"none", ""} and front_min > 0.0 and front_min < DEFAULT_ROUTE_STEP_CM * 10.0:
        return True, f"pointcloud geometry={geometry} ahead"
    return False, "front corridor clear"


def goal_alignment_action(
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    side_correction_cm: float,
    vertical_step_cm: float,
    allow_controlled_fence_descent: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, float]], str]:
    raw_progress, cross_track_cm = route_progress_raw_and_deviation(current_pose, start, goal)
    if raw_progress < GOAL_ALIGNMENT_PROGRESS and float(distance_to_goal_cm) > float(reach_tol_cm) + GOAL_ALIGNMENT_DISTANCE_MARGIN_CM:
        return None, None, "not in near-goal alignment zone"

    delta = body_delta_to_goal(current_pose, goal)
    front = as_float(delta.get("forward_cm"))
    right = as_float(delta.get("right_cm"))
    up = as_float(delta.get("up_cm"))
    yaw_delta = as_float(delta.get("yaw_delta_deg"))
    front_min = as_float(summary.get("front_min_depth_cm"))
    front_clear = front_min <= 0.0 or front_min >= POINTCLOUD_FORWARD_PROBE_MIN_FRONT_CM
    can_forward = bool(front_clear and bool(summary.get("forward_swept_clear", True)))

    payload = action_payload("hold")
    action = "hold"
    detail = ""
    if abs(up) > Z_ALIGN_TOL_CM:
        if up > 0.0 and bool(summary.get("up_swept_clear", True)):
            payload = action_payload("up")
            payload["up_cm"] = min(max(8.0, abs(up)), float(vertical_step_cm))
            action = "up"
            detail = "near-goal vertical alignment: climb toward goal height"
        elif up < 0.0 and (
            bool(summary.get("down_swept_clear", False))
            or (allow_controlled_fence_descent and controlled_fence_descent_clear(summary))
        ):
            payload = action_payload("down")
            payload["up_cm"] = -min(max(8.0, abs(up)), float(vertical_step_cm))
            action = "down"
            detail = "near-goal vertical alignment: descend toward goal height"
    if not detail and raw_progress >= 1.0 and front < -18.0 and bool(summary.get("backoff_swept_clear", True)):
        payload = action_payload("backoff")
        payload["forward_cm"] = -min(20.0, max(8.0, abs(front)))
        action = "backoff"
        detail = "near-goal overshoot: backoff toward goal instead of continuing forward"
    if not detail and abs(right) > 28.0:
        if right > 0.0 and bool(summary.get("right_swept_clear", True)):
            payload = action_payload("right")
            payload["right_cm"] = min(max(8.0, abs(right) * 0.35), float(side_correction_cm))
            action = "right"
            detail = "near-goal lateral alignment: move right toward goal"
        elif right < 0.0 and bool(summary.get("left_swept_clear", True)):
            payload = action_payload("left")
            payload["right_cm"] = -min(max(8.0, abs(right) * 0.35), float(side_correction_cm))
            action = "left"
            detail = "near-goal lateral alignment: move left toward goal"
    if not detail and raw_progress < 1.0 and front > 18.0 and can_forward:
        payload = {
            "forward_cm": min(18.0, max(8.0, front * 0.35)),
            "right_cm": 0.0,
            "up_cm": 0.0,
            "yaw_delta_deg": max(-15.0, min(15.0, yaw_delta)),
            "action_name": "goal_forward",
        }
        action = "goal_forward"
        detail = "near-goal final forward approach"
    if not detail and abs(yaw_delta) > GOAL_YAW_TOL_DEG:
        payload = action_payload("yaw_left" if yaw_delta < 0.0 else "yaw_right", yaw_delta_deg=max(-20.0, min(20.0, yaw_delta)))
        action = str(payload.get("action_name", "yaw_right"))
        detail = "near-goal yaw alignment"
    if not detail:
        detail = "near-goal alignment hold: no safe correction action"

    reason = (
        f"{detail}; distance={float(distance_to_goal_cm):.1f}cm raw_progress={raw_progress:.3f} "
        f"cross_track={cross_track_cm:.1f}cm body_goal_forward={front:.1f} "
        f"body_goal_right={right:.1f} body_goal_up={up:.1f}"
    )
    return action, payload, reason


def pointcloud_direction_rule_forward_probe(
    summary: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    *,
    obstacle_hint: str = "",
    environment_id: str = "",
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    route_step_cm: float,
    side_correction_cm: float,
) -> Tuple[Optional[Dict[str, float]], str]:
    """Small forward step after a lateral/vertical avoidance offset has opened room."""
    if not bool(summary.get("available", False)):
        return None, "pointcloud unavailable"
    front_min = as_float(summary.get("front_min_depth_cm"))
    if front_min <= 0.0 or front_min < POINTCLOUD_FORWARD_PROBE_MIN_FRONT_CM:
        return None, f"front too close for forward probe ({front_min:.1f}cm)"

    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    if geometry == "overhang_beam" and not bool(summary.get("forward_swept_clear", True)):
        return None, "overhang/beam still blocks forward swept volume"
    if is_tree_or_pole_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
        subtype, subtype_reason = classify_tree_or_pole_subtype(summary)
        if front_min < TREE_POLE_FORWARD_PROBE_CLEAR_CM:
            return None, (
                f"tree/pole split {subtype}: front={front_min:.1f}cm below "
                f"{TREE_POLE_FORWARD_PROBE_CLEAR_CM:.1f}cm, keep lateral avoidance before forward probe; "
                f"{subtype_reason}"
            )

    progress, route_offset_cm = route_progress_and_deviation(current_pose, start, goal)
    route_pose = point_on_route(start, goal, progress)
    vertical_offset_cm = abs(as_float(current_pose.get("z")) - as_float(route_pose.get("z")))
    building_needs_vantage, _required_z, _building_vertical_offset, building_reason = building_vantage_requirement(
        summary,
        current_pose,
        start,
        goal,
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    if building_needs_vantage:
        return None, building_reason
    fence_needs_overfly, fence_required_offset, _fence_offset, fence_reason = fence_overfly_requirement(
        summary,
        current_pose,
        start,
        goal,
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    if fence_needs_overfly and bool(summary.get("up_swept_clear", True)):
        return None, (
            f"fence/rail needs climb before forward probe "
            f"({vertical_offset_cm:.1f}/{fence_required_offset:.1f}cm vertical offset); {fence_reason}"
        )
    min_offset_cm = max(POINTCLOUD_FORWARD_PROBE_MIN_OFFSET_CM, min(90.0, float(side_correction_cm) * 1.4))
    if route_offset_cm < min_offset_cm and vertical_offset_cm < POINTCLOUD_FORWARD_PROBE_MIN_OFFSET_CM:
        return None, f"avoidance offset too small for probe ({route_offset_cm:.1f}cm)"

    step_cm = min(max(12.0, float(route_step_cm) * 0.45), POINTCLOUD_FORWARD_PROBE_STEP_CM)
    if front_min < POINTCLOUD_FORWARD_PROBE_CREEP_FRONT_CM:
        step_cm = min(step_cm, max(8.0, min(POINTCLOUD_FORWARD_CREEP_STEP_CM, front_min - 82.0)))
    final_alignment_distance_cm = max(220.0, float(reach_tol_cm) + POINTCLOUD_FINAL_ALIGNMENT_MARGIN_CM)
    far_from_goal = float(distance_to_goal_cm) > final_alignment_distance_cm
    correction_cm = 0.0 if far_from_goal else min(max(10.0, float(side_correction_cm) * 0.6), 22.0)
    payload, line_target, line_reason = straight_line_payload(
        current_pose,
        start,
        goal,
        route_step_cm=step_cm,
        side_correction_cm=correction_cm,
        vertical_step_cm=0.0,
    )
    payload["forward_cm"] = max(12.0, min(step_cm, as_float(payload.get("forward_cm")) or step_cm))
    if far_from_goal:
        payload["right_cm"] = 0.0
    payload["up_cm"] = 0.0
    payload["action_name"] = "forward_probe"
    approach_mode = "W-first goal approach" if far_from_goal else "near-goal A/D alignment enabled"
    reason = (
        f"forward probe after avoidance offset: front={front_min:.1f}cm "
        f"distance_to_goal={float(distance_to_goal_cm):.1f}cm "
        f"geometry={geometry} route_offset={route_offset_cm:.1f}cm "
        f"vertical_offset={vertical_offset_cm:.1f}cm mode={approach_mode}; {line_reason}; "
        f"line_target={line_target}"
    )
    return payload, reason


def select_route_action(
    summary: Dict[str, Any],
    scores: Dict[str, float],
    rel: Dict[str, float],
    *,
    method: str = "geometry_rule_v0",
    distance_to_goal_cm: float,
    reach_tol_cm: float,
    last_action: Dict[str, Any],
    current_pose: Dict[str, Any],
    start: Dict[str, Any],
    goal: Dict[str, Any],
    route_step_cm: float,
    side_correction_cm: float,
    vertical_step_cm: float,
    obstacle_hint: str = "",
    environment_id: str = "",
    building_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, float], str, str]:
    reached, completion_reason, _raw_progress, _cross_track_cm = route_completion_state(
        current_pose,
        start,
        goal,
        reach_tol_cm=reach_tol_cm,
    )
    if reached:
        payload = action_payload("hold")
        return "hold", payload, f"reached_goal:{completion_reason}", "REACHED"

    front_min = as_float(summary.get("front_min_depth_cm"))
    forward_swept_clear = bool(summary.get("forward_swept_clear", True))
    danger = is_hard_blocked(summary)
    distance_danger = (front_min > 0.0 and front_min < FORWARD_DANGER_CM) or not forward_swept_clear
    dz = as_float(rel.get("dz_cm"))
    method_id = str(method or "geometry_rule_v0").strip().lower()
    active_building_state: Optional[Dict[str, Any]] = None

    if method_id == "no_avoidance":
        step_cm = effective_route_step_cm(
            distance_to_goal_cm=distance_to_goal_cm,
            reach_tol_cm=reach_tol_cm,
            route_step_cm=route_step_cm,
            front_min_cm=front_min,
            respect_front_caution=False,
        )
        payload, line_target, line_reason = straight_line_payload(
            current_pose,
            start,
            goal,
            route_step_cm=step_cm,
            side_correction_cm=side_correction_cm,
            vertical_step_cm=vertical_step_cm,
        )
        if step_cm < float(route_step_cm):
            line_reason = f"{line_reason}; effective_route_step_cm={step_cm:.1f}"
        return str(payload.get("action_name", "forward")), payload, f"no_avoidance: {line_reason}; line_target={line_target}", "NO_AVOIDANCE"

    if method_id == "pointcloud_direction_rule":
        if not bool(summary.get("available", False)):
            payload = action_payload("hold")
            reason = (
                "pointcloud_direction_rule hold: pointcloud summary unavailable; "
                "cannot safely choose left/right/up/down"
            )
            return "hold", payload, reason, "POINTCLOUD_UNAVAILABLE"
        annotate_semantic_obstacle_context(
            summary,
            obstacle_hint=obstacle_hint,
            environment_id=environment_id,
        )

        state_status = ""
        if building_state is not None:
            active_building_state = update_building_obstacle_state(
                building_state,
                summary,
                current_pose,
                start,
                goal,
                obstacle_hint=obstacle_hint,
                environment_id=environment_id,
            )
            state_status = str(active_building_state.get("status", "")).lower()
            if state_status == "active":
                building_action, building_payload, building_reason = building_vantage_action(
                    summary,
                    current_pose,
                    start,
                    goal,
                    obstacle_hint=obstacle_hint,
                    environment_id=environment_id,
                    vertical_step_cm=vertical_step_cm,
                )
                if building_action is not None and building_payload is not None:
                    state_reason = str(active_building_state.get("state_reason", ""))
                    reason = (
                        "pointcloud_direction_rule building state active before goal alignment: "
                        f"{state_reason}; {building_reason}; selected={building_action}"
                    )
                    phase = "BUILDING_CRUISE" if building_action == "forward" else "BUILDING_VANTAGE"
                    return building_action, building_payload, reason, phase

                action, payload, cruise_reason, phase = building_cruise_forward_action(
                    current_pose,
                    start,
                    goal,
                    distance_to_goal_cm=distance_to_goal_cm,
                    reach_tol_cm=reach_tol_cm,
                    route_step_cm=route_step_cm,
                    front_min_cm=front_min,
                    reason_prefix=(
                        "pointcloud_direction_rule building state active cruise to clear obstacle: "
                        f"{active_building_state.get('state_reason', '')}"
                    ),
                )
                return action, payload, cruise_reason, phase

        if building_state is None:
            building_action, building_payload, building_reason = building_vantage_action(
                summary,
                current_pose,
                start,
                goal,
                obstacle_hint=obstacle_hint,
                environment_id=environment_id,
                vertical_step_cm=vertical_step_cm,
            )
            if building_action is not None and building_payload is not None:
                reason = (
                    "pointcloud_direction_rule building gate before goal alignment: "
                    f"{building_reason}; selected={building_action}"
                )
                phase = "BUILDING_CRUISE" if building_action == "forward" else "BUILDING_VANTAGE"
                return building_action, building_payload, reason, phase
            if is_building_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
                cruise_ready = front_min <= 0.0 or front_min >= BUILDING_CRUISE_MIN_FRONT_CM
                if cruise_ready:
                    action, payload, cruise_reason, phase = building_cruise_forward_action(
                        current_pose,
                        start,
                        goal,
                        distance_to_goal_cm=distance_to_goal_cm,
                        reach_tol_cm=reach_tol_cm,
                        route_step_cm=route_step_cm,
                        front_min_cm=front_min,
                        reason_prefix=(
                            "pointcloud_direction_rule building cruise after vantage: "
                            f"{building_reason}"
                        ),
                    )
                    return action, payload, cruise_reason, phase

        if dz < -Z_ALIGN_TOL_CM and is_building_context(obstacle_hint=obstacle_hint, environment_id=environment_id):
            building_down_allowed, building_down_reason = building_descent_allowed_after_clear(
                building_state,
                current_pose,
                start,
                goal,
                distance_to_goal_cm=distance_to_goal_cm,
                reach_tol_cm=reach_tol_cm,
                obstacle_hint=obstacle_hint,
                environment_id=environment_id,
            )
            if not building_down_allowed:
                return post_building_horizontal_approach_action(
                    current_pose,
                    start,
                    goal,
                    distance_to_goal_cm=distance_to_goal_cm,
                    reach_tol_cm=reach_tol_cm,
                    route_step_cm=route_step_cm,
                    front_min_cm=front_min,
                    gate_reason=building_down_reason,
                )

        fence_context = is_fence_or_rail_context(obstacle_hint=obstacle_hint, environment_id=environment_id)
        fence_xy_needed, fence_xy_distance, _fence_dz = fence_xy_alignment_needed(current_pose, goal)
        if fence_context and fence_xy_needed:
            action, payload, xy_reason, phase = fence_xy_alignment_action(
                current_pose,
                goal,
                route_step_cm=route_step_cm,
                side_correction_cm=side_correction_cm,
                reason_prefix=(
                    "pointcloud_direction_rule fence/rail close to target XY but not within "
                    f"{FENCE_FINAL_XY_TOL_CM:.1f}cm (xy={fence_xy_distance:.1f}cm)"
                ),
            )
            return action, payload, xy_reason, phase

        fence_down_allowed_before_overfly, fence_down_reason_before_overfly = fence_descent_allowed(
            summary,
            current_pose,
            start,
            goal,
            distance_to_goal_cm=distance_to_goal_cm,
            reach_tol_cm=reach_tol_cm,
            obstacle_hint=obstacle_hint,
            environment_id=environment_id,
        )
        if fence_context and dz < -Z_ALIGN_TOL_CM and fence_down_allowed_before_overfly:
            align_action, align_payload, align_reason = goal_alignment_action(
                summary,
                current_pose,
                start,
                goal,
                distance_to_goal_cm=distance_to_goal_cm,
                reach_tol_cm=reach_tol_cm,
                side_correction_cm=side_correction_cm,
                vertical_step_cm=vertical_step_cm,
                allow_controlled_fence_descent=True,
            )
            if align_action is not None and align_payload is not None:
                return (
                    align_action,
                    align_payload,
                    (
                        "pointcloud_direction_rule fence/rail target XY reached; "
                        f"prefer descent before re-triggering overfly: {fence_down_reason_before_overfly}; "
                        f"{align_reason}"
                    ),
                    "GOAL_ALIGNMENT",
                )
            if bool(summary.get("down_swept_clear", False)):
                payload = action_payload("down")
                payload["up_cm"] = -min(max(8.0, abs(dz)), float(vertical_step_cm))
                return (
                    "down",
                    payload,
                    (
                        "pointcloud_direction_rule fence/rail target XY reached; "
                        f"fallback descent before re-triggering overfly: {fence_down_reason_before_overfly}"
                    ),
                    "GOAL_ALIGNMENT",
                )

        fence_action, fence_payload, fence_reason = fence_overfly_action(
            summary,
            current_pose,
            start,
            goal,
            obstacle_hint=obstacle_hint,
            environment_id=environment_id,
            vertical_step_cm=vertical_step_cm,
        )
        if fence_action is not None and fence_payload is not None:
            return (
                fence_action,
                fence_payload,
                f"pointcloud_direction_rule fence/rail overfly before goal alignment: {fence_reason}",
                "FENCE_OVERFLY",
            )

        fence_down_allowed, fence_down_reason = fence_descent_allowed(
            summary,
            current_pose,
            start,
            goal,
            distance_to_goal_cm=distance_to_goal_cm,
            reach_tol_cm=reach_tol_cm,
            obstacle_hint=obstacle_hint,
            environment_id=environment_id,
        )
        if dz < -Z_ALIGN_TOL_CM and not fence_down_allowed:
            return post_fence_horizontal_approach_action(
                current_pose,
                start,
                goal,
                distance_to_goal_cm=distance_to_goal_cm,
                reach_tol_cm=reach_tol_cm,
                route_step_cm=route_step_cm,
                front_min_cm=front_min,
                gate_reason=fence_down_reason,
            )

        align_action, align_payload, align_reason = goal_alignment_action(
            summary,
            current_pose,
            start,
            goal,
            distance_to_goal_cm=distance_to_goal_cm,
            reach_tol_cm=reach_tol_cm,
            side_correction_cm=side_correction_cm,
            vertical_step_cm=vertical_step_cm,
        )
        if align_action is not None and align_payload is not None:
            return (
                align_action,
                align_payload,
                f"pointcloud_direction_rule near-goal alignment: {align_reason}",
                "GOAL_ALIGNMENT",
            )

        skip_recovery_after_building_clear = bool(
            building_state is not None and state_status == "cleared" and not distance_danger
        )
        expand, trigger_reason = pointcloud_direction_rule_should_expand(summary)
        if expand and not skip_recovery_after_building_clear:
            if building_state is None:
                building_action, building_payload, building_reason = building_vantage_action(
                    summary,
                    current_pose,
                    start,
                    goal,
                    obstacle_hint=obstacle_hint,
                    environment_id=environment_id,
                    vertical_step_cm=vertical_step_cm,
                )
                if building_action is not None and building_payload is not None:
                    reason = (
                        f"pointcloud_direction_rule trigger={trigger_reason}; "
                        f"{building_reason}; selected={building_action}"
                    )
                    phase = "BUILDING_CRUISE" if building_action == "forward" else "BUILDING_VANTAGE"
                    return building_action, building_payload, reason, phase
            tree_action, tree_payload, tree_reason = tree_or_pole_recovery_action(
                summary,
                rel,
                last_action,
                current_pose,
                start,
                goal,
                obstacle_hint=obstacle_hint,
                environment_id=environment_id,
                vertical_step_cm=vertical_step_cm,
            )
            if tree_action is not None and tree_payload is not None:
                reason = (
                    f"pointcloud_direction_rule trigger={trigger_reason}; "
                    f"{tree_reason}; selected={tree_action}"
                )
                return tree_action, tree_payload, reason, "TREE_POLE_LATERAL"
            probe_payload, probe_reason = pointcloud_direction_rule_forward_probe(
                summary,
                current_pose,
                start,
                goal,
                obstacle_hint=obstacle_hint,
                environment_id=environment_id,
                distance_to_goal_cm=distance_to_goal_cm,
                reach_tol_cm=reach_tol_cm,
                route_step_cm=route_step_cm,
                side_correction_cm=side_correction_cm,
            )
            if probe_payload is not None:
                reason = (
                    f"pointcloud_direction_rule trigger={trigger_reason}; "
                    f"{probe_reason}; selected=forward_probe"
                )
                return "forward_probe", probe_payload, reason, "POINTCLOUD_FORWARD_PROBE"
            action, detail = pointcloud_direction_rule_action(
                summary,
                rel,
                last_action,
                obstacle_hint=obstacle_hint,
                environment_id=environment_id,
            )
            payload = action_payload(action)
            reason = (
                f"pointcloud_direction_rule trigger={trigger_reason}; "
                f"front={front_min:.1f}cm geometry={summary.get('obstacle_geometry')}; "
                f"left={summary.get('left_min_depth_cm')} right={summary.get('right_min_depth_cm')} "
                f"up={summary.get('up_min_depth_cm')} down={summary.get('down_min_depth_cm')}; "
                f"{detail}; selected={action}"
            )
            return action, payload, reason, "POINTCLOUD_DIRECTION_RECOVERY"

    if method_id == "route_follow" and distance_danger:
        payload = action_payload("hold")
        reason = f"route_follow safety hold: front={front_min:.1f}cm forward_swept_clear={forward_swept_clear}"
        return "hold", payload, reason, "ROUTE_FOLLOW_HOLD"

    if method_id == "distance_rule" and distance_danger:
        action = distance_rule_recovery_action(summary, rel, last_action)
        payload = action_payload(action)
        reason = (
            f"distance_rule danger front={front_min:.1f}cm; "
            f"left={summary.get('left_min_depth_cm')} right={summary.get('right_min_depth_cm')} "
            f"up={summary.get('up_min_depth_cm')}; selected={action}"
        )
        return action, payload, reason, "DISTANCE_RULE_RECOVERY"

    if danger:
        action = choose_recovery_action(scores, rel, summary, last_action)
        if action == "backoff" and front_min >= 80.0:
            if as_float(scores.get("left")) >= as_float(scores.get("right")) and bool(summary.get("left_swept_clear", True)):
                action = "left"
            elif bool(summary.get("right_swept_clear", True)):
                action = "right"
            elif bool(summary.get("up_swept_clear", True)):
                action = "up"
        payload = action_payload(action)
        reason = (
            f"danger front={front_min:.1f}cm forward_swept_clear={forward_swept_clear}; "
            f"selected best safe recovery action {action}"
        )
        return action, payload, reason, "RECOVERY"

    if dz > Z_ALIGN_TOL_CM and bool(summary.get("up_swept_clear", True)):
        payload = action_payload("up")
        return "up", payload, f"goal is {dz:.1f}cm above current pose and up swept volume is clear", "ROUTE_FOLLOW"

    front_clear_for_down = front_min <= 0.0 or front_min >= 650.0
    descent_allowed, descent_gate_reason = building_descent_allowed_after_clear(
        building_state,
        current_pose,
        start,
        goal,
        distance_to_goal_cm=distance_to_goal_cm,
        reach_tol_cm=reach_tol_cm,
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    if dz < -Z_ALIGN_TOL_CM and not descent_allowed:
        return post_building_horizontal_approach_action(
            current_pose,
            start,
            goal,
            distance_to_goal_cm=distance_to_goal_cm,
            reach_tol_cm=reach_tol_cm,
            route_step_cm=route_step_cm,
            front_min_cm=front_min,
            gate_reason=descent_gate_reason,
        )

    fence_down_allowed, fence_down_reason = fence_descent_allowed(
        summary,
        current_pose,
        start,
        goal,
        distance_to_goal_cm=distance_to_goal_cm,
        reach_tol_cm=reach_tol_cm,
        obstacle_hint=obstacle_hint,
        environment_id=environment_id,
    )
    if dz < -Z_ALIGN_TOL_CM and not fence_down_allowed:
        fence_xy_needed, fence_xy_distance, _fence_dz = fence_xy_alignment_needed(current_pose, goal)
        if is_fence_or_rail_context(obstacle_hint=obstacle_hint, environment_id=environment_id) and fence_xy_needed:
            return fence_xy_alignment_action(
                current_pose,
                goal,
                route_step_cm=route_step_cm,
                side_correction_cm=side_correction_cm,
                reason_prefix=(
                    "fence/rail descent blocked until target XY is within "
                    f"{FENCE_FINAL_XY_TOL_CM:.1f}cm (xy={fence_xy_distance:.1f}cm); {fence_down_reason}"
                ),
            )
        return post_fence_horizontal_approach_action(
            current_pose,
            start,
            goal,
            distance_to_goal_cm=distance_to_goal_cm,
            reach_tol_cm=reach_tol_cm,
            route_step_cm=route_step_cm,
            front_min_cm=front_min,
            gate_reason=fence_down_reason,
        )

    if dz < -Z_ALIGN_TOL_CM and front_clear_for_down and bool(summary.get("down_swept_clear", False)):
        payload = action_payload("down")
        return "down", payload, f"goal is {abs(dz):.1f}cm below current pose and down swept volume is clear", "ROUTE_FOLLOW"

    step_cm = effective_route_step_cm(
        distance_to_goal_cm=distance_to_goal_cm,
        reach_tol_cm=reach_tol_cm,
        route_step_cm=route_step_cm,
        front_min_cm=front_min,
        respect_front_caution=True,
    )

    payload, line_target, line_reason = straight_line_payload(
        current_pose,
        start,
        goal,
        route_step_cm=step_cm,
        side_correction_cm=side_correction_cm,
        vertical_step_cm=vertical_step_cm,
    )
    if step_cm < float(route_step_cm):
        line_reason = f"{line_reason}; effective_route_step_cm={step_cm:.1f}"
    return str(payload.get("action_name", "forward")), payload, f"{line_reason}; line_target={line_target}", "ROUTE_FOLLOW"


def risk_state_from_summary(summary: Dict[str, Any], mission_phase: str) -> str:
    if mission_phase == "REACHED":
        return "SAFE"
    if mission_phase == "GOAL_ALIGNMENT":
        front_min = as_float(summary.get("front_min_depth_cm"))
        return "CAUTION" if front_min > 0.0 and front_min < SLOW_FORWARD_CAUTION_CM else "SAFE"
    if mission_phase == "BUILDING_VANTAGE":
        return "CAUTION"
    if mission_phase == "BUILDING_CRUISE":
        return "CAUTION"
    if mission_phase == "FENCE_OVERFLY":
        return "CAUTION"
    if mission_phase == "POST_FENCE_APPROACH":
        return "CAUTION"
    if mission_phase == "FENCE_XY_ALIGN":
        return "CAUTION"
    if mission_phase == "TREE_POLE_LATERAL":
        return "CAUTION"
    if mission_phase == "POINTCLOUD_FORWARD_PROBE":
        return "CAUTION"
    front_min = as_float(summary.get("front_min_depth_cm"))
    if is_hard_blocked(summary):
        return "BLOCKED"
    if front_min > 0.0 and front_min < SLOW_FORWARD_CAUTION_CM:
        return "CAUTION"
    return "SAFE"


def build_route_event(
    result: Dict[str, Any],
    *,
    session_dir: Path,
    frame_id: int,
    args: argparse.Namespace,
    episode: Dict[str, Any],
    episode_index: int,
    start: Dict[str, float],
    goal: Dict[str, float],
    last_action: Dict[str, Any],
) -> Dict[str, Any]:
    pose = result.get("pose", {}) if isinstance(result.get("pose"), dict) else {}
    rel = relative_target(pose, goal)
    progress, deviation = route_progress_and_deviation(pose, start, goal)
    raw_progress, cross_track_cm = route_progress_raw_and_deviation(pose, start, goal)
    event = {
        "frame_id": frame_id,
        "timestamp": result.get("capture_time", datetime.now().isoformat(timespec="milliseconds")),
        "session_id": session_dir.name,
        "episode_id": episode["episode_id"],
        "episode_index": episode_index,
        "collection_stage": args.stage,
        "scenario_id": episode.get("scenario_id") or f"route_episode_{episode['episode_id']}",
        "environment_id": episode.get("environment_id", ""),
        "method": args.method,
        "plan_method": episode.get("method", ""),
        "obstacle_hint": episode.get("obstacle_hint", ""),
        "run_id": args.run_id,
        "mission_phase": "ROUTE_EPISODE",
        "current_pose": pose,
        "pose": pose,
        "start_pose": start,
        "goal_pose": goal,
        "target_waypoint": goal,
        "relative_target": rel,
        "distance_to_goal_cm": round(distance_3d_cm(pose, goal), 3),
        "bearing_deg_body": round(as_float(rel.get("bearing_deg_body")), 3),
        "route_progress": round(progress, 5),
        "raw_route_progress": round(raw_progress, 5),
        "path_deviation_cm": round(deviation, 3),
        "route_cross_track_cm": round(cross_track_cm, 3),
        "rgb_path": result.get("rgb_path", ""),
        "depth_npy_path": result.get("depth_npy_path", ""),
        "depth_cm_path": result.get("depth_cm_path", ""),
        "pointcloud_path": result.get("point_cloud_world_standard_m_npy_path", ""),
        "point_cloud_world_standard_m_npy_path": result.get("point_cloud_world_standard_m_npy_path", ""),
        "point_cloud_image_aligned_standard_m_npy_path": result.get("point_cloud_image_aligned_standard_m_npy_path", ""),
        "point_cloud_pixels_aligned_standard_m_npy_path": result.get("point_cloud_pixels_aligned_standard_m_npy_path", ""),
        "rgb_depth_alignment_preview_path": result.get("rgb_depth_alignment_preview_path", ""),
        "pointcloud_alignment": result.get("pointcloud_alignment", {}),
        "capture_dir": result.get("capture_dir", ""),
        "pose_json_path": result.get("pose_json_path", ""),
        "action_json_path": result.get("action_json_path", ""),
        "depth_summary": result.get("depth_summary", {}),
        "depth_obstacle_summary": result.get("depth_obstacle_summary", {}),
        "minimal_capture": bool(result.get("minimal_capture", False)),
        "source_mode": result.get("source_mode", ""),
        "last_action": last_action,
        "obstacle_geometry_label": args.geometry_label,
        "operator_note": episode.get("operator_note") or args.note,
        "collision_state": False,
        "collision_source": "route_episode_collect",
        "avoidance_failed": False,
        "movement_mode": result.get("movement_mode", ""),
        "movement_enabled": bool(result.get("movement_enabled", False)),
        "point_count": int(result.get("point_count", 0) or 0),
        "coordinate_frame": result.get("coordinate_frame", ""),
        "coordinate_units": result.get("coordinate_units", ""),
    }
    base_dir = Path(str(result.get("capture_dir", ".") or ".")).parent
    summary = summarize_geometry_v0(event, base_dir=base_dir)
    event["pointcloud_summary"] = summary
    scores = score_candidate_actions(summary, rel, last_action)
    if is_soft_low_obstacle(summary):
        scores["forward"] = max(as_float(scores.get("forward")), 0.12)
        scores["slow_forward"] = max(as_float(scores.get("slow_forward")), 0.35)
    elif not bool(summary.get("forward_swept_clear", True)):
        scores["forward"] = 0.0
        scores["slow_forward"] = 0.0
    scores["down"] = 0.0
    event["candidate_action_scores"] = scores
    action, score = best_candidate_action(scores)
    event["v0_selected_action"] = {"action": action, "score": score}
    event["selected_action_reason"] = selected_action_reason(summary, scores)
    event["oscillation_risk"] = "LOW"
    return event


def route_episode_qa_report(session_dir: Path, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_rgb = 0
    missing_depth = 0
    missing_pointcloud = 0
    missing_pose = 0
    missing_action = 0
    hard_forward = 0
    controlled_forward_probe = 0
    unsafe_down = 0
    collision_count = 0
    impact_count = 0
    for event in events:
        if not Path(str(event.get("rgb_path", ""))).exists():
            missing_rgb += 1
        depth_path = Path(str(event.get("depth_npy_path", "") or event.get("depth_cm_path", "")))
        if not depth_path.exists():
            missing_depth += 1
        pointcloud_text = str(
            event.get("pointcloud_path", "")
            or event.get("point_cloud_world_standard_m_npy_path", "")
            or event.get("point_cloud_world_npy_path", "")
        ).strip()
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        minimal_pointcloud_ok = bool(event.get("minimal_capture", False)) or str(summary.get("source_mode", "")).lower() == "depth_sector_minimal"
        if not minimal_pointcloud_ok and (not pointcloud_text or not Path(pointcloud_text).is_file()):
            missing_pointcloud += 1
        if not isinstance(event.get("current_pose"), dict) or not event.get("current_pose"):
            missing_pose += 1
        action = event.get("executed_action")
        if not isinstance(action, dict):
            missing_action += 1
            action = {}
        action_name = str(action.get("action_name") or event.get("selected_action", "")).lower()
        if is_hard_blocked(summary) and as_float(action.get("forward_cm")) > 1.0 and action_name == "forward_probe":
            controlled_forward_probe += 1
        elif is_hard_blocked(summary) and as_float(action.get("forward_cm")) > 1.0:
            hard_forward += 1
        pose = event.get("current_pose") if isinstance(event.get("current_pose"), dict) else {}
        pose_z = as_float(pose.get("z"))
        if as_float(action.get("up_cm")) < -1.0 and (not summary.get("down_swept_clear", False) or pose_z <= 100.0):
            unsafe_down += 1
        if bool(event.get("collision_state", event.get("collision", False))):
            collision_count += 1
        if bool(event.get("impact_state", False)):
            impact_count += 1
    report = {
        "status": "ok",
        "session_dir": str(session_dir),
        "event_count": len(events),
        "valid_frame_count": max(0, len(events) - max(missing_rgb, missing_pointcloud, missing_pose, missing_action)),
        "missing_rgb_count": missing_rgb,
        "missing_depth_count": missing_depth,
        "missing_pointcloud_count": missing_pointcloud,
        "missing_pose_count": missing_pose,
        "missing_action_count": missing_action,
        "hard_blocked_forward_violation_count": hard_forward,
        "danger_forward_violation_count": hard_forward,
        "controlled_forward_probe_count": controlled_forward_probe,
        "unsafe_down_action_count": unsafe_down,
        "collision_count": collision_count,
        "impact_count": impact_count,
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    write_json(session_dir / "collection_quality_report.json", report)
    return report


def write_partial_episode_summary(
    session_dir: Path,
    events: List[Dict[str, Any]],
    *,
    start: Dict[str, Any],
    goal: Dict[str, Any],
    outcome: str,
    reach_tol_cm: float = 180.0,
) -> Dict[str, Any]:
    final_pose = events[-1].get("post_action_pose") or events[-1].get("current_pose") if events else start
    if not isinstance(final_pose, dict):
        final_pose = start
    final_reached, final_completion_reason, final_raw_progress, final_cross_track_cm = route_completion_state(
        final_pose,
        start,
        goal,
        reach_tol_cm=float(reach_tol_cm),
    )
    action_counts = Counter(str(event.get("selected_action", event.get("expert_action", ""))) for event in events)
    last_event = events[-1] if events else {}
    summary = {
        "status": "partial",
        "session_dir": str(session_dir),
        "episode_id": last_event.get("episode_id", "") if isinstance(last_event, dict) else "",
        "outcome_so_far": outcome,
        "frame_count": len(events),
        "start_pose": start,
        "goal_pose": goal,
        "latest_pose": final_pose,
        "latest_distance_to_goal_cm": round(distance_3d_cm(final_pose, goal), 3),
        "reach_tol_cm": float(reach_tol_cm),
        "goal_reached": bool(final_reached),
        "goal_status": "reached" if final_reached else "not_reached",
        "goal_completion_reason": final_completion_reason,
        "latest_raw_route_progress": round(final_raw_progress, 5),
        "latest_route_cross_track_cm": round(final_cross_track_cm, 3),
        "latest_action": last_event.get("selected_action", "") if isinstance(last_event, dict) else "",
        "latest_phase": last_event.get("mission_phase", "") if isinstance(last_event, dict) else "",
        "latest_risk_state": last_event.get("risk_state", "") if isinstance(last_event, dict) else "",
        "latest_reason": last_event.get("selected_action_reason", "") if isinstance(last_event, dict) else "",
        "action_counts": dict(sorted(action_counts.items())),
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    write_json(session_dir / "partial_episode_summary.json", summary)
    return summary


def summarize_episode(
    session_dir: Path,
    events: List[Dict[str, Any]],
    *,
    start: Dict[str, Any],
    goal: Dict[str, Any],
    outcome: str,
    reach_tol_cm: float = 180.0,
) -> Dict[str, Any]:
    final_pose = events[-1].get("post_action_pose") or events[-1].get("current_pose") if events else start
    if not isinstance(final_pose, dict):
        final_pose = start
    final_reached, final_completion_reason, final_raw_progress, final_cross_track_cm = route_completion_state(
        final_pose,
        start,
        goal,
        reach_tol_cm=float(reach_tol_cm),
    )
    action_counts = Counter(str(event.get("selected_action", event.get("expert_action", ""))) for event in events)
    geometry_counts = Counter(
        str((event.get("pointcloud_summary") or {}).get("obstacle_geometry", "unknown"))
        for event in events
        if isinstance(event.get("pointcloud_summary"), dict)
    )
    quality = route_episode_qa_report(session_dir, events)
    summary = {
        "session_dir": str(session_dir),
        "episode_id": events[0].get("episode_id", "") if events else "",
        "outcome": outcome,
        "frame_count": len(events),
        "valid_frame_count": quality.get("valid_frame_count", 0),
        "start_pose": start,
        "goal_pose": goal,
        "final_pose": final_pose,
        "final_distance_to_goal_cm": round(distance_3d_cm(final_pose, goal), 3),
        "reach_tol_cm": float(reach_tol_cm),
        "goal_reached": bool(final_reached),
        "goal_status": "reached" if final_reached else "not_reached",
        "final_raw_route_progress": round(final_raw_progress, 5),
        "final_route_cross_track_cm": round(final_cross_track_cm, 3),
        "goal_completion_reason": events[-1].get("goal_completion_reason", final_completion_reason) if events else final_completion_reason,
        "approach_failure_reason": ""
        if final_reached
        else f"{outcome}_before_goal: final_distance={distance_3d_cm(final_pose, goal):.1f}cm reach_tol={float(reach_tol_cm):.1f}cm",
        "collision_count": quality.get("collision_count", 0),
        "impact_count": quality.get("impact_count", 0),
        "had_collision": bool(quality.get("collision_count", 0)),
        "had_impact": bool(quality.get("impact_count", 0)),
        "action_counts": dict(sorted(action_counts.items())),
        "geometry_counts": dict(sorted(geometry_counts.items())),
        "danger_forward_violation_count": quality.get("danger_forward_violation_count", 0),
        "unsafe_down_action_count": quality.get("unsafe_down_action_count", 0),
        "quality_report_path": str(session_dir / "collection_quality_report.json"),
        "finished_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    write_json(session_dir / "episode_summary.json", summary)
    write_json(session_dir / "avoidance_session_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Route6_entrance_search-episode obstacle avoidance data from start/goal pairs.")
    parser.add_argument("--episodes-json", default="", help="Optional JSON list of episodes with start_pose and goal_pose.")
    parser.add_argument("--episode-ids", default="", help="Comma-separated episode ids to run, for example E03,E04.")
    parser.add_argument("--max-episodes", type=int, default=0, help="Limit episodes for smoke tests; 0 runs all.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved episodes without starting UE.")
    parser.add_argument("--data-root", default="obstacle_avoidance_data")
    parser.add_argument("--stage", default="route_episode")
    parser.add_argument("--method", default="geometry_rule_v0")
    parser.add_argument("--run-id", default="route_episode")
    parser.add_argument("--note", default="Route6_entrance_search episode collection")
    parser.add_argument("--geometry-label", default="unknown")
    parser.add_argument("--env-platform", default="win", choices=["auto", "win", "mac", "linux"])
    parser.add_argument("--launch-sleep", type=int, default=5)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--reach-tol-cm", type=float, default=180.0)
    parser.add_argument("--max-ticks-per-episode", type=int, default=220)
    parser.add_argument("--Route6_entrance_search-step-cm", type=float, default=DEFAULT_ROUTE_STEP_CM)
    parser.add_argument("--side-correction-cm", type=float, default=DEFAULT_ROUTE_SIDE_CORRECTION_CM)
    parser.add_argument("--vertical-step-cm", type=float, default=DEFAULT_ROUTE_VERTICAL_STEP_CM)
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--movement-mode", default="physics", choices=["pose_lock", "physics"])
    parser.add_argument("--time-dilation", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = load_episodes(args.episodes_json)
    wanted_ids = {item.strip().upper() for item in str(args.episode_ids or "").split(",") if item.strip()}
    if wanted_ids:
        episodes = [episode for episode in episodes if str(episode.get("episode_id", "")).upper() in wanted_ids]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    resolved = []
    for episode in episodes:
        item = {
            "episode_id": episode["episode_id"],
            "start_pose": pose_dict(episode["start_pose"]),
            "goal_pose": pose_dict(episode["goal_pose"]),
        }
        for key in ("scenario_id", "environment_id", "method", "obstacle_hint", "operator_note", "enabled"):
            if key in episode:
                item[key] = episode.get(key)
        resolved.append(item)
    if args.dry_run:
        print(json.dumps({"count": len(resolved), "episodes": resolved}, indent=2, ensure_ascii=False))
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage = sanitize(args.stage, "route_episode")
    method = sanitize(args.method, "geometry_rule_v0")
    run_id = sanitize(args.run_id, "route_episode")
    data_root = Path(args.data_root).resolve()
    batch_dir = data_root / "route_episode_batches" / f"{timestamp}_{stage}_{method}_{run_id}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    write_json(batch_dir / "route_episode_batch_config.json", {"args": vars(args), "episodes": resolved, "started_at": datetime.now().isoformat(timespec="milliseconds")})

    session_args = flight.default_session_args(
        env_platform=args.env_platform,
        output_dir=str(batch_dir / "run"),
        launch_sleep=args.launch_sleep,
        width=args.width,
        height=args.height,
        time_dilation=args.time_dilation,
        step_delay=0.05,
        save_every=0,
        lidar_capture_processing="minimal",
        movement_mode=args.movement_mode,
        force_kill_unreal_on_stop=True,
        log_level=args.log_level,
    )
    session = flight.DroneFlightSession(session_args)
    batch_summaries: List[Dict[str, Any]] = []
    try:
        start_result = session.start()
        write_json(batch_dir / "start_result.json", start_result if isinstance(start_result, dict) else {"result": str(start_result)})
        session.set_movement_mode(args.movement_mode)
        session.set_movement_enabled(True)

        for episode_index, episode in enumerate(resolved, start=1):
            episode_id = str(episode["episode_id"])
            start = dict(episode["start_pose"])
            goal = dict(episode["goal_pose"])
            session_dir = data_root / "sessions" / f"{timestamp}_{stage}_{episode_id}_{method}_{run_id}"
            session_dir.mkdir(parents=True, exist_ok=False)
            events_path = session_dir / "avoidance_events.jsonl"
            write_json(session_dir / "episode_config.json", {"episode_index": episode_index, "episode_id": episode_id, "start_pose": start, "goal_pose": goal, "args": vars(args)})

            print(f"EPISODE_START {episode_id} start={start} goal={goal}", flush=True)
            session.set_pose({"x": start["x"], "y": start["y"], "z": start["z"], "yaw": start["yaw"]})
            last_action = action_payload("hold")
            building_obstacle_state = make_building_obstacle_state()
            events: List[Dict[str, Any]] = []
            outcome = "timeout"

            for frame_id in range(1, max(1, args.max_ticks_per_episode) + 1):
                pre_state = session.get_state()
                pre_pose = pose_from_state(pre_state)
                pre_distance = distance_3d_cm(pre_pose, goal)
                action_detail = {
                    "source": "obstacle_avoidance_route_episode_collect",
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "collection_stage": args.stage,
                    "scenario_id": episode.get("scenario_id") or f"route_episode_{episode_id}",
                    "environment_id": episode.get("environment_id", ""),
                    "method": args.method,
                    "plan_method": episode.get("method", ""),
                    "obstacle_hint": episode.get("obstacle_hint", ""),
                    "run_id": args.run_id,
                    "mission_phase": "ROUTE_EPISODE",
                    "risk_state": "SAFE",
                    "expert_action": str(last_action.get("action_name", "hold")),
                    "expert_action_payload": last_action,
                    "start_pose": start,
                    "goal_pose": goal,
                    "target_waypoint": goal,
                    "operator_note": episode.get("operator_note") or args.note,
                    "building_obstacle_state": dict(building_obstacle_state),
                }
                result = session.capture_lidar_stream_frame(session_dir, frame_id, action_detail=action_detail)
                event = build_route_event(
                    result,
                    session_dir=session_dir,
                    frame_id=frame_id,
                    args=args,
                    episode=episode,
                    episode_index=episode_index,
                    start=start,
                    goal=goal,
                    last_action=last_action,
                )
                rel = event.get("relative_target") if isinstance(event.get("relative_target"), dict) else {}
                selected_action, payload, reason, phase = select_route_action(
                    event["pointcloud_summary"],
                    event["candidate_action_scores"],
                    rel,
                    method=args.method,
                    distance_to_goal_cm=pre_distance,
                    reach_tol_cm=float(args.reach_tol_cm),
                    last_action=last_action,
                    current_pose=pre_pose,
                    start=start,
                    goal=goal,
                    route_step_cm=float(args.route_step_cm),
                    side_correction_cm=float(args.side_correction_cm),
                    vertical_step_cm=float(args.vertical_step_cm),
                    obstacle_hint=str(episode.get("obstacle_hint", "")),
                    environment_id=str(episode.get("environment_id", "")),
                    building_state=building_obstacle_state,
                )
                risk_state = risk_state_from_summary(event["pointcloud_summary"], phase)
                event.update(
                    {
                        "mission_phase": phase,
                        "risk_state": risk_state,
                        "selected_action": selected_action,
                        "selected_action_payload": payload,
                        "selected_action_reason": reason,
                        "expert_action": str(payload.get("action_name", selected_action)),
                        "expert_action_payload": payload,
                        "nominal_action": payload,
                        "agent_action": payload,
                        "executed_action": payload,
                        "shield_state": "GOAL_REACHED" if phase == "REACHED" else "V0_SHIELD_APPLIED",
                        "episode_outcome": "running",
                        "building_obstacle_state": dict(building_obstacle_state),
                        "front_building_obstacle": bool(building_obstacle_state.get("front_building_obstacle", False)),
                        "current_front_building_obstacle": bool(building_obstacle_state.get("current_front_building_obstacle", False)),
                        "building_obstacle_cleared": bool(building_obstacle_state.get("cleared", False)),
                    }
                )

                if phase == "REACHED":
                    post_pose = pre_pose
                    post_distance = pre_distance
                else:
                    response = session.move_relative(payload)
                    post_pose = pose_from_state(response if isinstance(response, dict) else session.get_state())
                    post_distance = distance_3d_cm(post_pose, goal)

                post_progress, post_deviation = route_progress_and_deviation(post_pose, start, goal)
                post_reached, completion_reason, post_raw_progress, post_cross_track_cm = route_completion_state(
                    post_pose,
                    start,
                    goal,
                    reach_tol_cm=float(args.reach_tol_cm),
                )
                event["goal_completion_reason"] = completion_reason
                event["goal_passed"] = bool(post_raw_progress >= 1.0)
                event["goal_reached"] = bool(post_reached)
                event["goal_status"] = "reached" if post_reached else "not_reached"
                event["post_action_pose"] = post_pose
                event["post_distance_to_goal_cm"] = round(post_distance, 3)
                event["post_route_progress"] = round(post_progress, 5)
                event["post_raw_route_progress"] = round(post_raw_progress, 5)
                event["post_path_deviation_cm"] = round(post_deviation, 3)
                event["post_route_cross_track_cm"] = round(post_cross_track_cm, 3)
                annotate_collision_state(event, pre_pose=pre_pose, post_pose=post_pose, payload=payload)
                if bool(event.get("collision_state")):
                    outcome = "collision"
                    event["episode_outcome"] = "collision"
                    event["risk_state"] = "COLLISION"
                elif post_reached:
                    outcome = "reached"
                    event["episode_outcome"] = "reached"
                elif should_stop_for_repeated_hold([*events, event]):
                    outcome = "stalled_hold"
                    event["episode_outcome"] = "stalled_hold"
                    event["risk_state"] = "STALLED_HOLD"
                    event["selected_action_reason"] = (
                        f"{event.get('selected_action_reason', '')}; repeated hold "
                        f"{REPEATED_HOLD_STOP_COUNT} frames, ending episode"
                    )
                append_jsonl(events_path, event)
                events.append(event)
                write_partial_episode_summary(
                    session_dir,
                    events,
                    start=start,
                    goal=goal,
                    outcome=outcome,
                    reach_tol_cm=float(args.reach_tol_cm),
                )
                last_action = payload
                print(
                    f"[{episode_id} {frame_id}/{args.max_ticks_per_episode}] "
                    f"dist={post_distance:.1f}cm action={selected_action} risk={event.get('risk_state', risk_state)} "
                    f"building={building_obstacle_state.get('status', '')} "
                    f"impact={bool(event.get('collision_state'))} "
                    f"geometry={event['pointcloud_summary'].get('obstacle_geometry')} "
                    f"front={event['pointcloud_summary'].get('front_min_depth_cm')}",
                    flush=True,
                )
                if bool(event.get("collision_state")):
                    print(
                        f"COLLISION_ALERT {episode_id} frame={frame_id} source={event.get('collision_source')} "
                        f"detail={event.get('impact_detail')}",
                        flush=True,
                    )

                if outcome in {"reached", "collision", "stalled_hold"}:
                    break
                if frame_id < args.max_ticks_per_episode:
                    time.sleep(max(0.0, args.interval_s))

            summary = summarize_episode(session_dir, events, start=start, goal=goal, outcome=outcome, reach_tol_cm=float(args.reach_tol_cm))
            batch_summaries.append(summary)
            write_json(
                batch_dir / "route_episode_batch_summary.json",
                {
                    "batch_dir": str(batch_dir),
                    "episode_count": len(resolved),
                    "completed_count": len(batch_summaries),
                    "summaries": batch_summaries,
                    "updated_at": datetime.now().isoformat(timespec="milliseconds"),
                },
            )
            print(f"EPISODE_DONE {episode_id} outcome={outcome} final_distance={summary['final_distance_to_goal_cm']}", flush=True)
            if outcome != "reached" and not bool(args.continue_on_failure):
                break

        write_json(
            batch_dir / "route_episode_batch_summary.json",
            {
                "batch_dir": str(batch_dir),
                "episode_count": len(resolved),
                "completed_count": len(batch_summaries),
                "reached_count": sum(1 for item in batch_summaries if item.get("outcome") == "reached"),
                "collision_count": sum(1 for item in batch_summaries if item.get("outcome") == "collision" or item.get("had_collision")),
                "summaries": batch_summaries,
                "finished_at": datetime.now().isoformat(timespec="milliseconds"),
            },
        )
        print(f"ROUTE_EPISODE_BATCH_DONE {batch_dir}", flush=True)
    finally:
        session.close(force_kill_unreal=True)


if __name__ == "__main__":
    main()
