from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import run_drone_flight as flight
except Exception:  # pragma: no cover - keeps offline tools usable without Unreal runtime.
    flight = None  # type: ignore[assignment]


GEOMETRY_TYPES = ("none", "vertical_wall", "overhang_beam", "low_obstacle", "thin_structure", "unknown")
CANDIDATE_ACTIONS = (
    "forward",
    "slow_forward",
    "left",
    "right",
    "up",
    "down",
    "backoff",
    "hold",
    "yaw_left",
    "yaw_right",
)

DEFAULT_FRONT_RANGE_CM = 600.0
DEFAULT_SIDE_RANGE_CM = 300.0
DEFAULT_VERTICAL_RANGE_CM = 250.0
DEFAULT_DANGER_DEPTH_CM = 250.0
DEFAULT_CAUTION_DEPTH_CM = 450.0
DEFAULT_CLEAR_DEPTH_CM = 650.0
DEFAULT_MIN_SAFE_GROUND_CLEARANCE_CM = 80.0
DEFAULT_STEP_CM = 20.0


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
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "clear"}
    return False


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resolve_event_path(value: Any, *, base_dir: Optional[Path] = None) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("__missing_obstacle_avoidance_path__")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return path.resolve()


def first_existing_cloud_path(event: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Path:
    for key in ("pointcloud_path", "point_cloud_world_standard_m_npy_path", "point_cloud_world_npy_path"):
        path = resolve_event_path(event.get(key), base_dir=base_dir)
        if str(path) not in {"", "."} and path.is_file():
            return path
    capture_dir = resolve_event_path(event.get("capture_dir"), base_dir=base_dir)
    if str(capture_dir) not in {"", "."} and capture_dir.is_dir():
        for name in ("point_cloud_world_standard_m.npy", "point_cloud_world.npy"):
            candidate = capture_dir / name
            if candidate.exists():
                return candidate
    return Path("__missing_obstacle_avoidance_path__")


def pose_from_event(event: Dict[str, Any]) -> Dict[str, float]:
    pose = event.get("current_pose")
    if not isinstance(pose, dict):
        pose = event.get("pose")
    if not isinstance(pose, dict):
        pose = event.get("actual_pose")
    if not isinstance(pose, dict):
        pose = {}
    return {
        "x": as_float(pose.get("x")),
        "y": as_float(pose.get("y")),
        "z": as_float(pose.get("z")),
        "yaw": as_float(pose.get("yaw", pose.get("task_yaw", pose.get("yaw_deg", 0.0)))),
    }


def load_world_cloud_unreal_cm(path: Path, event: Dict[str, Any]) -> np.ndarray:
    cloud = np.load(path)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    if cloud.shape[0] > 200_000:
        stride = max(1, int(math.ceil(cloud.shape[0] / 200_000)))
        cloud = cloud[::stride]
    xyz = cloud[:, :3].astype(np.float32, copy=False)
    units = str(event.get("coordinate_units", "") or "").lower()
    is_standard_m = "standard_m" in path.name.lower() or units == "m"
    if is_standard_m:
        if flight is not None:
            try:
                converted = flight.standard_world_m_to_unreal_world_cm(cloud[:, :6])
                if converted.ndim == 2 and converted.shape[1] >= 3 and converted.shape[0] > 0:
                    return converted[:, :3].astype(np.float32, copy=False)
            except Exception:
                pass
        return np.column_stack((xyz[:, 0] * 100.0, -xyz[:, 1] * 100.0, xyz[:, 2] * 100.0)).astype(
            np.float32,
            copy=False,
        )
    return xyz.astype(np.float32, copy=False)


def world_to_body_xyz(world_xyz_cm: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
    if world_xyz_cm.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    yaw = math.radians(pose["yaw"])
    dx = world_xyz_cm[:, 0] - pose["x"]
    dy = world_xyz_cm[:, 1] - pose["y"]
    dz = world_xyz_cm[:, 2] - pose["z"]
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    up = dz
    return np.column_stack((forward, right, up)).astype(np.float32, copy=False)


def min_forward(body_xyz: np.ndarray, mask: np.ndarray) -> float:
    if body_xyz.size == 0:
        return 0.0
    values = body_xyz[mask, 0]
    return float(np.min(values)) if values.size else 0.0


def swept_clear(
    body_xyz: np.ndarray,
    *,
    delta_forward_cm: float = 0.0,
    delta_right_cm: float = 0.0,
    delta_up_cm: float = 0.0,
    body_forward_radius_cm: float = 70.0,
    body_side_radius_cm: float = 70.0,
    body_vertical_radius_cm: float = 70.0,
) -> bool:
    if body_xyz.size == 0:
        return True
    f = body_xyz[:, 0]
    r = body_xyz[:, 1]
    u = body_xyz[:, 2]
    f_min = min(0.0, delta_forward_cm) - body_forward_radius_cm
    f_max = max(0.0, delta_forward_cm) + body_forward_radius_cm
    r_min = min(0.0, delta_right_cm) - body_side_radius_cm
    r_max = max(0.0, delta_right_cm) + body_side_radius_cm
    u_min = min(0.0, delta_up_cm) - body_vertical_radius_cm
    u_max = max(0.0, delta_up_cm) + body_vertical_radius_cm
    occupied = (f >= f_min) & (f <= f_max) & (r >= r_min) & (r <= r_max) & (u >= u_min) & (u <= u_max)
    return not bool(np.any(occupied))


def classify_obstacle_geometry(body_xyz: np.ndarray, corridor_mask: np.ndarray, front_min_cm: float) -> Tuple[str, Dict[str, float]]:
    empty = {"obstacle_width_cm": 0.0, "obstacle_height_cm": 0.0, "obstacle_thickness_cm": 0.0}
    if body_xyz.size == 0 or front_min_cm <= 0.0 or front_min_cm >= DEFAULT_CLEAR_DEPTH_CM:
        return "none", empty
    near_mask = corridor_mask & (body_xyz[:, 0] <= min(DEFAULT_FRONT_RANGE_CM, front_min_cm + 140.0))
    points = body_xyz[near_mask]
    if points.size == 0:
        return "unknown", empty

    forward_span = float(np.ptp(points[:, 0])) if points.shape[0] > 1 else 0.0
    width = float(np.ptp(points[:, 1])) if points.shape[0] > 1 else 0.0
    height = float(np.ptp(points[:, 2])) if points.shape[0] > 1 else 0.0
    center_up = float(np.median(points[:, 2]))
    upper_q = float(np.percentile(points[:, 2], 75))
    lower_q = float(np.percentile(points[:, 2], 25))

    dims = {
        "obstacle_width_cm": round(width, 3),
        "obstacle_height_cm": round(height, 3),
        "obstacle_thickness_cm": round(forward_span, 3),
    }
    if points.shape[0] < 24 or min(width, max(height, 1.0)) < 25.0:
        return "thin_structure", dims
    if center_up > 60.0 and width > 100.0 and height < 180.0:
        return "overhang_beam", dims
    if upper_q < 80.0 and lower_q < -120.0:
        return "low_obstacle", dims
    if height > 160.0 and width > 120.0 and forward_span < 220.0:
        return "vertical_wall", dims
    return "unknown", dims


def summarize_geometry_v0(event: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    depth_summary = event.get("depth_obstacle_summary")
    if isinstance(depth_summary, dict) and depth_summary.get("available"):
        return dict(depth_summary)
    pointcloud_summary = event.get("pointcloud_summary")
    if isinstance(pointcloud_summary, dict) and pointcloud_summary.get("source_mode") == "depth_sector_minimal":
        return dict(pointcloud_summary)
    path = first_existing_cloud_path(event, base_dir=base_dir)
    pose = pose_from_event(event)
    base_summary: Dict[str, Any] = {
        "available": False,
        "point_count": 0,
        "front_min_depth_cm": 0.0,
        "front_mean_depth_cm": 0.0,
        "corridor_count": 0,
        "left_min_depth_cm": 0.0,
        "right_min_depth_cm": 0.0,
        "up_min_depth_cm": 0.0,
        "down_min_depth_cm": 0.0,
        "nearest_distance_cm": 0.0,
        "obstacle_geometry": "unknown",
        "obstacle_width_cm": 0.0,
        "obstacle_height_cm": 0.0,
        "obstacle_thickness_cm": 0.0,
        "left_swept_clear": True,
        "right_swept_clear": True,
        "up_swept_clear": True,
        "down_swept_clear": False,
        "forward_swept_clear": True,
        "backoff_swept_clear": True,
        "local_map_age_ms": 0,
    }
    if not path.exists():
        return base_summary
    try:
        world_xyz = load_world_cloud_unreal_cm(path, event)
    except Exception as exc:
        base_summary["error"] = str(exc)
        return base_summary
    if world_xyz.size == 0:
        return base_summary

    body = world_to_body_xyz(world_xyz, pose)
    forward = body[:, 0]
    right = body[:, 1]
    up = body[:, 2]
    front = (forward > 0.0) & (forward <= DEFAULT_FRONT_RANGE_CM)
    vertical = np.abs(up) <= DEFAULT_VERTICAL_RANGE_CM
    corridor = front & (np.abs(right) <= 180.0) & vertical
    left = front & (right < -40.0) & (right >= -DEFAULT_SIDE_RANGE_CM) & vertical
    right_side = front & (right > 40.0) & (right <= DEFAULT_SIDE_RANGE_CM) & vertical
    up_zone = front & (np.abs(right) <= 180.0) & (up > 40.0) & (up <= DEFAULT_SIDE_RANGE_CM)
    down_zone = front & (np.abs(right) <= 180.0) & (up < -40.0) & (up >= -DEFAULT_SIDE_RANGE_CM)
    nearest = np.sqrt(np.sum(body * body, axis=1))
    front_values = forward[corridor]
    front_min = float(np.min(front_values)) if front_values.size else 0.0
    geometry, dims = classify_obstacle_geometry(body, corridor, front_min)
    down_safe_height = pose["z"] > (DEFAULT_MIN_SAFE_GROUND_CLEARANCE_CM + DEFAULT_STEP_CM)

    summary = dict(base_summary)
    summary.update(
        {
            "available": True,
            "point_count": int(world_xyz.shape[0]),
            "front_min_depth_cm": round(front_min, 3),
            "front_mean_depth_cm": round(float(np.mean(front_values)) if front_values.size else 0.0, 3),
            "corridor_count": int(front_values.size),
            "left_min_depth_cm": round(min_forward(body, left), 3),
            "right_min_depth_cm": round(min_forward(body, right_side), 3),
            "up_min_depth_cm": round(min_forward(body, up_zone), 3),
            "down_min_depth_cm": round(min_forward(body, down_zone), 3),
            "nearest_distance_cm": round(float(np.min(nearest)) if nearest.size else 0.0, 3),
            "obstacle_geometry": geometry,
            "left_swept_clear": swept_clear(body, delta_right_cm=-DEFAULT_STEP_CM),
            "right_swept_clear": swept_clear(body, delta_right_cm=DEFAULT_STEP_CM),
            "up_swept_clear": swept_clear(body, delta_up_cm=DEFAULT_STEP_CM),
            "down_swept_clear": bool(down_safe_height and swept_clear(body, delta_up_cm=-DEFAULT_STEP_CM)),
            "forward_swept_clear": bool(front_min <= 0.0 or front_min >= DEFAULT_DANGER_DEPTH_CM),
            "backoff_swept_clear": swept_clear(body, delta_forward_cm=-DEFAULT_STEP_CM),
            "local_map_age_ms": int(as_float(event.get("local_map_age_ms"), 0.0)),
        }
    )
    summary.update(dims)
    label = str(event.get("obstacle_geometry_label", "") or "").strip()
    if label:
        summary["operator_obstacle_geometry_label"] = label
    return summary


def clearance_score(value_cm: Any) -> float:
    value = as_float(value_cm)
    if value <= 0.0:
        return 1.0
    return clamp01(value / DEFAULT_CLEAR_DEPTH_CM)


def score_candidate_actions(
    summary: Dict[str, Any],
    relative_target: Optional[Dict[str, Any]] = None,
    last_action: Optional[Any] = None,
) -> Dict[str, float]:
    relative_target = relative_target if isinstance(relative_target, dict) else {}
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    front_min = as_float(summary.get("front_min_depth_cm"))
    bearing = as_float(relative_target.get("bearing_deg_body"))
    blocked = front_min > 0.0 and front_min < DEFAULT_DANGER_DEPTH_CM
    caution = front_min > 0.0 and front_min < DEFAULT_CAUTION_DEPTH_CM

    if front_min <= 0.0 or front_min >= DEFAULT_CLEAR_DEPTH_CM:
        forward_score = 0.88
        slow_score = 0.65
    elif caution:
        forward_score = 0.12 if blocked else 0.28
        slow_score = 0.08 if blocked else 0.55
    else:
        forward_score = 0.62
        slow_score = 0.7

    left_score = 0.15 + 0.72 * clearance_score(summary.get("left_min_depth_cm"))
    right_score = 0.15 + 0.72 * clearance_score(summary.get("right_min_depth_cm"))
    up_score = 0.12 + 0.72 * clearance_score(summary.get("up_min_depth_cm"))
    down_score = 0.12 + 0.72 * clearance_score(summary.get("down_min_depth_cm"))
    backoff_score = 0.55 if blocked else 0.22
    hold_score = 0.42 if geometry in {"unknown", "thin_structure"} or blocked else 0.14
    yaw_score = 0.36 if geometry in {"unknown", "thin_structure"} else 0.16

    if not as_bool(summary.get("left_swept_clear", True)):
        left_score = 0.0
    if not as_bool(summary.get("right_swept_clear", True)):
        right_score = 0.0
    if not as_bool(summary.get("up_swept_clear", True)):
        up_score = 0.0
    if not as_bool(summary.get("down_swept_clear", False)):
        down_score = 0.0
    if not as_bool(summary.get("forward_swept_clear", True)):
        forward_score = min(forward_score, 0.05)
        slow_score = min(slow_score, 0.08)
    if not as_bool(summary.get("backoff_swept_clear", True)):
        backoff_score = 0.0

    if geometry == "vertical_wall":
        left_score += 0.1
        right_score += 0.1
        forward_score *= 0.35
        slow_score *= 0.45
    elif geometry == "overhang_beam":
        left_score += 0.1
        right_score += 0.1
        up_score *= 0.35
        down_score += 0.08
    elif geometry == "low_obstacle":
        up_score += 0.16
        forward_score *= 0.55
    elif geometry == "none":
        hold_score *= 0.55
        yaw_score *= 0.5

    if bearing > 8.0:
        right_score += min(0.08, bearing / 900.0)
    elif bearing < -8.0:
        left_score += min(0.08, abs(bearing) / 900.0)

    last_name = ""
    if isinstance(last_action, dict):
        last_name = str(last_action.get("action_name", "") or "").lower()
        if not last_name:
            if as_float(last_action.get("right_cm")) > 1.0:
                last_name = "right"
            elif as_float(last_action.get("right_cm")) < -1.0:
                last_name = "left"
            elif as_float(last_action.get("up_cm")) > 1.0:
                last_name = "up"
            elif as_float(last_action.get("up_cm")) < -1.0:
                last_name = "down"
    elif last_action is not None:
        last_name = str(last_action or "").lower()
    if "left" in last_name:
        right_score -= 0.08
    elif "right" in last_name:
        left_score -= 0.08
    elif "up" in last_name:
        down_score -= 0.08
    elif "down" in last_name:
        up_score -= 0.08

    scores = {
        "forward": forward_score,
        "slow_forward": slow_score,
        "left": left_score,
        "right": right_score,
        "up": up_score,
        "down": down_score,
        "backoff": backoff_score,
        "hold": hold_score,
        "yaw_left": yaw_score + (0.03 if bearing < 0 else 0.0),
        "yaw_right": yaw_score + (0.03 if bearing > 0 else 0.0),
    }
    return {key: round(clamp01(value), 4) for key, value in scores.items()}


def best_candidate_action(scores: Dict[str, float]) -> Tuple[str, float]:
    if not scores:
        return "hold", 0.0
    action, score = max(scores.items(), key=lambda item: (float(item[1]), item[0]))
    return str(action), float(score)


def selected_action_reason(summary: Dict[str, Any], scores: Dict[str, float]) -> str:
    action, score = best_candidate_action(scores)
    geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
    front_min = as_float(summary.get("front_min_depth_cm"))
    return (
        f"{action} has the highest v0 candidate score ({score:.2f}); "
        f"geometry={geometry}, front_min_depth_cm={front_min:.1f}"
    )
