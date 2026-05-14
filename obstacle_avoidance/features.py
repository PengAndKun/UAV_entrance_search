from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    import run_drone_flight as flight
except Exception:  # pragma: no cover - keeps offline dataset tools usable in minimal envs.
    flight = None  # type: ignore[assignment]


MISSION_PHASES = (
    "IDLE",
    "MANUAL",
    "NAV_TO_OBS",
    "NAV_TO_SCAN_POINT",
    "SCANNING",
    "NAVIGATING_TO_ENTRY",
    "UNKNOWN",
)

ACTION_PAYLOAD_TEMPLATES: Dict[str, Dict[str, float]] = {
    "hold": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "stop": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "forward": {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "slow_forward": {"forward_cm": 5.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "backoff": {"forward_cm": -20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "backward": {"forward_cm": -20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "left": {"forward_cm": 0.0, "right_cm": -20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "side_step_left": {"forward_cm": 0.0, "right_cm": -20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "right": {"forward_cm": 0.0, "right_cm": 20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "side_step_right": {"forward_cm": 0.0, "right_cm": 20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    "up": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 20.0, "yaw_delta_deg": 0.0},
    "down": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": -20.0, "yaw_delta_deg": 0.0},
    "yaw_left": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": -30.0},
    "yaw_right": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 30.0},
}

ACTION_VECTOR_NAMES = ("forward_cm", "right_cm", "up_cm", "yaw_delta_deg")

FEATURE_NAMES: List[str] = [
    "rgb_available",
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "rgb_dark_ratio",
    "rgb_bright_ratio",
    "depth_available",
    "depth_min_cm",
    "depth_max_cm",
    "depth_front_min_cm",
    "depth_front_mean_cm",
    "depth_valid_count_log",
    "depth_width",
    "depth_height",
    "pc_available",
    "pc_point_count_log",
    "pc_front_min_cm",
    "pc_front_mean_cm",
    "pc_corridor_count_log",
    "pc_left_min_cm",
    "pc_right_min_cm",
    "pc_up_min_cm",
    "pc_nearest_cm",
    "relative_distance_cm",
    "relative_bearing_deg_body",
    "relative_dz_cm",
    "last_forward_cm",
    "last_right_cm",
    "last_up_cm",
    "last_yaw_delta_deg",
    "nominal_forward_cm",
    "nominal_right_cm",
    "nominal_up_cm",
    "nominal_yaw_delta_deg",
    "pose_z_cm",
    "movement_enabled",
]
FEATURE_NAMES.extend(f"phase_{phase.lower()}" for phase in MISSION_PHASES)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if not math.isfinite(result):
        return default
    return result


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "collision", "collided"}
    return False


def normalize_angle_deg(angle: float) -> float:
    value = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 else value


def resolve_event_path(value: Any, *, base_dir: Optional[Path] = None) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return path.resolve()


def read_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def first_existing_path(event: Dict[str, Any], keys: Iterable[str], *, base_dir: Optional[Path] = None) -> Path:
    for key in keys:
        path = resolve_event_path(event.get(key), base_dir=base_dir)
        if str(path) not in {"", "."} and path.is_file():
            return path
    capture_dir = resolve_event_path(event.get("capture_dir"), base_dir=base_dir)
    if str(capture_dir) not in {"", "."} and capture_dir.is_dir():
        for key in keys:
            fallback_name = {
                "rgb_path": "rgb.png",
                "depth_npy_path": "depth.npy",
                "depth_cm_path": "depth_cm.png",
                "point_cloud_world_standard_m_npy_path": "point_cloud_world_standard_m.npy",
                "pointcloud_path": "point_cloud_world_standard_m.npy",
            }.get(key, "")
            if fallback_name:
                candidate = capture_dir / fallback_name
                if candidate.exists():
                    return candidate
    return Path()


def summarize_rgb(event: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, float]:
    path = first_existing_path(event, ("rgb_path",), base_dir=base_dir)
    if not path.exists():
        return {
            "rgb_available": 0.0,
            "rgb_mean_r": 0.0,
            "rgb_mean_g": 0.0,
            "rgb_mean_b": 0.0,
            "rgb_std_r": 0.0,
            "rgb_std_g": 0.0,
            "rgb_std_b": 0.0,
            "rgb_dark_ratio": 0.0,
            "rgb_bright_ratio": 0.0,
        }
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return summarize_rgb({}, base_dir=None)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    means = np.mean(rgb, axis=(0, 1))
    stds = np.std(rgb, axis=(0, 1))
    return {
        "rgb_available": 1.0,
        "rgb_mean_r": float(means[0]),
        "rgb_mean_g": float(means[1]),
        "rgb_mean_b": float(means[2]),
        "rgb_std_r": float(stds[0]),
        "rgb_std_g": float(stds[1]),
        "rgb_std_b": float(stds[2]),
        "rgb_dark_ratio": float(np.mean(gray < 0.12)),
        "rgb_bright_ratio": float(np.mean(gray > 0.88)),
    }


def coerce_depth(depth: Any) -> np.ndarray:
    if flight is not None:
        try:
            return flight.coerce_depth_planar_image(depth).astype(np.float32, copy=False)
        except Exception:
            pass
    array = np.asarray(depth)
    if array.ndim == 3:
        array = array[:, :, 0]
    return np.squeeze(array).astype(np.float32, copy=False)


def summarize_depth(event: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, float]:
    raw = event.get("depth_summary")
    if isinstance(raw, dict) and raw:
        front_min = raw.get("front_min_depth_cm", raw.get("front_min_depth", 0.0))
        front_mean = raw.get("front_mean_depth_cm", raw.get("front_mean_depth", 0.0))
        return {
            "depth_available": 1.0 if raw.get("available", True) else 0.0,
            "depth_min_cm": as_float(raw.get("min_depth", raw.get("min_depth_cm", 0.0))),
            "depth_max_cm": as_float(raw.get("max_depth", raw.get("max_depth_cm", 0.0))),
            "depth_front_min_cm": as_float(front_min),
            "depth_front_mean_cm": as_float(front_mean),
            "depth_valid_count_log": float(math.log1p(max(0.0, as_float(raw.get("valid_count", 0.0))))),
            "depth_width": as_float(raw.get("image_width", 0.0)),
            "depth_height": as_float(raw.get("image_height", 0.0)),
        }

    path = first_existing_path(event, ("depth_npy_path", "depth_cm_path"), base_dir=base_dir)
    if not path.exists():
        return {
            "depth_available": 0.0,
            "depth_min_cm": 0.0,
            "depth_max_cm": 0.0,
            "depth_front_min_cm": 0.0,
            "depth_front_mean_cm": 0.0,
            "depth_valid_count_log": 0.0,
            "depth_width": 0.0,
            "depth_height": 0.0,
        }
    try:
        if path.suffix.lower() == ".npy":
            depth = coerce_depth(np.load(path))
        else:
            depth = coerce_depth(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    except Exception:
        return summarize_depth({}, base_dir=None)
    finite = depth[np.isfinite(depth)]
    valid = finite[finite > 0.0]
    h, w = depth.shape[:2]
    patch = depth[int(h * 0.55):int(h * 0.9), int(w * 0.4):int(w * 0.6)]
    patch_valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return {
        "depth_available": 1.0 if finite.size else 0.0,
        "depth_min_cm": float(np.min(valid)) if valid.size else 0.0,
        "depth_max_cm": float(np.max(valid)) if valid.size else 0.0,
        "depth_front_min_cm": float(np.min(patch_valid)) if patch_valid.size else 0.0,
        "depth_front_mean_cm": float(np.mean(patch_valid)) if patch_valid.size else 0.0,
        "depth_valid_count_log": float(math.log1p(valid.size)),
        "depth_width": float(w),
        "depth_height": float(h),
    }


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


def cloud_xyz_unreal_cm(path: Path, event: Dict[str, Any]) -> np.ndarray:
    cloud = np.load(path)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = cloud[:, :3].astype(np.float32, copy=False)
    if xyz.shape[0] > 200_000:
        stride = max(1, int(math.ceil(xyz.shape[0] / 200_000)))
        xyz = xyz[::stride]
    units = str(event.get("coordinate_units", "") or "").lower()
    is_standard_m = "standard_m" in path.name.lower() or units == "m"
    if is_standard_m:
        if flight is not None:
            try:
                return flight.standard_world_m_to_unreal_world_cm(xyz).astype(np.float32, copy=False)
            except Exception:
                pass
        return (xyz * 100.0).astype(np.float32, copy=False)
    return xyz.astype(np.float32, copy=False)


def summarize_pointcloud(event: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, float]:
    raw = event.get("pointcloud_summary")
    if isinstance(raw, dict) and raw:
        return {
            "pc_available": 1.0 if raw.get("available", True) else 0.0,
            "pc_point_count_log": float(math.log1p(max(0.0, as_float(raw.get("point_count", 0.0))))),
            "pc_front_min_cm": as_float(raw.get("front_min_depth_cm", raw.get("front_min_cm", 0.0))),
            "pc_front_mean_cm": as_float(raw.get("front_mean_depth_cm", raw.get("front_mean_cm", 0.0))),
            "pc_corridor_count_log": float(math.log1p(max(0.0, as_float(raw.get("corridor_count", 0.0))))),
            "pc_left_min_cm": as_float(raw.get("left_min_depth_cm", raw.get("left_min_cm", 0.0))),
            "pc_right_min_cm": as_float(raw.get("right_min_depth_cm", raw.get("right_min_cm", 0.0))),
            "pc_up_min_cm": as_float(raw.get("up_min_depth_cm", raw.get("up_min_cm", 0.0))),
            "pc_nearest_cm": as_float(raw.get("nearest_distance_cm", 0.0)),
        }
    path = first_existing_path(
        event,
        (
            "pointcloud_path",
            "point_cloud_world_standard_m_npy_path",
            "point_cloud_world_npy_path",
        ),
        base_dir=base_dir,
    )
    if not path.exists():
        return {
            "pc_available": 0.0,
            "pc_point_count_log": 0.0,
            "pc_front_min_cm": 0.0,
            "pc_front_mean_cm": 0.0,
            "pc_corridor_count_log": 0.0,
            "pc_left_min_cm": 0.0,
            "pc_right_min_cm": 0.0,
            "pc_up_min_cm": 0.0,
            "pc_nearest_cm": 0.0,
        }
    try:
        xyz = cloud_xyz_unreal_cm(path, event)
    except Exception:
        return summarize_pointcloud({}, base_dir=None)
    if xyz.size == 0:
        return summarize_pointcloud({}, base_dir=None)
    pose = pose_from_event(event)
    yaw = math.radians(pose["yaw"])
    dx = xyz[:, 0] - pose["x"]
    dy = xyz[:, 1] - pose["y"]
    dz = xyz[:, 2] - pose["z"]
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    up = dz
    vertical = np.abs(up) <= 250.0
    front = (forward > 0.0) & (forward <= 600.0)
    corridor = front & (np.abs(right) <= 180.0) & vertical
    left = front & (right < -40.0) & (right >= -300.0) & vertical
    right_side = front & (right > 40.0) & (right <= 300.0) & vertical
    up_zone = front & (np.abs(right) <= 180.0) & (up > 40.0) & (up <= 300.0)
    nearest = np.sqrt(dx * dx + dy * dy + dz * dz)

    def min_forward(mask: np.ndarray) -> float:
        values = forward[mask]
        return float(np.min(values)) if values.size else 0.0

    front_values = forward[corridor]
    return {
        "pc_available": 1.0,
        "pc_point_count_log": float(math.log1p(xyz.shape[0])),
        "pc_front_min_cm": float(np.min(front_values)) if front_values.size else 0.0,
        "pc_front_mean_cm": float(np.mean(front_values)) if front_values.size else 0.0,
        "pc_corridor_count_log": float(math.log1p(front_values.size)),
        "pc_left_min_cm": min_forward(left),
        "pc_right_min_cm": min_forward(right_side),
        "pc_up_min_cm": min_forward(up_zone),
        "pc_nearest_cm": float(np.min(nearest)) if nearest.size else 0.0,
    }


def relative_target_features(event: Dict[str, Any]) -> Dict[str, float]:
    raw = event.get("relative_target")
    if isinstance(raw, dict):
        return {
            "relative_distance_cm": as_float(raw.get("distance_cm")),
            "relative_bearing_deg_body": as_float(raw.get("bearing_deg_body")),
            "relative_dz_cm": as_float(raw.get("dz_cm")),
        }
    target = event.get("target_waypoint")
    if not isinstance(target, dict) or not target:
        return {"relative_distance_cm": 0.0, "relative_bearing_deg_body": 0.0, "relative_dz_cm": 0.0}
    pose = pose_from_event(event)
    dx = as_float(target.get("x")) - pose["x"]
    dy = as_float(target.get("y")) - pose["y"]
    dz = as_float(target.get("z")) - pose["z"]
    absolute_bearing = math.degrees(math.atan2(dy, dx)) if dx or dy else pose["yaw"]
    return {
        "relative_distance_cm": float(math.hypot(dx, dy)),
        "relative_bearing_deg_body": normalize_angle_deg(absolute_bearing - pose["yaw"]),
        "relative_dz_cm": dz,
    }


def payload_from_label(label: Any) -> Dict[str, float]:
    text = str(label or "").strip().lower()
    return dict(ACTION_PAYLOAD_TEMPLATES.get(text, ACTION_PAYLOAD_TEMPLATES["hold"]))


def payload_to_vector(payload: Any) -> np.ndarray:
    if not isinstance(payload, dict):
        return np.zeros((4,), dtype=np.float32)
    return np.asarray([as_float(payload.get(name)) for name in ACTION_VECTOR_NAMES], dtype=np.float32)


def best_action_payload(event: Dict[str, Any]) -> Dict[str, float]:
    for key in ("executed_action", "agent_action", "expert_action_payload", "nominal_action", "last_action"):
        raw = event.get(key)
        if isinstance(raw, dict):
            return {
                "forward_cm": as_float(raw.get("forward_cm")),
                "right_cm": as_float(raw.get("right_cm")),
                "up_cm": as_float(raw.get("up_cm")),
                "yaw_delta_deg": as_float(raw.get("yaw_delta_deg")),
            }
    return payload_from_label(event.get("expert_action"))


def action_label_from_event(event: Dict[str, Any]) -> str:
    explicit = str(event.get("expert_action", "") or "").strip().lower()
    if explicit:
        return explicit
    for key in ("executed_action", "agent_action", "nominal_action", "last_action"):
        raw = event.get(key)
        if isinstance(raw, dict):
            name = str(raw.get("action_name", "") or "").strip().lower()
            if name:
                return name
    return "hold"


def risk_label_from_event(event: Dict[str, Any]) -> str:
    explicit = str(event.get("risk_state", "") or "").strip().upper()
    if explicit:
        return explicit
    return "BLOCKED" if as_bool(event.get("collision_state")) else "SAFE"


def extract_event_features(event: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    values: Dict[str, float] = {}
    values.update(summarize_rgb(event, base_dir=base_dir))
    values.update(summarize_depth(event, base_dir=base_dir))
    values.update(summarize_pointcloud(event, base_dir=base_dir))
    values.update(relative_target_features(event))
    last_payload = event.get("last_action") if isinstance(event.get("last_action"), dict) else {}
    nominal_payload = event.get("nominal_action") if isinstance(event.get("nominal_action"), dict) else {}
    for name, value in zip(("last_forward_cm", "last_right_cm", "last_up_cm", "last_yaw_delta_deg"), payload_to_vector(last_payload)):
        values[name] = float(value)
    for name, value in zip(("nominal_forward_cm", "nominal_right_cm", "nominal_up_cm", "nominal_yaw_delta_deg"), payload_to_vector(nominal_payload)):
        values[name] = float(value)
    pose = pose_from_event(event)
    values["pose_z_cm"] = pose["z"]
    values["movement_enabled"] = 1.0 if as_bool(event.get("movement_enabled")) else 0.0
    phase = str(event.get("mission_phase", "UNKNOWN") or "UNKNOWN").strip().upper()
    if phase not in MISSION_PHASES:
        phase = "UNKNOWN"
    for item in MISSION_PHASES:
        values[f"phase_{item.lower()}"] = 1.0 if item == phase else 0.0

    vector = np.asarray([values.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
    metadata = {
        "feature_values": values,
        "risk_state": risk_label_from_event(event),
        "expert_action": action_label_from_event(event),
        "collision_state": as_bool(event.get("collision_state")),
        "avoidance_failed": as_bool(event.get("collision_state")),
        "action_vector": payload_to_vector(best_action_payload(event)).astype(float).tolist(),
    }
    return vector, metadata
