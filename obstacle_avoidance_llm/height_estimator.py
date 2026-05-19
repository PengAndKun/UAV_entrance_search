from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


DEFAULT_HORIZONTAL_FOV_DEG = 90.0
DEFAULT_MAX_DEPTH_CM = 1200.0
DEFAULT_NEAR_BAND_BEHIND_CM = 220.0
DEFAULT_NEAR_BAND_IN_FRONT_CM = 80.0
DEFAULT_SAFETY_MARGIN_CM = 90.0
DEFAULT_MIN_FLYOVER_OFFSET_CM = 80.0
DEFAULT_MAX_FLYOVER_OFFSET_CM = 1400.0


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _resolve_path(value: Any) -> Path:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else Path("__missing_oa_llm_height_path__")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_depth_array(event: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    raw = event.get("depth_array_cm")
    if raw is not None:
        depth = np.asarray(raw, dtype=np.float32)
        return np.squeeze(depth), "event_depth_array"

    path = _resolve_path(event.get("depth_npy_path"))
    if not path.is_file():
        capture_dir = _resolve_path(event.get("capture_dir"))
        candidate = capture_dir / "depth.npy"
        if candidate.is_file():
            path = candidate
    if path.is_file():
        try:
            depth = np.load(path).astype(np.float32, copy=False)
            return np.squeeze(depth), str(path)
        except Exception:
            return None, str(path)
    return None, ""


def _camera_info_from_event(event: Dict[str, Any], depth_shape: Tuple[int, int]) -> Dict[str, Any]:
    info = event.get("camera_info")
    if isinstance(info, dict):
        return dict(info)
    path = _resolve_path(event.get("camera_info_path"))
    if not path.is_file():
        capture_dir = _resolve_path(event.get("capture_dir"))
        candidate = capture_dir / "camera_info.json"
        if candidate.is_file():
            path = candidate
    data = _load_json(path) if path.is_file() else {}
    data.setdefault("image_height", int(depth_shape[0]))
    data.setdefault("image_width", int(depth_shape[1]))
    data.setdefault("horizontal_fov_deg", DEFAULT_HORIZONTAL_FOV_DEG)
    return data


def _pose_z_from_event(event: Dict[str, Any], camera_info: Dict[str, Any]) -> float:
    for key in ("current_pose", "pose", "actual_pose"):
        pose = event.get(key)
        if isinstance(pose, dict):
            z = as_float(pose.get("z"), float("nan"))
            if math.isfinite(z):
                return z
    location = camera_info.get("location") if isinstance(camera_info.get("location"), dict) else {}
    return as_float(location.get("z"), 0.0)


def _camera_z_from_info(camera_info: Dict[str, Any], fallback_z: float) -> float:
    location = camera_info.get("location") if isinstance(camera_info.get("location"), dict) else {}
    z = as_float(location.get("z"), float("nan"))
    return z if math.isfinite(z) else fallback_z


def _camera_pitch_from_info(camera_info: Dict[str, Any]) -> float:
    rotation = camera_info.get("rotation") if isinstance(camera_info.get("rotation"), dict) else {}
    return as_float(rotation.get("pitch"), 0.0)


def _semantic_text(strategy: Optional[Dict[str, Any]]) -> str:
    data = strategy if isinstance(strategy, dict) else {}
    return f"{data.get('environment_id', '')} {data.get('obstacle_hint', '')}".lower()


def _flyover_recommended(strategy: Optional[Dict[str, Any]]) -> bool:
    text = _semantic_text(strategy)
    if any(token in text for token in ("fence", "rail", "building", "roof", "house", "wall")):
        return True
    if any(token in text for token in ("tree_trunk", "pole", "trunk")):
        return False
    return any(token in text for token in ("overhang", "canopy", "cluster"))


def _corridor_half_ratio(strategy: Optional[Dict[str, Any]]) -> float:
    text = _semantic_text(strategy)
    if any(token in text for token in ("fence", "rail", "building", "roof", "wall")):
        return 0.42
    if any(token in text for token in ("tree_trunk", "pole", "trunk")):
        return 0.24
    return 0.34


def _summary_fallback(
    event: Dict[str, Any],
    strategy: Optional[Dict[str, Any]],
    *,
    safety_margin_cm: float,
    min_flyover_offset_cm: float,
    max_flyover_offset_cm: float,
    reason: str,
) -> Dict[str, Any]:
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    pose_z = _pose_z_from_event(event, {})
    height = as_float(summary.get("obstacle_height_cm"))
    flyover = _flyover_recommended(strategy)
    if height <= 0.0:
        return {
            "available": False,
            "height_source": "unavailable",
            "reason": reason,
            "formula": "height unavailable because no depth array or positive summary obstacle_height_cm was found",
        }
    estimated_top = pose_z + max(0.0, height * 0.5)
    target_z = estimated_top + float(safety_margin_cm)
    offset = max(0.0, target_z - pose_z)
    if flyover:
        offset = min(max(float(min_flyover_offset_cm), offset), float(max_flyover_offset_cm))
        target_z = pose_z + offset
    else:
        offset = 0.0
        target_z = pose_z
    return {
        "available": True,
        "height_source": "pointcloud_summary_fallback",
        "flyover_recommended": bool(flyover),
        "current_z_cm": round(pose_z, 3),
        "camera_z_cm": round(pose_z, 3),
        "obstacle_base_z_cm": round(pose_z - max(0.0, height * 0.5), 3),
        "obstacle_top_z_cm": round(estimated_top, 3),
        "obstacle_height_cm": round(height, 3),
        "recommended_flyover_z_cm": round(target_z, 3),
        "recommended_vertical_offset_cm": round(offset, 3),
        "safety_margin_cm": round(float(safety_margin_cm), 3),
        "formula": "fallback: top_z ~= current_z + obstacle_height_cm/2; target_z = top_z + safety_margin_cm",
        "reason": reason,
    }


def estimate_pointcloud_flyover_height(
    event: Dict[str, Any],
    strategy: Optional[Dict[str, Any]] = None,
    *,
    safety_margin_cm: float = DEFAULT_SAFETY_MARGIN_CM,
    min_flyover_offset_cm: float = DEFAULT_MIN_FLYOVER_OFFSET_CM,
    max_flyover_offset_cm: float = DEFAULT_MAX_FLYOVER_OFFSET_CM,
) -> Dict[str, Any]:
    """Estimate obstacle height from the RGB-aligned depth point cloud.

    Formula:
      fx = W / (2 * tan(horizontal_fov / 2))
      fy = H / (2 * tan(vertical_fov / 2))
      up_camera_cm = -(pixel_y - cy) * depth_cm / fy
      point_z_cm = camera_z_cm + up_camera_cm * cos(pitch) + depth_cm * sin(pitch)
      obstacle_top_z_cm = percentile_95(point_z_cm for close obstacle pixels)
      recommended_flyover_z_cm = obstacle_top_z_cm + safety_margin_cm
    """
    data = event if isinstance(event, dict) else {}
    depth, depth_source = _load_depth_array(data)
    if depth is None or depth.ndim != 2 or depth.size == 0:
        return _summary_fallback(
            data,
            strategy,
            safety_margin_cm=safety_margin_cm,
            min_flyover_offset_cm=min_flyover_offset_cm,
            max_flyover_offset_cm=max_flyover_offset_cm,
            reason="depth array unavailable",
        )

    height_px, width_px = int(depth.shape[0]), int(depth.shape[1])
    camera_info = _camera_info_from_event(data, (height_px, width_px))
    current_z = _pose_z_from_event(data, camera_info)
    camera_z = _camera_z_from_info(camera_info, current_z)
    pitch_deg = _camera_pitch_from_info(camera_info)

    fov_x_deg = as_float(camera_info.get("horizontal_fov_deg"), DEFAULT_HORIZONTAL_FOV_DEG)
    fov_x_deg = min(150.0, max(10.0, fov_x_deg))
    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = 2.0 * math.atan(math.tan(fov_x_rad / 2.0) * (height_px / max(1.0, float(width_px))))
    fx = width_px / (2.0 * math.tan(fov_x_rad / 2.0))
    fy = height_px / (2.0 * math.tan(fov_y_rad / 2.0))
    cx = (width_px - 1.0) / 2.0
    cy = (height_px - 1.0) / 2.0

    summary = data.get("pointcloud_summary") if isinstance(data.get("pointcloud_summary"), dict) else {}
    depth_summary = data.get("depth_summary") if isinstance(data.get("depth_summary"), dict) else {}
    max_depth_cm = as_float(
        camera_info.get("max_depth_cm", depth_summary.get("max_depth", DEFAULT_MAX_DEPTH_CM)),
        DEFAULT_MAX_DEPTH_CM,
    )
    min_depth_cm = as_float(camera_info.get("min_depth_cm", depth_summary.get("min_depth", 1.0)), 1.0)
    valid = np.isfinite(depth) & (depth > max(1.0, min_depth_cm * 0.5))
    if max_depth_cm > 0.0:
        valid &= depth < (max_depth_cm - 1.0)

    if not bool(np.any(valid)):
        return _summary_fallback(
            data,
            strategy,
            safety_margin_cm=safety_margin_cm,
            min_flyover_offset_cm=min_flyover_offset_cm,
            max_flyover_offset_cm=max_flyover_offset_cm,
            reason="no valid depth pixels",
        )

    front_min = as_float(summary.get("front_min_depth_cm"))
    if front_min <= 0.0:
        front_min = float(np.percentile(depth[valid], 10.0))
    near_min = max(min_depth_cm, front_min - DEFAULT_NEAR_BAND_IN_FRONT_CM)
    near_max = min(max_depth_cm - 1.0 if max_depth_cm > 1.0 else front_min + DEFAULT_NEAR_BAND_BEHIND_CM, front_min + DEFAULT_NEAR_BAND_BEHIND_CM)

    cols = np.arange(width_px, dtype=np.float32)[None, :]
    bearing = as_float((data.get("relative_target") or {}).get("bearing_deg_body")) if isinstance(data.get("relative_target"), dict) else 0.0
    center_shift_px = math.tan(math.radians(bearing)) / max(0.01, math.tan(fov_x_rad / 2.0)) * (width_px / 2.0)
    center_shift_px = max(-width_px * 0.2, min(width_px * 0.2, center_shift_px))
    corridor_half_px = width_px * _corridor_half_ratio(strategy)
    corridor = np.abs(cols - (cx + center_shift_px)) <= corridor_half_px
    obstacle_mask = valid & corridor & (depth >= near_min) & (depth <= near_max)
    pixel_count = int(np.count_nonzero(obstacle_mask))
    if pixel_count < 16:
        obstacle_mask = valid & (np.abs(cols - (cx + center_shift_px)) <= width_px * min(0.5, _corridor_half_ratio(strategy) + 0.16)) & (depth <= front_min + 360.0)
        pixel_count = int(np.count_nonzero(obstacle_mask))
    if pixel_count < 8:
        return _summary_fallback(
            data,
            strategy,
            safety_margin_cm=safety_margin_cm,
            min_flyover_offset_cm=min_flyover_offset_cm,
            max_flyover_offset_cm=max_flyover_offset_cm,
            reason=f"too few close obstacle pixels: {pixel_count}",
        )

    rows_idx, _cols_idx = np.nonzero(obstacle_mask)
    selected_depth = depth[obstacle_mask].astype(np.float32, copy=False)
    up_camera_cm = -((rows_idx.astype(np.float32) - cy) * selected_depth / max(1.0, fy))
    pitch_rad = math.radians(pitch_deg)
    vertical_rel_cm = up_camera_cm * math.cos(pitch_rad) + selected_depth * math.sin(pitch_rad)
    z_world_cm = camera_z + vertical_rel_cm

    base_z = float(np.percentile(z_world_cm, 5.0))
    top_z = float(np.percentile(z_world_cm, 95.0))
    obstacle_height = max(0.0, top_z - base_z)
    clearance_z = top_z + float(safety_margin_cm)
    flyover = _flyover_recommended(strategy)
    if flyover:
        vertical_offset = clearance_z - current_z
        vertical_offset = min(max(float(min_flyover_offset_cm), vertical_offset), float(max_flyover_offset_cm))
        recommended_z = current_z + vertical_offset
    else:
        vertical_offset = 0.0
        recommended_z = current_z

    return {
        "available": True,
        "height_source": "depth_projection",
        "flyover_recommended": bool(flyover),
        "depth_source": depth_source,
        "selected_pixel_count": pixel_count,
        "front_min_depth_cm": round(front_min, 3),
        "near_depth_min_cm": round(float(near_min), 3),
        "near_depth_max_cm": round(float(near_max), 3),
        "current_z_cm": round(current_z, 3),
        "camera_z_cm": round(camera_z, 3),
        "camera_pitch_deg": round(pitch_deg, 3),
        "fx_px": round(float(fx), 3),
        "fy_px": round(float(fy), 3),
        "obstacle_base_z_cm": round(base_z, 3),
        "obstacle_top_z_cm": round(top_z, 3),
        "obstacle_height_cm": round(obstacle_height, 3),
        "clearance_z_cm": round(clearance_z, 3),
        "recommended_flyover_z_cm": round(recommended_z, 3),
        "recommended_vertical_offset_cm": round(vertical_offset, 3),
        "safety_margin_cm": round(float(safety_margin_cm), 3),
        "formula": (
            "fx=W/(2*tan(fov_x/2)); fy=H/(2*tan(fov_y/2)); "
            "up=-(v-cy)*depth/fy; point_z=camera_z+up*cos(pitch)+depth*sin(pitch); "
            "top_z=P95(point_z); target_z=top_z+safety_margin"
        ),
    }
