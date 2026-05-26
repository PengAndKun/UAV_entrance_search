from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .map_utils import image_to_world_with_calibration, world_to_image_with_calibration


ROUTE6_FACADES = ("south", "east", "north", "west")
ROUTE6_TERMINAL_HOUSE_STATUSES = {"searched", "searched_no_entry", "terminal_blocked"}
ROUTE6_COORDINATE_FRAME = {
    "pointcloud": "standard_m",
    "map_config": "unreal_cm",
    "standard_to_unreal": "x_cm=x_m*100, y_cm=-y_m*100, z_cm=z_m*100",
}
DEFAULT_ROUTE6_LAYER_Z_CM = tuple(range(50, 651, 50))
DEFAULT_ROUTE6_LAYER_BAND_CM = 25.0
DEFAULT_ROUTE6_LAYER_OCCUPIED_THRESHOLD = 2
DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM = {"min_x": -5000, "max_x": 5000, "min_y": -5000, "max_y": 5000}
DEFAULT_ROUTE6_UPDATE_MAP_VOXEL_SIZE_M = 0.25
DEFAULT_ROUTE6_UPDATE_MAP_MAX_POINTS = 320000


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _normalize_fixed_world_bounds_cm(bounds: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not isinstance(bounds, dict):
        return None
    result = {
        "min_x": _as_int(bounds.get("min_x"), -5000),
        "max_x": _as_int(bounds.get("max_x"), 5000),
        "min_y": _as_int(bounds.get("min_y"), -5000),
        "max_y": _as_int(bounds.get("max_y"), 5000),
    }
    if result["max_x"] <= result["min_x"] or result["max_y"] <= result["min_y"]:
        return None
    return result


def _house_id(house: Dict[str, Any]) -> str:
    return str(house.get("id", house.get("house_id", "")) or "").strip()


def _polygon_bbox(points: Iterable[Dict[str, float]]) -> Dict[str, float]:
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    if not xs or not ys:
        return {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0}
    return {
        "min_x": float(min(xs)),
        "max_x": float(max(xs)),
        "min_y": float(min(ys)),
        "max_y": float(max(ys)),
    }


def _bbox_center(bbox: Dict[str, Any]) -> Tuple[float, float]:
    return (
        0.5 * (_as_float(bbox.get("min_x")) + _as_float(bbox.get("max_x"))),
        0.5 * (_as_float(bbox.get("min_y")) + _as_float(bbox.get("max_y"))),
    )


def _bbox_distance_cm(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    dx = max(float(a.get("min_x", 0.0)) - float(b.get("max_x", 0.0)), float(b.get("min_x", 0.0)) - float(a.get("max_x", 0.0)), 0.0)
    dy = max(float(a.get("min_y", 0.0)) - float(b.get("max_y", 0.0)), float(b.get("min_y", 0.0)) - float(a.get("max_y", 0.0)), 0.0)
    edge_distance = math.hypot(dx, dy)
    if edge_distance > 0.0:
        return float(edge_distance)
    return float(math.hypot(acx - bcx, acy - bcy) * 0.01)


def house_world_bbox(config: Dict[str, Any], house: Dict[str, Any]) -> Dict[str, float]:
    for key in ("route6_corrected_bbox_world", "route6_candidate_bbox_world", "bbox", "bbox_world"):
        bbox = house.get(key)
        if isinstance(bbox, dict) and all(name in bbox for name in ("min_x", "max_x", "min_y", "max_y")):
            return {
                "min_x": _as_float(bbox["min_x"]),
                "max_x": _as_float(bbox["max_x"]),
                "min_y": _as_float(bbox["min_y"]),
                "max_y": _as_float(bbox["max_y"]),
            }

    overhead = config.get("overhead_map", {}) if isinstance(config.get("overhead_map"), dict) else {}
    calibration = overhead.get("calibration", {}) if isinstance(overhead.get("calibration"), dict) else {}
    bbox_image = house.get("map_bbox_image", {}) if isinstance(house.get("map_bbox_image"), dict) else {}
    if bbox_image and calibration:
        try:
            corners = [
                image_to_world_with_calibration(_as_float(bbox_image["x1"]), _as_float(bbox_image["y1"]), calibration),
                image_to_world_with_calibration(_as_float(bbox_image["x2"]), _as_float(bbox_image["y1"]), calibration),
                image_to_world_with_calibration(_as_float(bbox_image["x2"]), _as_float(bbox_image["y2"]), calibration),
                image_to_world_with_calibration(_as_float(bbox_image["x1"]), _as_float(bbox_image["y2"]), calibration),
            ]
            xs = [float(point[0]) for point in corners]
            ys = [float(point[1]) for point in corners]
            return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}
        except Exception:
            pass

    center_x = _as_float(house.get("center_x"))
    center_y = _as_float(house.get("center_y"))
    radius = max(150.0, _as_float(house.get("radius_cm"), 300.0))
    return {
        "min_x": center_x - radius,
        "max_x": center_x + radius,
        "min_y": center_y - radius,
        "max_y": center_y + radius,
    }


def nearest_boundary_distance_cm(pose: Dict[str, Any], bbox: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    px = _as_float(pose.get("x"))
    py = _as_float(pose.get("y"))
    nearest_x = min(max(px, float(bbox["min_x"])), float(bbox["max_x"]))
    nearest_y = min(max(py, float(bbox["min_y"])), float(bbox["max_y"]))
    return float(math.hypot(nearest_x - px, nearest_y - py)), {"x": round(nearest_x, 2), "y": round(nearest_y, 2)}


def _yaw_toward(source_x: float, source_y: float, target_x: float, target_y: float) -> float:
    return float(math.degrees(math.atan2(float(target_y) - float(source_y), float(target_x) - float(source_x))))


def facade_scan_pose_for_bbox(
    bbox: Dict[str, float],
    facade: str,
    *,
    current_pose: Optional[Dict[str, Any]] = None,
    standoff_cm: float = 850.0,
    scan_z_cm: float = 450.0,
) -> Dict[str, Any]:
    min_x = float(bbox["min_x"])
    max_x = float(bbox["max_x"])
    min_y = float(bbox["min_y"])
    max_y = float(bbox["max_y"])
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    px = _as_float((current_pose or {}).get("x"), center_x)
    py = _as_float((current_pose or {}).get("y"), center_y)
    facade = str(facade or "").lower()
    if facade == "south":
        x = min(max(px, min_x), max_x)
        y = min_y - float(standoff_cm)
    elif facade == "north":
        x = min(max(px, min_x), max_x)
        y = max_y + float(standoff_cm)
    elif facade == "east":
        x = max_x + float(standoff_cm)
        y = min(max(py, min_y), max_y)
    else:
        facade = "west"
        x = min_x - float(standoff_cm)
        y = min(max(py, min_y), max_y)
    return {
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "z": round(float(scan_z_cm), 2),
        "yaw_deg": round(_yaw_toward(x, y, center_x, center_y), 3),
        "facade": facade,
        "standoff_cm": round(float(standoff_cm), 2),
    }


def nearest_facade_scan_pose(
    bbox: Dict[str, float],
    pose: Dict[str, Any],
    *,
    standoff_cm: float = 850.0,
    scan_z_cm: float = 450.0,
) -> Dict[str, Any]:
    px = _as_float(pose.get("x"))
    py = _as_float(pose.get("y"))
    poses = [
        facade_scan_pose_for_bbox(bbox, facade, current_pose=pose, standoff_cm=standoff_cm, scan_z_cm=scan_z_cm)
        for facade in ROUTE6_FACADES
    ]
    return min(poses, key=lambda item: math.hypot(float(item["x"]) - px, float(item["y"]) - py))


def rank_house_candidates(
    map_config: Dict[str, Any],
    current_pose: Dict[str, Any],
    *,
    house_states: Optional[Dict[str, Dict[str, Any]]] = None,
    route_cost_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    standoff_cm: float = 850.0,
    scan_z_cm: float = 450.0,
) -> List[Dict[str, Any]]:
    houses = map_config.get("houses", []) if isinstance(map_config.get("houses"), list) else []
    states = house_states if isinstance(house_states, dict) else {}
    candidates: List[Dict[str, Any]] = []
    for house in houses:
        if not isinstance(house, dict):
            continue
        hid = _house_id(house)
        if not hid:
            continue
        state = states.get(hid, {}) if isinstance(states.get(hid, {}), dict) else {}
        status = str(state.get("status", house.get("route6_status", "unknown")) or "unknown").strip()
        if status in ROUTE6_TERMINAL_HOUSE_STATUSES or bool(state.get("cooldown_active", False)):
            continue
        bbox = house_world_bbox(map_config, house)
        boundary_distance, boundary_point = nearest_boundary_distance_cm(current_pose, bbox)
        scan_pose = nearest_facade_scan_pose(bbox, current_pose, standoff_cm=standoff_cm, scan_z_cm=scan_z_cm)
        base_cost = float(boundary_distance)
        route_result = route_cost_fn(scan_pose) if route_cost_fn is not None else {}
        route_cost = _as_float(route_result.get("route_cost_cm", route_result.get("cost_cm", base_cost)), base_cost)
        reachable = bool(route_result.get("reachable", True))
        blocked_reason = str(route_result.get("blocked_reason", "") or "")
        if not reachable:
            continue
        obstacle_state = str(route_result.get("obstacle_risk", route_result.get("safety_state", "safe")) or "safe").lower()
        obstacle_penalty = 1500.0 if "blocked" in obstacle_state else (500.0 if "caution" in obstacle_state else 0.0)
        blocked_history_penalty = 2000.0 if status in {"blocked", "needs_rescan"} else 0.0
        confidence = max(0.0, min(1.0, _as_float(state.get("map_confidence", house.get("route6_map_confidence", 0.0)), 0.0)))
        map_uncertainty_penalty = (1.0 - confidence) * 600.0
        information_gain_bonus = 800.0 if status in {"unknown", "queued", ""} else 250.0
        score = max(0.0, route_cost + boundary_distance * 0.4 + obstacle_penalty + blocked_history_penalty + map_uncertainty_penalty - information_gain_bonus)
        candidates.append({
            "house_id": hid,
            "status": status,
            "name": str(house.get("name", f"House_{hid}") or f"House_{hid}"),
            "center_x": _as_float(house.get("center_x")),
            "center_y": _as_float(house.get("center_y")),
            "bbox": {key: round(float(value), 2) for key, value in bbox.items()},
            "nearest_boundary_distance_cm": round(float(boundary_distance), 2),
            "nearest_boundary_point_world": boundary_point,
            "nearest_facade": str(scan_pose["facade"]),
            "nearest_scan_pose": scan_pose,
            "route_cost_cm": round(float(route_cost), 2),
            "reachable": True,
            "blocked_reason": blocked_reason,
            "map_confidence": round(float(confidence), 4),
            "information_gain": round(float(information_gain_bonus), 2),
            "score": round(float(score), 4),
        })
    candidates.sort(key=lambda item: (float(item["score"]), float(item["route_cost_cm"]), str(item["house_id"])))
    return candidates


def select_next_house_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return candidates[0] if candidates else {}


def _row_pointcloud_path(row: Dict[str, Any]) -> Path:
    for key in (
        "point_cloud_world_standard_m_npy_path",
        "pointcloud_path",
        "point_cloud_world_npy_path",
    ):
        value = str(row.get(key, "") or "")
        if value:
            return Path(value)
    return Path()


def filter_valid_pointcloud_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        facade = str(row.get("facade", "") or "").strip().lower()
        if facade not in ROUTE6_FACADES:
            continue
        if row.get("capture_guard_passed") is not True:
            continue
        if _as_int(row.get("point_count", 0)) <= 0:
            continue
        path = _row_pointcloud_path(row)
        if not path.is_file():
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
                continue
        except Exception:
            continue
        key = (str(row.get("scan_id", "")), str(path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        selected.append({**row, "facade": facade, "point_cloud_world_standard_m_npy_path": str(path)})
    return selected


def merge_pointcloud_rows(rows: Iterable[Dict[str, Any]], *, max_points: int = 320000) -> np.ndarray:
    clouds: List[np.ndarray] = []
    for row in rows:
        path = _row_pointcloud_path(row)
        if not path.is_file():
            continue
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
            continue
        if arr.shape[1] < 6:
            pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr[:, :3], pad])
        clouds.append(arr[:, :6])
    if not clouds:
        return np.zeros((0, 6), dtype=np.float32)
    merged = np.vstack(clouds).astype(np.float32, copy=False)
    if int(max_points) > 0 and merged.shape[0] > int(max_points):
        indices = np.linspace(0, merged.shape[0] - 1, num=int(max_points), dtype=np.int64)
        merged = merged[indices]
    return merged


def voxel_downsample_point_cloud(
    cloud: np.ndarray,
    *,
    voxel_size_m: float = DEFAULT_ROUTE6_UPDATE_MAP_VOXEL_SIZE_M,
    fixed_world_bounds_cm: Optional[Dict[str, Any]] = DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
    max_points: int = 0,
) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
        return np.zeros((0, 6), dtype=np.float32)
    if arr.shape[1] < 6:
        pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr[:, :3], pad])
    arr = arr[:, :6].astype(np.float32, copy=False)
    finite = np.isfinite(arr[:, :3]).all(axis=1)
    arr = arr[finite]
    fixed_bounds = _normalize_fixed_world_bounds_cm(fixed_world_bounds_cm)
    if fixed_bounds is not None and arr.shape[0] > 0:
        min_x = float(fixed_bounds["min_x"]) / 100.0
        max_x = float(fixed_bounds["max_x"]) / 100.0
        min_y = float(fixed_bounds["min_y"]) / 100.0
        max_y = float(fixed_bounds["max_y"]) / 100.0
        in_bounds = (
            (arr[:, 0] >= min_x)
            & (arr[:, 0] <= max_x)
            & (arr[:, 1] >= min_y)
            & (arr[:, 1] <= max_y)
        )
        arr = arr[in_bounds]
    if arr.shape[0] <= 0:
        return np.zeros((0, 6), dtype=np.float32)
    voxel = max(0.05, float(voxel_size_m))
    keys = np.floor(arr[:, :3] / voxel).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique_keys.shape[0]).astype(np.float64)
    sums = np.zeros((unique_keys.shape[0], 6), dtype=np.float64)
    np.add.at(sums, inverse, arr[:, :6].astype(np.float64, copy=False))
    reduced = (sums / counts[:, None]).astype(np.float32, copy=False)
    if int(max_points) > 0 and reduced.shape[0] > int(max_points):
        indices = np.linspace(0, reduced.shape[0] - 1, num=int(max_points), dtype=np.int64)
        reduced = reduced[indices]
    return reduced.astype(np.float32, copy=False)


def standard_m_to_unreal_cm_xy(cloud: np.ndarray) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([arr[:, 0] * 100.0, -arr[:, 1] * 100.0]).astype(np.float32, copy=False)


def filter_pointcloud_for_mapping(
    cloud: np.ndarray,
    *,
    z_min_m: float = 0.2,
    z_max_m: float = 8.0,
    bbox_unreal_cm: Optional[Dict[str, float]] = None,
    crop_margin_m: float = 2.0,
) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if arr.shape[1] < 6:
        pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr[:, :3], pad])
    xyz = arr[:, :3]
    mask = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= float(z_min_m)) & (xyz[:, 2] <= float(z_max_m))
    if bbox_unreal_cm:
        xy_cm = standard_m_to_unreal_cm_xy(arr)
        margin_cm = float(crop_margin_m) * 100.0
        mask &= (
            (xy_cm[:, 0] >= float(bbox_unreal_cm["min_x"]) - margin_cm)
            & (xy_cm[:, 0] <= float(bbox_unreal_cm["max_x"]) + margin_cm)
            & (xy_cm[:, 1] >= float(bbox_unreal_cm["min_y"]) - margin_cm)
            & (xy_cm[:, 1] <= float(bbox_unreal_cm["max_y"]) + margin_cm)
        )
    return arr[mask, :6].astype(np.float32, copy=False)


def build_occupancy_grid(
    cloud: np.ndarray,
    *,
    resolution_m: float = 0.25,
    occupied_threshold: int = 3,
    free_threshold: int = 1,
    fixed_world_bounds_cm: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    arr = np.asarray(cloud, dtype=np.float32)
    resolution = max(0.05, float(resolution_m))
    fixed_bounds = _normalize_fixed_world_bounds_cm(fixed_world_bounds_cm)
    fixed_origin: Optional[Tuple[float, float]] = None
    fixed_size: Optional[Tuple[int, int]] = None
    if fixed_bounds is not None:
        min_x = float(fixed_bounds["min_x"]) / 100.0
        max_x = float(fixed_bounds["max_x"]) / 100.0
        min_y = float(fixed_bounds["min_y"]) / 100.0
        max_y = float(fixed_bounds["max_y"]) / 100.0
        fixed_origin = (min_x, min_y)
        fixed_size = (
            max(1, int(math.ceil((max_x - min_x) / resolution))),
            max(1, int(math.ceil((max_y - min_y) / resolution))),
        )
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        width, height = fixed_size if fixed_size is not None else (0, 0)
        origin = list(fixed_origin) if fixed_origin is not None else [0.0, 0.0]
        grid = np.zeros((height, width), dtype=np.int16) if fixed_bounds is not None else np.zeros((0, 0), dtype=np.int16)
        counts = np.zeros((height, width), dtype=np.int32) if fixed_bounds is not None else np.zeros((0, 0), dtype=np.int32)
        return {
            "schema": "route6_occupancy_grid_v1",
            "resolution_m": resolution,
            "origin_standard_m": origin,
            "width": int(width),
            "height": int(height),
            "grid": grid,
            "counts": counts,
            "occupied_threshold": int(occupied_threshold),
            "free_threshold": int(free_threshold),
            "unknown_value": -1,
            "free_value": 0,
            "occupied_value": 100,
            "coordinate_frame": dict(ROUTE6_COORDINATE_FRAME),
            "fixed_world_bounds_cm": dict(fixed_bounds) if fixed_bounds is not None else {},
            "input_point_count": int(arr.shape[0]) if arr.ndim == 2 else 0,
            "finite_point_count": 0,
            "in_bounds_point_count": 0,
            "out_of_bounds_point_count": 0,
        }
    xy = arr[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.shape[0] == 0:
        return build_occupancy_grid(
            np.zeros((0, 6), dtype=np.float32),
            resolution_m=resolution,
            occupied_threshold=occupied_threshold,
            fixed_world_bounds_cm=fixed_bounds,
        )
    if fixed_bounds is not None and fixed_origin is not None and fixed_size is not None:
        min_x, min_y = fixed_origin
        max_x = float(fixed_bounds["max_x"]) / 100.0
        max_y = float(fixed_bounds["max_y"]) / 100.0
        width, height = fixed_size
        in_bounds = (
            (xy[:, 0] >= min_x)
            & (xy[:, 0] <= max_x)
            & (xy[:, 1] >= min_y)
            & (xy[:, 1] <= max_y)
        )
        bounded_xy = xy[in_bounds]
    else:
        min_x = float(np.min(xy[:, 0]))
        min_y = float(np.min(xy[:, 1]))
        max_x = float(np.max(xy[:, 0]))
        max_y = float(np.max(xy[:, 1]))
        width = max(1, int(math.floor((max_x - min_x) / resolution)) + 1)
        height = max(1, int(math.floor((max_y - min_y) / resolution)) + 1)
        in_bounds = np.ones((xy.shape[0],), dtype=bool)
        bounded_xy = xy
    counts = np.zeros((height, width), dtype=np.int32)
    if bounded_xy.shape[0] > 0:
        ix = np.floor((bounded_xy[:, 0] - min_x) / resolution).astype(np.int32)
        iy = np.floor((bounded_xy[:, 1] - min_y) / resolution).astype(np.int32)
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)
        np.add.at(counts, (iy, ix), 1)
    grid = np.zeros((height, width), dtype=np.int16) if fixed_bounds is not None else np.full((height, width), -1, dtype=np.int16)
    grid[counts >= int(max(1, occupied_threshold))] = 100
    return {
        "schema": "route6_occupancy_grid_v1",
        "resolution_m": resolution,
        "origin_standard_m": [min_x, min_y],
        "width": int(width),
        "height": int(height),
        "grid": grid,
        "counts": counts,
        "occupied_threshold": int(max(1, occupied_threshold)),
        "free_threshold": int(max(1, free_threshold)),
        "unknown_value": -1,
        "free_value": 0,
        "occupied_value": 100,
        "coordinate_frame": dict(ROUTE6_COORDINATE_FRAME),
        "fixed_world_bounds_cm": dict(fixed_bounds) if fixed_bounds is not None else {},
        "input_point_count": int(arr.shape[0]),
        "finite_point_count": int(xy.shape[0]),
        "in_bounds_point_count": int(bounded_xy.shape[0]),
        "out_of_bounds_point_count": int(xy.shape[0] - bounded_xy.shape[0]),
    }


def occupancy_metadata(occupancy: Dict[str, Any], *, house_id: str = "") -> Dict[str, Any]:
    return {
        "schema": "route6_occupancy_grid_v1",
        "house_id": str(house_id or ""),
        "resolution_m": float(occupancy.get("resolution_m", 0.25)),
        "origin_standard_m": list(occupancy.get("origin_standard_m", [0.0, 0.0])),
        "width": int(occupancy.get("width", 0) or 0),
        "height": int(occupancy.get("height", 0) or 0),
        "occupied_threshold": int(occupancy.get("occupied_threshold", 1) or 1),
        "free_threshold": int(occupancy.get("free_threshold", 1) or 1),
        "unknown_value": int(occupancy.get("unknown_value", -1)),
        "free_value": int(occupancy.get("free_value", 0)),
        "occupied_value": int(occupancy.get("occupied_value", 100)),
        "coordinate_frame": dict(occupancy.get("coordinate_frame", ROUTE6_COORDINATE_FRAME)),
        "fixed_world_bounds_cm": dict(occupancy.get("fixed_world_bounds_cm", {}) or {}),
        "input_point_count": int(occupancy.get("input_point_count", 0) or 0),
        "finite_point_count": int(occupancy.get("finite_point_count", 0) or 0),
        "in_bounds_point_count": int(occupancy.get("in_bounds_point_count", 0) or 0),
        "out_of_bounds_point_count": int(occupancy.get("out_of_bounds_point_count", 0) or 0),
        "occupied_cell_count": int(np.sum(np.asarray(occupancy.get("grid", np.zeros((0, 0), dtype=np.int16))) == 100)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def occupancy_preview_image(occupancy: Dict[str, Any]) -> np.ndarray:
    grid = np.asarray(occupancy.get("grid", np.zeros((0, 0), dtype=np.int16)))
    if grid.size == 0:
        return np.full((1, 1), 255, dtype=np.uint8)
    image = np.full(grid.shape, 255, dtype=np.uint8)
    image[grid >= 100] = 0
    return np.flipud(image)


def _normalize_layer_z_cm(layer_z_cm: Optional[Iterable[Any]] = None) -> List[int]:
    values = layer_z_cm if layer_z_cm is not None else DEFAULT_ROUTE6_LAYER_Z_CM
    layers: List[int] = []
    for value in values:
        try:
            number = int(round(float(value)))
        except Exception:
            continue
        if number not in layers:
            layers.append(number)
    return layers or list(DEFAULT_ROUTE6_LAYER_Z_CM)


def build_route6_layered_occupancy_maps(
    cloud: np.ndarray,
    *,
    layer_z_cm: Optional[Iterable[Any]] = None,
    layer_band_cm: float = DEFAULT_ROUTE6_LAYER_BAND_CM,
    resolution_m: float = 0.25,
    occupied_threshold: int = DEFAULT_ROUTE6_LAYER_OCCUPIED_THRESHOLD,
    fixed_world_bounds_cm: Optional[Dict[str, Any]] = DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
) -> Dict[str, Any]:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        arr = np.zeros((0, 6), dtype=np.float32)
    if arr.shape[1] < 6 and arr.shape[0] > 0:
        pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr[:, :3], pad])
    layers_z = _normalize_layer_z_cm(layer_z_cm)
    band = max(1.0, float(layer_band_cm))
    finite = np.isfinite(arr[:, :3]).all(axis=1) if arr.shape[0] > 0 else np.zeros((0,), dtype=bool)
    arr = arr[finite, :6] if arr.shape[0] > 0 else arr
    z_cm = arr[:, 2] * 100.0 if arr.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    layers: List[Dict[str, Any]] = []
    assigned_mask = np.zeros((arr.shape[0],), dtype=bool)
    for z in layers_z:
        z_float = float(z)
        mask = (z_cm >= z_float - band) & (z_cm <= z_float + band) if arr.shape[0] > 0 else np.zeros((0,), dtype=bool)
        layer_cloud = arr[mask, :6] if arr.shape[0] > 0 else np.zeros((0, 6), dtype=np.float32)
        if mask.size:
            assigned_mask |= mask
        occupancy = build_occupancy_grid(
            layer_cloud,
            resolution_m=float(resolution_m),
            occupied_threshold=int(max(1, occupied_threshold)),
            fixed_world_bounds_cm=fixed_world_bounds_cm,
        )
        layer_point_count = int(occupancy.get("in_bounds_point_count", layer_cloud.shape[0]) or 0)
        occupied_cell_count = int(np.sum(np.asarray(occupancy.get("grid", np.zeros((0, 0), dtype=np.int16))) >= 100))
        layers.append({
            "schema": "route6_layered_occupancy_layer_v1",
            "z_cm": int(z),
            "z_min_cm": round(z_float - band, 2),
            "z_max_cm": round(z_float + band, 2),
            "layer_band_cm": round(float(band), 2),
            "point_count": layer_point_count,
            "source_layer_point_count": int(layer_cloud.shape[0]),
            "out_of_bounds_point_count": int(occupancy.get("out_of_bounds_point_count", 0) or 0),
            "occupied_cell_count": int(occupied_cell_count),
            "occupancy": occupancy,
        })
    return {
        "schema": "route6_layered_occupancy_v1",
        "layer_z_cm": [int(value) for value in layers_z],
        "layer_band_cm": round(float(band), 2),
        "resolution_m": max(0.05, float(resolution_m)),
        "occupied_threshold": int(max(1, occupied_threshold)),
        "source_point_count": int(arr.shape[0]),
        "assigned_point_count": int(np.sum(assigned_mask)) if assigned_mask.size else 0,
        "unassigned_point_count": int(arr.shape[0] - int(np.sum(assigned_mask))) if assigned_mask.size else int(arr.shape[0]),
        "fixed_world_bounds_cm": dict(_normalize_fixed_world_bounds_cm(fixed_world_bounds_cm) or {}),
        "layers": layers,
        "coordinate_frame": dict(ROUTE6_COORDINATE_FRAME),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _layer_folder_name(z_cm: Any) -> str:
    try:
        value = int(round(float(z_cm)))
    except Exception:
        value = 0
    prefix = "z" if value < 0 else "z_"
    return f"{prefix}{abs(value):03d}"


def write_route6_layered_occupancy_artifacts(
    output_dir: Path,
    cloud: np.ndarray,
    *,
    layer_z_cm: Optional[Iterable[Any]] = None,
    layer_band_cm: float = DEFAULT_ROUTE6_LAYER_BAND_CM,
    resolution_m: float = 0.25,
    occupied_threshold: int = DEFAULT_ROUTE6_LAYER_OCCUPIED_THRESHOLD,
    fixed_world_bounds_cm: Optional[Dict[str, Any]] = DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    layered_dir = output_path / "map" / "layered_occupancy"
    layered_dir.mkdir(parents=True, exist_ok=True)
    layered = build_route6_layered_occupancy_maps(
        cloud,
        layer_z_cm=layer_z_cm,
        layer_band_cm=layer_band_cm,
        resolution_m=resolution_m,
        occupied_threshold=occupied_threshold,
        fixed_world_bounds_cm=fixed_world_bounds_cm,
    )
    layer_records: List[Dict[str, Any]] = []
    for layer in layered["layers"]:
        z_cm = int(layer.get("z_cm", 0) or 0)
        layer_dir = layered_dir / _layer_folder_name(z_cm)
        layer_dir.mkdir(parents=True, exist_ok=True)
        occupancy = layer.get("occupancy", {}) if isinstance(layer.get("occupancy", {}), dict) else {}
        grid_path = layer_dir / "occupancy_grid.npy"
        metadata_path = layer_dir / "occupancy_grid.json"
        preview_path = layer_dir / "occupancy_grid.png"
        np.save(grid_path, np.asarray(occupancy.get("grid", np.zeros((0, 0), dtype=np.int16)), dtype=np.int16))
        cv2.imwrite(str(preview_path), occupancy_preview_image(occupancy))
        metadata = occupancy_metadata(occupancy)
        metadata.update({
            "schema": "route6_layered_occupancy_grid_v1",
            "z_cm": z_cm,
            "z_min_cm": layer.get("z_min_cm", z_cm),
            "z_max_cm": layer.get("z_max_cm", z_cm),
            "layer_band_cm": layer.get("layer_band_cm", layer_band_cm),
            "point_count": int(layer.get("point_count", 0) or 0),
            "source_layer_point_count": int(layer.get("source_layer_point_count", layer.get("point_count", 0)) or 0),
            "out_of_bounds_point_count": int(layer.get("out_of_bounds_point_count", 0) or 0),
            "occupied_cell_count": int(layer.get("occupied_cell_count", metadata.get("occupied_cell_count", 0)) or 0),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        write_json(metadata_path, metadata)
        layer_records.append({
            "schema": "route6_layered_occupancy_layer_artifact_v1",
            "z_cm": z_cm,
            "label": f"z={z_cm} cm",
            "folder": str(layer_dir),
            "z_min_cm": metadata["z_min_cm"],
            "z_max_cm": metadata["z_max_cm"],
            "layer_band_cm": metadata["layer_band_cm"],
            "point_count": metadata["point_count"],
            "source_layer_point_count": metadata["source_layer_point_count"],
            "out_of_bounds_point_count": metadata["out_of_bounds_point_count"],
            "occupied_cell_count": metadata["occupied_cell_count"],
            "occupancy_grid_path": str(grid_path),
            "occupancy_metadata_path": str(metadata_path),
            "occupancy_preview_path": str(preview_path),
        })
    manifest_path = layered_dir / "route6_layered_occupancy.json"
    manifest = {
        "schema": "route6_layered_occupancy_manifest_v1",
        "layered_occupancy_dir": str(layered_dir),
        "layer_z_cm": layered["layer_z_cm"],
        "layer_count": len(layer_records),
        "layer_band_cm": layered["layer_band_cm"],
        "resolution_m": layered["resolution_m"],
        "occupied_threshold": layered["occupied_threshold"],
        "source_point_count": layered["source_point_count"],
        "assigned_point_count": layered["assigned_point_count"],
        "unassigned_point_count": layered["unassigned_point_count"],
        "fixed_world_bounds_cm": dict(layered.get("fixed_world_bounds_cm", {}) or {}),
        "layers": layer_records,
        "coordinate_frame": dict(ROUTE6_COORDINATE_FRAME),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(manifest_path, manifest)
    return {
        "schema": "route6_layered_occupancy_result_v1",
        "layered_occupancy_dir": str(layered_dir),
        "manifest_path": str(manifest_path),
        "layer_count": len(layer_records),
        "source_point_count": int(layered["source_point_count"]),
        "assigned_point_count": int(layered["assigned_point_count"]),
        "unassigned_point_count": int(layered["unassigned_point_count"]),
        "fixed_world_bounds_cm": dict(layered.get("fixed_world_bounds_cm", {}) or {}),
        "layers": layer_records,
    }


def _contour_points_to_standard(contour: np.ndarray, occupancy: Dict[str, Any]) -> List[Dict[str, float]]:
    resolution = float(occupancy["resolution_m"])
    origin_x, origin_y = [float(value) for value in occupancy["origin_standard_m"]]
    points: List[Dict[str, float]] = []
    for item in contour.reshape(-1, 2):
        col = float(item[0])
        row = float(item[1])
        x = origin_x + col * resolution
        y = origin_y + row * resolution
        points.append({"x": round(x * 100.0, 2), "y": round(-y * 100.0, 2)})
    return points


def _component_bbox_unreal_cm(component_mask: np.ndarray, occupancy: Dict[str, Any]) -> Dict[str, float]:
    ys, xs = np.where(component_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0}
    resolution = float(occupancy.get("resolution_m", 0.25))
    origin_x, origin_y = [float(value) for value in occupancy.get("origin_standard_m", [0.0, 0.0])]
    min_x_m = origin_x + float(xs.min()) * resolution
    max_x_m = origin_x + float(xs.max()) * resolution
    min_y_m = origin_y + float(ys.min()) * resolution
    max_y_m = origin_y + float(ys.max()) * resolution
    unreal_y_values = [-min_y_m * 100.0, -max_y_m * 100.0]
    return {
        "min_x": round(min_x_m * 100.0, 2),
        "max_x": round(max_x_m * 100.0, 2),
        "min_y": round(float(min(unreal_y_values)), 2),
        "max_y": round(float(max(unreal_y_values)), 2),
    }


def _select_component_mask(
    occupied: np.ndarray,
    occupancy: Dict[str, Any],
    *,
    target_bbox_unreal_cm: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(occupied, connectivity=8)
    if num_labels <= 1:
        component_cells = int(np.sum(occupied > 0))
        return occupied, {
            "component_count": 1 if component_cells > 0 else 0,
            "selected_component_reason": "single_component",
            "selected_component_label": 1 if component_cells > 0 else 0,
            "selected_component_distance_to_target_bbox_cm": 0.0,
        }
    candidates: List[Dict[str, Any]] = []
    for label in range(1, num_labels):
        area_cells = int(stats[label, cv2.CC_STAT_AREA])
        if area_cells <= 0:
            continue
        mask = (labels == label).astype(np.uint8)
        bbox = _component_bbox_unreal_cm(mask, occupancy)
        distance = _bbox_distance_cm(bbox, target_bbox_unreal_cm) if isinstance(target_bbox_unreal_cm, dict) else 0.0
        area_bonus = min(250.0, float(area_cells) * 0.1)
        score = float(distance) - area_bonus if isinstance(target_bbox_unreal_cm, dict) else -float(area_cells)
        candidates.append({
            "label": label,
            "area_cells": area_cells,
            "bbox": bbox,
            "distance_to_target_bbox_cm": round(float(distance), 2),
            "score": round(float(score), 4),
        })
    if not candidates:
        return occupied, {
            "component_count": 0,
            "selected_component_reason": "no_component_candidates",
            "selected_component_label": 0,
            "selected_component_distance_to_target_bbox_cm": 0.0,
        }
    best = min(candidates, key=lambda item: (float(item["score"]), -int(item["area_cells"]), int(item["label"])))
    reason = "nearest_target_bbox" if isinstance(target_bbox_unreal_cm, dict) else "largest_component"
    return (labels == int(best["label"])).astype(np.uint8), {
        "component_count": len(candidates),
        "selected_component_reason": reason,
        "selected_component_label": int(best["label"]),
        "selected_component_bbox": best["bbox"],
        "selected_component_distance_to_target_bbox_cm": float(best["distance_to_target_bbox_cm"]),
        "component_candidates": candidates,
    }


def extract_building_polygon(
    occupancy: Dict[str, Any],
    *,
    house_id: str = "",
    target_bbox_unreal_cm: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    grid = np.asarray(occupancy.get("grid", np.zeros((0, 0), dtype=np.int16)))
    occupied = (grid >= 100).astype(np.uint8)
    occupied_cell_count = int(np.sum(occupied > 0))
    if occupied_cell_count <= 0:
        return {
            "schema": "route6_building_polygon_v1",
            "house_id": str(house_id or ""),
            "source": "occupancy_connected_component",
            "coordinate_frame": "unreal_cm",
            "points": [],
            "bbox": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0},
            "quality": {"occupied_cell_count": 0, "component_area_m2": 0.0, "confidence": 0.0},
        }
    component_mask, selection = _select_component_mask(occupied, occupancy, target_bbox_unreal_cm=target_bbox_unreal_cm)
    contours, _hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(component_mask > 0)
        contour = np.array([[[int(xs.min()), int(ys.min())]], [[int(xs.max()), int(ys.min())]], [[int(xs.max()), int(ys.max())]], [[int(xs.min()), int(ys.max())]]], dtype=np.int32)
    else:
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.02 * cv2.arcLength(contour, True))
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if contour.reshape(-1, 2).shape[0] < 3:
            x, y, w, h = cv2.boundingRect(component_mask)
            contour = np.array([[[x, y]], [[x + w - 1, y]], [[x + w - 1, y + h - 1]], [[x, y + h - 1]]], dtype=np.int32)
    points = _contour_points_to_standard(contour, occupancy)
    bbox = _polygon_bbox(points)
    resolution = float(occupancy.get("resolution_m", 0.25))
    component_cells = int(np.sum(component_mask > 0))
    counts = np.asarray(occupancy.get("counts", np.zeros_like(grid, dtype=np.int32)))
    component_point_count = int(np.sum(counts[component_mask > 0])) if counts.shape == component_mask.shape else component_cells
    area_m2 = float(component_cells) * resolution * resolution
    point_score = min(1.0, float(component_point_count) / 5000.0)
    area_score = min(1.0, area_m2 / 10.0)
    cell_score = min(1.0, float(component_cells) / 20.0)
    confidence = max(0.0, min(1.0, 0.4 * cell_score + 0.3 * area_score + 0.3 * point_score))
    return {
        "schema": "route6_building_polygon_v1",
        "house_id": str(house_id or ""),
        "source": "occupancy_connected_component",
        "coordinate_frame": "unreal_cm",
        "points": points,
        "bbox": {key: round(float(value), 2) for key, value in bbox.items()},
        "quality": {
            "occupied_cell_count": component_cells,
            "point_count": component_point_count,
            "component_area_m2": round(area_m2, 4),
            "confidence": round(float(confidence), 4),
            **selection,
        },
    }


def build_corrected_map_config_from_polygon(
    map_config: Dict[str, Any],
    house_id: str,
    polygon: Dict[str, Any],
    *,
    min_confidence: float = 0.2,
    min_point_count: int = 5000,
    min_occupied_cell_count: int = 20,
    min_area_m2: float = 10.0,
    max_area_m2: float = 2000.0,
    max_allowed_shift_cm: float = 1500.0,
) -> Dict[str, Any]:
    corrected = copy.deepcopy(map_config)
    houses = corrected.get("houses", []) if isinstance(corrected.get("houses"), list) else []
    for house in houses:
        if not isinstance(house, dict) or _house_id(house) != str(house_id):
            continue
        bbox = polygon.get("bbox", {}) if isinstance(polygon.get("bbox"), dict) else {}
        if not bbox:
            house["route6_map_status"] = "candidate_only"
            house["route6_correction_rejected_reason"] = "missing_polygon_bbox"
            return corrected
        candidate_bbox = {key: round(_as_float(bbox.get(key)), 2) for key in ("min_x", "max_x", "min_y", "max_y")}
        house["route6_candidate_bbox_world"] = candidate_bbox
        center_x = 0.5 * (candidate_bbox["min_x"] + candidate_bbox["max_x"])
        center_y = 0.5 * (candidate_bbox["min_y"] + candidate_bbox["max_y"])
        old_x = _as_float(house.get("center_x"), center_x)
        old_y = _as_float(house.get("center_y"), center_y)
        shift = float(math.hypot(center_x - old_x, center_y - old_y))
        quality = polygon.get("quality", {}) if isinstance(polygon.get("quality"), dict) else {}
        confidence = _as_float(quality.get("confidence"), 0.0)
        point_count = _as_int(quality.get("point_count", 0), 0)
        occupied_cell_count = _as_int(quality.get("occupied_cell_count", 0), 0)
        area_m2 = _as_float(quality.get("component_area_m2", 0.0), 0.0)
        house["route6_map_confidence"] = round(float(confidence), 4)
        house["route6_center_shift_cm"] = round(float(shift), 2)
        house["route6_quality_gates"] = {
            "point_count": int(point_count),
            "min_point_count": int(min_point_count),
            "occupied_cell_count": int(occupied_cell_count),
            "min_occupied_cell_count": int(min_occupied_cell_count),
            "component_area_m2": round(float(area_m2), 4),
            "min_area_m2": round(float(min_area_m2), 4),
            "max_area_m2": round(float(max_area_m2), 4),
            "confidence": round(float(confidence), 4),
            "min_confidence": round(float(min_confidence), 4),
            "center_shift_cm": round(float(shift), 2),
            "max_allowed_shift_cm": round(float(max_allowed_shift_cm), 2),
        }
        if point_count < int(min_point_count):
            house["route6_map_status"] = "candidate_only"
            house["route6_correction_rejected_reason"] = "point_count_below_threshold"
            return corrected
        if occupied_cell_count < int(min_occupied_cell_count):
            house["route6_map_status"] = "candidate_only"
            house["route6_correction_rejected_reason"] = "occupied_cell_count_below_threshold"
            return corrected
        if area_m2 < float(min_area_m2):
            house["route6_map_status"] = "candidate_only"
            house["route6_correction_rejected_reason"] = "polygon_area_below_threshold"
            return corrected
        if area_m2 > float(max_area_m2):
            house["route6_map_status"] = "candidate_only"
            house["route6_correction_rejected_reason"] = "polygon_area_exceeds_threshold"
            return corrected
        if confidence < float(min_confidence):
            house["route6_map_status"] = "candidate_only"
            house["route6_correction_rejected_reason"] = "polygon_confidence_below_threshold"
            return corrected
        if shift > float(max_allowed_shift_cm):
            house["route6_map_status"] = "map_conflict"
            house["route6_correction_rejected_reason"] = "center_shift_exceeds_threshold"
            return corrected
        house["center_x"] = round(float(center_x), 2)
        house["center_y"] = round(float(center_y), 2)
        radius = max(
            300.0,
            0.5 * abs(candidate_bbox["max_x"] - candidate_bbox["min_x"]),
            0.5 * abs(candidate_bbox["max_y"] - candidate_bbox["min_y"]),
        )
        house["radius_cm"] = round(float(radius), 2)
        house["route6_corrected_bbox_world"] = candidate_bbox
        house["route6_map_status"] = "corrected"
        house.pop("route6_correction_rejected_reason", None)
        calibration = corrected.get("overhead_map", {}).get("calibration", {}) if isinstance(corrected.get("overhead_map"), dict) else {}
        if calibration:
            try:
                image_points = [
                    world_to_image_with_calibration(candidate_bbox["min_x"], candidate_bbox["min_y"], calibration),
                    world_to_image_with_calibration(candidate_bbox["max_x"], candidate_bbox["min_y"], calibration),
                    world_to_image_with_calibration(candidate_bbox["max_x"], candidate_bbox["max_y"], calibration),
                    world_to_image_with_calibration(candidate_bbox["min_x"], candidate_bbox["max_y"], calibration),
                ]
                xs = [point[0] for point in image_points]
                ys = [point[1] for point in image_points]
                house["map_bbox_image"] = {
                    "x1": round(float(min(xs)), 3),
                    "y1": round(float(min(ys)), 3),
                    "x2": round(float(max(xs)), 3),
                    "y2": round(float(max(ys)), 3),
                }
            except Exception as exc:
                house["route6_image_bbox_error"] = str(exc)
        return corrected
    return corrected


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.floating,)):
        return float(payload)
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_safe(value) for value in payload]
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_pointcloud_ply(path: Path, cloud: np.ndarray) -> None:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        arr = np.zeros((0, 6), dtype=np.float32)
    if arr.shape[1] < 6:
        pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr[:, :3], pad])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {int(arr.shape[0])}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for row in arr[:, :6]:
            red = int(max(0, min(255, round(float(row[3])))))
            green = int(max(0, min(255, round(float(row[4])))))
            blue = int(max(0, min(255, round(float(row[5])))))
            handle.write(f"{float(row[0]):.6f} {float(row[1]):.6f} {float(row[2]):.6f} {red} {green} {blue}\n")


def write_route6_map_artifacts(
    output_dir: Path,
    map_config: Dict[str, Any],
    house_id: str,
    cloud: np.ndarray,
    *,
    resolution_m: float = 0.25,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    map_dir = output_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    occupancy = build_occupancy_grid(cloud, resolution_m=resolution_m, occupied_threshold=1)
    houses = map_config.get("houses", []) if isinstance(map_config.get("houses"), list) else []
    target_house = next((item for item in houses if isinstance(item, dict) and _house_id(item) == str(house_id)), None)
    target_bbox = house_world_bbox(map_config, target_house) if isinstance(target_house, dict) else None
    polygon = extract_building_polygon(occupancy, house_id=house_id, target_bbox_unreal_cm=target_bbox)
    corrected = build_corrected_map_config_from_polygon(map_config, house_id, polygon)
    occupancy_path = map_dir / "route6_occupancy_grid.npy"
    occupancy_meta_path = map_dir / "route6_occupancy_grid.json"
    preview_path = map_dir / "route6_occupancy_grid.png"
    polygons_path = map_dir / "route6_polygons.json"
    corrected_path = map_dir / "route6_corrected_houses_config.json"
    quality_path = map_dir / "route6_map_quality_report.json"
    np.save(occupancy_path, np.asarray(occupancy["grid"], dtype=np.int16))
    write_json(occupancy_meta_path, occupancy_metadata(occupancy, house_id=house_id))
    cv2.imwrite(str(preview_path), occupancy_preview_image(occupancy))
    write_json(polygons_path, {"schema": "route6_polygons_v1", "polygons": [polygon], "updated_at": datetime.now().isoformat(timespec="seconds")})
    write_json(corrected_path, corrected)
    houses = corrected.get("houses", []) if isinstance(corrected.get("houses"), list) else []
    mapped = [
        house for house in houses
        if isinstance(house, dict) and str(house.get("route6_map_status", "")) in {"corrected", "candidate_only"}
    ]
    quality = {
        "schema": "route6_map_quality_report_v1",
        "target_house_id": str(house_id),
        "house_count_total": len(houses),
        "house_count_mapped": len(mapped),
        "global_occupied_cell_count": int(np.sum(np.asarray(occupancy["grid"]) == 100)),
        "global_polygon_count": 1 if polygon.get("points") else 0,
        "mean_map_confidence": round(float((polygon.get("quality", {}) if isinstance(polygon.get("quality"), dict) else {}).get("confidence", 0.0)), 4),
        "corrected_config_path": str(corrected_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(quality_path, quality)
    return {
        "occupancy_grid_path": str(occupancy_path),
        "occupancy_metadata_path": str(occupancy_meta_path),
        "occupancy_preview_path": str(preview_path),
        "polygons_path": str(polygons_path),
        "corrected_config_path": str(corrected_path),
        "quality_report_path": str(quality_path),
        "polygon": polygon,
        "quality_report": quality,
    }
