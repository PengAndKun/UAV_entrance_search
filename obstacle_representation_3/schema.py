from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


STOP_DEPTH_CM = 100.0
WARNING_DEPTH_CM = 250.0
CLEARANCE_DEPTH_CM = 450.0
DANGER_DEPTH_CM = 250.0
MAX_DEPTH_CM = 1200.0

GEOMETRY_FEATURE_NAMES: List[str] = [
    "front_min_depth_cm",
    "front_mean_depth_cm",
    "left_min_depth_cm",
    "right_min_depth_cm",
    "up_min_depth_cm",
    "down_min_depth_cm",
    "nearest_distance_cm",
    "obstacle_width_cm",
    "obstacle_height_cm",
    "obstacle_thickness_cm",
    "corridor_count",
    "valid_depth_count",
    "invalid_depth_count",
    "down_p05_depth_cm",
    "down_close_fraction",
    "distance_to_goal_cm",
    "bearing_deg_body",
    "dz_cm",
    "forward_swept_clear",
    "left_swept_clear",
    "right_swept_clear",
    "up_swept_clear",
    "down_swept_clear",
    "backoff_swept_clear",
]

DIRECTION_LABELS = ("forward", "left", "right", "up", "backoff", "hold")
DIRECTION_TO_INDEX = {label: idx for idx, label in enumerate(DIRECTION_LABELS)}
INDEX_TO_DIRECTION = {idx: label for label, idx in DIRECTION_TO_INDEX.items()}

MASK_CHANNELS = ("clearance_warning", "obstacle_warning", "must_stop")
RISK_STATES = ("clear", "clearance_warning", "obstacle_warning", "must_stop")
RISK_TO_INDEX = {label: idx for idx, label in enumerate(RISK_STATES)}
INDEX_TO_RISK = {idx: label for label, idx in RISK_TO_INDEX.items()}

PROJECTION_BOX = {
    "x0": 0.42,
    "x1": 0.58,
    "y0": 0.38,
    "y1": 0.72,
    "stop_fraction_threshold": 0.01,
}


def projection_box_dict() -> Dict[str, float]:
    return {key: float(value) for key, value in PROJECTION_BOX.items()}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "clear", "safe", "passed"}:
            return True
        if text in {"0", "false", "no", "n", "blocked", "unsafe", "failed"}:
            return False
    return default


def normalize_depth_cm(depth: np.ndarray, *, max_depth_cm: float = MAX_DEPTH_CM) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = np.clip(depth[valid], 0.0, float(max_depth_cm)) / float(max_depth_cm)
    return cleaned


def geometry_feature_dict(event: Dict[str, Any]) -> Dict[str, float]:
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    rel = event.get("relative_target") if isinstance(event.get("relative_target"), dict) else {}
    values = {
        "front_min_depth_cm": as_float(summary.get("front_min_depth_cm")),
        "front_mean_depth_cm": as_float(summary.get("front_mean_depth_cm")),
        "left_min_depth_cm": as_float(summary.get("left_min_depth_cm")),
        "right_min_depth_cm": as_float(summary.get("right_min_depth_cm")),
        "up_min_depth_cm": as_float(summary.get("up_min_depth_cm")),
        "down_min_depth_cm": as_float(summary.get("down_min_depth_cm")),
        "nearest_distance_cm": as_float(summary.get("nearest_distance_cm")),
        "obstacle_width_cm": as_float(summary.get("obstacle_width_cm")),
        "obstacle_height_cm": as_float(summary.get("obstacle_height_cm")),
        "obstacle_thickness_cm": as_float(summary.get("obstacle_thickness_cm")),
        "corridor_count": as_float(summary.get("corridor_count")),
        "valid_depth_count": as_float(summary.get("valid_depth_count")),
        "invalid_depth_count": as_float(summary.get("invalid_depth_count")),
        "down_p05_depth_cm": as_float(summary.get("down_p05_depth_cm")),
        "down_close_fraction": as_float(summary.get("down_close_fraction")),
        "distance_to_goal_cm": as_float(event.get("distance_to_goal_cm", rel.get("distance_cm"))),
        "bearing_deg_body": as_float(event.get("bearing_deg_body", rel.get("bearing_deg_body"))),
        "dz_cm": as_float(rel.get("dz_cm")),
        "forward_swept_clear": 1.0 if as_bool(summary.get("forward_swept_clear", True)) else 0.0,
        "left_swept_clear": 1.0 if as_bool(summary.get("left_swept_clear", True)) else 0.0,
        "right_swept_clear": 1.0 if as_bool(summary.get("right_swept_clear", True)) else 0.0,
        "up_swept_clear": 1.0 if as_bool(summary.get("up_swept_clear", True)) else 0.0,
        "down_swept_clear": 1.0 if as_bool(summary.get("down_swept_clear", False)) else 0.0,
        "backoff_swept_clear": 1.0 if as_bool(summary.get("backoff_swept_clear", True)) else 0.0,
    }
    return {name: float(values.get(name, 0.0)) for name in GEOMETRY_FEATURE_NAMES}


def geometry_vector(event: Dict[str, Any]) -> np.ndarray:
    values = geometry_feature_dict(event)
    return np.asarray([values[name] for name in GEOMETRY_FEATURE_NAMES], dtype=np.float32)


def event_geometry_vector(event: Dict[str, Any]) -> np.ndarray:
    return geometry_vector(event).astype(np.float32, copy=False)
