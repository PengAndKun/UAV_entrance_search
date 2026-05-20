from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np


OBSTACLE_LABELS = (
    "open_path",
    "tree_trunk_or_pole",
    "tree_canopy_or_cluster",
    "fence_or_rail",
    "building",
    "mixed",
    "unknown",
)
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(OBSTACLE_LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

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
        return value.strip().lower() in {"1", "true", "yes", "y", "clear", "passed"}
    return False


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


def counts(items: Iterable[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in items:
        result[str(item)] = result.get(str(item), 0) + 1
    return dict(sorted(result.items(), key=lambda kv: kv[0]))
