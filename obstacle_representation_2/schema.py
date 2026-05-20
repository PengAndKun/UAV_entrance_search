from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from obstacle_representation.schema import GEOMETRY_FEATURE_NAMES, geometry_vector


DANGER_DEPTH_CM = 250.0
CLEARANCE_DEPTH_CM = 450.0
MAX_DEPTH_CM = 1200.0

MASK_CHANNELS = ("danger", "insufficient_clearance")
DIRECTION_LABELS = ("forward", "left", "right", "up", "backoff", "hold")
DIRECTION_TO_INDEX = {label: idx for idx, label in enumerate(DIRECTION_LABELS)}
INDEX_TO_DIRECTION = {idx: label for label, idx in DIRECTION_TO_INDEX.items()}


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


def event_geometry_vector(event: Dict[str, Any]) -> np.ndarray:
    return geometry_vector(event).astype(np.float32, copy=False)


def empty_summary() -> Dict[str, Any]:
    return {
        "front_min_depth_cm": 0.0,
        "left_min_depth_cm": 0.0,
        "right_min_depth_cm": 0.0,
        "up_min_depth_cm": 0.0,
        "forward_swept_clear": True,
        "left_swept_clear": True,
        "right_swept_clear": True,
        "up_swept_clear": True,
        "obstacle_geometry": "unknown",
    }
