"""Obstacle Representation 2: RGB-D affordance and avoidance direction model."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .schema import (
    CLEARANCE_DEPTH_CM,
    DANGER_DEPTH_CM,
    DIRECTION_LABELS,
    DIRECTION_TO_INDEX,
    GEOMETRY_FEATURE_NAMES,
    MASK_CHANNELS,
)
from .teacher import compute_affordance_teacher

__all__ = [
    "CLEARANCE_DEPTH_CM",
    "DANGER_DEPTH_CM",
    "DIRECTION_LABELS",
    "DIRECTION_TO_INDEX",
    "GEOMETRY_FEATURE_NAMES",
    "MASK_CHANNELS",
    "compute_affordance_teacher",
]
