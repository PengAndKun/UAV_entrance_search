"""Obstacle Representation 3: projection-box constrained RGB-D front risk model."""

from .schema import (
    CLEARANCE_DEPTH_CM,
    MASK_CHANNELS,
    PROJECTION_BOX,
    RISK_STATES,
    RISK_TO_INDEX,
    STOP_DEPTH_CM,
    WARNING_DEPTH_CM,
)
from .teacher import (
    compute_affordance_teacher,
    depth_masks,
    front_risk_from_masks,
    projection_box_slices,
    projection_box_stats,
)

__all__ = [
    "CLEARANCE_DEPTH_CM",
    "MASK_CHANNELS",
    "PROJECTION_BOX",
    "RISK_STATES",
    "RISK_TO_INDEX",
    "STOP_DEPTH_CM",
    "WARNING_DEPTH_CM",
    "compute_affordance_teacher",
    "depth_masks",
    "front_risk_from_masks",
    "projection_box_slices",
    "projection_box_stats",
]
