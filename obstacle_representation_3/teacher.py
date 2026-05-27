from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .schema import (
    CLEARANCE_DEPTH_CM,
    PROJECTION_BOX,
    RISK_TO_INDEX,
    STOP_DEPTH_CM,
    WARNING_DEPTH_CM,
    as_float,
    projection_box_dict,
)


def resize_float_image(image: np.ndarray, image_size: int, *, nearest: bool = False) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    if data.shape == (int(image_size), int(image_size)):
        return data.astype(np.float32, copy=False)
    if data.ndim != 2 or data.size == 0:
        return np.zeros((int(image_size), int(image_size)), dtype=np.float32)

    src_h, src_w = data.shape
    dst = int(image_size)
    y = np.linspace(0.0, float(src_h - 1), dst, dtype=np.float32)
    x = np.linspace(0.0, float(src_w - 1), dst, dtype=np.float32)
    if nearest:
        yi = np.rint(y).astype(np.intp)
        xi = np.rint(x).astype(np.intp)
        return data[yi[:, None], xi[None, :]].astype(np.float32, copy=False)

    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (y - y0.astype(np.float32))[:, None]
    wx = (x - x0.astype(np.float32))[None, :]
    top = data[y0[:, None], x0[None, :]] * (1.0 - wx) + data[y0[:, None], x1[None, :]] * wx
    bottom = data[y1[:, None], x0[None, :]] * (1.0 - wx) + data[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32, copy=False)


def depth_masks(depth_cm: np.ndarray, image_size: int) -> np.ndarray:
    depth = np.squeeze(np.asarray(depth_cm, dtype=np.float32))
    if depth.ndim != 2:
        depth = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
    depth = resize_float_image(depth, image_size, nearest=False)
    valid = np.isfinite(depth) & (depth > 0.0)
    must_stop = valid & (depth <= STOP_DEPTH_CM)
    obstacle_warning = valid & (depth > STOP_DEPTH_CM) & (depth <= WARNING_DEPTH_CM)
    clearance_warning = valid & (depth > WARNING_DEPTH_CM) & (depth <= CLEARANCE_DEPTH_CM)
    return np.stack(
        [
            clearance_warning.astype(np.float32),
            obstacle_warning.astype(np.float32),
            must_stop.astype(np.float32),
        ],
        axis=0,
    )


def projection_box_slices(
    shape_or_size: Tuple[int, int] | int,
    projection_box: Dict[str, float] | None = None,
) -> Tuple[slice, slice]:
    box = projection_box or PROJECTION_BOX
    if isinstance(shape_or_size, int):
        height = width = int(shape_or_size)
    else:
        height, width = int(shape_or_size[0]), int(shape_or_size[1])
    x0 = max(0, min(width, int(round(width * float(box["x0"])))))
    x1 = max(0, min(width, int(round(width * float(box["x1"])))))
    y0 = max(0, min(height, int(round(height * float(box["y0"])))))
    y1 = max(0, min(height, int(round(height * float(box["y1"])))))
    return slice(y0, y1), slice(x0, x1)


def _fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def projection_box_stats(masks: np.ndarray, projection_box: Dict[str, float] | None = None) -> Dict[str, Any]:
    y_slice, x_slice = projection_box_slices(masks.shape[-2:], projection_box)
    region = masks[:, y_slice, x_slice]
    pixel_count = int(region.shape[-2] * region.shape[-1]) if region.ndim == 3 else 0
    return {
        "front_box_clearance_fraction": _fraction(region[0] > 0.5) if pixel_count else 0.0,
        "front_box_warning_fraction": _fraction(region[1] > 0.5) if pixel_count else 0.0,
        "front_box_stop_fraction": _fraction(region[2] > 0.5) if pixel_count else 0.0,
        "front_box_pixel_count": pixel_count,
    }


def front_risk_from_masks(event: Dict[str, Any], masks: np.ndarray) -> Dict[str, Any]:
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    front_min = as_float(summary.get("front_min_depth_cm"))
    box = projection_box_stats(masks)
    full_clearance_fraction = _fraction(masks[0] > 0.5)
    full_warning_fraction = _fraction(masks[1] > 0.5)
    full_stop_fraction = _fraction(masks[2] > 0.5)

    state = "clear"
    if box["front_box_stop_fraction"] > float(PROJECTION_BOX["stop_fraction_threshold"]):
        state = "must_stop"
    elif full_stop_fraction > 0.0 or (front_min > 0.0 and front_min <= STOP_DEPTH_CM):
        state = "obstacle_warning"
    elif box["front_box_warning_fraction"] > 0.0 or (front_min > STOP_DEPTH_CM and front_min <= WARNING_DEPTH_CM):
        state = "obstacle_warning"
    elif box["front_box_clearance_fraction"] > 0.0 or (front_min > WARNING_DEPTH_CM and front_min <= CLEARANCE_DEPTH_CM):
        state = "clearance_warning"

    return {
        "front_risk_state": state,
        "front_risk_index": int(RISK_TO_INDEX[state]),
        "can_forward": state != "must_stop",
        "must_stop": state == "must_stop",
        "front_clearance_fraction": float(box["front_box_clearance_fraction"]),
        "front_warning_fraction": float(box["front_box_warning_fraction"]),
        "front_stop_fraction": float(box["front_box_stop_fraction"]),
        "full_stop_fraction": float(full_stop_fraction),
        "full_warning_fraction": float(full_warning_fraction),
        "full_clearance_fraction": float(full_clearance_fraction),
        **box,
    }


def compute_affordance_teacher(event: Dict[str, Any], depth_cm: np.ndarray, image_size: int = 96) -> Dict[str, Any]:
    masks = depth_masks(depth_cm, image_size)
    risk = front_risk_from_masks(event, masks)
    return {
        "masks": masks.astype(np.float32, copy=False),
        "front_risk_state": risk["front_risk_state"],
        "front_risk_index": risk["front_risk_index"],
        "can_forward": bool(risk["can_forward"]),
        "must_stop": bool(risk["must_stop"]),
        "projection_box": projection_box_dict(),
        "front_box_clearance_fraction": risk["front_box_clearance_fraction"],
        "front_box_warning_fraction": risk["front_box_warning_fraction"],
        "front_box_stop_fraction": risk["front_box_stop_fraction"],
        "front_box_pixel_count": risk["front_box_pixel_count"],
        "front_clearance_fraction": risk["front_clearance_fraction"],
        "front_warning_fraction": risk["front_warning_fraction"],
        "front_stop_fraction": risk["front_stop_fraction"],
        "full_stop_fraction": risk["full_stop_fraction"],
        "full_warning_fraction": risk["full_warning_fraction"],
        "full_clearance_fraction": risk["full_clearance_fraction"],
        "teacher_source": "projection_box_depth_teacher_v1",
    }


__all__ = [
    "MASK_CHANNELS",
    "compute_affordance_teacher",
    "depth_masks",
    "front_risk_from_masks",
    "projection_box_slices",
    "projection_box_stats",
    "resize_float_image",
]
