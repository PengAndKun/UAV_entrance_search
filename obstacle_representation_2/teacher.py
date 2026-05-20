from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from .schema import (
    CLEARANCE_DEPTH_CM,
    DANGER_DEPTH_CM,
    DIRECTION_LABELS,
    DIRECTION_TO_INDEX,
    MASK_CHANNELS,
    as_bool,
    as_float,
)


def resize_float_image(image: np.ndarray, image_size: int, *, nearest: bool = False) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.float32), mode="F")
    resample = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(pil.resize((int(image_size), int(image_size)), resample), dtype=np.float32)


def depth_masks(depth_cm: np.ndarray, image_size: int) -> np.ndarray:
    depth = np.squeeze(np.asarray(depth_cm, dtype=np.float32))
    if depth.ndim != 2:
        depth = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
    depth = resize_float_image(depth, image_size, nearest=False)
    valid = np.isfinite(depth) & (depth > 0.0)
    danger = valid & (depth <= DANGER_DEPTH_CM)
    insufficient = valid & (depth > DANGER_DEPTH_CM) & (depth <= CLEARANCE_DEPTH_CM)
    return np.stack([danger.astype(np.float32), insufficient.astype(np.float32)], axis=0)


def _sector(mask: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    h, w = mask.shape[:2]
    ix0, ix1 = max(0, int(round(w * x0))), min(w, int(round(w * x1)))
    iy0, iy1 = max(0, int(round(h * y0))), min(h, int(round(h * y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return mask[0:0, 0:0]
    return mask[iy0:iy1, ix0:ix1]


def _fraction(mask: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    region = _sector(mask, x0, x1, y0, y1)
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region)) / float(region.size)


def _semantic_text(event: Dict[str, Any]) -> str:
    parts = []
    for key in ("environment_id", "obstacle_hint", "obstacle_geometry_label", "scenario_id", "operator_note"):
        parts.append(str(event.get(key, "") or ""))
    for key in ("llm_strategy", "llm_analysis_result"):
        data = event.get(key)
        if isinstance(data, dict):
            parts.extend(str(data.get(item, "") or "") for item in ("environment_id", "obstacle_hint", "strategy_reason"))
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    parts.append(str(summary.get("obstacle_geometry", "") or ""))
    return re.sub(r"[^a-z0-9]+", "_", " ".join(parts).lower())


def _depth_score(value_cm: float, swept_clear: bool, danger_fraction: float, yellow_fraction: float) -> float:
    if not swept_clear:
        return 0.0
    score = 0.25
    if value_cm <= 0.0:
        score = 0.55
    elif value_cm < DANGER_DEPTH_CM:
        score = 0.0
    elif value_cm < CLEARANCE_DEPTH_CM:
        score = 0.30 + 0.25 * ((value_cm - DANGER_DEPTH_CM) / max(1.0, CLEARANCE_DEPTH_CM - DANGER_DEPTH_CM))
    else:
        score = 0.65 + 0.35 * min(1.0, (value_cm - CLEARANCE_DEPTH_CM) / 550.0)
    score -= 0.85 * danger_fraction + 0.35 * yellow_fraction
    return float(max(0.0, min(1.0, score)))


def compute_direction_scores(event: Dict[str, Any], masks: np.ndarray) -> Dict[str, float]:
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    danger = masks[0] > 0.5
    yellow = masks[1] > 0.5

    front_red = _fraction(danger, 0.34, 0.66, 0.26, 0.82)
    left_red = _fraction(danger, 0.05, 0.38, 0.30, 0.86)
    right_red = _fraction(danger, 0.62, 0.95, 0.30, 0.86)
    up_red = _fraction(danger, 0.28, 0.72, 0.04, 0.36)
    front_yellow = _fraction(yellow, 0.34, 0.66, 0.26, 0.82)
    left_yellow = _fraction(yellow, 0.05, 0.38, 0.30, 0.86)
    right_yellow = _fraction(yellow, 0.62, 0.95, 0.30, 0.86)
    up_yellow = _fraction(yellow, 0.28, 0.72, 0.04, 0.36)

    front_min = as_float(summary.get("front_min_depth_cm"))
    left_min = as_float(summary.get("left_min_depth_cm"))
    right_min = as_float(summary.get("right_min_depth_cm"))
    up_min = as_float(summary.get("up_min_depth_cm"))
    width_cm = as_float(summary.get("obstacle_width_cm"))
    text = _semantic_text(event)

    scores = {
        "forward": _depth_score(front_min, as_bool(summary.get("forward_swept_clear", True), True), front_red, front_yellow),
        "left": _depth_score(left_min, as_bool(summary.get("left_swept_clear", True), True), left_red, left_yellow),
        "right": _depth_score(right_min, as_bool(summary.get("right_swept_clear", True), True), right_red, right_yellow),
        "up": _depth_score(up_min, as_bool(summary.get("up_swept_clear", True), True), up_red, up_yellow),
        "backoff": 0.38 if as_bool(summary.get("backoff_swept_clear", True), True) else 0.05,
        "hold": 0.15,
    }

    front_blocked = is_front_blocked(event, masks)
    wide_or_vertical = any(token in text for token in ("fence", "rail", "building", "wall", "roof", "low_obstacle", "vertical_wall"))
    thin_or_tree = any(token in text for token in ("tree_trunk", "pole", "thin_structure", "trunk"))

    if not front_blocked:
        scores["forward"] = max(scores["forward"], 0.78)
        scores["up"] *= 0.72
        scores["backoff"] *= 0.55
    else:
        scores["forward"] = 0.0
        if wide_or_vertical or width_cm >= 220.0:
            scores["up"] = max(scores["up"], 0.86 if scores["up"] > 0.05 else 0.0)
        if thin_or_tree and width_cm < 240.0:
            best_side = "left" if scores["left"] >= scores["right"] else "right"
            scores[best_side] = max(scores[best_side], 0.84)
            scores["up"] *= 0.82
        if scores["left"] <= 0.05 and scores["right"] <= 0.05 and scores["up"] <= 0.05:
            scores["backoff"] = max(scores["backoff"], 0.82)

    bearing = as_float(event.get("bearing_deg_body", (event.get("relative_target") or {}).get("bearing_deg_body") if isinstance(event.get("relative_target"), dict) else 0.0))
    if not front_blocked:
        if bearing < -8.0:
            scores["left"] = min(1.0, scores["left"] + 0.08)
        elif bearing > 8.0:
            scores["right"] = min(1.0, scores["right"] + 0.08)

    distance = as_float(event.get("distance_to_goal_cm", (event.get("relative_target") or {}).get("distance_cm") if isinstance(event.get("relative_target"), dict) else 0.0))
    if 0.0 < distance <= 180.0:
        scores = {key: value * 0.25 for key, value in scores.items()}
        scores["hold"] = 1.0

    return {label: float(max(0.0, min(1.0, scores.get(label, 0.0)))) for label in DIRECTION_LABELS}


def is_front_blocked(event: Dict[str, Any], masks: np.ndarray) -> bool:
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    front_min = as_float(summary.get("front_min_depth_cm"))
    red_fraction = _fraction(masks[0] > 0.5, 0.34, 0.66, 0.26, 0.82)
    yellow_fraction = _fraction(masks[1] > 0.5, 0.34, 0.66, 0.26, 0.82)
    return bool(
        red_fraction >= 0.025
        or yellow_fraction >= 0.22
        or (front_min > 0.0 and front_min < DANGER_DEPTH_CM)
        or not as_bool(summary.get("forward_swept_clear", True), True)
    )


def select_direction(scores: Dict[str, float], event: Dict[str, Any], masks: np.ndarray) -> str:
    front_blocked = is_front_blocked(event, masks)
    candidates = list(DIRECTION_LABELS)
    if front_blocked:
        candidates = [item for item in candidates if item != "forward"]
    best = max(candidates, key=lambda item: (float(scores.get(item, 0.0)), -DIRECTION_TO_INDEX[item]))
    return best if scores.get(best, 0.0) > 0.05 else "hold"


def flyover_delta_cm_for_direction(direction: str, event: Dict[str, Any]) -> float:
    if direction != "up":
        return 0.0
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    height = as_float(summary.get("obstacle_height_cm"))
    up_min = as_float(summary.get("up_min_depth_cm"))
    if height >= 450.0:
        return 320.0
    if height >= 220.0:
        return 180.0
    if up_min > 0.0 and up_min < CLEARANCE_DEPTH_CM:
        return 90.0
    return 120.0


def compute_affordance_teacher(event: Dict[str, Any], depth_cm: np.ndarray, image_size: int = 96) -> Dict[str, Any]:
    masks = depth_masks(depth_cm, image_size)
    scores = compute_direction_scores(event, masks)
    direction = select_direction(scores, event, masks)
    front_blocked = is_front_blocked(event, masks)
    red_fraction = _fraction(masks[0] > 0.5, 0.34, 0.66, 0.26, 0.82)
    yellow_fraction = _fraction(masks[1] > 0.5, 0.34, 0.66, 0.26, 0.82)
    return {
        "masks": masks.astype(np.float32, copy=False),
        "direction_label": direction,
        "direction_index": int(DIRECTION_TO_INDEX[direction]),
        "direction_scores": {label: float(scores[label]) for label in DIRECTION_LABELS},
        "direction_scores_vector": np.asarray([scores[label] for label in DIRECTION_LABELS], dtype=np.float32),
        "flyover_delta_cm": float(flyover_delta_cm_for_direction(direction, event)),
        "red_front_blocked": bool(front_blocked),
        "front_red_fraction": float(red_fraction),
        "front_insufficient_fraction": float(yellow_fraction),
        "teacher_source": "depth_pointcloud_vlm_cached_affordance",
    }
