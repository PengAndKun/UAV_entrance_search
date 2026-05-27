from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


DEFAULT_METHOD_ID = "obstacle_representation_direction_rule_v1"
STOP_FORWARD_FRACTION = 0.01
MOSTLY_STOP_FRACTION = 0.35
SIDE_BLOCKED_FRACTION = 0.35
FRONT_STOP_DEPTH_CM = 100.0
RED_RECOVERY_FRONT_CLEAR_CM = 320.0
RED_RECOVERY_MIN_TICKS = 4
RED_RECOVERY_CLEAR_TICKS = 2
RED_RECOVERY_SIDE_STEP_CM = 45.0
RED_RECOVERY_BACKOFF_CM = 35.0
RED_RECOVERY_VERTICAL_CM = 35.0


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _mask(prediction: Dict[str, Any], key: str) -> np.ndarray:
    value = prediction.get(key)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((96, 96), dtype=np.float32)
    return arr


def _resize_like(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.float32, copy=False)
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(mask.astype(np.float32), mode="F").resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)
    except Exception:
        return np.zeros(shape, dtype=np.float32)


def _region(mask: np.ndarray, *, y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    h, w = mask.shape[:2]
    ya = max(0, min(h, int(round(h * y0))))
    yb = max(ya + 1, min(h, int(round(h * y1))))
    xa = max(0, min(w, int(round(w * x0))))
    xb = max(xa + 1, min(w, int(round(w * x1))))
    return mask[ya:yb, xa:xb]


def _fraction(mask: np.ndarray, *, y0: float, y1: float, x0: float, x1: float) -> float:
    values = _region(mask, y0=y0, y1=y1, x0=x0, x1=x1)
    return float(np.mean(values >= 0.5)) if values.size else 0.0


def corridor_risk_stats(prediction: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    clearance = _mask(prediction, "clearance_warning_mask")
    warning = _mask(prediction, "obstacle_warning_mask")
    stop = _mask(prediction, "must_stop_mask")
    warning = _resize_like(warning, stop.shape)
    clearance = _resize_like(clearance, stop.shape)
    corridors = {
        "front_center": (0.26, 0.82, 0.34, 0.66),
        "left_corridor": (0.26, 0.82, 0.00, 0.34),
        "right_corridor": (0.26, 0.82, 0.66, 1.00),
        "up_corridor": (0.00, 0.36, 0.24, 0.76),
    }
    stats: Dict[str, Dict[str, float]] = {}
    for name, (y0, y1, x0, x1) in corridors.items():
        stop_fraction = _fraction(stop, y0=y0, y1=y1, x0=x0, x1=x1)
        warning_fraction = _fraction(warning, y0=y0, y1=y1, x0=x0, x1=x1)
        clearance_fraction = _fraction(clearance, y0=y0, y1=y1, x0=x0, x1=x1)
        stats[name] = {
            "stop_fraction": round(stop_fraction, 6),
            "warning_fraction": round(warning_fraction, 6),
            "clearance_fraction": round(clearance_fraction, 6),
            "risk_cost": round(stop_fraction * 3.0 + warning_fraction * 0.8 + clearance_fraction * 0.25, 6),
        }
    return stats


def _corridor_score(stats: Dict[str, float]) -> float:
    stop_fraction = as_float(stats.get("stop_fraction"))
    warning_fraction = as_float(stats.get("warning_fraction"))
    clearance_fraction = as_float(stats.get("clearance_fraction"))
    return max(0.0, 1.0 - stop_fraction * 3.0 - warning_fraction * 0.8 - clearance_fraction * 0.25)


def deep_red_stop_active(
    prediction: Dict[str, Any],
    pointcloud_summary: Dict[str, Any],
    corridor_risks: Dict[str, Dict[str, float]] | None = None,
) -> bool:
    summary = pointcloud_summary if isinstance(pointcloud_summary, dict) else {}
    stats = corridor_risks if isinstance(corridor_risks, dict) else corridor_risk_stats(prediction)
    front = stats.get("front_center", {}) if isinstance(stats.get("front_center"), dict) else {}
    front_min = as_float(summary.get("front_min_depth_cm"))
    return bool(
        str(prediction.get("front_risk_state", "") or "").lower() == "must_stop"
        or as_float(front.get("stop_fraction")) > STOP_FORWARD_FRACTION
        or (front_min > 0.0 and front_min <= FRONT_STOP_DEPTH_CM)
    )


def red_recovery_clear_enough(
    prediction: Dict[str, Any],
    pointcloud_summary: Dict[str, Any],
    corridor_risks: Dict[str, Dict[str, float]] | None = None,
) -> bool:
    summary = pointcloud_summary if isinstance(pointcloud_summary, dict) else {}
    stats = corridor_risks if isinstance(corridor_risks, dict) else corridor_risk_stats(prediction)
    front = stats.get("front_center", {}) if isinstance(stats.get("front_center"), dict) else {}
    front_min = as_float(summary.get("front_min_depth_cm"))
    return bool(
        front_min >= RED_RECOVERY_FRONT_CLEAR_CM
        and as_float(front.get("stop_fraction")) <= STOP_FORWARD_FRACTION
        and str(prediction.get("front_risk_state", "") or "").lower() != "must_stop"
    )


def choose_red_recovery_direction(rule: Dict[str, Any]) -> str:
    selected = str(rule.get("selected_direction", "") or "").lower()
    if selected in {"left", "right", "up", "backoff"}:
        return selected
    scores = rule.get("candidate_action_scores") if isinstance(rule.get("candidate_action_scores"), dict) else {}
    candidates = ("left", "right", "up", "backoff")
    return max(candidates, key=lambda name: as_float(scores.get(name)))


def select_or2_direction(
    prediction: Dict[str, Any],
    pointcloud_summary: Dict[str, Any],
    relative_target: Dict[str, Any] | None = None,
    *,
    last_action: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rel = relative_target if isinstance(relative_target, dict) else {}
    summary = pointcloud_summary if isinstance(pointcloud_summary, dict) else {}
    stats = corridor_risk_stats(prediction)
    front = stats["front_center"]
    left = stats["left_corridor"]
    right = stats["right_corridor"]
    up = stats["up_corridor"]
    front_min = as_float(summary.get("front_min_depth_cm"))
    front_risk_state = str(prediction.get("front_risk_state", "") or "").lower()
    front_blocked = bool(
        front_risk_state == "must_stop"
        or as_float(front.get("stop_fraction")) > STOP_FORWARD_FRACTION
        or (front_min > 0.0 and front_min <= FRONT_STOP_DEPTH_CM)
    )
    bearing = as_float(rel.get("bearing_deg_body"))
    dz = as_float(rel.get("dz_cm"))
    scores: Dict[str, float] = {
        "hold": 0.05,
        "backoff": 0.20,
        "forward": 0.0,
        "slow_forward": 0.0,
        "left": _corridor_score(left),
        "right": _corridor_score(right),
        "up": _corridor_score(up),
    }
    if not front_blocked:
        front_score = _corridor_score(front)
        goal_bonus = max(0.0, 0.18 - min(abs(bearing), 90.0) / 90.0 * 0.18)
        scores["forward"] = max(0.0, front_score + goal_bonus)
        scores["slow_forward"] = max(0.0, front_score + 0.08)
    if bearing < -12.0:
        scores["left"] += min(0.18, abs(bearing) / 180.0 * 0.32)
        scores["right"] -= 0.05
    elif bearing > 12.0:
        scores["right"] += min(0.18, abs(bearing) / 180.0 * 0.32)
        scores["left"] -= 0.05
    if dz > 45.0:
        scores["up"] += 0.12
    previous_name = str((last_action or {}).get("action_name", "")).lower()
    if "left" in previous_name and as_float(left.get("stop_fraction")) < SIDE_BLOCKED_FRACTION:
        scores["left"] += 0.05
    if "right" in previous_name and as_float(right.get("stop_fraction")) < SIDE_BLOCKED_FRACTION:
        scores["right"] += 0.05
    for name in ("left", "right", "up"):
        scores[name] = max(0.0, scores[name])

    all_escape_deep_red = all(
        as_float(region.get("stop_fraction")) >= MOSTLY_STOP_FRACTION
        for region in (left, right, up)
    )
    if front_blocked and all_escape_deep_red:
        scores["backoff"] = max(scores["backoff"], 0.75)
    elif front_blocked:
        scores["backoff"] = 0.10

    if front_blocked:
        candidates = ("left", "right", "up", "backoff", "hold")
    else:
        candidates = ("forward", "slow_forward", "left", "right", "up", "hold")
    selected = max(candidates, key=lambda name: (scores.get(name, 0.0), -candidates.index(name)))
    if front_blocked and selected in {"hold"}:
        selected = "backoff" if scores["backoff"] >= max(scores["left"], scores["right"], scores["up"]) else selected
    if front_blocked and selected in {"forward", "slow_forward"}:
        selected = "hold"
    reason = (
        f"or2_direction_rule selected={selected}; front_blocked={front_blocked}; "
        f"front_stop={front['stop_fraction']:.3f}; left_stop={left['stop_fraction']:.3f}; "
        f"right_stop={right['stop_fraction']:.3f}; up_stop={up['stop_fraction']:.3f}; "
        f"front_min={front_min:.1f}cm bearing={bearing:.1f} dz={dz:.1f}"
    )
    return {
        "selected_direction": selected,
        "candidate_action_scores": {key: round(float(value), 6) for key, value in scores.items()},
        "corridor_risks": stats,
        "front_blocked": bool(front_blocked),
        "reason": reason,
    }
