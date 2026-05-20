from __future__ import annotations

import re
from typing import Any, Dict

from .schema import OBSTACLE_LABELS, as_bool, as_float


def _summary(event: Dict[str, Any]) -> Dict[str, Any]:
    value = event.get("pointcloud_summary")
    return value if isinstance(value, dict) else {}


def _strategy(event: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("llm_strategy", "llm_analysis_result"):
        value = event.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _text_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def canonical_obstacle_label(value: Any, event: Dict[str, Any] | None = None) -> str:
    event = event if isinstance(event, dict) else {}
    summary = _summary(event)
    raw = str(value or "").strip()
    text = _text_key(raw)
    width_cm = as_float(summary.get("obstacle_width_cm"))
    geometry = str(summary.get("obstacle_geometry", "") or "").lower()

    if text in OBSTACLE_LABELS:
        return text
    if text in {"none", "no_obstacle", "clear", "open", "open_corridor"}:
        return "open_path"
    if any(token in text for token in ("fence", "rail", "railing", "barrier", "low_horizontal")):
        return "fence_or_rail"
    if any(token in text for token in ("building", "roof", "house", "wall", "facade", "entrance")):
        return "building"
    if any(token in text for token in ("mixed", "multiple")):
        return "mixed"
    if any(token in text for token in ("canopy", "cluster", "branches", "branch", "bush")):
        return "tree_canopy_or_cluster"
    if any(token in text for token in ("trunk", "pole", "tree", "thin_structure")):
        if width_cm >= 220.0 or "canopy" in geometry:
            return "tree_canopy_or_cluster"
        return "tree_trunk_or_pole"
    return "unknown"


def geometry_label_from_event(event: Dict[str, Any]) -> str:
    summary = _summary(event)
    geometry = str(summary.get("obstacle_geometry", "") or "").lower()
    width_cm = as_float(summary.get("obstacle_width_cm"))
    front_min = as_float(summary.get("front_min_depth_cm"))
    up_clear = as_bool(summary.get("up_swept_clear", True))
    if geometry == "none" or front_min >= 650.0:
        return "open_path"
    if geometry == "low_obstacle" and width_cm >= 220.0 and up_clear:
        return "fence_or_rail"
    if geometry in {"thin_structure"}:
        return "tree_trunk_or_pole"
    if geometry in {"vertical_wall", "overhang_beam"}:
        return "building"
    return "unknown"


def flyover_recommended_for_label(label: str, event: Dict[str, Any]) -> bool:
    if label in {"fence_or_rail", "building", "tree_canopy_or_cluster"}:
        return True
    if label in {"open_path", "tree_trunk_or_pole"}:
        return False
    summary = _summary(event)
    return bool(str(summary.get("obstacle_geometry", "") or "").lower() in {"low_obstacle", "overhang_beam"})


def teacher_label_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    manual_label = event.get("manual_label") or event.get("manual_obstacle_label")
    if manual_label:
        label = canonical_obstacle_label(manual_label, event)
        return {
            "label": label,
            "label_index": OBSTACLE_LABELS.index(label),
            "raw_label": str(manual_label or ""),
            "teacher_source": "manual_hard_label",
            "flyover_recommended": bool(
                event.get("flyover_recommended")
                if "flyover_recommended" in event
                else flyover_recommended_for_label(label, event)
            ),
            "sample_weight": float(as_float(event.get("sample_weight"), 3.0) or 3.0),
            "is_manual_hard_case": True,
        }
    strategy = _strategy(event)
    raw_label = (
        strategy.get("obstacle_hint")
        or strategy.get("environment_id")
        or event.get("obstacle_hint")
        or event.get("environment_id")
        or ""
    )
    label = canonical_obstacle_label(raw_label, event)
    source = "llm_strategy" if strategy else "event_metadata"
    if label == "unknown":
        label = geometry_label_from_event(event)
        source = "pointcloud_geometry" if label != "unknown" else source
    default_weight = {
        "llm_strategy": 1.5,
        "event_metadata": 0.8,
        "pointcloud_geometry": 0.7,
    }.get(source, 1.0)
    return {
        "label": label,
        "label_index": OBSTACLE_LABELS.index(label),
        "raw_label": str(raw_label or ""),
        "teacher_source": source,
        "flyover_recommended": bool(flyover_recommended_for_label(label, event)),
        "sample_weight": float(default_weight),
        "is_manual_hard_case": False,
    }
