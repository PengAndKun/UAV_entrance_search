from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


FACADE_ORDER = ("south", "east", "north", "west")


def normalize_facade_order(raw_order: Optional[Iterable[Any]]) -> List[str]:
    ordered: List[str] = []
    if raw_order is not None:
        for item in raw_order:
            facade = str(item or "").strip().lower()
            if facade in FACADE_ORDER and facade not in ordered:
                ordered.append(facade)
    for facade in FACADE_ORDER:
        if facade not in ordered:
            ordered.append(facade)
    return ordered


def _interpolate(start: float, end: float, index: int, count: int) -> float:
    if count <= 1:
        return 0.5 * (float(start) + float(end))
    ratio = float(index) / float(count - 1)
    return float(start) + (float(end) - float(start)) * ratio


def _facade_point_count(length_cm: float, spacing_cm: float) -> int:
    spacing = max(1.0, float(spacing_cm))
    return max(2, int(math.ceil(max(0.0, float(length_cm)) / spacing)) + 1)


def _yaw_to_target(x: float, y: float, target_x: float, target_y: float) -> float:
    return math.degrees(math.atan2(float(target_y) - float(y), float(target_x) - float(x)))


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def _oriented_axis_intervals(
    *,
    facade: str,
    full_min: float,
    full_max: float,
    raw_intervals: Optional[Any],
) -> List[Dict[str, Any]]:
    if raw_intervals is None:
        raw_items: List[Any] = [{"min": full_min, "max": full_max, "index": 0}]
    elif isinstance(raw_intervals, list):
        raw_items = raw_intervals
    else:
        raw_items = []

    normalized: List[Dict[str, Any]] = []
    for raw_index, item in enumerate(raw_items):
        if isinstance(item, dict):
            lo = _as_float(item.get("min", item.get("axis_min")))
            hi = _as_float(item.get("max", item.get("axis_max")))
            interval_index = item.get("index", item.get("safe_interval_index", raw_index))
            source = str(item.get("source", "") or "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lo = _as_float(item[0])
            hi = _as_float(item[1])
            interval_index = raw_index
            source = ""
        else:
            continue
        if lo is None or hi is None:
            continue
        axis_min = max(float(full_min), min(float(lo), float(hi)))
        axis_max = min(float(full_max), max(float(lo), float(hi)))
        if axis_max < axis_min:
            continue
        normalized.append(
            {
                "min": axis_min,
                "max": axis_max,
                "index": int(interval_index) if str(interval_index).lstrip("-").isdigit() else raw_index,
                "source": source,
            }
        )

    if facade in {"south", "east"}:
        ordered = sorted(normalized, key=lambda item: (float(item["min"]), float(item["max"])))
        for item in ordered:
            item["start"] = float(item["min"])
            item["end"] = float(item["max"])
    else:
        ordered = sorted(normalized, key=lambda item: (float(item["max"]), float(item["min"])), reverse=True)
        for item in ordered:
            item["start"] = float(item["max"])
            item["end"] = float(item["min"])
    return ordered


def generate_rule_scan_points(
    *,
    house_id: str,
    bbox_world: Dict[str, Any],
    facade_order: Optional[Iterable[Any]] = None,
    standoff_cm: float,
    scan_spacing_cm: float,
    altitude_cm: float,
    lidar_range_cm: Optional[List[float]] = None,
    facade_standoff_info: Optional[Dict[str, Dict[str, Any]]] = None,
    facade_axis_intervals: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    min_x = float(bbox_world["min_x"])
    max_x = float(bbox_world["max_x"])
    min_y = float(bbox_world["min_y"])
    max_y = float(bbox_world["max_y"])
    center_x = float(bbox_world.get("center_x", 0.5 * (min_x + max_x)))
    center_y = float(bbox_world.get("center_y", 0.5 * (min_y + max_y)))
    default_standoff = max(20.0, float(standoff_cm))
    spacing = max(1.0, float(scan_spacing_cm))
    altitude = float(altitude_cm)
    lidar_range = list(lidar_range_cm or [20.0, 1200.0])
    standoff_info_by_facade = facade_standoff_info if isinstance(facade_standoff_info, dict) else {}
    interval_limits_by_facade = facade_axis_intervals if isinstance(facade_axis_intervals, dict) else {}
    roles = ["front", "side", "back", "other_side"]

    points: List[Dict[str, Any]] = []
    for order_index, facade in enumerate(normalize_facade_order(facade_order)):
        role = roles[order_index] if order_index < len(roles) else f"facade_{order_index + 1}"
        if facade in {"south", "north"}:
            full_axis_min, full_axis_max = min_x, max_x
        else:
            full_axis_min, full_axis_max = min_y, max_y
        raw_intervals = interval_limits_by_facade.get(facade) if facade in interval_limits_by_facade else None
        axis_intervals = _oriented_axis_intervals(
            facade=facade,
            full_min=full_axis_min,
            full_max=full_axis_max,
            raw_intervals=raw_intervals,
        )
        facade_info = standoff_info_by_facade.get(facade, {}) if isinstance(standoff_info_by_facade.get(facade), dict) else {}
        try:
            standoff = max(20.0, float(facade_info.get("standoff_cm", default_standoff)))
        except Exception:
            standoff = default_standoff
        scan_index = 0
        for interval_position, interval in enumerate(axis_intervals):
            axis_start = float(interval["start"])
            axis_end = float(interval["end"])
            count = _facade_point_count(abs(axis_end - axis_start), spacing)
            for local_idx in range(count):
                axis_value = _interpolate(axis_start, axis_end, local_idx, count)
                if facade == "south":
                    x = axis_value
                    y = min_y - standoff
                    target_x, target_y = center_x, min_y
                elif facade == "east":
                    x = max_x + standoff
                    y = axis_value
                    target_x, target_y = max_x, center_y
                elif facade == "north":
                    x = axis_value
                    y = max_y + standoff
                    target_x, target_y = center_x, max_y
                else:
                    x = min_x - standoff
                    y = axis_value
                    target_x, target_y = min_x, center_y
                points.append(
                    {
                        "scan_id": f"{house_id}_{facade}_{scan_index:03d}",
                        "house_id": str(house_id),
                        "facade": facade,
                        "face_role": role,
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "z": round(float(altitude), 2),
                        "yaw_deg": round(_yaw_to_target(x, y, target_x, target_y), 2),
                        "standoff_cm": round(float(standoff), 2),
                        "scan_spacing_cm": round(float(spacing), 2),
                        "expected_lidar_range_cm": lidar_range,
                        "capture_trigger": "arrive_align_hover_capture",
                        "view_type": "face_view",
                        "status": "planned",
                        "safe_interval_index": int(interval.get("index", interval_position)),
                        "safe_interval_count": len(axis_intervals),
                        "safe_axis_min": round(float(interval["min"]), 2),
                        "safe_axis_max": round(float(interval["max"]), 2),
                        "safe_interval_source": str(interval.get("source", "") or ""),
                        "corridor_mode": str(facade_info.get("mode", "default") or "default"),
                        "corridor_gap_cm": facade_info.get("gap_cm"),
                        "corridor_side_margin_cm": facade_info.get("side_margin_cm"),
                        "corridor_blocking_house_id": str(facade_info.get("blocking_house_id", "") or ""),
                        "corridor_clearance_cm": facade_info.get("clearance_cm"),
                        "corridor_safe": bool(facade_info.get("safe", True)),
                    }
                )
                scan_index += 1
    return points
