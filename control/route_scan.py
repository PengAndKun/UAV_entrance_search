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


def generate_rule_scan_points(
    *,
    house_id: str,
    bbox_world: Dict[str, Any],
    facade_order: Optional[Iterable[Any]] = None,
    standoff_cm: float,
    scan_spacing_cm: float,
    altitude_cm: float,
    lidar_range_cm: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    min_x = float(bbox_world["min_x"])
    max_x = float(bbox_world["max_x"])
    min_y = float(bbox_world["min_y"])
    max_y = float(bbox_world["max_y"])
    center_x = float(bbox_world.get("center_x", 0.5 * (min_x + max_x)))
    center_y = float(bbox_world.get("center_y", 0.5 * (min_y + max_y)))
    standoff = max(20.0, float(standoff_cm))
    spacing = max(1.0, float(scan_spacing_cm))
    altitude = float(altitude_cm)
    lidar_range = list(lidar_range_cm or [20.0, 1200.0])
    roles = ["front", "side", "back", "other_side"]

    points: List[Dict[str, Any]] = []
    for order_index, facade in enumerate(normalize_facade_order(facade_order)):
        role = roles[order_index] if order_index < len(roles) else f"facade_{order_index + 1}"
        if facade in {"south", "north"}:
            count = _facade_point_count(max_x - min_x, spacing)
        else:
            count = _facade_point_count(max_y - min_y, spacing)
        for idx in range(count):
            if facade == "south":
                x = _interpolate(min_x, max_x, idx, count)
                y = min_y - standoff
                target_x, target_y = center_x, min_y
            elif facade == "east":
                x = max_x + standoff
                y = _interpolate(min_y, max_y, idx, count)
                target_x, target_y = max_x, center_y
            elif facade == "north":
                x = _interpolate(max_x, min_x, idx, count)
                y = max_y + standoff
                target_x, target_y = center_x, max_y
            else:
                x = min_x - standoff
                y = _interpolate(max_y, min_y, idx, count)
                target_x, target_y = min_x, center_y
            points.append(
                {
                    "scan_id": f"{house_id}_{facade}_{idx:03d}",
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
                }
            )
    return points
