from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def solve_affine_from_anchor_points(anchors: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
    if len(anchors) < 3:
        return None
    try:
        world = np.asarray(
            [[float(anchor["world_x"]), float(anchor["world_y"]), 1.0] for anchor in anchors],
            dtype=np.float64,
        )
        image_x = np.asarray([float(anchor["image_x"]) for anchor in anchors], dtype=np.float64)
        image_y = np.asarray([float(anchor["image_y"]) for anchor in anchors], dtype=np.float64)
        row_x, *_ = np.linalg.lstsq(world, image_x, rcond=None)
        row_y, *_ = np.linalg.lstsq(world, image_y, rcond=None)
        return [
            [float(row_x[0]), float(row_x[1]), float(row_x[2])],
            [float(row_y[0]), float(row_y[1]), float(row_y[2])],
        ]
    except Exception:
        return None


def world_to_image_with_affine(world_x: float, world_y: float, affine: List[List[float]]) -> Tuple[float, float]:
    matrix = np.asarray(affine, dtype=np.float64)
    image_x = float(matrix[0, 0]) * float(world_x) + float(matrix[0, 1]) * float(world_y) + float(matrix[0, 2])
    image_y = float(matrix[1, 0]) * float(world_x) + float(matrix[1, 1]) * float(world_y) + float(matrix[1, 2])
    return image_x, image_y


def image_to_world_with_affine(image_x: float, image_y: float, affine: List[List[float]]) -> Tuple[float, float]:
    matrix = np.asarray(affine, dtype=np.float64)
    linear = matrix[:, :2]
    offset = matrix[:, 2]
    world = np.linalg.solve(linear, np.asarray([float(image_x), float(image_y)], dtype=np.float64) - offset)
    return float(world[0]), float(world[1])


def affine_rmse_px(anchors: List[Dict[str, Any]], affine: List[List[float]]) -> float:
    errors: List[float] = []
    for anchor in anchors:
        predicted_x, predicted_y = world_to_image_with_affine(
            float(anchor["world_x"]),
            float(anchor["world_y"]),
            affine,
        )
        dx = predicted_x - float(anchor["image_x"])
        dy = predicted_y - float(anchor["image_y"])
        errors.append(dx * dx + dy * dy)
    if not errors:
        return 0.0
    return float(math.sqrt(sum(errors) / len(errors)))


def corrected_anchors_from_touch_state(
    original_anchors: List[Dict[str, Any]],
    touch_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    points = touch_state.get("points", []) if isinstance(touch_state.get("points"), list) else []
    done_by_label = {
        str(point.get("label", "")): point
        for point in points
        if isinstance(point, dict) and point.get("status") == "done"
    }
    corrected: List[Dict[str, Any]] = []
    for index, anchor in enumerate(original_anchors[:5], start=1):
        label = str(anchor.get("label", f"P{index}") or f"P{index}")
        point = done_by_label.get(label)
        if point is None:
            raise ValueError(f"Calibration point {label} is not completed")
        contact_x = point.get("contact_world_x")
        contact_y = point.get("contact_world_y")
        if contact_x is None or contact_y is None:
            pose = point.get("contact_pose", [])
            if not isinstance(pose, list) or len(pose) < 2:
                raise ValueError(f"Calibration point {label} has no contact pose")
            contact_x, contact_y = pose[0], pose[1]
        corrected.append({
            "index": int(float(anchor.get("index", index))),
            "label": label,
            "world_x": float(contact_x),
            "world_y": float(contact_y),
            "image_x": float(anchor["image_x"]),
            "image_y": float(anchor["image_y"]),
            "source_world_x": float(anchor["world_x"]),
            "source_world_y": float(anchor["world_y"]),
        })
    if len(corrected) < 5:
        raise ValueError("P1-P5 calibration requires five completed points")
    return corrected


def rebuild_houses_for_corrected_affine(config: Dict[str, Any], affine: List[List[float]]) -> List[Dict[str, Any]]:
    rebuilt: List[Dict[str, Any]] = []
    houses = config.get("houses", []) if isinstance(config.get("houses"), list) else []
    for house in houses:
        if not isinstance(house, dict):
            continue
        new_house = json.loads(json.dumps(house))
        bbox = new_house.get("map_bbox_image")
        if isinstance(bbox, dict):
            try:
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
                cx_img = (x1 + x2) * 0.5
                cy_img = (y1 + y2) * 0.5
                half_w = abs(x2 - x1) * 0.5
                half_h = abs(y2 - y1) * 0.5
                center_x, center_y = image_to_world_with_affine(cx_img, cy_img, affine)
                right_x, right_y = image_to_world_with_affine(cx_img + half_w, cy_img, affine)
                down_x, down_y = image_to_world_with_affine(cx_img, cy_img + half_h, affine)
                radius = max(
                    float(np.hypot(right_x - center_x, right_y - center_y)),
                    float(np.hypot(down_x - center_x, down_y - center_y)),
                    300.0,
                )
                new_house["center_x"] = center_x
                new_house["center_y"] = center_y
                new_house["radius_cm"] = radius
            except Exception as exc:
                new_house["map_rebuild_error"] = str(exc)
        rebuilt.append(new_house)
    return rebuilt


def build_corrected_map_config(config: Dict[str, Any], touch_state: Dict[str, Any]) -> Dict[str, Any]:
    corrected = json.loads(json.dumps(config))
    overhead = corrected.setdefault("overhead_map", {})
    calibration = overhead.setdefault("calibration", {})
    original_anchors = calibration.get("anchors", [])
    if not isinstance(original_anchors, list):
        raise ValueError("Map config has no calibration anchors")
    corrected_anchors = corrected_anchors_from_touch_state(original_anchors, touch_state)
    affine = solve_affine_from_anchor_points(corrected_anchors)
    if affine is None:
        raise ValueError("Failed to solve corrected affine")
    calibration["anchors"] = corrected_anchors
    calibration["affine_world_to_image"] = affine
    calibration["rmse_px"] = affine_rmse_px(corrected_anchors, affine)
    calibration["updated_at"] = time.time()
    calibration["touch_calibration_status"] = str(touch_state.get("status", ""))
    if "image_width" not in calibration:
        calibration["image_width"] = int(overhead.get("image_width", 0) or 0)
    if "image_height" not in calibration:
        calibration["image_height"] = int(overhead.get("image_height", 0) or 0)
    corrected["houses"] = rebuild_houses_for_corrected_affine(corrected, affine)
    corrected["map_touch_calibration"] = {
        "completed_at": touch_state.get("completed_at"),
        "completed_count": touch_state.get("completed_count"),
        "marker_z": touch_state.get("marker_z"),
        "points": touch_state.get("points", []),
    }
    return corrected
