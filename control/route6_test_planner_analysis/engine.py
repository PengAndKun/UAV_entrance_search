from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _json_safe(host: Any, payload: Any) -> Any:
    if hasattr(host, "route6_json_safe"):
        return host.route6_json_safe(payload)
    return payload


def active_reset_target(plan: Dict[str, Any]) -> str:
    scan_points = plan.get("selected_scan_observation_points", []) if isinstance(plan.get("selected_scan_observation_points", []), list) else []
    if scan_points and str(plan.get("scan_mode", "") or "") != "single_best_point":
        return "scan_observation_points"
    return "selected_observation_point"


def anchor_for_point(point: Dict[str, Any], selected_edge: Dict[str, Any]) -> Dict[str, float]:
    for source in (
        point.get("anchor_point_cm", {}) if isinstance(point.get("anchor_point_cm", {}), dict) else {},
        (point.get("algorithm_components", {}) if isinstance(point.get("algorithm_components", {}), dict) else {}).get("anchor_point_cm", {}),
        (selected_edge.get("algorithm_components", {}) if isinstance(selected_edge.get("algorithm_components", {}), dict) else {}).get("anchor_point_cm", {}),
        selected_edge.get("nearest_edge_point_cm", {}) if isinstance(selected_edge.get("nearest_edge_point_cm", {}), dict) else {},
        selected_edge.get("edge_center_cm", {}) if isinstance(selected_edge.get("edge_center_cm", {}), dict) else {},
    ):
        if isinstance(source, dict) and "x" in source and "y" in source:
            return {"x": float(source.get("x", 0.0) or 0.0), "y": float(source.get("y", 0.0) or 0.0)}
    return {"x": float(point.get("x", 0.0) or 0.0), "y": float(point.get("y", 0.0) or 0.0)}


def reset_summary(main_reset: Dict[str, Any], scan_resets: List[Dict[str, Any]]) -> Dict[str, Any]:
    main_status = str(main_reset.get("reset_status", "") or "")
    status_counts: Dict[str, int] = {}
    problems: List[str] = []
    failed_scan_indices: List[int] = []
    changed_scan_indices: List[int] = []
    report_only_scan_indices: List[int] = []
    for item in [main_reset] + [scan for scan in scan_resets if isinstance(scan, dict)]:
        status = str(item.get("reset_status", "") or "unknown")
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        for problem in item.get("problems", []) if isinstance(item.get("problems", []), list) else []:
            text = str(problem or "")
            if text and text not in problems:
                problems.append(text)
    for scan in scan_resets:
        if not isinstance(scan, dict):
            continue
        status = str(scan.get("reset_status", "") or "")
        point = scan.get("old_point", {}) if isinstance(scan.get("old_point", {}), dict) else {}
        try:
            scan_index = int(point.get("scan_index", 0) or 0)
        except Exception:
            scan_index = 0
        if status in {"failed", "micro_adjust_failed"}:
            failed_scan_indices.append(scan_index)
        elif status in {"ok", "micro_adjust_ok"}:
            changed_scan_indices.append(scan_index)
        elif status == "no_reset_needed":
            report_only_scan_indices.append(scan_index)

    if main_status in {"failed", "micro_adjust_failed"}:
        final_status = "failed"
    elif failed_scan_indices:
        final_status = "scan_reset_failed"
    elif main_status in {"ok", "micro_adjust_ok"}:
        final_status = "ok"
    elif changed_scan_indices:
        final_status = "scan_reset_ok"
    elif main_status == "no_reset_needed" or report_only_scan_indices:
        final_status = "no_reset_needed"
    elif main_status == "reference_not_reset" and int(status_counts.get("already_safe", 0)) > 0:
        final_status = "already_safe"
    else:
        final_status = main_status or "unknown"

    return {
        "reset_status": final_status,
        "main_reset_status": main_status,
        "status_counts": status_counts,
        "problems": problems,
        "failed_scan_indices": failed_scan_indices,
        "changed_scan_indices": changed_scan_indices,
        "report_only_scan_indices": report_only_scan_indices,
        "scan_point_count": len(scan_resets),
    }


def house_plan_for_point(plan: Dict[str, Any], point: Dict[str, Any]) -> Dict[str, Any]:
    house_id = str(point.get("house_id", "") or "")
    for house_plan in plan.get("house_plans", []) if isinstance(plan.get("house_plans", []), list) else []:
        if isinstance(house_plan, dict) and str(house_plan.get("house_id", "") or "") == house_id:
            return house_plan
    return {}


def edge_record_for_point(plan: Dict[str, Any], point: Dict[str, Any]) -> Dict[str, Any]:
    edge_name = str(point.get("edge", "") or "")
    house_plan = house_plan_for_point(plan, point)
    selected_edge = house_plan.get("selected_edge", {}) if isinstance(house_plan.get("selected_edge", {}), dict) else {}
    if str(selected_edge.get("edge", "") or "") == edge_name:
        return selected_edge
    for edge in house_plan.get("edge_calculations", []) if isinstance(house_plan.get("edge_calculations", []), list) else []:
        if isinstance(edge, dict) and str(edge.get("edge", "") or "") == edge_name:
            return edge
    return {}


def visual_formula(point: Dict[str, Any], anchor: Dict[str, float], bbox: Dict[str, Any], radar_distance_cm: float) -> str:
    edge = str(point.get("edge", "") or "")
    h = float(point.get("z", 0.0) or 0.0)
    if edge == "south":
        expression = "P_obs = (A.x, bbox.min_y - d, h)"
    elif edge == "north":
        expression = "P_obs = (A.x, bbox.max_y + d, h)"
    elif edge == "east":
        expression = "P_obs = (bbox.max_x + d, A.y, h)"
    else:
        expression = "P_obs = (bbox.min_x - d, A.y, h)"
    return (
        f"{expression}; A=({float(anchor.get('x', 0.0) or 0.0):.2f},"
        f"{float(anchor.get('y', 0.0) or 0.0):.2f}), d={float(radar_distance_cm):.2f}, h={h:.2f}, "
        f"P_obs=({float(point.get('x', 0.0) or 0.0):.2f},{float(point.get('y', 0.0) or 0.0):.2f})"
    )


def visual_calculation_records(
    host: Any,
    plan: Dict[str, Any],
    *,
    output_dir: Optional[Path] = None,
    selected_layer_key: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    out_path: Optional[Path] = Path(output_dir) if output_dir is not None else None
    if out_path is None and str(plan.get("output_dir", "") or ""):
        out_path = Path(str(plan.get("output_dir", "") or ""))
    layer_key = str(selected_layer_key or plan.get("selected_layer_key", "") or "")
    records: List[Dict[str, Any]] = []
    scan_points_for_display = plan.get("selected_scan_observation_points", []) if isinstance(plan.get("selected_scan_observation_points", []), list) else []
    scan_points_are_active = bool(scan_points_for_display) and active_reset_target(plan) == "scan_observation_points"

    def add_record(kind: str, point: Dict[str, Any]) -> None:
        if not isinstance(point, dict) or not point:
            return
        house_plan = house_plan_for_point(plan, point)
        edge_record = edge_record_for_point(plan, point)
        bbox = house_plan.get("bbox", {}) if isinstance(house_plan.get("bbox", {}), dict) else {}
        anchor = anchor_for_point(point, edge_record)
        safety = host.route6_test_planner_observation_safety_report(
            point,
            anchor=anchor,
            target_house_id=str(point.get("house_id", "") or ""),
            output_dir=out_path,
            selected_layer_key=layer_key,
        )
        distance = math.hypot(
            float(point.get("x", 0.0) or 0.0) - float(anchor.get("x", 0.0) or 0.0),
            float(point.get("y", 0.0) or 0.0) - float(anchor.get("y", 0.0) or 0.0),
        )
        coverage = point.get("coverage_segment_cm", {}) if isinstance(point.get("coverage_segment_cm", {}), dict) else {}
        coverage_start: Dict[str, float] = {}
        coverage_end: Dict[str, float] = {}
        if coverage and edge_record:
            coverage_start = host.route6_point_on_edge_progress(edge_record, float(coverage.get("start", 0.0) or 0.0))
            coverage_end = host.route6_point_on_edge_progress(edge_record, float(coverage.get("end", 0.0) or 0.0))
        try:
            scan_index = int(point.get("scan_index", 0) or 0)
        except Exception:
            scan_index = 0
        record = {
            "schema": "route6_visual_calculation_record_v1",
            "kind": kind,
            "label": f"S{scan_index}" if kind == "scan_observation" else f"H{point.get('house_id', '')} {point.get('edge', '')} obs",
            "display_on_map": not (kind == "selected_observation" and scan_points_are_active),
            "active_role": "reference" if kind == "selected_observation" and scan_points_are_active else "navigation",
            "house_id": str(point.get("house_id", "") or ""),
            "edge": str(point.get("edge", "") or ""),
            "scan_index": int(scan_index),
            "observation_point_cm": {
                "x": round(float(point.get("x", 0.0) or 0.0), 2),
                "y": round(float(point.get("y", 0.0) or 0.0), 2),
                "z": round(float(point.get("z", 0.0) or 0.0), 2),
                "yaw_deg": round(float(point.get("yaw_deg", 0.0) or 0.0), 3),
            },
            "anchor_point_cm": {"x": round(float(anchor.get("x", 0.0) or 0.0), 2), "y": round(float(anchor.get("y", 0.0) or 0.0), 2)},
            "ray_start_cm": {"x": round(float(point.get("x", 0.0) or 0.0), 2), "y": round(float(point.get("y", 0.0) or 0.0), 2)},
            "ray_end_cm": {"x": round(float(anchor.get("x", 0.0) or 0.0), 2), "y": round(float(anchor.get("y", 0.0) or 0.0), 2)},
            "radar_distance_cm": round(float(distance), 2),
            "edge_progress_cm": round(float(point.get("edge_progress_cm", 0.0) or 0.0), 2) if "edge_progress_cm" in point else None,
            "coverage_segment_cm": _json_safe(host, coverage),
            "coverage_start_point_cm": _json_safe(host, coverage_start),
            "coverage_end_point_cm": _json_safe(host, coverage_end),
            "formula": visual_formula(point, anchor, bbox, distance),
            "safety_report": _json_safe(host, safety),
        }
        records.append(_json_safe(host, record))

    selected = plan.get("selected_observation_point", {}) if isinstance(plan.get("selected_observation_point", {}), dict) else {}
    add_record("selected_observation", selected)
    scan_points = plan.get("selected_scan_observation_points", []) if isinstance(plan.get("selected_scan_observation_points", []), list) else []
    if not scan_points:
        scan_points = plan.get("scan_observation_points", []) if isinstance(plan.get("scan_observation_points", []), list) else []
    for scan_point in scan_points:
        add_record("scan_observation", scan_point)
    return records
