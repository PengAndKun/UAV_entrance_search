from __future__ import annotations

from PIL import ImageDraw

from .common import *
from . import route6_map_builder
from .route6_test_planner_analysis import (
    active_reset_target as route6_analysis_active_reset_target,
    anchor_for_point as route6_analysis_anchor_for_point,
    edge_record_for_point as route6_analysis_edge_record_for_point,
    house_plan_for_point as route6_analysis_house_plan_for_point,
    reset_summary as route6_analysis_reset_summary,
    visual_calculation_records as route6_analysis_visual_calculation_records,
    visual_formula as route6_analysis_visual_formula,
)


class Route6TestPlannerControlMixin:
    def route6_test_planner_normalize_house_id(self, value: Any) -> str:
        text = str(value or "").strip()
        if text.isdigit():
            return f"{int(text):03d}"
        return text

    def route6_test_planner_normalize_edge(self, value: Any) -> str:
        text = str(value or "").strip().lower().replace("_", " ")
        if text in {"south", "east", "north", "west"}:
            return text
        return ""

    def route6_test_planner_algorithm_options(self) -> List[str]:
        return [
            "nearest edge",
            "frontier-based",
            "nbv information gain",
            "surface edge explorer",
            "uav inspection contour",
        ]

    def route6_test_planner_normalize_algorithm(self, value: Any) -> str:
        text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
        text = " ".join(part for part in text.split() if part)
        mapping = {
            "": "nearest_edge",
            "auto": "nearest_edge",
            "auto nearest": "nearest_edge",
            "nearest": "nearest_edge",
            "nearest edge": "nearest_edge",
            "nearest exploration edge": "nearest_edge",
            "frontier": "frontier_based",
            "frontier based": "frontier_based",
            "frontier based exploration": "frontier_based",
            "nbv": "nbv_information_gain",
            "next best view": "nbv_information_gain",
            "nbv information gain": "nbv_information_gain",
            "information gain": "nbv_information_gain",
            "see": "surface_edge_explorer",
            "surface edge": "surface_edge_explorer",
            "surface edge explorer": "surface_edge_explorer",
            "uav inspection": "uav_inspection_contour",
            "inspection contour": "uav_inspection_contour",
            "uav inspection contour": "uav_inspection_contour",
            "multi layer inspection": "uav_inspection_contour",
        }
        return mapping.get(text, "nearest_edge")

    def route6_test_planner_algorithm_label(self, algorithm: str) -> str:
        labels = {
            "nearest_edge": "nearest edge",
            "frontier_based": "frontier-based",
            "nbv_information_gain": "nbv information gain",
            "surface_edge_explorer": "surface edge explorer",
            "uav_inspection_contour": "uav inspection contour",
        }
        return labels.get(str(algorithm or ""), "nearest edge")

    def route6_test_planner_scan_mode_options(self) -> List[str]:
        return [
            "single best point",
            "multi-point edge coverage",
            "greedy coverage nbv",
        ]

    def route6_test_planner_normalize_scan_mode(self, value: Any) -> str:
        text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
        text = " ".join(part for part in text.split() if part)
        mapping = {
            "": "single_best_point",
            "single": "single_best_point",
            "single best": "single_best_point",
            "single best point": "single_best_point",
            "multi": "multi_point_edge_coverage",
            "multi point": "multi_point_edge_coverage",
            "multi point edge coverage": "multi_point_edge_coverage",
            "edge coverage": "multi_point_edge_coverage",
            "coverage": "multi_point_edge_coverage",
            "greedy": "greedy_coverage_nbv",
            "greedy coverage": "greedy_coverage_nbv",
            "greedy coverage nbv": "greedy_coverage_nbv",
        }
        return mapping.get(text, "single_best_point")

    def route6_test_planner_scan_mode_label(self, scan_mode: str) -> str:
        labels = {
            "single_best_point": "single best point",
            "multi_point_edge_coverage": "multi-point edge coverage",
            "greedy_coverage_nbv": "greedy coverage nbv",
        }
        return labels.get(str(scan_mode or ""), "single best point")

    def route6_test_planner_house_records(self) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        records: List[Dict[str, Any]] = []
        try:
            known_records = self.route6_known_house_polygon_records()
        except Exception:
            known_records = []
        for raw in known_records if isinstance(known_records, list) else []:
            if not isinstance(raw, dict):
                continue
            house_id = self.route6_test_planner_normalize_house_id(raw.get("house_id", raw.get("id", "")))
            bbox = raw.get("bbox", {}) if isinstance(raw.get("bbox", {}), dict) else {}
            if not house_id or not bbox:
                continue
            try:
                normalized_bbox = {key: float(bbox[key]) for key in ("min_x", "max_x", "min_y", "max_y")}
            except Exception:
                continue
            records.append(
                {
                    **self.route6_json_safe(raw),
                    "house_id": house_id,
                    "id": house_id,
                    "name": str(raw.get("name", f"House_{house_id}") or f"House_{house_id}"),
                    "bbox": normalized_bbox,
                    "source": str(raw.get("source", "known_house_polygon") or "known_house_polygon"),
                }
            )
        if records:
            return records

        map_config = self.route6_get_runtime_map_config()
        houses = map_config.get("houses", []) if isinstance(map_config.get("houses", []), list) else []
        for raw in houses:
            if not isinstance(raw, dict):
                continue
            house_id = self.route6_test_planner_normalize_house_id(raw.get("house_id", raw.get("id", "")))
            if not house_id:
                continue
            try:
                bbox = route6_map_builder.house_world_bbox(map_config, raw)
                normalized_bbox = {key: float(bbox[key]) for key in ("min_x", "max_x", "min_y", "max_y")}
            except Exception:
                continue
            records.append(
                {
                    **self.route6_json_safe(raw),
                    "house_id": house_id,
                    "id": house_id,
                    "name": str(raw.get("name", f"House_{house_id}") or f"House_{house_id}"),
                    "bbox": normalized_bbox,
                    "source": "runtime_map_config",
                }
            )
        return records

    def route6_house_record_bbox(self, house_record: Dict[str, Any]) -> Dict[str, float]:
        item = house_record if isinstance(house_record, dict) else {}
        bbox = item.get("bbox", {}) if isinstance(item.get("bbox", {}), dict) else {}
        try:
            result = {
                "min_x": float(bbox["min_x"]),
                "max_x": float(bbox["max_x"]),
                "min_y": float(bbox["min_y"]),
                "max_y": float(bbox["max_y"]),
            }
        except Exception:
            return {}
        if result["max_x"] <= result["min_x"] or result["max_y"] <= result["min_y"]:
            return {}
        return result

    def route6_test_planner_nearest_point_on_segment(
        self,
        pose: Dict[str, Any],
        start: Dict[str, float],
        end: Dict[str, float],
    ) -> Dict[str, float]:
        px = float(pose.get("x", 0.0) or 0.0)
        py = float(pose.get("y", 0.0) or 0.0)
        x1 = float(start.get("x", 0.0) or 0.0)
        y1 = float(start.get("y", 0.0) or 0.0)
        x2 = float(end.get("x", 0.0) or 0.0)
        y2 = float(end.get("y", 0.0) or 0.0)
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        if denom <= 0.0:
            return {"x": round(x1, 2), "y": round(y1, 2)}
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
        return {"x": round(x1 + t * dx, 2), "y": round(y1 + t * dy, 2)}

    def route6_observation_point_for_edge(
        self,
        edge: str,
        nearest_point: Dict[str, float],
        bbox: Dict[str, float],
        *,
        radar_distance_cm: float,
        scan_z_cm: float,
    ) -> Dict[str, float]:
        edge_name = str(edge or "").strip().lower()
        x = float(nearest_point.get("x", 0.0) or 0.0)
        y = float(nearest_point.get("y", 0.0) or 0.0)
        distance = float(radar_distance_cm)
        if edge_name == "south":
            y = float(bbox["min_y"]) - distance
        elif edge_name == "north":
            y = float(bbox["max_y"]) + distance
        elif edge_name == "east":
            x = float(bbox["max_x"]) + distance
        else:
            x = float(bbox["min_x"]) - distance
        return {"x": round(x, 2), "y": round(y, 2), "z": round(float(scan_z_cm), 2)}

    def route6_exploration_edges_for_bbox(
        self,
        bbox: Dict[str, float],
        pose: Dict[str, Any],
        *,
        house_id: str = "",
        radar_distance_cm: float = 850.0,
        scan_z_cm: float = 450.0,
    ) -> List[Dict[str, Any]]:
        min_x = float(bbox["min_x"])
        max_x = float(bbox["max_x"])
        min_y = float(bbox["min_y"])
        max_y = float(bbox["max_y"])
        edge_specs = [
            ("south", {"x": min_x, "y": min_y}, {"x": max_x, "y": min_y}),
            ("east", {"x": max_x, "y": min_y}, {"x": max_x, "y": max_y}),
            ("north", {"x": max_x, "y": max_y}, {"x": min_x, "y": max_y}),
            ("west", {"x": min_x, "y": max_y}, {"x": min_x, "y": min_y}),
        ]
        edges: List[Dict[str, Any]] = []
        for edge_name, start, end in edge_specs:
            center = {"x": round(0.5 * (start["x"] + end["x"]), 2), "y": round(0.5 * (start["y"] + end["y"]), 2)}
            nearest = self.route6_test_planner_nearest_point_on_segment(pose, start, end)
            observation = self.route6_observation_point_for_edge(
                edge_name,
                nearest,
                bbox,
                radar_distance_cm=float(radar_distance_cm),
                scan_z_cm=float(scan_z_cm),
            )
            yaw = math.degrees(math.atan2(float(center["y"]) - float(observation["y"]), float(center["x"]) - float(observation["x"])))
            distance_to_edge = math.hypot(
                float(nearest["x"]) - float(pose.get("x", 0.0) or 0.0),
                float(nearest["y"]) - float(pose.get("y", 0.0) or 0.0),
            )
            distance_to_center = math.hypot(
                float(center["x"]) - float(pose.get("x", 0.0) or 0.0),
                float(center["y"]) - float(pose.get("y", 0.0) or 0.0),
            )
            edges.append(
                {
                    "schema": "route6_offline_exploration_edge_v1",
                    "house_id": str(house_id or ""),
                    "edge": edge_name,
                    "start_cm": {"x": round(float(start["x"]), 2), "y": round(float(start["y"]), 2)},
                    "end_cm": {"x": round(float(end["x"]), 2), "y": round(float(end["y"]), 2)},
                    "edge_center_cm": center,
                    "nearest_edge_point_cm": nearest,
                    "observation_point_cm": {**observation, "yaw_deg": round(float(yaw), 3)},
                    "radar_distance_cm": round(float(radar_distance_cm), 2),
                    "scan_z_cm": round(float(scan_z_cm), 2),
                    "distance_to_edge_cm": round(float(distance_to_edge), 2),
                    "distance_to_edge_center_cm": round(float(distance_to_center), 2),
                    "selection_reason": "candidate edge generated from house bbox and radar-distance standoff",
                }
            )
        return edges

    def route6_test_planner_observation_from_anchor(
        self,
        edge_record: Dict[str, Any],
        bbox: Dict[str, float],
        anchor: Dict[str, float],
    ) -> Dict[str, float]:
        observation = self.route6_observation_point_for_edge(
            str(edge_record.get("edge", "") or ""),
            anchor,
            bbox,
            radar_distance_cm=float(edge_record.get("radar_distance_cm", 850.0) or 850.0),
            scan_z_cm=float(edge_record.get("scan_z_cm", 450.0) or 450.0),
        )
        center = edge_record.get("edge_center_cm", {}) if isinstance(edge_record.get("edge_center_cm", {}), dict) else {}
        yaw = math.degrees(
            math.atan2(
                float(center.get("y", 0.0) or 0.0) - float(observation.get("y", 0.0) or 0.0),
                float(center.get("x", 0.0) or 0.0) - float(observation.get("x", 0.0) or 0.0),
            )
        )
        return {**observation, "yaw_deg": round(float(yaw), 3)}

    def route6_test_planner_edge_length_cm(self, edge_record: Dict[str, Any]) -> float:
        start = edge_record.get("start_cm", {}) if isinstance(edge_record.get("start_cm", {}), dict) else {}
        end = edge_record.get("end_cm", {}) if isinstance(edge_record.get("end_cm", {}), dict) else {}
        return float(
            math.hypot(
                float(end.get("x", 0.0) or 0.0) - float(start.get("x", 0.0) or 0.0),
                float(end.get("y", 0.0) or 0.0) - float(start.get("y", 0.0) or 0.0),
            )
        )

    def route6_apply_test_planner_algorithm_to_edge(
        self,
        edge_record: Dict[str, Any],
        bbox: Dict[str, float],
        pose: Dict[str, Any],
        algorithm: str,
    ) -> Dict[str, Any]:
        edge = dict(edge_record)
        normalized_algorithm = self.route6_test_planner_normalize_algorithm(algorithm)
        nearest = edge.get("nearest_edge_point_cm", {}) if isinstance(edge.get("nearest_edge_point_cm", {}), dict) else {}
        center = edge.get("edge_center_cm", {}) if isinstance(edge.get("edge_center_cm", {}), dict) else {}
        edge_length = self.route6_test_planner_edge_length_cm(edge)
        distance_to_edge = float(edge.get("distance_to_edge_cm", 0.0) or 0.0)
        distance_to_center = float(edge.get("distance_to_edge_center_cm", 0.0) or 0.0)
        radar = float(edge.get("radar_distance_cm", 850.0) or 850.0)
        px = float(pose.get("x", 0.0) or 0.0)
        py = float(pose.get("y", 0.0) or 0.0)

        anchor = nearest
        anchor_policy = "nearest_projected_edge_point"
        expected_gain = edge_length
        travel_basis = distance_to_edge
        algorithm_score = 1.0 / (distance_to_edge + 1.0)
        source_paper_family = "nearest exploration edge baseline"

        if normalized_algorithm == "frontier_based":
            anchor = nearest
            anchor_policy = "nearest_frontier_cell_standoff"
            expected_gain = edge_length
            travel_basis = distance_to_edge
            algorithm_score = edge_length / (distance_to_edge + 1.0)
            source_paper_family = "frontier-based exploration"
        elif normalized_algorithm == "nbv_information_gain":
            anchor = center
            anchor_policy = "edge_center_high_gain_sample"
            observation_for_cost = self.route6_observation_point_for_edge(
                str(edge.get("edge", "") or ""),
                center,
                bbox,
                radar_distance_cm=radar,
                scan_z_cm=float(edge.get("scan_z_cm", 450.0) or 450.0),
            )
            travel_basis = math.hypot(float(observation_for_cost.get("x", 0.0) or 0.0) - px, float(observation_for_cost.get("y", 0.0) or 0.0) - py)
            expected_gain = min(edge_length, max(edge_length * 0.25, 2.0 * radar))
            algorithm_score = expected_gain / (travel_basis + 1.0)
            source_paper_family = "next-best-view information gain"
        elif normalized_algorithm == "surface_edge_explorer":
            anchor = center
            anchor_policy = "edge_center_surface_standoff"
            travel_basis = max(distance_to_center, 1.0)
            expected_gain = edge_length
            algorithm_score = edge_length / (travel_basis + 1.0)
            source_paper_family = "surface edge explorer"
        elif normalized_algorithm == "uav_inspection_contour":
            anchor = center
            anchor_policy = "dilated_contour_midpoint"
            observation_for_cost = self.route6_observation_point_for_edge(
                str(edge.get("edge", "") or ""),
                center,
                bbox,
                radar_distance_cm=radar,
                scan_z_cm=float(edge.get("scan_z_cm", 450.0) or 450.0),
            )
            travel_basis = math.hypot(float(observation_for_cost.get("x", 0.0) or 0.0) - px, float(observation_for_cost.get("y", 0.0) or 0.0) - py)
            expected_gain = edge_length
            contour_order_bonus = {"south": 0.04, "east": 0.03, "north": 0.02, "west": 0.01}.get(str(edge.get("edge", "") or ""), 0.0)
            algorithm_score = (1.0 / (travel_basis + 1.0)) + contour_order_bonus
            source_paper_family = "UAV structural-inspection contour standoff"

        observation = self.route6_test_planner_observation_from_anchor(edge, bbox, anchor)
        edge["observation_point_cm"] = observation
        edge["algorithm"] = normalized_algorithm
        edge["algorithm_label"] = self.route6_test_planner_algorithm_label(normalized_algorithm)
        edge["algorithm_score"] = round(float(algorithm_score), 6)
        edge["algorithm_components"] = {
            "source_paper_family": source_paper_family,
            "anchor_policy": anchor_policy,
            "anchor_point_cm": self.route6_json_safe(anchor),
            "edge_length_cm": round(float(edge_length), 2),
            "expected_information_gain_cm": round(float(expected_gain), 2),
            "travel_cost_cm": round(float(travel_basis), 2),
            "score_formula": self.route6_test_planner_algorithm_score_formula(normalized_algorithm),
        }
        edge["selection_reason"] = f"{source_paper_family}: {anchor_policy}, score={edge['algorithm_score']}"
        return edge

    def route6_test_planner_algorithm_score_formula(self, algorithm: str) -> str:
        formulas = {
            "nearest_edge": "score = 1 / (distance_to_edge_cm + 1)",
            "frontier_based": "score = frontier_edge_length_cm / (distance_to_frontier_cm + 1)",
            "nbv_information_gain": "score = expected_information_gain_cm / (travel_cost_to_viewpoint_cm + 1)",
            "surface_edge_explorer": "score = surface_edge_length_cm / (distance_to_edge_center_cm + 1)",
            "uav_inspection_contour": "score = 1 / (travel_cost_to_dilated_contour_viewpoint_cm + 1) + contour_order_bonus",
        }
        return formulas.get(str(algorithm or ""), formulas["nearest_edge"])

    def route6_scan_coverage_parameters(
        self,
        *,
        radar_distance_cm: float,
        fov_deg: float,
        overlap_ratio: float,
    ) -> Dict[str, float]:
        radar = max(1.0, float(radar_distance_cm))
        fov = max(1.0, min(179.0, float(fov_deg)))
        overlap = max(0.0, min(0.95, float(overlap_ratio)))
        coverage_width = max(1.0, 2.0 * radar * math.tan(math.radians(fov) / 2.0))
        effective_step = max(1.0, coverage_width * (1.0 - overlap))
        return {
            "radar_distance_cm": round(float(radar), 2),
            "fov_deg": round(float(fov), 2),
            "overlap_ratio": round(float(overlap), 3),
            "coverage_width_cm": round(float(coverage_width), 2),
            "effective_step_cm": round(float(effective_step), 2),
        }

    def route6_point_on_edge_progress(self, edge_record: Dict[str, Any], progress_cm: float) -> Dict[str, float]:
        start = edge_record.get("start_cm", {}) if isinstance(edge_record.get("start_cm", {}), dict) else {}
        end = edge_record.get("end_cm", {}) if isinstance(edge_record.get("end_cm", {}), dict) else {}
        length = max(0.0, self.route6_test_planner_edge_length_cm(edge_record))
        t = 0.5 if length <= 0.0 else max(0.0, min(1.0, float(progress_cm) / length))
        return {
            "x": round(float(start.get("x", 0.0) or 0.0) + (float(end.get("x", 0.0) or 0.0) - float(start.get("x", 0.0) or 0.0)) * t, 2),
            "y": round(float(start.get("y", 0.0) or 0.0) + (float(end.get("y", 0.0) or 0.0) - float(start.get("y", 0.0) or 0.0)) * t, 2),
        }

    def route6_merged_interval_length(self, intervals: List[Tuple[float, float]]) -> float:
        cleaned = sorted((max(0.0, float(a)), max(0.0, float(b))) for a, b in intervals if float(b) > float(a))
        merged: List[Tuple[float, float]] = []
        for start, end in cleaned:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return float(sum(end - start for start, end in merged))

    def route6_scan_observation_from_anchor(
        self,
        edge_record: Dict[str, Any],
        bbox: Dict[str, float],
        anchor: Dict[str, float],
    ) -> Dict[str, float]:
        observation = self.route6_observation_point_for_edge(
            str(edge_record.get("edge", "") or ""),
            anchor,
            bbox,
            radar_distance_cm=float(edge_record.get("radar_distance_cm", 850.0) or 850.0),
            scan_z_cm=float(edge_record.get("scan_z_cm", 450.0) or 450.0),
        )
        yaw = math.degrees(
            math.atan2(
                float(anchor.get("y", 0.0) or 0.0) - float(observation.get("y", 0.0) or 0.0),
                float(anchor.get("x", 0.0) or 0.0) - float(observation.get("x", 0.0) or 0.0),
            )
        )
        return {**observation, "yaw_deg": round(float(yaw), 3)}

    def route6_build_scan_coverage_plan_for_edge(
        self,
        edge_record: Dict[str, Any],
        bbox: Dict[str, float],
        *,
        house_id: str,
        house_name: str,
        algorithm: str,
        scan_mode: str,
        fov_deg: float,
        overlap_ratio: float,
        coverage_threshold: float,
    ) -> Dict[str, Any]:
        normalized_mode = self.route6_test_planner_normalize_scan_mode(scan_mode)
        edge_length = self.route6_test_planner_edge_length_cm(edge_record)
        radar = float(edge_record.get("radar_distance_cm", 850.0) or 850.0)
        params = self.route6_scan_coverage_parameters(radar_distance_cm=radar, fov_deg=fov_deg, overlap_ratio=overlap_ratio)
        coverage_width = float(params["coverage_width_cm"])
        effective_step = float(params["effective_step_cm"])
        threshold = max(0.0, min(1.0, float(coverage_threshold)))

        if normalized_mode == "single_best_point":
            nearest = edge_record.get("nearest_edge_point_cm", {}) if isinstance(edge_record.get("nearest_edge_point_cm", {}), dict) else {}
            start = edge_record.get("start_cm", {}) if isinstance(edge_record.get("start_cm", {}), dict) else {}
            progress = math.hypot(
                float(nearest.get("x", 0.0) or 0.0) - float(start.get("x", 0.0) or 0.0),
                float(nearest.get("y", 0.0) or 0.0) - float(start.get("y", 0.0) or 0.0),
            )
            anchor_progress = [max(0.0, min(edge_length, progress))]
            scan_policy = "single_selected_observation_point"
        else:
            if edge_length <= coverage_width:
                point_count = 1
            else:
                point_count = int(math.ceil(max(edge_length - coverage_width, 0.0) / effective_step)) + 1
            point_count = max(1, point_count)
            anchor_progress = [edge_length * (index + 0.5) / float(point_count) for index in range(point_count)]
            scan_policy = "uniform_edge_coverage"
            if normalized_mode == "greedy_coverage_nbv":
                scan_policy = "greedy_coverage_nbv_score_order"

        intervals: List[Tuple[float, float]] = []
        scan_points: List[Dict[str, Any]] = []
        for index, progress in enumerate(anchor_progress, start=1):
            segment_start = max(0.0, float(progress) - coverage_width / 2.0)
            segment_end = min(edge_length, float(progress) + coverage_width / 2.0)
            intervals.append((segment_start, segment_end))
            anchor = self.route6_point_on_edge_progress(edge_record, progress)
            observation = self.route6_scan_observation_from_anchor(edge_record, bbox, anchor)
            scan_points.append(
                {
                    **observation,
                    "schema": "route6_scan_observation_point_v1",
                    "scan_index": int(index),
                    "house_id": str(house_id or ""),
                    "house_name": str(house_name or ""),
                    "edge": str(edge_record.get("edge", "") or ""),
                    "algorithm": self.route6_test_planner_normalize_algorithm(algorithm),
                    "scan_mode": normalized_mode,
                    "anchor_point_cm": self.route6_json_safe(anchor),
                    "edge_progress_cm": round(float(progress), 2),
                    "coverage_segment_cm": {
                        "start": round(float(segment_start), 2),
                        "end": round(float(segment_end), 2),
                    },
                    "coverage_width_cm": round(float(coverage_width), 2),
                    "selection_reason": f"{self.route6_test_planner_scan_mode_label(normalized_mode)} scan point {index}",
                }
            )

        covered_length = min(edge_length, self.route6_merged_interval_length(intervals))
        coverage_ratio = 1.0 if edge_length <= 0.0 else max(0.0, min(1.0, covered_length / edge_length))
        scan_satisfied = bool(coverage_ratio + 1e-9 >= threshold)
        return {
            "schema": "route6_scan_coverage_plan_v1",
            "scan_mode": normalized_mode,
            "scan_mode_label": self.route6_test_planner_scan_mode_label(normalized_mode),
            "scan_policy": scan_policy,
            "house_id": str(house_id or ""),
            "edge": str(edge_record.get("edge", "") or ""),
            "edge_length_cm": round(float(edge_length), 2),
            "coverage_width_cm": round(float(coverage_width), 2),
            "effective_step_cm": round(float(effective_step), 2),
            "fov_deg": float(params["fov_deg"]),
            "overlap_ratio": float(params["overlap_ratio"]),
            "coverage_threshold": round(float(threshold), 3),
            "required_point_count": len(scan_points),
            "covered_length_cm": round(float(covered_length), 2),
            "uncovered_length_cm": round(float(max(0.0, edge_length - covered_length)), 2),
            "coverage_ratio": round(float(coverage_ratio), 4),
            "scan_satisfied": scan_satisfied,
            "scan_observation_points": self.route6_json_safe(scan_points),
        }

    def route6_layer_grid_index_for_point(self, metadata: Dict[str, Any], x_cm: float, y_cm: float) -> Dict[str, Any]:
        try:
            width = int(metadata.get("width", 0) or 0)
            height = int(metadata.get("height", 0) or 0)
            resolution = float(metadata.get("resolution_m", 0.25) or 0.25)
            origin_x, origin_y = [float(value) for value in metadata.get("origin_standard_m", [0.0, 0.0])]
        except Exception:
            return {"in_bounds": False, "reason": "invalid_metadata"}
        if width <= 0 or height <= 0 or resolution <= 0.0:
            return {"in_bounds": False, "reason": "invalid_grid_shape"}
        standard_x = float(x_cm) / 100.0
        standard_y = -float(y_cm) / 100.0
        col = int(math.floor((standard_x - origin_x) / resolution))
        row = int(math.floor((standard_y - origin_y) / resolution))
        return {
            "in_bounds": bool(0 <= col < width and 0 <= row < height),
            "row": int(row),
            "col": int(col),
            "width": int(width),
            "height": int(height),
            "resolution_m": float(resolution),
        }

    def route6_test_planner_load_selected_layer_grid(
        self,
        *,
        output_dir: Optional[Path] = None,
        selected_layer_key: str = "",
    ) -> Dict[str, Any]:
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        if out_path is None:
            return {"status": "missing_output_dir"}
        manifest = self.route6_update_map_load_manifest(build_if_missing=False, output_dir=out_path)
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        layer_key = selected_layer_key or self.route6_choose_realtime_layer_key(layers, str(self.route6_update_map_layer_var.get() or ""))
        layer_record = next((layer for layer in layers if isinstance(layer, dict) and self._route6_update_map_layer_key(layer) == layer_key), {})
        if not layer_record:
            return {"status": "missing_layer", "selected_layer_key": layer_key}
        metadata = self.route6_update_map_load_layer_metadata(layer_record)
        grid_path = Path(str(layer_record.get("occupancy_grid_path", "") or ""))
        grid = np.zeros((0, 0), dtype=np.int16)
        if grid_path.is_file():
            try:
                grid = np.asarray(np.load(grid_path), dtype=np.int16)
            except Exception:
                grid = np.zeros((0, 0), dtype=np.int16)
        return {
            "status": "ok" if metadata and grid.size else "missing_grid",
            "output_dir": str(out_path),
            "selected_layer_key": layer_key,
            "layer_record": self.route6_json_safe(layer_record),
            "metadata": metadata,
            "grid": grid,
        }

    def route6_test_planner_house_containing_point(
        self,
        point: Dict[str, Any],
        *,
        target_house_id: str = "",
    ) -> List[Dict[str, Any]]:
        x = float(point.get("x", 0.0) or 0.0)
        y = float(point.get("y", 0.0) or 0.0)
        hits: List[Dict[str, Any]] = []
        for record in self.route6_test_planner_house_records():
            if not isinstance(record, dict):
                continue
            bbox = self.route6_house_record_bbox(record)
            if not bbox:
                continue
            if float(bbox["min_x"]) <= x <= float(bbox["max_x"]) and float(bbox["min_y"]) <= y <= float(bbox["max_y"]):
                house_id = self.route6_test_planner_normalize_house_id(record.get("house_id", record.get("id", "")))
                hits.append(
                    {
                        "house_id": house_id,
                        "name": str(record.get("name", f"House_{house_id}") or f"House_{house_id}"),
                        "is_target_house": house_id == self.route6_test_planner_normalize_house_id(target_house_id),
                        "bbox": self.route6_json_safe(bbox),
                    }
                )
        return hits

    def route6_test_planner_near_obstacle_report(
        self,
        point: Dict[str, Any],
        grid_info: Dict[str, Any],
        *,
        radius_cm: float = 150.0,
    ) -> Dict[str, Any]:
        metadata = grid_info.get("metadata", {}) if isinstance(grid_info.get("metadata", {}), dict) else {}
        grid = np.asarray(grid_info.get("grid", np.zeros((0, 0), dtype=np.int16)))
        if not metadata or grid.size <= 0:
            return {"problem": False, "reason": "missing_grid"}
        index = self.route6_layer_grid_index_for_point(metadata, float(point.get("x", 0.0) or 0.0), float(point.get("y", 0.0) or 0.0))
        if not bool(index.get("in_bounds", False)):
            return {"problem": True, "reason": "map_boundary", "grid_index": index}
        resolution = max(0.05, float(index.get("resolution_m", metadata.get("resolution_m", 0.25)) or 0.25))
        radius_cells = max(1, int(math.ceil((float(radius_cm) / 100.0) / resolution)))
        row = int(index.get("row", 0) or 0)
        col = int(index.get("col", 0) or 0)
        r0 = max(0, row - radius_cells)
        r1 = min(grid.shape[0], row + radius_cells + 1)
        c0 = max(0, col - radius_cells)
        c1 = min(grid.shape[1], col + radius_cells + 1)
        patch = grid[r0:r1, c0:c1]
        occupied_count = int(np.sum(patch >= 100))
        return {
            "problem": bool(occupied_count > 0),
            "reason": "near_obstacle" if occupied_count > 0 else "",
            "grid_index": index,
            "radius_cm": round(float(radius_cm), 2),
            "radius_cells": int(radius_cells),
            "occupied_cell_count": occupied_count,
        }

    def route6_test_planner_front_blocked_report(
        self,
        point: Dict[str, Any],
        anchor: Dict[str, Any],
        grid_info: Dict[str, Any],
        *,
        samples: int = 24,
        anchor_clearance_cm: float = 180.0,
    ) -> Dict[str, Any]:
        metadata = grid_info.get("metadata", {}) if isinstance(grid_info.get("metadata", {}), dict) else {}
        grid = np.asarray(grid_info.get("grid", np.zeros((0, 0), dtype=np.int16)))
        if not metadata or grid.size <= 0 or not anchor:
            return {"problem": False, "reason": "missing_grid_or_anchor"}
        blocked: List[Dict[str, Any]] = []
        ignored_anchor_adjacent = 0
        x0 = float(point.get("x", 0.0) or 0.0)
        y0 = float(point.get("y", 0.0) or 0.0)
        x1 = float(anchor.get("x", 0.0) or 0.0)
        y1 = float(anchor.get("y", 0.0) or 0.0)
        count = max(3, int(samples))
        clearance = max(0.0, float(anchor_clearance_cm))
        for idx in range(1, count):
            t = float(idx) / float(count)
            if t > 0.85:
                continue
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            distance_to_anchor = math.hypot(x1 - x, y1 - y)
            if distance_to_anchor <= clearance:
                ignored_anchor_adjacent += 1
                continue
            index = self.route6_layer_grid_index_for_point(metadata, x, y)
            if not bool(index.get("in_bounds", False)):
                continue
            row = int(index.get("row", 0) or 0)
            col = int(index.get("col", 0) or 0)
            if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and int(grid[row, col]) >= 100:
                blocked.append({"t": round(t, 3), "x": round(float(x), 2), "y": round(float(y), 2), "row": row, "col": col})
                if len(blocked) >= 8:
                    break
        return {
            "problem": bool(blocked),
            "reason": "front_blocked" if blocked else "",
            "blocked_samples": blocked,
            "sample_count": count,
            "anchor_clearance_cm": round(float(clearance), 2),
            "ignored_anchor_adjacent_sample_count": int(ignored_anchor_adjacent),
        }

    def route6_test_planner_observation_safety_report(
        self,
        point: Dict[str, Any],
        *,
        anchor: Optional[Dict[str, Any]] = None,
        target_house_id: str = "",
        output_dir: Optional[Path] = None,
        selected_layer_key: str = "",
        obstacle_radius_cm: float = 150.0,
        warning_radius_cm: float = 50.0,
    ) -> Dict[str, Any]:
        grid_info = self.route6_test_planner_load_selected_layer_grid(output_dir=output_dir, selected_layer_key=selected_layer_key)
        problems: List[str] = []
        details: Dict[str, Any] = {}
        house_hits = self.route6_test_planner_house_containing_point(point, target_house_id=target_house_id)
        if house_hits:
            problems.append("inside_house_bbox")
            details["inside_house_bbox"] = house_hits
        warning = self.route6_test_planner_near_obstacle_report(point, grid_info, radius_cm=float(warning_radius_cm))
        if bool(warning.get("problem", False)):
            problems.append("near_warning_50cm")
        details["warning_zone_50cm"] = self.route6_json_safe(warning)
        near = self.route6_test_planner_near_obstacle_report(point, grid_info, radius_cm=float(obstacle_radius_cm))
        if float(obstacle_radius_cm) > float(warning_radius_cm) and bool(near.get("problem", False)):
            problems.append(str(near.get("reason", "near_obstacle") or "near_obstacle"))
        details["near_obstacle"] = self.route6_json_safe(near)
        front = self.route6_test_planner_front_blocked_report(point, anchor or {}, grid_info)
        if bool(front.get("problem", False)):
            problems.append("front_blocked")
        details["front_blocked"] = self.route6_json_safe(front)
        return {
            "safe": not bool(problems),
            "problems": problems,
            "details": self.route6_json_safe(details),
            "grid_status": str(grid_info.get("status", "") or ""),
            "selected_layer_key": str(grid_info.get("selected_layer_key", selected_layer_key) or selected_layer_key),
        }

    def route6_test_planner_anchor_for_point(self, point: Dict[str, Any], selected_edge: Dict[str, Any]) -> Dict[str, float]:
        return route6_analysis_anchor_for_point(point, selected_edge)

    def route6_test_planner_reset_distance_candidates(
        self,
        *,
        base_distance_cm: float,
        reset_step_cm: float,
        max_attempts: int,
        problems: List[str],
        min_distance_cm: float = 50.0,
    ) -> List[float]:
        base = max(float(min_distance_cm), float(base_distance_cm))
        step = max(1.0, float(reset_step_cm))
        attempts = max(1, int(max_attempts))
        problem_set = {str(item or "") for item in problems if str(item or "")}
        values: List[float] = []
        if "front_blocked" not in problem_set:
            return values

        def add(value: float) -> None:
            rounded = round(max(float(min_distance_cm), float(value)), 2)
            if rounded not in values:
                values.append(rounded)

        for index in range(1, attempts + 1):
            add(base - step * index)
            if len(values) >= attempts:
                return values
        for index in range(1, attempts + 1):
            add(base + step * index)
            if len(values) >= attempts:
                return values
        return values

    def route6_test_planner_micro_adjust_candidates(
        self,
        point: Dict[str, Any],
        anchor: Dict[str, Any],
        *,
        step_cm: float = 50.0,
        max_attempts: int = 12,
    ) -> List[Dict[str, Any]]:
        x0 = float(point.get("x", 0.0) or 0.0)
        y0 = float(point.get("y", 0.0) or 0.0)
        ax = float(anchor.get("x", x0) or x0)
        ay = float(anchor.get("y", y0) or y0)
        nx = x0 - ax
        ny = y0 - ay
        norm = math.hypot(nx, ny)
        if norm <= 1e-6:
            edge = str(point.get("edge", "") or "")
            if edge == "south":
                nx, ny = 0.0, -1.0
            elif edge == "north":
                nx, ny = 0.0, 1.0
            elif edge == "east":
                nx, ny = 1.0, 0.0
            else:
                nx, ny = -1.0, 0.0
            norm = 1.0
        nx /= norm
        ny /= norm
        tx = -ny
        ty = nx
        step = max(10.0, float(step_cm))
        offsets: List[Tuple[float, float, str]] = []

        def add(dx: float, dy: float, reason: str) -> None:
            rounded = (round(float(dx), 2), round(float(dy), 2), reason)
            if not any(abs(item[0] - rounded[0]) < 1e-6 and abs(item[1] - rounded[1]) < 1e-6 for item in offsets):
                offsets.append(rounded)

        for multiplier in (1.0, 2.0, 3.0):
            add(tx * step * multiplier, ty * step * multiplier, "tangent_positive")
            add(-tx * step * multiplier, -ty * step * multiplier, "tangent_negative")
            add(nx * step * multiplier, ny * step * multiplier, "outward_normal")
            add(tx * step * multiplier + nx * step, ty * step * multiplier + ny * step, "tangent_positive_outward")
            add(-tx * step * multiplier + nx * step, -ty * step * multiplier + ny * step, "tangent_negative_outward")
            if len(offsets) >= max_attempts:
                break

        candidates: List[Dict[str, Any]] = []
        for attempt, (dx, dy, reason) in enumerate(offsets[: max(1, int(max_attempts))], start=1):
            candidate = dict(point)
            candidate["x"] = round(x0 + dx, 2)
            candidate["y"] = round(y0 + dy, 2)
            candidate["yaw_deg"] = round(math.degrees(math.atan2(ay - float(candidate["y"]), ax - float(candidate["x"]))), 3)
            candidate["micro_adjust_from_point"] = self.route6_json_safe(point)
            candidate["micro_adjust_attempt"] = int(attempt)
            candidate["micro_adjust_reason"] = reason
            candidate["micro_adjust_offset_cm"] = {"dx": round(dx, 2), "dy": round(dy, 2)}
            candidates.append(candidate)
        return candidates

    def route6_test_planner_micro_adjust_single_point(
        self,
        point: Dict[str, Any],
        selected_edge: Dict[str, Any],
        bbox: Dict[str, float],
        *,
        target_house_id: str,
        output_dir: Optional[Path],
        selected_layer_key: str,
        warning_radius_cm: float = 50.0,
        max_attempts: int = 12,
        initial_safety_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        anchor = self.route6_test_planner_anchor_for_point(point, selected_edge)
        old_point = self.route6_json_safe(point)
        initial = (
            initial_safety_report
            if isinstance(initial_safety_report, dict)
            else self.route6_test_planner_observation_safety_report(
                point,
                anchor=anchor,
                target_house_id=target_house_id,
                output_dir=output_dir,
                selected_layer_key=selected_layer_key,
            )
        )
        attempts: List[Dict[str, Any]] = []
        for candidate in self.route6_test_planner_micro_adjust_candidates(
            point,
            anchor,
            step_cm=float(warning_radius_cm),
            max_attempts=max_attempts,
        ):
            report = self.route6_test_planner_observation_safety_report(
                candidate,
                anchor=anchor,
                target_house_id=target_house_id,
                output_dir=output_dir,
                selected_layer_key=selected_layer_key,
            )
            attempts.append(
                {
                    "attempt": int(candidate.get("micro_adjust_attempt", len(attempts) + 1) or len(attempts) + 1),
                    "point": self.route6_json_safe(candidate),
                    "safety_report": self.route6_json_safe(report),
                }
            )
            problems = set(str(item or "") for item in report.get("problems", []) if str(item or ""))
            blocking_problems = {"near_warning_50cm", "front_blocked", "inside_house_bbox", "map_boundary"}
            if not bool(problems & blocking_problems):
                return {
                    "reset_status": "micro_adjust_ok",
                    "problems": list(initial.get("problems", [])),
                    "old_point": old_point,
                    "new_point": self.route6_json_safe(candidate),
                    "attempts": self.route6_json_safe(attempts),
                    "initial_safety_report": self.route6_json_safe(initial),
                    "final_safety_report": self.route6_json_safe(report),
                    "warning_radius_cm": round(float(warning_radius_cm), 2),
                }
        final_report = attempts[-1]["safety_report"] if attempts else initial
        return {
            "reset_status": "micro_adjust_failed",
            "problems": list(initial.get("problems", [])),
            "old_point": old_point,
            "new_point": self.route6_json_safe(point),
            "attempts": self.route6_json_safe(attempts),
            "initial_safety_report": self.route6_json_safe(initial),
            "final_safety_report": self.route6_json_safe(final_report),
            "warning_radius_cm": round(float(warning_radius_cm), 2),
        }

    def route6_test_planner_reset_single_point(
        self,
        point: Dict[str, Any],
        selected_edge: Dict[str, Any],
        bbox: Dict[str, float],
        *,
        target_house_id: str,
        output_dir: Optional[Path],
        selected_layer_key: str,
        reset_step_cm: float,
        max_attempts: int = 8,
    ) -> Dict[str, Any]:
        edge = str(point.get("edge", selected_edge.get("edge", "")) or selected_edge.get("edge", "") or "")
        anchor = self.route6_test_planner_anchor_for_point(point, selected_edge)
        old_point = self.route6_json_safe(point)
        initial = self.route6_test_planner_observation_safety_report(
            point,
            anchor=anchor,
            target_house_id=target_house_id,
            output_dir=output_dir,
            selected_layer_key=selected_layer_key,
        )
        base_distance = float(point.get("radar_distance_cm", selected_edge.get("radar_distance_cm", 850.0)) or selected_edge.get("radar_distance_cm", 850.0) or 850.0)
        attempts: List[Dict[str, Any]] = []
        if bool(initial.get("safe", False)):
            return {
                "reset_status": "already_safe",
                "problems": [],
                "old_point": old_point,
                "new_point": self.route6_json_safe(point),
                "attempts": [],
                "initial_safety_report": self.route6_json_safe(initial),
            }
        initial_problems = list(initial.get("problems", []))
        if "near_warning_50cm" in initial_problems and "front_blocked" not in initial_problems:
            return self.route6_test_planner_micro_adjust_single_point(
                point,
                selected_edge,
                bbox,
                target_house_id=target_house_id,
                output_dir=output_dir,
                selected_layer_key=selected_layer_key,
                initial_safety_report=initial,
            )
        if "front_blocked" not in initial_problems:
            return {
                "reset_status": "no_reset_needed",
                "problems": initial_problems,
                "old_point": old_point,
                "new_point": self.route6_json_safe(point),
                "attempts": [],
                "initial_safety_report": self.route6_json_safe(initial),
                "reset_policy": "report_only_without_front_blocked",
            }
        last_candidate: Dict[str, Any] = dict(point)
        last_report = initial
        distance_candidates = self.route6_test_planner_reset_distance_candidates(
            base_distance_cm=base_distance,
            reset_step_cm=reset_step_cm,
            max_attempts=max_attempts,
            problems=list(initial.get("problems", [])),
        )
        for attempt, distance in enumerate(distance_candidates, start=1):
            candidate_obs = self.route6_observation_point_for_edge(
                edge,
                anchor,
                bbox,
                radar_distance_cm=distance,
                scan_z_cm=float(point.get("z", selected_edge.get("scan_z_cm", 450.0)) or selected_edge.get("scan_z_cm", 450.0) or 450.0),
            )
            yaw = math.degrees(
                math.atan2(
                    float(anchor.get("y", 0.0) or 0.0) - float(candidate_obs.get("y", 0.0) or 0.0),
                    float(anchor.get("x", 0.0) or 0.0) - float(candidate_obs.get("x", 0.0) or 0.0),
                )
            )
            candidate = {
                **dict(point),
                **candidate_obs,
                "yaw_deg": round(float(yaw), 3),
                "radar_distance_cm": round(float(distance), 2),
                "reset_from_point": old_point,
                "reset_attempt": int(attempt),
            }
            report = self.route6_test_planner_observation_safety_report(
                candidate,
                anchor=anchor,
                target_house_id=target_house_id,
                output_dir=output_dir,
                selected_layer_key=selected_layer_key,
            )
            attempt_record = {
                "attempt": int(attempt),
                "radar_distance_cm": round(float(distance), 2),
                "point": self.route6_json_safe(candidate),
                "safety_report": self.route6_json_safe(report),
            }
            attempts.append(attempt_record)
            last_candidate = candidate
            last_report = report
            if bool(report.get("safe", False)):
                return {
                    "reset_status": "ok",
                    "problems": list(initial.get("problems", [])),
                    "old_point": old_point,
                    "new_point": self.route6_json_safe(candidate),
                    "attempts": self.route6_json_safe(attempts),
                    "initial_safety_report": self.route6_json_safe(initial),
                }
        return {
            "reset_status": "failed",
            "problems": list(initial.get("problems", [])),
            "old_point": old_point,
            "new_point": self.route6_json_safe(last_candidate),
            "attempts": self.route6_json_safe(attempts),
            "initial_safety_report": self.route6_json_safe(initial),
            "final_safety_report": self.route6_json_safe(last_report),
        }

    def route6_update_plan_point_after_reset(self, plan: Dict[str, Any], old_point: Dict[str, Any], new_point: Dict[str, Any]) -> None:
        house_id = str(old_point.get("house_id", "") or "")
        edge = str(old_point.get("edge", "") or "")
        scan_index = old_point.get("scan_index", None)
        def matches(item: Dict[str, Any]) -> bool:
            if str(item.get("house_id", "") or "") != house_id or str(item.get("edge", "") or "") != edge:
                return False
            if scan_index is not None:
                return int(item.get("scan_index", -1) or -1) == int(scan_index)
            return True

        if isinstance(plan.get("selected_observation_point", {}), dict) and matches(plan["selected_observation_point"]):
            plan["selected_observation_point"] = self.route6_json_safe(new_point)
        point_list_keys = ("scan_observation_points", "selected_scan_observation_points") if scan_index is not None else ("observation_points",)
        for key in point_list_keys:
            values = plan.get(key, []) if isinstance(plan.get(key, []), list) else []
            for index, item in enumerate(values):
                if isinstance(item, dict) and matches(item):
                    values[index] = self.route6_json_safe({**item, **new_point})
                    if scan_index is None:
                        break
        for house_plan in plan.get("house_plans", []) if isinstance(plan.get("house_plans", []), list) else []:
            if not isinstance(house_plan, dict) or str(house_plan.get("house_id", "") or "") != house_id:
                continue
            if isinstance(house_plan.get("observation_point", {}), dict) and matches(house_plan["observation_point"]):
                house_plan["observation_point"] = self.route6_json_safe(new_point)
            selected_edge = house_plan.get("selected_edge", {}) if isinstance(house_plan.get("selected_edge", {}), dict) else {}
            if str(selected_edge.get("edge", "") or "") == edge and scan_index is None:
                selected_edge["observation_point_cm"] = self.route6_json_safe(new_point)
            if scan_index is None:
                continue
            scan_plan = house_plan.get("scan_plan", {}) if isinstance(house_plan.get("scan_plan", {}), dict) else {}
            scan_points = scan_plan.get("scan_observation_points", []) if isinstance(scan_plan.get("scan_observation_points", []), list) else []
            for index, item in enumerate(scan_points):
                if isinstance(item, dict) and matches(item):
                    scan_points[index] = self.route6_json_safe({**item, **new_point})

    def route6_test_planner_reset_summary(self, main_reset: Dict[str, Any], scan_resets: List[Dict[str, Any]]) -> Dict[str, Any]:
        return route6_analysis_reset_summary(main_reset, scan_resets)

    def route6_reset_current_observation_point(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        self.ensure_route6_state()
        plan = self.llm_route6_state.get("route6_offline_test_plan", {}) if isinstance(self.llm_route6_state, dict) else {}
        if not isinstance(plan, dict) or not plan:
            plan = self.route6_build_offline_test_plan(output_dir=output_dir)
        out_path = Path(output_dir) if output_dir is not None else Path(str(plan.get("output_dir", "") or self.route6_update_map_latest_output_dir() or "."))
        selected = plan.get("selected_observation_point", {}) if isinstance(plan.get("selected_observation_point", {}), dict) else {}
        if not selected:
            result = {"reset_status": "no_current_observation_point", "problems": ["missing_selected_observation_point"]}
            self.route6_test_planner_status_var.set("Route 6 Test Planner reset: no current observation point.")
            return result
        house_id = str(selected.get("house_id", "") or "")
        selected_house_plan = next(
            (
                item
                for item in plan.get("house_plans", [])
                if isinstance(item, dict) and str(item.get("house_id", "") or "") == house_id
            ),
            {},
        )
        bbox = selected_house_plan.get("bbox", {}) if isinstance(selected_house_plan.get("bbox", {}), dict) else {}
        selected_edge = selected_house_plan.get("selected_edge", {}) if isinstance(selected_house_plan.get("selected_edge", {}), dict) else {}
        if not bbox or not selected_edge:
            result = {"reset_status": "missing_context", "problems": ["missing_house_bbox_or_edge"], "old_point": self.route6_json_safe(selected)}
            self.route6_test_planner_status_var.set("Route 6 Test Planner reset: missing bbox or edge context.")
            return result
        reset_step = self.route6_float_param(self.route6_test_planner_reset_step_cm_var, 150.0, min_value=1.0, max_value=5000.0)
        layer_key = str(plan.get("selected_layer_key", self.route6_update_map_layer_var.get() if hasattr(self.route6_update_map_layer_var, "get") else "") or "")
        selected_scan_points = list(plan.get("selected_scan_observation_points", []) if isinstance(plan.get("selected_scan_observation_points", []), list) else [])
        active_reset_target = route6_analysis_active_reset_target(plan)
        if active_reset_target == "scan_observation_points":
            main_reset = {
                "reset_status": "reference_not_reset",
                "problems": [],
                "old_point": self.route6_json_safe(selected),
                "new_point": self.route6_json_safe(selected),
                "attempts": [],
                "reset_policy": "multi_point_base_observation_is_reference_only",
            }
        else:
            main_reset = self.route6_test_planner_reset_single_point(
                selected,
                selected_edge,
                bbox,
                target_house_id=house_id,
                output_dir=out_path,
                selected_layer_key=layer_key,
                reset_step_cm=reset_step,
            )
        writeback_statuses = {"ok", "micro_adjust_ok"}
        if str(main_reset.get("reset_status", "")) in writeback_statuses and isinstance(main_reset.get("new_point", {}), dict):
            self.route6_update_plan_point_after_reset(plan, selected, main_reset["new_point"])
        scan_resets: List[Dict[str, Any]] = []
        for scan_point in selected_scan_points:
            if not isinstance(scan_point, dict):
                continue
            scan_reset = self.route6_test_planner_reset_single_point(
                scan_point,
                selected_edge,
                bbox,
                target_house_id=house_id,
                output_dir=out_path,
                selected_layer_key=layer_key,
                reset_step_cm=reset_step,
            )
            scan_resets.append(scan_reset)
            if str(scan_reset.get("reset_status", "")) in writeback_statuses and isinstance(scan_reset.get("new_point", {}), dict):
                self.route6_update_plan_point_after_reset(plan, scan_point, scan_reset["new_point"])
        reset_summary = self.route6_test_planner_reset_summary(main_reset, scan_resets)
        result = {
            **self.route6_json_safe(main_reset),
            "reset_status": reset_summary["reset_status"],
            "main_reset_status": reset_summary["main_reset_status"],
            "problems": self.route6_json_safe(reset_summary["problems"]),
            "schema": "route6_observation_point_reset_v1",
            "active_reset_target": active_reset_target,
            "reset_step_cm": round(float(reset_step), 2),
            "selected_layer_key": layer_key,
            "scan_point_resets": self.route6_json_safe(scan_resets),
            "scan_reset_summary": self.route6_json_safe(reset_summary),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        plan["observation_point_reset"] = self.route6_json_safe(result)
        plan["visual_calculation_records"] = self.route6_json_safe(
            self.route6_test_planner_visual_calculation_records(plan, output_dir=out_path, selected_layer_key=layer_key)
        )
        artifact_path = Path(str(plan.get("artifact_path", "") or (out_path / "route6_offline_test_plan.json")))
        plan["artifact_path"] = str(artifact_path)
        analysis_dir = Path(str(plan.get("analysis_dir", "") or (out_path / "route6_test_planner_analysis")))
        visual_records_path = analysis_dir / "visual_calculation_records.json"
        reset_report_path = analysis_dir / "observation_point_reset.json"
        plan["analysis_dir"] = str(analysis_dir)
        plan["visual_calculation_records_path"] = str(visual_records_path)
        plan["observation_point_reset_path"] = str(reset_report_path)
        self.route6_write_json_artifact(
            visual_records_path,
            {
                "schema": "route6_test_planner_visual_calculation_records_v1",
                "plan_artifact_path": str(artifact_path),
                "selected_layer_key": layer_key,
                "records": self.route6_json_safe(plan["visual_calculation_records"]),
            },
        )
        self.route6_write_json_artifact(
            reset_report_path,
            {
                "schema": "route6_test_planner_observation_point_reset_report_v1",
                "plan_artifact_path": str(artifact_path),
                "reset": self.route6_json_safe(result),
            },
        )
        self.route6_write_json_artifact(artifact_path, plan)
        self.llm_route6_state["route6_offline_test_plan"] = self.route6_json_safe(plan)
        self.llm_route6_state["route6_observation_point_reset"] = self.route6_json_safe(result)
        self.route6_write_state_artifact()
        self.refresh_route6_test_planner_point_status(plan)
        problems = ",".join(result.get("problems", [])) or "none"
        self.route6_test_planner_status_var.set(f"Route 6 Test Planner reset: {result.get('reset_status', 'n/a')} problems={problems}")
        return result

    def route6_selected_test_planner_house_ids(self) -> List[str]:
        listbox = getattr(self, "route6_test_planner_house_listbox", None)
        records = getattr(self, "route6_test_planner_house_record_cache", None)
        if not isinstance(records, list):
            records = self.route6_test_planner_house_records()
        if listbox is None:
            return []
        try:
            selected = [int(index) for index in listbox.curselection()]
        except Exception:
            selected = []
        house_ids: List[str] = []
        for index in selected:
            if 0 <= index < len(records):
                item = records[index] if isinstance(records[index], dict) else {}
                house_id = self.route6_test_planner_normalize_house_id(item.get("house_id", item.get("id", "")))
                if house_id:
                    house_ids.append(house_id)
        return house_ids

    def route6_build_offline_test_plan(
        self,
        output_dir: Optional[Path] = None,
        selected_house_ids: Optional[List[str]] = None,
        *,
        radar_distance_cm: Optional[float] = None,
        scan_z_cm: Optional[float] = None,
        preferred_edge: Optional[str] = None,
        planning_algorithm: Optional[str] = None,
        scan_mode: Optional[str] = None,
        fov_deg: Optional[float] = None,
        overlap_ratio: Optional[float] = None,
        coverage_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        if out_path is None:
            result = {
                "schema": "route6_offline_test_plan_v1",
                "status": "missing_route6_update_map",
                "error": "no Route 6 Update Map output directory is available",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.route6_test_planner_status_var.set("Route 6 Test Planner: no Route 6 Update Map output directory.")
            return result
        pose = self.route6_refresh_realtime_uav_pose()
        self.route6_test_planner_pose_var.set(self.route6_format_uav_pose_text(pose))
        manifest = self.route6_update_map_load_manifest(build_if_missing=False, output_dir=out_path)
        manifest_path = self.route6_update_map_manifest_path(out_path)
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        selected_layer_key = self.route6_choose_realtime_layer_key(layers, str(self.route6_update_map_layer_var.get() or ""))
        if selected_layer_key:
            self.route6_update_map_layer_var.set(selected_layer_key)
        radar = (
            float(radar_distance_cm)
            if radar_distance_cm is not None
            else self.route6_float_param(self.route6_test_planner_radar_distance_cm_var, 850.0, min_value=1.0, max_value=10000.0)
        )
        scan_z = (
            float(scan_z_cm)
            if scan_z_cm is not None
            else self.route6_float_param(self.route6_test_planner_scan_z_cm_var, 450.0, min_value=0.0, max_value=5000.0)
        )
        requested_edge = self.route6_test_planner_normalize_edge(
            preferred_edge if preferred_edge is not None else self.route6_test_planner_edge_var.get()
        )
        algorithm = self.route6_test_planner_normalize_algorithm(
            planning_algorithm if planning_algorithm is not None else self.route6_test_planner_algorithm_var.get()
        )
        selected_scan_mode = self.route6_test_planner_normalize_scan_mode(
            scan_mode if scan_mode is not None else self.route6_test_planner_scan_mode_var.get()
        )
        selected_fov = (
            float(fov_deg)
            if fov_deg is not None
            else self.route6_float_param(self.route6_test_planner_fov_deg_var, 60.0, min_value=1.0, max_value=179.0)
        )
        selected_overlap = (
            float(overlap_ratio)
            if overlap_ratio is not None
            else self.route6_float_param(self.route6_test_planner_overlap_var, 0.30, min_value=0.0, max_value=0.95)
        )
        selected_threshold = (
            float(coverage_threshold)
            if coverage_threshold is not None
            else self.route6_float_param(self.route6_test_planner_coverage_threshold_var, 0.90, min_value=0.0, max_value=1.0)
        )
        edge_selection_mode = "operator_selected_edge" if requested_edge else "auto_nearest_edge"
        requested_ids = [
            self.route6_test_planner_normalize_house_id(value)
            for value in (selected_house_ids if isinstance(selected_house_ids, list) else self.route6_selected_test_planner_house_ids())
            if str(value or "").strip()
        ]
        records = self.route6_test_planner_house_records()
        if requested_ids:
            requested_set = set(requested_ids)
            selected_records = [item for item in records if self.route6_test_planner_normalize_house_id(item.get("house_id", item.get("id", ""))) in requested_set]
        else:
            selected_records = list(records)
        house_plans: List[Dict[str, Any]] = []
        observation_points: List[Dict[str, Any]] = []
        scan_observation_points: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for record in selected_records:
            house_id = self.route6_test_planner_normalize_house_id(record.get("house_id", record.get("id", "")))
            house_name = str(record.get("name", f"House_{house_id}") or f"House_{house_id}")
            bbox = self.route6_house_record_bbox(record)
            if not house_id or not bbox:
                errors.append({"house_id": house_id, "reason": "invalid_house_bbox"})
                continue
            base_edges = self.route6_exploration_edges_for_bbox(bbox, pose, house_id=house_id, radar_distance_cm=radar, scan_z_cm=scan_z)
            edges = [self.route6_apply_test_planner_algorithm_to_edge(edge, bbox, pose, algorithm) for edge in base_edges]
            selected_edge = (
                next((item for item in edges if str(item.get("edge", "") or "") == requested_edge), {})
                if requested_edge
                else {}
            )
            if not selected_edge:
                selected_edge = (
                    max(
                        edges,
                        key=lambda item: (
                            float(item.get("algorithm_score", 0.0) or 0.0),
                            -float(item.get("distance_to_edge_cm", 0.0) or 0.0),
                            str(item.get("edge", "") or ""),
                        ),
                    )
                    if edges
                    else {}
                )
            observation = dict(selected_edge.get("observation_point_cm", {}) if isinstance(selected_edge.get("observation_point_cm", {}), dict) else {})
            observation_point = {
                **observation,
                "house_id": house_id,
                "house_name": house_name,
                "edge": str(selected_edge.get("edge", "") or ""),
                "algorithm": algorithm,
                "algorithm_label": self.route6_test_planner_algorithm_label(algorithm),
                "algorithm_score": float(selected_edge.get("algorithm_score", 0.0) or 0.0),
                "algorithm_components": self.route6_json_safe(selected_edge.get("algorithm_components", {})),
                "radar_distance_cm": round(float(radar), 2),
                "edge_center_cm": self.route6_json_safe(selected_edge.get("edge_center_cm", {})),
                "nearest_edge_point_cm": self.route6_json_safe(selected_edge.get("nearest_edge_point_cm", {})),
                "distance_to_edge_cm": float(selected_edge.get("distance_to_edge_cm", 0.0) or 0.0),
                "selection_reason": (
                    f"operator selected {requested_edge} exploration edge"
                    if requested_edge
                    else f"{self.route6_test_planner_algorithm_label(algorithm)} selected by score"
                ),
            }
            scan_plan = self.route6_build_scan_coverage_plan_for_edge(
                selected_edge,
                bbox,
                house_id=house_id,
                house_name=house_name,
                algorithm=algorithm,
                scan_mode=selected_scan_mode,
                fov_deg=selected_fov,
                overlap_ratio=selected_overlap,
                coverage_threshold=selected_threshold,
            )
            house_scan_points = scan_plan.get("scan_observation_points", []) if isinstance(scan_plan.get("scan_observation_points", []), list) else []
            scan_observation_points.extend(item for item in house_scan_points if isinstance(item, dict))
            observation_point["scan_point_count"] = int(scan_plan.get("required_point_count", 0) or 0)
            observation_point["scan_coverage_ratio"] = float(scan_plan.get("coverage_ratio", 0.0) or 0.0)
            observation_point["scan_satisfied"] = bool(scan_plan.get("scan_satisfied", False))
            house_plan = {
                "schema": "route6_offline_house_test_plan_v1",
                "house_id": house_id,
                "name": house_name,
                "source": str(record.get("source", "") or ""),
                "bbox": self.route6_json_safe(bbox),
                "edge_calculations": self.route6_json_safe(edges),
                "selected_edge": self.route6_json_safe(selected_edge),
                "observation_point": self.route6_json_safe(observation_point),
                "scan_plan": self.route6_json_safe(scan_plan),
            }
            house_plans.append(house_plan)
            observation_points.append(observation_point)
        observation_points.sort(
            key=lambda item: (
                -float(item.get("algorithm_score", 0.0) or 0.0),
                float(item.get("distance_to_edge_cm", 0.0) or 0.0),
                str(item.get("house_id", "")),
            )
        )
        selected_observation = dict(observation_points[0]) if observation_points else {}
        scan_observation_points.sort(key=lambda item: (str(item.get("house_id", "")), str(item.get("edge", "")), int(item.get("scan_index", 0) or 0)))
        selected_scan_observation_points = [
            item
            for item in scan_observation_points
            if str(item.get("house_id", "")) == str(selected_observation.get("house_id", ""))
            and str(item.get("edge", "")) == str(selected_observation.get("edge", ""))
        ]
        selected_scan_plan = next(
            (
                house_plan.get("scan_plan", {})
                for house_plan in house_plans
                if str(house_plan.get("house_id", "")) == str(selected_observation.get("house_id", ""))
                and isinstance(house_plan.get("scan_plan", {}), dict)
                and str(house_plan.get("scan_plan", {}).get("edge", "")) == str(selected_observation.get("edge", ""))
            ),
            {},
        )
        scan_satisfied_values = [
            bool(house_plan.get("scan_plan", {}).get("scan_satisfied", False))
            for house_plan in house_plans
            if isinstance(house_plan.get("scan_plan", {}), dict)
        ]
        scan_satisfied = bool(scan_satisfied_values and all(scan_satisfied_values))
        scan_coverage_summary = {
            key: selected_scan_plan.get(key)
            for key in (
                "scan_mode",
                "scan_mode_label",
                "edge",
                "edge_length_cm",
                "coverage_width_cm",
                "effective_step_cm",
                "fov_deg",
                "overlap_ratio",
                "coverage_threshold",
                "required_point_count",
                "covered_length_cm",
                "uncovered_length_cm",
                "coverage_ratio",
                "scan_satisfied",
            )
            if isinstance(selected_scan_plan, dict) and key in selected_scan_plan
        }
        selected_ids = [self.route6_test_planner_normalize_house_id(item.get("house_id", item.get("id", ""))) for item in selected_records]
        artifact_path = Path(out_path) / "route6_offline_test_plan.json"
        plan = {
            "schema": "route6_offline_test_plan_v1",
            "status": "ok" if observation_points else "no_observation_points",
            "output_dir": str(out_path),
            "manifest_path": str(manifest_path) if manifest_path else "",
            "manifest_available": bool(manifest),
            "selected_layer_key": selected_layer_key,
            "current_pose": self.route6_json_safe(pose),
            "radar_distance_cm": round(float(radar), 2),
            "scan_z_cm": round(float(scan_z), 2),
            "planning_algorithm": algorithm,
            "planning_algorithm_label": self.route6_test_planner_algorithm_label(algorithm),
            "scan_mode": selected_scan_mode,
            "scan_mode_label": self.route6_test_planner_scan_mode_label(selected_scan_mode),
            "scan_coverage_config": {
                "fov_deg": round(float(selected_fov), 2),
                "overlap_ratio": round(float(selected_overlap), 3),
                "coverage_threshold": round(float(selected_threshold), 3),
            },
            "requested_edge": requested_edge,
            "edge_selection_mode": edge_selection_mode,
            "selected_house_ids": selected_ids,
            "requested_house_ids": requested_ids,
            "evaluated_all_houses": not bool(requested_ids),
            "house_count": len(house_plans),
            "house_plans": self.route6_json_safe(house_plans),
            "observation_points": self.route6_json_safe(observation_points),
            "selected_observation_point": self.route6_json_safe(selected_observation),
            "scan_observation_points": self.route6_json_safe(scan_observation_points),
            "selected_scan_observation_points": self.route6_json_safe(selected_scan_observation_points),
            "scan_coverage_summary": self.route6_json_safe(scan_coverage_summary),
            "scan_satisfied": scan_satisfied,
            "errors": self.route6_json_safe(errors),
            "artifact_path": str(artifact_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        plan["visual_calculation_records"] = self.route6_json_safe(
            self.route6_test_planner_visual_calculation_records(plan, output_dir=Path(out_path), selected_layer_key=selected_layer_key)
        )
        analysis_dir = Path(out_path) / "route6_test_planner_analysis"
        visual_records_path = analysis_dir / "visual_calculation_records.json"
        plan["analysis_dir"] = str(analysis_dir)
        plan["visual_calculation_records_path"] = str(visual_records_path)
        self.route6_write_json_artifact(
            visual_records_path,
            {
                "schema": "route6_test_planner_visual_calculation_records_v1",
                "plan_artifact_path": str(artifact_path),
                "selected_layer_key": selected_layer_key,
                "records": self.route6_json_safe(plan["visual_calculation_records"]),
            },
        )
        self.route6_write_json_artifact(artifact_path, plan)
        self.llm_route6_state["route6_offline_test_plan"] = self.route6_json_safe(plan)
        self.llm_route6_state["route6_offline_test_plan_path"] = str(artifact_path)
        self.route6_write_state_artifact()
        self.refresh_route6_test_planner_point_status(plan)
        if selected_observation:
            self.route6_test_planner_status_var.set(
                f"Route 6 Test Planner: house={selected_observation.get('house_id', 'n/a')} "
                f"algorithm={self.route6_test_planner_algorithm_label(algorithm)} "
                f"edge={selected_observation.get('edge', 'n/a')} "
                f"scan={len(selected_scan_observation_points)} "
                f"x={float(selected_observation.get('x', 0.0) or 0.0):.1f} "
                f"y={float(selected_observation.get('y', 0.0) or 0.0):.1f}"
            )
        else:
            self.route6_test_planner_status_var.set("Route 6 Test Planner: no valid observation point.")
        return plan

    def route6_test_planner_layer_pixel(
        self,
        metadata: Dict[str, Any],
        x_cm: float,
        y_cm: float,
        *,
        scale: int = 1,
    ) -> Tuple[int, int]:
        px, py = self.route6_unreal_cm_to_layer_pixel(metadata, x_cm, y_cm)
        factor = max(1, int(scale))
        return int(px * factor + factor / 2), int(py * factor + factor / 2)

    def route6_test_planner_house_plan_for_point(self, plan: Dict[str, Any], point: Dict[str, Any]) -> Dict[str, Any]:
        return route6_analysis_house_plan_for_point(plan, point)

    def route6_test_planner_edge_record_for_point(self, plan: Dict[str, Any], point: Dict[str, Any]) -> Dict[str, Any]:
        return route6_analysis_edge_record_for_point(plan, point)

    def route6_test_planner_visual_formula(
        self,
        point: Dict[str, Any],
        anchor: Dict[str, float],
        bbox: Dict[str, Any],
        radar_distance_cm: float,
    ) -> str:
        return route6_analysis_visual_formula(point, anchor, bbox, radar_distance_cm)

    def route6_test_planner_visual_calculation_records(
        self,
        plan: Dict[str, Any],
        *,
        output_dir: Optional[Path] = None,
        selected_layer_key: str = "",
    ) -> List[Dict[str, Any]]:
        return route6_analysis_visual_calculation_records(
            self,
            plan,
            output_dir=output_dir,
            selected_layer_key=selected_layer_key,
        )

    def route6_draw_offline_test_plan_overlay(
        self,
        image: Image.Image,
        layer_record: Dict[str, Any],
        plan: Dict[str, Any],
        *,
        scale: int = 1,
    ) -> Image.Image:
        metadata = self.route6_update_map_load_layer_metadata(layer_record)
        if not metadata:
            return image
        source = image.convert("RGB")
        draw = ImageDraw.Draw(source)
        factor = max(1, int(scale))
        house_plans = plan.get("house_plans", []) if isinstance(plan.get("house_plans", []), list) else []
        selected_point = plan.get("selected_observation_point", {}) if isinstance(plan.get("selected_observation_point", {}), dict) else {}
        visual_records = self.route6_test_planner_visual_calculation_records(
            plan,
            output_dir=Path(str(plan.get("output_dir", "") or ".")),
            selected_layer_key=str(plan.get("selected_layer_key", "") or ""),
        )

        def record_key(record: Dict[str, Any]) -> Tuple[str, str, str, int]:
            return (
                str(record.get("kind", "") or ""),
                str(record.get("house_id", "") or ""),
                str(record.get("edge", "") or ""),
                int(record.get("scan_index", 0) or 0),
            )

        visual_by_key = {record_key(record): record for record in visual_records if isinstance(record, dict)}

        def point_record(kind: str, point: Dict[str, Any]) -> Dict[str, Any]:
            try:
                scan_index = int(point.get("scan_index", 0) or 0)
            except Exception:
                scan_index = 0
            return visual_by_key.get((kind, str(point.get("house_id", "") or ""), str(point.get("edge", "") or ""), scan_index), {})

        def draw_anchor_marker(anchor_px: Tuple[int, int], label: str) -> None:
            radius = max(3, factor * 2)
            draw.ellipse(
                (anchor_px[0] - radius, anchor_px[1] - radius, anchor_px[0] + radius, anchor_px[1] + radius),
                fill=(255, 220, 40),
                outline=(40, 40, 40),
                width=max(1, factor),
            )
            draw.text((anchor_px[0] + radius + 1, anchor_px[1] - radius - 1), label, fill=(90, 70, 0))

        def draw_blocked_samples(record: Dict[str, Any]) -> None:
            safety = record.get("safety_report", {}) if isinstance(record.get("safety_report", {}), dict) else {}
            details = safety.get("details", {}) if isinstance(safety.get("details", {}), dict) else {}
            front = details.get("front_blocked", {}) if isinstance(details.get("front_blocked", {}), dict) else {}
            samples = front.get("blocked_samples", []) if isinstance(front.get("blocked_samples", []), list) else []
            radius = max(2, factor)
            for sample in samples[:8]:
                if not isinstance(sample, dict):
                    continue
                px = self.route6_test_planner_layer_pixel(metadata, float(sample.get("x", 0.0) or 0.0), float(sample.get("y", 0.0) or 0.0), scale=factor)
                draw.line((px[0] - radius, px[1] - radius, px[0] + radius, px[1] + radius), fill=(240, 0, 0), width=max(1, factor))
                draw.line((px[0] - radius, px[1] + radius, px[0] + radius, px[1] - radius), fill=(240, 0, 0), width=max(1, factor))

        def draw_warning_zone(center_px: Tuple[int, int], record: Dict[str, Any]) -> None:
            safety = record.get("safety_report", {}) if isinstance(record.get("safety_report", {}), dict) else {}
            details = safety.get("details", {}) if isinstance(safety.get("details", {}), dict) else {}
            warning = details.get("warning_zone_50cm", {}) if isinstance(details.get("warning_zone_50cm", {}), dict) else {}
            radius_cm = float(warning.get("radius_cm", 50.0) or 50.0)
            resolution = max(0.05, float(metadata.get("resolution_m", 0.25) or 0.25))
            radius_px = max(2, int(round((radius_cm / 100.0) / resolution * factor)))
            problem = bool(warning.get("problem", False))
            outline = (255, 40, 40) if problem else (40, 180, 90)
            draw.ellipse(
                (
                    center_px[0] - radius_px,
                    center_px[1] - radius_px,
                    center_px[0] + radius_px,
                    center_px[1] + radius_px,
                ),
                outline=outline,
                width=max(1, factor),
            )

        for house_plan in house_plans:
            if not isinstance(house_plan, dict):
                continue
            bbox = house_plan.get("bbox", {}) if isinstance(house_plan.get("bbox", {}), dict) else {}
            if all(key in bbox for key in ("min_x", "max_x", "min_y", "max_y")):
                corners = [
                    (bbox["min_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["max_y"]),
                    (bbox["min_x"], bbox["max_y"]),
                ]
                pixels = [self.route6_test_planner_layer_pixel(metadata, float(x), float(y), scale=factor) for x, y in corners]
                if pixels:
                    draw.line(pixels + [pixels[0]], fill=(110, 170, 255), width=max(1, factor))
            edges = house_plan.get("edge_calculations", []) if isinstance(house_plan.get("edge_calculations", []), list) else []
            selected_edge_name = str((house_plan.get("selected_edge", {}) if isinstance(house_plan.get("selected_edge", {}), dict) else {}).get("edge", "") or "")
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                start = edge.get("start_cm", {}) if isinstance(edge.get("start_cm", {}), dict) else {}
                end = edge.get("end_cm", {}) if isinstance(edge.get("end_cm", {}), dict) else {}
                if not start or not end:
                    continue
                p1 = self.route6_test_planner_layer_pixel(metadata, float(start.get("x", 0.0) or 0.0), float(start.get("y", 0.0) or 0.0), scale=factor)
                p2 = self.route6_test_planner_layer_pixel(metadata, float(end.get("x", 0.0) or 0.0), float(end.get("y", 0.0) or 0.0), scale=factor)
                is_selected = str(edge.get("edge", "") or "") == selected_edge_name
                draw.line(p1 + p2, fill=(255, 80, 50) if is_selected else (245, 180, 60), width=max(2, factor * (3 if is_selected else 1)))

        points = plan.get("observation_points", []) if isinstance(plan.get("observation_points", []), list) else []
        for point in points:
            if not isinstance(point, dict):
                continue
            record = point_record("selected_observation", point)
            if record and bool(record.get("display_on_map", True)) is False:
                continue
            obs = self.route6_test_planner_layer_pixel(metadata, float(point.get("x", 0.0) or 0.0), float(point.get("y", 0.0) or 0.0), scale=factor)
            anchor = record.get("anchor_point_cm", {}) if isinstance(record.get("anchor_point_cm", {}), dict) else self.route6_test_planner_anchor_for_point(point, self.route6_test_planner_edge_record_for_point(plan, point))
            if anchor:
                target = self.route6_test_planner_layer_pixel(metadata, float(anchor.get("x", 0.0) or 0.0), float(anchor.get("y", 0.0) or 0.0), scale=factor)
                draw.line(obs + target, fill=(40, 180, 120), width=max(1, factor * 2))
                draw_anchor_marker(target, "A0")
                draw_blocked_samples(record)
            draw_warning_zone(obs, record)
            radius = max(4, factor * 3)
            is_first = str(point.get("house_id", "")) == str(selected_point.get("house_id", "")) and str(point.get("edge", "")) == str(selected_point.get("edge", ""))
            fill = (0, 105, 255) if is_first else (90, 150, 255)
            draw.ellipse((obs[0] - radius, obs[1] - radius, obs[0] + radius, obs[1] + radius), fill=fill, outline=(0, 0, 0), width=max(1, factor))
            draw.text((obs[0] + radius + 2, obs[1] - radius), f"H{point.get('house_id', '')} {point.get('edge', '')} obs", fill=(0, 0, 0))

        scan_points = plan.get("selected_scan_observation_points", [])
        if not isinstance(scan_points, list) or not scan_points:
            scan_points = plan.get("scan_observation_points", []) if isinstance(plan.get("scan_observation_points", []), list) else []
        for point in scan_points:
            if not isinstance(point, dict):
                continue
            record = point_record("scan_observation", point)
            if record and bool(record.get("display_on_map", True)) is False:
                continue
            obs = self.route6_test_planner_layer_pixel(metadata, float(point.get("x", 0.0) or 0.0), float(point.get("y", 0.0) or 0.0), scale=factor)
            coverage_start = record.get("coverage_start_point_cm", {}) if isinstance(record.get("coverage_start_point_cm", {}), dict) else {}
            coverage_end = record.get("coverage_end_point_cm", {}) if isinstance(record.get("coverage_end_point_cm", {}), dict) else {}
            if coverage_start and coverage_end:
                cov0 = self.route6_test_planner_layer_pixel(metadata, float(coverage_start.get("x", 0.0) or 0.0), float(coverage_start.get("y", 0.0) or 0.0), scale=factor)
                cov1 = self.route6_test_planner_layer_pixel(metadata, float(coverage_end.get("x", 0.0) or 0.0), float(coverage_end.get("y", 0.0) or 0.0), scale=factor)
                draw.line(cov0 + cov1, fill=(40, 210, 240), width=max(1, factor * 2))
            anchor = record.get("anchor_point_cm", {}) if isinstance(record.get("anchor_point_cm", {}), dict) else (point.get("anchor_point_cm", {}) if isinstance(point.get("anchor_point_cm", {}), dict) else {})
            if anchor:
                target = self.route6_test_planner_layer_pixel(metadata, float(anchor.get("x", 0.0) or 0.0), float(anchor.get("y", 0.0) or 0.0), scale=factor)
                draw.line(obs + target, fill=(140, 70, 220), width=max(1, factor))
                draw_anchor_marker(target, f"A{point.get('scan_index', '')}")
                draw_blocked_samples(record)
            draw_warning_zone(obs, record)
            radius = max(3, factor * 2)
            draw.rectangle((obs[0] - radius, obs[1] - radius, obs[0] + radius, obs[1] + radius), fill=(170, 80, 230), outline=(0, 0, 0), width=max(1, factor))
            draw.text((obs[0] + radius + 2, obs[1] + radius), f"S{point.get('scan_index', '')}", fill=(0, 0, 0))
        return source

    def route6_test_planner_uav_pose_text(self) -> str:
        pose = self.route6_refresh_realtime_uav_pose()
        text = self.route6_format_uav_pose_text(pose)
        self.route6_test_planner_pose_var.set(text)
        return text

    def route6_test_planner_schedule_pose_refresh(self) -> None:
        self.ensure_route6_state()
        window = getattr(self, "route6_test_planner_window", None)
        if window is None:
            return
        try:
            if not window.winfo_exists():
                return
            self.route6_test_planner_uav_pose_text()
            self.refresh_route6_test_planner_preview()
            self.route6_test_planner_pose_after_id = window.after(1000, self.route6_test_planner_schedule_pose_refresh)
        except tk.TclError:
            self.route6_test_planner_pose_after_id = None

    def route6_test_planner_los_monitor_trigger_key(self, record: Dict[str, Any]) -> str:
        safety = record.get("safety_report", {}) if isinstance(record.get("safety_report", {}), dict) else {}
        details = safety.get("details", {}) if isinstance(safety.get("details", {}), dict) else {}
        front = details.get("front_blocked", {}) if isinstance(details.get("front_blocked", {}), dict) else {}
        samples = front.get("blocked_samples", []) if isinstance(front.get("blocked_samples", []), list) else []
        sample_text = ";".join(
            f"{float(item.get('x', 0.0) or 0.0):.1f},{float(item.get('y', 0.0) or 0.0):.1f}"
            for item in samples[:3]
            if isinstance(item, dict)
        )
        warning = details.get("warning_zone_50cm", {}) if isinstance(details.get("warning_zone_50cm", {}), dict) else {}
        warning_index = warning.get("grid_index", {}) if isinstance(warning.get("grid_index", {}), dict) else {}
        warning_text = ""
        if bool(warning.get("problem", False)):
            warning_text = (
                f"warning50:{int(warning_index.get('row', -1) or -1)},"
                f"{int(warning_index.get('col', -1) or -1)},"
                f"{int(warning.get('occupied_cell_count', 0) or 0)}"
            )
        return (
            f"{record.get('house_id', '')}|{record.get('edge', '')}|"
            f"{int(record.get('scan_index', 0) or 0)}|{sample_text}|{warning_text}"
        )

    def route6_test_planner_los_monitor_report(
        self,
        output_dir: Optional[Path] = None,
        *,
        selected_layer_key: str = "",
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        plan = self.llm_route6_state.get("route6_offline_test_plan", {}) if isinstance(self.llm_route6_state, dict) else {}
        if not isinstance(plan, dict) or not plan:
            return {
                "schema": "route6_los_monitor_report_v1",
                "status": "missing_plan",
                "blocked_records": [],
                "warning_records": [],
                "message": "Analyze Search Task first.",
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is None and str(plan.get("output_dir", "") or ""):
            out_path = Path(str(plan.get("output_dir", "") or ""))
        layer_key = str(selected_layer_key or plan.get("selected_layer_key", "") or self.route6_update_map_layer_var.get() or "")
        records = self.route6_test_planner_visual_calculation_records(plan, output_dir=out_path, selected_layer_key=layer_key)
        active_records = [
            record
            for record in records
            if isinstance(record, dict)
            and str(record.get("active_role", "navigation") or "navigation") == "navigation"
            and bool(record.get("display_on_map", True))
        ]
        blocked_records: List[Dict[str, Any]] = []
        warning_records: List[Dict[str, Any]] = []
        for record in active_records:
            safety = record.get("safety_report", {}) if isinstance(record.get("safety_report", {}), dict) else {}
            details = safety.get("details", {}) if isinstance(safety.get("details", {}), dict) else {}
            front = details.get("front_blocked", {}) if isinstance(details.get("front_blocked", {}), dict) else {}
            if bool(front.get("problem", False)):
                blocked_records.append(record)
            warning = details.get("warning_zone_50cm", {}) if isinstance(details.get("warning_zone_50cm", {}), dict) else {}
            if bool(warning.get("problem", False)):
                warning_records.append(record)
        status = "blocked" if blocked_records else ("warning" if warning_records else "clear")
        return {
            "schema": "route6_los_monitor_report_v1",
            "status": status,
            "selected_layer_key": layer_key,
            "output_dir": str(out_path) if out_path is not None else "",
            "active_record_count": len(active_records),
            "blocked_record_count": len(blocked_records),
            "warning_record_count": len(warning_records),
            "blocked_records": self.route6_json_safe(blocked_records),
            "warning_records": self.route6_json_safe(warning_records),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }

    def route6_test_planner_check_los_monitor(
        self,
        output_dir: Optional[Path] = None,
        *,
        force_reset: bool = False,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        report = self.route6_test_planner_los_monitor_report(output_dir=output_dir)
        report_status = str(report.get("status", "") or "")
        if report_status not in {"blocked", "warning"}:
            self.llm_route6_state["route6_test_planner_los_monitor"] = self.route6_json_safe(report)
            if report.get("status") == "clear":
                self.route6_test_planner_los_monitor_last_trigger_key = ""
                self.route6_test_planner_los_monitor_var.set(
                    f"LOS Monitor: clear active_rays={int(report.get('active_record_count', 0) or 0)}"
                )
            else:
                self.route6_test_planner_los_monitor_var.set("LOS Monitor: missing plan; run Analyze Search Task first.")
            self.route6_write_state_artifact()
            self.refresh_route6_test_planner_point_status()
            return report

        trigger_reason = "front_blocked" if report_status == "blocked" else "near_warning_50cm"
        trigger_records_key = "blocked_records" if report_status == "blocked" else "warning_records"
        trigger_records = report.get(trigger_records_key, []) if isinstance(report.get(trigger_records_key, []), list) else []
        first_trigger = trigger_records[0] if trigger_records and isinstance(trigger_records[0], dict) else {}
        trigger_key = self.route6_test_planner_los_monitor_trigger_key(first_trigger)
        report["trigger_key"] = trigger_key
        report["trigger_reason"] = trigger_reason
        if trigger_key == str(getattr(self, "route6_test_planner_los_monitor_last_trigger_key", "") or "") and not force_reset:
            report["status"] = "blocked_repeat" if report_status == "blocked" else "warning_repeat"
            self.llm_route6_state["route6_test_planner_los_monitor"] = self.route6_json_safe(report)
            self.route6_test_planner_los_monitor_var.set(
                f"LOS Monitor: {trigger_reason} repeat S{int(first_trigger.get('scan_index', 0) or 0)}; waiting for geometry change."
            )
            self.route6_write_state_artifact()
            return report

        self.route6_test_planner_los_monitor_last_trigger_key = trigger_key
        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is None and str(report.get("output_dir", "") or ""):
            out_path = Path(str(report.get("output_dir", "") or ""))
        reset_result = self.route6_reset_current_observation_point(output_dir=out_path)
        report["status"] = "reset_triggered"
        report["reset_result"] = self.route6_json_safe(reset_result)
        self.llm_route6_state["route6_test_planner_los_monitor"] = self.route6_json_safe(report)
        self.route6_test_planner_los_monitor_var.set(
            f"LOS Monitor: {trigger_reason} S{int(first_trigger.get('scan_index', 0) or 0)} -> reset {reset_result.get('reset_status', 'n/a')}"
        )
        self.refresh_route6_test_planner_result_text(report)
        self.refresh_route6_test_planner_point_status()
        self.refresh_route6_test_planner_preview()
        self.route6_write_state_artifact()
        return report

    def route6_test_planner_los_monitor_tick(self) -> None:
        self.ensure_route6_state()
        if not bool(getattr(self, "route6_test_planner_los_monitor_active", False)):
            return
        try:
            self.route6_test_planner_check_los_monitor()
        except Exception as exc:
            self.route6_test_planner_los_monitor_var.set(f"LOS Monitor: error {exc}")
        self.route6_test_planner_schedule_los_monitor()

    def route6_test_planner_schedule_los_monitor(self) -> None:
        self.ensure_route6_state()
        window = getattr(self, "route6_test_planner_window", None)
        if window is None or not bool(getattr(self, "route6_test_planner_los_monitor_active", False)):
            return
        try:
            if not window.winfo_exists():
                return
            self.route6_test_planner_los_monitor_after_id = window.after(1000, self.route6_test_planner_los_monitor_tick)
        except tk.TclError:
            self.route6_test_planner_los_monitor_after_id = None

    def on_route6_test_planner_start_los_monitor(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        self.route6_test_planner_los_monitor_active = True
        self.route6_test_planner_los_monitor_last_trigger_key = ""
        report = self.route6_test_planner_check_los_monitor()
        self.route6_test_planner_schedule_los_monitor()
        return report

    def on_route6_test_planner_stop_los_monitor(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        self.route6_test_planner_los_monitor_active = False
        after_id = getattr(self, "route6_test_planner_los_monitor_after_id", None)
        window = getattr(self, "route6_test_planner_window", None)
        if after_id is not None and window is not None:
            try:
                window.after_cancel(after_id)
            except Exception:
                pass
        self.route6_test_planner_los_monitor_after_id = None
        self.route6_test_planner_los_monitor_var.set("LOS Monitor: stopped")
        result = {"schema": "route6_los_monitor_stop_v1", "status": "stopped"}
        self.llm_route6_state["route6_test_planner_los_monitor"] = self.route6_json_safe(result)
        self.route6_write_state_artifact()
        return result

    def _on_route6_test_planner_mousewheel(self, event: tk.Event):
        canvas = getattr(self, "route6_test_planner_scroll_canvas", None)
        if canvas is None:
            return
        try:
            delta = int(-1 * (event.delta / 120))
            if delta:
                canvas.yview_scroll(delta, "units")
        except tk.TclError:
            pass

    def _on_route6_test_planner_mousewheel_linux(self, event: tk.Event):
        canvas = getattr(self, "route6_test_planner_scroll_canvas", None)
        if canvas is None:
            return
        try:
            if getattr(event, "num", 0) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", 0) == 5:
                canvas.yview_scroll(3, "units")
        except tk.TclError:
            pass

    def _bind_route6_test_planner_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_route6_test_planner_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_route6_test_planner_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_route6_test_planner_mousewheel_linux, add="+")
        except tk.TclError:
            return
        for child in widget.winfo_children():
            self._bind_route6_test_planner_mousewheel_tree(child)

    def refresh_route6_test_planner_house_list(self) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        records = self.route6_test_planner_house_records()
        self.route6_test_planner_house_record_cache = records
        listbox = getattr(self, "route6_test_planner_house_listbox", None)
        if listbox is not None:
            try:
                listbox.delete(0, tk.END)
                for record in records:
                    bbox = self.route6_house_record_bbox(record)
                    house_id = str(record.get("house_id", record.get("id", "")) or "")
                    name = str(record.get("name", f"House_{house_id}") or f"House_{house_id}")
                    listbox.insert(
                        tk.END,
                        f"{house_id}  {name}  x=[{bbox.get('min_x', 0.0):.0f},{bbox.get('max_x', 0.0):.0f}] "
                        f"y=[{bbox.get('min_y', 0.0):.0f},{bbox.get('max_y', 0.0):.0f}]",
                    )
            except tk.TclError:
                pass
        return records

    def refresh_route6_test_planner_result_text(self, payload: Optional[Dict[str, Any]] = None) -> None:
        text_widget = getattr(self, "route6_test_planner_result_text", None)
        if text_widget is None:
            return
        result = payload if isinstance(payload, dict) else (self.llm_route6_state.get("route6_offline_test_plan", {}) if isinstance(self.llm_route6_state, dict) else {})
        try:
            text_widget.configure(state="normal")
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, json.dumps(self.route6_json_safe(result), indent=2, ensure_ascii=False))
            text_widget.configure(state="disabled")
        except tk.TclError:
            pass

    def route6_test_planner_point_status_lines(self, payload: Optional[Dict[str, Any]] = None) -> List[str]:
        self.ensure_route6_state()
        plan = payload if isinstance(payload, dict) and str(payload.get("schema", "")).startswith("route6_offline_test_plan") else {}
        if not plan:
            plan = self.llm_route6_state.get("route6_offline_test_plan", {}) if isinstance(self.llm_route6_state, dict) else {}
        if not isinstance(plan, dict) or not plan:
            return ["Observation Points: run Analyze Search Task first."]
        reset = plan.get("observation_point_reset", {}) if isinstance(plan.get("observation_point_reset", {}), dict) else {}
        reset_by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        for item in reset.get("scan_point_resets", []) if isinstance(reset.get("scan_point_resets", []), list) else []:
            if not isinstance(item, dict):
                continue
            old = item.get("old_point", {}) if isinstance(item.get("old_point", {}), dict) else {}
            key = (str(old.get("house_id", "") or ""), str(old.get("edge", "") or ""), int(old.get("scan_index", 0) or 0))
            reset_by_key[key] = item
        if "old_point" in reset and isinstance(reset.get("old_point", {}), dict):
            old = reset["old_point"]
            key = (str(old.get("house_id", "") or ""), str(old.get("edge", "") or ""), int(old.get("scan_index", 0) or 0))
            reset_by_key[key] = reset

        records = plan.get("visual_calculation_records", []) if isinstance(plan.get("visual_calculation_records", []), list) else []
        warning_by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or str(record.get("active_role", "navigation") or "navigation") != "navigation":
                continue
            safety = record.get("safety_report", {}) if isinstance(record.get("safety_report", {}), dict) else {}
            details = safety.get("details", {}) if isinstance(safety.get("details", {}), dict) else {}
            warning = details.get("warning_zone_50cm", {}) if isinstance(details.get("warning_zone_50cm", {}), dict) else {}
            key = (str(record.get("house_id", "") or ""), str(record.get("edge", "") or ""), int(record.get("scan_index", 0) or 0))
            warning_by_key[key] = warning

        points = plan.get("selected_scan_observation_points", []) if isinstance(plan.get("selected_scan_observation_points", []), list) else []
        if not points:
            selected = plan.get("selected_observation_point", {}) if isinstance(plan.get("selected_observation_point", {}), dict) else {}
            points = [selected] if selected else []
        lines: List[str] = []
        for index, point in enumerate(points, start=1):
            if not isinstance(point, dict):
                continue
            scan_index = int(point.get("scan_index", index if len(points) > 1 else 0) or 0)
            label = f"S{scan_index}" if scan_index else f"H{point.get('house_id', '')} {point.get('edge', '')}"
            key = (str(point.get("house_id", "") or ""), str(point.get("edge", "") or ""), int(scan_index))
            reset_item = reset_by_key.get(key, {})
            status = str(reset_item.get("reset_status", "not_reset") or "not_reset") if reset_item else "not_reset"
            warning = warning_by_key.get(key, {})
            warning_text = "warn50=blocked" if bool(warning.get("problem", False)) else "warn50=clear"
            xy = f"({float(point.get('x', 0.0) or 0.0):.1f},{float(point.get('y', 0.0) or 0.0):.1f})"
            detail = ""
            if reset_item and isinstance(reset_item.get("old_point", {}), dict) and isinstance(reset_item.get("new_point", {}), dict):
                old = reset_item["old_point"]
                new = reset_item["new_point"]
                if abs(float(old.get("x", 0.0) or 0.0) - float(new.get("x", 0.0) or 0.0)) > 1e-6 or abs(
                    float(old.get("y", 0.0) or 0.0) - float(new.get("y", 0.0) or 0.0)
                ) > 1e-6:
                    detail = (
                        f" reset {float(old.get('x', 0.0) or 0.0):.1f},{float(old.get('y', 0.0) or 0.0):.1f}"
                        f" -> {float(new.get('x', 0.0) or 0.0):.1f},{float(new.get('y', 0.0) or 0.0):.1f}"
                    )
            lines.append(f"{label}: {status} {warning_text} point={xy}{detail}")
        return lines or ["Observation Points: no active points."]

    def refresh_route6_test_planner_point_status(self, payload: Optional[Dict[str, Any]] = None) -> List[str]:
        lines = self.route6_test_planner_point_status_lines(payload)
        text = "\n".join(lines)
        self.route6_test_planner_point_status_var.set(text)
        return lines

    def route6_test_planner_formula_payload(self) -> Dict[str, Any]:
        return {
            "schema": "route6_offline_test_planner_formula_v1",
            "coordinate_frame": "Unreal/map cm: x,y,z are centimeters; yaw_deg uses atan2(delta_y, delta_x).",
            "variables": {
                "U": "current UAV point (u_x, u_y)",
                "A": "exploration edge start point (x1, y1)",
                "B": "exploration edge end point (x2, y2)",
                "v": "edge vector B - A",
                "d": "radar distance / standoff in cm",
                "h": "scan altitude in cm",
                "C": "edge center (A + B) / 2",
            },
            "formulas": [
                "v = B - A",
                "t = clamp(((U - A) dot v) / (v dot v), 0, 1)",
                "P_edge = A + t * v",
                "west:  P_obs = (bbox.min_x - d, P_edge.y, h)",
                "east:  P_obs = (bbox.max_x + d, P_edge.y, h)",
                "south: P_obs = (P_edge.x, bbox.min_y - d, h)",
                "north: P_obs = (P_edge.x, bbox.max_y + d, h)",
                "yaw_deg = atan2(C.y - P_obs.y, C.x - P_obs.x) * 180 / pi",
                "distance_to_edge_cm = sqrt((P_edge.x - U.x)^2 + (P_edge.y - U.y)^2)",
                "selected_edge = argmin(distance_to_edge_cm) for each house",
                "operator edge override: if edge in {south,east,north,west}, selected_edge = requested edge",
                "nearest edge: score = 1 / (distance_to_edge_cm + 1), anchor = nearest projected edge point",
                "frontier-based: score = frontier_edge_length_cm / (distance_to_frontier_cm + 1), anchor = nearest frontier point",
                "nbv information gain: score = expected_information_gain_cm / (travel_cost_to_viewpoint_cm + 1), anchor = edge center",
                "surface edge explorer: score = surface_edge_length_cm / (distance_to_edge_center_cm + 1), anchor = edge center",
                "uav inspection contour: score = 1 / (travel_cost_to_dilated_contour_viewpoint_cm + 1) + contour_order_bonus, anchor = dilated contour midpoint",
                "selected_observation = first observation after sorting by algorithm_score desc, distance_to_edge_cm asc, then house_id",
                "scan coverage width = 2 * radar_distance_cm * tan(horizontal_fov_deg / 2)",
                "effective_scan_step = scan_coverage_width * (1 - overlap_ratio)",
                "scan_point_count = ceil(max(edge_length_cm - scan_coverage_width, 0) / effective_scan_step) + 1",
                "multi-point anchors use centered bins: edge_progress_i = edge_length_cm * (i + 0.5) / scan_point_count",
                "visual ray target = A_i, not edge center: A_0 is the selected algorithm anchor, A_i is each scan anchor",
                "visual coverage segment_i = [edge_progress_i - coverage_width / 2, edge_progress_i + coverage_width / 2] clipped to the edge",
                "scan_coverage_ratio = covered_edge_length_cm / edge_length_cm",
                "scan_satisfied = scan_coverage_ratio >= coverage_threshold",
                "reset safety checks report inside_house_bbox OR near_obstacle OR front_blocked OR map_boundary",
                "front_blocked ray test ignores samples within 180 cm of A_i so target-edge/facade cells do not reset S_i",
                "LOS monitor periodically recomputes front_blocked for active S_i/A_i rays and calls reset when a new blocked ray appears",
                "warning zone: draw a 50 cm radius circle around each active observation point; if occupied cells enter the circle, micro-adjust x/y",
                "micro-adjust candidates = tangent offsets and outward-normal offsets in 50 cm steps; commit the first point whose warning circle and ray are clear",
                "reset movement trigger = front_blocked only; other problems are reported without changing P_obs",
                "if front_blocked: reset_distance_candidates = base_distance_cm - k * reset_step_cm, then outward fallback",
                "reset commit policy = commit P_reset only when reset_status == ok; failed candidates are reported but not written back to the plan",
                "multi-point reset target = S_i scan_observation_points; selected_observation is reference-only for edge sorting",
                "P_reset = edge outward-normal offset from the same anchor using each reset_distance candidate",
            ],
            "code_sources": [
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_test_planner_nearest_point_on_segment",
                    "purpose": "projects the UAV point onto an exploration edge segment to get P_edge",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_observation_point_for_edge",
                    "purpose": "offsets P_edge outward by radar distance d to get P_obs",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_exploration_edges_for_bbox",
                    "purpose": "enumerates south/east/north/west edges, computes yaw and edge distance",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_apply_test_planner_algorithm_to_edge",
                    "purpose": "turns each edge into a paper-inspired navigation-point candidate and algorithm score",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_test_planner_algorithm_score_formula",
                    "purpose": "documents the scoring formula for each selectable algorithm",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_build_scan_coverage_plan_for_edge",
                    "purpose": "samples multiple observation/navigation points until the selected edge meets scan coverage",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_scan_coverage_parameters",
                    "purpose": "computes FOV-derived coverage width and effective scan step",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_test_planner_reset_distance_candidates",
                    "purpose": "generates inward/toward-edge reset distances only when the observation ray is blocked",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_test_planner_front_blocked_report",
                    "purpose": "samples the scan ray and ignores target-edge-adjacent facade cells before reporting front_blocked",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_test_planner_observation_safety_report",
                    "purpose": "reports inside-house, near-obstacle, front-blocked, and map-boundary reset problems",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_test_planner_check_los_monitor",
                    "purpose": "periodically checks active S/A line-of-sight rays and triggers observation-point reset on front_blocked",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_test_planner_micro_adjust_single_point",
                    "purpose": "moves an observation point by small 50 cm tangent/normal offsets when the point warning circle contains obstacles",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_build_offline_test_plan",
                    "purpose": "sorts houses/edges, writes route6_offline_test_plan.json",
                },
                {
                    "file": "control/route6_test_planner_analysis/engine.py",
                    "symbol": "reset_summary",
                    "purpose": "summarizes main/scan reset outcomes and keeps multi-point base observations reference-only",
                },
                {
                    "file": "control/route6_test_planner_control.py",
                    "symbol": "route6_draw_offline_test_plan_overlay",
                    "purpose": "draws selected house edges, true anchor rays, scan coverage segments, and blocked ray samples on the map preview",
                },
                {
                    "file": "control/route6_test_planner_analysis/engine.py",
                    "symbol": "visual_calculation_records",
                    "purpose": "exports the visual calculation process: P_obs, A_i anchors, coverage intervals, formulas, and line-of-sight reports",
                },
                {
                    "file": "tools/verify_route6_test_planner.py",
                    "symbol": "test_route6_offline_planner_selects_nearest_edge_and_writes_artifact",
                    "purpose": "verifies the west-edge example and radar-distance offset",
                },
            ],
        }

    def open_route6_test_planner_formula_window(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self.ensure_route6_state()
        formula_payload = payload if isinstance(payload, dict) else self.route6_test_planner_formula_payload()
        window = tk.Toplevel(self.route6_test_planner_window if getattr(self, "route6_test_planner_window", None) is not None else self.root)
        window.title("Route 6 Formula / Code Sources")
        window.geometry("900x620")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)

        text = tk.Text(window, wrap="none", font=("Consolas", 10))
        y_scroll = tk.Scrollbar(window, orient="vertical", command=text.yview)
        x_scroll = tk.Scrollbar(window, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        text.insert(tk.END, json.dumps(self.route6_json_safe(formula_payload), indent=2, ensure_ascii=False))
        text.configure(state="disabled")

    def on_route6_test_planner_show_formula(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        payload = self.route6_test_planner_formula_payload()
        self.llm_route6_state["route6_offline_test_plan_formula"] = self.route6_json_safe(payload)
        self.refresh_route6_test_planner_result_text(payload)
        self.open_route6_test_planner_formula_window(payload)
        self.route6_test_planner_status_var.set("Route 6 Test Planner: formula window opened.")
        return payload

    def on_route6_test_planner_load_latest_map(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        manifest = self.route6_update_map_load_manifest(build_if_missing=False)
        if not manifest:
            self.route6_test_planner_status_var.set("Route 6 Test Planner: no Route 6 Update Map manifest to load.")
            self.refresh_route6_test_planner_preview()
            return {}
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        values = [self._route6_update_map_layer_key(layer) for layer in layers if isinstance(layer, dict)]
        combo = getattr(self, "route6_test_planner_layer_combo", None)
        if combo is not None:
            try:
                combo.configure(values=values)
            except tk.TclError:
                pass
        selected = self.route6_choose_realtime_layer_key(layers, str(self.route6_update_map_layer_var.get() or ""))
        if selected:
            self.route6_update_map_layer_var.set(selected)
        self.refresh_route6_test_planner_house_list()
        self.refresh_route6_test_planner_preview()
        self.route6_test_planner_status_var.set(f"Route 6 Test Planner: loaded latest map layer={selected or 'n/a'}.")
        return manifest

    def on_route6_test_planner_start_realtime_map(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        self.on_route6_update_map_start_realtime()
        self.route6_test_planner_uav_pose_text()
        self.refresh_route6_test_planner_preview()
        realtime_state = self.llm_route6_state.get("route6_update_map_realtime", {}) if isinstance(self.llm_route6_state, dict) else {}
        return self.route6_json_safe(realtime_state if isinstance(realtime_state, dict) else {})

    def on_route6_test_planner_stop_map_and_lock_movement(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        result = self.route6_stop_map_capture_and_lock_movement()
        self.route6_test_planner_uav_pose_text()
        self.refresh_route6_test_planner_preview()
        return result

    def on_route6_test_planner_analyze(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        selected = self.route6_selected_test_planner_house_ids()
        plan = self.route6_build_offline_test_plan(selected_house_ids=selected)
        self.refresh_route6_test_planner_result_text(plan)
        self.refresh_route6_test_planner_preview()
        return plan

    def on_route6_test_planner_reset_current_observation_point(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        result = self.route6_reset_current_observation_point()
        payload = self.llm_route6_state.get("route6_offline_test_plan", result) if isinstance(self.llm_route6_state, dict) else result
        self.refresh_route6_test_planner_result_text(payload if isinstance(payload, dict) else result)
        self.refresh_route6_test_planner_preview()
        return result

    def refresh_route6_test_planner_preview(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        manifest = self.route6_update_map_load_manifest(build_if_missing=False)
        preview = getattr(self, "route6_test_planner_preview_label", None)
        if not manifest:
            if preview is not None:
                try:
                    preview.configure(text="No Route 6 Update Map loaded.", image="")
                except tk.TclError:
                    pass
            return {}
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        values = [self._route6_update_map_layer_key(layer) for layer in layers if isinstance(layer, dict)]
        selected = str(self.route6_update_map_layer_var.get() or "")
        if selected not in values and values:
            selected = values[0]
            self.route6_update_map_layer_var.set(selected)
        layer_record = next((layer for layer in layers if isinstance(layer, dict) and self._route6_update_map_layer_key(layer) == selected), {})
        if preview is None:
            return manifest
        preview_path = self.route6_update_map_layer_preview_path(layer_record)
        if not preview_path.is_file():
            try:
                preview.configure(text=f"Preview missing: {preview_path}", image="")
            except tk.TclError:
                pass
            return manifest
        try:
            image = Image.open(preview_path).convert("RGB")
            width, height = image.size
            frame = getattr(self, "route6_test_planner_preview_frame", None)
            try:
                available_w = int(frame.winfo_width() or 820) if frame is not None else 820
            except Exception:
                available_w = 820
            scale = max(1, min(8, int((available_w - 40) / max(1, max(width, height)))))
            if scale > 1:
                image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
            image = self.route6_draw_update_map_uav_overlay(image, layer_record, scale=scale)
            plan = self.llm_route6_state.get("route6_offline_test_plan", {}) if isinstance(self.llm_route6_state, dict) else {}
            if isinstance(plan, dict) and plan:
                image = self.route6_draw_offline_test_plan_overlay(image, layer_record, plan, scale=scale)
            photo = ImageTk.PhotoImage(image)
            self.route6_test_planner_preview_photo = photo
            preview.configure(image=photo, text="")
        except Exception as exc:
            try:
                preview.configure(text=f"Route 6 Test Planner preview failed: {exc}", image="")
            except tk.TclError:
                pass
        return manifest

    def open_route6_test_planner_window(self) -> None:
        self.ensure_route6_state()
        if self.route6_test_planner_window is not None and self.route6_test_planner_window.winfo_exists():
            self.route6_test_planner_window.lift()
            self.route6_test_planner_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Route 6 Offline Test Planner")
        window.geometry("1080x760")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close_route6_test_planner_window)

        route6_test_planner_scroll_canvas = tk.Canvas(window, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(window, orient="vertical", command=route6_test_planner_scroll_canvas.yview)
        h_scrollbar = tk.Scrollbar(window, orient="horizontal", command=route6_test_planner_scroll_canvas.xview)
        route6_test_planner_scroll_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        route6_test_planner_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        content = tk.Frame(route6_test_planner_scroll_canvas)
        content_window = route6_test_planner_scroll_canvas.create_window((0, 0), window=content, anchor="nw")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(3, weight=1)
        content.grid_rowconfigure(4, weight=1)

        def _update_scroll_region(_event: Optional[tk.Event] = None) -> None:
            try:
                route6_test_planner_scroll_canvas.configure(scrollregion=route6_test_planner_scroll_canvas.bbox("all"))
            except tk.TclError:
                pass

        def _sync_content_width(event: tk.Event) -> None:
            try:
                route6_test_planner_scroll_canvas.itemconfigure(content_window, width=max(1040, int(event.width)))
                route6_test_planner_scroll_canvas.configure(scrollregion=route6_test_planner_scroll_canvas.bbox("all"))
            except tk.TclError:
                pass

        content.bind("<Configure>", _update_scroll_region)
        route6_test_planner_scroll_canvas.bind("<Configure>", _sync_content_width)

        toolbar = tk.LabelFrame(content, text="Offline Route 6 Test Planner")
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        toolbar.grid_columnconfigure(15, weight=1)
        tk.Label(toolbar, text="Layer").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        layer_combo = ttk.Combobox(
            toolbar,
            textvariable=self.route6_update_map_layer_var,
            values=[f"z_{int(value):03d}" for value in route6_map_builder.DEFAULT_ROUTE6_LAYER_Z_CM],
            state="readonly",
            width=10,
        )
        layer_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        layer_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_route6_test_planner_preview())
        tk.Button(toolbar, text="Load Latest Map", command=self.on_route6_test_planner_load_latest_map).grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Route 6 Update Map", command=self.open_route6_update_map_window).grid(row=0, column=3, sticky="w", padx=6, pady=6)
        tk.Label(toolbar, text="Edge").grid(row=0, column=4, sticky="e", padx=(18, 2), pady=6)
        edge_combo = ttk.Combobox(
            toolbar,
            textvariable=self.route6_test_planner_edge_var,
            values=["auto nearest", "south", "east", "north", "west"],
            state="readonly",
            width=12,
        )
        edge_combo.grid(row=0, column=5, sticky="w", padx=(0, 6), pady=6)
        tk.Label(toolbar, text="Radar cm").grid(row=0, column=6, sticky="e", padx=(12, 2), pady=6)
        tk.Entry(toolbar, textvariable=self.route6_test_planner_radar_distance_cm_var, width=8).grid(row=0, column=7, sticky="w", padx=(0, 6), pady=6)
        tk.Label(toolbar, text="Scan z cm").grid(row=0, column=8, sticky="e", padx=(12, 2), pady=6)
        tk.Entry(toolbar, textvariable=self.route6_test_planner_scan_z_cm_var, width=8).grid(row=0, column=9, sticky="w", padx=(0, 6), pady=6)
        tk.Label(toolbar, text="Algorithm").grid(row=1, column=0, sticky="e", padx=6, pady=(0, 6))
        algorithm_combo = ttk.Combobox(
            toolbar,
            textvariable=self.route6_test_planner_algorithm_var,
            values=self.route6_test_planner_algorithm_options(),
            state="readonly",
            width=24,
        )
        algorithm_combo.grid(row=1, column=1, columnspan=3, sticky="w", padx=6, pady=(0, 6))
        tk.Label(toolbar, textvariable=self.route6_test_planner_status_var, anchor="w").grid(row=1, column=4, columnspan=12, sticky="ew", padx=6, pady=(0, 6))
        tk.Label(toolbar, text="Scan Mode").grid(row=2, column=0, sticky="e", padx=6, pady=(0, 6))
        scan_mode_combo = ttk.Combobox(
            toolbar,
            textvariable=self.route6_test_planner_scan_mode_var,
            values=self.route6_test_planner_scan_mode_options(),
            state="readonly",
            width=24,
        )
        scan_mode_combo.grid(row=2, column=1, columnspan=3, sticky="w", padx=6, pady=(0, 6))
        tk.Label(toolbar, text="FOV deg").grid(row=2, column=4, sticky="e", padx=(12, 2), pady=(0, 6))
        tk.Entry(toolbar, textvariable=self.route6_test_planner_fov_deg_var, width=7).grid(row=2, column=5, sticky="w", padx=(0, 6), pady=(0, 6))
        tk.Label(toolbar, text="Overlap").grid(row=2, column=6, sticky="e", padx=(12, 2), pady=(0, 6))
        tk.Entry(toolbar, textvariable=self.route6_test_planner_overlap_var, width=7).grid(row=2, column=7, sticky="w", padx=(0, 6), pady=(0, 6))
        tk.Label(toolbar, text="Coverage").grid(row=2, column=8, sticky="e", padx=(12, 2), pady=(0, 6))
        tk.Entry(toolbar, textvariable=self.route6_test_planner_coverage_threshold_var, width=7).grid(row=2, column=9, sticky="w", padx=(0, 6), pady=(0, 6))
        tk.Button(toolbar, text="Analyze Search Task", command=self.on_route6_test_planner_analyze).grid(row=3, column=0, sticky="w", padx=6, pady=(2, 6))
        tk.Button(toolbar, text="Show Formula", command=self.on_route6_test_planner_show_formula).grid(row=3, column=1, sticky="w", padx=6, pady=(2, 6))
        tk.Button(toolbar, text="Reset Current Observation Point", command=self.on_route6_test_planner_reset_current_observation_point).grid(row=3, column=2, sticky="w", padx=6, pady=(2, 6))
        tk.Button(toolbar, text="Start Realtime Update", command=self.on_route6_test_planner_start_realtime_map).grid(row=3, column=3, sticky="w", padx=6, pady=(2, 6))
        tk.Button(toolbar, text="Stop Map + Lock Movement", command=self.on_route6_test_planner_stop_map_and_lock_movement).grid(row=3, column=4, sticky="w", padx=6, pady=(2, 6))
        tk.Button(toolbar, text="Start LOS Monitor", command=self.on_route6_test_planner_start_los_monitor).grid(row=4, column=0, sticky="w", padx=6, pady=(0, 6))
        tk.Button(toolbar, text="Stop LOS Monitor", command=self.on_route6_test_planner_stop_los_monitor).grid(row=4, column=1, sticky="w", padx=6, pady=(0, 6))
        tk.Label(toolbar, textvariable=self.route6_test_planner_los_monitor_var, anchor="w").grid(row=4, column=2, columnspan=14, sticky="ew", padx=6, pady=(0, 6))
        tk.Label(toolbar, textvariable=self.route6_test_planner_pose_var, anchor="w").grid(row=5, column=0, columnspan=16, sticky="ew", padx=6, pady=(0, 6))

        house_frame = tk.LabelFrame(content, text="Houses")
        house_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        house_frame.grid_columnconfigure(0, weight=1)
        house_list = tk.Listbox(house_frame, selectmode="extended", exportselection=False, height=6)
        house_scroll = tk.Scrollbar(house_frame, orient="vertical", command=house_list.yview)
        house_list.configure(yscrollcommand=house_scroll.set)
        house_list.grid(row=0, column=0, sticky="ew", padx=(6, 0), pady=6)
        house_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)

        point_status_frame = tk.LabelFrame(content, text="Observation Point Status")
        point_status_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        point_status_frame.grid_columnconfigure(0, weight=1)
        point_status_label = tk.Label(
            point_status_frame,
            textvariable=self.route6_test_planner_point_status_var,
            anchor="w",
            justify="left",
        )
        point_status_label.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        preview_frame = tk.LabelFrame(content, text="Route 6 Map Preview")
        preview_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_label = tk.Label(preview_frame, text="Load a Route 6 Update Map to preview the offline test plan.", anchor="center", justify="center")
        preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        result_frame = tk.LabelFrame(content, text="Calculation Process / Result")
        result_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(0, weight=1)
        result_text = tk.Text(result_frame, height=10, wrap="none", font=("Consolas", 9))
        result_y = tk.Scrollbar(result_frame, orient="vertical", command=result_text.yview)
        result_x = tk.Scrollbar(result_frame, orient="horizontal", command=result_text.xview)
        result_text.configure(yscrollcommand=result_y.set, xscrollcommand=result_x.set, state="disabled")
        result_text.grid(row=0, column=0, sticky="nsew")
        result_y.grid(row=0, column=1, sticky="ns")
        result_x.grid(row=1, column=0, sticky="ew")

        self.route6_test_planner_window = window
        self.route6_test_planner_scroll_canvas = route6_test_planner_scroll_canvas
        self.route6_test_planner_content_frame = content
        self.route6_test_planner_layer_combo = layer_combo
        self.route6_test_planner_edge_combo = edge_combo
        self.route6_test_planner_algorithm_combo = algorithm_combo
        self.route6_test_planner_scan_mode_combo = scan_mode_combo
        self.route6_test_planner_house_listbox = house_list
        self.route6_test_planner_preview_frame = preview_frame
        self.route6_test_planner_point_status_label = point_status_label
        self.route6_test_planner_preview_label = preview_label
        self.route6_test_planner_result_text = result_text
        self.refresh_route6_test_planner_house_list()
        self.route6_test_planner_uav_pose_text()
        self.refresh_route6_test_planner_point_status()
        self.refresh_route6_test_planner_result_text()
        self.refresh_route6_test_planner_preview()
        self._bind_route6_test_planner_mousewheel_tree(window)
        self.route6_test_planner_schedule_pose_refresh()

    def close_route6_test_planner_window(self) -> None:
        self.ensure_route6_state()
        after_id = getattr(self, "route6_test_planner_pose_after_id", None)
        if after_id is not None and self.route6_test_planner_window is not None:
            try:
                self.route6_test_planner_window.after_cancel(after_id)
            except Exception:
                pass
        self.route6_test_planner_pose_after_id = None
        monitor_after_id = getattr(self, "route6_test_planner_los_monitor_after_id", None)
        if monitor_after_id is not None and self.route6_test_planner_window is not None:
            try:
                self.route6_test_planner_window.after_cancel(monitor_after_id)
            except Exception:
                pass
        self.route6_test_planner_los_monitor_after_id = None
        self.route6_test_planner_los_monitor_active = False
        if self.route6_test_planner_window is not None:
            try:
                self.route6_test_planner_window.destroy()
            except Exception:
                pass
        self.route6_test_planner_window = None
        self.route6_test_planner_scroll_canvas = None
        self.route6_test_planner_content_frame = None
        self.route6_test_planner_house_listbox = None
        self.route6_test_planner_result_text = None
        self.route6_test_planner_preview_label = None
        self.route6_test_planner_point_status_label = None
        self.route6_test_planner_preview_photo = None
        self.route6_test_planner_layer_combo = None
        self.route6_test_planner_algorithm_combo = None
        self.route6_test_planner_scan_mode_combo = None
        self.route6_test_planner_preview_frame = None

