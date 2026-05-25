from __future__ import annotations

from .common import *


ACTIVE_NBV_FACADES = ("south", "east", "north", "west")
ACTIVE_NBV_REWARD_WEIGHTS = {
    "w_cov": 1.0,
    "w_quality": 0.35,
    "w_safe": 0.5,
    "w_repeat": 0.6,
    "w_path": 0.2,
    "w_risk": 0.8,
}


class ActiveNBVScanControlMixin:
    def ensure_active_nbv_state(self) -> None:
        if not hasattr(self, "active_nbv_window"):
            self.active_nbv_window = None
        if not hasattr(self, "active_nbv_thread"):
            self.active_nbv_thread = None
        if not hasattr(self, "active_nbv_stop_event"):
            self.active_nbv_stop_event = threading.Event()
        if not hasattr(self, "active_nbv_pause_event"):
            self.active_nbv_pause_event = threading.Event()
        if not hasattr(self, "active_nbv_state"):
            self.active_nbv_state = {}
        if not hasattr(self, "active_nbv_output_dir"):
            self.active_nbv_output_dir = None
        if not hasattr(self, "active_nbv_status_var"):
            self.active_nbv_status_var = tk.StringVar(value="Active NBV: idle")
        if not hasattr(self, "active_nbv_stage_var"):
            self.active_nbv_stage_var = tk.StringVar(value="Stage: idle")
        if not hasattr(self, "active_nbv_target_var"):
            self.active_nbv_target_var = tk.StringVar(value="Target: n/a")
        if not hasattr(self, "active_nbv_progress_var"):
            self.active_nbv_progress_var = tk.DoubleVar(value=0.0)
        if not hasattr(self, "active_nbv_progress_text_var"):
            self.active_nbv_progress_text_var = tk.StringVar(value="Progress: 0%")
        if not hasattr(self, "active_nbv_standoff_cm_var"):
            default = getattr(self, "llm_route_standoff_cm_var", None)
            self.active_nbv_standoff_cm_var = tk.StringVar(value=str(default.get() if default is not None else "850"))
        if not hasattr(self, "active_nbv_initial_view_count_var"):
            self.active_nbv_initial_view_count_var = tk.StringVar(value="3")
        if not hasattr(self, "active_nbv_scan_z_cm_var"):
            self.active_nbv_scan_z_cm_var = tk.StringVar(value="450")
        if not hasattr(self, "active_nbv_max_rescan_rounds_var"):
            self.active_nbv_max_rescan_rounds_var = tk.StringVar(value="2")
        if not hasattr(self, "active_nbv_facade_threshold_var"):
            self.active_nbv_facade_threshold_var = tk.StringVar(value="0.75")
        if not hasattr(self, "active_nbv_mean_threshold_var"):
            self.active_nbv_mean_threshold_var = tk.StringVar(value="0.85")
        if not hasattr(self, "active_nbv_preview_text"):
            self.active_nbv_preview_text = None
        if not hasattr(self, "active_nbv_map_text"):
            self.active_nbv_map_text = None
        if not hasattr(self, "active_nbv_map_widget"):
            self.active_nbv_map_widget = None
        if not hasattr(self, "active_nbv_map_frame"):
            self.active_nbv_map_frame = None
        if not hasattr(self, "active_nbv_map_status_var"):
            self.active_nbv_map_status_var = tk.StringVar(value="Active NBV Map: idle")
        if not hasattr(self, "active_nbv_scroll_canvas"):
            self.active_nbv_scroll_canvas = None
        if not hasattr(self, "active_nbv_scroll_frame"):
            self.active_nbv_scroll_frame = None

    def active_nbv_float_param(self, var: Any, default: float, *, min_value: float, max_value: float) -> float:
        try:
            value = float(var.get())
        except Exception:
            value = float(default)
        if not math.isfinite(value):
            value = float(default)
        return max(float(min_value), min(float(max_value), float(value)))

    def active_nbv_int_param(self, var: Any, default: int, *, min_value: int, max_value: int) -> int:
        try:
            value = int(float(var.get()))
        except Exception:
            value = int(default)
        return max(int(min_value), min(int(max_value), int(value)))

    def active_nbv_route5_oa3_config(self) -> Dict[str, Any]:
        self.ensure_route5_state()
        return {
            "or2_model_path": str(self.llm_route5_representation_model_var.get() or self.default_route5_or2_model_path()),
            "oa3_plan_path": str(self.llm_route5_oa3_plan_var.get() or self.default_route5_oa3_plan_path()),
            "method": "active_nbv_or2_rule_v1",
            "fallback_policy": "active_nbv_depth_pointcloud_rule_on_or2_unavailable",
        }

    def active_nbv_output_root(self) -> Path:
        return self.resolve_project_path("active_nbv_scan_runs")

    def make_active_nbv_output_dir(self, target_house_id: str) -> Path:
        root = self.active_nbv_output_root()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_active_nbv_or2_{timestamp}"
        candidate = root / base_name
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = root / f"{base_name}_{suffix}"
        (candidate / "frames").mkdir(parents=True, exist_ok=True)
        (candidate / "facade_observations").mkdir(parents=True, exist_ok=True)
        (candidate / "reconstruction").mkdir(parents=True, exist_ok=True)
        return candidate

    def active_nbv_write_state_artifact(self) -> None:
        output_dir = getattr(self, "active_nbv_output_dir", None)
        if output_dir is None:
            return
        payload = dict(getattr(self, "active_nbv_state", {}) if isinstance(getattr(self, "active_nbv_state", {}), dict) else {})
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.write_json_artifact(Path(output_dir) / "active_nbv_scan_state.json", self.route5_json_safe(payload))

    def active_nbv_log_event(self, output_dir: Path, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "event_type": str(event_type),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            **(payload if isinstance(payload, dict) else {}),
        }
        self.append_jsonl(output_dir / "active_nbv_events.jsonl", self.route5_json_safe(record))

    def active_nbv_set_stage(self, stage: str, *, output_dir: Optional[Path] = None, message: str = "", facade: str = "") -> None:
        self.ensure_active_nbv_state()
        self.active_nbv_state["stage"] = str(stage)
        self.active_nbv_state["current_facade"] = str(facade or "")
        self.active_nbv_state["stage_message"] = str(message or "")
        self.active_nbv_stage_var.set(f"Stage: {stage}")
        if message:
            self.active_nbv_status_var.set(f"Active NBV: {message}")
        if output_dir is not None:
            self.active_nbv_log_event(output_dir, "stage", {"stage": stage, "facade": facade, "message": message})
            self.active_nbv_write_state_artifact()

    def active_nbv_facade_axis_range(self, bbox: Dict[str, Any], facade: str) -> Tuple[float, float]:
        if facade in {"south", "north"}:
            return float(bbox["min_x"]), float(bbox["max_x"])
        return float(bbox["min_y"]), float(bbox["max_y"])

    def active_nbv_axis_values(self, axis_min: float, axis_max: float, view_count: int) -> List[Tuple[str, float, float]]:
        low = float(min(axis_min, axis_max))
        high = float(max(axis_min, axis_max))
        span = max(1.0, high - low)
        templates = [
            ("face_center", 0.50, 0.0),
            ("left_oblique", 0.25, -35.0),
            ("right_oblique", 0.75, 35.0),
            ("edge_left_oblique", 0.10, -45.0),
            ("edge_right_oblique", 0.90, 45.0),
        ]
        return [(name, low + span * ratio, yaw_offset) for name, ratio, yaw_offset in templates[: max(1, min(5, int(view_count)))]]

    def active_nbv_initial_scan_points(self, house_id: str) -> List[Dict[str, Any]]:
        bbox = self.house_world_bbox_for_id(str(house_id))
        if not bbox:
            return []
        standoff = self.active_nbv_float_param(
            getattr(self, "active_nbv_standoff_cm_var", getattr(self, "llm_route_standoff_cm_var", None)),
            self.route_scan_standoff_cm(),
            min_value=150.0,
            max_value=2000.0,
        )
        z_cm = self.active_nbv_float_param(getattr(self, "active_nbv_scan_z_cm_var", None), 450.0, min_value=100.0, max_value=2000.0)
        view_count = self.active_nbv_int_param(getattr(self, "active_nbv_initial_view_count_var", None), 3, min_value=1, max_value=5)
        points: List[Dict[str, Any]] = []
        order = 0
        for facade in ACTIVE_NBV_FACADES:
            axis_min, axis_max = self.active_nbv_facade_axis_range(bbox, facade)
            for view_type, axis_value, yaw_offset in self.active_nbv_axis_values(axis_min, axis_max, view_count):
                pose = self.route2_facade_pose_from_axis(bbox, facade, axis_value, standoff, z_cm)
                yaw = float(pose.get("yaw_deg", 0.0) or 0.0) + float(yaw_offset)
                point = {
                    "scan_id": f"{house_id}_{facade}_active_nbv_initial_{order:03d}",
                    "global_scan_order": order,
                    "local_scan_index": order,
                    "house_id": str(house_id),
                    "facade": facade,
                    "facade_id": self.route2_facade_id(str(house_id), facade),
                    "height_band": "active_nbv",
                    "floor_index": 0,
                    "semantic_region": "active_nbv_sparse_facade",
                    "view_type": view_type,
                    "active_nbv_round": 0,
                    "active_nbv_source": "initial_sparse_views",
                    "x": round(float(pose["x"]), 2),
                    "y": round(float(pose["y"]), 2),
                    "z": round(float(pose["z"]), 2),
                    "yaw_deg": round(yaw, 3),
                    "standoff_cm": round(float(standoff), 2),
                    "scan_spacing_cm": round(float(self.route_scan_spacing_cm()), 2),
                    "lidar_max_range_cm": float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
                    "capture_trigger": "arrive_align_hover_capture",
                    "status": "planned",
                }
                points.append(point)
                order += 1
        return points

    def active_nbv_score_view_candidate(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        weights = ACTIVE_NBV_REWARD_WEIGHTS
        coverage_gain = float(context.get("coverage_gain", 0.0) or 0.0)
        view_quality = float(context.get("view_quality", 0.0) or 0.0)
        safety_score = float(context.get("safety_score", 0.0) or 0.0)
        redundant = float(context.get("redundant_view_penalty", 0.0) or 0.0)
        path_cost = float(context.get("path_length_cost", 0.0) or 0.0)
        obstacle_risk = float(context.get("obstacle_risk", 0.0) or 0.0)
        reward = (
            weights["w_cov"] * coverage_gain
            + weights["w_quality"] * view_quality
            + weights["w_safe"] * safety_score
            - weights["w_repeat"] * redundant
            - weights["w_path"] * path_cost
            - weights["w_risk"] * obstacle_risk
        )
        return {
            "scan_id": str(candidate.get("scan_id", "")),
            "facade": str(candidate.get("facade", "")),
            "view_type": str(candidate.get("view_type", "")),
            "coverage_gain": round(coverage_gain, 6),
            "view_quality": round(view_quality, 6),
            "safety_score": round(safety_score, 6),
            "redundant_view_penalty": round(redundant, 6),
            "path_length_cost": round(path_cost, 6),
            "obstacle_risk": round(obstacle_risk, 6),
            "reward": round(float(reward), 6),
            "weights": dict(weights),
        }

    def active_nbv_capture_point_count(self, capture: Dict[str, Any]) -> int:
        if not isinstance(capture, dict):
            return 0
        candidates = [capture.get("point_count", 0)]
        execution = capture.get("execution", {})
        if isinstance(execution, dict):
            candidates.append(execution.get("point_count", 0))
        rows = capture.get("rows", [])
        if isinstance(rows, list):
            candidates.append(sum(int(row.get("point_count", 0) or 0) for row in rows if isinstance(row, dict)))
        best = 0
        for value in candidates:
            try:
                best = max(best, int(value or 0))
            except Exception:
                continue
        return best

    def active_nbv_capture_dirs(self, capture: Dict[str, Any]) -> List[str]:
        if not isinstance(capture, dict):
            return []
        dirs: List[str] = []
        capture_dir = str(capture.get("capture_dir", "") or "")
        if capture_dir:
            dirs.append(capture_dir)
        execution = capture.get("execution", {})
        if isinstance(execution, dict):
            for item in execution.get("capture_dirs", []) if isinstance(execution.get("capture_dirs", []), list) else []:
                value = str(item or "")
                if value and value not in dirs:
                    dirs.append(value)
        for row in capture.get("rows", []) if isinstance(capture.get("rows", []), list) else []:
            if isinstance(row, dict):
                value = str(row.get("capture_dir", "") or "")
                if value and value not in dirs:
                    dirs.append(value)
        return dirs

    def active_nbv_successful_scan_rows(self, output_dir: Path) -> List[Dict[str, Any]]:
        rows = []
        for row in self.read_jsonl_artifact(output_dir / "lidar_capture_log.jsonl"):
            if not isinstance(row, dict):
                continue
            status = str(row.get("capture_status", row.get("status", "")) or "").strip().lower()
            if status not in {"ok", "captured", "done"}:
                continue
            if row.get("capture_guard_passed") is not True:
                continue
            if int(row.get("point_count", 0) or 0) <= 0:
                continue
            rows.append(row)
        return rows

    def active_nbv_build_coverage_report(
        self,
        house_id: str,
        points: List[Dict[str, Any]],
        *,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        threshold = self.active_nbv_float_param(getattr(self, "active_nbv_facade_threshold_var", None), 0.75, min_value=0.0, max_value=1.0)
        mean_threshold = self.active_nbv_float_param(getattr(self, "active_nbv_mean_threshold_var", None), 0.85, min_value=0.0, max_value=1.0)
        capture_rows = self.active_nbv_successful_scan_rows(output_dir) if output_dir is not None else []
        captured_ids = {str(row.get("scan_id", "") or "") for row in capture_rows}
        row_point_count: Dict[str, int] = {}
        for row in capture_rows:
            scan_id = str(row.get("scan_id", "") or "")
            row_point_count[scan_id] = int(row_point_count.get(scan_id, 0)) + int(row.get("point_count", 0) or 0)
        facades: Dict[str, Dict[str, Any]] = {}
        for facade in ACTIVE_NBV_FACADES:
            planned = [point for point in points if isinstance(point, dict) and str(point.get("facade", "") or "") == facade]
            if output_dir is not None:
                captured = [point for point in planned if str(point.get("scan_id", "") or "") in captured_ids]
            else:
                captured = [
                    point
                    for point in planned
                    if str(point.get("status", "") or "") in {"captured", "visited"}
                    and int(point.get("point_count", 0) or 0) > 0
                    and bool((point.get("route5_capture_guard") or {}).get("capture_guard_passed", True))
                ]
            planned_count = len(planned)
            captured_count = len(captured)
            coverage = round(float(captured_count) / float(max(1, planned_count)), 4)
            point_count = sum(int(row_point_count.get(str(point.get("scan_id", "") or ""), int(point.get("point_count", 0) or 0))) for point in captured)
            facades[facade] = {
                "facade": facade,
                "planned_scan_count": planned_count,
                "captured_scan_count": captured_count,
                "scan_completion_ratio": coverage,
                "point_cloud_coverage": coverage,
                "covered_cells": captured_count,
                "total_cells": max(1, planned_count),
                "point_count": int(point_count),
                "needs_rescan": bool(coverage < threshold),
            }
        coverage_values = [float(item["point_cloud_coverage"]) for item in facades.values()]
        mean_coverage = float(sum(coverage_values) / max(1, len(coverage_values)))
        merged_point_count = int(sum(int(item.get("point_count", 0) or 0) for item in facades.values()))
        complete = bool(
            all(float(item["point_cloud_coverage"]) >= threshold for item in facades.values())
            and mean_coverage >= mean_threshold
            and merged_point_count > 0
        )
        return {
            "schema": "active_nbv_coverage_report_v1",
            "target_house_id": str(house_id),
            "facades": facades,
            "mean_facade_coverage": round(mean_coverage, 4),
            "facade_coverage_threshold": round(float(threshold), 4),
            "mean_coverage_threshold": round(float(mean_threshold), 4),
            "merged_point_count": merged_point_count,
            "valid_scan_capture_count": len(capture_rows),
            "source_frame_count": len(capture_rows),
            "complete": complete,
            "coverage_mode": "active_nbv_guarded_scan_completion",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def active_nbv_build_rescan_plan(
        self,
        house_id: str,
        coverage_report: Dict[str, Any],
        existing_points: List[Dict[str, Any]],
        *,
        round_index: int = 1,
    ) -> Dict[str, Any]:
        threshold = self.active_nbv_float_param(getattr(self, "active_nbv_facade_threshold_var", None), 0.75, min_value=0.0, max_value=1.0)
        facades = coverage_report.get("facades", {}) if isinstance(coverage_report.get("facades"), dict) else {}
        rescan_points: List[Dict[str, Any]] = []
        for facade, report in facades.items():
            if str(facade) not in ACTIVE_NBV_FACADES or not isinstance(report, dict):
                continue
            coverage = report.get("point_cloud_coverage")
            if coverage is None:
                coverage = report.get("scan_completion_ratio", 0.0)
            if float(coverage or 0.0) >= threshold:
                continue
            candidates = [
                point for point in existing_points
                if isinstance(point, dict) and str(point.get("facade", "") or "") == str(facade)
            ]
            if not candidates:
                generated = self.active_nbv_initial_scan_points(house_id)
                candidates = [point for point in generated if str(point.get("facade", "") or "") == str(facade)]
            if not candidates:
                continue
            center = dict(candidates[len(candidates) // 2])
            edge = dict(candidates[0 if round_index % 2 else -1])
            high = dict(center)
            edge["yaw_deg"] = round(float(edge.get("yaw_deg", 0.0) or 0.0) + (35.0 if facade in {"east", "north"} else -35.0), 3)
            high["z"] = round(float(high.get("z", 450.0) or 450.0) + 120.0, 2)
            templates = [
                ("hole_center_face_view", center, 0.78, 0.10),
                ("oblique_edge_view", edge, 0.72, 0.16),
                ("high_center", high, 0.68, 0.18),
            ]
            for view_type, base_pose, view_quality, repeat_penalty in templates:
                base = dict(base_pose)
                base.update(
                    {
                        "scan_id": f"{house_id}_{facade}_active_nbv_rescan_r{round_index:02d}_{len(rescan_points):03d}",
                        "status": "planned",
                        "view_type": view_type,
                        "active_nbv_round": int(round_index),
                        "active_nbv_source": "coverage_nbv_rescan",
                        "rescan_reason": f"{facade} coverage {float(coverage or 0.0):.4f} below threshold {threshold:.4f}",
                        "coverage_before_rescan": round(float(coverage or 0.0), 4),
                    }
                )
                score = self.active_nbv_score_view_candidate(
                    base,
                    {
                        "coverage_gain": max(0.0, threshold - float(coverage or 0.0)),
                        "view_quality": view_quality,
                        "safety_score": 0.8,
                        "redundant_view_penalty": repeat_penalty,
                        "path_length_cost": 0.1,
                        "obstacle_risk": 0.05,
                    },
                )
                base["active_nbv_score"] = score
                rescan_points.append(base)
        return {
            "schema": "active_nbv_rescan_plan_v1",
            "house_id": str(house_id),
            "round_index": int(round_index),
            "rescan_reason": "coverage below threshold" if rescan_points else "",
            "rescan_points": rescan_points,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def active_nbv_write_scan_points(self, output_dir: Path, house_id: str, points: List[Dict[str, Any]]) -> None:
        payload = {
            "schema": "active_nbv_scan_points_v1",
            "target_house_id": str(house_id),
            "scan_points": self.route5_json_safe(points),
            "total_scan_count": len(points),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(output_dir / "scan_points.json", payload)

    def active_nbv_write_summary_csv(self, output_dir: Path, coverage_report: Dict[str, Any], points: List[Dict[str, Any]]) -> None:
        captured = [point for point in points if str(point.get("status", "") or "") == "captured"]
        headers = [
            "target_house_id",
            "planned_scan_count",
            "captured_scan_count",
            "merged_point_count",
            "mean_facade_coverage",
            "complete",
            "max_rescan_rounds",
        ]
        row = {
            "target_house_id": str(coverage_report.get("target_house_id", "")),
            "planned_scan_count": str(len(points)),
            "captured_scan_count": str(len(captured)),
            "merged_point_count": str(coverage_report.get("merged_point_count", 0)),
            "mean_facade_coverage": str(coverage_report.get("mean_facade_coverage", 0.0)),
            "complete": str(bool(coverage_report.get("complete", False))),
            "max_rescan_rounds": str(self.active_nbv_int_param(getattr(self, "active_nbv_max_rescan_rounds_var", None), 2, min_value=0, max_value=10)),
        }
        lines = [",".join(headers), ",".join(row.get(header, "") for header in headers)]
        (output_dir / "active_scan_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def active_nbv_current_output_dir(self) -> Optional[Path]:
        output_dir = getattr(self, "active_nbv_output_dir", None)
        if output_dir is not None:
            return Path(output_dir)
        state = getattr(self, "active_nbv_state", {}) if isinstance(getattr(self, "active_nbv_state", {}), dict) else {}
        raw = str(state.get("output_dir", "") or "").strip()
        return Path(raw) if raw else None

    def active_nbv_read_json_artifact(self, path: Path) -> Dict[str, Any]:
        try:
            if Path(path).is_file():
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
        return {}

    def active_nbv_scan_points_for_map(self, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        state = getattr(self, "active_nbv_state", {}) if isinstance(getattr(self, "active_nbv_state", {}), dict) else {}
        state_points = state.get("scan_points", [])
        if isinstance(state_points, list) and state_points:
            return [dict(item) for item in state_points if isinstance(item, dict)]
        if output_dir is None:
            output_dir = self.active_nbv_current_output_dir()
        if output_dir is None:
            return []
        payload = self.active_nbv_read_json_artifact(Path(output_dir) / "scan_points.json")
        raw_points = payload.get("scan_points", [])
        return [dict(item) for item in raw_points if isinstance(item, dict)] if isinstance(raw_points, list) else []

    def active_nbv_map_style(self, status: str) -> Dict[str, str]:
        if hasattr(self, "route5_map_status_style"):
            try:
                return self.route5_map_status_style(status)
            except Exception:
                pass
        colors = {
            "active": "#22d3ee",
            "captured": "#22c55e",
            "done": "#22c55e",
            "visited": "#22c55e",
            "blocked": "#ef4444",
            "failed": "#ef4444",
            "capture_failed": "#ef4444",
            "invalid_capture_pose_retryable": "#fb923c",
            "planned": "#facc15",
            "pending": "#facc15",
        }
        return {"color": colors.get(str(status or "").strip().lower(), "#ffd166"), "outline_color": "#111827"}

    def active_nbv_map_route_points(self, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        if output_dir is None:
            output_dir = self.active_nbv_current_output_dir()
        state = getattr(self, "active_nbv_state", {}) if isinstance(getattr(self, "active_nbv_state", {}), dict) else {}
        active_scan_id = str(state.get("last_scan_id", "") or "")
        route_points: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for idx, point in enumerate(self.active_nbv_scan_points_for_map(output_dir), start=1):
            try:
                float(point.get("x"))
                float(point.get("y"))
            except Exception:
                continue
            scan_id = str(point.get("scan_id", point.get("point_id", "")) or f"active_nbv_scan_{idx}")
            status = str(point.get("status", "planned") or "planned").strip().lower()
            if active_scan_id and scan_id == active_scan_id and status in {"planned", "pending"}:
                status = "active"
            route_points.append(
                {
                    **self.route5_json_safe(point),
                    **self.active_nbv_map_style(status),
                    "label": scan_id,
                    "status": status,
                    "route_point_type": "active_nbv_scan_point",
                    "scan_id": scan_id,
                    "target_id": scan_id,
                    "facade": str(point.get("facade", "") or ""),
                    "overlay": "active_nbv_scan_points",
                }
            )
            seen.add(scan_id)
        if output_dir is not None:
            nav_rows = self.read_jsonl_artifact(Path(output_dir) / "route5_navigation_plan.jsonl")[-80:]
            for nav in nav_rows:
                if not isinstance(nav, dict):
                    continue
                target_id = str(nav.get("target_id", "") or "")
                plan = nav.get("plan", {}) if isinstance(nav.get("plan"), dict) else {}
                waypoints = plan.get("waypoints", []) if isinstance(plan.get("waypoints"), list) else []
                for idx, waypoint in enumerate(waypoints, start=1):
                    if not isinstance(waypoint, dict):
                        continue
                    try:
                        float(waypoint.get("x"))
                        float(waypoint.get("y"))
                    except Exception:
                        continue
                    label = f"{target_id or 'active_nbv'}_wp_{idx}"
                    if label in seen:
                        continue
                    route_points.append(
                        {
                            **self.route5_json_safe(waypoint),
                            **self.active_nbv_map_style("pending"),
                            "label": label,
                            "status": "pending",
                            "route_point_type": "active_nbv_waypoint",
                            "target_id": target_id,
                            "facade": str(nav.get("facade", "") or ""),
                            "overlay": "active_nbv_navigation_waypoints",
                        }
                    )
                    seen.add(label)
            reset_rows = self.read_jsonl_artifact(Path(output_dir) / "route5_target_resets.jsonl")[-80:]
            for reset in reset_rows:
                if not isinstance(reset, dict):
                    continue
                pose = reset.get("reset_target_pose", {}) if isinstance(reset.get("reset_target_pose"), dict) else {}
                try:
                    float(pose.get("x"))
                    float(pose.get("y"))
                except Exception:
                    continue
                label = str(reset.get("reset_target_id", "") or f"active_nbv_reset_{len(route_points) + 1}")
                if label in seen:
                    continue
                route_points.append(
                    {
                        **self.route5_json_safe(pose),
                        **self.active_nbv_map_style("active"),
                        "label": label,
                        "status": "active",
                        "route_point_type": "active_nbv_target_reset",
                        "target_id": str(reset.get("target_id", "") or ""),
                        "facade": str(reset.get("facade", "") or ""),
                        "overlay": "active_nbv_target_resets",
                    }
                )
                seen.add(label)
        return route_points[-500:]

    def active_nbv_map_trajectory(self, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        if output_dir is None:
            output_dir = self.active_nbv_current_output_dir()
        if output_dir is None:
            return []
        points: List[Dict[str, Any]] = []
        trajectory_payload = self.active_nbv_read_json_artifact(Path(output_dir) / "trajectory.json")
        raw_trajectory = trajectory_payload.get("trajectory", [])
        if isinstance(raw_trajectory, list):
            for row in raw_trajectory[-300:]:
                if not isinstance(row, dict):
                    continue
                for key in ("current_pose", "actual_pose", "pose"):
                    pose = row.get(key, {})
                    if isinstance(pose, dict) and "x" in pose and "y" in pose:
                        points.append(dict(pose))
                        break
        for row in self.read_jsonl_artifact(Path(output_dir) / "route5_movement_trace.jsonl")[-400:]:
            if not isinstance(row, dict):
                continue
            pose = row.get("current_pose", row.get("post_action_pose", {}))
            if isinstance(pose, dict) and "x" in pose and "y" in pose:
                points.append(dict(pose))
        return points[-500:]

    def active_nbv_map_obstacle_points(self, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        if output_dir is None:
            output_dir = self.active_nbv_current_output_dir()
        if output_dir is None:
            return []
        summary = self.active_nbv_read_json_artifact(Path(output_dir) / "local_obstacle_summary.json")
        raw_points = summary.get("preview_points", []) if isinstance(summary.get("preview_points"), list) else []
        points: List[Dict[str, Any]] = []
        for idx, point in enumerate(raw_points[:300], start=1):
            if not isinstance(point, dict):
                continue
            try:
                x_value = float(point.get("x"))
                y_value = float(point.get("y"))
            except Exception:
                continue
            confidence = 0.0
            try:
                confidence = float(point.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            points.append(
                {
                    "x": x_value,
                    "y": y_value,
                    "z": point.get("z", 0.0),
                    "label": str(point.get("semantic_hint", "") or f"obs_{idx}"),
                    "status": "local_3d_obstacle",
                    "route_point_type": "local_3d_obstacle",
                    "overlay": "local_3d_obstacle_map",
                    "color": "#f97316" if confidence >= 0.7 else "#f59e0b",
                    "outline_color": "#111827",
                    "radius_px": 3 + min(5, max(0, int(round(confidence * 4.0)))),
                    "confidence": round(max(0.0, min(1.0, confidence)), 4),
                    "source": str(point.get("source", "") or ""),
                }
            )
        return points

    def active_nbv_latest_map_pose(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        latest = self.latest_state.get("pose", {}) if isinstance(getattr(self, "latest_state", {}), dict) and isinstance(self.latest_state.get("pose"), dict) else {}
        if latest:
            return dict(latest)
        trajectory = self.active_nbv_map_trajectory(output_dir)
        return dict(trajectory[-1]) if trajectory else {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}

    def refresh_active_nbv_map(self) -> None:
        self.ensure_active_nbv_state()
        widget = getattr(self, "active_nbv_map_widget", None)
        if widget is None:
            return
        try:
            if not self.load_map_resources(force=not bool(getattr(self, "map_config", {}))):
                self.active_nbv_map_status_var.set("Active NBV Map: map unavailable")
                return
            output_dir = self.active_nbv_current_output_dir()
            pose = self.active_nbv_latest_map_pose(output_dir)
            pose_x = float(pose.get("x", 0.0) or 0.0)
            pose_y = float(pose.get("y", 0.0) or 0.0)
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0)) or 0.0)
            map_frame = getattr(self, "active_nbv_map_frame", None)
            if map_frame is not None and hasattr(widget, "resize_canvas") and hasattr(self, "route5_map_canvas_size_for_width"):
                try:
                    available_w = float(map_frame.winfo_width() or widget.canvas.winfo_width() or 760)
                    size = self.route5_map_canvas_size_for_width(available_w, image_size=self.map_image_size())
                    target_h = max(260, min(420, int(size.get("height", 320))))
                    if abs(int(getattr(widget, "_canvas_w", 0)) - int(size["width"])) > 12 or abs(int(getattr(widget, "_canvas_h", 0)) - target_h) > 12:
                        widget.resize_canvas(int(size["width"]), target_h)
                except Exception:
                    pass
            _houses, boxes = self.build_map_display(pose)
            widget.set_background_image(self.map_image)
            widget.set_calibration(
                self.map_calibration.get("affine_world_to_image"),
                self.map_image_size(),
                [],
                self.map_calibration.get("homography_world_to_image"),
            )
            widget.set_image_layer_offset(*self.map_display_offset_px)
            widget.set_house_boxes(boxes)
            widget.update_houses([])
            widget.update_uav(pose_x, pose_y, pose_yaw)
            route_points = self.active_nbv_map_route_points(output_dir)
            trajectory = self.active_nbv_map_trajectory(output_dir)
            obstacle_points = self.active_nbv_map_obstacle_points(output_dir)
            widget.set_route_plan({"route_points": route_points})
            widget.set_trajectory(trajectory)
            if hasattr(widget, "set_point_overlay_points"):
                widget.set_point_overlay_points(obstacle_points)
            run_name = Path(output_dir).name if output_dir is not None else "current"
            self.active_nbv_map_status_var.set(
                f"Active NBV Map: run={run_name} route_points={len(route_points)} trajectory={len(trajectory)} obstacles={len(obstacle_points)}"
            )
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh Active NBV map failed: %s", exc)
            self.active_nbv_map_status_var.set(f"Active NBV Map: failed: {exc}")

    def on_active_nbv_map_frame_configure(self, _event: Any) -> None:
        self.refresh_active_nbv_map()

    def active_nbv_refresh_preview(self) -> None:
        self.ensure_active_nbv_state()
        text = getattr(self, "active_nbv_preview_text", None)
        if text is None:
            return
        state = getattr(self, "active_nbv_state", {}) if isinstance(getattr(self, "active_nbv_state", {}), dict) else {}
        payload = {
            "stage": state.get("stage", "idle"),
            "target_house_id": state.get("target_house_id", ""),
            "output_dir": state.get("output_dir", ""),
            "planned_scan_count": state.get("planned_scan_count", 0),
            "captured_scan_count": state.get("captured_scan_count", 0),
            "coverage": state.get("coverage_report", {}),
            "rescan": state.get("last_rescan_plan", {}),
        }
        try:
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("end", json.dumps(self.route5_json_safe(payload), indent=2, ensure_ascii=False))
            text.configure(state="disabled")
        except tk.TclError:
            return
        map_text = getattr(self, "active_nbv_map_text", None)
        if map_text is not None:
            try:
                map_text.configure(state="normal")
                map_text.delete("1.0", "end")
                for point in state.get("scan_points", []) if isinstance(state.get("scan_points", []), list) else []:
                    map_text.insert(
                        "end",
                        f"{point.get('status', 'planned'):>18}  {point.get('scan_id', ''):<40} "
                        f"{point.get('facade', ''):<5} {point.get('view_type', '')}\n",
                    )
                map_text.configure(state="disabled")
            except tk.TclError:
                return
        self.refresh_active_nbv_map()

    def open_active_nbv_scan_window(self) -> None:
        self.ensure_active_nbv_state()
        self.ensure_route5_state()
        if self.active_nbv_window is not None and self.active_nbv_window.winfo_exists():
            self.active_nbv_window.lift()
            self.active_nbv_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Active NBV Pointcloud Scan + OR2 Avoidance")
        window.geometry("1120x820")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)

        scroll_host = tk.Frame(window)
        scroll_host.grid(row=0, column=0, sticky="nsew")
        scroll_host.grid_columnconfigure(0, weight=1)
        scroll_host.grid_rowconfigure(0, weight=1)
        scroll_canvas = tk.Canvas(scroll_host, highlightthickness=0, borderwidth=0)
        scroll_bar = ttk.Scrollbar(scroll_host, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scroll_bar.set)
        scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scroll_bar.grid(row=0, column=1, sticky="ns")
        content = tk.Frame(scroll_canvas)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(4, weight=1)
        content_window = scroll_canvas.create_window((0, 0), window=content, anchor="nw")

        def refresh_scroll_region(_event: Optional[tk.Event] = None) -> None:
            try:
                scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
            except tk.TclError:
                return

        def resize_scroll_content(event: tk.Event) -> None:
            try:
                scroll_canvas.itemconfigure(content_window, width=int(event.width))
            except tk.TclError:
                return

        def on_scroll_wheel(event: tk.Event) -> str:
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            else:
                raw_delta = int(getattr(event, "delta", 0) or 0)
                delta = -1 * int(raw_delta / 120) if raw_delta else 0
            if delta:
                try:
                    scroll_canvas.yview_scroll(delta, "units")
                except tk.TclError:
                    return "break"
            return "break"

        def bind_scroll_wheel(_event: Optional[tk.Event] = None) -> None:
            scroll_canvas.bind_all("<MouseWheel>", on_scroll_wheel)
            scroll_canvas.bind_all("<Button-4>", on_scroll_wheel)
            scroll_canvas.bind_all("<Button-5>", on_scroll_wheel)

        def unbind_scroll_wheel(_event: Optional[tk.Event] = None) -> None:
            scroll_canvas.unbind_all("<MouseWheel>")
            scroll_canvas.unbind_all("<Button-4>")
            scroll_canvas.unbind_all("<Button-5>")

        content.bind("<Configure>", refresh_scroll_region)
        scroll_canvas.bind("<Configure>", resize_scroll_content)
        scroll_canvas.bind("<Enter>", bind_scroll_wheel)
        scroll_canvas.bind("<Leave>", unbind_scroll_wheel)
        content.bind("<Enter>", bind_scroll_wheel)
        content.bind("<Leave>", unbind_scroll_wheel)
        self.active_nbv_scroll_canvas = scroll_canvas
        self.active_nbv_scroll_frame = content

        top = tk.LabelFrame(content, text="Active NBV Scan Config")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        for col in (1, 3, 5):
            top.grid_columnconfigure(col, weight=1)
        tk.Label(top, text="Target House").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        combo = ttk.Combobox(top, textvariable=self.llm_route_target_var, values=list(self.house_choice_map.keys()), state="readonly", width=22)
        combo.grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        if combo not in self.house_target_combos:
            self.house_target_combos.append(combo)
        tk.Label(top, text="OR2 Model").grid(row=0, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.llm_route5_representation_model_var).grid(row=0, column=3, sticky="ew", padx=6, pady=5)
        tk.Button(top, text="Default", command=lambda: self.llm_route5_representation_model_var.set(str(self.default_route5_representation_model_path()))).grid(row=0, column=4, sticky="w", padx=6, pady=5)
        tk.Label(top, text="OA3 Plan").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.llm_route5_oa3_plan_var).grid(row=1, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(top, text="Sense interval s").grid(row=1, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.llm_route5_sensing_interval_s_var, width=7).grid(row=1, column=3, sticky="w", padx=6, pady=5)
        tk.Label(top, text="Standoff cm").grid(row=2, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.active_nbv_standoff_cm_var, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=5)
        tk.Label(top, text="Initial sparse views").grid(row=2, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.active_nbv_initial_view_count_var, width=7).grid(row=2, column=3, sticky="w", padx=6, pady=5)
        tk.Label(top, text="Scan z cm").grid(row=2, column=4, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.active_nbv_scan_z_cm_var, width=7).grid(row=2, column=5, sticky="w", padx=6, pady=5)
        tk.Label(top, text="Max rescan rounds").grid(row=3, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.active_nbv_max_rescan_rounds_var, width=7).grid(row=3, column=1, sticky="w", padx=6, pady=5)
        tk.Label(top, text="Facade threshold").grid(row=3, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.active_nbv_facade_threshold_var, width=7).grid(row=3, column=3, sticky="w", padx=6, pady=5)
        tk.Label(top, text="Mean threshold").grid(row=3, column=4, sticky="w", padx=6, pady=5)
        tk.Entry(top, textvariable=self.active_nbv_mean_threshold_var, width=7).grid(row=3, column=5, sticky="w", padx=6, pady=5)

        actions = tk.Frame(content)
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(actions, text="Start", command=self.on_active_nbv_start).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Pause/Resume", command=self.on_active_nbv_toggle_pause).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=self.on_active_nbv_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_active_nbv_clear).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Validate", command=self.on_active_nbv_validate).pack(side="left", padx=6, pady=4)
        tk.Label(actions, textvariable=self.active_nbv_stage_var, anchor="w").pack(side="left", padx=(18, 8), pady=4)
        tk.Label(actions, textvariable=self.active_nbv_progress_text_var, anchor="w").pack(side="left", padx=8, pady=4)
        ttk.Progressbar(actions, variable=self.active_nbv_progress_var, maximum=100.0, length=150, mode="determinate").pack(side="left", padx=8, pady=4)

        tk.Label(content, textvariable=self.active_nbv_status_var, anchor="w", justify="left", wraplength=1080).grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 4))

        monitor = self.build_route5_or2_monitor_panel(content)
        monitor.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 4))

        body = tk.PanedWindow(content, orient="horizontal", sashrelief="raised")
        body.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        preview_frame = tk.LabelFrame(body, text="Preview / Status")
        map_frame = tk.LabelFrame(body, text="Map / Scan Point Text")
        body.add(preview_frame, stretch="always", minsize=480)
        body.add(map_frame, stretch="always", minsize=480)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(1, weight=1)
        map_frame.grid_rowconfigure(2, weight=1)
        map_frame.grid_columnconfigure(0, weight=1)
        preview = tk.Text(preview_frame, height=12, width=64, wrap="none", font=("Consolas", 9))
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=preview.yview)
        preview.configure(yscrollcommand=preview_scroll.set)
        preview.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        preview_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        preview.configure(state="disabled")
        map_toolbar = tk.Frame(map_frame)
        map_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        map_toolbar.grid_columnconfigure(0, weight=1)
        tk.Label(map_toolbar, textvariable=self.active_nbv_map_status_var, anchor="w", wraplength=500, justify="left").grid(row=0, column=0, sticky="ew")
        tk.Button(map_toolbar, text="Refresh Map", command=self.refresh_active_nbv_map).grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.load_map_resources(force=True)
        initial_map_size = self.route5_map_canvas_size_for_width(760, image_size=self.map_image_size()) if hasattr(self, "route5_map_canvas_size_for_width") else {"width": 760, "height": 320}
        map_widget = OverheadMapWidget(
            map_frame,
            world_bounds=self.map_world_bounds,
            canvas_w=int(initial_map_size.get("width", 760)),
            canvas_h=max(260, min(420, int(initial_map_size.get("height", 320)))),
        )
        map_widget.canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        map_text = tk.Text(map_frame, height=12, width=64, wrap="none", font=("Consolas", 9))
        map_scroll = ttk.Scrollbar(map_frame, orient="vertical", command=map_text.yview)
        map_text.configure(yscrollcommand=map_scroll.set)
        map_text.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 6))
        map_scroll.grid(row=2, column=1, sticky="ns", padx=(0, 6), pady=(0, 6))
        map_text.configure(state="disabled")
        self.active_nbv_preview_text = preview
        self.active_nbv_map_text = map_text
        self.active_nbv_map_widget = map_widget
        self.active_nbv_map_frame = map_frame
        map_frame.bind("<Configure>", self.on_active_nbv_map_frame_configure, add="+")

        def close_window() -> None:
            if combo in self.house_target_combos:
                self.house_target_combos.remove(combo)
            self.active_nbv_window = None
            self.active_nbv_preview_text = None
            self.active_nbv_map_text = None
            self.active_nbv_map_widget = None
            self.active_nbv_map_frame = None
            self.active_nbv_scroll_canvas = None
            self.active_nbv_scroll_frame = None
            unbind_scroll_wheel()
            self.route5_or2_state_label = None
            self.route5_or2_rgb_label = None
            self.route5_or2_mask_label = None
            self.route5_or2_rgb_photo = None
            self.route5_or2_mask_photo = None
            self.route5_or2_report_text = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.active_nbv_window = window
        self.active_nbv_refresh_preview()
        self.refresh_active_nbv_map()

    def on_active_nbv_toggle_pause(self) -> None:
        self.ensure_active_nbv_state()
        self.ensure_route5_state()
        if self.active_nbv_pause_event.is_set():
            self.active_nbv_pause_event.clear()
            self.llm_route5_pause_event.clear()
            self.active_nbv_status_var.set("Active NBV: resumed.")
        else:
            self.active_nbv_pause_event.set()
            self.llm_route5_pause_event.set()
            self.active_nbv_status_var.set("Active NBV: paused.")

    def on_active_nbv_start(self) -> None:
        self.ensure_active_nbv_state()
        self.ensure_route5_state()
        session = self.active_session()
        if session is None:
            self.active_nbv_status_var.set("Active NBV: start a session first.")
            return
        if self.active_nbv_thread is not None and self.active_nbv_thread.is_alive():
            self.active_nbv_status_var.set("Active NBV: already running.")
            return
        if getattr(self, "llm_route5_thread", None) is not None and self.llm_route5_thread.is_alive():
            self.active_nbv_status_var.set("Active NBV: stop Route 5 before starting Active NBV.")
            return
        self.active_nbv_stop_event.clear()
        self.active_nbv_pause_event.clear()
        self.llm_route5_stop_event.clear()
        self.llm_route5_pause_event.clear()
        self.active_nbv_thread = threading.Thread(target=lambda: self.active_nbv_scan_worker(session), daemon=True)
        self.active_nbv_thread.start()

    def on_active_nbv_stop(self) -> None:
        self.ensure_active_nbv_state()
        self.ensure_route5_state()
        self.active_nbv_stop_event.set()
        self.active_nbv_pause_event.clear()
        self.llm_route5_stop_event.set()
        self.llm_route5_pause_event.clear()
        self.active_nbv_status_var.set("Active NBV: stop requested.")
        try:
            session = self.active_session()
            if session is not None:
                self.route5_hold(session, output_dir=getattr(self, "active_nbv_output_dir", None), reason="active_nbv_stop_button")
        except Exception:
            pass

    def on_active_nbv_clear(self) -> None:
        self.ensure_active_nbv_state()
        if self.active_nbv_thread is not None and self.active_nbv_thread.is_alive():
            self.active_nbv_status_var.set("Active NBV: stop before clearing.")
            return
        self.active_nbv_state = {}
        self.active_nbv_output_dir = None
        self.active_nbv_progress_var.set(0.0)
        self.active_nbv_progress_text_var.set("Progress: 0%")
        self.active_nbv_stage_var.set("Stage: idle")
        self.active_nbv_status_var.set("Active NBV: cleared.")
        self.active_nbv_refresh_preview()

    def on_active_nbv_validate(self) -> None:
        self.ensure_active_nbv_state()
        output_dir = getattr(self, "active_nbv_output_dir", None)
        if output_dir is None:
            self.active_nbv_status_var.set("Active NBV: no run to validate.")
            return
        points = self.active_nbv_state.get("scan_points", []) if isinstance(self.active_nbv_state.get("scan_points"), list) else []
        house_id = str(self.active_nbv_state.get("target_house_id", "") or "")
        coverage = self.active_nbv_build_coverage_report(house_id, points, output_dir=Path(output_dir))
        self.write_json_artifact(Path(output_dir) / "coverage_report.json", coverage)
        self.active_nbv_state["coverage_report"] = coverage
        self.active_nbv_write_summary_csv(Path(output_dir), coverage, points)
        self.active_nbv_write_state_artifact()
        self.active_nbv_refresh_preview()
        self.active_nbv_status_var.set(f"Active NBV: validation {'PASS' if coverage.get('complete') else 'CHECK'} -> {Path(output_dir) / 'coverage_report.json'}")

    def active_nbv_wait_if_paused(self) -> bool:
        while self.active_nbv_pause_event.is_set() and not self.active_nbv_stop_event.is_set():
            time.sleep(0.1)
        return not self.active_nbv_stop_event.is_set()

    def active_nbv_capture_context_from_nav(self, point: Dict[str, Any], nav_result: Dict[str, Any], capture: Dict[str, Any]) -> Dict[str, Any]:
        status_ok = str(capture.get("status", capture.get("capture_status", "")) or "").strip().lower() == "ok"
        point_count = int(capture.get("point_count", point.get("point_count", 0)) or 0)
        pose_error = nav_result.get("pose_error", {}) if isinstance(nav_result.get("pose_error"), dict) else {}
        distance_error = float(pose_error.get("distance_cm", pose_error.get("xy_error_cm", 0.0)) or 0.0)
        reset_count = len(nav_result.get("target_resets", [])) if isinstance(nav_result.get("target_resets"), list) else int(bool(nav_result.get("route5_target_reset_navigation")))
        return {
            "coverage_gain": 1.0 if status_ok and point_count > 0 else 0.0,
            "view_quality": min(1.0, max(0.0, point_count / 12000.0)) if point_count > 0 else (0.4 if status_ok else 0.0),
            "safety_score": 1.0 if nav_result.get("status") == "ok" else 0.0,
            "redundant_view_penalty": 0.0 if status_ok and point_count > 0 else 0.4,
            "path_length_cost": min(1.0, max(0.0, distance_error / 1000.0)),
            "obstacle_risk": min(1.0, 0.15 * reset_count),
        }

    def active_nbv_execute_scan_points(
        self,
        session: flight.DroneFlightSession,
        output_dir: Path,
        target_house_id: str,
        points: List[Dict[str, Any]],
        *,
        round_index: int,
        all_points: List[Dict[str, Any]],
    ) -> None:
        total = max(1, len(points))
        for idx, point in enumerate(points, start=1):
            if self.active_nbv_stop_event.is_set():
                break
            if not self.active_nbv_wait_if_paused():
                break
            facade = str(point.get("facade", "") or "")
            scan_id = str(point.get("scan_id", "") or f"active_nbv_{idx}")
            facade_dir = output_dir / "facade_observations" / facade
            facade_dir.mkdir(parents=True, exist_ok=True)
            original_pose = self.route3_target_pose_from_point(point)
            target_pose = dict(original_pose)
            self.active_nbv_set_stage("NAV_TO_ACTIVE_NBV_VIEW", output_dir=output_dir, facade=facade, message=f"round {round_index} scan {idx}/{total} {scan_id}")
            nav_result = self.route5_navigate_to_pose_with_fusion(
                session,
                target_pose,
                output_dir=output_dir,
                stage="ACTIVE_NBV_NAV_TO_SCAN_POINT",
                facade=facade,
                target_id=scan_id,
                target_house_id=target_house_id,
            )
            self.active_nbv_log_event(output_dir, "active_nbv_navigation_result", {"scan_id": scan_id, "facade": facade, "navigation": nav_result})
            if nav_result.get("status") != "ok":
                point["status"] = "blocked"
                point["block_reason"] = str(nav_result.get("reason", "navigation_failed") or "navigation_failed")
                episode = {"scan_id": scan_id, "facade": facade, "status": point["status"], "navigation": nav_result}
                self.append_jsonl(output_dir / "active_scan_episode.jsonl", self.route5_json_safe(episode))
                continue
            if isinstance(nav_result.get("final_target_pose"), dict):
                target_pose = dict(nav_result.get("final_target_pose", target_pose))
                point["route5_runtime_target_pose"] = self.route5_json_safe(target_pose)
            capture_pose = self.route3_current_pose(session)
            arrival = self.route5_capture_guard_arrival_state(
                nav_result=nav_result,
                stage="ACTIVE_NBV_NAV_TO_SCAN_POINT",
                facade=facade,
                target_id=scan_id,
                original_target_pose=original_pose,
                runtime_target_pose=target_pose,
                capture_pose=capture_pose,
                config=self.route5_nav_config(),
            )
            guard = self.route5_capture_guard_state(
                target_house_id=target_house_id,
                stage="ACTIVE_NBV_NAV_TO_SCAN_POINT",
                facade=facade,
                target_id=scan_id,
                original_target_pose=original_pose,
                runtime_target_pose=target_pose,
                capture_pose=capture_pose,
                pose_error=nav_result.get("pose_error", {}) if isinstance(nav_result.get("pose_error"), dict) else {},
                arrival_state=arrival,
                config=self.route5_nav_config(),
                capture_kind="active_nbv_scan",
            )
            guard = self.route5_record_capture_guard(output_dir, guard)
            point["route5_capture_guard"] = self.route5_json_safe(guard)
            if not bool(guard.get("capture_guard_passed", False)):
                point["status"] = "invalid_capture_pose_retryable"
                point["block_reason"] = str(guard.get("reason", "capture_guard_failed") or "capture_guard_failed")
                self.route5_write_capture_guard_blocked_decision(output_dir, guard, current_pose=capture_pose)
                episode = {"scan_id": scan_id, "facade": facade, "status": point["status"], "capture_guard": guard}
                self.append_jsonl(output_dir / "active_scan_episode.jsonl", self.route5_json_safe(episode))
                continue
            self.active_nbv_set_stage("CAPTURE_ACTIVE_NBV_VIEW", output_dir=output_dir, facade=facade, message=f"capturing {scan_id}")
            capture = self.route3_capture_scan_point_current(session, output_dir=output_dir, facade_dir=facade_dir, point=point, planned_pose=target_pose)
            capture = self.route5_annotate_scan_capture_guard(
                output_dir,
                facade_dir,
                scan_id=scan_id,
                guard=guard,
                original_target_pose=original_pose,
                runtime_target_pose=target_pose,
                capture_pose=capture_pose,
                capture=capture if isinstance(capture, dict) else {},
            )
            capture_status = str(capture.get("status", capture.get("capture_status", "")) or "").strip().lower()
            point_count = self.active_nbv_capture_point_count(capture if isinstance(capture, dict) else {})
            capture_dirs = self.active_nbv_capture_dirs(capture if isinstance(capture, dict) else {})
            if isinstance(capture, dict):
                capture["point_count"] = int(point_count)
                capture["capture_dirs"] = capture_dirs
                if capture_dirs and not capture.get("capture_dir"):
                    capture["capture_dir"] = capture_dirs[0]
            point["status"] = "captured" if capture_status == "ok" and point_count > 0 else "capture_failed"
            point["point_count"] = int(point_count)
            point["capture_result"] = self.route5_json_safe(capture)
            context = self.active_nbv_capture_context_from_nav(point, nav_result, capture if isinstance(capture, dict) else {})
            score = self.active_nbv_score_view_candidate(point, context)
            point["active_nbv_score"] = score
            episode = {
                "episode_id": output_dir.name,
                "step": len(self.read_jsonl_artifact(output_dir / "active_scan_episode.jsonl")) + 1,
                "house_id": target_house_id,
                "facade": facade,
                "scan_id": scan_id,
                "view_type": point.get("view_type", ""),
                "candidate_pose": original_pose,
                "runtime_target_pose": target_pose,
                "status": point["status"],
                "capture_guard_passed": bool(guard.get("capture_guard_passed", False)),
                "point_count": int(point.get("point_count", 0) or 0),
                "navigation": nav_result,
                "capture": capture,
            }
            self.append_jsonl(output_dir / "active_scan_episode.jsonl", self.route5_json_safe(episode))
            self.append_jsonl(output_dir / "active_scan_reward.jsonl", self.route5_json_safe(score))
            if point["status"] == "captured" and int(point.get("point_count", 0) or 0) > 0:
                self.append_jsonl(
                    output_dir / "selected_keyframes.jsonl",
                    self.route5_json_safe(
                        {
                            "frame_id": scan_id,
                            "house_id": target_house_id,
                            "facade": facade,
                            "source_capture_dir": capture_dirs[0] if capture_dirs else "",
                            "source_capture_dirs": capture_dirs,
                            "selected": True,
                            "selection_reason": ["active_nbv_captured_positive_point_count"],
                            "coverage_gain": score.get("coverage_gain", 0.0),
                            "point_count": int(point.get("point_count", 0) or 0),
                        }
                    ),
                )
            captured = len([item for item in all_points if str(item.get("status", "")) == "captured"])
            self.active_nbv_state.update(
                scan_points=all_points,
                captured_scan_count=captured,
                planned_scan_count=len(all_points),
                last_scan_id=scan_id,
            )
            progress = 100.0 * min(1.0, float(captured) / float(max(1, len(all_points))))
            self.root.after(0, lambda v=progress: self.active_nbv_progress_var.set(max(0.0, min(100.0, v))))
            self.root.after(0, lambda c=captured, p=len(all_points): self.active_nbv_progress_text_var.set(f"Progress: {c}/{p}"))
            self.root.after(0, self.active_nbv_refresh_preview)
            self.active_nbv_write_state_artifact()

    def active_nbv_scan_worker(self, session: flight.DroneFlightSession) -> None:
        self.ensure_active_nbv_state()
        self.ensure_route5_state()
        target_house_id = self.selected_route_target_house_id()
        if not target_house_id:
            self.root.after(0, lambda: self.active_nbv_status_var.set("Active NBV: select a target house first."))
            return
        output_dir = self.make_active_nbv_output_dir(target_house_id)
        self.active_nbv_output_dir = output_dir
        status = "done"
        all_points: List[Dict[str, Any]] = []
        try:
            self.route5_set_control_lock(True)
            self.active_nbv_set_stage("PLAN_INITIAL_SPARSE_VIEWS", output_dir=output_dir, message=f"planning sparse NBV views for house={target_house_id}")
            initial_points = self.active_nbv_initial_scan_points(target_house_id)
            all_points.extend(initial_points)
            self.active_nbv_state = {
                "schema": "active_nbv_scan_state_v1",
                "target_house_id": target_house_id,
                "output_dir": str(output_dir),
                "or2_config": self.active_nbv_route5_oa3_config(),
                "planned_scan_count": len(all_points),
                "captured_scan_count": 0,
                "scan_points": all_points,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.active_nbv_target_var.set(f"Target: {target_house_id}")
            self.active_nbv_write_scan_points(output_dir, target_house_id, all_points)
            self.write_json_artifact(output_dir / "active_nbv_config.json", self.route5_json_safe(self.active_nbv_state))
            self.root.after(0, self.active_nbv_refresh_preview)

            self.active_nbv_execute_scan_points(session, output_dir, target_house_id, initial_points, round_index=0, all_points=all_points)
            coverage = self.active_nbv_build_coverage_report(target_house_id, all_points, output_dir=output_dir)
            self.write_json_artifact(output_dir / "coverage_report.json", coverage)
            self.active_nbv_state["coverage_report"] = coverage
            max_rounds = self.active_nbv_int_param(getattr(self, "active_nbv_max_rescan_rounds_var", None), 2, min_value=0, max_value=10)
            for round_index in range(1, max_rounds + 1):
                if self.active_nbv_stop_event.is_set() or bool(coverage.get("complete", False)):
                    break
                self.active_nbv_set_stage("PLAN_RESCAN", output_dir=output_dir, message=f"planning NBV rescan round {round_index}")
                rescan_plan = self.active_nbv_build_rescan_plan(target_house_id, coverage, all_points, round_index=round_index)
                self.write_json_artifact(output_dir / "rescan_plan.json", rescan_plan)
                self.active_nbv_state["last_rescan_plan"] = rescan_plan
                rescan_points = [point for point in rescan_plan.get("rescan_points", []) if isinstance(point, dict)]
                if not rescan_points:
                    break
                all_points.extend(rescan_points)
                self.active_nbv_write_scan_points(output_dir, target_house_id, all_points)
                self.active_nbv_execute_scan_points(session, output_dir, target_house_id, rescan_points, round_index=round_index, all_points=all_points)
                coverage = self.active_nbv_build_coverage_report(target_house_id, all_points, output_dir=output_dir)
                self.write_json_artifact(output_dir / "coverage_report.json", coverage)
                self.active_nbv_state["coverage_report"] = coverage
                self.active_nbv_write_state_artifact()
            if self.active_nbv_stop_event.is_set():
                status = "stopped"
            self.active_nbv_state["scan_points"] = all_points
            self.active_nbv_state["final_status"] = status
            self.active_nbv_state["coverage_report"] = coverage
            self.active_nbv_write_scan_points(output_dir, target_house_id, all_points)
            if not (output_dir / "rescan_plan.json").is_file():
                self.write_json_artifact(output_dir / "rescan_plan.json", {"schema": "active_nbv_rescan_plan_v1", "house_id": target_house_id, "rescan_points": []})
            self.active_nbv_write_summary_csv(output_dir, coverage, all_points)
            self.active_nbv_write_state_artifact()
            self.root.after(0, self.active_nbv_refresh_preview)
            self.root.after(0, lambda: self.active_nbv_status_var.set(f"Active NBV: {status}, output={output_dir}"))
        except Exception as exc:
            status = "failed"
            LOGGER.exception("Active NBV scan failed")
            self.active_nbv_state["final_status"] = status
            self.active_nbv_state["error"] = str(exc)
            self.active_nbv_write_state_artifact()
            self.root.after(0, lambda e=str(exc): self.active_nbv_status_var.set(f"Active NBV: failed: {e}"))
        finally:
            try:
                self.route5_set_control_lock(False)
            except Exception:
                pass
            self.llm_route5_stop_event.clear()
            self.llm_route5_pause_event.clear()
            self.active_nbv_stop_event.clear()
            self.active_nbv_pause_event.clear()
