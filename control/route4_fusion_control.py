from __future__ import annotations

from copy import deepcopy

from .common import *

from obstacle_avoidance.collect_route_episodes import (
    DEFAULT_ROUTE_SIDE_CORRECTION_CM,
    DEFAULT_ROUTE_VERTICAL_STEP_CM,
    action_payload,
    annotate_collision_state,
    build_route_event,
    distance_3d_cm,
    pose_from_state,
    risk_state_from_summary,
    route_completion_state,
    select_route_action,
)
from obstacle_avoidance_llm.policy import (
    LLM_STRATEGY_METHOD_ID,
    refine_strategy_with_pointcloud_context,
    strategy_from_episode_metadata,
)


class Route4FusionControlMixin:
    def default_route4_representation_model_path(self) -> Path:
        plus = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_plus_model.pt"
        legacy = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_model.pt"
        return plus if plus.is_file() else legacy

    def ensure_route4_state(self) -> None:
        if not hasattr(self, "llm_route4_status_var"):
            self.llm_route4_status_var = tk.StringVar(value="LLM Route V4: idle")
        if not hasattr(self, "llm_route4_map_status_var"):
            self.llm_route4_map_status_var = tk.StringVar(value="Route V4 Map: idle")
        if not hasattr(self, "llm_route4_stage_var"):
            self.llm_route4_stage_var = tk.StringVar(value="Stage: idle")
        if not hasattr(self, "llm_route4_active_var"):
            self.llm_route4_active_var = tk.StringVar(value="Active: n/a")
        if not hasattr(self, "llm_route4_target_var"):
            self.llm_route4_target_var = tk.StringVar(value="Target: n/a")
        if not hasattr(self, "llm_route4_error_var"):
            self.llm_route4_error_var = tk.StringVar(value="Error: n/a")
        if not hasattr(self, "llm_route4_payload_var"):
            self.llm_route4_payload_var = tk.StringVar(value="Payload: hold")
        if not hasattr(self, "llm_route4_progress_text_var"):
            self.llm_route4_progress_text_var = tk.StringVar(value="Fusion: 0%")
        if not hasattr(self, "llm_route4_progress_var"):
            self.llm_route4_progress_var = tk.DoubleVar(value=0.0)
        if not hasattr(self, "llm_route4_current_status_var"):
            self.llm_route4_current_status_var = tk.StringVar(value="Current: idle")
        if not hasattr(self, "llm_route4_next_status_var"):
            self.llm_route4_next_status_var = tk.StringVar(value="Next: n/a")
        if not hasattr(self, "llm_route4_avoidance_status_var"):
            self.llm_route4_avoidance_status_var = tk.StringVar(value="Avoidance: idle")
        if not hasattr(self, "llm_route4_representation_status_var"):
            self.llm_route4_representation_status_var = tk.StringVar(value="Representation: idle")
        if not hasattr(self, "llm_route4_thinking_status_var"):
            self.llm_route4_thinking_status_var = tk.StringVar(value="Thinking: idle")
        if not hasattr(self, "llm_route4_paused_var"):
            self.llm_route4_paused_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route4_auto_refresh_var"):
            self.llm_route4_auto_refresh_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route4_move_tick_ms_var"):
            self.llm_route4_move_tick_ms_var = tk.StringVar(value="150")
        if not hasattr(self, "llm_route4_nav_step_cm_var"):
            self.llm_route4_nav_step_cm_var = tk.StringVar(value="20")
        if not hasattr(self, "llm_route4_reach_tol_cm_var"):
            self.llm_route4_reach_tol_cm_var = tk.StringVar(value="60")
        if not hasattr(self, "llm_route4_z_tol_cm_var"):
            self.llm_route4_z_tol_cm_var = tk.StringVar(value="40")
        if not hasattr(self, "llm_route4_yaw_tol_deg_var"):
            self.llm_route4_yaw_tol_deg_var = tk.StringVar(value="10")
        if not hasattr(self, "llm_route4_max_stage_s_var"):
            self.llm_route4_max_stage_s_var = tk.StringVar(value="90")
        if not hasattr(self, "llm_route4_sensing_interval_s_var"):
            self.llm_route4_sensing_interval_s_var = tk.StringVar(value="1.0")
        if not hasattr(self, "llm_route4_representation_model_var"):
            self.llm_route4_representation_model_var = tk.StringVar(value=str(self.default_route4_representation_model_path()))
        if not hasattr(self, "llm_route4_window"):
            self.llm_route4_window = None
        if not hasattr(self, "llm_route4_window_canvas"):
            self.llm_route4_window_canvas = None
        if not hasattr(self, "llm_route4_window_content"):
            self.llm_route4_window_content = None
        if not hasattr(self, "llm_route4_window_content_window"):
            self.llm_route4_window_content_window = None
        if not hasattr(self, "llm_route4_map_widget"):
            self.llm_route4_map_widget = None
        if not hasattr(self, "llm_route4_preview_text"):
            self.llm_route4_preview_text = None
        if not hasattr(self, "llm_route4_analysis_text"):
            self.llm_route4_analysis_text = None
        if not hasattr(self, "llm_route4_rgb_label"):
            self.llm_route4_rgb_label = None
        if not hasattr(self, "llm_route4_rgb_photo"):
            self.llm_route4_rgb_photo = None
        if not hasattr(self, "llm_route4_state"):
            self.llm_route4_state = {}
        if not hasattr(self, "llm_route4_completed_facades"):
            self.llm_route4_completed_facades = set()
        if not hasattr(self, "llm_route4_blocked_facades"):
            self.llm_route4_blocked_facades = set()
        if not hasattr(self, "llm_route4_thread"):
            self.llm_route4_thread = None
        if not hasattr(self, "llm_route4_stop_event"):
            self.llm_route4_stop_event = threading.Event()
        if not hasattr(self, "llm_route4_pause_event"):
            self.llm_route4_pause_event = threading.Event()
        if not hasattr(self, "llm_route4_control_locked"):
            self.llm_route4_control_locked = False
        if not hasattr(self, "llm_route4_auto_refresh_job"):
            self.llm_route4_auto_refresh_job = None

    def route4_float_param(self, var: tk.StringVar, default: float, *, min_value: float, max_value: float) -> float:
        try:
            value = float(var.get().strip())
        except Exception:
            value = float(default)
        return max(float(min_value), min(float(max_value), float(value)))

    def route4_nav_config(self) -> Dict[str, float]:
        self.ensure_route4_state()
        return {
            "move_tick_ms": self.route4_float_param(self.llm_route4_move_tick_ms_var, 150.0, min_value=50.0, max_value=2000.0),
            "nav_step_cm": self.route4_float_param(self.llm_route4_nav_step_cm_var, 20.0, min_value=5.0, max_value=200.0),
            "reach_tol_cm": self.route4_float_param(self.llm_route4_reach_tol_cm_var, 60.0, min_value=5.0, max_value=500.0),
            "z_tol_cm": self.route4_float_param(self.llm_route4_z_tol_cm_var, 40.0, min_value=5.0, max_value=500.0),
            "yaw_tol_deg": self.route4_float_param(self.llm_route4_yaw_tol_deg_var, 10.0, min_value=1.0, max_value=90.0),
            "max_stage_s": self.route4_float_param(self.llm_route4_max_stage_s_var, 90.0, min_value=5.0, max_value=900.0),
        }

    def route4_sensing_config(self) -> Dict[str, Any]:
        self.ensure_route4_state()
        interval_s = self.route4_float_param(self.llm_route4_sensing_interval_s_var, 1.0, min_value=0.2, max_value=30.0)
        return {
            "sensing_interval_s": float(interval_s),
            "representation_model_path": str(self.llm_route4_representation_model_var.get() or ""),
            "llm_strategy_mode": LLM_STRATEGY_METHOD_ID,
            "capture_processing": "minimal",
        }

    def route4_scan_boundary_policy(self) -> Dict[str, Any]:
        max_yaw = min(45.0, float(LLM_ROUTE3_PANORAMA_MAX_YAW_DELTA_DEG))
        yaw_offset = min(35.0, max_yaw)
        return {
            "source": "route4_scan_boundary_compaction_v1",
            "applies_to": "open_llm_route_window_4_only",
            "clamp_axis_to_safe_interval": True,
            "clamp_axis_to_house_facade_bbox": True,
            "axis_clamp_source": "safe_interval_intersect_house_facade_bbox",
            "max_physical_axis_samples_per_band": 7,
            "required_physical_roles": ["left_boundary", "center", "right_boundary"],
            "boundary_coverage_mode": "in_place_yaw_supplement",
            "boundary_yaw_offset_deg": float(yaw_offset),
            "boundary_yaw_max_abs_offset_deg": float(max_yaw),
            "reason": "Avoid dense lateral edge driving; use boundary yaw to supplement edge coverage.",
        }

    def route4_scan_axis_key(self, facade: str) -> str:
        facade = str(facade or "").strip().lower()
        return "x" if facade in {"south", "north"} else "y"

    def route4_scan_axis_value(self, point: Dict[str, Any], facade: str) -> Optional[float]:
        key = self.route4_scan_axis_key(facade)
        value = self._as_float_or_none(point.get(key))
        return None if value is None else float(value)

    def route4_safe_axis_bounds_for_points(
        self,
        points: List[Dict[str, Any]],
        facade: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        mins: List[float] = []
        maxs: List[float] = []
        axes: List[float] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            axis = self.route4_scan_axis_value(point, facade)
            if axis is not None:
                axes.append(float(axis))
            axis_min = self._as_float_or_none(point.get("safe_axis_min"))
            axis_max = self._as_float_or_none(point.get("safe_axis_max"))
            if axis_min is not None and axis_max is not None:
                low = min(float(axis_min), float(axis_max))
                high = max(float(axis_min), float(axis_max))
                mins.append(low)
                maxs.append(high)
        if mins and maxs:
            return min(mins), max(maxs)
        if axes:
            return min(axes), max(axes)
        return None, None

    def route4_house_facade_axis_bounds(self, house_id: str, facade: str) -> Tuple[Optional[float], Optional[float]]:
        bbox = self.house_world_bbox_for_id(str(house_id or ""))
        if not bbox:
            return None, None
        try:
            axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
            return min(float(axis_min), float(axis_max)), max(float(axis_min), float(axis_max))
        except Exception:
            return None, None

    def route4_effective_scan_axis_bounds(
        self,
        points: List[Dict[str, Any]],
        *,
        house_id: str,
        facade: str,
    ) -> Dict[str, Any]:
        safe_min, safe_max = self.route4_safe_axis_bounds_for_points(points, facade)
        house_min, house_max = self.route4_house_facade_axis_bounds(house_id, facade)
        if safe_min is not None and safe_max is not None:
            safe_low = min(float(safe_min), float(safe_max))
            safe_high = max(float(safe_min), float(safe_max))
        else:
            safe_low = safe_high = None
        if house_min is not None and house_max is not None:
            house_low = min(float(house_min), float(house_max))
            house_high = max(float(house_min), float(house_max))
        else:
            house_low = house_high = None

        if safe_low is not None and safe_high is not None and house_low is not None and house_high is not None:
            low = max(float(safe_low), float(house_low))
            high = min(float(safe_high), float(house_high))
            source = "safe_interval_intersect_house_facade_bbox"
            if low > high:
                low, high = float(house_low), float(house_high)
                source = "house_facade_bbox_no_safe_overlap"
        elif house_low is not None and house_high is not None:
            low, high = float(house_low), float(house_high)
            source = "house_facade_bbox"
        elif safe_low is not None and safe_high is not None:
            low, high = float(safe_low), float(safe_high)
            source = "safe_interval_only_no_house_bbox"
        else:
            low = high = None
            source = "missing_axis_bounds"
        return {
            "axis_min": low,
            "axis_max": high,
            "safe_axis_min_original": safe_low,
            "safe_axis_max_original": safe_high,
            "house_facade_axis_min": house_low,
            "house_facade_axis_max": house_high,
            "axis_clamp_source": source,
        }

    def route4_boundary_axis_targets(self, axis_min: float, axis_max: float, raw_unique_count: int) -> List[float]:
        low = float(min(axis_min, axis_max))
        high = float(max(axis_min, axis_max))
        if abs(high - low) <= 1e-6:
            return [round(low, 2)]
        max_count = int(self.route4_scan_boundary_policy()["max_physical_axis_samples_per_band"])
        count = max(2, min(max_count, max(1, int(raw_unique_count))))
        if count >= 3 and abs(high - low) >= 1.0:
            return [round(low + (high - low) * float(idx) / float(count - 1), 2) for idx in range(count)]
        return [round(low, 2), round(high, 2)]

    def route4_apply_axis_pose_to_scan_point(
        self,
        point: Dict[str, Any],
        *,
        house_id: str,
        facade: str,
        axis_value: float,
    ) -> Dict[str, Any]:
        item = dict(point)
        axis_key = self.route4_scan_axis_key(facade)
        standoff = self._as_float_or_none(item.get("standoff_cm"))
        z_cm = self._as_float_or_none(item.get("z"))
        bbox = self.house_world_bbox_for_id(house_id)
        if bbox and standoff is not None and z_cm is not None:
            pose = self.route2_facade_pose_from_axis(bbox, facade, float(axis_value), float(standoff), float(z_cm))
            item.update(pose)
        else:
            item[axis_key] = round(float(axis_value), 2)
        return item

    def route4_boundary_yaw_supplement_meta(self, base_yaw_deg: float, boundary_role: str) -> Dict[str, Any]:
        policy = self.route4_scan_boundary_policy()
        offset_abs = float(policy["boundary_yaw_offset_deg"])
        offset = -offset_abs if str(boundary_role) == "left_boundary" else offset_abs
        offset = max(-float(policy["boundary_yaw_max_abs_offset_deg"]), min(float(policy["boundary_yaw_max_abs_offset_deg"]), offset))
        supplement_yaw = self._normalize_angle_deg(float(base_yaw_deg) + float(offset))
        return {
            "enabled": True,
            "source": "route4_boundary_yaw_supplement",
            "coverage_mode": "yaw_in_place_no_lateral_overrun",
            "base_yaw_deg": round(float(base_yaw_deg), 2),
            "offset_deg": round(float(offset), 2),
            "supplement_yaw_deg": round(float(supplement_yaw), 2),
            "max_abs_offset_deg": float(policy["boundary_yaw_max_abs_offset_deg"]),
        }

    def route4_compact_boundary_scan_points(
        self,
        points: List[Dict[str, Any]],
        *,
        house_id: str,
        facade: str,
    ) -> Dict[str, Any]:
        policy = self.route4_scan_boundary_policy()
        facade = str(facade or "").strip().lower()
        axis_key = self.route4_scan_axis_key(facade)
        raw_points = [dict(point) for point in points if isinstance(point, dict)]
        full_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        passthrough: List[Dict[str, Any]] = []
        for point in raw_points:
            if str(point.get("view_type", "") or "") == "facade_floor_band_scan":
                key = (
                    str(point.get("height_band", "") or "band"),
                    int(float(point.get("floor_index", 0) or 0)),
                )
                full_groups.setdefault(key, []).append(point)
            else:
                passthrough.append(point)

        compacted: List[Dict[str, Any]] = []
        raw_physical_count = 0
        physical_axis_sample_count = 0
        yaw_supplement_record_count = 0
        next_local_index = 0

        for (_height_band, _floor_index), group in full_groups.items():
            group.sort(key=lambda item: (
                self.route4_scan_axis_value(item, facade) if self.route4_scan_axis_value(item, facade) is not None else 0.0,
                int(item.get("local_scan_index", 0) or 0),
            ))
            raw_axes = [
                round(float(axis), 2)
                for axis in (self.route4_scan_axis_value(point, facade) for point in group)
                if axis is not None
            ]
            raw_unique_axes = sorted(set(raw_axes))
            raw_physical_count += len(raw_unique_axes)
            bounds = self.route4_effective_scan_axis_bounds(group, house_id=house_id, facade=facade)
            axis_min = bounds.get("axis_min")
            axis_max = bounds.get("axis_max")
            if axis_min is None or axis_max is None:
                compacted.extend(dict(point) for point in group)
                physical_axis_sample_count += len(raw_unique_axes)
                continue
            targets = self.route4_boundary_axis_targets(float(axis_min), float(axis_max), len(raw_unique_axes))
            physical_axis_sample_count += len(targets)
            for target_index, target_axis in enumerate(targets):
                template = min(
                    group,
                    key=lambda point: abs(float(self.route4_scan_axis_value(point, facade) or target_axis) - float(target_axis)),
                )
                item = deepcopy(template)
                item["source_original_scan_id"] = template.get("scan_id", "")
                item["source_original_local_scan_index"] = template.get("local_scan_index")
                item = self.route4_apply_axis_pose_to_scan_point(item, house_id=house_id, facade=facade, axis_value=float(target_axis))
                item["safe_axis_min"] = round(float(axis_min), 2)
                item["safe_axis_max"] = round(float(axis_max), 2)
                if bounds.get("safe_axis_min_original") is not None:
                    item["safe_axis_min_original"] = round(float(bounds["safe_axis_min_original"]), 2)
                if bounds.get("safe_axis_max_original") is not None:
                    item["safe_axis_max_original"] = round(float(bounds["safe_axis_max_original"]), 2)
                if bounds.get("house_facade_axis_min") is not None:
                    item["house_facade_axis_min"] = round(float(bounds["house_facade_axis_min"]), 2)
                if bounds.get("house_facade_axis_max") is not None:
                    item["house_facade_axis_max"] = round(float(bounds["house_facade_axis_max"]), 2)
                item["axis_clamp_source"] = str(bounds.get("axis_clamp_source", "") or "")
                item["route4_scan_boundary_policy"] = dict(policy)
                item["route4_axis_key"] = axis_key
                item["route4_axis_value"] = round(float(target_axis), 2)
                item["route4_axis_ratio"] = round(float(target_index) / float(max(1, len(targets) - 1)), 4)
                item["route4_compacted_from_raw_count"] = len(group)
                item["route4_raw_physical_axis_sample_count"] = len(raw_unique_axes)
                item["physical_axis_sample_count"] = len(targets)
                item["planned_facade_sample_count"] = len(targets)
                item["is_yaw_supplement"] = False
                if target_index == 0:
                    boundary_role = "left_boundary"
                elif target_index == len(targets) - 1:
                    boundary_role = "right_boundary"
                elif len(targets) >= 3 and target_index == len(targets) // 2:
                    boundary_role = "center"
                else:
                    boundary_role = "interior"
                item["boundary_role"] = boundary_role
                item["axis_clamped"] = boundary_role in {"left_boundary", "right_boundary"} or (
                    self.route4_scan_axis_value(template, facade) is not None
                    and abs(float(self.route4_scan_axis_value(template, facade) or target_axis) - float(target_axis)) > 0.01
                )
                item["local_scan_index"] = next_local_index
                next_local_index += 1
                if boundary_role in {"left_boundary", "right_boundary"}:
                    yaw_meta = self.route4_boundary_yaw_supplement_meta(float(item.get("yaw_deg", item.get("yaw", 0.0)) or 0.0), boundary_role)
                    yaw_meta["supplement_record_planned"] = True
                    item["yaw_supplement"] = yaw_meta
                    compacted.append(item)

                    supplement = deepcopy(item)
                    supplement["view_type"] = "boundary_yaw_supplement_scan"
                    supplement["semantic_region"] = f"{boundary_role}_yaw_supplement"
                    supplement["is_yaw_supplement"] = True
                    supplement["yaw_deg"] = yaw_meta["supplement_yaw_deg"]
                    supplement["capture_trigger"] = "arrive_boundary_yaw_hover_capture"
                    supplement["route4_boundary_yaw_source_scan_id"] = item.get("scan_id", "")
                    supplement["local_scan_index"] = next_local_index
                    supplement["yaw_supplement"] = dict(yaw_meta, applied_to_capture=True)
                    next_local_index += 1
                    compacted.append(supplement)
                    yaw_supplement_record_count += 1
                else:
                    item["yaw_supplement"] = {"enabled": False, "reason": "not_boundary_point"}
                    compacted.append(item)

        for point in passthrough:
            item = deepcopy(point)
            bounds = self.route4_effective_scan_axis_bounds([item], house_id=house_id, facade=facade)
            axis_min = bounds.get("axis_min")
            axis_max = bounds.get("axis_max")
            axis_value = self.route4_scan_axis_value(item, facade)
            if axis_min is not None and axis_max is not None and axis_value is not None:
                low = min(float(axis_min), float(axis_max))
                high = max(float(axis_min), float(axis_max))
                clamped = max(low, min(high, float(axis_value)))
                if abs(clamped - float(axis_value)) > 0.01:
                    item = self.route4_apply_axis_pose_to_scan_point(item, house_id=house_id, facade=facade, axis_value=clamped)
                    item["axis_clamped"] = True
                else:
                    item["axis_clamped"] = False
                item["safe_axis_min"] = round(low, 2)
                item["safe_axis_max"] = round(high, 2)
                if bounds.get("safe_axis_min_original") is not None:
                    item["safe_axis_min_original"] = round(float(bounds["safe_axis_min_original"]), 2)
                if bounds.get("safe_axis_max_original") is not None:
                    item["safe_axis_max_original"] = round(float(bounds["safe_axis_max_original"]), 2)
                if bounds.get("house_facade_axis_min") is not None:
                    item["house_facade_axis_min"] = round(float(bounds["house_facade_axis_min"]), 2)
                if bounds.get("house_facade_axis_max") is not None:
                    item["house_facade_axis_max"] = round(float(bounds["house_facade_axis_max"]), 2)
                item["axis_clamp_source"] = str(bounds.get("axis_clamp_source", "") or "")
                item["route4_axis_key"] = axis_key
                item["route4_axis_value"] = round(clamped, 2)
            item["route4_scan_boundary_policy"] = dict(policy)
            item["boundary_role"] = str(item.get("boundary_role", "") or "semantic_confirm")
            item["physical_axis_sample_count"] = int(policy["max_physical_axis_samples_per_band"])
            item["yaw_supplement"] = {"enabled": False, "reason": "semantic_confirm_or_non_full_facade_point"}
            item["local_scan_index"] = next_local_index
            next_local_index += 1
            compacted.append(item)

        return {
            "points": compacted,
            "policy": policy,
            "raw_point_count": len(raw_points),
            "raw_physical_axis_sample_count": raw_physical_count,
            "physical_axis_sample_count": physical_axis_sample_count,
            "yaw_supplement_record_count": yaw_supplement_record_count,
            "total_capture_record_count": len(compacted),
        }

    def route4_write_merged_scan_points(
        self,
        output_dir: Path,
        house_id: str,
        *,
        policy: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        points = self.route2_existing_facade_scan_points(output_dir)
        points.sort(key=lambda item: (self.route2_scan_order_from_point(item), str(item.get("scan_id", "") or "")))
        facade_counts: Dict[str, int] = {}
        facade_policies: Dict[str, Any] = {}
        for point in points:
            facade = str(point.get("facade", "") or "")
            facade_counts[facade] = facade_counts.get(facade, 0) + 1
        root = output_dir / "facade_observations"
        if root.exists():
            for path in sorted(root.glob("*/facade_search_plan.json")):
                payload = flight.read_json_object(path)
                if not isinstance(payload, dict):
                    continue
                facade = str(payload.get("facade", "") or "")
                facade_policy = payload.get("route4_scan_boundary_policy")
                if facade and isinstance(facade_policy, dict):
                    facade_policies[facade] = facade_policy
        payload = {
            "schema": "facade_v4_fused_global_scan_points",
            "target_house_id": house_id,
            "scan_points": points,
            "facade_counts": facade_counts,
            "total_scan_count": len(points),
            "route4_scan_boundary_policy": policy or self.route4_scan_boundary_policy(),
            "route4_scan_boundary_policy_by_facade": facade_policies,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(output_dir / "scan_points.json", payload)
        return points

    def route4_plan_facade_scan_current(self) -> Dict[str, Any]:
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        if not analysis:
            analysis = self.route2_fallback_facade_analysis("Route V4 used fallback before VLM analysis.")
        raw_points = self.route2_generate_facade_scan_points(analysis)
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            raise RuntimeError("missing facade output directory")
        compacted = self.route4_compact_boundary_scan_points(raw_points, house_id=house_id, facade=facade)
        points = [point for point in compacted.get("points", []) if isinstance(point, dict)]
        next_hint = state.get("next_facade_hint", {}) if isinstance(state.get("next_facade_hint"), dict) else {}
        next_observation = next_hint.get("observation_point", {}) if isinstance(next_hint.get("observation_point"), dict) else {}
        points = self.route2_order_scan_points_continuously(points, next_observation_pose=next_observation)
        points = self.route2_assign_global_scan_ids(output_dir, house_id, facade, points)
        validation = self.scan_point_validation_report(house_id, points)
        counts = {
            "raw_point_count": int(compacted.get("raw_point_count", len(raw_points))),
            "raw_physical_axis_sample_count": int(compacted.get("raw_physical_axis_sample_count", 0)),
            "physical_axis_sample_count": int(compacted.get("physical_axis_sample_count", 0)),
            "yaw_supplement_record_count": int(compacted.get("yaw_supplement_record_count", 0)),
            "total_capture_record_count": len(points),
        }
        search_plan = {
            "schema": "facade_v4_fused_scan_plan",
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "observation_point": state.get("observation_point", {}),
            "next_facade_hint": next_hint,
            "facade_analysis": analysis,
            "scan_points": points,
            "scan_point_validation_report": validation,
            "route_blocked_by_safety": not bool(validation.get("valid", False)),
            "route4_scan_boundary_policy": compacted.get("policy", self.route4_scan_boundary_policy()),
            "route4_scan_counts": counts,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(facade_dir / "facade_search_plan.json", search_plan)
        merged_points = self.route4_write_merged_scan_points(
            output_dir,
            house_id,
            policy=search_plan["route4_scan_boundary_policy"],
        )
        self.route2_update_state(facade_analysis=analysis, facade_search_plan=search_plan, facade_scan_points=points, validation_report=validation)
        self.route2_write_state_artifact()
        self.route4_update_state(
            route4_scan_boundary_policy=search_plan["route4_scan_boundary_policy"],
            last_scan_plan_counts=counts,
        )
        self.route4_write_state_artifact()
        return {
            "search_plan": search_plan,
            "points": points,
            "validation": validation,
            "merged_points": merged_points,
            "boundary_policy": search_plan["route4_scan_boundary_policy"],
            "scan_counts": counts,
        }

    def make_route4_fused_output_dir(self, target_house_id: str) -> Path:
        root = self.resolve_project_path("llm_route4_fusion_runs")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_autosearch_v4_fused_{timestamp}"
        root.mkdir(parents=True, exist_ok=True)
        candidate = root / base_name
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = root / f"{base_name}_{suffix}"
        (candidate / "frames").mkdir(parents=True, exist_ok=True)
        (candidate / "reconstruction").mkdir(parents=True, exist_ok=True)
        (candidate / "facade_observations").mkdir(parents=True, exist_ok=True)
        return candidate

    def route4_state_output_dir(self) -> Optional[Path]:
        self.ensure_route4_state()
        state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        raw = str(state.get("output_dir", "") or "")
        if not raw:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        (path / "reconstruction").mkdir(parents=True, exist_ok=True)
        (path / "facade_observations").mkdir(parents=True, exist_ok=True)
        return path

    def route4_update_state(self, **updates: Any) -> Dict[str, Any]:
        self.ensure_route4_state()
        state = dict(self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {})
        state.update(updates)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route4_state = state
        return state

    def route4_write_state_artifact(self) -> None:
        output_dir = self.route4_state_output_dir()
        if output_dir is None:
            return
        self.write_json_artifact(output_dir / "route4_fusion_state.json", self.llm_route4_state)

    def route4_log_event(self, output_dir: Optional[Path], event_type: str, payload: Dict[str, Any]) -> None:
        if output_dir is None:
            return
        row = {
            "event_type": str(event_type),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "payload": payload,
        }
        self.append_jsonl(output_dir / "route4_fusion_events.jsonl", row)

    def route4_json_safe(self, value: Any, *, max_string: int = 4000) -> Any:
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                key_lower = key_text.lower()
                if key_lower in {"api_key", "apikey", "authorization"} or "api_key" in key_lower:
                    continue
                result[key_text] = self.route4_json_safe(item, max_string=max_string)
            return result
        if isinstance(value, list):
            return [self.route4_json_safe(item, max_string=max_string) for item in value]
        if isinstance(value, tuple):
            return [self.route4_json_safe(item, max_string=max_string) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return str(value)
        if isinstance(value, str) and len(value) > max_string:
            return value[:max_string] + "...<truncated>"
        return value

    def route4_next_llm_call_id(self, call_type: str) -> str:
        self.ensure_route4_state()
        seq = int(self.llm_route4_state.get("llm_call_seq", 0) or 0) + 1
        self.route4_update_state(llm_call_seq=seq)
        safe_type = re.sub(r"[^a-zA-Z0-9_]+", "_", str(call_type or "llm")).strip("_").lower() or "llm"
        return f"route4_llm_{seq:04d}_{safe_type}"

    def route4_log_llm_call(
        self,
        output_dir: Optional[Path],
        call_type: str,
        context: Dict[str, Any],
        response: Dict[str, Any],
        *,
        frame_id: Optional[int] = None,
        facade: str = "",
        target_id: str = "",
        decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        call_id = self.route4_next_llm_call_id(call_type)
        raw_text = str((response if isinstance(response, dict) else {}).get("raw_text", "") or "")
        record = {
            "call_id": call_id,
            "call_type": str(call_type or "llm"),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "frame_id": frame_id,
            "facade": str(facade or (context if isinstance(context, dict) else {}).get("facade", "") or ""),
            "target_id": str(target_id or (context if isinstance(context, dict) else {}).get("target_id", "") or ""),
            "context": self.route4_json_safe(context if isinstance(context, dict) else {}),
            "response": self.route4_json_safe(response if isinstance(response, dict) else {}),
            "decision": self.route4_json_safe(decision if isinstance(decision, dict) else {}),
            "raw_text_preview": raw_text[:1000],
        }
        if output_dir is not None:
            self.append_jsonl(output_dir / "route4_llm_calls.jsonl", record)
        return {
            "call_id": call_id,
            "call_type": record["call_type"],
            "frame_id": frame_id,
            "facade": record["facade"],
            "target_id": record["target_id"],
            "decision_reason": str((decision if isinstance(decision, dict) else {}).get("reason", "") or ""),
            "raw_text_preview": record["raw_text_preview"],
        }

    def route4_frame_dir_for_event(self, output_dir: Path, event: Dict[str, Any]) -> Path:
        capture_dir = str((event if isinstance(event, dict) else {}).get("capture_dir", "") or "")
        if capture_dir:
            return Path(capture_dir)
        frame_id = int(float((event if isinstance(event, dict) else {}).get("frame_id", 0) or 0))
        return output_dir / "frames" / f"frame_{frame_id:06d}"

    def route4_strategy_cache_summary(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        strategy = strategy if isinstance(strategy, dict) else {}
        return {
            "key": str(strategy.get("strategy_cache_key", "") or ""),
            "hit": bool(strategy.get("strategy_cache_hit", False)),
            "source": str(strategy.get("strategy_source", "") or ""),
            "reason": str(strategy.get("strategy_cache_reason", "") or ""),
        }

    def route4_write_frame_decision(
        self,
        output_dir: Path,
        event: Dict[str, Any],
        *,
        final_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = event if isinstance(event, dict) else {}
        frame_id = int(float(event.get("frame_id", 0) or 0))
        strategy = event.get("llm_strategy", {}) if isinstance(event.get("llm_strategy"), dict) else {}
        gate = event.get("avoidance_gate", {}) if isinstance(event.get("avoidance_gate"), dict) else {}
        selected_reason = str(event.get("selected_action_reason", "") or gate.get("reason", "") or "")
        decision = {
            "schema": "route4_frame_decision_v1",
            "frame_id": frame_id,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "stage": str(event.get("route4_stage", event.get("stage", "")) or ""),
            "facade": str(event.get("facade", "") or ""),
            "target_id": str(event.get("target_id", "") or ""),
            "current_pose": self.route4_json_safe(event.get("current_pose", event.get("pose", {}))),
            "target_pose": self.route4_json_safe(event.get("target_waypoint", event.get("goal_pose", {}))),
            "lookahead_plan": self.route4_json_safe(event.get("depth_lookahead_plan", {})),
            "pointcloud_summary": self.route4_json_safe(event.get("pointcloud_summary", event.get("depth_obstacle_summary", {}))),
            "representation_prediction": self.route4_json_safe(event.get("representation_prediction", {})),
            "avoidance_gate": self.route4_json_safe(event.get("avoidance_gate", {})),
            "selected_action": str(event.get("selected_action", "") or ""),
            "selected_action_reason": selected_reason,
            "decision_reason": str(selected_reason or event.get("goal_completion_reason", "") or ""),
            "risk_state": str(event.get("risk_state", "") or ""),
            "nominal_payload": self.route4_json_safe(event.get("nominal_action", event.get("nominal_payload", {}))),
            "selected_action_payload": self.route4_json_safe(event.get("selected_action_payload", {})),
            "final_payload": self.route4_json_safe(final_payload if isinstance(final_payload, dict) else event.get("selected_action_payload", {})),
            "collision_state": bool(event.get("collision_state", False)),
            "avoidance_failed": bool(event.get("avoidance_failed", False)),
            "llm_calls": self.route4_json_safe(event.get("llm_call_refs", [])),
            "strategy_cache": self.route4_strategy_cache_summary(strategy),
            "strategy_source": str(strategy.get("strategy_source", "") or ""),
            "memory_updates": self.route4_json_safe(event.get("house_memory_updates", [])),
        }
        frame_dir = self.route4_frame_dir_for_event(output_dir, event)
        frame_dir.mkdir(parents=True, exist_ok=True)
        self.write_json_artifact(frame_dir / "decision.json", decision)
        self.append_jsonl(output_dir / "route4_frame_decisions.jsonl", decision)
        return decision

    def route4_empty_house_memory(self, target_house_id: str, *, history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "schema": "route4_house_exploration_memory_v1",
            "target_house_id": str(target_house_id or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "history": history if isinstance(history, dict) else {},
            "facades": {
                facade: {
                    "facade": facade,
                    "status": "pending",
                    "attempt_count": 0,
                    "observation_attempts": [],
                    "failure_reasons": [],
                    "degraded_reasons": [],
                    "obstacle_label_conflicts": [],
                    "success_frames": [],
                    "scan_coverage": {},
                    "entrance_candidates": [],
                    "llm_decision_reasons": [],
                    "last_safe_observation_pose": {},
                    "last_updated_at": "",
                }
                for facade in ("west", "south", "east", "north")
            },
        }

    def route4_house_memory_path(self, output_dir: Path) -> Path:
        return output_dir / "house_exploration_memory.json"

    def route4_house_memory_events_path(self, output_dir: Path) -> Path:
        return output_dir / "house_exploration_memory_events.jsonl"

    def route4_cross_run_house_memory_path(self, output_dir: Path, target_house_id: str) -> Path:
        return output_dir.parent / "house_memory" / f"house_{str(target_house_id or '').strip()}.json"

    def route4_initialize_house_memory(self, output_dir: Path, target_house_id: str) -> Dict[str, Any]:
        history_path = self.route4_cross_run_house_memory_path(output_dir, target_house_id)
        history = flight.read_json_object(history_path) if history_path.is_file() else {}
        memory = self.route4_empty_house_memory(target_house_id, history=history)
        self.write_json_artifact(self.route4_house_memory_path(output_dir), memory)
        agenda = {facade: item.get("status", "pending") for facade, item in memory["facades"].items()}
        self.route4_update_state(house_memory=self.route4_house_memory_summary(memory), mandatory_facade_agenda=agenda)
        self.route4_write_state_artifact()
        self.append_jsonl(
            self.route4_house_memory_events_path(output_dir),
            {
                "event_type": "memory_initialized",
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                "target_house_id": str(target_house_id or ""),
                "history_loaded": bool(history),
            },
        )
        return memory

    def route4_read_house_memory(self, output_dir: Path, target_house_id: str) -> Dict[str, Any]:
        path = self.route4_house_memory_path(output_dir)
        if path.is_file():
            payload = flight.read_json_object(path)
            if isinstance(payload, dict) and isinstance(payload.get("facades"), dict):
                return payload
        return self.route4_initialize_house_memory(output_dir, target_house_id)

    def route4_house_memory_summary(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        facades = memory.get("facades", {}) if isinstance(memory, dict) and isinstance(memory.get("facades"), dict) else {}
        return {
            "target_house_id": str(memory.get("target_house_id", "") or "") if isinstance(memory, dict) else "",
            "facade_status": {str(facade): str(item.get("status", "") or "") for facade, item in facades.items() if isinstance(item, dict)},
            "updated_at": str(memory.get("updated_at", "") or "") if isinstance(memory, dict) else "",
        }

    def route4_append_unique(self, values: List[Any], item: Any) -> List[Any]:
        safe_item = self.route4_json_safe(item)
        encoded = json.dumps(safe_item, sort_keys=True, ensure_ascii=False)
        seen = {json.dumps(self.route4_json_safe(value), sort_keys=True, ensure_ascii=False) for value in values}
        if encoded not in seen:
            values.append(safe_item)
        return values

    def route4_update_house_memory(
        self,
        output_dir: Path,
        target_house_id: str,
        facade: str,
        *,
        status: str,
        reason: str = "",
        observation_attempt: Optional[Dict[str, Any]] = None,
        nav_result: Optional[Dict[str, Any]] = None,
        frame_id: Optional[int] = None,
        scan_coverage: Optional[Dict[str, Any]] = None,
        entrance_candidates: Optional[List[Any]] = None,
        llm_decision_reason: str = "",
        obstacle_label_conflict: Optional[Dict[str, Any]] = None,
        safe_observation_pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        facade = str(facade or "").strip().lower()
        memory = self.route4_read_house_memory(output_dir, target_house_id)
        facades = memory.setdefault("facades", {})
        item = facades.setdefault(facade, self.route4_empty_house_memory(target_house_id)["facades"].get(facade, {"facade": facade}))
        previous_status = str(item.get("status", "pending") or "pending")
        item["status"] = str(status or previous_status)
        item["attempt_count"] = int(item.get("attempt_count", 0) or 0) + 1
        item["last_reason"] = str(reason or item.get("last_reason", "") or "")
        item["last_updated_at"] = datetime.now().isoformat(timespec="milliseconds")
        if observation_attempt:
            self.route4_append_unique(item.setdefault("observation_attempts", []), observation_attempt)
        if reason and str(status).lower() in {"soft_blocked", "failed_blocked"}:
            self.route4_append_unique(item.setdefault("failure_reasons", []), reason)
        if reason and str(status).lower() == "degraded_completed":
            self.route4_append_unique(item.setdefault("degraded_reasons", []), reason)
        if nav_result:
            item["last_navigation_result"] = self.route4_json_safe(nav_result)
        if frame_id is not None:
            self.route4_append_unique(item.setdefault("success_frames", []), int(frame_id))
        if scan_coverage:
            item["scan_coverage"] = self.route4_json_safe(scan_coverage)
        if entrance_candidates:
            item["entrance_candidates"] = self.route4_json_safe(entrance_candidates)
        if llm_decision_reason:
            self.route4_append_unique(item.setdefault("llm_decision_reasons", []), llm_decision_reason)
        if obstacle_label_conflict:
            self.route4_append_unique(item.setdefault("obstacle_label_conflicts", []), obstacle_label_conflict)
        if safe_observation_pose:
            item["last_safe_observation_pose"] = self.route4_json_safe(safe_observation_pose)
        memory["updated_at"] = datetime.now().isoformat(timespec="seconds")
        facades[facade] = item
        self.write_json_artifact(self.route4_house_memory_path(output_dir), memory)
        cross_path = self.route4_cross_run_house_memory_path(output_dir, target_house_id)
        cross_payload = dict(memory)
        cross_payload["history"] = {}
        self.write_json_artifact(cross_path, cross_payload)
        event = {
            "event_type": "facade_memory_update",
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "target_house_id": str(target_house_id or ""),
            "facade": facade,
            "previous_status": previous_status,
            "status": item["status"],
            "reason": reason,
        }
        self.append_jsonl(self.route4_house_memory_events_path(output_dir), event)
        agenda = {name: str(data.get("status", "pending") or "pending") for name, data in facades.items() if isinstance(data, dict)}
        self.route4_update_state(house_memory=self.route4_house_memory_summary(memory), mandatory_facade_agenda=agenda)
        self.route4_write_state_artifact()
        return memory

    def route4_set_stage(
        self,
        stage: str,
        *,
        output_dir: Optional[Path] = None,
        facade: str = "",
        target: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> None:
        self.ensure_route4_state()
        updates: Dict[str, Any] = {"stage": stage}
        if facade:
            updates["current_facade"] = facade
        if target:
            updates["target_pose"] = target
        if error:
            updates["last_error"] = error
        self.route4_update_state(**updates)
        self.route4_write_state_artifact()
        self.route4_log_event(output_dir or self.route4_state_output_dir(), "stage", {"stage": stage, "facade": facade, "message": message})
        try:
            self.root.after(0, lambda s=stage: self.llm_route4_stage_var.set(f"Stage: {s}"))
            if facade:
                self.root.after(0, lambda f=facade: self.llm_route4_active_var.set(f"Active: facade={f}"))
            if target:
                self.root.after(
                    0,
                    lambda t=target: self.llm_route4_target_var.set(
                        f"Target: x={float(t.get('x', 0.0)):.1f} y={float(t.get('y', 0.0)):.1f} z={float(t.get('z', 0.0)):.1f}"
                    ),
                )
            if error:
                self.root.after(0, lambda e=error: self.llm_route4_error_var.set(f"Error: {e.get('reason', e.get('status', 'CHECK'))}"))
            if message:
                self.root.after(0, lambda m=message: self.llm_route4_status_var.set(f"LLM Route V4: {m}"))
        except Exception:
            pass

    def route4_initialize_run(self, target_house_id: str, *, force_new: bool = False) -> Path:
        self.ensure_route4_state()
        current = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        current_target = str(current.get("target_house_id", "") or "")
        output_dir = self.route4_state_output_dir()
        created_new_run = False
        if force_new or output_dir is None or current_target != str(target_house_id):
            created_new_run = True
            output_dir = self.make_route4_fused_output_dir(target_house_id)
            self.llm_route4_completed_facades = set()
            self.llm_route4_blocked_facades = set()
            self.llm_route4_state = {
                "mode": "route4_llm_route_avoidance_fusion",
                "target_house_id": target_house_id,
                "output_dir": str(output_dir),
                "stage": "INIT_RUN",
                "nav_config": self.route4_nav_config(),
                "sensing_config": self.route4_sensing_config(),
                "last_obstacle_event": {},
                "thinking_status": "Thinking: INIT_RUN",
                "last_navigation_decision": {},
                "house_memory": {},
                "mandatory_facade_agenda": {facade: "pending" for facade in ("west", "south", "east", "north")},
                "facade_completion_status": {},
                "llm_call_seq": 0,
                "obstacle_strategy_cache": {},
                "completed_facades": [],
                "completed_facade_order": [],
                "last_completed_facade": "",
                "blocked_facades": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            for artifact_name in (
                "route4_fusion_events.jsonl",
                "route4_navigation_plan.jsonl",
                "route4_movement_trace.jsonl",
                "route4_frame_decisions.jsonl",
                "route4_llm_calls.jsonl",
                "house_exploration_memory_events.jsonl",
                "avoidance_events.jsonl",
            ):
                (output_dir / artifact_name).touch(exist_ok=True)
        self.llm_route2_state = {
            "mode": "facade_by_facade_vlm_v4_fused",
            "target_house_id": target_house_id,
            "output_dir": str(output_dir),
            "facade": "",
            "facade_id": "",
            "completed_facades": sorted(self.llm_route4_completed_facades),
        }
        self.llm_route2_completed_facades = set(self.llm_route4_completed_facades)
        self.route4_write_state_artifact()
        if created_new_run:
            self.route4_write_avoidance_summary(output_dir, status="initialized")
        return output_dir

    def route4_task_facade_priority(self) -> List[str]:
        state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        plan = state.get("task_plan", {}) if isinstance(state.get("task_plan"), dict) else {}
        raw = plan.get("facade_priority", []) if isinstance(plan, dict) else []
        result = [str(item).strip().lower() for item in raw if str(item).strip().lower() in {"south", "east", "north", "west"}]
        for facade in ("south", "east", "north", "west"):
            if facade not in result:
                result.append(facade)
        return result

    def route4_analyze_task_plan(self, target_house_id: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        task_text = self.llm_task_text_var.get().strip()
        registry = self.route3_house_registry_for_task_plan()
        parsed: Dict[str, Any] = {}
        response_payload: Dict[str, Any] = {}
        if self.effective_llm_api_key() and self.effective_llm_model():
            try:
                context = {
                    "selected_target_house_id": str(target_house_id or ""),
                    "task_text": task_text,
                    "available_facades": ["south", "east", "north", "west"],
                    "house_registry": registry,
                    "fusion_mode": "route4_llm_route_avoidance_fusion",
                }
                response_payload = self.call_configured_llm_text(
                    system_prompt=(
                        "You are the high-level task planner for a UAV house search with local obstacle avoidance. "
                        "Return strict compact JSON only. Do not output low-level movement commands."
                    ),
                    user_prompt=(
                        "Plan ordered target houses and facade priority for the fused route + obstacle-avoidance run.\n"
                        f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}\n"
                        f"Expected JSON:\n{json.dumps(LLM_ROUTE3_TASK_PLAN_SCHEMA, indent=2, ensure_ascii=False)}"
                    ),
                    max_output_tokens=700,
                    json_schema=LLM_ROUTE3_TASK_PLAN_SCHEMA,
                )
                parsed = extract_json_object(str(response_payload.get("raw_text", "") or ""))
                if output_dir is not None:
                    ref = self.route4_log_llm_call(
                        output_dir,
                        "task_plan_analysis",
                        context,
                        response_payload,
                        decision=parsed if isinstance(parsed, dict) else {},
                    )
                    response_payload["route4_llm_call_ref"] = ref
            except Exception as exc:
                response_payload = {"error": str(exc), "fallback_used": True}
        if not parsed:
            explicit_targets = self.route3_explicit_house_sequence_from_text(task_text, registry)
            fallback_target = explicit_targets[0] if explicit_targets else str(target_house_id or "")
            local_targets = explicit_targets or ([fallback_target] if fallback_target else [])
            parsed = {
                "target_house_id": str(fallback_target or ""),
                "ordered_targets": self.route3_task_target_entries_from_ids(
                    local_targets,
                    source="local_fallback_task_text" if explicit_targets else "selected_target_house",
                ),
                "major_task": task_text or "Search selected house entrance with obstacle avoidance.",
                "facade_priority": self.route3_default_facade_priority(task_text),
                "preferred_start_facade": "",
                "completion_criteria": "Reachable facades have RGB/VLM analysis, scan captures, validation, and avoidance logs.",
                "reason": "Local fallback task analysis.",
                "planner_source": "route4_local_fallback_no_or_failed_api",
            }
        plan = self.route3_normalize_task_plan(parsed, target_house_id, task_text)
        self.route4_update_state(task_plan=plan)
        if output_dir is not None:
            self.write_json_artifact(output_dir / "route4_task_plan.json", {"plan": plan, "llm_response": response_payload})
            self.route4_log_event(output_dir, "task_plan_analysis", {"task_plan": plan, "llm_response": response_payload})
            self.route4_write_state_artifact()
        return plan

    def route4_rank_observation_candidates(
        self,
        target_house_id: str,
        candidates: List[Dict[str, Any]],
        completed: set[str],
        blocked: set[str],
        *,
        start_pose: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        current = dict(start_pose or self.route3_current_pose() or self.current_route_pose() or {})
        priority = self.route4_task_facade_priority()
        priority_index = {facade: idx for idx, facade in enumerate(priority)}
        last_completed = self.route4_last_completed_facade(completed)
        base_by_facade: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            if isinstance(candidate, dict):
                facade = str(candidate.get("facade", "") or "").strip().lower()
                if facade in {"south", "east", "north", "west"} and facade not in base_by_facade:
                    base_by_facade[facade] = candidate
        ranked: List[Dict[str, Any]] = []
        for facade in ("south", "east", "north", "west"):
            if facade in completed or facade in blocked:
                continue
            base = base_by_facade.get(facade, {})
            attempts = self.route3_observation_attempts_for_facade(target_house_id, facade, base)
            ranked_attempts: List[Dict[str, Any]] = []
            for attempt in attempts:
                item = dict(attempt)
                pose = self.route3_target_pose_from_point(item)
                if not current:
                    item["route4_navigation_status"] = "unknown_no_current_pose"
                    item["route4_navigation_cost_cm"] = float(item.get("observation_selection_score", item.get("distance_to_uav_cm", 0.0)) or 0.0)
                    ranked_attempts.append(item)
                    continue
                plan = self.route3_plan_navigation_waypoints(current, pose, target_house_id, grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM))
                item["route4_navigation_plan"] = plan
                item["route4_navigation_status"] = str(plan.get("status", "blocked") or "blocked")
                item["route4_navigation_reason"] = str(plan.get("reason", "") or "")
                item["route3_navigation_status"] = item["route4_navigation_status"]
                if plan.get("status") == "ok":
                    item["route4_navigation_cost_cm"] = round(self.route3_navigation_plan_cost_cm(current, plan), 2)
                    item["status"] = "planned"
                else:
                    item["route4_navigation_cost_cm"] = float("inf")
                    item["status"] = "blocked"
                    item["observation_block_reason"] = str(plan.get("reason", item.get("observation_block_reason", "navigation_blocked")) or "navigation_blocked")
                ranked_attempts.append(item)
            ranked_attempts.sort(
                key=lambda item: (
                    0 if item.get("route4_navigation_status") == "ok" else 1,
                    float(item.get("route4_navigation_cost_cm", float("inf"))),
                    int(item.get("observation_attempt_index", 999) or 999),
                )
            )
            feasible = next((item for item in ranked_attempts if item.get("route4_navigation_status") == "ok"), None)
            selected = dict(feasible or (ranked_attempts[0] if ranked_attempts else base))
            nav_cost = float(selected.get("route4_navigation_cost_cm", selected.get("distance_to_uav_cm", 0.0)) or 0.0)
            if not math.isfinite(nav_cost):
                nav_cost = 1_000_000.0
            transition_rank = self.route4_facade_transition_rank(facade, completed, last_completed_facade=last_completed)
            selected["route4_observation_rank_score"] = round(float(nav_cost + 25.0 * float(priority_index.get(facade, len(priority))) + 2000.0 * float(transition_rank)), 2)
            selected["route4_facade_priority_index"] = priority_index.get(facade, len(priority))
            selected["route4_facade_transition_rank"] = int(transition_rank)
            selected["route4_last_completed_facade"] = last_completed
            selected["selected_observation_attempt"] = dict(feasible or selected)
            selected["observation_attempts"] = ranked_attempts
            selected["observation_attempt_count"] = len(ranked_attempts)
            if feasible is None:
                selected["status"] = "blocked"
                selected["route4_navigation_status"] = str(selected.get("route4_navigation_status", "blocked") or "blocked")
            ranked.append(selected)
        ranked.sort(
            key=lambda item: (
                0 if item.get("route4_navigation_status") == "ok" and item.get("status") != "blocked" else 1,
                float(item.get("route4_observation_rank_score", 1_000_000.0)),
                int(item.get("route4_facade_priority_index", 99) or 99),
            )
        )
        return ranked

    def route4_last_completed_facade(self, completed: set[str]) -> str:
        completed_set = {str(item).strip().lower() for item in completed if str(item).strip().lower()}
        state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        last = str(state.get("last_completed_facade", "") or "").strip().lower()
        if last in completed_set:
            return last
        order = state.get("completed_facade_order", []) if isinstance(state.get("completed_facade_order"), list) else []
        for item in reversed(order):
            facade = str(item or "").strip().lower()
            if facade in completed_set:
                return facade
        return ""

    def route4_facade_transition_rank(
        self,
        facade: str,
        completed: set[str],
        *,
        last_completed_facade: str = "",
    ) -> int:
        facade = str(facade or "").strip().lower()
        completed_set = {str(item).strip().lower() for item in completed if str(item).strip().lower()}
        last = str(last_completed_facade or "").strip().lower()
        if not last:
            last = self.route4_last_completed_facade(completed_set)
        if facade in completed_set:
            return 99
        if last == "north":
            order = ["east", "south", "west"]
        elif last == "south":
            order = ["east", "north", "west"]
        elif last == "west":
            order = ["north", "south", "east"]
        elif last == "east":
            order = ["south", "north", "west"]
        else:
            order = self.route4_task_facade_priority()
        if facade in order:
            return order.index(facade)
        return len(order)

    def route4_nav_result_current_pose(self, nav_result: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(nav_result, dict):
            return {}
        candidates = [
            nav_result.get("current_pose", {}),
            nav_result.get("post_action_pose", {}),
        ]
        last_result = nav_result.get("last_result", {}) if isinstance(nav_result.get("last_result"), dict) else {}
        candidates.extend([last_result.get("current_pose", {}), last_result.get("post_action_pose", {})])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            x = self._as_float_or_none(candidate.get("x"))
            y = self._as_float_or_none(candidate.get("y"))
            if x is None or y is None:
                continue
            z = self._as_float_or_none(candidate.get("z"))
            yaw = self._as_float_or_none(candidate.get("yaw", candidate.get("task_yaw", candidate.get("yaw_deg"))))
            return {
                "x": float(x),
                "y": float(y),
                "z": float(z if z is not None else LLM_ROUTE2_OBSERVATION_Z_CM),
                "yaw": float(yaw if yaw is not None else 0.0),
            }
        return {}

    def route4_observation_rescue_candidate(
        self,
        *,
        target_house_id: str,
        facade: str,
        original_observation: Dict[str, Any],
        nav_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(nav_result, dict) or str(nav_result.get("status", "") or "") == "ok":
            return {}
        reason = str(nav_result.get("reason", "") or "").strip()
        if reason not in {"non_target_house_clearance", "target_house_bbox", "replan_exhausted", "movement_failed", "navigation_plan_failed", "nav_timeout"}:
            return {}
        hid = str(target_house_id or "").strip()
        facade = str(facade or "").strip().lower()
        bbox = self.house_world_bbox_for_id(hid)
        current = self.route4_nav_result_current_pose(nav_result)
        if not hid or facade not in {"south", "east", "north", "west"} or not bbox or not current:
            return {}
        x = float(current["x"])
        y = float(current["y"])
        z = float(current.get("z", LLM_ROUTE2_OBSERVATION_Z_CM))
        if self.point_inside_open_bbox(x, y, bbox):
            return {}
        axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
        axis_low = min(float(axis_min), float(axis_max))
        axis_high = max(float(axis_min), float(axis_max))
        axis_center = self.route2_facade_center_axis(bbox, facade)
        facade_length = max(1.0, axis_high - axis_low)
        axis_margin = min(450.0, 0.35 * facade_length)
        min_x = float(bbox["min_x"])
        max_x = float(bbox["max_x"])
        min_y = float(bbox["min_y"])
        max_y = float(bbox["max_y"])
        if facade == "north":
            if y <= max_y:
                return {}
            axis_value = x
            standoff = y - max_y
            target_x = max(axis_low, min(axis_high, axis_value))
            target_y = max_y
        elif facade == "south":
            if y >= min_y:
                return {}
            axis_value = x
            standoff = min_y - y
            target_x = max(axis_low, min(axis_high, axis_value))
            target_y = min_y
        elif facade == "east":
            if x <= max_x:
                return {}
            axis_value = y
            standoff = x - max_x
            target_x = max_x
            target_y = max(axis_low, min(axis_high, axis_value))
        else:
            if x >= min_x:
                return {}
            axis_value = y
            standoff = min_x - x
            target_x = min_x
            target_y = max(axis_low, min(axis_high, axis_value))
        if axis_value < axis_low - axis_margin or axis_value > axis_high + axis_margin:
            return {}
        min_standoff = 0.5 * float(LLM_ROUTE2_OBSERVATION_MIN_STANDOFF_CM)
        max_standoff = float(LLM_ROUTE2_OBSERVATION_MAX_STANDOFF_CM)
        if not (min_standoff <= float(standoff) <= max_standoff):
            return {}
        bounds_report = self.route2_observation_map_bounds_report(x, y)
        if not bool(bounds_report.get("in_bounds", True)):
            return {}
        blocking = self.route2_observation_blocking_house(hid, x, y)
        yaw = math.degrees(math.atan2(float(target_y) - y, float(target_x) - x))
        required_standoff = self._as_float_or_none(original_observation.get("observation_required_standoff_cm"))
        original_standoff = self._as_float_or_none(original_observation.get("standoff_cm"))
        return {
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(float(z), 2),
            "yaw_deg": round(float(yaw), 2),
            "target_x": round(float(target_x), 2),
            "target_y": round(float(target_y), 2),
            "label": f"{hid}_{facade}_obs_rescue",
            "route_point_type": "observation_point",
            "house_id": hid,
            "facade": facade,
            "facade_id": self.route2_facade_id(hid, facade),
            "axis_value": round(float(max(axis_low, min(axis_high, axis_value))), 2),
            "axis_center_cm": round(float(axis_center), 2),
            "axis_center_error_cm": round(abs(float(axis_value) - float(axis_center)), 2),
            "standoff_cm": round(float(standoff), 2),
            "observation_required_standoff_cm": round(float(required_standoff if required_standoff is not None else LLM_ROUTE2_OBSERVATION_MIN_STANDOFF_CM), 2),
            "observation_actual_standoff_cm": round(float(standoff), 2),
            "observation_attempt_index": int(original_observation.get("observation_attempt_index", 0) or 0),
            "observation_attempt_source": "route4_rescue_current_pose",
            "observation_map_bounds": bounds_report,
            "observation_blocking_house_id": str(blocking.get("house_id", "") or ""),
            "observation_block_reason": "route4_rescue_allowed_non_target_clearance" if blocking else "",
            "observation_boundary_adjustment": {},
            "status": "planned",
            "route4_observation_rescue": True,
            "route4_rescue_capture_without_additional_navigation": True,
            "route4_rescue_reason": "navigation_blocked_near_facade_observe_from_current_pose",
            "route4_rescue_navigation_reason": reason,
            "route4_rescue_target_id": str(nav_result.get("target_id", "") or ""),
            "route4_rescue_original_observation": {
                "x": original_observation.get("x"),
                "y": original_observation.get("y"),
                "z": original_observation.get("z"),
                "yaw_deg": original_observation.get("yaw_deg", original_observation.get("yaw")),
                "standoff_cm": original_standoff,
                "target_x": original_observation.get("target_x"),
                "target_y": original_observation.get("target_y"),
            },
            "route4_rescue_current_pose": current,
        }

    def route4_observation_rescue_decision(
        self,
        *,
        target_house_id: str,
        facade: str,
        original_observation: Dict[str, Any],
        candidate: Dict[str, Any],
        nav_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        fallback = {
            "next_action": "capture_current_observation",
            "accepted": bool(candidate),
            "reason": "Deterministic rescue: current pose is on the requested facade side and inside map bounds.",
            "planner_source": "route4_rescue_rule_fallback",
        }
        if not candidate or not self.effective_llm_api_key() or not self.effective_llm_model():
            return fallback
        try:
            context = {
                "target_house_id": target_house_id,
                "facade": facade,
                "original_observation": original_observation,
                "rescue_candidate": candidate,
                "navigation_result": {
                    "status": nav_result.get("status"),
                    "reason": nav_result.get("reason"),
                    "target_id": nav_result.get("target_id"),
                    "current_pose": nav_result.get("current_pose"),
                    "pose_error": nav_result.get("pose_error"),
                    "safety": nav_result.get("safety"),
                },
                "instruction": "If the UAV is near the facade and blocked on the way to the planned observation point, decide whether to capture from the current pose or keep searching for another observation point.",
            }
            response = self.call_configured_llm_text(
                system_prompt=(
                    "You are a UAV observation rescue planner. Return compact JSON only. "
                    "Prefer capture_current_observation when the current pose can view the requested facade."
                ),
                user_prompt=(
                    "Choose a rescue action for a blocked NAV_TO_OBS segment. "
                    "Allowed next_action values: capture_current_observation, try_next_observation, mark_blocked.\n"
                    f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
                ),
                max_output_tokens=350,
                json_schema={
                    "next_action": "capture_current_observation",
                    "accepted": True,
                    "reason": "",
                },
            )
            parsed = extract_json_object(str(response.get("raw_text", "") or ""))
            output_dir = self.route4_state_output_dir()
            llm_ref = self.route4_log_llm_call(
                output_dir,
                "observation_rescue_decision",
                context,
                response,
                facade=facade,
                target_id=str(nav_result.get("target_id", "") or ""),
                decision=parsed if isinstance(parsed, dict) else {},
            )
            action = str(parsed.get("next_action", "") or "").strip().lower()
            if action in {"capture_current_observation", "try_next_observation", "mark_blocked"}:
                return {
                    **fallback,
                    **parsed,
                    "accepted": action == "capture_current_observation" and bool(candidate),
                    "planner_source": "route4_rescue_llm",
                    "llm_response": response,
                    "llm_call_ref": llm_ref,
                }
        except Exception as exc:
            return {**fallback, "llm_error": str(exc)}
        return fallback

    def route4_try_observation_rescue(
        self,
        *,
        output_dir: Path,
        target_house_id: str,
        facade: str,
        original_observation: Dict[str, Any],
        ranked_candidates: List[Dict[str, Any]],
        nav_result: Dict[str, Any],
        attempt_index: int,
    ) -> Dict[str, Any]:
        candidate = self.route4_observation_rescue_candidate(
            target_house_id=target_house_id,
            facade=facade,
            original_observation=original_observation,
            nav_result=nav_result,
        )
        decision = self.route4_observation_rescue_decision(
            target_house_id=target_house_id,
            facade=facade,
            original_observation=original_observation,
            candidate=candidate,
            nav_result=nav_result,
        )
        rescue_payload = {
            "facade": facade,
            "attempt_index": attempt_index,
            "candidate": candidate,
            "decision": decision,
            "navigation": nav_result,
        }
        self.route4_log_event(output_dir, "observation_rescue_candidate", rescue_payload)
        if not candidate or not bool(decision.get("accepted", False)):
            return {"status": "skipped", "reason": str(decision.get("next_action", "no_rescue_candidate") or "no_rescue_candidate"), **rescue_payload}
        self.apply_route2_observation_plan(
            target_house_id,
            candidate,
            ranked_candidates,
            status_label=f"v4 rescue observation {facade}",
        )
        result = {
            "status": "ok",
            "reason": "route4_observation_rescue_capture_current",
            "observation": candidate,
            "target_pose": self.route3_target_pose_from_point(candidate),
            "decision": decision,
            "source_navigation": nav_result,
        }
        self.route4_update_state(last_observation_rescue=result)
        self.route4_write_state_artifact()
        self.route4_log_event(output_dir, "observation_rescue_applied", result)
        return result

    def route4_degraded_observation_candidate(
        self,
        *,
        target_house_id: str,
        facade: str,
        original_observation: Dict[str, Any],
        nav_result: Dict[str, Any],
        fallback_pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hid = str(target_house_id or "").strip()
        facade = str(facade or "").strip().lower()
        bbox = self.house_world_bbox_for_id(hid)
        current = self.route4_nav_result_current_pose(nav_result) or (dict(fallback_pose) if isinstance(fallback_pose, dict) else {})
        if not hid or facade not in {"south", "east", "north", "west"} or not bbox or not current:
            return {}
        x = self._as_float_or_none(current.get("x"))
        y = self._as_float_or_none(current.get("y"))
        z = self._as_float_or_none(current.get("z"))
        if x is None or y is None:
            return {}
        x = float(x)
        y = float(y)
        z = float(z if z is not None else LLM_ROUTE2_OBSERVATION_Z_CM)
        if self.point_inside_open_bbox(x, y, bbox):
            return {}
        bounds_report = self.route2_observation_map_bounds_report(x, y)
        if not bool(bounds_report.get("in_bounds", True)):
            return {}
        axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
        axis_low = min(float(axis_min), float(axis_max))
        axis_high = max(float(axis_min), float(axis_max))
        axis_center = self.route2_facade_center_axis(bbox, facade)
        if facade == "north":
            if y <= float(bbox["max_y"]):
                return {}
            axis_value = x
            standoff = y - float(bbox["max_y"])
            target_x = max(axis_low, min(axis_high, axis_value))
            target_y = float(bbox["max_y"])
        elif facade == "south":
            if y >= float(bbox["min_y"]):
                return {}
            axis_value = x
            standoff = float(bbox["min_y"]) - y
            target_x = max(axis_low, min(axis_high, axis_value))
            target_y = float(bbox["min_y"])
        elif facade == "east":
            if x <= float(bbox["max_x"]):
                return {}
            axis_value = y
            standoff = x - float(bbox["max_x"])
            target_x = float(bbox["max_x"])
            target_y = max(axis_low, min(axis_high, axis_value))
        else:
            if x >= float(bbox["min_x"]):
                return {}
            axis_value = y
            standoff = float(bbox["min_x"]) - x
            target_x = float(bbox["min_x"])
            target_y = max(axis_low, min(axis_high, axis_value))
        yaw = math.degrees(math.atan2(float(target_y) - y, float(target_x) - x))
        blocking = self.route2_observation_blocking_house(hid, x, y)
        required_standoff = self._as_float_or_none(original_observation.get("observation_required_standoff_cm"))
        return {
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "yaw_deg": round(float(yaw), 2),
            "target_x": round(float(target_x), 2),
            "target_y": round(float(target_y), 2),
            "label": f"{hid}_{facade}_obs_degraded",
            "route_point_type": "observation_point",
            "house_id": hid,
            "facade": facade,
            "facade_id": self.route2_facade_id(hid, facade),
            "axis_value": round(float(max(axis_low, min(axis_high, axis_value))), 2),
            "axis_center_cm": round(float(axis_center), 2),
            "axis_center_error_cm": round(abs(float(axis_value) - float(axis_center)), 2),
            "standoff_cm": round(float(standoff), 2),
            "observation_required_standoff_cm": round(float(required_standoff if required_standoff is not None else LLM_ROUTE2_OBSERVATION_MIN_STANDOFF_CM), 2),
            "observation_actual_standoff_cm": round(float(standoff), 2),
            "observation_attempt_index": int(original_observation.get("observation_attempt_index", 0) or 0),
            "observation_attempt_source": "route4_degraded_current_pose",
            "observation_map_bounds": bounds_report,
            "observation_blocking_house_id": str(blocking.get("house_id", "") or ""),
            "observation_block_reason": "degraded_current_pose_with_non_target_clearance" if blocking else "",
            "status": "planned",
            "route4_degraded_observation": True,
            "route4_completion_status": "degraded_completed",
            "route4_degraded_reason": str(nav_result.get("reason", "navigation_failed") or "navigation_failed") if isinstance(nav_result, dict) else "navigation_failed",
            "route4_degraded_navigation": self.route4_json_safe(nav_result if isinstance(nav_result, dict) else {}),
            "route4_degraded_original_observation": self.route4_json_safe(original_observation if isinstance(original_observation, dict) else {}),
            "route4_degraded_current_pose": self.route4_json_safe(current),
        }

    def route4_mark_facade_degraded_completed(
        self,
        output_dir: Path,
        target_house_id: str,
        facade: str,
        *,
        observation: Dict[str, Any],
        reason: str,
        nav_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        facade = str(facade or "").strip().lower()
        self.llm_route4_completed_facades.add(facade)
        self.llm_route4_blocked_facades.discard(facade)
        completion_status = dict(self.llm_route4_state.get("facade_completion_status", {}) if isinstance(self.llm_route4_state.get("facade_completion_status"), dict) else {})
        completion_status[facade] = "degraded_completed"
        completed_order = [
            str(item).strip().lower()
            for item in (
                self.llm_route4_state.get("completed_facade_order", [])
                if isinstance(self.llm_route4_state.get("completed_facade_order"), list)
                else []
            )
            if str(item).strip().lower()
        ]
        if facade not in completed_order:
            completed_order.append(facade)
        result = {
            "status": "degraded_completed",
            "facade": facade,
            "target_house_id": str(target_house_id or ""),
            "reason": str(reason or "degraded_observation"),
            "observation": self.route4_json_safe(observation),
            "navigation": self.route4_json_safe(nav_result),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.route4_update_state(
            completed_facades=sorted(self.llm_route4_completed_facades),
            blocked_facades=sorted(self.llm_route4_blocked_facades),
            completed_facade_order=completed_order,
            last_completed_facade=facade,
            facade_completion_status=completion_status,
            last_degraded_facade_completion=result,
        )
        self.route4_update_house_memory(
            output_dir,
            target_house_id,
            facade,
            status="degraded_completed",
            reason=str(reason or "degraded_observation"),
            observation_attempt=observation,
            nav_result=nav_result,
            obstacle_label_conflict=(
                self.llm_route4_state.get("last_obstacle_event", {}).get("obstacle_label_conflict")
                if isinstance(self.llm_route4_state.get("last_obstacle_event"), dict)
                else None
            ),
            safe_observation_pose=observation,
        )
        self.route4_log_event(output_dir, "facade_degraded_completed", result)
        self.route4_write_state_artifact()
        return result

    def route4_decide_next_facade(
        self,
        target_house_id: str,
        candidates: List[Dict[str, Any]],
        completed: set[str],
        blocked: set[str],
    ) -> Dict[str, Any]:
        available = [
            candidate for candidate in candidates
            if isinstance(candidate, dict)
            and str(candidate.get("facade", "") or "") not in completed
            and str(candidate.get("facade", "") or "") not in set(self.llm_route4_state.get("terminal_failed_facades", []) if isinstance(self.llm_route4_state.get("terminal_failed_facades"), list) else [])
        ]
        fallback = {
            "next_action": "select_facade" if available else "done",
            "target_facade": str(available[0].get("facade", "") or "") if available else "",
            "reason": "fallback nearest feasible uncompleted facade",
            "rescan_required": False,
            "stop_condition_met": not bool(available),
            "planner_source": "route4_rule_fallback",
            "ranked_candidates_considered": len(candidates),
        }
        if not available or not self.effective_llm_api_key():
            return fallback
        try:
            context = {
                "target_house_id": target_house_id,
                "completed_facades": sorted(completed),
                "blocked_facades": sorted(blocked),
                "candidate_observation_points": available,
                "current_pose": self.route3_current_pose(),
                "task": self.llm_task_text_var.get().strip(),
                "fusion_mode": "route4_llm_route_avoidance_fusion",
            }
            response = self.call_configured_llm_text(
                system_prompt=(
                    "You are a high-level UAV house facade search supervisor for a fused route and obstacle-avoidance run. "
                    "Choose the next facade only; do not output low-level movement commands. Return compact JSON."
                ),
                user_prompt=(
                    "Choose the next safe facade to search. Return JSON with keys "
                    "next_action(select_facade|done), target_facade, reason, rescan_required, stop_condition_met.\n"
                    f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
                ),
                max_output_tokens=400,
                json_schema={
                    "next_action": "select_facade",
                    "target_facade": "west",
                    "reason": "",
                    "rescan_required": False,
                    "stop_condition_met": False,
                },
            )
            parsed = extract_json_object(str(response.get("raw_text", "") or ""))
            output_dir = self.route4_state_output_dir()
            llm_ref = self.route4_log_llm_call(
                output_dir,
                "high_level_facade_decision",
                context,
                response,
                decision=parsed if isinstance(parsed, dict) else {},
            )
            chosen = str(parsed.get("target_facade", "") or "").strip().lower()
            valid_facades = {str(item.get("facade", "") or "") for item in available}
            if chosen in valid_facades:
                nearest = str(available[0].get("facade", "") or "") if available else chosen
                if chosen != nearest:
                    parsed["llm_requested_facade"] = chosen
                    parsed["target_facade"] = nearest
                    parsed["rank_correction_reason"] = "nearest feasible observation point has priority over LLM facade preference"
                    parsed["planner_source"] = "route4_llm_high_level_rank_corrected"
                else:
                    parsed["planner_source"] = "route4_llm_high_level"
                parsed["llm_call_ref"] = llm_ref
                return parsed
        except Exception as exc:
            return {**fallback, "llm_error": str(exc)}
        return fallback

    def route4_set_control_lock(self, locked: bool) -> None:
        self.ensure_route4_state()
        self.llm_route4_control_locked = bool(locked)
        try:
            if locked:
                self.root.after(0, lambda: self.stop_keyboard_control(send_hold=False))
                self.root.after(0, lambda: self.update_keyboard_status("locked by LLM Route V4"))
            else:
                self.root.after(0, self.update_keyboard_status)
        except Exception:
            pass

    def route4_enable_physics_movement(self, session: flight.DroneFlightSession) -> None:
        self.route4_set_control_lock(True)
        result = self.safe("Route V4 movement mode physics", lambda: session.set_movement_mode("physics"))
        self.safe("Route V4 enable movement", lambda: session.set_movement_enabled(True))
        self.movement_mode_state = "physics"
        try:
            self.root.after(0, lambda: self.movement_mode_var.set("physics"))
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_state(r))
        except Exception:
            pass

    def route4_hold(self, session: flight.DroneFlightSession, *, output_dir: Optional[Path] = None, reason: str = "hold") -> Dict[str, Any]:
        payload = action_payload("hold")
        result = self.safe("Route V4 hold", lambda: session.move_relative(payload))
        self.route4_log_event(output_dir, "hold", {"reason": reason, "payload": payload})
        return result if isinstance(result, dict) else {}

    def route4_wait_if_paused(self, session: flight.DroneFlightSession, output_dir: Optional[Path]) -> bool:
        self.ensure_route4_state()
        held = False
        while self.llm_route4_pause_event.is_set() and not self.llm_route4_stop_event.is_set():
            if not held:
                self.route4_hold(session, output_dir=output_dir, reason="paused")
                held = True
            self.root.after(0, lambda: self.llm_route4_status_var.set("LLM Route V4: paused."))
            time.sleep(0.2)
        return bool(self.llm_route4_stop_event.is_set())

    def route4_semantic_hint_from_prediction(self, prediction: Dict[str, Any]) -> str:
        label = str((prediction if isinstance(prediction, dict) else {}).get("predicted_label", "") or "").strip().lower()
        mapping = {
            "open_path": "unknown",
            "none": "unknown",
            "unknown": "unknown",
            "tree_trunk_or_pole": "tree_trunk_or_pole",
            "tree_canopy_or_cluster": "tree_canopy_or_cluster",
            "fence_or_rail": "fence_or_rail",
            "building": "building",
            "mixed": "mixed",
        }
        return mapping.get(label, "unknown")

    def route4_event_float(self, value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if math.isfinite(number) else float(default)
        except Exception:
            return float(default)

    def route4_avoidance_trigger_distance_cm(self, semantic_hint: str) -> float:
        hint = str(semantic_hint or "unknown").strip().lower()
        if hint == "building":
            return 500.0
        if hint in {"tree_trunk_or_pole", "tree_canopy_or_cluster", "fence_or_rail", "mixed", "unknown"}:
            return 120.0
        return 120.0

    def route4_avoidance_gate_semantic_hint(self, event: Dict[str, Any], strategy: Dict[str, Any]) -> Tuple[str, str]:
        prediction = event.get("representation_prediction", {}) if isinstance(event.get("representation_prediction"), dict) else {}
        representation_hint = self.route4_semantic_hint_from_prediction(prediction)
        if representation_hint != "unknown":
            return representation_hint, "representation_prediction"
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        geometry = str(summary.get("obstacle_geometry", "") or "").strip().lower()
        if geometry in {"vertical_wall", "overhang_beam", "building", "wall", "facade"}:
            return "building", "pointcloud_geometry"
        if geometry in {"low_obstacle", "thin_vertical", "fence", "rail"}:
            return "fence_or_rail", "pointcloud_geometry"
        strategy_hint = self.route4_semantic_hint_from_prediction({"predicted_label": str(strategy.get("obstacle_hint", "") or "")})
        if strategy_hint != "unknown":
            return strategy_hint, "llm_strategy"
        return "unknown", "default"

    def route4_should_apply_avoidance(self, event: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        semantic_hint, semantic_source = self.route4_avoidance_gate_semantic_hint(event, strategy if isinstance(strategy, dict) else {})
        trigger_distance_cm = self.route4_avoidance_trigger_distance_cm(semantic_hint)
        front_min_depth_cm = self.route4_event_float(summary.get("front_min_depth_cm"), default=0.0)
        forward_swept_clear = bool(summary.get("forward_swept_clear", True))
        pointcloud_available = bool(summary.get("available", bool(summary)))
        geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
        has_front_depth = front_min_depth_cm > 0.0
        distance_triggered = has_front_depth and front_min_depth_cm <= trigger_distance_cm
        blocked_without_depth = (not forward_swept_clear) and not has_front_depth
        avoidance_active = bool(pointcloud_available and (distance_triggered or blocked_without_depth))
        if not pointcloud_available:
            reason = "pointcloud_unavailable"
        elif distance_triggered:
            reason = f"front_depth_within_{trigger_distance_cm:.0f}cm"
        elif blocked_without_depth:
            reason = "forward_swept_blocked_without_depth"
        elif not forward_swept_clear:
            reason = f"forward_swept_blocked_but_front_depth_{front_min_depth_cm:.1f}cm_exceeds_{trigger_distance_cm:.0f}cm"
        elif has_front_depth:
            reason = f"front_clear_depth_{front_min_depth_cm:.1f}cm_exceeds_{trigger_distance_cm:.0f}cm"
        else:
            reason = "no_front_obstacle_depth"
        return {
            "avoidance_active": avoidance_active,
            "semantic_hint": semantic_hint,
            "semantic_source": semantic_source,
            "trigger_distance_cm": float(trigger_distance_cm),
            "front_min_depth_cm": float(front_min_depth_cm),
            "forward_swept_clear": forward_swept_clear,
            "pointcloud_available": pointcloud_available,
            "obstacle_geometry": geometry,
            "reason": reason,
        }

    def route4_depth_lookahead_stages(self) -> set:
        return {"NAV_TO_OBS", "NAV_TO_SCAN_POINT"}

    def route4_payload_blocked_directions_from_depth(
        self,
        summary: Dict[str, Any],
        nominal_payload: Dict[str, Any],
    ) -> List[str]:
        if not isinstance(summary, dict) or not isinstance(nominal_payload, dict):
            return []
        deadband_cm = 0.5
        blocked: List[str] = []
        forward_cm = self.route4_event_float(nominal_payload.get("forward_cm"), default=0.0)
        right_cm = self.route4_event_float(nominal_payload.get("right_cm"), default=0.0)
        up_cm = self.route4_event_float(nominal_payload.get("up_cm"), default=0.0)
        if forward_cm > deadband_cm and summary.get("forward_swept_clear") is False:
            blocked.append("forward")
        if right_cm > deadband_cm and summary.get("right_swept_clear") is False:
            blocked.append("right")
        if right_cm < -deadband_cm and summary.get("left_swept_clear") is False:
            blocked.append("left")
        if up_cm > deadband_cm and summary.get("up_swept_clear") is False:
            blocked.append("up")
        if up_cm < -deadband_cm and summary.get("down_swept_clear") is False:
            blocked.append("down")
        return blocked

    def route4_should_precheck_depth_before_payload(self, stage: str, nominal_payload: Dict[str, Any]) -> bool:
        stage_name = str(stage or "").strip().upper()
        if stage_name not in self.route4_depth_lookahead_stages() or not isinstance(nominal_payload, dict):
            return False
        deadband_cm = 0.5
        return any(
            abs(self.route4_event_float(nominal_payload.get(key), default=0.0)) > deadband_cm
            for key in ("forward_cm", "right_cm", "up_cm")
        )

    def route4_navigation_travel_yaw_deg(self, current: Dict[str, Any], target: Dict[str, Any]) -> float:
        dx = self.route4_event_float((target if isinstance(target, dict) else {}).get("x"), default=0.0) - self.route4_event_float((current if isinstance(current, dict) else {}).get("x"), default=0.0)
        dy = self.route4_event_float((target if isinstance(target, dict) else {}).get("y"), default=0.0) - self.route4_event_float((current if isinstance(current, dict) else {}).get("y"), default=0.0)
        if math.hypot(dx, dy) <= 1e-6:
            return self._normalize_angle_deg(self.route4_event_float((current if isinstance(current, dict) else {}).get("yaw"), default=0.0))
        return self._normalize_angle_deg(math.degrees(math.atan2(dy, dx)))

    def route4_depth_lookahead_plan(
        self,
        current: Dict[str, Any],
        target: Dict[str, Any],
        *,
        stage: str,
        facade: str,
        target_id: str,
    ) -> Dict[str, Any]:
        current_dict = current if isinstance(current, dict) else {}
        target_dict = target if isinstance(target, dict) else {}
        stage_name = str(stage or "").strip().upper()
        current_yaw = self.route4_event_float(current_dict.get("yaw"), default=0.0)
        capture_yaw = self.route4_event_float(target_dict.get("yaw", target_dict.get("yaw_deg")), default=current_yaw)
        dx = self.route4_event_float(target_dict.get("x"), default=0.0) - self.route4_event_float(current_dict.get("x"), default=0.0)
        dy = self.route4_event_float(target_dict.get("y"), default=0.0) - self.route4_event_float(current_dict.get("y"), default=0.0)
        horizontal_distance_cm = float(math.hypot(dx, dy))
        enabled = bool(stage_name in self.route4_depth_lookahead_stages() and horizontal_distance_cm > 1.0)
        look_yaw = self.route4_navigation_travel_yaw_deg(current_dict, target_dict) if enabled else current_yaw
        yaw_error = self._normalize_angle_deg(float(look_yaw) - float(current_yaw))
        status = "ALIGN_LOOK_YAW" if enabled and abs(yaw_error) > 1.0 else ("SENSE_LOOK_YAW" if enabled else "NO_LOOKAHEAD")
        thinking = (
            f"Thinking: {stage_name or 'NAV'} {status} look waypoint {target_id} "
            f"yaw={float(look_yaw):.1f} current={float(current_yaw):.1f} "
            f"capture={float(capture_yaw):.1f} dist={horizontal_distance_cm:.1f}cm"
        )
        return {
            "enabled": enabled,
            "stage": stage_name,
            "facade": facade,
            "target_id": target_id,
            "look_direction": "waypoint_bearing" if enabled else "current_yaw",
            "look_yaw_deg": round(float(look_yaw), 3),
            "current_yaw_deg": round(float(current_yaw), 3),
            "capture_yaw_deg": round(float(capture_yaw), 3),
            "yaw_error_deg": round(float(yaw_error), 3),
            "horizontal_distance_cm": round(horizontal_distance_cm, 3),
            "segment_start": {
                "x": self.route4_event_float(current_dict.get("x"), default=0.0),
                "y": self.route4_event_float(current_dict.get("y"), default=0.0),
                "z": self.route4_event_float(current_dict.get("z"), default=0.0),
            },
            "segment_goal": {
                "x": self.route4_event_float(target_dict.get("x"), default=0.0),
                "y": self.route4_event_float(target_dict.get("y"), default=0.0),
                "z": self.route4_event_float(target_dict.get("z"), default=0.0),
            },
            "decision_state": status,
            "thinking_status": thinking,
            "policy": "face_waypoint_bearing_for_depth_then_capture_house_yaw",
        }

    def route4_set_thinking_status(
        self,
        decision: Any,
        *,
        output_dir: Optional[Path] = None,
        write_artifact: bool = False,
    ) -> None:
        self.ensure_route4_state()
        if isinstance(decision, dict):
            payload = dict(decision)
            text = str(payload.get("thinking_status", "") or payload.get("status", "") or "Thinking: active")
        else:
            payload = {"thinking_status": str(decision)}
            text = str(decision)
        self.route4_update_state(thinking_status=text, last_navigation_decision=payload)
        if write_artifact:
            self.route4_write_state_artifact()
        if output_dir is not None:
            self.route4_log_event(output_dir, "thinking_status", payload)
        try:
            self.root.after(0, lambda t=text: self.llm_route4_thinking_status_var.set(t))
        except Exception:
            try:
                self.llm_route4_thinking_status_var.set(text)
            except Exception:
                pass

    def route4_movement_payload_for_target_with_lookahead(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        config: Dict[str, float],
        *,
        stage: str,
    ) -> Dict[str, Any]:
        payload = self.route3_movement_payload_for_target(current, target, config)
        stage_name = str(stage or "").strip().upper()
        if stage_name not in self.route4_depth_lookahead_stages():
            return payload
        dx = float(target["x"]) - float(current["x"])
        dy = float(target["y"]) - float(current["y"])
        dist_xy = float(math.hypot(dx, dy))
        if dist_xy <= float(config["reach_tol_cm"]):
            return payload
        look_yaw = self.route4_navigation_travel_yaw_deg(current, target)
        yaw_error = self._normalize_angle_deg(float(look_yaw) - float(current.get("yaw", 0.0)))
        yaw_delta = 0.0 if abs(yaw_error) <= float(config["yaw_tol_deg"]) else max(-30.0, min(30.0, float(yaw_error)))
        payload = dict(payload)
        payload["yaw_delta_deg"] = round(float(yaw_delta), 3)
        payload["action_name"] = "route4_nav_lookahead" if any(abs(float(payload.get(key, 0.0) or 0.0)) > 1e-6 for key in ("forward_cm", "right_cm", "up_cm", "yaw_delta_deg")) else "hold"
        payload["yaw_policy"] = "waypoint_bearing_until_capture"
        payload["look_yaw_deg"] = round(float(look_yaw), 3)
        payload["capture_yaw_deg"] = round(float(target.get("yaw", target.get("yaw_deg", 0.0)) or 0.0), 3)
        return payload

    def route4_align_to_navigation_depth_yaw(
        self,
        session: flight.DroneFlightSession,
        current: Dict[str, float],
        target_pose: Dict[str, float],
        *,
        output_dir: Path,
        stage: str,
        facade: str,
        target_id: str,
        config: Dict[str, float],
    ) -> Dict[str, Any]:
        plan = self.route4_depth_lookahead_plan(current, target_pose, stage=stage, facade=facade, target_id=target_id)
        if not bool(plan.get("enabled", False)):
            self.route4_set_thinking_status(plan, output_dir=output_dir)
            return {"status": "skipped", "reason": "lookahead_not_required", "current_pose": current, "lookahead_plan": plan}
        attempts = 0
        max_attempts = max(1, min(8, int(math.ceil(abs(float(plan.get("yaw_error_deg", 0.0))) / 30.0)) + 2))
        updated_current = dict(current)
        last_payload = action_payload("hold")
        while (
            attempts < max_attempts
            and abs(float(plan.get("yaw_error_deg", 0.0))) > float(config["yaw_tol_deg"])
            and not self.llm_route4_stop_event.is_set()
        ):
            yaw_delta = max(-30.0, min(30.0, float(plan.get("yaw_error_deg", 0.0))))
            yaw_payload = {
                "forward_cm": 0.0,
                "right_cm": 0.0,
                "up_cm": 0.0,
                "yaw_delta_deg": round(float(yaw_delta), 3),
                "action_name": "route4_depth_lookahead_yaw",
                "yaw_policy": "align_to_waypoint_bearing_before_depth",
                "look_yaw_deg": plan.get("look_yaw_deg"),
                "capture_yaw_deg": plan.get("capture_yaw_deg"),
            }
            tick_plan = dict(plan, decision_state="ALIGN_LOOK_YAW", thinking_status=f"{plan['thinking_status']} attempt={attempts + 1}/{max_attempts}")
            self.route4_set_thinking_status(tick_plan, output_dir=output_dir)
            self.route4_log_event(
                output_dir,
                "depth_lookahead_yaw_align",
                {
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "attempt": attempts + 1,
                    "payload": yaw_payload,
                    "lookahead_plan": plan,
                },
            )
            result = self.safe("Route V4 depth lookahead yaw", lambda p=yaw_payload: session.move_relative(p))
            if not isinstance(result, dict):
                return {"status": "failed", "reason": "lookahead_yaw_failed", "current_pose": updated_current, "lookahead_plan": plan}
            try:
                self.root.after(0, lambda r=result: self.apply_state(r))
            except Exception:
                pass
            updated_current = self.route3_pose_from_payload(result) or self.route3_current_pose(session) or self.route3_predict_next_pose(updated_current, yaw_payload)
            last_payload = yaw_payload
            attempts += 1
            plan = self.route4_depth_lookahead_plan(updated_current, target_pose, stage=stage, facade=facade, target_id=target_id)
        done_plan = dict(plan, decision_state="SENSE_LOOK_YAW", thinking_status=f"{plan['thinking_status']} -> sensing depth")
        self.route4_set_thinking_status(done_plan, output_dir=output_dir)
        return {
            "status": "ok",
            "reason": "lookahead_yaw_ready",
            "current_pose": updated_current,
            "lookahead_plan": done_plan,
            "attempts": attempts,
            "last_action": last_payload,
        }

    def route4_depth_lookahead_semantic_hint(self, summary: Dict[str, Any], gate: Dict[str, Any]) -> str:
        geometry = str(summary.get("obstacle_geometry", "") or "").strip().lower() if isinstance(summary, dict) else ""
        if geometry in {"vertical_wall", "overhang_beam", "building", "wall", "facade"}:
            return "building"
        if geometry in {"low_obstacle", "thin_vertical", "fence", "rail"}:
            return "fence_or_rail"
        hint = str(gate.get("semantic_hint", "unknown") or "unknown").strip().lower() if isinstance(gate, dict) else "unknown"
        return hint if hint else "unknown"

    def route4_depth_lookahead_gate(
        self,
        event: Dict[str, Any],
        gate: Dict[str, Any],
        nominal_payload: Dict[str, Any],
        *,
        stage: str = "",
    ) -> Dict[str, Any]:
        result = dict(gate) if isinstance(gate, dict) else {}
        result["depth_lookahead_checked"] = True
        result["lookahead_blocked_directions"] = []
        event_dict = event if isinstance(event, dict) else {}
        stage_name = str(stage or event_dict.get("stage") or event_dict.get("navigation_stage") or "").strip().upper()
        if stage_name not in self.route4_depth_lookahead_stages():
            result["depth_lookahead_reason"] = "stage_not_collection_navigation"
            return result
        summary = event_dict.get("pointcloud_summary", {}) if isinstance(event_dict.get("pointcloud_summary"), dict) else {}
        pointcloud_available = bool(summary.get("available", bool(summary)))
        if not pointcloud_available:
            result["depth_lookahead_reason"] = "pointcloud_unavailable"
            return result
        blocked_directions = self.route4_payload_blocked_directions_from_depth(summary, nominal_payload if isinstance(nominal_payload, dict) else {})
        if not blocked_directions:
            if bool(result.get("avoidance_active", False)):
                result["depth_lookahead_reason"] = "base_gate_already_active"
                return result
            result["depth_lookahead_reason"] = "nominal_direction_clear"
            return result
        semantic_hint = self.route4_depth_lookahead_semantic_hint(summary, result)
        trigger_distance_cm = self.route4_avoidance_trigger_distance_cm(semantic_hint)
        result.update(
            {
                "avoidance_active": True,
                "semantic_hint": semantic_hint,
                "semantic_source": "depth_lookahead_pointcloud",
                "trigger_distance_cm": float(trigger_distance_cm),
                "front_min_depth_cm": float(self.route4_event_float(summary.get("front_min_depth_cm"), default=0.0)),
                "forward_swept_clear": bool(summary.get("forward_swept_clear", True)),
                "pointcloud_available": pointcloud_available,
                "obstacle_geometry": str(summary.get("obstacle_geometry", "unknown") or "unknown"),
                "reason": "depth_lookahead_blocked_before_collection",
                "depth_lookahead_reason": "nominal_payload_intersects_blocked_swept_direction",
                "lookahead_blocked_directions": blocked_directions,
                "lookahead_stage": stage_name,
                "lookahead_nominal_payload": dict(nominal_payload) if isinstance(nominal_payload, dict) else {},
                "lookahead_policy": "observe_depth_before_advancing_to_capture_point",
            }
        )
        return result

    def route4_strategy_cache_key(self, event: Dict[str, Any]) -> str:
        prediction = event.get("representation_prediction", {}) if isinstance(event, dict) else {}
        hint = self.route4_semantic_hint_from_prediction(prediction if isinstance(prediction, dict) else {})
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown").strip().lower()
        waypoint = str(event.get("target_id", "") or event.get("waypoint_id", "") or "unknown_waypoint").strip()
        return f"{waypoint}|{hint}|{geometry}"

    def route4_strategy_for_event(self, event: Dict[str, Any], *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        self.ensure_route4_state()
        cache = self.llm_route4_state.get("obstacle_strategy_cache", {}) if isinstance(self.llm_route4_state.get("obstacle_strategy_cache"), dict) else {}
        key = self.route4_strategy_cache_key(event)
        if key in cache and isinstance(cache[key], dict):
            cached = dict(cache[key])
            cached["strategy_source"] = "representation_prediction_cached"
            cached["strategy_cache_key"] = key
            cached["strategy_cache_hit"] = True
            cached["strategy_cache_reason"] = "reused cached strategy for waypoint/semantic/geometry"
            return cached
        prediction = event.get("representation_prediction", {}) if isinstance(event.get("representation_prediction"), dict) else {}
        hint = self.route4_semantic_hint_from_prediction(prediction)
        environment = hint if hint != "building" else "building_or_roof"
        if environment == "unknown":
            environment = "default_unreal_scene"
        episode = {
            "episode_id": str(event.get("episode_id", "route4_fusion") or "route4_fusion"),
            "environment_id": environment,
            "obstacle_hint": hint,
            "method": LLM_STRATEGY_METHOD_ID,
            "operator_note": "Route4 fused navigation strategy.",
        }
        strategy: Dict[str, Any]
        if self.effective_llm_api_key() and hasattr(self, "obstacle_avoidance_llm_strategy_decision"):
            try:
                strategy = self.obstacle_avoidance_llm_strategy_decision(event, episode)
                strategy["strategy_source"] = "route4_oa_llm_strategy"
                llm_raw = strategy.get("llm_raw", {}) if isinstance(strategy.get("llm_raw"), dict) else {}
                if output_dir is None:
                    output_dir = self.route4_state_output_dir()
                if output_dir is not None:
                    ref = self.route4_log_llm_call(
                        output_dir,
                        "oa_strategy",
                        {
                            "frame_id": event.get("frame_id"),
                            "facade": event.get("facade"),
                            "target_id": event.get("target_id"),
                            "episode": episode,
                            "representation_prediction": event.get("representation_prediction", {}),
                            "pointcloud_summary": event.get("pointcloud_summary", {}),
                        },
                        llm_raw or {"raw_text": json.dumps(strategy, ensure_ascii=False)},
                        frame_id=int(event.get("frame_id", 0) or 0) if event.get("frame_id") is not None else None,
                        facade=str(event.get("facade", "") or ""),
                        target_id=str(event.get("target_id", "") or ""),
                        decision=strategy,
                    )
                    event.setdefault("llm_call_refs", []).append(ref)
            except Exception as exc:
                strategy = strategy_from_episode_metadata(episode)
                strategy["strategy_source"] = "representation_prediction_no_api"
                strategy["llm_error"] = str(exc)
        else:
            strategy = strategy_from_episode_metadata(episode)
            strategy["strategy_source"] = "representation_prediction_no_api"
        strategy_source = str(strategy.get("strategy_source", "representation_prediction_no_api") or "representation_prediction_no_api")
        llm_error = str(strategy.get("llm_error", "") or "")
        strategy = refine_strategy_with_pointcloud_context(strategy, event)
        strategy["strategy_source"] = strategy_source
        if llm_error:
            strategy["llm_error"] = llm_error
        strategy["strategy_cache_key"] = key
        strategy["strategy_cache_hit"] = False
        strategy["strategy_cache_reason"] = "new strategy computed and cached"
        cache[key] = dict(strategy)
        self.route4_update_state(obstacle_strategy_cache=cache)
        return strategy

    def route4_obstacle_args(self, config: Dict[str, float]) -> argparse.Namespace:
        return argparse.Namespace(
            stage="route4_fused_navigation",
            method=LLM_STRATEGY_METHOD_ID,
            run_id="route4",
            geometry_label="auto",
            note="Route4 fused route and obstacle avoidance.",
            reach_tol_cm=float(config.get("reach_tol_cm", 60.0)),
            route_step_cm=float(config.get("nav_step_cm", 20.0)),
            side_correction_cm=max(float(DEFAULT_ROUTE_SIDE_CORRECTION_CM), float(config.get("nav_step_cm", 20.0))),
            vertical_step_cm=max(float(DEFAULT_ROUTE_VERTICAL_STEP_CM), float(config.get("nav_step_cm", 20.0))),
        )

    def route4_predict_obstacle_representation(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_route4_state()
        model_path = Path(str(self.llm_route4_representation_model_var.get() or "")).expanduser()
        rgb_path = Path(str(event.get("rgb_path", "") or "")).expanduser()
        if not model_path.is_file():
            return {"status": "skipped", "reason": "model_not_found", "model_path": str(model_path)}
        if not rgb_path.is_file():
            return {"status": "skipped", "reason": "rgb_not_found", "rgb_path": str(rgb_path)}
        try:
            from obstacle_representation.demo import predict_obstacle_representation

            prediction = predict_obstacle_representation(model_path, rgb_path, event)
            prediction["status"] = "ok"
            return prediction
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "model_path": str(model_path), "rgb_path": str(rgb_path)}

    def route4_capture_obstacle_event(
        self,
        session: flight.DroneFlightSession,
        *,
        output_dir: Path,
        target_pose: Dict[str, float],
        start_pose: Dict[str, float],
        last_action: Dict[str, Any],
        stage: str,
        facade: str,
        target_id: str,
        frame_id: int,
        config: Dict[str, float],
        lookahead_plan: Optional[Dict[str, Any]] = None,
        nominal_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        action_detail = {
            "source": "llm_route_v4_fused_navigation",
            "stage": stage,
            "facade": facade,
            "target_id": target_id,
            "target_waypoint": target_pose,
            "last_action": last_action,
            "depth_lookahead_plan": lookahead_plan or {},
            "nominal_payload": nominal_payload or {},
        }
        old_mode = str(getattr(session.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING))
        try:
            session.args.lidar_capture_processing = "minimal"
            capture_result = session.capture_lidar_stream_frame(output_dir, frame_id, action_detail=action_detail)
        finally:
            try:
                session.args.lidar_capture_processing = old_mode
            except Exception:
                pass
        if not isinstance(capture_result, dict):
            raise RuntimeError("capture_lidar_stream_frame returned non-dict")
        episode = {
            "episode_id": f"route4_{target_id}_{frame_id}",
            "scenario_id": f"route4_{target_id}",
            "environment_id": "default_unreal_scene",
            "method": LLM_STRATEGY_METHOD_ID,
            "obstacle_hint": "unknown",
        }
        event = build_route_event(
            capture_result,
            session_dir=output_dir,
            frame_id=frame_id,
            args=self.route4_obstacle_args(config),
            episode=episode,
            episode_index=1,
            start=start_pose,
            goal=target_pose,
            last_action=last_action,
        )
        event["source"] = "llm_route_v4_fused_navigation"
        event["route4_stage"] = stage
        event["facade"] = facade
        event["target_id"] = target_id
        event["depth_lookahead_plan"] = lookahead_plan or {}
        event["nominal_payload"] = nominal_payload or {}
        prediction = self.route4_predict_obstacle_representation(event)
        event["representation_prediction"] = prediction
        hint = self.route4_semantic_hint_from_prediction(prediction)
        event["representation_obstacle_hint"] = hint
        self.route4_update_state(last_obstacle_event=event)
        self.route4_log_event(output_dir, "obstacle_sensing", event)
        try:
            label = str(prediction.get("predicted_label", prediction.get("status", "unknown")) or "unknown")
            confidence = float(prediction.get("confidence", 0.0) or 0.0)
            self.root.after(0, lambda l=label, c=confidence: self.llm_route4_representation_status_var.set(f"Representation: {l} conf={c:.2f}"))
        except Exception:
            pass
        return event

    def route4_normalize_avoidance_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(event if isinstance(event, dict) else {})
        collision = bool(normalized.get("collision_state", False))
        normalized["collision_state"] = collision
        normalized["avoidance_failed"] = collision
        return normalized

    def route4_write_avoidance_summary(self, output_dir: Path, *, status: str = "running") -> Dict[str, Any]:
        events = self.read_jsonl_artifact(output_dir / "avoidance_events.jsonl")
        collision_count = sum(1 for item in events if bool(item.get("collision_state", False)))
        summary = {
            "source": "llm_route_v4_fused_navigation",
            "status": status,
            "event_count": len(events),
            "collision_count": collision_count,
            "avoidance_failed_count": sum(1 for item in events if bool(item.get("avoidance_failed", False))),
            "last_event": events[-1] if events else {},
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(output_dir / "avoidance_session_summary.json", summary)
        return summary

    def route4_follow_navigation_waypoint_with_fusion(
        self,
        session: flight.DroneFlightSession,
        target_pose: Dict[str, float],
        *,
        output_dir: Path,
        stage: str,
        facade: str,
        target_id: str,
        target_house_id: str,
        waypoint_index: int,
        waypoint_count: int,
        config: Dict[str, float],
        escape_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = self.route3_current_pose(session)
        if not current:
            return {"status": "failed", "reason": "missing_current_pose"}
        started_at = time.time()
        tick_s = float(config["move_tick_ms"]) / 1000.0
        sensing_interval_s = float(self.route4_sensing_config()["sensing_interval_s"])
        step_index = 0
        last_error: Dict[str, Any] = {}
        last_sense_at = 0.0
        last_action = action_payload("hold")
        start_pose = dict(current)
        target_safety = self.route3_safety_report_for_pose(target_house_id, target_pose)
        if not bool(target_safety.get("safe", False)):
            self.route4_hold(session, output_dir=output_dir, reason="unsafe_waypoint")
            return {"status": "blocked", "reason": "unsafe_waypoint", "safety": target_safety}
        while not self.llm_route4_stop_event.is_set():
            if self.route4_wait_if_paused(session, output_dir):
                break
            error = self.route3_pose_error(current, target_pose, config)
            last_error = error
            self.root.after(
                0,
                lambda e=error: self.llm_route4_error_var.set(
                    f"Error: xy={float(e['dist_xy_cm']):.1f} z={float(e['dz']):.1f} yaw={float(e['yaw_error_deg']):.1f}"
                ),
            )
            if bool(error.get("reached", False)):
                self.route4_hold(session, output_dir=output_dir, reason="target_reached")
                return {
                    "status": "ok",
                    "reason": "target_reached",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "waypoint_index": waypoint_index,
                    "waypoint_count": waypoint_count,
                    "pose_error": error,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "current_pose": current,
                }
            if time.time() - started_at > float(config["max_stage_s"]):
                self.route4_hold(session, output_dir=output_dir, reason="nav_timeout")
                return {
                    "status": "timeout",
                    "reason": "nav_timeout",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "pose_error": error,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "current_pose": current,
                }
            payload = self.route4_movement_payload_for_target_with_lookahead(current, target_pose, config, stage=stage)
            event: Dict[str, Any] = {}
            force_depth_precheck = self.route4_should_precheck_depth_before_payload(stage, payload)
            should_sense = force_depth_precheck or step_index == 0 or (time.time() - last_sense_at) >= sensing_interval_s
            lookahead_plan = self.route4_depth_lookahead_plan(current, target_pose, stage=stage, facade=facade, target_id=target_id)
            if force_depth_precheck:
                align_result = self.route4_align_to_navigation_depth_yaw(
                    session,
                    current,
                    target_pose,
                    output_dir=output_dir,
                    stage=stage,
                    facade=facade,
                    target_id=target_id,
                    config=config,
                )
                if align_result.get("status") == "failed":
                    self.route4_hold(session, output_dir=output_dir, reason=str(align_result.get("reason", "lookahead_yaw_failed")))
                    return {
                        "status": "failed",
                        "reason": str(align_result.get("reason", "lookahead_yaw_failed")),
                        "stage": stage,
                        "facade": facade,
                        "target_id": target_id,
                        "pose_error": error,
                        "current_pose": current,
                    }
                current = dict(align_result.get("current_pose", current))
                last_action = dict(align_result.get("last_action", last_action))
                error = self.route3_pose_error(current, target_pose, config)
                last_error = error
                payload = self.route4_movement_payload_for_target_with_lookahead(current, target_pose, config, stage=stage)
                lookahead_plan = dict(align_result.get("lookahead_plan", lookahead_plan))
                lookahead_plan["nominal_payload"] = dict(payload)
                self.route4_set_thinking_status(lookahead_plan, output_dir=output_dir, write_artifact=True)
            if should_sense:
                try:
                    frame_id = self.route2_next_frame_index(output_dir)
                    event = self.route4_capture_obstacle_event(
                        session,
                        output_dir=output_dir,
                        target_pose=target_pose,
                        start_pose=start_pose,
                        last_action=last_action,
                        stage=stage,
                        facade=facade,
                        target_id=target_id,
                        frame_id=frame_id,
                        config=config,
                        lookahead_plan=lookahead_plan,
                        nominal_payload=payload,
                    )
                    last_sense_at = time.time()
                    gate = self.route4_should_apply_avoidance(event, {})
                    gate = self.route4_depth_lookahead_gate(event, gate, payload, stage=stage)
                    strategy: Dict[str, Any] = {
                        "strategy_source": "skipped_gate_inactive",
                        "obstacle_hint": gate.get("semantic_hint", "unknown"),
                        "environment_id": "building_or_roof" if gate.get("semantic_hint") == "building" else "default_unreal_scene",
                        "llm_call_required": False,
                    }
                    if bool(gate.get("avoidance_active", False)):
                        strategy = self.route4_strategy_for_event(event, output_dir=output_dir)
                        gate = self.route4_should_apply_avoidance(event, strategy)
                        gate = self.route4_depth_lookahead_gate(event, gate, payload, stage=stage)
                    event["llm_strategy"] = deepcopy(strategy)
                    event["avoidance_gate"] = gate
                    event["avoidance_active"] = bool(gate.get("avoidance_active", False))
                    if str(gate.get("reason", "") or "") == "depth_lookahead_blocked_before_collection":
                        self.route4_log_event(
                            output_dir,
                            "depth_lookahead_blocked",
                            {
                                "stage": stage,
                                "facade": facade,
                                "target_id": target_id,
                                "gate": gate,
                            },
                        )
                    decision_status = "AVOIDANCE" if bool(gate.get("avoidance_active", False)) else "CLEAR"
                    decision = dict(
                        lookahead_plan,
                        decision_state=decision_status,
                        avoidance_gate=gate,
                        thinking_status=(
                            f"Thinking: {stage} {decision_status} look waypoint yaw={float(lookahead_plan.get('look_yaw_deg', 0.0)):.1f} "
                            f"front={float(gate.get('front_min_depth_cm', 0.0)):.1f}/{float(gate.get('trigger_distance_cm', 0.0)):.0f}cm "
                            f"{gate.get('reason', '')}"
                        ),
                    )
                    self.route4_set_thinking_status(decision, output_dir=output_dir, write_artifact=True)
                    self.route4_log_event(output_dir, "depth_lookahead_decision", decision)
                    if bool(gate.get("avoidance_active", False)):
                        obstacle_hint = str(gate.get("semantic_hint", strategy.get("obstacle_hint", event.get("representation_obstacle_hint", "unknown"))) or "unknown")
                        environment_id = "building_or_roof" if obstacle_hint == "building" else str(strategy.get("environment_id", "default_unreal_scene") or "default_unreal_scene")
                        selected_action, avoidance_payload, reason, phase = select_route_action(
                            event["pointcloud_summary"],
                            event["candidate_action_scores"],
                            event.get("relative_target", {}),
                            method="pointcloud_direction_rule",
                            distance_to_goal_cm=float(event.get("distance_to_goal_cm", distance_3d_cm(current, target_pose)) or 0.0),
                            reach_tol_cm=float(config["reach_tol_cm"]),
                            last_action=last_action,
                            current_pose=current,
                            start=start_pose,
                            goal=target_pose,
                            route_step_cm=float(config["nav_step_cm"]),
                            side_correction_cm=max(float(DEFAULT_ROUTE_SIDE_CORRECTION_CM), float(config["nav_step_cm"])),
                            vertical_step_cm=max(float(DEFAULT_ROUTE_VERTICAL_STEP_CM), float(config["nav_step_cm"])),
                            obstacle_hint=obstacle_hint,
                            environment_id=environment_id,
                            building_state={},
                        )
                        risk_state = risk_state_from_summary(event["pointcloud_summary"], phase)
                        event.update(
                            {
                                "mission_phase": phase,
                                "risk_state": risk_state,
                                "selected_action": selected_action,
                                "selected_action_payload": avoidance_payload,
                                "selected_action_reason": reason,
                                "expert_action": str(avoidance_payload.get("action_name", selected_action)),
                                "expert_action_payload": avoidance_payload,
                                "agent_action": avoidance_payload,
                                "nominal_action": payload,
                            }
                        )
                        payload = avoidance_payload
                        self.root.after(
                            0,
                            lambda h=obstacle_hint, a=selected_action, r=risk_state, g=gate: self.llm_route4_avoidance_status_var.set(
                                f"Avoidance: active=YES hint={h} action={a} risk={r} front={float(g.get('front_min_depth_cm', 0.0)):.1f}/{float(g.get('trigger_distance_cm', 0.0)):.0f}cm"
                            ),
                        )
                    else:
                        event.update(
                            {
                                "mission_phase": "ROUTE_NAVIGATION",
                                "risk_state": "CLEAR",
                                "selected_action": "route3_nav",
                                "selected_action_payload": payload,
                                "selected_action_reason": str(gate.get("reason", "avoidance_gate_inactive")),
                                "expert_action": str(payload.get("action_name", "route3_nav")),
                                "expert_action_payload": payload,
                                "agent_action": payload,
                                "nominal_action": payload,
                            }
                        )
                        self.route4_log_event(output_dir, "avoidance_gate_inactive", {"stage": stage, "target_id": target_id, "gate": gate})
                        self.root.after(
                            0,
                            lambda g=gate: self.llm_route4_avoidance_status_var.set(
                                f"Avoidance: active=NO hint={g.get('semantic_hint', 'unknown')} front={float(g.get('front_min_depth_cm', 0.0)):.1f}/{float(g.get('trigger_distance_cm', 0.0)):.0f}cm {g.get('reason', '')}"
                            ),
                        )
                    self.route4_update_state(last_obstacle_event=event, avoidance_active=bool(gate.get("avoidance_active", False)))
                except Exception as exc:
                    self.route4_log_event(output_dir, "obstacle_sensing_failed", {"error": str(exc), "stage": stage, "target_id": target_id})
                    self.root.after(0, lambda e=exc: self.llm_route4_avoidance_status_var.set(f"Avoidance: fallback navigation ({e})"))
            predicted = self.route3_predict_next_pose(current, payload)
            safety = self.route3_safety_report_for_pose(target_house_id, predicted)
            escape_safety_allowed = self.route3_escape_safety_allowed(current, predicted, target_pose, safety, escape_obstacle)
            trace = {
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "waypoint_index": waypoint_index,
                "waypoint_count": waypoint_count,
                "step_index": step_index,
                "current_pose": current,
                "target_pose": target_pose,
                "pose_error": error,
                "payload": payload,
                "predicted_pose": predicted,
                "safety": safety,
                "obstacle_event_frame_id": event.get("frame_id") if event else None,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            self.append_jsonl(output_dir / "route4_movement_trace.jsonl", trace)
            self.root.after(
                0,
                lambda p=payload: self.llm_route4_payload_var.set(
                    f"Payload: f={p['forward_cm']} r={p['right_cm']} u={p['up_cm']} yaw={p['yaw_delta_deg']}"
                ),
            )
            if not bool(safety.get("safe", False)) and not escape_safety_allowed:
                self.route4_hold(session, output_dir=output_dir, reason=str(safety.get("reason", "unsafe_next_step")))
                self.route4_log_event(output_dir, "navigation_blocked", trace)
                return {
                    "status": "blocked",
                    "reason": str(safety.get("reason", "unsafe_next_step")),
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "pose_error": error,
                    "safety": safety,
                    "current_pose": current,
                }
            result = self.safe("Route V4 fused movement tick", lambda p=payload: session.move_relative(p))
            if not isinstance(result, dict):
                self.route4_hold(session, output_dir=output_dir, reason="movement_failed")
                return {"status": "failed", "reason": "movement_failed", "pose_error": error, "current_pose": current}
            self.root.after(0, lambda r=result: self.apply_state(r))
            post_pose = self.route3_pose_from_payload(result) or predicted
            if event:
                hit_state = self.read_obstacle_avoidance_2_hit_state(session) if hasattr(self, "read_obstacle_avoidance_2_hit_state") else {}
                post_reached, completion_reason, post_raw_progress, post_cross_track_cm = route_completion_state(
                    post_pose,
                    start_pose,
                    target_pose,
                    reach_tol_cm=float(config["reach_tol_cm"]),
                )
                event["goal_completion_reason"] = completion_reason
                event["goal_reached"] = bool(post_reached)
                event["post_action_pose"] = post_pose
                event["post_distance_to_goal_cm"] = round(distance_3d_cm(post_pose, target_pose), 3)
                event["post_raw_route_progress"] = round(post_raw_progress, 5)
                event["post_route_cross_track_cm"] = round(post_cross_track_cm, 3)
                annotate_collision_state(event, pre_pose=current, post_pose=post_pose, payload=payload, explicit_collision=hit_state)
                event = self.route4_normalize_avoidance_event(event)
                prediction_hint = self.route4_semantic_hint_from_prediction(event.get("representation_prediction", {}) if isinstance(event.get("representation_prediction"), dict) else {})
                gate_hint = str((event.get("avoidance_gate", {}) if isinstance(event.get("avoidance_gate"), dict) else {}).get("semantic_hint", "") or "")
                gate_source = str((event.get("avoidance_gate", {}) if isinstance(event.get("avoidance_gate"), dict) else {}).get("semantic_source", "") or "")
                if prediction_hint != "unknown" and gate_hint and prediction_hint != gate_hint:
                    conflict = {
                        "frame_id": event.get("frame_id"),
                        "representation_hint": prediction_hint,
                        "gate_hint": gate_hint,
                        "gate_source": gate_source,
                        "pointcloud_geometry": (event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}).get("obstacle_geometry", ""),
                    }
                    event["obstacle_label_conflict"] = conflict
                    event.setdefault("house_memory_updates", []).append({"facade": facade, "status": "in_progress", "obstacle_label_conflict": conflict})
                self.route4_write_frame_decision(output_dir, event, final_payload=payload)
                if bool(event.get("avoidance_active", False)) or bool(event.get("collision_state", False)):
                    self.append_jsonl(output_dir / "avoidance_events.jsonl", event)
                    self.route4_write_avoidance_summary(output_dir, status="running")
                if bool(event.get("collision_state")):
                    self.route4_hold(session, output_dir=output_dir, reason="collision")
                    return {"status": "collision", "reason": "collision", "event": event, "current_pose": post_pose}
            current = post_pose
            last_action = payload
            step_index += 1
            if self.llm_route4_stop_event.wait(max(0.01, tick_s)):
                break
        self.route4_hold(session, output_dir=output_dir, reason="stopped")
        return {"status": "stopped", "reason": "stopped", "pose_error": last_error, "current_pose": current}

    def route4_navigate_to_pose_with_fusion(
        self,
        session: flight.DroneFlightSession,
        target_pose: Dict[str, float],
        *,
        output_dir: Path,
        stage: str,
        facade: str,
        target_id: str,
        target_house_id: str,
    ) -> Dict[str, Any]:
        base_config = self.route4_nav_config()
        self.route4_enable_physics_movement(session)
        current = self.route3_current_pose(session)
        if not current:
            return {"status": "failed", "reason": "missing_current_pose"}
        max_replans = int(LLM_ROUTE3_ASTAR_MAX_REPLANS)
        replan_count = 0
        started_at = time.time()
        last_result: Dict[str, Any] = {}
        while replan_count <= max_replans and not self.llm_route4_stop_event.is_set():
            plan = self.route3_plan_navigation_waypoints(current, target_pose, target_house_id, grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM))
            plan_log = {
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "target_pose": target_pose,
                "current_pose": current,
                "replan_count": replan_count,
                "plan": plan,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            self.append_jsonl(output_dir / "route4_navigation_plan.jsonl", plan_log)
            self.route4_log_event(output_dir, "navigation_plan", plan_log)
            if plan.get("status") != "ok":
                self.route4_hold(session, output_dir=output_dir, reason=str(plan.get("reason", "navigation_plan_failed")))
                return {
                    "status": "blocked",
                    "reason": str(plan.get("reason", "navigation_plan_failed")),
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "navigation_plan": plan,
                    "replan_count": replan_count,
                    "elapsed_s": round(time.time() - started_at, 3),
                }
            waypoints = [dict(item) for item in plan.get("waypoints", []) if isinstance(item, dict)] or [dict(target_pose)]
            waypoint_count = len(waypoints)
            blocked_for_replan = False
            for idx, waypoint in enumerate(waypoints, start=1):
                waypoint_pose = {
                    "x": float(waypoint.get("x", target_pose["x"])),
                    "y": float(waypoint.get("y", target_pose["y"])),
                    "z": float(waypoint.get("z", target_pose["z"])),
                    "yaw": float(waypoint.get("yaw", target_pose["yaw"])),
                }
                segment_config = dict(base_config)
                is_escape_waypoint = isinstance(waypoint.get("escape_from_obstacle"), dict)
                if is_escape_waypoint:
                    segment_config["reach_tol_cm"] = min(
                        float(segment_config["reach_tol_cm"]),
                        float(waypoint.get("strict_reach_tol_cm", LLM_ROUTE3_ESCAPE_REACH_TOL_CM) or LLM_ROUTE3_ESCAPE_REACH_TOL_CM),
                    )
                    segment_config["yaw_tol_deg"] = float(waypoint.get("strict_yaw_tol_deg", 180.0) or 180.0)
                elif idx < waypoint_count:
                    segment_config["reach_tol_cm"] = max(float(segment_config["reach_tol_cm"]), float(LLM_ROUTE3_NAV_SEGMENT_REACH_TOL_CM))
                    segment_config["yaw_tol_deg"] = 180.0
                result = self.route4_follow_navigation_waypoint_with_fusion(
                    session,
                    waypoint_pose,
                    output_dir=output_dir,
                    stage=stage,
                    facade=facade,
                    target_id=target_id,
                    target_house_id=target_house_id,
                    waypoint_index=idx,
                    waypoint_count=waypoint_count,
                    config=segment_config,
                    escape_obstacle=waypoint.get("escape_from_obstacle") if isinstance(waypoint.get("escape_from_obstacle"), dict) else None,
                )
                last_result = result
                if result.get("status") == "ok":
                    current = result.get("current_pose", waypoint_pose) if isinstance(result.get("current_pose"), dict) else waypoint_pose
                    continue
                if result.get("status") == "blocked" and replan_count < max_replans:
                    fresh = self.route3_current_pose(session)
                    current = fresh or (result.get("current_pose", current) if isinstance(result.get("current_pose"), dict) else current)
                    replan_count += 1
                    blocked_for_replan = True
                    break
                result["navigation_plan"] = plan
                result["replan_count"] = replan_count
                result["elapsed_s"] = round(time.time() - started_at, 3)
                return result
            if blocked_for_replan:
                continue
            return {
                "status": "ok",
                "reason": "target_reached",
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "navigation_plan": plan,
                "replan_count": replan_count,
                "waypoint_count": waypoint_count,
                "pose_error": last_result.get("pose_error", {}),
                "elapsed_s": round(time.time() - started_at, 3),
            }
        self.route4_hold(session, output_dir=output_dir, reason="replan_exhausted")
        return {"status": "blocked", "reason": "replan_exhausted", "last_result": last_result, "replan_count": replan_count}

    def route4_run_summary(self, output_dir: Path, *, status: str) -> Dict[str, Any]:
        capture_rows = self.read_jsonl_artifact(output_dir / "lidar_capture_log.jsonl")
        avoidance_rows = self.read_jsonl_artifact(output_dir / "avoidance_events.jsonl")
        attempted_facades = set(self.llm_route4_completed_facades) | set(self.llm_route4_blocked_facades)
        completion_status = dict(self.llm_route4_state.get("facade_completion_status", {}) or {})
        house_memory = self.llm_route4_state.get("house_memory", {})
        summary = {
            "mode": "route4_llm_route_avoidance_fusion",
            "status": status,
            "target_house_id": str(self.llm_route4_state.get("target_house_id", "") or ""),
            "output_dir": str(output_dir),
            "completed_facades": sorted(self.llm_route4_completed_facades),
            "blocked_facades": sorted(self.llm_route4_blocked_facades),
            "full_completed_facades": sorted(name for name, state in completion_status.items() if state == "full_completed"),
            "degraded_completed_facades": sorted(name for name, state in completion_status.items() if state == "degraded_completed"),
            "facade_completion_status": self.route4_json_safe(completion_status),
            "mandatory_facade_agenda": self.route4_json_safe(self.llm_route4_state.get("mandatory_facade_agenda", {})),
            "house_memory": self.route4_json_safe(house_memory),
            "attempted_facades": sorted(attempted_facades),
            "capture_count": len(capture_rows),
            "avoidance_event_count": len(avoidance_rows),
            "collision_count": sum(1 for item in avoidance_rows if bool(item.get("collision_state", False))),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(output_dir / "route4_fusion_summary.json", summary)
        self.route4_write_avoidance_summary(output_dir, status=status)
        return summary

    def route4_full_search_worker(self, session: flight.DroneFlightSession, *, single_facade: bool = False, force_new: bool = False) -> None:
        self.ensure_route4_state()
        selected_target_house_id = self.selected_route_target_house_id()
        if not selected_target_house_id:
            self.root.after(0, lambda: self.llm_route4_status_var.set("LLM Route V4: select a target house first."))
            return
        output_dir: Optional[Path] = None
        status = "done"
        try:
            output_dir = self.route4_initialize_run(selected_target_house_id, force_new=force_new)
            self.route4_set_stage("TASK_ANALYSIS", output_dir=output_dir, message=f"analyzing task for selected house={selected_target_house_id}")
            task_plan = self.route4_analyze_task_plan(selected_target_house_id, output_dir=output_dir)
            target_house_id = str(task_plan.get("target_house_id", selected_target_house_id) or selected_target_house_id)
            self.route4_update_state(output_dir=str(output_dir), target_house_id=target_house_id)
            self.route4_initialize_house_memory(output_dir, target_house_id)
            self.route4_set_stage("PLAN_4_FACADES", output_dir=output_dir, message=f"planning facade candidates for house={target_house_id}")
            self.route4_set_control_lock(True)
            while not self.llm_route4_stop_event.is_set():
                completed = set(self.llm_route4_completed_facades)
                blocked = set(self.llm_route4_blocked_facades)
                if len(completed | blocked) >= 4:
                    break
                facade_candidates = self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False)
                ranked_candidates = self.route4_rank_observation_candidates(target_house_id, facade_candidates, completed, blocked, start_pose=self.route3_current_pose(session))
                self.route4_update_state(ranked_facade_candidates=ranked_candidates)
                decision = self.route4_decide_next_facade(target_house_id, ranked_candidates, completed, blocked)
                self.route4_log_event(output_dir, "high_level_decision", decision)
                if decision.get("next_action") == "done" or decision.get("stop_condition_met"):
                    break
                facade = str(decision.get("target_facade", "") or "").strip().lower()
                selected = next((item for item in ranked_candidates if str(item.get("facade", "") or "") == facade), {})
                if not facade or not selected:
                    status = "blocked_no_facade"
                    break
                self.route4_set_stage("SELECT_NEXT_FACADE", output_dir=output_dir, facade=facade, message=f"selected facade={facade}")
                self.route4_update_house_memory(
                    output_dir,
                    target_house_id,
                    facade,
                    status="in_progress",
                    reason=str(decision.get("reason", "selected_for_exploration") or "selected_for_exploration"),
                    observation_attempt=selected.get("selected_observation_attempt", selected) if isinstance(selected, dict) else {},
                    llm_decision_reason=str(decision.get("reason", "") or ""),
                )
                self.llm_route2_state = {"target_house_id": target_house_id, "output_dir": str(output_dir)}
                facade_dir = self.route2_facade_dir(output_dir, target_house_id, facade)
                observation_attempts = self.route3_ordered_observation_attempts(selected)
                observation: Dict[str, Any] = {}
                obs_pose: Dict[str, float] = {}
                nav_result: Dict[str, Any] = {}
                for attempt_index, attempt in enumerate(observation_attempts, start=1):
                    if self.llm_route4_stop_event.is_set():
                        break
                    attempt_status = str(attempt.get("status", "") or "")
                    if attempt_status == "blocked" and str(attempt.get("route4_navigation_status", attempt.get("route3_navigation_status", "")) or "") != "ok":
                        self.route4_log_event(output_dir, "observation_attempt_skipped", {"facade": facade, "attempt_index": attempt_index, "attempt": attempt})
                        continue
                    self.apply_route2_observation_plan(target_house_id, attempt, ranked_candidates, status_label=f"v4 selected attempt {attempt_index}/{len(observation_attempts)}")
                    observation = self.route2_selected_state().get("observation_point", attempt)
                    obs_pose = self.route3_target_pose_from_point(observation)
                    self.route4_update_state(
                        selected_observation_attempt=observation,
                        current_exploration_status={
                            "stage": "NAV_TO_OBS",
                            "facade": facade,
                            "observation_attempt_index": attempt_index,
                            "observation_attempt_count": len(observation_attempts),
                            "target_pose": obs_pose,
                        },
                    )
                    self.route4_set_stage("NAV_TO_OBS", output_dir=output_dir, facade=facade, target=obs_pose, message=f"navigating to {facade} observation attempt {attempt_index}/{len(observation_attempts)}")
                    nav_result = self.route4_navigate_to_pose_with_fusion(
                        session,
                        obs_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_OBS",
                        facade=facade,
                        target_id=f"{target_house_id}_{facade}_obs_attempt_{attempt_index}",
                        target_house_id=target_house_id,
                    )
                    self.route4_log_event(output_dir, "observation_attempt_navigation_result", {"facade": facade, "attempt_index": attempt_index, "navigation": nav_result})
                    if nav_result.get("status") != "ok":
                        rescue_result = self.route4_try_observation_rescue(
                            output_dir=output_dir,
                            target_house_id=target_house_id,
                            facade=facade,
                            original_observation=observation,
                            ranked_candidates=ranked_candidates,
                            nav_result=nav_result,
                            attempt_index=attempt_index,
                        )
                        if rescue_result.get("status") == "ok":
                            observation = dict(rescue_result.get("observation", observation))
                            obs_pose = dict(rescue_result.get("target_pose", self.route3_target_pose_from_point(observation)))
                            nav_result = {
                                "status": "ok",
                                "reason": "route4_observation_rescue_capture_current",
                                "stage": "NAV_TO_OBS",
                                "facade": facade,
                                "target_id": f"{target_house_id}_{facade}_obs_rescue_{attempt_index}",
                                "route4_observation_rescue": rescue_result,
                                "source_navigation_status": rescue_result.get("source_navigation", {}).get("status"),
                                "source_navigation_reason": rescue_result.get("source_navigation", {}).get("reason"),
                            }
                            self.route4_log_event(output_dir, "observation_attempt_navigation_rescued", {"facade": facade, "attempt_index": attempt_index, "navigation": nav_result})
                            break
                    if nav_result.get("status") == "ok":
                        break
                if nav_result.get("status") != "ok":
                    degraded_observation = self.route4_degraded_observation_candidate(
                        target_house_id=target_house_id,
                        facade=facade,
                        original_observation=observation if isinstance(observation, dict) else selected,
                        nav_result=nav_result,
                        fallback_pose=self.route3_current_pose(session),
                    )
                    if degraded_observation:
                        self.apply_route2_observation_plan(
                            target_house_id,
                            degraded_observation,
                            ranked_candidates,
                            status_label=f"v4 degraded observation {facade}",
                        )
                        observation = degraded_observation
                        obs_pose = self.route3_target_pose_from_point(degraded_observation)
                        nav_result = {
                            "status": "ok",
                            "reason": "route4_degraded_observation_capture",
                            "stage": "NAV_TO_OBS",
                            "facade": facade,
                            "target_id": f"{target_house_id}_{facade}_obs_degraded",
                            "route4_degraded_observation": degraded_observation,
                            "source_navigation_status": nav_result.get("status"),
                            "source_navigation_reason": nav_result.get("reason"),
                        }
                        self.route4_update_house_memory(
                            output_dir,
                            target_house_id,
                            facade,
                            status="degraded_completed",
                            reason="observation_navigation_degraded",
                            observation_attempt=degraded_observation,
                            nav_result=nav_result,
                            safe_observation_pose=degraded_observation,
                        )
                        self.route4_log_event(output_dir, "observation_navigation_degraded", {"facade": facade, "observation": degraded_observation, "navigation": nav_result})
                    else:
                        self.route4_update_house_memory(
                            output_dir,
                            target_house_id,
                            facade,
                            status="soft_blocked",
                            reason=str(nav_result.get("reason", "observation_navigation_failed") or "observation_navigation_failed"),
                            observation_attempt=observation if isinstance(observation, dict) else selected,
                            nav_result=nav_result,
                        )
                        self.llm_route4_blocked_facades.add(facade)
                        blocked_reasons = dict(self.llm_route4_state.get("blocked_facade_reasons", {}) if isinstance(self.llm_route4_state.get("blocked_facade_reasons"), dict) else {})
                        blocked_reasons[facade] = {"reason": str(nav_result.get("reason", "observation_navigation_failed") or "observation_navigation_failed"), "last_navigation_result": nav_result}
                        terminal_failed = sorted(set(self.llm_route4_state.get("terminal_failed_facades", []) if isinstance(self.llm_route4_state.get("terminal_failed_facades"), list) else []) | {facade})
                        self.route4_update_state(blocked_facades=sorted(self.llm_route4_blocked_facades), blocked_facade_reasons=blocked_reasons, terminal_failed_facades=terminal_failed)
                        self.route4_write_state_artifact()
                        if single_facade:
                            break
                        continue
                self.route4_set_stage("CAPTURE_RGB", output_dir=output_dir, facade=facade, target=obs_pose, message=f"capturing {facade} RGB")
                rgb_result = self.route3_capture_facade_rgb_current(session, output_dir=output_dir, facade_dir=facade_dir, house_id=target_house_id, facade=facade, planned_pose=obs_pose)
                self.route4_log_event(output_dir, "facade_rgb_capture", rgb_result)
                self.root.after(0, self.refresh_route4_support_views)
                self.route4_set_stage("ANALYZE_VLM", output_dir=output_dir, facade=facade, message=f"analyzing {facade} facade")
                self.route2_analyze_facade_vlm_worker()
                self.root.after(0, self.refresh_route4_support_views)
                self.route4_set_stage("PLAN_SCAN", output_dir=output_dir, facade=facade, message=f"planning {facade} scan")
                plan_result = self.route4_plan_facade_scan_current()
                points = [point for point in plan_result.get("points", []) if isinstance(point, dict)]
                scan_counts = plan_result.get("scan_counts", {}) if isinstance(plan_result.get("scan_counts"), dict) else {}
                self.route4_log_event(
                    output_dir,
                    "facade_scan_plan",
                    {
                        "facade": facade,
                        "point_count": len(points),
                        "physical_axis_sample_count": scan_counts.get("physical_axis_sample_count"),
                        "total_capture_record_count": scan_counts.get("total_capture_record_count", len(points)),
                        "yaw_supplement_record_count": scan_counts.get("yaw_supplement_record_count"),
                        "route4_scan_boundary_policy": plan_result.get("boundary_policy", {}),
                        "validation": plan_result.get("validation", {}),
                    },
                )
                if scan_counts:
                    physical = int(scan_counts.get("physical_axis_sample_count", len(points)) or 0)
                    total_records = int(scan_counts.get("total_capture_record_count", len(points)) or len(points))
                    self.root.after(
                        0,
                        lambda f=facade, p=physical, t=total_records: self.llm_route4_status_var.set(
                            f"LLM Route V4: planned {f} scan physical={p} capture/yaw={t}"
                        ),
                    )
                if not points:
                    self.route4_mark_facade_degraded_completed(
                        output_dir,
                        target_house_id,
                        facade,
                        observation=observation if isinstance(observation, dict) else {},
                        reason="scan_plan_empty_degraded_completion",
                        nav_result=nav_result if isinstance(nav_result, dict) else {},
                    )
                    if single_facade:
                        break
                    continue
                total = len(points)
                for idx, point in enumerate(points, start=1):
                    if self.llm_route4_stop_event.is_set():
                        break
                    scan_id = str(point.get("scan_id", "") or f"{facade}_{idx}")
                    target_pose = self.route3_target_pose_from_point(point)
                    self.route4_update_state(
                        current_exploration_status={
                            "stage": "NAV_TO_SCAN_POINT",
                            "facade": facade,
                            "point_index": idx,
                            "point_total": total,
                            "scan_id": scan_id,
                            "target_pose": target_pose,
                        }
                    )
                    self.route4_set_stage("NAV_TO_SCAN_POINT", output_dir=output_dir, facade=facade, target=target_pose, message=f"scan {idx}/{total} {scan_id}")
                    nav_scan = self.route4_navigate_to_pose_with_fusion(
                        session,
                        target_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_SCAN_POINT",
                        facade=facade,
                        target_id=scan_id,
                        target_house_id=target_house_id,
                    )
                    self.route4_log_event(output_dir, "scan_navigation_result", nav_scan)
                    if nav_scan.get("status") != "ok":
                        point["status"] = "blocked"
                        point["block_reason"] = nav_scan.get("reason", "navigation_failed")
                        continue
                    self.route4_set_stage("CAPTURE_SCAN", output_dir=output_dir, facade=facade, target=target_pose, message=f"capturing {scan_id}")
                    capture = self.route3_capture_scan_point_current(session, output_dir=output_dir, facade_dir=facade_dir, point=point, planned_pose=target_pose)
                    self.route4_log_event(output_dir, "scan_capture", {"scan_id": scan_id, **capture})
                    progress = 100.0 * (len(self.llm_route4_completed_facades) + (idx / max(1, total))) / 4.0
                    self.root.after(0, lambda v=progress: self.llm_route4_progress_var.set(max(0.0, min(100.0, v))))
                    self.root.after(0, lambda i=idx, t=total, f=facade: self.llm_route4_progress_text_var.set(f"Fusion: {f} {i}/{t}"))
                    self.root.after(0, self.refresh_route4_support_views)
                self.route2_write_lidar_summary(output_dir, running=False)
                self.route4_set_stage("VALIDATE_FACADE", output_dir=output_dir, facade=facade, message=f"validating {facade}")
                validation = self.route2_validate_facade()
                self.route4_log_event(output_dir, "facade_validation", validation)
                self.llm_route4_completed_facades.add(facade)
                completion_kind = "degraded_completed" if bool(observation.get("route4_degraded_observation", False)) else "full_completed"
                completed_order = [
                    str(item).strip().lower()
                    for item in (
                        self.llm_route4_state.get("completed_facade_order", [])
                        if isinstance(self.llm_route4_state.get("completed_facade_order"), list)
                        else []
                    )
                    if str(item).strip().lower()
                ]
                if facade not in completed_order:
                    completed_order.append(facade)
                completion_status = dict(self.llm_route4_state.get("facade_completion_status", {}) if isinstance(self.llm_route4_state.get("facade_completion_status"), dict) else {})
                completion_status[facade] = completion_kind
                self.route4_update_state(
                    completed_facades=sorted(self.llm_route4_completed_facades),
                    completed_facade_order=completed_order,
                    last_completed_facade=facade,
                    blocked_facades=sorted(self.llm_route4_blocked_facades),
                    facade_completion_status=completion_status,
                )
                self.route4_update_house_memory(
                    output_dir,
                    target_house_id,
                    facade,
                    status=completion_kind,
                    reason=str(validation.get("reason", completion_kind) if isinstance(validation, dict) else completion_kind),
                    observation_attempt=observation,
                    scan_coverage=scan_counts,
                    entrance_candidates=validation.get("entrance_candidates", []) if isinstance(validation, dict) else [],
                    obstacle_label_conflict=(
                        self.llm_route4_state.get("last_obstacle_event", {}).get("obstacle_label_conflict")
                        if isinstance(self.llm_route4_state.get("last_obstacle_event"), dict)
                        else None
                    ),
                    safe_observation_pose=observation,
                )
                self.route4_write_state_artifact()
                self.root.after(0, self.refresh_route4_support_views)
                if single_facade:
                    status = "single_facade_complete"
                    break
                self.route4_set_stage("DECIDE_NEXT", output_dir=output_dir, facade=facade, message=f"{facade} complete; deciding next facade")
            if self.llm_route4_stop_event.is_set():
                status = "stopped"
            if status == "done" and self.llm_route4_blocked_facades:
                status = "done_with_blocked"
            final_stage = "DONE" if status == "done" else ("DONE_WITH_BLOCKED" if status == "done_with_blocked" else "DECIDE_NEXT")
            self.route4_set_stage(final_stage, output_dir=output_dir, message=f"fused autosearch {status}")
            summary = self.route4_run_summary(output_dir, status=status)
            self.route4_log_event(output_dir, "summary", summary)
            self.root.after(0, lambda s=status, d=output_dir: self.llm_route4_status_var.set(f"LLM Route V4: {s} -> {d}"))
            self.root.after(0, self.refresh_route4_support_views)
        except Exception as exc:
            LOGGER.warning("Route V4 fused autosearch failed: %s", exc)
            if output_dir is not None:
                self.route4_log_event(output_dir, "error", {"reason": str(exc)})
                self.route4_run_summary(output_dir, status="failed")
            self.root.after(0, lambda e=exc: self.llm_route4_status_var.set(f"LLM Route V4: failed: {e}"))
        finally:
            self.route4_set_control_lock(False)

    def refresh_route4_preview(self) -> None:
        preview_text = getattr(self, "llm_route4_preview_text", None)
        if preview_text is None:
            return
        payload = {
            "fusion_state": self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {},
            "active_facade_state": self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {},
        }
        try:
            if not preview_text.winfo_exists():
                self.llm_route4_preview_text = None
                return
            preview_text.configure(state="normal")
            preview_text.delete("1.0", "end")
            preview_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            preview_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route4_preview_text = None

    def refresh_route4_analysis_view(self) -> None:
        analysis_text = getattr(self, "llm_route4_analysis_text", None)
        if analysis_text is None:
            return
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        obstacle_event = self.llm_route4_state.get("last_obstacle_event", {}) if isinstance(self.llm_route4_state, dict) else {}
        payload = {"facade_analysis": analysis or {"status": "No facade analysis yet."}, "last_obstacle_event": obstacle_event}
        try:
            if not analysis_text.winfo_exists():
                self.llm_route4_analysis_text = None
                return
            analysis_text.configure(state="normal")
            analysis_text.delete("1.0", "end")
            analysis_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            analysis_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route4_analysis_text = None

    def refresh_route4_rgb_display(self) -> None:
        widget = getattr(self, "llm_route4_rgb_label", None)
        if widget is None:
            return
        image_path = self.route2_current_rgb_path()
        if image_path is None:
            try:
                self.route2_draw_rgb_preview_message(widget, "No facade RGB")
                self.llm_route4_rgb_photo = None
            except tk.TclError:
                self.llm_route4_rgb_label = None
            return
        try:
            image = Image.open(image_path).convert("RGB")
            photo = ImageTk.PhotoImage(self.route2_rgb_preview_image(image, widget))
            self.route2_draw_rgb_preview_photo(widget, photo)
            self.llm_route4_rgb_photo = photo
        except Exception as exc:
            LOGGER.warning("Refresh route v4 facade RGB failed: %s", exc)
            try:
                self.route2_draw_rgb_preview_message(widget, f"RGB preview failed: {exc}")
                self.llm_route4_rgb_photo = None
            except tk.TclError:
                self.llm_route4_rgb_label = None

    def refresh_llm_route4_map(self) -> None:
        self.ensure_route4_state()
        widget = getattr(self, "llm_route4_map_widget", None)
        if widget is None:
            return
        try:
            if not self.load_map_resources(force=not bool(self.map_config)):
                self.llm_route4_map_status_var.set("Route V4 Map: map unavailable")
                return
            pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
            pose_x = float(pose.get("x", 0.0)) if pose else 0.0
            pose_y = float(pose.get("y", 0.0)) if pose else 0.0
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0))) if pose else 0.0
            houses, boxes = self.build_map_display(pose)
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
            route_points: List[Dict[str, Any]] = []
            state = self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {}
            route4_state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
            candidates_for_map = route4_state.get("ranked_facade_candidates", [])
            for candidate in candidates_for_map if isinstance(candidates_for_map, list) else []:
                if isinstance(candidate, dict):
                    selected_attempt = candidate.get("selected_observation_attempt", {}) if isinstance(candidate.get("selected_observation_attempt"), dict) else {}
                    item = dict(selected_attempt or candidate)
                    item["label"] = str(item.get("label", "") or f"{item.get('facade', '')}_obs")
                    item["route_point_type"] = "observation_point"
                    route_points.append(item)
            for point in state.get("facade_scan_points", []) if isinstance(state.get("facade_scan_points"), list) else []:
                if isinstance(point, dict):
                    item = dict(point)
                    item["label"] = str(item.get("scan_id", "") or f"scan_{len(route_points)}")
                    item["route_point_type"] = "scan_point"
                    route_points.append(item)
            widget.set_route_plan({"route_points": route_points})
            output_dir = self.route4_state_output_dir()
            if output_dir is not None:
                trace_rows = self.read_jsonl_artifact(output_dir / "route4_movement_trace.jsonl")
                trajectory = [row.get("current_pose", {}) for row in trace_rows[-400:] if isinstance(row.get("current_pose"), dict)]
                widget.set_trajectory(trajectory)
            self.llm_route4_map_status_var.set(
                f"Route V4 Map: houses={len(houses)} route_points={len(route_points)} completed={len(self.llm_route4_completed_facades)}/4"
            )
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM route v4 map failed: %s", exc)
            self.llm_route4_map_status_var.set(f"Route V4 Map: failed: {exc}")

    def schedule_route4_auto_refresh(self) -> None:
        self.ensure_route4_state()
        try:
            enabled = bool(self.llm_route4_auto_refresh_var.get())
        except tk.TclError:
            enabled = False
        if not enabled:
            self.cancel_route4_auto_refresh()
            return
        self.cancel_route4_auto_refresh()

        def tick() -> None:
            self.llm_route4_auto_refresh_job = None
            try:
                if not bool(self.llm_route4_auto_refresh_var.get()):
                    return
            except tk.TclError:
                return
            self.refresh_route4_support_views()
            try:
                self.llm_route4_auto_refresh_job = self.root.after(int(LLM_ROUTE3_AUTO_REFRESH_MS), tick)
            except tk.TclError:
                self.llm_route4_auto_refresh_job = None

        try:
            self.llm_route4_auto_refresh_job = self.root.after(int(LLM_ROUTE3_AUTO_REFRESH_MS), tick)
        except tk.TclError:
            self.llm_route4_auto_refresh_job = None

    def cancel_route4_auto_refresh(self) -> None:
        job = getattr(self, "llm_route4_auto_refresh_job", None)
        if job:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
            except Exception:
                pass
        self.llm_route4_auto_refresh_job = None

    def on_route4_auto_refresh_toggle(self) -> None:
        self.ensure_route4_state()
        try:
            enabled = bool(self.llm_route4_auto_refresh_var.get())
        except tk.TclError:
            enabled = False
        if enabled:
            self.refresh_route4_support_views()
            self.schedule_route4_auto_refresh()
        else:
            self.cancel_route4_auto_refresh()

    def refresh_route4_support_views(self) -> None:
        self.ensure_route4_state()
        completed = len(getattr(self, "llm_route4_completed_facades", set()) or set())
        blocked = len(getattr(self, "llm_route4_blocked_facades", set()) or set())
        progress = 100.0 * float(min(4, completed + blocked)) / 4.0
        state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        current_status = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status"), dict) else {}
        try:
            self.llm_route4_progress_var.set(max(0.0, min(100.0, progress)))
            self.llm_route4_progress_text_var.set(f"Fusion: completed={completed} blocked={blocked}")
            if current_status:
                facade = str(current_status.get("facade", state.get("current_facade", "")) or state.get("current_facade", "") or "-")
                stage = str(current_status.get("stage", state.get("stage", "")) or state.get("stage", "") or "-")
                point_index = current_status.get("point_index")
                point_total = current_status.get("point_total")
                suffix = f" {point_index}/{point_total}" if point_index is not None and point_total is not None else ""
                self.llm_route4_current_status_var.set(f"Current: {stage} {facade}{suffix}")
            else:
                self.llm_route4_current_status_var.set(f"Current: {state.get('stage', 'idle')} {state.get('current_facade', '')}")
            next_status = state.get("next_exploration_status", {}) if isinstance(state.get("next_exploration_status"), dict) else {}
            next_facade = str(next_status.get("facade", next_status.get("target_facade", "")) or "")
            self.llm_route4_next_status_var.set(f"Next: {next_facade}" if next_facade else "Next: n/a")
        except tk.TclError:
            pass
        self.refresh_route4_preview()
        self.refresh_route4_analysis_view()
        self.refresh_route4_rgb_display()
        self.refresh_llm_route4_map()

    def _on_llm_route4_window_content_configure(self, _event: tk.Event) -> None:
        canvas = getattr(self, "llm_route4_window_canvas", None)
        if canvas is None:
            return
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_llm_route4_window_canvas_configure(self, event: tk.Event) -> None:
        canvas = getattr(self, "llm_route4_window_canvas", None)
        content = getattr(self, "llm_route4_window_content", None)
        content_window = getattr(self, "llm_route4_window_content_window", None)
        if canvas is None or content is None or content_window is None:
            return
        requested_width = max(content.winfo_reqwidth(), int(event.width))
        canvas.itemconfigure(content_window, width=requested_width)
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_llm_route4_window_mousewheel(self, event: tk.Event):
        canvas = getattr(self, "llm_route4_window_canvas", None)
        if canvas is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            canvas.xview_scroll(delta, "units")
        else:
            canvas.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route4_window_mousewheel_linux(self, event: tk.Event):
        canvas = getattr(self, "llm_route4_window_canvas", None)
        if canvas is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        canvas.yview_scroll(direction, "units")
        return "break"

    def _on_llm_route4_text_mousewheel(self, event: tk.Event):
        widget = getattr(event, "widget", None)
        if widget is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            widget.xview_scroll(delta, "units")
        else:
            widget.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route4_text_mousewheel_linux(self, event: tk.Event):
        widget = getattr(event, "widget", None)
        if widget is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        widget.yview_scroll(direction, "units")
        return "break"

    def _bind_llm_route4_text_mousewheel(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_llm_route4_text_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route4_text_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route4_text_mousewheel_linux, add="+")
        except tk.TclError:
            pass

    def _bind_llm_route4_window_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            if isinstance(widget, tk.Text):
                self._bind_llm_route4_text_mousewheel(widget)
                return
            widget.bind("<MouseWheel>", self._on_llm_route4_window_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route4_window_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route4_window_mousewheel_linux, add="+")
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            self._bind_llm_route4_window_mousewheel_tree(child)

    def _build_llm_route4_section(self, parent: tk.Misc) -> tk.LabelFrame:
        self.ensure_route4_state()
        route = tk.LabelFrame(parent, text="LLM House Entrance Route V4 Fused Route + Avoidance")
        for col in (1, 3, 5):
            route.grid_columnconfigure(col, weight=1)
        route.grid_rowconfigure(9, weight=1)
        tk.Label(route, text="Target House").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        combo = ttk.Combobox(route, textvariable=self.llm_route_target_var, values=list(self.house_choice_map.keys()), state="readonly", width=22)
        combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        if combo not in self.house_target_combos:
            self.house_target_combos.append(combo)
        tk.Label(route, text="Task").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_task_text_var).grid(row=0, column=3, columnspan=3, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="API").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        api_combo = ttk.Combobox(route, textvariable=self.llm_api_style_var, values=LLM_API_STYLE_OPTIONS, state="readonly", width=17)
        api_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        api_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_llm_api_defaults(force=False))
        tk.Label(route, text="Base URL").grid(row=1, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_base_url_var).grid(row=1, column=3, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="Model").grid(row=1, column=4, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_model_var, width=18).grid(row=1, column=5, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="API Key").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_api_key_var, show="*").grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="Timeout s").grid(row=2, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_timeout_s_var, width=8).grid(row=2, column=3, sticky="w", padx=6, pady=6)
        tk.Label(route, textvariable=self.llm_route4_active_var, anchor="w").grid(row=2, column=4, columnspan=2, sticky="ew", padx=6, pady=6)

        rep = tk.Frame(route)
        rep.grid(row=3, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(rep, text="Representation Model").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(rep, textvariable=self.llm_route4_representation_model_var, width=54).pack(side="left", fill="x", expand=True, padx=(0, 6), pady=4)
        tk.Button(rep, text="Default", command=lambda: self.llm_route4_representation_model_var.set(str(self.default_route4_representation_model_path()))).pack(side="left", padx=6, pady=4)
        tk.Label(rep, text="Sense interval s").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(rep, textvariable=self.llm_route4_sensing_interval_s_var, width=7).pack(side="left", padx=(0, 8), pady=4)

        config = tk.Frame(route)
        config.grid(row=4, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(config, text="Floor height m").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_floor_height_m_var, width=6).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Default floors").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_default_floors_var, width=5).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Density mode").pack(side="left", padx=(6, 2), pady=4)
        ttk.Combobox(config, textvariable=self.llm_route2_density_mode_var, values=("auto", "high", "medium", "low"), state="readonly", width=8).pack(side="left", padx=(0, 8), pady=4)

        nav = tk.Frame(route)
        nav.grid(row=5, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        for label, var, width in (
            ("Move tick ms", self.llm_route4_move_tick_ms_var, 6),
            ("Nav step cm", self.llm_route4_nav_step_cm_var, 6),
            ("Reach tol cm", self.llm_route4_reach_tol_cm_var, 6),
            ("Z tol cm", self.llm_route4_z_tol_cm_var, 6),
            ("Yaw tol deg", self.llm_route4_yaw_tol_deg_var, 6),
            ("Max stage s", self.llm_route4_max_stage_s_var, 6),
        ):
            tk.Label(nav, text=label).pack(side="left", padx=(6, 2), pady=4)
            tk.Entry(nav, textvariable=var, width=width).pack(side="left", padx=(0, 6), pady=4)

        actions = tk.Frame(route)
        actions.grid(row=6, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 4))
        tk.Button(actions, text="Start Fused Route+Avoidance", command=self.on_route4_start_fused_search).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Step Facade", command=self.on_route4_step_facade).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Pause/Resume", command=self.on_route4_toggle_pause).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=self.on_route4_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_route4_clear).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Validate Run", command=self.on_route4_validate_run).pack(side="left", padx=6, pady=4)

        status = tk.Frame(route)
        status.grid(row=7, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        status.grid_columnconfigure(1, weight=1)
        tk.Label(status, textvariable=self.llm_route4_stage_var, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 12))
        tk.Label(status, textvariable=self.llm_route4_target_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=(0, 12))
        tk.Label(status, textvariable=self.llm_route4_error_var, anchor="w").grid(row=0, column=2, sticky="w")
        tk.Label(status, textvariable=self.llm_route4_payload_var, anchor="w").grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(status, textvariable=self.llm_route4_avoidance_status_var, anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(status, textvariable=self.llm_route4_representation_status_var, anchor="w").grid(row=3, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(status, textvariable=self.llm_route4_thinking_status_var, anchor="w").grid(row=4, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(route, textvariable=self.llm_route4_status_var, anchor="w").grid(row=8, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        preview_frame = tk.Frame(route)
        preview_frame.grid(row=9, column=0, columnspan=6, sticky="nsew", padx=6, pady=(0, 6))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_text = tk.Text(preview_frame, height=7, width=96, wrap="none", font=("Consolas", 9))
        preview_y = tk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
        preview_x = tk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
        preview_text.configure(yscrollcommand=preview_y.set, xscrollcommand=preview_x.set)
        preview_text.grid(row=0, column=0, sticky="nsew")
        preview_y.grid(row=0, column=1, sticky="ns")
        preview_x.grid(row=1, column=0, sticky="ew")
        preview_text.configure(state="disabled")
        self.llm_route4_preview_text = preview_text
        setattr(route, "_llm_route4_combo", combo)
        return route

    def open_llm_route_window4(self) -> None:
        self.ensure_route4_state()
        if self.llm_route4_window is not None and self.llm_route4_window.winfo_exists():
            self.llm_route4_window.lift()
            self.llm_route4_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route V4 Fused Route + Avoidance")
        window.geometry("980x760")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)

        window_canvas = tk.Canvas(window, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(window, orient="vertical", command=window_canvas.yview)
        h_scrollbar = tk.Scrollbar(window, orient="horizontal", command=window_canvas.xview)
        window_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        window_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        content = tk.Frame(window_canvas)
        content_window = window_canvas.create_window((0, 0), window=content, anchor="nw")
        self.llm_route4_window_canvas = window_canvas
        self.llm_route4_window_content = content
        self.llm_route4_window_content_window = content_window
        content.bind("<Configure>", self._on_llm_route4_window_content_configure, add="+")
        window_canvas.bind("<Configure>", self._on_llm_route4_window_canvas_configure, add="+")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(2, weight=1)

        route = self._build_llm_route4_section(content)
        route.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        support = tk.Frame(content)
        support.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 4))
        support.grid_columnconfigure(0, weight=0)
        support.grid_columnconfigure(1, weight=1)
        rgb_frame = tk.LabelFrame(support, text="Facade RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        self.llm_route4_rgb_label = tk.Canvas(rgb_frame, width=330, height=220, bg="#202020", highlightthickness=0)
        self.llm_route4_rgb_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route4_rgb_label.bind("<Configure>", lambda _event: self.refresh_route4_rgb_display(), add="+")
        analysis_frame = tk.LabelFrame(support, text="Fusion Analysis")
        analysis_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_rowconfigure(0, weight=1)
        analysis_text = tk.Text(analysis_frame, height=10, width=64, wrap="none", font=("Consolas", 9))
        analysis_y = tk.Scrollbar(analysis_frame, orient="vertical", command=analysis_text.yview)
        analysis_x = tk.Scrollbar(analysis_frame, orient="horizontal", command=analysis_text.xview)
        analysis_text.configure(yscrollcommand=analysis_y.set, xscrollcommand=analysis_x.set)
        analysis_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        analysis_y.grid(row=0, column=1, sticky="ns", pady=(6, 0))
        analysis_x.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
        analysis_text.configure(state="disabled")
        self.llm_route4_analysis_text = analysis_text

        map_frame = tk.LabelFrame(content, text="Map Facade V4 Fusion")
        map_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(1, weight=1)
        map_toolbar = tk.Frame(map_frame)
        map_toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        tk.Label(map_toolbar, textvariable=self.llm_route4_map_status_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(map_toolbar, textvariable=self.llm_route4_current_status_var, anchor="w").pack(side="left", padx=(8, 2))
        tk.Label(map_toolbar, textvariable=self.llm_route4_next_status_var, anchor="w").pack(side="left", padx=(8, 2))
        tk.Label(map_toolbar, textvariable=self.llm_route4_progress_text_var, anchor="e").pack(side="left", padx=(8, 2))
        ttk.Progressbar(map_toolbar, variable=self.llm_route4_progress_var, maximum=100.0, length=150, mode="determinate").pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            map_toolbar,
            text="Auto Refresh",
            variable=self.llm_route4_auto_refresh_var,
            command=self.on_route4_auto_refresh_toggle,
        ).pack(side="left", padx=(0, 8))
        tk.Button(map_toolbar, text="Refresh Map", command=self.refresh_llm_route4_map).pack(side="right", padx=6)
        self.load_map_resources(force=True)
        self.llm_route4_map_widget = OverheadMapWidget(map_frame, world_bounds=self.map_world_bounds, canvas_w=800, canvas_h=320)
        self.llm_route4_map_widget.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route4_window = window
        self._bind_llm_route4_window_mousewheel_tree(content)
        window_canvas.bind("<MouseWheel>", self._on_llm_route4_window_mousewheel, add="+")
        window_canvas.bind("<Button-4>", self._on_llm_route4_window_mousewheel_linux, add="+")
        window_canvas.bind("<Button-5>", self._on_llm_route4_window_mousewheel_linux, add="+")

        def close_window() -> None:
            combo = getattr(route, "_llm_route4_combo", None)
            if combo is not None and combo in self.house_target_combos:
                self.house_target_combos.remove(combo)
            self.llm_route4_window = None
            self.llm_route4_window_canvas = None
            self.llm_route4_window_content = None
            self.llm_route4_window_content_window = None
            self.llm_route4_map_widget = None
            self.llm_route4_preview_text = None
            self.llm_route4_analysis_text = None
            self.llm_route4_rgb_label = None
            self.llm_route4_rgb_photo = None
            self.cancel_route4_auto_refresh()
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.refresh_route4_support_views()

    def on_route4_start_fused_search(self) -> None:
        self.ensure_route4_state()
        session = self.active_session()
        if session is None:
            return
        if self.llm_route4_thread is not None and self.llm_route4_thread.is_alive():
            self.llm_route4_status_var.set("LLM Route V4: already running.")
            return
        self.llm_route4_stop_event.clear()
        self.llm_route4_pause_event.clear()
        self.llm_route4_paused_var.set(False)
        self.llm_route4_thread = threading.Thread(target=lambda: self.route4_full_search_worker(session, single_facade=False, force_new=True), daemon=True)
        self.llm_route4_thread.start()

    def on_route4_step_facade(self) -> None:
        self.ensure_route4_state()
        session = self.active_session()
        if session is None:
            return
        if self.llm_route4_thread is not None and self.llm_route4_thread.is_alive():
            self.llm_route4_status_var.set("LLM Route V4: wait for current worker.")
            return
        self.llm_route4_stop_event.clear()
        self.llm_route4_pause_event.clear()
        self.llm_route4_thread = threading.Thread(target=lambda: self.route4_full_search_worker(session, single_facade=True, force_new=False), daemon=True)
        self.llm_route4_thread.start()

    def on_route4_toggle_pause(self) -> None:
        self.ensure_route4_state()
        if self.llm_route4_pause_event.is_set():
            self.llm_route4_pause_event.clear()
            self.llm_route4_paused_var.set(False)
            self.llm_route4_status_var.set("LLM Route V4: resumed.")
        else:
            self.llm_route4_pause_event.set()
            self.llm_route4_paused_var.set(True)
            session = self.active_session()
            if session is not None:
                self.route4_hold(session, output_dir=self.route4_state_output_dir(), reason="pause_button")
            self.llm_route4_status_var.set("LLM Route V4: paused.")

    def on_route4_stop(self) -> None:
        self.ensure_route4_state()
        self.llm_route4_stop_event.set()
        self.llm_route4_pause_event.clear()
        session = self.active_session()
        if session is not None:
            self.route4_hold(session, output_dir=self.route4_state_output_dir(), reason="stop_button")
        self.llm_route4_status_var.set("LLM Route V4: stop requested.")

    def on_route4_clear(self) -> None:
        self.ensure_route4_state()
        if self.llm_route4_thread is not None and self.llm_route4_thread.is_alive():
            self.llm_route4_status_var.set("LLM Route V4: stop before clearing.")
            return
        self.llm_route4_state = {}
        self.llm_route4_completed_facades = set()
        self.llm_route4_blocked_facades = set()
        self.llm_route4_stage_var.set("Stage: idle")
        self.llm_route4_active_var.set("Active: n/a")
        self.llm_route4_target_var.set("Target: n/a")
        self.llm_route4_error_var.set("Error: n/a")
        self.llm_route4_payload_var.set("Payload: hold")
        self.llm_route4_current_status_var.set("Current: idle")
        self.llm_route4_next_status_var.set("Next: n/a")
        self.llm_route4_avoidance_status_var.set("Avoidance: idle")
        self.llm_route4_representation_status_var.set("Representation: idle")
        self.llm_route4_thinking_status_var.set("Thinking: idle")
        self.llm_route4_status_var.set("LLM Route V4: cleared.")
        self.refresh_route4_support_views()

    def on_route4_validate_run(self) -> None:
        self.ensure_route4_state()
        output_dir = self.route4_state_output_dir()
        if output_dir is None:
            self.llm_route4_status_var.set("LLM Route V4: no run to validate.")
            return
        summary = self.route4_run_summary(output_dir, status=str(self.llm_route4_state.get("stage", "manual_validate") or "manual_validate"))
        self.llm_route4_status_var.set(f"LLM Route V4: run summary -> {output_dir / 'route4_fusion_summary.json'}")
        self.route4_log_event(output_dir, "manual_validate_run", summary)
