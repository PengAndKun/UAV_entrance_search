from __future__ import annotations

from copy import deepcopy
from PIL import ImageDraw

from .common import *
from .local_obstacle_map import LocalObstacleMap, LocalObstacleMapConfig
from . import lidar_yolo_analysis, route6_map_builder
from .route5_route7_planning import Route7PlanningMixin
from .route5_route7_map_window import Route7MapWindowMixin
from .route5_house_memory import Route5HouseMemoryMixin
from .route5_capture_guard import Route5CaptureGuardMixin
from .route5_avoidance_decision import Route5AvoidanceDecisionMixin

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
from obstacle_avoidance_3.control_utils import serializable_or2_prediction
from obstacle_avoidance_3.or2_direction_rule import select_or2_direction
from obstacle_avoidance_3.plans import DEFAULT_PLAN_FILENAME as OA3_DEFAULT_PLAN_FILENAME
from obstacle_representation_2.demo import ObstacleRepresentation2Predictor, render_affordance_overlay

try:
    from obstacle_representation_3.demo import (
        ObstacleRepresentation3Predictor,
        render_affordance_overlay as render_or3_affordance_overlay,
    )
except Exception:
    ObstacleRepresentation3Predictor = None
    render_or3_affordance_overlay = None


class _Route5FallbackVar:
    def __init__(self, value: Any = "") -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


def _route5_string_var(value: str) -> Any:
    try:
        return tk.StringVar(value=value)
    except RuntimeError:
        return _Route5FallbackVar(value)


class Route5FusionControlMixin(
    Route7PlanningMixin,
    Route7MapWindowMixin,
    Route5HouseMemoryMixin,
    Route5CaptureGuardMixin,
    Route5AvoidanceDecisionMixin,
):
    def default_route5_or2_model_path(self) -> Path:
        return PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"

    def default_route5_or3_model_path(self) -> Path:
        return PROJECT_ROOT / "obstacle_representation_3_data" / "models" / "a_plus_3_model.pt"

    def default_route7_or31_model_path(self) -> Path:
        return PROJECT_ROOT / "obstacle_representation_3_data" / "models" / "a_plus_3_1_model.pt"

    def default_route5_oa3_plan_path(self) -> Path:
        return PROJECT_ROOT / "obstacle_avoidance_3_data" / "plans" / OA3_DEFAULT_PLAN_FILENAME

    def default_route5_representation_model_path(self) -> Path:
        return self.default_route5_or2_model_path()

    def route5_event_is_route7(self, event: Dict[str, Any] | None = None) -> bool:
        event_dict = event if isinstance(event, dict) else {}
        label = str(event_dict.get("route_window_label", "") or "").strip().upper()
        if label == "V7":
            return True
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        if str(state.get("route_window_label", "") or "").strip().upper() == "V7":
            return True
        for key in ("route5_output_dir", "output_dir", "capture_dir"):
            raw = str(event_dict.get(key, state.get(key, "")) or "").replace("/", "\\").lower()
            if "llm_route_7_fusion_runs" in raw or "_v7_or3_1_fused_" in raw or "_v7_or2_fused_" in raw:
                return True
        return False

    def route5_or3_model_path_for_event(self, event: Dict[str, Any] | None = None) -> Path:
        return self.default_route7_or31_model_path() if self.route5_event_is_route7(event) else self.default_route5_or3_model_path()

    def ensure_route5_state(self) -> None:
        if not hasattr(self, "llm_route5_status_var"):
            self.llm_route5_status_var = tk.StringVar(value="LLM Route V5: idle")
        if not hasattr(self, "llm_route5_map_status_var"):
            self.llm_route5_map_status_var = tk.StringVar(value="Route V5 Map: idle")
        if not hasattr(self, "llm_route5_stage_var"):
            self.llm_route5_stage_var = tk.StringVar(value="Stage: idle")
        if not hasattr(self, "llm_route5_active_var"):
            self.llm_route5_active_var = tk.StringVar(value="Active: n/a")
        if not hasattr(self, "llm_route5_target_var"):
            self.llm_route5_target_var = tk.StringVar(value="Target: n/a")
        if not hasattr(self, "llm_route5_error_var"):
            self.llm_route5_error_var = tk.StringVar(value="Error: n/a")
        if not hasattr(self, "llm_route5_payload_var"):
            self.llm_route5_payload_var = tk.StringVar(value="Payload: hold")
        if not hasattr(self, "llm_route5_progress_text_var"):
            self.llm_route5_progress_text_var = tk.StringVar(value="Fusion: 0%")
        if not hasattr(self, "llm_route5_progress_var"):
            self.llm_route5_progress_var = tk.DoubleVar(value=0.0)
        if not hasattr(self, "llm_route5_current_status_var"):
            self.llm_route5_current_status_var = tk.StringVar(value="Current: idle")
        if not hasattr(self, "llm_route5_next_status_var"):
            self.llm_route5_next_status_var = tk.StringVar(value="Next: n/a")
        if not hasattr(self, "llm_route5_avoidance_status_var"):
            self.llm_route5_avoidance_status_var = tk.StringVar(value="Avoidance: idle")
        if not hasattr(self, "llm_route5_representation_status_var"):
            self.llm_route5_representation_status_var = tk.StringVar(value="Representation: idle")
        if not hasattr(self, "llm_route5_thinking_status_var"):
            self.llm_route5_thinking_status_var = tk.StringVar(value="Thinking: idle")
        if not hasattr(self, "llm_route5_paused_var"):
            self.llm_route5_paused_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route5_auto_refresh_var"):
            self.llm_route5_auto_refresh_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route5_show_all_obs_points_var"):
            self.llm_route5_show_all_obs_points_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route5_show_facade_captures_var"):
            self.llm_route5_show_facade_captures_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route5_move_tick_ms_var"):
            self.llm_route5_move_tick_ms_var = tk.StringVar(value="150")
        if not hasattr(self, "llm_route5_nav_step_cm_var"):
            self.llm_route5_nav_step_cm_var = tk.StringVar(value="20")
        if not hasattr(self, "llm_route5_reach_tol_cm_var"):
            self.llm_route5_reach_tol_cm_var = tk.StringVar(value="60")
        if not hasattr(self, "llm_route5_z_tol_cm_var"):
            self.llm_route5_z_tol_cm_var = tk.StringVar(value="40")
        if not hasattr(self, "llm_route5_yaw_tol_deg_var"):
            self.llm_route5_yaw_tol_deg_var = tk.StringVar(value="10")
        if not hasattr(self, "llm_route5_max_stage_s_var"):
            self.llm_route5_max_stage_s_var = tk.StringVar(value="90")
        if not hasattr(self, "llm_route5_sensing_interval_s_var"):
            self.llm_route5_sensing_interval_s_var = tk.StringVar(value="1.0")
        if not hasattr(self, "llm_route5_representation_model_var"):
            self.llm_route5_representation_model_var = tk.StringVar(value=str(self.default_route5_representation_model_path()))
        if not hasattr(self, "llm_route5_oa3_plan_var"):
            self.llm_route5_oa3_plan_var = tk.StringVar(value=str(self.default_route5_oa3_plan_path()))
        if not hasattr(self, "route5_or2_state_var"):
            self.route5_or2_state_var = tk.StringVar(value="State: --")
        if not hasattr(self, "route5_or2_frame_count_var"):
            self.route5_or2_frame_count_var = tk.StringVar(value="Frames: 0")
        if not hasattr(self, "route5_or2_front_depth_var"):
            self.route5_or2_front_depth_var = tk.StringVar(value="Front depth: --")
        if not hasattr(self, "route5_or2_can_forward_var"):
            self.route5_or2_can_forward_var = tk.StringVar(value="Can forward: --")
        if not hasattr(self, "route5_or2_selected_direction_var"):
            self.route5_or2_selected_direction_var = tk.StringVar(value="Selected direction: --")
        if not hasattr(self, "route5_or2_corridor_var"):
            self.route5_or2_corridor_var = tk.StringVar(value="Corridors: --")
        if not hasattr(self, "route5_or2_capture_dir_var"):
            self.route5_or2_capture_dir_var = tk.StringVar(value="OR2 capture: --")
        if not hasattr(self, "route5_or2_interval_s_var"):
            self.route5_or2_interval_s_var = tk.StringVar(value="1.0")
        if not hasattr(self, "route5_or2_monitor_stop_event"):
            self.route5_or2_monitor_stop_event = threading.Event()
        if not hasattr(self, "route5_or2_monitor_thread"):
            self.route5_or2_monitor_thread = None
        if not hasattr(self, "route5_or2_state_label"):
            self.route5_or2_state_label = None
        if not hasattr(self, "route5_or2_rgb_label"):
            self.route5_or2_rgb_label = None
        if not hasattr(self, "route5_or2_mask_label"):
            self.route5_or2_mask_label = None
        if not hasattr(self, "route5_or2_rgb_photo"):
            self.route5_or2_rgb_photo = None
        if not hasattr(self, "route5_or2_mask_photo"):
            self.route5_or2_mask_photo = None
        if not hasattr(self, "route5_or2_report_text"):
            self.route5_or2_report_text = None
        if not hasattr(self, "route5_or2_predictor"):
            self.route5_or2_predictor = None
        if not hasattr(self, "route5_or2_predictor_path"):
            self.route5_or2_predictor_path = ""
        if not hasattr(self, "route5_or3_predictor"):
            self.route5_or3_predictor = None
        if not hasattr(self, "route5_or3_predictor_path"):
            self.route5_or3_predictor_path = ""
        if not hasattr(self, "llm_route5_window"):
            self.llm_route5_window = None
        if not hasattr(self, "llm_route5_window_canvas"):
            self.llm_route5_window_canvas = None
        if not hasattr(self, "llm_route5_window_content"):
            self.llm_route5_window_content = None
        if not hasattr(self, "llm_route5_window_content_window"):
            self.llm_route5_window_content_window = None
        if not hasattr(self, "llm_route5_map_widget"):
            self.llm_route5_map_widget = None
        if not hasattr(self, "llm_route5_map_frame"):
            self.llm_route5_map_frame = None
        if not hasattr(self, "route5_local_obstacle_map"):
            self.route5_local_obstacle_map = None
        if not hasattr(self, "route5_local_obstacle_output_dir"):
            self.route5_local_obstacle_output_dir = None
        if not hasattr(self, "llm_route5_preview_text"):
            self.llm_route5_preview_text = None
        if not hasattr(self, "llm_route5_analysis_text"):
            self.llm_route5_analysis_text = None
        if not hasattr(self, "llm_route5_capture_analysis_status_var"):
            self.llm_route5_capture_analysis_status_var = tk.StringVar(value="V5 Capture Analysis: idle")
        if not hasattr(self, "llm_route5_capture_analysis_run_dir_var"):
            self.llm_route5_capture_analysis_run_dir_var = tk.StringVar(value="")
        if not hasattr(self, "llm_route5_capture_analysis_thread"):
            self.llm_route5_capture_analysis_thread = None
        if not hasattr(self, "llm_route5_capture_analysis_stop_event"):
            self.llm_route5_capture_analysis_stop_event = threading.Event()
        if not hasattr(self, "llm_route5_rgb_label"):
            self.llm_route5_rgb_label = None
        if not hasattr(self, "llm_route5_rgb_photo"):
            self.llm_route5_rgb_photo = None
        if not hasattr(self, "llm_route5_state"):
            self.llm_route5_state = {}
        if not hasattr(self, "llm_route5_completed_facades"):
            self.llm_route5_completed_facades = set()
        if not hasattr(self, "llm_route5_blocked_facades"):
            self.llm_route5_blocked_facades = set()
        if not hasattr(self, "llm_route5_thread"):
            self.llm_route5_thread = None
        if not hasattr(self, "llm_route5_stop_event"):
            self.llm_route5_stop_event = threading.Event()
        if not hasattr(self, "llm_route5_pause_event"):
            self.llm_route5_pause_event = threading.Event()
        if not hasattr(self, "llm_route5_control_locked"):
            self.llm_route5_control_locked = False
        if not hasattr(self, "llm_route5_auto_refresh_job"):
            self.llm_route5_auto_refresh_job = None
        if not hasattr(self, "llm_route7_window"):
            self.llm_route7_window = None
        if not hasattr(self, "llm_route7_window_canvas"):
            self.llm_route7_window_canvas = None
        if not hasattr(self, "llm_route7_window_content"):
            self.llm_route7_window_content = None
        if not hasattr(self, "llm_route7_window_content_window"):
            self.llm_route7_window_content_window = None
        if not hasattr(self, "llm_route7_update_map_frame"):
            self.llm_route7_update_map_frame = None
        if not hasattr(self, "llm_route7_update_map_preview_label"):
            self.llm_route7_update_map_preview_label = None
        if not hasattr(self, "llm_route7_update_map_preview_photo"):
            self.llm_route7_update_map_preview_photo = None
        if not hasattr(self, "llm_route7_task_switch_canvas"):
            self.llm_route7_task_switch_canvas = None
        if not hasattr(self, "llm_route7_task_switch_text"):
            self.llm_route7_task_switch_text = _route5_string_var("Task switch: n/a")
        if not hasattr(self, "llm_route7_update_map_layer_combo"):
            self.llm_route7_update_map_layer_combo = None
        if not hasattr(self, "llm_route7_update_map_after_id"):
            self.llm_route7_update_map_after_id = None
        if not hasattr(self, "llm_route7_map_layer_var"):
            self.llm_route7_map_layer_var = _route5_string_var("z_300")
        if not hasattr(self, "llm_route7_map_status_var"):
            self.llm_route7_map_status_var = _route5_string_var("Route V7 Map: idle")

    def route5_status_label(self) -> str:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        label = str(state.get("route_window_label", "V5") or "V5").strip().upper()
        return label if label.startswith("V") else "V5"

    def route5_status_prefix(self) -> str:
        return f"LLM Route {self.route5_status_label()}"

    def route5_float_param(self, var: tk.StringVar, default: float, *, min_value: float, max_value: float) -> float:
        try:
            value = float(var.get().strip())
        except Exception:
            value = float(default)
        return max(float(min_value), min(float(max_value), float(value)))

    def route5_nav_config(self) -> Dict[str, float]:
        self.ensure_route5_state()
        return {
            "move_tick_ms": self.route5_float_param(self.llm_route5_move_tick_ms_var, 150.0, min_value=50.0, max_value=2000.0),
            "nav_step_cm": self.route5_float_param(self.llm_route5_nav_step_cm_var, 20.0, min_value=5.0, max_value=200.0),
            "reach_tol_cm": self.route5_float_param(self.llm_route5_reach_tol_cm_var, 60.0, min_value=5.0, max_value=500.0),
            "z_tol_cm": self.route5_float_param(self.llm_route5_z_tol_cm_var, 40.0, min_value=5.0, max_value=500.0),
            "yaw_tol_deg": self.route5_float_param(self.llm_route5_yaw_tol_deg_var, 10.0, min_value=1.0, max_value=90.0),
            "max_stage_s": self.route5_float_param(self.llm_route5_max_stage_s_var, 90.0, min_value=5.0, max_value=900.0),
        }

    def route5_sensing_config(self) -> Dict[str, Any]:
        self.ensure_route5_state()
        interval_s = self.route5_float_param(self.llm_route5_sensing_interval_s_var, 1.0, min_value=0.2, max_value=30.0)
        route7_primary = self.route5_event_is_route7({})
        representation_path = str(self.default_route7_or31_model_path() if route7_primary else self.llm_route5_representation_model_var.get() or "")
        return {
            "sensing_interval_s": float(interval_s),
            "representation_model_path": representation_path,
            "representation_source": "or3_1" if route7_primary else "or2",
            "or2_model_path": str(self.llm_route5_representation_model_var.get() or ""),
            "or3_1_model_path": str(self.default_route7_or31_model_path()),
            "llm_strategy_mode": LLM_STRATEGY_METHOD_ID,
            "capture_processing": "minimal",
        }

    def route5_oa3_config(self) -> Dict[str, Any]:
        self.ensure_route5_state()
        route7_primary = self.route5_event_is_route7({})
        return {
            "or2_model_path": str(self.llm_route5_representation_model_var.get() or self.default_route5_or2_model_path()),
            "or3_1_model_path": str(self.default_route7_or31_model_path()),
            "representation_source": "or3_1" if route7_primary else "or2",
            "oa3_plan_path": str(self.llm_route5_oa3_plan_var.get() or self.default_route5_oa3_plan_path()),
            "method": "obstacle_representation_direction_rule_v1",
            "fallback_policy": "route5_depth_pointcloud_rule_on_or2_unavailable",
        }

    def route5_scan_boundary_policy(self) -> Dict[str, Any]:
        max_yaw = min(45.0, float(LLM_ROUTE3_PANORAMA_MAX_YAW_DELTA_DEG))
        yaw_offset = min(35.0, max_yaw)
        return {
            "source": "route5_scan_boundary_compaction_v1",
            "applies_to": "open_llm_route_window_5_only",
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

    def route5_scan_axis_key(self, facade: str) -> str:
        facade = str(facade or "").strip().lower()
        return "x" if facade in {"south", "north"} else "y"

    def route5_scan_axis_value(self, point: Dict[str, Any], facade: str) -> Optional[float]:
        key = self.route5_scan_axis_key(facade)
        value = self._as_float_or_none(point.get(key))
        return None if value is None else float(value)

    def route5_safe_axis_bounds_for_points(
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
            axis = self.route5_scan_axis_value(point, facade)
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

    def route5_house_facade_axis_bounds(self, house_id: str, facade: str) -> Tuple[Optional[float], Optional[float]]:
        bbox = self.house_world_bbox_for_id(str(house_id or ""))
        if not bbox:
            return None, None
        try:
            axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
            return min(float(axis_min), float(axis_max)), max(float(axis_min), float(axis_max))
        except Exception:
            return None, None

    def route5_effective_scan_axis_bounds(
        self,
        points: List[Dict[str, Any]],
        *,
        house_id: str,
        facade: str,
    ) -> Dict[str, Any]:
        safe_min, safe_max = self.route5_safe_axis_bounds_for_points(points, facade)
        house_min, house_max = self.route5_house_facade_axis_bounds(house_id, facade)
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

    def route5_boundary_axis_targets(self, axis_min: float, axis_max: float, raw_unique_count: int) -> List[float]:
        low = float(min(axis_min, axis_max))
        high = float(max(axis_min, axis_max))
        if abs(high - low) <= 1e-6:
            return [round(low, 2)]
        max_count = int(self.route5_scan_boundary_policy()["max_physical_axis_samples_per_band"])
        count = max(2, min(max_count, max(1, int(raw_unique_count))))
        if count >= 3 and abs(high - low) >= 1.0:
            return [round(low + (high - low) * float(idx) / float(count - 1), 2) for idx in range(count)]
        return [round(low, 2), round(high, 2)]

    def route5_apply_axis_pose_to_scan_point(
        self,
        point: Dict[str, Any],
        *,
        house_id: str,
        facade: str,
        axis_value: float,
    ) -> Dict[str, Any]:
        item = dict(point)
        axis_key = self.route5_scan_axis_key(facade)
        standoff = self._as_float_or_none(item.get("standoff_cm"))
        z_cm = self._as_float_or_none(item.get("z"))
        bbox = self.house_world_bbox_for_id(house_id)
        if bbox and standoff is not None and z_cm is not None:
            pose = self.route2_facade_pose_from_axis(bbox, facade, float(axis_value), float(standoff), float(z_cm))
            item.update(pose)
        else:
            item[axis_key] = round(float(axis_value), 2)
        return item

    def route5_boundary_yaw_supplement_meta(self, base_yaw_deg: float, boundary_role: str) -> Dict[str, Any]:
        policy = self.route5_scan_boundary_policy()
        offset_abs = float(policy["boundary_yaw_offset_deg"])
        offset = -offset_abs if str(boundary_role) == "left_boundary" else offset_abs
        offset = max(-float(policy["boundary_yaw_max_abs_offset_deg"]), min(float(policy["boundary_yaw_max_abs_offset_deg"]), offset))
        supplement_yaw = self._normalize_angle_deg(float(base_yaw_deg) + float(offset))
        return {
            "enabled": True,
            "source": "route5_boundary_yaw_supplement",
            "coverage_mode": "yaw_in_place_no_lateral_overrun",
            "base_yaw_deg": round(float(base_yaw_deg), 2),
            "offset_deg": round(float(offset), 2),
            "supplement_yaw_deg": round(float(supplement_yaw), 2),
            "max_abs_offset_deg": float(policy["boundary_yaw_max_abs_offset_deg"]),
        }

    def route5_compact_boundary_scan_points(
        self,
        points: List[Dict[str, Any]],
        *,
        house_id: str,
        facade: str,
    ) -> Dict[str, Any]:
        policy = self.route5_scan_boundary_policy()
        facade = str(facade or "").strip().lower()
        axis_key = self.route5_scan_axis_key(facade)
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
                self.route5_scan_axis_value(item, facade) if self.route5_scan_axis_value(item, facade) is not None else 0.0,
                int(item.get("local_scan_index", 0) or 0),
            ))
            raw_axes = [
                round(float(axis), 2)
                for axis in (self.route5_scan_axis_value(point, facade) for point in group)
                if axis is not None
            ]
            raw_unique_axes = sorted(set(raw_axes))
            raw_physical_count += len(raw_unique_axes)
            bounds = self.route5_effective_scan_axis_bounds(group, house_id=house_id, facade=facade)
            axis_min = bounds.get("axis_min")
            axis_max = bounds.get("axis_max")
            if axis_min is None or axis_max is None:
                compacted.extend(dict(point) for point in group)
                physical_axis_sample_count += len(raw_unique_axes)
                continue
            targets = self.route5_boundary_axis_targets(float(axis_min), float(axis_max), len(raw_unique_axes))
            physical_axis_sample_count += len(targets)
            for target_index, target_axis in enumerate(targets):
                template = min(
                    group,
                    key=lambda point: abs(float(self.route5_scan_axis_value(point, facade) or target_axis) - float(target_axis)),
                )
                item = deepcopy(template)
                item["source_original_scan_id"] = template.get("scan_id", "")
                item["source_original_local_scan_index"] = template.get("local_scan_index")
                item = self.route5_apply_axis_pose_to_scan_point(item, house_id=house_id, facade=facade, axis_value=float(target_axis))
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
                item["route5_scan_boundary_policy"] = dict(policy)
                item["route5_axis_key"] = axis_key
                item["route5_axis_value"] = round(float(target_axis), 2)
                item["route5_axis_ratio"] = round(float(target_index) / float(max(1, len(targets) - 1)), 4)
                item["route5_compacted_from_raw_count"] = len(group)
                item["route5_raw_physical_axis_sample_count"] = len(raw_unique_axes)
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
                    self.route5_scan_axis_value(template, facade) is not None
                    and abs(float(self.route5_scan_axis_value(template, facade) or target_axis) - float(target_axis)) > 0.01
                )
                item["local_scan_index"] = next_local_index
                next_local_index += 1
                if boundary_role in {"left_boundary", "right_boundary"}:
                    yaw_meta = self.route5_boundary_yaw_supplement_meta(float(item.get("yaw_deg", item.get("yaw", 0.0)) or 0.0), boundary_role)
                    yaw_meta["supplement_record_planned"] = True
                    item["yaw_supplement"] = yaw_meta
                    compacted.append(item)

                    supplement = deepcopy(item)
                    supplement["view_type"] = "boundary_yaw_supplement_scan"
                    supplement["semantic_region"] = f"{boundary_role}_yaw_supplement"
                    supplement["is_yaw_supplement"] = True
                    supplement["yaw_deg"] = yaw_meta["supplement_yaw_deg"]
                    supplement["capture_trigger"] = "arrive_boundary_yaw_hover_capture"
                    supplement["route5_boundary_yaw_source_scan_id"] = item.get("scan_id", "")
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
            bounds = self.route5_effective_scan_axis_bounds([item], house_id=house_id, facade=facade)
            axis_min = bounds.get("axis_min")
            axis_max = bounds.get("axis_max")
            axis_value = self.route5_scan_axis_value(item, facade)
            if axis_min is not None and axis_max is not None and axis_value is not None:
                low = min(float(axis_min), float(axis_max))
                high = max(float(axis_min), float(axis_max))
                clamped = max(low, min(high, float(axis_value)))
                if abs(clamped - float(axis_value)) > 0.01:
                    item = self.route5_apply_axis_pose_to_scan_point(item, house_id=house_id, facade=facade, axis_value=clamped)
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
                item["route5_axis_key"] = axis_key
                item["route5_axis_value"] = round(clamped, 2)
            item["route5_scan_boundary_policy"] = dict(policy)
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

    def route5_write_merged_scan_points(
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
                facade_policy = payload.get("route5_scan_boundary_policy")
                if facade and isinstance(facade_policy, dict):
                    facade_policies[facade] = facade_policy
        payload = {
            "schema": "facade_v5_or2_fused_global_scan_points",
            "target_house_id": house_id,
            "scan_points": points,
            "facade_counts": facade_counts,
            "total_scan_count": len(points),
            "route5_scan_boundary_policy": policy or self.route5_scan_boundary_policy(),
            "route5_scan_boundary_policy_by_facade": facade_policies,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(output_dir / "scan_points.json", payload)
        return points

    def route5_plan_facade_scan_current(self) -> Dict[str, Any]:
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        if not analysis:
            analysis = self.route2_fallback_facade_analysis("Route V5 used fallback before VLM analysis.")
        raw_points = self.route2_generate_facade_scan_points(analysis)
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            raise RuntimeError("missing facade output directory")
        compacted = self.route5_compact_boundary_scan_points(raw_points, house_id=house_id, facade=facade)
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
            "schema": "facade_v5_or2_fused_scan_plan",
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "observation_point": state.get("observation_point", {}),
            "next_facade_hint": next_hint,
            "facade_analysis": analysis,
            "scan_points": points,
            "scan_point_validation_report": validation,
            "route_blocked_by_safety": not bool(validation.get("valid", False)),
            "route5_scan_boundary_policy": compacted.get("policy", self.route5_scan_boundary_policy()),
            "route5_scan_counts": counts,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(facade_dir / "facade_search_plan.json", search_plan)
        merged_points = self.route5_write_merged_scan_points(
            output_dir,
            house_id,
            policy=search_plan["route5_scan_boundary_policy"],
        )
        self.route2_update_state(facade_analysis=analysis, facade_search_plan=search_plan, facade_scan_points=points, validation_report=validation)
        self.route2_write_state_artifact()
        self.route5_update_state(
            route5_scan_boundary_policy=search_plan["route5_scan_boundary_policy"],
            last_scan_plan_counts=counts,
        )
        self.route5_write_state_artifact()
        return {
            "search_plan": search_plan,
            "points": points,
            "validation": validation,
            "merged_points": merged_points,
            "boundary_policy": search_plan["route5_scan_boundary_policy"],
            "scan_counts": counts,
        }

    def make_route5_fused_output_dir(self, target_house_id: str) -> Path:
        override = getattr(self, "llm_route5_output_root_override", None)
        root = Path(override) if override is not None else self.resolve_project_path("llm_route5_fusion_runs")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_autosearch_v5_or2_fused_{timestamp}"
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

    def make_route7_fused_output_dir(self, target_house_id: str) -> Path:
        override = getattr(self, "llm_route7_output_root_override", None)
        root = Path(override) if override is not None else self.resolve_project_path("llm_route_7_fusion_runs")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_autosearch_v7_or3_1_fused_{timestamp}"
        root.mkdir(parents=True, exist_ok=True)
        candidate = root / base_name
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = root / f"{base_name}_{suffix}"
        for subdir in ("frames", "open3d_frames", "reconstruction", "facade_observations", "map"):
            (candidate / subdir).mkdir(parents=True, exist_ok=True)
        return candidate

    def route5_state_output_dir(self) -> Optional[Path]:
        self.ensure_route5_state()
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        raw = str(state.get("output_dir", "") or "")
        if not raw:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        if str(state.get("route_window_label", "") or "").strip().upper() == "V7" or "llm_route_7_fusion_runs" in str(path).replace("/", "\\"):
            (path / "open3d_frames").mkdir(parents=True, exist_ok=True)
        (path / "reconstruction").mkdir(parents=True, exist_ok=True)
        (path / "facade_observations").mkdir(parents=True, exist_ok=True)
        return path

    def route5_update_state(self, **updates: Any) -> Dict[str, Any]:
        self.ensure_route5_state()
        state = dict(self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {})
        state.update(updates)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route5_state = state
        return state

    def route5_write_state_artifact(self) -> None:
        output_dir = self.route5_state_output_dir()
        if output_dir is None:
            return
        self.write_json_artifact(output_dir / "route5_fusion_state.json", self.llm_route5_state)

    def route5_log_event(self, output_dir: Optional[Path], event_type: str, payload: Dict[str, Any]) -> None:
        if output_dir is None:
            return
        row = {
            "event_type": str(event_type),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "payload": self.route5_json_safe(payload),
        }
        self.append_jsonl(output_dir / "route5_fusion_events.jsonl", row)

    def route5_json_safe(self, value: Any, *, max_string: int = 4000) -> Any:
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            return {
                "array_shape": list(arr.shape),
                "array_dtype": str(arr.dtype),
                "array_mean": float(np.mean(arr)) if arr.size else 0.0,
                "array_max": float(np.max(arr)) if arr.size else 0.0,
            }
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                key_lower = key_text.lower()
                if key_lower in {"api_key", "apikey", "authorization"} or "api_key" in key_lower:
                    continue
                result[key_text] = self.route5_json_safe(item, max_string=max_string)
            return result
        if isinstance(value, list):
            return [self.route5_json_safe(item, max_string=max_string) for item in value]
        if isinstance(value, tuple):
            return [self.route5_json_safe(item, max_string=max_string) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return str(value)
        if isinstance(value, str) and len(value) > max_string:
            return value[:max_string] + "...<truncated>"
        return value

    def route5_next_llm_call_id(self, call_type: str) -> str:
        self.ensure_route5_state()
        seq = int(self.llm_route5_state.get("llm_call_seq", 0) or 0) + 1
        self.route5_update_state(llm_call_seq=seq)
        safe_type = re.sub(r"[^a-zA-Z0-9_]+", "_", str(call_type or "llm")).strip("_").lower() or "llm"
        return f"route5_llm_{seq:04d}_{safe_type}"

    def route5_log_llm_call(
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
        call_id = self.route5_next_llm_call_id(call_type)
        raw_text = str((response if isinstance(response, dict) else {}).get("raw_text", "") or "")
        record = {
            "call_id": call_id,
            "call_type": str(call_type or "llm"),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "frame_id": frame_id,
            "facade": str(facade or (context if isinstance(context, dict) else {}).get("facade", "") or ""),
            "target_id": str(target_id or (context if isinstance(context, dict) else {}).get("target_id", "") or ""),
            "context": self.route5_json_safe(context if isinstance(context, dict) else {}),
            "response": self.route5_json_safe(response if isinstance(response, dict) else {}),
            "decision": self.route5_json_safe(decision if isinstance(decision, dict) else {}),
            "raw_text_preview": raw_text[:1000],
        }
        if output_dir is not None:
            self.append_jsonl(output_dir / "route5_llm_calls.jsonl", record)
        return {
            "call_id": call_id,
            "call_type": record["call_type"],
            "frame_id": frame_id,
            "facade": record["facade"],
            "target_id": record["target_id"],
            "decision_reason": str((decision if isinstance(decision, dict) else {}).get("reason", "") or ""),
            "raw_text_preview": record["raw_text_preview"],
        }

    def route5_frame_dir_for_event(self, output_dir: Path, event: Dict[str, Any]) -> Path:
        capture_dir = str((event if isinstance(event, dict) else {}).get("capture_dir", "") or "")
        if capture_dir:
            return Path(capture_dir)
        frame_id = int(float((event if isinstance(event, dict) else {}).get("frame_id", 0) or 0))
        return output_dir / "frames" / f"frame_{frame_id:06d}"

    def route5_strategy_cache_summary(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        strategy = strategy if isinstance(strategy, dict) else {}
        return {
            "key": str(strategy.get("strategy_cache_key", "") or ""),
            "hit": bool(strategy.get("strategy_cache_hit", False)),
            "source": str(strategy.get("strategy_source", "") or ""),
            "reason": str(strategy.get("strategy_cache_reason", "") or ""),
        }

    def route5_write_frame_decision(
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
        or2_prediction = event.get("or2_prediction", {}) if isinstance(event.get("or2_prediction"), dict) else {}
        or2_rule = event.get("or2_rule", {}) if isinstance(event.get("or2_rule"), dict) else {}
        representation_source = str(event.get("route7_primary_representation", "") or ("or3_1" if bool(event.get("or3_1_primary", False)) else "or2"))
        selected_reason = str(event.get("selected_action_reason", "") or gate.get("reason", "") or "")
        decision = {
            "schema": "route5_frame_decision_v1",
            "frame_id": frame_id,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "stage": str(event.get("route5_stage", event.get("stage", "")) or ""),
            "facade": str(event.get("facade", "") or ""),
            "target_id": str(event.get("target_id", "") or ""),
            "current_pose": self.route5_json_safe(event.get("current_pose", event.get("pose", {}))),
            "target_pose": self.route5_json_safe(event.get("target_waypoint", event.get("goal_pose", {}))),
            "lookahead_plan": self.route5_json_safe(event.get("depth_lookahead_plan", {})),
            "pointcloud_summary": self.route5_json_safe(event.get("pointcloud_summary", event.get("depth_obstacle_summary", {}))),
            "representation_prediction": self.route5_json_safe(event.get("representation_prediction", {})),
            "representation_source": representation_source,
            "route7_primary_representation": str(event.get("route7_primary_representation", "") or ""),
            "or2": {
                "front_risk_state": str(or2_prediction.get("front_risk_state", event.get("or2_front_risk_state", "")) or ""),
                "can_forward": bool(or2_prediction.get("can_forward", event.get("or2_can_forward", False))),
                "must_stop": bool(or2_prediction.get("must_stop", event.get("or2_must_stop", False))),
                "corridor_risks": self.route5_json_safe(or2_rule.get("corridor_risks", event.get("or2_corridor_risks", {}))),
                "selected_direction": str(or2_rule.get("selected_direction", event.get("or2_selected_direction", "")) or ""),
                "candidate_action_scores": self.route5_json_safe(or2_rule.get("candidate_action_scores", event.get("or2_candidate_action_scores", {}))),
                "risk_overlay_path": str(or2_prediction.get("risk_overlay_path", event.get("or2_risk_overlay_path", "")) or ""),
                "prediction_json_path": str(or2_prediction.get("prediction_json_path", event.get("or2_prediction_json_path", "")) or ""),
                "rule_reason": str(or2_rule.get("reason", event.get("or2_rule_reason", "")) or ""),
                "selected_action_rejected_reason": str(event.get("or2_selected_action_rejected_reason", "") or ""),
                "safe_alternative_action": str(event.get("safe_alternative_action", "") or ""),
                "candidate_safety_scores": self.route5_json_safe(event.get("candidate_safety_scores", {})),
            },
            "avoidance_gate": self.route5_json_safe(event.get("avoidance_gate", {})),
            "selected_action": str(event.get("selected_action", "") or ""),
            "or2_selected_action_rejected_reason": str(event.get("or2_selected_action_rejected_reason", "") or ""),
            "safe_alternative_action": str(event.get("safe_alternative_action", "") or ""),
            "candidate_safety_scores": self.route5_json_safe(event.get("candidate_safety_scores", {})),
            "selected_action_reason": selected_reason,
            "decision_reason": str(selected_reason or event.get("goal_completion_reason", "") or ""),
            "risk_state": str(event.get("risk_state", "") or ""),
            "arrival_policy": str(event.get("arrival_policy", "") or ""),
            "near_obstacle_reached": bool(event.get("near_obstacle_reached", False)),
            "front_depth_cm": self.route5_event_float(event.get("front_depth_cm"), default=0.0),
            "distance_to_goal_cm": self.route5_event_float(event.get("distance_to_goal_cm", event.get("post_distance_to_goal_cm", 0.0)), default=0.0),
            "arrival_reason": str(event.get("arrival_reason", event.get("goal_completion_reason", "")) or ""),
            "capture_guard_blocked": bool(event.get("capture_guard_blocked", False)),
            "capture_guard_passed": bool(event.get("capture_guard_passed", False)),
            "capture_guard_recomputed_arrival": bool(
                event.get(
                    "capture_guard_recomputed_arrival",
                    (event.get("capture_guard", {}) if isinstance(event.get("capture_guard"), dict) else {}).get("capture_guard_recomputed_arrival", False),
                )
            ),
            "capture_guard_repeat_count": int(
                self.route5_event_float(
                    event.get(
                        "capture_guard_repeat_count",
                        (event.get("capture_guard", {}) if isinstance(event.get("capture_guard"), dict) else {}).get("capture_guard_repeat_count", 0),
                    ),
                    default=0.0,
                )
            ),
            "next_retry_action": str(
                event.get(
                    "next_retry_action",
                    (event.get("capture_guard", {}) if isinstance(event.get("capture_guard"), dict) else {}).get("next_retry_action", ""),
                )
                or ""
            ),
            "capture_guard": self.route5_json_safe(event.get("capture_guard", {})),
            "original_target_pose": self.route5_json_safe(event.get("original_target_pose", {})),
            "runtime_target_pose": self.route5_json_safe(event.get("runtime_target_pose", event.get("target_waypoint", {}))),
            "capture_pose": self.route5_json_safe(event.get("capture_pose", event.get("current_pose", {}))),
            "target_reset_candidate": self.route5_json_safe(event.get("target_reset_candidate", {})),
            "target_reset_applied": bool(event.get("target_reset_applied", False)),
            "reset_reason": str(event.get("reset_reason", "") or ""),
            "plan_deviation": self.route5_json_safe(event.get("plan_deviation", {})),
            "plan_repair_applied": bool(event.get("plan_repair_applied", False)),
            "plan_repair": self.route5_json_safe(event.get("plan_repair", event.get("plan_repair_record", {}))),
            "repair_action": str(event.get("repair_action", "") or ""),
            "repair_reason": str(event.get("repair_reason", "") or ""),
            "route7_soft_obstacle_policy": self.route5_json_safe(event.get("route7_soft_obstacle_policy", {})),
            "route7_map_route_replan_request": bool(event.get("route7_map_route_replan_request", False)),
            "route7_navigation_point_reset_request": bool(event.get("route7_navigation_point_reset_request", False)),
            "forward_blocked": bool(event.get("route7_forward_blocked", event.get("forward_blocked", False))),
            "yaw_error_deg": self.route5_event_float(event.get("route7_yaw_error_deg", event.get("yaw_error_deg", 0.0)), default=0.0),
            "route7_realtime_route_plan": self.route5_json_safe(event.get("route7_realtime_route_plan", {})),
            "route7_route_decision_reason": str(event.get("route7_route_decision_reason", "") or ""),
            "nominal_payload": self.route5_json_safe(event.get("nominal_action", event.get("nominal_payload", {}))),
            "selected_action_payload": self.route5_json_safe(event.get("selected_action_payload", {})),
            "final_payload": self.route5_json_safe(final_payload if isinstance(final_payload, dict) else event.get("selected_action_payload", {})),
            "collision_state": bool(event.get("collision_state", False)),
            "avoidance_failed": bool(event.get("avoidance_failed", False)),
            "llm_calls": self.route5_json_safe(event.get("llm_call_refs", [])),
            "strategy_cache": self.route5_strategy_cache_summary(strategy),
            "strategy_source": str(strategy.get("strategy_source", "") or ""),
            "memory_updates": self.route5_json_safe(event.get("house_memory_updates", [])),
        }
        frame_dir = self.route5_frame_dir_for_event(output_dir, event)
        frame_dir.mkdir(parents=True, exist_ok=True)
        self.write_json_artifact(frame_dir / "decision.json", decision)
        self.append_jsonl(output_dir / "route5_frame_decisions.jsonl", decision)
        is_route7 = (
            str(event.get("route_window_label", "") or "").strip().upper() == "V7"
            or "llm_route_7_fusion_runs" in str(output_dir).replace("/", "\\")
            or bool(decision.get("route7_soft_obstacle_policy", {}))
        )
        if is_route7:
            concise = {
                "schema": "route7_decision_log_v1",
                "frame_id": frame_id,
                "created_at": decision["created_at"],
                "stage": decision["stage"],
                "facade": decision["facade"],
                "target_id": decision["target_id"],
                "current_pose": decision["current_pose"],
                "target_pose": decision["target_pose"],
                "front_depth_cm": decision["front_depth_cm"],
                "distance_to_goal_cm": decision["distance_to_goal_cm"],
                "risk_state": decision["or2"]["front_risk_state"],
                "representation_source": decision["representation_source"],
                "can_forward": decision["or2"]["can_forward"],
                "must_stop": decision["or2"]["must_stop"],
                "or2_selected_direction": decision["or2"]["selected_direction"],
                "selected_action": decision["selected_action"],
                "selected_action_payload": decision["selected_action_payload"],
                "final_payload": decision["final_payload"],
                "reason": decision["selected_action_reason"] or decision["decision_reason"],
                "or2_rule_reason": decision["or2"]["rule_reason"],
                "rejected_reason": decision["or2_selected_action_rejected_reason"],
                "safe_alternative_action": decision["safe_alternative_action"],
                "route7_soft_obstacle_policy": decision["route7_soft_obstacle_policy"],
                "route7_map_route_replan_request": decision["route7_map_route_replan_request"],
                "route7_navigation_point_reset_request": decision["route7_navigation_point_reset_request"],
                "forward_blocked": decision["forward_blocked"],
                "yaw_error_deg": decision["yaw_error_deg"],
                "route7_realtime_route_plan": decision["route7_realtime_route_plan"],
                "route7_route_decision_reason": decision["route7_route_decision_reason"],
                "collision_state": decision["collision_state"],
            }
            self.append_jsonl(output_dir / "route7_frame_decision_log.jsonl", concise)
        return decision

    def route5_write_safety_blocked_frame_decision(
        self,
        output_dir: Path,
        event: Dict[str, Any],
        payload: Dict[str, Any],
        safety: Dict[str, Any],
    ) -> Dict[str, Any]:
        event = dict(event if isinstance(event, dict) else {})
        event["safety_blocked_before_move"] = True
        event["selected_action_reason"] = "safety_blocked_before_move"
        event["selected_action_payload"] = dict(payload if isinstance(payload, dict) else {})
        event["final_payload"] = dict(payload if isinstance(payload, dict) else {})
        event["safety"] = self.route5_json_safe(safety if isinstance(safety, dict) else {})
        event = self.route5_normalize_avoidance_event(event)
        decision = self.route5_write_frame_decision(output_dir, event, final_payload=payload)
        if bool(event.get("avoidance_active", False)) or bool(event.get("collision_state", False)):
            self.append_jsonl(output_dir / "avoidance_events.jsonl", self.route5_json_safe(event))
            self.route5_write_avoidance_summary(output_dir, status="running")
        return decision

    def route5_set_stage(
        self,
        stage: str,
        *,
        output_dir: Optional[Path] = None,
        facade: str = "",
        target: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> None:
        self.ensure_route5_state()
        updates: Dict[str, Any] = {"stage": stage}
        if facade:
            updates["current_facade"] = facade
        if target:
            updates["target_pose"] = target
        if error:
            updates["last_error"] = error
        self.route5_update_state(**updates)
        self.route5_write_state_artifact()
        self.route5_log_event(output_dir or self.route5_state_output_dir(), "stage", {"stage": stage, "facade": facade, "message": message})
        try:
            self.root.after(0, lambda s=stage: self.llm_route5_stage_var.set(f"Stage: {s}"))
            if facade:
                self.root.after(0, lambda f=facade: self.llm_route5_active_var.set(f"Active: facade={f}"))
            if target:
                self.root.after(
                    0,
                    lambda t=target: self.llm_route5_target_var.set(
                        f"Target: x={float(t.get('x', 0.0)):.1f} y={float(t.get('y', 0.0)):.1f} z={float(t.get('z', 0.0)):.1f}"
                    ),
                )
            if error:
                self.root.after(0, lambda e=error: self.llm_route5_error_var.set(f"Error: {e.get('reason', e.get('status', 'CHECK'))}"))
            if message:
                self.root.after(0, lambda m=message, p=self.route5_status_prefix(): self.llm_route5_status_var.set(f"{p}: {m}"))
        except Exception:
            pass

    def route5_initialize_run(
        self,
        target_house_id: str,
        *,
        force_new: bool = False,
        output_dir_override: Optional[Path] = None,
        route_window_label: str = "V5",
    ) -> Path:
        self.ensure_route5_state()
        current = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        current_target = str(current.get("target_house_id", "") or "")
        output_dir = self.route5_state_output_dir()
        created_new_run = False
        label = str(route_window_label or "V5").strip().upper() or "V5"
        if force_new or output_dir_override is not None or output_dir is None or current_target != str(target_house_id):
            created_new_run = True
            output_dir = Path(output_dir_override) if output_dir_override is not None else (
                self.make_route7_fused_output_dir(target_house_id) if label == "V7" else self.make_route5_fused_output_dir(target_house_id)
            )
            for subdir in ("frames", "reconstruction", "facade_observations"):
                (output_dir / subdir).mkdir(parents=True, exist_ok=True)
            if label == "V7":
                (output_dir / "map").mkdir(parents=True, exist_ok=True)
                (output_dir / "open3d_frames").mkdir(parents=True, exist_ok=True)
            self.route5_local_obstacle_map = LocalObstacleMap(self.route5_local_obstacle_config())
            self.route5_local_obstacle_output_dir = output_dir
            self.llm_route5_completed_facades = set()
            self.llm_route5_blocked_facades = set()
            self.llm_route5_state = {
                "mode": "route7_llm_route_oa3_or3_1_fusion" if label == "V7" else "route5_llm_route_oa3_or2_fusion",
                "route_window_label": label,
                "target_house_id": target_house_id,
                "output_dir": str(output_dir),
                "route7_map_output_dir": str(output_dir) if label == "V7" else "",
                "route7_primary_representation": "or3_1" if label == "V7" else "",
                "or3_1_model_path": str(self.default_route7_or31_model_path()) if label == "V7" else "",
                "stage": "INIT_RUN",
                "nav_config": self.route5_nav_config(),
                "sensing_config": self.route5_sensing_config(),
                "oa3_config": self.route5_oa3_config(),
                "or2_monitor": {},
                "last_or2_event": {},
                "last_obstacle_event": {},
                "last_target_reset": {},
                "target_reset_count": 0,
                "reset_route_points": [],
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
                "route5_fusion_events.jsonl",
            "route5_navigation_plan.jsonl",
            "route5_movement_trace.jsonl",
            "route5_frame_decisions.jsonl",
            "route5_or2_risk_events.jsonl",
            "route5_target_resets.jsonl",
            "route5_plan_deviation_repairs.jsonl",
            "route5_scan_postprocess_events.jsonl",
            "route5_capture_guard_events.jsonl",
            "route5_llm_calls.jsonl",
            "house_exploration_memory_events.jsonl",
            "avoidance_events.jsonl",
            "local_obstacle_map.jsonl",
            "local_3d_safety_events.jsonl",
        ):
                (output_dir / artifact_name).touch(exist_ok=True)
            self.write_json_artifact(
                output_dir / "route5_or2_monitor_summary.json",
                {
                    "status": "initialized",
                    "frame_count": 0,
                    "state_counts": {},
                    "created_at": datetime.now().isoformat(timespec="milliseconds"),
                    "model_path": self.route5_oa3_config().get("or2_model_path", ""),
                },
            )
        self.llm_route2_state = {
            "mode": "facade_by_facade_vlm_v5_or2_fused",
            "target_house_id": target_house_id,
            "output_dir": str(output_dir),
            "facade": "",
            "facade_id": "",
            "completed_facades": sorted(self.llm_route5_completed_facades),
        }
        self.llm_route2_completed_facades = set(self.llm_route5_completed_facades)
        self.route5_write_state_artifact()
        if created_new_run:
            self.route5_write_avoidance_summary(output_dir, status="initialized")
        return output_dir

    def route5_task_facade_priority(self) -> List[str]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        plan = state.get("task_plan", {}) if isinstance(state.get("task_plan"), dict) else {}
        raw = plan.get("facade_priority", []) if isinstance(plan, dict) else []
        result = [str(item).strip().lower() for item in raw if str(item).strip().lower() in {"south", "east", "north", "west"}]
        for facade in ("south", "east", "north", "west"):
            if facade not in result:
                result.append(facade)
        return result

    def route5_analyze_task_plan(self, target_house_id: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
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
                    "fusion_mode": "route5_llm_route_oa3_or2_fusion",
                }
                response_payload = self.call_configured_llm_text(
                    system_prompt=(
                        "You are the high-level task planner for a UAV house search with local obstacle avoidance. "
                        "Return strict compact JSON only. Do not output low-level movement commands."
                    ),
                    user_prompt=(
                        "Plan ordered target houses and facade priority for the fused Route6_entrance_search + obstacle-avoidance run.\n"
                        f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}\n"
                        f"Expected JSON:\n{json.dumps(LLM_ROUTE3_TASK_PLAN_SCHEMA, indent=2, ensure_ascii=False)}"
                    ),
                    max_output_tokens=700,
                    json_schema=LLM_ROUTE3_TASK_PLAN_SCHEMA,
                )
                parsed = extract_json_object(str(response_payload.get("raw_text", "") or ""))
                if output_dir is not None:
                    ref = self.route5_log_llm_call(
                        output_dir,
                        "task_plan_analysis",
                        context,
                        response_payload,
                        decision=parsed if isinstance(parsed, dict) else {},
                    )
                    response_payload["route5_llm_call_ref"] = ref
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
                "planner_source": "route5_local_fallback_no_or_failed_api",
            }
        plan = self.route3_normalize_task_plan(parsed, target_house_id, task_text)
        self.route5_update_state(task_plan=plan)
        if output_dir is not None:
            self.write_json_artifact(output_dir / "route5_task_plan.json", {"plan": plan, "llm_response": response_payload})
            self.route5_log_event(output_dir, "task_plan_analysis", {"task_plan": plan, "llm_response": response_payload})
            self.route5_write_state_artifact()
        return plan

    def route5_rank_observation_candidates(
        self,
        target_house_id: str,
        candidates: List[Dict[str, Any]],
        completed: set[str],
        blocked: set[str],
        *,
        start_pose: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        current = dict(start_pose or self.route3_current_pose() or self.current_route_pose() or {})
        route7_mode = self.route5_event_is_route7({})
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        route7_output_dir_raw = str(state.get("route7_map_output_dir", state.get("output_dir", "")) or "")
        route7_output_dir = Path(route7_output_dir_raw) if route7_output_dir_raw else None
        priority = self.route5_task_facade_priority()
        priority_index = {facade: idx for idx, facade in enumerate(priority)}
        last_completed = self.route5_last_completed_facade(completed)
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
            if route7_mode:
                route7_attempts = base.get("route7_edge_observation_attempts", []) if isinstance(base.get("route7_edge_observation_attempts", []), list) else []
                if not route7_attempts:
                    route7_attempts = base.get("observation_attempts", []) if isinstance(base.get("observation_attempts", []), list) else []
                attempts = [dict(item) for item in route7_attempts if isinstance(item, dict)]
                if not attempts:
                    attempts = self.route7_edge_observation_attempts_for_facade(
                        target_house_id,
                        facade,
                        output_dir=route7_output_dir,
                    )
            else:
                attempts = self.route3_observation_attempts_for_facade(target_house_id, facade, base)
            ranked_attempts: List[Dict[str, Any]] = []
            for attempt in attempts:
                item = dict(attempt)
                if str(item.get("status", "") or "").strip().lower() == "blocked":
                    item["route5_navigation_status"] = "blocked"
                    item["route5_navigation_reason"] = str(
                        item.get("route7_observation_replan_reason", item.get("observation_block_reason", "observation_attempt_blocked")) or "observation_attempt_blocked"
                    )
                    item["route3_navigation_status"] = "blocked"
                    item["route3_navigation_plan"] = {"status": "blocked", "reason": item["route5_navigation_reason"]}
                    item["route5_navigation_plan"] = dict(item["route3_navigation_plan"])
                    item["route5_navigation_cost_cm"] = float("inf")
                    ranked_attempts.append(item)
                    continue
                pose = self.route3_target_pose_from_point(item)
                if not current:
                    item["route5_navigation_status"] = "unknown_no_current_pose"
                    item["route5_navigation_cost_cm"] = float(item.get("observation_selection_score", item.get("distance_to_uav_cm", 0.0)) or 0.0)
                    ranked_attempts.append(item)
                    continue
                if route7_mode:
                    plan = self.route7_plan_spatial_navigation_path(
                        current,
                        pose,
                        output_dir=route7_output_dir,
                        stage="NAV_TO_OBS",
                        target_id=str(item.get("target_id", item.get("label", "")) or ""),
                        target_house_id=target_house_id,
                    )
                else:
                    plan = self.route3_plan_navigation_waypoints(current, pose, target_house_id, grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM))
                item["route5_navigation_plan"] = plan
                item["route5_navigation_status"] = str(plan.get("status", "blocked") or "blocked")
                item["route5_navigation_reason"] = str(plan.get("reason", "") or "")
                item["route3_navigation_status"] = item["route5_navigation_status"]
                item["route3_navigation_plan"] = plan
                if plan.get("status") == "ok":
                    item["route5_navigation_cost_cm"] = round(self.route3_navigation_plan_cost_cm(current, plan), 2)
                    item["status"] = "planned"
                else:
                    item["route5_navigation_cost_cm"] = float("inf")
                    item["status"] = "blocked"
                    item["observation_block_reason"] = str(plan.get("reason", item.get("observation_block_reason", "navigation_blocked")) or "navigation_blocked")
                ranked_attempts.append(item)
            ranked_attempts.sort(
                key=lambda item: (
                    0 if item.get("route5_navigation_status") == "ok" else 1,
                    float(item.get("route5_navigation_cost_cm", float("inf"))),
                    int(item.get("observation_attempt_index", 999) or 999),
                )
            )
            feasible = next((item for item in ranked_attempts if item.get("route5_navigation_status") == "ok"), None)
            selected = dict(feasible or (ranked_attempts[0] if ranked_attempts else base))
            nav_cost = float(selected.get("route5_navigation_cost_cm", selected.get("distance_to_uav_cm", 0.0)) or 0.0)
            if not math.isfinite(nav_cost):
                nav_cost = 1_000_000.0
            transition_rank = self.route5_facade_transition_rank(facade, completed, last_completed_facade=last_completed)
            selected["route5_observation_rank_score"] = round(float(nav_cost), 2)
            selected["route5_selection_policy"] = "nearest_from_current_uav_pose"
            selected["route5_facade_priority_index"] = priority_index.get(facade, len(priority))
            selected["route5_facade_transition_rank"] = int(transition_rank)
            selected["route5_last_completed_facade"] = last_completed
            selected["selected_observation_attempt"] = dict(feasible or selected)
            selected["observation_attempts"] = ranked_attempts
            selected["observation_attempt_count"] = len(ranked_attempts)
            if feasible is None:
                selected["status"] = "blocked"
                selected["route5_navigation_status"] = str(selected.get("route5_navigation_status", "blocked") or "blocked")
            ranked.append(selected)
        ranked.sort(
            key=lambda item: (
                0 if item.get("route5_navigation_status") == "ok" and item.get("status") != "blocked" else 1,
                float(item.get("route5_observation_rank_score", 1_000_000.0)),
                int(item.get("route5_facade_priority_index", 99) or 99),
                int(item.get("route5_facade_transition_rank", 99) or 99),
            )
        )
        return ranked

    def route5_last_completed_facade(self, completed: set[str]) -> str:
        completed_set = {str(item).strip().lower() for item in completed if str(item).strip().lower()}
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        last = str(state.get("last_completed_facade", "") or "").strip().lower()
        if last in completed_set:
            return last
        order = state.get("completed_facade_order", []) if isinstance(state.get("completed_facade_order"), list) else []
        for item in reversed(order):
            facade = str(item or "").strip().lower()
            if facade in completed_set:
                return facade
        return ""

    def route5_facade_transition_rank(
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
            last = self.route5_last_completed_facade(completed_set)
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
            order = self.route5_task_facade_priority()
        if facade in order:
            return order.index(facade)
        return len(order)

    def route5_nav_result_current_pose(self, nav_result: Dict[str, Any]) -> Dict[str, float]:
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

    def route5_observation_rescue_candidate(
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
        current = self.route5_nav_result_current_pose(nav_result)
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
            "observation_attempt_source": "route5_rescue_current_pose",
            "observation_map_bounds": bounds_report,
            "observation_blocking_house_id": str(blocking.get("house_id", "") or ""),
            "observation_block_reason": "route5_rescue_allowed_non_target_clearance" if blocking else "",
            "observation_boundary_adjustment": {},
            "status": "planned",
            "route5_observation_rescue": True,
            "route5_rescue_capture_without_additional_navigation": True,
            "route5_rescue_reason": "navigation_blocked_near_facade_observe_from_current_pose",
            "route5_rescue_navigation_reason": reason,
            "route5_rescue_target_id": str(nav_result.get("target_id", "") or ""),
            "route5_rescue_original_observation": {
                "x": original_observation.get("x"),
                "y": original_observation.get("y"),
                "z": original_observation.get("z"),
                "yaw_deg": original_observation.get("yaw_deg", original_observation.get("yaw")),
                "standoff_cm": original_standoff,
                "target_x": original_observation.get("target_x"),
                "target_y": original_observation.get("target_y"),
            },
            "route5_rescue_current_pose": current,
        }

    def route5_observation_failure_retry_status(self, nav_result: Dict[str, Any]) -> Dict[str, Any]:
        nav_result = nav_result if isinstance(nav_result, dict) else {}
        reason = str(nav_result.get("reason", "") or "").strip()
        pose_error = nav_result.get("pose_error", {}) if isinstance(nav_result.get("pose_error"), dict) else {}
        distance = self.route5_event_float(pose_error.get("dist_xy_cm", pose_error.get("dist_3d_cm", 0.0)), default=0.0)
        retryable_reasons = {
            "non_target_house_clearance",
            "target_house_bbox",
            "replan_exhausted",
            "movement_failed",
            "navigation_plan_failed",
            "nav_timeout",
            "route7_map_route_replan_required",
            "target_reset_failed",
            "target_reset_limit_exhausted",
        }
        if reason in retryable_reasons:
            remote_block = reason == "non_target_house_clearance" and distance > 300.0
            return {
                "terminal": False,
                "status": "soft_blocked_retryable",
                "reason": reason,
                "retry_action": "try_next_observation_or_detour" if remote_block else "retry_with_rescue_or_degraded",
                "distance_to_goal_cm": round(float(distance), 3),
                "remote_blocked": bool(remote_block),
            }
        return {
            "terminal": True,
            "status": "terminal_blocked",
            "reason": reason or "observation_navigation_failed",
            "retry_action": "",
            "distance_to_goal_cm": round(float(distance), 3),
            "remote_blocked": False,
        }

    def route5_observation_rescue_decision(
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
            "planner_source": "route5_rescue_rule_fallback",
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
            output_dir = self.route5_state_output_dir()
            llm_ref = self.route5_log_llm_call(
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
                    "planner_source": "route5_rescue_llm",
                    "llm_response": response,
                    "llm_call_ref": llm_ref,
                }
        except Exception as exc:
            return {**fallback, "llm_error": str(exc)}
        return fallback

    def route5_try_observation_rescue(
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
        candidate = self.route5_observation_rescue_candidate(
            target_house_id=target_house_id,
            facade=facade,
            original_observation=original_observation,
            nav_result=nav_result,
        )
        decision = self.route5_observation_rescue_decision(
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
        self.route5_log_event(output_dir, "observation_rescue_candidate", rescue_payload)
        if not candidate or not bool(decision.get("accepted", False)):
            return {"status": "skipped", "reason": str(decision.get("next_action", "no_rescue_candidate") or "no_rescue_candidate"), **rescue_payload}
        self.apply_route2_observation_plan(
            target_house_id,
            candidate,
            ranked_candidates,
            status_label=f"v5 rescue observation {facade}",
        )
        result = {
            "status": "ok",
            "reason": "route5_observation_rescue_capture_current",
            "observation": candidate,
            "target_pose": self.route3_target_pose_from_point(candidate),
            "decision": decision,
            "source_navigation": nav_result,
        }
        self.route5_update_state(last_observation_rescue=result)
        self.route5_write_state_artifact()
        self.route5_log_event(output_dir, "observation_rescue_applied", result)
        return result

    def route5_degraded_observation_candidate(
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
        current = self.route5_nav_result_current_pose(nav_result) or (dict(fallback_pose) if isinstance(fallback_pose, dict) else {})
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
            "observation_attempt_source": "route5_degraded_current_pose",
            "observation_map_bounds": bounds_report,
            "observation_blocking_house_id": str(blocking.get("house_id", "") or ""),
            "observation_block_reason": "degraded_current_pose_with_non_target_clearance" if blocking else "",
            "status": "planned",
            "route5_degraded_observation": True,
            "route5_completion_status": "degraded_completed",
            "route5_degraded_reason": str(nav_result.get("reason", "navigation_failed") or "navigation_failed") if isinstance(nav_result, dict) else "navigation_failed",
            "route5_degraded_navigation": self.route5_json_safe(nav_result if isinstance(nav_result, dict) else {}),
            "route5_degraded_original_observation": self.route5_json_safe(original_observation if isinstance(original_observation, dict) else {}),
            "route5_degraded_current_pose": self.route5_json_safe(current),
        }

    def route5_mark_facade_degraded_completed(
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
        self.llm_route5_completed_facades.add(facade)
        self.llm_route5_blocked_facades.discard(facade)
        completion_status = dict(self.llm_route5_state.get("facade_completion_status", {}) if isinstance(self.llm_route5_state.get("facade_completion_status"), dict) else {})
        completion_status[facade] = "degraded_completed"
        completed_order = [
            str(item).strip().lower()
            for item in (
                self.llm_route5_state.get("completed_facade_order", [])
                if isinstance(self.llm_route5_state.get("completed_facade_order"), list)
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
            "observation": self.route5_json_safe(observation),
            "navigation": self.route5_json_safe(nav_result),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.route5_update_state(
            completed_facades=sorted(self.llm_route5_completed_facades),
            blocked_facades=sorted(self.llm_route5_blocked_facades),
            completed_facade_order=completed_order,
            last_completed_facade=facade,
            facade_completion_status=completion_status,
            last_degraded_facade_completion=result,
        )
        self.route5_update_house_memory(
            output_dir,
            target_house_id,
            facade,
            status="degraded_completed",
            reason=str(reason or "degraded_observation"),
            observation_attempt=observation,
            nav_result=nav_result,
            obstacle_label_conflict=(
                self.llm_route5_state.get("last_obstacle_event", {}).get("obstacle_label_conflict")
                if isinstance(self.llm_route5_state.get("last_obstacle_event"), dict)
                else None
            ),
            safe_observation_pose=observation,
        )
        self.route5_log_event(output_dir, "facade_degraded_completed", result)
        self.route5_write_state_artifact()
        return result

    def route5_decide_next_facade(
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
            and str(candidate.get("facade", "") or "") not in set(self.llm_route5_state.get("terminal_failed_facades", []) if isinstance(self.llm_route5_state.get("terminal_failed_facades"), list) else [])
        ]
        fallback = {
            "next_action": "select_facade" if available else "done",
            "target_facade": str(available[0].get("facade", "") or "") if available else "",
            "reason": "nearest_from_current_uav_pose" if available else "no_available_facade",
            "rescan_required": False,
            "stop_condition_met": not bool(available),
            "planner_source": "route5_rule_fallback",
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
                "fusion_mode": "route5_llm_route_oa3_or2_fusion",
            }
            response = self.call_configured_llm_text(
                system_prompt=(
                    "You are a high-level UAV house facade search supervisor for a fused Route6_entrance_search and obstacle-avoidance run. "
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
            output_dir = self.route5_state_output_dir()
            llm_ref = self.route5_log_llm_call(
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
                    parsed["planner_source"] = "route5_llm_high_level_rank_corrected"
                else:
                    parsed["planner_source"] = "route5_llm_high_level"
                parsed["llm_call_ref"] = llm_ref
                return parsed
        except Exception as exc:
            return {**fallback, "llm_error": str(exc)}
        return fallback

    def route5_rewrite_jsonl_artifact(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(self.route5_json_safe(row), ensure_ascii=False) + "\n")

    def route5_resolve_capture_dir(self, output_dir: Path, raw_capture_dir: Any) -> Optional[Path]:
        raw = str(raw_capture_dir or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = (output_dir / path).resolve()
        return path

    def route5_refresh_lidar_rows_for_file(
        self,
        path: Path,
        *,
        output_dir: Path,
        facade: str,
        result_cache: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        rows = self.read_jsonl_artifact(path)
        updated_rows: List[Dict[str, Any]] = []
        updated_count = 0
        for row in rows:
            item = dict(row if isinstance(row, dict) else {})
            if facade and str(item.get("facade", "") or "").strip().lower() != str(facade).strip().lower():
                updated_rows.append(item)
                continue
            capture_dir = self.route5_resolve_capture_dir(output_dir, item.get("capture_dir", ""))
            if capture_dir is None:
                updated_rows.append(item)
                continue
            if "capture_guard_passed" in item and not bool(item.get("capture_guard_passed", False)):
                item.update({"point_count": 0, "postprocess_status": "invalid_capture_pose", "postprocess_error": str(item.get("capture_guard_reason", "capture_guard_failed") or "capture_guard_failed")})
                updated_rows.append(item)
                continue
            cache_key = str(capture_dir.resolve())
            if cache_key not in result_cache:
                try:
                    capture_payload = {}
                    capture_json = capture_dir / "capture.json"
                    if capture_json.is_file():
                        capture_payload = json.loads(capture_json.read_text(encoding="utf-8"))
                    args = getattr(self, "args", None)
                    result_cache[cache_key] = flight.ensure_standard_world_cloud_for_capture(
                        capture_dir,
                        capture_payload=capture_payload if isinstance(capture_payload, dict) else {},
                        lidar_depth_projection=str(getattr(args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
                        min_depth_cm=float(getattr(args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
                        max_depth_cm=float(getattr(args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
                    )
                except Exception as exc:
                    result_cache[cache_key] = {"postprocess_status": "failed", "postprocess_error": f"{type(exc).__name__}: {exc}", "point_count": 0}
            result = result_cache.get(cache_key, {})
            point_count = int(result.get("point_count", item.get("point_count", 0)) or 0)
            item.update(
                {
                    "point_count": point_count,
                    "postprocess_status": str(result.get("postprocess_status", "done" if point_count > 0 else "failed") or ""),
                    "postprocess_error": str(result.get("postprocess_error", "") or ""),
                }
            )
            for key in (
                "point_cloud_world_standard_m_npy_path",
                "point_cloud_world_standard_m_ply_path",
                "projection_diagnostics_path",
                "depth_projection",
                "depth_projection_selected",
                "coordinate_frame",
                "coordinate_units",
            ):
                if result.get(key):
                    item[key] = result.get(key)
            updated_count += 1
            updated_rows.append(item)
        if rows:
            self.route5_rewrite_jsonl_artifact(path, updated_rows)
        return updated_rows, updated_count

    def route5_refresh_scan_execution_rows(
        self,
        path: Path,
        *,
        output_dir: Path,
        facade: str,
        capture_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows = self.read_jsonl_artifact(path)
        if not rows:
            return []
        point_count_by_scan: Dict[str, int] = {}
        point_count_by_dir: Dict[str, int] = {}
        for row in capture_rows:
            if facade and str(row.get("facade", "") or "").strip().lower() != str(facade).strip().lower():
                continue
            scan_id = str(row.get("scan_id", "") or "")
            point_count = int(row.get("point_count", 0) or 0)
            if scan_id:
                point_count_by_scan[scan_id] = int(point_count_by_scan.get(scan_id, 0) or 0) + point_count
            capture_dir = self.route5_resolve_capture_dir(output_dir, row.get("capture_dir", ""))
            if capture_dir is not None:
                point_count_by_dir[str(capture_dir.resolve())] = point_count
        updated_rows: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row if isinstance(row, dict) else {})
            if facade and str(item.get("facade", "") or "").strip().lower() != str(facade).strip().lower():
                updated_rows.append(item)
                continue
            scan_id = str(item.get("scan_id", "") or "")
            point_count = point_count_by_scan.get(scan_id)
            if point_count is None:
                capture_dirs = item.get("capture_dirs", []) if isinstance(item.get("capture_dirs"), list) else []
                point_count = 0
                for raw_dir in capture_dirs:
                    capture_dir = self.route5_resolve_capture_dir(output_dir, raw_dir)
                    if capture_dir is not None:
                        point_count += int(point_count_by_dir.get(str(capture_dir.resolve()), 0) or 0)
            item["point_count"] = int(point_count or 0)
            item["postprocess_status"] = "done" if int(point_count or 0) > 0 else "failed"
            updated_rows.append(item)
        self.route5_rewrite_jsonl_artifact(path, updated_rows)
        return updated_rows

    def route5_scan_capture_guard_passed(self, row: Dict[str, Any]) -> bool:
        row = row if isinstance(row, dict) else {}
        return bool(row.get("capture_guard_passed", False))

    def route5_valid_scan_capture_count_from_logs(self, output_dir: Path, facade_dir: Path, facade: str) -> int:
        rows = self.read_jsonl_artifact(facade_dir / "lidar_capture_log.jsonl") or self.read_jsonl_artifact(output_dir / "lidar_capture_log.jsonl")
        valid_scan_ids: set[str] = set()
        for row in rows:
            if str(row.get("facade", "") or "").strip().lower() != str(facade or "").strip().lower():
                continue
            scan_id = str(row.get("scan_id", "") or "").strip()
            if not scan_id:
                continue
            status = str(row.get("capture_status", row.get("status", "ok")) or "ok").strip().lower()
            if status not in {"ok", "captured", "done", ""}:
                continue
            if self.route5_scan_capture_guard_passed(row):
                valid_scan_ids.add(scan_id)
        return len(valid_scan_ids)

    def route5_annotate_scan_capture_guard(
        self,
        output_dir: Path,
        facade_dir: Path,
        *,
        scan_id: str,
        guard: Dict[str, Any],
        original_target_pose: Dict[str, Any],
        runtime_target_pose: Dict[str, Any],
        capture_pose: Dict[str, Any],
        capture: Dict[str, Any],
    ) -> Dict[str, Any]:
        guard_payload = self.route5_json_safe(guard if isinstance(guard, dict) else {})
        metadata = {
            "capture_guard_passed": bool(guard_payload.get("capture_guard_passed", False)),
            "capture_guard_reason": str(guard_payload.get("reason", "") or ""),
            "capture_guard": guard_payload,
            "original_target_pose": self.route5_json_safe(original_target_pose),
            "runtime_target_pose": self.route5_json_safe(runtime_target_pose),
            "capture_pose": self.route5_json_safe(capture_pose),
        }
        capture = dict(capture if isinstance(capture, dict) else {})
        capture["capture_guard"] = guard_payload
        capture["capture_guard_passed"] = metadata["capture_guard_passed"]
        capture["original_target_pose"] = metadata["original_target_pose"]
        capture["runtime_target_pose"] = metadata["runtime_target_pose"]
        capture["capture_pose"] = metadata["capture_pose"]

        for path in (output_dir / "lidar_capture_log.jsonl", facade_dir / "lidar_capture_log.jsonl"):
            rows = self.read_jsonl_artifact(path)
            if not rows:
                continue
            updated: List[Dict[str, Any]] = []
            for row in rows:
                item = dict(row if isinstance(row, dict) else {})
                if str(item.get("scan_id", "") or "") == str(scan_id or ""):
                    item.update(metadata)
                    item["capture_status"] = str(item.get("capture_status", item.get("status", "ok")) or "ok")
                updated.append(item)
            self.route5_rewrite_jsonl_artifact(path, updated)

        for path in (output_dir / "scan_execution_log.jsonl", facade_dir / "scan_execution_log.jsonl"):
            rows = self.read_jsonl_artifact(path)
            if not rows:
                continue
            updated = []
            for row in rows:
                item = dict(row if isinstance(row, dict) else {})
                if str(item.get("scan_id", "") or "") == str(scan_id or ""):
                    item.update(metadata)
                updated.append(item)
            self.route5_rewrite_jsonl_artifact(path, updated)
        return capture

    def route5_refresh_facade_scan_pointcloud_rows(self, output_dir: Path, facade_dir: Path, facade: str) -> Dict[str, Any]:
        result_cache: Dict[str, Dict[str, Any]] = {}
        root_capture_rows, root_updates = self.route5_refresh_lidar_rows_for_file(
            output_dir / "lidar_capture_log.jsonl",
            output_dir=output_dir,
            facade=facade,
            result_cache=result_cache,
        )
        facade_capture_rows, facade_updates = self.route5_refresh_lidar_rows_for_file(
            facade_dir / "lidar_capture_log.jsonl",
            output_dir=output_dir,
            facade=facade,
            result_cache=result_cache,
        )
        capture_rows_for_scan = facade_capture_rows or root_capture_rows
        self.route5_refresh_scan_execution_rows(
            output_dir / "scan_execution_log.jsonl",
            output_dir=output_dir,
            facade=facade,
            capture_rows=capture_rows_for_scan,
        )
        self.route5_refresh_scan_execution_rows(
            facade_dir / "scan_execution_log.jsonl",
            output_dir=output_dir,
            facade=facade,
            capture_rows=capture_rows_for_scan,
        )
        source_point_count = sum(int(item.get("point_count", 0) or 0) for item in capture_rows_for_scan if str(item.get("facade", "") or "").strip().lower() == str(facade).strip().lower())
        failed = [
            {"capture_dir": key, "error": str(value.get("postprocess_error", "") or ""), "point_count": int(value.get("point_count", 0) or 0)}
            for key, value in result_cache.items()
            if int(value.get("point_count", 0) or 0) <= 0
        ]
        record = {
            "facade": str(facade or ""),
            "processed_frame_count": len(result_cache),
            "updated_row_count": int(root_updates + facade_updates),
            "source_point_count": int(source_point_count),
            "failed_count": len(failed),
            "failed_rows": failed,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.append_jsonl(output_dir / "route5_scan_postprocess_events.jsonl", record)
        self.route5_log_event(output_dir, "scan_pointcloud_postprocess", record)
        return record

    def route5_new_target_reset_tracker(self, target_id: str, target_pose: Dict[str, Any], original_target_pose: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        original = original_target_pose if isinstance(original_target_pose, dict) and original_target_pose else target_pose
        return {
            "target_id": str(target_id or ""),
            "original_target_pose": self.route5_json_safe(original if isinstance(original, dict) else {}),
            "avoidance_tick_count": 0,
            "consecutive_avoidance_ticks": 0,
            "consecutive_must_stop_ticks": 0,
            "z_deviation_tick_count": 0,
            "first_distance_cm": None,
            "best_distance_cm": None,
            "last_distance_cm": None,
            "max_cross_track_cm": 0.0,
            "last_direction": "",
            "direction_counts": {},
            "candidate_action_scores": {},
            "risk_counts": {},
            "last_front_depth_cm": 0.0,
            "last_risk_state": "",
            "history": [],
        }

    def route5_record_target_reset_tick(
        self,
        tracker: Dict[str, Any],
        event: Dict[str, Any],
        gate: Dict[str, Any],
        error: Dict[str, Any],
        *,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
    ) -> Dict[str, Any]:
        tracker = dict(tracker if isinstance(tracker, dict) else self.route5_new_target_reset_tracker("", target_pose))
        event = event if isinstance(event, dict) else {}
        gate = gate if isinstance(gate, dict) else {}
        error = error if isinstance(error, dict) else {}
        prediction = event.get("or2_prediction", {}) if isinstance(event.get("or2_prediction"), dict) else {}
        rule = event.get("or2_rule", {}) if isinstance(event.get("or2_rule"), dict) else {}
        selected = str(gate.get("selected_direction", event.get("or2_selected_direction", rule.get("selected_direction", ""))) or "").strip().lower()
        risk = str(gate.get("front_risk_state", prediction.get("front_risk_state", event.get("or2_front_risk_state", ""))) or "").strip().lower()
        active = bool(gate.get("avoidance_active", False)) or risk in {"obstacle_warning", "must_stop"}
        route7_policy = gate.get("route7_soft_obstacle_policy", {}) if isinstance(gate.get("route7_soft_obstacle_policy", {}), dict) else {}
        if not route7_policy:
            route7_policy = event.get("route7_soft_obstacle_policy", {}) if isinstance(event.get("route7_soft_obstacle_policy", {}), dict) else {}
        route7_policy_mode = str(route7_policy.get("mode", "") or "").strip().lower()
        route7_reset_request = bool(gate.get("route7_navigation_point_reset_request", False)) or bool(event.get("route7_navigation_point_reset_request", False))
        route7_continue_planned_route = route7_policy_mode == "continue_planned_route" and not route7_reset_request
        if route7_continue_planned_route:
            active = False
        if route7_reset_request:
            active = True
        distance = self.route5_event_float(event.get("distance_to_goal_cm", error.get("dist_xy_cm", 0.0)), default=0.0)
        cross_track = self.route5_event_float(
            event.get("route_cross_track_cm", event.get("path_deviation_cm", event.get("post_route_cross_track_cm", 0.0))),
            default=0.0,
        )
        front_depth = self.route5_event_float(gate.get("front_min_depth_cm", (event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}).get("front_min_depth_cm", 0.0)), default=0.0)
        original_pose = self.route5_original_target_pose_from_tracker(tracker, target_pose)
        corridor = self.route5_height_corridor_for_target(str(event.get("route5_stage", event.get("stage", "")) or ""), original_pose)
        current_z = self.route5_event_float((current_pose if isinstance(current_pose, dict) else {}).get("z"), default=0.0)
        planned_z = float(corridor["planned_z_cm"])
        z_error = current_z - planned_z
        tracker["last_z_error_cm"] = round(float(z_error), 3)
        tracker["height_corridor"] = corridor
        if abs(z_error) > 120.0:
            tracker["z_deviation_tick_count"] = int(tracker.get("z_deviation_tick_count", 0) or 0) + 1
        else:
            tracker["z_deviation_tick_count"] = 0
        if tracker.get("first_distance_cm") is None and distance > 0.0:
            tracker["first_distance_cm"] = float(distance)
        best = tracker.get("best_distance_cm")
        if distance > 0.0 and (best is None or distance < float(best)):
            tracker["best_distance_cm"] = float(distance)
        tracker["last_distance_cm"] = float(distance)
        tracker["max_cross_track_cm"] = max(float(tracker.get("max_cross_track_cm", 0.0) or 0.0), abs(float(cross_track)))
        tracker["last_direction"] = selected
        tracker["last_front_depth_cm"] = float(front_depth)
        tracker["last_risk_state"] = risk
        tracker["last_route7_policy_mode"] = route7_policy_mode
        tracker["route7_navigation_point_reset_request"] = bool(route7_reset_request)
        tracker["candidate_action_scores"] = self.route5_json_safe(rule.get("candidate_action_scores", event.get("or2_candidate_action_scores", {})))
        directions = dict(tracker.get("direction_counts", {}) if isinstance(tracker.get("direction_counts"), dict) else {})
        if selected:
            directions[selected] = int(directions.get(selected, 0) or 0) + 1
        tracker["direction_counts"] = directions
        risks = dict(tracker.get("risk_counts", {}) if isinstance(tracker.get("risk_counts"), dict) else {})
        if risk:
            risks[risk] = int(risks.get(risk, 0) or 0) + 1
        tracker["risk_counts"] = risks
        if active:
            tracker["avoidance_tick_count"] = int(tracker.get("avoidance_tick_count", 0) or 0) + 1
            tracker["consecutive_avoidance_ticks"] = int(tracker.get("consecutive_avoidance_ticks", 0) or 0) + 1
        else:
            tracker["consecutive_avoidance_ticks"] = 0
        if route7_continue_planned_route:
            tracker["route7_continue_planned_route_tick_count"] = int(tracker.get("route7_continue_planned_route_tick_count", 0) or 0) + 1
        if risk == "must_stop" and not route7_continue_planned_route:
            tracker["consecutive_must_stop_ticks"] = int(tracker.get("consecutive_must_stop_ticks", 0) or 0) + 1
        else:
            tracker["consecutive_must_stop_ticks"] = 0
        history = list(tracker.get("history", []) if isinstance(tracker.get("history"), list) else [])
        history.append(
            {
                "frame_id": event.get("frame_id"),
                "risk": risk,
                "selected_direction": selected,
                "avoidance_active": active,
                "route7_policy_mode": route7_policy_mode,
                "route7_navigation_point_reset_request": bool(route7_reset_request),
                "front_min_depth_cm": float(front_depth),
                "distance_to_goal_cm": float(distance),
                "cross_track_cm": float(cross_track),
                "z_error_cm": round(float(z_error), 3),
                "current_pose": self.route5_json_safe(current_pose),
            }
        )
        tracker["history"] = history[-40:]
        return tracker

    def route5_should_reset_target(self, tracker: Dict[str, Any], error: Dict[str, Any]) -> Dict[str, Any]:
        tracker = tracker if isinstance(tracker, dict) else {}
        error = error if isinstance(error, dict) else {}
        if bool(error.get("reached", False)):
            return {"should_reset": False, "reason": "target_reached"}
        if bool(tracker.get("route7_navigation_point_reset_request", False)):
            return {"should_reset": True, "reason": "route7_navigation_point_reset_request", "value": True}
        consecutive = int(tracker.get("consecutive_avoidance_ticks", 0) or 0)
        must_stop = int(tracker.get("consecutive_must_stop_ticks", 0) or 0)
        total = int(tracker.get("avoidance_tick_count", 0) or 0)
        first_distance = tracker.get("first_distance_cm")
        best_distance = tracker.get("best_distance_cm")
        max_cross_track = float(tracker.get("max_cross_track_cm", 0.0) or 0.0)
        if consecutive >= 8:
            return {"should_reset": True, "reason": "consecutive_or2_avoidance_ticks", "value": consecutive, "threshold": 8}
        if must_stop >= 2:
            return {"should_reset": True, "reason": "consecutive_must_stop_ticks", "value": must_stop, "threshold": 2}
        if total >= 20 and first_distance is not None and best_distance is not None and float(best_distance) >= float(first_distance) - 50.0:
            return {
                "should_reset": True,
                "reason": "or2_avoidance_no_distance_progress",
                "value": total,
                "threshold": 20,
                "first_distance_cm": float(first_distance),
                "best_distance_cm": float(best_distance),
            }
        if max_cross_track > 250.0:
            return {"should_reset": True, "reason": "path_deviation_exceeded", "value": round(max_cross_track, 3), "threshold": 250.0}
        return {"should_reset": False, "reason": "reset_not_needed"}

    def route5_height_corridor_for_target(self, stage: str, planned_pose: Dict[str, Any]) -> Dict[str, float]:
        planned_z = self.route5_event_float((planned_pose if isinstance(planned_pose, dict) else {}).get("z"), default=300.0)
        stage_name = str(stage or "").strip().upper()
        allowance = 80.0 if stage_name == "NAV_TO_SCAN_POINT" else 120.0
        return {
            "planned_z_cm": round(float(planned_z), 3),
            "max_z_cm": round(float(planned_z + allowance), 3),
            "allowance_cm": float(allowance),
        }

    def route5_original_target_pose_from_tracker(self, tracker: Dict[str, Any], target_pose: Dict[str, Any]) -> Dict[str, Any]:
        tracker = tracker if isinstance(tracker, dict) else {}
        original = tracker.get("original_target_pose", {}) if isinstance(tracker.get("original_target_pose"), dict) else {}
        return dict(original or (target_pose if isinstance(target_pose, dict) else {}))

    def route5_target_reset_directions(
        self,
        stage: str,
        tracker: Dict[str, Any],
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
    ) -> List[str]:
        primary = self.route5_target_reset_direction(tracker)
        original = self.route5_original_target_pose_from_tracker(tracker, target_pose)
        corridor = self.route5_height_corridor_for_target(stage, original)
        current_z = self.route5_event_float((current_pose if isinstance(current_pose, dict) else {}).get("z"), default=0.0)
        over_corridor = current_z >= float(corridor["max_z_cm"])
        base = [primary] + [name for name in ("left", "right", "up", "backoff") if name != primary]
        if over_corridor:
            base = [name for name in base if name != "up"]
            if primary == "up":
                base = ["backoff"] + [name for name in base if name != "backoff"]
        return base or ["backoff", "left", "right"]

    def route5_plan_deviation_state(
        self,
        stage: str,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        tracker: Dict[str, Any],
        error: Dict[str, Any],
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        tracker = tracker if isinstance(tracker, dict) else {}
        error = error if isinstance(error, dict) else {}
        event = event if isinstance(event, dict) else {}
        original = self.route5_original_target_pose_from_tracker(tracker, target_pose)
        corridor = self.route5_height_corridor_for_target(stage, original)
        current_z = self.route5_event_float((current_pose if isinstance(current_pose, dict) else {}).get("z"), default=0.0)
        target_z = self.route5_event_float((target_pose if isinstance(target_pose, dict) else {}).get("z"), default=current_z)
        planned_z = float(corridor["planned_z_cm"])
        z_error = current_z - planned_z
        target_z_excess = target_z - float(corridor["max_z_cm"])
        cross_track = self.route5_event_float(
            event.get("route_cross_track_cm", event.get("path_deviation_cm", event.get("post_route_cross_track_cm", tracker.get("max_cross_track_cm", 0.0)))),
            default=0.0,
        )
        best = tracker.get("best_distance_cm")
        first = tracker.get("first_distance_cm")
        no_progress = bool(first is not None and best is not None and float(best) >= float(first) - 50.0)
        z_ticks = int(tracker.get("z_deviation_tick_count", 0) or 0)
        up_count = int((tracker.get("direction_counts", {}) if isinstance(tracker.get("direction_counts"), dict) else {}).get("up", 0) or 0)
        reasons: List[str] = []
        if abs(z_error) > 120.0 and z_ticks >= 3:
            reasons.append("z_error_exceeded_for_3_ticks")
        if target_z_excess > 0.0:
            reasons.append("reset_target_z_exceeds_planned_corridor")
        if abs(cross_track) > 250.0 and no_progress:
            reasons.append("cross_track_no_progress")
        if up_count >= 3 and current_z > float(corridor["max_z_cm"]):
            reasons.append("repeated_up_moving_away_from_planned_band")
        return {
            "stage": str(stage or ""),
            "z_error_cm": round(float(z_error), 3),
            "target_z_excess_cm": round(float(max(0.0, target_z_excess)), 3),
            "distance_to_goal_cm": round(self.route5_event_float(event.get("distance_to_goal_cm", error.get("dist_xy_cm", 0.0)), default=0.0), 3),
            "cross_track_cm": round(float(cross_track), 3),
            "z_deviation_tick_count": z_ticks,
            "up_direction_count": up_count,
            "height_corridor": corridor,
            "original_target_pose": self.route5_json_safe(original),
            "current_target_pose": self.route5_json_safe(target_pose if isinstance(target_pose, dict) else {}),
            "should_repair": bool(reasons),
            "repair_reasons": reasons,
        }

    def route5_plan_repair_fallback_action(self, deviation: Dict[str, Any]) -> str:
        reasons = set(deviation.get("repair_reasons", []) if isinstance(deviation.get("repair_reasons"), list) else [])
        if "cross_track_no_progress" in reasons:
            return "lateral_reset_then_descend"
        if "reset_target_z_exceeds_planned_corridor" in reasons or "z_error_exceeded_for_3_ticks" in reasons:
            return "descend_to_target_band"
        return "backoff_then_descend"

    def route5_plan_repair_pose(
        self,
        *,
        target_house_id: str,
        action: str,
        stage: str,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        deviation: Dict[str, Any],
    ) -> Dict[str, Any]:
        corridor = deviation.get("height_corridor", {}) if isinstance(deviation.get("height_corridor"), dict) else self.route5_height_corridor_for_target(stage, target_pose)
        target_z = float(corridor.get("planned_z_cm", (target_pose if isinstance(target_pose, dict) else {}).get("z", 300.0)) or 300.0)
        yaw = float((target_pose if isinstance(target_pose, dict) else {}).get("yaw", (target_pose if isinstance(target_pose, dict) else {}).get("yaw_deg", (current_pose if isinstance(current_pose, dict) else {}).get("yaw", 0.0))) or 0.0)
        original_target = deviation.get("original_target_pose", {}) if isinstance(deviation.get("original_target_pose"), dict) else {}
        if action == "descend_to_target_band":
            pose = dict(original_target or (target_pose if isinstance(target_pose, dict) else {}) or (current_pose if isinstance(current_pose, dict) else {}))
            pose["z"] = target_z
            pose["yaw"] = yaw
            safety = self.route3_safety_report_for_pose(target_house_id, pose)
            pose["route5_plan_repair_safety"] = self.route5_json_safe(safety)
            return pose
        candidates: List[Dict[str, Any]] = []
        if action == "lateral_reset_then_descend":
            left_payload = action_payload("left")
            left_payload.update({"forward_cm": 0.0, "right_cm": -120.0, "up_cm": 0.0, "yaw_delta_deg": 0.0})
            right_payload = action_payload("right")
            right_payload.update({"forward_cm": 0.0, "right_cm": 120.0, "up_cm": 0.0, "yaw_delta_deg": 0.0})
            candidates.extend(
                [
                    self.route3_predict_next_pose(current_pose, left_payload),
                    self.route3_predict_next_pose(current_pose, right_payload),
                ]
            )
        elif action == "backoff_then_descend":
            payload = action_payload("backoff")
            payload.update({"forward_cm": -120.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0})
            candidates.append(self.route3_predict_next_pose(current_pose, payload))
        candidates.append(dict(current_pose if isinstance(current_pose, dict) else {}))
        for pose in candidates:
            pose = dict(pose)
            pose["z"] = target_z
            pose["yaw"] = yaw
            safety = self.route3_safety_report_for_pose(target_house_id, pose)
            if bool(safety.get("safe", False)):
                pose["route5_plan_repair_safety"] = self.route5_json_safe(safety)
                return pose
        pose = dict(current_pose if isinstance(current_pose, dict) else {})
        pose["z"] = target_z
        pose["yaw"] = yaw
        pose["route5_plan_repair_safety"] = {"safe": False, "reason": "no_safe_repair_pose_found"}
        return pose

    def route5_plan_deviation_repair_decision(
        self,
        output_dir: Optional[Path],
        *,
        target_house_id: str,
        stage: str,
        facade: str,
        target_id: str,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        tracker: Dict[str, Any],
        error: Dict[str, Any],
        event: Dict[str, Any],
        repair_index: int,
    ) -> Dict[str, Any]:
        deviation = self.route5_plan_deviation_state(stage, current_pose, target_pose, tracker, error, event)
        if not bool(deviation.get("should_repair", False)):
            return {"status": "not_needed", "plan_deviation": deviation}
        allowed = {"descend_to_target_band", "lateral_reset_then_descend", "backoff_then_descend", "try_next_observation", "degraded_capture_current", "skip_current_scan_point"}
        action = self.route5_plan_repair_fallback_action(deviation)
        reason = "deterministic_plan_deviation_repair"
        llm_ref: Dict[str, Any] = {}
        api_key = ""
        try:
            api_key = str(self.effective_llm_api_key() or "")
        except Exception:
            api_key = ""
        if api_key:
            context = {
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "current_pose": current_pose,
                "target_pose": target_pose,
                "deviation": deviation,
                "tracker": tracker,
            }
            try:
                response = self.call_configured_llm_text(
                    system_prompt="Return compact JSON only for UAV Route6_entrance_search repair. Choose one allowed action and explain briefly.",
                    user_prompt=json.dumps(
                        {
                            "allowed_actions": sorted(allowed),
                            "context": self.route5_json_safe(context),
                            "expected_json": {"repair_action": "descend_to_target_band", "reason": ""},
                        },
                        ensure_ascii=False,
                    ),
                    max_output_tokens=350,
                    json_schema={"repair_action": "descend_to_target_band", "reason": ""},
                )
                parsed = extract_json_object(str(response.get("raw_text", "") or ""))
                parsed_action = str(parsed.get("repair_action", parsed.get("action", "")) or "").strip()
                if parsed_action in allowed:
                    action = parsed_action
                    reason = str(parsed.get("reason", "llm_plan_deviation_repair") or "llm_plan_deviation_repair")
                llm_ref = self.route5_log_llm_call(output_dir, "plan_deviation_repair", context, response, facade=facade, target_id=target_id, decision={"repair_action": action, "reason": reason})
            except Exception as exc:
                reason = f"deterministic_fallback_after_llm_error:{exc}"
        repair_pose = self.route5_plan_repair_pose(
            target_house_id=target_house_id,
            action=action,
            stage=stage,
            current_pose=current_pose,
            target_pose=target_pose,
            deviation=deviation,
        )
        record = {
            "status": "ok",
            "stage": str(stage or ""),
            "facade": str(facade or ""),
            "target_id": str(target_id or ""),
            "repair_target_id": f"{target_id}_repair_{int(repair_index)}",
            "repair_index": int(repair_index),
            "repair_action": action,
            "repair_reason": reason,
            "plan_deviation": self.route5_json_safe(deviation),
            "current_pose": self.route5_json_safe(current_pose),
            "original_target_pose": self.route5_json_safe(deviation.get("original_target_pose", {})),
            "current_target_pose": self.route5_json_safe(target_pose),
            "repair_target_pose": self.route5_json_safe(repair_pose),
            "llm_call_ref": llm_ref,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        if output_dir is not None:
            self.append_jsonl(output_dir / "route5_plan_deviation_repairs.jsonl", record)
            self.route5_log_event(output_dir, "plan_deviation_repair", record)
        return record

    def route5_near_obstacle_arrival_state(
        self,
        error: Dict[str, Any],
        gate: Dict[str, Any],
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        error = error if isinstance(error, dict) else {}
        gate = gate if isinstance(gate, dict) else {}
        event = event if isinstance(event, dict) else {}
        summary = event.get("pointcloud_summary", event.get("depth_obstacle_summary", {}))
        summary = summary if isinstance(summary, dict) else {}
        distance = self.route5_event_float(event.get("distance_to_goal_cm", error.get("dist_3d_cm", error.get("dist_xy_cm", 0.0))), default=0.0)
        front_depth = self.route5_event_float(gate.get("front_min_depth_cm", summary.get("front_min_depth_cm", 0.0)), default=0.0)
        risk = str(gate.get("front_risk_state", event.get("or2_front_risk_state", "")) or "").strip().lower()
        route7_policy = gate.get("route7_soft_obstacle_policy", {}) if isinstance(gate.get("route7_soft_obstacle_policy", {}), dict) else {}
        if not route7_policy:
            route7_policy = event.get("route7_soft_obstacle_policy", {}) if isinstance(event.get("route7_soft_obstacle_policy", {}), dict) else {}
        route7_policy_mode = str(route7_policy.get("mode", "") or "").strip().lower()
        forward_clear = bool(summary.get("forward_swept_clear", True))
        obstacle_confirmed = bool(
            risk in {"obstacle_warning", "must_stop"}
            or (front_depth > 0.0 and front_depth <= 300.0 and not forward_clear)
            or bool(gate.get("must_stop", False))
        )
        if route7_policy_mode == "continue_planned_route":
            obstacle_confirmed = False
        if bool(error.get("reached", False)):
            policy = "standard_reach_tolerance"
            near_reached = False
            reason = "standard_reach_tolerance"
        elif obstacle_confirmed and distance <= 150.0:
            policy = "near_obstacle_reached"
            near_reached = True
            reason = "confirmed_obstacle_within_150cm_goal_tolerance"
        elif obstacle_confirmed and distance <= 300.0:
            policy = "approach_with_caution"
            near_reached = False
            reason = "confirmed_obstacle_within_300cm_caution_zone"
        else:
            policy = "continue_navigation"
            near_reached = False
            reason = "no_confirmed_near_obstacle_arrival"
        return {
            "arrival_policy": policy,
            "near_obstacle_reached": bool(near_reached),
            "front_depth_cm": round(float(front_depth), 3),
            "distance_to_goal_cm": round(float(distance), 3),
            "arrival_reason": reason,
            "near_obstacle_confirmed": obstacle_confirmed,
            "route7_policy_mode": route7_policy_mode,
            "near_obstacle_arrival_threshold_cm": 150.0,
            "near_obstacle_caution_threshold_cm": 300.0,
        }

    def route5_target_reset_direction(self, tracker: Dict[str, Any]) -> str:
        tracker = tracker if isinstance(tracker, dict) else {}
        last_direction = str(tracker.get("last_direction", "") or "").strip().lower()
        if last_direction in {"left", "right", "up", "backoff"}:
            return last_direction
        counts = tracker.get("direction_counts", {}) if isinstance(tracker.get("direction_counts"), dict) else {}
        counted = [(name, int(counts.get(name, 0) or 0)) for name in ("left", "right", "up", "backoff")]
        best_count = max((count for _name, count in counted), default=0)
        if best_count > 0:
            return max(counted, key=lambda item: (item[1], {"left": 3, "right": 3, "up": 2, "backoff": 1}.get(item[0], 0)))[0]
        scores = tracker.get("candidate_action_scores", {}) if isinstance(tracker.get("candidate_action_scores"), dict) else {}
        return max(("left", "right", "up", "backoff"), key=lambda name: self.route5_event_float(scores.get(name), default=0.0))

    def route5_target_reset_payload(self, direction: str, step_cm: float) -> Dict[str, Any]:
        direction = str(direction or "backoff").strip().lower()
        payload = action_payload(direction)
        step = float(step_cm)
        if direction == "left":
            payload.update({"forward_cm": 0.0, "right_cm": -step, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_target_reset_left"})
        elif direction == "right":
            payload.update({"forward_cm": 0.0, "right_cm": step, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_target_reset_right"})
        elif direction == "up":
            payload.update({"forward_cm": 0.0, "right_cm": 0.0, "up_cm": step, "yaw_delta_deg": 0.0, "action_name": "route5_target_reset_up"})
        else:
            payload.update({"forward_cm": -step, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_target_reset_backoff"})
        return payload

    def route5_build_target_reset_candidate(
        self,
        *,
        target_house_id: str,
        stage: str,
        facade: str,
        target_id: str,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        tracker: Dict[str, Any],
        reset_reason: str,
        reset_index: int,
    ) -> Dict[str, Any]:
        original_target_pose = self.route5_original_target_pose_from_tracker(tracker, target_pose)
        corridor = self.route5_height_corridor_for_target(stage, original_target_pose)
        directions = self.route5_target_reset_directions(stage, tracker, current_pose, target_pose)
        primary = directions[0] if directions else self.route5_target_reset_direction(tracker)
        rejected: List[Dict[str, Any]] = []
        for direction in directions:
            for step_cm in (120.0, 180.0, 240.0):
                payload = self.route5_target_reset_payload(direction, step_cm)
                pose = self.route3_predict_next_pose(current_pose, payload)
                height_clamped = False
                if self.route5_event_float(pose.get("z"), default=0.0) > float(corridor["max_z_cm"]):
                    pose["z"] = float(corridor["max_z_cm"])
                    height_clamped = True
                pose["yaw"] = float(target_pose.get("yaw", target_pose.get("yaw_deg", current_pose.get("yaw", 0.0))) or 0.0)
                facade_corridor_check = self.route5_facade_corridor_check(
                    target_house_id,
                    facade,
                    pose,
                    original_target_pose=original_target_pose,
                )
                safety = self.route3_safety_report_for_pose(target_house_id, pose)
                accepted = bool(safety.get("safe", False)) and bool(facade_corridor_check.get("same_facade_corridor", True))
                candidate = {
                    "status": "ok" if accepted else "rejected",
                    "stage": str(stage or ""),
                    "facade": str(facade or ""),
                    "target_id": str(target_id or ""),
                    "reset_target_id": f"{target_id}_reset_{int(reset_index)}",
                    "reset_index": int(reset_index),
                    "reset_direction": direction,
                    "reset_step_cm": float(step_cm),
                    "reset_reason": str(reset_reason or ""),
                    "original_target_pose": self.route5_json_safe(original_target_pose),
                    "runtime_target_pose": self.route5_json_safe(target_pose),
                    "planned_target_pose": self.route5_json_safe(original_target_pose),
                    "current_pose": self.route5_json_safe(current_pose),
                    "reset_target_pose": self.route5_json_safe(pose),
                    "height_corridor": self.route5_json_safe(corridor),
                    "height_clamped": bool(height_clamped),
                    "facade_corridor_check": self.route5_json_safe(facade_corridor_check),
                    "safety": self.route5_json_safe(safety),
                    "tracker": self.route5_json_safe(tracker),
                    "created_at": datetime.now().isoformat(timespec="milliseconds"),
                }
                if accepted:
                    return candidate
                rejected.append(candidate)
        return {
            "status": "failed",
            "stage": str(stage or ""),
            "facade": str(facade or ""),
            "target_id": str(target_id or ""),
            "reset_index": int(reset_index),
            "reset_direction": primary,
            "reset_reason": str(reset_reason or ""),
            "original_target_pose": self.route5_json_safe(original_target_pose),
            "runtime_target_pose": self.route5_json_safe(target_pose),
            "planned_target_pose": self.route5_json_safe(original_target_pose),
            "height_corridor": self.route5_json_safe(corridor),
            "current_pose": self.route5_json_safe(current_pose),
            "rejected_candidates": rejected[-8:],
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }

    def route5_record_target_reset(self, output_dir: Path, candidate: Dict[str, Any]) -> Dict[str, Any]:
        record = self.route5_json_safe(candidate if isinstance(candidate, dict) else {})
        self.append_jsonl(output_dir / "route5_target_resets.jsonl", record)
        self.route5_log_event(output_dir, "target_reset_applied", record)
        reset_points = list(self.llm_route5_state.get("reset_route_points", []) if isinstance(self.llm_route5_state.get("reset_route_points"), list) else [])
        reset_pose = record.get("reset_target_pose", {}) if isinstance(record.get("reset_target_pose"), dict) else {}
        if reset_pose:
            reset_points.append(
                {
                    **reset_pose,
                    "label": str(record.get("reset_target_id", "target_reset") or "target_reset"),
                    "route_point_type": "target_reset",
                    "facade": str(record.get("facade", "") or ""),
                    "reset_direction": str(record.get("reset_direction", "") or ""),
                }
            )
        self.route5_update_state(
            last_target_reset=record,
            target_reset_count=int(self.llm_route5_state.get("target_reset_count", 0) or 0) + 1,
            reset_route_points=reset_points[-80:],
        )
        return record

    def route5_capture_obstacle_event(
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
            "source": "llm_route_v5_or2_fused_navigation",
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
            "episode_id": f"route5_{target_id}_{frame_id}",
            "scenario_id": f"route5_{target_id}",
            "environment_id": "default_unreal_scene",
            "method": "obstacle_representation_direction_rule_v1",
            "obstacle_hint": "unknown",
        }
        event = build_route_event(
            capture_result,
            session_dir=output_dir,
            frame_id=frame_id,
            args=self.route5_obstacle_args(config),
            episode=episode,
            episode_index=1,
            start=start_pose,
            goal=target_pose,
            last_action=last_action,
        )
        route7_primary = self.route5_event_is_route7(event)
        if route7_primary:
            event["route_window_label"] = "V7"
        event["source"] = "llm_route_v7_or3_1_fused_navigation" if route7_primary else "llm_route_v5_or2_fused_navigation"
        event["route5_stage"] = stage
        event["facade"] = facade
        event["target_id"] = target_id
        event["route5_output_dir"] = str(output_dir)
        event["depth_lookahead_plan"] = lookahead_plan or {}
        event["nominal_payload"] = nominal_payload or {}
        prediction = self.route5_predict_obstacle_representation(event)
        event["or2_prediction"] = prediction
        if route7_primary:
            event["or3_prediction"] = prediction if isinstance(prediction, dict) else {}
            event["or3_1_prediction"] = prediction if isinstance(prediction, dict) else {}
            event["route7_primary_representation"] = "or3_1"
        if isinstance(prediction, dict) and isinstance(prediction.get("or3_prediction"), dict):
            event["or3_prediction"] = prediction.get("or3_prediction", {})
        event["representation_prediction"] = (
            {
                "status": "replaced_by_or3_1",
                "route7_primary_representation": "or3_1",
                "or3_1_front_risk_state": prediction.get("front_risk_state", "") if isinstance(prediction, dict) else "",
            }
            if route7_primary
            else {"status": "replaced_by_or2", "or2_front_risk_state": prediction.get("front_risk_state", "")}
        )
        event["representation_obstacle_hint"] = "or3_1_risk_region" if route7_primary else "or2_risk_region"
        try:
            event["local_obstacle_map_update"] = self.update_local_obstacle_map_from_event(
                event,
                event.get("current_pose", {}) if isinstance(event.get("current_pose"), dict) else {},
                output_dir,
            )
        except Exception as exc:
            event["local_obstacle_map_update"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            self.route5_log_event(output_dir, "local_obstacle_map_update_failed", event["local_obstacle_map_update"])
        self.route5_update_state(last_obstacle_event=self.route5_json_safe(event), last_or2_event=self.route5_json_safe(event))
        self.route5_log_event(output_dir, "obstacle_sensing", event)
        self.append_jsonl(
            output_dir / "route5_or2_risk_events.jsonl",
            {
                "event_type": "or3_1_prediction" if route7_primary else "or2_prediction",
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                "frame_id": frame_id,
                "facade": facade,
                "target_id": target_id,
                "route7_primary_representation": "or3_1" if route7_primary else "",
                "prediction": serializable_or2_prediction(prediction) if isinstance(prediction, dict) else {},
                "or3_prediction": serializable_or2_prediction(event.get("or3_prediction", {})) if isinstance(event.get("or3_prediction"), dict) else {},
                "risk_overlay_path": str(prediction.get("risk_overlay_path", "") if isinstance(prediction, dict) else ""),
                "prediction_json_path": str(prediction.get("prediction_json_path", "") if isinstance(prediction, dict) else ""),
            },
        )
        if isinstance(event.get("or3_prediction"), dict) and event["or3_prediction"].get("status") == "ok":
            self.append_jsonl(
                output_dir / "route5_or3_risk_events.jsonl",
                {
                    "event_type": "or3_1_prediction" if route7_primary else "or3_prediction",
                    "created_at": datetime.now().isoformat(timespec="milliseconds"),
                    "frame_id": frame_id,
                    "facade": facade,
                    "target_id": target_id,
                    "route7_primary_representation": "or3_1" if route7_primary else "",
                    "prediction": serializable_or2_prediction(event["or3_prediction"]),
                    "risk_overlay_path": str(event["or3_prediction"].get("risk_overlay_path", "")),
                    "prediction_json_path": str(event["or3_prediction"].get("prediction_json_path", "")),
                },
            )
        monitor = dict(self.llm_route5_state.get("or2_monitor", {}) if isinstance(self.llm_route5_state.get("or2_monitor"), dict) else {})
        state_counts = dict(monitor.get("state_counts", {}) if isinstance(monitor.get("state_counts"), dict) else {})
        risk_state = str(prediction.get("front_risk_state", prediction.get("status", "unknown")) if isinstance(prediction, dict) else "unknown")
        state_counts[risk_state] = int(state_counts.get(risk_state, 0) or 0) + 1
        monitor.update(
            {
                "status": "follows_route_navigation",
                "frame_count": int(monitor.get("frame_count", 0) or 0) + 1,
                "state_counts": state_counts,
                "last_frame_id": frame_id,
                "last_facade": facade,
                "last_target_id": target_id,
                "last_risk_state": risk_state,
                "last_updated_at": datetime.now().isoformat(timespec="milliseconds"),
                "representation_source": "or3_1" if route7_primary else "or2",
                "model_path": str(self.default_route7_or31_model_path() if route7_primary else self.llm_route5_representation_model_var.get() or ""),
            }
        )
        self.route5_update_state(or2_monitor=self.route5_json_safe(monitor))
        self.write_json_artifact(output_dir / "route5_or2_monitor_summary.json", self.route5_json_safe(monitor))
        try:
            risk = str(prediction.get("front_risk_state", prediction.get("status", "unknown")) or "unknown")
            front = float((event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}).get("front_min_depth_cm", 0.0) or 0.0)
            label = "OR3_1" if route7_primary else "OR2"
            self.root.after(0, lambda r=risk, f=front, label=label: self.llm_route5_representation_status_var.set(f"{label}: state={r} front={f:.1f}cm"))
        except Exception:
            pass
        return event

    def route5_normalize_avoidance_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(event if isinstance(event, dict) else {})
        collision = bool(normalized.get("collision_state", False))
        normalized["collision_state"] = collision
        normalized["avoidance_failed"] = collision
        return normalized

    def route5_write_avoidance_summary(self, output_dir: Path, *, status: str = "running") -> Dict[str, Any]:
        events = self.read_jsonl_artifact(output_dir / "avoidance_events.jsonl")
        collision_count = sum(1 for item in events if bool(item.get("collision_state", False)))
        summary = {
            "source": "llm_route_v5_or2_fused_navigation",
            "status": status,
            "event_count": len(events),
            "collision_count": collision_count,
            "avoidance_failed_count": sum(1 for item in events if bool(item.get("avoidance_failed", False))),
            "last_event": events[-1] if events else {},
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(output_dir / "avoidance_session_summary.json", summary)
        return summary

    def route5_follow_navigation_waypoint_with_fusion(
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
        sensing_interval_s = float(self.route5_sensing_config()["sensing_interval_s"])
        step_index = 0
        last_error: Dict[str, Any] = {}
        last_sense_at = 0.0
        last_action = action_payload("hold")
        start_pose = dict(current)
        reset_count = 0
        repair_count = 0
        max_target_resets = 3
        max_plan_repairs = 2
        original_planned_target_pose = dict(target_pose)
        reset_tracker = self.route5_new_target_reset_tracker(target_id, target_pose, original_planned_target_pose)
        target_safety = self.route7_navigation_safety_report(
            target_house_id,
            target_pose,
            stage=stage,
            facade=facade,
            target_id=target_id,
            output_dir=output_dir,
        )
        if not bool(target_safety.get("safe", False)):
            self.route5_hold(session, output_dir=output_dir, reason="unsafe_waypoint")
            return {"status": "blocked", "reason": "unsafe_waypoint", "safety": target_safety}
        while not self.llm_route5_stop_event.is_set():
            if self.route5_route6_full_stop_requested():
                self.llm_route5_stop_event.set()
                self.route5_hold(session, output_dir=output_dir, reason="route6_full_stop")
                return {
                    "status": "stopped",
                    "reason": "route6_full_stop",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "final_target_id": target_id,
                    "final_target_pose": target_pose,
                    "target_reset_count": reset_count,
                    "pose_error": last_error,
                    "current_pose": current,
                }
            if self.route5_wait_if_paused(session, output_dir):
                break
            error = self.route3_pose_error(current, target_pose, config)
            last_error = error
            self.root.after(
                0,
                lambda e=error: self.llm_route5_error_var.set(
                    f"Error: xy={float(e['dist_xy_cm']):.1f} z={float(e['dz']):.1f} yaw={float(e['yaw_error_deg']):.1f}"
                ),
            )
            if bool(error.get("reached", False)):
                self.route5_hold(session, output_dir=output_dir, reason="target_reached")
                return {
                    "status": "ok",
                    "reason": "target_reached",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "final_target_id": target_id,
                    "final_target_pose": target_pose,
                    "target_reset_count": reset_count,
                    "waypoint_index": waypoint_index,
                    "waypoint_count": waypoint_count,
                    "pose_error": error,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "current_pose": current,
                    "final_current_pose": current,
                    "arrival_state": {
                        "arrival_policy": "standard_reach_tolerance",
                        "near_obstacle_reached": False,
                        "near_obstacle_confirmed": False,
                        "distance_to_goal_cm": round(float(error.get("dist_3d_cm", error.get("dist_xy_cm", 0.0)) or 0.0), 3),
                        "arrival_reason": "standard_reach_tolerance",
                    },
                    "final_arrival_reason": "standard_reach_tolerance",
                }
            if time.time() - started_at > float(config["max_stage_s"]):
                self.route5_hold(session, output_dir=output_dir, reason="nav_timeout")
                return {
                    "status": "timeout",
                    "reason": "nav_timeout",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "final_target_id": target_id,
                    "final_target_pose": target_pose,
                    "target_reset_count": reset_count,
                    "pose_error": error,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "current_pose": current,
                }
            route7_runtime_plan: Dict[str, Any] = {}
            if self.route7_should_use_map_route_planner(stage, target_id, output_dir=output_dir):
                route7_runtime_plan = self.route7_update_realtime_navigation_route(
                    current,
                    target_pose,
                    output_dir=output_dir,
                    stage=stage,
                    target_id=target_id,
                    target_house_id=target_house_id,
                    update_reason=f"movement_tick_{step_index}",
                )
                if str(route7_runtime_plan.get("status", "") or "") == "blocked":
                    reason = "route7_realtime_route_blocked"
                    self.route5_hold(session, output_dir=output_dir, reason=reason)
                    self.route5_log_event(
                        output_dir,
                        "route7_realtime_route_blocked",
                        {
                            "stage": stage,
                            "facade": facade,
                            "target_id": target_id,
                            "current_pose": current,
                            "target_pose": target_pose,
                            "route7_realtime_route_plan": self.route5_json_safe(route7_runtime_plan),
                        },
                    )
                    return {
                        "status": "blocked",
                        "reason": reason,
                        "stage": stage,
                        "facade": facade,
                        "target_id": target_id,
                        "final_target_id": target_id,
                        "final_target_pose": target_pose,
                        "target_reset_count": reset_count,
                        "pose_error": error,
                        "route7_realtime_route_plan": self.route5_json_safe(route7_runtime_plan),
                        "current_pose": current,
                    }
            payload = self.route5_movement_payload_for_target_with_lookahead(current, target_pose, config, stage=stage)
            event: Dict[str, Any] = {}
            force_depth_precheck = self.route5_should_precheck_depth_before_payload(stage, payload)
            should_sense = force_depth_precheck or step_index == 0 or (time.time() - last_sense_at) >= sensing_interval_s
            lookahead_plan = self.route5_depth_lookahead_plan(current, target_pose, stage=stage, facade=facade, target_id=target_id)
            if route7_runtime_plan:
                lookahead_plan["route7_realtime_route_plan"] = self.route5_json_safe(route7_runtime_plan.get("route7_realtime_route_state", route7_runtime_plan))
            if force_depth_precheck:
                align_result = self.route5_align_to_navigation_depth_yaw(
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
                    self.route5_hold(session, output_dir=output_dir, reason=str(align_result.get("reason", "lookahead_yaw_failed")))
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
                payload = self.route5_movement_payload_for_target_with_lookahead(current, target_pose, config, stage=stage)
                lookahead_plan = dict(align_result.get("lookahead_plan", lookahead_plan))
                lookahead_plan["nominal_payload"] = dict(payload)
                self.route5_set_thinking_status(lookahead_plan, output_dir=output_dir, write_artifact=True)
            if should_sense:
                try:
                    frame_id = self.route2_next_frame_index(output_dir)
                    event = self.route5_capture_obstacle_event(
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
                    if route7_runtime_plan:
                        event["route7_realtime_route_plan"] = self.route5_json_safe(route7_runtime_plan.get("route7_realtime_route_state", route7_runtime_plan))
                        event["route7_route_decision_reason"] = str(route7_runtime_plan.get("reason", "") or "")
                    last_sense_at = time.time()
                    or2_decision = self.route5_or2_decision_for_event(
                        event,
                        nominal_payload=payload,
                        config=config,
                        current_pose=current,
                        start_pose=start_pose,
                        target_pose=target_pose,
                        last_action=last_action,
                    )
                    gate = or2_decision["gate"]
                    representation_label = "OR3_1" if str(gate.get("source", "") or "") == "route7_or3_1_a_plus_3_1" else "OR2"
                    event["llm_strategy"] = {
                        "strategy_source": str(gate.get("source", "route5_or2_a_plus_2") or "route5_or2_a_plus_2"),
                        "llm_call_required": False,
                        "strategy_cache_hit": False,
                        "strategy_cache_reason": f"{representation_label} direction rule, no LLM strategy required for local tick",
                    }
                    event.update(or2_decision["event_updates"])
                    payload = dict(or2_decision["payload"])
                    arrival_state = self.route5_near_obstacle_arrival_state(error, gate, event)
                    event.update(arrival_state)
                    if bool(arrival_state.get("near_obstacle_reached", False)):
                        payload = action_payload("hold")
                        event["goal_reached"] = True
                        event["goal_completion_reason"] = "near_obstacle_reached"
                        event["selected_action"] = "route5_near_obstacle_arrival_hold"
                        event["selected_action_reason"] = str(arrival_state.get("arrival_reason", "near_obstacle_reached") or "near_obstacle_reached")
                        event["selected_action_payload"] = payload
                        event["final_payload"] = payload
                        event = self.route5_normalize_avoidance_event(event)
                        try:
                            monitor_result = self.route5_or2_monitor_result_from_event(event)
                            self.root.after(0, lambda r=monitor_result: self.apply_route5_or2_monitor_result(r))
                        except Exception as exc:
                            self.route5_log_event(output_dir, "or2_monitor_update_failed", {"error": str(exc), "stage": stage, "target_id": target_id})
                        self.route5_write_frame_decision(output_dir, event, final_payload=payload)
                        if bool(event.get("avoidance_active", False)) or bool(event.get("collision_state", False)):
                            self.append_jsonl(output_dir / "avoidance_events.jsonl", self.route5_json_safe(event))
                            self.route5_write_avoidance_summary(output_dir, status="running")
                        self.route5_hold(session, output_dir=output_dir, reason="near_obstacle_reached")
                        return {
                            "status": "ok",
                            "reason": "near_obstacle_reached",
                            "stage": stage,
                            "facade": facade,
                            "target_id": target_id,
                            "final_target_id": target_id,
                            "final_target_pose": target_pose,
                            "target_reset_count": reset_count,
                            "waypoint_index": waypoint_index,
                            "waypoint_count": waypoint_count,
                            "pose_error": error,
                            "elapsed_s": round(time.time() - started_at, 3),
                            "current_pose": current,
                            "final_current_pose": current,
                            "arrival_state": arrival_state,
                            "final_obstacle_event": self.route5_json_safe(event),
                            "final_arrival_reason": str(arrival_state.get("arrival_reason", "near_obstacle_reached") or "near_obstacle_reached"),
                        }
                    try:
                        monitor_result = self.route5_or2_monitor_result_from_event(event)
                        self.root.after(0, lambda r=monitor_result: self.apply_route5_or2_monitor_result(r))
                    except Exception as exc:
                        self.route5_log_event(output_dir, "or2_monitor_update_failed", {"error": str(exc), "stage": stage, "target_id": target_id})
                    reset_tracker = self.route5_record_target_reset_tick(
                        reset_tracker,
                        event,
                        gate,
                        error,
                        current_pose=current,
                        target_pose=target_pose,
                    )
                    plan_deviation = self.route5_plan_deviation_state(stage, current, target_pose, reset_tracker, error, event)
                    event["plan_deviation"] = plan_deviation
                    repair_record: Dict[str, Any] = {}
                    if bool(plan_deviation.get("should_repair", False)) and repair_count < max_plan_repairs:
                        repair_record = self.route5_plan_deviation_repair_decision(
                            output_dir,
                            target_house_id=target_house_id,
                            stage=stage,
                            facade=facade,
                            target_id=target_id,
                            current_pose=current,
                            target_pose=target_pose,
                            tracker=reset_tracker,
                            error=error,
                            event=event,
                            repair_index=repair_count + 1,
                        )
                        if repair_record.get("status") == "ok":
                            repair_count += 1
                            target_pose = dict(repair_record.get("repair_target_pose", target_pose))
                            target_id = str(repair_record.get("repair_target_id", f"{target_id}_repair_{repair_count}") or f"{target_id}_repair_{repair_count}")
                            start_pose = dict(current)
                            reset_tracker = self.route5_new_target_reset_tracker(target_id, target_pose, original_planned_target_pose)
                            event["plan_repair_applied"] = True
                            event["plan_repair"] = repair_record
                            event["repair_action"] = str(repair_record.get("repair_action", "") or "")
                            event["repair_reason"] = str(repair_record.get("repair_reason", "") or "")
                            event["target_waypoint"] = target_pose
                            event["goal_pose"] = target_pose
                            event["selected_action"] = "route5_plan_deviation_repair"
                            event["selected_action_reason"] = event["repair_reason"]
                            payload = self.route5_movement_payload_for_target_with_lookahead(current, target_pose, config, stage=stage)
                            payload["action_name"] = "route5_plan_deviation_repair_nav"
                            event["selected_action_payload"] = payload
                            event["nominal_action"] = payload
                            repair_plan_log = {
                                "stage": stage,
                                "facade": facade,
                                "target_id": target_id,
                                "target_pose": target_pose,
                                "current_pose": current,
                                "plan": {
                                    "status": "ok",
                                    "reason": "plan_deviation_repair",
                                    "waypoints": [target_pose],
                                    "plan_repair": repair_record,
                                },
                                "plan_repair": repair_record,
                                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                            }
                            self.append_jsonl(output_dir / "route5_navigation_plan.jsonl", self.route5_json_safe(repair_plan_log))
                            self.route5_update_state(
                                current_navigation_plan=self.route5_json_safe(repair_plan_log),
                                current_exploration_status={
                                    "stage": stage,
                                    "facade": facade,
                                    "target_id": target_id,
                                    "target_pose": target_pose,
                                    "plan_repair": repair_record,
                                },
                            )
                            repair_decision = dict(
                                lookahead_plan,
                                decision_state="PLAN_REPAIR",
                                plan_deviation=plan_deviation,
                                plan_repair=repair_record,
                                thinking_status=(
                                    f"Thinking: {stage} PLAN_REPAIR action={repair_record.get('repair_action', '')} "
                                    f"reason={repair_record.get('repair_reason', '')} new_target={target_id}"
                                ),
                            )
                            self.route5_set_thinking_status(repair_decision, output_dir=output_dir, write_artifact=True)
                    reset_trigger = self.route5_should_reset_target(reset_tracker, error)
                    if bool(reset_trigger.get("should_reset", False)) and not bool(event.get("plan_repair_applied", False)):
                        if reset_count >= max_target_resets:
                            failed_reset = {
                                "status": "failed",
                                "reason": "target_reset_limit_exhausted",
                                "reset_trigger": reset_trigger,
                                "target_id": target_id,
                                "target_pose": self.route5_json_safe(target_pose),
                                "tracker": self.route5_json_safe(reset_tracker),
                            }
                            event["target_reset_candidate"] = failed_reset
                            event["target_reset_applied"] = False
                            event["reset_reason"] = "target_reset_limit_exhausted"
                            event["selected_action_payload"] = payload
                            event = self.route5_normalize_avoidance_event(event)
                            self.route5_write_frame_decision(output_dir, event, final_payload=payload)
                            self.route5_log_event(output_dir, "target_reset_failed", failed_reset)
                            self.route5_hold(session, output_dir=output_dir, reason="target_reset_limit_exhausted")
                            return {
                                "status": "blocked",
                                "reason": "target_reset_limit_exhausted",
                                "stage": stage,
                                "facade": facade,
                                "target_id": target_id,
                                "pose_error": error,
                                "current_pose": current,
                                "target_reset": failed_reset,
                            }
                        candidate = self.route5_build_target_reset_candidate(
                            target_house_id=target_house_id,
                            stage=stage,
                            facade=facade,
                            target_id=target_id,
                            current_pose=current,
                            target_pose=target_pose,
                            tracker=reset_tracker,
                            reset_reason=str(reset_trigger.get("reason", "or2_avoidance_conflict") or "or2_avoidance_conflict"),
                            reset_index=reset_count + 1,
                        )
                        event["target_reset_candidate"] = candidate
                        event["reset_reason"] = str(candidate.get("reset_reason", reset_trigger.get("reason", "")) or "")
                        if candidate.get("status") == "ok":
                            reset_record = self.route5_record_target_reset(output_dir, candidate)
                            if str(candidate.get("reset_reason", "") or "") == "route7_navigation_point_reset_request":
                                self.route5_log_event(output_dir, "route7_navigation_point_reset", reset_record)
                            reset_count += 1
                            target_pose = dict(candidate.get("reset_target_pose", target_pose))
                            target_id = str(candidate.get("reset_target_id", f"{target_id}_reset_{reset_count}") or f"{target_id}_reset_{reset_count}")
                            start_pose = dict(current)
                            reset_tracker = self.route5_new_target_reset_tracker(target_id, target_pose, original_planned_target_pose)
                            event["target_reset_applied"] = True
                            event["target_reset_record"] = reset_record
                            event["target_waypoint"] = target_pose
                            event["goal_pose"] = target_pose
                            event["selected_action"] = "route5_target_reset"
                            event["selected_action_reason"] = str(candidate.get("reset_reason", "target_reset_away_from_obstacle") or "target_reset_away_from_obstacle")
                            payload = self.route5_movement_payload_for_target_with_lookahead(current, target_pose, config, stage=stage)
                            payload["action_name"] = "route5_target_reset_nav"
                            event["selected_action_payload"] = payload
                            event["nominal_action"] = payload
                            lookahead_plan = self.route5_depth_lookahead_plan(current, target_pose, stage=stage, facade=facade, target_id=target_id)
                            reset_plan_log = {
                                "stage": stage,
                                "facade": facade,
                                "target_id": target_id,
                                "target_pose": target_pose,
                                "current_pose": current,
                                "plan": {
                                    "status": "ok",
                                    "reason": "target_reset_away_from_obstacle",
                                    "waypoints": [target_pose],
                                    "target_reset": reset_record,
                                },
                                "target_reset": reset_record,
                                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                            }
                            self.append_jsonl(output_dir / "route5_navigation_plan.jsonl", self.route5_json_safe(reset_plan_log))
                            self.route5_log_event(output_dir, "target_reset_navigation_plan", reset_plan_log)
                            self.route5_update_state(
                                current_navigation_plan=self.route5_json_safe(reset_plan_log),
                                current_exploration_status={
                                    "stage": stage,
                                    "facade": facade,
                                    "target_id": target_id,
                                    "target_pose": target_pose,
                                    "target_reset": reset_record,
                                },
                            )
                            reset_decision = dict(
                                lookahead_plan,
                                decision_state="TARGET_RESET",
                                avoidance_gate=gate,
                                target_reset=reset_record,
                                thinking_status=(
                                    f"Thinking: {stage} TARGET_RESET away_from_obstacle direction={candidate.get('reset_direction', '')} "
                                    f"reason={candidate.get('reset_reason', '')} new_target={target_id}"
                                ),
                            )
                            self.route5_set_thinking_status(reset_decision, output_dir=output_dir, write_artifact=True)
                        else:
                            event["target_reset_applied"] = False
                            event["selected_action_payload"] = payload
                            event = self.route5_normalize_avoidance_event(event)
                            self.route5_write_frame_decision(output_dir, event, final_payload=payload)
                            self.route5_log_event(output_dir, "target_reset_failed", candidate)
                            self.route5_hold(session, output_dir=output_dir, reason=str(candidate.get("reset_reason", "target_reset_failed") or "target_reset_failed"))
                            return {
                                "status": "blocked",
                                "reason": "target_reset_failed",
                                "stage": stage,
                                "facade": facade,
                                "target_id": target_id,
                                "pose_error": error,
                                "current_pose": current,
                                "target_reset": candidate,
                            }
                    if str(gate.get("reason", "") or "") == "depth_lookahead_blocked_before_collection":
                        self.route5_log_event(
                            output_dir,
                            "depth_lookahead_blocked",
                            {
                                "stage": stage,
                                "facade": facade,
                                "target_id": target_id,
                                "gate": gate,
                            },
                        )
                    decision_status = "TARGET_RESET" if bool(event.get("target_reset_applied", False)) else ("AVOIDANCE" if bool(gate.get("avoidance_active", False)) else "CLEAR")
                    decision = dict(
                        lookahead_plan,
                        decision_state=decision_status,
                        avoidance_gate=gate,
                        target_reset=event.get("target_reset_record", event.get("target_reset_candidate", {})),
                        thinking_status=(
                            f"Thinking: {stage} {decision_status} look waypoint yaw={float(lookahead_plan.get('look_yaw_deg', 0.0)):.1f} "
                            f"{representation_label} state={gate.get('front_risk_state', 'fallback')} selected={gate.get('selected_direction', event.get('or2_selected_direction', '--'))} "
                            f"front={float(gate.get('front_min_depth_cm', 0.0)):.1f}cm {gate.get('reason', '')}"
                        ),
                    )
                    self.route5_set_thinking_status(decision, output_dir=output_dir, write_artifact=True)
                    self.route5_log_event(output_dir, "or2_navigation_decision", decision)
                    self.root.after(
                        0,
                        lambda d=or2_decision, g=gate, label=representation_label: self.llm_route5_avoidance_status_var.set(
                            f"{label}: state={g.get('front_risk_state', 'fallback')} selected={d.get('selected_direction', '--')} "
                            f"front={float(g.get('front_min_depth_cm', 0.0)):.1f}cm can_forward={'yes' if g.get('can_forward', False) else 'no'}"
                        ),
                    )
                    self.route5_update_state(last_obstacle_event=self.route5_json_safe(event), last_or2_event=self.route5_json_safe(event), avoidance_active=bool(gate.get("avoidance_active", False)))
                    if bool(event.get("route7_map_route_replan_request", False)) or bool(gate.get("route7_map_route_replan_request", False)):
                        reason = "route7_map_route_replan_required"
                        event["avoidance_active"] = True
                        event["selected_action"] = "route7_map_route_replan_required"
                        event["selected_action_reason"] = str(
                            (event.get("route7_soft_obstacle_policy", {}) if isinstance(event.get("route7_soft_obstacle_policy", {}), dict) else {}).get("reason", reason)
                            or reason
                        )
                        event["selected_action_payload"] = action_payload("hold")
                        event["final_payload"] = action_payload("hold")
                        event = self.route5_normalize_avoidance_event(event)
                        self.route5_write_frame_decision(output_dir, event, final_payload=action_payload("hold"))
                        self.append_jsonl(output_dir / "avoidance_events.jsonl", self.route5_json_safe(event))
                        self.route5_write_avoidance_summary(output_dir, status="running")
                        self.route5_hold(session, output_dir=output_dir, reason=reason)
                        self.route5_log_event(output_dir, "route7_map_route_replan_required", {"stage": stage, "facade": facade, "target_id": target_id, "event": self.route5_json_safe(event)})
                        return {
                            "status": "blocked",
                            "reason": reason,
                            "stage": stage,
                            "facade": facade,
                            "target_id": target_id,
                            "final_target_id": target_id,
                            "final_target_pose": target_pose,
                            "target_reset_count": reset_count,
                            "pose_error": error,
                            "route7_soft_obstacle_policy": self.route5_json_safe(event.get("route7_soft_obstacle_policy", {})),
                            "current_pose": current,
                        }
                except Exception as exc:
                    self.route5_log_event(output_dir, "obstacle_sensing_failed", {"error": str(exc), "stage": stage, "target_id": target_id})
                    self.root.after(0, lambda e=exc: self.llm_route5_avoidance_status_var.set(f"Avoidance: fallback navigation ({e})"))
            predicted = self.route3_predict_next_pose(current, payload)
            safety = self.route7_navigation_safety_report(
                target_house_id,
                predicted,
                stage=stage,
                facade=facade,
                target_id=target_id,
                output_dir=output_dir,
            )
            local_3d_safety = self.query_local_3d_safety(
                current,
                payload,
                config,
                output_dir=output_dir,
                stage=stage,
                target_id=target_id,
                candidate_direction=str(payload.get("action_name", "") or ""),
            )
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
                "local_3d_safety": local_3d_safety,
                "route7_realtime_route_plan": self.route5_json_safe(route7_runtime_plan.get("route7_realtime_route_state", route7_runtime_plan)) if route7_runtime_plan else {},
                "obstacle_event_frame_id": event.get("frame_id") if event else None,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            if event:
                event["local_3d_safety"] = local_3d_safety
            self.append_jsonl(output_dir / "route5_movement_trace.jsonl", trace)
            self.root.after(
                0,
                lambda p=payload: self.llm_route5_payload_var.set(
                    f"Payload: f={p['forward_cm']} r={p['right_cm']} u={p['up_cm']} yaw={p['yaw_delta_deg']}"
                ),
            )
            if not bool(safety.get("safe", False)) and not escape_safety_allowed:
                if event:
                    self.route5_write_safety_blocked_frame_decision(output_dir, event, payload, safety)
                self.route5_hold(session, output_dir=output_dir, reason=str(safety.get("reason", "unsafe_next_step")))
                self.route5_log_event(output_dir, "navigation_blocked", trace)
                return {
                    "status": "blocked",
                    "reason": str(safety.get("reason", "unsafe_next_step")),
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "final_target_id": target_id,
                    "final_target_pose": target_pose,
                    "target_reset_count": reset_count,
                    "pose_error": error,
                    "safety": safety,
                    "current_pose": current,
                }
            if not bool(local_3d_safety.get("safe", True)):
                reason = str(local_3d_safety.get("reason", "local_3d_occupancy_blocked") or "local_3d_occupancy_blocked")
                route7_decision: Dict[str, Any] = {}
                if self.route7_should_use_map_route_planner(stage, target_id, output_dir=output_dir):
                    route7_decision = self.route7_local_3d_replan_decision(
                        current,
                        target_pose,
                        payload,
                        local_3d_safety,
                        event.get("avoidance_gate", event) if isinstance(event, dict) else {},
                        output_dir=output_dir,
                        stage=stage,
                        target_id=target_id,
                    )
                    trace["route7_local_3d_replan_decision"] = route7_decision
                    if event:
                        event["route7_local_3d_replan_decision"] = route7_decision
                    if str(route7_decision.get("action", "") or "") == "continue_cautious":
                        self.route5_log_event(output_dir, "route7_local_3d_continue_cautious", trace)
                    elif str(route7_decision.get("action", "") or "") == "replan":
                        reason = "route7_map_route_replan_required"
                        if event:
                            event["local_3d_safety"] = local_3d_safety
                            event["avoidance_active"] = True
                            event["selected_action"] = "route7_map_route_replan_required"
                            event["selected_action_reason"] = str(route7_decision.get("reason", reason) or reason)
                            event["selected_action_payload"] = action_payload("hold")
                            event["final_payload"] = action_payload("hold")
                            event = self.route5_normalize_avoidance_event(event)
                            self.route5_write_frame_decision(output_dir, event, final_payload=action_payload("hold"))
                            self.append_jsonl(output_dir / "avoidance_events.jsonl", self.route5_json_safe(event))
                            self.route5_write_avoidance_summary(output_dir, status="running")
                        self.route5_hold(session, output_dir=output_dir, reason=reason)
                        self.route5_log_event(output_dir, "route7_map_route_replan_required", trace)
                        return {
                            "status": "blocked",
                            "reason": reason,
                            "stage": stage,
                            "facade": facade,
                            "target_id": target_id,
                            "final_target_id": target_id,
                            "final_target_pose": target_pose,
                            "target_reset_count": reset_count,
                            "pose_error": error,
                            "local_3d_safety": local_3d_safety,
                            "route7_local_3d_replan_decision": route7_decision,
                            "current_pose": current,
                        }
                if str(route7_decision.get("action", "") or "") == "continue_cautious":
                    pass
                else:
                    if event:
                        event["local_3d_safety"] = local_3d_safety
                        event["avoidance_active"] = True
                        event["selected_action"] = "route5_local_3d_safety_hold"
                        event["selected_action_reason"] = reason
                        event["selected_action_payload"] = action_payload("hold")
                        event["final_payload"] = action_payload("hold")
                        event = self.route5_normalize_avoidance_event(event)
                        self.route5_write_frame_decision(output_dir, event, final_payload=action_payload("hold"))
                        self.append_jsonl(output_dir / "avoidance_events.jsonl", self.route5_json_safe(event))
                        self.route5_write_avoidance_summary(output_dir, status="running")
                    self.route5_hold(session, output_dir=output_dir, reason=reason)
                    self.route5_log_event(output_dir, "local_3d_navigation_blocked", trace)
                    return {
                        "status": "blocked",
                        "reason": reason,
                        "stage": stage,
                        "facade": facade,
                        "target_id": target_id,
                        "final_target_id": target_id,
                        "final_target_pose": target_pose,
                        "target_reset_count": reset_count,
                        "pose_error": error,
                        "local_3d_safety": local_3d_safety,
                        "current_pose": current,
                    }
            result = self.safe("Route V5 fused movement tick", lambda p=payload: session.move_relative(p))
            if not isinstance(result, dict):
                self.route5_hold(session, output_dir=output_dir, reason="movement_failed")
                return {"status": "failed", "reason": "movement_failed", "pose_error": error, "current_pose": current, "final_target_id": target_id, "final_target_pose": target_pose, "target_reset_count": reset_count}
            self.root.after(0, lambda r=result: self.apply_state(r))
            post_pose = self.route3_pose_from_payload(result) or predicted
            if event:
                if hasattr(self, "read_obstacle_avoidance_3_hit_state"):
                    hit_state = self.read_obstacle_avoidance_3_hit_state(session)
                else:
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
                event = self.route5_normalize_avoidance_event(event)
                prediction_hint = self.route5_semantic_hint_from_prediction(event.get("representation_prediction", {}) if isinstance(event.get("representation_prediction"), dict) else {})
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
                self.route5_write_frame_decision(output_dir, event, final_payload=payload)
                if bool(event.get("avoidance_active", False)) or bool(event.get("collision_state", False)):
                    self.append_jsonl(output_dir / "avoidance_events.jsonl", self.route5_json_safe(event))
                    self.route5_write_avoidance_summary(output_dir, status="running")
                if bool(event.get("collision_state")):
                    self.route5_hold(session, output_dir=output_dir, reason="collision")
                    return {"status": "collision", "reason": "collision", "event": event, "current_pose": post_pose, "final_target_id": target_id, "final_target_pose": target_pose, "target_reset_count": reset_count}
            current = post_pose
            last_action = payload
            step_index += 1
            if self.llm_route5_stop_event.wait(max(0.01, tick_s)):
                break
        self.route5_hold(session, output_dir=output_dir, reason="stopped")
        return {"status": "stopped", "reason": "stopped", "pose_error": last_error, "current_pose": current, "final_target_id": target_id, "final_target_pose": target_pose, "target_reset_count": reset_count}

    def route5_navigate_to_pose_with_fusion(
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
        base_config = self.route5_nav_config()
        self.route5_enable_physics_movement(session)
        current = self.route3_current_pose(session)
        if not current:
            return {"status": "failed", "reason": "missing_current_pose"}
        max_replans = int(LLM_ROUTE3_ASTAR_MAX_REPLANS)
        replan_count = 0
        started_at = time.time()
        last_result: Dict[str, Any] = {}
        while replan_count <= max_replans and not self.llm_route5_stop_event.is_set():
            if self.route5_route6_full_stop_requested():
                self.llm_route5_stop_event.set()
                self.route5_hold(session, output_dir=output_dir, reason="route6_full_stop")
                return {
                    "status": "stopped",
                    "reason": "route6_full_stop",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "final_target_pose": target_pose,
                    "current_pose": current,
                    "replan_count": replan_count,
                    "elapsed_s": round(time.time() - started_at, 3),
                }
            if self.route7_should_use_map_route_planner(stage, target_id, output_dir=output_dir):
                route7_plan = self.route7_update_realtime_navigation_route(
                    current,
                    target_pose,
                    output_dir=output_dir,
                    stage=stage,
                    target_id=target_id,
                    target_house_id=target_house_id,
                    update_reason=f"navigation_plan_replan_{replan_count}",
                )
                if route7_plan.get("status") == "fallback":
                    legacy_route7_plan = self.route7_plan_navigation_waypoints_from_map(
                        current,
                        target_pose,
                        output_dir=output_dir,
                        stage=stage,
                        target_id=target_id,
                        target_house_id=target_house_id,
                    )
                    plan = self.route3_plan_navigation_waypoints(current, target_pose, target_house_id, grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM))
                    plan["route7_map_plan_fallback"] = self.route5_json_safe(route7_plan)
                    plan["route7_legacy_layer_plan"] = self.route5_json_safe(legacy_route7_plan)
                else:
                    plan = route7_plan
            else:
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
            if str(plan.get("planner_source", "") or "") == "route7_layered_occupancy_astar":
                visual = self.route7_write_navigation_plan_visualization(
                    output_dir,
                    plan,
                    current_pose=current,
                    target_pose=target_pose,
                    target_id=target_id,
                )
                plan_log["route7_plan_visualization"] = self.route5_json_safe(visual)
                if visual.get("status") == "ok":
                    plan["route7_plan_visualization_path"] = str(visual.get("visualization_path", "") or "")
            self.append_jsonl(output_dir / "route5_navigation_plan.jsonl", plan_log)
            self.route5_log_event(output_dir, "navigation_plan", plan_log)
            self.route5_update_state(current_navigation_plan=self.route5_json_safe(plan_log))
            if plan.get("status") != "ok":
                self.route5_hold(session, output_dir=output_dir, reason=str(plan.get("reason", "navigation_plan_failed")))
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
                if self.route5_route6_full_stop_requested():
                    self.llm_route5_stop_event.set()
                    self.route5_hold(session, output_dir=output_dir, reason="route6_full_stop")
                    return {
                        "status": "stopped",
                        "reason": "route6_full_stop",
                        "stage": stage,
                        "facade": facade,
                        "target_id": target_id,
                        "final_target_pose": target_pose,
                        "current_pose": current,
                        "navigation_plan": plan,
                        "replan_count": replan_count,
                        "elapsed_s": round(time.time() - started_at, 3),
                    }
                waypoint_pose = dict(waypoint)
                waypoint_pose.update(
                    {
                        "x": float(waypoint.get("x", target_pose["x"])),
                        "y": float(waypoint.get("y", target_pose["y"])),
                        "z": float(waypoint.get("z", target_pose["z"])),
                        "yaw": float(waypoint.get("yaw", target_pose["yaw"])),
                    }
                )
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
                result = self.route5_follow_navigation_waypoint_with_fusion(
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
                "reason": str(last_result.get("reason", "target_reached") if isinstance(last_result, dict) else "target_reached") or "target_reached",
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "final_target_id": str(last_result.get("final_target_id", target_id) if isinstance(last_result, dict) else target_id),
                "final_target_pose": last_result.get("final_target_pose", target_pose) if isinstance(last_result, dict) else target_pose,
                "current_pose": last_result.get("current_pose", current) if isinstance(last_result, dict) else current,
                "final_current_pose": last_result.get("final_current_pose", last_result.get("current_pose", current)) if isinstance(last_result, dict) else current,
                "target_reset_count": int(last_result.get("target_reset_count", 0) or 0) if isinstance(last_result, dict) else 0,
                "navigation_plan": plan,
                "replan_count": replan_count,
                "waypoint_count": waypoint_count,
                "pose_error": last_result.get("pose_error", {}),
                "arrival_state": last_result.get("arrival_state", {}) if isinstance(last_result.get("arrival_state", {}), dict) else {},
                "final_obstacle_event": last_result.get("final_obstacle_event", {}) if isinstance(last_result.get("final_obstacle_event", {}), dict) else {},
                "final_arrival_reason": str(
                    last_result.get(
                        "final_arrival_reason",
                        (last_result.get("arrival_state", {}) if isinstance(last_result.get("arrival_state", {}), dict) else {}).get("arrival_reason", last_result.get("reason", "")),
                    )
                    or ""
                ),
                "elapsed_s": round(time.time() - started_at, 3),
            }
        self.route5_hold(session, output_dir=output_dir, reason="replan_exhausted")
        return {"status": "blocked", "reason": "replan_exhausted", "last_result": last_result, "replan_count": replan_count}

    def route5_run_summary(self, output_dir: Path, *, status: str) -> Dict[str, Any]:
        capture_rows = self.read_jsonl_artifact(output_dir / "lidar_capture_log.jsonl")
        avoidance_rows = self.read_jsonl_artifact(output_dir / "avoidance_events.jsonl")
        attempted_facades = set(self.llm_route5_completed_facades) | set(self.llm_route5_blocked_facades)
        completion_status = dict(self.llm_route5_state.get("facade_completion_status", {}) or {})
        house_memory = self.llm_route5_state.get("house_memory", {})
        summary = {
            "mode": "route5_llm_route_oa3_or2_fusion",
            "status": status,
            "target_house_id": str(self.llm_route5_state.get("target_house_id", "") or ""),
            "output_dir": str(output_dir),
            "completed_facades": sorted(self.llm_route5_completed_facades),
            "blocked_facades": sorted(self.llm_route5_blocked_facades),
            "full_completed_facades": sorted(name for name, state in completion_status.items() if state == "full_completed"),
            "degraded_completed_facades": sorted(name for name, state in completion_status.items() if state == "degraded_completed"),
            "facade_completion_status": self.route5_json_safe(completion_status),
            "mandatory_facade_agenda": self.route5_json_safe(self.llm_route5_state.get("mandatory_facade_agenda", {})),
            "house_memory": self.route5_json_safe(house_memory),
            "oa3_config": self.route5_json_safe(self.llm_route5_state.get("oa3_config", self.route5_oa3_config())),
            "last_or2_event": self.route5_json_safe(self.llm_route5_state.get("last_or2_event", {})),
            "attempted_facades": sorted(attempted_facades),
            "capture_count": len(capture_rows),
            "avoidance_event_count": len(avoidance_rows),
            "collision_count": sum(1 for item in avoidance_rows if bool(item.get("collision_state", False))),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(output_dir / "route5_fusion_summary.json", summary)
        self.route5_write_avoidance_summary(output_dir, status=status)
        return summary

    def route5_full_search_worker(
        self,
        session: flight.DroneFlightSession,
        *,
        single_facade: bool = False,
        force_new: bool = False,
        observation_z_cm: Optional[float] = None,
        route_window_label: str = "V5",
        output_dir_override: Optional[Path] = None,
    ) -> None:
        self.ensure_route5_state()
        previous_observation_z = getattr(self, "route_observation_z_override_cm", None)
        had_previous_observation_z = hasattr(self, "route_observation_z_override_cm")
        observation_z = None
        try:
            if observation_z_cm is not None:
                observation_z = max(80.0, min(900.0, float(observation_z_cm)))
                self.route_observation_z_override_cm = observation_z
        except Exception:
            observation_z = None
        selected_target_house_id = self.selected_route_target_house_id()
        if not selected_target_house_id:
            self.root.after(0, lambda p=self.route5_status_prefix(): self.llm_route5_status_var.set(f"{p}: select a target house first."))
            if had_previous_observation_z:
                self.route_observation_z_override_cm = previous_observation_z
            elif hasattr(self, "route_observation_z_override_cm"):
                delattr(self, "route_observation_z_override_cm")
            return
        output_dir: Optional[Path] = None
        status = "done"
        try:
            output_dir = self.route5_initialize_run(
                selected_target_house_id,
                force_new=force_new,
                output_dir_override=output_dir_override,
                route_window_label=route_window_label,
            )
            self.route5_update_state(
                route_window_label=str(route_window_label or "V5").strip().upper() or "V5",
                observation_z_override_cm=observation_z,
                route7_map_output_dir=str(output_dir) if str(route_window_label or "").strip().upper() == "V7" else self.llm_route5_state.get("route7_map_output_dir", ""),
            )
            self.route5_set_stage("TASK_ANALYSIS", output_dir=output_dir, message=f"analyzing task for selected house={selected_target_house_id}")
            task_plan = self.route5_analyze_task_plan(selected_target_house_id, output_dir=output_dir)
            target_house_id = str(task_plan.get("target_house_id", selected_target_house_id) or selected_target_house_id)
            self.route5_update_state(output_dir=str(output_dir), target_house_id=target_house_id)
            self.route5_initialize_house_memory(output_dir, target_house_id)
            self.route5_set_stage("PLAN_4_FACADES", output_dir=output_dir, message=f"planning facade candidates for house={target_house_id}")
            self.route5_set_control_lock(True)
            while not self.llm_route5_stop_event.is_set():
                completed = set(self.llm_route5_completed_facades)
                terminal_failed = set(self.llm_route5_blocked_facades)
                terminal_failed.update(
                    str(item).strip().lower()
                    for item in (
                        self.llm_route5_state.get("terminal_failed_facades", [])
                        if isinstance(self.llm_route5_state.get("terminal_failed_facades", []), list)
                        else []
                    )
                    if str(item).strip().lower()
                )
                blocked = set(terminal_failed)
                if len(completed | terminal_failed) >= 4:
                    break
                if str(route_window_label or "").strip().upper() == "V7":
                    facade_candidates = self.route7_edge_observation_candidates_for_house(target_house_id, output_dir=output_dir)
                else:
                    facade_candidates = self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False)
                ranked_candidates = self.route5_rank_observation_candidates(target_house_id, facade_candidates, completed, blocked, start_pose=self.route3_current_pose(session))
                self.route5_update_state(ranked_facade_candidates=ranked_candidates)
                decision = self.route5_decide_next_facade(target_house_id, ranked_candidates, completed, blocked)
                self.route5_log_event(output_dir, "high_level_decision", decision)
                if self.llm_route5_stop_event.is_set():
                    status = "stopped"
                    break
                if decision.get("next_action") == "done" or decision.get("stop_condition_met"):
                    break
                facade = str(decision.get("target_facade", "") or "").strip().lower()
                selected = next((item for item in ranked_candidates if str(item.get("facade", "") or "") == facade), {})
                if not facade or not selected:
                    status = "blocked_no_facade"
                    break
                self.route5_set_stage("SELECT_NEXT_FACADE", output_dir=output_dir, facade=facade, message=f"selected facade={facade}")
                self.route5_update_house_memory(
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
                original_obs_pose: Dict[str, float] = {}
                nav_result: Dict[str, Any] = {}
                for attempt_index, attempt in enumerate(observation_attempts, start=1):
                    if self.llm_route5_stop_event.is_set():
                        break
                    attempt_status = str(attempt.get("status", "") or "")
                    if attempt_status == "blocked" and str(attempt.get("route5_navigation_status", attempt.get("route3_navigation_status", "")) or "") != "ok":
                        self.route5_log_event(output_dir, "observation_attempt_skipped", {"facade": facade, "attempt_index": attempt_index, "attempt": attempt})
                        continue
                    self.apply_route2_observation_plan(target_house_id, attempt, ranked_candidates, status_label=f"v5 selected attempt {attempt_index}/{len(observation_attempts)}")
                    observation = self.route2_selected_state().get("observation_point", attempt)
                    obs_pose = self.route3_target_pose_from_point(observation)
                    original_obs_pose = dict(obs_pose)
                    obs_target_id = f"{target_house_id}_{facade}_obs_attempt_{attempt_index}"
                    skip_targets = self.llm_route5_state.get("capture_guard_skip_targets", {}) if isinstance(self.llm_route5_state, dict) else {}
                    if isinstance(skip_targets, dict) and obs_target_id in skip_targets:
                        self.route5_log_event(
                            output_dir,
                            "observation_attempt_skipped_capture_guard_repeat",
                            {"facade": facade, "attempt_index": attempt_index, "target_id": obs_target_id, "skip": skip_targets.get(obs_target_id, {})},
                        )
                        continue
                    self.route5_update_state(
                        selected_observation_attempt=observation,
                        current_exploration_status={
                            "stage": "NAV_TO_OBS",
                            "facade": facade,
                            "target_id": obs_target_id,
                            "observation_attempt_index": attempt_index,
                            "observation_attempt_count": len(observation_attempts),
                            "target_pose": obs_pose,
                        },
                    )
                    self.route5_set_stage("NAV_TO_OBS", output_dir=output_dir, facade=facade, target=obs_pose, message=f"navigating to {facade} observation attempt {attempt_index}/{len(observation_attempts)}")
                    nav_result = self.route5_navigate_to_pose_with_fusion(
                        session,
                        obs_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_OBS",
                        facade=facade,
                        target_id=obs_target_id,
                        target_house_id=target_house_id,
                    )
                    self.route5_log_event(output_dir, "observation_attempt_navigation_result", {"facade": facade, "attempt_index": attempt_index, "navigation": nav_result})
                    if nav_result.get("status") != "ok":
                        rescue_result = self.route5_try_observation_rescue(
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
                                "reason": "route5_observation_rescue_capture_current",
                                "stage": "NAV_TO_OBS",
                                "facade": facade,
                                "target_id": f"{target_house_id}_{facade}_obs_rescue_{attempt_index}",
                                "route5_observation_rescue": rescue_result,
                                "source_navigation_status": rescue_result.get("source_navigation", {}).get("status"),
                                "source_navigation_reason": rescue_result.get("source_navigation", {}).get("reason"),
                            }
                            self.route5_log_event(output_dir, "observation_attempt_navigation_rescued", {"facade": facade, "attempt_index": attempt_index, "navigation": nav_result})
                            break
                    if nav_result.get("status") == "ok":
                        if isinstance(nav_result.get("final_target_pose"), dict):
                            obs_pose = dict(nav_result.get("final_target_pose", obs_pose))
                            observation = dict(observation)
                            observation.update(
                                {
                                    "x": obs_pose.get("x", observation.get("x")),
                                    "y": obs_pose.get("y", observation.get("y")),
                                    "z": obs_pose.get("z", observation.get("z")),
                                    "yaw": obs_pose.get("yaw", observation.get("yaw", observation.get("yaw_deg", 0.0))),
                                    "route5_target_reset_navigation": self.route5_json_safe(nav_result),
                                }
                            )
                        break
                if self.llm_route5_stop_event.is_set():
                    status = "stopped"
                    break
                if nav_result.get("status") != "ok":
                    degraded_observation = self.route5_degraded_observation_candidate(
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
                            status_label=f"v5 degraded observation {facade}",
                        )
                        observation = degraded_observation
                        obs_pose = self.route3_target_pose_from_point(degraded_observation)
                        original_obs_pose = dict(obs_pose)
                        nav_result = {
                            "status": "ok",
                            "reason": "route5_degraded_observation_capture",
                            "stage": "NAV_TO_OBS",
                            "facade": facade,
                            "target_id": f"{target_house_id}_{facade}_obs_degraded",
                            "route5_degraded_observation": degraded_observation,
                            "source_navigation_status": nav_result.get("status"),
                            "source_navigation_reason": nav_result.get("reason"),
                        }
                        self.route5_update_house_memory(
                            output_dir,
                            target_house_id,
                            facade,
                            status="degraded_completed",
                            reason="observation_navigation_degraded",
                            observation_attempt=degraded_observation,
                            nav_result=nav_result,
                            safe_observation_pose=degraded_observation,
                        )
                        self.route5_log_event(output_dir, "observation_navigation_degraded", {"facade": facade, "observation": degraded_observation, "navigation": nav_result})
                    else:
                        retry_info = self.route5_observation_failure_retry_status(nav_result)
                        retry_counts = dict(self.llm_route5_state.get("facade_retry_counts", {}) if isinstance(self.llm_route5_state.get("facade_retry_counts"), dict) else {})
                        retry_count = int(retry_counts.get(facade, 0) or 0) + 1
                        retry_counts[facade] = retry_count
                        terminal = bool(retry_info.get("terminal", False)) or retry_count >= self.route5_max_facade_retry_count()
                        retry_status = "terminal_blocked" if terminal else str(retry_info.get("status", "soft_blocked_retryable") or "soft_blocked_retryable")
                        self.route5_update_house_memory(
                            output_dir,
                            target_house_id,
                            facade,
                            status=retry_status,
                            reason=str(nav_result.get("reason", "observation_navigation_failed") or "observation_navigation_failed"),
                            observation_attempt=observation if isinstance(observation, dict) else selected,
                            nav_result=nav_result,
                        )
                        if terminal:
                            self.llm_route5_blocked_facades.add(facade)
                        blocked_reasons = dict(self.llm_route5_state.get("blocked_facade_reasons", {}) if isinstance(self.llm_route5_state.get("blocked_facade_reasons"), dict) else {})
                        blocked_reasons[facade] = {
                            "reason": str(nav_result.get("reason", "observation_navigation_failed") or "observation_navigation_failed"),
                            "last_navigation_result": nav_result,
                            "retry_info": retry_info,
                            "retry_count": retry_count,
                            "terminal": terminal,
                        }
                        terminal_failed = sorted(
                            set(self.llm_route5_state.get("terminal_failed_facades", []) if isinstance(self.llm_route5_state.get("terminal_failed_facades"), list) else [])
                            | ({facade} if terminal else set())
                        )
                        self.route5_update_state(
                            blocked_facades=sorted(self.llm_route5_blocked_facades),
                            blocked_facade_reasons=blocked_reasons,
                            terminal_failed_facades=terminal_failed,
                            facade_retry_counts=retry_counts,
                        )
                        self.route5_write_state_artifact()
                        if single_facade:
                            break
                        continue
                capture_pose = self.route3_current_pose(session)
                observation_arrival = self.route5_capture_guard_arrival_state(
                    nav_result=nav_result,
                    stage="NAV_TO_OBS",
                    facade=facade,
                    target_id=str(nav_result.get("target_id", f"{target_house_id}_{facade}_obs") or f"{target_house_id}_{facade}_obs"),
                    original_target_pose=original_obs_pose or obs_pose,
                    runtime_target_pose=obs_pose,
                    capture_pose=capture_pose,
                    config=self.route5_nav_config(),
                )
                observation_guard = self.route5_capture_guard_state(
                    target_house_id=target_house_id,
                    stage="NAV_TO_OBS",
                    facade=facade,
                    target_id=str(nav_result.get("target_id", f"{target_house_id}_{facade}_obs") or f"{target_house_id}_{facade}_obs"),
                    original_target_pose=original_obs_pose or obs_pose,
                    runtime_target_pose=obs_pose,
                    capture_pose=capture_pose,
                    pose_error=nav_result.get("pose_error", {}) if isinstance(nav_result.get("pose_error"), dict) else {},
                    arrival_state=observation_arrival,
                    config=self.route5_nav_config(),
                    capture_kind="observation",
                )
                observation_guard = self.route5_record_capture_guard(output_dir, observation_guard)
                if not bool(observation_guard.get("capture_guard_passed", False)):
                    self.route5_write_capture_guard_blocked_decision(output_dir, observation_guard, current_pose=capture_pose)
                    self.route5_update_house_memory(
                        output_dir,
                        target_house_id,
                        facade,
                        status="invalid_capture_pose_retryable",
                        reason=str(observation_guard.get("reason", "capture_guard_failed") or "capture_guard_failed"),
                        observation_attempt=observation if isinstance(observation, dict) else selected,
                        nav_result=nav_result,
                    )
                    self.route5_log_event(output_dir, "observation_capture_guard_blocked", observation_guard)
                    if single_facade:
                        status = "single_facade_capture_guard_blocked"
                        break
                    continue
                self.route5_set_stage("CAPTURE_RGB", output_dir=output_dir, facade=facade, target=obs_pose, message=f"capturing {facade} RGB")
                rgb_result = self.route3_capture_facade_rgb_current(session, output_dir=output_dir, facade_dir=facade_dir, house_id=target_house_id, facade=facade, planned_pose=obs_pose)
                self.route5_log_event(output_dir, "facade_rgb_capture", rgb_result)
                self.root.after(0, self.refresh_route5_support_views)
                if str(route_window_label or "").strip().upper() == "V7":
                    self.route5_set_stage("BUILD_OBSTACLE_MAP", output_dir=output_dir, facade=facade, target=obs_pose, message=f"building {facade} obstacle map from observation")
                    map_build = self.route7_build_update_map_after_observation(
                        output_dir,
                        facade=facade,
                        observation=observation if isinstance(observation, dict) else {},
                        rgb_result=rgb_result if isinstance(rgb_result, dict) else {},
                    )
                    self.route5_log_event(output_dir, "route7_observation_map_build", map_build)
                    fallback_analysis = self.route2_fallback_facade_analysis("Route V7 uses selected layered occupancy map for direct house-edge capture.")
                    self.route2_update_state(facade_analysis=fallback_analysis)
                    self.route2_write_state_artifact()
                    self.route5_set_stage("PLAN_MAP_LAYER_SCAN", output_dir=output_dir, facade=facade, message=f"planning {facade} map-layer edge capture")
                    plan_result = self.route7_plan_facade_map_layer_scan_current()
                else:
                    self.route5_set_stage("ANALYZE_VLM", output_dir=output_dir, facade=facade, message=f"analyzing {facade} facade")
                    self.route2_analyze_facade_vlm_worker()
                    self.root.after(0, self.refresh_route5_support_views)
                    self.route5_set_stage("PLAN_SCAN", output_dir=output_dir, facade=facade, message=f"planning {facade} scan")
                    plan_result = self.route5_plan_facade_scan_current()
                points = [point for point in plan_result.get("points", []) if isinstance(point, dict)]
                scan_counts = plan_result.get("scan_counts", {}) if isinstance(plan_result.get("scan_counts"), dict) else {}
                self.route5_log_event(
                    output_dir,
                    "facade_scan_plan",
                    {
                        "facade": facade,
                        "point_count": len(points),
                        "physical_axis_sample_count": scan_counts.get("physical_axis_sample_count"),
                        "total_capture_record_count": scan_counts.get("total_capture_record_count", len(points)),
                        "yaw_supplement_record_count": scan_counts.get("yaw_supplement_record_count"),
                        "route5_scan_boundary_policy": plan_result.get("boundary_policy", {}),
                        "validation": plan_result.get("validation", {}),
                    },
                )
                if scan_counts:
                    physical = int(scan_counts.get("physical_axis_sample_count", len(points)) or 0)
                    total_records = int(scan_counts.get("total_capture_record_count", len(points)) or len(points))
                    self.root.after(
                        0,
                        lambda f=facade, p=physical, t=total_records, prefix=self.route5_status_prefix(): self.llm_route5_status_var.set(
                            f"{prefix}: planned {f} scan physical={p} capture/yaw={t}"
                        ),
                    )
                if not points:
                    self.route5_mark_facade_degraded_completed(
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
                scan_capture_count = 0
                scan_loop_stopped = False
                for idx, point in enumerate(points, start=1):
                    if self.llm_route5_stop_event.is_set():
                        scan_loop_stopped = True
                        break
                    scan_id = str(point.get("scan_id", "") or f"{facade}_{idx}")
                    original_scan_pose = self.route3_target_pose_from_point(point)
                    target_pose = dict(original_scan_pose)
                    point["route5_original_target_pose"] = self.route5_json_safe(original_scan_pose)
                    self.route5_update_state(
                        current_exploration_status={
                            "stage": "NAV_TO_SCAN_POINT",
                            "facade": facade,
                            "target_id": scan_id,
                            "point_index": idx,
                            "point_total": total,
                            "scan_id": scan_id,
                            "target_pose": target_pose,
                        }
                    )
                    self.route5_set_stage("NAV_TO_SCAN_POINT", output_dir=output_dir, facade=facade, target=target_pose, message=f"scan {idx}/{total} {scan_id}")
                    nav_scan = self.route5_navigate_to_pose_with_fusion(
                        session,
                        target_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_SCAN_POINT",
                        facade=facade,
                        target_id=scan_id,
                        target_house_id=target_house_id,
                    )
                    self.route5_log_event(output_dir, "scan_navigation_result", nav_scan)
                    if nav_scan.get("status") != "ok":
                        point["status"] = "blocked"
                        point["block_reason"] = nav_scan.get("reason", "navigation_failed")
                        continue
                    if isinstance(nav_scan.get("final_target_pose"), dict):
                        target_pose = dict(nav_scan.get("final_target_pose", target_pose))
                        point["route5_target_reset_navigation"] = self.route5_json_safe(nav_scan)
                        point["route5_runtime_target_pose"] = self.route5_json_safe(target_pose)
                    capture_pose = self.route3_current_pose(session)
                    scan_arrival = self.route5_capture_guard_arrival_state(
                        nav_result=nav_scan,
                        stage="NAV_TO_SCAN_POINT",
                        facade=facade,
                        target_id=scan_id,
                        original_target_pose=original_scan_pose,
                        runtime_target_pose=target_pose,
                        capture_pose=capture_pose,
                        config=self.route5_nav_config(),
                    )
                    scan_guard = self.route5_capture_guard_state(
                        target_house_id=target_house_id,
                        stage="NAV_TO_SCAN_POINT",
                        facade=facade,
                        target_id=scan_id,
                        original_target_pose=original_scan_pose,
                        runtime_target_pose=target_pose,
                        capture_pose=capture_pose,
                        pose_error=nav_scan.get("pose_error", {}) if isinstance(nav_scan.get("pose_error"), dict) else {},
                        arrival_state=scan_arrival,
                        config=self.route5_nav_config(),
                        capture_kind="scan",
                    )
                    scan_guard = self.route5_record_capture_guard(output_dir, scan_guard)
                    point["route5_capture_guard"] = self.route5_json_safe(scan_guard)
                    if not bool(scan_guard.get("capture_guard_passed", False)):
                        point["status"] = "invalid_capture_pose_retryable"
                        point["block_reason"] = str(scan_guard.get("reason", "capture_guard_failed") or "capture_guard_failed")
                        self.route5_write_capture_guard_blocked_decision(output_dir, scan_guard, current_pose=capture_pose)
                        self.route5_log_event(output_dir, "scan_capture_guard_blocked", {"scan_id": scan_id, "guard": scan_guard})
                        continue
                    self.route5_set_stage("CAPTURE_SCAN", output_dir=output_dir, facade=facade, target=target_pose, message=f"capturing {scan_id}")
                    capture = self.route3_capture_scan_point_current(session, output_dir=output_dir, facade_dir=facade_dir, point=point, planned_pose=target_pose)
                    capture = self.route5_annotate_scan_capture_guard(
                        output_dir,
                        facade_dir,
                        scan_id=scan_id,
                        guard=scan_guard,
                        original_target_pose=original_scan_pose,
                        runtime_target_pose=target_pose,
                        capture_pose=capture_pose,
                        capture=capture if isinstance(capture, dict) else {},
                    )
                    self.route5_log_event(output_dir, "scan_capture", {"scan_id": scan_id, **capture})
                    capture_status = str((capture if isinstance(capture, dict) else {}).get("status", (capture if isinstance(capture, dict) else {}).get("capture_status", "")) or "").strip().lower()
                    if capture_status == "ok":
                        scan_capture_count += 1
                    progress = 100.0 * (len(self.llm_route5_completed_facades) + (idx / max(1, total))) / 4.0
                    self.root.after(0, lambda v=progress: self.llm_route5_progress_var.set(max(0.0, min(100.0, v))))
                    self.root.after(0, lambda i=idx, t=total, f=facade: self.llm_route5_progress_text_var.set(f"Fusion: {f} {i}/{t}"))
                    self.root.after(0, self.refresh_route5_support_views)
                if scan_loop_stopped or self.llm_route5_stop_event.is_set():
                    status = "stopped"
                    self.route5_log_event(output_dir, "scan_loop_stopped", {"facade": facade, "captured_scan_count": scan_capture_count})
                    break
                self.route5_set_stage("VALIDATE_FACADE", output_dir=output_dir, facade=facade, message=f"validating {facade}")
                postprocess_result = self.route5_refresh_facade_scan_pointcloud_rows(output_dir, facade_dir, facade)
                valid_scan_capture_count = self.route5_valid_scan_capture_count_from_logs(output_dir, facade_dir, facade)
                self.route2_write_lidar_summary(output_dir, running=False)
                validation = self.route2_validate_facade()
                self.route5_log_event(output_dir, "facade_validation", validation)
                completion_gate = self.route5_facade_completion_gate(
                    validation if isinstance(validation, dict) else {},
                    observation=observation if isinstance(observation, dict) else {},
                    rgb_result=rgb_result if isinstance(rgb_result, dict) else {},
                    scan_capture_count=scan_capture_count,
                    valid_scan_capture_count=valid_scan_capture_count,
                    scan_loop_stopped=False,
                    postprocess_result=postprocess_result,
                )
                completion_kind = str(completion_gate.get("completion_status", "scan_incomplete") or "scan_incomplete")
                completed_order = [
                    str(item).strip().lower()
                    for item in (
                        self.llm_route5_state.get("completed_facade_order", [])
                        if isinstance(self.llm_route5_state.get("completed_facade_order"), list)
                        else []
                    )
                    if str(item).strip().lower()
                ]
                completion_status = dict(self.llm_route5_state.get("facade_completion_status", {}) if isinstance(self.llm_route5_state.get("facade_completion_status"), dict) else {})
                completion_status[facade] = completion_kind
                if not bool(completion_gate.get("complete", False)):
                    retry_counts = dict(self.llm_route5_state.get("facade_retry_counts", {}) if isinstance(self.llm_route5_state.get("facade_retry_counts"), dict) else {})
                    retry_count = int(retry_counts.get(facade, 0) or 0) + 1
                    retry_counts[facade] = retry_count
                    terminal = retry_count >= self.route5_max_facade_retry_count()
                    if terminal:
                        completion_kind = "terminal_blocked"
                        completion_status[facade] = completion_kind
                        self.llm_route5_blocked_facades.add(facade)
                    blocked_reasons = dict(self.llm_route5_state.get("blocked_facade_reasons", {}) if isinstance(self.llm_route5_state.get("blocked_facade_reasons"), dict) else {})
                    blocked_reasons[facade] = {
                        "reason": str(completion_gate.get("reason", "scan_incomplete") or "scan_incomplete"),
                        "completion_gate": self.route5_json_safe(completion_gate),
                        "validation": self.route5_json_safe(validation if isinstance(validation, dict) else {}),
                        "postprocess_result": self.route5_json_safe(postprocess_result),
                        "retry_count": retry_count,
                        "terminal": terminal,
                    }
                    terminal_failed = sorted(
                        set(self.llm_route5_state.get("terminal_failed_facades", []) if isinstance(self.llm_route5_state.get("terminal_failed_facades"), list) else [])
                        | ({facade} if terminal else set())
                    )
                    self.route5_update_state(
                        completed_facades=sorted(self.llm_route5_completed_facades),
                        blocked_facades=sorted(self.llm_route5_blocked_facades),
                        blocked_facade_reasons=blocked_reasons,
                        facade_completion_status=completion_status,
                        facade_retry_counts=retry_counts,
                        terminal_failed_facades=terminal_failed,
                    )
                    self.route5_update_house_memory(
                        output_dir,
                        target_house_id,
                        facade,
                        status=completion_kind if terminal else str(completion_gate.get("completion_status", completion_kind) or completion_kind),
                        reason=str(completion_gate.get("reason", completion_kind) or completion_kind),
                        observation_attempt=observation,
                        scan_coverage={**scan_counts, **completion_gate},
                        entrance_candidates=validation.get("entrance_candidates", []) if isinstance(validation, dict) else [],
                        safe_observation_pose=observation,
                    )
                    self.route5_log_event(
                        output_dir,
                        "facade_scan_incomplete",
                        {
                            "facade": facade,
                            "completion_gate": completion_gate,
                            "validation": validation,
                            "captured_scan_count": scan_capture_count,
                        },
                    )
                    self.route5_write_state_artifact()
                    self.root.after(0, self.refresh_route5_support_views)
                    if single_facade:
                        status = "single_facade_incomplete"
                        break
                    self.route5_set_stage("DECIDE_NEXT", output_dir=output_dir, facade=facade, message=f"{facade} scan incomplete; deciding next facade")
                    continue
                self.llm_route5_completed_facades.add(facade)
                self.llm_route5_blocked_facades.discard(facade)
                if facade not in completed_order:
                    completed_order.append(facade)
                self.route5_update_state(
                    completed_facades=sorted(self.llm_route5_completed_facades),
                    completed_facade_order=completed_order,
                    last_completed_facade=facade,
                    blocked_facades=sorted(self.llm_route5_blocked_facades),
                    facade_completion_status=completion_status,
                )
                self.route5_update_house_memory(
                    output_dir,
                    target_house_id,
                    facade,
                    status=completion_kind,
                    reason=str(validation.get("reason", completion_kind) if isinstance(validation, dict) else completion_kind),
                    observation_attempt=observation,
                    scan_coverage={**scan_counts, **completion_gate},
                    entrance_candidates=validation.get("entrance_candidates", []) if isinstance(validation, dict) else [],
                    obstacle_label_conflict=(
                        self.llm_route5_state.get("last_obstacle_event", {}).get("obstacle_label_conflict")
                        if isinstance(self.llm_route5_state.get("last_obstacle_event"), dict)
                        else None
                    ),
                    safe_observation_pose=observation,
                )
                self.route5_write_state_artifact()
                self.root.after(0, self.refresh_route5_support_views)
                if single_facade:
                    status = "single_facade_complete"
                    break
                self.route5_set_stage("DECIDE_NEXT", output_dir=output_dir, facade=facade, message=f"{facade} complete; deciding next facade")
            if self.llm_route5_stop_event.is_set():
                status = "stopped"
            status = self.route5_final_status_for_task_lock(status)
            final_stage = "DONE" if status == "done" else ("DONE_WITH_BLOCKED" if status == "done_with_blocked" else ("BLOCKED_SELECTED_HOUSE" if status == "blocked_selected_house_incomplete" else "DECIDE_NEXT"))
            self.route5_set_stage(final_stage, output_dir=output_dir, message=f"fused autosearch {status}")
            summary = self.route5_run_summary(output_dir, status=status)
            self.route5_log_event(output_dir, "summary", summary)
            self.root.after(0, lambda s=status, d=output_dir, p=self.route5_status_prefix(): self.llm_route5_status_var.set(f"{p}: {s} -> {d}"))
            self.root.after(0, self.refresh_route5_support_views)
        except Exception as exc:
            LOGGER.warning("Route V5 fused autosearch failed: %s", exc)
            if output_dir is not None:
                self.route5_log_event(output_dir, "error", {"reason": str(exc)})
                self.route5_run_summary(output_dir, status="failed")
            self.root.after(0, lambda e=exc, p=self.route5_status_prefix(): self.llm_route5_status_var.set(f"{p}: failed: {e}"))
        finally:
            self.route5_set_control_lock(False)
            if had_previous_observation_z:
                self.route_observation_z_override_cm = previous_observation_z
            elif hasattr(self, "route_observation_z_override_cm"):
                delattr(self, "route_observation_z_override_cm")

    def refresh_route5_preview(self) -> None:
        preview_text = getattr(self, "llm_route5_preview_text", None)
        if preview_text is None:
            return
        payload = {
            "fusion_state": self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {},
            "active_facade_state": self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {},
        }
        try:
            if not preview_text.winfo_exists():
                self.llm_route5_preview_text = None
                return
            preview_text.configure(state="normal")
            preview_text.delete("1.0", "end")
            preview_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            preview_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route5_preview_text = None

    def refresh_route5_analysis_view(self) -> None:
        analysis_text = getattr(self, "llm_route5_analysis_text", None)
        if analysis_text is None:
            return
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        obstacle_event = self.llm_route5_state.get("last_obstacle_event", {}) if isinstance(self.llm_route5_state, dict) else {}
        capture_analysis = self.llm_route5_state.get("last_capture_analysis", {}) if isinstance(self.llm_route5_state, dict) else {}
        payload = {
            "facade_analysis": analysis or {"status": "No facade analysis yet."},
            "last_obstacle_event": obstacle_event,
            "last_capture_analysis": capture_analysis,
        }
        try:
            if not analysis_text.winfo_exists():
                self.llm_route5_analysis_text = None
                return
            analysis_text.configure(state="normal")
            analysis_text.delete("1.0", "end")
            analysis_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            analysis_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route5_analysis_text = None

    def refresh_route5_rgb_display(self) -> None:
        widget = getattr(self, "llm_route5_rgb_label", None)
        if widget is None:
            return
        image_path = self.route2_current_rgb_path()
        if image_path is None:
            try:
                self.route2_draw_rgb_preview_message(widget, "No facade RGB")
                self.llm_route5_rgb_photo = None
            except tk.TclError:
                self.llm_route5_rgb_label = None
            return
        try:
            image = Image.open(image_path).convert("RGB")
            photo = ImageTk.PhotoImage(self.route2_rgb_preview_image(image, widget))
            self.route2_draw_rgb_preview_photo(widget, photo)
            self.llm_route5_rgb_photo = photo
        except Exception as exc:
            LOGGER.warning("Refresh Route6_entrance_search v5 facade RGB failed: %s", exc)
            try:
                self.route2_draw_rgb_preview_message(widget, f"RGB preview failed: {exc}")
                self.llm_route5_rgb_photo = None
            except tk.TclError:
                self.llm_route5_rgb_label = None

    def route5_capture_analysis_dir(self, run_dir: Path) -> Path:
        return Path(run_dir).resolve() / "route5_capture_analysis"

    def route5_capture_analysis_paths_for_row(self, run_dir: Path, row: Dict[str, Any]) -> Dict[str, Path]:
        row = row if isinstance(row, dict) else {}
        capture_dir = self.route5_resolve_capture_dir(run_dir, row.get("capture_dir", ""))
        if capture_dir is None:
            capture_dir = Path()
        rgb_path = self.route5_resolve_capture_dir(run_dir, row.get("rgb_path", ""))
        if rgb_path is None or not rgb_path.is_file():
            rgb_path = capture_dir / "rgb.png"
        depth_path = self.route5_resolve_capture_dir(run_dir, row.get("depth_npy_path", row.get("depth_path", "")))
        if depth_path is None or not depth_path.is_file():
            depth_path = capture_dir / "depth.npy"
        camera_info_path = self.route5_resolve_capture_dir(run_dir, row.get("camera_info_path", ""))
        if camera_info_path is None or not camera_info_path.is_file():
            camera_info_path = capture_dir / "camera_info.json"
        return {"capture_dir": capture_dir, "rgb_path": rgb_path, "depth_npy_path": depth_path, "camera_info_path": camera_info_path}

    def route5_read_capture_analysis_log_rows(self, run_dir: Path) -> List[Dict[str, Any]]:
        run_dir = Path(run_dir).resolve()
        sources = [run_dir / "lidar_capture_log.jsonl"]
        facade_root = run_dir / "facade_observations"
        if facade_root.is_dir():
            sources.extend(sorted(facade_root.glob("*/lidar_capture_log.jsonl")))
        rows: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for path in sources:
            for row in self.read_jsonl_artifact(path):
                item = dict(row if isinstance(row, dict) else {})
                scan_id = str(item.get("scan_id", "") or "")
                capture_dir = str(item.get("capture_dir", "") or "")
                key = (scan_id, capture_dir)
                if key in seen:
                    continue
                seen.add(key)
                item["source_log_path"] = str(path)
                rows.append(item)
        return rows

    def route5_build_capture_analysis_manifest(self, run_dir: Path) -> Dict[str, Any]:
        run_dir = Path(run_dir).resolve()
        analysis_dir = self.route5_capture_analysis_dir(run_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        included: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        for index, row in enumerate(self.route5_read_capture_analysis_log_rows(run_dir), start=1):
            item = dict(row)
            scan_id = str(item.get("scan_id", "") or "").strip()
            capture_kind = str(item.get("capture_kind", "") or "").strip().lower()
            status = str(item.get("capture_status", item.get("status", "ok")) or "ok").strip().lower()
            paths = self.route5_capture_analysis_paths_for_row(run_dir, item)
            reason = ""
            if not scan_id:
                reason = "not_scan_capture"
            elif "observation" in capture_kind and "scan" not in capture_kind:
                reason = "not_scan_capture"
            elif status not in {"ok", "captured", "done", ""}:
                reason = f"capture_status_{status or 'unknown'}"
            elif "capture_guard_passed" not in item:
                reason = "legacy_unverified"
            elif not bool(item.get("capture_guard_passed", False)):
                reason = "capture_guard_failed"
            elif not paths["capture_dir"].is_dir():
                reason = "capture_dir_missing"
            elif not paths["rgb_path"].is_file():
                reason = "rgb_missing"
            elif not paths["depth_npy_path"].is_file():
                reason = "depth_missing"
            elif not paths["camera_info_path"].is_file():
                reason = "camera_info_missing"
            manifest_row = {
                **self.route5_json_safe(item),
                "manifest_index": index,
                "capture_dir": str(paths["capture_dir"]),
                "rgb_path": str(paths["rgb_path"]),
                "depth_npy_path": str(paths["depth_npy_path"]),
                "camera_info_path": str(paths["camera_info_path"]),
            }
            if reason:
                excluded.append({**manifest_row, "reason": reason})
            else:
                included.append(manifest_row)
        manifest = {
            "schema": "route5_capture_analysis_manifest_v1",
            "run_dir": str(run_dir),
            "analysis_dir": str(analysis_dir),
            "included_count": len(included),
            "excluded_count": len(excluded),
            "included_captures": included,
            "excluded_captures": excluded,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(analysis_dir / "selected_capture_manifest.json", manifest)
        return manifest

    def route5_capture_manifest_frames(self, manifest: Dict[str, Any]) -> List[lidar_yolo_analysis.LidarYoloFrame]:
        frames: List[lidar_yolo_analysis.LidarYoloFrame] = []
        for row in manifest.get("included_captures", []) if isinstance(manifest.get("included_captures"), list) else []:
            if not isinstance(row, dict):
                continue
            capture_dir = Path(str(row.get("capture_dir", "") or ""))
            rgb_path = Path(str(row.get("rgb_path", "") or ""))
            depth_path = Path(str(row.get("depth_npy_path", "") or ""))
            camera_info_path = Path(str(row.get("camera_info_path", "") or ""))
            try:
                frame_index = int(float(row.get("frame_index", 0) or lidar_yolo_analysis.parse_frame_index(capture_dir.name)))
            except Exception:
                frame_index = lidar_yolo_analysis.parse_frame_index(capture_dir.name)
            if capture_dir.is_dir() and rgb_path.is_file() and depth_path.is_file() and camera_info_path.is_file():
                frames.append(
                    lidar_yolo_analysis.LidarYoloFrame(
                        frame_index=frame_index,
                        frame_name=capture_dir.name,
                        capture_dir=capture_dir,
                        rgb_path=rgb_path,
                        depth_npy_path=depth_path,
                        camera_info_path=camera_info_path,
                        capture_json_path=capture_dir / "capture.json",
                    )
                )
        return frames

    def route5_write_capture_analysis_reports(
        self,
        run_dir: Path,
        manifest: Dict[str, Any],
        yolo_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        analysis_dir = self.route5_capture_analysis_dir(run_dir)
        yolo_result = yolo_result if isinstance(yolo_result, dict) else {}
        yolo_rows: List[Dict[str, Any]] = []
        for frame in yolo_result.get("frames", []) if isinstance(yolo_result.get("frames"), list) else []:
            row = dict(frame if isinstance(frame, dict) else {})
            yolo_rows.append(row)
        with (analysis_dir / "yolo_detections.jsonl").open("w", encoding="utf-8") as fh:
            for row in yolo_rows:
                fh.write(json.dumps(self.route5_json_safe(row), ensure_ascii=False) + "\n")
        pointcloud_manifest = {
            "semantic_point_count": int(yolo_result.get("semantic_point_count", 0) or 0),
            "semantic_points_path": str(yolo_result.get("semantic_points_path", "") or ""),
            "semantic_points_ply_path": str(yolo_result.get("semantic_points_ply_path", "") or ""),
            "semantic_points_pcd_path": str(yolo_result.get("semantic_points_pcd_path", "") or ""),
            "selected_capture_count": int(manifest.get("included_count", 0) or 0),
        }
        self.write_json_artifact(analysis_dir / "pointcloud_manifest.json", pointcloud_manifest)
        by_facade: Dict[str, Dict[str, Any]] = {}
        for row in manifest.get("included_captures", []) if isinstance(manifest.get("included_captures"), list) else []:
            facade = str(row.get("facade", "") or "unknown")
            slot = by_facade.setdefault(facade, {"facade": facade, "capture_count": 0, "scan_ids": []})
            slot["capture_count"] += 1
            scan_id = str(row.get("scan_id", "") or "")
            if scan_id:
                slot["scan_ids"].append(scan_id)
        coverage = {"facades": sorted(by_facade.values(), key=lambda item: item["facade"]), "included_capture_count": int(manifest.get("included_count", 0) or 0)}
        self.write_json_artifact(analysis_dir / "facade_coverage_report.json", coverage)
        labels_path = str(yolo_result.get("semantic_labels_path", "") or "")
        entrance_candidates: List[Dict[str, Any]] = []
        if labels_path and Path(labels_path).is_file():
            labels_payload = lidar_yolo_analysis.load_json(Path(labels_path), {})
            for label in labels_payload.get("labels", []) if isinstance(labels_payload.get("labels"), list) else []:
                if isinstance(label, dict):
                    class_name = str(label.get("class_name", label.get("class_name_normalized", "")) or "").lower()
                    if "door" in class_name:
                        entrance_candidates.append(label)
        self.write_json_artifact(analysis_dir / "entrance_candidates.json", {"candidate_count": len(entrance_candidates), "candidates": entrance_candidates})
        summary = {
            "status": str(yolo_result.get("status", "manifest_only") or "manifest_only"),
            "run_dir": str(Path(run_dir).resolve()),
            "analysis_dir": str(analysis_dir),
            "included_count": int(manifest.get("included_count", 0) or 0),
            "excluded_count": int(manifest.get("excluded_count", 0) or 0),
            "yolo_processed_frame_count": int(yolo_result.get("processed_frame_count", len(yolo_rows)) or 0),
            "semantic_label_count": int(yolo_result.get("semantic_label_count", 0) or 0),
            "semantic_point_count": int(yolo_result.get("semantic_point_count", 0) or 0),
            "latest_annotated_path": str((yolo_rows[-1].get("yolo_annotated_path", "") if yolo_rows else "") or ""),
            "manifest_path": str(analysis_dir / "selected_capture_manifest.json"),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(analysis_dir / "analysis_summary.json", summary)
        return summary

    def route5_run_capture_analysis(
        self,
        run_dir: Path,
        *,
        weights_path: Optional[Path] = None,
        stop_event: Optional[threading.Event] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir).resolve()
        manifest = self.route5_build_capture_analysis_manifest(run_dir)
        frames = self.route5_capture_manifest_frames(manifest)
        if not frames:
            summary = self.route5_write_capture_analysis_reports(run_dir, manifest, {"status": "no_valid_scan_captures", "frames": []})
            self.route5_update_state(last_capture_analysis=summary)
            self.route5_write_state_artifact()
            return {**summary, "manifest": manifest}
        raw_weights = ""
        weights_var = getattr(self, "lidar_yolo_weights_var", None)
        if weights_var is not None and hasattr(weights_var, "get"):
            try:
                raw_weights = str(weights_var.get() or "")
            except Exception:
                raw_weights = ""
        selected_weights = weights_path or Path(raw_weights or str(lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_WEIGHTS_PATH))
        yolo_result = lidar_yolo_analysis.run_lidar_yolo_analysis(
            stream_dir=run_dir,
            weights_path=selected_weights,
            device=lidar_yolo_analysis.default_lidar_yolo_device(),
            selected_frames_override=frames,
            stop_event=stop_event,
            progress_callback=progress_callback,
        )
        summary = self.route5_write_capture_analysis_reports(run_dir, manifest, yolo_result)
        self.route5_update_state(last_capture_analysis=summary)
        self.route5_write_state_artifact()
        return {**summary, "manifest": manifest, "yolo_result": self.route5_json_safe(yolo_result)}

    def open_route5_capture_analysis_window(self) -> None:
        self.ensure_route5_state()
        output_dir = self.route5_state_output_dir()
        initial_dir = output_dir or self.resolve_project_path("llm_route5_fusion_runs")
        if not self.llm_route5_capture_analysis_run_dir_var.get().strip() and output_dir is not None:
            self.llm_route5_capture_analysis_run_dir_var.set(str(output_dir))
        dialog = tk.Toplevel(self.llm_route5_window if getattr(self, "llm_route5_window", None) is not None else self.root)
        dialog.title("Analyze V5 Captures")
        dialog.geometry("920x420")
        dialog.grid_columnconfigure(1, weight=1)
        tk.Label(dialog, text="V5 Run").grid(row=0, column=0, sticky="w", padx=8, pady=(10, 4))
        tk.Entry(dialog, textvariable=self.llm_route5_capture_analysis_run_dir_var).grid(row=0, column=1, sticky="ew", padx=4, pady=(10, 4))

        def browse() -> None:
            selected = filedialog.askdirectory(title="Select V5 Route6_entrance_search run folder", initialdir=str(initial_dir))
            if selected:
                self.llm_route5_capture_analysis_run_dir_var.set(str(Path(selected)))

        def use_current() -> None:
            current = self.route5_state_output_dir()
            if current is not None:
                self.llm_route5_capture_analysis_run_dir_var.set(str(current))

        tk.Button(dialog, text="Browse", command=browse).grid(row=0, column=2, padx=4, pady=(10, 4))
        tk.Button(dialog, text="Current", command=use_current).grid(row=0, column=3, padx=4, pady=(10, 4))
        status = tk.Label(dialog, textvariable=self.llm_route5_capture_analysis_status_var, anchor="w", justify="left", wraplength=880)
        status.grid(row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
        preview = tk.Text(dialog, height=14, wrap="word", font=("Consolas", 9))
        preview.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=8, pady=6)
        dialog.grid_rowconfigure(2, weight=1)

        def set_preview(payload: Dict[str, Any]) -> None:
            try:
                preview.configure(state="normal")
                preview.delete("1.0", "end")
                preview.insert("1.0", json.dumps(self.route5_json_safe(payload), indent=2, ensure_ascii=False))
                preview.configure(state="disabled")
            except tk.TclError:
                pass

        def start() -> None:
            if self.llm_route5_capture_analysis_thread is not None and self.llm_route5_capture_analysis_thread.is_alive():
                self.llm_route5_capture_analysis_status_var.set("V5 Capture Analysis: already running")
                return
            run_dir = Path(self.llm_route5_capture_analysis_run_dir_var.get().strip()).resolve()
            if not run_dir.exists():
                self.llm_route5_capture_analysis_status_var.set(f"V5 Capture Analysis: folder missing: {run_dir}")
                return
            self.llm_route5_capture_analysis_stop_event.clear()
            self.llm_route5_capture_analysis_status_var.set(f"V5 Capture Analysis: running -> {run_dir / 'route5_capture_analysis'}")

            def progress(payload: Dict[str, Any]) -> None:
                self.root.after(0, lambda p=payload: self.llm_route5_capture_analysis_status_var.set(str(p.get("message", "V5 Capture Analysis running"))))

            def worker() -> None:
                try:
                    result = self.route5_run_capture_analysis(run_dir, stop_event=self.llm_route5_capture_analysis_stop_event, progress_callback=progress)
                except Exception as exc:
                    self.root.after(0, lambda e=exc: self.llm_route5_capture_analysis_status_var.set(f"V5 Capture Analysis failed: {e}"))
                    return
                self.root.after(
                    0,
                    lambda r=result: (
                        self.llm_route5_capture_analysis_status_var.set(
                            f"V5 Capture Analysis: {r.get('status')} included={r.get('included_count')} excluded={r.get('excluded_count')} -> {r.get('analysis_dir')}"
                        ),
                        set_preview(r),
                    ),
                )

            self.llm_route5_capture_analysis_thread = threading.Thread(target=worker, daemon=True)
            self.llm_route5_capture_analysis_thread.start()

        buttons = tk.Frame(dialog)
        buttons.grid(row=3, column=0, columnspan=4, sticky="e", padx=8, pady=(0, 8))
        tk.Button(buttons, text="Run V5 Capture Analysis", command=start).pack(side="left", padx=4)
        tk.Button(buttons, text="Stop", command=lambda: self.llm_route5_capture_analysis_stop_event.set()).pack(side="left", padx=4)
        tk.Button(buttons, text="Close", command=dialog.destroy).pack(side="left", padx=4)

    def route5_map_canvas_size_for_width(self, available_width_px: float, *, image_size: Optional[Tuple[int, int]] = None) -> Dict[str, int]:
        width = max(760, min(1800, int(float(available_width_px or 0.0)) - 24))
        if image_size and len(image_size) >= 2 and float(image_size[0] or 0.0) > 0:
            aspect = float(image_size[1]) / max(1.0, float(image_size[0]))
            height = int(round(width * aspect))
        else:
            height = 360
        return {"width": int(width), "height": int(max(300, min(620, height)))}

    def route5_map_status_style(self, status: str) -> Dict[str, str]:
        status = str(status or "").strip().lower()
        colors = {
            "active": "#22d3ee",
            "captured": "#22c55e",
            "done": "#22c55e",
            "visited": "#22c55e",
            "pending": "#facc15",
            "planned": "#facc15",
            "blocked": "#ef4444",
            "failed": "#ef4444",
        }
        return {"color": colors.get(status, "#ffd166"), "outline_color": "#111827"}

    def route5_bool_var(self, name: str) -> bool:
        var = getattr(self, name, None)
        try:
            return bool(var.get())
        except Exception:
            return False

    def route5_completed_scan_ids(self, output_dir: Optional[Path] = None) -> set[str]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        completed: set[str] = set()
        raw = state.get("captured_scan_ids", [])
        if isinstance(raw, dict):
            completed.update(str(key) for key, value in raw.items() if value)
        elif isinstance(raw, (list, tuple, set)):
            completed.update(str(item) for item in raw if str(item).strip())
        if output_dir is None:
            output_dir = self.route5_state_output_dir()
        if output_dir is not None:
            for name in ("scan_execution_log.jsonl", "lidar_capture_log.jsonl"):
                try:
                    rows = self.read_jsonl_artifact(output_dir / name)
                except Exception:
                    rows = []
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    status = str(row.get("status", row.get("capture_status", "")) or "").strip().lower()
                    if status and status != "ok":
                        continue
                    if "capture_guard_passed" in row and not bool(row.get("capture_guard_passed", False)):
                        continue
                    scan_id = str(row.get("scan_id", row.get("point_id", row.get("capture_id", ""))) or "")
                    if scan_id:
                        completed.add(scan_id)
        return completed

    def route5_observation_overlay_points(self, target_house_id: str, active: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        if hasattr(self, "route2_all_facade_observation_candidates"):
            try:
                candidates = list(self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False))
            except Exception:
                candidates = []
        active_target = str(active.get("target_id", "") or "")
        completed = {str(item).strip().lower() for item in getattr(self, "llm_route5_completed_facades", set()) or set()}
        blocked = {str(item).strip().lower() for item in getattr(self, "llm_route5_blocked_facades", set()) or set()}
        points: List[Dict[str, Any]] = []
        for idx, item in enumerate(candidates, start=1):
            if not isinstance(item, dict):
                continue
            try:
                float(item.get("x"))
                float(item.get("y"))
            except Exception:
                continue
            facade = str(item.get("facade", "") or "").strip().lower()
            label = str(item.get("label", "") or f"{target_house_id}_{facade}_obs_{idx}")
            raw_status = str(item.get("status", "") or "").strip().lower()
            if active_target and label == active_target:
                status = "active"
            elif facade in completed or raw_status in {"captured", "done", "visited"}:
                status = "captured"
            elif facade in blocked or raw_status in {"blocked", "failed", "unsafe"}:
                status = "blocked"
            else:
                status = "pending"
            points.append(
                {
                    **self.route5_json_safe(item),
                    **self.route5_map_status_style(status),
                    "label": label,
                    "status": status,
                    "route_point_type": "observation_point",
                    "target_id": label,
                    "facade": facade,
                    "overlay": "all_observation_points",
                }
            )
        return points

    def route7_observation_overlay_points(self, target_house_id: str, active: Dict[str, Any]) -> List[Dict[str, Any]]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        facade = str(active.get("facade", state.get("current_facade", "")) or "").strip().lower()
        if facade not in {"south", "east", "north", "west"}:
            return []
        output_dir = self.route7_current_map_output_dir()
        attempts: List[Dict[str, Any]] = []
        for candidate in state.get("ranked_facade_candidates", []) if isinstance(state.get("ranked_facade_candidates", []), list) else []:
            if not isinstance(candidate, dict) or str(candidate.get("facade", "") or "").strip().lower() != facade:
                continue
            raw_attempts = candidate.get("route7_edge_observation_attempts", []) if isinstance(candidate.get("route7_edge_observation_attempts", []), list) else []
            if not raw_attempts:
                raw_attempts = candidate.get("observation_attempts", []) if isinstance(candidate.get("observation_attempts", []), list) else []
            attempts.extend([dict(item) for item in raw_attempts if isinstance(item, dict)])
            break
        if not attempts:
            attempts = self.route7_edge_observation_attempts_for_facade(target_house_id, facade, output_dir=output_dir)
        active_target = str(active.get("target_id", "") or "")
        points: List[Dict[str, Any]] = []
        for idx, item in enumerate(attempts, start=1):
            try:
                float(item.get("x"))
                float(item.get("y"))
            except Exception:
                continue
            target_id = str(item.get("target_id", item.get("label", f"{target_house_id}_{facade}_edge_obs_{idx:03d}")) or "")
            status = "active" if active_target and target_id == active_target else str(item.get("status", "pending") or "pending")
            points.append(
                {
                    **self.route5_json_safe(item),
                    **self.route5_map_status_style(status),
                    "label": target_id or f"{target_house_id}_{facade}_edge_obs_{idx:03d}",
                    "target_id": target_id or f"{target_house_id}_{facade}_edge_obs_{idx:03d}",
                    "facade": facade,
                    "status": status,
                    "route_point_type": "route7_edge_observation_point",
                    "overlay": "current_facade_edge_observation_points",
                }
            )
        return points

    def route5_facade_capture_overlay_points(self, active: Dict[str, Any]) -> List[Dict[str, Any]]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        facade = str(active.get("facade", state.get("current_facade", "")) or "").strip().lower()
        if not facade:
            return []
        active_target = str(active.get("target_id", active.get("scan_id", "")) or "")
        raw_points: List[Dict[str, Any]] = []
        primary_sources = (state.get("facade_scan_points", []), state.get("current_facade_scan_points", []))
        for source in primary_sources:
            if isinstance(source, list):
                raw_points.extend([dict(item) for item in source if isinstance(item, dict)])
        if not raw_points:
            source = (getattr(self, "llm_route2_state", {}) if isinstance(getattr(self, "llm_route2_state", {}), dict) else {}).get("facade_scan_points", [])
            if isinstance(source, list):
                raw_points.extend([dict(item) for item in source if isinstance(item, dict)])
        completed_scan_ids = self.route5_completed_scan_ids()
        points: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for idx, point in enumerate(raw_points, start=1):
            if str(point.get("facade", "") or "").strip().lower() != facade:
                continue
            try:
                float(point.get("x"))
                float(point.get("y"))
            except Exception:
                continue
            scan_id = str(point.get("scan_id", point.get("point_id", "")) or f"{facade}_scan_{idx}")
            if scan_id in seen:
                continue
            seen.add(scan_id)
            raw_status = str(point.get("status", point.get("capture_status", "")) or "").strip().lower()
            if active_target and scan_id == active_target:
                status = "active"
            elif scan_id in completed_scan_ids or raw_status in {"captured", "done", "visited"}:
                status = "captured"
            elif raw_status in {"blocked", "failed", "unsafe"}:
                status = "blocked"
            else:
                status = "pending"
            points.append(
                {
                    **self.route5_json_safe(point),
                    **self.route5_map_status_style(status),
                    "label": scan_id,
                    "status": status,
                    "route_point_type": "scan_point",
                    "scan_id": scan_id,
                    "target_id": scan_id,
                    "facade": facade,
                    "overlay": "current_facade_captures",
                }
            )
        return points

    def route5_active_map_route_points(self) -> List[Dict[str, Any]]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        active = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status"), dict) else {}
        stage = str(active.get("stage", state.get("stage", "")) or "")
        facade = str(active.get("facade", state.get("current_facade", "")) or "")
        target_id = str(active.get("target_id", active.get("scan_id", "")) or "")
        target_pose = active.get("target_pose", state.get("target", {}))
        route_points: List[Dict[str, Any]] = []
        if isinstance(target_pose, dict) and target_pose:
            route_points.append(
                {
                    **self.route5_json_safe(target_pose),
                    "label": target_id or f"{facade}_target",
                    "route_point_type": "current_target",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    **self.route5_map_status_style("active"),
                }
            )
        nav = state.get("current_navigation_plan", {}) if isinstance(state.get("current_navigation_plan"), dict) else {}
        nav_target = str(nav.get("target_id", "") or "")
        if nav and (not target_id or not nav_target or nav_target == target_id):
            plan = nav.get("plan", {}) if isinstance(nav.get("plan"), dict) else {}
            waypoints = plan.get("waypoints", []) if isinstance(plan.get("waypoints"), list) else []
            for idx, waypoint in enumerate(waypoints, start=1):
                if not isinstance(waypoint, dict):
                    continue
                route_points.append(
                    {
                        **self.route5_json_safe(waypoint),
                        "label": f"{target_id or nav_target or facade}_wp_{idx}",
                        "route_point_type": "navigation_waypoint",
                        "stage": str(nav.get("stage", stage) or stage),
                        "facade": str(nav.get("facade", facade) or facade),
                        "target_id": nav_target or target_id,
                        **self.route5_map_status_style("pending"),
                    }
                )
        reset = state.get("last_target_reset", {}) if isinstance(state.get("last_target_reset"), dict) else {}
        reset_target = str(reset.get("target_id", "") or "")
        reset_pose = reset.get("reset_target_pose", {}) if isinstance(reset.get("reset_target_pose"), dict) else {}
        original_pose = reset.get("original_target_pose", {}) if isinstance(reset.get("original_target_pose"), dict) else {}
        reset_id = str(reset.get("reset_target_id", "") or "")
        if reset_pose and (not target_id or reset_target == target_id or (target_id and reset_id.startswith(target_id))):
            if original_pose:
                route_points.append(
                    {
                        **self.route5_json_safe(original_pose),
                        "label": reset_target or f"{target_id}_original",
                        "route_point_type": "original_navigation_target",
                        "stage": str(reset.get("stage", stage) or stage),
                        "facade": str(reset.get("facade", facade) or facade),
                        "target_id": reset_target or target_id,
                        "reset_target_id": reset_id,
                        **self.route5_map_status_style("pending"),
                    }
                )
            route_points.append(
                {
                    **self.route5_json_safe(reset_pose),
                    "label": reset_id or f"{target_id}_reset",
                    "route_point_type": "target_reset",
                    "stage": str(reset.get("stage", stage) or stage),
                    "facade": str(reset.get("facade", facade) or facade),
                    "target_id": reset_target or target_id,
                    "reset_direction": str(reset.get("reset_direction", "") or ""),
                    **self.route5_map_status_style("active"),
                }
            )
        target_house_id = str(state.get("target_house_id", "") or self.selected_route_target_house_id() or "")
        if self.route5_bool_var("llm_route5_show_all_obs_points_var"):
            route_points.extend(self.route5_observation_overlay_points(target_house_id, active))
        if self.route5_bool_var("llm_route5_show_facade_captures_var"):
            route_points.extend(self.route5_facade_capture_overlay_points(active))
        return route_points

    def refresh_llm_route5_map(self) -> None:
        self.ensure_route5_state()
        widget = getattr(self, "llm_route5_map_widget", None)
        if widget is None:
            return
        try:
            if not self.load_map_resources(force=not bool(self.map_config)):
                self.llm_route5_map_status_var.set("Route V5 Map: map unavailable")
                return
            pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
            pose_x = float(pose.get("x", 0.0)) if pose else 0.0
            pose_y = float(pose.get("y", 0.0)) if pose else 0.0
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0))) if pose else 0.0
            houses, boxes = self.build_map_display(pose)
            map_frame = getattr(self, "llm_route5_map_frame", None)
            if map_frame is not None and hasattr(widget, "resize_canvas"):
                try:
                    available_w = float(map_frame.winfo_width() or widget.canvas.winfo_width() or 1120)
                    size = self.route5_map_canvas_size_for_width(available_w, image_size=self.map_image_size())
                    if abs(int(getattr(widget, "_canvas_w", 0)) - size["width"]) > 12 or abs(int(getattr(widget, "_canvas_h", 0)) - size["height"]) > 12:
                        widget.resize_canvas(size["width"], size["height"])
                except Exception:
                    pass
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
            route5_state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
            active = route5_state.get("current_exploration_status", {}) if isinstance(route5_state.get("current_exploration_status"), dict) else {}
            active_stage = str(active.get("stage", route5_state.get("stage", "")) or "")
            active_facade = str(active.get("facade", route5_state.get("current_facade", "")) or "")
            active_target = str(active.get("target_id", active.get("scan_id", "")) or "")
            route_points = self.route5_active_map_route_points()
            widget.set_route_plan({"route_points": route_points})
            output_dir = self.route5_state_output_dir()
            if output_dir is not None:
                trace_rows = self.read_jsonl_artifact(output_dir / "route5_movement_trace.jsonl")
                trajectory = [
                    row.get("current_pose", {})
                    for row in trace_rows[-400:]
                    if isinstance(row.get("current_pose"), dict)
                    and (not active_target or str(row.get("target_id", "") or "") == active_target)
                    and (not active_stage or str(row.get("stage", "") or "") == active_stage)
                ][-200:]
                widget.set_trajectory(trajectory)
            self.llm_route5_map_status_var.set(
                f"Route V5 Map: active={active_facade or '-'} target={active_target or '-'} visible_points={len(route_points)} completed={len(self.llm_route5_completed_facades)}/4"
            )
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM Route6_entrance_search v5 map failed: %s", exc)
            self.llm_route5_map_status_var.set(f"Route V5 Map: failed: {exc}")

    def schedule_route5_auto_refresh(self) -> None:
        self.ensure_route5_state()
        try:
            enabled = bool(self.llm_route5_auto_refresh_var.get())
        except tk.TclError:
            enabled = False
        if not enabled:
            self.cancel_route5_auto_refresh()
            return
        self.cancel_route5_auto_refresh()

        def tick() -> None:
            self.llm_route5_auto_refresh_job = None
            try:
                if not bool(self.llm_route5_auto_refresh_var.get()):
                    return
            except tk.TclError:
                return
            self.refresh_route5_support_views()
            try:
                self.llm_route5_auto_refresh_job = self.root.after(int(LLM_ROUTE3_AUTO_REFRESH_MS), tick)
            except tk.TclError:
                self.llm_route5_auto_refresh_job = None

        try:
            self.llm_route5_auto_refresh_job = self.root.after(int(LLM_ROUTE3_AUTO_REFRESH_MS), tick)
        except tk.TclError:
            self.llm_route5_auto_refresh_job = None

    def cancel_route5_auto_refresh(self) -> None:
        job = getattr(self, "llm_route5_auto_refresh_job", None)
        if job:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
            except Exception:
                pass
        self.llm_route5_auto_refresh_job = None

    def on_route5_auto_refresh_toggle(self) -> None:
        self.ensure_route5_state()
        try:
            enabled = bool(self.llm_route5_auto_refresh_var.get())
        except tk.TclError:
            enabled = False
        if enabled:
            self.refresh_route5_support_views()
            self.schedule_route5_auto_refresh()
        else:
            self.cancel_route5_auto_refresh()

    def on_route5_toggle_all_obs_points(self) -> None:
        self.ensure_route5_state()
        try:
            self.llm_route5_show_all_obs_points_var.set(not bool(self.llm_route5_show_all_obs_points_var.get()))
        except Exception:
            return
        self.refresh_llm_route5_map()

    def on_route5_toggle_facade_captures(self) -> None:
        self.ensure_route5_state()
        try:
            self.llm_route5_show_facade_captures_var.set(not bool(self.llm_route5_show_facade_captures_var.get()))
        except Exception:
            return
        self.refresh_llm_route5_map()

    def on_route5_map_frame_configure(self, event: Any) -> None:
        widget = getattr(self, "llm_route5_map_widget", None)
        if widget is None or not hasattr(widget, "resize_canvas"):
            return
        try:
            image_size = self.map_image_size()
        except Exception:
            image_size = None
        try:
            size = self.route5_map_canvas_size_for_width(float(getattr(event, "width", 0) or 0), image_size=image_size)
            if abs(int(getattr(widget, "_canvas_w", 0)) - size["width"]) > 12 or abs(int(getattr(widget, "_canvas_h", 0)) - size["height"]) > 12:
                widget.resize_canvas(size["width"], size["height"])
        except Exception:
            pass

    def refresh_route5_support_views(self) -> None:
        self.ensure_route5_state()
        completed = len(getattr(self, "llm_route5_completed_facades", set()) or set())
        blocked = len(getattr(self, "llm_route5_blocked_facades", set()) or set())
        progress = 100.0 * float(min(4, completed + blocked)) / 4.0
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        current_status = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status"), dict) else {}
        try:
            self.llm_route5_progress_var.set(max(0.0, min(100.0, progress)))
            self.llm_route5_progress_text_var.set(f"Fusion: completed={completed} blocked={blocked}")
            if current_status:
                facade = str(current_status.get("facade", state.get("current_facade", "")) or state.get("current_facade", "") or "-")
                stage = str(current_status.get("stage", state.get("stage", "")) or state.get("stage", "") or "-")
                point_index = current_status.get("point_index")
                point_total = current_status.get("point_total")
                suffix = f" {point_index}/{point_total}" if point_index is not None and point_total is not None else ""
                self.llm_route5_current_status_var.set(f"Current: {stage} {facade}{suffix}")
            else:
                self.llm_route5_current_status_var.set(f"Current: {state.get('stage', 'idle')} {state.get('current_facade', '')}")
            next_status = state.get("next_exploration_status", {}) if isinstance(state.get("next_exploration_status"), dict) else {}
            next_facade = str(next_status.get("facade", next_status.get("target_facade", "")) or "")
            self.llm_route5_next_status_var.set(f"Next: {next_facade}" if next_facade else "Next: n/a")
        except tk.TclError:
            pass
        self.refresh_route5_preview()
        self.refresh_route5_analysis_view()
        self.refresh_route5_rgb_display()
        self.refresh_llm_route5_map()

    def _on_llm_route5_window_content_configure(self, _event: tk.Event) -> None:
        canvas = getattr(self, "llm_route5_window_canvas", None)
        if canvas is None:
            return
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_llm_route5_window_canvas_configure(self, event: tk.Event) -> None:
        canvas = getattr(self, "llm_route5_window_canvas", None)
        content = getattr(self, "llm_route5_window_content", None)
        content_window = getattr(self, "llm_route5_window_content_window", None)
        if canvas is None or content is None or content_window is None:
            return
        requested_width = max(content.winfo_reqwidth(), int(event.width))
        canvas.itemconfigure(content_window, width=requested_width)
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_llm_route5_window_mousewheel(self, event: tk.Event):
        canvas = getattr(self, "llm_route5_window_canvas", None)
        if canvas is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            canvas.xview_scroll(delta, "units")
        else:
            canvas.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route5_window_mousewheel_linux(self, event: tk.Event):
        canvas = getattr(self, "llm_route5_window_canvas", None)
        if canvas is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        canvas.yview_scroll(direction, "units")
        return "break"

    def _on_llm_route5_text_mousewheel(self, event: tk.Event):
        widget = getattr(event, "widget", None)
        if widget is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            widget.xview_scroll(delta, "units")
        else:
            widget.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route5_text_mousewheel_linux(self, event: tk.Event):
        widget = getattr(event, "widget", None)
        if widget is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        widget.yview_scroll(direction, "units")
        return "break"

    def _bind_llm_route5_text_mousewheel(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_llm_route5_text_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route5_text_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route5_text_mousewheel_linux, add="+")
        except tk.TclError:
            pass

    def _bind_llm_route5_window_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            if isinstance(widget, tk.Text):
                self._bind_llm_route5_text_mousewheel(widget)
                return
            widget.bind("<MouseWheel>", self._on_llm_route5_window_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route5_window_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route5_window_mousewheel_linux, add="+")
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            self._bind_llm_route5_window_mousewheel_tree(child)

    def route5_or2_corridor_text(self, corridor_risks: Dict[str, Any]) -> str:
        if not isinstance(corridor_risks, dict) or not corridor_risks:
            return "Corridors: --"
        parts = []
        for name in ("front_center", "left_corridor", "right_corridor", "up_corridor"):
            stats = corridor_risks.get(name) if isinstance(corridor_risks.get(name), dict) else {}
            parts.append(
                f"{name}:D{float(stats.get('stop_fraction', 0.0) or 0.0):.2f}/"
                f"R{float(stats.get('warning_fraction', 0.0) or 0.0):.2f}/"
                f"Y{float(stats.get('clearance_fraction', 0.0) or 0.0):.2f}"
            )
        return "Corridors: " + " | ".join(parts)

    def route5_or2_array_to_photo(self, image: np.ndarray, *, max_width: int = 360, max_height: int = 240) -> ImageTk.PhotoImage:
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        scale = min(max_width / max(1, pil.width), max_height / max(1, pil.height), 1.0)
        if scale < 1.0:
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def route5_update_or2_state_color(self, risk: str) -> None:
        label = getattr(self, "route5_or2_state_label", None)
        if label is None:
            return
        colors = {
            "clear": ("#3cbe5a", "black"),
            "clearance_warning": ("#f5c82d", "black"),
            "obstacle_warning": ("#f5825f", "black"),
            "must_stop": ("#be1423", "white"),
            "starting": ("#d9d9d9", "black"),
            "predict_error": ("#7a1f1f", "white"),
            "failed": ("#7a1f1f", "white"),
        }
        bg, fg = colors.get(str(risk), ("#d9d9d9", "black"))
        try:
            label.configure(bg=bg, fg=fg)
        except tk.TclError:
            pass

    def route5_or2_monitor_result_from_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event = event if isinstance(event, dict) else {}
        route7_or3_1 = self.route5_event_is_route7(event) and (
            isinstance(event.get("or3_1_prediction"), dict) or isinstance(event.get("or3_prediction"), dict)
        )
        if route7_or3_1 and isinstance(event.get("or3_1_prediction"), dict):
            prediction = event.get("or3_1_prediction", {})
        elif route7_or3_1 and isinstance(event.get("or3_prediction"), dict):
            prediction = event.get("or3_prediction", {})
        else:
            prediction = event.get("or2_prediction", {}) if isinstance(event.get("or2_prediction"), dict) else {}
        rule = event.get("or2_rule", {}) if isinstance(event.get("or2_rule"), dict) else {}
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        if not summary and isinstance(event.get("depth_obstacle_summary"), dict):
            summary = event.get("depth_obstacle_summary", {})
        rgb_image = None
        mask_image = None
        rgb_path = str(event.get("rgb_path", "") or "")
        overlay_path = str(prediction.get("risk_overlay_path", event.get("or2_risk_overlay_path", "")) or "")
        representation_label = "OR3_1" if route7_or3_1 else "OR2"
        try:
            if rgb_path and Path(rgb_path).is_file():
                rgb_image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            rgb_image = None
        try:
            if overlay_path and Path(overlay_path).is_file():
                mask_image = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            mask_image = None
        front_depth = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_index": int(float(event.get("frame_id", 0) or 0)),
            "frame_dir": str(event.get("capture_dir", "")),
            "rgb_path": rgb_path,
            "depth_path": str(event.get("depth_npy_path", "")),
            "risk_overlay_path": overlay_path,
            "prediction_json_path": str(prediction.get("prediction_json_path", event.get("or2_prediction_json_path", "")) or ""),
            "front_risk_state": str(prediction.get("front_risk_state", event.get("or2_front_risk_state", prediction.get("status", "predict_error"))) or "predict_error"),
            "can_forward": bool(prediction.get("can_forward", event.get("or2_can_forward", False))),
            "must_stop": bool(prediction.get("must_stop", event.get("or2_must_stop", False))),
            "front_min_depth_cm": front_depth,
            "or2_corridor_risks": rule.get("corridor_risks", event.get("or2_corridor_risks", {})),
            "or2_selected_direction": rule.get("selected_direction", event.get("or2_selected_direction", "")),
            "or2_candidate_action_scores": rule.get("candidate_action_scores", event.get("or2_candidate_action_scores", {})),
            "pointcloud_summary": summary,
            "representation_label": representation_label,
            "route7_primary_representation": "or3_1" if route7_or3_1 else "",
        }
        return {
            "event_record": record,
            "prediction": prediction,
            "rule": rule,
            "summary": summary,
            "rgb_image": rgb_image,
            "mask_image": mask_image,
            "frame_count": int(record["frame_index"]),
        }

    def apply_route5_or2_monitor_result(self, result: Dict[str, Any]) -> None:
        prediction = result.get("prediction") if isinstance(result.get("prediction"), dict) else {}
        rule = result.get("rule") if isinstance(result.get("rule"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        record = result.get("event_record") if isinstance(result.get("event_record"), dict) else {}
        risk = str(prediction.get("front_risk_state", record.get("front_risk_state", "predict_error")) or "predict_error")
        selected = str(rule.get("selected_direction", record.get("or2_selected_direction", "--")) or "--")
        front = float(summary.get("front_min_depth_cm", record.get("front_min_depth_cm", 0.0)) or 0.0)
        frame_count = int(result.get("frame_count", 0) or 0)
        representation_label = str(record.get("representation_label", "OR2") or "OR2")
        self.route5_or2_state_var.set(f"State: {risk}")
        self.route5_or2_frame_count_var.set(f"Frames: {frame_count}")
        self.route5_or2_front_depth_var.set(f"Front depth: {front:.1f} cm")
        self.route5_or2_can_forward_var.set(f"Can forward: {'yes' if prediction.get('can_forward', False) else 'no'}")
        self.route5_or2_selected_direction_var.set(f"Selected direction: {selected}")
        self.route5_or2_corridor_var.set(self.route5_or2_corridor_text(rule.get("corridor_risks", {})))
        self.route5_or2_capture_dir_var.set(f"{representation_label} capture: {record.get('frame_dir', '--')}")
        self.llm_route5_representation_status_var.set(f"{representation_label}: state={risk} selected={selected} front={front:.1f}cm")
        self.route5_update_or2_state_color(risk)
        if self.route5_or2_rgb_label is not None and isinstance(result.get("rgb_image"), np.ndarray):
            self.route5_or2_rgb_photo = self.route5_or2_array_to_photo(result["rgb_image"])
            self.route5_or2_rgb_label.configure(image=self.route5_or2_rgb_photo, text="")
        if self.route5_or2_mask_label is not None and isinstance(result.get("mask_image"), np.ndarray):
            self.route5_or2_mask_photo = self.route5_or2_array_to_photo(result["mask_image"])
            self.route5_or2_mask_label.configure(image=self.route5_or2_mask_photo, text="")
        if self.route5_or2_report_text is not None:
            payload = {
                "event": record,
                "prediction": serializable_or2_prediction(prediction),
                "rule": rule,
            }
            self.route5_or2_report_text.delete("1.0", "end")
            self.route5_or2_report_text.insert("end", json.dumps(self.route5_json_safe(payload), indent=2, ensure_ascii=False))

    def stop_route5_or2_monitor(self) -> None:
        self.ensure_route5_state()
        self.route5_or2_monitor_stop_event.set()

    def build_route5_or2_monitor_panel(self, parent: tk.Misc, *, representation_label: str = "OR2") -> tk.LabelFrame:
        monitor = tk.LabelFrame(parent, text=f"{representation_label} Live Risk Monitor")
        monitor.grid_columnconfigure(0, weight=1)
        controls = tk.Frame(monitor)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 3))
        self.route5_or2_state_label = tk.Label(controls, textvariable=self.route5_or2_state_var, width=28, anchor="center", bg="#d9d9d9", fg="black")
        self.route5_or2_state_label.pack(side="left", padx=(4, 8), pady=3)
        for var in (
            self.route5_or2_frame_count_var,
            self.route5_or2_front_depth_var,
            self.route5_or2_can_forward_var,
            self.route5_or2_selected_direction_var,
        ):
            tk.Label(controls, textvariable=var, anchor="w").pack(side="left", padx=7, pady=3)

        legend = tk.Frame(monitor)
        legend.grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        for label, color, text in (
            ("clear", "#3cbe5a", "safe"),
            ("clearance_warning", "#f5c82d", "250-450cm"),
            ("obstacle_warning", "#f5825f", "100-250cm"),
            ("must_stop", "#be1423", "<=100cm"),
        ):
            tk.Label(legend, text=f"{label}: {text}", bg=color, fg="white" if label == "must_stop" else "black").pack(side="left", padx=4, pady=3, ipadx=8, ipady=2)
        tk.Label(legend, textvariable=self.route5_or2_corridor_var, anchor="w").pack(side="left", fill="x", expand=True, padx=(18, 4), pady=3)

        image_row = tk.Frame(monitor)
        image_row.grid(row=2, column=0, sticky="w", padx=6, pady=4)
        rgb_frame = tk.LabelFrame(image_row, text="RGB", width=380, height=260)
        rgb_frame.pack(side="left", padx=(0, 8), pady=0)
        rgb_frame.pack_propagate(False)
        overlay_title = "A+3.1 Risk Overlay" if str(representation_label).upper() == "OR3_1" else "A+2 Risk Overlay"
        mask_frame = tk.LabelFrame(image_row, text=overlay_title, width=380, height=260)
        mask_frame.pack(side="left", padx=(8, 0), pady=0)
        mask_frame.pack_propagate(False)
        self.route5_or2_rgb_label = tk.Label(rgb_frame, text="No RGB yet")
        self.route5_or2_rgb_label.pack(fill="both", expand=True, padx=6, pady=6)
        self.route5_or2_mask_label = tk.Label(mask_frame, text="No overlay yet")
        self.route5_or2_mask_label.pack(fill="both", expand=True, padx=6, pady=6)

        bottom = tk.Frame(monitor)
        bottom.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        bottom.grid_columnconfigure(1, weight=1)
        tk.Label(bottom, textvariable=self.route5_or2_capture_dir_var, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.route5_or2_report_text = tk.Text(bottom, height=4, wrap="none", font=("Consolas", 8))
        self.route5_or2_report_text.grid(row=0, column=1, sticky="ew")
        return monitor

    def route5_fixed_status_label(
        self,
        parent: tk.Misc,
        textvariable: Any,
        *,
        row: int,
        column: int,
        width_px: int,
        height_px: int,
        columnspan: int = 1,
        padx: Any = 0,
        pady: Any = 0,
        sticky: str = "ew",
    ) -> tk.Label:
        frame = tk.Frame(parent, width=max(40, int(width_px)), height=max(20, int(height_px)))
        frame.grid(row=row, column=column, columnspan=columnspan, sticky=sticky, padx=padx, pady=pady)
        frame.grid_propagate(False)
        label = tk.Label(
            frame,
            textvariable=textvariable,
            anchor="nw",
            justify="left",
            wraplength=max(20, int(width_px) - 8),
        )
        label.place(x=0, y=0, relwidth=1.0, relheight=1.0)
        return label

    def _build_llm_route5_section(
        self,
        parent: tk.Misc,
        *,
        route_label: str = "V5",
        start_command: Optional[Any] = None,
        step_command: Optional[Any] = None,
        stop_command: Optional[Any] = None,
    ) -> tk.LabelFrame:
        self.ensure_route5_state()
        label = str(route_label or "V5").strip().upper() or "V5"
        representation_label = "OR3_1" if label == "V7" else "OR2"
        representation_default_path = self.default_route7_or31_model_path() if label == "V7" else self.default_route5_representation_model_path()
        if label == "V7":
            try:
                current_model = str(self.llm_route5_representation_model_var.get() or "")
                if not current_model or current_model == str(self.default_route5_representation_model_path()):
                    self.llm_route5_representation_model_var.set(str(representation_default_path))
            except Exception:
                pass
        route = tk.LabelFrame(parent, text=f"LLM House Entrance Route {label} Fused Route + Avoidance")
        for col in (1, 3, 5):
            route.grid_columnconfigure(col, weight=1)
        route.grid_rowconfigure(7, minsize=132)
        route.grid_rowconfigure(8, minsize=34)
        route.grid_rowconfigure(9, minsize=156, weight=0)
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
        tk.Label(route, textvariable=self.llm_route5_active_var, anchor="w").grid(row=2, column=4, columnspan=2, sticky="ew", padx=6, pady=6)

        rep = tk.Frame(route)
        rep.grid(row=3, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(rep, text=f"{representation_label} Model").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(rep, textvariable=self.llm_route5_representation_model_var, width=42).pack(side="left", fill="x", expand=True, padx=(0, 6), pady=4)
        tk.Button(rep, text="Default", command=lambda path=representation_default_path: self.llm_route5_representation_model_var.set(str(path))).pack(side="left", padx=6, pady=4)
        tk.Label(rep, text="OA3 Plan").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(rep, textvariable=self.llm_route5_oa3_plan_var, width=34).pack(side="left", fill="x", expand=True, padx=(0, 6), pady=4)
        tk.Label(rep, text="Sense interval s").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(rep, textvariable=self.llm_route5_sensing_interval_s_var, width=7).pack(side="left", padx=(0, 8), pady=4)

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
            ("Move tick ms", self.llm_route5_move_tick_ms_var, 6),
            ("Nav step cm", self.llm_route5_nav_step_cm_var, 6),
            ("Reach tol cm", self.llm_route5_reach_tol_cm_var, 6),
            ("Z tol cm", self.llm_route5_z_tol_cm_var, 6),
            ("Yaw tol deg", self.llm_route5_yaw_tol_deg_var, 6),
            ("Max stage s", self.llm_route5_max_stage_s_var, 6),
        ):
            tk.Label(nav, text=label).pack(side="left", padx=(6, 2), pady=4)
            tk.Entry(nav, textvariable=var, width=width).pack(side="left", padx=(0, 6), pady=4)

        actions = tk.Frame(route)
        actions.grid(row=6, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 4))
        tk.Button(actions, text="Start Fused Route+Avoidance", command=start_command or self.on_route5_start_fused_search).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Step Facade", command=step_command or self.on_route5_step_facade).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Pause/Resume", command=self.on_route5_toggle_pause).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=stop_command or self.on_route5_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_route5_clear).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Validate Run", command=self.on_route5_validate_run).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text=f"Analyze {label} Captures", command=self.open_route5_capture_analysis_window).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Global Pause All", command=self.on_route5_global_pause_all).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Release Movement", command=self.on_route5_release_movement).pack(side="left", padx=6, pady=4)

        status = tk.Frame(route)
        status.grid(row=7, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        status.configure(height=132)
        status.grid_propagate(False)
        status.grid_columnconfigure(1, weight=1)
        status.grid_columnconfigure(0, minsize=250)
        status.grid_columnconfigure(2, minsize=360)
        for row_index, minsize in ((0, 24), (1, 20), (2, 20), (3, 20), (4, 38), (5, 20)):
            status.grid_rowconfigure(row_index, minsize=minsize)
        self.route5_fixed_status_label(status, self.llm_route5_stage_var, row=0, column=0, width_px=250, height_px=24, padx=(0, 12), sticky="w")
        self.route5_fixed_status_label(status, self.llm_route5_target_var, row=0, column=1, width_px=520, height_px=24, padx=(0, 12))
        self.route5_fixed_status_label(status, self.llm_route5_error_var, row=0, column=2, width_px=360, height_px=24, sticky="w")
        self.route5_fixed_status_label(status, self.llm_route5_payload_var, row=1, column=0, columnspan=3, width_px=1040, height_px=20, pady=(2, 0))
        self.route5_fixed_status_label(status, self.llm_route5_avoidance_status_var, row=2, column=0, columnspan=3, width_px=1040, height_px=20, pady=(2, 0))
        self.route5_fixed_status_label(status, self.llm_route5_representation_status_var, row=3, column=0, columnspan=3, width_px=1040, height_px=20, pady=(2, 0))
        self.route5_fixed_status_label(status, self.llm_route5_thinking_status_var, row=4, column=0, columnspan=3, width_px=1040, height_px=38, pady=(2, 0))
        self.route5_fixed_status_label(status, self.llm_route5_capture_analysis_status_var, row=5, column=0, columnspan=3, width_px=1040, height_px=20, pady=(2, 0))
        self.route5_fixed_status_label(route, self.llm_route5_status_var, row=8, column=0, columnspan=6, width_px=1080, height_px=28, padx=6, pady=(0, 4))
        preview_frame = tk.Frame(route, height=156)
        preview_frame.grid(row=9, column=0, columnspan=6, sticky="nsew", padx=6, pady=(0, 6))
        preview_frame.grid_propagate(False)
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
        self.llm_route5_preview_text = preview_text
        setattr(route, "_llm_route5_combo", combo)
        return route

    def open_llm_route_window5(self) -> None:
        self.ensure_route5_state()
        if self.llm_route5_window is not None and self.llm_route5_window.winfo_exists():
            self.llm_route5_window.lift()
            self.llm_route5_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route V5 Fused Route + Avoidance")
        window.geometry("1120x860")
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
        self.llm_route5_window_canvas = window_canvas
        self.llm_route5_window_content = content
        self.llm_route5_window_content_window = content_window
        content.bind("<Configure>", self._on_llm_route5_window_content_configure, add="+")
        window_canvas.bind("<Configure>", self._on_llm_route5_window_canvas_configure, add="+")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(3, weight=1)

        route = self._build_llm_route5_section(content)
        route.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        support = tk.Frame(content)
        support.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 4))
        support.grid_columnconfigure(0, weight=0)
        support.grid_columnconfigure(1, weight=1)
        rgb_frame = tk.LabelFrame(support, text="Facade RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        self.llm_route5_rgb_label = tk.Canvas(rgb_frame, width=330, height=220, bg="#202020", highlightthickness=0)
        self.llm_route5_rgb_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route5_rgb_label.bind("<Configure>", lambda _event: self.refresh_route5_rgb_display(), add="+")
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
        self.llm_route5_analysis_text = analysis_text

        monitor = self.build_route5_or2_monitor_panel(content)
        monitor.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 4))

        map_frame = tk.LabelFrame(content, text="Map Facade V5 Fusion")
        map_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 8))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(2, weight=1)
        map_status_row = tk.Frame(map_frame)
        map_status_row.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        map_status_row.grid_columnconfigure(0, weight=1)
        tk.Label(map_status_row, textvariable=self.llm_route5_map_status_var, anchor="w", wraplength=920, justify="left").grid(row=0, column=0, sticky="ew")
        tk.Label(map_status_row, textvariable=self.llm_route5_progress_text_var, anchor="e", wraplength=220, justify="left").grid(row=0, column=1, sticky="e", padx=(8, 2))
        ttk.Progressbar(map_status_row, variable=self.llm_route5_progress_var, maximum=100.0, length=150, mode="determinate").grid(row=0, column=2, sticky="e", padx=(0, 8))
        map_toolbar = tk.Frame(map_frame)
        map_toolbar.grid(row=1, column=0, sticky="ew", padx=6, pady=(4, 0))
        tk.Label(map_toolbar, textvariable=self.llm_route5_current_status_var, anchor="w", wraplength=360, justify="left").pack(side="left", padx=(0, 8))
        tk.Label(map_toolbar, textvariable=self.llm_route5_next_status_var, anchor="w", wraplength=300, justify="left").pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            map_toolbar,
            text="Auto Refresh",
            variable=self.llm_route5_auto_refresh_var,
            command=self.on_route5_auto_refresh_toggle,
        ).pack(side="left", padx=(0, 8))
        tk.Button(map_toolbar, text="Show/Hide All Obs Points", command=self.on_route5_toggle_all_obs_points).pack(side="left", padx=(0, 8))
        tk.Button(map_toolbar, text="Show/Hide Facade Captures", command=self.on_route5_toggle_facade_captures).pack(side="left", padx=(0, 8))
        tk.Button(map_toolbar, text="Refresh Map", command=self.refresh_llm_route5_map).pack(side="right", padx=6)
        self.load_map_resources(force=True)
        initial_map_size = self.route5_map_canvas_size_for_width(1120, image_size=self.map_image_size())
        self.llm_route5_map_widget = OverheadMapWidget(map_frame, world_bounds=self.map_world_bounds, canvas_w=initial_map_size["width"], canvas_h=initial_map_size["height"])
        self.llm_route5_map_widget.canvas.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route5_map_frame = map_frame
        map_frame.bind("<Configure>", self.on_route5_map_frame_configure, add="+")
        self.llm_route5_window = window
        self._bind_llm_route5_window_mousewheel_tree(content)
        window_canvas.bind("<MouseWheel>", self._on_llm_route5_window_mousewheel, add="+")
        window_canvas.bind("<Button-4>", self._on_llm_route5_window_mousewheel_linux, add="+")
        window_canvas.bind("<Button-5>", self._on_llm_route5_window_mousewheel_linux, add="+")

        def close_window() -> None:
            combo = getattr(route, "_llm_route5_combo", None)
            if combo is not None and combo in self.house_target_combos:
                self.house_target_combos.remove(combo)
            self.llm_route5_window = None
            self.llm_route5_window_canvas = None
            self.llm_route5_window_content = None
            self.llm_route5_window_content_window = None
            self.llm_route5_map_widget = None
            self.llm_route5_map_frame = None
            self.llm_route5_preview_text = None
            self.llm_route5_analysis_text = None
            self.llm_route5_rgb_label = None
            self.llm_route5_rgb_photo = None
            self.route5_or2_state_label = None
            self.route5_or2_rgb_label = None
            self.route5_or2_mask_label = None
            self.route5_or2_rgb_photo = None
            self.route5_or2_mask_photo = None
            self.route5_or2_report_text = None
            self.stop_route5_or2_monitor()
            try:
                self.llm_route5_capture_analysis_stop_event.set()
            except Exception:
                pass
            self.cancel_route5_auto_refresh()
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.refresh_route5_support_views()

    def on_route5_start_fused_search(self) -> None:
        self.ensure_route5_state()
        session = self.active_session()
        if session is None:
            return
        if self.llm_route5_thread is not None and self.llm_route5_thread.is_alive():
            self.llm_route5_status_var.set("LLM Route V5: already running.")
            return
        self.llm_route5_stop_event.clear()
        self.llm_route5_pause_event.clear()
        self.llm_route5_paused_var.set(False)
        self.llm_route5_thread = threading.Thread(target=lambda: self.route5_full_search_worker(session, single_facade=False, force_new=True), daemon=True)
        self.llm_route5_thread.start()

    def on_route5_step_facade(self) -> None:
        self.ensure_route5_state()
        session = self.active_session()
        if session is None:
            return
        if self.llm_route5_thread is not None and self.llm_route5_thread.is_alive():
            self.llm_route5_status_var.set("LLM Route V5: wait for current worker.")
            return
        self.llm_route5_stop_event.clear()
        self.llm_route5_pause_event.clear()
        self.llm_route5_thread = threading.Thread(target=lambda: self.route5_full_search_worker(session, single_facade=True, force_new=False), daemon=True)
        self.llm_route5_thread.start()

    def on_route5_toggle_pause(self) -> None:
        self.ensure_route5_state()
        if self.llm_route5_pause_event.is_set():
            self.llm_route5_pause_event.clear()
            self.llm_route5_paused_var.set(False)
            self.llm_route5_status_var.set("LLM Route V5: resumed.")
        else:
            self.llm_route5_pause_event.set()
            self.llm_route5_paused_var.set(True)
            session = self.active_session()
            if session is not None:
                self.route5_hold(session, output_dir=self.route5_state_output_dir(), reason="pause_button")
            self.llm_route5_status_var.set("LLM Route V5: paused.")

    def on_route5_global_pause_all(self) -> Dict[str, Any]:
        self.ensure_route5_state()
        paused: List[str] = []
        for name, attr in (
            ("route3", "llm_route3_pause_event"),
            ("route4", "llm_route4_pause_event"),
            ("route5", "llm_route5_pause_event"),
        ):
            event = getattr(self, attr, None)
            if event is not None and hasattr(event, "set"):
                event.set()
                paused.append(name)
        try:
            self.llm_route5_paused_var.set(True)
        except Exception:
            pass
        try:
            self.stop_keyboard_control(send_hold=False)
        except Exception:
            pass
        session = self.active_session()
        hold_result: Dict[str, Any] = {}
        if session is not None:
            hold_result = self.route5_hold(session, output_dir=self.route5_state_output_dir(), reason="global_pause_all_button")
        for status_attr, text in (
            ("llm_route3_status_var", "LLM Route V3: paused by V5 global pause."),
            ("llm_route4_status_var", "LLM Route V4: paused by V5 global pause."),
            ("llm_route5_status_var", "LLM Route V5: global pause requested."),
        ):
            var = getattr(self, status_attr, None)
            if var is not None and hasattr(var, "set"):
                try:
                    var.set(text)
                except Exception:
                    pass
        try:
            self.llm_route5_thinking_status_var.set("Thinking: GLOBAL_PAUSE_ALL manual operator pause")
        except Exception:
            pass
        result = {"paused": paused, "hold_result": self.route5_json_safe(hold_result), "created_at": datetime.now().isoformat(timespec="milliseconds")}
        self.route5_log_event(self.route5_state_output_dir(), "global_pause_all", result)
        return result

    def on_route5_release_movement(self) -> Dict[str, Any]:
        self.ensure_route5_state()
        pause_result = self.on_route5_global_pause_all()
        for attr in ("llm_route3_control_locked", "llm_route4_control_locked", "llm_route5_control_locked"):
            if hasattr(self, attr):
                setattr(self, attr, False)
        try:
            self.stop_keyboard_control(send_hold=False)
            self.update_keyboard_status("movement released")
        except Exception:
            pass
        session = self.active_session()
        movement_result: Dict[str, Any] = {}
        hold_result: Dict[str, Any] = {}
        if session is not None:
            try:
                movement_result = self.safe("Route V5 release movement enable", lambda: session.set_movement_enabled(True))
                hold_result = self.safe("Route V5 release movement hold", lambda: session.move_relative(action_payload("hold")))
                self.root.after(0, lambda r=movement_result: self.apply_state(r))
            except Exception as exc:
                movement_result = {"status": "error", "reason": str(exc)}
        try:
            self.movement_enabled_state = True
            self.llm_route5_status_var.set("LLM Route V5: movement released for manual control.")
            self.llm_route5_payload_var.set("Payload: manual movement released")
            self.llm_route5_thinking_status_var.set("Thinking: MOVEMENT_RELEASED global pause active; manual control unlocked")
        except Exception:
            pass
        result = {
            "released": True,
            "pause_result": self.route5_json_safe(pause_result),
            "movement_result": self.route5_json_safe(movement_result),
            "hold_result": self.route5_json_safe(hold_result),
            "locks": {
                "route3": bool(getattr(self, "llm_route3_control_locked", False)),
                "route4": bool(getattr(self, "llm_route4_control_locked", False)),
                "route5": bool(getattr(self, "llm_route5_control_locked", False)),
            },
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.route5_log_event(self.route5_state_output_dir(), "movement_released", result)
        return result

    def on_route5_stop(self) -> None:
        self.ensure_route5_state()
        self.llm_route5_stop_event.set()
        self.llm_route5_pause_event.clear()
        session = self.active_session()
        if session is not None:
            self.route5_hold(session, output_dir=self.route5_state_output_dir(), reason="stop_button")
        self.llm_route5_status_var.set("LLM Route V5: stop requested.")

    def on_route5_clear(self) -> None:
        self.ensure_route5_state()
        if self.llm_route5_thread is not None and self.llm_route5_thread.is_alive():
            self.llm_route5_status_var.set("LLM Route V5: stop before clearing.")
            return
        self.llm_route5_state = {}
        self.llm_route5_completed_facades = set()
        self.llm_route5_blocked_facades = set()
        self.llm_route5_stage_var.set("Stage: idle")
        self.llm_route5_active_var.set("Active: n/a")
        self.llm_route5_target_var.set("Target: n/a")
        self.llm_route5_error_var.set("Error: n/a")
        self.llm_route5_payload_var.set("Payload: hold")
        self.llm_route5_current_status_var.set("Current: idle")
        self.llm_route5_next_status_var.set("Next: n/a")
        self.llm_route5_avoidance_status_var.set("Avoidance: idle")
        self.llm_route5_representation_status_var.set("Representation: idle")
        self.llm_route5_thinking_status_var.set("Thinking: idle")
        self.llm_route5_status_var.set("LLM Route V5: cleared.")
        self.refresh_route5_support_views()

    def on_route5_validate_run(self) -> None:
        self.ensure_route5_state()
        output_dir = self.route5_state_output_dir()
        if output_dir is None:
            self.llm_route5_status_var.set("LLM Route V5: no run to validate.")
            return
        summary = self.route5_run_summary(output_dir, status=str(self.llm_route5_state.get("stage", "manual_validate") or "manual_validate"))
        self.llm_route5_status_var.set(f"LLM Route V5: run summary -> {output_dir / 'route5_fusion_summary.json'}")
        self.route5_log_event(output_dir, "manual_validate_run", summary)
