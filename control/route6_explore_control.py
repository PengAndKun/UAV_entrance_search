from __future__ import annotations

import csv

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


ROUTE6_DESIGN_DOC = "overleaf/Route6_entrance_search/v002/route6_window6_realtime_map_llm_targeting.md"
ROUTE6_V003_REQUIREMENTS_DOC = "overleaf/Route6_entrance_search/v003/route6_v003_full_stop_llm_semantic_navigation_requirements.md"
ROUTE6_DEFAULT_KNOWN_HOUSE_POLYGONS = [
    {
        "house_id": "001",
        "name": "house_1",
        "points": [
            {"x": 2800.0, "y": -1400.0},
            {"x": 4100.0, "y": -1400.0},
            {"x": 4100.0, "y": 1800.0},
            {"x": 2800.0, "y": 1800.0},
        ],
    },
    {
        "house_id": "002",
        "name": "house_2",
        "points": [
            {"x": -250.0, "y": 1450.0},
            {"x": 1680.0, "y": 1450.0},
            {"x": 1680.0, "y": 2200.0},
            {"x": -250.0, "y": 2200.0},
        ],
    },
    {
        "house_id": "003",
        "name": "house_3",
        "points": [
            {"x": -800.0, "y": 1450.0},
            {"x": -3000.0, "y": 1450.0},
            {"x": -3000.0, "y": 2450.0},
            {"x": -800.0, "y": 2450.0},
        ],
    },
    {
        "house_id": "004",
        "name": "house_4",
        "points": [
            {"x": -3600.0, "y": -1550.0},
            {"x": -3600.0, "y": -2400.0},
            {"x": -1100.0, "y": -2400.0},
            {"x": -1100.0, "y": -1550.0},
        ],
    },
    {
        "house_id": "005",
        "name": "house_5",
        "points": [
            {"x": -300.0, "y": -1400.0},
            {"x": -300.0, "y": -2350.0},
            {"x": 1700.0, "y": -2350.0},
            {"x": 1700.0, "y": -1400.0},
        ],
    },
]


class _Route6FallbackVar:
    def __init__(self, value: Any = "") -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


def _route6_string_var(value: str) -> Any:
    try:
        return tk.StringVar(value=value)
    except RuntimeError:
        return _Route6FallbackVar(value)


class Route6ExploreControlMixin:
    def ensure_route6_state(self) -> None:
        if not hasattr(self, "llm_route6_window"):
            self.llm_route6_window = None
        if not hasattr(self, "llm_route6_summary_text"):
            self.llm_route6_summary_text = None
        if not hasattr(self, "llm_route6_scroll_canvas"):
            self.llm_route6_scroll_canvas = None
        if not hasattr(self, "llm_route6_content_frame"):
            self.llm_route6_content_frame = None
        if not hasattr(self, "llm_route6_map_widget"):
            self.llm_route6_map_widget = None
        if not hasattr(self, "llm_route6_map_frame"):
            self.llm_route6_map_frame = None
        if not hasattr(self, "llm_route6_realtime_map_frame"):
            self.llm_route6_realtime_map_frame = None
        if not hasattr(self, "llm_route6_realtime_map_preview_label"):
            self.llm_route6_realtime_map_preview_label = None
        if not hasattr(self, "llm_route6_realtime_map_preview_photo"):
            self.llm_route6_realtime_map_preview_photo = None
        if not hasattr(self, "llm_route6_realtime_map_layer_combo"):
            self.llm_route6_realtime_map_layer_combo = None
        if not hasattr(self, "llm_route6_realtime_map_after_id"):
            self.llm_route6_realtime_map_after_id = None
        if not hasattr(self, "route6_update_map_window"):
            self.route6_update_map_window = None
        if not hasattr(self, "route6_update_map_scroll_canvas"):
            self.route6_update_map_scroll_canvas = None
        if not hasattr(self, "route6_update_map_content_frame"):
            self.route6_update_map_content_frame = None
        if not hasattr(self, "route6_update_map_preview_label"):
            self.route6_update_map_preview_label = None
        if not hasattr(self, "route6_update_map_preview_photo"):
            self.route6_update_map_preview_photo = None
        if not hasattr(self, "route6_update_map_layer_combo"):
            self.route6_update_map_layer_combo = None
        if not hasattr(self, "route6_update_map_capture_thread"):
            self.route6_update_map_capture_thread = None
        if not hasattr(self, "route6_update_map_capture_stop_event"):
            self.route6_update_map_capture_stop_event = threading.Event()
        if not hasattr(self, "route6_update_map_realtime_thread"):
            self.route6_update_map_realtime_thread = None
        if not hasattr(self, "route6_update_map_realtime_stop_event"):
            self.route6_update_map_realtime_stop_event = threading.Event()
        if not hasattr(self, "route6_update_map_pose_after_id"):
            self.route6_update_map_pose_after_id = None
        if not hasattr(self, "route6_capture_folder_reader_window"):
            self.route6_capture_folder_reader_window = None
        if not hasattr(self, "route6_capture_folder_listbox"):
            self.route6_capture_folder_listbox = None
        if not hasattr(self, "route6_pointcloud_report_text"):
            self.route6_pointcloud_report_text = None
        if not hasattr(self, "route6_capture_folder_records"):
            self.route6_capture_folder_records = []
        if not hasattr(self, "route6_test_planner_window"):
            self.route6_test_planner_window = None
        if not hasattr(self, "route6_test_planner_scroll_canvas"):
            self.route6_test_planner_scroll_canvas = None
        if not hasattr(self, "route6_test_planner_content_frame"):
            self.route6_test_planner_content_frame = None
        if not hasattr(self, "route6_test_planner_house_listbox"):
            self.route6_test_planner_house_listbox = None
        if not hasattr(self, "route6_test_planner_result_text"):
            self.route6_test_planner_result_text = None
        if not hasattr(self, "route6_test_planner_preview_label"):
            self.route6_test_planner_preview_label = None
        if not hasattr(self, "route6_test_planner_preview_photo"):
            self.route6_test_planner_preview_photo = None
        if not hasattr(self, "route6_test_planner_house_record_cache"):
            self.route6_test_planner_house_record_cache = []
        if not hasattr(self, "llm_route6_state"):
            self.llm_route6_state = {}
        if not hasattr(self, "llm_route6_thread"):
            self.llm_route6_thread = None
        if not hasattr(self, "llm_route6_stop_event"):
            self.llm_route6_stop_event = threading.Event()
        if not hasattr(self, "llm_route6_pause_event"):
            self.llm_route6_pause_event = threading.Event()
        if not hasattr(self, "llm_route6_force_next_event"):
            self.llm_route6_force_next_event = threading.Event()
        if not hasattr(self, "route6_full_stop_event"):
            self.route6_full_stop_event = threading.Event()
        if not hasattr(self, "route6_visual_orbit_stop_event"):
            self.route6_visual_orbit_stop_event = threading.Event()
        if not hasattr(self, "route6_runtime_map_config"):
            self.route6_runtime_map_config = None
        if not hasattr(self, "llm_route6_status_var"):
            self.llm_route6_status_var = tk.StringVar(value="LLM Route V6: idle")
        if not hasattr(self, "llm_route6_stage_var"):
            self.llm_route6_stage_var = tk.StringVar(value="Stage: idle")
        if not hasattr(self, "llm_route6_current_house_var"):
            self.llm_route6_current_house_var = tk.StringVar(value="Current house: n/a")
        if not hasattr(self, "llm_route6_queue_var"):
            self.llm_route6_queue_var = tk.StringVar(value="House queue: n/a")
        if not hasattr(self, "llm_route6_map_status_var"):
            self.llm_route6_map_status_var = tk.StringVar(value="Map: idle")
        if not hasattr(self, "llm_route6_output_dir_var"):
            self.llm_route6_output_dir_var = tk.StringVar(value="Output: n/a")
        if not hasattr(self, "llm_route6_metrics_var"):
            self.llm_route6_metrics_var = tk.StringVar(value="Metrics: mapped=0 searched=0 blocked=0 confidence=n/a corrected=n/a")
        if not hasattr(self, "llm_route6_max_houses_var"):
            self.llm_route6_max_houses_var = tk.StringVar(value="3")
        if not hasattr(self, "llm_route6_runtime_min_var"):
            self.llm_route6_runtime_min_var = tk.StringVar(value="30")
        if not hasattr(self, "llm_route6_standoff_cm_var"):
            default = getattr(self, "llm_route_standoff_cm_var", None)
            self.llm_route6_standoff_cm_var = tk.StringVar(value=str(default.get() if default is not None else "850"))
        if not hasattr(self, "llm_route6_scan_z_cm_var"):
            self.llm_route6_scan_z_cm_var = tk.StringVar(value="450")
        if not hasattr(self, "llm_route6_occupancy_resolution_m_var"):
            self.llm_route6_occupancy_resolution_m_var = tk.StringVar(value="0.25")
        if not hasattr(self, "llm_route6_coverage_threshold_var"):
            self.llm_route6_coverage_threshold_var = tk.StringVar(value="0.75")
        if not hasattr(self, "llm_route6_allow_save_corrected_var"):
            self.llm_route6_allow_save_corrected_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route6_task_prompt_var"):
            self.llm_route6_task_prompt_var = _route6_string_var("Explore the first house north of current UAV.")
        if not hasattr(self, "llm_route6_selected_target_var"):
            self.llm_route6_selected_target_var = _route6_string_var("Selected target: n/a")
        if not hasattr(self, "llm_route6_realtime_map_status_var"):
            self.llm_route6_realtime_map_status_var = _route6_string_var("Realtime map: idle")
        if not hasattr(self, "llm_route6_map_analysis_status_var"):
            self.llm_route6_map_analysis_status_var = _route6_string_var("LLM Map Analysis: idle")
        if not hasattr(self, "llm_route6_map_analysis_detail_var"):
            self.llm_route6_map_analysis_detail_var = _route6_string_var("Semantic target: n/a")
        if not hasattr(self, "llm_route6_navigation_target_var"):
            self.llm_route6_navigation_target_var = _route6_string_var("Navigation target: n/a")
        if not hasattr(self, "llm_route6_visual_status_var"):
            self.llm_route6_visual_status_var = _route6_string_var("LLM Visual Direction Analysis: idle")
        if not hasattr(self, "llm_route6_visual_detail_var"):
            self.llm_route6_visual_detail_var = _route6_string_var("Visual target: n/a")
        if not hasattr(self, "llm_route6_conflict_status_var"):
            self.llm_route6_conflict_status_var = _route6_string_var("Height Conflict / Replan: idle")
        if not hasattr(self, "llm_route6_conflict_detail_var"):
            self.llm_route6_conflict_detail_var = _route6_string_var("Conflict detail: n/a")
        if not hasattr(self, "llm_route6_or_status_var"):
            self.llm_route6_or_status_var = _route6_string_var("OR Avoidance: idle")
        if not hasattr(self, "llm_route6_or_detail_var"):
            self.llm_route6_or_detail_var = _route6_string_var("OR detail: n/a")
        if not hasattr(self, "route6_update_map_layer_var"):
            self.route6_update_map_layer_var = _route6_string_var("z_050")
        if not hasattr(self, "route6_update_map_status_var"):
            self.route6_update_map_status_var = _route6_string_var("Route 6 Update Map: idle")
        if not hasattr(self, "route6_update_map_pose_var"):
            self.route6_update_map_pose_var = _route6_string_var("UAV x=n/a y=n/a z=n/a yaw=n/a")
        if not hasattr(self, "route6_update_map_capture_interval_s_var"):
            self.route6_update_map_capture_interval_s_var = _route6_string_var("1.0")
        if not hasattr(self, "route6_update_map_min_move_cm_var"):
            self.route6_update_map_min_move_cm_var = _route6_string_var("50")
        if not hasattr(self, "route6_update_map_min_yaw_deg_var"):
            self.route6_update_map_min_yaw_deg_var = _route6_string_var("5")
        if not hasattr(self, "route6_capture_folder_status_var"):
            self.route6_capture_folder_status_var = _route6_string_var("Route 6 Capture Folder Reader: idle")
        if not hasattr(self, "route6_selected_capture_folder_var"):
            self.route6_selected_capture_folder_var = _route6_string_var("")
        if not hasattr(self, "route6_test_planner_status_var"):
            self.route6_test_planner_status_var = _route6_string_var("Route 6 Test Planner: idle")
        if not hasattr(self, "route6_test_planner_radar_distance_cm_var"):
            default = getattr(self, "llm_route6_standoff_cm_var", None)
            self.route6_test_planner_radar_distance_cm_var = _route6_string_var(str(default.get() if default is not None else "850"))
        if not hasattr(self, "route6_test_planner_scan_z_cm_var"):
            default = getattr(self, "llm_route6_scan_z_cm_var", None)
            self.route6_test_planner_scan_z_cm_var = _route6_string_var(str(default.get() if default is not None else "450"))
        if not hasattr(self, "route6_test_planner_edge_var"):
            self.route6_test_planner_edge_var = _route6_string_var("auto nearest")
        if not hasattr(self, "route6_test_planner_algorithm_var"):
            self.route6_test_planner_algorithm_var = _route6_string_var("nearest edge")
        if not hasattr(self, "route6_test_planner_scan_mode_var"):
            self.route6_test_planner_scan_mode_var = _route6_string_var("single best point")
        if not hasattr(self, "route6_test_planner_fov_deg_var"):
            self.route6_test_planner_fov_deg_var = _route6_string_var("60")
        if not hasattr(self, "route6_test_planner_overlap_var"):
            self.route6_test_planner_overlap_var = _route6_string_var("0.30")
        if not hasattr(self, "route6_test_planner_coverage_threshold_var"):
            self.route6_test_planner_coverage_threshold_var = _route6_string_var("0.90")
        if not hasattr(self, "route6_test_planner_reset_step_cm_var"):
            self.route6_test_planner_reset_step_cm_var = _route6_string_var("150")

    def route6_design_doc_path(self) -> Path:
        return PROJECT_ROOT / ROUTE6_DESIGN_DOC

    def route6_output_root(self) -> Path:
        override = getattr(self, "llm_route6_output_root_override", None)
        return Path(override) if override is not None else self.resolve_project_path("route6_explore_runs")

    def make_route6_output_dir(self) -> Path:
        root = self.route6_output_root()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        candidate = root / f"route6_nearest_map_{timestamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = root / f"route6_nearest_map_{timestamp}_{suffix}"
        (candidate / "map").mkdir(parents=True, exist_ok=True)
        (candidate / "houses").mkdir(parents=True, exist_ok=True)
        return candidate

    def route6_json_safe(self, payload: Any) -> Any:
        if isinstance(payload, Path):
            return str(payload)
        if isinstance(payload, np.ndarray):
            return payload.tolist()
        if isinstance(payload, np.integer):
            return int(payload)
        if isinstance(payload, np.floating):
            return float(payload)
        if isinstance(payload, dict):
            return {str(key): self.route6_json_safe(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self.route6_json_safe(value) for value in payload]
        return payload

    def route6_write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        if hasattr(self, "write_json_artifact"):
            self.write_json_artifact(path, self.route6_json_safe(payload))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.route6_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    def route6_append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        if hasattr(self, "append_jsonl"):
            self.append_jsonl(path, self.route6_json_safe(payload))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.route6_json_safe(payload), ensure_ascii=False) + "\n")

    def route6_current_pose(self) -> Dict[str, float]:
        try:
            pose = self.route3_current_pose()
            if isinstance(pose, dict) and pose:
                return {
                    "x": float(pose.get("x", 0.0) or 0.0),
                    "y": float(pose.get("y", 0.0) or 0.0),
                    "z": float(pose.get("z", 0.0) or 0.0),
                    "yaw": float(pose.get("yaw", pose.get("yaw_deg", 0.0)) or 0.0),
                }
        except Exception:
            pass
        state = getattr(self, "latest_state", {}) if isinstance(getattr(self, "latest_state", {}), dict) else {}
        raw_pose = state.get("pose", state.get("location", []))
        if isinstance(raw_pose, dict):
            return {
                "x": float(raw_pose.get("x", 0.0) or 0.0),
                "y": float(raw_pose.get("y", 0.0) or 0.0),
                "z": float(raw_pose.get("z", 0.0) or 0.0),
                "yaw": float(raw_pose.get("yaw", raw_pose.get("yaw_deg", 0.0)) or 0.0),
            }
        if isinstance(raw_pose, (list, tuple)) and len(raw_pose) >= 3:
            return {
                "x": float(raw_pose[0] or 0.0),
                "y": float(raw_pose[1] or 0.0),
                "z": float(raw_pose[2] or 0.0),
                "yaw": float(raw_pose[3] if len(raw_pose) > 3 else 0.0),
            }
        return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}

    def route6_house_states(self) -> Dict[str, Dict[str, Any]]:
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        house_states = state.get("house_states", {}) if isinstance(state.get("house_states", {}), dict) else {}
        return dict(house_states)

    def route6_get_runtime_map_config(self) -> Dict[str, Any]:
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        if bool(state.get("route6_known_house_coordinate_mode_enabled", False)):
            return self.route6_known_house_map_config()
        runtime = getattr(self, "route6_runtime_map_config", None)
        if isinstance(runtime, dict) and runtime:
            return runtime
        config = getattr(self, "map_config", {}) if isinstance(getattr(self, "map_config", {}), dict) else {}
        try:
            runtime = json.loads(json.dumps(config))
        except Exception:
            runtime = dict(config)
        self.route6_runtime_map_config = runtime
        return runtime

    def route6_rank_house_candidates(self) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        map_config = self.route6_get_runtime_map_config()
        standoff = float(self.llm_route6_standoff_cm_var.get() or 850.0)
        scan_z = float(self.llm_route6_scan_z_cm_var.get() or 450.0)
        return route6_map_builder.rank_house_candidates(
            map_config,
            self.route6_current_pose(),
            house_states=self.route6_house_states(),
            standoff_cm=standoff,
            scan_z_cm=scan_z,
        )

    def route6_int_param(self, variable: Any, default: int, *, min_value: int = 1, max_value: int = 100) -> int:
        try:
            value = variable.get() if hasattr(variable, "get") else variable
            number = int(float(value))
        except Exception:
            number = int(default)
        return max(int(min_value), min(int(max_value), int(number)))

    def route6_float_param(self, variable: Any, default: float, *, min_value: float = 0.0, max_value: float = 1e9) -> float:
        try:
            value = variable.get() if hasattr(variable, "get") else variable
            number = float(value)
            if not math.isfinite(number):
                number = float(default)
        except Exception:
            number = float(default)
        return max(float(min_value), min(float(max_value), float(number)))

    def route6_mapping_house_states(self) -> Dict[str, Dict[str, Any]]:
        states = {str(hid): dict(state) for hid, state in self.route6_house_states().items() if isinstance(state, dict)}
        for state in states.values():
            status = str(state.get("status", "") or "").strip()
            if status in {"mapped_complete", "mapped_partial", "needs_capture", "searched", "searched_no_entry", "terminal_blocked"}:
                state["cooldown_active"] = True
        return states

    def route6_rank_mapping_house_candidates(self) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        map_config = self.route6_get_runtime_map_config()
        standoff = float(self.llm_route6_standoff_cm_var.get() or 850.0)
        scan_z = float(self.llm_route6_scan_z_cm_var.get() or 450.0)
        return route6_map_builder.rank_house_candidates(
            map_config,
            self.route6_current_pose(),
            house_states=self.route6_mapping_house_states(),
            standoff_cm=standoff,
            scan_z_cm=scan_z,
        )

    def route6_parse_task_direction(self, prompt: str) -> str:
        text = str(prompt or "").strip().lower()
        direction_markers = [
            ("north", ("north", "北")),
            ("south", ("south", "南")),
            ("east", ("east", "东", "東")),
            ("west", ("west", "西")),
        ]
        for direction, markers in direction_markers:
            if any(marker in text for marker in markers):
                return direction
        return ""

    def route6_candidate_direction_relation(self, candidate: Dict[str, Any], pose: Dict[str, Any]) -> Dict[str, Any]:
        cx = float((candidate or {}).get("center_x", 0.0) or 0.0)
        cy = float((candidate or {}).get("center_y", 0.0) or 0.0)
        px = float((pose or {}).get("x", 0.0) or 0.0)
        py = float((pose or {}).get("y", 0.0) or 0.0)
        dx = cx - px
        dy = cy - py
        relation: List[str] = []
        if dy < -1.0:
            relation.append("north")
        elif dy > 1.0:
            relation.append("south")
        if dx > 1.0:
            relation.append("east")
        elif dx < -1.0:
            relation.append("west")
        return {
            "relation": relation,
            "dx_cm": round(float(dx), 2),
            "dy_cm": round(float(dy), 2),
            "north_distance_cm": round(max(0.0, -float(dy)), 2),
            "south_distance_cm": round(max(0.0, float(dy)), 2),
            "east_distance_cm": round(max(0.0, float(dx)), 2),
            "west_distance_cm": round(max(0.0, -float(dx)), 2),
        }

    def route6_candidate_matches_direction(self, candidate: Dict[str, Any], pose: Dict[str, Any], direction: str) -> bool:
        if not direction:
            return True
        relation = self.route6_candidate_direction_relation(candidate, pose)
        return str(direction or "") in relation.get("relation", [])

    def route6_candidate_direction_distance(self, candidate: Dict[str, Any], pose: Dict[str, Any], direction: str) -> float:
        relation = self.route6_candidate_direction_relation(candidate, pose)
        key = f"{str(direction or '').lower()}_distance_cm"
        try:
            return float(relation.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    def route6_candidate_blocked_for_realtime_map(self, candidate: Dict[str, Any]) -> bool:
        item = candidate if isinstance(candidate, dict) else {}
        if bool(item.get("realtime_map_blocked", False)):
            return True
        risk_text = " ".join(
            str(item.get(key, "") or "").lower()
            for key in ("obstacle_risk", "safety_state", "blocked_reason", "route6_map_gate")
        )
        return "blocked" in risk_text or "collision" in risk_text

    def route6_object_center_xy(self, obj: Dict[str, Any]) -> Tuple[float, float]:
        item = obj if isinstance(obj, dict) else {}
        center = item.get("center_cm", {}) if isinstance(item.get("center_cm", {}), dict) else {}
        try:
            cx = float(center.get("x", item.get("center_x", 0.0)) or 0.0)
        except Exception:
            cx = 0.0
        try:
            cy = float(center.get("y", item.get("center_y", 0.0)) or 0.0)
        except Exception:
            cy = 0.0
        return cx, cy

    def route6_object_direction_relation(self, obj: Dict[str, Any], pose: Dict[str, Any]) -> Dict[str, Any]:
        cx, cy = self.route6_object_center_xy(obj)
        candidate = {"center_x": cx, "center_y": cy}
        relation = self.route6_candidate_direction_relation(candidate, pose)
        explicit = (obj if isinstance(obj, dict) else {}).get("direction_from_uav", [])
        if isinstance(explicit, list) and explicit:
            relation["relation"] = [str(item).strip().lower() for item in explicit if str(item).strip()]
        return relation

    def route6_object_matches_direction(self, obj: Dict[str, Any], pose: Dict[str, Any], direction: str) -> bool:
        if not direction:
            return True
        relation = self.route6_object_direction_relation(obj, pose)
        return str(direction or "").strip().lower() in relation.get("relation", [])

    def route6_object_direction_distance(self, obj: Dict[str, Any], pose: Dict[str, Any], direction: str) -> float:
        item = obj if isinstance(obj, dict) else {}
        key = f"{str(direction or '').strip().lower()}_distance_cm"
        if key in item:
            try:
                return float(item.get(key, 0.0) or 0.0)
            except Exception:
                pass
        relation = self.route6_object_direction_relation(item, pose)
        try:
            return float(relation.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    def route6_object_semantic_label(self, obj: Dict[str, Any]) -> str:
        item = obj if isinstance(obj, dict) else {}
        label = str(item.get("semantic_label", "") or "").strip().lower()
        if label:
            return label
        overlap = item.get("map_config_house_overlap", {}) if isinstance(item.get("map_config_house_overlap", {}), dict) else {}
        try:
            iou = float(overlap.get("iou", 0.0) or 0.0)
        except Exception:
            iou = 0.0
        try:
            persistence = float(item.get("height_persistence_score", 0.0) or 0.0)
        except Exception:
            persistence = 0.0
        try:
            linearity = float(item.get("linearity_score", 0.0) or 0.0)
        except Exception:
            linearity = 0.0
        try:
            area = float(item.get("area_score", 0.0) or 0.0)
        except Exception:
            area = 0.0
        if str(item.get("house_id", "") or "").strip() or iou >= 0.35 or persistence >= 0.55 or area >= 0.45:
            return "house"
        if linearity >= 0.75 and area <= 0.25:
            return "fence_or_rail"
        return "unknown"

    def route6_layer_z_from_key(self, key: str) -> int:
        try:
            return int(str(key or "").lower().replace("z_", ""))
        except Exception:
            return 0

    def route6_choose_realtime_layer_key(self, layers: List[Dict[str, Any]], selected_key: str = "") -> str:
        values = [self._route6_update_map_layer_key(layer) for layer in layers if isinstance(layer, dict)]
        if not values:
            return str(selected_key or "z_050")
        pose_z = float(self.route6_current_pose().get("z", 0.0) or 0.0)
        current = str(selected_key or "")
        if current in values and current != "z_050":
            return current
        return min(values, key=lambda key: abs(float(self.route6_layer_z_from_key(key)) - pose_z))

    def route6_build_realtime_map_planning_context(
        self,
        *,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        prompt = str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else "")
        pose = self.route6_current_pose()
        manifest = self.route6_update_map_load_manifest(build_if_missing=False)
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        selected_layer_key = self.route6_choose_realtime_layer_key(layers, str(self.route6_update_map_layer_var.get() or ""))
        if layers:
            self.route6_update_map_layer_var.set(selected_layer_key)
        selected_layer = next(
            (layer for layer in layers if isinstance(layer, dict) and self._route6_update_map_layer_key(layer) == selected_layer_key),
            {},
        )
        ranked_candidates = list(candidates) if isinstance(candidates, list) else self.route6_rank_mapping_house_candidates()
        return {
            "schema": "route6_realtime_map_planning_context_v1",
            "task_prompt": prompt,
            "requested_direction": self.route6_parse_task_direction(prompt),
            "current_pose": pose,
            "map_coordinate_frame": dict(route6_map_builder.ROUTE6_COORDINATE_FRAME),
            "north_rule": "candidate.center_y < uav.y in Unreal cm; standard_y = -unreal_y / 100",
            "selected_layer_key": selected_layer_key,
            "selected_layer_z_cm": self.route6_layer_z_from_key(selected_layer_key),
            "realtime_map_manifest_path": str((self.llm_route6_state.get("route6_update_map", {}) if isinstance(self.llm_route6_state.get("route6_update_map", {}), dict) else {}).get("manifest_path", "") or ""),
            "realtime_map_available": bool(manifest),
            "realtime_map_layer": self.route6_json_safe(selected_layer),
            "candidate_count": len(ranked_candidates),
            "candidates": self.route6_json_safe(ranked_candidates),
            "obstacle_summary": self.route6_json_safe(
                {
                    "last_obstacle_validation": self.llm_route6_state.get("last_obstacle_validation", {}),
                    "last_house_state": self.llm_route6_state.get("last_house_state", {}),
                }
            ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def route6_manifest_layer_records(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = manifest if isinstance(manifest, dict) else {}
        return [dict(layer) for layer in payload.get("layers", []) if isinstance(layer, dict)]

    def route6_layer_occupancy_summary(self, layers: List[Dict[str, Any]]) -> Dict[str, Any]:
        values: List[Dict[str, Any]] = []
        occupied_counts: List[int] = []
        for layer in layers if isinstance(layers, list) else []:
            key = self._route6_update_map_layer_key(layer)
            try:
                occupied = int(layer.get("occupied_cell_count", 0) or 0)
            except Exception:
                occupied = 0
            try:
                points = int(layer.get("point_count", 0) or 0)
            except Exception:
                points = 0
            occupied_counts.append(occupied)
            values.append({"layer_key": key, "z_cm": self.route6_layer_z_from_key(key), "occupied_cell_count": occupied, "point_count": points})
        nonzero = [count for count in occupied_counts if count > 0]
        if nonzero:
            sorted_counts = sorted(nonzero)
            median = sorted_counts[len(sorted_counts) // 2]
            open_threshold = max(1, int(median * 0.75))
        else:
            open_threshold = 0
        open_layers = [
            item["layer_key"]
            for item in values
            if int(item.get("occupied_cell_count", 0) or 0) <= open_threshold
        ]
        return {
            "layer_count": len(values),
            "available_layer_keys": [item["layer_key"] for item in values],
            "open_corridor_layers": open_layers,
            "layer_stats": values,
            "open_corridor_policy": "layers with occupied cells <= 75% of nonzero median; empty maps treated as open",
        }

    def route6_candidate_to_semantic_object(
        self,
        candidate: Dict[str, Any],
        pose: Dict[str, Any],
        layers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        item = candidate if isinstance(candidate, dict) else {}
        house_id = str(item.get("house_id", "") or "").strip()
        cx = float(item.get("center_x", 0.0) or 0.0)
        cy = float(item.get("center_y", 0.0) or 0.0)
        relation = self.route6_candidate_direction_relation(item, pose)
        nonzero_layers = [layer for layer in layers if int(layer.get("occupied_cell_count", 0) or 0) > 0]
        persistence = 1.0 if not layers else min(1.0, max(0.35, len(nonzero_layers) / max(1.0, float(len(layers)))))
        bbox = item.get("bbox", {}) if isinstance(item.get("bbox", {}), dict) else {}
        if bbox:
            try:
                area_score = min(
                    1.0,
                    abs(float(bbox.get("max_x", cx)) - float(bbox.get("min_x", cx)))
                    * abs(float(bbox.get("max_y", cy)) - float(bbox.get("min_y", cy)))
                    / 1000000.0,
                )
            except Exception:
                area_score = 0.6
        else:
            area_score = 0.6
        layer_occupancy = {
            self._route6_update_map_layer_key(layer): int(layer.get("occupied_cell_count", 0) or 0)
            for layer in layers
            if isinstance(layer, dict)
        }
        blocked = self.route6_candidate_blocked_for_realtime_map(item)
        return {
            "schema": "route6_semantic_object_candidate_v1",
            "object_id": f"house_{house_id}" if house_id else f"object_{len(str(item))}",
            "semantic_label": "house",
            "house_id": house_id,
            "name": str(item.get("name", f"House_{house_id}") or f"House_{house_id}"),
            "center_cm": {"x": round(cx, 2), "y": round(cy, 2)},
            "bbox_cm": self.route6_json_safe(bbox),
            "bbox": self.route6_json_safe(bbox),
            "direction_from_uav": relation.get("relation", []),
            "north_distance_cm": relation.get("north_distance_cm", 0.0),
            "south_distance_cm": relation.get("south_distance_cm", 0.0),
            "east_distance_cm": relation.get("east_distance_cm", 0.0),
            "west_distance_cm": relation.get("west_distance_cm", 0.0),
            "visible_from_uav": True,
            "layer_occupancy": layer_occupancy,
            "height_persistence_score": round(float(persistence), 4),
            "linearity_score": 0.2,
            "area_score": round(float(area_score), 4),
            "enclosure_score": round(float(min(1.0, area_score * persistence)), 4),
            "map_config_house_overlap": {"house_id": house_id, "iou": 1.0 if house_id else 0.0},
            "approach_blocked": bool(blocked),
            "selected_candidate": self.route6_json_safe(item),
        }

    def route6_extract_layered_object_candidates(
        self,
        manifest: Optional[Dict[str, Any]] = None,
        *,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        layers = self.route6_manifest_layer_records(manifest if isinstance(manifest, dict) else {})
        pose = self.route6_current_pose()
        ranked = list(candidates) if isinstance(candidates, list) else self.route6_rank_mapping_house_candidates()
        objects = [self.route6_candidate_to_semantic_object(candidate, pose, layers) for candidate in ranked if isinstance(candidate, dict)]
        return sorted(
            objects,
            key=lambda item: (
                0 if self.route6_object_semantic_label(item) == "house" else 1,
                self.route6_object_direction_distance(item, pose, "north"),
                str(item.get("object_id", "")),
            ),
        )

    def route6_build_semantic_map_context(
        self,
        *,
        output_dir: Optional[Path] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        manifest = self.route6_update_map_load_manifest(build_if_missing=False, output_dir=out_path)
        manifest_path = self.route6_update_map_manifest_path(out_path) if out_path is not None else Path()
        layers = self.route6_manifest_layer_records(manifest)
        summary = self.route6_layer_occupancy_summary(layers)
        prompt = str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else "")
        pose = self.route6_current_pose()
        objects = self.route6_extract_layered_object_candidates(manifest, candidates=candidates)
        context = {
            "schema": "route6_semantic_map_context_v1",
            "mission_prompt": prompt,
            "task_prompt": prompt,
            "requested_direction": self.route6_parse_task_direction(prompt),
            "current_pose": pose,
            "map_coordinate_frame": dict(route6_map_builder.ROUTE6_COORDINATE_FRAME),
            "north_rule": "Unreal/map_config north uses smaller y; standard_y = -unreal_y / 100",
            "manifest_path": str(manifest_path) if manifest_path else "",
            "manifest_available": bool(manifest),
            "layer_count": int(manifest.get("layer_count", len(layers)) if isinstance(manifest, dict) else len(layers)),
            "available_layer_keys": summary["available_layer_keys"],
            "open_corridor_layers": summary["open_corridor_layers"],
            "layer_stats": summary["layer_stats"],
            "object_candidates": self.route6_json_safe(objects),
            "house_candidate_count": len([item for item in objects if self.route6_object_semantic_label(item) == "house"]),
            "obstacle_summary": self.route6_json_safe(
                {
                    "last_obstacle_validation": self.llm_route6_state.get("last_obstacle_validation", {}),
                    "last_house_state": self.llm_route6_state.get("last_house_state", {}),
                }
            ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.llm_route6_state["llm_semantic_map_context"] = self.route6_json_safe(context)
        self.llm_route6_map_analysis_status_var.set(
            f"LLM Map Analysis: layers={context['layer_count']} objects={len(objects)} manifest={'yes' if manifest else 'no'}"
        )
        self.route6_write_state_artifact()
        return context

    def route6_select_llm_semantic_target(
        self,
        context: Dict[str, Any],
        *,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        ctx = context if isinstance(context, dict) else {}
        pose = ctx.get("current_pose", {}) if isinstance(ctx.get("current_pose", {}), dict) else self.route6_current_pose()
        requested_direction = str(ctx.get("requested_direction", "") or "").strip().lower()
        objects = [dict(item) for item in ctx.get("object_candidates", []) if isinstance(item, dict)]
        selectable = [item for item in objects if self.route6_object_matches_direction(item, pose, requested_direction)] if requested_direction else list(objects)
        fallback_used = bool(requested_direction and not selectable)
        pool = selectable if selectable else objects
        rejected: List[Dict[str, Any]] = []

        def _sort_key(item: Dict[str, Any]) -> Tuple[float, float, str]:
            distance = self.route6_object_direction_distance(item, pose, requested_direction)
            if not requested_direction or fallback_used:
                distance = math.hypot(
                    float(self.route6_object_center_xy(item)[0]) - float(pose.get("x", 0.0) or 0.0),
                    float(self.route6_object_center_xy(item)[1]) - float(pose.get("y", 0.0) or 0.0),
                )
            label_rank = 0.0 if self.route6_object_semantic_label(item) == "house" else 1.0
            return (distance, label_rank, str(item.get("object_id", "")))

        selected: Dict[str, Any] = {}
        for obj in sorted(pool, key=_sort_key):
            label = self.route6_object_semantic_label(obj)
            if bool(obj.get("approach_blocked", False)):
                rejected.append(
                    {
                        "object_id": str(obj.get("object_id", "") or ""),
                        "semantic_label": label,
                        "reason": "approach_blocked",
                        "direction_distance_cm": self.route6_object_direction_distance(obj, pose, requested_direction),
                    }
                )
                continue
            if label == "house":
                selected = obj
                break
            rejected.append(
                {
                    "object_id": str(obj.get("object_id", "") or ""),
                    "semantic_label": label,
                    "reason": "not_house_target",
                    "direction_distance_cm": self.route6_object_direction_distance(obj, pose, requested_direction),
                }
            )
        if not selected and pool:
            selected = sorted(pool, key=_sort_key)[0]
            fallback_used = True
        selected_label = self.route6_object_semantic_label(selected) if selected else ""
        relation = self.route6_object_direction_relation(selected, pose) if selected else {"relation": []}
        selected_candidate = selected.get("selected_candidate", {}) if isinstance(selected.get("selected_candidate", {}), dict) else {}
        house_id = str(selected.get("house_id", selected_candidate.get("house_id", "")) or "").strip()
        confidence = 0.82 if selected_label == "house" and not fallback_used else (0.52 if selected else 0.0)
        selection = {
            "schema": "route6_llm_semantic_target_selection_v1",
            "source": "deterministic_semantic_llm_fallback",
            "mission_prompt": str(ctx.get("mission_prompt", ctx.get("task_prompt", "")) or ""),
            "requested_direction": requested_direction,
            "selected_object_id": str(selected.get("object_id", "") or "") if selected else "",
            "semantic_label": selected_label,
            "house_id": house_id,
            "selected_candidate": self.route6_json_safe(selected_candidate),
            "direction_relation": relation.get("relation", []),
            "direction_distance_cm": self.route6_object_direction_distance(selected, pose, requested_direction) if selected else 0.0,
            "recommended_facade": str(selected_candidate.get("nearest_facade", "") or ""),
            "recommended_layer_key": self.route6_choose_realtime_layer_key(
                [{"z_cm": self.route6_layer_z_from_key(key)} for key in ctx.get("open_corridor_layers", []) if isinstance(key, str)]
                or [{"z_cm": self.route6_layer_z_from_key(key)} for key in ctx.get("available_layer_keys", []) if isinstance(key, str)]
            ),
            "confidence": round(float(confidence), 4),
            "fallback_used": bool(fallback_used),
            "rejected_objects": self.route6_json_safe(rejected),
            "risk_notes": [
                "nearer non-house objects are skipped before selecting a house"
                if rejected
                else "no nearer non-house rejection needed",
                "LLM unavailable or not configured; deterministic semantic ranking used",
            ],
            "why_this_is_first_house": (
                f"selected nearest {requested_direction or 'reachable'} object classified as house after skipping non-house obstacles"
                if selected
                else "no semantic target available"
            ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.llm_route6_state["llm_semantic_target_selection"] = self.route6_json_safe(selection)
        self.llm_route6_map_analysis_detail_var.set(
            f"Semantic target: house={house_id or 'n/a'} object={selection['selected_object_id'] or 'n/a'} "
            f"label={selected_label or 'n/a'} confidence={selection['confidence']}"
        )
        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is None:
            state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
            out_path = Path(state_output) if state_output else None
        if out_path is not None:
            path = out_path / "route6_llm_semantic_target_selection.json"
            self.route6_write_json_artifact(path, selection)
            self.llm_route6_state["llm_semantic_target_selection_path"] = str(path)
            self.route6_write_state_artifact()
        return selection

    def route6_plan_llm_navigation_target(
        self,
        output_dir: Path,
        selection: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        ctx = context if isinstance(context, dict) else {}
        selected = selection if isinstance(selection, dict) else {}
        candidate = selected.get("selected_candidate", {}) if isinstance(selected.get("selected_candidate", {}), dict) else {}
        house_id = str(selected.get("house_id", candidate.get("house_id", "")) or "").strip()
        if not candidate and house_id:
            for item in self.route6_rank_mapping_house_candidates():
                if str(item.get("house_id", "") or "") == house_id:
                    candidate = dict(item)
                    break
        target_pose = dict(candidate.get("nearest_scan_pose", {}) if isinstance(candidate.get("nearest_scan_pose", {}), dict) else {})
        if not target_pose:
            pose = ctx.get("current_pose", {}) if isinstance(ctx.get("current_pose", {}), dict) else self.route6_current_pose()
            cx = float(pose.get("x", 0.0) or 0.0)
            cy = float(pose.get("y", 0.0) or 0.0) - 700.0
            target_pose = {"x": cx, "y": cy, "z": float(pose.get("z", 300.0) or 300.0), "yaw": 180.0, "facade": "south"}
        available_layers = [str(key) for key in ctx.get("available_layer_keys", []) if str(key).strip()]
        open_layers = [str(key) for key in ctx.get("open_corridor_layers", []) if str(key).strip()]
        if not available_layers:
            available_layers = [str(self.route6_update_map_layer_var.get() or "z_050")]
        preferred_layers = [key for key in open_layers if key in available_layers] or available_layers
        try:
            pose_z = float((ctx.get("current_pose", {}) if isinstance(ctx.get("current_pose", {}), dict) else {}).get("z", self.route6_current_pose().get("z", 0.0)) or 0.0)
        except Exception:
            pose_z = 0.0
        layer_key = min(preferred_layers, key=lambda key: abs(float(self.route6_layer_z_from_key(key)) - pose_z))
        target_pose["z"] = float(target_pose.get("z", self.route6_layer_z_from_key(layer_key)) or self.route6_layer_z_from_key(layer_key))
        target = {
            "schema": "route6_llm_navigation_target_v1",
            "source": "route6_semantic_layered_map_planner",
            "target_object_id": str(selected.get("selected_object_id", "") or ""),
            "house_id": house_id,
            "selected_object_id": str(selected.get("selected_object_id", "") or ""),
            "semantic_label": str(selected.get("semantic_label", "") or ""),
            "target_pose_cm": self.route6_json_safe(target_pose),
            "expected_facade": str(selected.get("recommended_facade", candidate.get("nearest_facade", target_pose.get("facade", ""))) or ""),
            "recommended_facade": str(selected.get("recommended_facade", candidate.get("nearest_facade", target_pose.get("facade", ""))) or ""),
            "approach_layer_key": layer_key,
            "approach_layer_z_cm": int(self.route6_layer_z_from_key(layer_key)),
            "standoff_cm": self.route6_float_param(self.llm_route6_standoff_cm_var, 850.0, min_value=0.0, max_value=10000.0),
            "available_layer_keys": available_layers,
            "open_corridor_layers": open_layers,
            "path_summary": {
                "status": "candidate_clear" if layer_key in open_layers else "candidate_unknown",
                "blocked_layers": [key for key in available_layers if key not in open_layers],
                "open_corridor_score": round(float(len(open_layers)) / float(max(1, len(available_layers))), 4),
                "policy": "choose layer closest to current UAV z among open 13-layer corridor candidates",
                "fallback_used": bool(layer_key not in open_layers),
            },
            "replan_policy": {
                "replan_on_blocked": True,
                "min_move_cm": 150,
                "min_yaw_deg": 20,
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        out_path = Path(output_dir)
        self.route6_write_json_artifact(out_path / "route6_llm_navigation_target.json", target)
        self.llm_route6_state["llm_navigation_target"] = self.route6_json_safe(target)
        self.llm_route6_state["llm_navigation_target_path"] = str(out_path / "route6_llm_navigation_target.json")
        self.llm_route6_navigation_target_var.set(
            f"Navigation target: house={house_id or 'n/a'} layer={layer_key} "
            f"x={float(target_pose.get('x', 0.0) or 0.0):.1f} y={float(target_pose.get('y', 0.0) or 0.0):.1f}"
        )
        self.route6_write_state_artifact()
        return target

    def refresh_llm_route6_map_analysis_panel(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        output_dir = None
        state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        if state_output:
            output_dir = Path(state_output)
        context = self.route6_build_semantic_map_context(output_dir=None)
        selection = self.route6_select_llm_semantic_target(context, output_dir=output_dir)
        target: Dict[str, Any] = {}
        if output_dir is not None:
            target = self.route6_plan_llm_navigation_target(output_dir, selection, context)
        self.llm_route6_map_analysis_status_var.set(
            f"LLM Map Analysis: target={selection.get('house_id', '') or 'n/a'} "
            f"layer={target.get('approach_layer_key', selection.get('recommended_layer_key', '') or 'n/a')} "
            f"objects={len(context.get('object_candidates', []))}"
        )
        return {
            "schema": "route6_llm_map_analysis_panel_v1",
            "context": self.route6_json_safe(context),
            "selection": self.route6_json_safe(selection),
            "navigation_target": self.route6_json_safe(target),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def route6_select_llm_target_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_route6_state()
        ctx = context if isinstance(context, dict) else {}
        pose = ctx.get("current_pose", {}) if isinstance(ctx.get("current_pose", {}), dict) else self.route6_current_pose()
        requested_direction = str(ctx.get("requested_direction", "") or "").strip().lower()
        candidates = [dict(item) for item in ctx.get("candidates", []) if isinstance(item, dict)]
        skipped_blocked = [item for item in candidates if self.route6_candidate_blocked_for_realtime_map(item)]
        selectable = [
            item
            for item in candidates
            if bool(item.get("reachable", True)) and not self.route6_candidate_blocked_for_realtime_map(item)
        ]
        directional = [
            item
            for item in selectable
            if self.route6_candidate_matches_direction(item, pose, requested_direction)
        ] if requested_direction else list(selectable)
        fallback_used = bool(requested_direction and not directional)
        pool = directional if directional else selectable

        def _sort_key(item: Dict[str, Any]) -> Tuple[float, float, float]:
            direction_distance = self.route6_candidate_direction_distance(item, pose, requested_direction)
            try:
                score = float(item.get("score", item.get("route_cost_cm", 0.0)) or 0.0)
            except Exception:
                score = 0.0
            try:
                route_cost = float(item.get("route_cost_cm", score) or score)
            except Exception:
                route_cost = score
            return (direction_distance if requested_direction and not fallback_used else score, route_cost, score)

        selected = min(pool, key=_sort_key) if pool else {}
        relation = self.route6_candidate_direction_relation(selected, pose) if selected else {"relation": []}
        house_id = str(selected.get("house_id", "") or "") if selected else ""
        recommended_facade = str(selected.get("nearest_facade", "") or "")
        if not recommended_facade:
            nearest_pose = selected.get("nearest_scan_pose", {}) if isinstance(selected.get("nearest_scan_pose", {}), dict) else {}
            recommended_facade = str(nearest_pose.get("facade", "") or "")
        confidence = 0.72 if house_id and requested_direction and not fallback_used else (0.55 if house_id else 0.0)
        risk_notes = []
        if skipped_blocked:
            risk_notes.append(f"skipped {len(skipped_blocked)} blocked realtime-map/obstacle candidates")
        if fallback_used:
            risk_notes.append(f"no {requested_direction} candidate available; used nearest reachable fallback")
        return {
            "schema": "route6_llm_target_selection_v1",
            "source": "deterministic_llm_target_fallback",
            "task_prompt": str(ctx.get("task_prompt", "") or ""),
            "requested_direction": requested_direction,
            "house_id": house_id,
            "selected_candidate": self.route6_json_safe(selected),
            "direction_relation": relation.get("relation", []),
            "direction_delta_cm": {
                "dx": relation.get("dx_cm", 0.0),
                "dy": relation.get("dy_cm", 0.0),
            },
            "direction_distance_cm": self.route6_candidate_direction_distance(selected, pose, requested_direction) if selected else 0.0,
            "recommended_facade": recommended_facade,
            "approach_layer_key": str(ctx.get("selected_layer_key", "") or ""),
            "approach_layer_z_cm": int(ctx.get("selected_layer_z_cm", 0) or 0),
            "confidence": round(float(confidence), 4),
            "fallback_used": bool(fallback_used),
            "skipped_blocked_candidate_count": len(skipped_blocked),
            "risk_notes": risk_notes,
            "selection_reason": (
                f"selected first reachable {requested_direction} house from realtime map context"
                if house_id and requested_direction and not fallback_used
                else ("selected nearest reachable fallback from realtime map context" if house_id else "no reachable house candidate")
            ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def route6_apply_selected_target_to_scan_plan(
        self,
        output_dir: Optional[Path] = None,
        selection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        target_selection = dict(selection) if isinstance(selection, dict) else self.route6_select_llm_target_from_context(
            self.route6_build_realtime_map_planning_context()
        )
        selected_candidate = target_selection.get("selected_candidate", {}) if isinstance(target_selection.get("selected_candidate", {}), dict) else {}
        selected_house_id = str(target_selection.get("house_id", selected_candidate.get("house_id", "")) or "")
        if selected_house_id:
            self.llm_route6_state["selected_house_id"] = selected_house_id
            self.llm_route6_state["selected_candidate"] = selected_candidate
        self.llm_route6_state["llm_target_selection"] = target_selection
        if hasattr(self, "llm_route6_selected_target_var"):
            self.llm_route6_selected_target_var.set(
                f"Selected target: house={selected_house_id or 'n/a'} "
                f"direction={target_selection.get('requested_direction', '') or 'any'} "
                f"layer={target_selection.get('approach_layer_key', '') or 'n/a'}"
            )
        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is None:
            state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
            out_path = Path(state_output) if state_output else None
        if out_path is not None:
            path = Path(out_path) / "route6_llm_target_selection.json"
            self.route6_write_json_artifact(path, target_selection)
            self.llm_route6_state["llm_target_selection_path"] = str(path)
            self.route6_write_state_artifact()
        return target_selection

    def route6_known_house_polygon_records(self) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        raw_records = state.get("route6_known_house_polygons", ROUTE6_DEFAULT_KNOWN_HOUSE_POLYGONS)
        if not isinstance(raw_records, list) or not raw_records:
            raw_records = ROUTE6_DEFAULT_KNOWN_HOUSE_POLYGONS
        records: List[Dict[str, Any]] = []
        for index, raw in enumerate(raw_records, start=1):
            item = raw if isinstance(raw, dict) else {}
            raw_points = item.get("points", []) if isinstance(item.get("points", []), list) else []
            points: List[Dict[str, float]] = []
            for point in raw_points:
                if not isinstance(point, dict):
                    continue
                try:
                    points.append({"x": float(point.get("x", 0.0) or 0.0), "y": float(point.get("y", 0.0) or 0.0)})
                except Exception:
                    continue
            if len(points) < 2:
                continue
            xs = [float(point["x"]) for point in points]
            ys = [float(point["y"]) for point in points]
            bbox = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}
            ordered_points = [
                {"x": bbox["min_x"], "y": bbox["min_y"]},
                {"x": bbox["max_x"], "y": bbox["min_y"]},
                {"x": bbox["max_x"], "y": bbox["max_y"]},
                {"x": bbox["min_x"], "y": bbox["max_y"]},
            ]
            center_x = 0.5 * (bbox["min_x"] + bbox["max_x"])
            center_y = 0.5 * (bbox["min_y"] + bbox["max_y"])
            house_id = str(item.get("house_id", item.get("id", f"{index:03d}")) or f"{index:03d}").strip()
            if house_id.isdigit():
                house_id = f"{int(house_id):03d}"
            records.append(
                {
                    "schema": "route6_known_house_polygon_v1",
                    "house_id": house_id,
                    "id": house_id,
                    "name": str(item.get("name", f"house_{index}") or f"house_{index}"),
                    "points": ordered_points,
                    "bbox": {key: round(float(value), 2) for key, value in bbox.items()},
                    "center_x": round(float(center_x), 2),
                    "center_y": round(float(center_y), 2),
                    "radius_cm": round(max(abs(bbox["max_x"] - bbox["min_x"]), abs(bbox["max_y"] - bbox["min_y"])) * 0.5, 2),
                    "source": "operator_known_coordinates",
                }
            )
        return records

    def route6_known_house_map_config(self) -> Dict[str, Any]:
        base = getattr(self, "map_config", {}) if isinstance(getattr(self, "map_config", {}), dict) else {}
        try:
            config = json.loads(json.dumps(base))
        except Exception:
            config = dict(base)
        houses = []
        for record in self.route6_known_house_polygon_records():
            bbox = record.get("bbox", {}) if isinstance(record.get("bbox", {}), dict) else {}
            houses.append(
                {
                    "id": str(record.get("house_id", "") or ""),
                    "house_id": str(record.get("house_id", "") or ""),
                    "name": str(record.get("name", "") or ""),
                    "center_x": float(record.get("center_x", 0.0) or 0.0),
                    "center_y": float(record.get("center_y", 0.0) or 0.0),
                    "radius_cm": float(record.get("radius_cm", 300.0) or 300.0),
                    "route6_candidate_bbox_world": self.route6_json_safe(bbox),
                    "route6_manual_polygon_points": self.route6_json_safe(record.get("points", [])),
                    "route6_status": "unknown",
                    "route6_source": "operator_known_coordinates",
                }
            )
        config["houses"] = houses
        config["route6_known_house_coordinate_mode"] = True
        config["route6_known_house_count"] = len(houses)
        return config

    def route6_enable_known_house_coordinate_mode(self) -> List[Dict[str, Any]]:
        records = self.route6_known_house_polygon_records()
        self.llm_route6_state["route6_known_house_coordinate_mode_enabled"] = True
        self.llm_route6_state["route6_known_house_polygons"] = self.route6_json_safe(records)
        self.route6_runtime_map_config = self.route6_known_house_map_config()
        self.route6_write_state_artifact()
        return records

    def route6_unreal_cm_to_layer_pixel(self, metadata: Dict[str, Any], x_cm: float, y_cm: float) -> Tuple[int, int]:
        width = int(metadata.get("width", 0) or 0)
        height = int(metadata.get("height", 0) or 0)
        resolution = float(metadata.get("resolution_m", 0.25) or 0.25)
        origin_x, origin_y = [float(value) for value in metadata.get("origin_standard_m", [0.0, 0.0])]
        standard_x = float(x_cm) / 100.0
        standard_y = -float(y_cm) / 100.0
        col = int(math.floor((standard_x - origin_x) / resolution))
        row = int(math.floor((standard_y - origin_y) / resolution))
        x_px = max(0, min(width - 1, col))
        y_px = max(0, min(height - 1, height - 1 - row))
        return x_px, y_px

    def route6_draw_known_house_overlay_on_image(
        self,
        image: Image.Image,
        metadata: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> Image.Image:
        source = image.convert("RGB")
        base = Image.new("RGB", source.size, "white")
        draw = ImageDraw.Draw(base)
        colors = [(170, 205, 255), (170, 225, 190), (235, 190, 150), (215, 180, 235), (180, 220, 235)]
        for index, record in enumerate(records):
            color = colors[index % len(colors)]
            points = record.get("points", []) if isinstance(record.get("points", []), list) else []
            projected = [
                self.route6_unreal_cm_to_layer_pixel(metadata, float(point.get("x", 0.0) or 0.0), float(point.get("y", 0.0) or 0.0))
                for point in points
                if isinstance(point, dict)
            ]
            if len(projected) >= 2:
                draw.line(projected + [projected[0]], fill=color, width=2)
            for x_px, y_px in projected:
                draw.ellipse((x_px - 3, y_px - 3, x_px + 3, y_px + 3), fill=color)
            bbox = record.get("bbox", {}) if isinstance(record.get("bbox", {}), dict) else {}
            cx = 0.5 * (float(bbox.get("min_x", 0.0) or 0.0) + float(bbox.get("max_x", 0.0) or 0.0))
            cy = 0.5 * (float(bbox.get("min_y", 0.0) or 0.0) + float(bbox.get("max_y", 0.0) or 0.0))
            tx, ty = self.route6_unreal_cm_to_layer_pixel(metadata, cx, cy)
            draw.text((tx + 4, ty + 4), f"H{str(record.get('house_id', ''))}", fill=color)
        source_arr = np.asarray(source)
        base_arr = np.asarray(base).copy()
        obstacle_mask = np.all(source_arr <= 32, axis=2)
        base_arr[obstacle_mask] = source_arr[obstacle_mask]
        return Image.fromarray(base_arr.astype(np.uint8), mode="RGB")

    def route6_apply_known_house_polygons_to_update_map(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        if out_path is None:
            self.route6_update_map_status_var.set("Route 6 Update Map: no output folder for known house overlay.")
            return {}
        manifest = self.route6_update_map_load_manifest(build_if_missing=False, output_dir=out_path)
        if not manifest:
            self.route6_update_map_status_var.set(f"Route 6 Update Map: no layered map manifest under {out_path}")
            return {}
        records = self.route6_enable_known_house_coordinate_mode()
        polygon_artifact = {
            "schema": "route6_known_house_polygons_v1",
            "source": "operator_known_coordinates",
            "run_dir": str(out_path),
            "coordinate_frame": "unreal_cm",
            "house_count": len(records),
            "houses": self.route6_json_safe(records),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        polygon_path = out_path / "map" / "known_house_polygons.json"
        self.route6_write_json_artifact(polygon_path, polygon_artifact)
        updated_layers: List[Dict[str, Any]] = []
        for layer in manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []:
            layer_record = dict(layer) if isinstance(layer, dict) else {}
            metadata = self.route6_update_map_load_layer_metadata(layer_record)
            preview_path = Path(str(layer_record.get("occupancy_preview_path", "") or ""))
            if not metadata or not preview_path.is_file():
                updated_layers.append(layer_record)
                continue
            try:
                image = Image.open(preview_path).convert("RGB")
                image = self.route6_draw_known_house_overlay_on_image(image, metadata, records)
                overlay_path = preview_path.with_name("known_house_overlay.png")
                image.save(overlay_path)
                layer_record["known_house_overlay_preview_path"] = str(overlay_path)
                layer_record["known_house_polygon_count"] = len(records)
            except Exception as exc:
                layer_record["known_house_overlay_error"] = str(exc)
            updated_layers.append(layer_record)
        manifest["layers"] = updated_layers
        manifest["known_house_overlay"] = {
            "schema": "route6_known_house_overlay_v1",
            "source": "operator_known_coordinates",
            "house_count": len(records),
            "polygon_artifact_path": str(polygon_path),
            "updated_layer_count": len([layer for layer in updated_layers if str(layer.get("known_house_overlay_preview_path", "") or "")]),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path = self.route6_update_map_manifest_path(out_path)
        self.route6_write_json_artifact(manifest_path, manifest)
        self.llm_route6_state["route6_known_house_overlay"] = self.route6_json_safe(manifest["known_house_overlay"])
        self.llm_route6_state["route6_known_house_polygon_path"] = str(polygon_path)
        self.route6_write_state_artifact()
        self.route6_update_map_status_var.set(
            f"Route 6 Update Map: known houses drawn on {manifest['known_house_overlay']['updated_layer_count']} layers -> {out_path}"
        )
        return manifest

    def route6_update_map_layer_preview_path(self, layer_record: Dict[str, Any]) -> Path:
        layer = layer_record if isinstance(layer_record, dict) else {}
        for key in ("known_house_overlay_preview_path", "house_overlay_preview_path", "occupancy_preview_path"):
            path = Path(str(layer.get(key, "") or ""))
            if path.is_file():
                return path
        return Path(str(layer.get("occupancy_preview_path", "") or ""))

    def route6_build_known_house_navigation_plan(
        self,
        output_dir: Optional[Path] = None,
        *,
        target_house_id: str = "",
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        if out_path is None:
            out_path = self.route6_current_or_new_output_dir()
        records = self.route6_enable_known_house_coordinate_mode()
        config = self.route6_known_house_map_config()
        pose = self.route6_current_pose()
        candidates = route6_map_builder.rank_house_candidates(
            config,
            pose,
            house_states=self.route6_house_states(),
            standoff_cm=self.route6_float_param(self.llm_route6_standoff_cm_var, 850.0, min_value=0.0, max_value=10000.0),
            scan_z_cm=self.route6_float_param(self.llm_route6_scan_z_cm_var, 450.0, min_value=0.0, max_value=5000.0),
        )
        target_id = str(target_house_id or "").strip()
        if target_id.isdigit():
            target_id = f"{int(target_id):03d}"
        selected = next((item for item in candidates if str(item.get("house_id", "") or "") == target_id), {}) if target_id else {}
        if not selected:
            context = {
                "task_prompt": str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else ""),
                "requested_direction": self.route6_parse_task_direction(str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else "")),
                "current_pose": pose,
                "candidates": candidates,
            }
            selection = self.route6_select_llm_target_from_context(context)
            selected = selection.get("selected_candidate", {}) if isinstance(selection.get("selected_candidate", {}), dict) else {}
        house_id = str(selected.get("house_id", "") or target_id or "")
        manifest = self.route6_update_map_load_manifest(build_if_missing=False, output_dir=out_path)
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        layer_keys = [self._route6_update_map_layer_key(layer) for layer in layers if isinstance(layer, dict)] or ["z_050"]
        selected_layer = self.route6_choose_realtime_layer_key(layers, str(self.route6_update_map_layer_var.get() or ""))
        context = {
            "schema": "route6_known_house_navigation_context_v1",
            "task_prompt": str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else ""),
            "requested_direction": self.route6_parse_task_direction(str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else "")),
            "current_pose": pose,
            "available_layer_keys": layer_keys,
            "open_corridor_layers": layer_keys,
            "selected_layer_key": selected_layer,
            "selected_layer_z_cm": self.route6_layer_z_from_key(selected_layer),
            "candidates": self.route6_json_safe(candidates),
        }
        selection = {
            "schema": "route6_known_house_target_selection_v1",
            "source": "route5_style_known_coordinate_planner",
            "house_id": house_id,
            "selected_candidate": self.route6_json_safe(selected),
            "requested_direction": context["requested_direction"],
            "recommended_facade": str(selected.get("nearest_facade", "") or ""),
            "approach_layer_key": selected_layer,
            "fallback_used": False,
            "selection_reason": "operator-provided known house polygon coordinates",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.llm_route6_state["selected_house_id"] = house_id
        self.llm_route6_state["selected_candidate"] = self.route6_json_safe(selected)
        self.llm_route6_state["llm_target_selection"] = self.route6_json_safe(selection)
        navigation_target = self.route6_plan_llm_navigation_target(Path(out_path), selection, context)
        plan = {
            "schema": "route6_known_house_navigation_plan_v1",
            "source": "route5_style_known_coordinate_planner",
            "run_dir": str(out_path),
            "target_house_id": house_id,
            "known_house_count": len(records),
            "selected_candidate": self.route6_json_safe(selected),
            "target_pose_cm": self.route6_json_safe(navigation_target.get("target_pose_cm", {})),
            "navigation_target_path": str(Path(out_path) / "route6_llm_navigation_target.json"),
            "scan_policy": "use known polygon bbox to generate nearest facade scout point before NBV",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(Path(out_path) / "route6_known_house_navigation_plan.json", plan)
        self.llm_route6_state["route6_known_house_navigation_plan"] = self.route6_json_safe(plan)
        self.route6_write_state_artifact()
        self.llm_route6_navigation_target_var.set(
            f"Known coordinate nav: house={house_id or 'n/a'} "
            f"x={float(plan.get('target_pose_cm', {}).get('x', 0.0) or 0.0):.1f} "
            f"y={float(plan.get('target_pose_cm', {}).get('y', 0.0) or 0.0):.1f}"
        )
        return plan

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
    ) -> Dict[str, Any]:
        grid_info = self.route6_test_planner_load_selected_layer_grid(output_dir=output_dir, selected_layer_key=selected_layer_key)
        problems: List[str] = []
        details: Dict[str, Any] = {}
        house_hits = self.route6_test_planner_house_containing_point(point, target_house_id=target_house_id)
        if house_hits:
            problems.append("inside_house_bbox")
            details["inside_house_bbox"] = house_hits
        near = self.route6_test_planner_near_obstacle_report(point, grid_info, radius_cm=float(obstacle_radius_cm))
        if bool(near.get("problem", False)):
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
        if str(main_reset.get("reset_status", "")) == "ok" and isinstance(main_reset.get("new_point", {}), dict):
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
            if str(scan_reset.get("reset_status", "")) == "ok" and isinstance(scan_reset.get("new_point", {}), dict):
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
        pose = self.route6_current_pose()
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
            radius = max(3, factor * 2)
            draw.rectangle((obs[0] - radius, obs[1] - radius, obs[0] + radius, obs[1] + radius), fill=(170, 80, 230), outline=(0, 0, 0), width=max(1, factor))
            draw.text((obs[0] + radius + 2, obs[1] + radius), f"S{point.get('scan_index', '')}", fill=(0, 0, 0))
        return source

    def route6_direction_yaw_deg(self, direction: str, *, default: float = 0.0) -> float:
        text = str(direction or "").strip().lower()
        mapping = {
            "east": 0.0,
            "south": 90.0,
            "west": 180.0,
            "north": -90.0,
        }
        return float(mapping.get(text, default))

    def route6_capture_direction_sweep(
        self,
        session: Any,
        output_dir: Path,
        *,
        requested_direction: str = "",
        yaw_offsets: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        sweep_dir = out_path / "route6_visual_direction_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        prompt = str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else "")
        direction = str(requested_direction or self.route6_parse_task_direction(prompt) or "north").strip().lower()
        pose = self.route6_current_pose()
        base_yaw = self.route6_direction_yaw_deg(direction, default=float(pose.get("yaw", 0.0) or 0.0))
        offsets = list(yaw_offsets) if isinstance(yaw_offsets, list) and yaw_offsets else [0.0, -30.0, 30.0]
        labels = ["center", "left_30", "right_30"]
        sweep_rows: List[Dict[str, Any]] = []
        for index, offset in enumerate(offsets, start=1):
            yaw_deg = float(base_yaw) + float(offset)
            label = labels[index - 1] if index - 1 < len(labels) else f"offset_{int(offset):+d}"
            action_detail = {
                "schema": "route6_visual_direction_sweep_action_v1",
                "capture_kind": "visual_direction_sweep",
                "requested_direction": direction,
                "sweep_label": label,
                "frame_index": int(index),
                "target_yaw_deg": round(float(yaw_deg), 3),
                "yaw_offset_deg": round(float(offset), 3),
                "map_capture_required": False,
            }
            move_result: Dict[str, Any] = {}
            if session is not None and callable(getattr(session, "move_relative", None)):
                try:
                    move_result = session.move_relative(
                        {
                            "action_name": "route6_visual_direction_yaw",
                            "yaw_deg": float(yaw_deg),
                            "yaw_delta_deg": float(offset),
                            "forward_cm": 0.0,
                            "right_cm": 0.0,
                            "up_cm": 0.0,
                        }
                    )
                except Exception as exc:
                    move_result = {"status": "failed", "error": str(exc)}
            capture = {"status": "skipped", "reason": "missing_session"}
            if session is not None and callable(getattr(session, "capture_lidar_stream_frame", None)):
                try:
                    capture = session.capture_lidar_stream_frame(sweep_dir, int(index), action_detail=action_detail)
                    capture = capture if isinstance(capture, dict) else {"status": "failed", "error": "capture_returned_non_dict"}
                except Exception as exc:
                    capture = {"status": "failed", "error": str(exc)}
            capture_dir = Path(str(capture.get("capture_dir", sweep_dir / "frames" / f"frame_{index:06d}") or sweep_dir / "frames" / f"frame_{index:06d}"))
            capture_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = Path(str(capture.get("rgb_path", capture_dir / "rgb.png") or capture_dir / "rgb.png"))
            depth_path = Path(str(capture.get("depth_npy_path", capture_dir / "depth.npy") or capture_dir / "depth.npy"))
            camera_info_path = Path(str(capture.get("camera_info_path", capture_dir / "camera_info.json") or capture_dir / "camera_info.json"))
            if not rgb_path.is_file():
                Image.new("RGB", (48, 36), (96, 132, 168)).save(rgb_path)
            if not depth_path.is_file():
                np.save(depth_path, np.full((36, 48), float(pose.get("z", 300.0) or 300.0), dtype=np.float32))
            if not camera_info_path.is_file():
                self.route6_write_json_artifact(
                    camera_info_path,
                    {
                        "schema": "route6_visual_camera_info_v1",
                        "width": 48,
                        "height": 36,
                        "yaw_deg": float(yaw_deg),
                        "pose_cm": self.route6_json_safe(pose),
                    },
                )
            pose_path = capture_dir / "pose.json"
            self.route6_write_json_artifact(
                pose_path,
                {
                    "schema": "route6_visual_direction_pose_v1",
                    "pose_cm": self.route6_json_safe({**pose, "yaw": float(yaw_deg)}),
                    "requested_direction": direction,
                    "sweep_label": label,
                },
            )
            sweep_rows.append(
                {
                    "frame_id": f"sweep_{index:03d}",
                    "frame_index": int(index),
                    "label": label,
                    "requested_direction": direction,
                    "yaw_deg": round(float(yaw_deg), 3),
                    "yaw_offset_deg": round(float(offset), 3),
                    "capture_dir": str(capture_dir),
                    "rgb_path": str(rgb_path),
                    "depth_npy_path": str(depth_path),
                    "camera_info_path": str(camera_info_path),
                    "pose_path": str(pose_path),
                    "capture_status": str(capture.get("capture_status", capture.get("status", "")) or ""),
                    "move_result": self.route6_json_safe(move_result),
                }
            )
        manifest = {
            "schema": "route6_visual_direction_sweep_manifest_v1",
            "run_dir": str(out_path),
            "sweep_dir": str(sweep_dir),
            "task_prompt": prompt,
            "requested_direction": direction,
            "base_yaw_deg": round(float(base_yaw), 3),
            "current_pose_cm": self.route6_json_safe(pose),
            "capture_policy": {
                "map_capture_required": False,
                "purpose": "visual house-vs-fence target judgement before map approach",
            },
            "sweep_yaws": sweep_rows,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        manifest_path = sweep_dir / "sweep_manifest.json"
        self.route6_write_json_artifact(manifest_path, manifest)
        self.llm_route6_state["route6_visual_direction_sweep"] = self.route6_json_safe(manifest)
        self.llm_route6_state["route6_visual_direction_sweep_path"] = str(manifest_path)
        self.llm_route6_visual_status_var.set(f"LLM Visual Direction Analysis: captured {len(sweep_rows)} views toward {direction}")
        self.llm_route6_visual_detail_var.set(f"Visual sweep: {manifest_path}")
        self.route6_write_state_artifact()
        return manifest

    def route6_build_visual_direction_context(
        self,
        output_dir: Path,
        sweep_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        sweep = sweep_manifest if isinstance(sweep_manifest, dict) and sweep_manifest else self.route6_load_json_artifact(
            out_path / "route6_visual_direction_sweep" / "sweep_manifest.json",
            {},
        )
        planning = self.route6_build_realtime_map_planning_context()
        requested_direction = str(sweep.get("requested_direction", planning.get("requested_direction", "")) or "north").strip().lower()
        return {
            "schema": "route6_visual_direction_context_v1",
            "run_dir": str(out_path),
            "task_prompt": str(planning.get("task_prompt", "") or ""),
            "requested_direction": requested_direction,
            "current_pose": self.route6_json_safe(planning.get("current_pose", self.route6_current_pose())),
            "sweep_manifest_path": str(out_path / "route6_visual_direction_sweep" / "sweep_manifest.json"),
            "sweep_yaws": self.route6_json_safe(sweep.get("sweep_yaws", []) if isinstance(sweep.get("sweep_yaws", []), list) else []),
            "realtime_map_context": self.route6_json_safe(planning),
            "candidates": self.route6_json_safe(planning.get("candidates", [])),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }

    def route6_select_visual_house_target(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctx = context if isinstance(context, dict) else {}
        planning = ctx.get("realtime_map_context", {}) if isinstance(ctx.get("realtime_map_context", {}), dict) else {}
        if not planning:
            planning = {
                "task_prompt": ctx.get("task_prompt", ""),
                "requested_direction": ctx.get("requested_direction", ""),
                "current_pose": ctx.get("current_pose", self.route6_current_pose()),
                "candidates": ctx.get("candidates", []),
            }
        return self.route6_select_llm_target_from_context(planning)

    def route6_call_visual_house_llm(
        self,
        output_dir: Path,
        sweep_or_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        source = sweep_or_context if isinstance(sweep_or_context, dict) else {}
        context = source if str(source.get("schema", "") or "") == "route6_visual_direction_context_v1" else self.route6_build_visual_direction_context(out_path, source)
        selection = self.route6_select_visual_house_target(context)
        candidate = selection.get("selected_candidate", {}) if isinstance(selection.get("selected_candidate", {}), dict) else {}
        house_id = str(selection.get("house_id", candidate.get("house_id", "")) or "")
        sweep_rows = context.get("sweep_yaws", []) if isinstance(context.get("sweep_yaws", []), list) else []
        selected_frame = min(sweep_rows, key=lambda item: abs(float((item if isinstance(item, dict) else {}).get("yaw_offset_deg", 0.0) or 0.0))) if sweep_rows else {}
        requested_direction = str(context.get("requested_direction", selection.get("requested_direction", "")) or "").strip().lower()
        judgement = {
            "schema": "route6_visual_house_llm_judgement_v1",
            "source": "deterministic_visual_house_fallback",
            "run_dir": str(out_path),
            "task_prompt": str(context.get("task_prompt", "") or ""),
            "requested_direction": requested_direction,
            "target_visible": bool(house_id),
            "semantic_label": "house" if house_id else "unknown",
            "house_id": house_id,
            "selected_candidate": self.route6_json_safe(candidate),
            "selected_frame_id": str((selected_frame if isinstance(selected_frame, dict) else {}).get("frame_id", "") or ""),
            "selected_rgb_path": str((selected_frame if isinstance(selected_frame, dict) else {}).get("rgb_path", "") or ""),
            "selected_camera_yaw_deg": float((selected_frame if isinstance(selected_frame, dict) else {}).get("yaw_deg", self.route6_direction_yaw_deg(requested_direction, default=0.0)) or 0.0),
            "direction_relation": selection.get("direction_relation", []),
            "visual_confidence": round(float(selection.get("confidence", 0.55) or 0.55), 4) if house_id else 0.0,
            "image_region": {"bbox_norm": [0.35, 0.25, 0.65, 0.75], "source": "fallback_center_region"},
            "why_house_not_fence": (
                "selected target comes from map_config/realtime layered-map house candidates and satisfies the requested visual direction"
                if house_id else "no visually supported house candidate was available"
            ),
            "rejected_visual_objects": [
                {"semantic_label": "fence_or_rail", "reason": "linear/thin obstacle candidates are not accepted as house targets"}
            ],
            "recommended_action": "mark_visual_house_and_plan_approach" if house_id else "capture_more_direction_views",
            "needs_more_views": not bool(house_id),
            "risk_notes": selection.get("risk_notes", []),
            "fallback_used": True,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        judgement_path = out_path / "route6_visual_house_llm_judgement.json"
        self.route6_write_json_artifact(judgement_path, judgement)
        self.llm_route6_state["route6_visual_house_llm_judgement"] = self.route6_json_safe(judgement)
        self.llm_route6_state["route6_visual_house_llm_judgement_path"] = str(judgement_path)
        self.llm_route6_visual_status_var.set(f"LLM Visual Direction Analysis: target={house_id or 'n/a'} direction={requested_direction or 'any'}")
        self.llm_route6_visual_detail_var.set(
            f"Visual judgement: label={judgement['semantic_label']} confidence={judgement['visual_confidence']:.2f} frame={judgement['selected_frame_id'] or 'n/a'}"
        )
        self.route6_write_state_artifact()
        return judgement

    def route6_build_visual_marker_overlay(self, output_dir: Path, marker: Dict[str, Any]) -> Path:
        out_path = Path(output_dir)
        overlay_path = out_path / "route6_visual_house_marker_overlay.png"
        size = 720
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)
        bounds = (-5000.0, 5000.0, -5000.0, 5000.0)

        def project(x_cm: float, y_cm: float) -> Tuple[int, int]:
            min_x, max_x, min_y, max_y = bounds
            px = int(round((float(x_cm) - min_x) / max(1.0, max_x - min_x) * float(size - 1)))
            py = int(round((float(y_cm) - min_y) / max(1.0, max_y - min_y) * float(size - 1)))
            return max(0, min(size - 1, px)), max(0, min(size - 1, py))

        for value in range(-5000, 5001, 1000):
            x0, y0 = project(value, -5000.0)
            x1, y1 = project(value, 5000.0)
            draw.line((x0, y0, x1, y1), fill=(230, 230, 230), width=1)
            x0, y0 = project(-5000.0, value)
            x1, y1 = project(5000.0, value)
            draw.line((x0, y0, x1, y1), fill=(230, 230, 230), width=1)
        pose = self.route6_current_pose()
        ux, uy = project(float(pose.get("x", 0.0) or 0.0), float(pose.get("y", 0.0) or 0.0))
        center = marker.get("estimated_center_cm", {}) if isinstance(marker.get("estimated_center_cm", {}), dict) else {}
        tx, ty = project(float(center.get("x", 0.0) or 0.0), float(center.get("y", -1200.0) or -1200.0))
        yaw = float(pose.get("yaw", 0.0) or 0.0)
        dx, dy = self.route6_yaw_screen_vector(yaw)
        arrow_len = 32.0
        tip = (ux + dx * arrow_len, uy + dy * arrow_len)
        draw.line((ux, uy, tip[0], tip[1]), fill=(220, 0, 0), width=4)
        draw.polygon(
            [
                tip,
                (tip[0] - dx * 10.0 + dy * 7.0, tip[1] - dy * 10.0 - dx * 7.0),
                (tip[0] - dx * 10.0 - dy * 7.0, tip[1] - dy * 10.0 + dx * 7.0),
            ],
            fill=(220, 0, 0),
        )
        draw.ellipse((ux - 7, uy - 7, ux + 7, uy + 7), outline=(220, 0, 0), width=3)
        draw.line((ux, uy, tx, ty), fill=(40, 110, 220), width=2)
        draw.ellipse((tx - 12, ty - 12, tx + 12, ty + 12), outline=(40, 110, 220), width=4)
        draw.text((tx + 14, ty - 10), str(marker.get("object_id", "visual_house")), fill=(0, 0, 0))
        self.route6_draw_update_map_compass_overlay(draw, image, yaw, scale=2)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(overlay_path)
        return overlay_path

    def route6_mark_visual_house_candidate_on_map(
        self,
        output_dir: Path,
        judgement: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        item = judgement if isinstance(judgement, dict) else {}
        candidate = item.get("selected_candidate", {}) if isinstance(item.get("selected_candidate", {}), dict) else {}
        house_id = str(item.get("house_id", candidate.get("house_id", "")) or "").strip()
        pose = self.route6_current_pose()
        try:
            cx = float(candidate.get("center_x", candidate.get("x", 0.0)) or 0.0)
            cy = float(candidate.get("center_y", candidate.get("y", 0.0)) or 0.0)
        except Exception:
            cx = 0.0
            cy = 0.0
        if abs(cx) < 1e-6 and abs(cy) < 1e-6:
            yaw = float(item.get("selected_camera_yaw_deg", self.route6_direction_yaw_deg(str(item.get("requested_direction", "north") or "north"))) or 0.0)
            dx, dy = self.route6_yaw_screen_vector(yaw)
            cx = float(pose.get("x", 0.0) or 0.0) + dx * 1200.0
            cy = float(pose.get("y", 0.0) or 0.0) + dy * 1200.0
        object_id = f"visual_house_{house_id or '0001'}"
        selected_layer = str(self.route6_update_map_layer_var.get() or "z_050")
        marker = {
            "schema": "route6_visual_house_marker_v1",
            "object_id": object_id,
            "semantic_label": "house_candidate",
            "house_id": house_id,
            "requested_direction": str(item.get("requested_direction", "") or ""),
            "estimated_center_cm": {"x": round(float(cx), 2), "y": round(float(cy), 2)},
            "bearing_deg": float(item.get("selected_camera_yaw_deg", self.route6_direction_yaw_deg(str(item.get("requested_direction", "north") or "north"))) or 0.0),
            "supporting_layers": [selected_layer],
            "source_judgement_path": str(out_path / "route6_visual_house_llm_judgement.json"),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        overlay_path = self.route6_build_visual_marker_overlay(out_path, marker)
        marker["overlay_preview_path"] = str(overlay_path)
        marker_path = out_path / "route6_visual_house_marker.json"
        self.route6_write_json_artifact(marker_path, marker)
        self.llm_route6_state["route6_visual_house_marker"] = self.route6_json_safe(marker)
        self.llm_route6_state["route6_visual_house_marker_path"] = str(marker_path)
        self.llm_route6_visual_detail_var.set(
            f"Visual marker: {object_id} x={marker['estimated_center_cm']['x']:.1f} y={marker['estimated_center_cm']['y']:.1f}"
        )
        self.route6_write_state_artifact()
        return marker

    def route6_detect_height_conflict(
        self,
        output_dir: Path,
        target: Dict[str, Any],
        layer_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        tgt = target if isinstance(target, dict) else {}
        state = layer_state if isinstance(layer_state, dict) else {}
        approach_layer = str(tgt.get("approach_layer_key", self.route6_update_map_layer_var.get() or "z_050") or "z_050")
        blocked_layers = sorted({str(key) for key in state.get("blocked_layers", []) if str(key).strip()}, key=self.route6_layer_z_from_key)
        open_layers = sorted({str(key) for key in state.get("open_layers", tgt.get("open_corridor_layers", [])) if str(key).strip()}, key=self.route6_layer_z_from_key)
        or_state = state.get("or_state", {}) if isinstance(state.get("or_state", {}), dict) else {}
        risk_state = str(or_state.get("risk_state", "") or "").lower()
        blocked_z = [self.route6_layer_z_from_key(key) for key in blocked_layers]
        longest_run = 0
        current_run = 0
        previous = None
        for value in blocked_z:
            if previous is None or abs(value - previous) <= 50:
                current_run += 1
            else:
                current_run = 1
            longest_run = max(longest_run, current_run)
            previous = value
        if risk_state in {"must_stop", "collision", "collision_imminent"}:
            conflict_type = "local_or_blocked"
            decision = "hold_and_replan"
        elif not blocked_layers:
            conflict_type = "clear"
            decision = "continue"
        elif len(blocked_layers) == 1:
            conflict_type = "single_layer_conflict"
            decision = "change_layer"
        elif longest_run >= 3:
            conflict_type = "multi_layer_boundary"
            decision = "orbit_capture"
        else:
            conflict_type = "adjacent_layer_conflict"
            decision = "side_step_or_change_layer"
        report = {
            "schema": "route6_height_conflict_report_v1",
            "run_dir": str(out_path),
            "target_object_id": str(tgt.get("target_object_id", tgt.get("selected_object_id", "")) or ""),
            "house_id": str(tgt.get("house_id", "") or ""),
            "approach_layer_key": approach_layer,
            "blocked_layers": blocked_layers,
            "open_layers": open_layers,
            "or_state": self.route6_json_safe(or_state),
            "conflict_type": conflict_type,
            "decision": decision,
            "risk_summary": {
                "blocked_layer_count": len(blocked_layers),
                "longest_adjacent_blocked_run": int(longest_run),
                "risk_state": risk_state or "n/a",
            },
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.route6_write_json_artifact(out_path / "route6_height_conflict_report.json", report)
        self.llm_route6_state["route6_height_conflict_report"] = self.route6_json_safe(report)
        self.llm_route6_conflict_status_var.set(f"Height Conflict / Replan: {conflict_type} -> {decision}")
        self.llm_route6_conflict_detail_var.set(f"Blocked layers: {', '.join(blocked_layers) or 'none'}; open: {', '.join(open_layers) or 'n/a'}")
        self.route6_write_state_artifact()
        return report

    def route6_decide_height_conflict_avoidance(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_route6_state()
        item = conflict if isinstance(conflict, dict) else {}
        approach = str(item.get("approach_layer_key", "z_050") or "z_050")
        open_layers = [str(key) for key in item.get("open_layers", []) if str(key).strip()]
        conflict_type = str(item.get("conflict_type", "") or "")
        target_layer = ""
        next_action = "continue"
        if conflict_type == "single_layer_conflict":
            approach_z = self.route6_layer_z_from_key(approach)
            adjacent = [key for key in open_layers if abs(self.route6_layer_z_from_key(key) - approach_z) <= 50]
            target_layer = min(adjacent or open_layers, key=lambda key: abs(self.route6_layer_z_from_key(key) - approach_z)) if (adjacent or open_layers) else ""
            next_action = "change_layer" if target_layer else "hold_and_replan"
        elif conflict_type == "multi_layer_boundary":
            next_action = "orbit_capture"
        elif conflict_type in {"adjacent_layer_conflict", "local_or_blocked"}:
            next_action = "side_step_or_replan"
            target_layer = min(open_layers, key=lambda key: abs(self.route6_layer_z_from_key(key) - self.route6_layer_z_from_key(approach))) if open_layers else ""
        decision = {
            "schema": "route6_replan_llm_decision_v1",
            "source": "deterministic_height_conflict_replan",
            "conflict_type": conflict_type,
            "next_action": next_action,
            "target_layer_key": target_layer,
            "recommended_capture": "route6_building_orbit_capture" if next_action == "orbit_capture" else "",
            "reason": str(item.get("decision", "") or ""),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        out_dir = str(item.get("run_dir", "") or "")
        if out_dir:
            self.route6_write_json_artifact(Path(out_dir) / "route6_replan_llm_decision.json", decision)
        self.llm_route6_state["route6_replan_llm_decision"] = self.route6_json_safe(decision)
        self.llm_route6_conflict_detail_var.set(f"Replan: {next_action} layer={target_layer or 'n/a'}")
        self.route6_write_state_artifact()
        return decision

    def route6_call_replan_llm(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        return self.route6_decide_height_conflict_avoidance(conflict)

    def route6_plan_building_orbit_capture(
        self,
        output_dir: Path,
        marker: Dict[str, Any],
        conflict: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        item = marker if isinstance(marker, dict) else {}
        object_id = str(item.get("object_id", "visual_house_0001") or "visual_house_0001")
        object_dir = out_path / "route6_visual_orbit_captures" / f"object_{object_id}"
        object_dir.mkdir(parents=True, exist_ok=True)
        center = item.get("estimated_center_cm", {}) if isinstance(item.get("estimated_center_cm", {}), dict) else {}
        cx = float(center.get("x", 0.0) or 0.0)
        cy = float(center.get("y", -1200.0) or -1200.0)
        z_cm = float(self.route6_current_pose().get("z", self.route6_layer_z_from_key(str(self.route6_update_map_layer_var.get() or "z_300"))) or 300.0)
        standoff = self.route6_float_param(self.llm_route6_standoff_cm_var, 850.0, min_value=100.0, max_value=5000.0)
        viewpoints: List[Dict[str, Any]] = []
        for index, angle_deg in enumerate([0.0, 90.0, 180.0, 270.0]):
            angle = math.radians(angle_deg)
            x = cx + math.cos(angle) * standoff
            y = cy + math.sin(angle) * standoff
            yaw = math.degrees(math.atan2(cy - y, cx - x))
            viewpoints.append(
                {
                    "view_id": f"orbit_{index:03d}",
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "z": round(float(z_cm), 2),
                    "yaw_deg": round(float(yaw), 2),
                    "standoff_cm": round(float(standoff), 2),
                }
            )
        plan = {
            "schema": "route6_building_orbit_capture_manifest_v1",
            "run_dir": str(out_path),
            "object_id": object_id,
            "object_dir": str(object_dir),
            "marker": self.route6_json_safe(marker),
            "height_conflict": self.route6_json_safe(conflict),
            "planned_viewpoints": viewpoints,
            "capture_count": 0,
            "capture_status": "planned",
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.route6_write_json_artifact(object_dir / "orbit_manifest.json", plan)
        self.llm_route6_state["route6_building_orbit_capture_plan"] = self.route6_json_safe(plan)
        self.route6_write_state_artifact()
        return plan

    def route6_execute_building_orbit_capture(
        self,
        session: Any,
        output_dir: Path,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        self.route6_visual_orbit_stop_event.clear()
        out_path = Path(output_dir)
        payload = plan if isinstance(plan, dict) else {}
        object_dir = Path(str(payload.get("object_dir", out_path / "route6_visual_orbit_captures" / "object_visual_house_0001") or out_path))
        frames_dir = object_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        executed: List[Dict[str, Any]] = []
        for index, view in enumerate(payload.get("planned_viewpoints", []) if isinstance(payload.get("planned_viewpoints", []), list) else [], start=1):
            if self.route6_visual_orbit_stop_event.is_set() or not self.route6_movement_allowed():
                break
            if not isinstance(view, dict):
                continue
            view_id = str(view.get("view_id", f"orbit_{index - 1:03d}") or f"orbit_{index - 1:03d}")
            view_dir = frames_dir / view_id
            view_dir.mkdir(parents=True, exist_ok=True)
            move_result: Dict[str, Any] = {}
            if session is not None and callable(getattr(session, "move_relative", None)):
                try:
                    move_result = session.move_relative(
                        {
                            "action_name": "route6_orbit_viewpoint",
                            "x": float(view.get("x", 0.0) or 0.0),
                            "y": float(view.get("y", 0.0) or 0.0),
                            "z": float(view.get("z", 0.0) or 0.0),
                            "yaw_deg": float(view.get("yaw_deg", 0.0) or 0.0),
                        }
                    )
                except Exception as exc:
                    move_result = {"status": "failed", "error": str(exc)}
            capture = {"status": "skipped", "reason": "missing_session"}
            if session is not None and callable(getattr(session, "capture_lidar_stream_frame", None)):
                try:
                    capture = session.capture_lidar_stream_frame(
                        view_dir,
                        int(index),
                        action_detail={
                            "schema": "route6_building_orbit_capture_action_v1",
                            "capture_kind": "route6_building_orbit_capture",
                            "view_id": view_id,
                        },
                    )
                    capture = capture if isinstance(capture, dict) else {"status": "failed", "error": "capture_returned_non_dict"}
                except Exception as exc:
                    capture = {"status": "failed", "error": str(exc)}
            executed.append(
                {
                    "view_id": view_id,
                    "view_dir": str(view_dir),
                    "move_result": self.route6_json_safe(move_result),
                    "capture": self.route6_json_safe(capture),
                }
            )
        result = dict(payload)
        result["executed_viewpoints"] = executed
        result["capture_count"] = len(executed)
        result["capture_status"] = "stopped" if self.route6_visual_orbit_stop_event.is_set() else "executed"
        result["updated_at"] = datetime.now().isoformat(timespec="milliseconds")
        self.route6_write_json_artifact(object_dir / "orbit_manifest.json", result)
        self.llm_route6_state["route6_building_orbit_capture_manifest"] = self.route6_json_safe(result)
        self.llm_route6_conflict_status_var.set(f"Height Conflict / Replan: orbit capture {result['capture_status']} count={len(executed)}")
        self.route6_write_state_artifact()
        return result

    def route6_write_house_queue(self, output_dir: Path, candidates: List[Dict[str, Any]], selected: Dict[str, Any]) -> None:
        selected_house_id = str((selected or {}).get("house_id", "") or "")
        queue = {
            "schema": "route6_house_queue_v1",
            "selected_house_id": selected_house_id,
            "selected_candidate": selected,
            "llm_target_selection": self.llm_route6_state.get("llm_target_selection", {}),
            "candidates": candidates,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(Path(output_dir) / "route6_house_queue.json", queue)
        self.llm_route6_state["selected_house_id"] = selected_house_id
        self.llm_route6_state["selected_candidate"] = selected
        self.llm_route6_state["candidate_count"] = len(candidates)
        self.llm_route6_state["current_pose"] = self.route6_current_pose()
        self.llm_route6_state["stage"] = "SELECT_HOUSE" if selected else "DONE"
        self.llm_route6_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route6_stage_var.set(f"Stage: {self.llm_route6_state['stage']}")
        self.llm_route6_current_house_var.set(f"Current house: {selected_house_id or 'n/a'}")
        self.llm_route6_queue_var.set(f"House queue: candidates={len(candidates)}")
        self.route6_write_state_artifact()
        self.route6_log_event(
            Path(output_dir),
            "select_house",
            {
                "stage": self.llm_route6_state["stage"],
                "selected_house_id": selected_house_id,
                "candidate_count": len(candidates),
            },
        )

    def route6_select_next_house_for_mapping(self, output_dir: Path) -> Dict[str, Any]:
        candidates = self.route6_rank_mapping_house_candidates()
        context = self.route6_build_realtime_map_planning_context(candidates=candidates)
        selection = self.route6_select_llm_target_from_context(context)
        semantic_context = self.route6_build_semantic_map_context(candidates=candidates)
        semantic_selection = self.route6_select_llm_semantic_target(semantic_context, output_dir=Path(output_dir))
        semantic_house_id = str(semantic_selection.get("house_id", "") or "")
        if semantic_house_id:
            semantic_candidate = next((item for item in candidates if str(item.get("house_id", "") or "") == semantic_house_id), {})
            if semantic_candidate:
                selection["house_id"] = semantic_house_id
                selection["selected_candidate"] = semantic_candidate
                selection["semantic_target_selection"] = self.route6_json_safe(semantic_selection)
        selected = selection.get("selected_candidate", {}) if isinstance(selection.get("selected_candidate", {}), dict) else {}
        if not selected:
            selected = route6_map_builder.select_next_house_candidate(candidates)
            if selected:
                selection["house_id"] = str(selected.get("house_id", "") or "")
                selection["selected_candidate"] = selected
                selection["fallback_used"] = True
        self.route6_apply_selected_target_to_scan_plan(Path(output_dir), selection)
        self.route6_plan_llm_navigation_target(Path(output_dir), semantic_selection, semantic_context)
        self.route6_write_house_queue(Path(output_dir), candidates, selected)
        return selected

    def route6_wait_if_paused(self, output_dir: Path) -> bool:
        paused_logged = False
        while self.llm_route6_pause_event.is_set() and not self.llm_route6_stop_event.is_set():
            if not paused_logged:
                self.route6_set_stage("PAUSED", "Route 6 paused.")
                self.route6_log_event(Path(output_dir), "pause", {"stage": "PAUSED"})
                paused_logged = True
            time.sleep(0.1)
        return not self.llm_route6_stop_event.is_set()

    def route6_movement_allowed(self) -> bool:
        self.ensure_route6_state()
        if getattr(self, "route6_full_stop_event", None) is not None and self.route6_full_stop_event.is_set():
            return False
        if getattr(self, "llm_route6_stop_event", None) is not None and self.llm_route6_stop_event.is_set():
            return False
        return True

    def route6_apply_full_stop(
        self,
        session: Any = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is None:
            state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
            out_path = Path(state_output) if state_output else None
        if session is None:
            session = getattr(self, "session", None)
        self.llm_route6_state["stage"] = "FULL_STOP_REQUESTED"
        self.llm_route6_stage_var.set("Stage: FULL_STOP_REQUESTED")
        self.llm_route6_status_var.set("LLM Route V6: full stop requested; stopping movement chains.")
        if out_path is not None:
            self.route6_log_event(
                out_path,
                "full_stop_requested",
                {"stage": "FULL_STOP_REQUESTED", "created_at": datetime.now().isoformat(timespec="milliseconds")},
            )
        for event_name in (
            "route6_full_stop_event",
            "llm_route6_stop_event",
            "route6_update_map_realtime_stop_event",
            "route6_update_map_capture_stop_event",
            "active_nbv_stop_event",
            "llm_route5_stop_event",
            "llm_route4_stop_event",
            "llm_route3_stop_event",
            "route5_or2_monitor_stop_event",
        ):
            event = getattr(self, event_name, None)
            if hasattr(event, "set"):
                try:
                    event.set()
                except Exception:
                    pass
        try:
            self.llm_route6_pause_event.clear()
        except Exception:
            pass
        hold_result: Dict[str, Any] = {"status": "skipped", "reason": "missing_session"}
        movement_disable_result: Dict[str, Any] = {"status": "skipped", "reason": "missing_session"}
        if session is not None:
            if callable(getattr(session, "move_relative", None)):
                try:
                    hold_payload = action_payload("hold") if callable(globals().get("action_payload")) else {"action_name": "hold"}
                    hold_result = session.move_relative(hold_payload)
                except Exception as exc:
                    hold_result = {"status": "failed", "error": str(exc)}
            if callable(getattr(session, "set_movement_enabled", None)):
                try:
                    movement_disable_result = session.set_movement_enabled(False)
                except Exception as exc:
                    movement_disable_result = {"status": "failed", "error": str(exc)}
        self.movement_enabled_state = False
        try:
            if hasattr(self, "movement_enabled_var"):
                self.movement_enabled_var.set(False)
        except Exception:
            pass
        record = {
            "schema": "route6_full_stop_v1",
            "status": "full_stopped",
            "reason": "operator_full_stop",
            "hold_result": self.route6_json_safe(hold_result),
            "movement_disable_result": self.route6_json_safe(movement_disable_result),
            "events_set": [
                name
                for name in (
                    "route6_full_stop_event",
                    "llm_route6_stop_event",
                    "route6_update_map_realtime_stop_event",
                    "route6_update_map_capture_stop_event",
                    "active_nbv_stop_event",
                    "llm_route5_stop_event",
                    "llm_route4_stop_event",
                    "llm_route3_stop_event",
                    "route5_or2_monitor_stop_event",
                )
                if hasattr(getattr(self, name, None), "is_set") and getattr(self, name).is_set()
            ],
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.llm_route6_state["stage"] = "FULL_STOPPED"
        self.llm_route6_state["full_stop"] = self.route6_json_safe(record)
        self.llm_route6_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route6_stage_var.set("Stage: FULL_STOPPED")
        if session is None:
            self.llm_route6_status_var.set("LLM Route V6: FULL_STOPPED: no active session; all Route stop events set.")
        else:
            self.llm_route6_status_var.set("LLM Route V6: full stop applied; UAV hold sent and movement disabled.")
        if out_path is not None:
            self.route6_write_json_artifact(out_path / "route6_full_stop.json", record)
            self.route6_write_state_artifact()
            self.route6_log_event(out_path, "full_stop_applied", record)
        self.route6_update_summary_text()
        return record

    def route6_log_event(self, output_dir: Path, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "event_type": str(event_type),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            **(payload if isinstance(payload, dict) else {}),
        }
        self.route6_append_jsonl(Path(output_dir) / "route6_events.jsonl", record)

    def route6_write_state_artifact(self) -> None:
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        output_dir = state.get("output_dir")
        if not output_dir:
            return
        self.route6_write_json_artifact(Path(str(output_dir)) / "route6_state.json", state)

    def route6_house_output_dir(self, output_dir: Path, house_id: str) -> Path:
        house_dir = Path(output_dir) / "houses" / f"house_{str(house_id or '').strip()}"
        (house_dir / "pointcloud").mkdir(parents=True, exist_ok=True)
        (house_dir / "map").mkdir(parents=True, exist_ok=True)
        (house_dir / "entrance").mkdir(parents=True, exist_ok=True)
        return house_dir

    def route6_initialize_run(self, *, force_new: bool = False) -> Path:
        self.ensure_route6_state()
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        prior_house_states = self.route6_house_states()
        preserved_update_map_state = {
            key: state[key]
            for key in (
                "route6_update_map_output_dir",
                "route6_update_map_output_source",
                "route6_update_map_realtime",
                "route6_update_map_capture",
                "route6_update_map",
                "route6_update_map_pointcloud_merge",
                "route6_update_map_capture_frame_index",
                "route6_target_search_realtime_map",
                "route6_movement_preflight",
                "route6_known_house_coordinate_mode_enabled",
                "route6_known_house_polygons",
                "route6_known_house_navigation_plan",
            )
            if key in state
        }
        output_dir = Path(str(state.get("output_dir", ""))) if state.get("output_dir") else None
        if force_new or output_dir is None or not output_dir.exists():
            output_dir = self.make_route6_output_dir()
        base_config = getattr(self, "map_config", {}) if isinstance(getattr(self, "map_config", {}), dict) else {}
        try:
            self.route6_runtime_map_config = json.loads(json.dumps(base_config))
        except Exception:
            self.route6_runtime_map_config = dict(base_config)
        self.llm_route6_state = {
            "schema": "route6_state_v1",
            "mode": "route6_nearest_house_pointcloud_map",
            "stage": "INIT_RUN",
            "output_dir": str(output_dir),
            "design_doc": str(self.route6_design_doc_path()),
            "house_states": prior_house_states,
            "processed_house_ids": list(state.get("processed_house_ids", [])) if isinstance(state.get("processed_house_ids", []), list) and not force_new else [],
            "created_at": state.get("created_at", datetime.now().isoformat(timespec="seconds")),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **preserved_update_map_state,
        }
        self.route6_write_json_artifact(output_dir / "route6_state.json", self.llm_route6_state)
        self.route6_set_stage("INIT_RUN", "initializing Route 6 nearest-house exploration.")
        self.route6_set_stage("LOAD_MAP", "using current map config for Route 6 ranking.")
        self.route6_set_stage("RANK_HOUSES", "ranking nearest reachable houses.")
        candidates = self.route6_rank_mapping_house_candidates()
        context = self.route6_build_realtime_map_planning_context(candidates=candidates)
        target_selection = self.route6_select_llm_target_from_context(context)
        semantic_context = self.route6_build_semantic_map_context(candidates=candidates)
        semantic_selection = self.route6_select_llm_semantic_target(semantic_context, output_dir=output_dir)
        semantic_house_id = str(semantic_selection.get("house_id", "") or "")
        if semantic_house_id:
            semantic_candidate = next((item for item in candidates if str(item.get("house_id", "") or "") == semantic_house_id), {})
            if semantic_candidate:
                target_selection["house_id"] = semantic_house_id
                target_selection["selected_candidate"] = semantic_candidate
                target_selection["selection_reason"] = str(semantic_selection.get("why_this_is_first_house", target_selection.get("selection_reason", "")) or "")
                target_selection["semantic_target_selection"] = self.route6_json_safe(semantic_selection)
        selected = target_selection.get("selected_candidate", {}) if isinstance(target_selection.get("selected_candidate", {}), dict) else {}
        if not selected:
            selected = route6_map_builder.select_next_house_candidate(candidates)
            if selected:
                target_selection["house_id"] = str(selected.get("house_id", "") or "")
                target_selection["selected_candidate"] = selected
                target_selection["fallback_used"] = True
        selected_house_id = str(selected.get("house_id", "") or "")
        self.llm_route6_state.update({
            "stage": "SELECT_HOUSE" if selected else "FAILED",
            "current_pose": self.route6_current_pose(),
            "selected_house_id": selected_house_id,
            "selected_candidate": selected,
            "llm_target_selection": target_selection,
            "candidate_count": len(candidates),
            "max_houses": self.llm_route6_max_houses_var.get(),
            "runtime_minutes": self.llm_route6_runtime_min_var.get(),
            "standoff_cm": self.llm_route6_standoff_cm_var.get(),
            "scan_z_cm": self.llm_route6_scan_z_cm_var.get(),
            "occupancy_resolution_m": self.llm_route6_occupancy_resolution_m_var.get(),
            "coverage_threshold": self.llm_route6_coverage_threshold_var.get(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        self.route6_write_json_artifact(output_dir / "route6_state.json", self.llm_route6_state)
        self.route6_write_house_queue(output_dir, candidates, selected)
        self.route6_apply_selected_target_to_scan_plan(output_dir, target_selection)
        self.route6_plan_llm_navigation_target(output_dir, semantic_selection, semantic_context)
        self.route6_set_stage("SELECT_HOUSE" if selected else "FAILED", f"selected house={selected_house_id or 'n/a'} from candidates={len(candidates)}.")
        self.route6_log_event(
            output_dir,
            "init_run",
            {
                "stage": self.llm_route6_state["stage"],
                "selected_house_id": selected_house_id,
                "candidate_count": len(candidates),
            },
        )
        self.llm_route6_stage_var.set(f"Stage: {self.llm_route6_state['stage']}")
        self.llm_route6_current_house_var.set(f"Current house: {selected_house_id or 'n/a'}")
        self.llm_route6_queue_var.set(f"House queue: candidates={len(candidates)}")
        self.llm_route6_output_dir_var.set(f"Output: {output_dir}")
        self.llm_route6_map_status_var.set("Map: waiting for pointcloud artifacts")
        return output_dir

    def route6_write_cumulative_polygon_artifact(self, output_dir: Path, house_id: str, polygon: Dict[str, Any]) -> Path:
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        by_house = self.llm_route6_state.setdefault("map_polygons_by_house", {})
        if not isinstance(by_house, dict):
            by_house = {}
            self.llm_route6_state["map_polygons_by_house"] = by_house
        if hid and isinstance(polygon, dict):
            by_house[hid] = self.route6_json_safe(polygon)
        polygons = [
            item
            for _hid, item in sorted(by_house.items(), key=lambda pair: str(pair[0]))
            if isinstance(item, dict)
        ]
        path = out_path / "map" / "route6_polygons.json"
        self.route6_write_json_artifact(
            path,
            {
                "schema": "route6_polygons_v1",
                "polygons": polygons,
                "polygon_count": len(polygons),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self.route6_write_state_artifact()
        return path

    def route6_write_cumulative_corrected_config(
        self,
        output_dir: Path,
        house_id: str,
        corrected_house: Dict[str, Any],
    ) -> Path:
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        records = self.llm_route6_state.setdefault("corrected_house_records_by_id", {})
        if not isinstance(records, dict):
            records = {}
            self.llm_route6_state["corrected_house_records_by_id"] = records
        if hid and isinstance(corrected_house, dict) and corrected_house:
            records[hid] = self.route6_json_safe(corrected_house)
        base_config = getattr(self, "map_config", {}) if isinstance(getattr(self, "map_config", {}), dict) else {}
        try:
            cumulative = json.loads(json.dumps(base_config))
        except Exception:
            cumulative = dict(base_config)
        houses = cumulative.get("houses", []) if isinstance(cumulative.get("houses"), list) else []
        for house in houses:
            if not isinstance(house, dict):
                continue
            record = records.get(str(house.get("id", house.get("house_id", "")) or "").strip())
            if isinstance(record, dict):
                house.update(record)
        path = out_path / "map" / "route6_corrected_houses_config.json"
        self.route6_write_json_artifact(path, cumulative)
        self.route6_runtime_map_config = cumulative
        self.route6_write_state_artifact()
        return path

    def route6_write_cumulative_global_occupancy(self, output_dir: Path) -> Dict[str, Any]:
        out_path = Path(output_dir)
        map_dir = out_path / "map"
        map_dir.mkdir(parents=True, exist_ok=True)
        states = self.route6_house_states()
        clouds: List[np.ndarray] = []
        house_ids: List[str] = []
        source_paths: List[str] = []
        for hid, state in sorted(states.items(), key=lambda pair: str(pair[0])):
            item = state if isinstance(state, dict) else {}
            status = str(item.get("status", "") or "")
            if status not in {"mapped_complete", "mapped_partial", "searched", "searched_no_entry"}:
                continue
            pointcloud_path = Path(str(item.get("merged_pointcloud_path", "") or ""))
            if not pointcloud_path.is_file():
                continue
            try:
                cloud = np.load(pointcloud_path)
            except Exception as exc:
                self.route6_log_event(out_path, "global_occupancy_warning", {"house_id": str(hid), "path": str(pointcloud_path), "error": str(exc)})
                continue
            cloud = np.asarray(cloud)
            if cloud.ndim != 2 or cloud.shape[0] <= 0 or cloud.shape[1] < 3:
                continue
            clouds.append(cloud.astype(np.float32, copy=False))
            house_ids.append(str(hid))
            source_paths.append(str(pointcloud_path))
        if not clouds:
            self.llm_route6_state["global_occupancy"] = {}
            self.route6_write_state_artifact()
            return {}
        raw_point_count = int(sum(int(cloud.shape[0]) for cloud in clouds))
        resolution_m = self.route6_float_param(self.llm_route6_occupancy_resolution_m_var, 0.25, min_value=0.05, max_value=5.0)
        merged = route6_map_builder.voxel_downsample_point_cloud(
            np.concatenate(clouds, axis=0),
            voxel_size_m=float(resolution_m),
            fixed_world_bounds_cm=route6_map_builder.DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
            max_points=route6_map_builder.DEFAULT_ROUTE6_UPDATE_MAP_MAX_POINTS,
        )
        max_points_global = int(route6_map_builder.DEFAULT_ROUTE6_UPDATE_MAP_MAX_POINTS)
        occupancy = route6_map_builder.build_occupancy_grid(
            merged,
            resolution_m=float(resolution_m),
            occupied_threshold=1,
            fixed_world_bounds_cm=route6_map_builder.DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
        )
        grid_path = map_dir / "route6_occupancy_grid.npy"
        metadata_path = map_dir / "route6_occupancy_grid.json"
        preview_path = map_dir / "route6_occupancy_grid.png"
        np.save(grid_path, np.asarray(occupancy["grid"], dtype=np.int16))
        cv2.imwrite(str(preview_path), route6_map_builder.occupancy_preview_image(occupancy))
        metadata = route6_map_builder.occupancy_metadata(occupancy)
        metadata.update({
            "schema": "route6_global_occupancy_grid_v1",
            "scope": "run_global",
            "house_ids": house_ids,
            "house_count": len(house_ids),
            "source_pointcloud_count": len(source_paths),
            "source_pointcloud_paths": source_paths,
            "raw_point_count": int(raw_point_count),
            "merged_point_count": int(merged.shape[0]),
            "max_points_global": int(max_points_global),
            "pointcloud_voxel_size_m": float(resolution_m),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        self.route6_write_json_artifact(metadata_path, metadata)
        record = {
            "schema": metadata["schema"],
            "scope": metadata["scope"],
            "house_ids": house_ids,
            "source_pointcloud_count": len(source_paths),
            "raw_point_count": int(raw_point_count),
            "merged_point_count": int(merged.shape[0]),
            "pointcloud_voxel_size_m": float(resolution_m),
            "occupied_cell_count": int(metadata.get("occupied_cell_count", 0) or 0),
            "grid_path": str(grid_path),
            "metadata_path": str(metadata_path),
            "preview_path": str(preview_path),
            "updated_at": metadata["updated_at"],
        }
        self.llm_route6_state["global_occupancy"] = record
        try:
            layered = route6_map_builder.write_route6_layered_occupancy_artifacts(
                out_path,
                merged,
                layer_z_cm=route6_map_builder.DEFAULT_ROUTE6_LAYER_Z_CM,
                layer_band_cm=route6_map_builder.DEFAULT_ROUTE6_LAYER_BAND_CM,
                resolution_m=float(resolution_m),
                occupied_threshold=route6_map_builder.DEFAULT_ROUTE6_LAYER_OCCUPIED_THRESHOLD,
            )
            record["layered_occupancy_dir"] = str(layered.get("layered_occupancy_dir", "") or "")
            record["layered_manifest_path"] = str(layered.get("manifest_path", "") or "")
            record["layer_count"] = int(layered.get("layer_count", 0) or 0)
            self.llm_route6_state["route6_update_map"] = {
                "schema": "route6_update_map_state_v1",
                "source": "route6_global_occupancy",
                "manifest_path": record["layered_manifest_path"],
                "layered_occupancy_dir": record["layered_occupancy_dir"],
                "layer_count": record["layer_count"],
                "source_pointcloud_count": len(source_paths),
                "source_pointcloud_paths": source_paths,
                "raw_point_count": int(raw_point_count),
                "merged_point_count": int(merged.shape[0]),
                "pointcloud_voxel_size_m": float(resolution_m),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as exc:
            self.route6_log_event(out_path, "route6_update_map_warning", {"error": str(exc)})
        self.route6_log_event(out_path, "global_occupancy_updated", record)
        self.route6_write_state_artifact()
        return record

    def route6_plan_selected_house_scan_points(self, output_dir: Path, house_id: str) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        hid = str(house_id or "").strip()
        points: List[Dict[str, Any]] = []
        selected = self.llm_route6_state.get("selected_candidate", {}) if isinstance(self.llm_route6_state.get("selected_candidate"), dict) else {}
        pose = dict(selected.get("nearest_scan_pose", {}) if isinstance(selected.get("nearest_scan_pose"), dict) else {})
        scout_point: Dict[str, Any] = {}
        if hid and pose:
            facade = str(pose.get("facade", selected.get("nearest_facade", "west")) or "west")
            scout_point = {
                "scan_id": f"{hid}_{facade}_route6_nearest_scout_000",
                "house_id": hid,
                "facade": facade,
                "facade_id": f"{hid}_{facade}",
                "x": float(pose.get("x", 0.0) or 0.0),
                "y": float(pose.get("y", 0.0) or 0.0),
                "z": float(pose.get("z", self.llm_route6_scan_z_cm_var.get() or 450.0) or 450.0),
                "yaw_deg": float(pose.get("yaw_deg", 0.0) or 0.0),
                "standoff_cm": float(pose.get("standoff_cm", self.llm_route6_standoff_cm_var.get() or 850.0) or 850.0),
                "capture_trigger": "arrive_align_hover_capture",
                "status": "planned",
                "view_type": "route6_nearest_facade_scout",
                "route6_source": "nearest_candidate_scout",
            }
        if hid and callable(getattr(self, "active_nbv_initial_scan_points", None)):
            try:
                points = [
                    dict(point)
                    for point in self.active_nbv_initial_scan_points(hid)
                    if isinstance(point, dict)
                ]
            except Exception as exc:
                self.route6_log_event(Path(output_dir), "scan_point_plan_warning", {"house_id": hid, "error": str(exc)})
                points = []
        if scout_point:
            def _same_scan_pose(point: Dict[str, Any]) -> bool:
                try:
                    return (
                        str(point.get("facade", "") or "") == str(scout_point.get("facade", "") or "")
                        and abs(float(point.get("x", 0.0)) - float(scout_point.get("x", 0.0))) <= 1.0
                        and abs(float(point.get("y", 0.0)) - float(scout_point.get("y", 0.0))) <= 1.0
                    )
                except Exception:
                    return False

            if not any(_same_scan_pose(point) for point in points if isinstance(point, dict)):
                points = [scout_point] + points
            elif not points:
                points = [scout_point]
        try:
            scan_point_limit = int(float(getattr(self, "llm_route6_scan_point_limit_override", 0) or 0))
        except Exception:
            scan_point_limit = 0
        if scan_point_limit > 0:
            points = points[:scan_point_limit]
        house_dir = self.route6_house_output_dir(Path(output_dir), hid)
        payload = {
            "schema": "route6_scan_points_v1",
            "target_house_id": hid,
            "scan_points": points,
            "planned_scan_count": len(points),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(house_dir / "scan_points.json", payload)
        self.route6_write_json_artifact(Path(output_dir) / "scan_points.json", payload)
        self.llm_route6_state["scan_points"] = points
        self.llm_route6_state["planned_scan_count"] = len(points)
        self.route6_write_state_artifact()
        self.route6_log_event(Path(output_dir), "scan_points_planned", {"house_id": hid, "planned_scan_count": len(points)})
        return points

    def route6_read_lidar_rows(self, output_dir: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        candidates = [Path(output_dir) / "lidar_capture_log.jsonl"]
        candidates.extend(sorted((Path(output_dir) / "facade_observations").glob("*/lidar_capture_log.jsonl")))
        candidates.extend(sorted((Path(output_dir) / "houses").glob("house_*/lidar_capture_log.jsonl")))
        for path in candidates:
            if not path.is_file():
                continue
            if callable(getattr(self, "read_jsonl_artifact", None)):
                rows.extend(self.read_jsonl_artifact(path))
            else:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    text = line.strip()
                    if text:
                        rows.append(json.loads(text))
        return rows

    def route6_filter_rows_for_house(
        self,
        rows: List[Dict[str, Any]],
        house_id: str,
        scan_points: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        hid = str(house_id or "").strip()
        if not hid:
            return list(rows)
        scan_ids = {
            str(point.get("scan_id", "") or "").strip()
            for point in (scan_points or [])
            if isinstance(point, dict) and str(point.get("scan_id", "") or "").strip()
        }
        has_house_tags = any(str(row.get("house_id", row.get("target_house_id", "")) or "").strip() for row in rows if isinstance(row, dict))
        selected: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_house = str(row.get("house_id", row.get("target_house_id", "")) or "").strip()
            row_scan = str(row.get("scan_id", "") or "").strip()
            if has_house_tags:
                if row_house == hid:
                    selected.append(row)
            elif not scan_ids or row_scan in scan_ids:
                selected.append(row)
        return selected

    def route6_resolve_artifact_path(self, output_dir: Path, value: Any) -> Path:
        raw = str(value or "").strip()
        if not raw:
            return Path(output_dir) / ".route6_missing_artifact"
        path = Path(raw)
        if path.is_absolute():
            return path
        return (Path(output_dir) / path).resolve()

    def route6_build_entrance_capture_manifest(
        self,
        output_dir: Path,
        house_id: str,
        scan_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        rows = self.route6_filter_rows_for_house(self.route6_read_lidar_rows(out_path), hid, scan_points)
        included: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            item = dict(row if isinstance(row, dict) else {})
            capture_dir = self.route6_resolve_artifact_path(out_path, item.get("capture_dir", ""))
            rgb_path = self.route6_resolve_artifact_path(out_path, item.get("rgb_path", ""))
            depth_path = self.route6_resolve_artifact_path(out_path, item.get("depth_npy_path", item.get("depth_path", "")))
            camera_info_path = self.route6_resolve_artifact_path(out_path, item.get("camera_info_path", ""))
            if not str(item.get("rgb_path", "") or "").strip() and capture_dir.is_dir():
                rgb_path = capture_dir / "rgb.png"
            if not str(item.get("depth_npy_path", item.get("depth_path", "")) or "").strip() and capture_dir.is_dir():
                depth_path = capture_dir / "depth.npy"
            if not str(item.get("camera_info_path", "") or "").strip() and capture_dir.is_dir():
                camera_info_path = capture_dir / "camera_info.json"
            try:
                point_count = int(float(item.get("point_count", 0) or 0))
            except Exception:
                point_count = 0
            reason = ""
            if not str(item.get("scan_id", "") or "").strip():
                reason = "not_scan_capture"
            elif item.get("capture_guard_passed") is not True:
                reason = "capture_guard_failed_or_missing"
            elif point_count <= 0:
                reason = "point_count_not_positive"
            elif not capture_dir.is_dir():
                reason = "capture_dir_missing"
            elif not rgb_path.is_file():
                reason = "rgb_missing"
            elif not depth_path.is_file():
                reason = "depth_missing"
            elif not camera_info_path.is_file():
                reason = "camera_info_missing"
            manifest_row = {
                **self.route6_json_safe(item),
                "manifest_index": index,
                "capture_dir": str(capture_dir),
                "rgb_path": str(rgb_path),
                "depth_npy_path": str(depth_path),
                "camera_info_path": str(camera_info_path),
            }
            if reason:
                excluded.append({**manifest_row, "reason": reason})
            else:
                included.append(manifest_row)
        return {
            "schema": "route6_entrance_capture_manifest_v1",
            "run_dir": str(out_path),
            "house_id": hid,
            "included_count": len(included),
            "excluded_count": len(excluded),
            "included_captures": included,
            "excluded_captures": excluded,
            "route5_capture_analysis_available": bool(callable(getattr(self, "route5_run_capture_analysis", None))),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }

    def route6_load_json_artifact(self, path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            target = Path(path)
            if target.is_file():
                payload = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
        except Exception:
            pass
        return dict(default or {})

    def route6_build_close_confirm_scan_plan(
        self,
        output_dir: Path,
        house_id: str,
        candidates: List[Dict[str, Any]],
        house_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        house_dir = self.route6_house_output_dir(out_path, hid)
        entrance_dir = house_dir / "entrance"
        entrance_dir.mkdir(parents=True, exist_ok=True)
        state = house_state if isinstance(house_state, dict) else {}
        map_artifacts = state.get("map_artifacts", {}) if isinstance(state.get("map_artifacts", {}), dict) else {}
        polygon = map_artifacts.get("polygon", {}) if isinstance(map_artifacts.get("polygon", {}), dict) else {}
        bbox = polygon.get("bbox", {}) if isinstance(polygon.get("bbox", {}), dict) else {}
        map_config = self.route6_get_runtime_map_config()
        if not bbox:
            houses = map_config.get("houses", []) if isinstance(map_config.get("houses"), list) else []
            house = next(
                (item for item in houses if isinstance(item, dict) and str(item.get("id", item.get("house_id", ""))) == hid),
                None,
            )
            bbox = route6_map_builder.house_world_bbox(map_config, house) if isinstance(house, dict) else {}
        scan_points = self.llm_route6_state.get("scan_points", []) if isinstance(self.llm_route6_state.get("scan_points", []), list) else []
        selected = self.llm_route6_state.get("selected_candidate", {}) if isinstance(self.llm_route6_state.get("selected_candidate", {}), dict) else {}
        fallback_facade = str(selected.get("nearest_facade", "") or "")
        if not fallback_facade and scan_points:
            fallback_facade = str((scan_points[0] if isinstance(scan_points[0], dict) else {}).get("facade", "") or "")
        if fallback_facade not in route6_map_builder.ROUTE6_FACADES:
            fallback_facade = "west"
        base_standoff = self.route6_float_param(self.llm_route6_standoff_cm_var, 850.0, min_value=350.0, max_value=5000.0)
        confirm_standoff = max(350.0, min(float(base_standoff), float(base_standoff) * 0.65))
        scan_z = self.route6_float_param(self.llm_route6_scan_z_cm_var, 450.0, min_value=100.0, max_value=5000.0)
        valid_candidates = [item for item in candidates if isinstance(item, dict)]
        planned_points: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(valid_candidates[:3]):
            facade = str(candidate.get("facade", candidate.get("source_facade", fallback_facade)) or fallback_facade).lower()
            if facade not in route6_map_builder.ROUTE6_FACADES:
                facade = fallback_facade
            pose: Dict[str, Any] = {}
            if isinstance(bbox, dict) and all(key in bbox for key in ("min_x", "max_x", "min_y", "max_y")):
                pose = route6_map_builder.facade_scan_pose_for_bbox(
                    bbox,
                    facade,
                    current_pose=self.route6_current_pose(),
                    standoff_cm=confirm_standoff,
                    scan_z_cm=scan_z,
                )
            source_scan = scan_points[min(idx, len(scan_points) - 1)] if scan_points and isinstance(scan_points[min(idx, len(scan_points) - 1)], dict) else {}
            if not pose and source_scan:
                pose = {
                    "x": float(source_scan.get("x", 0.0) or 0.0),
                    "y": float(source_scan.get("y", 0.0) or 0.0),
                    "z": float(source_scan.get("z", scan_z) or scan_z),
                    "yaw_deg": float(source_scan.get("yaw_deg", 0.0) or 0.0),
                    "facade": facade,
                    "standoff_cm": confirm_standoff,
                }
            if not pose:
                continue
            planned_points.append({
                **self.route6_json_safe(pose),
                "scan_id": f"{hid}_{facade}_route6_close_confirm_{idx:03d}",
                "house_id": hid,
                "facade": facade,
                "candidate_index": idx,
                "candidate_class": str(candidate.get("class_name", candidate.get("class_name_normalized", "candidate")) or "candidate"),
                "candidate_confidence": candidate.get("confidence", ""),
                "source_frame_name": str(candidate.get("frame_name", candidate.get("image_name", "")) or ""),
                "source_candidate": self.route6_json_safe(candidate),
                "view_type": "route6_close_confirm_scan",
                "capture_trigger": "arrive_align_hover_capture",
                "status": "planned",
                "recommended_validator": "route5_capture_guard_plus_obstacle_validation",
            })
        payload = {
            "schema": "route6_close_confirm_scan_plan_v1",
            "house_id": hid,
            "candidate_count": len(valid_candidates),
            "planned_scan_count": len(planned_points),
            "scan_points": planned_points,
            "source": "route5_entrance_candidates",
            "obstacle_validation_status": "pending",
            "recommended_executor": "active_nbv_execute_scan_points",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(entrance_dir / "close_confirm_scan_plan.json", payload)
        return payload

    def route6_run_route5_capture_analysis_for_house(
        self,
        output_dir: Path,
        house_id: str,
        manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        included_count = int(manifest.get("included_count", 0) or 0)
        if included_count <= 0:
            return {"ran": False, "reason": "no_included_captures"}
        runner = getattr(self, "route5_run_capture_analysis", None)
        if not callable(runner):
            return {"ran": False, "reason": "route5_capture_analysis_unavailable"}
        out_path = Path(output_dir)
        try:
            try:
                summary = runner(out_path, stop_event=self.llm_route6_stop_event)
            except TypeError:
                summary = runner(out_path)
        except Exception as exc:
            return {"ran": False, "reason": "route5_capture_analysis_failed", "error": str(exc)}
        summary = summary if isinstance(summary, dict) else {}
        analysis_dir = Path(str(summary.get("analysis_dir", "") or out_path / "route5_capture_analysis"))
        if not analysis_dir.is_absolute():
            analysis_dir = (out_path / analysis_dir).resolve()
        candidates_path = analysis_dir / "entrance_candidates.json"
        candidates_payload = self.route6_load_json_artifact(candidates_path, {"candidate_count": 0, "candidates": []})
        candidates = candidates_payload.get("candidates", []) if isinstance(candidates_payload.get("candidates", []), list) else []
        candidate_count = int(candidates_payload.get("candidate_count", len(candidates)) or len(candidates))
        return {
            "ran": True,
            "house_id": str(house_id),
            "status": str(summary.get("status", "unknown") or "unknown"),
            "summary": self.route6_json_safe(summary),
            "analysis_dir": str(analysis_dir),
            "summary_path": str(analysis_dir / "analysis_summary.json"),
            "entrance_candidates_path": str(candidates_path),
            "candidate_count": candidate_count,
            "candidates": candidates,
        }

    def route6_write_exploration_summary(self, output_dir: Path) -> Path:
        out_path = Path(output_dir)
        path = out_path / "route6_exploration_summary.csv"
        fields = [
            "house_id",
            "status",
            "search_status",
            "entrance_status",
            "valid_scan_capture_count",
            "merged_point_count",
            "map_status",
            "map_confidence",
            "entrance_report_path",
            "coverage_report_path",
            "corrected_config_path",
        ]
        states = self.route6_house_states()
        rows: List[Dict[str, Any]] = []
        for hid, state in sorted(states.items(), key=lambda item: str(item[0])):
            item = state if isinstance(state, dict) else {}
            artifacts = item.get("map_artifacts", {}) if isinstance(item.get("map_artifacts", {}), dict) else {}
            quality = artifacts.get("quality_report", {}) if isinstance(artifacts.get("quality_report", {}), dict) else {}
            polygon = artifacts.get("polygon", {}) if isinstance(artifacts.get("polygon", {}), dict) else {}
            polygon_quality = polygon.get("quality", {}) if isinstance(polygon.get("quality", {}), dict) else {}
            confidence = item.get("map_confidence", quality.get("mean_map_confidence", polygon_quality.get("confidence", "")))
            rows.append({
                "house_id": str(hid),
                "status": str(item.get("status", "") or ""),
                "search_status": str(item.get("search_status", "") or ""),
                "entrance_status": str(item.get("entrance_status", "") or ""),
                "valid_scan_capture_count": int(item.get("valid_scan_capture_count", 0) or 0),
                "merged_point_count": int(item.get("merged_point_count", 0) or 0),
                "map_status": str(item.get("map_status", "") or ""),
                "map_confidence": confidence,
                "entrance_report_path": str(item.get("entrance_report_path", "") or ""),
                "coverage_report_path": str(item.get("coverage_report_path", "") or ""),
                "corrected_config_path": str(artifacts.get("corrected_config_path", "") or ""),
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def route6_write_run_quality_report(self, output_dir: Path) -> Dict[str, Any]:
        out_path = Path(output_dir)
        states = self.route6_house_states()
        map_config = getattr(self, "map_config", {}) if isinstance(getattr(self, "map_config", {}), dict) else {}
        houses = map_config.get("houses", []) if isinstance(map_config.get("houses"), list) else []
        total = len(houses)
        mapped_statuses = {"mapped_complete", "mapped_partial", "searched", "searched_no_entry"}
        searched_statuses = {"searched", "searched_no_entry"}
        blocked_statuses = {"blocked", "terminal_blocked"}
        occupied_cells = 0
        polygon_count = 0
        confidences: List[float] = []
        corrected_path = str(out_path / "map" / "route6_corrected_houses_config.json")
        warnings: List[str] = []
        for hid, state in states.items():
            item = state if isinstance(state, dict) else {}
            artifacts = item.get("map_artifacts", {}) if isinstance(item.get("map_artifacts", {}), dict) else {}
            quality = artifacts.get("quality_report", {}) if isinstance(artifacts.get("quality_report", {}), dict) else {}
            polygon = artifacts.get("polygon", {}) if isinstance(artifacts.get("polygon", {}), dict) else {}
            polygon_quality = polygon.get("quality", {}) if isinstance(polygon.get("quality", {}), dict) else {}
            occupied_cells += int(quality.get("global_occupied_cell_count", 0) or 0)
            polygon_count += int(quality.get("global_polygon_count", 1 if polygon.get("points") else 0) or 0)
            confidence_value = polygon_quality.get("confidence", quality.get("mean_map_confidence", item.get("map_confidence", None)))
            try:
                confidences.append(float(confidence_value))
            except Exception:
                pass
            if artifacts.get("corrected_config_path"):
                corrected_path = str(artifacts.get("corrected_config_path"))
            if item.get("blocked_reason"):
                warnings.append(f"house_{hid}: {item.get('blocked_reason')}")
        statuses = [str(state.get("status", "") or "") for state in states.values() if isinstance(state, dict)]
        global_occupancy = self.llm_route6_state.get("global_occupancy", {}) if isinstance(self.llm_route6_state.get("global_occupancy", {}), dict) else {}
        global_occupied_cells = occupied_cells
        try:
            if global_occupancy.get("occupied_cell_count", "") != "":
                global_occupied_cells = int(global_occupancy.get("occupied_cell_count", 0) or 0)
        except Exception:
            global_occupied_cells = occupied_cells
        quality_report = {
            "schema": "route6_map_quality_report_v1",
            "run_dir": str(out_path),
            "house_count_total": total,
            "house_count_mapped": len([status for status in statuses if status in mapped_statuses]),
            "house_count_searched": len([status for status in statuses if status in searched_statuses]),
            "house_count_blocked": len([status for status in statuses if status in blocked_statuses]),
            "house_count_needs_rescan": len([status for status in statuses if status == "needs_rescan"]),
            "global_occupied_cell_count": int(global_occupied_cells),
            "global_polygon_count": int(polygon_count),
            "mean_map_confidence": round(float(sum(confidences) / max(1, len(confidences))), 4) if confidences else 0.0,
            "corrected_config_path": corrected_path,
            "exploration_summary_path": str(out_path / "route6_exploration_summary.csv"),
            "global_occupancy_grid_path": str(global_occupancy.get("grid_path", "") or ""),
            "global_occupancy_metadata_path": str(global_occupancy.get("metadata_path", "") or ""),
            "global_occupancy_preview_path": str(global_occupancy.get("preview_path", "") or ""),
            "global_occupancy_house_ids": list(global_occupancy.get("house_ids", [])) if isinstance(global_occupancy.get("house_ids", []), list) else [],
            "global_occupancy_source_pointcloud_count": int(global_occupancy.get("source_pointcloud_count", 0) or 0),
            "global_occupancy_merged_point_count": int(global_occupancy.get("merged_point_count", 0) or 0),
            "warnings": warnings,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(out_path / "map" / "route6_map_quality_report.json", quality_report)
        return quality_report

    def route6_write_run_artifacts(self, output_dir: Path) -> Dict[str, Any]:
        out_path = Path(output_dir)
        summary_path = self.route6_write_exploration_summary(out_path)
        global_occupancy = self.route6_write_cumulative_global_occupancy(out_path)
        quality = self.route6_write_run_quality_report(out_path)
        self.llm_route6_state["exploration_summary_path"] = str(summary_path)
        self.llm_route6_state["run_quality_report"] = quality
        self.route6_write_state_artifact()
        return {"summary_path": str(summary_path), "quality_report": quality, "global_occupancy": global_occupancy}

    def route6_map_status_style(self, status: str) -> Dict[str, str]:
        status = str(status or "").strip().lower()
        colors = {
            "active": "#22d3ee",
            "captured": "#22c55e",
            "searched": "#22c55e",
            "searched_no_entry": "#94a3b8",
            "mapped_complete": "#22c55e",
            "mapped_partial": "#facc15",
            "needs_rescan": "#fb923c",
            "blocked": "#ef4444",
            "terminal_blocked": "#ef4444",
            "planned": "#facc15",
        }
        return {"color": colors.get(status, "#a78bfa"), "outline_color": "#111827"}

    def route6_map_route_points(self) -> List[Dict[str, Any]]:
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        points: List[Dict[str, Any]] = []
        selected = state.get("selected_candidate", {}) if isinstance(state.get("selected_candidate", {}), dict) else {}
        pose = selected.get("nearest_scan_pose", {}) if isinstance(selected.get("nearest_scan_pose", {}), dict) else {}
        if pose:
            points.append({
                **self.route6_json_safe(pose),
                "label": f"{selected.get('house_id', '')}_nearest",
                "route_point_type": "current_target",
                "status": "active",
                **self.route6_map_status_style("active"),
            })
        for idx, point in enumerate(state.get("scan_points", []) if isinstance(state.get("scan_points", []), list) else [], start=1):
            if not isinstance(point, dict):
                continue
            try:
                float(point.get("x"))
                float(point.get("y"))
            except Exception:
                continue
            status = str(point.get("status", "planned") or "planned").strip().lower()
            scan_id = str(point.get("scan_id", "") or f"route6_scan_{idx}")
            points.append({
                **self.route6_json_safe(point),
                "label": scan_id,
                "route_point_type": "scan_point",
                "status": status,
                "scan_id": scan_id,
                "facade": str(point.get("facade", "") or ""),
                **self.route6_map_status_style(status),
            })
        return points

    def route6_map_overlay_points(self) -> List[Dict[str, Any]]:
        states = self.route6_house_states()
        points: List[Dict[str, Any]] = []
        for hid, state in sorted(states.items(), key=lambda item: str(item[0])):
            item = state if isinstance(state, dict) else {}
            artifacts = item.get("map_artifacts", {}) if isinstance(item.get("map_artifacts", {}), dict) else {}
            polygon = artifacts.get("polygon", {}) if isinstance(artifacts.get("polygon", {}), dict) else {}
            polygon_points = polygon.get("points", []) if isinstance(polygon.get("points", []), list) else []
            for idx, point in enumerate(polygon_points):
                if not isinstance(point, dict):
                    continue
                try:
                    wx = float(point.get("x"))
                    wy = float(point.get("y"))
                except Exception:
                    continue
                points.append({
                    "x": wx,
                    "y": wy,
                    "label": f"{hid}_poly_{idx}",
                    "status": "polygon",
                    "color": "#a78bfa",
                    "outline_color": "#312e81",
                    "radius_px": 3,
                })
            bbox = item.get("map_artifacts", {}).get("polygon", {}).get("bbox", {}) if isinstance(item.get("map_artifacts", {}), dict) else {}
            if isinstance(bbox, dict) and all(key in bbox for key in ("min_x", "max_x", "min_y", "max_y")):
                corners = [
                    (bbox["min_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["max_y"]),
                    (bbox["min_x"], bbox["max_y"]),
                ]
                for idx, (wx, wy) in enumerate(corners):
                    points.append({
                        "x": float(wx),
                        "y": float(wy),
                        "label": f"{hid}_bbox_{idx}",
                        "status": "corrected_bbox",
                        "color": "#38bdf8",
                        "outline_color": "#0f172a",
                        "radius_px": 4,
                    })
        return points

    def refresh_llm_route6_realtime_map(self, *, build_if_missing: bool = False) -> Dict[str, Any]:
        self.ensure_route6_state()
        self.route6_update_map_uav_pose_text()
        manifest = self.route6_update_map_load_manifest(build_if_missing=bool(build_if_missing))
        combo = getattr(self, "llm_route6_realtime_map_layer_combo", None)
        preview = getattr(self, "llm_route6_realtime_map_preview_label", None)
        if not manifest:
            self.llm_route6_realtime_map_status_var.set("Realtime map: no layered map artifact yet.")
            if preview is not None:
                try:
                    preview.configure(text="No realtime layered occupancy map available.", image="")
                except tk.TclError:
                    pass
            return {}
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        values = [self._route6_update_map_layer_key(layer) for layer in layers if isinstance(layer, dict)]
        if combo is not None:
            try:
                combo.configure(values=values)
            except tk.TclError:
                pass
        selected = self.route6_choose_realtime_layer_key(layers, str(self.route6_update_map_layer_var.get() or ""))
        if values and selected not in values:
            selected = values[0]
        if selected:
            self.route6_update_map_layer_var.set(selected)
        layer_record = next(
            (layer for layer in layers if isinstance(layer, dict) and self._route6_update_map_layer_key(layer) == selected),
            {},
        )
        point_count = int((layer_record or {}).get("point_count", 0) or 0)
        occupied_count = int((layer_record or {}).get("occupied_cell_count", 0) or 0)
        status = f"Realtime map loaded: {selected or 'n/a'} points={point_count} occupied={occupied_count}"
        self.llm_route6_realtime_map_status_var.set(status)
        self.route6_update_map_status_var.set(f"Route 6 Update Map: {selected or 'n/a'} points={point_count} occupied={occupied_count}")
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
            frame = getattr(self, "llm_route6_realtime_map_frame", None)
            try:
                available_w = int(frame.winfo_width() or 920) if frame is not None else 920
            except Exception:
                available_w = 920
            scale = max(1, min(8, int((available_w - 40) / max(1, max(width, height)))))
            if scale > 1:
                image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
            image = self.route6_draw_update_map_uav_overlay(image, layer_record, scale=scale)
            photo = ImageTk.PhotoImage(image)
            self.llm_route6_realtime_map_preview_photo = photo
            preview.configure(image=photo, text="")
        except Exception as exc:
            try:
                preview.configure(text=f"Realtime map preview load failed: {exc}", image="")
            except tk.TclError:
                pass
        return manifest

    def route6_schedule_llm_realtime_map_refresh(self) -> None:
        window = getattr(self, "llm_route6_window", None)
        if window is None:
            self.llm_route6_realtime_map_after_id = None
            return
        try:
            self.refresh_llm_route6_realtime_map(build_if_missing=False)
            self.refresh_llm_route6_or_avoidance_display()
            self.llm_route6_realtime_map_after_id = window.after(1000, self.route6_schedule_llm_realtime_map_refresh)
        except tk.TclError:
            self.llm_route6_realtime_map_after_id = None

    def route6_latest_avoidance_jsonl_event(self, output_dir: Optional[Path]) -> Dict[str, Any]:
        if output_dir is None:
            return {}
        path = Path(output_dir) / "avoidance_events.jsonl"
        rows: List[Dict[str, Any]] = []
        if callable(getattr(self, "read_jsonl_artifact", None)):
            try:
                rows = [row for row in self.read_jsonl_artifact(path) if isinstance(row, dict)]
            except Exception:
                rows = []
        elif path.is_file():
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                rows = [row for row in rows if isinstance(row, dict)]
            except Exception:
                rows = []
        return dict(rows[-1]) if rows else {}

    def route6_read_avoidance_summary(self, output_dir: Optional[Path]) -> Dict[str, Any]:
        if output_dir is None:
            return {}
        path = Path(output_dir) / "avoidance_session_summary.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def route6_current_output_dir_from_state(self) -> Optional[Path]:
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        output_dir = str(state.get("output_dir", "") or "")
        return Path(output_dir) if output_dir else None

    def route6_or_avoidance_display_payload(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        route5_state = getattr(self, "llm_route5_state", {}) if isinstance(getattr(self, "llm_route5_state", {}), dict) else {}
        output_dir = self.route6_current_output_dir_from_state()
        event = {}
        for source in (
            state.get("last_or2_event", {}),
            state.get("last_obstacle_event", {}),
            route5_state.get("last_or2_event", {}),
            route5_state.get("last_obstacle_event", {}),
            self.route6_latest_avoidance_jsonl_event(output_dir),
        ):
            if isinstance(source, dict) and source:
                event = source
                break
        summary = self.route6_read_avoidance_summary(output_dir)
        prediction = event.get("prediction", {}) if isinstance(event.get("prediction", {}), dict) else {}
        rule = event.get("rule", {}) if isinstance(event.get("rule", {}), dict) else {}
        pointcloud_summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary", {}), dict) else {}
        depth_summary = event.get("depth_obstacle_summary", {}) if isinstance(event.get("depth_obstacle_summary", {}), dict) else {}
        risk_state = str(
            event.get(
                "front_risk_state",
                prediction.get("front_risk_state", event.get("risk_state", "unknown")),
            )
            or "unknown"
        )
        selected_direction = str(
            event.get(
                "or2_selected_direction",
                rule.get("selected_direction", event.get("selected_direction", "--")),
            )
            or "--"
        )
        front_depth = 0.0
        for source in (event, pointcloud_summary, depth_summary):
            if not isinstance(source, dict):
                continue
            for key in ("front_min_depth_cm", "front_depth_cm", "front_obstacle_depth_cm"):
                try:
                    value = float(source.get(key, 0.0) or 0.0)
                except Exception:
                    value = 0.0
                if value > 0.0:
                    front_depth = value
                    break
            if front_depth > 0.0:
                break
        payload = {
            "schema": "route6_or_avoidance_display_v1",
            "risk_state": risk_state,
            "selected_direction": selected_direction,
            "front_min_depth_cm": round(float(front_depth), 2),
            "avoidance_active": bool(event.get("avoidance_active", False)),
            "collision_state": bool(event.get("collision_state", False)),
            "avoidance_failed": bool(event.get("avoidance_failed", False)),
            "event_count": int(summary.get("event_count", summary.get("avoidance_event_count", 0)) or 0),
            "collision_count": int(summary.get("collision_count", 0) or 0),
            "avoidance_failed_count": int(summary.get("avoidance_failed_count", 0) or 0),
            "summary_status": str(summary.get("status", "") or ""),
            "frame_id": str(event.get("frame_id", event.get("capture_frame_id", "")) or ""),
            "source_output_dir": str(output_dir or ""),
        }
        return payload

    def refresh_llm_route6_or_avoidance_display(self) -> Dict[str, Any]:
        payload = self.route6_or_avoidance_display_payload()
        risk = str(payload.get("risk_state", "unknown") or "unknown")
        selected = str(payload.get("selected_direction", "--") or "--")
        front = float(payload.get("front_min_depth_cm", 0.0) or 0.0)
        active = "yes" if bool(payload.get("avoidance_active", False)) else "no"
        collision = "yes" if bool(payload.get("collision_state", False)) else "no"
        self.llm_route6_or_status_var.set(
            f"OR Avoidance: risk={risk} selected={selected} front={front:.1f}cm active={active} collision={collision}"
        )
        self.llm_route6_or_detail_var.set(
            "OR detail: "
            f"events={int(payload.get('event_count', 0) or 0)} "
            f"collisions={int(payload.get('collision_count', 0) or 0)} "
            f"failed={int(payload.get('avoidance_failed_count', 0) or 0)} "
            f"frame={payload.get('frame_id', '') or 'n/a'} "
            f"source={payload.get('source_output_dir', '') or 'n/a'}"
        )
        return payload

    def _on_llm_route6_mousewheel(self, event: tk.Event):
        canvas = getattr(self, "llm_route6_scroll_canvas", None)
        if canvas is None:
            return None
        try:
            delta = int(-1 * (float(event.delta) / 120.0))
            if delta == 0:
                delta = -1 if float(event.delta) > 0 else 1
            canvas.yview_scroll(delta, "units")
        except tk.TclError:
            pass
        return "break"

    def _on_llm_route6_mousewheel_linux(self, event: tk.Event):
        canvas = getattr(self, "llm_route6_scroll_canvas", None)
        if canvas is None:
            return None
        try:
            canvas.yview_scroll(-1 if int(getattr(event, "num", 0) or 0) == 4 else 1, "units")
        except tk.TclError:
            pass
        return "break"

    def _bind_llm_route6_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_llm_route6_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route6_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route6_mousewheel_linux, add="+")
        except tk.TclError:
            return
        for child in widget.winfo_children():
            self._bind_llm_route6_mousewheel_tree(child)

    def refresh_llm_route6_map(self) -> None:
        self.ensure_route6_state()
        widget = getattr(self, "llm_route6_map_widget", None)
        if widget is None:
            return
        try:
            if callable(getattr(self, "load_map_resources", None)) and not self.load_map_resources(force=not bool(getattr(self, "map_config", {}))):
                self.llm_route6_map_status_var.set("Map: unavailable")
                return
            pose = self.route6_current_pose()
            pose_x = float(pose.get("x", 0.0) or 0.0)
            pose_y = float(pose.get("y", 0.0) or 0.0)
            pose_yaw = float(pose.get("yaw", 0.0) or 0.0)
            houses: List[Dict[str, Any]] = []
            boxes: List[Dict[str, Any]] = []
            if callable(getattr(self, "build_map_display", None)):
                try:
                    houses, boxes = self.build_map_display({"x": pose_x, "y": pose_y, "yaw": pose_yaw})
                except Exception:
                    houses, boxes = [], []
            map_frame = getattr(self, "llm_route6_map_frame", None)
            if map_frame is not None and hasattr(widget, "resize_canvas"):
                try:
                    available_w = float(map_frame.winfo_width() or widget.canvas.winfo_width() or 1040)
                    if callable(getattr(self, "route5_map_canvas_size_for_width", None)):
                        size = self.route5_map_canvas_size_for_width(available_w, image_size=self.map_image_size())
                    else:
                        size = {"width": max(760, int(available_w) - 24), "height": 320}
                    if abs(int(getattr(widget, "_canvas_w", 0)) - size["width"]) > 12 or abs(int(getattr(widget, "_canvas_h", 0)) - size["height"]) > 12:
                        widget.resize_canvas(size["width"], size["height"])
                except Exception:
                    pass
            widget.set_background_image(getattr(self, "map_image", None))
            calibration = getattr(self, "map_calibration", {}) if isinstance(getattr(self, "map_calibration", {}), dict) else {}
            widget.set_calibration(
                calibration.get("affine_world_to_image"),
                self.map_image_size() if callable(getattr(self, "map_image_size", None)) else None,
                [],
                calibration.get("homography_world_to_image"),
            )
            widget.set_image_layer_offset(*getattr(self, "map_display_offset_px", (0.0, 0.0)))
            widget.set_house_boxes(boxes)
            widget.update_houses([])
            widget.update_uav(pose_x, pose_y, pose_yaw)
            route_points = self.route6_map_route_points()
            overlay_points = self.route6_map_overlay_points()
            widget.set_route_plan({"route_points": route_points})
            widget.set_point_overlay_points(overlay_points)
            self.llm_route6_map_status_var.set(
                f"Map: route_points={len(route_points)} overlay_points={len(overlay_points)}"
            )
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM Route6_entrance_search v6 map failed: %s", exc)
            self.llm_route6_map_status_var.set(f"Map: failed: {exc}")

    def route6_update_map_latest_output_dir(self) -> Optional[Path]:
        self.ensure_route6_state()
        state = self.llm_route6_state if isinstance(self.llm_route6_state, dict) else {}
        for key in (
            "route6_update_map_realtime",
            "route6_update_map",
            "route6_update_map_capture",
        ):
            record = state.get(key, {}) if isinstance(state.get(key, {}), dict) else {}
            raw = str(record.get("output_dir", "") or "")
            if raw:
                path = Path(raw)
                if path.exists():
                    return path
        raw_update_output = str(state.get("route6_update_map_output_dir", "") or "")
        if raw_update_output:
            path = Path(raw_update_output)
            if path.exists():
                return path
        state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        if state_output:
            path = Path(state_output)
            manifest_path = path / "map" / "layered_occupancy" / "route6_layered_occupancy.json"
            if path.exists() and (path.name.startswith("route6_update_map_") or manifest_path.is_file()):
                return path
        root = self.route6_output_root()
        if not root.exists():
            return None
        candidates = [
            path
            for pattern in ("route6_update_map_*", "route6_nearest_map_*")
            for path in root.glob(pattern)
            if path.is_dir()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def route6_output_dir_kind(self, output_dir: Path) -> str:
        name = Path(output_dir).name
        if name.startswith("route6_update_map_"):
            return "update_map"
        if name.startswith("route6_nearest_map_"):
            return "target_search"
        return "unknown"

    def route6_update_map_should_preserve_search_output(self, output_dir: Path) -> bool:
        self.ensure_route6_state()
        state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        if not state_output:
            return False
        current = Path(state_output)
        target = Path(output_dir)
        return (
            self.route6_output_dir_kind(current) == "target_search"
            and self.route6_output_dir_kind(target) == "update_map"
            and str(current) != str(target)
        )

    def route6_record_update_map_output_dir(self, output_dir: Path, *, source: str = "") -> bool:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        preserve_search_output = self.route6_update_map_should_preserve_search_output(out_path)
        self.llm_route6_state["route6_update_map_output_dir"] = str(out_path)
        if source:
            self.llm_route6_state["route6_update_map_output_source"] = str(source)
        if not preserve_search_output:
            self.llm_route6_state["output_dir"] = str(out_path)
        return preserve_search_output

    def route6_path_is_under(self, path: Path, parent: Path) -> bool:
        try:
            Path(path).resolve().relative_to(Path(parent).resolve())
            return True
        except Exception:
            return False

    def route6_capture_folder_pointcloud_paths(self, output_dir: Path) -> List[Path]:
        out_path = Path(output_dir)
        paths: List[Path] = []
        for path in sorted(out_path.glob("houses/house_*/pointcloud/merged_point_cloud_world_standard_m.npy")):
            if path.is_file() and path not in paths:
                paths.append(path)
        logged_frame_dirs = self.route6_logged_capture_frame_dirs(out_path)
        if logged_frame_dirs:
            frame_candidates = [frame_dir / "point_cloud_world_standard_m.npy" for frame_dir in logged_frame_dirs]
        else:
            frame_candidates = sorted(out_path.glob("frames/frame_*/point_cloud_world_standard_m.npy"))
        for path in frame_candidates:
            if path.is_file() and path not in paths:
                paths.append(path)
        return paths

    def route6_logged_capture_frame_dirs(self, output_dir: Path) -> List[Path]:
        out_path = Path(output_dir)
        log_paths = [
            out_path / "route6_update_map_capture_log.jsonl",
            out_path / "lidar_capture_log.jsonl",
        ]
        frame_dirs: List[Path] = []
        seen: set[str] = set()
        for log_path in log_paths:
            if not log_path.is_file():
                continue
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if str(row.get("capture_kind", "") or "") != "route6_update_map_capture":
                    continue
                capture_dir = Path(str(row.get("capture_dir", "") or ""))
                if not capture_dir:
                    frame_index = int(row.get("frame_index", 0) or 0)
                    capture_dir = out_path / "frames" / f"frame_{frame_index:06d}"
                if not self.route6_path_is_under(capture_dir, out_path):
                    continue
                key = str(capture_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    frame_dirs.append(capture_dir)
            if frame_dirs:
                break
        return frame_dirs

    def route6_capture_folder_pointcloud_report_path(self, output_dir: Path) -> Path:
        return Path(output_dir) / "map" / "pointcloud_processing_report.json"

    def route6_capture_folder_candidate_pointcloud_paths(self, output_dir: Path) -> List[Path]:
        out_path = Path(output_dir)
        paths: List[Path] = []
        allowed_suffixes = {".npy", ".npz", ".csv", ".txt", ".xyz"}
        name_tokens = ("point", "cloud", "lidar", "xyz")
        for path in sorted(out_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            relative = path.relative_to(out_path)
            parts = {part.lower() for part in relative.parts}
            if "map" in parts or "frames" in parts or "houses" in parts:
                continue
            if path.name == "point_cloud_world_standard_m.npy":
                continue
            lowered = path.name.lower()
            if path.suffix.lower() == ".xyz" or any(token in lowered for token in name_tokens):
                paths.append(path)
        return paths

    def route6_capture_folder_raw_depth_frame_dirs(self, output_dir: Path) -> List[Path]:
        frames_root = Path(output_dir) / "frames"
        if not frames_root.exists():
            return []
        frame_dirs: List[Path] = []
        for frame_dir in sorted(path for path in frames_root.glob("frame_*") if path.is_dir()):
            if (frame_dir / "point_cloud_world_standard_m.npy").is_file():
                continue
            if (frame_dir / "depth.npy").is_file() and (frame_dir / "rgb.png").is_file() and (frame_dir / "camera_info.json").is_file():
                frame_dirs.append(frame_dir)
        return frame_dirs

    def route6_load_candidate_pointcloud(self, path: Path) -> Tuple[np.ndarray, str]:
        point_path = Path(path)
        suffix = point_path.suffix.lower()
        if suffix == ".npy":
            cloud = np.asarray(np.load(point_path, allow_pickle=False), dtype=np.float32)
            return cloud, "npy"
        if suffix == ".npz":
            with np.load(point_path, allow_pickle=False) as payload:
                preferred = [key for key in ("points", "pointcloud", "point_cloud", "cloud", "xyz") if key in payload.files]
                keys = preferred + [key for key in payload.files if key not in preferred]
                for key in keys:
                    cloud = np.asarray(payload[key], dtype=np.float32)
                    if cloud.ndim == 2 and cloud.shape[1] >= 3 and cloud.shape[0] > 0:
                        return cloud, f"npz:{key}"
            return np.zeros((0, 6), dtype=np.float32), "npz:no_usable_array"
        if suffix in {".csv", ".txt", ".xyz"}:
            try:
                cloud = np.asarray(np.genfromtxt(point_path, delimiter=","), dtype=np.float32)
                if cloud.ndim == 2 and cloud.shape[1] >= 3 and cloud.shape[0] > 0:
                    return cloud, f"{suffix[1:]}:comma"
            except Exception:
                pass
            cloud = np.asarray(np.genfromtxt(point_path), dtype=np.float32)
            return cloud, f"{suffix[1:]}:whitespace"
        return np.zeros((0, 6), dtype=np.float32), f"{suffix}:unsupported"

    def route6_standardize_pointcloud_array(self, cloud: np.ndarray) -> np.ndarray:
        arr = np.asarray(cloud, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size < 3 or arr.size % 3 != 0:
                return np.zeros((0, 6), dtype=np.float32)
            arr = arr.reshape((-1, 3))
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
            return np.zeros((0, 6), dtype=np.float32)
        arr = arr[np.all(np.isfinite(arr[:, :3]), axis=1)]
        if arr.shape[0] <= 0:
            return np.zeros((0, 6), dtype=np.float32)
        if arr.shape[1] >= 6:
            return arr[:, :6].astype(np.float32, copy=False)
        pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr[:, :3], pad]).astype(np.float32, copy=False)

    def route6_update_pointcloud_report_text(self, report: Dict[str, Any]) -> None:
        text_widget = getattr(self, "route6_pointcloud_report_text", None)
        if text_widget is None:
            return
        try:
            text_widget.configure(state="normal")
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, json.dumps(self.route6_json_safe(report), indent=2, ensure_ascii=False))
            text_widget.configure(state="disabled")
        except tk.TclError:
            pass

    def route6_process_capture_folder_pointclouds(self, output_dir: Path) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = self.route6_capture_folder_pointcloud_report_path(out_path)
        standard_paths = self.route6_capture_folder_pointcloud_paths(out_path)
        candidate_paths = self.route6_capture_folder_candidate_pointcloud_paths(out_path)
        raw_depth_frame_dirs = self.route6_capture_folder_raw_depth_frame_dirs(out_path)
        log_files = [
            path
            for path in (
                out_path / "lidar_capture_log.jsonl",
                out_path / "route6_update_map_capture_log.jsonl",
                out_path / "route6_events.jsonl",
            )
            if path.is_file()
        ]
        frame_entries: List[Dict[str, Any]] = []
        postprocessed_paths: List[str] = []
        args = getattr(self, "args", None)
        lidar_depth_projection = str(getattr(args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION))
        min_depth_cm = float(getattr(args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
        max_depth_cm = float(getattr(args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
        for frame_dir in raw_depth_frame_dirs:
            frame_entry: Dict[str, Any] = {
                "capture_dir": str(frame_dir),
                "status": "pending",
            }
            try:
                capture_payload = flight.read_json_object(frame_dir / "capture.json")
                ensured = flight.ensure_standard_world_cloud_for_capture(
                    frame_dir,
                    capture_payload=capture_payload,
                    lidar_depth_projection=lidar_depth_projection,
                    min_depth_cm=min_depth_cm,
                    max_depth_cm=max_depth_cm,
                )
                pointcloud_path = Path(str(ensured.get("point_cloud_world_standard_m_npy_path", "") or ""))
                point_count = int(ensured.get("point_count", 0) or 0)
                if pointcloud_path.is_file():
                    if point_count <= 0:
                        try:
                            point_count = int(np.load(pointcloud_path, mmap_mode="r").shape[0])
                        except Exception:
                            point_count = 0
                    frame_entry.update(
                        {
                            "status": "depth_postprocessed",
                            "point_cloud_world_standard_m_npy_path": str(pointcloud_path),
                            "point_count": int(point_count),
                            "depth_projection_selected": ensured.get("depth_projection_selected", ""),
                        }
                    )
                    postprocessed_paths.append(str(pointcloud_path))
                else:
                    frame_entry.update({"status": "no_pointcloud_output", "result": ensured})
            except Exception as exc:
                frame_entry.update({"status": "failed", "error": str(exc)})
            frame_entries.append(frame_entry)
        standard_paths = self.route6_capture_folder_pointcloud_paths(out_path)
        entries: List[Dict[str, Any]] = []
        created_paths: List[str] = []
        next_index = len(standard_paths) + 1
        for candidate in candidate_paths:
            entry: Dict[str, Any] = {
                "path": str(candidate),
                "size_bytes": int(candidate.stat().st_size),
                "status": "pending",
            }
            try:
                raw_cloud, loader = self.route6_load_candidate_pointcloud(candidate)
                standard_cloud = self.route6_standardize_pointcloud_array(raw_cloud)
                entry["loader"] = loader
                entry["raw_shape"] = list(np.asarray(raw_cloud).shape)
                entry["standard_shape"] = list(standard_cloud.shape)
                if standard_cloud.shape[0] <= 0:
                    entry["status"] = "invalid_shape"
                    entries.append(entry)
                    continue
                frame_dir = out_path / "frames" / f"frame_{next_index:06d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                standard_path = frame_dir / "point_cloud_world_standard_m.npy"
                np.save(standard_path, standard_cloud)
                entry["status"] = "standardized"
                entry["standard_pointcloud_path"] = str(standard_path)
                entry["point_count"] = int(standard_cloud.shape[0])
                created_paths.append(str(standard_path))
                next_index += 1
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
            entries.append(entry)
        refreshed_standard_paths = self.route6_capture_folder_pointcloud_paths(out_path)
        all_created_paths = postprocessed_paths + created_paths
        if refreshed_standard_paths:
            status = "processed" if all_created_paths else "ready"
        elif candidate_paths:
            status = "failed"
        else:
            status = "missing_pointcloud"
        report = {
            "schema": "route6_pointcloud_processing_report_v1",
            "status": status,
            "output_dir": str(out_path),
            "report_path": str(report_path),
            "standard_pointcloud_count": len(refreshed_standard_paths),
            "standard_pointcloud_paths": [str(path) for path in refreshed_standard_paths],
            "candidate_file_count": len(candidate_paths),
            "candidate_files": [str(path) for path in candidate_paths],
            "raw_depth_frame_count": len(raw_depth_frame_dirs),
            "postprocessed_frame_count": len(postprocessed_paths),
            "failed_frame_count": len([entry for entry in frame_entries if entry.get("status") == "failed"]),
            "created_standard_pointcloud_count": len(all_created_paths),
            "created_standard_pointcloud_paths": all_created_paths,
            "log_file_count": len(log_files),
            "log_files": [str(path) for path in log_files],
            "frame_entries": frame_entries,
            "entries": entries,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(report_path, report)
        self.route6_update_pointcloud_report_text(report)
        return report

    def route6_list_capture_folders(self, root: Optional[Path] = None) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        root_path = Path(root) if root is not None else self.route6_output_root()
        if not root_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for run_dir in sorted((path for path in root_path.iterdir() if path.is_dir()), key=lambda path: path.name):
            manifest_path = run_dir / "map" / "layered_occupancy" / "route6_layered_occupancy.json"
            pointcloud_paths = self.route6_capture_folder_pointcloud_paths(run_dir)
            looks_like_capture = (
                run_dir.name.startswith(("route6_update_map_", "route6_nearest_map_"))
                or (run_dir / "frames").exists()
                or (run_dir / "houses").exists()
                or (run_dir / "lidar_capture_log.jsonl").is_file()
                or (run_dir / "route6_update_map_capture_log.jsonl").is_file()
                or manifest_path.is_file()
            )
            if not looks_like_capture:
                continue
            mtime_candidates = [run_dir]
            for candidate in (
                manifest_path,
                run_dir / "lidar_capture_log.jsonl",
                run_dir / "route6_update_map_capture_log.jsonl",
            ):
                if candidate.exists():
                    mtime_candidates.append(candidate)
            mtime_candidates.extend(pointcloud_paths)
            try:
                mtime = max(path.stat().st_mtime for path in mtime_candidates if path.exists())
            except Exception:
                mtime = 0.0
            try:
                updated_at = datetime.fromtimestamp(float(mtime)).isoformat(timespec="seconds")
            except Exception:
                updated_at = ""
            records.append(
                {
                    "schema": "route6_capture_folder_record_v1",
                    "name": run_dir.name,
                    "path": str(run_dir),
                    "mtime": float(mtime),
                    "updated_at": updated_at,
                    "pointcloud_count": len(pointcloud_paths),
                    "has_layered_map": manifest_path.is_file(),
                    "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
                }
            )
        records.sort(key=lambda record: (float(record.get("mtime", 0.0) or 0.0), str(record.get("name", ""))), reverse=True)
        return records

    def route6_set_selected_capture_folder(self, folder_path: Path) -> Path:
        self.ensure_route6_state()
        selected = Path(folder_path)
        self.route6_selected_capture_folder_var.set(str(selected))
        if self.route6_output_dir_kind(selected) == "update_map":
            self.route6_record_update_map_output_dir(selected, source="capture_folder_reader")
        else:
            self.llm_route6_state["output_dir"] = str(selected)
        return selected

    def route6_selected_capture_folder_path(self) -> Optional[Path]:
        self.ensure_route6_state()
        listbox = getattr(self, "route6_capture_folder_listbox", None)
        records = getattr(self, "route6_capture_folder_records", []) or []
        if listbox is not None and records:
            try:
                selection = listbox.curselection()
                if selection:
                    record = records[int(selection[0])]
                    path = Path(str(record.get("path", "") or ""))
                    if path.exists():
                        return self.route6_set_selected_capture_folder(path)
            except Exception:
                pass
        selected_raw = str(self.route6_selected_capture_folder_var.get() or "")
        if selected_raw:
            path = Path(selected_raw)
            if path.exists():
                return self.route6_set_selected_capture_folder(path)
        if records:
            path = Path(str(records[0].get("path", "") or ""))
            if path.exists():
                return self.route6_set_selected_capture_folder(path)
        return None

    def on_route6_capture_folder_select(self, _event: Optional[tk.Event] = None) -> None:
        selected = self.route6_selected_capture_folder_path()
        if selected is not None:
            self.route6_capture_folder_status_var.set(f"Route 6 Capture Folder Reader: selected {selected}")

    def on_route6_capture_folder_process_pointcloud(self) -> Dict[str, Any]:
        selected = self.route6_selected_capture_folder_path()
        if selected is None:
            self.route6_capture_folder_status_var.set("Route 6 Capture Folder Reader: select a capture folder first.")
            self.route6_update_map_status_var.set("Route 6 Update Map: select a capture folder first.")
            return {}
        self.route6_set_selected_capture_folder(selected)
        report = self.route6_process_capture_folder_pointclouds(selected)
        self.refresh_route6_capture_folder_list()
        status = str(report.get("status", "") or "")
        standard_count = int(report.get("standard_pointcloud_count", 0) or 0)
        candidate_count = int(report.get("candidate_file_count", 0) or 0)
        if status == "missing_pointcloud":
            message = f"Route 6 Capture Folder Reader: no pointcloud files found under {selected}"
        elif status == "failed":
            message = f"Route 6 Capture Folder Reader: pointcloud processing failed candidates={candidate_count} -> {selected}"
        else:
            message = f"Route 6 Capture Folder Reader: pointcloud {status} standard={standard_count} candidates={candidate_count}"
        self.route6_capture_folder_status_var.set(message)
        self.route6_update_map_status_var.set(message.replace("Route 6 Capture Folder Reader", "Route 6 Update Map"))
        return report

    def refresh_route6_capture_folder_list(self) -> List[Dict[str, Any]]:
        self.ensure_route6_state()
        records = self.route6_list_capture_folders()
        self.route6_capture_folder_records = records
        listbox = getattr(self, "route6_capture_folder_listbox", None)
        selected_raw = str(self.route6_selected_capture_folder_var.get() or "")
        selected_index = 0
        if listbox is not None:
            try:
                listbox.delete(0, tk.END)
                for index, record in enumerate(records):
                    if selected_raw and str(record.get("path", "") or "") == selected_raw:
                        selected_index = index
                    map_flag = "map=yes" if record.get("has_layered_map") else "map=no"
                    label = (
                        f"{record.get('name', '')} | pc={int(record.get('pointcloud_count', 0) or 0)} "
                        f"| {map_flag} | {record.get('updated_at', '')}"
                    )
                    listbox.insert(tk.END, label)
                if records:
                    selected_index = min(max(0, selected_index), len(records) - 1)
                    listbox.selection_set(selected_index)
                    listbox.activate(selected_index)
                    self.route6_set_selected_capture_folder(Path(str(records[selected_index].get("path", "") or "")))
            except tk.TclError:
                pass
        root = self.route6_output_root()
        self.route6_capture_folder_status_var.set(f"Route 6 Capture Folder Reader: folders={len(records)} root={root}")
        return records

    def on_route6_capture_folder_generate_map(self) -> Dict[str, Any]:
        selected = self.route6_selected_capture_folder_path()
        if selected is None:
            self.route6_capture_folder_status_var.set("Route 6 Capture Folder Reader: select a capture folder first.")
            self.route6_update_map_status_var.set("Route 6 Update Map: select a capture folder first.")
            return {}
        self.route6_set_selected_capture_folder(selected)
        result = self.route6_update_map_build_from_pointcloud(selected)
        self.refresh_route6_update_map_window()
        self.refresh_route6_capture_folder_list()
        if result:
            self.route6_capture_folder_status_var.set(
                f"Route 6 Capture Folder Reader: generated map layers={int(result.get('layer_count', 0) or 0)} -> {selected}"
            )
        else:
            self.route6_capture_folder_status_var.set(f"Route 6 Capture Folder Reader: no pointcloud found under {selected}")
        return result

    def on_route6_capture_folder_load_map(self) -> Dict[str, Any]:
        selected = self.route6_selected_capture_folder_path()
        if selected is None:
            self.route6_capture_folder_status_var.set("Route 6 Capture Folder Reader: select a capture folder first.")
            self.route6_update_map_status_var.set("Route 6 Update Map: select a capture folder first.")
            return {}
        self.route6_set_selected_capture_folder(selected)
        manifest = self.route6_update_map_load_manifest(build_if_missing=False, output_dir=selected)
        if not manifest:
            self.route6_capture_folder_status_var.set(f"Route 6 Capture Folder Reader: no map found under {selected}")
            self.route6_update_map_status_var.set(f"Route 6 Update Map: no layered map under {selected}")
            return {}
        manifest_path = self.route6_update_map_manifest_path(selected)
        record = {
            "schema": "route6_update_map_state_v1",
            "source": "manual_route6_update_map_load",
            "output_dir": str(selected),
            "manifest_path": str(manifest_path),
            "layered_occupancy_dir": str(manifest_path.parent),
            "layer_count": int(manifest.get("layer_count", len(manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else [])) or 0),
            "fixed_world_bounds_cm": dict(manifest.get("fixed_world_bounds_cm", {}) or {}),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_record_update_map_output_dir(selected, source="manual_route6_update_map_load")
        self.llm_route6_state["route6_update_map"] = record
        self.route6_write_state_artifact()
        self.refresh_route6_update_map_window()
        self.refresh_route6_capture_folder_list()
        self.route6_update_map_status_var.set(
            f"Route 6 Update Map: loaded map layers={record['layer_count']} -> {selected}"
        )
        self.route6_capture_folder_status_var.set(
            f"Route 6 Capture Folder Reader: loaded map layers={record['layer_count']} -> {selected}"
        )
        return manifest

    def make_route6_update_map_output_dir(self) -> Path:
        self.ensure_route6_state()
        root = self.route6_output_root()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        candidate = root / f"route6_update_map_{timestamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = root / f"route6_update_map_{timestamp}_{suffix}"
        (candidate / "map").mkdir(parents=True, exist_ok=True)
        (candidate / "frames").mkdir(parents=True, exist_ok=True)
        return candidate

    def route6_update_map_capture_interval_s(self) -> float:
        return self.route6_float_param(
            self.route6_update_map_capture_interval_s_var,
            1.0,
            min_value=0.05,
            max_value=60.0,
        )

    def route6_update_map_min_move_cm(self) -> float:
        return self.route6_float_param(
            self.route6_update_map_min_move_cm_var,
            50.0,
            min_value=0.0,
            max_value=5000.0,
        )

    def route6_update_map_min_yaw_deg(self) -> float:
        return self.route6_float_param(
            self.route6_update_map_min_yaw_deg_var,
            5.0,
            min_value=0.0,
            max_value=180.0,
        )

    def route6_update_map_pose_delta(self, current_pose: Dict[str, Any], last_pose: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not isinstance(last_pose, dict) or not last_pose:
            return {"position_cm": float("inf"), "yaw_deg": float("inf")}
        try:
            dx = float(current_pose.get("x", 0.0) or 0.0) - float(last_pose.get("x", 0.0) or 0.0)
            dy = float(current_pose.get("y", 0.0) or 0.0) - float(last_pose.get("y", 0.0) or 0.0)
            dz = float(current_pose.get("z", 0.0) or 0.0) - float(last_pose.get("z", 0.0) or 0.0)
            yaw_delta = abs(float(current_pose.get("yaw", 0.0) or 0.0) - float(last_pose.get("yaw", 0.0) or 0.0))
            yaw_delta = min(yaw_delta % 360.0, (360.0 - yaw_delta) % 360.0)
            return {"position_cm": float(math.sqrt(dx * dx + dy * dy + dz * dz)), "yaw_deg": float(yaw_delta)}
        except Exception:
            return {"position_cm": float("inf"), "yaw_deg": float("inf")}

    def route6_update_map_should_capture_pose(self, current_pose: Dict[str, Any], last_pose: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, float]]:
        delta = self.route6_update_map_pose_delta(current_pose, last_pose)
        min_move = self.route6_update_map_min_move_cm()
        min_yaw = self.route6_update_map_min_yaw_deg()
        if not isinstance(last_pose, dict) or not last_pose:
            return True, delta
        if min_move <= 0.0 and min_yaw <= 0.0:
            return True, delta
        should_capture = bool(float(delta.get("position_cm", 0.0) or 0.0) >= min_move or float(delta.get("yaw_deg", 0.0) or 0.0) >= min_yaw)
        return should_capture, delta

    def route6_should_capture_map_after_motion(
        self,
        current_pose: Dict[str, Any],
        last_pose: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        should_capture, delta = self.route6_update_map_should_capture_pose(current_pose, last_pose)
        reason = "first_pose" if not isinstance(last_pose, dict) or not last_pose else ("motion_threshold_met" if should_capture else "stationary_pose")
        return {
            "schema": "route6_update_map_capture_gate_v1",
            "should_capture": bool(should_capture),
            "reason": reason,
            "position_delta_cm": float(delta.get("position_cm", 0.0) or 0.0),
            "yaw_delta_deg": float(delta.get("yaw_deg", 0.0) or 0.0),
            "min_move_cm": self.route6_update_map_min_move_cm(),
            "min_yaw_deg": self.route6_update_map_min_yaw_deg(),
            "current_pose": self.route6_json_safe(current_pose if isinstance(current_pose, dict) else {}),
            "last_accepted_pose": self.route6_json_safe(last_pose if isinstance(last_pose, dict) else {}),
        }

    def route6_update_map_capture_once(self, session: Any, output_dir: Path) -> Dict[str, Any]:
        self.ensure_route6_state()
        if session is None:
            return {"capture_status": "failed", "error": "missing_session"}
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        frame_index = int((self.llm_route6_state or {}).get("route6_update_map_capture_frame_index", 0) or 0) + 1
        action_detail = {
            "schema_version": 1,
            "source": "route6_update_map",
            "capture_kind": "route6_update_map_capture",
            "frame_index": frame_index,
            "layer_z_cm": list(route6_map_builder.DEFAULT_ROUTE6_LAYER_Z_CM),
            "capture_interval_s": self.route6_update_map_capture_interval_s(),
        }
        capture = session.capture_lidar_stream_frame(out_path, frame_index, action_detail=action_detail)
        capture = capture if isinstance(capture, dict) else {"status": "failed", "error": "capture_returned_non_dict"}
        pointcloud_path = str(capture.get("point_cloud_world_standard_m_npy_path", "") or "")
        if not pointcloud_path:
            capture_dir = Path(str(capture.get("capture_dir", "") or ""))
            candidate = capture_dir / "point_cloud_world_standard_m.npy"
            if candidate.is_file():
                pointcloud_path = str(candidate)
                capture["point_cloud_world_standard_m_npy_path"] = pointcloud_path
        point_count = int(capture.get("point_count", 0) or 0)
        if point_count <= 0 and pointcloud_path and Path(pointcloud_path).is_file():
            try:
                point_count = int(np.asarray(np.load(pointcloud_path)).shape[0])
                capture["point_count"] = point_count
            except Exception:
                pass
        capture_status = str(capture.get("capture_status", capture.get("status", "")) or "").strip().lower()
        if capture_status == "ok":
            capture_status = "ok"
        elif str(capture.get("status", "") or "").strip().lower() == "ok":
            capture_status = "ok"
        else:
            capture_status = capture_status or "failed"
        row = {
            **self.route6_json_safe(capture),
            "schema": "route6_update_map_capture_row_v1",
            "capture_kind": "route6_update_map_capture",
            "capture_status": capture_status,
            "frame_index": frame_index,
            "point_count": int(point_count),
            "point_cloud_world_standard_m_npy_path": pointcloud_path,
            "capture_guard_passed": bool(point_count > 0 and Path(pointcloud_path).is_file()) if pointcloud_path else False,
            "capture_time": capture.get("capture_time", datetime.now().isoformat(timespec="seconds")),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_append_jsonl(out_path / "lidar_capture_log.jsonl", row)
        self.route6_append_jsonl(out_path / "route6_update_map_capture_log.jsonl", row)
        self.route6_record_update_map_output_dir(out_path, source="route6_update_map_capture")
        self.llm_route6_state["route6_update_map_capture_frame_index"] = frame_index
        self.llm_route6_state["route6_update_map_capture"] = {
            "schema": "route6_update_map_capture_state_v1",
            "running": not self.route6_update_map_capture_stop_event.is_set(),
            "output_dir": str(out_path),
            "frame_count": frame_index,
            "last_capture_status": capture_status,
            "last_point_count": int(point_count),
            "last_pointcloud_path": pointcloud_path,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_state_artifact()
        self.route6_update_map_status_var.set(f"Route 6 Update Map: captured frame={frame_index} points={point_count}")
        return row

    def route6_update_map_capture_worker(self, session: Any, output_dir: Path) -> None:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        self.route6_update_map_status_var.set(f"Route 6 Update Map: capture running -> {out_path}")
        while not self.route6_update_map_capture_stop_event.is_set():
            try:
                self.route6_update_map_capture_once(session, out_path)
            except Exception as exc:
                self.route6_update_map_status_var.set(f"Route 6 Update Map: capture failed: {exc}")
                self.route6_log_event(out_path, "route6_update_map_capture_error", {"error": str(exc)})
                break
            if self.route6_update_map_capture_stop_event.wait(self.route6_update_map_capture_interval_s()):
                break
        capture_state = self.llm_route6_state.get("route6_update_map_capture", {}) if isinstance(self.llm_route6_state.get("route6_update_map_capture", {}), dict) else {}
        capture_state["running"] = False
        capture_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route6_state["route6_update_map_capture"] = capture_state
        self.route6_write_state_artifact()
        self.route6_update_map_status_var.set(f"Route 6 Update Map: capture stopped -> {out_path}")

    def route6_update_map_request_window_refresh(self) -> None:
        window = getattr(self, "route6_update_map_window", None)
        if window is None:
            return
        try:
            if not window.winfo_exists():
                return
            window.after(0, lambda: self.refresh_route6_update_map_window(build_if_missing=False))
        except Exception:
            return

    def route6_update_map_ensure_capture_pointcloud(self, capture: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        out_path = Path(output_dir)
        payload = capture if isinstance(capture, dict) else {}
        pointcloud_path = Path(str(payload.get("point_cloud_world_standard_m_npy_path", "") or ""))
        point_count = int(payload.get("point_count", 0) or 0)
        if pointcloud_path.is_file() and point_count > 0:
            return {
                "status": "ready",
                "point_cloud_world_standard_m_npy_path": str(pointcloud_path),
                "point_count": int(point_count),
                "postprocessed_frame_count": 0,
            }
        capture_dir = Path(str(payload.get("capture_dir", "") or ""))
        if not capture_dir or not self.route6_path_is_under(capture_dir, out_path):
            return {"status": "missing_capture_dir", "point_cloud_world_standard_m_npy_path": "", "point_count": 0, "postprocessed_frame_count": 0}
        required = [capture_dir / "depth.npy", capture_dir / "rgb.png", capture_dir / "camera_info.json"]
        if not all(path.is_file() for path in required):
            return {"status": "missing_raw_depth_inputs", "point_cloud_world_standard_m_npy_path": "", "point_count": 0, "postprocessed_frame_count": 0}
        args = getattr(self, "args", None)
        lidar_depth_projection = str(getattr(args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION))
        min_depth_cm = float(getattr(args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
        max_depth_cm = float(getattr(args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
        try:
            capture_payload = flight.read_json_object(capture_dir / "capture.json")
        except Exception:
            capture_payload = payload
        try:
            ensured = flight.ensure_standard_world_cloud_for_capture(
                capture_dir,
                capture_payload=capture_payload,
                lidar_depth_projection=lidar_depth_projection,
                min_depth_cm=min_depth_cm,
                max_depth_cm=max_depth_cm,
            )
            pointcloud_path = Path(str(ensured.get("point_cloud_world_standard_m_npy_path", "") or ""))
            point_count = int(ensured.get("point_count", 0) or 0)
            if pointcloud_path.is_file() and point_count <= 0:
                try:
                    point_count = int(np.load(pointcloud_path, mmap_mode="r").shape[0])
                except Exception:
                    point_count = 0
            return {
                "status": "processed" if pointcloud_path.is_file() else "no_pointcloud_output",
                "point_cloud_world_standard_m_npy_path": str(pointcloud_path) if pointcloud_path.is_file() else "",
                "point_count": int(point_count),
                "postprocessed_frame_count": 1 if pointcloud_path.is_file() else 0,
                "depth_projection_selected": ensured.get("depth_projection_selected", ""),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "point_cloud_world_standard_m_npy_path": "",
                "point_count": 0,
                "postprocessed_frame_count": 0,
            }

    def route6_update_map_realtime_worker(
        self,
        session: Any,
        output_dir: Path,
        *,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.route6_record_update_map_output_dir(out_path, source="route6_update_map_realtime")
        capture_count = 0
        map_count = 0
        loop_count = 0
        skipped_stationary_count = 0
        last_manifest_path = ""
        last_pointcloud_process_status = ""
        last_postprocessed_frame_count = 0
        last_accepted_pose: Optional[Dict[str, Any]] = None
        last_pose_delta: Dict[str, float] = {}
        final_error = ""
        self.llm_route6_state["route6_update_map_realtime"] = {
            "schema": "route6_update_map_realtime_state_v1",
            "running": True,
            "output_dir": str(out_path),
            "capture_count": 0,
            "map_count": 0,
            "skipped_stationary_count": 0,
            "last_manifest_path": "",
            "last_pointcloud_process_status": "",
            "last_postprocessed_frame_count": 0,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_state_artifact()
        self.route6_update_map_status_var.set(f"Route 6 Update Map: realtime update running -> {out_path}")

        while not self.route6_update_map_realtime_stop_event.is_set():
            loop_count += 1
            current_pose = self.route6_current_pose()
            should_capture, last_pose_delta = self.route6_update_map_should_capture_pose(current_pose, last_accepted_pose)
            if not should_capture:
                skipped_stationary_count += 1
                skip_event = {
                    "schema": "route6_update_map_skip_event_v1",
                    "reason": "stationary_pose",
                    "loop_index": int(loop_count),
                    "skipped_stationary_count": int(skipped_stationary_count),
                    "position_delta_cm": float(last_pose_delta.get("position_cm", 0.0) or 0.0),
                    "yaw_delta_deg": float(last_pose_delta.get("yaw_deg", 0.0) or 0.0),
                    "min_move_cm": self.route6_update_map_min_move_cm(),
                    "min_yaw_deg": self.route6_update_map_min_yaw_deg(),
                    "current_pose": self.route6_json_safe(current_pose),
                    "last_accepted_pose": self.route6_json_safe(last_accepted_pose or {}),
                    "created_at": datetime.now().isoformat(timespec="milliseconds"),
                }
                self.route6_append_jsonl(out_path / "route6_update_map_skip_events.jsonl", skip_event)
                self.llm_route6_state["route6_update_map_realtime"] = {
                    "schema": "route6_update_map_realtime_state_v1",
                    "running": True,
                    "output_dir": str(out_path),
                    "capture_count": int(capture_count),
                    "map_count": int(map_count),
                    "skipped_stationary_count": int(skipped_stationary_count),
                    "last_manifest_path": last_manifest_path,
                    "last_pointcloud_process_status": last_pointcloud_process_status,
                    "last_postprocessed_frame_count": int(last_postprocessed_frame_count),
                    "last_pose_delta": last_pose_delta,
                    "last_accepted_pose": last_accepted_pose or {},
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                self.route6_write_state_artifact()
                self.route6_update_map_status_var.set(
                    "Route 6 Update Map: realtime waiting for movement "
                    f"move={last_pose_delta.get('position_cm', 0.0):.1f}cm yaw={last_pose_delta.get('yaw_deg', 0.0):.1f}deg "
                    f"skipped={skipped_stationary_count}"
                )
                if max_iterations is not None and loop_count >= int(max_iterations):
                    break
                if self.route6_update_map_realtime_stop_event.wait(self.route6_update_map_capture_interval_s()):
                    break
                continue
            try:
                capture = self.route6_update_map_capture_once(session, out_path)
                capture_count += 1
                pointcloud_report = self.route6_update_map_ensure_capture_pointcloud(capture, out_path)
                last_pointcloud_process_status = str(pointcloud_report.get("status", "") or "")
                last_postprocessed_frame_count = int(pointcloud_report.get("postprocessed_frame_count", 0) or 0)
                build = self.route6_update_map_build_from_pointcloud(out_path)
                if build:
                    map_count += 1
                    last_manifest_path = str(build.get("manifest_path", "") or "")
                    self.route6_update_map_request_window_refresh()
                last_accepted_pose = current_pose
            except Exception as exc:
                final_error = str(exc)
                self.route6_update_map_status_var.set(f"Route 6 Update Map: realtime update failed: {exc}")
                self.route6_log_event(out_path, "route6_update_map_realtime_error", {"error": final_error})
                break

            self.llm_route6_state["route6_update_map_realtime"] = {
                "schema": "route6_update_map_realtime_state_v1",
                "running": True,
                "output_dir": str(out_path),
                "capture_count": int(capture_count),
                "map_count": int(map_count),
                "skipped_stationary_count": int(skipped_stationary_count),
                "last_manifest_path": last_manifest_path,
                "last_pointcloud_process_status": last_pointcloud_process_status,
                "last_postprocessed_frame_count": int(last_postprocessed_frame_count),
                "last_pose_delta": last_pose_delta,
                "last_accepted_pose": last_accepted_pose or {},
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.route6_write_state_artifact()
            if max_iterations is not None and loop_count >= int(max_iterations):
                break
            if self.route6_update_map_realtime_stop_event.wait(self.route6_update_map_capture_interval_s()):
                break

        final_state = {
            "schema": "route6_update_map_realtime_state_v1",
            "running": False,
            "output_dir": str(out_path),
            "capture_count": int(capture_count),
            "map_count": int(map_count),
            "skipped_stationary_count": int(skipped_stationary_count),
            "last_manifest_path": last_manifest_path,
            "last_pointcloud_process_status": last_pointcloud_process_status,
            "last_postprocessed_frame_count": int(last_postprocessed_frame_count),
            "last_pose_delta": last_pose_delta,
            "last_accepted_pose": last_accepted_pose or {},
            "last_error": final_error,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.llm_route6_state["route6_update_map_realtime"] = final_state
        capture_state = self.llm_route6_state.get("route6_update_map_capture", {}) if isinstance(self.llm_route6_state.get("route6_update_map_capture", {}), dict) else {}
        if capture_state:
            capture_state["running"] = False
            capture_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.llm_route6_state["route6_update_map_capture"] = capture_state
        self.route6_write_state_artifact()
        if final_error:
            self.route6_update_map_status_var.set(f"Route 6 Update Map: realtime stopped with error after frames={capture_count} maps={map_count}")
        else:
            self.route6_update_map_status_var.set(f"Route 6 Update Map: realtime stopped frames={capture_count} maps={map_count} -> {out_path}")
        return final_state

    def on_route6_update_map_start_capture(self) -> None:
        self.ensure_route6_state()
        thread = getattr(self, "route6_update_map_capture_thread", None)
        if thread is not None and thread.is_alive():
            self.route6_update_map_status_var.set("Route 6 Update Map: capture already running.")
            return
        realtime_thread = getattr(self, "route6_update_map_realtime_thread", None)
        if realtime_thread is not None and realtime_thread.is_alive():
            self.route6_update_map_status_var.set("Route 6 Update Map: realtime update already running.")
            return
        session = getattr(self, "session", None)
        if session is None:
            self.route6_update_map_status_var.set("Route 6 Update Map: no active session for capture.")
            return
        output_dir = self.make_route6_update_map_output_dir()
        self.route6_record_update_map_output_dir(output_dir, source="manual_capture")
        self.llm_route6_state["route6_update_map_capture_frame_index"] = 0
        self.route6_update_map_capture_stop_event.clear()
        self.route6_update_map_capture_thread = threading.Thread(
            target=lambda: self.route6_update_map_capture_worker(session, output_dir),
            daemon=True,
        )
        self.route6_update_map_capture_thread.start()
        self.route6_update_map_status_var.set(f"Route 6 Update Map: capture started -> {output_dir}")

    def on_route6_update_map_stop_capture(self) -> None:
        self.ensure_route6_state()
        self.route6_update_map_capture_stop_event.set()
        self.route6_update_map_status_var.set("Route 6 Update Map: stop capture requested.")

    def on_route6_update_map_start_realtime(self) -> None:
        self.ensure_route6_state()
        realtime_thread = getattr(self, "route6_update_map_realtime_thread", None)
        if realtime_thread is not None and realtime_thread.is_alive():
            self.route6_update_map_status_var.set("Route 6 Update Map: realtime update already running.")
            return
        capture_thread = getattr(self, "route6_update_map_capture_thread", None)
        if capture_thread is not None and capture_thread.is_alive():
            self.route6_update_map_status_var.set("Route 6 Update Map: capture already running; stop capture before realtime update.")
            return
        session = getattr(self, "session", None)
        if session is None:
            self.route6_update_map_status_var.set("Route 6 Update Map: no active session for realtime update.")
            return
        output_dir = self.make_route6_update_map_output_dir()
        self.route6_record_update_map_output_dir(output_dir, source="manual_realtime")
        self.llm_route6_state["route6_update_map_capture_frame_index"] = 0
        self.route6_update_map_capture_stop_event.clear()
        self.route6_update_map_realtime_stop_event.clear()
        self.route6_update_map_realtime_thread = threading.Thread(
            target=lambda: self.route6_update_map_realtime_worker(session, output_dir),
            daemon=True,
        )
        self.route6_update_map_realtime_thread.start()
        self.route6_update_map_status_var.set(f"Route 6 Update Map: realtime update started -> {output_dir}")

    def on_route6_update_map_stop_realtime(self) -> None:
        self.ensure_route6_state()
        self.route6_update_map_realtime_stop_event.set()
        self.route6_update_map_status_var.set("Route 6 Update Map: stop realtime update requested.")

    def on_route6_update_map_generate_map(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        result = self.route6_update_map_build_from_pointcloud()
        if result:
            self.refresh_route6_update_map_window()
        else:
            root = self.route6_output_root()
            self.route6_update_map_status_var.set(
                f"Route 6 Update Map: no map generated. Open Capture Folders, select a run under {root}, then Generate Map or Load Map."
            )
        return result

    def on_route6_apply_known_houses_to_map(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        output_dir = self.route6_update_map_latest_output_dir()
        if output_dir is None:
            self.route6_update_map_status_var.set("Route 6 Update Map: no map folder for known house overlay.")
            return {}
        manifest = self.route6_apply_known_house_polygons_to_update_map(output_dir)
        if manifest:
            self.route6_build_known_house_navigation_plan(output_dir, target_house_id="001")
            self.refresh_route6_update_map_window(build_if_missing=False)
        return manifest

    def route6_update_map_collect_pointclouds(self, output_dir: Path, *, voxel_size_m: Optional[float] = None) -> Tuple[np.ndarray, List[str]]:
        out_path = Path(output_dir)
        paths: List[Path] = []
        map_merged_path = out_path / "map" / "route6_update_map_merged_point_cloud_world_standard_m.npy"
        if map_merged_path.is_file():
            paths.append(map_merged_path)
        states = self.route6_house_states()
        for _hid, state in sorted(states.items(), key=lambda item: str(item[0])):
            item = state if isinstance(state, dict) else {}
            pointcloud_path = Path(str(item.get("merged_pointcloud_path", "") or ""))
            if pointcloud_path.is_file() and self.route6_path_is_under(pointcloud_path, out_path) and pointcloud_path not in paths:
                paths.append(pointcloud_path)
        for path in self.route6_capture_folder_pointcloud_paths(out_path):
            if path not in paths:
                paths.append(path)
        clouds: List[np.ndarray] = []
        used_paths: List[str] = []
        raw_point_count = 0
        per_source_voxel_point_count = 0
        voxel_size = float(voxel_size_m if voxel_size_m is not None else route6_map_builder.DEFAULT_ROUTE6_UPDATE_MAP_VOXEL_SIZE_M)
        for path in paths:
            try:
                cloud = np.asarray(np.load(path), dtype=np.float32)
            except Exception as exc:
                self.route6_log_event(out_path, "route6_update_map_pointcloud_warning", {"path": str(path), "error": str(exc)})
                continue
            if cloud.ndim != 2 or cloud.shape[0] <= 0 or cloud.shape[1] < 3:
                continue
            if cloud.shape[1] < 6:
                pad = np.zeros((cloud.shape[0], 6 - cloud.shape[1]), dtype=np.float32)
                cloud = np.hstack([cloud[:, :3], pad])
            raw_point_count += int(cloud.shape[0])
            reduced = route6_map_builder.voxel_downsample_point_cloud(
                cloud[:, :6],
                voxel_size_m=voxel_size,
                fixed_world_bounds_cm=route6_map_builder.DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
            )
            if reduced.shape[0] <= 0:
                continue
            per_source_voxel_point_count += int(reduced.shape[0])
            clouds.append(reduced.astype(np.float32, copy=False))
            used_paths.append(str(path))
        if not clouds:
            self.llm_route6_state["route6_update_map_pointcloud_merge"] = {
                "schema": "route6_update_map_pointcloud_merge_v1",
                "raw_point_count": int(raw_point_count),
                "per_source_voxel_point_count": int(per_source_voxel_point_count),
                "pre_global_voxel_point_count": 0,
                "merged_point_count": 0,
                "voxel_size_m": float(voxel_size),
                "max_points": int(route6_map_builder.DEFAULT_ROUTE6_UPDATE_MAP_MAX_POINTS),
                "source_pointcloud_count": len(used_paths),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            return np.zeros((0, 6), dtype=np.float32), []
        merged = np.concatenate(clouds, axis=0)
        pre_global_voxel_point_count = int(merged.shape[0])
        max_points = int(route6_map_builder.DEFAULT_ROUTE6_UPDATE_MAP_MAX_POINTS)
        merged = route6_map_builder.voxel_downsample_point_cloud(
            merged,
            voxel_size_m=voxel_size,
            fixed_world_bounds_cm=route6_map_builder.DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
            max_points=max_points,
        )
        self.llm_route6_state["route6_update_map_pointcloud_merge"] = {
            "schema": "route6_update_map_pointcloud_merge_v1",
            "raw_point_count": int(raw_point_count),
            "per_source_voxel_point_count": int(per_source_voxel_point_count),
            "pre_global_voxel_point_count": int(pre_global_voxel_point_count),
            "merged_point_count": int(merged.shape[0]),
            "voxel_size_m": float(voxel_size),
            "max_points": int(max_points),
            "source_pointcloud_count": len(used_paths),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return merged.astype(np.float32, copy=False), used_paths

    def route6_cleanup_update_map_frame_pointclouds(
        self,
        output_dir: Path,
        source_paths: List[str],
        *,
        reason: str = "merged_into_route6_update_map",
    ) -> Dict[str, Any]:
        out_path = Path(output_dir)
        deleted_paths: List[str] = []
        skipped_paths: List[Dict[str, str]] = []
        delete_candidates: List[Path] = []
        for raw_path in source_paths:
            path = Path(str(raw_path or ""))
            frame_dir = None
            for parent in [path.parent, *path.parents]:
                if parent.name.startswith("frame_") and self.route6_path_is_under(parent, out_path / "frames"):
                    frame_dir = parent
                    break
            if frame_dir is None:
                skipped_paths.append({"path": str(path), "reason": "not_frame_pointcloud"})
                continue
            for pattern in ("point_cloud*.npy", "point_cloud*.ply"):
                for candidate in sorted(frame_dir.rglob(pattern)):
                    if candidate.is_file() and candidate not in delete_candidates:
                        delete_candidates.append(candidate)
        for path in delete_candidates:
            if not self.route6_path_is_under(path, out_path / "frames"):
                skipped_paths.append({"path": str(path), "reason": "outside_frames"})
                continue
            try:
                path.unlink()
                deleted_paths.append(str(path))
            except Exception as exc:
                skipped_paths.append({"path": str(path), "reason": f"delete_failed:{exc}"})
        report = {
            "schema": "route6_update_map_pointcloud_cleanup_v1",
            "reason": str(reason),
            "output_dir": str(out_path),
            "deleted_count": len(deleted_paths),
            "deleted_paths": deleted_paths,
            "skipped_count": len(skipped_paths),
            "skipped_paths": skipped_paths,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if deleted_paths or skipped_paths:
            self.route6_write_json_artifact(out_path / "map" / "route6_update_map_pointcloud_cleanup.json", report)
            self.route6_log_event(out_path, "route6_update_map_pointcloud_cleanup", report)
            self.llm_route6_state["route6_update_map_pointcloud_cleanup"] = self.route6_json_safe(report)
            update_state = self.llm_route6_state.get("route6_update_map", {}) if isinstance(self.llm_route6_state.get("route6_update_map", {}), dict) else {}
            if update_state:
                update_state["pointcloud_cleanup"] = self.route6_json_safe(report)
                self.llm_route6_state["route6_update_map"] = update_state
            self.route6_write_state_artifact()
        return report

    def route6_update_map_manifest_path(self, output_dir: Optional[Path] = None) -> Path:
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        if out_path is None:
            return Path()
        state_record = (self.llm_route6_state or {}).get("route6_update_map", {})
        candidate = Path(str((state_record if isinstance(state_record, dict) else {}).get("manifest_path", "") or ""))
        if candidate.is_file() and (output_dir is None or self.route6_path_is_under(candidate, out_path)):
            return candidate
        return out_path / "map" / "layered_occupancy" / "route6_layered_occupancy.json"

    def route6_update_map_build_from_pointcloud(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        if out_path is None:
            self.route6_update_map_status_var.set("Route 6 Update Map: no Route 6 output directory found.")
            return {}
        resolution_m = self.route6_float_param(self.llm_route6_occupancy_resolution_m_var, 0.25, min_value=0.05, max_value=5.0)
        merged, source_paths = self.route6_update_map_collect_pointclouds(out_path, voxel_size_m=float(resolution_m))
        if merged.shape[0] <= 0:
            self.route6_update_map_status_var.set(f"Route 6 Update Map: no pointcloud found under {out_path}")
            return {}
        result = route6_map_builder.write_route6_layered_occupancy_artifacts(
            out_path,
            merged,
            layer_z_cm=route6_map_builder.DEFAULT_ROUTE6_LAYER_Z_CM,
            layer_band_cm=route6_map_builder.DEFAULT_ROUTE6_LAYER_BAND_CM,
            resolution_m=float(resolution_m),
            occupied_threshold=route6_map_builder.DEFAULT_ROUTE6_LAYER_OCCUPIED_THRESHOLD,
        )
        merged_pointcloud_path = out_path / "map" / "route6_update_map_merged_point_cloud_world_standard_m.npy"
        merged_pointcloud_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(merged_pointcloud_path, merged.astype(np.float32, copy=False))
        overlay_manifest = self.route6_apply_known_house_polygons_to_update_map(out_path)
        manifest_path = str((overlay_manifest if isinstance(overlay_manifest, dict) else {}).get("manifest_path", "") or result.get("manifest_path", "") or "")
        cleanup = self.route6_cleanup_update_map_frame_pointclouds(out_path, source_paths)
        record = {
            "schema": "route6_update_map_state_v1",
            "source": "manual_route6_update_map",
            "output_dir": str(out_path),
            "manifest_path": manifest_path,
            "layered_occupancy_dir": str(result.get("layered_occupancy_dir", "") or ""),
            "layer_count": int(result.get("layer_count", 0) or 0),
            "fixed_world_bounds_cm": dict(result.get("fixed_world_bounds_cm", {}) or {}),
            "source_pointcloud_count": len(source_paths),
            "source_pointcloud_paths": source_paths,
            "merged_pointcloud_path": str(merged_pointcloud_path),
            "pointcloud_cleanup": self.route6_json_safe(cleanup),
            "raw_point_count": int((self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}) if isinstance(self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}), dict) else {}).get("raw_point_count", merged.shape[0]) or 0),
            "per_source_voxel_point_count": int((self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}) if isinstance(self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}), dict) else {}).get("per_source_voxel_point_count", merged.shape[0]) or 0),
            "pre_global_voxel_point_count": int((self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}) if isinstance(self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}), dict) else {}).get("pre_global_voxel_point_count", merged.shape[0]) or 0),
            "merged_point_count": int(merged.shape[0]),
            "pointcloud_voxel_size_m": float(resolution_m),
            "pointcloud_max_points": int(route6_map_builder.DEFAULT_ROUTE6_UPDATE_MAP_MAX_POINTS),
            "pointcloud_reduction_ratio": round(float(merged.shape[0]) / max(1.0, float((self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}) if isinstance(self.llm_route6_state.get("route6_update_map_pointcloud_merge", {}), dict) else {}).get("raw_point_count", merged.shape[0]) or merged.shape[0])), 6),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_record_update_map_output_dir(out_path, source=str(record.get("source", "route6_update_map_build") or "route6_update_map_build"))
        self.llm_route6_state["route6_update_map"] = record
        self.route6_write_state_artifact()
        self.route6_update_map_status_var.set(
            f"Route 6 Update Map: layers={record['layer_count']} points={record['merged_point_count']} raw={record['raw_point_count']}"
        )
        return record

    def route6_update_map_load_manifest(self, *, build_if_missing: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        out_path = Path(output_dir) if output_dir is not None else self.route6_update_map_latest_output_dir()
        if out_path is None:
            return {}
        manifest_path = self.route6_update_map_manifest_path(out_path)
        if not manifest_path.is_file() and build_if_missing:
            self.route6_update_map_build_from_pointcloud(out_path)
            manifest_path = self.route6_update_map_manifest_path(out_path)
        if not manifest_path.is_file():
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            self.route6_update_map_status_var.set(f"Route 6 Update Map: failed to load manifest: {exc}")
            return {}

    def _route6_update_map_layer_key(self, layer: Dict[str, Any]) -> str:
        try:
            return f"z_{int(float(layer.get('z_cm', 0))):03d}"
        except Exception:
            return "z_000"

    def route6_update_map_uav_pose_text(self) -> str:
        pose = self.route6_current_pose()
        text = (
            f"UAV x={float(pose.get('x', 0.0) or 0.0):.1f}cm "
            f"y={float(pose.get('y', 0.0) or 0.0):.1f}cm "
            f"z={float(pose.get('z', 0.0) or 0.0):.1f}cm "
            f"yaw={float(pose.get('yaw', 0.0) or 0.0):.1f}deg"
        )
        self.route6_update_map_pose_var.set(text)
        return text

    def route6_update_map_load_layer_metadata(self, layer_record: Dict[str, Any]) -> Dict[str, Any]:
        metadata_path = Path(str((layer_record if isinstance(layer_record, dict) else {}).get("occupancy_metadata_path", "") or ""))
        if not metadata_path.is_file():
            return {}
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def route6_heading_label_from_yaw(self, yaw_deg: float) -> str:
        yaw = float(yaw_deg) % 360.0
        labels = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
        index = int(round(yaw / 45.0)) % len(labels)
        return labels[index]

    def route6_yaw_screen_vector(self, yaw_deg: float) -> Tuple[float, float]:
        yaw_rad = math.radians(float(yaw_deg))
        return math.cos(yaw_rad), math.sin(yaw_rad)

    def route6_draw_update_map_compass_overlay(self, draw: ImageDraw.ImageDraw, image: Image.Image, yaw_deg: float, *, scale: int = 1) -> None:
        factor = max(1, int(scale))
        margin = max(5, 5 * factor)
        radius = max(12, 11 * factor)
        cx = margin + radius
        cy = margin + radius
        box_pad = max(4, 4 * factor)
        box_right = min(image.width - 1, cx + radius + max(44, 28 * factor))
        box_bottom = min(image.height - 1, cy + radius + max(18, 10 * factor))
        draw.rectangle((margin - box_pad, margin - box_pad, box_right, box_bottom), fill=(255, 255, 255), outline=(120, 120, 120), width=max(1, factor))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(60, 60, 60), width=max(1, factor))
        draw.line((cx, cy - radius, cx, cy + radius), fill=(150, 150, 150), width=max(1, factor))
        draw.line((cx - radius, cy, cx + radius, cy), fill=(150, 150, 150), width=max(1, factor))
        draw.text((cx - 3 * factor, cy - radius - 1), "N", fill=(0, 0, 0))
        draw.text((cx + radius + 2, cy - 5), "E", fill=(0, 0, 0))
        draw.text((cx - 3 * factor, cy + radius + 1), "S", fill=(0, 0, 0))
        draw.text((max(1, cx - radius - 7 * factor), cy - 5), "W", fill=(0, 0, 0))
        dx, dy = self.route6_yaw_screen_vector(yaw_deg)
        tip_x = cx + dx * (radius - max(3, factor))
        tip_y = cy + dy * (radius - max(3, factor))
        tail_x = cx - dx * max(3, radius * 0.25)
        tail_y = cy - dy * max(3, radius * 0.25)
        draw.line((tail_x, tail_y, tip_x, tip_y), fill=(220, 0, 0), width=max(2, factor * 2))
        left_x = tip_x - dx * max(5, factor * 4) + dy * max(4, factor * 3)
        left_y = tip_y - dy * max(5, factor * 4) - dx * max(4, factor * 3)
        right_x = tip_x - dx * max(5, factor * 4) - dy * max(4, factor * 3)
        right_y = tip_y - dy * max(5, factor * 4) + dx * max(4, factor * 3)
        draw.polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], fill=(220, 0, 0))
        heading = self.route6_heading_label_from_yaw(yaw_deg)
        draw.text((cx + radius + 8, cy + 6), f"{heading} {float(yaw_deg):.0f}deg", fill=(0, 0, 0))

    def route6_draw_update_map_uav_overlay(self, image: Image.Image, layer_record: Dict[str, Any], *, scale: int = 1) -> Image.Image:
        metadata = self.route6_update_map_load_layer_metadata(layer_record)
        if not metadata:
            return image
        try:
            width = int(metadata.get("width", 0) or 0)
            height = int(metadata.get("height", 0) or 0)
            resolution = float(metadata.get("resolution_m", 0.25) or 0.25)
            origin_x, origin_y = [float(value) for value in metadata.get("origin_standard_m", [0.0, 0.0])]
        except Exception:
            return image
        if width <= 0 or height <= 0 or resolution <= 0:
            return image
        pose = self.route6_current_pose()
        standard_x_m = float(pose.get("x", 0.0) or 0.0) / 100.0
        standard_y_m = -float(pose.get("y", 0.0) or 0.0) / 100.0
        col = int(math.floor((standard_x_m - origin_x) / resolution))
        row = int(math.floor((standard_y_m - origin_y) / resolution))
        if col < 0 or col >= width or row < 0 or row >= height:
            return image
        draw = ImageDraw.Draw(image)
        factor = max(1, int(scale))
        x_px = int(col * factor + factor / 2)
        y_px = int((height - 1 - row) * factor + factor / 2)
        radius = max(4, factor * 2)
        yaw_deg = float(pose.get("yaw", pose.get("yaw_deg", 0.0)) or 0.0)
        color = (220, 0, 0)
        self.route6_draw_update_map_compass_overlay(draw, image, yaw_deg, scale=scale)
        dx, dy = self.route6_yaw_screen_vector(yaw_deg)
        arrow_len = max(radius * 3.0, factor * 12.0)
        tip_x = x_px + dx * arrow_len
        tip_y = y_px + dy * arrow_len
        tail_x = x_px - dx * max(2.0, radius * 0.4)
        tail_y = y_px - dy * max(2.0, radius * 0.4)
        draw.line((tail_x, tail_y, tip_x, tip_y), fill=color, width=max(2, factor * 2))
        head_len = max(5.0, factor * 4.0)
        head_w = max(4.0, factor * 3.0)
        left_x = tip_x - dx * head_len + dy * head_w
        left_y = tip_y - dy * head_len - dx * head_w
        right_x = tip_x - dx * head_len - dy * head_w
        right_y = tip_y - dy * head_len + dx * head_w
        draw.polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], fill=color)
        draw.line((x_px - radius, y_px, x_px + radius, y_px), fill=color, width=max(2, factor))
        draw.line((x_px, y_px - radius, x_px, y_px + radius), fill=color, width=max(2, factor))
        draw.ellipse((x_px - radius, y_px - radius, x_px + radius, y_px + radius), outline=color, width=max(2, factor))
        return image

    def refresh_route6_update_map_window(self, *, build_if_missing: bool = True) -> None:
        self.ensure_route6_state()
        self.route6_update_map_uav_pose_text()
        manifest = self.route6_update_map_load_manifest(build_if_missing=build_if_missing)
        combo = getattr(self, "route6_update_map_layer_combo", None)
        preview = getattr(self, "route6_update_map_preview_label", None)
        if not manifest:
            self.route6_update_map_status_var.set("Route 6 Update Map: no layered map artifact yet.")
            if preview is not None:
                try:
                    preview.configure(text="No layered occupancy map available.", image="")
                except tk.TclError:
                    pass
            return
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        values = [self._route6_update_map_layer_key(layer) for layer in layers if isinstance(layer, dict)]
        if combo is not None:
            try:
                combo.configure(values=values)
            except tk.TclError:
                pass
        selected = str(self.route6_update_map_layer_var.get() or "")
        if selected not in values and values:
            selected = values[0]
            self.route6_update_map_layer_var.set(selected)
        layer_record = next((layer for layer in layers if isinstance(layer, dict) and self._route6_update_map_layer_key(layer) == selected), {})
        preview_path = self.route6_update_map_layer_preview_path(layer_record)
        status = (
            f"Route 6 Update Map: {selected} "
            f"points={int((layer_record or {}).get('point_count', 0) or 0)} "
            f"occupied={int((layer_record or {}).get('occupied_cell_count', 0) or 0)}"
        )
        self.route6_update_map_status_var.set(status)
        if preview is None:
            return
        if not preview_path.is_file():
            try:
                preview.configure(text=f"Preview missing: {preview_path}", image="")
            except tk.TclError:
                pass
            return
        try:
            image = Image.open(preview_path).convert("RGB")
            width, height = image.size
            scale = max(1, min(8, int(780 / max(1, max(width, height)))))
            if scale > 1:
                image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
            image = self.route6_draw_update_map_uav_overlay(image, layer_record, scale=scale)
            photo = ImageTk.PhotoImage(image)
            self.route6_update_map_preview_photo = photo
            preview.configure(image=photo, text="")
        except Exception as exc:
            try:
                preview.configure(text=f"Preview load failed: {exc}", image="")
            except tk.TclError:
                pass

    def route6_update_map_schedule_pose_refresh(self) -> None:
        self.ensure_route6_state()
        window = getattr(self, "route6_update_map_window", None)
        if window is None:
            return
        try:
            if not window.winfo_exists():
                return
            self.refresh_route6_update_map_window(build_if_missing=False)
            self.route6_update_map_pose_after_id = window.after(1000, self.route6_update_map_schedule_pose_refresh)
        except tk.TclError:
            self.route6_update_map_pose_after_id = None

    def _on_route6_update_map_mousewheel(self, event: tk.Event):
        canvas = getattr(self, "route6_update_map_scroll_canvas", None)
        if canvas is None:
            return None
        delta = -1 if int(getattr(event, "delta", 0) or 0) > 0 else 1
        try:
            if int(getattr(event, "state", 0) or 0) & 0x0001:
                canvas.xview_scroll(delta, "units")
            else:
                canvas.yview_scroll(delta, "units")
            return "break"
        except tk.TclError:
            return None

    def _on_route6_update_map_mousewheel_linux(self, event: tk.Event):
        canvas = getattr(self, "route6_update_map_scroll_canvas", None)
        if canvas is None:
            return None
        direction = -1 if int(getattr(event, "num", 0) or 0) == 4 else 1
        try:
            canvas.yview_scroll(direction, "units")
            return "break"
        except tk.TclError:
            return None

    def _bind_route6_update_map_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_route6_update_map_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_route6_update_map_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_route6_update_map_mousewheel_linux, add="+")
        except tk.TclError:
            return
        for child in widget.winfo_children():
            self._bind_route6_update_map_mousewheel_tree(child)

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
                "reset movement trigger = front_blocked only; other problems are reported without changing P_obs",
                "if front_blocked: reset_distance_candidates = base_distance_cm - k * reset_step_cm, then outward fallback",
                "reset commit policy = commit P_reset only when reset_status == ok; failed candidates are reported but not written back to the plan",
                "multi-point reset target = S_i scan_observation_points; selected_observation is reference-only for edge sorting",
                "P_reset = edge outward-normal offset from the same anchor using each reset_distance candidate",
            ],
            "code_sources": [
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_test_planner_nearest_point_on_segment",
                    "purpose": "projects the UAV point onto an exploration edge segment to get P_edge",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_observation_point_for_edge",
                    "purpose": "offsets P_edge outward by radar distance d to get P_obs",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_exploration_edges_for_bbox",
                    "purpose": "enumerates south/east/north/west edges, computes yaw and edge distance",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_apply_test_planner_algorithm_to_edge",
                    "purpose": "turns each edge into a paper-inspired navigation-point candidate and algorithm score",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_test_planner_algorithm_score_formula",
                    "purpose": "documents the scoring formula for each selectable algorithm",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_build_scan_coverage_plan_for_edge",
                    "purpose": "samples multiple observation/navigation points until the selected edge meets scan coverage",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_scan_coverage_parameters",
                    "purpose": "computes FOV-derived coverage width and effective scan step",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_test_planner_reset_distance_candidates",
                    "purpose": "generates inward/toward-edge reset distances only when the observation ray is blocked",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_test_planner_front_blocked_report",
                    "purpose": "samples the scan ray and ignores target-edge-adjacent facade cells before reporting front_blocked",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_test_planner_observation_safety_report",
                    "purpose": "reports inside-house, near-obstacle, front-blocked, and map-boundary reset problems",
                },
                {
                    "file": "control/route6_explore_control.py",
                    "symbol": "route6_build_offline_test_plan",
                    "purpose": "sorts houses/edges, writes route6_offline_test_plan.json",
                },
                {
                    "file": "control/route6_test_planner_analysis/engine.py",
                    "symbol": "reset_summary",
                    "purpose": "summarizes main/scan reset outcomes and keeps multi-point base observations reference-only",
                },
                {
                    "file": "control/route6_explore_control.py",
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
        content.grid_rowconfigure(2, weight=1)
        content.grid_rowconfigure(3, weight=1)

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
        toolbar.grid_columnconfigure(13, weight=1)
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
        tk.Button(toolbar, text="Analyze Search Task", command=self.on_route6_test_planner_analyze).grid(row=0, column=10, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Show Formula", command=self.on_route6_test_planner_show_formula).grid(row=0, column=11, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Reset Current Observation Point", command=self.on_route6_test_planner_reset_current_observation_point).grid(row=0, column=12, sticky="w", padx=6, pady=6)
        tk.Label(toolbar, text="Algorithm").grid(row=1, column=0, sticky="e", padx=6, pady=(0, 6))
        algorithm_combo = ttk.Combobox(
            toolbar,
            textvariable=self.route6_test_planner_algorithm_var,
            values=self.route6_test_planner_algorithm_options(),
            state="readonly",
            width=24,
        )
        algorithm_combo.grid(row=1, column=1, columnspan=3, sticky="w", padx=6, pady=(0, 6))
        tk.Label(toolbar, textvariable=self.route6_test_planner_status_var, anchor="w").grid(row=1, column=4, columnspan=10, sticky="ew", padx=6, pady=(0, 6))
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

        house_frame = tk.LabelFrame(content, text="Houses")
        house_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        house_frame.grid_columnconfigure(0, weight=1)
        house_list = tk.Listbox(house_frame, selectmode="extended", exportselection=False, height=6)
        house_scroll = tk.Scrollbar(house_frame, orient="vertical", command=house_list.yview)
        house_list.configure(yscrollcommand=house_scroll.set)
        house_list.grid(row=0, column=0, sticky="ew", padx=(6, 0), pady=6)
        house_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)

        preview_frame = tk.LabelFrame(content, text="Route 6 Map Preview")
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_label = tk.Label(preview_frame, text="Load a Route 6 Update Map to preview the offline test plan.", anchor="center", justify="center")
        preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        result_frame = tk.LabelFrame(content, text="Calculation Process / Result")
        result_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 8))
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
        self.route6_test_planner_preview_label = preview_label
        self.route6_test_planner_result_text = result_text
        self.refresh_route6_test_planner_house_list()
        self.refresh_route6_test_planner_result_text()
        self.refresh_route6_test_planner_preview()
        self._bind_route6_test_planner_mousewheel_tree(window)

    def close_route6_test_planner_window(self) -> None:
        self.ensure_route6_state()
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
        self.route6_test_planner_preview_photo = None
        self.route6_test_planner_layer_combo = None
        self.route6_test_planner_algorithm_combo = None
        self.route6_test_planner_scan_mode_combo = None
        self.route6_test_planner_preview_frame = None

    def open_route6_update_map_window(self) -> None:
        self.ensure_route6_state()
        if self.route6_update_map_window is not None and self.route6_update_map_window.winfo_exists():
            self.route6_update_map_window.lift()
            self.route6_update_map_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Route 6 Update Map")
        window.geometry("980x760")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close_route6_update_map_window)

        route6_update_map_scroll_canvas = tk.Canvas(window, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(window, orient="vertical", command=route6_update_map_scroll_canvas.yview)
        h_scrollbar = tk.Scrollbar(window, orient="horizontal", command=route6_update_map_scroll_canvas.xview)
        route6_update_map_scroll_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        route6_update_map_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        content = tk.Frame(route6_update_map_scroll_canvas)
        content_window = route6_update_map_scroll_canvas.create_window((0, 0), window=content, anchor="nw")
        content.grid_columnconfigure(0, weight=1)

        def _sync_scrollregion(_event: tk.Event) -> None:
            try:
                route6_update_map_scroll_canvas.configure(scrollregion=route6_update_map_scroll_canvas.bbox("all"))
            except tk.TclError:
                pass

        def _sync_content_width(event: tk.Event) -> None:
            try:
                route6_update_map_scroll_canvas.itemconfigure(content_window, width=max(920, int(event.width)))
            except tk.TclError:
                pass

        content.bind("<Configure>", _sync_scrollregion)
        route6_update_map_scroll_canvas.bind("<Configure>", _sync_content_width)

        toolbar = tk.LabelFrame(content, text="Route 6 Update Map")
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        toolbar.grid_columnconfigure(14, weight=1)
        tk.Label(toolbar, text="Layer").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        route6_update_map_layer_combo = ttk.Combobox(
            toolbar,
            textvariable=self.route6_update_map_layer_var,
            values=[f"z_{int(value):03d}" for value in route6_map_builder.DEFAULT_ROUTE6_LAYER_Z_CM],
            state="readonly",
            width=10,
        )
        route6_update_map_layer_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        route6_update_map_layer_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_route6_update_map_window())
        tk.Button(toolbar, text="Start Capture", command=self.on_route6_update_map_start_capture).grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Stop Capture", command=self.on_route6_update_map_stop_capture).grid(row=0, column=3, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Generate Map", command=self.on_route6_update_map_generate_map).grid(row=0, column=4, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Start Realtime Update", command=self.on_route6_update_map_start_realtime).grid(row=0, column=5, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Stop Realtime Update", command=self.on_route6_update_map_stop_realtime).grid(row=0, column=6, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Open Capture Folders", command=self.open_route6_capture_folder_reader_window).grid(row=0, column=7, sticky="w", padx=6, pady=6)
        tk.Button(toolbar, text="Apply Known Houses", command=self.on_route6_apply_known_houses_to_map).grid(row=0, column=14, sticky="w", padx=6, pady=6)
        tk.Label(toolbar, text="Interval s").grid(row=0, column=8, sticky="e", padx=(18, 2), pady=6)
        tk.Entry(toolbar, textvariable=self.route6_update_map_capture_interval_s_var, width=6).grid(row=0, column=9, sticky="w", padx=(0, 6), pady=6)
        tk.Label(toolbar, text="Min move cm").grid(row=0, column=10, sticky="e", padx=(12, 2), pady=6)
        tk.Entry(toolbar, textvariable=self.route6_update_map_min_move_cm_var, width=6).grid(row=0, column=11, sticky="w", padx=(0, 6), pady=6)
        tk.Label(toolbar, text="Min yaw deg").grid(row=0, column=12, sticky="e", padx=(12, 2), pady=6)
        tk.Entry(toolbar, textvariable=self.route6_update_map_min_yaw_deg_var, width=6).grid(row=0, column=13, sticky="w", padx=(0, 6), pady=6)
        tk.Label(toolbar, textvariable=self.route6_update_map_status_var, anchor="w").grid(row=1, column=0, columnspan=15, sticky="ew", padx=6, pady=6)
        tk.Label(toolbar, textvariable=self.route6_update_map_pose_var, anchor="w").grid(row=2, column=0, columnspan=15, sticky="ew", padx=6, pady=(0, 6))

        preview_frame = tk.LabelFrame(content, text="Layer Preview")
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        preview_frame.grid_columnconfigure(0, weight=1)
        route6_update_map_preview_label = tk.Label(preview_frame, text="Loading layered map...", anchor="center", justify="center")
        route6_update_map_preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.route6_update_map_window = window
        self.route6_update_map_scroll_canvas = route6_update_map_scroll_canvas
        self.route6_update_map_content_frame = content
        self.route6_update_map_layer_combo = route6_update_map_layer_combo
        self.route6_update_map_preview_label = route6_update_map_preview_label
        self._bind_route6_update_map_mousewheel_tree(window)
        self.refresh_route6_update_map_window()
        self.route6_update_map_schedule_pose_refresh()

    def open_route6_capture_folder_reader_window(self) -> None:
        self.ensure_route6_state()
        if self.route6_capture_folder_reader_window is not None and self.route6_capture_folder_reader_window.winfo_exists():
            self.route6_capture_folder_reader_window.lift()
            self.route6_capture_folder_reader_window.focus_force()
            self.refresh_route6_capture_folder_list()
            return
        window = tk.Toplevel(self.root)
        window.title("Route 6 Capture Folder Reader")
        window.geometry("980x700")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(1, weight=1)
        window.grid_rowconfigure(3, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close_route6_capture_folder_reader_window)

        header = tk.Frame(window)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header.grid_columnconfigure(1, weight=1)
        tk.Label(header, text="Root").grid(row=0, column=0, sticky="w", padx=(0, 6))
        tk.Label(header, text=str(self.route6_output_root()), anchor="w").grid(row=0, column=1, sticky="ew")

        list_frame = tk.LabelFrame(window, text="Capture Folders")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        listbox = tk.Listbox(list_frame, exportselection=False, height=14)
        list_scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=list_scrollbar.set)
        listbox.grid(row=0, column=0, sticky="nsew")
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        listbox.bind("<<ListboxSelect>>", self.on_route6_capture_folder_select)

        actions = tk.Frame(window)
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        actions.grid_columnconfigure(4, weight=1)
        tk.Button(actions, text="Refresh", command=self.refresh_route6_capture_folder_list).grid(row=0, column=0, sticky="w", padx=(0, 6))
        tk.Button(actions, text="Process Pointcloud Data", command=self.on_route6_capture_folder_process_pointcloud).grid(row=0, column=1, sticky="w", padx=6)
        tk.Button(actions, text="Generate Map", command=self.on_route6_capture_folder_generate_map).grid(row=0, column=2, sticky="w", padx=6)
        tk.Button(actions, text="Load Map", command=self.on_route6_capture_folder_load_map).grid(row=0, column=3, sticky="w", padx=6)
        tk.Label(actions, textvariable=self.route6_selected_capture_folder_var, anchor="w").grid(row=0, column=4, sticky="ew", padx=10)

        report_frame = tk.LabelFrame(window, text="Pointcloud Report")
        report_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        report_frame.grid_columnconfigure(0, weight=1)
        report_frame.grid_rowconfigure(0, weight=1)
        report_text = tk.Text(report_frame, height=10, wrap="none", font=("Consolas", 9))
        report_y = tk.Scrollbar(report_frame, orient="vertical", command=report_text.yview)
        report_x = tk.Scrollbar(report_frame, orient="horizontal", command=report_text.xview)
        report_text.configure(yscrollcommand=report_y.set, xscrollcommand=report_x.set, state="disabled")
        report_text.grid(row=0, column=0, sticky="nsew")
        report_y.grid(row=0, column=1, sticky="ns")
        report_x.grid(row=1, column=0, sticky="ew")

        tk.Label(window, textvariable=self.route6_capture_folder_status_var, anchor="w").grid(row=4, column=0, sticky="ew", padx=8, pady=(4, 8))

        self.route6_capture_folder_reader_window = window
        self.route6_capture_folder_listbox = listbox
        self.route6_pointcloud_report_text = report_text
        self.refresh_route6_capture_folder_list()

    def close_route6_capture_folder_reader_window(self) -> None:
        self.ensure_route6_state()
        if self.route6_capture_folder_reader_window is not None:
            try:
                self.route6_capture_folder_reader_window.destroy()
            except Exception:
                pass
        self.route6_capture_folder_reader_window = None
        self.route6_capture_folder_listbox = None
        self.route6_pointcloud_report_text = None

    def close_route6_update_map_window(self) -> None:
        self.ensure_route6_state()
        self.route6_update_map_capture_stop_event.set()
        self.route6_update_map_realtime_stop_event.set()
        after_id = getattr(self, "route6_update_map_pose_after_id", None)
        if after_id is not None and self.route6_update_map_window is not None:
            try:
                self.route6_update_map_window.after_cancel(after_id)
            except Exception:
                pass
        self.route6_update_map_pose_after_id = None
        if self.route6_update_map_window is not None:
            try:
                self.route6_update_map_window.destroy()
            except Exception:
                pass
        self.route6_update_map_window = None
        self.route6_update_map_scroll_canvas = None
        self.route6_update_map_content_frame = None
        self.route6_update_map_layer_combo = None
        self.route6_update_map_preview_label = None
        self.route6_update_map_preview_photo = None

    def route6_run_entrance_search(
        self,
        output_dir: Path,
        house_id: str,
        house_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        house_dir = self.route6_house_output_dir(out_path, hid)
        entrance_dir = house_dir / "entrance"
        entrance_dir.mkdir(parents=True, exist_ok=True)
        self.route6_set_stage("SEARCH_ENTRANCE", f"preparing Route 6 entrance evidence for house={hid}")
        state = house_state if isinstance(house_state, dict) else {}
        scan_points = self.llm_route6_state.get("scan_points", []) if isinstance(self.llm_route6_state.get("scan_points"), list) else []
        manifest = self.route6_build_entrance_capture_manifest(out_path, hid, scan_points)
        analysis = self.route6_run_route5_capture_analysis_for_house(out_path, hid, manifest)
        analysis_ran = bool(analysis.get("ran", False))
        analysis_error = str(analysis.get("error", "") or "")
        route5_candidates = analysis.get("candidates", []) if isinstance(analysis.get("candidates", []), list) else []
        route5_candidate_count = int(analysis.get("candidate_count", len(route5_candidates)) or len(route5_candidates))
        close_confirm_plan = (
            self.route6_build_close_confirm_scan_plan(out_path, hid, route5_candidates, state)
            if route5_candidate_count > 0
            else {}
        )
        close_confirm_plan_path = str(entrance_dir / "close_confirm_scan_plan.json") if close_confirm_plan else ""
        yolo_manifest = {
            "schema": "route6_entrance_yolo_manifest_v1",
            "house_id": hid,
            "source": "route5_capture_analysis" if analysis_ran else "route6_capture_manifest",
            "included_count": int(manifest.get("included_count", 0) or 0),
            "excluded_count": int(manifest.get("excluded_count", 0) or 0),
            "included_captures": manifest.get("included_captures", []),
            "excluded_captures": manifest.get("excluded_captures", []),
            "recommended_next_tool": "route5_run_capture_analysis",
            "route5_analysis": analysis,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        candidates_payload = {
            "schema": "route6_entrance_candidates_v1",
            "house_id": hid,
            "candidate_count": route5_candidate_count,
            "candidates": route5_candidates,
            "source": "route5_capture_analysis" if analysis_ran else "pending_yolo_or_pointcloud_validation",
            "route5_entrance_candidates_path": str(analysis.get("entrance_candidates_path", "") or ""),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(entrance_dir / "yolo_manifest.json", yolo_manifest)
        self.route6_write_json_artifact(entrance_dir / "entrance_candidates.json", candidates_payload)
        self.route6_write_json_artifact(entrance_dir / "capture_manifest.json", manifest)
        corrected_config_path = str((state.get("map_artifacts", {}) if isinstance(state.get("map_artifacts", {}), dict) else {}).get("corrected_config_path", ""))
        polygons_path = str((state.get("map_artifacts", {}) if isinstance(state.get("map_artifacts", {}), dict) else {}).get("polygons_path", ""))
        coverage_report_path = str(state.get("coverage_report_path", "") or "")
        map_ready = str(state.get("status", "")) in {"mapped_complete", "mapped_partial"}
        included_count = int(manifest.get("included_count", 0) or 0)
        if not map_ready:
            status = "not_ready_for_entrance_search"
            recommended = "finish_route6_pointcloud_mapping"
            entry_search_complete = False
        elif analysis_error:
            status = "route5_capture_analysis_failed"
            recommended = "inspect_route5_capture_analysis_error"
            entry_search_complete = False
        elif analysis_ran and route5_candidate_count > 0:
            status = "entrance_candidates_need_close_confirm"
            recommended = "run_close_confirm_scan_and_obstacle_validation"
            entry_search_complete = False
        elif analysis_ran:
            status = "no_entry_candidate_after_full_coverage"
            recommended = "select_next_house_or_rescan_if_coverage_is_low"
            entry_search_complete = True
        elif included_count > 0:
            status = "ready_for_route5_capture_analysis"
            recommended = "run_route5_capture_analysis_on_route6_output"
            entry_search_complete = False
        else:
            status = "map_ready_waiting_for_rgb_depth_analysis"
            recommended = "capture_rgb_depth_pointcloud_with_route5_guard"
            entry_search_complete = False
        report = {
            "schema": "route6_entrance_validation_report_v1",
            "house_id": hid,
            "status": status,
            "entry_search_complete": entry_search_complete,
            "recommended_next_action": recommended,
            "corrected_config_path": corrected_config_path,
            "polygons_path": polygons_path,
            "coverage_report_path": coverage_report_path,
            "yolo_manifest_path": str(entrance_dir / "yolo_manifest.json"),
            "entrance_candidates_path": str(entrance_dir / "entrance_candidates.json"),
            "capture_manifest_path": str(entrance_dir / "capture_manifest.json"),
            "close_confirm_scan_plan_path": close_confirm_plan_path,
            "route5_analysis_summary_path": str(analysis.get("summary_path", "") or ""),
            "route5_analysis_status": str(analysis.get("status", "") or ""),
            "route5_analysis_error": analysis_error,
            "included_capture_count": included_count,
            "excluded_capture_count": int(manifest.get("excluded_count", 0) or 0),
            "candidate_count": route5_candidate_count,
            "close_confirm_planned_scan_count": int(close_confirm_plan.get("planned_scan_count", 0) or 0) if close_confirm_plan else 0,
            "notes": [
                "Route 6 keeps entrance search open until close-confirm scan and obstacle validation finish.",
            ],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(entrance_dir / "entrance_validation_report.json", report)
        house_states = self.llm_route6_state.setdefault("house_states", {})
        current = dict(house_states.get(hid, state) if isinstance(house_states.get(hid, state), dict) else state)
        current["entrance_status"] = status
        if route5_candidate_count > 0:
            current["search_status"] = "pending_close_confirm_scan"
            current["entrance_status"] = "candidates_need_close_confirm"
            current["close_confirm_scan_plan_path"] = close_confirm_plan_path
        elif entry_search_complete:
            current["status"] = "searched_no_entry"
            current["search_status"] = "searched_no_entry"
            current["entrance_status"] = "no_entry_candidate_after_full_coverage"
        else:
            current["search_status"] = "pending_capture_analysis" if included_count > 0 else "pending_entrance_capture_evidence"
        current["entrance_report_path"] = str(entrance_dir / "entrance_validation_report.json")
        current["route5_analysis_summary_path"] = str(analysis.get("summary_path", "") or "")
        current["entrance_candidate_count"] = route5_candidate_count
        current["updated_at"] = datetime.now().isoformat(timespec="seconds")
        house_states[hid] = current
        self.llm_route6_state["last_entrance_report"] = report
        self.route6_write_json_artifact(house_dir / "house_state.json", current)
        self.route6_write_run_artifacts(out_path)
        self.route6_write_state_artifact()
        self.route6_log_event(out_path, "entrance_search_prepared", {"house_id": hid, "status": status, "included_capture_count": included_count})
        return report

    def route6_execute_close_confirm_scan(
        self,
        session: Optional[flight.DroneFlightSession],
        output_dir: Path,
        house_id: str,
        entrance_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        house_dir = self.route6_house_output_dir(out_path, hid)
        entrance_dir = house_dir / "entrance"
        report_path = entrance_dir / "close_confirm_execution_report.json"
        plan_path = Path(str((entrance_report or {}).get("close_confirm_scan_plan_path", "") or ""))
        plan = self.route6_load_json_artifact(plan_path, {}) if plan_path else {}
        points = plan.get("scan_points", []) if isinstance(plan.get("scan_points", []), list) else []
        runner = getattr(self, "active_nbv_execute_scan_points", None)
        if session is None or not points or not callable(runner):
            execution = {
                "schema": "route6_close_confirm_execution_report_v1",
                "house_id": hid,
                "status": "pending",
                "reason": "session_or_executor_unavailable" if session is None or not callable(runner) else "no_confirm_scan_points",
                "plan_path": str(plan_path),
                "planned_scan_count": len(points),
                "executed_scan_count": 0,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.route6_write_json_artifact(report_path, execution)
            return execution
        self.route6_set_stage("VALIDATE_HOUSE", f"executing Route 6 close-confirm scan for house={hid}")
        status = "executed"
        error = ""
        try:
            if callable(getattr(self, "ensure_active_nbv_state", None)):
                self.ensure_active_nbv_state()
            if hasattr(self, "active_nbv_output_dir"):
                self.active_nbv_output_dir = out_path
            if hasattr(self, "active_nbv_state"):
                self.active_nbv_state = {
                    "schema": "route6_close_confirm_active_nbv_state_v1",
                    "target_house_id": hid,
                    "output_dir": str(out_path),
                    "scan_points": points,
                }
            runner(session, out_path, hid, points, round_index=90, all_points=points)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            self.route6_log_event(out_path, "close_confirm_scan_failed", {"house_id": hid, "error": error})
        executed_count = len(points) if status == "executed" else 0
        execution = {
            "schema": "route6_close_confirm_execution_report_v1",
            "house_id": hid,
            "status": status,
            "error": error,
            "plan_path": str(plan_path),
            "planned_scan_count": len(points),
            "executed_scan_count": executed_count,
            "scan_points": self.route6_json_safe(points),
            "recommended_next_action": "run_route5_capture_analysis_on_close_confirm_captures" if status == "executed" else "inspect_close_confirm_scan_failure",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(report_path, execution)
        validation_path = entrance_dir / "entrance_validation_report.json"
        validation = self.route6_load_json_artifact(validation_path, {})
        if validation:
            validation["close_confirm_execution_report_path"] = str(report_path)
            validation["close_confirm_status"] = status
            validation["close_confirm_executed_scan_count"] = executed_count
            validation["entry_search_complete"] = False
            validation["recommended_next_action"] = execution["recommended_next_action"]
            validation["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.route6_write_json_artifact(validation_path, validation)
            self.llm_route6_state["last_entrance_report"] = validation
        house_states = self.llm_route6_state.setdefault("house_states", {})
        current = dict(house_states.get(hid, {}) if isinstance(house_states.get(hid, {}), dict) else {})
        current["close_confirm_status"] = status
        current["close_confirm_execution_report_path"] = str(report_path)
        if status == "executed":
            current["search_status"] = "pending_close_confirm_analysis"
            current["entrance_status"] = "close_confirm_scan_executed"
        elif status == "failed":
            current["search_status"] = "close_confirm_scan_failed"
            current["entrance_status"] = "close_confirm_scan_failed"
        current["updated_at"] = datetime.now().isoformat(timespec="seconds")
        house_states[hid] = current
        self.route6_write_json_artifact(house_dir / "house_state.json", current)
        self.route6_write_run_artifacts(out_path)
        self.route6_write_state_artifact()
        self.route6_log_event(out_path, "close_confirm_scan_execution", {"house_id": hid, "status": status, "executed_scan_count": executed_count})
        return execution

    def route6_write_close_confirm_obstacle_validation(
        self,
        output_dir: Path,
        house_id: str,
        included_captures: List[Dict[str, Any]],
        excluded_captures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        house_dir = self.route6_house_output_dir(out_path, hid)
        entrance_dir = house_dir / "entrance"
        blockers: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for row in included_captures:
            if not isinstance(row, dict):
                continue
            safety_state = str(row.get("safety_state", row.get("front_risk_state", row.get("risk_state", ""))) or "").lower()
            obstacle_status = str(row.get("obstacle_validation_status", row.get("obstacle_status", "")) or "").lower()
            collision = bool(row.get("collision_state", False))
            avoidance_failed = bool(row.get("avoidance_failed", False))
            terminal = any(token in safety_state for token in ("terminal", "blocked", "collision", "must_stop"))
            blocked = collision or avoidance_failed or terminal or obstacle_status in {"blocked", "terminal_blocked", "collision"}
            if blocked:
                blockers.append({
                    "scan_id": str(row.get("scan_id", "") or ""),
                    "collision_state": collision,
                    "avoidance_failed": avoidance_failed,
                    "safety_state": safety_state,
                    "obstacle_validation_status": obstacle_status,
                    "front_min_depth_cm": row.get("front_min_depth_cm", row.get("min_depth_cm", "")),
                    "reason": "collision_or_terminal_obstacle_risk",
                })
        for row in excluded_captures:
            if isinstance(row, dict) and str(row.get("reason", "") or "") in {"capture_guard_failed_or_missing", "point_count_not_positive"}:
                warnings.append(f"{row.get('scan_id', 'unknown')}: {row.get('reason')}")
        passable = len(included_captures) > 0 and not blockers
        if blockers:
            status = "blocked"
            recommended = "run_avoidance_or_rescan_candidate_from_safer_standoff"
        elif passable:
            status = "clear"
            recommended = "accept_confirmed_entrance"
        else:
            status = "insufficient_evidence"
            recommended = "capture_more_close_confirm_frames"
        report = {
            "schema": "route6_obstacle_validation_report_v1",
            "house_id": hid,
            "status": status,
            "passable_space_confirmed": passable,
            "included_confirm_capture_count": len(included_captures),
            "excluded_confirm_capture_count": len(excluded_captures),
            "blocking_capture_count": len(blockers),
            "blocking_captures": blockers,
            "warnings": warnings,
            "recommended_next_action": recommended,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(entrance_dir / "obstacle_validation_report.json", report)
        return report

    def route6_run_close_confirm_analysis(
        self,
        output_dir: Path,
        house_id: str,
        execution_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        out_path = Path(output_dir)
        hid = str(house_id or "").strip()
        house_dir = self.route6_house_output_dir(out_path, hid)
        entrance_dir = house_dir / "entrance"
        analysis_report_path = entrance_dir / "close_confirm_analysis_report.json"
        plan_path = Path(str((execution_report or {}).get("plan_path", "") or ""))
        plan = self.route6_load_json_artifact(plan_path, {}) if plan_path else {}
        plan_points = plan.get("scan_points", []) if isinstance(plan.get("scan_points", []), list) else []
        confirm_scan_ids = {
            str(point.get("scan_id", "") or "").strip()
            for point in plan_points
            if isinstance(point, dict) and str(point.get("scan_id", "") or "").strip()
        }
        manifest = self.route6_build_entrance_capture_manifest(out_path, hid, plan_points)
        included = [
            row for row in manifest.get("included_captures", [])
            if isinstance(row, dict)
            and (
                str(row.get("scan_id", "") or "").strip() in confirm_scan_ids
                or str(row.get("view_type", "") or "") == "route6_close_confirm_scan"
            )
        ]
        excluded = [
            row for row in manifest.get("excluded_captures", [])
            if isinstance(row, dict)
            and (
                str(row.get("scan_id", "") or "").strip() in confirm_scan_ids
                or str(row.get("view_type", "") or "") == "route6_close_confirm_scan"
            )
        ]
        confirm_manifest = {
            "schema": "route6_close_confirm_capture_manifest_v1",
            "run_dir": str(out_path),
            "house_id": hid,
            "plan_path": str(plan_path),
            "confirm_scan_ids": sorted(confirm_scan_ids),
            "included_count": len(included),
            "excluded_count": len(excluded),
            "included_captures": included,
            "excluded_captures": excluded,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        manifest_path = entrance_dir / "close_confirm_capture_manifest.json"
        self.route6_write_json_artifact(manifest_path, confirm_manifest)
        analysis = self.route6_run_route5_capture_analysis_for_house(out_path, hid, confirm_manifest)
        analysis_ran = bool(analysis.get("ran", False))
        analysis_error = str(analysis.get("error", "") or "")
        candidate_count = int(analysis.get("candidate_count", 0) or 0)
        obstacle_validation = self.route6_write_close_confirm_obstacle_validation(out_path, hid, included, excluded)
        obstacle_status = str(obstacle_validation.get("status", "") or "")
        passable_space_confirmed = bool(obstacle_validation.get("passable_space_confirmed", False))
        if analysis_error:
            status = "analysis_failed"
            recommended = "inspect_close_confirm_analysis_error"
        elif analysis_ran and candidate_count > 0 and passable_space_confirmed:
            status = "confirmed"
            recommended = "navigate_to_confirmed_entrance_or_finish_house"
        elif analysis_ran and candidate_count > 0:
            status = "blocked_by_obstacle"
            recommended = "run_avoidance_or_rescan_candidate_from_safer_standoff"
        elif analysis_ran:
            status = "not_confirmed"
            recommended = "rescan_candidate_facade_or_select_next_house"
        else:
            status = "pending"
            recommended = "run_route5_capture_analysis_on_close_confirm_captures"
        report = {
            "schema": "route6_close_confirm_analysis_report_v1",
            "house_id": hid,
            "status": status,
            "recommended_next_action": recommended,
            "capture_manifest_path": str(manifest_path),
            "execution_report_path": str(entrance_dir / "close_confirm_execution_report.json"),
            "route5_analysis_summary_path": str(analysis.get("summary_path", "") or ""),
            "route5_analysis_status": str(analysis.get("status", "") or ""),
            "route5_analysis_error": analysis_error,
            "obstacle_validation_report_path": str(entrance_dir / "obstacle_validation_report.json"),
            "obstacle_validation_status": obstacle_status,
            "passable_space_confirmed": passable_space_confirmed,
            "included_confirm_capture_count": len(included),
            "excluded_confirm_capture_count": len(excluded),
            "candidate_count": candidate_count,
            "candidates": analysis.get("candidates", []) if isinstance(analysis.get("candidates", []), list) else [],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(analysis_report_path, report)
        validation_path = entrance_dir / "entrance_validation_report.json"
        validation = self.route6_load_json_artifact(validation_path, {})
        if validation:
            validation["close_confirm_analysis_report_path"] = str(analysis_report_path)
            validation["close_confirm_analysis_status"] = status
            validation["close_confirm_candidate_count"] = candidate_count
            validation["obstacle_validation_report_path"] = str(entrance_dir / "obstacle_validation_report.json")
            validation["obstacle_validation_status"] = obstacle_status
            validation["passable_space_confirmed"] = passable_space_confirmed
            validation["entry_search_complete"] = status == "confirmed"
            validation["status"] = "entrance_confirmed" if status == "confirmed" else f"close_confirm_{status}"
            validation["recommended_next_action"] = recommended
            validation["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.route6_write_json_artifact(validation_path, validation)
            self.llm_route6_state["last_entrance_report"] = validation
        house_states = self.llm_route6_state.setdefault("house_states", {})
        current = dict(house_states.get(hid, {}) if isinstance(house_states.get(hid, {}), dict) else {})
        current["close_confirm_status"] = status
        current["close_confirm_analysis_report_path"] = str(analysis_report_path)
        current["close_confirm_candidate_count"] = candidate_count
        current["obstacle_validation_report_path"] = str(entrance_dir / "obstacle_validation_report.json")
        current["obstacle_validation_status"] = obstacle_status
        if status == "confirmed":
            current["status"] = "searched"
            current["search_status"] = "entrance_confirmed"
            current["entrance_status"] = "confirmed_by_close_confirm_scan"
        elif status == "blocked_by_obstacle":
            current["status"] = "blocked"
            current["search_status"] = "entrance_candidate_blocked_by_obstacle"
            current["entrance_status"] = "blocked_by_obstacle"
            current["blocked_reason"] = "close_confirm_obstacle_validation_blocked"
        elif status == "not_confirmed":
            current["search_status"] = "close_confirm_not_confirmed"
            current["entrance_status"] = "needs_rescan_or_next_candidate"
        elif status == "analysis_failed":
            current["search_status"] = "close_confirm_analysis_failed"
            current["entrance_status"] = "close_confirm_analysis_failed"
        current["updated_at"] = datetime.now().isoformat(timespec="seconds")
        house_states[hid] = current
        self.route6_write_json_artifact(house_dir / "house_state.json", current)
        self.route6_write_run_artifacts(out_path)
        self.route6_write_state_artifact()
        self.route6_log_event(out_path, "close_confirm_analysis", {"house_id": hid, "status": status, "candidate_count": candidate_count})
        return report

    def route6_write_house_coverage_report(
        self,
        output_dir: Path,
        house_id: str,
        scan_points: List[Dict[str, Any]],
        valid_rows: List[Dict[str, Any]],
        merged_cloud: np.ndarray,
    ) -> Dict[str, Any]:
        coverage: Dict[str, Any] = {}
        if callable(getattr(self, "active_nbv_build_coverage_report", None)):
            try:
                coverage = self.active_nbv_build_coverage_report(house_id, scan_points, output_dir=Path(output_dir))
            except Exception:
                coverage = {}
        if not coverage:
            captured_ids = {str(row.get("scan_id", "") or "") for row in valid_rows}
            facades: Dict[str, Dict[str, Any]] = {}
            for facade in route6_map_builder.ROUTE6_FACADES:
                planned = [point for point in scan_points if str(point.get("facade", "") or "") == facade]
                captured = [point for point in planned if str(point.get("scan_id", "") or "") in captured_ids]
                facades[facade] = {
                    "facade": facade,
                    "planned_scan_count": len(planned),
                    "captured_scan_count": len(captured),
                    "scan_completion_ratio": round(float(len(captured)) / float(max(1, len(planned))), 4),
                    "point_cloud_coverage": round(float(len(captured)) / float(max(1, len(planned))), 4),
                    "needs_rescan": bool(len(captured) < len(planned)),
                }
            values = [float(item["point_cloud_coverage"]) for item in facades.values()]
            coverage = {
                "schema": "route6_house_coverage_report_v1",
                "target_house_id": str(house_id),
                "facades": facades,
                "valid_scan_capture_count": len(valid_rows),
                "merged_point_count": int(merged_cloud.shape[0]),
                "mean_facade_coverage": round(float(sum(values)) / float(max(1, len(values))), 4),
                "complete": bool(valid_rows and merged_cloud.shape[0] > 0),
                "coverage_mode": "route6_valid_scan_completion",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        coverage["valid_scan_capture_count"] = int(coverage.get("valid_scan_capture_count", len(valid_rows)) or len(valid_rows))
        coverage["merged_point_count"] = int(coverage.get("merged_point_count", merged_cloud.shape[0]) or merged_cloud.shape[0])
        house_dir = self.route6_house_output_dir(Path(output_dir), house_id)
        self.route6_write_json_artifact(house_dir / "coverage_report.json", coverage)
        return coverage

    def route6_postprocess_selected_house(self, output_dir: Path, house_id: str) -> Dict[str, Any]:
        self.ensure_route6_state()
        out_path = Path(output_dir)
        hid = str(house_id or self.llm_route6_state.get("selected_house_id", "") or "").strip()
        house_dir = self.route6_house_output_dir(out_path, hid)
        self.route6_set_stage("POSTPROCESS_POINTCLOUD", f"postprocessing Route 6 pointcloud for house={hid}")
        rows = self.route6_read_lidar_rows(out_path)
        scan_points = self.llm_route6_state.get("scan_points", []) if isinstance(self.llm_route6_state.get("scan_points"), list) else []
        house_rows = self.route6_filter_rows_for_house(rows, hid, scan_points)
        valid_rows = route6_map_builder.filter_valid_pointcloud_rows(house_rows)
        if not valid_rows:
            result = {
                "schema": "route6_house_state_v1",
                "house_id": hid,
                "status": "needs_capture",
                "reason": "no_valid_pointcloud_rows",
                "valid_scan_capture_count": 0,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.route6_write_json_artifact(house_dir / "house_state.json", result)
            self.llm_route6_state.setdefault("house_states", {})[hid] = result
            self.route6_write_run_artifacts(out_path)
            self.route6_write_state_artifact()
            self.route6_set_stage("SCAN_HOUSE_FACADES", f"house={hid} needs live scan captures before map build.")
            return result
        merged = route6_map_builder.merge_pointcloud_rows(valid_rows)
        map_config = getattr(self, "map_config", {}) if isinstance(getattr(self, "map_config", {}), dict) else {}
        house = next(
            (item for item in map_config.get("houses", []) if isinstance(item, dict) and str(item.get("id", "")) == hid),
            None,
        ) if isinstance(map_config.get("houses", []), list) else None
        bbox = route6_map_builder.house_world_bbox(map_config, house) if isinstance(house, dict) else None
        filtered = route6_map_builder.filter_pointcloud_for_mapping(merged, bbox_unreal_cm=bbox)
        if filtered.shape[0] <= 0:
            filtered = merged
        pointcloud_dir = house_dir / "pointcloud"
        pointcloud_dir.mkdir(parents=True, exist_ok=True)
        merged_path = pointcloud_dir / "merged_point_cloud_world_standard_m.npy"
        np.save(merged_path, filtered.astype(np.float32, copy=False))
        merged_ply_path = pointcloud_dir / "merged_point_cloud_world_standard_m.ply"
        route6_map_builder.write_pointcloud_ply(merged_ply_path, filtered)
        coverage = self.route6_write_house_coverage_report(out_path, hid, scan_points, valid_rows, filtered)
        self.route6_set_stage("BUILD_OCCUPANCY", f"building Route 6 occupancy map for house={hid}")
        self.route6_set_stage("EXTRACT_POLYGONS", f"extracting Route 6 building polygon for house={hid}")
        artifacts = route6_map_builder.write_route6_map_artifacts(
            out_path,
            map_config,
            hid,
            filtered,
            resolution_m=float(self.llm_route6_occupancy_resolution_m_var.get() or 0.25),
        )
        map_dir = house_dir / "map"
        map_dir.mkdir(parents=True, exist_ok=True)
        for source_key, target_name in (
            ("occupancy_grid_path", "occupancy_grid.npy"),
            ("occupancy_metadata_path", "occupancy_grid.json"),
            ("occupancy_preview_path", "occupancy_grid.png"),
        ):
            source = Path(str(artifacts.get(source_key, "") or ""))
            if source.is_file():
                target = map_dir / target_name
                target.write_bytes(source.read_bytes())
                artifacts[f"house_{source_key}"] = str(target)
        building_polygon_path = map_dir / "building_polygon.json"
        polygon_payload = artifacts.get("polygon", {}) if isinstance(artifacts.get("polygon", {}), dict) else {}
        self.route6_write_json_artifact(building_polygon_path, polygon_payload)
        artifacts["building_polygon_path"] = str(building_polygon_path)
        corrected_house: Dict[str, Any] = {}
        corrected_config_path = Path(str(artifacts.get("corrected_config_path", "") or ""))
        if corrected_config_path.is_file():
            try:
                corrected_config = json.loads(corrected_config_path.read_text(encoding="utf-8"))
                if isinstance(corrected_config, dict):
                    self.route6_runtime_map_config = corrected_config
                corrected_houses = corrected_config.get("houses", []) if isinstance(corrected_config.get("houses"), list) else []
                corrected_house = next(
                    (
                        item for item in corrected_houses
                        if isinstance(item, dict) and str(item.get("id", item.get("house_id", ""))) == hid
                    ),
                    {},
                )
            except Exception as exc:
                corrected_house = {"route6_corrected_config_read_error": str(exc)}
        if corrected_house:
            corrected_config_path = self.route6_write_cumulative_corrected_config(out_path, hid, corrected_house)
            artifacts["corrected_config_path"] = str(corrected_config_path)
        corrected_bbox_path = map_dir / "corrected_bbox.json"
        corrected_record = {
            "schema": "route6_corrected_bbox_record_v1",
            "house_id": hid,
            "rough_bbox_world": bbox or {},
            "candidate_bbox_world": corrected_house.get("route6_candidate_bbox_world", polygon_payload.get("bbox", {})),
            "corrected_bbox_world": corrected_house.get("route6_corrected_bbox_world", {}),
            "route6_map_status": str(corrected_house.get("route6_map_status", "candidate_only") or "candidate_only"),
            "route6_correction_rejected_reason": str(corrected_house.get("route6_correction_rejected_reason", "") or ""),
            "route6_map_confidence": corrected_house.get("route6_map_confidence", (polygon_payload.get("quality", {}) if isinstance(polygon_payload.get("quality", {}), dict) else {}).get("confidence", 0.0)),
            "route6_center_shift_cm": corrected_house.get("route6_center_shift_cm", ""),
            "route6_quality_gates": corrected_house.get("route6_quality_gates", {}),
            "map_bbox_image": corrected_house.get("map_bbox_image", {}),
            "corrected_config_path": str(corrected_config_path) if corrected_config_path else "",
            "building_polygon_path": str(building_polygon_path),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(corrected_bbox_path, corrected_record)
        artifacts["corrected_bbox_path"] = str(corrected_bbox_path)
        cumulative_polygons_path = self.route6_write_cumulative_polygon_artifact(out_path, hid, polygon_payload)
        artifacts["polygons_path"] = str(cumulative_polygons_path)
        status = "mapped_complete" if bool(coverage.get("complete", False)) else "mapped_partial"
        polygon_quality = polygon_payload.get("quality", {}) if isinstance(polygon_payload.get("quality", {}), dict) else {}
        map_confidence = corrected_record.get("route6_map_confidence", polygon_quality.get("confidence", 0.0))
        house_state = {
            "schema": "route6_house_state_v1",
            "house_id": hid,
            "status": status,
            "search_status": "pending_entrance_search",
            "map_status": str(corrected_record.get("route6_map_status", "candidate_only") or "candidate_only"),
            "facades": coverage.get("facades", {}) if isinstance(coverage.get("facades", {}), dict) else {},
            "valid_scan_capture_count": len(valid_rows),
            "merged_point_count": int(filtered.shape[0]),
            "map_confidence": map_confidence,
            "entrance_status": "not_started",
            "blocked_reason": "",
            "coverage_report_path": str(house_dir / "coverage_report.json"),
            "merged_pointcloud_path": str(merged_path),
            "merged_pointcloud_ply_path": str(merged_ply_path),
            "map_artifacts": artifacts,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.route6_write_json_artifact(house_dir / "house_state.json", house_state)
        house_states = self.llm_route6_state.setdefault("house_states", {})
        house_states[hid] = house_state
        self.llm_route6_state["last_house_state"] = house_state
        self.llm_route6_state["map_artifacts"] = artifacts
        self.route6_write_run_artifacts(out_path)
        self.route6_write_state_artifact()
        self.route6_set_stage("CORRECT_MAP_CONFIG", f"Route 6 corrected map artifact written for house={hid}")
        self.llm_route6_map_status_var.set(f"Map: {status}, points={int(filtered.shape[0])}")
        return house_state

    def route6_full_explore_worker(self, session: Optional[flight.DroneFlightSession] = None, *, force_new: bool = False) -> None:
        self.ensure_route6_state()
        output_dir = self.route6_initialize_run(force_new=force_new)
        if session is not None:
            try:
                self.route6_set_stage("VISUAL_DIRECTION_ANALYSIS", "capturing direction sweep for visual house target selection.")
                self.route6_run_visual_direction_analysis(session, output_dir)
            except Exception as exc:
                self.route6_log_event(output_dir, "visual_direction_analysis_failed", {"error": str(exc)})
                self.llm_route6_visual_status_var.set(f"LLM Visual Direction Analysis: failed: {exc}")
        max_houses = self.route6_int_param(self.llm_route6_max_houses_var, 3, min_value=1, max_value=100)
        max_runtime_minutes = self.route6_float_param(self.llm_route6_runtime_min_var, 30.0, min_value=0.0, max_value=1440.0)
        max_runtime_s = float(max_runtime_minutes) * 60.0
        started_monotonic = time.monotonic()
        processed = list(self.llm_route6_state.get("processed_house_ids", [])) if isinstance(self.llm_route6_state.get("processed_house_ids", []), list) else []
        terminal_message = ""
        runtime_exhausted = False
        while len(processed) < max_houses:
            if max_runtime_s <= 0.0 or (time.monotonic() - started_monotonic) >= max_runtime_s:
                runtime_exhausted = True
                terminal_message = f"max_runtime_minutes={max_runtime_minutes:g} exhausted."
                break
            if self.llm_route6_stop_event.is_set():
                terminal_message = "stop requested."
                break
            if not self.route6_wait_if_paused(output_dir):
                terminal_message = "stop requested while paused."
                break
            if not self.route6_movement_allowed():
                terminal_message = "full stop requested."
                break
            selected_candidate = self.route6_select_next_house_for_mapping(output_dir)
            selected = str(selected_candidate.get("house_id", "") or "")
            if not selected:
                terminal_message = "no more reachable Route 6 house candidates."
                break
            if self.llm_route6_force_next_event.is_set():
                self.llm_route6_force_next_event.clear()
                house_states = self.llm_route6_state.setdefault("house_states", {})
                house_states[selected] = {
                    "schema": "route6_house_state_v1",
                    "house_id": selected,
                    "status": "needs_rescan",
                    "search_status": "force_next_requested",
                    "cooldown_active": True,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                self.route6_write_state_artifact()
                self.route6_log_event(output_dir, "force_next_house", {"house_id": selected})
                continue
            self.route6_set_stage("PLAN_TO_HOUSE", f"planning Route 6 scan points for house={selected}")
            points = self.route6_plan_selected_house_scan_points(output_dir, selected)
            if not self.route6_movement_allowed():
                terminal_message = "full stop requested before live scan."
                break
            if session is not None and points and callable(getattr(self, "active_nbv_execute_scan_points", None)):
                self.route6_set_stage("SCAN_HOUSE_FACADES", f"executing Route 6 live scan for house={selected}")
                try:
                    if callable(getattr(self, "ensure_active_nbv_state", None)):
                        self.ensure_active_nbv_state()
                    if hasattr(self, "active_nbv_output_dir"):
                        self.active_nbv_output_dir = output_dir
                    if hasattr(self, "active_nbv_state"):
                        self.active_nbv_state = {
                            "schema": "route6_active_nbv_bridge_state_v1",
                            "target_house_id": selected,
                            "output_dir": str(output_dir),
                            "scan_points": points,
                        }
                    self.active_nbv_execute_scan_points(session, output_dir, selected, points, round_index=0, all_points=points)
                except Exception as exc:
                    self.route6_log_event(output_dir, "live_scan_failed", {"house_id": selected, "error": str(exc)})
                    self.route6_set_stage("SCAN_HOUSE_FACADES", f"live scan failed for house={selected}: {exc}")
            if not self.route6_movement_allowed():
                terminal_message = "full stop requested after live scan."
                break
            if self.llm_route6_force_next_event.is_set():
                self.llm_route6_force_next_event.clear()
                house_states = self.llm_route6_state.setdefault("house_states", {})
                house_states[selected] = {
                    "schema": "route6_house_state_v1",
                    "house_id": selected,
                    "status": "needs_rescan",
                    "search_status": "force_next_requested",
                    "cooldown_active": True,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                self.route6_write_state_artifact()
                self.route6_log_event(output_dir, "force_next_house", {"house_id": selected})
                continue
            result = self.route6_postprocess_selected_house(output_dir, selected)
            status = str(result.get("status", ""))
            if selected not in processed:
                processed.append(selected)
                self.llm_route6_state["processed_house_ids"] = processed
                self.route6_write_state_artifact()
            if status in {"mapped_complete", "mapped_partial"}:
                entrance_report = self.route6_run_entrance_search(output_dir, selected, result)
                if entrance_report.get("close_confirm_scan_plan_path"):
                    confirm_execution = self.route6_execute_close_confirm_scan(session, output_dir, selected, entrance_report)
                    if str(confirm_execution.get("status", "") or "") == "executed":
                        self.route6_run_close_confirm_analysis(output_dir, selected, confirm_execution)
                self.route6_set_stage("SELECT_NEXT_HOUSE", f"Route 6 finished map artifacts for house={selected}; selecting next house.")
                continue
            if status == "needs_capture":
                self.route6_set_stage("SCAN_HOUSE_FACADES", f"Route 6 waiting for captures, house={selected}")
                if session is None:
                    terminal_message = f"house={selected} needs live scan captures before continuing."
                    break
                continue
            self.route6_set_stage("FAILED", f"Route 6 house processing failed, house={selected}")
            return
        if getattr(self, "route6_full_stop_event", None) is not None and self.route6_full_stop_event.is_set():
            self.llm_route6_state["stage"] = "FULL_STOPPED"
            self.llm_route6_stage_var.set("Stage: FULL_STOPPED")
            self.llm_route6_status_var.set(f"LLM Route V6: {terminal_message or 'full stop requested.'}")
            self.route6_write_state_artifact()
        elif self.llm_route6_stop_event.is_set() or runtime_exhausted:
            self.route6_set_stage("STOPPED", terminal_message or "stop requested.")
        else:
            self.llm_route6_state["processed_house_ids"] = processed
            self.route6_write_run_artifacts(output_dir)
            self.route6_write_state_artifact()
            self.route6_set_stage("DONE", terminal_message or f"processed max_houses={len(processed)}.")

    def route6_update_summary_text(self) -> None:
        text_widget = getattr(self, "llm_route6_summary_text", None)
        if text_widget is None:
            return
        design_path = self.route6_design_doc_path()
        payload = {
            "mode": "route6_nearest_house_pointcloud_map",
            "design_doc": str(design_path),
            "v003_requirements_doc": str(PROJECT_ROOT / ROUTE6_V003_REQUIREMENTS_DOC),
            "current_stage": self.llm_route6_stage_var.get(),
            "planned_pipeline": [
                "consume 13-layer realtime 2D occupancy map from Route 6 Update Map",
                "run LLM semantic map analysis to skip fences/rails and choose the requested house target",
                "write route6_llm_semantic_target_selection.json and route6_llm_navigation_target.json",
                "navigate to the selected observation side with OR avoidance gating",
                "merge valid point clouds",
                "build 2D occupancy grid",
                "extract building/obstacle polygons",
                "write route6 corrected map artifact",
                "run entrance search after local map confidence is sufficient",
            ],
            "implementation_status": "v003 adds Full Stop UAV, LLM semantic layered-map target selection, navigation target artifacts, and OR-aware search handoff",
        }
        try:
            text_widget.configure(state="normal")
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, json.dumps(payload, indent=2, ensure_ascii=False))
            text_widget.configure(state="disabled")
        except tk.TclError:
            pass

    def route6_update_runtime_metrics(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        states = self.route6_house_states()
        statuses = [str(item.get("status", "") or "") for item in states.values() if isinstance(item, dict)]
        mapped = len([status for status in statuses if status in {"mapped_complete", "mapped_partial", "searched", "searched_no_entry"}])
        searched = len([status for status in statuses if status in {"searched", "searched_no_entry"}])
        blocked = len([status for status in statuses if status in {"blocked", "terminal_blocked"}])
        last_house = self.llm_route6_state.get("last_house_state", {}) if isinstance(self.llm_route6_state.get("last_house_state", {}), dict) else {}
        artifacts = last_house.get("map_artifacts", {}) if isinstance(last_house.get("map_artifacts", {}), dict) else {}
        confidence = last_house.get("map_confidence", "")
        if confidence == "":
            quality = artifacts.get("polygon", {}).get("quality", {}) if isinstance(artifacts.get("polygon", {}), dict) else {}
            confidence = quality.get("confidence", "")
        corrected_path = str(artifacts.get("corrected_config_path", "") or "")
        selected = self.llm_route6_state.get("selected_candidate", {}) if isinstance(self.llm_route6_state.get("selected_candidate", {}), dict) else {}
        current_facade = str(selected.get("nearest_facade", "") or "")
        scan_points = self.llm_route6_state.get("scan_points", []) if isinstance(self.llm_route6_state.get("scan_points", []), list) else []
        for point in scan_points:
            if isinstance(point, dict) and str(point.get("status", "") or "") in {"planned", "captured", "active"}:
                current_facade = str(point.get("facade", current_facade) or current_facade)
                break
        confidence_text = "n/a"
        try:
            confidence_text = f"{float(confidence):.3f}"
        except Exception:
            pass
        corrected_text = Path(corrected_path).name if corrected_path else "n/a"
        metrics = {
            "mapped_count": mapped,
            "searched_count": searched,
            "blocked_count": blocked,
            "current_facade": current_facade or "n/a",
            "map_confidence": confidence_text,
            "latest_corrected_config_path": corrected_path,
        }
        self.llm_route6_metrics_var.set(
            f"Metrics: facade={metrics['current_facade']} mapped={mapped} searched={searched} "
            f"blocked={blocked} confidence={confidence_text} corrected={corrected_text}"
        )
        self.llm_route6_state["runtime_metrics"] = metrics
        return metrics

    def route6_set_stage(self, stage: str, message: str = "") -> None:
        self.ensure_route6_state()
        self.llm_route6_stage_var.set(f"Stage: {stage}")
        if message:
            self.llm_route6_status_var.set(f"LLM Route V6: {message}")
        self.llm_route6_state["stage"] = str(stage)
        self.llm_route6_state["message"] = str(message or "")
        self.llm_route6_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        output_dir = self.llm_route6_state.get("output_dir") if isinstance(self.llm_route6_state, dict) else ""
        if output_dir:
            out_path = Path(str(output_dir))
            self.route6_write_state_artifact()
            self.route6_log_event(out_path, "stage", {"stage": str(stage), "message": str(message or "")})
        self.route6_update_runtime_metrics()
        if output_dir:
            self.route6_write_state_artifact()
        self.route6_update_summary_text()
        if getattr(self, "llm_route6_map_widget", None) is not None:
            self.refresh_llm_route6_map()
        if getattr(self, "llm_route6_realtime_map_preview_label", None) is not None:
            self.refresh_llm_route6_realtime_map(build_if_missing=False)
        if hasattr(self, "llm_route6_or_status_var"):
            self.refresh_llm_route6_or_avoidance_display()

    def open_llm_route_window6(self) -> None:
        self.ensure_route6_state()
        if self.llm_route6_window is not None and self.llm_route6_window.winfo_exists():
            self.llm_route6_window.lift()
            self.llm_route6_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route 6")
        window.geometry("1120x760")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close_llm_route_window6)

        llm_route6_scroll_canvas = tk.Canvas(window, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(window, orient="vertical", command=llm_route6_scroll_canvas.yview)
        h_scrollbar = tk.Scrollbar(window, orient="horizontal", command=llm_route6_scroll_canvas.xview)
        llm_route6_scroll_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        llm_route6_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        content = tk.Frame(llm_route6_scroll_canvas)
        content_window = llm_route6_scroll_canvas.create_window((0, 0), window=content, anchor="nw")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(3, weight=2)
        content.grid_rowconfigure(8, weight=1)

        def _sync_scrollregion(_event: tk.Event) -> None:
            try:
                llm_route6_scroll_canvas.configure(scrollregion=llm_route6_scroll_canvas.bbox("all"))
            except tk.TclError:
                pass

        def _sync_content_width(event: tk.Event) -> None:
            try:
                llm_route6_scroll_canvas.itemconfigure(content_window, width=max(1060, int(event.width)))
            except tk.TclError:
                pass

        content.bind("<Configure>", _sync_scrollregion)
        llm_route6_scroll_canvas.bind("<Configure>", _sync_content_width)

        header = tk.LabelFrame(content, text="Route 6 Nearest House Pointcloud Map")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        for col in (1, 3, 5, 7):
            header.grid_columnconfigure(col, weight=1)

        tk.Label(header, text="Max houses").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_max_houses_var, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Runtime min").grid(row=0, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_runtime_min_var, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Standoff cm").grid(row=0, column=4, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_standoff_cm_var, width=8).grid(row=0, column=5, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Scan z cm").grid(row=0, column=6, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_scan_z_cm_var, width=8).grid(row=0, column=7, sticky="w", padx=6, pady=5)

        tk.Label(header, text="Occupancy m").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_occupancy_resolution_m_var, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Coverage").grid(row=1, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_coverage_threshold_var, width=8).grid(row=1, column=3, sticky="w", padx=6, pady=5)
        tk.Checkbutton(
            header,
            text="Allow save corrected config",
            variable=self.llm_route6_allow_save_corrected_var,
        ).grid(row=1, column=4, columnspan=3, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Mission").grid(row=2, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_task_prompt_var, width=72).grid(row=2, column=1, columnspan=5, sticky="ew", padx=6, pady=5)
        tk.Button(header, text="Select LLM Target", command=self.on_route6_select_llm_target).grid(row=2, column=6, columnspan=2, sticky="ew", padx=6, pady=5)
        tk.Label(header, textvariable=self.llm_route6_selected_target_var, anchor="w").grid(row=3, column=0, columnspan=8, sticky="ew", padx=6, pady=(0, 5))

        actions = tk.Frame(content)
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(actions, text="Start LLM Target Map Search", command=self.on_route6_start_nearest_map_search).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Pause", command=self.on_route6_pause).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Resume", command=self.on_route6_resume).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=self.on_route6_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Full Stop UAV", command=self.on_route6_full_stop_uav).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Force Next House", command=self.on_route6_force_next_house).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_route6_clear).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Save Corrected Map Config", command=self.on_route6_save_corrected_map_config).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Open Latest Route 6 Output", command=self.on_route6_open_latest_output).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Route 6 Update Map", command=self.open_route6_update_map_window).pack(side="left", padx=6, pady=4)

        status = tk.LabelFrame(content, text="Status")
        status.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        status.grid_columnconfigure(0, weight=1)
        status.grid_columnconfigure(1, weight=1)
        tk.Label(status, textvariable=self.llm_route6_stage_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_current_house_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_queue_var, anchor="w").grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_map_status_var, anchor="w").grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_output_dir_var, anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_metrics_var, anchor="w").grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_status_var, anchor="w", wraplength=1040, justify="left").grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=3)

        realtime_section = tk.LabelFrame(content, text="Realtime Layered Map")
        realtime_section.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        realtime_section.grid_columnconfigure(0, weight=1)
        realtime_section.grid_rowconfigure(2, weight=1)
        realtime_toolbar = tk.Frame(realtime_section)
        realtime_toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        tk.Label(realtime_toolbar, text="Layer").pack(side="left", padx=(0, 4))
        realtime_layer_combo = ttk.Combobox(
            realtime_toolbar,
            textvariable=self.route6_update_map_layer_var,
            values=[f"z_{int(value):03d}" for value in route6_map_builder.DEFAULT_ROUTE6_LAYER_Z_CM],
            state="readonly",
            width=10,
        )
        realtime_layer_combo.pack(side="left", padx=4)
        realtime_layer_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_llm_route6_realtime_map(build_if_missing=False))
        tk.Button(realtime_toolbar, text="Load Latest Map", command=lambda: self.refresh_llm_route6_realtime_map(build_if_missing=False)).pack(side="left", padx=6)
        tk.Button(realtime_toolbar, text="Start Realtime Update", command=self.on_route6_update_map_start_realtime).pack(side="left", padx=6)
        tk.Button(realtime_toolbar, text="Stop Realtime Update", command=self.on_route6_update_map_stop_realtime).pack(side="left", padx=6)
        tk.Button(realtime_toolbar, text="Route 6 Update Map", command=self.open_route6_update_map_window).pack(side="left", padx=6)
        tk.Label(realtime_toolbar, textvariable=self.llm_route6_realtime_map_status_var, anchor="w").pack(side="left", fill="x", expand=True, padx=8)
        realtime_status = tk.Frame(realtime_section)
        realtime_status.grid(row=1, column=0, sticky="ew", padx=6, pady=(6, 0))
        realtime_status.grid_columnconfigure(0, weight=1)
        realtime_status.grid_columnconfigure(1, weight=1)
        tk.Label(realtime_status, textvariable=self.route6_update_map_status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=(0, 8))
        tk.Label(realtime_status, textvariable=self.route6_update_map_pose_var, anchor="w").grid(row=0, column=1, sticky="ew")
        realtime_preview = tk.Label(realtime_section, text="Loading realtime layered occupancy map...", anchor="center", justify="center")
        realtime_preview.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route6_realtime_map_frame = realtime_section
        self.llm_route6_realtime_map_layer_combo = realtime_layer_combo
        self.llm_route6_realtime_map_preview_label = realtime_preview

        analysis_section = tk.LabelFrame(content, text="LLM Map Analysis")
        analysis_section.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        analysis_section.grid_columnconfigure(0, weight=1)
        tk.Label(analysis_section, textvariable=self.llm_route6_map_analysis_status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 3))
        tk.Label(analysis_section, textvariable=self.llm_route6_map_analysis_detail_var, anchor="w", wraplength=980, justify="left").grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        tk.Label(analysis_section, textvariable=self.llm_route6_navigation_target_var, anchor="w", wraplength=980, justify="left").grid(row=2, column=0, sticky="ew", padx=6, pady=(3, 6))
        tk.Button(analysis_section, text="Refresh LLM Map Analysis", command=self.refresh_llm_route6_map_analysis_panel).grid(row=0, column=1, rowspan=3, sticky="e", padx=6, pady=6)

        visual_section = tk.LabelFrame(content, text="LLM Visual Direction Analysis")
        visual_section.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        visual_section.grid_columnconfigure(0, weight=1)
        visual_buttons = tk.Frame(visual_section)
        visual_buttons.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 3))
        tk.Button(visual_buttons, text="Capture Direction Sweep", command=self.on_route6_capture_direction_sweep).pack(side="left", padx=(0, 6))
        tk.Button(visual_buttons, text="Analyze Direction Images", command=self.on_route6_analyze_direction_images).pack(side="left", padx=6)
        tk.Button(visual_buttons, text="Apply Visual House Marker", command=self.on_route6_apply_visual_house_marker).pack(side="left", padx=6)
        tk.Button(visual_buttons, text="Start Visual Target Approach", command=self.on_route6_start_visual_target_approach).pack(side="left", padx=6)
        tk.Label(visual_section, textvariable=self.llm_route6_visual_status_var, anchor="w").grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        tk.Label(visual_section, textvariable=self.llm_route6_visual_detail_var, anchor="w", wraplength=1010, justify="left").grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))

        conflict_section = tk.LabelFrame(content, text="Height Conflict / Replan")
        conflict_section.grid(row=6, column=0, sticky="ew", padx=8, pady=4)
        conflict_section.grid_columnconfigure(0, weight=1)
        conflict_buttons = tk.Frame(conflict_section)
        conflict_buttons.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 3))
        tk.Button(conflict_buttons, text="Check Height Conflict", command=self.on_route6_check_height_conflict).pack(side="left", padx=(0, 6))
        tk.Button(conflict_buttons, text="Start Orbit Capture", command=self.on_route6_start_orbit_capture).pack(side="left", padx=6)
        tk.Button(conflict_buttons, text="Stop Orbit Capture", command=self.on_route6_stop_orbit_capture).pack(side="left", padx=6)
        tk.Label(conflict_section, textvariable=self.llm_route6_conflict_status_var, anchor="w").grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        tk.Label(conflict_section, textvariable=self.llm_route6_conflict_detail_var, anchor="w", wraplength=1010, justify="left").grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))

        or_section = tk.LabelFrame(content, text="OR Avoidance")
        or_section.grid(row=7, column=0, sticky="ew", padx=8, pady=4)
        or_section.grid_columnconfigure(1, weight=1)
        tk.Label(or_section, textvariable=self.llm_route6_or_status_var, anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 3))
        tk.Label(or_section, textvariable=self.llm_route6_or_detail_var, anchor="w", wraplength=1010, justify="left").grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        tk.Button(or_section, text="Refresh OR", command=self.refresh_llm_route6_or_avoidance_display).grid(row=1, column=1, sticky="e", padx=6, pady=(0, 6))

        summary = tk.LabelFrame(content, text="Route 6 Implementation Contract")
        summary.grid(row=8, column=0, sticky="nsew", padx=8, pady=(4, 8))
        summary.grid_columnconfigure(0, weight=1)
        summary.grid_rowconfigure(0, weight=1)
        text = tk.Text(summary, height=18, wrap="none", font=("Consolas", 9))
        text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        scroll = tk.Scrollbar(summary, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns", pady=6)
        text.configure(yscrollcommand=scroll.set, state="disabled")

        self.llm_route6_window = window
        self.llm_route6_scroll_canvas = llm_route6_scroll_canvas
        self.llm_route6_content_frame = content
        self.llm_route6_summary_text = text
        self.route6_update_summary_text()
        self.refresh_llm_route6_realtime_map(build_if_missing=False)
        self.refresh_llm_route6_or_avoidance_display()
        self._bind_llm_route6_mousewheel_tree(window)
        self.route6_schedule_llm_realtime_map_refresh()

    def close_llm_route_window6(self) -> None:
        self.ensure_route6_state()
        after_id = getattr(self, "llm_route6_realtime_map_after_id", None)
        if after_id is not None and self.llm_route6_window is not None:
            try:
                self.llm_route6_window.after_cancel(after_id)
            except Exception:
                pass
        self.llm_route6_realtime_map_after_id = None
        if self.llm_route6_window is not None:
            try:
                self.llm_route6_window.destroy()
            except Exception:
                pass
        self.llm_route6_window = None
        self.llm_route6_summary_text = None
        self.llm_route6_scroll_canvas = None
        self.llm_route6_content_frame = None
        self.llm_route6_map_widget = None
        self.llm_route6_map_frame = None
        self.llm_route6_realtime_map_frame = None
        self.llm_route6_realtime_map_preview_label = None
        self.llm_route6_realtime_map_preview_photo = None
        self.llm_route6_realtime_map_layer_combo = None

    def on_route6_select_llm_target(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        output_dir = None
        state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        if state_output:
            output_dir = Path(state_output)
        elif self.route6_update_map_latest_output_dir() is not None:
            output_dir = self.route6_update_map_latest_output_dir()
        context = self.route6_build_realtime_map_planning_context()
        selection = self.route6_select_llm_target_from_context(context)
        semantic_panel = self.refresh_llm_route6_map_analysis_panel()
        semantic_selection = semantic_panel.get("selection", {}) if isinstance(semantic_panel.get("selection", {}), dict) else {}
        semantic_house_id = str(semantic_selection.get("house_id", "") or "")
        if semantic_house_id:
            semantic_candidate = next((item for item in context.get("candidates", []) if isinstance(item, dict) and str(item.get("house_id", "") or "") == semantic_house_id), {})
            if semantic_candidate:
                selection["house_id"] = semantic_house_id
                selection["selected_candidate"] = semantic_candidate
                selection["semantic_target_selection"] = self.route6_json_safe(semantic_selection)
        applied = self.route6_apply_selected_target_to_scan_plan(output_dir, selection)
        house_id = str(applied.get("house_id", "") or "")
        self.llm_route6_status_var.set(f"LLM Route V6: selected target house={house_id or 'n/a'}.")
        self.route6_update_summary_text()
        return applied

    def route6_current_or_new_output_dir(self) -> Path:
        self.ensure_route6_state()
        state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        if state_output and Path(state_output).exists():
            return Path(state_output)
        return self.route6_initialize_run(force_new=False)

    def route6_run_visual_direction_analysis(
        self,
        session: Any,
        output_dir: Path,
    ) -> Dict[str, Any]:
        prompt = str(self.llm_route6_task_prompt_var.get() if hasattr(self.llm_route6_task_prompt_var, "get") else "")
        direction = self.route6_parse_task_direction(prompt) or "north"
        sweep = self.route6_capture_direction_sweep(session, Path(output_dir), requested_direction=direction)
        judgement = self.route6_call_visual_house_llm(Path(output_dir), sweep)
        marker = self.route6_mark_visual_house_candidate_on_map(Path(output_dir), judgement)
        result = {
            "schema": "route6_visual_direction_analysis_result_v1",
            "sweep_manifest_path": str(Path(output_dir) / "route6_visual_direction_sweep" / "sweep_manifest.json"),
            "judgement_path": str(Path(output_dir) / "route6_visual_house_llm_judgement.json"),
            "marker_path": str(Path(output_dir) / "route6_visual_house_marker.json"),
            "target_house_id": str(judgement.get("house_id", "") or ""),
            "marker_object_id": str(marker.get("object_id", "") or ""),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.llm_route6_state["route6_visual_direction_analysis_result"] = self.route6_json_safe(result)
        self.route6_write_state_artifact()
        return result

    def on_route6_capture_direction_sweep(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        session = getattr(self, "session", None)
        if session is None:
            self.llm_route6_visual_status_var.set("LLM Visual Direction Analysis: no active session for visual sweep.")
            return {}
        output_dir = self.route6_current_or_new_output_dir()
        direction = self.route6_parse_task_direction(str(self.llm_route6_task_prompt_var.get() or "")) or "north"
        return self.route6_capture_direction_sweep(session, output_dir, requested_direction=direction)

    def on_route6_analyze_direction_images(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        output_dir = self.route6_current_or_new_output_dir()
        sweep = self.route6_load_json_artifact(output_dir / "route6_visual_direction_sweep" / "sweep_manifest.json", {})
        if not sweep:
            self.llm_route6_visual_status_var.set("LLM Visual Direction Analysis: capture direction sweep first.")
            return {}
        return self.route6_call_visual_house_llm(output_dir, sweep)

    def on_route6_apply_visual_house_marker(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        output_dir = self.route6_current_or_new_output_dir()
        judgement = self.route6_load_json_artifact(output_dir / "route6_visual_house_llm_judgement.json", {})
        if not judgement:
            judgement = self.on_route6_analyze_direction_images()
        if not judgement:
            self.llm_route6_visual_status_var.set("LLM Visual Direction Analysis: no visual judgement available for marker.")
            return {}
        return self.route6_mark_visual_house_candidate_on_map(output_dir, judgement)

    def on_route6_start_visual_target_approach(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        output_dir = self.route6_current_or_new_output_dir()
        marker = self.route6_load_json_artifact(output_dir / "route6_visual_house_marker.json", {})
        if not marker:
            marker = self.on_route6_apply_visual_house_marker()
        if not marker:
            return {}
        target_selection = self.route6_apply_selected_target_to_scan_plan(output_dir)
        semantic_context = self.route6_build_semantic_map_context()
        target = self.route6_plan_llm_navigation_target(output_dir, target_selection, semantic_context)
        self.llm_route6_visual_status_var.set(
            f"LLM Visual Direction Analysis: marker applied; approach house={target.get('house_id', '') or 'n/a'} layer={target.get('approach_layer_key', '') or 'n/a'}"
        )
        return target

    def on_route6_check_height_conflict(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        output_dir = self.route6_current_or_new_output_dir()
        target = self.route6_load_json_artifact(output_dir / "route6_llm_navigation_target.json", {})
        if not target:
            context = self.route6_build_realtime_map_planning_context()
            selection = self.route6_select_llm_target_from_context(context)
            target = self.route6_plan_llm_navigation_target(output_dir, selection, context)
        path_summary = target.get("path_summary", {}) if isinstance(target.get("path_summary", {}), dict) else {}
        layer_state = {
            "blocked_layers": path_summary.get("blocked_layers", []),
            "open_layers": target.get("open_corridor_layers", []),
            "or_state": self.route6_or_avoidance_display_payload().get("or_state", {}) if callable(getattr(self, "route6_or_avoidance_display_payload", None)) else {},
        }
        conflict = self.route6_detect_height_conflict(output_dir, target, layer_state)
        self.route6_decide_height_conflict_avoidance(conflict)
        return conflict

    def on_route6_start_orbit_capture(self) -> Dict[str, Any]:
        self.ensure_route6_state()
        session = getattr(self, "session", None)
        output_dir = self.route6_current_or_new_output_dir()
        marker = self.route6_load_json_artifact(output_dir / "route6_visual_house_marker.json", {})
        if not marker:
            marker = self.on_route6_apply_visual_house_marker()
        conflict = self.route6_load_json_artifact(output_dir / "route6_height_conflict_report.json", {})
        if not conflict:
            conflict = self.on_route6_check_height_conflict()
        plan = self.route6_plan_building_orbit_capture(output_dir, marker, conflict)
        return self.route6_execute_building_orbit_capture(session, output_dir, plan)

    def on_route6_stop_orbit_capture(self) -> None:
        self.ensure_route6_state()
        self.route6_visual_orbit_stop_event.set()
        self.llm_route6_conflict_status_var.set("Height Conflict / Replan: stop orbit capture requested.")

    def route6_release_movement_for_target_search(self, session: Any) -> Dict[str, Any]:
        self.ensure_route6_state()
        if not self.route6_movement_allowed():
            return {"status": "blocked", "reason": "route6_full_stop_or_stop_active"}
        if session is None:
            return {"status": "skipped", "reason": "missing_session"}
        result: Dict[str, Any] = {
            "status": "attempted",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            enable_route5 = getattr(self, "route5_enable_physics_movement", None)
            if callable(enable_route5):
                enable_route5(session)
                result["route5_enable_physics_movement"] = "called"
            else:
                if callable(getattr(session, "set_movement_mode", None)):
                    result["movement_mode"] = session.set_movement_mode("physics")
                if callable(getattr(session, "set_movement_enabled", None)):
                    result["movement_enabled"] = session.set_movement_enabled(True)
            result["status"] = "ok"
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
        self.llm_route6_state["route6_movement_preflight"] = self.route6_json_safe(result)
        self.route6_write_state_artifact()
        return result

    def route6_start_realtime_map_for_target_search(self, session: Any = None) -> Dict[str, Any]:
        self.ensure_route6_state()
        if session is None:
            session = getattr(self, "session", None)
        if session is None:
            result = {"status": "skipped", "reason": "missing_session"}
            self.llm_route6_state["route6_target_search_realtime_map"] = result
            self.route6_update_map_status_var.set("Route 6 Update Map: no active session for target-search realtime update.")
            return result
        realtime_thread = getattr(self, "route6_update_map_realtime_thread", None)
        if realtime_thread is not None and realtime_thread.is_alive():
            output_dir = str((self.llm_route6_state.get("route6_update_map_realtime", {}) if isinstance(self.llm_route6_state.get("route6_update_map_realtime", {}), dict) else {}).get("output_dir", "") or self.llm_route6_state.get("route6_update_map_output_dir", "") or "")
            result = {"status": "already_running", "output_dir": output_dir}
            self.llm_route6_state["route6_target_search_realtime_map"] = result
            self.route6_update_map_status_var.set(f"Route 6 Update Map: realtime already running -> {output_dir or 'n/a'}")
            return result
        capture_thread = getattr(self, "route6_update_map_capture_thread", None)
        if capture_thread is not None and capture_thread.is_alive():
            result = {"status": "blocked", "reason": "capture_already_running"}
            self.llm_route6_state["route6_target_search_realtime_map"] = result
            self.route6_update_map_status_var.set("Route 6 Update Map: stop manual capture before target-search realtime update.")
            return result
        output_dir = self.make_route6_update_map_output_dir()
        self.route6_record_update_map_output_dir(output_dir, source="llm_target_search_realtime")
        self.llm_route6_state["route6_update_map_capture_frame_index"] = 0
        self.route6_update_map_capture_stop_event.clear()
        self.route6_update_map_realtime_stop_event.clear()
        self.route6_update_map_realtime_thread = threading.Thread(
            target=lambda: self.route6_update_map_realtime_worker(session, output_dir),
            daemon=True,
        )
        result = {
            "status": "started",
            "output_dir": str(output_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.llm_route6_state["route6_target_search_realtime_map"] = result
        self.route6_write_state_artifact()
        self.route6_update_map_realtime_thread.start()
        self.route6_update_map_status_var.set(f"Route 6 Update Map: realtime started with LLM target search -> {output_dir}")
        return result

    def on_route6_start_nearest_map_search(self) -> None:
        self.ensure_route6_state()
        if self.llm_route6_thread is not None and self.llm_route6_thread.is_alive():
            self.llm_route6_status_var.set("LLM Route V6: already running.")
            return
        if getattr(self, "route6_full_stop_event", None) is not None and self.route6_full_stop_event.is_set():
            self.llm_route6_status_var.set("LLM Route V6: full stop is active; press Clear before starting movement again.")
            return
        self.llm_route6_stop_event.clear()
        session = getattr(self, "session", None)
        realtime_result = self.route6_start_realtime_map_for_target_search(session)
        movement_result = self.route6_release_movement_for_target_search(session)
        self.llm_route6_thread = threading.Thread(
            target=lambda: self.route6_full_explore_worker(session, force_new=True),
            daemon=True,
        )
        self.llm_route6_thread.start()
        realtime_status = str(realtime_result.get("status", "n/a") if isinstance(realtime_result, dict) else "n/a")
        movement_status = str(movement_result.get("status", "n/a") if isinstance(movement_result, dict) else "n/a")
        self.llm_route6_status_var.set(f"LLM Route V6: worker started; realtime_map={realtime_status}; movement={movement_status}.")

    def on_route6_pause(self) -> None:
        self.ensure_route6_state()
        self.llm_route6_pause_event.set()
        if hasattr(self, "active_nbv_pause_event"):
            try:
                self.active_nbv_pause_event.set()
            except Exception:
                pass
        self.route6_set_stage("PAUSED", "pause requested.")

    def on_route6_resume(self) -> None:
        self.ensure_route6_state()
        self.llm_route6_pause_event.clear()
        if hasattr(self, "active_nbv_pause_event"):
            try:
                self.active_nbv_pause_event.clear()
            except Exception:
                pass
        self.route6_set_stage("SELECT_NEXT_HOUSE", "resume requested.")

    def on_route6_stop(self) -> None:
        self.ensure_route6_state()
        self.llm_route6_stop_event.set()
        self.llm_route6_pause_event.clear()
        self.route6_update_map_realtime_stop_event.set()
        if hasattr(self, "active_nbv_stop_event"):
            try:
                self.active_nbv_stop_event.set()
            except Exception:
                pass
        self.route6_set_stage("STOPPED", "stop requested.")

    def on_route6_full_stop_uav(self) -> None:
        self.ensure_route6_state()
        state_output = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        output_dir = Path(state_output) if state_output else None
        self.route6_apply_full_stop(getattr(self, "session", None), output_dir)

    def on_route6_force_next_house(self) -> None:
        self.ensure_route6_state()
        selected = str((self.llm_route6_state or {}).get("selected_house_id", "") or "")
        if selected:
            house_states = self.llm_route6_state.setdefault("house_states", {})
            house_states[selected] = {
                "schema": "route6_house_state_v1",
                "house_id": selected,
                "status": "needs_rescan",
                "search_status": "force_next_requested",
                "cooldown_active": True,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.route6_write_state_artifact()
        self.llm_route6_force_next_event.set()
        self.route6_set_stage("SELECT_NEXT_HOUSE", f"force next requested; skipped house={selected or 'n/a'}.")

    def on_route6_clear(self) -> None:
        self.ensure_route6_state()
        self.llm_route6_stop_event.clear()
        self.llm_route6_pause_event.clear()
        self.llm_route6_force_next_event.clear()
        self.route6_full_stop_event.clear()
        self.llm_route6_thread = None
        self.llm_route6_state = {}
        self.llm_route6_stage_var.set("Stage: idle")
        self.llm_route6_current_house_var.set("Current house: n/a")
        self.llm_route6_queue_var.set("House queue: n/a")
        self.llm_route6_map_status_var.set("Map: idle")
        self.llm_route6_output_dir_var.set("Output: n/a")
        self.llm_route6_metrics_var.set("Metrics: mapped=0 searched=0 blocked=0 confidence=n/a corrected=n/a")
        self.llm_route6_map_analysis_status_var.set("LLM Map Analysis: idle")
        self.llm_route6_map_analysis_detail_var.set("Semantic target: n/a")
        self.llm_route6_navigation_target_var.set("Navigation target: n/a")
        self.llm_route6_status_var.set("LLM Route V6: cleared.")
        self.route6_update_summary_text()

    def on_route6_save_corrected_map_config(self) -> None:
        self.ensure_route6_state()
        if not bool(self.llm_route6_allow_save_corrected_var.get()):
            self.llm_route6_status_var.set("LLM Route V6: enable corrected-config save before writing global map config.")
            return
        output_dir = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        candidate = Path(output_dir) / "map" / "route6_corrected_houses_config.json" if output_dir else Path()
        if not candidate.is_file():
            self.llm_route6_status_var.set("LLM Route V6: no Route 6 corrected config artifact is available yet.")
            return
        target = PROJECT_ROOT / "assets" / "overhead_map" / DEFAULT_CORRECTED_MAP_CONFIG_NAME
        try:
            target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
            self.llm_route6_status_var.set(f"LLM Route V6: saved corrected config -> {target}")
        except Exception as exc:
            self.llm_route6_status_var.set(f"LLM Route V6: save corrected config failed: {exc}")

    def on_route6_open_latest_output(self) -> None:
        self.ensure_route6_state()
        output_dir = str((self.llm_route6_state or {}).get("output_dir", "") or "")
        path = Path(output_dir) if output_dir else self.route6_output_root()
        if not path.exists():
            self.llm_route6_status_var.set(f"LLM Route V6: output path not found: {path}")
            return
        try:
            os.startfile(str(path))
            self.llm_route6_status_var.set(f"LLM Route V6: opened output -> {path}")
        except Exception as exc:
            self.llm_route6_status_var.set(f"LLM Route V6: open output failed: {exc}; path={path}")
