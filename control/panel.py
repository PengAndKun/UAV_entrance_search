from __future__ import annotations

from .common import *
from .analysis_control import AnalysisControlMixin
from .flight_control import FlightControlMixin
from .lidar_analysis_control import LidarAnalysisControlMixin
from .map_control import MapControlMixin
from .obstacle_avoidance_2_control import ObstacleAvoidance2ControlMixin
from .obstacle_avoidance_llm_control import ObstacleAvoidanceLLMControlMixin
from .obstacle_avoidance_control import ObstacleAvoidanceControlMixin
from .obstacle_representation_control import ObstacleRepresentationControlMixin
from .route_control import RouteControlMixin
from .route4_fusion_control import Route4FusionControlMixin


class RunDroneFlightPanel(
    FlightControlMixin,
    MapControlMixin,
    RouteControlMixin,
    Route4FusionControlMixin,
    AnalysisControlMixin,
    LidarAnalysisControlMixin,
    ObstacleAvoidanceControlMixin,
    ObstacleAvoidance2ControlMixin,
    ObstacleAvoidanceLLMControlMixin,
    ObstacleRepresentationControlMixin,
):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session: Optional[flight.DroneFlightSession] = None
        self.latest_state: Dict[str, Any] = {}
        self.manual_request_inflight = False
        self.move_request_inflight = False
        self.keyboard_request_inflight = False
        self.state_refresh_inflight = False
        self.preview_refresh_inflight = False
        self.temp_capture_inflight = False
        self.temp_capture_lidar_inflight = False
        self.stream_capture_thread: Optional[threading.Thread] = None
        self.stream_capture_stop_event = threading.Event()
        self.stream_capture_dir: Optional[Path] = None
        self.stream_capture_trajectory: List[Dict[str, Any]] = []
        self.lidar_stream_capture_thread: Optional[threading.Thread] = None
        self.lidar_stream_capture_stop_event = threading.Event()
        self.lidar_stream_capture_dir: Optional[Path] = None
        self.lidar_stream_capture_trajectory: List[Dict[str, Any]] = []
        self.lidar_stream_reconstruction_cloud: Optional[np.ndarray] = None
        self.lidar_stream_source_point_count = 0
        self.lidar_stream_last_reconstruction: Dict[str, Any] = {}
        self.lidar_stream_last_map_refresh = 0.0
        self.sequence_thread: Optional[threading.Thread] = None
        self.sequence_stop_event = threading.Event()
        self.route_thread: Optional[threading.Thread] = None
        self.route_stop_event = threading.Event()

        self.root = tk.Tk()
        self.root.title("Run Drone Flight Controller")
        self.root.geometry("1040x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.pose_var = tk.StringVar(value="Pose: not started")
        self.session_var = tk.StringVar(value="Session: idle")
        self.control_var = tk.StringVar(value="Movement enabled=0")
        self.map_status_var = tk.StringVar(value="Map: closed")
        self.map_pose_var = tk.StringVar(value="Map pose: n/a")
        self.map_calibration_var = tk.StringVar(value="Calibration: idle")
        self.xy_tolerance_var = tk.StringVar(value="60")
        self.z_tolerance_var = tk.StringVar(value="80")
        self.marker_class_var = tk.StringVar(value=flight.DEFAULT_CALIBRATION_MARKER_CLASS)
        self.marker_scale_var = tk.StringVar(
            value=" ".join(str(value) for value in flight.DEFAULT_CALIBRATION_MARKER_SCALE)
        )
        self.llm_route_status_var = tk.StringVar(value="LLM Route: idle")
        self.llm_route_map_status_var = tk.StringVar(value="Route Map: idle")
        self.llm_route_target_var = tk.StringVar(value="")
        self.llm_task_text_var = tk.StringVar(value="Search the selected house entrance.")
        self.llm_api_style_var = tk.StringVar(value=normalize_llm_api_style(getattr(args, "llm_api_style", default_llm_api_style())))
        self.llm_base_url_var = tk.StringVar(value=str(getattr(args, "llm_base_url", "") or ""))
        self.llm_api_key_var = tk.StringVar(value=str(getattr(args, "llm_api_key", "") or ""))
        self.llm_model_var = tk.StringVar(value=str(getattr(args, "llm_model", "") or ""))
        self.llm_timeout_s_var = tk.StringVar(value=str(getattr(args, "llm_route_timeout_s", 60.0)))
        self.route_step_cm_var = tk.StringVar(value=str(getattr(args, "route_step_cm", 120.0)))
        self.route_delay_ms_var = tk.StringVar(value=str(getattr(args, "route_delay_ms", 100.0)))
        self.llm_route_standoff_cm_var = tk.StringVar(value=str(LLM_ROUTE_SCAN_STANDOFF_CM))
        self.llm_route_scan_spacing_cm_var = tk.StringVar(value=str(LLM_ROUTE_SCAN_SPACING_CM))
        self.llm_route_capture_count_var = tk.StringVar(value=str(LLM_ROUTE_CAPTURE_COUNT))
        self.route_capture_interval_s_var = tk.StringVar(
            value=str(getattr(args, "route_capture_interval_s", LLM_ROUTE_PATH_CAPTURE_INTERVAL_S))
        )
        self.llm_route2_status_var = tk.StringVar(value="LLM Route V2: idle")
        self.llm_route2_map_status_var = tk.StringVar(value="Route V2 Map: idle")
        self.llm_route2_facade_var = tk.StringVar(value="Facade: n/a")
        self.llm_route2_selected_facade_var = tk.StringVar(value="auto")
        self.llm_route2_facade_status_var = tk.StringVar(value="Processed: no facade")
        self.llm_route2_rgb_status_var = tk.StringVar(value="Facade RGB: none")
        self.llm_route2_progress_text_var = tk.StringVar(value="Explore: 0%")
        self.llm_route2_progress_var = tk.DoubleVar(value=0.0)
        self.llm_route2_auto_refresh_var = tk.BooleanVar(value=False)
        self.llm_route2_floor_height_m_var = tk.StringVar(value=str(LLM_ROUTE2_DEFAULT_FLOOR_HEIGHT_M))
        self.llm_route2_default_floors_var = tk.StringVar(value=str(LLM_ROUTE2_DEFAULT_FLOORS))
        self.llm_route2_low_z_cm_var = tk.StringVar(value=str(LLM_ROUTE2_LOW_Z_CM))
        self.llm_route2_z_step_cm_var = tk.StringVar(value=str(LLM_ROUTE2_Z_STEP_CM))
        self.llm_route2_density_mode_var = tk.StringVar(value="auto")
        self.llm_route3_status_var = tk.StringVar(value="LLM Route V3: idle")
        self.llm_route3_map_status_var = tk.StringVar(value="Route V3 Map: idle")
        self.llm_route3_stage_var = tk.StringVar(value="Stage: idle")
        self.llm_route3_active_var = tk.StringVar(value="Active: n/a")
        self.llm_route3_target_var = tk.StringVar(value="Target: n/a")
        self.llm_route3_error_var = tk.StringVar(value="Error: n/a")
        self.llm_route3_payload_var = tk.StringVar(value="Payload: hold")
        self.llm_route3_progress_text_var = tk.StringVar(value="Autonomy: 0%")
        self.llm_route3_progress_var = tk.DoubleVar(value=0.0)
        self.llm_route3_task_status_var = tk.StringVar(value="Task Plan: n/a")
        self.llm_route3_target_sequence_var = tk.StringVar(value="Targets: n/a")
        self.llm_route3_current_status_var = tk.StringVar(value="Current: idle")
        self.llm_route3_next_status_var = tk.StringVar(value="Next: n/a")
        self.llm_route3_auto_refresh_var = tk.BooleanVar(value=False)
        self.llm_route3_paused_var = tk.BooleanVar(value=False)
        self.llm_route3_move_tick_ms_var = tk.StringVar(value="150")
        self.llm_route3_nav_step_cm_var = tk.StringVar(value="20")
        self.llm_route3_reach_tol_cm_var = tk.StringVar(value="60")
        self.llm_route3_z_tol_cm_var = tk.StringVar(value="40")
        self.llm_route3_yaw_tol_deg_var = tk.StringVar(value="10")
        self.llm_route3_max_stage_s_var = tk.StringVar(value="90")

        self.env_platform_var = tk.StringVar(value=args.env_platform)
        self.env_root_var = tk.StringVar(value=args.env_root or "UnrealEnv")
        self.env_bin_var = tk.StringVar(value=args.env_bin or "")
        self.output_dir_var = tk.StringVar(value=args.output_dir)
        self.width_var = tk.StringVar(value=str(args.width))
        self.height_var = tk.StringVar(value=str(args.height))
        self.launch_sleep_var = tk.StringVar(value=str(args.launch_sleep))
        self.time_dilation_var = tk.StringVar(value=str(args.time_dilation))
        self.step_delay_var = tk.StringVar(value=str(args.step_delay))
        self.save_every_var = tk.StringVar(value=str(args.save_every))
        self.movement_mode_var = tk.StringVar(value=args.movement_mode)
        self.keyboard_enabled_var = tk.BooleanVar(value=True)
        self.keyboard_interval_ms_var = tk.StringVar(
            value=str(getattr(args, "keyboard_interval_ms", DEFAULT_KEYBOARD_INTERVAL_MS))
        )
        self.keyboard_status_var = tk.StringVar(value="Keyboard: idle")
        self.fpv_camera_x_var = tk.StringVar(value="0")
        self.fpv_camera_y_var = tk.StringVar(value="0")
        self.fpv_camera_z_var = tk.StringVar(value="0")
        self.fpv_camera_roll_var = tk.StringVar(value="0")
        self.fpv_camera_pitch_var = tk.StringVar(value="0")
        self.fpv_camera_yaw_var = tk.StringVar(value="0")
        self.fpv_camera_status_var = tk.StringVar(value="FPV Camera: idle")
        self.fpv_camera_info_var = tk.StringVar(value="Camera Info: not loaded")
        self.initial_pose_var = tk.StringVar(value=" ".join(str(value) for value in args.initial_pos))

        self.sequence_var = tk.StringVar(value="")
        self.sequence_delay_var = tk.StringVar(value="150")
        self.auto_rgb_var = tk.BooleanVar(value=False)
        self.enhance_rgb_var = tk.BooleanVar(value=bool(args.enhance_rgb))
        self.rgb_source_order_var = tk.StringVar(
            value=str(getattr(args, "rgb_source_order", flight.DEFAULT_RGB_SOURCE_ORDER) or flight.DEFAULT_RGB_SOURCE_ORDER)
        )
        self.lidar_depth_min_cm_var = tk.StringVar(
            value=str(getattr(args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
        )
        self.lidar_depth_max_cm_var = tk.StringVar(
            value=str(getattr(args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
        )
        self.lidar_capture_processing_var = tk.StringVar(
            value=flight.normalize_lidar_capture_processing(
                getattr(args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING)
            )
        )
        self.stream_task_title_var = tk.StringVar(value="stream_task")
        self.stream_interval_s_var = tk.StringVar(
            value=str(getattr(args, "stream_interval_s", flight.DEFAULT_STREAM_CAPTURE_INTERVAL_S))
        )
        self.stream_status_var = tk.StringVar(value="Stream Capture: idle")
        self.lidar_stream_status_var = tk.StringVar(value="Lidar Stream: idle")
        self.lidar_stream_analysis_status_var = tk.StringVar(value="Lidar Analysis: idle")
        self.stream_player_status_var = tk.StringVar(value="Stream Player: idle")
        self.stream_player_image_mode_var = tk.StringVar(value="rgb")
        self.obstacle_avoidance_data_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "obstacle_avoidance_data"))
        self.obstacle_avoidance_session_var = tk.StringVar(value="obstacle_avoidance")
        self.obstacle_avoidance_interval_s_var = tk.StringVar(value="0.5")
        self.obstacle_avoidance_stage_var = tk.StringVar(value="manual_expert")
        self.obstacle_avoidance_scenario_var = tk.StringVar(value="S0")
        self.obstacle_avoidance_method_var = tk.StringVar(value="manual_keyboard")
        self.obstacle_avoidance_run_id_var = tk.StringVar(value="001")
        self.obstacle_avoidance_geometry_label_var = tk.StringVar(value="unknown")
        self.obstacle_avoidance_operator_note_var = tk.StringVar(value="")
        self.obstacle_avoidance_risk_var = tk.StringVar(value="SAFE")
        self.obstacle_avoidance_expert_action_var = tk.StringVar(value="hold")
        self.obstacle_avoidance_collision_var = tk.BooleanVar(value=False)
        self.obstacle_avoidance_status_var = tk.StringVar(value="Obstacle Avoidance: idle")
        self.obstacle_plan_json_path_var = tk.StringVar(
            value=str(PROJECT_ROOT / "obstacle_avoidance_data" / "plans" / "obstacle_avoidance_plans.json")
        )
        self.obstacle_plan_project_var = tk.StringVar(value="default_route_episodes")
        self.obstacle_plan_project_name_var = tk.StringVar(value="Default 10 route episodes")
        self.obstacle_plan_environment_var = tk.StringVar(value="default_unreal_scene")
        self.obstacle_plan_method_var = tk.StringVar(value="geometry_rule_v0")
        self.obstacle_plan_selected_episode_var = tk.StringVar(value="")
        self.obstacle_plan_runner_status_var = tk.StringVar(value="Plan runner: idle")
        self.obstacle_plan_episode_enabled_var = tk.BooleanVar(value=True)
        self.obstacle_plan_start_pose_var = tk.StringVar(value="")
        self.obstacle_plan_goal_pose_var = tk.StringVar(value="")
        self.obstacle_plan_scenario_var = tk.StringVar(value="")
        self.obstacle_plan_obstacle_hint_var = tk.StringVar(value="unknown")
        self.obstacle_plan_operator_note_var = tk.StringVar(value="")
        self.obstacle_avoidance_2_data_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "obstacle_avoidance_2_data"))
        self.obstacle_avoidance_2_plan_json_var = tk.StringVar(
            value=str(PROJECT_ROOT / "obstacle_avoidance_2_data" / "plans" / "obstacle_avoidance_2_plans.json")
        )
        self.obstacle_avoidance_2_project_var = tk.StringVar(value="route_obstacle_collection_v2")
        self.obstacle_avoidance_2_project_name_var = tk.StringVar(value="OA2 default route obstacle collection")
        self.obstacle_avoidance_2_environment_var = tk.StringVar(value="default_unreal_scene")
        self.obstacle_avoidance_2_method_var = tk.StringVar(value="geometry_rule_v0")
        self.obstacle_avoidance_2_status_var = tk.StringVar(value="Obstacle Avoidance 2: idle")
        self.obstacle_avoidance_2_episode_id_var = tk.StringVar(value="")
        self.obstacle_avoidance_2_enabled_var = tk.BooleanVar(value=True)
        self.obstacle_avoidance_2_start_pose_var = tk.StringVar(value="")
        self.obstacle_avoidance_2_goal_pose_var = tk.StringVar(value="")
        self.obstacle_avoidance_2_scenario_var = tk.StringVar(value="")
        self.obstacle_avoidance_2_obstacle_hint_var = tk.StringVar(value="unknown")
        self.obstacle_avoidance_2_note_var = tk.StringVar(value="")
        obstacle_representation_plus_model = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_plus_model.pt"
        obstacle_representation_legacy_model = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_model.pt"
        self.obstacle_representation_model_var = tk.StringVar(
            value=str(obstacle_representation_plus_model if obstacle_representation_plus_model.is_file() else obstacle_representation_legacy_model)
        )
        self.obstacle_representation_status_var = tk.StringVar(value="Obstacle Representation: idle")
        self.obstacle_representation_result_var = tk.StringVar(value="Result: --")
        self.obstacle_representation_capture_dir_var = tk.StringVar(value="Capture: --")
        self.stream_analysis_status_var = tk.StringVar(value="Analysis: idle")
        self.stream_analysis_stream_dir_var = tk.StringVar(value="")
        self.stream_analysis_weights_var = tk.StringVar(value="")
        self.stream_analysis_device_var = tk.StringVar(value="0")
        self.stream_analysis_conf_var = tk.StringVar(value="0.25")
        self.stream_analysis_imgsz_var = tk.StringVar(value="640")
        self.stream_analysis_stride_var = tk.StringVar(value="1")
        self.stream_analysis_max_frames_var = tk.StringVar(value="0")
        self.stream_analysis_progress_var = tk.DoubleVar(value=0.0)
        self.stream_analysis_map_pose_var = tk.StringVar(value="Map pose: n/a")
        self.stream_analysis_map_shift_step_var = tk.StringVar(value="5")
        self.stream_analysis_live_pose_var = tk.BooleanVar(value=False)
        self.show_houses_var = tk.BooleanVar(value=True)
        self.show_trajectory_var = tk.BooleanVar(value=True)
        self.show_calibration_points_var = tk.BooleanVar(value=False)

        self.orbit_center_x_var = tk.StringVar(value=str(args.orbit_center[0]))
        self.orbit_center_y_var = tk.StringVar(value=str(args.orbit_center[1]))
        self.orbit_radius_var = tk.StringVar(value=str(args.orbit_radius))
        self.orbit_altitude_var = tk.StringVar(value=str(args.orbit_altitude))
        self.orbit_steps_var = tk.StringVar(value=str(args.orbit_steps))
        self.orbit_start_angle_var = tk.StringVar(value=str(args.orbit_start_angle))
        self.orbit_clockwise_var = tk.BooleanVar(value=bool(args.orbit_clockwise))

        self.movement_enabled_state = False
        self.movement_mode_state = args.movement_mode
        self.movement_toggle_button: Optional[tk.Button] = None
        self.keyboard_pressed_symbols: set[str] = set()
        self.camera_toggle_key_down = False
        self.keyboard_loop_after_id: Optional[str] = None
        self.fpv_camera_window: Optional[tk.Toplevel] = None
        self.fpv_camera_info_text: Optional[tk.Text] = None
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.stream_player_window: Optional[tk.Toplevel] = None
        self.stream_player_label: Optional[tk.Label] = None
        self.stream_player_photo: Optional[ImageTk.PhotoImage] = None
        self.stream_task_entry: Optional[tk.Entry] = None
        self.stream_player_frames: List[Path] = []
        self.stream_player_index = 0
        self.stream_player_after_id: Optional[str] = None
        self.stream_player_playing = False
        self.stream_player_dir: Optional[Path] = None
        self.stream_player_interval_ms = int(float(getattr(args, "stream_interval_s", flight.DEFAULT_STREAM_CAPTURE_INTERVAL_S)) * 1000)
        self.stream_analysis_window: Optional[tk.Toplevel] = None
        self.stream_analysis_thread: Optional[threading.Thread] = None
        self.stream_analysis_stop_event = threading.Event()
        self.stream_analysis_rows: List[Dict[str, Any]] = []
        self.stream_analysis_result: Dict[str, Any] = {}
        self.stream_analysis_listbox: Optional[tk.Listbox] = None
        self.stream_analysis_preview_label: Optional[tk.Label] = None
        self.stream_analysis_preview_photo: Optional[ImageTk.PhotoImage] = None
        self.stream_analysis_summary_text: Optional[tk.Text] = None
        self.stream_analysis_json_text: Optional[tk.Text] = None
        self.stream_analysis_progressbar: Optional[ttk.Progressbar] = None
        self.stream_analysis_map_widget: Optional[OverheadMapWidget] = None
        self.stream_analysis_pose_cache: Dict[str, Dict[str, Any]] = {}
        self.stream_analysis_live_after_id: Optional[str] = None
        self.stream_analysis_current_row: Optional[Dict[str, Any]] = None
        self.lidar_analysis_window: Optional[tk.Toplevel] = None
        self.lidar_analysis_stream_dir_var = tk.StringVar(value="")
        self.lidar_analysis_mode_var = tk.StringVar(value="Cumulative")
        self.lidar_analysis_max_points_var = tk.StringVar(value="60000")
        self.lidar_analysis_color_mode_var = tk.StringVar(value="RGB")
        self.lidar_analysis_point_size_var = tk.StringVar(value="1.0")
        self.lidar_analysis_view_preset_var = tk.StringVar(value="Perspective")
        self.lidar_analysis_rows: List[Dict[str, Any]] = []
        self.lidar_analysis_listbox: Optional[tk.Listbox] = None
        self.lidar_analysis_summary_text: Optional[tk.Text] = None
        self.lidar_analysis_json_text: Optional[tk.Text] = None
        self.lidar_analysis_fig: Any = None
        self.lidar_analysis_ax: Any = None
        self.lidar_analysis_canvas: Any = None
        self.lidar_analysis_toolbar: Any = None
        self.lidar_analysis_after_id: Optional[str] = None
        self.lidar_analysis_playing = False
        self.lidar_analysis_index = 0
        self.lidar_analysis_cached_index = -1
        self.lidar_analysis_cached_cloud: Optional[np.ndarray] = None
        self.lidar_analysis_rebuild_thread: Optional[threading.Thread] = None
        self.lidar_analysis_export_thread: Optional[threading.Thread] = None
        self.lidar_analysis_open3d_thread: Optional[threading.Thread] = None
        self.obstacle_avoidance_window: Optional[tk.Toplevel] = None
        self.obstacle_avoidance_report_text: Optional[tk.Text] = None
        self.obstacle_avoidance_capture_thread: Optional[threading.Thread] = None
        self.obstacle_avoidance_task_thread: Optional[threading.Thread] = None
        self.obstacle_avoidance_stop_event = threading.Event()
        self.obstacle_avoidance_session_dir: Optional[Path] = None
        self.obstacle_avoidance_frame_index = 0
        self.obstacle_representation_window: Optional[tk.Toplevel] = None
        self.obstacle_representation_rgb_label: Optional[tk.Label] = None
        self.obstacle_representation_mask_label: Optional[tk.Label] = None
        self.obstacle_representation_rgb_photo: Optional[ImageTk.PhotoImage] = None
        self.obstacle_representation_mask_photo: Optional[ImageTk.PhotoImage] = None
        self.obstacle_representation_report_text: Optional[tk.Text] = None
        self.obstacle_representation_thread: Optional[threading.Thread] = None
        self.obstacle_representation_frame_index = 0
        self.map_window: Optional[tk.Toplevel] = None
        self.map_widget: Optional[OverheadMapWidget] = None
        self.map_refresh_inflight = False
        self.map_config: Dict[str, Any] = {}
        self.map_config_path: Optional[Path] = None
        self.map_image_path: Optional[Path] = None
        self.map_image: Optional[np.ndarray] = None
        self.map_calibration: Dict[str, Any] = {}
        self.map_display_offset_px: Tuple[float, float] = (0.0, 0.0)
        self.map_world_bounds: Tuple[float, float, float, float] = DEFAULT_MAP_BOUNDS
        self.map_touch_state: Dict[str, Any] = {}
        self.map_touch_poll_inflight = False
        self.map_touch_auto_saved = False
        self.house_target_combo: Optional[ttk.Combobox] = None
        self.house_target_combos: List[ttk.Combobox] = []
        self.house_choice_map: Dict[str, str] = {}
        self.house_display_by_id: Dict[str, str] = {}
        self.llm_task_plan: Dict[str, Any] = {}
        self.llm_route_plan: Dict[str, Any] = {}
        self.llm_route_scan_points: List[Dict[str, Any]] = []
        self.llm_route_lidar_trajectory: List[Dict[str, Any]] = []
        self.llm_route_execution_summary: Dict[str, Any] = {}
        self.llm_route_validation_report: Dict[str, Any] = {}
        self.house_search_dir: Optional[Path] = None
        self.llm_route_window: Optional[tk.Toplevel] = None
        self.llm_route_window_canvas: Optional[tk.Canvas] = None
        self.llm_route_window_content: Optional[tk.Frame] = None
        self.llm_route_window_content_window: Optional[int] = None
        self.llm_route_map_widget: Optional[OverheadMapWidget] = None
        self.llm_route2_window: Optional[tk.Toplevel] = None
        self.llm_route2_window_canvas: Optional[tk.Canvas] = None
        self.llm_route2_window_content: Optional[tk.Frame] = None
        self.llm_route2_window_content_window: Optional[int] = None
        self.llm_route2_map_widget: Optional[OverheadMapWidget] = None
        self.llm_route2_preview_text: Optional[tk.Text] = None
        self.llm_route2_analysis_text: Optional[tk.Text] = None
        self.llm_route2_rgb_label: Optional[tk.Widget] = None
        self.llm_route2_rgb_photo: Optional[ImageTk.PhotoImage] = None
        self.llm_route2_auto_refresh_job: Optional[str] = None
        self.llm_route2_state: Dict[str, Any] = {}
        self.llm_route2_completed_facades: set[str] = set()
        self.llm_route3_window: Optional[tk.Toplevel] = None
        self.llm_route3_window_canvas: Optional[tk.Canvas] = None
        self.llm_route3_window_content: Optional[tk.Frame] = None
        self.llm_route3_window_content_window: Optional[int] = None
        self.llm_route3_map_widget: Optional[OverheadMapWidget] = None
        self.llm_route3_preview_text: Optional[tk.Text] = None
        self.llm_route3_analysis_text: Optional[tk.Text] = None
        self.llm_route3_rgb_label: Optional[tk.Widget] = None
        self.llm_route3_rgb_photo: Optional[ImageTk.PhotoImage] = None
        self.llm_route3_auto_refresh_job: Optional[str] = None
        self.llm_route3_state: Dict[str, Any] = {}
        self.llm_route3_completed_facades: set[str] = set()
        self.llm_route3_blocked_facades: set[str] = set()
        self.llm_route3_thread: Optional[threading.Thread] = None
        self.llm_route3_stop_event = threading.Event()
        self.llm_route3_pause_event = threading.Event()
        self.llm_route3_control_locked = False
        self.llm_route_preview_text: Optional[tk.Text] = None
        self.llm_route_preview_texts: List[tk.Text] = []
        self.main_canvas: Optional[tk.Canvas] = None
        self.content_frame: Optional[tk.Frame] = None
        self.content_window: Optional[int] = None

        self._build_ui()
        self.apply_llm_api_defaults(force=False)
        self.load_map_resources(force=True)
        self.refresh_house_target_choices()
        self.refresh_route_preview()
        self._bind_hotkeys()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)
        self.root.after(self.args.map_interval_ms, self.schedule_map_refresh)

    def _on_content_frame_configure(self, _event: tk.Event) -> None:
        if self.main_canvas is None:
            return
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event: tk.Event) -> None:
        if self.main_canvas is None or self.content_window is None or self.content_frame is None:
            return
        requested_width = max(self.content_frame.winfo_reqwidth(), int(event.width))
        self.main_canvas.itemconfigure(self.content_window, width=requested_width)
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.main_canvas is None:
            return
        delta = -1 if event.delta > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            self.main_canvas.xview_scroll(delta, "units")
        else:
            self.main_canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        if self.main_canvas is None:
            return
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        self.main_canvas.yview_scroll(direction, "units")

    def _on_llm_route_window_content_configure(self, _event: tk.Event) -> None:
        if self.llm_route_window_canvas is None:
            return
        self.llm_route_window_canvas.configure(scrollregion=self.llm_route_window_canvas.bbox("all"))

    def _on_llm_route_window_canvas_configure(self, event: tk.Event) -> None:
        if (
            self.llm_route_window_canvas is None
            or self.llm_route_window_content is None
            or self.llm_route_window_content_window is None
        ):
            return
        requested_width = max(self.llm_route_window_content.winfo_reqwidth(), int(event.width))
        self.llm_route_window_canvas.itemconfigure(self.llm_route_window_content_window, width=requested_width)
        self.llm_route_window_canvas.configure(scrollregion=self.llm_route_window_canvas.bbox("all"))

    def _on_llm_route_window_mousewheel(self, event: tk.Event):
        if self.llm_route_window_canvas is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            self.llm_route_window_canvas.xview_scroll(delta, "units")
        else:
            self.llm_route_window_canvas.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route_window_mousewheel_linux(self, event: tk.Event):
        if self.llm_route_window_canvas is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        self.llm_route_window_canvas.yview_scroll(direction, "units")
        return "break"

    def _bind_llm_route_window_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_llm_route_window_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route_window_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route_window_mousewheel_linux, add="+")
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            self._bind_llm_route_window_mousewheel_tree(child)

    def _on_llm_route2_window_content_configure(self, _event: tk.Event) -> None:
        if self.llm_route2_window_canvas is None:
            return
        self.llm_route2_window_canvas.configure(scrollregion=self.llm_route2_window_canvas.bbox("all"))

    def _on_llm_route2_window_canvas_configure(self, event: tk.Event) -> None:
        if (
            self.llm_route2_window_canvas is None
            or self.llm_route2_window_content is None
            or self.llm_route2_window_content_window is None
        ):
            return
        requested_width = max(self.llm_route2_window_content.winfo_reqwidth(), int(event.width))
        self.llm_route2_window_canvas.itemconfigure(self.llm_route2_window_content_window, width=requested_width)
        self.llm_route2_window_canvas.configure(scrollregion=self.llm_route2_window_canvas.bbox("all"))

    def _on_llm_route2_window_mousewheel(self, event: tk.Event):
        if self.llm_route2_window_canvas is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            self.llm_route2_window_canvas.xview_scroll(delta, "units")
        else:
            self.llm_route2_window_canvas.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route2_window_mousewheel_linux(self, event: tk.Event):
        if self.llm_route2_window_canvas is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        self.llm_route2_window_canvas.yview_scroll(direction, "units")
        return "break"

    def _bind_llm_route2_window_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_llm_route2_window_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route2_window_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route2_window_mousewheel_linux, add="+")
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            self._bind_llm_route2_window_mousewheel_tree(child)

    def _on_llm_route3_window_content_configure(self, _event: tk.Event) -> None:
        if self.llm_route3_window_canvas is None:
            return
        self.llm_route3_window_canvas.configure(scrollregion=self.llm_route3_window_canvas.bbox("all"))

    def _on_llm_route3_window_canvas_configure(self, event: tk.Event) -> None:
        if (
            self.llm_route3_window_canvas is None
            or self.llm_route3_window_content is None
            or self.llm_route3_window_content_window is None
        ):
            return
        requested_width = max(self.llm_route3_window_content.winfo_reqwidth(), int(event.width))
        self.llm_route3_window_canvas.itemconfigure(self.llm_route3_window_content_window, width=requested_width)
        self.llm_route3_window_canvas.configure(scrollregion=self.llm_route3_window_canvas.bbox("all"))

    def _on_llm_route3_window_mousewheel(self, event: tk.Event):
        if self.llm_route3_window_canvas is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            self.llm_route3_window_canvas.xview_scroll(delta, "units")
        else:
            self.llm_route3_window_canvas.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route3_window_mousewheel_linux(self, event: tk.Event):
        if self.llm_route3_window_canvas is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        self.llm_route3_window_canvas.yview_scroll(direction, "units")
        return "break"

    def _bind_llm_route3_window_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_llm_route3_window_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route3_window_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route3_window_mousewheel_linux, add="+")
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            self._bind_llm_route3_window_mousewheel_tree(child)

    def _on_pose_text_mousewheel(self, event: tk.Event):
        if not hasattr(self, "pose_text"):
            return None
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            self.pose_text.xview_scroll(delta, "units")
        else:
            self.pose_text.yview_scroll(delta, "units")
        return "break"

    def _on_pose_text_mousewheel_linux(self, event: tk.Event):
        if not hasattr(self, "pose_text"):
            return None
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        self.pose_text.yview_scroll(direction, "units")
        return "break"

    def _build_ui(self) -> None:
        self.main_canvas = tk.Canvas(self.root, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        h_scrollbar = tk.Scrollbar(self.root, orient="horizontal", command=self.main_canvas.xview)
        self.main_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        outer = tk.Frame(self.main_canvas)
        self.content_frame = outer
        self.content_window = self.main_canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", self._on_content_frame_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self._on_mousewheel_linux)
        outer.grid_columnconfigure(0, weight=1)

        status = tk.LabelFrame(outer, text="Status")
        status.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        status.grid_columnconfigure(0, weight=1)
        for idx, var in enumerate((self.status_var, self.pose_var, self.session_var, self.control_var)):
            tk.Label(status, textvariable=var, anchor="w").grid(row=idx, column=0, sticky="ew", padx=6, pady=2)

        launch = tk.LabelFrame(outer, text="Launch")
        launch.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3, 5):
            launch.grid_columnconfigure(col, weight=1)
        tk.Label(launch, text="Platform").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(launch, textvariable=self.env_platform_var, values=("auto", "win", "mac", "linux"),
                     width=8, state="readonly").grid(row=0, column=1, sticky="w", padx=6, pady=4)
        tk.Label(launch, text="Env Root").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        tk.Entry(launch, textvariable=self.env_root_var).grid(row=0, column=3, sticky="ew", padx=6, pady=4)
        tk.Button(launch, text="Browse", command=self.on_browse_env_root).grid(row=0, column=4, padx=6, pady=4)
        tk.Label(launch, text="Env Bin").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        tk.Entry(launch, textvariable=self.env_bin_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=4)
        tk.Label(launch, text="Output").grid(row=1, column=4, sticky="w", padx=6, pady=4)
        tk.Entry(launch, textvariable=self.output_dir_var).grid(row=1, column=5, sticky="ew", padx=6, pady=4)

        row = tk.Frame(launch)
        row.grid(row=2, column=0, columnspan=6, sticky="ew", padx=0, pady=2)
        for label, var, width in (
            ("W", self.width_var, 7),
            ("H", self.height_var, 7),
            ("Sleep", self.launch_sleep_var, 7),
            ("TimeDil", self.time_dilation_var, 7),
            ("Delay", self.step_delay_var, 7),
            ("SaveEvery", self.save_every_var, 7),
            ("Initial Pose", self.initial_pose_var, 34),
        ):
            tk.Label(row, text=label).pack(side="left", padx=(6, 2))
            tk.Entry(row, textvariable=var, width=width).pack(side="left", padx=(0, 8))
        tk.Button(row, text="Start Unreal", command=self.on_start_session).pack(side="left", padx=6)
        tk.Button(row, text="Stop", command=self.on_stop_session).pack(side="left", padx=6)

        move = tk.LabelFrame(outer, text="Basic Movement")
        move.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        for col in range(4):
            move.grid_columnconfigure(col, weight=1)
        tk.Label(move, text="Movement Mode").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        mode_combo = ttk.Combobox(
            move,
            textvariable=self.movement_mode_var,
            values=("pose_lock", "physics"),
            state="readonly",
            width=12,
        )
        mode_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        mode_combo.bind("<<ComboboxSelected>>", self.on_movement_mode_selected)
        self.movement_toggle_button = tk.Button(move, text="Enable Basic Movement", command=self.on_toggle_movement)
        self.movement_toggle_button.grid(row=0, column=2, columnspan=2, sticky="ew", padx=6, pady=6)
        layout = [
            ("Q Yaw Left", "q", 1, 0),
            ("W Forward", "w", 1, 1),
            ("E Yaw Right", "e", 1, 2),
            ("R Up", "r", 1, 3),
            ("A Left", "a", 2, 0),
            ("X Hold", "x", 2, 1),
            ("D Right", "d", 2, 2),
            ("F Down", "f", 2, 3),
            ("S Backward", "s", 3, 1),
        ]
        for label, symbol, row_idx, col_idx in layout:
            tk.Button(move, text=label, command=lambda s=symbol: self.send_move_symbol(s)).grid(
                row=row_idx, column=col_idx, sticky="ew", padx=6, pady=4
            )
        tk.Button(move, text="Toggle View (Z)", command=self.on_toggle_camera_view).grid(
            row=3, column=2, sticky="ew", padx=6, pady=4
        )
        tk.Button(move, text="FPV Camera", command=self.open_first_person_camera_window).grid(
            row=3, column=3, sticky="ew", padx=6, pady=4
        )
        keyboard_row = tk.Frame(move)
        keyboard_row.grid(row=4, column=0, columnspan=4, sticky="ew", padx=0, pady=(2, 6))
        tk.Checkbutton(
            keyboard_row,
            text="Keyboard Hold",
            variable=self.keyboard_enabled_var,
            command=self.on_keyboard_enabled_changed,
        ).pack(side="left", padx=6)
        tk.Label(keyboard_row, text="Key ms").pack(side="left", padx=(12, 2))
        tk.Entry(keyboard_row, textvariable=self.keyboard_interval_ms_var, width=6).pack(side="left", padx=(0, 12))
        tk.Button(keyboard_row, text="Focus Keys", command=self.focus_keyboard_control).pack(side="left", padx=(0, 12))
        tk.Button(keyboard_row, text="Stream Start (,)", command=self.on_start_stream_capture).pack(side="left", padx=(0, 6))
        tk.Button(keyboard_row, text="Stream Stop (.)", command=self.on_stop_stream_capture).pack(side="left", padx=(0, 12))
        tk.Label(keyboard_row, textvariable=self.keyboard_status_var, anchor="w").pack(
            side="left", fill="x", expand=True, padx=6
        )

        seq = tk.LabelFrame(outer, text="Sequence")
        seq.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        seq.grid_columnconfigure(1, weight=1)
        tk.Label(seq, text="Symbols").grid(row=0, column=0, padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(seq, text="Delay ms").grid(row=0, column=2, padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_delay_var, width=8).grid(row=0, column=3, padx=6, pady=6)
        tk.Button(seq, text="Execute", command=self.on_execute_sequence).grid(row=0, column=4, padx=6, pady=6)
        tk.Button(seq, text="Stop Sequence", command=self.on_stop_sequence).grid(row=0, column=5, padx=6, pady=6)

        pose = tk.LabelFrame(outer, text="Pose")
        pose.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        pose_text_frame = tk.Frame(pose)
        pose_text_frame.grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.pose_text = tk.Text(pose_text_frame, height=4, width=42, wrap="none")
        pose_y_scroll = tk.Scrollbar(pose_text_frame, orient="vertical", command=self.pose_text.yview)
        pose_x_scroll = tk.Scrollbar(pose_text_frame, orient="horizontal", command=self.pose_text.xview)
        self.pose_text.configure(yscrollcommand=pose_y_scroll.set, xscrollcommand=pose_x_scroll.set)
        self.pose_text.grid(row=0, column=0, sticky="nsew")
        pose_y_scroll.grid(row=0, column=1, sticky="ns")
        pose_x_scroll.grid(row=1, column=0, sticky="ew")
        self.pose_text.bind("<MouseWheel>", self._on_pose_text_mousewheel, add="+")
        self.pose_text.bind("<Button-4>", self._on_pose_text_mousewheel_linux, add="+")
        self.pose_text.bind("<Button-5>", self._on_pose_text_mousewheel_linux, add="+")
        self.pose_text.insert("1.0", json.dumps({"x": 0, "y": 0, "z": 100, "yaw": 0}, indent=2))
        tk.Button(pose, text="Set Pose", command=self.on_set_pose).grid(row=0, column=1, padx=6, pady=6, sticky="ns")

        preview = tk.LabelFrame(outer, text="Preview")
        preview.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(preview, text="Toggle RGB", command=self.toggle_preview_window).pack(side="left", padx=6, pady=6)
        tk.Button(preview, text="Refresh RGB", command=self.refresh_preview_window).pack(side="left", padx=6, pady=6)
        tk.Button(preview, text="Save Frame", command=self.on_save_frame).pack(side="left", padx=6, pady=6)
        tk.Button(preview, text="Temp Capture", command=self.on_temp_capture).pack(side="left", padx=6, pady=6)
        tk.Button(preview, text="Temp Capture Lidar", command=self.on_temp_capture_lidar).pack(side="left", padx=6, pady=6)
        tk.Label(preview, text="Lidar cm").pack(side="left", padx=(8, 2), pady=6)
        tk.Entry(preview, textvariable=self.lidar_depth_min_cm_var, width=6).pack(side="left", padx=(0, 2), pady=6)
        tk.Label(preview, text="-").pack(side="left", padx=(0, 2), pady=6)
        tk.Entry(preview, textvariable=self.lidar_depth_max_cm_var, width=6).pack(side="left", padx=(0, 6), pady=6)
        tk.Checkbutton(preview, text="Auto RGB", variable=self.auto_rgb_var).pack(side="left", padx=6, pady=6)
        tk.Checkbutton(preview, text="Enhance RGB", variable=self.enhance_rgb_var).pack(side="left", padx=6, pady=6)
        tk.Label(preview, text="RGB Source").pack(side="left", padx=(12, 2), pady=6)
        ttk.Combobox(
            preview,
            textvariable=self.rgb_source_order_var,
            values=("bgr", "rgb"),
            state="readonly",
            width=5,
        ).pack(side="left", padx=(0, 6), pady=6)
        tk.Button(preview, text="Refresh State", command=self.refresh_state_once).pack(side="left", padx=6, pady=6)

        stream = tk.LabelFrame(outer, text="Stream Capture")
        stream.grid(row=6, column=0, sticky="ew", padx=8, pady=4)
        stream.grid_columnconfigure(1, weight=1)
        stream.grid_columnconfigure(7, weight=1)
        tk.Label(stream, text="Task").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.stream_task_entry = tk.Entry(
            stream,
            textvariable=self.stream_task_title_var,
            readonlybackground="#f0f0f0",
        )
        self.stream_task_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=6)
        tk.Label(stream, text="Interval s").grid(row=0, column=4, sticky="w", padx=(12, 2), pady=6)
        tk.Entry(stream, textvariable=self.stream_interval_s_var, width=7).grid(row=0, column=5, sticky="w", padx=(0, 8), pady=6)
        tk.Button(stream, text="Start Timed Capture", command=self.on_start_stream_capture).grid(row=1, column=0, padx=6, pady=(0, 6))
        tk.Button(stream, text="Stop Timed Capture", command=self.on_stop_stream_capture).grid(row=1, column=1, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Open Player", command=self.open_stream_player_window).grid(row=1, column=2, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Play Latest", command=self.play_latest_stream_capture).grid(row=1, column=3, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Analyze Stream", command=self.open_stream_analysis_window).grid(row=1, column=4, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Obstacle Avoidance", command=self.open_obstacle_avoidance_window).grid(row=1, column=5, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Obstacle Avoidance 2", command=self.open_obstacle_avoidance_2_window).grid(row=1, column=6, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Obstacle Avoidance LLM", command=self.open_obstacle_avoidance_llm_window).grid(row=1, column=7, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Start Lidar Capture", command=self.on_start_lidar_stream_capture).grid(row=2, column=0, padx=6, pady=(0, 6))
        tk.Button(stream, text="Stop Lidar Capture", command=self.on_stop_lidar_stream_capture).grid(row=2, column=1, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Analyze Lidar", command=self.open_lidar_analysis_window).grid(row=2, column=2, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Analyze Lidar 2", command=self.open_lidar2_analysis_window).grid(row=2, column=3, sticky="w", padx=6, pady=(0, 6))
        tk.Button(stream, text="Export Open3D", command=self.export_lidar_analysis_open3d).grid(row=2, column=4, sticky="w", padx=6, pady=(0, 6))
        tk.Label(stream, text="Lidar mode").grid(row=2, column=5, sticky="e", padx=(12, 2), pady=(0, 6))
        ttk.Combobox(
            stream,
            textvariable=self.lidar_capture_processing_var,
            values=("smooth", "minimal", "full"),
            state="readonly",
            width=8,
        ).grid(row=2, column=6, sticky="w", padx=(0, 8), pady=(0, 6))
        tk.Button(stream, text="Obstacle Representation", command=self.open_obstacle_representation_window).grid(
            row=2, column=7, sticky="w", padx=6, pady=(0, 6)
        )
        tk.Button(stream, text="Obstacle Representation 2", command=self.open_obstacle_representation_2_monitor_window).grid(
            row=3, column=7, sticky="w", padx=6, pady=(0, 6)
        )
        tk.Label(stream, textvariable=self.stream_status_var, anchor="w").grid(row=4, column=0, columnspan=8, sticky="ew", padx=6, pady=(0, 3))
        tk.Label(stream, textvariable=self.lidar_stream_status_var, anchor="w").grid(row=5, column=0, columnspan=8, sticky="ew", padx=6, pady=(0, 3))
        tk.Label(stream, textvariable=self.lidar_stream_analysis_status_var, anchor="w").grid(row=6, column=0, columnspan=8, sticky="ew", padx=6, pady=(0, 3))
        tk.Label(stream, textvariable=self.stream_player_status_var, anchor="w").grid(row=7, column=0, columnspan=8, sticky="ew", padx=6, pady=(0, 6))

        map_frame = tk.LabelFrame(outer, text="Map")
        map_frame.grid(row=7, column=0, sticky="ew", padx=8, pady=4)
        map_frame.grid_columnconfigure(11, weight=1)
        tk.Button(map_frame, text="Open Map", command=self.toggle_map_window).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(map_frame, text="Refresh Map", command=lambda: self.refresh_map_once(force_reload=True)).grid(row=0, column=1, padx=6, pady=6)
        tk.Button(map_frame, text="Setting Map", command=self.open_setting_map_window).grid(row=0, column=2, padx=6, pady=6)
        tk.Checkbutton(map_frame, text="Show Houses", variable=self.show_houses_var, command=self.refresh_map_once).grid(row=0, column=3, padx=6, pady=6)
        tk.Checkbutton(map_frame, text="Show Trajectory", variable=self.show_trajectory_var, command=self.refresh_map_once).grid(row=0, column=4, padx=6, pady=6)
        tk.Checkbutton(map_frame, text="Show P Points", variable=self.show_calibration_points_var, command=self.refresh_map_once).grid(row=0, column=5, padx=6, pady=6)
        tk.Label(map_frame, textvariable=self.map_status_var, anchor="w").grid(row=0, column=6, columnspan=6, sticky="ew", padx=6, pady=6)
        tk.Label(map_frame, textvariable=self.map_pose_var, anchor="w").grid(row=1, column=0, columnspan=12, sticky="ew", padx=6, pady=(0, 4))
        tk.Button(map_frame, text="Start P Calibration", command=self.on_start_map_touch_calibration).grid(row=2, column=0, padx=6, pady=6)
        tk.Button(map_frame, text="Stop/Clear", command=self.on_stop_map_touch_calibration).grid(row=2, column=1, padx=6, pady=6)
        tk.Button(map_frame, text="Reset Marker", command=self.on_reset_map_touch_markers).grid(row=2, column=2, padx=6, pady=6)
        tk.Button(map_frame, text="Save Corrected Config", command=self.on_save_corrected_map_config).grid(row=2, column=3, padx=6, pady=6)
        tk.Label(map_frame, text="XY Tol cm").grid(row=2, column=4, padx=(12, 2), pady=6)
        tk.Entry(map_frame, textvariable=self.xy_tolerance_var, width=7).grid(row=2, column=5, padx=(0, 8), pady=6)
        tk.Label(map_frame, text="Z Tol cm").grid(row=2, column=6, padx=(6, 2), pady=6)
        tk.Entry(map_frame, textvariable=self.z_tolerance_var, width=7).grid(row=2, column=7, padx=(0, 8), pady=6)
        tk.Label(map_frame, text="Marker Class").grid(row=3, column=0, padx=(6, 2), pady=(0, 6))
        ttk.Combobox(
            map_frame,
            textvariable=self.marker_class_var,
            values=("BP_drone01_C", "target_C", "bp_character_C", "Cube_C"),
            state="readonly",
            width=14,
        ).grid(row=3, column=1, padx=(0, 8), pady=(0, 6), sticky="w")
        tk.Label(map_frame, text="Marker Scale").grid(row=3, column=2, padx=(6, 2), pady=(0, 6))
        tk.Entry(map_frame, textvariable=self.marker_scale_var, width=7).grid(row=3, column=3, padx=(0, 8), pady=(0, 6))
        tk.Label(map_frame, textvariable=self.map_calibration_var, anchor="w").grid(row=3, column=4, columnspan=7, sticky="ew", padx=6, pady=(0, 6))

        route = self._build_llm_route_section(outer, include_window_button=True)
        route.grid(row=8, column=0, sticky="ew", padx=8, pady=4)

        orbit = tk.LabelFrame(outer, text="Orbit Plan")
        orbit.grid(row=9, column=0, sticky="ew", padx=8, pady=(4, 8))
        for label, var, width in (
            ("Center X", self.orbit_center_x_var, 10),
            ("Center Y", self.orbit_center_y_var, 10),
            ("Radius", self.orbit_radius_var, 8),
            ("Altitude", self.orbit_altitude_var, 8),
            ("Steps", self.orbit_steps_var, 6),
            ("Start Angle", self.orbit_start_angle_var, 8),
        ):
            tk.Label(orbit, text=label).pack(side="left", padx=(6, 2), pady=6)
            tk.Entry(orbit, textvariable=var, width=width).pack(side="left", padx=(0, 6), pady=6)
        tk.Checkbutton(orbit, text="Clockwise", variable=self.orbit_clockwise_var).pack(side="left", padx=6)
        tk.Button(orbit, text="Run Orbit", command=self.on_run_orbit).pack(side="left", padx=6)
        tk.Button(orbit, text="Run Scripted Smoke", command=self.on_run_scripted).pack(side="left", padx=6)

    def _register_llm_route_widgets(self, combo: ttk.Combobox, preview_text: tk.Text) -> None:
        if self.house_target_combo is None:
            self.house_target_combo = combo
        if combo not in self.house_target_combos:
            self.house_target_combos.append(combo)
        if self.llm_route_preview_text is None:
            self.llm_route_preview_text = preview_text
        if preview_text not in self.llm_route_preview_texts:
            self.llm_route_preview_texts.append(preview_text)
        try:
            combo["values"] = list(self.house_choice_map.keys())
        except tk.TclError:
            pass

    def _unregister_llm_route_widgets(self, combo: Optional[ttk.Combobox], preview_text: Optional[tk.Text]) -> None:
        if combo is not None and combo in self.house_target_combos:
            self.house_target_combos.remove(combo)
        if preview_text is not None and preview_text in self.llm_route_preview_texts:
            self.llm_route_preview_texts.remove(preview_text)
        if self.house_target_combo is combo:
            self.house_target_combo = self.house_target_combos[0] if self.house_target_combos else None
        if self.llm_route_preview_text is preview_text:
            self.llm_route_preview_text = self.llm_route_preview_texts[0] if self.llm_route_preview_texts else None

    def _build_llm_route_section(self, parent: tk.Misc, *, include_window_button: bool) -> tk.LabelFrame:
        route = tk.LabelFrame(parent, text="LLM House Entrance Route")
        for col in (1, 3, 5):
            route.grid_columnconfigure(col, weight=1)
        tk.Label(route, text="Target House").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        combo = ttk.Combobox(
            route,
            textvariable=self.llm_route_target_var,
            values=list(self.house_choice_map.keys()),
            state="readonly",
            width=20,
        )
        combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="Task").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_task_text_var).grid(row=0, column=3, columnspan=3, sticky="ew", padx=6, pady=6)

        tk.Label(route, text="API").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        api_combo = ttk.Combobox(
            route,
            textvariable=self.llm_api_style_var,
            values=LLM_API_STYLE_OPTIONS,
            state="readonly",
            width=17,
        )
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
        tk.Label(route, text="Step cm").grid(row=2, column=4, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.route_step_cm_var, width=8).grid(row=2, column=5, sticky="w", padx=6, pady=6)

        scan = tk.Frame(route)
        scan.grid(row=3, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(scan, text="Standoff cm").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(scan, textvariable=self.llm_route_standoff_cm_var, width=8).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(scan, text="Scan spacing cm").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(scan, textvariable=self.llm_route_scan_spacing_cm_var, width=8).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(scan, text="Capture frames").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(scan, textvariable=self.llm_route_capture_count_var, width=6).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(scan, text="Path capture s").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(scan, textvariable=self.route_capture_interval_s_var, width=6).pack(side="left", padx=(0, 8), pady=4)
        if include_window_button:
            tk.Button(scan, text="Open LLM Route Window", command=self.open_llm_route_window).pack(side="left", padx=(18, 6), pady=4)
            tk.Button(scan, text="Open LLM Route Window 2", command=self.open_llm_route_window2).pack(side="left", padx=6, pady=4)
            tk.Button(scan, text="Open LLM Route Window 3", command=self.open_llm_route_window3).pack(side="left", padx=6, pady=4)
            tk.Button(scan, text="Open LLM Route Window 4", command=self.open_llm_route_window4).pack(side="left", padx=6, pady=4)

        actions = tk.Frame(route)
        actions.grid(row=4, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 4))
        tk.Button(actions, text="Analyze Task", command=self.on_llm_task_analyze).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Plan Route", command=self.on_llm_route_plan).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Fallback Route", command=self.on_fallback_route_plan).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Direct Capture Test", command=self.on_direct_scan_capture_test).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Route Capture Lidar", command=self.on_route_lidar_capture).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Validate Data", command=self.on_validate_house_search_data).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Follow Next", command=self.on_follow_route_next).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Auto Follow", command=self.on_follow_route_auto).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop Route", command=self.on_stop_route_follow).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear Route", command=self.on_clear_route_plan).pack(side="left", padx=6, pady=4)
        tk.Label(actions, text="Delay ms").pack(side="left", padx=(18, 2), pady=4)
        tk.Entry(actions, textvariable=self.route_delay_ms_var, width=7).pack(side="left", padx=(0, 6), pady=4)
        tk.Label(route, textvariable=self.llm_route_status_var, anchor="w").grid(row=5, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        preview_text = tk.Text(route, height=8, wrap="none", font=("Consolas", 9))
        preview_text.grid(row=6, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 6))
        preview_text.configure(state="disabled")
        self._register_llm_route_widgets(combo, preview_text)
        setattr(route, "_llm_route_combo", combo)
        setattr(route, "_llm_route_preview_text", preview_text)
        return route

    def open_llm_route_window(self) -> None:
        if self.llm_route_window is not None and self.llm_route_window.winfo_exists():
            self.llm_route_window.lift()
            self.llm_route_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route")
        window.geometry("1280x920")
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
        content.grid_columnconfigure(0, weight=1)
        content_window = window_canvas.create_window((0, 0), window=content, anchor="nw")
        self.llm_route_window_canvas = window_canvas
        self.llm_route_window_content = content
        self.llm_route_window_content_window = content_window
        content.bind("<Configure>", self._on_llm_route_window_content_configure)
        window_canvas.bind("<Configure>", self._on_llm_route_window_canvas_configure)
        window.bind("<MouseWheel>", self._on_llm_route_window_mousewheel, add="+")
        window.bind("<Button-4>", self._on_llm_route_window_mousewheel_linux, add="+")
        window.bind("<Button-5>", self._on_llm_route_window_mousewheel_linux, add="+")

        route = self._build_llm_route_section(content, include_window_button=False)
        route.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        map_frame = tk.LabelFrame(content, text="Map Scan Points")
        map_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(1, weight=1)
        map_toolbar = tk.Frame(map_frame)
        map_toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        tk.Label(map_toolbar, textvariable=self.llm_route_map_status_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(map_toolbar, text="Refresh Map", command=self.refresh_llm_route_map).pack(side="right", padx=6)
        self.load_map_resources(force=True)
        self.llm_route_map_widget = OverheadMapWidget(map_frame, world_bounds=self.map_world_bounds, canvas_w=1200, canvas_h=480)
        self.llm_route_map_widget.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route_window = window
        self._bind_llm_route_window_mousewheel_tree(content)

        def close_window() -> None:
            self._unregister_llm_route_widgets(
                getattr(route, "_llm_route_combo", None),
                getattr(route, "_llm_route_preview_text", None),
            )
            self.llm_route_window = None
            self.llm_route_window_canvas = None
            self.llm_route_window_content = None
            self.llm_route_window_content_window = None
            self.llm_route_map_widget = None
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.refresh_house_target_choices()
        self.refresh_route_preview()
        self.refresh_llm_route_map()

    def _build_llm_route2_section(self, parent: tk.Misc) -> tk.LabelFrame:
        route = tk.LabelFrame(parent, text="LLM House Entrance Route V2")
        for col in (1, 3, 5):
            route.grid_columnconfigure(col, weight=1)
        route.grid_rowconfigure(6, weight=1)
        tk.Label(route, text="Target House").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        combo = ttk.Combobox(
            route,
            textvariable=self.llm_route_target_var,
            values=list(self.house_choice_map.keys()),
            state="readonly",
            width=22,
        )
        combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        if combo not in self.house_target_combos:
            self.house_target_combos.append(combo)
        if self.house_target_combo is None:
            self.house_target_combo = combo
        tk.Label(route, text="Task").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_task_text_var).grid(row=0, column=3, columnspan=3, sticky="ew", padx=6, pady=6)

        tk.Label(route, text="API").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        api_combo = ttk.Combobox(
            route,
            textvariable=self.llm_api_style_var,
            values=LLM_API_STYLE_OPTIONS,
            state="readonly",
            width=17,
        )
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
        facade_controls = tk.Frame(route)
        facade_controls.grid(row=2, column=4, columnspan=2, sticky="ew", padx=6, pady=4)
        tk.Label(facade_controls, textvariable=self.llm_route2_facade_var, anchor="w").pack(side="left", padx=(0, 8))
        tk.Label(facade_controls, text="Select").pack(side="left", padx=(0, 2))
        ttk.Combobox(
            facade_controls,
            textvariable=self.llm_route2_selected_facade_var,
            values=LLM_ROUTE2_FACADE_OPTIONS,
            state="readonly",
            width=7,
        ).pack(side="left", padx=(0, 6))
        tk.Button(
            facade_controls,
            text="Plan Selected Facade",
            command=self.on_route2_plan_selected_facade,
        ).pack(side="left", padx=(0, 8))
        tk.Label(facade_controls, textvariable=self.llm_route2_facade_status_var, anchor="w").pack(side="left")

        config = tk.Frame(route)
        config.grid(row=3, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(config, text="Floor height m").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_floor_height_m_var, width=6).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Default floors").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_default_floors_var, width=5).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Low Z cm").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_low_z_cm_var, width=7).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Z step cm").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_z_step_cm_var, width=7).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Density mode").pack(side="left", padx=(6, 2), pady=4)
        ttk.Combobox(
            config,
            textvariable=self.llm_route2_density_mode_var,
            values=("auto", "high", "medium", "low"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=(0, 8), pady=4)

        actions = tk.Frame(route)
        actions.grid(row=4, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 4))
        tk.Button(actions, text="Plan 4 Facades", command=self.on_route2_plan_nearest_facade).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Move To Obs Point", command=self.on_route2_move_to_observation).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Capture Facade RGB", command=self.on_route2_capture_facade_rgb).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Analyze Facade VLM", command=self.on_route2_analyze_facade_vlm).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Plan Facade Scan", command=self.on_route2_plan_facade_scan).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Capture Facade Scan", command=self.on_route2_capture_facade_scan).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Validate Facade", command=self.on_route2_validate_facade).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=self.on_route2_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_route2_clear).pack(side="left", padx=6, pady=4)
        tk.Label(route, textvariable=self.llm_route2_status_var, anchor="w").grid(row=5, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        preview_frame = tk.Frame(route)
        preview_frame.grid(row=6, column=0, columnspan=6, sticky="nsew", padx=6, pady=(0, 6))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_text = tk.Text(preview_frame, height=6, width=96, wrap="none", font=("Consolas", 9))
        preview_y = tk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
        preview_x = tk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
        preview_text.configure(yscrollcommand=preview_y.set, xscrollcommand=preview_x.set)
        preview_text.grid(row=0, column=0, sticky="nsew")
        preview_y.grid(row=0, column=1, sticky="ns")
        preview_x.grid(row=1, column=0, sticky="ew")
        preview_text.configure(state="disabled")
        self.llm_route2_preview_text = preview_text
        setattr(route, "_llm_route2_combo", combo)
        return route

    def open_llm_route_window2(self) -> None:
        if self.llm_route2_window is not None and self.llm_route2_window.winfo_exists():
            self.llm_route2_window.lift()
            self.llm_route2_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route V2")
        window.geometry("760x560")
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
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(2, weight=1)
        content_window = window_canvas.create_window((0, 0), window=content, anchor="nw")
        self.llm_route2_window_canvas = window_canvas
        self.llm_route2_window_content = content
        self.llm_route2_window_content_window = content_window
        content.bind("<Configure>", self._on_llm_route2_window_content_configure)
        window_canvas.bind("<Configure>", self._on_llm_route2_window_canvas_configure)
        window.bind("<MouseWheel>", self._on_llm_route2_window_mousewheel, add="+")
        window.bind("<Button-4>", self._on_llm_route2_window_mousewheel_linux, add="+")
        window.bind("<Button-5>", self._on_llm_route2_window_mousewheel_linux, add="+")

        route = self._build_llm_route2_section(content)
        route.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        facade_panel = tk.Frame(content)
        facade_panel.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 4))
        facade_panel.grid_columnconfigure(0, weight=0)
        facade_panel.grid_columnconfigure(1, weight=1)

        rgb_frame = tk.LabelFrame(facade_panel, text="Facade RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        rgb_frame.grid_columnconfigure(0, weight=1)
        rgb_frame.grid_rowconfigure(1, weight=1)
        tk.Label(rgb_frame, textvariable=self.llm_route2_rgb_status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 2))
        self.llm_route2_rgb_label = tk.Canvas(
            rgb_frame,
            width=330,
            height=230,
            bg="#202020",
            highlightthickness=0,
        )
        self.llm_route2_rgb_label.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.llm_route2_rgb_label.bind("<Configure>", lambda _event: self.refresh_route2_rgb_display(), add="+")

        analysis_frame = tk.LabelFrame(facade_panel, text="Facade Analysis")
        analysis_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_rowconfigure(0, weight=1)
        analysis_text = tk.Text(analysis_frame, height=11, width=58, wrap="none", font=("Consolas", 9))
        analysis_y = tk.Scrollbar(analysis_frame, orient="vertical", command=analysis_text.yview)
        analysis_x = tk.Scrollbar(analysis_frame, orient="horizontal", command=analysis_text.xview)
        analysis_text.configure(yscrollcommand=analysis_y.set, xscrollcommand=analysis_x.set)
        analysis_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        analysis_y.grid(row=0, column=1, sticky="ns", pady=(6, 0))
        analysis_x.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
        analysis_text.configure(state="disabled")
        self.llm_route2_analysis_text = analysis_text

        map_frame = tk.LabelFrame(content, text="Map Facade V2 Points")
        map_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(1, weight=1)
        map_toolbar = tk.Frame(map_frame)
        map_toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        tk.Label(map_toolbar, textvariable=self.llm_route2_map_status_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(map_toolbar, textvariable=self.llm_route2_progress_text_var, anchor="e").pack(side="left", padx=(8, 2))
        ttk.Progressbar(
            map_toolbar,
            variable=self.llm_route2_progress_var,
            maximum=100.0,
            length=150,
            mode="determinate",
        ).pack(side="left", padx=(0, 8))
        tk.Button(map_toolbar, text="Refresh Map", command=self.refresh_llm_route2_map).pack(side="right", padx=6)
        tk.Checkbutton(
            map_toolbar,
            text="Auto Refresh",
            variable=self.llm_route2_auto_refresh_var,
            command=self.on_route2_auto_refresh_toggle,
        ).pack(side="right", padx=6)
        self.load_map_resources(force=True)
        self.llm_route2_map_widget = OverheadMapWidget(map_frame, world_bounds=self.map_world_bounds, canvas_w=610, canvas_h=260)
        self.llm_route2_map_widget.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route2_window = window
        self._bind_llm_route2_window_mousewheel_tree(content)

        def close_window() -> None:
            combo = getattr(route, "_llm_route2_combo", None)
            if combo is not None and combo in self.house_target_combos:
                self.house_target_combos.remove(combo)
            if self.house_target_combo is combo:
                self.house_target_combo = self.house_target_combos[0] if self.house_target_combos else None
            self.llm_route2_window = None
            self.llm_route2_window_canvas = None
            self.llm_route2_window_content = None
            self.llm_route2_window_content_window = None
            self.llm_route2_map_widget = None
            self.llm_route2_preview_text = None
            self.llm_route2_analysis_text = None
            self.llm_route2_rgb_label = None
            self.llm_route2_rgb_photo = None
            self.cancel_route2_auto_refresh()
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.refresh_house_target_choices()
        self.refresh_route2_preview()
        self.refresh_llm_route2_map()

    def _build_llm_route3_section(self, parent: tk.Misc) -> tk.LabelFrame:
        route = tk.LabelFrame(parent, text="LLM House Entrance Route V3 Autonomy")
        for col in (1, 3, 5):
            route.grid_columnconfigure(col, weight=1)
        route.grid_rowconfigure(8, weight=1)

        tk.Label(route, text="Target House").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        combo = ttk.Combobox(
            route,
            textvariable=self.llm_route_target_var,
            values=list(self.house_choice_map.keys()),
            state="readonly",
            width=22,
        )
        combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        if combo not in self.house_target_combos:
            self.house_target_combos.append(combo)
        if self.house_target_combo is None:
            self.house_target_combo = combo
        tk.Label(route, text="Task").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_task_text_var).grid(row=0, column=3, columnspan=2, sticky="ew", padx=6, pady=6)
        tk.Button(route, text="Analyze Task Plan", command=self.on_route3_analyze_task_plan).grid(row=0, column=5, sticky="ew", padx=6, pady=6)

        tk.Label(route, text="API").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        api_combo = ttk.Combobox(
            route,
            textvariable=self.llm_api_style_var,
            values=LLM_API_STYLE_OPTIONS,
            state="readonly",
            width=17,
        )
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
        tk.Label(route, textvariable=self.llm_route3_active_var, anchor="w").grid(row=2, column=4, columnspan=2, sticky="ew", padx=6, pady=6)

        config = tk.Frame(route)
        config.grid(row=3, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(config, text="Floor height m").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_floor_height_m_var, width=6).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Default floors").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_default_floors_var, width=5).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Low Z cm").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_low_z_cm_var, width=7).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Z step cm").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_z_step_cm_var, width=7).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Density mode").pack(side="left", padx=(6, 2), pady=4)
        ttk.Combobox(
            config,
            textvariable=self.llm_route2_density_mode_var,
            values=("auto", "high", "medium", "low"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=(0, 8), pady=4)

        nav = tk.Frame(route)
        nav.grid(row=4, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        for label, var, width in (
            ("Move tick ms", self.llm_route3_move_tick_ms_var, 6),
            ("Nav step cm", self.llm_route3_nav_step_cm_var, 6),
            ("Reach tol cm", self.llm_route3_reach_tol_cm_var, 6),
            ("Z tol cm", self.llm_route3_z_tol_cm_var, 6),
            ("Yaw tol deg", self.llm_route3_yaw_tol_deg_var, 6),
            ("Max stage s", self.llm_route3_max_stage_s_var, 6),
        ):
            tk.Label(nav, text=label).pack(side="left", padx=(6, 2), pady=4)
            tk.Entry(nav, textvariable=var, width=width).pack(side="left", padx=(0, 6), pady=4)

        actions = tk.Frame(route)
        actions.grid(row=5, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 4))
        tk.Button(actions, text="Start Full Search", command=self.on_route3_start_full_search).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Step Stage", command=self.on_route3_step_stage).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Pause/Resume", command=self.on_route3_toggle_pause).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=self.on_route3_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_route3_clear).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Validate Run", command=self.on_route3_validate_run).pack(side="left", padx=6, pady=4)

        status = tk.Frame(route)
        status.grid(row=6, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        status.grid_columnconfigure(1, weight=1)
        tk.Label(status, textvariable=self.llm_route3_stage_var, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 12))
        tk.Label(status, textvariable=self.llm_route3_target_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=(0, 12))
        tk.Label(status, textvariable=self.llm_route3_error_var, anchor="w").grid(row=0, column=2, sticky="w")
        tk.Label(status, textvariable=self.llm_route3_payload_var, anchor="w").grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(status, textvariable=self.llm_route3_task_status_var, anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(status, textvariable=self.llm_route3_target_sequence_var, anchor="w").grid(row=3, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        tk.Label(route, textvariable=self.llm_route3_status_var, anchor="w").grid(row=7, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        preview_frame = tk.Frame(route)
        preview_frame.grid(row=8, column=0, columnspan=6, sticky="nsew", padx=6, pady=(0, 6))
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
        self.llm_route3_preview_text = preview_text
        setattr(route, "_llm_route3_combo", combo)
        return route

    def open_llm_route_window3(self) -> None:
        if self.llm_route3_window is not None and self.llm_route3_window.winfo_exists():
            self.llm_route3_window.lift()
            self.llm_route3_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route V3 Autonomy")
        window.geometry("900x680")
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
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(2, weight=1)
        content_window = window_canvas.create_window((0, 0), window=content, anchor="nw")
        self.llm_route3_window_canvas = window_canvas
        self.llm_route3_window_content = content
        self.llm_route3_window_content_window = content_window
        content.bind("<Configure>", self._on_llm_route3_window_content_configure)
        window_canvas.bind("<Configure>", self._on_llm_route3_window_canvas_configure)
        window.bind("<MouseWheel>", self._on_llm_route3_window_mousewheel, add="+")
        window.bind("<Button-4>", self._on_llm_route3_window_mousewheel_linux, add="+")
        window.bind("<Button-5>", self._on_llm_route3_window_mousewheel_linux, add="+")

        route = self._build_llm_route3_section(content)
        route.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        facade_panel = tk.Frame(content)
        facade_panel.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 4))
        facade_panel.grid_columnconfigure(0, weight=0)
        facade_panel.grid_columnconfigure(1, weight=1)

        rgb_frame = tk.LabelFrame(facade_panel, text="Facade RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        tk.Label(rgb_frame, textvariable=self.llm_route2_rgb_status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 2))
        self.llm_route3_rgb_label = tk.Canvas(rgb_frame, width=330, height=230, bg="#202020", highlightthickness=0)
        self.llm_route3_rgb_label.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.llm_route3_rgb_label.bind("<Configure>", lambda _event: self.refresh_route3_rgb_display(), add="+")

        analysis_frame = tk.LabelFrame(facade_panel, text="Facade Analysis")
        analysis_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_rowconfigure(0, weight=1)
        analysis_text = tk.Text(analysis_frame, height=11, width=58, wrap="none", font=("Consolas", 9))
        analysis_y = tk.Scrollbar(analysis_frame, orient="vertical", command=analysis_text.yview)
        analysis_x = tk.Scrollbar(analysis_frame, orient="horizontal", command=analysis_text.xview)
        analysis_text.configure(yscrollcommand=analysis_y.set, xscrollcommand=analysis_x.set)
        analysis_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        analysis_y.grid(row=0, column=1, sticky="ns", pady=(6, 0))
        analysis_x.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
        analysis_text.configure(state="disabled")
        self.llm_route3_analysis_text = analysis_text

        map_frame = tk.LabelFrame(content, text="Map Facade V3 Autonomy")
        map_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(1, weight=1)
        map_toolbar = tk.Frame(map_frame)
        map_toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        tk.Label(map_toolbar, textvariable=self.llm_route3_map_status_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(map_toolbar, textvariable=self.llm_route3_current_status_var, anchor="w").pack(side="left", padx=(8, 2))
        tk.Label(map_toolbar, textvariable=self.llm_route3_next_status_var, anchor="w").pack(side="left", padx=(8, 2))
        tk.Label(map_toolbar, textvariable=self.llm_route3_progress_text_var, anchor="e").pack(side="left", padx=(8, 2))
        ttk.Progressbar(map_toolbar, variable=self.llm_route3_progress_var, maximum=100.0, length=150, mode="determinate").pack(side="left", padx=(0, 8))
        tk.Checkbutton(
            map_toolbar,
            text="Auto Refresh",
            variable=self.llm_route3_auto_refresh_var,
            command=self.on_route3_auto_refresh_toggle,
        ).pack(side="left", padx=(0, 8))
        tk.Button(map_toolbar, text="Refresh Map", command=self.refresh_llm_route3_map).pack(side="right", padx=6)
        self.load_map_resources(force=True)
        self.llm_route3_map_widget = OverheadMapWidget(map_frame, world_bounds=self.map_world_bounds, canvas_w=760, canvas_h=320)
        self.llm_route3_map_widget.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route3_window = window
        self._bind_llm_route3_window_mousewheel_tree(content)

        def close_window() -> None:
            combo = getattr(route, "_llm_route3_combo", None)
            if combo is not None and combo in self.house_target_combos:
                self.house_target_combos.remove(combo)
            if self.house_target_combo is combo:
                self.house_target_combo = self.house_target_combos[0] if self.house_target_combos else None
            self.llm_route3_window = None
            self.llm_route3_window_canvas = None
            self.llm_route3_window_content = None
            self.llm_route3_window_content_window = None
            self.llm_route3_map_widget = None
            self.llm_route3_preview_text = None
            self.llm_route3_analysis_text = None
            self.llm_route3_rgb_label = None
            self.llm_route3_rgb_photo = None
            self.cancel_route3_auto_refresh()
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.refresh_house_target_choices()
        self.refresh_route3_preview()
        self.refresh_route3_support_views()

    def _bind_hotkeys(self) -> None:
        self.root.bind_all("<KeyPress>", self._on_keyboard_press, add="+")
        self.root.bind_all("<KeyRelease>", self._on_keyboard_release, add="+")

    def _event_widget_accepts_text(self, event: tk.Event) -> bool:
        widget = getattr(event, "widget", None)
        if isinstance(widget, ttk.Combobox):
            try:
                return "readonly" not in widget.state()
            except Exception:
                return False
        if isinstance(widget, (tk.Entry, tk.Text, ttk.Entry)):
            return True
        try:
            return str(widget.winfo_class()).lower() in {"entry", "text", "tentry", "spinbox", "tspinbox"}
        except Exception:
            return False

    def _movement_symbol_from_event(self, event: tk.Event) -> Optional[str]:
        char = str(getattr(event, "char", "") or "").lower()
        keysym = str(getattr(event, "keysym", "") or "").lower()
        if char in MOVE_COMMANDS:
            return char
        if keysym in MOVE_COMMANDS:
            return keysym
        alias = KEYBOARD_SYMBOL_ALIASES.get(keysym) or KEYBOARD_SYMBOL_ALIASES.get(char)
        if alias in MOVE_COMMANDS:
            return alias
        return None

    def _stream_capture_action_from_event(self, event: tk.Event) -> Optional[str]:
        char = str(getattr(event, "char", "") or "")
        keysym = str(getattr(event, "keysym", "") or "").lower()
        if char in {",", "，"} or keysym == "comma":
            return "start"
        if char in {".", "。"} or keysym == "period":
            return "stop"
        return None

    def _camera_view_toggle_from_event(self, event: tk.Event) -> bool:
        char = str(getattr(event, "char", "") or "").lower()
        keysym = str(getattr(event, "keysym", "") or "").lower()
        return char == "z" or keysym == "z"

    def should_capture_movement_key(self, event: tk.Event, symbol: str) -> bool:
        if not self._event_widget_accepts_text(event):
            return True
        if not self.keyboard_enabled_var.get():
            return False
        if symbol == "x":
            return self.session is not None and self.session.started
        return self.session is not None and self.session.started and self.movement_enabled_state

    def focus_keyboard_control(self) -> None:
        try:
            self.root.focus_force()
            if self.content_frame is not None:
                self.content_frame.focus_set()
        except Exception:
            pass
        self.update_keyboard_status("ready")

    def _on_keyboard_press(self, event: tk.Event):
        stream_action = self._stream_capture_action_from_event(event)
        if stream_action is not None and not self._event_widget_accepts_text(event):
            if stream_action == "start":
                self.on_start_stream_capture()
            else:
                self.on_stop_stream_capture()
            return "break"

        if (
            self.keyboard_enabled_var.get()
            and self._camera_view_toggle_from_event(event)
            and not self._event_widget_accepts_text(event)
        ):
            if not self.camera_toggle_key_down:
                self.camera_toggle_key_down = True
                self.on_toggle_camera_view()
            return "break"

        symbol = self._movement_symbol_from_event(event)
        if symbol is None:
            return None
        if not self.should_capture_movement_key(event, symbol):
            return None
        if not self.keyboard_enabled_var.get():
            self.send_move_symbol(symbol)
            return "break"
        if symbol == "x":
            self.stop_keyboard_control(send_hold=True, force_hold=True)
            return "break"
        if self.session is None or not self.session.started:
            self.keyboard_status_var.set("Keyboard: start session first")
            return "break"
        if not self.movement_enabled_state:
            self.keyboard_status_var.set("Keyboard: enable Basic Movement first")
            return "break"
        already_pressed = symbol in self.keyboard_pressed_symbols
        self.keyboard_pressed_symbols.add(symbol)
        self.update_keyboard_status()
        if not already_pressed:
            self.start_keyboard_loop()
        return "break"

    def _on_keyboard_release(self, event: tk.Event):
        if (
            self.keyboard_enabled_var.get()
            and self._camera_view_toggle_from_event(event)
            and not self._event_widget_accepts_text(event)
        ):
            self.camera_toggle_key_down = False
            return "break"

        symbol = self._movement_symbol_from_event(event)
        if symbol is None or symbol == "x":
            return None
        if not self.should_capture_movement_key(event, symbol) and symbol not in self.keyboard_pressed_symbols:
            return None
        if symbol in self.keyboard_pressed_symbols:
            self.keyboard_pressed_symbols.discard(symbol)
            self.update_keyboard_status()
        if not self.keyboard_pressed_symbols:
            self.cancel_keyboard_loop()
        return "break"

    def on_browse_env_root(self) -> None:
        path = filedialog.askdirectory(title="Select UnrealEnv directory")
        if path:
            self.env_root_var.set(path)

    def parse_float_list(self, text: str) -> List[float]:
        return [float(part) for part in text.replace(",", " ").split()]

    def build_flight_args(self) -> argparse.Namespace:
        env_root = self.env_root_var.get().strip() or None
        env_bin = self.env_bin_var.get().strip() or None
        output_dir = self.output_dir_var.get().strip() or "results/drone_flight_controller"
        initial_pos = self.parse_float_list(self.initial_pose_var.get().strip())
        lidar_min_cm, lidar_max_cm = self.parse_lidar_depth_range()
        return flight.default_session_args(
            env_platform=self.env_platform_var.get().strip() or "auto",
            env_root=env_root,
            env_bin=env_bin,
            output_dir=output_dir,
            width=int(float(self.width_var.get().strip())),
            height=int(float(self.height_var.get().strip())),
            launch_sleep=int(float(self.launch_sleep_var.get().strip())),
            time_dilation=int(float(self.time_dilation_var.get().strip())),
            step_delay=max(0.0, float(self.step_delay_var.get().strip())),
            control_dt=max(0.03, min(1.0, self.keyboard_interval_ms() / 1000.0)),
            save_every=max(0, int(float(self.save_every_var.get().strip()))),
            movement_mode=self.movement_mode_var.get().strip() or flight.DEFAULT_MOVEMENT_MODE,
            initial_pos=initial_pos,
            mode="keyboard",
            auto_action="none",
            max_steps=0,
            enhance_rgb=bool(self.enhance_rgb_var.get()),
            rgb_enhance_gamma=float(self.args.rgb_enhance_gamma),
            rgb_enhance_gain=float(self.args.rgb_enhance_gain),
            rgb_source_order=self.rgb_source_order_var.get().strip() or flight.DEFAULT_RGB_SOURCE_ORDER,
            first_person_camera_config=str(getattr(self.args, "first_person_camera_config", flight.DEFAULT_FIRST_PERSON_CAMERA_CONFIG)),
            native_viewport_camera_config=str(getattr(self.args, "native_viewport_camera_config", flight.DEFAULT_NATIVE_VIEWPORT_CAMERA_CONFIG)),
            temp_capture_dir=str(getattr(self.args, "temp_capture_dir", flight.DEFAULT_TEMP_CAPTURE_DIR)),
            temp_capture_lidar_dir=str(getattr(self.args, "temp_capture_lidar_dir", flight.DEFAULT_TEMP_CAPTURE_LIDAR_DIR)),
            stream_capture_dir=str(getattr(self.args, "stream_capture_dir", flight.DEFAULT_STREAM_CAPTURE_DIR)),
            stream_capture_lidar_dir=str(getattr(self.args, "stream_capture_lidar_dir", flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR)),
            stream_interval_s=self.parse_stream_interval_s(),
            depth_min_cm=float(getattr(self.args, "depth_min_cm", flight.DEFAULT_DEPTH_MIN_CM)),
            depth_max_cm=float(getattr(self.args, "depth_max_cm", flight.DEFAULT_DEPTH_MAX_CM)),
            lidar_depth_min_cm=lidar_min_cm,
            lidar_depth_max_cm=lidar_max_cm,
            lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
            lidar_capture_processing=flight.normalize_lidar_capture_processing(
                self.lidar_capture_processing_var.get()
            ),
            force_kill_unreal_on_stop=bool(self.args.force_kill_unreal_on_stop),
            log_level=self.args.log_level,
        )

    def safe(self, label: str, fn):
        try:
            return fn()
        except Exception as exc:
            LOGGER.warning("%s failed: %s", label, exc)
            self.root.after(0, lambda e=exc, l=label: self.status_var.set(f"{l} failed: {e}"))
            return None

    def call_async(self, desc: str, fn) -> None:
        if self.manual_request_inflight:
            self.status_var.set(f"{desc} skipped while another request is running.")
            return

        def worker() -> None:
            self.manual_request_inflight = True
            self.root.after(0, lambda: self.status_var.set(f"{desc}..."))
            try:
                result = self.safe(desc, fn)
                if isinstance(result, dict):
                    self.root.after(0, lambda r=result: self.apply_state(r))
            finally:
                self.manual_request_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def active_session(self) -> Optional[flight.DroneFlightSession]:
        if self.session is None or not self.session.started:
            self.root.after(0, lambda: self.status_var.set("Start run_drone_flight first."))
            return None
        return self.session

    def apply_state(self, state: Dict[str, Any]) -> None:
        self.latest_state = state if isinstance(state, dict) else {}
        message = str(self.latest_state.get("message", "") or self.latest_state.get("status", ""))
        if message:
            self.status_var.set(message)
        pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        if pose:
            self.pose_var.set(
                "Pose "
                f"x={float(pose.get('x', 0.0)):.1f} "
                f"y={float(pose.get('y', 0.0)):.1f} "
                f"z={float(pose.get('z', 0.0)):.1f} "
                f"yaw={float(pose.get('task_yaw', pose.get('yaw', 0.0))):.1f} "
                f"action={self.latest_state.get('last_action', 'idle')}"
            )
        else:
            self.pose_var.set("Pose: not available")
        self.movement_enabled_state = bool(self.latest_state.get("movement_enabled", False))
        self.movement_mode_state = str(self.latest_state.get("movement_mode", self.movement_mode_state) or self.movement_mode_state)
        if self.movement_mode_var.get() != self.movement_mode_state:
            self.movement_mode_var.set(self.movement_mode_state)
        commanded = self.latest_state.get("commanded_pose", {}) if isinstance(self.latest_state.get("commanded_pose"), dict) else {}
        pose_error = self.latest_state.get("pose_error", {}) if isinstance(self.latest_state.get("pose_error"), dict) else {}
        commanded_yaw = commanded.get("task_yaw", commanded.get("yaw", "n/a"))
        actual_yaw = pose.get("task_yaw", pose.get("yaw", "n/a")) if pose else "n/a"
        yaw_error = pose_error.get("yaw_deg", "n/a")
        camera_view = str(
            self.latest_state.get("camera_view_mode", flight.DEFAULT_CAMERA_VIEW_MODE)
            or flight.DEFAULT_CAMERA_VIEW_MODE
        )
        self.control_var.set(
            f"Movement enabled={1 if self.movement_enabled_state else 0} "
            f"mode={self.movement_mode_state} "
            f"view={camera_view} "
            f"commanded_yaw={self._fmt_float(commanded_yaw)} "
            f"actual_yaw={self._fmt_float(actual_yaw)} "
            f"yaw_error={self._fmt_float(yaw_error)}"
        )
        run_dir = str(self.latest_state.get("run_dir", "") or "")
        step_count = int(self.latest_state.get("step_count", 0) or 0)
        drone_name = str(self.latest_state.get("drone_name", "") or "n/a")
        started = int(bool(self.latest_state.get("started", False)))
        self.session_var.set(f"Session started={started} drone={drone_name} steps={step_count} dir={run_dir or 'none'}")
        if self.movement_toggle_button is not None:
            self.movement_toggle_button.configure(
                text="Disable Basic Movement" if self.movement_enabled_state else "Enable Basic Movement"
            )
        touch_state = self.latest_state.get("map_touch_calibration")
        if isinstance(touch_state, dict):
            self.apply_map_touch_state(touch_state, refresh=False)
        self.refresh_map_once()

    def _fmt_float(self, value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return str(value)

    def _as_float_or_none(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            number = float(value)
            if not math.isfinite(number):
                return None
            return number
        except Exception:
            return None

    def _normalize_angle_deg(self, angle_deg: float) -> float:
        return flight.normalize_angle_deg(float(angle_deg))

    def resolve_project_path(self, value: str, *, base_dir: Optional[Path] = None) -> Path:
        raw = str(value or "").strip()
        if not raw:
            return Path()
        path = Path(raw)
        if path.is_absolute():
            return path
        base = base_dir if base_dir is not None else PROJECT_ROOT
        return (base / path).resolve()

    def on_close(self) -> None:
        self.stop_keyboard_control(send_hold=False)
        self.llm_route3_stop_event.set()
        self.llm_route3_pause_event.clear()
        route4_stop_event = getattr(self, "llm_route4_stop_event", None)
        if route4_stop_event is not None:
            route4_stop_event.set()
        route4_pause_event = getattr(self, "llm_route4_pause_event", None)
        if route4_pause_event is not None:
            route4_pause_event.clear()
        if self.llm_route_window is not None:
            try:
                self.llm_route_window.destroy()
            except Exception:
                pass
            self.llm_route_window = None
            self.llm_route_window_canvas = None
            self.llm_route_window_content = None
            self.llm_route_window_content_window = None
            self.llm_route_map_widget = None
        if self.llm_route2_window is not None:
            try:
                self.llm_route2_window.destroy()
            except Exception:
                pass
            self.llm_route2_window = None
            self.llm_route2_window_canvas = None
            self.llm_route2_window_content = None
            self.llm_route2_window_content_window = None
            self.llm_route2_map_widget = None
            self.llm_route2_preview_text = None
            self.llm_route2_analysis_text = None
            self.llm_route2_rgb_label = None
            self.llm_route2_rgb_photo = None
            self.cancel_route2_auto_refresh()
        if self.llm_route3_window is not None:
            try:
                self.llm_route3_window.destroy()
            except Exception:
                pass
            self.llm_route3_window = None
            self.llm_route3_window_canvas = None
            self.llm_route3_window_content = None
            self.llm_route3_window_content_window = None
            self.llm_route3_map_widget = None
            self.llm_route3_preview_text = None
            self.llm_route3_analysis_text = None
            self.llm_route3_rgb_label = None
            self.llm_route3_rgb_photo = None
        if getattr(self, "llm_route4_window", None) is not None:
            try:
                self.llm_route4_window.destroy()
            except Exception:
                pass
            self.llm_route4_window = None
            self.llm_route4_window_canvas = None
            self.llm_route4_window_content = None
            self.llm_route4_window_content_window = None
            self.llm_route4_map_widget = None
            self.llm_route4_preview_text = None
            self.llm_route4_analysis_text = None
            self.llm_route4_rgb_label = None
            self.llm_route4_rgb_photo = None
        self.stop_stream_player()
        self.stop_stream_analysis()
        self.close_lidar_analysis_window()
        self.stream_capture_stop_event.set()
        self.lidar_stream_capture_stop_event.set()
        self.obstacle_avoidance_stop_event.set()
        or2_monitor_stop_event = getattr(self, "or2_monitor_stop_event", None)
        if or2_monitor_stop_event is not None:
            or2_monitor_stop_event.set()
        if self.obstacle_avoidance_window is not None:
            try:
                self.obstacle_avoidance_window.destroy()
            except Exception:
                pass
            self.obstacle_avoidance_window = None
            self.obstacle_avoidance_report_text = None
        self.sequence_stop_event.set()
        self.route_stop_event.set()
        session = self.session
        if session is not None and session.started:
            try:
                session.close(force_kill_unreal=True)
            except Exception as exc:
                LOGGER.warning("Failed to close session: %s", exc)
        self.close_map_window()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

