from __future__ import annotations

from .common import *
from .flight_control import FlightControlMixin
from .map_control import MapControlMixin
from .route_control import RouteControlMixin


class RunDroneFlightPanel(FlightControlMixin, MapControlMixin, RouteControlMixin):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session: Optional[flight.DroneFlightSession] = None
        self.latest_state: Dict[str, Any] = {}
        self.manual_request_inflight = False
        self.move_request_inflight = False
        self.keyboard_request_inflight = False
        self.state_refresh_inflight = False
        self.preview_refresh_inflight = False
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
        self.llm_route_target_var = tk.StringVar(value="")
        self.llm_task_text_var = tk.StringVar(value="Search the selected house entrance.")
        self.llm_api_style_var = tk.StringVar(value=normalize_llm_api_style(getattr(args, "llm_api_style", default_llm_api_style())))
        self.llm_base_url_var = tk.StringVar(value=str(getattr(args, "llm_base_url", "") or ""))
        self.llm_api_key_var = tk.StringVar(value=str(getattr(args, "llm_api_key", "") or ""))
        self.llm_model_var = tk.StringVar(value=str(getattr(args, "llm_model", "") or ""))
        self.llm_timeout_s_var = tk.StringVar(value=str(getattr(args, "llm_route_timeout_s", 60.0)))
        self.route_step_cm_var = tk.StringVar(value=str(getattr(args, "route_step_cm", 120.0)))
        self.route_delay_ms_var = tk.StringVar(value=str(getattr(args, "route_delay_ms", 100.0)))

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
        self.initial_pose_var = tk.StringVar(value=" ".join(str(value) for value in args.initial_pos))

        self.sequence_var = tk.StringVar(value="")
        self.sequence_delay_var = tk.StringVar(value="150")
        self.auto_rgb_var = tk.BooleanVar(value=False)
        self.enhance_rgb_var = tk.BooleanVar(value=bool(args.enhance_rgb))
        self.rgb_source_order_var = tk.StringVar(
            value=str(getattr(args, "rgb_source_order", flight.DEFAULT_RGB_SOURCE_ORDER) or flight.DEFAULT_RGB_SOURCE_ORDER)
        )
        self.show_houses_var = tk.BooleanVar(value=True)
        self.show_trajectory_var = tk.BooleanVar(value=True)

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
        self.keyboard_loop_after_id: Optional[str] = None
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.map_window: Optional[tk.Toplevel] = None
        self.map_widget: Optional[OverheadMapWidget] = None
        self.map_refresh_inflight = False
        self.map_config: Dict[str, Any] = {}
        self.map_config_path: Optional[Path] = None
        self.map_image_path: Optional[Path] = None
        self.map_image: Optional[np.ndarray] = None
        self.map_calibration: Dict[str, Any] = {}
        self.map_world_bounds: Tuple[float, float, float, float] = DEFAULT_MAP_BOUNDS
        self.map_touch_state: Dict[str, Any] = {}
        self.map_touch_poll_inflight = False
        self.map_touch_auto_saved = False
        self.house_target_combo: Optional[ttk.Combobox] = None
        self.house_choice_map: Dict[str, str] = {}
        self.house_display_by_id: Dict[str, str] = {}
        self.llm_task_plan: Dict[str, Any] = {}
        self.llm_route_plan: Dict[str, Any] = {}
        self.llm_route_preview_text: Optional[tk.Text] = None
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
        pose.grid_columnconfigure(0, weight=1)
        self.pose_text = tk.Text(pose, height=3, wrap="word")
        self.pose_text.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.pose_text.insert("1.0", json.dumps({"x": 0, "y": 0, "z": 100, "yaw": 0}, indent=2))
        tk.Button(pose, text="Set Pose", command=self.on_set_pose).grid(row=0, column=1, padx=6, pady=6, sticky="ns")

        preview = tk.LabelFrame(outer, text="Preview")
        preview.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(preview, text="Toggle RGB", command=self.toggle_preview_window).pack(side="left", padx=6, pady=6)
        tk.Button(preview, text="Refresh RGB", command=self.refresh_preview_window).pack(side="left", padx=6, pady=6)
        tk.Button(preview, text="Save Frame", command=self.on_save_frame).pack(side="left", padx=6, pady=6)
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

        map_frame = tk.LabelFrame(outer, text="Map")
        map_frame.grid(row=6, column=0, sticky="ew", padx=8, pady=4)
        map_frame.grid_columnconfigure(10, weight=1)
        tk.Button(map_frame, text="Open Map", command=self.toggle_map_window).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(map_frame, text="Refresh Map", command=lambda: self.refresh_map_once(force_reload=True)).grid(row=0, column=1, padx=6, pady=6)
        tk.Checkbutton(map_frame, text="Show Houses", variable=self.show_houses_var, command=self.refresh_map_once).grid(row=0, column=2, padx=6, pady=6)
        tk.Checkbutton(map_frame, text="Show Trajectory", variable=self.show_trajectory_var, command=self.refresh_map_once).grid(row=0, column=3, padx=6, pady=6)
        tk.Label(map_frame, textvariable=self.map_status_var, anchor="w").grid(row=0, column=4, columnspan=7, sticky="ew", padx=6, pady=6)
        tk.Label(map_frame, textvariable=self.map_pose_var, anchor="w").grid(row=1, column=0, columnspan=11, sticky="ew", padx=6, pady=(0, 4))
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

        route = tk.LabelFrame(outer, text="LLM House Entrance Route")
        route.grid(row=7, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3, 5):
            route.grid_columnconfigure(col, weight=1)
        tk.Label(route, text="Target House").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.house_target_combo = ttk.Combobox(
            route,
            textvariable=self.llm_route_target_var,
            values=(),
            state="readonly",
            width=20,
        )
        self.house_target_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
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

        actions = tk.Frame(route)
        actions.grid(row=3, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 4))
        tk.Button(actions, text="Analyze Task", command=self.on_llm_task_analyze).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Plan Route", command=self.on_llm_route_plan).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Fallback Route", command=self.on_fallback_route_plan).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Follow Next", command=self.on_follow_route_next).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Auto Follow", command=self.on_follow_route_auto).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop Route", command=self.on_stop_route_follow).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear Route", command=self.on_clear_route_plan).pack(side="left", padx=6, pady=4)
        tk.Label(actions, text="Delay ms").pack(side="left", padx=(18, 2), pady=4)
        tk.Entry(actions, textvariable=self.route_delay_ms_var, width=7).pack(side="left", padx=(0, 6), pady=4)
        tk.Label(route, textvariable=self.llm_route_status_var, anchor="w").grid(row=4, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        self.llm_route_preview_text = tk.Text(route, height=5, wrap="none", font=("Consolas", 9))
        self.llm_route_preview_text.grid(row=5, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 6))
        self.llm_route_preview_text.configure(state="disabled")

        orbit = tk.LabelFrame(outer, text="Orbit Plan")
        orbit.grid(row=8, column=0, sticky="ew", padx=8, pady=(4, 8))
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
        self.control_var.set(
            f"Movement enabled={1 if self.movement_enabled_state else 0} "
            f"mode={self.movement_mode_state} "
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

