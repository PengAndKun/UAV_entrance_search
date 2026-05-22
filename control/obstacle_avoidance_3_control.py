from __future__ import annotations

from copy import deepcopy
from tkinter import messagebox

from .common import *

from obstacle_avoidance.collect import append_jsonl, write_json
from obstacle_avoidance.collect_route_episodes import (
    DEFAULT_ROUTE_SIDE_CORRECTION_CM,
    DEFAULT_ROUTE_STEP_CM,
    DEFAULT_ROUTE_VERTICAL_STEP_CM,
    action_payload,
    annotate_collision_state,
    build_route_event,
    distance_3d_cm,
    pose_dict,
    pose_from_state,
    route_completion_state,
    route_progress_and_deviation,
    should_stop_for_repeated_hold,
    straight_line_payload,
    summarize_episode,
    write_partial_episode_summary,
)
from obstacle_avoidance_3.control_utils import jsonable, serializable_or2_prediction
from obstacle_avoidance_3.or2_direction_rule import (
    DEFAULT_METHOD_ID,
    RED_RECOVERY_BACKOFF_CM,
    RED_RECOVERY_CLEAR_TICKS,
    RED_RECOVERY_MIN_TICKS,
    RED_RECOVERY_SIDE_STEP_CM,
    RED_RECOVERY_VERTICAL_CM,
    SIDE_BLOCKED_FRACTION,
    choose_red_recovery_direction,
    deep_red_stop_active,
    red_recovery_clear_enough,
    select_or2_direction,
)
from obstacle_avoidance_3.plans import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_PLAN_FILENAME,
    DEFAULT_PROJECT_ID,
    coerce_pose,
    load_plans,
    make_default_plans,
    sanitize_id,
    save_plans,
    sync_oa2_plan_into_oa3,
)
from obstacle_representation_2.demo import ObstacleRepresentation2Predictor, render_affordance_overlay


class ObstacleAvoidance3ControlMixin:
    def ensure_obstacle_avoidance_3_state(self) -> None:
        if not hasattr(self, "obstacle_avoidance_3_data_dir_var"):
            self.obstacle_avoidance_3_data_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "obstacle_avoidance_3_data"))
        if not hasattr(self, "obstacle_avoidance_3_plan_json_var"):
            self.obstacle_avoidance_3_plan_json_var = tk.StringVar(
                value=str(PROJECT_ROOT / "obstacle_avoidance_3_data" / "plans" / DEFAULT_PLAN_FILENAME)
            )
        if not hasattr(self, "obstacle_avoidance_3_model_var"):
            self.obstacle_avoidance_3_model_var = tk.StringVar(
                value=str(PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt")
            )
        if not hasattr(self, "obstacle_avoidance_3_project_var"):
            self.obstacle_avoidance_3_project_var = tk.StringVar(value=DEFAULT_PROJECT_ID)
        if not hasattr(self, "obstacle_avoidance_3_project_name_var"):
            self.obstacle_avoidance_3_project_name_var = tk.StringVar(value="OA3 default OR2 risk-region collection")
        if not hasattr(self, "obstacle_avoidance_3_environment_var"):
            self.obstacle_avoidance_3_environment_var = tk.StringVar(value=DEFAULT_ENVIRONMENT_ID)
        if not hasattr(self, "obstacle_avoidance_3_method_var"):
            self.obstacle_avoidance_3_method_var = tk.StringVar(value=DEFAULT_METHOD_ID)
        if not hasattr(self, "obstacle_avoidance_3_status_var"):
            self.obstacle_avoidance_3_status_var = tk.StringVar(value="Obstacle Avoidance 3: idle")
        if not hasattr(self, "obstacle_avoidance_3_episode_id_var"):
            self.obstacle_avoidance_3_episode_id_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_3_enabled_var"):
            self.obstacle_avoidance_3_enabled_var = tk.BooleanVar(value=True)
        if not hasattr(self, "obstacle_avoidance_3_start_pose_var"):
            self.obstacle_avoidance_3_start_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_3_goal_pose_var"):
            self.obstacle_avoidance_3_goal_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_3_scenario_var"):
            self.obstacle_avoidance_3_scenario_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_3_obstacle_hint_var"):
            self.obstacle_avoidance_3_obstacle_hint_var = tk.StringVar(value="unknown")
        if not hasattr(self, "obstacle_avoidance_3_note_var"):
            self.obstacle_avoidance_3_note_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_3_goal_distance_var"):
            self.obstacle_avoidance_3_goal_distance_var = tk.StringVar(value="Goal distance: --")
        if not hasattr(self, "obstacle_avoidance_3_goal_reached_var"):
            self.obstacle_avoidance_3_goal_reached_var = tk.StringVar(value="Goal reached: --")
        if not hasattr(self, "obstacle_avoidance_3_impact_var"):
            self.obstacle_avoidance_3_impact_var = tk.StringVar(value="Impact: unknown")
        if not hasattr(self, "obstacle_avoidance_3_or2_risk_var"):
            self.obstacle_avoidance_3_or2_risk_var = tk.StringVar(value="OR2 risk: --")
        if not hasattr(self, "obstacle_avoidance_3_or2_state_var"):
            self.obstacle_avoidance_3_or2_state_var = tk.StringVar(value="State: --")
        if not hasattr(self, "obstacle_avoidance_3_or2_frame_count_var"):
            self.obstacle_avoidance_3_or2_frame_count_var = tk.StringVar(value="Frames: 0")
        if not hasattr(self, "obstacle_avoidance_3_or2_front_depth_var"):
            self.obstacle_avoidance_3_or2_front_depth_var = tk.StringVar(value="Front depth: --")
        if not hasattr(self, "obstacle_avoidance_3_or2_can_forward_var"):
            self.obstacle_avoidance_3_or2_can_forward_var = tk.StringVar(value="Can forward: --")
        if not hasattr(self, "obstacle_avoidance_3_or2_selected_direction_var"):
            self.obstacle_avoidance_3_or2_selected_direction_var = tk.StringVar(value="Selected direction: --")
        if not hasattr(self, "obstacle_avoidance_3_or2_corridor_var"):
            self.obstacle_avoidance_3_or2_corridor_var = tk.StringVar(value="Corridors: --")
        if not hasattr(self, "obstacle_avoidance_3_or2_capture_dir_var"):
            self.obstacle_avoidance_3_or2_capture_dir_var = tk.StringVar(value="OR2 capture: --")
        if not hasattr(self, "obstacle_avoidance_3_or2_interval_s_var"):
            self.obstacle_avoidance_3_or2_interval_s_var = tk.StringVar(value="1.0")
        if not hasattr(self, "obstacle_avoidance_3_record_var"):
            self.obstacle_avoidance_3_record_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_3_record_summary_var"):
            self.obstacle_avoidance_3_record_summary_var = tk.StringVar(value="Select an episode to inspect records.")
        if not hasattr(self, "obstacle_avoidance_3_plan_data"):
            self.obstacle_avoidance_3_plan_data = None
        if not hasattr(self, "obstacle_avoidance_3_window"):
            self.obstacle_avoidance_3_window = None
        if not hasattr(self, "obstacle_avoidance_3_report_text"):
            self.obstacle_avoidance_3_report_text = None
        if not hasattr(self, "obstacle_avoidance_3_project_combo"):
            self.obstacle_avoidance_3_project_combo = None
        if not hasattr(self, "obstacle_avoidance_3_environment_combo"):
            self.obstacle_avoidance_3_environment_combo = None
        if not hasattr(self, "obstacle_avoidance_3_method_combo"):
            self.obstacle_avoidance_3_method_combo = None
        if not hasattr(self, "obstacle_avoidance_3_episode_tree"):
            self.obstacle_avoidance_3_episode_tree = None
        if not hasattr(self, "obstacle_avoidance_3_tree_iids"):
            self.obstacle_avoidance_3_tree_iids = {}
        if not hasattr(self, "obstacle_avoidance_3_record_combo"):
            self.obstacle_avoidance_3_record_combo = None
        if not hasattr(self, "obstacle_avoidance_3_or2_state_label"):
            self.obstacle_avoidance_3_or2_state_label = None
        if not hasattr(self, "obstacle_avoidance_3_or2_rgb_label"):
            self.obstacle_avoidance_3_or2_rgb_label = None
        if not hasattr(self, "obstacle_avoidance_3_or2_mask_label"):
            self.obstacle_avoidance_3_or2_mask_label = None
        if not hasattr(self, "obstacle_avoidance_3_or2_rgb_photo"):
            self.obstacle_avoidance_3_or2_rgb_photo = None
        if not hasattr(self, "obstacle_avoidance_3_or2_mask_photo"):
            self.obstacle_avoidance_3_or2_mask_photo = None
        if not hasattr(self, "obstacle_avoidance_3_or2_report_text"):
            self.obstacle_avoidance_3_or2_report_text = None
        if not hasattr(self, "obstacle_avoidance_3_record_entries"):
            self.obstacle_avoidance_3_record_entries = []
        if not hasattr(self, "obstacle_avoidance_3_runner_thread"):
            self.obstacle_avoidance_3_runner_thread = None
        if not hasattr(self, "obstacle_avoidance_3_or2_monitor_thread"):
            self.obstacle_avoidance_3_or2_monitor_thread = None
        if not hasattr(self, "obstacle_avoidance_3_or2_monitor_stop_event"):
            self.obstacle_avoidance_3_or2_monitor_stop_event = threading.Event()
        if not hasattr(self, "obstacle_avoidance_3_or2_monitor_frame_index"):
            self.obstacle_avoidance_3_or2_monitor_frame_index = 1
        if not hasattr(self, "obstacle_avoidance_3_stop_event"):
            self.obstacle_avoidance_3_stop_event = threading.Event()
        if not hasattr(self, "obstacle_avoidance_3_collision_alert_keys"):
            self.obstacle_avoidance_3_collision_alert_keys = set()

    def obstacle_avoidance_3_data_root(self) -> Path:
        self.ensure_obstacle_avoidance_3_state()
        raw = str(self.obstacle_avoidance_3_data_dir_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else PROJECT_ROOT / "obstacle_avoidance_3_data"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def obstacle_avoidance_3_plan_path(self) -> Path:
        self.ensure_obstacle_avoidance_3_state()
        raw = str(self.obstacle_avoidance_3_plan_json_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else self.obstacle_avoidance_3_data_root() / "plans" / DEFAULT_PLAN_FILENAME
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def obstacle_avoidance_3_oa2_plan_path(self) -> Path:
        return (PROJECT_ROOT / "obstacle_avoidance_2_data" / "plans" / "obstacle_avoidance_2_plans.json").resolve()

    def obstacle_avoidance_3_model_path(self) -> Path:
        self.ensure_obstacle_avoidance_3_state()
        raw = str(self.obstacle_avoidance_3_model_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def open_obstacle_avoidance_3_window(self) -> None:
        self.ensure_obstacle_avoidance_3_state()
        if self.obstacle_avoidance_3_window is not None and self.obstacle_avoidance_3_window.winfo_exists():
            self.obstacle_avoidance_3_window.lift()
            self.obstacle_avoidance_3_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Obstacle Avoidance 3 - OR2 Risk-Region Route Control")
        window.geometry("1180x760")
        window.minsize(1000, 620)
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        self.obstacle_avoidance_3_window = window

        canvas = tk.Canvas(window, highlightthickness=0)
        y_scroll = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
        x_scroll = tk.Scrollbar(window, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        content = tk.Frame(canvas)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.grid_columnconfigure(0, weight=1)

        def on_content_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(content_window, width=max(content.winfo_reqwidth(), int(event.width)))
            canvas.configure(scrollregion=canvas.bbox("all"))

        content.bind("<Configure>", on_content_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        self.build_obstacle_avoidance_3_window(content)
        try:
            self.load_obstacle_avoidance_3_plans()
        except Exception as exc:
            self.obstacle_avoidance_3_status_var.set(f"OA3 load failed: {exc}")
            self.obstacle_avoidance_3_report({"status": "error", "task": "load_plans", "error": str(exc)})

        def close_window() -> None:
            self.stop_obstacle_avoidance_3_or2_monitor()
            self.obstacle_avoidance_3_window = None
            self.obstacle_avoidance_3_report_text = None
            self.obstacle_avoidance_3_record_combo = None
            self.obstacle_avoidance_3_or2_state_label = None
            self.obstacle_avoidance_3_or2_rgb_label = None
            self.obstacle_avoidance_3_or2_mask_label = None
            self.obstacle_avoidance_3_or2_report_text = None
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)

    def build_obstacle_avoidance_3_window(self, window: tk.Widget) -> None:
        config = tk.LabelFrame(window, text="Plan / Model Storage")
        config.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        config.grid_columnconfigure(1, weight=1)
        config.grid_columnconfigure(4, weight=1)
        tk.Label(config, text="Data Dir").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(config, textvariable=self.obstacle_avoidance_3_data_dir_var).grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        tk.Button(config, text="Browse", command=self.select_obstacle_avoidance_3_data_dir).grid(row=0, column=2, padx=6, pady=5)
        tk.Label(config, text="Plan JSON").grid(row=0, column=3, sticky="w", padx=6, pady=5)
        tk.Entry(config, textvariable=self.obstacle_avoidance_3_plan_json_var).grid(row=0, column=4, sticky="ew", padx=6, pady=5)
        tk.Button(config, text="Reload", command=self.load_obstacle_avoidance_3_plans).grid(row=0, column=5, padx=6, pady=5)
        tk.Button(config, text="Save", command=self.save_obstacle_avoidance_3_plans).grid(row=0, column=6, padx=6, pady=5)
        tk.Button(config, text="Reset 10 Points", command=self.reset_obstacle_avoidance_3_plans).grid(row=0, column=7, padx=6, pady=5)
        tk.Button(config, text="Sync Points From OA2", command=self.sync_obstacle_avoidance_3_points_from_oa2).grid(
            row=0, column=8, padx=6, pady=5
        )
        tk.Label(config, text="OR2 Model").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(config, textvariable=self.obstacle_avoidance_3_model_var).grid(row=1, column=1, columnspan=4, sticky="ew", padx=6, pady=5)
        tk.Button(config, text="Browse", command=self.select_obstacle_avoidance_3_model).grid(row=1, column=5, padx=6, pady=5)
        tk.Button(config, text="Default", command=self.reset_obstacle_avoidance_3_model).grid(row=1, column=6, padx=6, pady=5)

        project = tk.LabelFrame(window, text="Project / Method")
        project.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3):
            project.grid_columnconfigure(col, weight=1)
        tk.Label(project, text="Project").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_3_project_combo = ttk.Combobox(project, textvariable=self.obstacle_avoidance_3_project_var, state="readonly", width=32)
        self.obstacle_avoidance_3_project_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        self.obstacle_avoidance_3_project_combo.bind("<<ComboboxSelected>>", lambda _event: self.select_obstacle_avoidance_3_project())
        tk.Label(project, text="Name").grid(row=0, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(project, textvariable=self.obstacle_avoidance_3_project_name_var).grid(row=0, column=3, sticky="ew", padx=6, pady=5)
        tk.Button(project, text="Apply", command=self.apply_obstacle_avoidance_3_project).grid(row=0, column=4, padx=6, pady=5)
        tk.Label(project, text="Method").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_3_method_combo = ttk.Combobox(project, textvariable=self.obstacle_avoidance_3_method_var, state="readonly", width=42)
        self.obstacle_avoidance_3_method_combo.grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=5)
        tk.Label(project, textvariable=self.obstacle_avoidance_3_status_var, anchor="w").grid(row=2, column=0, columnspan=5, sticky="ew", padx=6, pady=(3, 5))

        actions = tk.LabelFrame(window, text="Start-Goal OA3 Experiment")
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(actions, text="1. Locate Start", command=self.locate_obstacle_avoidance_3_start).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="2. Apply Algorithm", command=self.apply_obstacle_avoidance_3_algorithm).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="3. Execute To Goal", command=self.execute_obstacle_avoidance_3_to_goal).pack(side="left", padx=6, pady=6)
        tk.Button(
            actions,
            text="Stop All",
            command=self.emergency_stop_obstacle_avoidance_3_all,
            bg="#b00020",
            fg="white",
            activebackground="#7a0016",
            activeforeground="white",
        ).pack(side="left", padx=(18, 10), pady=6)
        for var in (
            self.obstacle_avoidance_3_goal_distance_var,
            self.obstacle_avoidance_3_goal_reached_var,
            self.obstacle_avoidance_3_impact_var,
            self.obstacle_avoidance_3_or2_risk_var,
        ):
            tk.Label(actions, textvariable=var, anchor="w").pack(side="left", padx=12, pady=6)

        self.build_obstacle_avoidance_3_or2_monitor_panel(window, row=3)

        body = tk.Frame(window)
        body.grid(row=4, column=0, sticky="nsew", padx=8, pady=4)
        body.grid_columnconfigure(0, weight=1)
        episodes = tk.LabelFrame(body, text="Episodes")
        episodes.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        episodes.grid_columnconfigure(0, weight=1)
        episodes.grid_rowconfigure(0, weight=1)
        columns = ("on", "episode_id", "start", "goal", "method", "hint", "note")
        tree = ttk.Treeview(episodes, columns=columns, show="headings", height=11, selectmode="extended")
        widths = (55, 80, 240, 240, 260, 150, 220)
        labels = ("On", "ID", "Start [x,y,z,yaw]", "Goal [x,y,z,yaw]", "Method", "Hint", "Note")
        for column, label, width in zip(columns, labels, widths):
            tree.heading(column, text=label)
            tree.column(column, width=width, anchor="w", stretch=True)
        y_scroll = tk.Scrollbar(episodes, orient="vertical", command=tree.yview)
        x_scroll = tk.Scrollbar(episodes, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        tree.bind("<<TreeviewSelect>>", lambda _event: self.select_obstacle_avoidance_3_episode_from_tree())
        self.obstacle_avoidance_3_episode_tree = tree

        editor = tk.LabelFrame(body, text="Episode Editor")
        editor.grid(row=0, column=1, sticky="ns", padx=(6, 0), pady=0)
        tk.Checkbutton(editor, text="Enabled", variable=self.obstacle_avoidance_3_enabled_var).grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(editor, textvariable=self.obstacle_avoidance_3_episode_id_var, width=16).grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        rows = [
            ("Start Pose", self.obstacle_avoidance_3_start_pose_var),
            ("Goal Pose", self.obstacle_avoidance_3_goal_pose_var),
            ("Scenario", self.obstacle_avoidance_3_scenario_var),
            ("Hint", self.obstacle_avoidance_3_obstacle_hint_var),
            ("Note", self.obstacle_avoidance_3_note_var),
        ]
        for index, (label, var) in enumerate(rows, start=1):
            tk.Label(editor, text=label).grid(row=index, column=0, sticky="w", padx=6, pady=5)
            tk.Entry(editor, textvariable=var, width=28).grid(row=index, column=1, sticky="ew", padx=6, pady=5)
        tk.Button(editor, text="Apply Episode", command=self.apply_obstacle_avoidance_3_episode).grid(row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 5))
        tk.Button(editor, text="Add Episode", command=self.add_obstacle_avoidance_3_episode).grid(row=7, column=0, sticky="ew", padx=6, pady=5)
        tk.Button(editor, text="Delete Episode", command=self.delete_obstacle_avoidance_3_episode).grid(row=7, column=1, sticky="ew", padx=6, pady=5)

        runner = tk.LabelFrame(window, text="Collection Runner")
        runner.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(runner, text="Run Current Episode", command=self.run_obstacle_avoidance_3_current_episode).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Run Selected Episodes", command=self.run_obstacle_avoidance_3_selected).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Stop Run", command=self.stop_obstacle_avoidance_3_runner).pack(side="left", padx=(18, 6), pady=6)

        records = tk.LabelFrame(window, text="Run Records")
        records.grid(row=6, column=0, sticky="ew", padx=8, pady=4)
        records.grid_columnconfigure(1, weight=1)
        tk.Label(records, text="Record").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_3_record_combo = ttk.Combobox(records, textvariable=self.obstacle_avoidance_3_record_var, state="readonly")
        self.obstacle_avoidance_3_record_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        self.obstacle_avoidance_3_record_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_obstacle_avoidance_3_record_summary())
        tk.Button(records, text="Reload Records", command=lambda: self.refresh_obstacle_avoidance_3_records()).grid(row=0, column=2, padx=6, pady=5)
        tk.Button(records, text="View Record", command=self.view_obstacle_avoidance_3_selected_record).grid(row=0, column=3, padx=6, pady=5)
        tk.Label(records, textvariable=self.obstacle_avoidance_3_record_summary_var, anchor="w", justify="left").grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 6))

        report_frame = tk.LabelFrame(window, text="Report")
        report_frame.grid(row=7, column=0, sticky="nsew", padx=8, pady=(4, 8))
        report_frame.grid_columnconfigure(0, weight=1)
        report_frame.grid_rowconfigure(0, weight=1)
        report = tk.Text(report_frame, height=8, wrap="none", font=("Consolas", 9))
        report_y = tk.Scrollbar(report_frame, orient="vertical", command=report.yview)
        report_x = tk.Scrollbar(report_frame, orient="horizontal", command=report.xview)
        report.configure(yscrollcommand=report_y.set, xscrollcommand=report_x.set)
        report.grid(row=0, column=0, sticky="nsew")
        report_y.grid(row=0, column=1, sticky="ns")
        report_x.grid(row=1, column=0, sticky="ew")
        self.obstacle_avoidance_3_report_text = report

    def build_obstacle_avoidance_3_or2_monitor_panel(self, window: tk.Widget, *, row: int) -> None:
        monitor = tk.LabelFrame(window, text="OR2 Live Risk Monitor")
        monitor.grid(row=row, column=0, sticky="ew", padx=8, pady=4)
        monitor.grid_columnconfigure(0, weight=1)

        controls = tk.Frame(monitor)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 3))
        tk.Button(controls, text="Start OR2 Monitor", command=self.start_obstacle_avoidance_3_or2_monitor).pack(side="left", padx=4, pady=3)
        tk.Button(controls, text="Stop OR2 Monitor", command=self.stop_obstacle_avoidance_3_or2_monitor).pack(side="left", padx=4, pady=3)
        tk.Button(controls, text="Capture OR2 Once", command=self.capture_obstacle_avoidance_3_or2_once).pack(side="left", padx=4, pady=3)
        tk.Label(controls, text="Interval s").pack(side="left", padx=(18, 3), pady=3)
        tk.Entry(controls, textvariable=self.obstacle_avoidance_3_or2_interval_s_var, width=7).pack(side="left", padx=3, pady=3)
        self.obstacle_avoidance_3_or2_state_label = tk.Label(
            controls,
            textvariable=self.obstacle_avoidance_3_or2_state_var,
            width=28,
            anchor="center",
            bg="#d9d9d9",
            fg="black",
        )
        self.obstacle_avoidance_3_or2_state_label.pack(side="left", padx=(18, 8), pady=3)
        for var in (
            self.obstacle_avoidance_3_or2_frame_count_var,
            self.obstacle_avoidance_3_or2_front_depth_var,
            self.obstacle_avoidance_3_or2_can_forward_var,
            self.obstacle_avoidance_3_or2_selected_direction_var,
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
            tk.Label(legend, text=f"{label}: {text}", bg=color, fg="white" if label == "must_stop" else "black").pack(
                side="left", padx=4, pady=3, ipadx=8, ipady=2
            )
        tk.Label(legend, textvariable=self.obstacle_avoidance_3_or2_corridor_var, anchor="w").pack(
            side="left", fill="x", expand=True, padx=(18, 4), pady=3
        )

        image_row = tk.Frame(monitor)
        image_row.grid(row=2, column=0, sticky="w", padx=6, pady=4)
        rgb_frame = tk.LabelFrame(image_row, text="RGB", width=380, height=260)
        rgb_frame.pack(side="left", padx=(0, 8), pady=0)
        rgb_frame.pack_propagate(False)
        mask_frame = tk.LabelFrame(image_row, text="A+2 Risk Overlay", width=380, height=260)
        mask_frame.pack(side="left", padx=(8, 0), pady=0)
        mask_frame.pack_propagate(False)
        self.obstacle_avoidance_3_or2_rgb_label = tk.Label(rgb_frame, text="No RGB yet")
        self.obstacle_avoidance_3_or2_rgb_label.pack(fill="both", expand=True, padx=6, pady=6)
        self.obstacle_avoidance_3_or2_mask_label = tk.Label(mask_frame, text="No overlay yet")
        self.obstacle_avoidance_3_or2_mask_label.pack(fill="both", expand=True, padx=6, pady=6)

        bottom = tk.Frame(monitor)
        bottom.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        bottom.grid_columnconfigure(1, weight=1)
        tk.Label(bottom, textvariable=self.obstacle_avoidance_3_or2_capture_dir_var, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.obstacle_avoidance_3_or2_report_text = tk.Text(bottom, height=4, wrap="none", font=("Consolas", 8))
        self.obstacle_avoidance_3_or2_report_text.grid(row=0, column=1, sticky="ew")

    def obstacle_avoidance_3_report(self, payload: Any) -> None:
        text = payload.rstrip() if isinstance(payload, str) else json.dumps(jsonable(payload), indent=2, ensure_ascii=False)
        widget = self.obstacle_avoidance_3_report_text
        if widget is not None:
            try:
                widget.insert("end", text + "\n\n")
                widget.see("end")
            except tk.TclError:
                pass

    def obstacle_avoidance_3_set_report(self, payload: Any) -> None:
        widget = self.obstacle_avoidance_3_report_text
        if widget is not None:
            try:
                widget.delete("1.0", "end")
            except tk.TclError:
                pass
        self.obstacle_avoidance_3_report(payload)

    def select_obstacle_avoidance_3_data_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self.obstacle_avoidance_3_data_root()))
        if path:
            self.obstacle_avoidance_3_data_dir_var.set(str(Path(path).resolve()))

    def select_obstacle_avoidance_3_model(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str((PROJECT_ROOT / "obstacle_representation_2_data" / "models").resolve()),
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.obstacle_avoidance_3_model_var.set(str(Path(path).resolve()))

    def reset_obstacle_avoidance_3_model(self) -> None:
        self.obstacle_avoidance_3_model_var.set(str(PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"))

    def parse_obstacle_avoidance_3_or2_interval_s(self) -> float:
        try:
            value = float(self.obstacle_avoidance_3_or2_interval_s_var.get().strip())
        except Exception:
            value = 1.0
        if not math.isfinite(value):
            value = 1.0
        return max(0.2, min(30.0, value))

    def obstacle_avoidance_3_array_to_photo(
        self,
        image: np.ndarray,
        *,
        max_width: int = 340,
        max_height: int = 230,
    ) -> ImageTk.PhotoImage:
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        scale = min(max_width / max(1, pil.width), max_height / max(1, pil.height), 1.0)
        if scale < 1.0:
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def update_obstacle_avoidance_3_or2_state_color(self, risk: str) -> None:
        label = self.obstacle_avoidance_3_or2_state_label
        if label is None:
            return
        colors = {
            "clear": ("#3cbe5a", "black"),
            "clearance_warning": ("#f5c82d", "black"),
            "obstacle_warning": ("#f5825f", "black"),
            "must_stop": ("#be1423", "white"),
            "starting": ("#d9d9d9", "black"),
            "failed": ("#7a1f1f", "white"),
            "predict_error": ("#7a1f1f", "white"),
        }
        bg, fg = colors.get(str(risk), ("#d9d9d9", "black"))
        try:
            label.configure(bg=bg, fg=fg)
        except tk.TclError:
            pass

    def obstacle_avoidance_3_corridor_text(self, corridor_risks: Dict[str, Any]) -> str:
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

    def obstacle_avoidance_3_or2_run_dir(self, suffix: str) -> Path:
        root = PROJECT_ROOT / "obstacle_representation_2_data" / "live_monitor"
        return root / datetime.now().strftime(f"%Y%m%d-%H%M%S_{suffix}")

    def capture_obstacle_avoidance_3_or2_once(self) -> None:
        self.ensure_obstacle_avoidance_3_state()
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_3_status_var.set("OA3 OR2: start Unreal from the main window first")
            return
        thread = getattr(self, "obstacle_avoidance_3_or2_monitor_thread", None)
        if thread is not None and thread.is_alive():
            self.obstacle_avoidance_3_status_var.set("OA3 OR2 monitor is already running")
            return
        model_path = self.obstacle_avoidance_3_model_path()
        if not model_path.is_file():
            self.obstacle_avoidance_3_status_var.set(f"OA3 OR2 model not found: {model_path}")
            return
        run_dir = self.obstacle_avoidance_3_or2_run_dir("oa3_or2_capture_once")

        def worker() -> None:
            try:
                predictor = ObstacleRepresentation2Predictor(model_path)
                result = self.capture_obstacle_avoidance_3_or2_frame(
                    session,
                    predictor,
                    model_path,
                    run_dir,
                    1,
                    action_name="oa3_or2_capture_once",
                )
                self.root.after(0, lambda r=result: self.apply_obstacle_avoidance_3_or2_result(r))
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_avoidance_3_status_var.set(f"OA3 OR2 capture failed: {e}"),
                        self.update_obstacle_avoidance_3_or2_state_color("failed"),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def start_obstacle_avoidance_3_or2_monitor(self) -> None:
        self.ensure_obstacle_avoidance_3_state()
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_3_status_var.set("OA3 OR2: start Unreal from the main window first")
            return
        thread = getattr(self, "obstacle_avoidance_3_or2_monitor_thread", None)
        if thread is not None and thread.is_alive():
            self.obstacle_avoidance_3_status_var.set("OA3 OR2 monitor already running")
            return
        model_path = self.obstacle_avoidance_3_model_path()
        if not model_path.is_file():
            self.obstacle_avoidance_3_status_var.set(f"OA3 OR2 model not found: {model_path}")
            return
        interval_s = self.parse_obstacle_avoidance_3_or2_interval_s()
        self.obstacle_avoidance_3_or2_interval_s_var.set(f"{interval_s:g}")
        run_dir = self.obstacle_avoidance_3_or2_run_dir("oa3_or2_risk_monitor")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.obstacle_avoidance_3_or2_monitor_stop_event.clear()
        self.obstacle_avoidance_3_or2_capture_dir_var.set(f"OR2 capture: {run_dir}")
        self.obstacle_avoidance_3_or2_state_var.set("State: starting")
        self.update_obstacle_avoidance_3_or2_state_color("starting")
        self.obstacle_avoidance_3_status_var.set("OA3 OR2 monitor starting")

        def worker() -> None:
            self.run_obstacle_avoidance_3_or2_monitor(session, model_path, run_dir, interval_s)

        self.obstacle_avoidance_3_or2_monitor_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_3_or2_monitor_thread.start()

    def stop_obstacle_avoidance_3_or2_monitor(self) -> None:
        self.ensure_obstacle_avoidance_3_state()
        self.obstacle_avoidance_3_or2_monitor_stop_event.set()
        thread = getattr(self, "obstacle_avoidance_3_or2_monitor_thread", None)
        if thread is not None and thread.is_alive():
            self.obstacle_avoidance_3_status_var.set("OA3 OR2 monitor stopping")

    def run_obstacle_avoidance_3_or2_monitor(
        self,
        session: flight.DroneFlightSession,
        model_path: Path,
        run_dir: Path,
        interval_s: float,
    ) -> None:
        started_at = datetime.now().isoformat(timespec="milliseconds")
        events_path = run_dir / "risk_events.jsonl"
        summary_path = run_dir / "risk_monitor_summary.json"
        predictor = ObstacleRepresentation2Predictor(model_path)
        frame_count = 0
        state_counts: Dict[str, int] = {}
        min_front_depth_cm: Optional[float] = None
        old_mode = str(getattr(session.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING))
        try:
            if hasattr(self, "sync_capture_options_to_session"):
                self.sync_capture_options_to_session(session)
            session.args.lidar_capture_processing = "minimal"
            frame_index = 1
            while not self.obstacle_avoidance_3_or2_monitor_stop_event.is_set():
                tick_start = time.time()
                result = self.capture_obstacle_avoidance_3_or2_frame(
                    session,
                    predictor,
                    model_path,
                    run_dir,
                    frame_index,
                    action_name="oa3_or2_risk_monitor",
                    events_path=events_path,
                )
                frame_count += 1
                risk = str(result.get("prediction", {}).get("front_risk_state", "clear"))
                state_counts[risk] = state_counts.get(risk, 0) + 1
                front_depth = float(result.get("summary", {}).get("front_min_depth_cm", 0.0) or 0.0)
                if front_depth > 0.0:
                    min_front_depth_cm = front_depth if min_front_depth_cm is None else min(min_front_depth_cm, front_depth)
                result["frame_count"] = frame_count
                self.root.after(0, lambda r=result: self.apply_obstacle_avoidance_3_or2_result(r))
                frame_index += 1
                remaining = max(0.0, interval_s - (time.time() - tick_start))
                if self.obstacle_avoidance_3_or2_monitor_stop_event.wait(remaining):
                    break
        except Exception as exc:
            self.root.after(
                0,
                lambda e=exc: (
                    self.obstacle_avoidance_3_status_var.set(f"OA3 OR2 monitor failed: {e}"),
                    self.update_obstacle_avoidance_3_or2_state_color("failed"),
                ),
            )
        finally:
            try:
                session.args.lidar_capture_processing = old_mode
            except Exception:
                pass
            summary = {
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="milliseconds"),
                "capture_dir": str(run_dir),
                "frame_count": frame_count,
                "state_counts": state_counts,
                "min_front_depth_cm": min_front_depth_cm,
                "model_path": str(model_path),
                "events_path": str(events_path),
            }
            write_json(summary_path, summary)
            self.root.after(0, lambda n=frame_count, p=summary_path: self.obstacle_avoidance_3_status_var.set(f"OA3 OR2 monitor stopped, frames={n}, summary={p}"))

    def capture_obstacle_avoidance_3_or2_frame(
        self,
        session: flight.DroneFlightSession,
        predictor: ObstacleRepresentation2Predictor,
        model_path: Path,
        run_dir: Path,
        frame_index: int,
        *,
        action_name: str,
        events_path: Optional[Path] = None,
        relative_target: Optional[Dict[str, Any]] = None,
        last_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        capture = session.capture_lidar_stream_frame(
            str(run_dir),
            frame_index,
            {
                "source": "obstacle_avoidance_3_or2_monitor",
                "action_name": action_name,
            },
        )
        summary = capture.get("pointcloud_summary")
        if not isinstance(summary, dict):
            summary = capture.get("depth_obstacle_summary") if isinstance(capture.get("depth_obstacle_summary"), dict) else {}
        event = {
            "rgb_path": capture.get("rgb_path", ""),
            "depth_npy_path": capture.get("depth_npy_path", ""),
            "capture_dir": capture.get("capture_dir", ""),
            "pose": capture.get("pose", {}),
            "pointcloud_summary": summary,
            "depth_obstacle_summary": capture.get("depth_obstacle_summary", {}),
            "relative_target": relative_target or {},
        }
        predict_start = time.time()
        prediction = predictor.predict(event["rgb_path"], event)
        prediction_latency_ms = round((time.time() - predict_start) * 1000.0, 3)
        rgb_image = np.asarray(Image.open(event["rgb_path"]).convert("RGB"), dtype=np.uint8)
        mask_image = render_affordance_overlay(rgb_image, prediction)
        capture_dir = Path(str(capture.get("capture_dir", "")))
        overlay_path = capture_dir / "or2_risk_overlay.png"
        prediction_path = capture_dir / "or2_risk_prediction.json"
        Image.fromarray(mask_image).save(overlay_path)
        rule = select_or2_direction(prediction, summary, relative_target or {}, last_action=last_action)
        prediction_payload = {
            **serializable_or2_prediction(prediction),
            "prediction_latency_ms": prediction_latency_ms,
            "risk_overlay_path": str(overlay_path),
            "or2_selected_direction": rule.get("selected_direction", ""),
            "or2_corridor_risks": rule.get("corridor_risks", {}),
            "or2_candidate_action_scores": rule.get("candidate_action_scores", {}),
        }
        write_json(prediction_path, prediction_payload)
        front_depth = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
        event_record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_index": int(frame_index),
            "frame_dir": str(capture_dir),
            "rgb_path": str(capture.get("rgb_path", "")),
            "depth_path": str(capture.get("depth_npy_path", "")),
            "depth_preview_path": str(capture.get("depth_preview_path", "")),
            "risk_overlay_path": str(overlay_path),
            "prediction_json_path": str(prediction_path),
            "model_path": str(model_path),
            "front_risk_state": str(prediction.get("front_risk_state", "clear")),
            "can_forward": bool(prediction.get("can_forward", False)),
            "must_stop": bool(prediction.get("must_stop", False)),
            "front_min_depth_cm": front_depth,
            "risk_fractions": {
                "clearance_warning": float(prediction.get("front_clearance_fraction", 0.0) or 0.0),
                "obstacle_warning": float(prediction.get("front_warning_fraction", 0.0) or 0.0),
                "must_stop": float(prediction.get("front_stop_fraction", 0.0) or 0.0),
            },
            "or2_corridor_risks": rule.get("corridor_risks", {}),
            "or2_selected_direction": rule.get("selected_direction", ""),
            "or2_candidate_action_scores": rule.get("candidate_action_scores", {}),
            "pointcloud_summary": summary,
            "prediction_latency_ms": prediction_latency_ms,
            "session_pose": capture.get("pose", {}),
        }
        if events_path is not None:
            append_jsonl(events_path, event_record)
        return {
            "capture": capture,
            "event": event,
            "event_record": event_record,
            "prediction": prediction,
            "rule": rule,
            "summary": summary,
            "rgb_image": rgb_image,
            "mask_image": mask_image,
            "frame_count": frame_index,
        }

    def apply_obstacle_avoidance_3_or2_result(self, result: Dict[str, Any]) -> None:
        prediction = result.get("prediction") if isinstance(result.get("prediction"), dict) else {}
        rule = result.get("rule") if isinstance(result.get("rule"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        event_record = result.get("event_record") if isinstance(result.get("event_record"), dict) else {}
        risk = str(prediction.get("front_risk_state", "clear"))
        front_depth = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
        selected_direction = str(rule.get("selected_direction") or event_record.get("or2_selected_direction") or "--")
        frame_count = int(result.get("frame_count", 0) or 0)
        self.obstacle_avoidance_3_or2_state_var.set(f"State: {risk}")
        self.obstacle_avoidance_3_or2_frame_count_var.set(f"Frames: {frame_count}")
        self.obstacle_avoidance_3_or2_front_depth_var.set(f"Front depth: {front_depth:.1f} cm")
        self.obstacle_avoidance_3_or2_can_forward_var.set(
            f"Can forward: {'yes' if prediction.get('can_forward', False) else 'no'} / must stop: {'yes' if prediction.get('must_stop', False) else 'no'}"
        )
        self.obstacle_avoidance_3_or2_selected_direction_var.set(f"Selected direction: {selected_direction}")
        self.obstacle_avoidance_3_or2_corridor_var.set(self.obstacle_avoidance_3_corridor_text(rule.get("corridor_risks", {})))
        self.obstacle_avoidance_3_or2_capture_dir_var.set(f"OR2 capture: {event_record.get('frame_dir', '--')}")
        self.obstacle_avoidance_3_or2_risk_var.set(f"OR2 risk: {risk}")
        self.update_obstacle_avoidance_3_or2_state_color(risk)
        if self.obstacle_avoidance_3_or2_rgb_label is not None and isinstance(result.get("rgb_image"), np.ndarray):
            self.obstacle_avoidance_3_or2_rgb_photo = self.obstacle_avoidance_3_array_to_photo(result["rgb_image"])
            self.obstacle_avoidance_3_or2_rgb_label.configure(image=self.obstacle_avoidance_3_or2_rgb_photo, text="")
        if self.obstacle_avoidance_3_or2_mask_label is not None and isinstance(result.get("mask_image"), np.ndarray):
            self.obstacle_avoidance_3_or2_mask_photo = self.obstacle_avoidance_3_array_to_photo(result["mask_image"])
            self.obstacle_avoidance_3_or2_mask_label.configure(image=self.obstacle_avoidance_3_or2_mask_photo, text="")
        if self.obstacle_avoidance_3_or2_report_text is not None:
            payload = {
                "event": event_record,
                "prediction": serializable_or2_prediction(prediction),
                "rule": rule,
            }
            self.obstacle_avoidance_3_or2_report_text.delete("1.0", "end")
            self.obstacle_avoidance_3_or2_report_text.insert("end", json.dumps(jsonable(payload), indent=2, ensure_ascii=False))

    def load_obstacle_avoidance_3_plans(self) -> None:
        data = load_plans(self.obstacle_avoidance_3_plan_path())
        synced = self.sync_obstacle_avoidance_3_points_from_oa2(silent=True, base_data=data)
        if isinstance(synced, dict):
            data = synced
        self.obstacle_avoidance_3_plan_data = data
        active_id = str(data.get("active_project_id", DEFAULT_PROJECT_ID) or DEFAULT_PROJECT_ID)
        self.obstacle_avoidance_3_project_var.set(active_id)
        self.refresh_obstacle_avoidance_3_project_combos()
        self.select_obstacle_avoidance_3_project()
        count = self.ensure_obstacle_avoidance_3_project_points()
        self.refresh_obstacle_avoidance_3_tree()
        self.obstacle_avoidance_3_status_var.set(f"OA3 plans loaded: {count} episode(s)")

    def sync_obstacle_avoidance_3_points_from_oa2(
        self,
        *,
        silent: bool = False,
        base_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        oa2_path = self.obstacle_avoidance_3_oa2_plan_path()
        if not oa2_path.is_file():
            if not silent:
                self.obstacle_avoidance_3_status_var.set(f"OA3 sync skipped; OA2 plan not found: {oa2_path}")
            return base_data
        try:
            oa2_raw = json.loads(oa2_path.read_text(encoding="utf-8"))
            current = base_data if isinstance(base_data, dict) else (
                self.obstacle_avoidance_3_plan_data
                if isinstance(self.obstacle_avoidance_3_plan_data, dict)
                else load_plans(self.obstacle_avoidance_3_plan_path())
            )
            synced = sync_oa2_plan_into_oa3(current, oa2_raw if isinstance(oa2_raw, dict) else {})
            projects = synced.get("projects", []) if isinstance(synced, dict) else []
            episodes = projects[0].get("episodes", []) if projects and isinstance(projects[0], dict) else []
            count = len(episodes) if isinstance(episodes, list) else 0
            if count <= 0:
                if not silent:
                    self.obstacle_avoidance_3_status_var.set("OA3 sync skipped; OA2 active project has no valid episodes")
                return base_data
            self.obstacle_avoidance_3_plan_data = synced
            save_plans(self.obstacle_avoidance_3_plan_path(), synced)
            self.obstacle_avoidance_3_project_var.set(DEFAULT_PROJECT_ID)
            self.refresh_obstacle_avoidance_3_project_combos()
            self.select_obstacle_avoidance_3_project()
            if not silent:
                self.obstacle_avoidance_3_status_var.set(f"OA3 synced {count} point(s) from OA2")
                self.obstacle_avoidance_3_report(
                    {
                        "status": "synced_points_from_oa2",
                        "oa2_plan_path": str(oa2_path),
                        "oa3_plan_path": str(self.obstacle_avoidance_3_plan_path()),
                        "episode_count": count,
                    }
                )
            return synced
        except Exception as exc:
            if not silent:
                self.obstacle_avoidance_3_status_var.set(f"OA3 sync from OA2 failed: {exc}")
                self.obstacle_avoidance_3_report({"status": "error", "task": "sync_points_from_oa2", "error": str(exc)})
            return base_data

    def save_obstacle_avoidance_3_plans(self) -> None:
        data = self.obstacle_avoidance_3_plan_data if isinstance(self.obstacle_avoidance_3_plan_data, dict) else make_default_plans()
        save_plans(self.obstacle_avoidance_3_plan_path(), data)
        self.obstacle_avoidance_3_status_var.set(f"OA3 plans saved: {self.obstacle_avoidance_3_plan_path()}")

    def reset_obstacle_avoidance_3_plans(self) -> None:
        self.obstacle_avoidance_3_plan_data = make_default_plans()
        save_plans(self.obstacle_avoidance_3_plan_path(), self.obstacle_avoidance_3_plan_data)
        self.refresh_obstacle_avoidance_3_project_combos()
        self.select_obstacle_avoidance_3_project()
        self.obstacle_avoidance_3_status_var.set("OA3 default 10 episodes restored")

    def refresh_obstacle_avoidance_3_project_combos(self) -> None:
        data = self.obstacle_avoidance_3_plan_data if isinstance(self.obstacle_avoidance_3_plan_data, dict) else make_default_plans()
        project_values = tuple(str(project.get("project_id", "")) for project in data.get("projects", []) if isinstance(project, dict))
        env_values = tuple(str(item.get("environment_id", "")) for item in data.get("environments", []) if isinstance(item, dict))
        method_values = tuple(str(item.get("method_id", "")) for item in data.get("methods", []) if isinstance(item, dict))
        for combo, values in (
            (self.obstacle_avoidance_3_project_combo, project_values),
            (self.obstacle_avoidance_3_environment_combo, env_values),
            (self.obstacle_avoidance_3_method_combo, method_values),
        ):
            if combo is not None:
                try:
                    combo.configure(values=values)
                except tk.TclError:
                    pass

    def current_obstacle_avoidance_3_project(self) -> Dict[str, Any]:
        if self.obstacle_avoidance_3_plan_data is None:
            self.obstacle_avoidance_3_plan_data = load_plans(self.obstacle_avoidance_3_plan_path())
        data = self.obstacle_avoidance_3_plan_data if isinstance(self.obstacle_avoidance_3_plan_data, dict) else make_default_plans()
        wanted = str(self.obstacle_avoidance_3_project_var.get() or data.get("active_project_id", DEFAULT_PROJECT_ID)).strip()
        projects = data.setdefault("projects", [])
        for project in projects:
            if isinstance(project, dict) and str(project.get("project_id", "")) == wanted:
                return project
        default = make_default_plans()["projects"][0]
        projects.append(default)
        data["active_project_id"] = default["project_id"]
        self.obstacle_avoidance_3_project_var.set(default["project_id"])
        return default

    def ensure_obstacle_avoidance_3_project_points(self) -> int:
        project = self.current_obstacle_avoidance_3_project()
        episodes = project.get("episodes")
        if isinstance(episodes, list) and episodes:
            return len(episodes)
        fallback = make_default_plans()["projects"][0]
        project.setdefault("project_id", fallback["project_id"])
        project.setdefault("name", fallback["name"])
        project.setdefault("environment_id", fallback["environment_id"])
        project["default_method"] = DEFAULT_METHOD_ID
        project["episodes"] = deepcopy(fallback["episodes"])
        project.setdefault("experiment_defaults", deepcopy(fallback["experiment_defaults"]))
        if isinstance(self.obstacle_avoidance_3_plan_data, dict):
            self.obstacle_avoidance_3_plan_data["active_project_id"] = str(project.get("project_id", DEFAULT_PROJECT_ID))
            try:
                save_plans(self.obstacle_avoidance_3_plan_path(), self.obstacle_avoidance_3_plan_data)
            except Exception:
                pass
        return len(project["episodes"])

    def select_obstacle_avoidance_3_project(self) -> None:
        project = self.current_obstacle_avoidance_3_project()
        self.ensure_obstacle_avoidance_3_project_points()
        if isinstance(self.obstacle_avoidance_3_plan_data, dict):
            self.obstacle_avoidance_3_plan_data["active_project_id"] = project.get("project_id", DEFAULT_PROJECT_ID)
        self.obstacle_avoidance_3_project_name_var.set(str(project.get("name", "")))
        self.obstacle_avoidance_3_environment_var.set(str(project.get("environment_id", DEFAULT_ENVIRONMENT_ID)))
        self.obstacle_avoidance_3_method_var.set(str(project.get("default_method", DEFAULT_METHOD_ID)))
        self.refresh_obstacle_avoidance_3_tree()

    def apply_obstacle_avoidance_3_project(self, *, silent: bool = False) -> None:
        project = self.current_obstacle_avoidance_3_project()
        project["name"] = str(self.obstacle_avoidance_3_project_name_var.get() or "").strip() or str(project.get("name", "OA3 project"))
        project["environment_id"] = str(self.obstacle_avoidance_3_environment_var.get() or DEFAULT_ENVIRONMENT_ID).strip()
        project["default_method"] = str(self.obstacle_avoidance_3_method_var.get() or DEFAULT_METHOD_ID).strip()
        if isinstance(self.obstacle_avoidance_3_plan_data, dict):
            self.obstacle_avoidance_3_plan_data["active_project_id"] = str(project.get("project_id", DEFAULT_PROJECT_ID))
            save_plans(self.obstacle_avoidance_3_plan_path(), self.obstacle_avoidance_3_plan_data)
        self.refresh_obstacle_avoidance_3_tree()
        if not silent:
            self.obstacle_avoidance_3_status_var.set("OA3 project applied")

    def obstacle_avoidance_3_pose_text(self, pose: Any) -> str:
        try:
            return ", ".join(f"{float(value):g}" for value in coerce_pose(pose))
        except Exception:
            return str(pose)

    def refresh_obstacle_avoidance_3_tree(self) -> None:
        tree = self.obstacle_avoidance_3_episode_tree
        if tree is None:
            return
        project = self.current_obstacle_avoidance_3_project()
        wanted_id = str(self.obstacle_avoidance_3_episode_id_var.get() or "").strip()
        for item in tree.get_children():
            tree.delete(item)
        self.obstacle_avoidance_3_tree_iids = {}
        for episode in project.get("episodes", []):
            if not isinstance(episode, dict):
                continue
            episode_id = str(episode.get("episode_id", ""))
            iid = tree.insert(
                "",
                "end",
                values=(
                    "yes" if episode.get("enabled", True) else "no",
                    episode_id,
                    self.obstacle_avoidance_3_pose_text(episode.get("start_pose", "")),
                    self.obstacle_avoidance_3_pose_text(episode.get("goal_pose", "")),
                    str(episode.get("method", "")),
                    str(episode.get("obstacle_hint", "")),
                    str(episode.get("operator_note", "")),
                ),
            )
            self.obstacle_avoidance_3_tree_iids[episode_id] = iid
        if project.get("episodes"):
            selected = self.find_obstacle_avoidance_3_episode(wanted_id) if wanted_id else None
            if selected is None:
                selected = next((item for item in project.get("episodes", []) if isinstance(item, dict)), None)
            if selected is not None:
                iid = self.obstacle_avoidance_3_tree_iids.get(str(selected.get("episode_id", "")))
                if iid:
                    tree.selection_set(iid)
                    self.load_obstacle_avoidance_3_episode_into_editor(selected)

    def selected_obstacle_avoidance_3_episode_ids(self) -> List[str]:
        tree = self.obstacle_avoidance_3_episode_tree
        if tree is None:
            return []
        ids: List[str] = []
        for iid in tree.selection():
            values = tree.item(iid, "values")
            if len(values) > 1:
                ids.append(str(values[1]))
        return ids

    def find_obstacle_avoidance_3_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        project = self.current_obstacle_avoidance_3_project()
        for episode in project.get("episodes", []):
            if isinstance(episode, dict) and str(episode.get("episode_id", "")) == str(episode_id):
                return episode
        return None

    def select_obstacle_avoidance_3_episode_from_tree(self) -> None:
        ids = self.selected_obstacle_avoidance_3_episode_ids()
        if not ids:
            return
        episode = self.find_obstacle_avoidance_3_episode(ids[0])
        if episode is not None:
            self.load_obstacle_avoidance_3_episode_into_editor(episode)
            self.refresh_obstacle_avoidance_3_records(str(episode.get("episode_id", "")))

    def load_obstacle_avoidance_3_episode_into_editor(self, episode: Dict[str, Any]) -> None:
        self.obstacle_avoidance_3_enabled_var.set(bool(episode.get("enabled", True)))
        self.obstacle_avoidance_3_episode_id_var.set(str(episode.get("episode_id", "")))
        self.obstacle_avoidance_3_start_pose_var.set(self.obstacle_avoidance_3_pose_text(episode.get("start_pose", "")))
        self.obstacle_avoidance_3_goal_pose_var.set(self.obstacle_avoidance_3_pose_text(episode.get("goal_pose", "")))
        self.obstacle_avoidance_3_scenario_var.set(str(episode.get("scenario_id", "")))
        self.obstacle_avoidance_3_environment_var.set(str(episode.get("environment_id", DEFAULT_ENVIRONMENT_ID)))
        self.obstacle_avoidance_3_method_var.set(str(episode.get("method", DEFAULT_METHOD_ID)))
        self.obstacle_avoidance_3_obstacle_hint_var.set(str(episode.get("obstacle_hint", "unknown")))
        self.obstacle_avoidance_3_note_var.set(str(episode.get("operator_note", "")))

    def next_obstacle_avoidance_3_episode_id(self) -> str:
        project = self.current_obstacle_avoidance_3_project()
        used = {str(item.get("episode_id", "")) for item in project.get("episodes", []) if isinstance(item, dict)}
        for index in range(1, 1000):
            candidate = f"E{index:02d}"
            if candidate not in used:
                return candidate
        return f"E{len(used) + 1:03d}"

    def build_obstacle_avoidance_3_editor_episode(self) -> Optional[Dict[str, Any]]:
        if not str(self.obstacle_avoidance_3_start_pose_var.get() or "").strip() or not str(
            self.obstacle_avoidance_3_goal_pose_var.get() or ""
        ).strip():
            self.ensure_obstacle_avoidance_3_project_points()
            project = self.current_obstacle_avoidance_3_project()
            first_episode = next((item for item in project.get("episodes", []) if isinstance(item, dict)), None)
            if first_episode is not None:
                self.load_obstacle_avoidance_3_episode_into_editor(first_episode)
        episode_id = str(self.obstacle_avoidance_3_episode_id_var.get() or "").strip() or self.next_obstacle_avoidance_3_episode_id()
        self.obstacle_avoidance_3_episode_id_var.set(episode_id)
        try:
            start_pose = coerce_pose(self.obstacle_avoidance_3_start_pose_var.get())
            goal_pose = coerce_pose(self.obstacle_avoidance_3_goal_pose_var.get())
        except Exception as exc:
            self.obstacle_avoidance_3_status_var.set(f"OA3 invalid pose: {exc}")
            return None
        project = self.current_obstacle_avoidance_3_project()
        method = str(self.obstacle_avoidance_3_method_var.get() or DEFAULT_METHOD_ID).strip()
        return {
            "episode_id": episode_id,
            "enabled": bool(self.obstacle_avoidance_3_enabled_var.get()),
            "start_pose": start_pose,
            "goal_pose": goal_pose,
            "scenario_id": str(self.obstacle_avoidance_3_scenario_var.get() or f"oa3_route_{episode_id}").strip(),
            "environment_id": str(self.obstacle_avoidance_3_environment_var.get() or project.get("environment_id", DEFAULT_ENVIRONMENT_ID)).strip(),
            "method": method,
            "obstacle_hint": str(self.obstacle_avoidance_3_obstacle_hint_var.get() or "unknown").strip(),
            "operator_note": str(self.obstacle_avoidance_3_note_var.get() or "").strip(),
        }

    def apply_obstacle_avoidance_3_episode(self, *, silent: bool = False) -> Optional[Dict[str, Any]]:
        episode = self.build_obstacle_avoidance_3_editor_episode()
        if episode is None:
            return None
        project = self.current_obstacle_avoidance_3_project()
        existing = self.find_obstacle_avoidance_3_episode(str(episode.get("episode_id", "")))
        if existing is None:
            project.setdefault("episodes", []).append(dict(episode))
        else:
            existing.clear()
            existing.update(episode)
        if isinstance(self.obstacle_avoidance_3_plan_data, dict):
            save_plans(self.obstacle_avoidance_3_plan_path(), self.obstacle_avoidance_3_plan_data)
        self.refresh_obstacle_avoidance_3_tree()
        self.refresh_obstacle_avoidance_3_records(str(episode.get("episode_id", "")))
        if not silent:
            self.obstacle_avoidance_3_status_var.set(f"OA3 episode applied: {episode.get('episode_id', '')}")
        return episode

    def add_obstacle_avoidance_3_episode(self) -> None:
        self.obstacle_avoidance_3_episode_id_var.set(self.next_obstacle_avoidance_3_episode_id())
        self.apply_obstacle_avoidance_3_episode()

    def delete_obstacle_avoidance_3_episode(self) -> None:
        episode_id = str(self.obstacle_avoidance_3_episode_id_var.get() or "").strip()
        if not episode_id:
            return
        project = self.current_obstacle_avoidance_3_project()
        project["episodes"] = [item for item in project.get("episodes", []) if str(item.get("episode_id", "")) != episode_id]
        if isinstance(self.obstacle_avoidance_3_plan_data, dict):
            save_plans(self.obstacle_avoidance_3_plan_path(), self.obstacle_avoidance_3_plan_data)
        self.refresh_obstacle_avoidance_3_tree()
        self.obstacle_avoidance_3_status_var.set(f"OA3 episode deleted: {episode_id}")

    def scan_obstacle_avoidance_3_records(self, episode_id: str) -> List[Dict[str, Any]]:
        wanted = str(episode_id or "").strip()
        sessions_dir = self.obstacle_avoidance_3_data_root() / "sessions"
        if not wanted or not sessions_dir.exists():
            return []
        records: List[Dict[str, Any]] = []
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            summary = self.read_obstacle_avoidance_3_json(session_dir / "episode_summary.json")
            config = self.read_obstacle_avoidance_3_json(session_dir / "episode_config.json")
            found = str(summary.get("episode_id", "") or (config.get("episode", {}) if isinstance(config.get("episode"), dict) else {}).get("episode_id", ""))
            if not found:
                match = re.search(r"_(E[0-9A-Za-z-]+)_", session_dir.name)
                found = match.group(1) if match else ""
            if found != wanted:
                continue
            try:
                mtime = session_dir.stat().st_mtime
            except OSError:
                mtime = 0.0
            records.append(
                {
                    "episode_id": found,
                    "session_dir": str(session_dir),
                    "summary_path": str(session_dir / "episode_summary.json"),
                    "events_path": str(session_dir / "avoidance_events.jsonl"),
                    "outcome": str(summary.get("outcome", "unknown")),
                    "frame_count": summary.get("frame_count", 0),
                    "final_distance_to_goal_cm": summary.get("final_distance_to_goal_cm", ""),
                    "summary": summary,
                    "mtime": mtime,
                }
            )
        records.sort(key=lambda item: float(item.get("mtime", 0.0) or 0.0), reverse=True)
        for index, record in enumerate(records, start=1):
            finished = str(record.get("summary", {}).get("finished_at", "")).replace("T", " ")[:19]
            distance = record.get("final_distance_to_goal_cm", "")
            distance_text = f"{float(distance):.1f}cm" if isinstance(distance, (int, float)) else str(distance or "?")
            record["display"] = f"{index:02d} {finished} | {record.get('outcome', 'unknown')} | d={distance_text}"
        return records

    def read_obstacle_avoidance_3_json(self, path: Path) -> Dict[str, Any]:
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
        return {}

    def refresh_obstacle_avoidance_3_records(self, episode_id: Optional[str] = None) -> None:
        episode_id = str(episode_id or self.obstacle_avoidance_3_episode_id_var.get() or "").strip()
        records = self.scan_obstacle_avoidance_3_records(episode_id)
        self.obstacle_avoidance_3_record_entries = records
        values = tuple(str(record.get("display", "")) for record in records)
        if self.obstacle_avoidance_3_record_combo is not None:
            try:
                self.obstacle_avoidance_3_record_combo.configure(values=values)
            except tk.TclError:
                pass
        if not episode_id:
            self.obstacle_avoidance_3_record_var.set("")
            self.obstacle_avoidance_3_record_summary_var.set("Select an episode to inspect records.")
            return
        if not records:
            self.obstacle_avoidance_3_record_var.set("")
            self.obstacle_avoidance_3_record_summary_var.set(f"{episode_id}: no OA3 run records yet.")
            return
        current = str(self.obstacle_avoidance_3_record_var.get() or "")
        if current not in values:
            self.obstacle_avoidance_3_record_var.set(values[0])
        self.update_obstacle_avoidance_3_record_summary()

    def selected_obstacle_avoidance_3_record(self) -> Optional[Dict[str, Any]]:
        selected = str(self.obstacle_avoidance_3_record_var.get() or "")
        for record in self.obstacle_avoidance_3_record_entries:
            if str(record.get("display", "")) == selected:
                return record
        return self.obstacle_avoidance_3_record_entries[0] if self.obstacle_avoidance_3_record_entries else None

    def update_obstacle_avoidance_3_record_summary(self) -> None:
        record = self.selected_obstacle_avoidance_3_record()
        if record is None:
            return
        summary = record.get("summary", {}) if isinstance(record.get("summary"), dict) else {}
        self.obstacle_avoidance_3_record_summary_var.set(
            "\n".join(
                [
                    f"Episode: {record.get('episode_id', '')}",
                    f"Outcome: {record.get('outcome', '')}",
                    f"Frames: {record.get('frame_count', 0)}",
                    f"Final distance: {record.get('final_distance_to_goal_cm', '?')} cm",
                    f"Goal reached: {summary.get('goal_reached', False)}",
                    f"Collision: {summary.get('had_collision', False)}",
                    f"Session: {Path(str(record.get('session_dir', ''))).name}",
                ]
            )
        )

    def view_obstacle_avoidance_3_selected_record(self) -> None:
        self.update_obstacle_avoidance_3_record_summary()
        record = self.selected_obstacle_avoidance_3_record()
        if record is None:
            self.obstacle_avoidance_3_set_report({"status": "no_record", "episode_id": self.obstacle_avoidance_3_episode_id_var.get()})
            return
        self.obstacle_avoidance_3_set_report(
            {
                "status": "record_selected",
                "episode_id": record.get("episode_id", ""),
                "session_dir": record.get("session_dir", ""),
                "events_path": record.get("events_path", ""),
                "summary": record.get("summary", {}),
            }
        )

    def locate_obstacle_avoidance_3_start(self) -> None:
        if self.obstacle_avoidance_3_runner_thread is not None and self.obstacle_avoidance_3_runner_thread.is_alive():
            self.obstacle_avoidance_3_status_var.set("OA3 runner is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_3_status_var.set("OA3: start Unreal from the main window first")
            return
        episode = self.apply_obstacle_avoidance_3_episode(silent=True)
        if episode is None:
            return
        start = pose_dict(episode["start_pose"])
        goal = pose_dict(episode["goal_pose"])
        try:
            response = session.set_pose({"x": start["x"], "y": start["y"], "z": start["z"], "yaw": start["yaw"]})
            mode_result = session.set_movement_mode("physics")
            enable_result = session.set_movement_enabled(True)
        except Exception as exc:
            self.obstacle_avoidance_3_status_var.set(f"OA3 locate start failed: {exc}")
            self.obstacle_avoidance_3_report({"status": "error", "task": "locate_start", "error": str(exc)})
            return
        state_payload = response if isinstance(response, dict) else enable_result if isinstance(enable_result, dict) else mode_result
        if isinstance(state_payload, dict):
            try:
                self.apply_state(state_payload)
            except Exception:
                pass
        self.movement_enabled_state = True
        self.movement_mode_state = "physics"
        self.movement_mode_var.set("physics")
        start_distance = distance_3d_cm(start, goal)
        self.obstacle_avoidance_3_goal_distance_var.set(f"Goal distance: {start_distance:.1f} cm")
        self.obstacle_avoidance_3_goal_reached_var.set("Goal reached: no")
        self.obstacle_avoidance_3_impact_var.set("Impact: no")
        self.obstacle_avoidance_3_or2_risk_var.set("OR2 risk: --")
        self.obstacle_avoidance_3_status_var.set(f"OA3 located start: {episode['episode_id']} dist={start_distance:.1f}cm")
        self.obstacle_avoidance_3_report(
            {
                "status": "located_start",
                "episode_id": episode["episode_id"],
                "start_pose": start,
                "goal_pose": goal,
                "distance_to_goal_cm": round(start_distance, 3),
                "movement_mode_result": mode_result,
                "movement_enabled_result": enable_result,
                "set_pose_result": response,
            }
        )

    def prepare_obstacle_avoidance_3_algorithm(
        self,
        *,
        episode: Optional[Dict[str, Any]] = None,
        silent: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if episode is None:
            episode = self.apply_obstacle_avoidance_3_episode(silent=True)
        if episode is None:
            return None
        method = str(self.obstacle_avoidance_3_method_var.get() or DEFAULT_METHOD_ID).strip()
        self.obstacle_avoidance_3_method_var.set(method)
        model_path = self.obstacle_avoidance_3_model_path()
        if method != DEFAULT_METHOD_ID:
            self.obstacle_avoidance_3_status_var.set(f"OA3 method not executable: {method}")
            self.obstacle_avoidance_3_report({"status": "blocked", "reason": "method_not_executable", "method": method})
            return None
        if not model_path.is_file():
            self.obstacle_avoidance_3_status_var.set(f"OA3 model not found: {model_path}")
            self.obstacle_avoidance_3_report({"status": "blocked", "reason": "model_not_found", "model_path": str(model_path)})
            return None
        episode["method"] = DEFAULT_METHOD_ID
        if not silent:
            session = self.active_session()
            hint = "preview capture starting" if session is not None else "ready; start Unreal to execute"
            self.obstacle_avoidance_3_status_var.set(f"OA3 algorithm ready: {method} ({hint})")
            self.obstacle_avoidance_3_report(
                {
                    "status": "algorithm_ready",
                    "episode_id": episode.get("episode_id", ""),
                    "method": method,
                    "model_path": str(model_path),
                    "model_exists": True,
                    "execute_to_goal": "will automatically run OR2 avoidance every tick",
                }
            )
        return {"episode": episode, "method": method, "model_path": model_path}

    def apply_obstacle_avoidance_3_algorithm(self) -> None:
        prepared = self.prepare_obstacle_avoidance_3_algorithm(silent=False)
        if prepared is None:
            return
        if self.active_session() is not None:
            self.capture_obstacle_avoidance_3_or2_once()

    def execute_obstacle_avoidance_3_to_goal(self) -> None:
        prepared = self.prepare_obstacle_avoidance_3_algorithm(silent=False)
        if prepared is None:
            return
        episode = prepared["episode"]
        self.obstacle_avoidance_3_status_var.set("OA3 Execute To Goal: OR2 avoidance active")
        self.start_obstacle_avoidance_3_runner([self.resolved_obstacle_avoidance_3_episode(episode)], selected_label=str(episode["episode_id"]))

    def resolved_obstacle_avoidance_3_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "episode_id": str(episode["episode_id"]),
            "start_pose": pose_dict(episode["start_pose"]),
            "goal_pose": pose_dict(episode["goal_pose"]),
            "scenario_id": episode.get("scenario_id", ""),
            "environment_id": episode.get("environment_id", ""),
            "method": episode.get("method", DEFAULT_METHOD_ID),
            "obstacle_hint": episode.get("obstacle_hint", ""),
            "operator_note": episode.get("operator_note", ""),
            "enabled": episode.get("enabled", True),
        }

    def run_obstacle_avoidance_3_current_episode(self) -> None:
        prepared = self.prepare_obstacle_avoidance_3_algorithm(silent=True)
        if prepared is None:
            return
        episode = prepared["episode"]
        self.start_obstacle_avoidance_3_runner([self.resolved_obstacle_avoidance_3_episode(episode)], selected_label=str(episode["episode_id"]))

    def run_obstacle_avoidance_3_selected(self) -> None:
        if self.obstacle_avoidance_3_plan_data is None:
            self.load_obstacle_avoidance_3_plans()
        self.apply_obstacle_avoidance_3_project(silent=True)
        self.apply_obstacle_avoidance_3_episode(silent=True)
        project = self.current_obstacle_avoidance_3_project()
        selected_ids = set(self.selected_obstacle_avoidance_3_episode_ids())
        rows = []
        for episode in project.get("episodes", []):
            if not isinstance(episode, dict):
                continue
            if selected_ids and str(episode.get("episode_id", "")) not in selected_ids:
                continue
            if not selected_ids and not bool(episode.get("enabled", True)):
                continue
            rows.append(self.resolved_obstacle_avoidance_3_episode(episode))
        self.start_obstacle_avoidance_3_runner(rows, selected_label="selected" if selected_ids else "enabled")

    def start_obstacle_avoidance_3_runner(self, episodes: List[Dict[str, Any]], *, selected_label: str) -> None:
        if self.obstacle_avoidance_3_runner_thread is not None and self.obstacle_avoidance_3_runner_thread.is_alive():
            self.obstacle_avoidance_3_status_var.set("OA3 runner is already active")
            return
        monitor_thread = getattr(self, "obstacle_avoidance_3_or2_monitor_thread", None)
        if monitor_thread is not None and monitor_thread.is_alive():
            self.obstacle_avoidance_3_or2_monitor_stop_event.set()
            monitor_thread.join(timeout=1.0)
        if not episodes:
            self.obstacle_avoidance_3_status_var.set("OA3: no episodes selected")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_3_status_var.set("OA3: start Unreal from the main window first")
            return
        method = str(self.obstacle_avoidance_3_method_var.get() or DEFAULT_METHOD_ID).strip()
        if method != DEFAULT_METHOD_ID:
            self.obstacle_avoidance_3_status_var.set(f"OA3 method not executable: {method}")
            return
        model_path = self.obstacle_avoidance_3_model_path()
        if not model_path.is_file():
            self.obstacle_avoidance_3_status_var.set(f"OA3 model not found: {model_path}")
            self.obstacle_avoidance_3_report({"status": "blocked", "reason": "model_not_found", "model_path": str(model_path)})
            return
        project = self.current_obstacle_avoidance_3_project()
        run_id = sanitize_id(f"{selected_label}_{datetime.now().strftime('%H%M%S')}", "oa3_run", max_len=32)
        note = (
            f"oa3_project={project.get('project_id', '')}; "
            f"environment={self.obstacle_avoidance_3_environment_var.get()}; selected={selected_label}"
        )
        self.obstacle_avoidance_3_stop_event = threading.Event()
        self.obstacle_avoidance_3_status_var.set(f"OA3 runner starting with OR2 avoidance: {selected_label}")
        self.obstacle_avoidance_3_report(
            {
                "status": "starting",
                "session_source": "main_window_start_unreal",
                "episode_count": len(episodes),
                "method": method,
                "model_path": str(model_path),
                "run_id": run_id,
            }
        )

        def worker() -> None:
            try:
                result = self.run_obstacle_avoidance_3_collection_on_session(
                    session,
                    episodes,
                    method=method,
                    run_id=run_id,
                    note=note,
                    project_id=str(project.get("project_id", "")),
                    model_path=model_path,
                )
                self.root.after(
                    0,
                    lambda r=result: (
                        self.obstacle_avoidance_3_status_var.set(
                            f"OA3 done: reached {r.get('reached_count', 0)}/{r.get('episode_count', 0)}, collision {r.get('collision_count', 0)}"
                        ),
                        self.refresh_obstacle_avoidance_3_records(),
                        self.obstacle_avoidance_3_report(r),
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_avoidance_3_status_var.set(f"OA3 runner failed: {e}"),
                        self.obstacle_avoidance_3_report({"status": "error", "task": "oa3_collection", "error": str(e)}),
                    ),
                )

        self.obstacle_avoidance_3_runner_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_3_runner_thread.start()

    def obstacle_avoidance_3_args(self, *, method: str, run_id: str, note: str) -> argparse.Namespace:
        return argparse.Namespace(
            stage="route_episode_3",
            method=method,
            run_id=run_id,
            note=note,
            geometry_label="unknown",
            reach_tol_cm=180.0,
            max_ticks_per_episode=220,
            route_step_cm=DEFAULT_ROUTE_STEP_CM,
            side_correction_cm=DEFAULT_ROUTE_SIDE_CORRECTION_CM,
            vertical_step_cm=DEFAULT_ROUTE_VERTICAL_STEP_CM,
            interval_s=0.5,
            continue_on_failure=True,
            movement_mode="physics",
            lidar_capture_processing="minimal",
        )

    def run_obstacle_avoidance_3_collection_on_session(
        self,
        session: flight.DroneFlightSession,
        episodes: List[Dict[str, Any]],
        *,
        method: str,
        run_id: str,
        note: str,
        project_id: str,
        model_path: Path,
    ) -> Dict[str, Any]:
        args = self.obstacle_avoidance_3_args(method=method, run_id=run_id, note=note)
        stop_event = self.obstacle_avoidance_3_stop_event
        predictor = ObstacleRepresentation2Predictor(model_path)
        previous_lidar_processing = str(
            getattr(getattr(session, "args", object()), "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING)
            or flight.DEFAULT_LIDAR_CAPTURE_PROCESSING
        )
        try:
            session.args.lidar_capture_processing = "minimal"
        except Exception:
            pass
        if getattr(self, "lidar_capture_processing_var", None) is not None:
            self.root.after(0, lambda: self.lidar_capture_processing_var.set("minimal"))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        data_root = self.obstacle_avoidance_3_data_root()
        method_path_id = "or2dir_v1" if method == DEFAULT_METHOD_ID else sanitize_id(method, "method", max_len=24)
        batch_dir = data_root / "route_episode_batches" / f"{timestamp}_oa3_{method_path_id}_{run_id}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            batch_dir / "route_episode_batch_config.json",
            {
                "source": "obstacle_avoidance_3_gui_active_session",
                "project_id": project_id,
                "args": vars(args),
                "model_path": str(model_path),
                "method_path_id": method_path_id,
                "previous_lidar_capture_processing": previous_lidar_processing,
                "forced_lidar_capture_processing": "minimal",
                "episodes": episodes,
                "started_at": datetime.now().isoformat(timespec="milliseconds"),
            },
        )
        mode_result = session.set_movement_mode("physics")
        enable_result = session.set_movement_enabled(True)
        self.root.after(
            0,
            lambda: (
                self.apply_state(enable_result if isinstance(enable_result, dict) else mode_result)
                if isinstance(enable_result if isinstance(enable_result, dict) else mode_result, dict)
                else None,
                setattr(self, "movement_enabled_state", True),
                setattr(self, "movement_mode_state", "physics"),
                self.movement_mode_var.set("physics"),
                self.obstacle_avoidance_3_report(
                    {
                        "status": "movement_physics_enabled",
                        "lidar_capture_processing": "minimal",
                        "movement_mode_result": mode_result,
                        "movement_enabled_result": enable_result,
                    }
                ),
            ),
        )
        batch_summaries: List[Dict[str, Any]] = []

        for episode_index, episode in enumerate(episodes, start=1):
            if stop_event.is_set():
                break
            episode_id = str(episode["episode_id"])
            start = dict(episode["start_pose"])
            goal = dict(episode["goal_pose"])
            session_dir = data_root / "sessions" / f"{timestamp}_oa3_{episode_id}_{method_path_id}_{run_id}"
            session_dir.mkdir(parents=True, exist_ok=False)
            events_path = session_dir / "avoidance_events.jsonl"
            write_json(
                session_dir / "episode_config.json",
                {
                    "source": "obstacle_avoidance_3_gui_active_session",
                    "project_id": project_id,
                    "episode_index": episode_index,
                    "episode": episode,
                    "args": vars(args),
                    "model_path": str(model_path),
                },
            )
            self.root.after(0, lambda eid=episode_id, s=start, g=goal: self.obstacle_avoidance_3_report(f"OA3_EPISODE_START {eid} start={s} goal={g}"))
            session.set_pose({"x": start["x"], "y": start["y"], "z": start["z"], "yaw": start["yaw"]})
            last_action = action_payload("hold")
            red_recovery = {"active": False, "direction": "", "ticks": 0, "clear_ticks": 0}
            events: List[Dict[str, Any]] = []
            outcome = "timeout"
            episode_error = ""

            for frame_id in range(1, int(args.max_ticks_per_episode) + 1):
                if stop_event.is_set():
                    outcome = "stopped"
                    break
                try:
                    pre_state = session.get_state()
                except Exception as exc:
                    outcome = "unrealcv_error"
                    episode_error = f"get_state failed at frame {frame_id}: {exc}"
                    self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_3_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                    break
                pre_pose = pose_from_state(pre_state if isinstance(pre_state, dict) else {})
                pre_distance = distance_3d_cm(pre_pose, goal)
                action_detail = {
                    "source": "obstacle_avoidance_3_gui_active_session",
                    "project_id": project_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "collection_stage": args.stage,
                    "scenario_id": episode.get("scenario_id") or f"oa3_route_{episode_id}",
                    "environment_id": episode.get("environment_id", ""),
                    "method": args.method,
                    "plan_method": episode.get("method", ""),
                    "obstacle_hint": episode.get("obstacle_hint", ""),
                    "run_id": args.run_id,
                    "mission_phase": "ROUTE_EPISODE_3",
                    "risk_state": "SAFE",
                    "expert_action": str(last_action.get("action_name", "hold")),
                    "expert_action_payload": last_action,
                    "start_pose": start,
                    "goal_pose": goal,
                    "target_waypoint": goal,
                    "operator_note": episode.get("operator_note") or args.note,
                }
                try:
                    result = session.capture_lidar_stream_frame(session_dir, frame_id, action_detail=action_detail)
                except Exception as exc:
                    outcome = "unrealcv_error"
                    episode_error = f"capture_lidar_stream_frame failed at frame {frame_id}: {exc}"
                    self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_3_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                    break
                if not isinstance(result, dict):
                    outcome = "unrealcv_error"
                    episode_error = f"capture_lidar_stream_frame returned non-dict at frame {frame_id}"
                    break
                event = build_route_event(
                    result,
                    session_dir=session_dir,
                    frame_id=frame_id,
                    args=args,
                    episode=episode,
                    episode_index=episode_index,
                    start=start,
                    goal=goal,
                    last_action=last_action,
                )
                event["source"] = "obstacle_avoidance_3_gui_active_session"
                event["project_id"] = project_id
                rel = event.get("relative_target") if isinstance(event.get("relative_target"), dict) else {}
                prediction_start = time.perf_counter()
                rgb_array = None
                overlay_array = None
                try:
                    prediction = predictor.predict(str(event.get("rgb_path", "")), event)
                    latency_ms = (time.perf_counter() - prediction_start) * 1000.0
                    capture_dir = Path(str(event.get("capture_dir") or result.get("capture_dir") or session_dir))
                    prediction_json_path = capture_dir / "or2_risk_prediction.json"
                    overlay_path = capture_dir / "or2_risk_overlay.png"
                    try:
                        rgb_array = np.asarray(Image.open(str(event.get("rgb_path", ""))).convert("RGB"), dtype=np.uint8)
                        overlay_array = render_affordance_overlay(rgb_array, prediction)
                        Image.fromarray(overlay_array).save(overlay_path)
                    except Exception as exc:
                        overlay_path = None
                        prediction["overlay_error"] = str(exc)
                    rule = select_or2_direction(prediction, event["pointcloud_summary"], rel, last_action=last_action)
                    write_json(
                        prediction_json_path,
                        {
                            **serializable_or2_prediction(prediction),
                            "prediction_latency_ms": round(latency_ms, 3),
                            "risk_overlay_path": str(overlay_path) if overlay_path else "",
                            "or2_selected_direction": rule.get("selected_direction", ""),
                            "or2_corridor_risks": rule.get("corridor_risks", {}),
                            "or2_candidate_action_scores": rule.get("candidate_action_scores", {}),
                        },
                    )
                except Exception as exc:
                    latency_ms = (time.perf_counter() - prediction_start) * 1000.0
                    prediction = {
                        "front_risk_state": "predict_error",
                        "can_forward": False,
                        "must_stop": False,
                        "reason": str(exc),
                    }
                    prediction_json_path = None
                    overlay_path = None
                    rule = {
                        "selected_direction": "hold",
                        "candidate_action_scores": {"hold": 1.0},
                        "corridor_risks": {},
                        "front_blocked": False,
                        "reason": f"or2 prediction failed: {exc}",
                    }
                selected_direction = str(rule.get("selected_direction", "hold"))
                recovery_reason = ""
                red_now = deep_red_stop_active(prediction, event["pointcloud_summary"], rule.get("corridor_risks", {}))
                if red_now and not bool(red_recovery.get("active")):
                    red_recovery = {
                        "active": True,
                        "direction": choose_red_recovery_direction(rule),
                        "ticks": 0,
                        "clear_ticks": 0,
                    }
                if bool(red_recovery.get("active")):
                    red_recovery["ticks"] = int(red_recovery.get("ticks", 0) or 0) + 1
                    if (not red_now) and red_recovery_clear_enough(prediction, event["pointcloud_summary"], rule.get("corridor_risks", {})):
                        red_recovery["clear_ticks"] = int(red_recovery.get("clear_ticks", 0) or 0) + 1
                    else:
                        red_recovery["clear_ticks"] = 0
                    can_release = (
                        int(red_recovery.get("ticks", 0) or 0) >= RED_RECOVERY_MIN_TICKS
                        and int(red_recovery.get("clear_ticks", 0) or 0) >= RED_RECOVERY_CLEAR_TICKS
                    )
                    if can_release:
                        recovery_reason = (
                            f"red_recovery released after ticks={red_recovery.get('ticks')} "
                            f"clear_ticks={red_recovery.get('clear_ticks')}"
                        )
                        red_recovery = {"active": False, "direction": "", "ticks": 0, "clear_ticks": 0}
                    else:
                        locked_direction = str(red_recovery.get("direction") or choose_red_recovery_direction(rule))
                        locked_region = {
                            "left": "left_corridor",
                            "right": "right_corridor",
                            "up": "up_corridor",
                        }.get(locked_direction)
                        corridor_risks = rule.get("corridor_risks", {}) if isinstance(rule.get("corridor_risks"), dict) else {}
                        locked_stats = corridor_risks.get(locked_region, {}) if locked_region else {}
                        if isinstance(locked_stats, dict) and float(locked_stats.get("stop_fraction", 0.0) or 0.0) >= SIDE_BLOCKED_FRACTION:
                            locked_direction = choose_red_recovery_direction(rule)
                            red_recovery["direction"] = locked_direction
                        if locked_direction in {"forward", "slow_forward", ""}:
                            locked_direction = "backoff"
                        selected_direction = locked_direction
                        recovery_reason = (
                            f"red_recovery active: lock={locked_direction}, ticks={red_recovery.get('ticks')}, "
                            f"clear_ticks={red_recovery.get('clear_ticks')}, red_now={red_now}; forward disabled until "
                            f"the deep-red obstacle leaves the front view"
                        )
                if pre_distance <= float(args.reach_tol_cm):
                    selected_action = "hold"
                    payload = action_payload("hold")
                    reason = f"goal reached before action; distance={pre_distance:.1f}cm"
                    phase = "REACHED"
                elif selected_direction in {"forward", "slow_forward"}:
                    step = float(args.route_step_cm) if selected_direction == "forward" else max(10.0, float(args.route_step_cm) * 0.35)
                    payload, _target, route_reason = straight_line_payload(
                        pre_pose,
                        start,
                        goal,
                        route_step_cm=step,
                        side_correction_cm=float(args.side_correction_cm),
                        vertical_step_cm=float(args.vertical_step_cm),
                    )
                    payload["action_name"] = selected_direction
                    selected_action = selected_direction
                    reason = f"{rule.get('reason', '')}; {route_reason}"
                    phase = "ROUTE_FOLLOW"
                else:
                    selected_action = selected_direction
                    payload = action_payload(selected_direction)
                    reason = str(rule.get("reason", "or2 direction rule"))
                    phase = "OR2_AVOIDANCE"
                if recovery_reason and phase != "REACHED":
                    if selected_action in {"left", "right"}:
                        payload["forward_cm"] = 0.0
                        payload["right_cm"] = -RED_RECOVERY_SIDE_STEP_CM if selected_action == "left" else RED_RECOVERY_SIDE_STEP_CM
                        payload["up_cm"] = 0.0
                        payload["action_name"] = "side_step_left" if selected_action == "left" else "side_step_right"
                    elif selected_action == "up":
                        payload["forward_cm"] = 0.0
                        payload["right_cm"] = 0.0
                        payload["up_cm"] = RED_RECOVERY_VERTICAL_CM
                        payload["action_name"] = "up"
                    elif selected_action == "backoff":
                        payload["forward_cm"] = -RED_RECOVERY_BACKOFF_CM
                        payload["right_cm"] = 0.0
                        payload["up_cm"] = 0.0
                        payload["action_name"] = "backoff"
                    reason = f"{reason}; {recovery_reason}"
                event.update(
                    {
                        "mission_phase": phase,
                        "risk_state": str(prediction.get("front_risk_state", "unknown")).upper(),
                        "selected_action": selected_action,
                        "selected_action_payload": payload,
                        "selected_action_reason": reason,
                        "expert_action": str(payload.get("action_name", selected_action)),
                        "expert_action_payload": payload,
                        "nominal_action": payload,
                        "agent_action": payload,
                        "executed_action": payload,
                        "shield_state": "GOAL_REACHED" if phase == "REACHED" else "OR2_SHIELD_APPLIED",
                        "episode_outcome": "running",
                        "or2_front_risk_state": prediction.get("front_risk_state", "unknown"),
                        "or2_can_forward": bool(prediction.get("can_forward", False)),
                        "or2_must_stop": bool(prediction.get("must_stop", False)),
                        "or2_corridor_risks": rule.get("corridor_risks", {}),
                        "or2_candidate_action_scores": rule.get("candidate_action_scores", {}),
                        "or2_selected_direction": selected_direction,
                        "or2_front_blocked": bool(rule.get("front_blocked", False)),
                        "or2_prediction_latency_ms": round(latency_ms, 3),
                        "or2_prediction_reason": prediction.get("reason", ""),
                        "or2_rule_reason": rule.get("reason", ""),
                        "or2_red_recovery_active": bool(red_recovery.get("active")),
                        "or2_red_recovery_direction": str(red_recovery.get("direction", "")),
                        "or2_red_recovery_ticks": int(red_recovery.get("ticks", 0) or 0),
                        "or2_red_recovery_clear_ticks": int(red_recovery.get("clear_ticks", 0) or 0),
                        "or2_red_recovery_reason": recovery_reason,
                        "or2_risk_overlay_path": str(overlay_path) if overlay_path else "",
                        "or2_prediction_json_path": str(prediction_json_path) if prediction_json_path else "",
                    }
                )
                or2_ui_result = {
                    "prediction": prediction,
                    "rule": rule,
                    "summary": event["pointcloud_summary"],
                    "event_record": {
                        "frame_index": frame_id,
                        "frame_dir": str(event.get("capture_dir", "")),
                        "risk_overlay_path": str(overlay_path) if overlay_path else "",
                        "prediction_json_path": str(prediction_json_path) if prediction_json_path else "",
                        "or2_selected_direction": selected_direction,
                        "or2_corridor_risks": rule.get("corridor_risks", {}),
                    },
                    "rgb_image": rgb_array,
                    "mask_image": overlay_array,
                    "frame_count": frame_id,
                }
                if phase == "REACHED":
                    post_pose = pre_pose
                    post_distance = pre_distance
                    hit_state = self.read_obstacle_avoidance_3_hit_state(session)
                else:
                    try:
                        response = session.move_relative(payload)
                    except Exception as exc:
                        outcome = "unrealcv_error"
                        episode_error = f"move_relative failed at frame {frame_id}: {exc}"
                        self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_3_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                        break
                    post_pose = pose_from_state(response if isinstance(response, dict) else session.get_state())
                    post_distance = distance_3d_cm(post_pose, goal)
                    hit_state = self.read_obstacle_avoidance_3_hit_state(session)
                post_progress, post_deviation = route_progress_and_deviation(post_pose, start, goal)
                post_reached, completion_reason, post_raw_progress, post_cross_track_cm = route_completion_state(
                    post_pose,
                    start,
                    goal,
                    reach_tol_cm=float(args.reach_tol_cm),
                )
                event["goal_completion_reason"] = completion_reason
                event["goal_passed"] = bool(post_raw_progress >= 1.0)
                event["goal_reached"] = bool(post_reached)
                event["goal_status"] = "reached" if post_reached else "not_reached"
                event["post_action_pose"] = post_pose
                event["post_distance_to_goal_cm"] = round(post_distance, 3)
                event["post_route_progress"] = round(post_progress, 5)
                event["post_raw_route_progress"] = round(post_raw_progress, 5)
                event["post_path_deviation_cm"] = round(post_deviation, 3)
                event["post_route_cross_track_cm"] = round(post_cross_track_cm, 3)
                annotate_collision_state(event, pre_pose=pre_pose, post_pose=post_pose, payload=payload, explicit_collision=hit_state)
                if bool(event.get("collision_state")):
                    outcome = "collision"
                    event["episode_outcome"] = "collision"
                    event["risk_state"] = "COLLISION"
                elif post_reached:
                    outcome = "reached"
                    event["episode_outcome"] = "reached"
                elif should_stop_for_repeated_hold([*events, event]):
                    outcome = "stalled_hold"
                    event["episode_outcome"] = "stalled_hold"
                    event["risk_state"] = "STALLED_HOLD"
                    event["selected_action_reason"] = f"{event.get('selected_action_reason', '')}; repeated hold detected, ending episode"
                append_jsonl(events_path, event)
                events.append(event)
                write_partial_episode_summary(session_dir, events, start=start, goal=goal, outcome=outcome, reach_tol_cm=float(args.reach_tol_cm))
                last_action = payload
                if bool(event.get("collision_state")):
                    detail = deepcopy(event.get("impact_detail", {})) if isinstance(event.get("impact_detail"), dict) else {}
                    self.root.after(
                        0,
                        lambda eid=episode_id, fid=frame_id, sdir=str(session_dir), d=detail: self.notify_obstacle_avoidance_3_collision(
                            episode_id=eid,
                            frame_id=fid,
                            session_dir=sdir,
                            impact_detail=d,
                        ),
                    )
                self.root.after(
                    0,
                    lambda eid=episode_id, fid=frame_id, dist=post_distance, reached=post_reached, completion=completion_reason, action=selected_action, risk=event.get("or2_front_risk_state", ""), hit=bool(event.get("collision_state")), reason=event.get("selected_action_reason", reason), ui=or2_ui_result: (
                        self.apply_obstacle_avoidance_3_or2_result(ui),
                        self.obstacle_avoidance_3_goal_distance_var.set(f"Goal distance: {dist:.1f} cm"),
                        self.obstacle_avoidance_3_goal_reached_var.set(f"Goal reached: {'YES' if reached else 'no'} ({completion})"),
                        self.obstacle_avoidance_3_impact_var.set("Impact: YES" if hit else "Impact: no"),
                        self.obstacle_avoidance_3_or2_risk_var.set(f"OR2 risk: {risk}"),
                        self.obstacle_avoidance_3_status_var.set(
                            f"OA3 {eid} frame {fid}: dist={dist:.1f}cm reached={reached} action={action} risk={risk} impact={hit}"
                        ),
                        self.obstacle_avoidance_3_report(
                            f"OA3_FRAME {eid} {fid}/{args.max_ticks_per_episode} dist={dist:.1f} reached={reached} completion={completion} action={action} risk={risk} impact={hit} reason={reason}"
                        ),
                    ),
                )
                if outcome in {"reached", "collision", "stalled_hold"}:
                    break
                if frame_id < int(args.max_ticks_per_episode):
                    if stop_event.wait(max(0.0, float(args.interval_s))):
                        outcome = "stopped"
                        break
            summary = summarize_episode(session_dir, events, start=start, goal=goal, outcome=outcome, reach_tol_cm=float(args.reach_tol_cm))
            if episode_error:
                summary["episode_error"] = episode_error
                summary["approach_failure_reason"] = episode_error
                write_json(session_dir / "episode_summary.json", summary)
                write_json(session_dir / "avoidance_session_summary.json", summary)
            batch_summaries.append(summary)
            self.write_obstacle_avoidance_3_batch_summary(batch_dir, episodes, batch_summaries)
            self.root.after(
                0,
                lambda eid=episode_id, o=outcome, s=summary: (
                    self.obstacle_avoidance_3_goal_distance_var.set(f"Goal distance: {float(s.get('final_distance_to_goal_cm', 0.0)):.1f} cm"),
                    self.obstacle_avoidance_3_goal_reached_var.set(
                        f"Goal reached: {'YES' if s.get('goal_reached') else 'no'} ({s.get('goal_completion_reason', '')})"
                    ),
                    self.obstacle_avoidance_3_report({"episode_done": eid, "outcome": o, "summary": s}),
                ),
            )
            if outcome not in {"reached", "stopped"} and not bool(args.continue_on_failure):
                break
            if outcome == "stopped":
                break
        return self.write_obstacle_avoidance_3_batch_summary(batch_dir, episodes, batch_summaries, finished=True)

    def write_obstacle_avoidance_3_batch_summary(
        self,
        batch_dir: Path,
        episodes: List[Dict[str, Any]],
        batch_summaries: List[Dict[str, Any]],
        *,
        finished: bool = False,
    ) -> Dict[str, Any]:
        summary = {
            "batch_dir": str(batch_dir),
            "source": "obstacle_avoidance_3_gui_active_session",
            "episode_count": len(episodes),
            "completed_count": len(batch_summaries),
            "reached_count": sum(1 for item in batch_summaries if item.get("outcome") == "reached"),
            "collision_count": sum(1 for item in batch_summaries if item.get("outcome") == "collision" or item.get("had_collision")),
            "summaries": batch_summaries,
            ("finished_at" if finished else "updated_at"): datetime.now().isoformat(timespec="milliseconds"),
        }
        write_json(batch_dir / "route_episode_batch_summary.json", summary)
        return summary

    def read_obstacle_avoidance_3_hit_state(self, session: flight.DroneFlightSession) -> Dict[str, Any]:
        lock = getattr(session, "api_lock", None)
        acquired = False
        try:
            if lock is not None:
                lock.acquire()
                acquired = True
            env = getattr(session, "env", None)
            drone_name = getattr(session, "drone_name", None)
            unrealcv = getattr(getattr(env, "unwrapped", env), "unrealcv", None)
            if unrealcv is None or not drone_name or not hasattr(unrealcv, "get_hit"):
                return {"available": False, "collision_state": False, "collision_source": "unavailable"}
            raw = unrealcv.get_hit(drone_name)
            try:
                collision = bool(float(raw) > 0.0)
            except Exception:
                collision = str(raw).strip().lower() in {"1", "true", "yes", "hit", "collision", "collided", "impact"}
            return {
                "available": True,
                "collision_state": collision,
                "impact_state": collision,
                "collision_source": "unreal_get_hit",
                "raw_hit": raw,
            }
        except Exception as exc:
            return {
                "available": False,
                "collision_state": False,
                "impact_state": False,
                "collision_source": "unreal_get_hit_error",
                "error": str(exc),
            }
        finally:
            if acquired:
                try:
                    lock.release()
                except Exception:
                    pass

    def notify_obstacle_avoidance_3_collision(self, *, episode_id: str, frame_id: int, session_dir: str, impact_detail: Dict[str, Any]) -> None:
        key = f"{episode_id}:{session_dir}"
        if key in self.obstacle_avoidance_3_collision_alert_keys:
            return
        self.obstacle_avoidance_3_collision_alert_keys.add(key)
        message = f"OA3 collision detected: {episode_id} frame {frame_id}"
        self.obstacle_avoidance_3_impact_var.set(f"Impact: YES ({episode_id} F{frame_id})")
        self.obstacle_avoidance_3_status_var.set(message)
        self.obstacle_avoidance_3_report(
            {
                "status": "collision_alert",
                "episode_id": episode_id,
                "frame_id": frame_id,
                "session_dir": session_dir,
                "impact_detail": impact_detail,
            }
        )
        try:
            self.root.bell()
        except Exception:
            pass
        try:
            messagebox.showwarning("OA3 Collision Alert", message)
        except Exception:
            pass

    def stop_obstacle_avoidance_3_runner(self) -> None:
        self.obstacle_avoidance_3_stop_event.set()
        self.obstacle_avoidance_3_status_var.set("OA3 stop requested")

    def emergency_stop_obstacle_avoidance_3_all(self) -> None:
        self.obstacle_avoidance_3_stop_event.set()
        self.obstacle_avoidance_3_or2_monitor_stop_event.set()
        self.stop_keyboard_control(send_hold=False)
        session = self.session
        if session is not None and getattr(session, "started", False):
            try:
                session.move_relative(action_payload("hold"))
                session.set_movement_enabled(False)
            except Exception as exc:
                self.obstacle_avoidance_3_report({"status": "emergency_stop_error", "error": str(exc)})
        self.obstacle_avoidance_3_status_var.set("OA3 emergency stop requested")
