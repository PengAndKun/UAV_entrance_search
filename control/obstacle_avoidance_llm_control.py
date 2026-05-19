from __future__ import annotations

from copy import deepcopy
from tkinter import messagebox

from .common import *

from obstacle_avoidance.collect import append_jsonl, write_json
from obstacle_avoidance.collect_route_episodes import (
    DEFAULT_ROUTE_SIDE_CORRECTION_CM,
    DEFAULT_ROUTE_STEP_CM,
    DEFAULT_ROUTE_VERTICAL_STEP_CM,
    annotate_collision_state,
    action_payload,
    build_route_event,
    distance_3d_cm,
    load_episodes,
    make_building_obstacle_state,
    pose_dict,
    pose_from_state,
    risk_state_from_summary,
    route_completion_state,
    route_progress_and_deviation,
    select_route_action,
    should_stop_for_repeated_hold,
    summarize_episode,
    write_partial_episode_summary,
)
from obstacle_avoidance_llm.plans import (
    DEFAULT_ENVIRONMENT_ID as LLM_DEFAULT_ENVIRONMENT_ID,
    DEFAULT_METHOD_ID as LLM_DEFAULT_METHOD_ID,
    DEFAULT_PLAN_FILENAME as LLM_DEFAULT_PLAN_FILENAME,
    DEFAULT_PROJECT_ID as LLM_DEFAULT_PROJECT_ID,
    LLM_DIRECT_METHOD_ID,
    LLM_STRATEGY_METHOD_ID,
    coerce_pose,
    export_selected_episodes,
    load_plans,
    make_default_plans,
    method_is_runnable,
    project_by_id,
    sanitize_id,
    save_plans,
    validate_plan_episode,
)
from obstacle_avoidance_llm.height_estimator import estimate_pointcloud_flyover_height
from obstacle_avoidance_llm.policy import (
    DIRECT_DECISION_SCHEMA,
    STRATEGY_DECISION_SCHEMA,
    apply_strategy_to_episode_metadata,
    build_direct_prompts,
    build_strategy_prompts,
    normalize_direct_decision,
    normalize_strategy_decision,
    refine_strategy_with_pointcloud_context,
    shield_direct_payload,
    strategy_from_episode_metadata,
)


class ObstacleAvoidanceLLMControlMixin:
    def ensure_obstacle_avoidance_llm_state(self) -> None:
        if not hasattr(self, "obstacle_avoidance_llm_data_dir_var"):
            self.obstacle_avoidance_llm_data_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "obstacle_avoidance_llm_data"))
        if not hasattr(self, "obstacle_avoidance_llm_plan_json_var"):
            self.obstacle_avoidance_llm_plan_json_var = tk.StringVar(
                value=str(PROJECT_ROOT / "obstacle_avoidance_llm_data" / "plans" / LLM_DEFAULT_PLAN_FILENAME)
            )
        if not hasattr(self, "obstacle_avoidance_llm_project_var"):
            self.obstacle_avoidance_llm_project_var = tk.StringVar(value=LLM_DEFAULT_PROJECT_ID)
        if not hasattr(self, "obstacle_avoidance_llm_project_name_var"):
            self.obstacle_avoidance_llm_project_name_var = tk.StringVar(value="OA-LLM default route obstacle collection")
        if not hasattr(self, "obstacle_avoidance_llm_environment_var"):
            self.obstacle_avoidance_llm_environment_var = tk.StringVar(value=LLM_DEFAULT_ENVIRONMENT_ID)
        if not hasattr(self, "obstacle_avoidance_llm_method_var"):
            self.obstacle_avoidance_llm_method_var = tk.StringVar(value=LLM_DEFAULT_METHOD_ID)
        if not hasattr(self, "obstacle_avoidance_llm_status_var"):
            self.obstacle_avoidance_llm_status_var = tk.StringVar(value="Obstacle Avoidance LLM: idle")
        if not hasattr(self, "obstacle_avoidance_llm_goal_distance_var"):
            self.obstacle_avoidance_llm_goal_distance_var = tk.StringVar(value="Goal distance: --")
        if not hasattr(self, "obstacle_avoidance_llm_goal_reached_var"):
            self.obstacle_avoidance_llm_goal_reached_var = tk.StringVar(value="Goal reached: --")
        if not hasattr(self, "obstacle_avoidance_llm_impact_var"):
            self.obstacle_avoidance_llm_impact_var = tk.StringVar(value="Impact: unknown")
        if not hasattr(self, "obstacle_avoidance_llm_analysis_env_var"):
            self.obstacle_avoidance_llm_analysis_env_var = tk.StringVar(value="LLM Environment: --")
        if not hasattr(self, "obstacle_avoidance_llm_analysis_hint_var"):
            self.obstacle_avoidance_llm_analysis_hint_var = tk.StringVar(value="LLM Hint: --")
        if not hasattr(self, "obstacle_avoidance_llm_analysis_summary_var"):
            self.obstacle_avoidance_llm_analysis_summary_var = tk.StringVar(value="LLM Analysis: not run")
        if not hasattr(self, "obstacle_avoidance_llm_pointcloud_height_var"):
            self.obstacle_avoidance_llm_pointcloud_height_var = tk.StringVar(value="PointCloud Height: --")
        if not hasattr(self, "obstacle_avoidance_llm_episode_id_var"):
            self.obstacle_avoidance_llm_episode_id_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_llm_enabled_var"):
            self.obstacle_avoidance_llm_enabled_var = tk.BooleanVar(value=True)
        if not hasattr(self, "obstacle_avoidance_llm_start_pose_var"):
            self.obstacle_avoidance_llm_start_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_llm_goal_pose_var"):
            self.obstacle_avoidance_llm_goal_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_llm_scenario_var"):
            self.obstacle_avoidance_llm_scenario_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_llm_obstacle_hint_var"):
            self.obstacle_avoidance_llm_obstacle_hint_var = tk.StringVar(value="unknown")
        if not hasattr(self, "obstacle_avoidance_llm_note_var"):
            self.obstacle_avoidance_llm_note_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_llm_record_var"):
            self.obstacle_avoidance_llm_record_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_llm_record_summary_var"):
            self.obstacle_avoidance_llm_record_summary_var = tk.StringVar(value="Select an episode to inspect LLM records.")
        if not hasattr(self, "obstacle_avoidance_llm_plan_data"):
            self.obstacle_avoidance_llm_plan_data = None
        if not hasattr(self, "obstacle_avoidance_llm_window"):
            self.obstacle_avoidance_llm_window = None
        if not hasattr(self, "obstacle_avoidance_llm_report_text"):
            self.obstacle_avoidance_llm_report_text = None
        if not hasattr(self, "obstacle_avoidance_llm_project_combo"):
            self.obstacle_avoidance_llm_project_combo = None
        if not hasattr(self, "obstacle_avoidance_llm_environment_combo"):
            self.obstacle_avoidance_llm_environment_combo = None
        if not hasattr(self, "obstacle_avoidance_llm_method_combo"):
            self.obstacle_avoidance_llm_method_combo = None
        if not hasattr(self, "obstacle_avoidance_llm_episode_tree"):
            self.obstacle_avoidance_llm_episode_tree = None
        if not hasattr(self, "obstacle_avoidance_llm_tree_iids"):
            self.obstacle_avoidance_llm_tree_iids = {}
        if not hasattr(self, "obstacle_avoidance_llm_record_combo"):
            self.obstacle_avoidance_llm_record_combo = None
        if not hasattr(self, "obstacle_avoidance_llm_record_entries"):
            self.obstacle_avoidance_llm_record_entries = []
        if not hasattr(self, "obstacle_avoidance_llm_runner_thread"):
            self.obstacle_avoidance_llm_runner_thread = None
        if not hasattr(self, "obstacle_avoidance_llm_analysis_thread"):
            self.obstacle_avoidance_llm_analysis_thread = None
        if not hasattr(self, "obstacle_avoidance_llm_stop_event"):
            self.obstacle_avoidance_llm_stop_event = threading.Event()

    def obstacle_avoidance_llm_data_root(self) -> Path:
        self.ensure_obstacle_avoidance_llm_state()
        raw = str(self.obstacle_avoidance_llm_data_dir_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else PROJECT_ROOT / "obstacle_avoidance_llm_data"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def obstacle_avoidance_llm_plan_path(self) -> Path:
        self.ensure_obstacle_avoidance_llm_state()
        raw = str(self.obstacle_avoidance_llm_plan_json_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else self.obstacle_avoidance_llm_data_root() / "plans" / LLM_DEFAULT_PLAN_FILENAME
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def open_obstacle_avoidance_llm_window(self) -> None:
        self.ensure_obstacle_avoidance_llm_state()
        if self.obstacle_avoidance_llm_window is not None and self.obstacle_avoidance_llm_window.winfo_exists():
            self.obstacle_avoidance_llm_window.lift()
            self.obstacle_avoidance_llm_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Obstacle Avoidance LLM - Route Plan Collection")
        window.geometry("1160x760")
        window.minsize(980, 620)
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        self.obstacle_avoidance_llm_window = window

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
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda event: (
                canvas.itemconfigure(content_window, width=max(content.winfo_reqwidth(), int(event.width))),
                canvas.configure(scrollregion=canvas.bbox("all")),
            ),
        )
        self.build_obstacle_avoidance_llm_window(content)
        try:
            self.load_obstacle_avoidance_llm_plans()
        except Exception as exc:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM load failed: {exc}")
            self.obstacle_avoidance_llm_report({"status": "error", "task": "load_plans", "error": str(exc)})

        def close_window() -> None:
            self.obstacle_avoidance_llm_window = None
            self.obstacle_avoidance_llm_report_text = None
            self.obstacle_avoidance_llm_record_combo = None
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)

    def build_obstacle_avoidance_llm_window(self, window: tk.Widget) -> None:
        storage = tk.LabelFrame(window, text="Plan Storage")
        storage.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        storage.grid_columnconfigure(1, weight=1)
        storage.grid_columnconfigure(4, weight=1)
        tk.Label(storage, text="Data Dir").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(storage, textvariable=self.obstacle_avoidance_llm_data_dir_var).grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        tk.Button(storage, text="Browse", command=self.select_obstacle_avoidance_llm_data_dir).grid(row=0, column=2, padx=6, pady=5)
        tk.Label(storage, text="Plan JSON").grid(row=0, column=3, sticky="w", padx=6, pady=5)
        tk.Entry(storage, textvariable=self.obstacle_avoidance_llm_plan_json_var).grid(row=0, column=4, sticky="ew", padx=6, pady=5)
        tk.Button(storage, text="Browse", command=self.select_obstacle_avoidance_llm_plan_json).grid(row=0, column=5, padx=6, pady=5)
        tk.Button(storage, text="Reload", command=self.load_obstacle_avoidance_llm_plans).grid(row=0, column=6, padx=6, pady=5)
        tk.Button(storage, text="Save", command=self.save_obstacle_avoidance_llm_plans).grid(row=0, column=7, padx=6, pady=5)
        tk.Button(storage, text="Reset 10 Points", command=self.reset_obstacle_avoidance_llm_plans).grid(row=0, column=8, padx=6, pady=5)

        llm = tk.LabelFrame(window, text="LLM Config (shared with LLM Route)")
        llm.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3, 5, 7):
            llm.grid_columnconfigure(col, weight=1)
        tk.Label(llm, text="API").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        ttk.Combobox(llm, textvariable=self.llm_api_style_var, values=LLM_API_STYLE_OPTIONS, state="readonly", width=16).grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(llm, text="Base URL").grid(row=0, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(llm, textvariable=self.llm_base_url_var).grid(row=0, column=3, sticky="ew", padx=6, pady=5)
        tk.Label(llm, text="Model").grid(row=0, column=4, sticky="w", padx=6, pady=5)
        tk.Entry(llm, textvariable=self.llm_model_var).grid(row=0, column=5, sticky="ew", padx=6, pady=5)
        tk.Label(llm, text="Timeout s").grid(row=0, column=6, sticky="w", padx=6, pady=5)
        tk.Entry(llm, textvariable=self.llm_timeout_s_var, width=8).grid(row=0, column=7, sticky="w", padx=6, pady=5)
        tk.Button(llm, text="Apply Defaults", command=lambda: self.apply_llm_api_defaults(force=True)).grid(row=0, column=8, padx=6, pady=5)

        project = tk.LabelFrame(window, text="Project / Environment / Method")
        project.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3, 5):
            project.grid_columnconfigure(col, weight=1)
        tk.Label(project, text="Project").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_llm_project_combo = ttk.Combobox(project, textvariable=self.obstacle_avoidance_llm_project_var, state="readonly", width=30)
        self.obstacle_avoidance_llm_project_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        self.obstacle_avoidance_llm_project_combo.bind("<<ComboboxSelected>>", lambda _event: self.select_obstacle_avoidance_llm_project())
        tk.Label(project, text="Name").grid(row=0, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(project, textvariable=self.obstacle_avoidance_llm_project_name_var).grid(row=0, column=3, sticky="ew", padx=6, pady=5)
        tk.Button(project, text="Apply", command=self.apply_obstacle_avoidance_llm_project).grid(row=0, column=4, padx=6, pady=5)
        tk.Button(project, text="New", command=self.new_obstacle_avoidance_llm_project).grid(row=0, column=5, padx=6, pady=5)
        tk.Button(project, text="Copy", command=self.copy_obstacle_avoidance_llm_project).grid(row=0, column=6, padx=6, pady=5)
        tk.Button(project, text="Delete", command=self.delete_obstacle_avoidance_llm_project).grid(row=0, column=7, padx=6, pady=5)
        tk.Label(project, text="Environment").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_llm_environment_combo = ttk.Combobox(project, textvariable=self.obstacle_avoidance_llm_environment_var, state="readonly", width=24)
        self.obstacle_avoidance_llm_environment_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(project, text="Method").grid(row=1, column=2, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_llm_method_combo = ttk.Combobox(project, textvariable=self.obstacle_avoidance_llm_method_var, state="readonly", width=30)
        self.obstacle_avoidance_llm_method_combo.grid(row=1, column=3, sticky="ew", padx=6, pady=5)
        tk.Label(project, textvariable=self.obstacle_avoidance_llm_status_var, anchor="w").grid(row=2, column=0, columnspan=8, sticky="ew", padx=6, pady=5)

        experiment = tk.LabelFrame(window, text="Start-Goal LLM Experiment")
        experiment.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        for col in (4, 6, 8):
            experiment.grid_columnconfigure(col, weight=1)
        tk.Button(experiment, text="1. Locate Start", command=self.locate_obstacle_avoidance_llm_start).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(experiment, text="2. LLM Analyze", command=self.analyze_obstacle_avoidance_llm_current_episode).grid(row=0, column=1, padx=6, pady=6)
        tk.Button(experiment, text="3. Apply Algorithm", command=self.apply_obstacle_avoidance_llm_algorithm).grid(row=0, column=2, padx=6, pady=6)
        tk.Button(experiment, text="4. Execute To Goal", command=self.execute_obstacle_avoidance_llm_to_goal).grid(row=0, column=3, padx=6, pady=6)
        tk.Button(experiment, text="Stop All", command=self.emergency_stop_obstacle_avoidance_llm_all).grid(row=0, column=4, padx=6, pady=6)
        tk.Label(experiment, textvariable=self.obstacle_avoidance_llm_goal_distance_var, anchor="w").grid(row=0, column=5, sticky="ew", padx=12, pady=6)
        tk.Label(experiment, textvariable=self.obstacle_avoidance_llm_goal_reached_var, anchor="w").grid(row=0, column=6, sticky="ew", padx=12, pady=6)
        tk.Label(experiment, textvariable=self.obstacle_avoidance_llm_impact_var, anchor="w").grid(row=0, column=8, sticky="ew", padx=12, pady=6)
        tk.Label(experiment, textvariable=self.obstacle_avoidance_llm_analysis_env_var, anchor="w").grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
        tk.Label(experiment, textvariable=self.obstacle_avoidance_llm_analysis_hint_var, anchor="w").grid(row=1, column=2, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
        tk.Label(experiment, textvariable=self.obstacle_avoidance_llm_analysis_summary_var, anchor="w", wraplength=900, justify="left").grid(row=2, column=0, columnspan=9, sticky="ew", padx=6, pady=(0, 6))
        tk.Label(experiment, textvariable=self.obstacle_avoidance_llm_pointcloud_height_var, anchor="w", wraplength=900, justify="left").grid(row=3, column=0, columnspan=9, sticky="ew", padx=6, pady=(0, 6))

        body = tk.Frame(window)
        body.grid(row=4, column=0, sticky="nsew", padx=8, pady=4)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)
        tree_frame = tk.LabelFrame(body, text="Episodes")
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        tree_frame.grid_columnconfigure(0, weight=1)
        columns = ("enabled", "episode", "start", "goal", "environment", "method", "hint", "note")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended", height=11)
        headings = {
            "enabled": "On",
            "episode": "ID",
            "start": "Start [x,y,z,yaw]",
            "goal": "Goal [x,y,z,yaw]",
            "environment": "Environment",
            "method": "Method",
            "hint": "Hint",
            "note": "Note",
        }
        widths = {"enabled": 55, "episode": 70, "start": 210, "goal": 210, "environment": 150, "method": 210, "hint": 120, "note": 180}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w", stretch=True)
        ybar = tk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        xbar = tk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        tree.bind("<<TreeviewSelect>>", lambda _event: self.on_obstacle_avoidance_llm_episode_selected())
        self.obstacle_avoidance_llm_episode_tree = tree

        editor = tk.LabelFrame(body, text="Episode Editor")
        editor.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
        editor.grid_columnconfigure(1, weight=1)
        tk.Checkbutton(editor, text="Enabled", variable=self.obstacle_avoidance_llm_enabled_var).grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(editor, textvariable=self.obstacle_avoidance_llm_episode_id_var, width=12).grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        for row, (label, var) in enumerate(
            (
                ("Start Pose", self.obstacle_avoidance_llm_start_pose_var),
                ("Goal Pose", self.obstacle_avoidance_llm_goal_pose_var),
                ("Scenario", self.obstacle_avoidance_llm_scenario_var),
                ("Note", self.obstacle_avoidance_llm_note_var),
            ),
            start=1,
        ):
            tk.Label(editor, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=5)
            tk.Entry(editor, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(editor, text="Obstacle Hint").grid(row=5, column=0, sticky="w", padx=6, pady=5)
        ttk.Combobox(
            editor,
            textvariable=self.obstacle_avoidance_llm_obstacle_hint_var,
            values=("unknown", "tree_trunk_or_pole", "tree_canopy_or_cluster", "tree", "pole", "fence_or_rail", "building", "mixed"),
            state="readonly",
        ).grid(row=5, column=1, sticky="ew", padx=6, pady=5)
        tk.Button(editor, text="Apply Episode", command=self.apply_obstacle_avoidance_llm_episode).grid(row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 5))
        tk.Button(editor, text="Add Episode", command=self.add_obstacle_avoidance_llm_episode).grid(row=7, column=0, sticky="ew", padx=6, pady=5)
        tk.Button(editor, text="Delete Episode", command=self.delete_obstacle_avoidance_llm_episode).grid(row=7, column=1, sticky="ew", padx=6, pady=5)

        runner = tk.LabelFrame(window, text="Collection Runner")
        runner.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(runner, text="Export Selected JSON", command=self.export_obstacle_avoidance_llm_selected_json).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Dry Run Selected", command=self.dry_run_obstacle_avoidance_llm_selected).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Run Selected Episodes", command=self.run_obstacle_avoidance_llm_selected).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Run Current Episode", command=self.run_obstacle_avoidance_llm_current_episode).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Stop Run", command=self.stop_obstacle_avoidance_llm_runner).pack(side="left", padx=6, pady=6)

        records = tk.LabelFrame(window, text="Run Records")
        records.grid(row=6, column=0, sticky="ew", padx=8, pady=4)
        records.grid_columnconfigure(1, weight=1)
        tk.Label(records, text="Record").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_llm_record_combo = ttk.Combobox(records, textvariable=self.obstacle_avoidance_llm_record_var, state="readonly")
        self.obstacle_avoidance_llm_record_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        self.obstacle_avoidance_llm_record_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_obstacle_avoidance_llm_record_summary())
        tk.Button(records, text="Refresh Records", command=self.refresh_obstacle_avoidance_llm_records).grid(row=0, column=2, padx=6, pady=5)
        tk.Button(records, text="View Selected", command=self.view_obstacle_avoidance_llm_selected_record).grid(row=0, column=3, padx=6, pady=5)
        tk.Label(records, textvariable=self.obstacle_avoidance_llm_record_summary_var, anchor="w").grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 5))

        report = tk.LabelFrame(window, text="Report")
        report.grid(row=7, column=0, sticky="nsew", padx=8, pady=(4, 8))
        report.grid_columnconfigure(0, weight=1)
        report.grid_rowconfigure(0, weight=1)
        text = tk.Text(report, height=10, wrap="none")
        y = tk.Scrollbar(report, orient="vertical", command=text.yview)
        x = tk.Scrollbar(report, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        text.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        self.obstacle_avoidance_llm_report_text = text

    def obstacle_avoidance_llm_report(self, payload: Any) -> None:
        text = self.obstacle_avoidance_llm_report_text
        if text is None:
            return
        try:
            rendered = payload if isinstance(payload, str) else json.dumps(payload, indent=2, ensure_ascii=False)
            text.insert("end", rendered + "\n")
            text.see("end")
        except tk.TclError:
            pass

    def select_obstacle_avoidance_llm_data_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self.obstacle_avoidance_llm_data_root()))
        if path:
            self.obstacle_avoidance_llm_data_dir_var.set(path)

    def select_obstacle_avoidance_llm_plan_json(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(self.obstacle_avoidance_llm_plan_path().parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.obstacle_avoidance_llm_plan_json_var.set(path)
            self.load_obstacle_avoidance_llm_plans()

    def load_obstacle_avoidance_llm_plans(self) -> None:
        data = load_plans(self.obstacle_avoidance_llm_plan_path())
        self.obstacle_avoidance_llm_plan_data = data
        save_plans(self.obstacle_avoidance_llm_plan_path(), data)
        self.refresh_obstacle_avoidance_llm_options()
        active = str(data.get("active_project_id", LLM_DEFAULT_PROJECT_ID))
        self.obstacle_avoidance_llm_project_var.set(active)
        self.select_obstacle_avoidance_llm_project()
        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM plans loaded: {self.obstacle_avoidance_llm_plan_path()}")

    def save_obstacle_avoidance_llm_plans(self) -> None:
        if not isinstance(self.obstacle_avoidance_llm_plan_data, dict):
            self.obstacle_avoidance_llm_plan_data = make_default_plans()
        self.apply_obstacle_avoidance_llm_project(silent=True)
        self.apply_obstacle_avoidance_llm_episode(silent=True)
        save_plans(self.obstacle_avoidance_llm_plan_path(), self.obstacle_avoidance_llm_plan_data)
        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM plans saved: {self.obstacle_avoidance_llm_plan_path()}")

    def reset_obstacle_avoidance_llm_plans(self) -> None:
        self.obstacle_avoidance_llm_plan_data = make_default_plans()
        save_plans(self.obstacle_avoidance_llm_plan_path(), self.obstacle_avoidance_llm_plan_data)
        self.load_obstacle_avoidance_llm_plans()
        self.obstacle_avoidance_llm_status_var.set("OA-LLM default 10 episodes restored")

    def refresh_obstacle_avoidance_llm_options(self) -> None:
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        projects = [str(item.get("project_id", "")) for item in data.get("projects", []) if isinstance(item, dict)]
        environments = [str(item.get("environment_id", "")) for item in data.get("environments", []) if isinstance(item, dict)]
        methods = [str(item.get("method_id", "")) for item in data.get("methods", []) if isinstance(item, dict)]
        if self.obstacle_avoidance_llm_project_combo is not None:
            self.obstacle_avoidance_llm_project_combo.configure(values=tuple(projects))
        if self.obstacle_avoidance_llm_environment_combo is not None:
            self.obstacle_avoidance_llm_environment_combo.configure(values=tuple(environments))
        if self.obstacle_avoidance_llm_method_combo is not None:
            self.obstacle_avoidance_llm_method_combo.configure(values=tuple(methods))

    def current_obstacle_avoidance_llm_project(self) -> Dict[str, Any]:
        if not isinstance(self.obstacle_avoidance_llm_plan_data, dict):
            self.obstacle_avoidance_llm_plan_data = make_default_plans()
        return project_by_id(self.obstacle_avoidance_llm_plan_data, self.obstacle_avoidance_llm_project_var.get())

    def select_obstacle_avoidance_llm_project(self) -> None:
        project = self.current_obstacle_avoidance_llm_project()
        self.obstacle_avoidance_llm_project_name_var.set(str(project.get("name", "")))
        self.obstacle_avoidance_llm_environment_var.set(str(project.get("environment_id", LLM_DEFAULT_ENVIRONMENT_ID)))
        self.obstacle_avoidance_llm_method_var.set(str(project.get("default_method", LLM_DEFAULT_METHOD_ID)))
        self.refresh_obstacle_avoidance_llm_tree()

    def apply_obstacle_avoidance_llm_project(self, *, silent: bool = False) -> None:
        project = self.current_obstacle_avoidance_llm_project()
        project["name"] = str(self.obstacle_avoidance_llm_project_name_var.get() or project.get("project_id", "")).strip()
        project["environment_id"] = str(self.obstacle_avoidance_llm_environment_var.get() or LLM_DEFAULT_ENVIRONMENT_ID).strip()
        project["default_method"] = str(self.obstacle_avoidance_llm_method_var.get() or LLM_DEFAULT_METHOD_ID).strip()
        if isinstance(self.obstacle_avoidance_llm_plan_data, dict):
            self.obstacle_avoidance_llm_plan_data["active_project_id"] = project["project_id"]
        if not silent:
            self.refresh_obstacle_avoidance_llm_tree()
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM project applied: {project['project_id']}")

    def unique_obstacle_avoidance_llm_project_id(self, base: str) -> str:
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        existing = {str(item.get("project_id", "")) for item in data.get("projects", []) if isinstance(item, dict)}
        stem = sanitize_id(base, "llm_project")
        candidate = stem
        index = 2
        while candidate in existing:
            candidate = f"{stem}_{index}"
            index += 1
        return candidate

    def new_obstacle_avoidance_llm_project(self) -> None:
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        project_id = self.unique_obstacle_avoidance_llm_project_id("llm_project")
        project = {
            "project_id": project_id,
            "name": f"OA-LLM {project_id}",
            "environment_id": str(self.obstacle_avoidance_llm_environment_var.get() or LLM_DEFAULT_ENVIRONMENT_ID),
            "default_method": str(self.obstacle_avoidance_llm_method_var.get() or LLM_DEFAULT_METHOD_ID),
            "episodes": [],
            "experiment_defaults": {"stage": "route_episode_llm", "interval_s": 5.0, "reach_tol_cm": 180},
        }
        data.setdefault("projects", []).append(project)
        data["active_project_id"] = project_id
        self.obstacle_avoidance_llm_plan_data = data
        self.obstacle_avoidance_llm_project_var.set(project_id)
        self.refresh_obstacle_avoidance_llm_options()
        self.select_obstacle_avoidance_llm_project()

    def copy_obstacle_avoidance_llm_project(self) -> None:
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        source = deepcopy(self.current_obstacle_avoidance_llm_project())
        project_id = self.unique_obstacle_avoidance_llm_project_id(f"{source.get('project_id', 'project')}_copy")
        source["project_id"] = project_id
        source["name"] = f"{source.get('name', project_id)} copy"
        data.setdefault("projects", []).append(source)
        data["active_project_id"] = project_id
        self.obstacle_avoidance_llm_project_var.set(project_id)
        self.refresh_obstacle_avoidance_llm_options()
        self.select_obstacle_avoidance_llm_project()

    def delete_obstacle_avoidance_llm_project(self) -> None:
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        projects = [p for p in data.get("projects", []) if isinstance(p, dict)]
        if len(projects) <= 1:
            self.obstacle_avoidance_llm_status_var.set("OA-LLM cannot delete the last project")
            return
        current = str(self.obstacle_avoidance_llm_project_var.get())
        data["projects"] = [p for p in projects if str(p.get("project_id", "")) != current]
        data["active_project_id"] = str(data["projects"][0]["project_id"])
        self.obstacle_avoidance_llm_project_var.set(data["active_project_id"])
        self.refresh_obstacle_avoidance_llm_options()
        self.select_obstacle_avoidance_llm_project()

    def format_obstacle_avoidance_llm_pose(self, pose: Any) -> str:
        try:
            values = coerce_pose(pose)
            return ", ".join(f"{float(v):g}" for v in values)
        except Exception:
            return str(pose)

    def refresh_obstacle_avoidance_llm_tree(self) -> None:
        tree = self.obstacle_avoidance_llm_episode_tree
        if tree is None:
            return
        selected = set(self.selected_obstacle_avoidance_llm_episode_ids())
        for iid in tree.get_children():
            tree.delete(iid)
        self.obstacle_avoidance_llm_tree_iids = {}
        project = self.current_obstacle_avoidance_llm_project()
        for index, episode in enumerate(project.get("episodes", []), start=1):
            episode_id = str(episode.get("episode_id", f"E{index:02d}"))
            iid = f"episode_{index}"
            self.obstacle_avoidance_llm_tree_iids[iid] = episode_id
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    "yes" if bool(episode.get("enabled", True)) else "no",
                    episode_id,
                    self.format_obstacle_avoidance_llm_pose(episode.get("start_pose", "")),
                    self.format_obstacle_avoidance_llm_pose(episode.get("goal_pose", "")),
                    str(episode.get("environment_id", project.get("environment_id", LLM_DEFAULT_ENVIRONMENT_ID))),
                    str(episode.get("method", project.get("default_method", LLM_DEFAULT_METHOD_ID))),
                    str(episode.get("obstacle_hint", "unknown")),
                    str(episode.get("operator_note", "")),
                ),
            )
        target_iids = [iid for iid, eid in self.obstacle_avoidance_llm_tree_iids.items() if eid in selected]
        if target_iids:
            tree.selection_set(target_iids)
            tree.see(target_iids[0])

    def selected_obstacle_avoidance_llm_episode_ids(self) -> List[str]:
        tree = self.obstacle_avoidance_llm_episode_tree
        if tree is None:
            return []
        return [self.obstacle_avoidance_llm_tree_iids.get(str(iid), "") for iid in tree.selection() if self.obstacle_avoidance_llm_tree_iids.get(str(iid), "")]

    def find_obstacle_avoidance_llm_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        project = self.current_obstacle_avoidance_llm_project()
        for episode in project.get("episodes", []):
            if str(episode.get("episode_id", "")) == str(episode_id):
                return episode
        return None

    def on_obstacle_avoidance_llm_episode_selected(self) -> None:
        ids = self.selected_obstacle_avoidance_llm_episode_ids()
        if not ids:
            return
        episode = self.find_obstacle_avoidance_llm_episode(ids[0])
        if not isinstance(episode, dict):
            return
        self.obstacle_avoidance_llm_enabled_var.set(bool(episode.get("enabled", True)))
        self.obstacle_avoidance_llm_episode_id_var.set(str(episode.get("episode_id", "")))
        self.obstacle_avoidance_llm_start_pose_var.set(self.format_obstacle_avoidance_llm_pose(episode.get("start_pose", "")))
        self.obstacle_avoidance_llm_goal_pose_var.set(self.format_obstacle_avoidance_llm_pose(episode.get("goal_pose", "")))
        self.obstacle_avoidance_llm_scenario_var.set(str(episode.get("scenario_id", "")))
        self.obstacle_avoidance_llm_environment_var.set(str(episode.get("environment_id", self.obstacle_avoidance_llm_environment_var.get())))
        self.obstacle_avoidance_llm_method_var.set(str(episode.get("method", self.obstacle_avoidance_llm_method_var.get())))
        self.obstacle_avoidance_llm_obstacle_hint_var.set(str(episode.get("obstacle_hint", "unknown")))
        self.obstacle_avoidance_llm_note_var.set(str(episode.get("operator_note", "")))
        self.refresh_obstacle_avoidance_llm_records(str(episode.get("episode_id", "")))

    def build_obstacle_avoidance_llm_editor_episode(self) -> Optional[Dict[str, Any]]:
        episode_id = str(self.obstacle_avoidance_llm_episode_id_var.get() or "").strip()
        if not episode_id:
            episode_id = self.next_obstacle_avoidance_llm_episode_id()
            self.obstacle_avoidance_llm_episode_id_var.set(episode_id)
        try:
            start_pose = coerce_pose(self.obstacle_avoidance_llm_start_pose_var.get())
            goal_pose = coerce_pose(self.obstacle_avoidance_llm_goal_pose_var.get())
        except Exception as exc:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM invalid pose: {exc}")
            return None
        project = self.current_obstacle_avoidance_llm_project()
        episode = {
            "episode_id": episode_id,
            "enabled": bool(self.obstacle_avoidance_llm_enabled_var.get()),
            "start_pose": start_pose,
            "goal_pose": goal_pose,
            "scenario_id": str(self.obstacle_avoidance_llm_scenario_var.get() or f"oa_llm_route_{episode_id}").strip(),
            "environment_id": str(self.obstacle_avoidance_llm_environment_var.get() or project.get("environment_id", LLM_DEFAULT_ENVIRONMENT_ID)).strip(),
            "method": str(self.obstacle_avoidance_llm_method_var.get() or project.get("default_method", LLM_DEFAULT_METHOD_ID)).strip(),
            "obstacle_hint": str(self.obstacle_avoidance_llm_obstacle_hint_var.get() or "unknown").strip(),
            "operator_note": str(self.obstacle_avoidance_llm_note_var.get() or "").strip(),
        }
        errors = validate_plan_episode(episode)
        if errors:
            self.obstacle_avoidance_llm_status_var.set("; ".join(errors))
            return None
        return episode

    def upsert_obstacle_avoidance_llm_editor_episode(self) -> Optional[Dict[str, Any]]:
        episode = self.build_obstacle_avoidance_llm_editor_episode()
        if episode is None:
            return None
        project = self.current_obstacle_avoidance_llm_project()
        existing = self.find_obstacle_avoidance_llm_episode(str(episode.get("episode_id", "")))
        if existing is None:
            project.setdefault("episodes", []).append(dict(episode))
        else:
            existing.clear()
            existing.update(episode)
        if isinstance(self.obstacle_avoidance_llm_plan_data, dict):
            save_plans(self.obstacle_avoidance_llm_plan_path(), self.obstacle_avoidance_llm_plan_data)
        self.refresh_obstacle_avoidance_llm_tree()
        self.refresh_obstacle_avoidance_llm_records(str(episode.get("episode_id", "")))
        return episode

    def apply_obstacle_avoidance_llm_episode(self, *, silent: bool = False) -> None:
        episode = self.upsert_obstacle_avoidance_llm_editor_episode()
        if episode is not None and not silent:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM episode applied: {episode['episode_id']}")

    def next_obstacle_avoidance_llm_episode_id(self) -> str:
        project = self.current_obstacle_avoidance_llm_project()
        used = {str(item.get("episode_id", "")) for item in project.get("episodes", [])}
        index = 1
        while f"E{index:02d}" in used:
            index += 1
        return f"E{index:02d}"

    def add_obstacle_avoidance_llm_episode(self) -> None:
        episode_id = self.next_obstacle_avoidance_llm_episode_id()
        project = self.current_obstacle_avoidance_llm_project()
        episode = {
            "episode_id": episode_id,
            "enabled": True,
            "start_pose": [0.0, 0.0, 100.0, 0.0],
            "goal_pose": [500.0, 0.0, 100.0, 0.0],
            "scenario_id": f"oa_llm_route_{episode_id}",
            "environment_id": str(self.obstacle_avoidance_llm_environment_var.get() or project.get("environment_id", LLM_DEFAULT_ENVIRONMENT_ID)),
            "method": str(self.obstacle_avoidance_llm_method_var.get() or project.get("default_method", LLM_DEFAULT_METHOD_ID)),
            "obstacle_hint": "unknown",
            "operator_note": "",
        }
        project.setdefault("episodes", []).append(episode)
        self.refresh_obstacle_avoidance_llm_tree()
        self.obstacle_avoidance_llm_episode_id_var.set(episode_id)
        self.on_obstacle_avoidance_llm_episode_selected()

    def delete_obstacle_avoidance_llm_episode(self) -> None:
        ids = set(self.selected_obstacle_avoidance_llm_episode_ids() or [self.obstacle_avoidance_llm_episode_id_var.get()])
        project = self.current_obstacle_avoidance_llm_project()
        project["episodes"] = [episode for episode in project.get("episodes", []) if str(episode.get("episode_id", "")) not in ids]
        self.refresh_obstacle_avoidance_llm_tree()

    def obstacle_avoidance_llm_export_path(self, purpose: str) -> Path:
        project_id = sanitize_id(self.obstacle_avoidance_llm_project_var.get() or LLM_DEFAULT_PROJECT_ID, LLM_DEFAULT_PROJECT_ID)
        method = sanitize_id(self.obstacle_avoidance_llm_method_var.get() or LLM_DEFAULT_METHOD_ID, LLM_DEFAULT_METHOD_ID)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.obstacle_avoidance_llm_data_root() / "plans" / purpose / f"{stamp}_{project_id}_{method}_episodes.json"

    def export_obstacle_avoidance_llm_selected_json(self) -> Optional[Path]:
        try:
            self.apply_obstacle_avoidance_llm_project(silent=True)
            self.apply_obstacle_avoidance_llm_episode(silent=True)
            path = export_selected_episodes(
                self.current_obstacle_avoidance_llm_project(),
                self.selected_obstacle_avoidance_llm_episode_ids(),
                self.obstacle_avoidance_llm_export_path("exports"),
            )
        except Exception as exc:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM export failed: {exc}")
            return None
        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM exported: {path}")
        self.obstacle_avoidance_llm_report({"status": "exported", "path": str(path)})
        return path

    def dry_run_obstacle_avoidance_llm_selected(self) -> None:
        path = self.export_obstacle_avoidance_llm_selected_json()
        if path is None:
            return
        try:
            rows = load_episodes(str(path))
            self.obstacle_avoidance_llm_report({"status": "dry_run", "count": len(rows), "episodes": rows})
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM dry run: {len(rows)} episode(s)")
        except Exception as exc:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM dry run failed: {exc}")

    def resolve_obstacle_avoidance_llm_episodes(self, episodes_json: Path) -> List[Dict[str, Any]]:
        rows = load_episodes(str(episodes_json))
        resolved: List[Dict[str, Any]] = []
        for row in rows:
            item = {
                "episode_id": str(row["episode_id"]),
                "start_pose": pose_dict(row["start_pose"]),
                "goal_pose": pose_dict(row["goal_pose"]),
            }
            for key in ("scenario_id", "environment_id", "method", "obstacle_hint", "operator_note", "enabled"):
                if key in row:
                    item[key] = row.get(key)
            resolved.append(item)
        return resolved

    def scan_obstacle_avoidance_llm_records(self, episode_id: str = "") -> List[Dict[str, Any]]:
        root = self.obstacle_avoidance_llm_data_root() / "sessions"
        if not root.exists():
            return []
        records: List[Dict[str, Any]] = []
        for summary_path in root.glob("*/episode_summary.json"):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            session_dir = summary_path.parent
            config = {}
            config_path = session_dir / "episode_config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    config = {}
            episode = config.get("episode", {}) if isinstance(config.get("episode"), dict) else {}
            eid = str(summary.get("episode_id", "") or episode.get("episode_id", ""))
            if episode_id and eid != episode_id:
                continue
            records.append(
                {
                    "episode_id": eid,
                    "session_dir": str(session_dir),
                    "outcome": summary.get("outcome", ""),
                    "goal_reached": bool(summary.get("goal_reached", False)),
                    "final_distance_to_goal_cm": summary.get("final_distance_to_goal_cm", ""),
                    "method": str(summary.get("method", "") or config.get("args", {}).get("method", "") or episode.get("method", "")),
                    "finished_at": summary.get("finished_at", ""),
                }
            )
        records.sort(key=lambda item: str(item.get("session_dir", "")), reverse=True)
        return records

    def refresh_obstacle_avoidance_llm_records(self, episode_id: Optional[str] = None) -> None:
        eid = episode_id if episode_id is not None else str(self.obstacle_avoidance_llm_episode_id_var.get() or "")
        records = self.scan_obstacle_avoidance_llm_records(eid)
        self.obstacle_avoidance_llm_record_entries = records
        values = [
            f"{idx + 1:02d} {record.get('episode_id', '')} | {record.get('outcome', '')} | d={record.get('final_distance_to_goal_cm', '')} | {Path(str(record.get('session_dir', ''))).name}"
            for idx, record in enumerate(records)
        ]
        if self.obstacle_avoidance_llm_record_combo is not None:
            self.obstacle_avoidance_llm_record_combo.configure(values=tuple(values))
        if values:
            self.obstacle_avoidance_llm_record_var.set(values[0])
            self.update_obstacle_avoidance_llm_record_summary()
        else:
            self.obstacle_avoidance_llm_record_var.set("")
            self.obstacle_avoidance_llm_record_summary_var.set(f"{eid or 'episode'}: no LLM run records yet.")

    def selected_obstacle_avoidance_llm_record(self) -> Optional[Dict[str, Any]]:
        value = str(self.obstacle_avoidance_llm_record_var.get() or "")
        if not value:
            return None
        try:
            index = int(value.split(" ", 1)[0]) - 1
        except Exception:
            return None
        if 0 <= index < len(self.obstacle_avoidance_llm_record_entries):
            return self.obstacle_avoidance_llm_record_entries[index]
        return None

    def update_obstacle_avoidance_llm_record_summary(self) -> None:
        record = self.selected_obstacle_avoidance_llm_record()
        if record is None:
            return
        self.obstacle_avoidance_llm_record_summary_var.set(
            f"{record.get('episode_id', '')}: outcome={record.get('outcome', '')}, "
            f"reached={record.get('goal_reached')}, distance={record.get('final_distance_to_goal_cm')}cm, "
            f"method={record.get('method', '')}"
        )

    def view_obstacle_avoidance_llm_selected_record(self) -> None:
        record = self.selected_obstacle_avoidance_llm_record()
        if record is None:
            self.obstacle_avoidance_llm_status_var.set("OA-LLM no record selected")
            return
        session_dir = Path(str(record.get("session_dir", "")))
        payload = {"record": record}
        for name in ("episode_summary.json", "episode_config.json", "route_episode_batch_summary.json"):
            path = session_dir / name
            if path.exists():
                try:
                    payload[name] = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload[name] = {"error": str(exc)}
        self.obstacle_avoidance_llm_report(payload)

    def obstacle_avoidance_llm_args(self, *, method: str, run_id: str, note: str) -> argparse.Namespace:
        return argparse.Namespace(
            stage="route_episode_llm",
            method=method,
            run_id=run_id,
            note=note,
            geometry_label="unknown",
            reach_tol_cm=180.0,
            max_ticks_per_episode=220,
            route_step_cm=DEFAULT_ROUTE_STEP_CM,
            side_correction_cm=DEFAULT_ROUTE_SIDE_CORRECTION_CM,
            vertical_step_cm=DEFAULT_ROUTE_VERTICAL_STEP_CM,
            interval_s=5.0,
            continue_on_failure=True,
            movement_mode="physics",
            lidar_capture_processing="minimal",
        )

    def locate_obstacle_avoidance_llm_start(self) -> None:
        if self.obstacle_avoidance_llm_runner_thread is not None and self.obstacle_avoidance_llm_runner_thread.is_alive():
            self.obstacle_avoidance_llm_status_var.set("OA-LLM runner is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_llm_status_var.set("OA-LLM: start Unreal from the main window first")
            return
        episode = self.upsert_obstacle_avoidance_llm_editor_episode()
        if episode is None:
            return
        start = pose_dict(episode["start_pose"])
        goal = pose_dict(episode["goal_pose"])
        try:
            response = session.set_pose({"x": start["x"], "y": start["y"], "z": start["z"], "yaw": start["yaw"]})
            mode_result = session.set_movement_mode("physics")
            enable_result = session.set_movement_enabled(True)
        except Exception as exc:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM locate start failed: {exc}")
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
        self.obstacle_avoidance_llm_impact_var.set("Impact: no")
        self.obstacle_avoidance_llm_goal_distance_var.set(f"Goal distance: {start_distance:.1f} cm")
        self.obstacle_avoidance_llm_goal_reached_var.set("Goal reached: no")
        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM located start: {episode['episode_id']} dist={start_distance:.1f}cm")
        self.obstacle_avoidance_llm_report({"status": "located_start", "episode_id": episode["episode_id"], "start_pose": start, "goal_pose": goal})

    def apply_obstacle_avoidance_llm_algorithm(self) -> None:
        if self.obstacle_avoidance_llm_plan_data is None:
            self.load_obstacle_avoidance_llm_plans()
        method = str(self.obstacle_avoidance_llm_method_var.get() or LLM_DEFAULT_METHOD_ID).strip()
        project = self.current_obstacle_avoidance_llm_project()
        project["default_method"] = method
        episode = self.upsert_obstacle_avoidance_llm_editor_episode()
        if episode is None:
            return
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        runnable = method_is_runnable(data, method)
        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM algorithm selected: {method} ({'executable' if runnable else 'metadata only'})")
        self.obstacle_avoidance_llm_report({"status": "algorithm_selected", "episode_id": episode.get("episode_id", ""), "method": method, "runnable": runnable})

    def execute_obstacle_avoidance_llm_to_goal(self) -> None:
        self.run_obstacle_avoidance_llm_current_episode()

    def obstacle_avoidance_llm_image_b64(self, event: Dict[str, Any]) -> str:
        rgb_path = Path(str(event.get("rgb_path", "") or ""))
        if not rgb_path.is_file():
            return ""
        try:
            return base64.b64encode(rgb_path.read_bytes()).decode("ascii")
        except Exception:
            return ""

    def obstacle_avoidance_llm_json_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        image_b64: str,
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.effective_llm_api_key():
            return {
                "parsed": dict(fallback),
                "raw": {"planner_source": "deterministic_fallback_no_api_key"},
                "error": "missing LLM API key",
            }
        try:
            raw = self.call_configured_llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=700,
                json_schema=schema,
                image_b64=image_b64,
            )
            parsed = extract_json_object(str(raw.get("raw_text", "") or ""))
            if not parsed:
                parsed = dict(fallback)
                error_text = "LLM returned no JSON object"
            else:
                error_text = ""
            return {"parsed": parsed, "raw": raw, "error": error_text}
        except Exception as exc:
            return {"parsed": dict(fallback), "raw": {}, "error": str(exc)}

    def obstacle_avoidance_llm_direct_decision(
        self,
        event: Dict[str, Any],
        episode: Dict[str, Any],
        last_action: Dict[str, Any],
        current_pose: Dict[str, Any],
    ) -> Tuple[str, Dict[str, float], str, Dict[str, Any]]:
        system_prompt, user_prompt = build_direct_prompts(event, episode, last_action)
        fallback = {
            "action_name": "hold",
            "forward_cm": 0.0,
            "right_cm": 0.0,
            "up_cm": 0.0,
            "yaw_delta_deg": 0.0,
            "reason": "fallback hold because LLM direct decision was unavailable",
            "confidence": 0.0,
        }
        call = self.obstacle_avoidance_llm_json_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=DIRECT_DECISION_SCHEMA,
            image_b64=self.obstacle_avoidance_llm_image_b64(event),
            fallback=fallback,
        )
        llm_payload, meta = normalize_direct_decision(call.get("parsed", {}))
        shielded, shield = shield_direct_payload(llm_payload, event.get("pointcloud_summary", {}), current_pose)
        decision = {
            "mode": LLM_DIRECT_METHOD_ID,
            "llm_decision": meta,
            "llm_payload": llm_payload,
            "shielded_payload": shielded,
            "shield": shield,
            "llm_error": call.get("error", ""),
            "llm_raw": call.get("raw", {}),
        }
        action = str(shielded.get("action_name", "hold"))
        reason = (
            f"llm_direct_control_v1: action={meta.get('action_name')} confidence={meta.get('confidence')}; "
            f"{meta.get('reason', '')}; shield={shield.get('state')} {shield.get('reason', '')}"
        )
        if call.get("error"):
            reason = f"{reason}; llm_error={call.get('error')}"
        return action, shielded, reason, decision

    def obstacle_avoidance_llm_strategy_decision(self, event: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt, user_prompt = build_strategy_prompts(event, episode)
        fallback = {
            "environment_id": episode.get("environment_id", LLM_DEFAULT_ENVIRONMENT_ID),
            "obstacle_hint": episode.get("obstacle_hint", "unknown"),
            "recommended_method": "pointcloud_direction_rule",
            "flyover_z_cm": 0.0,
            "lateral_preference": "auto",
            "vertical_policy": "auto",
            "strategy_reason": "fallback strategy because LLM strategy decision was unavailable",
        }
        call = self.obstacle_avoidance_llm_json_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=STRATEGY_DECISION_SCHEMA,
            image_b64=self.obstacle_avoidance_llm_image_b64(event),
            fallback=fallback,
        )
        strategy = normalize_strategy_decision(call.get("parsed", {}))
        strategy["llm_error"] = call.get("error", "")
        strategy["llm_raw"] = call.get("raw", {})
        height_estimate = estimate_pointcloud_flyover_height(event, strategy)
        strategy["pointcloud_height_estimate"] = height_estimate
        if height_estimate.get("available"):
            strategy["pointcloud_recommended_flyover_z_cm"] = height_estimate.get("recommended_flyover_z_cm", 0.0)
            strategy["pointcloud_recommended_vertical_offset_cm"] = height_estimate.get("recommended_vertical_offset_cm", 0.0)
            strategy["pointcloud_obstacle_height_cm"] = height_estimate.get("obstacle_height_cm", 0.0)
            strategy["pointcloud_obstacle_top_z_cm"] = height_estimate.get("obstacle_top_z_cm", 0.0)
            strategy["pointcloud_height_source"] = height_estimate.get("height_source", "")
            strategy["pointcloud_flyover_recommended"] = bool(height_estimate.get("flyover_recommended", False))
            strategy["pointcloud_clearance_z_cm"] = height_estimate.get("clearance_z_cm", 0.0)
            strategy["flyover_z_cm"] = height_estimate.get("recommended_vertical_offset_cm", 0.0)
        strategy = refine_strategy_with_pointcloud_context(strategy, event)
        return strategy

    def show_obstacle_avoidance_llm_analysis(self, episode: Dict[str, Any], strategy: Dict[str, Any], *, analysis_path: str = "") -> None:
        environment_id = str(strategy.get("environment_id", LLM_DEFAULT_ENVIRONMENT_ID) or LLM_DEFAULT_ENVIRONMENT_ID)
        obstacle_hint = str(strategy.get("obstacle_hint", "unknown") or "unknown")
        flyover_z = strategy.get("flyover_z_cm", 0.0)
        lateral = str(strategy.get("lateral_preference", "auto") or "auto")
        vertical = str(strategy.get("vertical_policy", "auto") or "auto")
        reason = str(strategy.get("strategy_reason", "") or "")
        error_text = str(strategy.get("llm_error", "") or "")
        height_estimate = strategy.get("pointcloud_height_estimate") if isinstance(strategy.get("pointcloud_height_estimate"), dict) else {}

        self.obstacle_avoidance_llm_analysis_env_var.set(f"LLM Environment: {environment_id}")
        self.obstacle_avoidance_llm_analysis_hint_var.set(f"LLM Hint: {obstacle_hint}")
        summary = (
            f"LLM Analysis: flyover_z={flyover_z}cm, lateral={lateral}, vertical={vertical}; "
            f"reason={reason or 'n/a'}"
        )
        if error_text:
            summary = f"{summary}; error={error_text}"
        self.obstacle_avoidance_llm_analysis_summary_var.set(summary)
        if height_estimate.get("available"):
            self.obstacle_avoidance_llm_pointcloud_height_var.set(
                "PointCloud Height: "
                f"height={height_estimate.get('obstacle_height_cm', 0.0)}cm, "
                f"top_z={height_estimate.get('obstacle_top_z_cm', 0.0)}cm, "
                f"target_z={height_estimate.get('recommended_flyover_z_cm', 0.0)}cm, "
                f"offset={height_estimate.get('recommended_vertical_offset_cm', 0.0)}cm, "
                f"source={height_estimate.get('height_source', '')}, "
                f"flyover={height_estimate.get('flyover_recommended', False)}"
            )
        else:
            self.obstacle_avoidance_llm_pointcloud_height_var.set(
                f"PointCloud Height: unavailable; {height_estimate.get('reason', 'no estimate')}"
            )

        original_method = str(episode.get("method", self.obstacle_avoidance_llm_method_var.get() or LLM_DEFAULT_METHOD_ID))
        updated = apply_strategy_to_episode_metadata(episode, strategy)
        updated["method"] = original_method
        existing = self.find_obstacle_avoidance_llm_episode(str(updated.get("episode_id", "")))
        if existing is not None:
            existing.clear()
            existing.update(updated)
        self.obstacle_avoidance_llm_environment_var.set(environment_id)
        self.obstacle_avoidance_llm_obstacle_hint_var.set(obstacle_hint)
        self.obstacle_avoidance_llm_method_var.set(original_method)
        self.obstacle_avoidance_llm_note_var.set(str(updated.get("operator_note", "")))
        if isinstance(self.obstacle_avoidance_llm_plan_data, dict):
            save_plans(self.obstacle_avoidance_llm_plan_path(), self.obstacle_avoidance_llm_plan_data)
        self.refresh_obstacle_avoidance_llm_tree()
        self.obstacle_avoidance_llm_status_var.set(
            f"OA-LLM analysis applied: env={environment_id}, hint={obstacle_hint}"
        )
        self.obstacle_avoidance_llm_report(
            {
                "status": "llm_analysis_applied",
                "episode_id": updated.get("episode_id", ""),
                "environment_id": environment_id,
                "obstacle_hint": obstacle_hint,
                "strategy": strategy,
                "analysis_path": analysis_path,
            }
        )

    def analyze_obstacle_avoidance_llm_current_episode(self) -> None:
        if self.obstacle_avoidance_llm_runner_thread is not None and self.obstacle_avoidance_llm_runner_thread.is_alive():
            self.obstacle_avoidance_llm_status_var.set("OA-LLM runner is already active")
            return
        if self.obstacle_avoidance_llm_analysis_thread is not None and self.obstacle_avoidance_llm_analysis_thread.is_alive():
            self.obstacle_avoidance_llm_status_var.set("OA-LLM analysis is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_llm_status_var.set("OA-LLM: start Unreal from the main window first")
            return
        episode = self.upsert_obstacle_avoidance_llm_editor_episode()
        if episode is None:
            return

        episode = deepcopy(episode)
        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM analyzing current view: {episode.get('episode_id', '')}")
        self.obstacle_avoidance_llm_analysis_summary_var.set("LLM Analysis: running...")
        self.obstacle_avoidance_llm_pointcloud_height_var.set("PointCloud Height: calculating after LLM type...")

        def worker() -> None:
            episode_id = str(episode.get("episode_id", "E00"))
            start = pose_dict(episode["start_pose"])
            goal = pose_dict(episode["goal_pose"])
            run_id = sanitize_id(f"analysis_{episode_id}_{datetime.now().strftime('%H%M%S')}", "analysis")
            args = self.obstacle_avoidance_llm_args(method=LLM_STRATEGY_METHOD_ID, run_id=run_id, note="llm one-frame analysis")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            session_dir = self.obstacle_avoidance_llm_data_root() / "analysis" / f"{timestamp}_llm_analysis_{episode_id}_{run_id}"
            events_path = session_dir / "analysis_events.jsonl"
            try:
                session_dir.mkdir(parents=True, exist_ok=False)
                try:
                    session.args.lidar_capture_processing = "minimal"
                except Exception:
                    pass
                pre_state = session.get_state()
                pre_pose = pose_from_state(pre_state if isinstance(pre_state, dict) else {})
                action_detail = {
                    "source": "obstacle_avoidance_llm_analysis",
                    "episode_id": episode_id,
                    "episode_index": 1,
                    "collection_stage": "route_episode_llm_analysis",
                    "scenario_id": episode.get("scenario_id") or f"oa_llm_route_{episode_id}",
                    "environment_id": episode.get("environment_id", ""),
                    "method": LLM_STRATEGY_METHOD_ID,
                    "plan_method": episode.get("method", ""),
                    "obstacle_hint": episode.get("obstacle_hint", ""),
                    "run_id": run_id,
                    "mission_phase": "LLM_ANALYSIS",
                    "risk_state": "SAFE",
                    "expert_action": "hold",
                    "expert_action_payload": action_payload("hold"),
                    "start_pose": start,
                    "goal_pose": goal,
                    "target_waypoint": goal,
                    "operator_note": episode.get("operator_note", ""),
                }
                result = session.capture_lidar_stream_frame(session_dir, 1, action_detail=action_detail)
                if not isinstance(result, dict):
                    raise RuntimeError("capture_lidar_stream_frame returned non-dict")
                event = build_route_event(
                    result,
                    session_dir=session_dir,
                    frame_id=1,
                    args=args,
                    episode=episode,
                    episode_index=1,
                    start=start,
                    goal=goal,
                    last_action=action_payload("hold"),
                )
                event["source"] = "obstacle_avoidance_llm_analysis"
                event["current_pose_before_analysis"] = pre_pose
                strategy = self.obstacle_avoidance_llm_strategy_decision(event, episode)
                event["llm_strategy"] = deepcopy(strategy)
                event["llm_analysis_result"] = deepcopy(strategy)
                append_jsonl(events_path, event)
                summary = {
                    "status": "ok",
                    "episode_id": episode_id,
                    "session_dir": str(session_dir),
                    "analysis_event_path": str(events_path),
                    "strategy": strategy,
                    "finished_at": datetime.now().isoformat(timespec="milliseconds"),
                }
                write_json(session_dir / "analysis_summary.json", summary)
                self.root.after(
                    0,
                    lambda e=deepcopy(episode), s=deepcopy(strategy), p=str(session_dir): self.show_obstacle_avoidance_llm_analysis(
                        e,
                        s,
                        analysis_path=p,
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda err=str(exc): (
                        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM analysis failed: {err}"),
                        self.obstacle_avoidance_llm_analysis_summary_var.set(f"LLM Analysis: failed; error={err}"),
                        self.obstacle_avoidance_llm_report({"status": "error", "task": "llm_analysis", "error": err}),
                    ),
                )

        self.obstacle_avoidance_llm_analysis_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_llm_analysis_thread.start()

    def prepare_obstacle_avoidance_llm_run(self) -> Optional[Tuple[Dict[str, Any], List[str], Path, str, str, str]]:
        if self.obstacle_avoidance_llm_plan_data is None:
            self.load_obstacle_avoidance_llm_plans()
        self.apply_obstacle_avoidance_llm_project(silent=True)
        self.apply_obstacle_avoidance_llm_episode(silent=True)
        method = str(self.obstacle_avoidance_llm_method_var.get() or LLM_DEFAULT_METHOD_ID).strip()
        project = self.current_obstacle_avoidance_llm_project()
        selected_ids = self.selected_obstacle_avoidance_llm_episode_ids()
        try:
            episodes_json = export_selected_episodes(project, selected_ids, self.obstacle_avoidance_llm_export_path("tmp"))
        except Exception as exc:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM export failed: {exc}")
            return None
        run_id = sanitize_id(f"{project.get('project_id', LLM_DEFAULT_PROJECT_ID)}_{method}_{datetime.now().strftime('%H%M%S')}", "oa_llm_run")
        note = (
            f"oa_llm_project={project.get('project_id', '')}; "
            f"environment={self.obstacle_avoidance_llm_environment_var.get()}; "
            f"selected={','.join(selected_ids) if selected_ids else 'enabled'}"
        )
        return project, selected_ids, episodes_json, method, run_id, note

    def run_obstacle_avoidance_llm_current_episode(self) -> None:
        if self.obstacle_avoidance_llm_runner_thread is not None and self.obstacle_avoidance_llm_runner_thread.is_alive():
            self.obstacle_avoidance_llm_status_var.set("OA-LLM runner is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_llm_status_var.set("OA-LLM: start Unreal from the main window first")
            return
        episode = self.upsert_obstacle_avoidance_llm_editor_episode()
        if episode is None:
            return
        method = str(episode.get("method", "") or self.obstacle_avoidance_llm_method_var.get() or LLM_DEFAULT_METHOD_ID).strip()
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        if not method_is_runnable(data, method):
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM method not executable: {method}")
            self.obstacle_avoidance_llm_report({"status": "blocked", "method": method, "reason": "runner_not_implemented"})
            return
        project = self.current_obstacle_avoidance_llm_project()
        resolved = [
            {
                "episode_id": str(episode["episode_id"]),
                "start_pose": pose_dict(episode["start_pose"]),
                "goal_pose": pose_dict(episode["goal_pose"]),
                "scenario_id": episode.get("scenario_id", ""),
                "environment_id": episode.get("environment_id", ""),
                "method": episode.get("method", method),
                "obstacle_hint": episode.get("obstacle_hint", ""),
                "operator_note": episode.get("operator_note", ""),
                "enabled": episode.get("enabled", True),
            }
        ]
        run_id = sanitize_id(
            f"{project.get('project_id', LLM_DEFAULT_PROJECT_ID)}_{method}_{episode['episode_id']}_{datetime.now().strftime('%H%M%S')}",
            "oa_llm_single_run",
        )
        note = f"oa_llm_project={project.get('project_id', '')}; environment={episode.get('environment_id', '')}; single_episode={episode['episode_id']}"
        self.obstacle_avoidance_llm_stop_event = threading.Event()
        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM current episode starting: {episode['episode_id']}")

        def worker() -> None:
            try:
                result = self.run_obstacle_avoidance_llm_collection_on_session(
                    session,
                    resolved,
                    method=method,
                    run_id=run_id,
                    note=note,
                    project_id=str(project.get("project_id", "")),
                )
                self.root.after(
                    0,
                    lambda r=result, eid=str(episode["episode_id"]): (
                        self.obstacle_avoidance_llm_status_var.set(
                            f"OA-LLM current episode done: {eid}, reached {r.get('reached_count', 0)}/{r.get('episode_count', 0)}, collision {r.get('collision_count', 0)}"
                        ),
                        self.refresh_obstacle_avoidance_llm_records(eid),
                        self.obstacle_avoidance_llm_report(r),
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM current episode failed: {e}"),
                        self.obstacle_avoidance_llm_report({"status": "error", "task": "current_episode_collection", "error": str(e)}),
                    ),
                )

        self.obstacle_avoidance_llm_runner_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_llm_runner_thread.start()

    def run_obstacle_avoidance_llm_selected(self) -> None:
        if self.obstacle_avoidance_llm_runner_thread is not None and self.obstacle_avoidance_llm_runner_thread.is_alive():
            self.obstacle_avoidance_llm_status_var.set("OA-LLM runner is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_llm_status_var.set("OA-LLM: start Unreal from the main window first")
            return
        prepared = self.prepare_obstacle_avoidance_llm_run()
        if prepared is None:
            return
        project, selected_ids, episodes_json, method, run_id, note = prepared
        data = self.obstacle_avoidance_llm_plan_data if isinstance(self.obstacle_avoidance_llm_plan_data, dict) else make_default_plans()
        if not method_is_runnable(data, method):
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM method not executable: {method}")
            self.obstacle_avoidance_llm_report({"status": "blocked", "method": method, "reason": "runner_not_implemented"})
            return
        try:
            episodes = self.resolve_obstacle_avoidance_llm_episodes(episodes_json)
        except Exception as exc:
            self.obstacle_avoidance_llm_status_var.set(f"OA-LLM episode parse failed: {exc}")
            return
        self.obstacle_avoidance_llm_stop_event = threading.Event()
        self.obstacle_avoidance_llm_status_var.set("OA-LLM in-session collection starting...")
        self.obstacle_avoidance_llm_report({"status": "starting_in_session_collection", "episode_count": len(episodes), "method": method, "selected_episode_ids": selected_ids})

        def worker() -> None:
            try:
                result = self.run_obstacle_avoidance_llm_collection_on_session(
                    session,
                    episodes,
                    method=method,
                    run_id=run_id,
                    note=note,
                    project_id=str(project.get("project_id", "")),
                )
                self.root.after(
                    0,
                    lambda r=result: (
                        self.obstacle_avoidance_llm_status_var.set(
                            f"OA-LLM collection done: reached {r.get('reached_count', 0)}/{r.get('episode_count', 0)}, collision {r.get('collision_count', 0)}"
                        ),
                        self.refresh_obstacle_avoidance_llm_records(),
                        self.obstacle_avoidance_llm_report(r),
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_avoidance_llm_status_var.set(f"OA-LLM collection failed: {e}"),
                        self.obstacle_avoidance_llm_report({"status": "error", "task": "in_session_collection", "error": str(e)}),
                    ),
                )

        self.obstacle_avoidance_llm_runner_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_llm_runner_thread.start()

    def run_obstacle_avoidance_llm_collection_on_session(
        self,
        session: flight.DroneFlightSession,
        episodes: List[Dict[str, Any]],
        *,
        method: str,
        run_id: str,
        note: str,
        project_id: str,
    ) -> Dict[str, Any]:
        args = self.obstacle_avoidance_llm_args(method=method, run_id=run_id, note=note)
        stop_event = self.obstacle_avoidance_llm_stop_event
        previous_lidar_processing = str(
            getattr(getattr(session, "args", object()), "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING)
            or flight.DEFAULT_LIDAR_CAPTURE_PROCESSING
        )
        try:
            session.args.lidar_capture_processing = "minimal"
        except Exception:
            pass
        if getattr(self, "lidar_capture_processing_var", None) is not None:
            try:
                self.root.after(0, lambda: self.lidar_capture_processing_var.set("minimal"))
            except Exception:
                pass
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        data_root = self.obstacle_avoidance_llm_data_root()
        batch_dir = data_root / "route_episode_batches" / f"{timestamp}_route_episode_llm_{method}_{run_id}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            batch_dir / "route_episode_batch_config.json",
            {
                "source": "obstacle_avoidance_llm_gui_active_session",
                "project_id": project_id,
                "args": vars(args),
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
                self.obstacle_avoidance_llm_report(
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
            session_dir = data_root / "sessions" / f"{timestamp}_route_episode_llm_{episode_id}_{method}_{run_id}"
            session_dir.mkdir(parents=True, exist_ok=False)
            events_path = session_dir / "avoidance_events.jsonl"
            write_json(
                session_dir / "episode_config.json",
                {
                    "source": "obstacle_avoidance_llm_gui_active_session",
                    "project_id": project_id,
                    "episode_index": episode_index,
                    "episode": episode,
                    "args": vars(args),
                    "llm_method": method,
                },
            )
            self.root.after(0, lambda eid=episode_id, s=start, g=goal: self.obstacle_avoidance_llm_report(f"OA_LLM_EPISODE_START {eid} start={s} goal={g}"))
            session.set_pose({"x": start["x"], "y": start["y"], "z": start["z"], "yaw": start["yaw"]})
            last_action = action_payload("hold")
            building_obstacle_state = make_building_obstacle_state()
            llm_strategy: Optional[Dict[str, Any]] = None
            events: List[Dict[str, Any]] = []
            outcome = "timeout"
            episode_error = ""

            for frame_id in range(1, int(args.max_ticks_per_episode) + 1):
                if stop_event.is_set():
                    outcome = "stopped"
                    break
                try:
                    pre_state = session.get_state()
                    pre_pose = pose_from_state(pre_state if isinstance(pre_state, dict) else {})
                    pre_distance = distance_3d_cm(pre_pose, goal)
                    runtime_episode = dict(episode)
                    if method == LLM_STRATEGY_METHOD_ID and isinstance(llm_strategy, dict):
                        runtime_episode["environment_id"] = llm_strategy.get("environment_id", runtime_episode.get("environment_id", ""))
                        runtime_episode["obstacle_hint"] = llm_strategy.get("obstacle_hint", runtime_episode.get("obstacle_hint", ""))
                    action_detail = {
                        "source": "obstacle_avoidance_llm_gui_active_session",
                        "project_id": project_id,
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "collection_stage": args.stage,
                        "scenario_id": runtime_episode.get("scenario_id") or f"oa_llm_route_{episode_id}",
                        "environment_id": runtime_episode.get("environment_id", ""),
                        "method": args.method,
                        "plan_method": episode.get("method", ""),
                        "obstacle_hint": runtime_episode.get("obstacle_hint", ""),
                        "run_id": args.run_id,
                        "mission_phase": "ROUTE_EPISODE_LLM",
                        "risk_state": "SAFE",
                        "expert_action": str(last_action.get("action_name", "hold")),
                        "expert_action_payload": last_action,
                        "start_pose": start,
                        "goal_pose": goal,
                        "target_waypoint": goal,
                        "operator_note": episode.get("operator_note") or args.note,
                        "building_obstacle_state": dict(building_obstacle_state),
                        "llm_strategy": llm_strategy or {},
                    }
                    result = session.capture_lidar_stream_frame(session_dir, frame_id, action_detail=action_detail)
                    if not isinstance(result, dict):
                        raise RuntimeError("capture_lidar_stream_frame returned non-dict")
                    event = build_route_event(
                        result,
                        session_dir=session_dir,
                        frame_id=frame_id,
                        args=args,
                        episode=runtime_episode,
                        episode_index=episode_index,
                        start=start,
                        goal=goal,
                        last_action=last_action,
                    )
                except Exception as exc:
                    outcome = "unrealcv_error"
                    episode_error = f"capture/get_state failed at frame {frame_id}: {exc}"
                    self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_llm_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                    break

                event["source"] = "obstacle_avoidance_llm_gui_active_session"
                event["project_id"] = project_id
                event["llm_method"] = method
                rel = event.get("relative_target") if isinstance(event.get("relative_target"), dict) else {}

                if method == LLM_DIRECT_METHOD_ID:
                    selected_action, payload, reason, llm_decision = self.obstacle_avoidance_llm_direct_decision(
                        event,
                        episode,
                        last_action,
                        pre_pose,
                    )
                    phase = "LLM_DIRECT_CONTROL"
                    event["llm_decision"] = llm_decision
                else:
                    if llm_strategy is None:
                        llm_strategy = strategy_from_episode_metadata(episode)
                        self.root.after(0, lambda eid=episode_id, s=deepcopy(llm_strategy): self.obstacle_avoidance_llm_report({"status": "llm_strategy_loaded_from_episode", "episode_id": eid, "strategy": s}))
                    runtime_env = str(llm_strategy.get("environment_id", episode.get("environment_id", "")) if isinstance(llm_strategy, dict) else episode.get("environment_id", ""))
                    runtime_hint = str(llm_strategy.get("obstacle_hint", episode.get("obstacle_hint", "")) if isinstance(llm_strategy, dict) else episode.get("obstacle_hint", ""))
                    selected_action, payload, rule_reason, phase = select_route_action(
                        event["pointcloud_summary"],
                        event["candidate_action_scores"],
                        rel,
                        method="pointcloud_direction_rule",
                        distance_to_goal_cm=pre_distance,
                        reach_tol_cm=float(args.reach_tol_cm),
                        last_action=last_action,
                        current_pose=pre_pose,
                        start=start,
                        goal=goal,
                        route_step_cm=float(args.route_step_cm),
                        side_correction_cm=float(args.side_correction_cm),
                        vertical_step_cm=float(args.vertical_step_cm),
                        obstacle_hint=runtime_hint,
                        environment_id=runtime_env,
                        building_state=building_obstacle_state,
                    )
                    reason = (
                        "llm_strategy_pointcloud_rule_v1: "
                        f"env={runtime_env} hint={runtime_hint} "
                        f"source={llm_strategy.get('strategy_source', '') if isinstance(llm_strategy, dict) else ''} "
                        f"policy={llm_strategy.get('vertical_policy', '') if isinstance(llm_strategy, dict) else ''}; "
                        f"{rule_reason}"
                    )
                    event["llm_strategy"] = deepcopy(llm_strategy)
                    event["environment_id"] = runtime_env
                    event["obstacle_hint"] = runtime_hint

                risk_state = risk_state_from_summary(event["pointcloud_summary"], phase)
                event.update(
                    {
                        "mission_phase": phase,
                        "risk_state": risk_state,
                        "selected_action": selected_action,
                        "selected_action_payload": payload,
                        "selected_action_reason": reason,
                        "expert_action": str(payload.get("action_name", selected_action)),
                        "expert_action_payload": payload,
                        "nominal_action": payload,
                        "agent_action": payload,
                        "executed_action": payload,
                        "shield_state": "GOAL_REACHED" if phase == "REACHED" else "LLM_SHIELD_APPLIED",
                        "episode_outcome": "running",
                        "building_obstacle_state": dict(building_obstacle_state),
                        "front_building_obstacle": bool(building_obstacle_state.get("front_building_obstacle", False)),
                        "current_front_building_obstacle": bool(building_obstacle_state.get("current_front_building_obstacle", False)),
                        "building_obstacle_cleared": bool(building_obstacle_state.get("cleared", False)),
                    }
                )
                if phase == "REACHED":
                    post_pose = pre_pose
                    post_distance = pre_distance
                    hit_state = self.read_obstacle_avoidance_2_hit_state(session) if hasattr(self, "read_obstacle_avoidance_2_hit_state") else {}
                else:
                    try:
                        response = session.move_relative(payload)
                    except Exception as exc:
                        outcome = "unrealcv_error"
                        episode_error = f"move_relative failed at frame {frame_id}: {exc}"
                        self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_llm_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                        break
                    post_pose = pose_from_state(response if isinstance(response, dict) else session.get_state())
                    post_distance = distance_3d_cm(post_pose, goal)
                    hit_state = self.read_obstacle_avoidance_2_hit_state(session) if hasattr(self, "read_obstacle_avoidance_2_hit_state") else {}
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
                self.root.after(
                    0,
                    lambda eid=episode_id, fid=frame_id, dist=post_distance, reached=post_reached, completion=completion_reason, action=selected_action, risk=event.get("risk_state", risk_state), hit=bool(event.get("collision_state")), phase=event.get("mission_phase", phase), reason=event.get("selected_action_reason", reason): (
                        self.obstacle_avoidance_llm_goal_distance_var.set(f"Goal distance: {dist:.1f} cm"),
                        self.obstacle_avoidance_llm_goal_reached_var.set(f"Goal reached: {'YES' if reached else 'no'} ({completion})"),
                        self.obstacle_avoidance_llm_status_var.set(
                            f"OA-LLM {eid} frame {fid}: dist={dist:.1f}cm reached={reached} action={action} phase={phase} risk={risk} impact={hit}"
                        ),
                        self.obstacle_avoidance_llm_impact_var.set("Impact: YES" if hit else "Impact: no"),
                        self.obstacle_avoidance_llm_report(
                            f"OA_LLM_FRAME {eid} {fid}/{args.max_ticks_per_episode} dist={dist:.1f} reached={reached} completion={completion} action={action} phase={phase} risk={risk} impact={hit} reason={reason}"
                        ),
                    ),
                )
                if bool(event.get("collision_state")) and hasattr(self, "notify_obstacle_avoidance_2_collision"):
                    alert_detail = deepcopy(event.get("impact_detail", {})) if isinstance(event.get("impact_detail"), dict) else {}
                    self.root.after(
                        0,
                        lambda eid=episode_id, fid=frame_id, sdir=str(session_dir), detail=alert_detail: self.notify_obstacle_avoidance_2_collision(
                            episode_id=eid,
                            frame_id=fid,
                            session_dir=sdir,
                            impact_detail=detail,
                        ),
                    )
                if outcome in {"reached", "collision", "stalled_hold"}:
                    break
                if frame_id < int(args.max_ticks_per_episode):
                    if stop_event.wait(max(0.0, float(args.interval_s))):
                        outcome = "stopped"
                        break

            summary = summarize_episode(session_dir, events, start=start, goal=goal, outcome=outcome, reach_tol_cm=float(args.reach_tol_cm))
            summary["source"] = "obstacle_avoidance_llm_gui_active_session"
            summary["method"] = method
            summary["llm_strategy"] = llm_strategy or {}
            if episode_error:
                summary["episode_error"] = episode_error
                summary["approach_failure_reason"] = episode_error
                write_json(session_dir / "episode_summary.json", summary)
                write_json(session_dir / "avoidance_session_summary.json", summary)
            batch_summaries.append(summary)
            write_json(
                batch_dir / "route_episode_batch_summary.json",
                {
                    "batch_dir": str(batch_dir),
                    "source": "obstacle_avoidance_llm_gui_active_session",
                    "episode_count": len(episodes),
                    "completed_count": len(batch_summaries),
                    "collision_count": sum(1 for item in batch_summaries if item.get("outcome") == "collision" or item.get("had_collision")),
                    "summaries": batch_summaries,
                    "updated_at": datetime.now().isoformat(timespec="milliseconds"),
                },
            )
            self.root.after(
                0,
                lambda eid=episode_id, o=outcome, s=summary: (
                    self.obstacle_avoidance_llm_goal_distance_var.set(f"Goal distance: {float(s.get('final_distance_to_goal_cm', 0.0)):.1f} cm"),
                    self.obstacle_avoidance_llm_goal_reached_var.set(f"Goal reached: {'YES' if s.get('goal_reached') else 'no'} ({s.get('goal_completion_reason', '')})"),
                    self.obstacle_avoidance_llm_report({"episode_done": eid, "outcome": o, "summary": s}),
                ),
            )
            if outcome not in {"reached", "stopped"} and not bool(args.continue_on_failure):
                break
            if outcome == "stopped":
                break

        batch_summary = {
            "batch_dir": str(batch_dir),
            "source": "obstacle_avoidance_llm_gui_active_session",
            "episode_count": len(episodes),
            "completed_count": len(batch_summaries),
            "reached_count": sum(1 for item in batch_summaries if item.get("outcome") == "reached"),
            "collision_count": sum(1 for item in batch_summaries if item.get("outcome") == "collision" or item.get("had_collision")),
            "summaries": batch_summaries,
            "finished_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        write_json(batch_dir / "route_episode_batch_summary.json", batch_summary)
        return batch_summary

    def stop_obstacle_avoidance_llm_runner(self) -> None:
        self.obstacle_avoidance_llm_stop_event.set()
        self.obstacle_avoidance_llm_status_var.set("OA-LLM stop requested")

    def emergency_stop_obstacle_avoidance_llm_all(self) -> None:
        self.ensure_obstacle_avoidance_llm_state()
        self.obstacle_avoidance_llm_stop_event.set()
        try:
            self.stop_keyboard_control(send_hold=True, force_hold=True)
        except Exception:
            pass
        if hasattr(self, "obstacle_avoidance_2_stop_event"):
            try:
                self.obstacle_avoidance_2_stop_event.set()
            except Exception:
                pass
        self.obstacle_avoidance_llm_runner_thread = None
        self.obstacle_avoidance_llm_status_var.set("OA-LLM EMERGENCY STOP ALL: stopping runner and disabling movement...")
        self.obstacle_avoidance_llm_report({"status": "emergency_stop_all_requested", "runner_state": "cleared"})

        def worker() -> None:
            halt_result = self.halt_obstacle_avoidance_2_motion_now() if hasattr(self, "halt_obstacle_avoidance_2_motion_now") else {"status": "halt_unavailable"}
            self.root.after(
                0,
                lambda r=halt_result: (
                    self.obstacle_avoidance_llm_status_var.set("OA-LLM EMERGENCY STOP ALL complete; runner lock cleared"),
                    self.obstacle_avoidance_llm_report({"status": "emergency_stop_all_complete", "halt_result": r}),
                ),
            )

        threading.Thread(target=worker, daemon=True).start()
