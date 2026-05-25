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
from obstacle_avoidance_2.plans import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_METHOD_ID,
    DEFAULT_PLAN_FILENAME,
    DEFAULT_PROJECT_ID,
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


class ObstacleAvoidance2ControlMixin:
    def ensure_obstacle_avoidance_2_state(self) -> None:
        if not hasattr(self, "obstacle_avoidance_2_data_dir_var"):
            self.obstacle_avoidance_2_data_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "obstacle_avoidance_2_data"))
        if not hasattr(self, "obstacle_avoidance_2_plan_json_var"):
            self.obstacle_avoidance_2_plan_json_var = tk.StringVar(
                value=str(PROJECT_ROOT / "obstacle_avoidance_2_data" / "plans" / DEFAULT_PLAN_FILENAME)
            )
        if not hasattr(self, "obstacle_avoidance_2_project_var"):
            self.obstacle_avoidance_2_project_var = tk.StringVar(value=DEFAULT_PROJECT_ID)
        if not hasattr(self, "obstacle_avoidance_2_project_name_var"):
            self.obstacle_avoidance_2_project_name_var = tk.StringVar(value="OA2 default Route6_entrance_search obstacle collection")
        if not hasattr(self, "obstacle_avoidance_2_environment_var"):
            self.obstacle_avoidance_2_environment_var = tk.StringVar(value=DEFAULT_ENVIRONMENT_ID)
        if not hasattr(self, "obstacle_avoidance_2_method_var"):
            self.obstacle_avoidance_2_method_var = tk.StringVar(value=DEFAULT_METHOD_ID)
        if not hasattr(self, "obstacle_avoidance_2_status_var"):
            self.obstacle_avoidance_2_status_var = tk.StringVar(value="Obstacle Avoidance 2: idle")
        if not hasattr(self, "obstacle_avoidance_2_episode_id_var"):
            self.obstacle_avoidance_2_episode_id_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_2_enabled_var"):
            self.obstacle_avoidance_2_enabled_var = tk.BooleanVar(value=True)
        if not hasattr(self, "obstacle_avoidance_2_start_pose_var"):
            self.obstacle_avoidance_2_start_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_2_goal_pose_var"):
            self.obstacle_avoidance_2_goal_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_2_scenario_var"):
            self.obstacle_avoidance_2_scenario_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_2_obstacle_hint_var"):
            self.obstacle_avoidance_2_obstacle_hint_var = tk.StringVar(value="unknown")
        if not hasattr(self, "obstacle_avoidance_2_note_var"):
            self.obstacle_avoidance_2_note_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_2_plan_data"):
            self.obstacle_avoidance_2_plan_data = None
        if not hasattr(self, "obstacle_avoidance_2_window"):
            self.obstacle_avoidance_2_window = None
        if not hasattr(self, "obstacle_avoidance_2_report_text"):
            self.obstacle_avoidance_2_report_text = None
        if not hasattr(self, "obstacle_avoidance_2_project_combo"):
            self.obstacle_avoidance_2_project_combo = None
        if not hasattr(self, "obstacle_avoidance_2_environment_combo"):
            self.obstacle_avoidance_2_environment_combo = None
        if not hasattr(self, "obstacle_avoidance_2_method_combo"):
            self.obstacle_avoidance_2_method_combo = None
        if not hasattr(self, "obstacle_avoidance_2_episode_tree"):
            self.obstacle_avoidance_2_episode_tree = None
        if not hasattr(self, "obstacle_avoidance_2_tree_iids"):
            self.obstacle_avoidance_2_tree_iids = {}
        if not hasattr(self, "obstacle_avoidance_2_record_var"):
            self.obstacle_avoidance_2_record_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_2_record_summary_var"):
            self.obstacle_avoidance_2_record_summary_var = tk.StringVar(value="Select an episode to inspect records.")
        if not hasattr(self, "obstacle_avoidance_2_impact_var"):
            self.obstacle_avoidance_2_impact_var = tk.StringVar(value="Impact: unknown")
        if not hasattr(self, "obstacle_avoidance_2_goal_distance_var"):
            self.obstacle_avoidance_2_goal_distance_var = tk.StringVar(value="Goal distance: --")
        if not hasattr(self, "obstacle_avoidance_2_goal_reached_var"):
            self.obstacle_avoidance_2_goal_reached_var = tk.StringVar(value="Goal reached: --")
        if not hasattr(self, "obstacle_avoidance_2_record_combo"):
            self.obstacle_avoidance_2_record_combo = None
        if not hasattr(self, "obstacle_avoidance_2_record_entries"):
            self.obstacle_avoidance_2_record_entries = []
        if not hasattr(self, "obstacle_avoidance_2_runner_thread"):
            self.obstacle_avoidance_2_runner_thread = None
        if not hasattr(self, "obstacle_avoidance_2_runner_process"):
            self.obstacle_avoidance_2_runner_process = None
        if not hasattr(self, "obstacle_avoidance_2_stop_event"):
            self.obstacle_avoidance_2_stop_event = threading.Event()
        if not hasattr(self, "obstacle_avoidance_2_collision_alert_keys"):
            self.obstacle_avoidance_2_collision_alert_keys = set()

    def obstacle_avoidance_2_data_root(self) -> Path:
        self.ensure_obstacle_avoidance_2_state()
        raw = str(self.obstacle_avoidance_2_data_dir_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else PROJECT_ROOT / "obstacle_avoidance_2_data"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def obstacle_avoidance_2_plan_path(self) -> Path:
        self.ensure_obstacle_avoidance_2_state()
        raw = str(self.obstacle_avoidance_2_plan_json_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else self.obstacle_avoidance_2_data_root() / "plans" / DEFAULT_PLAN_FILENAME
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def open_obstacle_avoidance_2_window(self) -> None:
        self.ensure_obstacle_avoidance_2_state()
        if self.obstacle_avoidance_2_window is not None and self.obstacle_avoidance_2_window.winfo_exists():
            self.obstacle_avoidance_2_window.lift()
            self.obstacle_avoidance_2_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Obstacle Avoidance 2 - Route Plan Collection")
        window.geometry("1120x720")
        window.minsize(980, 620)
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        self.obstacle_avoidance_2_window = window

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

        self.build_obstacle_avoidance_2_window(content)
        try:
            self.load_obstacle_avoidance_2_plans()
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 load failed: {exc}")
            self.obstacle_avoidance_2_report({"status": "error", "task": "load_plans", "error": str(exc)})

        def close_window() -> None:
            self.obstacle_avoidance_2_window = None
            self.obstacle_avoidance_2_report_text = None
            self.obstacle_avoidance_2_record_combo = None
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)

    def build_obstacle_avoidance_2_window(self, window: tk.Widget) -> None:
        config = tk.LabelFrame(window, text="Plan Storage")
        config.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        config.grid_columnconfigure(1, weight=1)
        config.grid_columnconfigure(4, weight=1)
        tk.Label(config, text="Data Dir").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(config, textvariable=self.obstacle_avoidance_2_data_dir_var).grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        tk.Button(config, text="Browse", command=self.select_obstacle_avoidance_2_data_dir).grid(row=0, column=2, padx=6, pady=5)
        tk.Label(config, text="Plan JSON").grid(row=0, column=3, sticky="w", padx=6, pady=5)
        tk.Entry(config, textvariable=self.obstacle_avoidance_2_plan_json_var).grid(row=0, column=4, sticky="ew", padx=6, pady=5)
        tk.Button(config, text="Browse", command=self.select_obstacle_avoidance_2_plan_json).grid(row=0, column=5, padx=6, pady=5)
        tk.Button(config, text="Reload", command=self.load_obstacle_avoidance_2_plans).grid(row=0, column=6, padx=6, pady=5)
        tk.Button(config, text="Save", command=self.save_obstacle_avoidance_2_plans).grid(row=0, column=7, padx=6, pady=5)
        tk.Button(config, text="Reset 10 Points", command=self.reset_obstacle_avoidance_2_plans).grid(row=0, column=8, padx=6, pady=5)

        project = tk.LabelFrame(window, text="Project / Environment / Method")
        project.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3, 5):
            project.grid_columnconfigure(col, weight=1)
        tk.Label(project, text="Project").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_2_project_combo = ttk.Combobox(
            project,
            textvariable=self.obstacle_avoidance_2_project_var,
            state="readonly",
            width=30,
        )
        self.obstacle_avoidance_2_project_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        self.obstacle_avoidance_2_project_combo.bind("<<ComboboxSelected>>", lambda _event: self.select_obstacle_avoidance_2_project())
        tk.Label(project, text="Name").grid(row=0, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(project, textvariable=self.obstacle_avoidance_2_project_name_var).grid(row=0, column=3, sticky="ew", padx=6, pady=5)
        tk.Button(project, text="Apply", command=self.apply_obstacle_avoidance_2_project).grid(row=0, column=4, padx=6, pady=5)
        tk.Button(project, text="New", command=self.new_obstacle_avoidance_2_project).grid(row=0, column=5, padx=6, pady=5)
        tk.Button(project, text="Copy", command=self.copy_obstacle_avoidance_2_project).grid(row=0, column=6, padx=6, pady=5)
        tk.Button(project, text="Delete", command=self.delete_obstacle_avoidance_2_project).grid(row=0, column=7, padx=6, pady=5)
        tk.Label(project, text="Environment").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_2_environment_combo = ttk.Combobox(
            project,
            textvariable=self.obstacle_avoidance_2_environment_var,
            width=30,
        )
        self.obstacle_avoidance_2_environment_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(project, text="Method").grid(row=1, column=2, sticky="w", padx=6, pady=5)
        self.obstacle_avoidance_2_method_combo = ttk.Combobox(
            project,
            textvariable=self.obstacle_avoidance_2_method_var,
            width=30,
        )
        self.obstacle_avoidance_2_method_combo.grid(row=1, column=3, sticky="ew", padx=6, pady=5)
        tk.Label(project, text="Status").grid(row=2, column=0, sticky="nw", padx=6, pady=5)
        tk.Label(project, textvariable=self.obstacle_avoidance_2_status_var, anchor="w", justify="left", wraplength=980).grid(
            row=2, column=1, columnspan=7, sticky="ew", padx=6, pady=5
        )

        experiment = tk.LabelFrame(window, text="Start-Goal Experiment")
        experiment.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(experiment, text="1. Locate Start", command=self.locate_obstacle_avoidance_2_start).pack(side="left", padx=6, pady=6)
        tk.Button(experiment, text="2. Apply Algorithm", command=self.apply_obstacle_avoidance_2_algorithm).pack(side="left", padx=6, pady=6)
        tk.Button(experiment, text="3. Execute To Goal", command=self.execute_obstacle_avoidance_2_to_goal).pack(side="left", padx=6, pady=6)
        tk.Button(
            experiment,
            text="MAX STOP ALL",
            command=self.emergency_stop_obstacle_avoidance_2_all,
            bg="#b00020",
            fg="white",
            activebackground="#7a0016",
            activeforeground="white",
        ).pack(side="left", padx=(18, 6), pady=6)
        tk.Label(experiment, textvariable=self.obstacle_avoidance_2_goal_distance_var, anchor="w", width=24).pack(side="left", padx=(18, 6), pady=6)
        tk.Label(experiment, textvariable=self.obstacle_avoidance_2_goal_reached_var, anchor="w", width=22).pack(side="left", padx=(6, 6), pady=6)
        tk.Label(experiment, textvariable=self.obstacle_avoidance_2_impact_var, anchor="w").pack(side="left", padx=(18, 6), pady=6)

        body = tk.Frame(window)
        body.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        body.grid_columnconfigure(0, weight=4)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        tree_frame = tk.LabelFrame(body, text="Episodes")
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        columns = ("enabled", "episode", "start", "goal", "environment", "method", "hint", "note")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended", height=12)
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
        widths = {
            "enabled": 44,
            "episode": 56,
            "start": 170,
            "goal": 170,
            "environment": 135,
            "method": 160,
            "hint": 90,
            "note": 150,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w", stretch=True)
        tree_y = tk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree_x = tk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        tree.bind("<<TreeviewSelect>>", lambda _event: self.on_obstacle_avoidance_2_episode_selected())
        self.obstacle_avoidance_2_episode_tree = tree

        side = tk.Frame(body)
        side.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1)

        editor = tk.LabelFrame(side, text="Episode Editor")
        editor.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        editor.grid_columnconfigure(1, weight=1)
        tk.Checkbutton(editor, text="Enabled", variable=self.obstacle_avoidance_2_enabled_var).grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(editor, textvariable=self.obstacle_avoidance_2_episode_id_var, width=12).grid(row=0, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(editor, text="Start Pose").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(editor, textvariable=self.obstacle_avoidance_2_start_pose_var).grid(row=1, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(editor, text="Goal Pose").grid(row=2, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(editor, textvariable=self.obstacle_avoidance_2_goal_pose_var).grid(row=2, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(editor, text="Scenario").grid(row=3, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(editor, textvariable=self.obstacle_avoidance_2_scenario_var).grid(row=3, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(editor, text="Obstacle Hint").grid(row=4, column=0, sticky="w", padx=6, pady=5)
        ttk.Combobox(
            editor,
            textvariable=self.obstacle_avoidance_2_obstacle_hint_var,
            values=(
                "unknown",
                "tree_trunk_or_pole",
                "tree_canopy_or_cluster",
                "tree",
                "pole",
                "fence_or_rail",
                "building",
                "roof",
                "low_obstacle",
                "mixed",
            ),
        ).grid(row=4, column=1, sticky="ew", padx=6, pady=5)
        tk.Label(editor, text="Note").grid(row=5, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(editor, textvariable=self.obstacle_avoidance_2_note_var).grid(row=5, column=1, sticky="ew", padx=6, pady=5)
        tk.Button(editor, text="Apply Episode", command=self.apply_obstacle_avoidance_2_episode).grid(row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=(12, 5))
        tk.Button(editor, text="Add Episode", command=self.add_obstacle_avoidance_2_episode).grid(row=7, column=0, sticky="ew", padx=6, pady=5)
        tk.Button(editor, text="Delete Episode", command=self.delete_obstacle_avoidance_2_episode).grid(row=7, column=1, sticky="ew", padx=6, pady=5)

        records = tk.LabelFrame(side, text="Episode Records")
        records.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        records.grid_columnconfigure(0, weight=1)
        tk.Label(records, text="Record").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        self.obstacle_avoidance_2_record_combo = ttk.Combobox(
            records,
            textvariable=self.obstacle_avoidance_2_record_var,
            state="readonly",
            width=34,
        )
        self.obstacle_avoidance_2_record_combo.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=2)
        self.obstacle_avoidance_2_record_combo.bind("<<ComboboxSelected>>", lambda _event: self.view_obstacle_avoidance_2_selected_record())
        tk.Label(
            records,
            textvariable=self.obstacle_avoidance_2_record_summary_var,
            justify="left",
            anchor="nw",
            wraplength=260,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=5)
        tk.Button(records, text="Reload Records", command=self.refresh_obstacle_avoidance_2_records).grid(
            row=3, column=0, sticky="ew", padx=6, pady=5
        )
        tk.Button(records, text="View Record", command=self.view_obstacle_avoidance_2_selected_record).grid(
            row=3, column=1, sticky="ew", padx=6, pady=5
        )
        tk.Button(records, text="Run Current Episode", command=self.run_obstacle_avoidance_2_current_episode).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 6)
        )

        run_frame = tk.LabelFrame(window, text="Collection Runner")
        run_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(run_frame, text="Export Selected JSON", command=self.export_obstacle_avoidance_2_selected_json).pack(side="left", padx=6, pady=6)
        tk.Button(run_frame, text="Dry Run Selected", command=self.dry_run_obstacle_avoidance_2_selected).pack(side="left", padx=6, pady=6)
        tk.Button(run_frame, text="Run Selected Episodes", command=self.run_obstacle_avoidance_2_selected).pack(side="left", padx=6, pady=6)
        tk.Button(run_frame, text="Run Current Episode", command=self.run_obstacle_avoidance_2_current_episode).pack(side="left", padx=6, pady=6)
        tk.Button(run_frame, text="Stop Run", command=self.stop_obstacle_avoidance_2_runner).pack(side="left", padx=(18, 6), pady=6)
        tk.Button(run_frame, text="Reset Runner", command=self.reset_obstacle_avoidance_2_runner).pack(side="left", padx=6, pady=6)
        tk.Button(
            run_frame,
            text="Emergency Stop All",
            command=self.emergency_stop_obstacle_avoidance_2_all,
            bg="#b00020",
            fg="white",
            activebackground="#7a0016",
            activeforeground="white",
        ).pack(side="left", padx=(18, 6), pady=6)

        report_frame = tk.LabelFrame(window, text="Report")
        report_frame.grid(row=5, column=0, sticky="nsew", padx=8, pady=(4, 8))
        report_frame.grid_columnconfigure(0, weight=1)
        report_frame.grid_rowconfigure(0, weight=1)
        report = tk.Text(report_frame, height=7, wrap="none", font=("Consolas", 9))
        report_y = tk.Scrollbar(report_frame, orient="vertical", command=report.yview)
        report_x = tk.Scrollbar(report_frame, orient="horizontal", command=report.xview)
        report.configure(yscrollcommand=report_y.set, xscrollcommand=report_x.set)
        report.grid(row=0, column=0, sticky="nsew")
        report_y.grid(row=0, column=1, sticky="ns")
        report_x.grid(row=1, column=0, sticky="ew")
        self.obstacle_avoidance_2_report_text = report

    def obstacle_avoidance_2_report(self, payload: Any) -> None:
        if isinstance(payload, str):
            text = payload.rstrip()
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        widget = self.obstacle_avoidance_2_report_text
        if widget is not None:
            try:
                widget.insert("end", text + "\n\n")
                widget.see("end")
            except tk.TclError:
                pass

    def obstacle_avoidance_2_set_report(self, payload: Any) -> None:
        widget = self.obstacle_avoidance_2_report_text
        if widget is not None:
            try:
                widget.delete("1.0", "end")
            except tk.TclError:
                pass
        self.obstacle_avoidance_2_report(payload)

    def read_obstacle_avoidance_2_json(self, path: Path) -> Dict[str, Any]:
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
        except Exception:
            return {}
        return {}

    def infer_obstacle_avoidance_2_episode_id(self, session_dir: Path, summary: Dict[str, Any], config: Dict[str, Any]) -> str:
        episode_id = str(summary.get("episode_id", "") or "").strip()
        if episode_id:
            return episode_id
        episode = config.get("episode") if isinstance(config.get("episode"), dict) else {}
        episode_id = str(episode.get("episode_id", "") or "").strip()
        if episode_id:
            return episode_id
        match = re.search(r"_(E[0-9A-Za-z-]+)_", session_dir.name)
        return match.group(1) if match else ""

    def scan_obstacle_avoidance_2_records(self, episode_id: str) -> List[Dict[str, Any]]:
        wanted = str(episode_id or "").strip()
        sessions_dir = self.obstacle_avoidance_2_data_root() / "sessions"
        if not wanted or not sessions_dir.exists():
            return []
        records: List[Dict[str, Any]] = []
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            summary = self.read_obstacle_avoidance_2_json(session_dir / "episode_summary.json")
            config = self.read_obstacle_avoidance_2_json(session_dir / "episode_config.json")
            found_id = self.infer_obstacle_avoidance_2_episode_id(session_dir, summary, config)
            if found_id != wanted:
                continue
            args = config.get("args") if isinstance(config.get("args"), dict) else {}
            episode = config.get("episode") if isinstance(config.get("episode"), dict) else {}
            method = str(summary.get("method", "") or args.get("method", "") or episode.get("method", "") or "")
            run_id = str(args.get("run_id", "") or "")
            try:
                mtime = session_dir.stat().st_mtime
            except OSError:
                mtime = 0.0
            finished_at = str(summary.get("finished_at", "") or summary.get("updated_at", "") or "")
            if not finished_at and mtime > 0:
                finished_at = datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
            distance = summary.get("final_distance_to_goal_cm", "")
            records.append(
                {
                    "episode_id": found_id,
                    "session_dir": str(session_dir),
                    "summary_path": str(session_dir / "episode_summary.json"),
                    "events_path": str(session_dir / "avoidance_events.jsonl"),
                    "quality_report_path": str(session_dir / "collection_quality_report.json"),
                    "outcome": str(summary.get("outcome", "unknown")),
                    "frame_count": summary.get("frame_count", 0),
                    "final_distance_to_goal_cm": distance,
                    "method": method,
                    "run_id": run_id,
                    "finished_at": finished_at,
                    "mtime": mtime,
                    "summary": summary,
                }
            )
        records.sort(key=lambda item: float(item.get("mtime", 0.0) or 0.0), reverse=True)
        for index, record in enumerate(records, start=1):
            finished = str(record.get("finished_at", "")).replace("T", " ")[:19]
            distance = record.get("final_distance_to_goal_cm", "")
            if isinstance(distance, (int, float)):
                distance_text = f"{float(distance):.1f}cm"
            else:
                distance_text = str(distance or "?")
            record["display"] = (
                f"{index:02d} {finished} | {record.get('outcome', 'unknown')} | "
                f"d={distance_text} | {record.get('method', '') or 'method?'}"
            )
        return records

    def refresh_obstacle_avoidance_2_records(self, episode_id: Optional[str] = None) -> None:
        episode_id = str(episode_id or self.obstacle_avoidance_2_episode_id_var.get() or "").strip()
        records = self.scan_obstacle_avoidance_2_records(episode_id)
        self.obstacle_avoidance_2_record_entries = records
        values = tuple(str(record.get("display", "")) for record in records)
        if self.obstacle_avoidance_2_record_combo is not None:
            try:
                self.obstacle_avoidance_2_record_combo.configure(values=values)
            except tk.TclError:
                pass
        if not episode_id:
            self.obstacle_avoidance_2_record_var.set("")
            self.obstacle_avoidance_2_record_summary_var.set("Select an episode to inspect records.")
            return
        if not records:
            self.obstacle_avoidance_2_record_var.set("")
            self.obstacle_avoidance_2_record_summary_var.set(
                f"{episode_id}: no run records yet.\nUse Run Current Episode to create one with the current start/goal/method."
            )
            self.obstacle_avoidance_2_status_var.set(f"OA2 {episode_id}: no records")
            return
        current = str(self.obstacle_avoidance_2_record_var.get() or "")
        if current not in values:
            self.obstacle_avoidance_2_record_var.set(values[0])
        self.update_obstacle_avoidance_2_record_summary()
        self.obstacle_avoidance_2_status_var.set(f"OA2 {episode_id}: {len(records)} record(s)")

    def selected_obstacle_avoidance_2_record(self) -> Optional[Dict[str, Any]]:
        selected = str(self.obstacle_avoidance_2_record_var.get() or "")
        for record in self.obstacle_avoidance_2_record_entries:
            if str(record.get("display", "")) == selected:
                return record
        if self.obstacle_avoidance_2_record_entries:
            return self.obstacle_avoidance_2_record_entries[0]
        return None

    def update_obstacle_avoidance_2_record_summary(self) -> None:
        record = self.selected_obstacle_avoidance_2_record()
        if record is None:
            return
        self.obstacle_avoidance_2_record_summary_var.set(
            "\n".join(
                [
                    f"Episode: {record.get('episode_id', '')}",
                    f"Outcome: {record.get('outcome', '')}",
                    f"Frames: {record.get('frame_count', 0)}",
                    f"Final distance: {record.get('final_distance_to_goal_cm', '?')} cm",
                    f"Collision: {record.get('summary', {}).get('had_collision', False)}",
                    f"Impact count: {record.get('summary', {}).get('impact_count', 0)}",
                    f"Session: {Path(str(record.get('session_dir', ''))).name}",
                ]
            )
        )

    def view_obstacle_avoidance_2_selected_record(self) -> None:
        self.update_obstacle_avoidance_2_record_summary()
        record = self.selected_obstacle_avoidance_2_record()
        if record is None:
            episode_id = str(self.obstacle_avoidance_2_episode_id_var.get() or "").strip()
            self.obstacle_avoidance_2_set_report(
                {
                    "status": "no_record",
                    "episode_id": episode_id,
                    "hint": "Run Current Episode will collect a new run record for the current start/goal/method.",
                }
            )
            return
        self.obstacle_avoidance_2_set_report(
            {
                "status": "record_selected",
                "episode_id": record.get("episode_id", ""),
                "display": record.get("display", ""),
                "session_dir": record.get("session_dir", ""),
                "events_path": record.get("events_path", ""),
                "quality_report_path": record.get("quality_report_path", ""),
                "summary": record.get("summary", {}),
            }
        )

    def read_obstacle_avoidance_2_hit_state(self, session: flight.DroneFlightSession) -> Dict[str, Any]:
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

    def notify_obstacle_avoidance_2_collision(
        self,
        *,
        episode_id: str,
        frame_id: int,
        session_dir: str,
        impact_detail: Dict[str, Any],
    ) -> None:
        key = f"{episode_id}:{session_dir}"
        if key in self.obstacle_avoidance_2_collision_alert_keys:
            return
        self.obstacle_avoidance_2_collision_alert_keys.add(key)
        reason = str(impact_detail.get("reason", "collision detected") if isinstance(impact_detail, dict) else "collision detected")
        message = f"OA2 collision detected: {episode_id} frame {frame_id}\n{reason}"
        self.obstacle_avoidance_2_impact_var.set(f"Impact: YES ({episode_id} F{frame_id})")
        self.obstacle_avoidance_2_status_var.set(message.replace("\n", " | "))
        self.obstacle_avoidance_2_report(
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
            messagebox.showwarning("OA2 Collision Alert", message)
        except Exception:
            pass

    def select_obstacle_avoidance_2_data_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select obstacle_avoidance_2_data folder", initialdir=str(self.obstacle_avoidance_2_data_root()))
        if selected:
            self.obstacle_avoidance_2_data_dir_var.set(selected)
            self.obstacle_avoidance_2_plan_json_var.set(str(Path(selected) / "plans" / DEFAULT_PLAN_FILENAME))
            self.refresh_obstacle_avoidance_2_records()

    def select_obstacle_avoidance_2_plan_json(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select OA2 plan JSON",
            initialdir=str(self.obstacle_avoidance_2_plan_path().parent),
            initialfile=self.obstacle_avoidance_2_plan_path().name,
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if selected:
            self.obstacle_avoidance_2_plan_json_var.set(selected)
            self.load_obstacle_avoidance_2_plans()

    def load_obstacle_avoidance_2_plans(self) -> None:
        plan_path = self.obstacle_avoidance_2_plan_path()
        if not plan_path.exists():
            save_plans(plan_path, make_default_plans())
        self.obstacle_avoidance_2_plan_data = load_plans(plan_path)
        self.obstacle_avoidance_2_project_var.set(str(self.obstacle_avoidance_2_plan_data.get("active_project_id", DEFAULT_PROJECT_ID)))
        self.refresh_obstacle_avoidance_2_options()
        self.select_obstacle_avoidance_2_project()
        self.obstacle_avoidance_2_status_var.set(f"OA2 plans loaded: {plan_path}")
        self.obstacle_avoidance_2_report({"status": "loaded", "plan_path": str(plan_path)})

    def save_obstacle_avoidance_2_plans(self) -> None:
        if self.obstacle_avoidance_2_plan_data is None:
            self.obstacle_avoidance_2_plan_data = make_default_plans()
        self.apply_obstacle_avoidance_2_project(silent=True)
        self.apply_obstacle_avoidance_2_episode(silent=True)
        save_plans(self.obstacle_avoidance_2_plan_path(), self.obstacle_avoidance_2_plan_data)
        self.obstacle_avoidance_2_status_var.set("OA2 plans saved")
        self.obstacle_avoidance_2_report({"status": "saved", "plan_path": str(self.obstacle_avoidance_2_plan_path())})

    def reset_obstacle_avoidance_2_plans(self) -> None:
        if not messagebox.askyesno("Reset OA2 Plans", "Reset OA2 plan JSON to the default 10 start-goal episodes?"):
            return
        self.obstacle_avoidance_2_plan_data = make_default_plans()
        save_plans(self.obstacle_avoidance_2_plan_path(), self.obstacle_avoidance_2_plan_data)
        self.refresh_obstacle_avoidance_2_options()
        self.select_obstacle_avoidance_2_project()
        self.obstacle_avoidance_2_status_var.set("OA2 reset to default 10 points")
        self.obstacle_avoidance_2_report({"status": "reset_default", "plan_path": str(self.obstacle_avoidance_2_plan_path())})

    def refresh_obstacle_avoidance_2_options(self) -> None:
        data = self.obstacle_avoidance_2_plan_data if isinstance(self.obstacle_avoidance_2_plan_data, dict) else make_default_plans()
        projects = [str(project.get("project_id", "")) for project in data.get("projects", []) if str(project.get("project_id", ""))]
        environments = [str(item.get("environment_id", "")) for item in data.get("environments", []) if str(item.get("environment_id", ""))]
        methods = [str(item.get("method_id", "")) for item in data.get("methods", []) if str(item.get("method_id", ""))]
        if self.obstacle_avoidance_2_project_combo is not None:
            self.obstacle_avoidance_2_project_combo.configure(values=tuple(projects))
        if self.obstacle_avoidance_2_environment_combo is not None:
            self.obstacle_avoidance_2_environment_combo.configure(values=tuple(environments))
        if self.obstacle_avoidance_2_method_combo is not None:
            self.obstacle_avoidance_2_method_combo.configure(values=tuple(methods))

    def current_obstacle_avoidance_2_project(self) -> Dict[str, Any]:
        if self.obstacle_avoidance_2_plan_data is None:
            self.load_obstacle_avoidance_2_plans()
        data = self.obstacle_avoidance_2_plan_data if isinstance(self.obstacle_avoidance_2_plan_data, dict) else make_default_plans()
        try:
            return project_by_id(data, str(self.obstacle_avoidance_2_project_var.get() or data.get("active_project_id", DEFAULT_PROJECT_ID)))
        except Exception:
            project = data["projects"][0]
            self.obstacle_avoidance_2_project_var.set(str(project.get("project_id", DEFAULT_PROJECT_ID)))
            return project

    def select_obstacle_avoidance_2_project(self) -> None:
        project = self.current_obstacle_avoidance_2_project()
        if isinstance(self.obstacle_avoidance_2_plan_data, dict):
            self.obstacle_avoidance_2_plan_data["active_project_id"] = str(project.get("project_id", DEFAULT_PROJECT_ID))
        self.obstacle_avoidance_2_project_name_var.set(str(project.get("name", "")))
        self.obstacle_avoidance_2_environment_var.set(str(project.get("environment_id", DEFAULT_ENVIRONMENT_ID)))
        self.obstacle_avoidance_2_method_var.set(str(project.get("default_method", DEFAULT_METHOD_ID)))
        self.refresh_obstacle_avoidance_2_tree()
        self.refresh_obstacle_avoidance_2_records()

    def apply_obstacle_avoidance_2_project(self, *, silent: bool = False) -> None:
        project = self.current_obstacle_avoidance_2_project()
        project["name"] = str(self.obstacle_avoidance_2_project_name_var.get() or project.get("project_id", "")).strip()
        project["environment_id"] = str(self.obstacle_avoidance_2_environment_var.get() or DEFAULT_ENVIRONMENT_ID).strip()
        project["default_method"] = str(self.obstacle_avoidance_2_method_var.get() or DEFAULT_METHOD_ID).strip()
        if isinstance(self.obstacle_avoidance_2_plan_data, dict):
            self.obstacle_avoidance_2_plan_data["active_project_id"] = str(project.get("project_id", DEFAULT_PROJECT_ID))
        if not silent:
            self.refresh_obstacle_avoidance_2_tree()
            self.obstacle_avoidance_2_status_var.set("OA2 project applied")

    def unique_obstacle_avoidance_2_project_id(self, base: str) -> str:
        data = self.obstacle_avoidance_2_plan_data if isinstance(self.obstacle_avoidance_2_plan_data, dict) else make_default_plans()
        existing = {str(project.get("project_id", "")) for project in data.get("projects", [])}
        root = sanitize_id(base, "oa2_project")
        candidate = root
        index = 2
        while candidate in existing:
            candidate = f"{root}_{index}"
            index += 1
        return candidate

    def new_obstacle_avoidance_2_project(self) -> None:
        if self.obstacle_avoidance_2_plan_data is None:
            self.load_obstacle_avoidance_2_plans()
        name = str(self.obstacle_avoidance_2_project_name_var.get() or "OA2 new collection project")
        project_id = self.unique_obstacle_avoidance_2_project_id(name)
        project = {
            "project_id": project_id,
            "name": name,
            "environment_id": str(self.obstacle_avoidance_2_environment_var.get() or DEFAULT_ENVIRONMENT_ID),
            "default_method": str(self.obstacle_avoidance_2_method_var.get() or DEFAULT_METHOD_ID),
            "episodes": [],
            "experiment_defaults": make_default_plans()["projects"][0]["experiment_defaults"],
        }
        self.obstacle_avoidance_2_plan_data.setdefault("projects", []).append(project)
        self.obstacle_avoidance_2_project_var.set(project_id)
        self.refresh_obstacle_avoidance_2_options()
        self.select_obstacle_avoidance_2_project()
        self.obstacle_avoidance_2_status_var.set(f"OA2 project created: {project_id}")

    def copy_obstacle_avoidance_2_project(self) -> None:
        project = deepcopy(self.current_obstacle_avoidance_2_project())
        old_id = str(project.get("project_id", "oa2_project"))
        project["project_id"] = self.unique_obstacle_avoidance_2_project_id(f"{old_id}_copy")
        project["name"] = f"{project.get('name', old_id)} copy"
        self.obstacle_avoidance_2_plan_data.setdefault("projects", []).append(project)
        self.obstacle_avoidance_2_project_var.set(str(project["project_id"]))
        self.refresh_obstacle_avoidance_2_options()
        self.select_obstacle_avoidance_2_project()
        self.obstacle_avoidance_2_status_var.set(f"OA2 project copied: {old_id}")

    def delete_obstacle_avoidance_2_project(self) -> None:
        data = self.obstacle_avoidance_2_plan_data
        if not isinstance(data, dict) or len(data.get("projects", [])) <= 1:
            self.obstacle_avoidance_2_status_var.set("OA2 cannot delete the last project")
            return
        project = self.current_obstacle_avoidance_2_project()
        project_id = str(project.get("project_id", ""))
        if not messagebox.askyesno("Delete OA2 Project", f"Delete OA2 project {project_id}?"):
            return
        data["projects"] = [item for item in data.get("projects", []) if str(item.get("project_id", "")) != project_id]
        self.obstacle_avoidance_2_project_var.set(str(data["projects"][0].get("project_id", DEFAULT_PROJECT_ID)))
        self.refresh_obstacle_avoidance_2_options()
        self.select_obstacle_avoidance_2_project()
        self.obstacle_avoidance_2_status_var.set(f"OA2 project deleted: {project_id}")

    def format_obstacle_avoidance_2_pose(self, pose: Any) -> str:
        try:
            return ", ".join(f"{value:g}" for value in coerce_pose(pose))
        except Exception:
            return ""

    def refresh_obstacle_avoidance_2_tree(self) -> None:
        tree = self.obstacle_avoidance_2_episode_tree
        if tree is None:
            return
        previous_ids = [
            self.obstacle_avoidance_2_tree_iids.get(str(iid), "")
            for iid in tree.selection()
        ]
        previous_ids = [item for item in previous_ids if item]
        previous_id = str(self.obstacle_avoidance_2_episode_id_var.get() or "")
        if previous_id and previous_id not in previous_ids:
            previous_ids.append(previous_id)
        for iid in tree.get_children():
            tree.delete(iid)
        self.obstacle_avoidance_2_tree_iids = {}
        project = self.current_obstacle_avoidance_2_project()
        first_iid = ""
        selected_iids: List[str] = []
        for index, episode in enumerate(project.get("episodes", []), start=1):
            episode_id = str(episode.get("episode_id", f"E{index:02d}"))
            iid = f"{episode_id}::{index}"
            self.obstacle_avoidance_2_tree_iids[iid] = episode_id
            if not first_iid:
                first_iid = iid
            if episode_id in previous_ids:
                selected_iids.append(iid)
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    "yes" if bool(episode.get("enabled", True)) else "no",
                    episode_id,
                    self.format_obstacle_avoidance_2_pose(episode.get("start_pose")),
                    self.format_obstacle_avoidance_2_pose(episode.get("goal_pose")),
                    str(episode.get("environment_id", project.get("environment_id", DEFAULT_ENVIRONMENT_ID))),
                    str(episode.get("method", project.get("default_method", DEFAULT_METHOD_ID))),
                    str(episode.get("obstacle_hint", "unknown")),
                    str(episode.get("operator_note", "")),
                ),
            )
        target_iids = selected_iids or ([first_iid] if first_iid else [])
        if target_iids:
            try:
                tree.selection_set(target_iids)
                tree.see(target_iids[0])
                self.on_obstacle_avoidance_2_episode_selected()
            except tk.TclError:
                pass

    def selected_obstacle_avoidance_2_episode_ids(self) -> List[str]:
        tree = self.obstacle_avoidance_2_episode_tree
        if tree is None:
            return []
        ids: List[str] = []
        for iid in tree.selection():
            episode_id = self.obstacle_avoidance_2_tree_iids.get(str(iid), "")
            if episode_id:
                ids.append(episode_id)
        return ids

    def find_obstacle_avoidance_2_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        project = self.current_obstacle_avoidance_2_project()
        for episode in project.get("episodes", []):
            if str(episode.get("episode_id", "")) == str(episode_id):
                return episode
        return None

    def on_obstacle_avoidance_2_episode_selected(self) -> None:
        selected = self.selected_obstacle_avoidance_2_episode_ids()
        if not selected:
            return
        episode = self.find_obstacle_avoidance_2_episode(selected[0])
        if not isinstance(episode, dict):
            return
        self.obstacle_avoidance_2_episode_id_var.set(str(episode.get("episode_id", "")))
        self.obstacle_avoidance_2_enabled_var.set(bool(episode.get("enabled", True)))
        self.obstacle_avoidance_2_start_pose_var.set(self.format_obstacle_avoidance_2_pose(episode.get("start_pose")))
        self.obstacle_avoidance_2_goal_pose_var.set(self.format_obstacle_avoidance_2_pose(episode.get("goal_pose")))
        self.obstacle_avoidance_2_scenario_var.set(str(episode.get("scenario_id", "")))
        self.obstacle_avoidance_2_environment_var.set(str(episode.get("environment_id", self.obstacle_avoidance_2_environment_var.get())))
        self.obstacle_avoidance_2_method_var.set(str(episode.get("method", self.obstacle_avoidance_2_method_var.get())))
        self.obstacle_avoidance_2_obstacle_hint_var.set(str(episode.get("obstacle_hint", "unknown")))
        self.obstacle_avoidance_2_note_var.set(str(episode.get("operator_note", "")))
        self.refresh_obstacle_avoidance_2_records(str(episode.get("episode_id", "")))

    def apply_obstacle_avoidance_2_episode(self, *, silent: bool = False) -> None:
        episode_id = str(self.obstacle_avoidance_2_episode_id_var.get() or "").strip()
        if not episode_id:
            if not silent:
                self.obstacle_avoidance_2_status_var.set("OA2 select or enter an episode id")
            return
        episode = self.find_obstacle_avoidance_2_episode(episode_id)
        if episode is None:
            if not silent:
                self.obstacle_avoidance_2_status_var.set(f"OA2 episode not found: {episode_id}")
            return
        try:
            episode["start_pose"] = coerce_pose(self.obstacle_avoidance_2_start_pose_var.get())
            episode["goal_pose"] = coerce_pose(self.obstacle_avoidance_2_goal_pose_var.get())
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 invalid pose: {exc}")
            return
        episode["enabled"] = bool(self.obstacle_avoidance_2_enabled_var.get())
        episode["scenario_id"] = str(self.obstacle_avoidance_2_scenario_var.get() or f"oa2_route_{episode_id}").strip()
        episode["environment_id"] = str(self.obstacle_avoidance_2_environment_var.get() or DEFAULT_ENVIRONMENT_ID).strip()
        episode["method"] = str(self.obstacle_avoidance_2_method_var.get() or DEFAULT_METHOD_ID).strip()
        episode["obstacle_hint"] = str(self.obstacle_avoidance_2_obstacle_hint_var.get() or "unknown").strip()
        episode["operator_note"] = str(self.obstacle_avoidance_2_note_var.get() or "").strip()
        errors = validate_plan_episode(episode)
        if errors:
            self.obstacle_avoidance_2_status_var.set("; ".join(errors))
            return
        self.refresh_obstacle_avoidance_2_tree()
        if not silent:
            self.obstacle_avoidance_2_status_var.set(f"OA2 episode applied: {episode_id}")

    def next_obstacle_avoidance_2_episode_id(self) -> str:
        project = self.current_obstacle_avoidance_2_project()
        existing = {str(episode.get("episode_id", "")) for episode in project.get("episodes", [])}
        index = len(existing) + 1
        while True:
            candidate = f"E{index:02d}"
            if candidate not in existing:
                return candidate
            index += 1

    def add_obstacle_avoidance_2_episode(self) -> None:
        project = self.current_obstacle_avoidance_2_project()
        episode_id = str(self.obstacle_avoidance_2_episode_id_var.get() or "").strip() or self.next_obstacle_avoidance_2_episode_id()
        if self.find_obstacle_avoidance_2_episode(episode_id) is not None:
            episode_id = self.next_obstacle_avoidance_2_episode_id()
        try:
            start_pose = coerce_pose(self.obstacle_avoidance_2_start_pose_var.get())
            goal_pose = coerce_pose(self.obstacle_avoidance_2_goal_pose_var.get())
        except Exception:
            start_pose = [0.0, 0.0, 200.0, 0.0]
            goal_pose = [300.0, 0.0, 200.0, 0.0]
        project.setdefault("episodes", []).append(
            {
                "episode_id": episode_id,
                "enabled": bool(self.obstacle_avoidance_2_enabled_var.get()),
                "start_pose": start_pose,
                "goal_pose": goal_pose,
                "scenario_id": str(self.obstacle_avoidance_2_scenario_var.get() or f"oa2_route_{episode_id}"),
                "environment_id": str(self.obstacle_avoidance_2_environment_var.get() or project.get("environment_id", DEFAULT_ENVIRONMENT_ID)),
                "method": str(self.obstacle_avoidance_2_method_var.get() or project.get("default_method", DEFAULT_METHOD_ID)),
                "obstacle_hint": str(self.obstacle_avoidance_2_obstacle_hint_var.get() or "unknown"),
                "operator_note": str(self.obstacle_avoidance_2_note_var.get() or ""),
            }
        )
        self.obstacle_avoidance_2_episode_id_var.set(episode_id)
        self.refresh_obstacle_avoidance_2_tree()
        self.obstacle_avoidance_2_status_var.set(f"OA2 episode added: {episode_id}")

    def delete_obstacle_avoidance_2_episode(self) -> None:
        selected = self.selected_obstacle_avoidance_2_episode_ids()
        if not selected:
            selected = [str(self.obstacle_avoidance_2_episode_id_var.get() or "").strip()]
        selected = [item for item in selected if item]
        if not selected:
            self.obstacle_avoidance_2_status_var.set("OA2 select episode(s) to delete")
            return
        if not messagebox.askyesno("Delete OA2 Episodes", f"Delete {len(selected)} selected OA2 episode(s)?"):
            return
        project = self.current_obstacle_avoidance_2_project()
        wanted = set(selected)
        project["episodes"] = [episode for episode in project.get("episodes", []) if str(episode.get("episode_id", "")) not in wanted]
        self.refresh_obstacle_avoidance_2_tree()
        self.obstacle_avoidance_2_status_var.set(f"OA2 deleted {len(selected)} episode(s)")

    def obstacle_avoidance_2_export_path(self, purpose: str) -> Path:
        project = self.current_obstacle_avoidance_2_project()
        project_id = sanitize_id(project.get("project_id", DEFAULT_PROJECT_ID), DEFAULT_PROJECT_ID)
        method = sanitize_id(self.obstacle_avoidance_2_method_var.get() or DEFAULT_METHOD_ID, DEFAULT_METHOD_ID)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.obstacle_avoidance_2_data_root() / "plans" / purpose / f"{stamp}_{project_id}_{method}_episodes.json"

    def export_obstacle_avoidance_2_selected_json(self) -> Optional[Path]:
        self.apply_obstacle_avoidance_2_project(silent=True)
        self.apply_obstacle_avoidance_2_episode(silent=True)
        project = self.current_obstacle_avoidance_2_project()
        selected_ids = self.selected_obstacle_avoidance_2_episode_ids()
        try:
            path = export_selected_episodes(project, selected_ids, self.obstacle_avoidance_2_export_path("exports"))
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 export failed: {exc}")
            self.obstacle_avoidance_2_report({"status": "error", "task": "export", "error": str(exc)})
            return None
        self.obstacle_avoidance_2_status_var.set(f"OA2 exported: {path}")
        self.obstacle_avoidance_2_report({"status": "exported", "episodes_json": str(path), "selected_episode_ids": selected_ids})
        return path

    def dry_run_obstacle_avoidance_2_selected(self) -> None:
        prepared = self.prepare_obstacle_avoidance_2_run()
        if prepared is None:
            return
        _project, selected_ids, episodes_json, method, run_id, note = prepared
        try:
            episodes = self.resolve_obstacle_avoidance_2_episodes(episodes_json)
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 dry-run failed: {exc}")
            self.obstacle_avoidance_2_report({"status": "error", "task": "dry_run", "error": str(exc)})
            return
        self.obstacle_avoidance_2_status_var.set(f"OA2 dry-run done: {len(episodes)} episode(s)")
        self.obstacle_avoidance_2_report(
            {
                "status": "dry_run_done",
                "episodes_json": str(episodes_json),
                "method": method,
                "run_id": run_id,
                "note": note,
                "selected_episode_ids": selected_ids,
                "count": len(episodes),
                "episodes": episodes,
            }
        )

    def locate_obstacle_avoidance_2_start(self) -> None:
        if self.obstacle_avoidance_2_runner_thread is not None and self.obstacle_avoidance_2_runner_thread.is_alive():
            self.obstacle_avoidance_2_status_var.set("OA2 runner is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_2_status_var.set("OA2: start Unreal from the main window first")
            return
        episode = self.upsert_obstacle_avoidance_2_editor_episode()
        if episode is None:
            return
        start = pose_dict(episode["start_pose"])
        goal = pose_dict(episode["goal_pose"])
        try:
            response = session.set_pose({"x": start["x"], "y": start["y"], "z": start["z"], "yaw": start["yaw"]})
            mode_result = session.set_movement_mode("physics")
            enable_result = session.set_movement_enabled(True)
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 locate start failed: {exc}")
            self.obstacle_avoidance_2_report({"status": "error", "task": "locate_start", "error": str(exc)})
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
        self.obstacle_avoidance_2_impact_var.set("Impact: no")
        start_distance = distance_3d_cm(start, goal)
        self.obstacle_avoidance_2_goal_distance_var.set(f"Goal distance: {start_distance:.1f} cm")
        self.obstacle_avoidance_2_goal_reached_var.set("Goal reached: no")
        self.obstacle_avoidance_2_status_var.set(f"OA2 located start with physics: {episode['episode_id']} dist={start_distance:.1f}cm")
        self.obstacle_avoidance_2_report(
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

    def apply_obstacle_avoidance_2_algorithm(self) -> None:
        if self.obstacle_avoidance_2_plan_data is None:
            self.load_obstacle_avoidance_2_plans()
        method = str(self.obstacle_avoidance_2_method_var.get() or DEFAULT_METHOD_ID).strip()
        project = self.current_obstacle_avoidance_2_project()
        project["default_method"] = method
        episode = self.upsert_obstacle_avoidance_2_editor_episode()
        if episode is None:
            return
        data = self.obstacle_avoidance_2_plan_data if isinstance(self.obstacle_avoidance_2_plan_data, dict) else make_default_plans()
        runnable = method_is_runnable(data, method)
        status = "executable" if runnable else "metadata only"
        self.obstacle_avoidance_2_status_var.set(f"OA2 algorithm selected: {method} ({status})")
        self.obstacle_avoidance_2_report(
            {
                "status": "algorithm_selected",
                "episode_id": episode.get("episode_id", ""),
                "method": method,
                "runnable": runnable,
                "hint": "" if runnable else "This method is saved as metadata, but no runner is implemented yet.",
            }
        )

    def execute_obstacle_avoidance_2_to_goal(self) -> None:
        self.run_obstacle_avoidance_2_current_episode()

    def build_obstacle_avoidance_2_editor_episode(self) -> Optional[Dict[str, Any]]:
        episode_id = str(self.obstacle_avoidance_2_episode_id_var.get() or "").strip()
        if not episode_id:
            episode_id = self.next_obstacle_avoidance_2_episode_id()
            self.obstacle_avoidance_2_episode_id_var.set(episode_id)
        try:
            start_pose = coerce_pose(self.obstacle_avoidance_2_start_pose_var.get())
            goal_pose = coerce_pose(self.obstacle_avoidance_2_goal_pose_var.get())
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 invalid current episode pose: {exc}")
            return None
        project = self.current_obstacle_avoidance_2_project()
        episode = {
            "episode_id": episode_id,
            "enabled": bool(self.obstacle_avoidance_2_enabled_var.get()),
            "start_pose": start_pose,
            "goal_pose": goal_pose,
            "scenario_id": str(self.obstacle_avoidance_2_scenario_var.get() or f"oa2_route_{episode_id}").strip(),
            "environment_id": str(self.obstacle_avoidance_2_environment_var.get() or project.get("environment_id", DEFAULT_ENVIRONMENT_ID)).strip(),
            "method": str(self.obstacle_avoidance_2_method_var.get() or project.get("default_method", DEFAULT_METHOD_ID)).strip(),
            "obstacle_hint": str(self.obstacle_avoidance_2_obstacle_hint_var.get() or "unknown").strip(),
            "operator_note": str(self.obstacle_avoidance_2_note_var.get() or "").strip(),
        }
        errors = validate_plan_episode(episode)
        if errors:
            self.obstacle_avoidance_2_status_var.set("; ".join(errors))
            return None
        return episode

    def upsert_obstacle_avoidance_2_editor_episode(self) -> Optional[Dict[str, Any]]:
        episode = self.build_obstacle_avoidance_2_editor_episode()
        if episode is None:
            return None
        project = self.current_obstacle_avoidance_2_project()
        existing = self.find_obstacle_avoidance_2_episode(str(episode.get("episode_id", "")))
        if existing is None:
            project.setdefault("episodes", []).append(dict(episode))
        else:
            existing.clear()
            existing.update(episode)
        if isinstance(self.obstacle_avoidance_2_plan_data, dict):
            save_plans(self.obstacle_avoidance_2_plan_path(), self.obstacle_avoidance_2_plan_data)
        self.refresh_obstacle_avoidance_2_tree()
        self.refresh_obstacle_avoidance_2_records(str(episode.get("episode_id", "")))
        return episode

    def run_obstacle_avoidance_2_current_episode(self) -> None:
        if self.obstacle_avoidance_2_runner_thread is not None and self.obstacle_avoidance_2_runner_thread.is_alive():
            self.obstacle_avoidance_2_status_var.set("OA2 runner is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_2_status_var.set("OA2: start Unreal from the main window first")
            return
        episode = self.upsert_obstacle_avoidance_2_editor_episode()
        if episode is None:
            return
        project = self.current_obstacle_avoidance_2_project()
        method = str(episode.get("method", "") or self.obstacle_avoidance_2_method_var.get() or DEFAULT_METHOD_ID).strip()
        data = self.obstacle_avoidance_2_plan_data if isinstance(self.obstacle_avoidance_2_plan_data, dict) else make_default_plans()
        if not method_is_runnable(data, method):
            self.obstacle_avoidance_2_status_var.set(f"OA2 method not executable: {method}")
            self.obstacle_avoidance_2_report(
                {
                    "status": "blocked",
                    "method": method,
                    "reason": "runner_not_implemented",
                    "hint": "Executable methods: geometry_rule_v0, pointcloud_direction_rule, distance_rule, route_follow, no_avoidance.",
                }
            )
            return
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
        episode_id = str(episode["episode_id"])
        run_id = sanitize_id(
            f"{project.get('project_id', DEFAULT_PROJECT_ID)}_{method}_{episode_id}_{datetime.now().strftime('%H%M%S')}",
            "oa2_single_run",
        )
        note = (
            f"oa2_project={project.get('project_id', '')}; "
            f"environment={episode.get('environment_id', '')}; "
            f"single_episode={episode_id}"
        )
        self.obstacle_avoidance_2_stop_event = threading.Event()
        self.obstacle_avoidance_2_status_var.set(f"OA2 current episode starting: {episode_id}")
        self.obstacle_avoidance_2_report(
            {
                "status": "starting_current_episode",
                "session_source": "main_window_start_unreal",
                "project_id": project.get("project_id", ""),
                "episode_id": episode_id,
                "method": method,
                "run_id": run_id,
                "start_pose": episode.get("start_pose"),
                "goal_pose": episode.get("goal_pose"),
            }
        )

        def worker() -> None:
            try:
                result = self.run_obstacle_avoidance_2_collection_on_session(
                    session,
                    resolved,
                    method=method,
                    run_id=run_id,
                    note=note,
                    project_id=str(project.get("project_id", "")),
                )
                self.root.after(
                    0,
                    lambda r=result, eid=episode_id: (
                        self.obstacle_avoidance_2_status_var.set(
                            f"OA2 current episode done: {eid}, reached {r.get('reached_count', 0)}/{r.get('episode_count', 0)}, collision {r.get('collision_count', 0)}"
                        ),
                        self.refresh_obstacle_avoidance_2_records(eid),
                        self.obstacle_avoidance_2_report(r),
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_avoidance_2_status_var.set(f"OA2 current episode failed: {e}"),
                        self.obstacle_avoidance_2_report({"status": "error", "task": "current_episode_collection", "error": str(e)}),
                    ),
                )

        self.obstacle_avoidance_2_runner_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_2_runner_thread.start()

    def run_obstacle_avoidance_2_selected(self) -> None:
        if self.obstacle_avoidance_2_runner_thread is not None and self.obstacle_avoidance_2_runner_thread.is_alive():
            self.obstacle_avoidance_2_status_var.set("OA2 runner is already active")
            return
        session = self.active_session()
        if session is None:
            self.obstacle_avoidance_2_status_var.set("OA2: start Unreal from the main window first")
            return
        prepared = self.prepare_obstacle_avoidance_2_run()
        if prepared is None:
            return
        project, selected_ids, episodes_json, method, run_id, note = prepared
        data = self.obstacle_avoidance_2_plan_data if isinstance(self.obstacle_avoidance_2_plan_data, dict) else make_default_plans()
        if not method_is_runnable(data, method):
            self.obstacle_avoidance_2_status_var.set(f"OA2 method not executable: {method}")
            self.obstacle_avoidance_2_report(
                {
                    "status": "blocked",
                    "method": method,
                    "reason": "runner_not_implemented",
                    "hint": "Executable methods: geometry_rule_v0, pointcloud_direction_rule, distance_rule, route_follow, no_avoidance.",
                }
            )
            return
        try:
            episodes = self.resolve_obstacle_avoidance_2_episodes(episodes_json)
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 episode parse failed: {exc}")
            self.obstacle_avoidance_2_report({"status": "error", "task": "parse_episodes", "error": str(exc)})
            return
        self.obstacle_avoidance_2_stop_event = threading.Event()
        self.obstacle_avoidance_2_status_var.set("OA2 in-session collection starting...")
        self.obstacle_avoidance_2_report(
            {
                "status": "starting_in_session_collection",
                "session_source": "main_window_start_unreal",
                "episodes_json": str(episodes_json),
                "project_id": project.get("project_id", ""),
                "selected_episode_ids": selected_ids,
                "method": method,
                "run_id": run_id,
                "episode_count": len(episodes),
            }
        )

        def worker() -> None:
            try:
                result = self.run_obstacle_avoidance_2_collection_on_session(
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
                        self.obstacle_avoidance_2_status_var.set(
                            f"OA2 in-session collection done: reached {r.get('reached_count', 0)}/{r.get('episode_count', 0)}, collision {r.get('collision_count', 0)}"
                        ),
                        self.refresh_obstacle_avoidance_2_records(),
                        self.obstacle_avoidance_2_report(r),
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_avoidance_2_status_var.set(f"OA2 collection failed: {e}"),
                        self.obstacle_avoidance_2_report({"status": "error", "task": "in_session_collection", "error": str(e)}),
                    ),
                )

        self.obstacle_avoidance_2_runner_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_2_runner_thread.start()

    def prepare_obstacle_avoidance_2_run(self) -> Optional[Tuple[Dict[str, Any], List[str], Path, str, str, str]]:
        if self.obstacle_avoidance_2_plan_data is None:
            self.load_obstacle_avoidance_2_plans()
        self.apply_obstacle_avoidance_2_project(silent=True)
        self.apply_obstacle_avoidance_2_episode(silent=True)
        method = str(self.obstacle_avoidance_2_method_var.get() or DEFAULT_METHOD_ID).strip()
        project = self.current_obstacle_avoidance_2_project()
        selected_ids = self.selected_obstacle_avoidance_2_episode_ids()
        try:
            episodes_json = export_selected_episodes(project, selected_ids, self.obstacle_avoidance_2_export_path("tmp"))
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 export failed: {exc}")
            self.obstacle_avoidance_2_report({"status": "error", "task": "runner_export", "error": str(exc)})
            return None
        run_id = sanitize_id(f"{project.get('project_id', DEFAULT_PROJECT_ID)}_{method}_{datetime.now().strftime('%H%M%S')}", "oa2_run")
        note = (
            f"oa2_project={project.get('project_id', '')}; "
            f"environment={self.obstacle_avoidance_2_environment_var.get()}; "
            f"selected={','.join(selected_ids) if selected_ids else 'enabled'}"
        )
        return project, selected_ids, episodes_json, method, run_id, note

    def resolve_obstacle_avoidance_2_episodes(self, episodes_json: Path) -> List[Dict[str, Any]]:
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

    def obstacle_avoidance_2_args(self, *, method: str, run_id: str, note: str) -> argparse.Namespace:
        return argparse.Namespace(
            stage="route_episode_2",
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

    def run_obstacle_avoidance_2_collection_on_session(
        self,
        session: flight.DroneFlightSession,
        episodes: List[Dict[str, Any]],
        *,
        method: str,
        run_id: str,
        note: str,
        project_id: str,
    ) -> Dict[str, Any]:
        args = self.obstacle_avoidance_2_args(method=method, run_id=run_id, note=note)
        stop_event = self.obstacle_avoidance_2_stop_event
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
        data_root = self.obstacle_avoidance_2_data_root()
        batch_dir = data_root / "route_episode_batches" / f"{timestamp}_route_episode_2_{method}_{run_id}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            batch_dir / "route_episode_batch_config.json",
            {
                "source": "obstacle_avoidance_2_gui_active_session",
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
                self.obstacle_avoidance_2_report(
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
            session_dir = data_root / "sessions" / f"{timestamp}_route_episode_2_{episode_id}_{method}_{run_id}"
            session_dir.mkdir(parents=True, exist_ok=False)
            events_path = session_dir / "avoidance_events.jsonl"
            write_json(
                session_dir / "episode_config.json",
                {
                    "source": "obstacle_avoidance_2_gui_active_session",
                    "project_id": project_id,
                    "episode_index": episode_index,
                    "episode": episode,
                    "args": vars(args),
                },
            )
            self.root.after(
                0,
                lambda eid=episode_id, s=start, g=goal: self.obstacle_avoidance_2_report(
                    f"OA2_EPISODE_START {eid} start={s} goal={g}"
                ),
            )
            session.set_pose({"x": start["x"], "y": start["y"], "z": start["z"], "yaw": start["yaw"]})
            last_action = action_payload("hold")
            building_obstacle_state = make_building_obstacle_state()
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
                    self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_2_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                    break
                pre_pose = pose_from_state(pre_state if isinstance(pre_state, dict) else {})
                pre_distance = distance_3d_cm(pre_pose, goal)
                action_detail = {
                    "source": "obstacle_avoidance_2_gui_active_session",
                    "project_id": project_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "collection_stage": args.stage,
                    "scenario_id": episode.get("scenario_id") or f"oa2_route_{episode_id}",
                    "environment_id": episode.get("environment_id", ""),
                    "method": args.method,
                    "plan_method": episode.get("method", ""),
                    "obstacle_hint": episode.get("obstacle_hint", ""),
                    "run_id": args.run_id,
                    "mission_phase": "ROUTE_EPISODE_2",
                    "risk_state": "SAFE",
                    "expert_action": str(last_action.get("action_name", "hold")),
                    "expert_action_payload": last_action,
                    "start_pose": start,
                    "goal_pose": goal,
                    "target_waypoint": goal,
                    "operator_note": episode.get("operator_note") or args.note,
                    "building_obstacle_state": dict(building_obstacle_state),
                }
                try:
                    result = session.capture_lidar_stream_frame(session_dir, frame_id, action_detail=action_detail)
                except Exception as exc:
                    outcome = "unrealcv_error"
                    episode_error = f"capture_lidar_stream_frame failed at frame {frame_id}: {exc}"
                    self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_2_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                    break
                if not isinstance(result, dict):
                    outcome = "unrealcv_error"
                    episode_error = f"capture_lidar_stream_frame returned non-dict at frame {frame_id}"
                    self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_2_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                    break
                if stop_event.is_set():
                    outcome = "stopped"
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
                event["source"] = "obstacle_avoidance_2_gui_active_session"
                event["project_id"] = project_id
                rel = event.get("relative_target") if isinstance(event.get("relative_target"), dict) else {}
                selected_action, payload, reason, phase = select_route_action(
                    event["pointcloud_summary"],
                    event["candidate_action_scores"],
                    rel,
                    method=args.method,
                    distance_to_goal_cm=pre_distance,
                    reach_tol_cm=float(args.reach_tol_cm),
                    last_action=last_action,
                    current_pose=pre_pose,
                    start=start,
                    goal=goal,
                    route_step_cm=float(args.route_step_cm),
                    side_correction_cm=float(args.side_correction_cm),
                    vertical_step_cm=float(args.vertical_step_cm),
                    obstacle_hint=str(episode.get("obstacle_hint", "")),
                    environment_id=str(episode.get("environment_id", "")),
                    building_state=building_obstacle_state,
                )
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
                        "shield_state": "GOAL_REACHED" if phase == "REACHED" else "V0_SHIELD_APPLIED",
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
                    hit_state = self.read_obstacle_avoidance_2_hit_state(session)
                else:
                    try:
                        response = session.move_relative(payload)
                    except Exception as exc:
                        outcome = "unrealcv_error"
                        episode_error = f"move_relative failed at frame {frame_id}: {exc}"
                        self.root.after(0, lambda eid=episode_id, err=episode_error: self.obstacle_avoidance_2_report({"status": "unrealcv_error", "episode_id": eid, "error": err}))
                        break
                    post_pose = pose_from_state(response if isinstance(response, dict) else session.get_state())
                    post_distance = distance_3d_cm(post_pose, goal)
                    hit_state = self.read_obstacle_avoidance_2_hit_state(session)
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
                annotate_collision_state(
                    event,
                    pre_pose=pre_pose,
                    post_pose=post_pose,
                    payload=payload,
                    explicit_collision=hit_state,
                )
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
                    event["selected_action_reason"] = (
                        f"{event.get('selected_action_reason', '')}; repeated hold detected, ending episode"
                    )
                append_jsonl(events_path, event)
                events.append(event)
                write_partial_episode_summary(
                    session_dir,
                    events,
                    start=start,
                    goal=goal,
                    outcome=outcome,
                    reach_tol_cm=float(args.reach_tol_cm),
                )
                last_action = payload
                if bool(event.get("collision_state")):
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
                self.root.after(
                    0,
                    lambda eid=episode_id, fid=frame_id, dist=post_distance, reached=post_reached, completion=completion_reason, action=selected_action, risk=event.get("risk_state", risk_state), hit=bool(event.get("collision_state")), phase=event.get("mission_phase", phase), building=(event.get("building_obstacle_state", {}) if isinstance(event.get("building_obstacle_state"), dict) else {}).get("status", ""), reason=event.get("selected_action_reason", reason): (
                        self.obstacle_avoidance_2_goal_distance_var.set(f"Goal distance: {dist:.1f} cm"),
                        self.obstacle_avoidance_2_goal_reached_var.set(
                            f"Goal reached: {'YES' if reached else 'no'} ({completion})"
                        ),
                        self.obstacle_avoidance_2_status_var.set(
                            f"OA2 {eid} frame {fid}: dist={dist:.1f}cm reached={reached} action={action} phase={phase} building={building} risk={risk} impact={hit}"
                        ),
                        self.obstacle_avoidance_2_impact_var.set("Impact: YES" if hit else "Impact: no"),
                        self.obstacle_avoidance_2_report(
                            f"OA2_FRAME {eid} {fid}/{args.max_ticks_per_episode} dist={dist:.1f} reached={reached} completion={completion} action={action} phase={phase} building={building} risk={risk} impact={hit} reason={reason}"
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
            write_json(
                batch_dir / "route_episode_batch_summary.json",
                {
                    "batch_dir": str(batch_dir),
                    "source": "obstacle_avoidance_2_gui_active_session",
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
                    self.obstacle_avoidance_2_goal_distance_var.set(
                        f"Goal distance: {float(s.get('final_distance_to_goal_cm', 0.0)):.1f} cm"
                    ),
                    self.obstacle_avoidance_2_goal_reached_var.set(
                        f"Goal reached: {'YES' if s.get('goal_reached') else 'no'} ({s.get('goal_completion_reason', '')})"
                    ),
                    self.obstacle_avoidance_2_report({"episode_done": eid, "outcome": o, "summary": s}),
                ),
            )
            if outcome not in {"reached", "stopped"} and not bool(args.continue_on_failure):
                break
            if outcome == "stopped":
                break

        batch_summary = {
            "batch_dir": str(batch_dir),
            "source": "obstacle_avoidance_2_gui_active_session",
            "episode_count": len(episodes),
            "completed_count": len(batch_summaries),
            "reached_count": sum(1 for item in batch_summaries if item.get("outcome") == "reached"),
            "collision_count": sum(1 for item in batch_summaries if item.get("outcome") == "collision" or item.get("had_collision")),
            "summaries": batch_summaries,
            "finished_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        write_json(batch_dir / "route_episode_batch_summary.json", batch_summary)
        return batch_summary

    def stop_obstacle_avoidance_2_runner(self) -> None:
        self.obstacle_avoidance_2_stop_event.set()
        process = self.obstacle_avoidance_2_runner_process
        if process is None or process.poll() is not None:
            self.obstacle_avoidance_2_status_var.set("OA2 stop requested")
            return
        try:
            process.terminate()
            self.obstacle_avoidance_2_status_var.set("OA2 runner terminate requested")
        except Exception as exc:
            self.obstacle_avoidance_2_status_var.set(f"OA2 stop failed: {exc}")

    def terminate_obstacle_avoidance_2_process(self, process: Any) -> bool:
        if process is None:
            return False
        try:
            if process.poll() is not None:
                return False
        except Exception:
            pass
        try:
            process.terminate()
            time.sleep(0.15)
            if process.poll() is None:
                process.kill()
            return True
        except Exception:
            try:
                process.kill()
                return True
            except Exception:
                return False

    def halt_obstacle_avoidance_2_motion_now(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"status": "halt_requested"}
        session = self.active_session()
        if session is None:
            result["session"] = "none"
            return result
        try:
            env = getattr(session, "env", None)
            drone_name = getattr(session, "drone_name", None)
            unrealcv = getattr(getattr(env, "unwrapped", env), "unrealcv", None)
            lock = getattr(session, "api_lock", None)
            acquired = False
            try:
                if lock is not None:
                    acquired = bool(lock.acquire(timeout=0.75))
                if (lock is None or acquired) and unrealcv is not None and drone_name:
                    unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
                    result["velocity"] = "zeroed"
                elif lock is not None and not acquired:
                    result["velocity"] = "api_lock_busy"
            finally:
                if acquired:
                    lock.release()
        except Exception as exc:
            result["velocity_error"] = str(exc)
        try:
            session.set_movement_enabled(False)
            result["movement_enabled"] = False
        except Exception as exc:
            result["movement_disable_error"] = str(exc)
        return result

    def emergency_stop_obstacle_avoidance_2_all(self) -> None:
        self.ensure_obstacle_avoidance_2_state()
        try:
            self.stop_keyboard_control(send_hold=True, force_hold=True)
        except Exception:
            pass
        stopped_events: List[str] = []
        for name in (
            "stream_capture_stop_event",
            "lidar_stream_capture_stop_event",
            "obstacle_avoidance_stop_event",
            "obstacle_avoidance_2_stop_event",
            "route_stop_event",
            "llm_route3_stop_event",
            "stream_analysis_stop_event",
            "sequence_stop_event",
        ):
            event = getattr(self, name, None)
            if event is not None:
                try:
                    event.set()
                    stopped_events.append(name)
                except Exception:
                    pass

        killed_processes: List[str] = []
        if self.terminate_obstacle_avoidance_2_process(getattr(self, "obstacle_avoidance_2_runner_process", None)):
            killed_processes.append("obstacle_avoidance_2_runner_process")
        if self.terminate_obstacle_avoidance_2_process(getattr(self, "obstacle_plan_runner_process", None)):
            killed_processes.append("obstacle_plan_runner_process")
        self.obstacle_avoidance_2_runner_process = None
        if hasattr(self, "obstacle_plan_runner_process"):
            self.obstacle_plan_runner_process = None

        old_stop_event = self.obstacle_avoidance_2_stop_event
        try:
            old_stop_event.set()
        except Exception:
            pass
        self.obstacle_avoidance_2_runner_thread = None
        self.obstacle_avoidance_2_stop_event = threading.Event()
        self.obstacle_avoidance_2_collision_alert_keys.clear()
        self.obstacle_avoidance_2_impact_var.set("Impact: unknown")
        self.obstacle_avoidance_2_status_var.set("OA2 EMERGENCY STOP ALL: stopping runners and disabling movement...")
        self.obstacle_avoidance_2_report(
            {
                "status": "emergency_stop_all_requested",
                "stopped_events": stopped_events,
                "killed_processes": killed_processes,
                "runner_state": "cleared",
            }
        )

        def worker() -> None:
            halt_result = self.halt_obstacle_avoidance_2_motion_now()
            self.root.after(
                0,
                lambda r=halt_result: (
                    self.obstacle_avoidance_2_status_var.set("OA2 EMERGENCY STOP ALL complete; runner lock cleared"),
                    self.obstacle_avoidance_2_report({"status": "emergency_stop_all_complete", "halt_result": r}),
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def finish_obstacle_avoidance_2_runner_reset(self) -> None:
        thread = self.obstacle_avoidance_2_runner_thread
        if thread is not None and thread.is_alive():
            self.obstacle_avoidance_2_status_var.set("OA2 reset requested; runner is still stopping")
            try:
                self.root.after(1500, self.finish_obstacle_avoidance_2_runner_reset)
            except Exception:
                pass
            return
        self.obstacle_avoidance_2_runner_thread = None
        self.obstacle_avoidance_2_runner_process = None
        self.obstacle_avoidance_2_stop_event.clear()
        self.obstacle_avoidance_2_collision_alert_keys.clear()
        self.obstacle_avoidance_2_impact_var.set("Impact: unknown")
        self.obstacle_avoidance_2_goal_distance_var.set("Goal distance: --")
        self.obstacle_avoidance_2_goal_reached_var.set("Goal reached: --")
        self.obstacle_avoidance_2_status_var.set("OA2 runner reset")

    def reset_obstacle_avoidance_2_runner(self) -> None:
        self.obstacle_avoidance_2_stop_event.set()
        process = self.obstacle_avoidance_2_runner_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        session = self.active_session()
        if session is not None:
            try:
                session.set_movement_enabled(False)
            except Exception:
                pass
            try:
                env = getattr(session, "env", None)
                drone_name = getattr(session, "drone_name", None)
                unrealcv = getattr(getattr(env, "unwrapped", env), "unrealcv", None)
                lock = getattr(session, "api_lock", None)
                acquired = False
                try:
                    if lock is not None:
                        acquired = bool(lock.acquire(timeout=0.75))
                    if (lock is None or acquired) and unrealcv is not None and drone_name:
                        unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
                finally:
                    if acquired:
                        lock.release()
            except Exception:
                pass
        self.obstacle_avoidance_2_status_var.set("OA2 reset requested")
        try:
            self.root.after(800, self.finish_obstacle_avoidance_2_runner_reset)
        except Exception:
            self.finish_obstacle_avoidance_2_runner_reset()
