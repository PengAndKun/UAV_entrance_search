from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from tkinter import messagebox

from .common import *

from obstacle_avoidance.dataset import build_dataset
from obstacle_avoidance.features import ACTION_PAYLOAD_TEMPLATES, extract_event_features
from obstacle_avoidance.geometry_v0 import (
    best_candidate_action,
    score_candidate_actions,
    selected_action_reason,
    summarize_geometry_v0,
)
from obstacle_avoidance.plans import (
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
from obstacle_avoidance.train import train_baseline
from obstacle_avoidance.validate import validate_model


class ObstacleAvoidanceControlMixin:
    def ensure_obstacle_avoidance_state(self) -> None:
        if not hasattr(self, "obstacle_avoidance_data_dir_var"):
            self.obstacle_avoidance_data_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "obstacle_avoidance_data"))
        if not hasattr(self, "obstacle_avoidance_session_var"):
            self.obstacle_avoidance_session_var = tk.StringVar(value="obstacle_avoidance")
        if not hasattr(self, "obstacle_avoidance_interval_s_var"):
            self.obstacle_avoidance_interval_s_var = tk.StringVar(value="0.5")
        if not hasattr(self, "obstacle_avoidance_stage_var"):
            self.obstacle_avoidance_stage_var = tk.StringVar(value="manual_expert")
        if not hasattr(self, "obstacle_avoidance_scenario_var"):
            self.obstacle_avoidance_scenario_var = tk.StringVar(value="S0")
        if not hasattr(self, "obstacle_avoidance_method_var"):
            self.obstacle_avoidance_method_var = tk.StringVar(value="manual_keyboard")
        if not hasattr(self, "obstacle_avoidance_run_id_var"):
            self.obstacle_avoidance_run_id_var = tk.StringVar(value="001")
        if not hasattr(self, "obstacle_avoidance_geometry_label_var"):
            self.obstacle_avoidance_geometry_label_var = tk.StringVar(value="unknown")
        if not hasattr(self, "obstacle_avoidance_operator_note_var"):
            self.obstacle_avoidance_operator_note_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_avoidance_risk_var"):
            self.obstacle_avoidance_risk_var = tk.StringVar(value="SAFE")
        if not hasattr(self, "obstacle_avoidance_expert_action_var"):
            self.obstacle_avoidance_expert_action_var = tk.StringVar(value="hold")
        if not hasattr(self, "obstacle_avoidance_collision_var"):
            self.obstacle_avoidance_collision_var = tk.BooleanVar(value=False)
        if not hasattr(self, "obstacle_avoidance_status_var"):
            self.obstacle_avoidance_status_var = tk.StringVar(value="Obstacle Avoidance: idle")
        if not hasattr(self, "obstacle_avoidance_stop_event"):
            self.obstacle_avoidance_stop_event = threading.Event()
        if not hasattr(self, "obstacle_avoidance_capture_thread"):
            self.obstacle_avoidance_capture_thread = None
        if not hasattr(self, "obstacle_avoidance_task_thread"):
            self.obstacle_avoidance_task_thread = None
        if not hasattr(self, "obstacle_avoidance_session_dir"):
            self.obstacle_avoidance_session_dir = None
        if not hasattr(self, "obstacle_avoidance_frame_index"):
            self.obstacle_avoidance_frame_index = 0
        if not hasattr(self, "obstacle_avoidance_report_text"):
            self.obstacle_avoidance_report_text = None
        if not hasattr(self, "obstacle_avoidance_window"):
            self.obstacle_avoidance_window = None
        if not hasattr(self, "obstacle_plan_json_path_var"):
            self.obstacle_plan_json_path_var = tk.StringVar(
                value=str(PROJECT_ROOT / "obstacle_avoidance_data" / "plans" / DEFAULT_PLAN_FILENAME)
            )
        if not hasattr(self, "obstacle_plan_project_var"):
            self.obstacle_plan_project_var = tk.StringVar(value=DEFAULT_PROJECT_ID)
        if not hasattr(self, "obstacle_plan_project_name_var"):
            self.obstacle_plan_project_name_var = tk.StringVar(value="Default 10 route episodes")
        if not hasattr(self, "obstacle_plan_environment_var"):
            self.obstacle_plan_environment_var = tk.StringVar(value=DEFAULT_ENVIRONMENT_ID)
        if not hasattr(self, "obstacle_plan_method_var"):
            self.obstacle_plan_method_var = tk.StringVar(value=DEFAULT_METHOD_ID)
        if not hasattr(self, "obstacle_plan_selected_episode_var"):
            self.obstacle_plan_selected_episode_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_plan_runner_status_var"):
            self.obstacle_plan_runner_status_var = tk.StringVar(value="Plan runner: idle")
        if not hasattr(self, "obstacle_plan_episode_enabled_var"):
            self.obstacle_plan_episode_enabled_var = tk.BooleanVar(value=True)
        if not hasattr(self, "obstacle_plan_start_pose_var"):
            self.obstacle_plan_start_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_plan_goal_pose_var"):
            self.obstacle_plan_goal_pose_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_plan_scenario_var"):
            self.obstacle_plan_scenario_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_plan_obstacle_hint_var"):
            self.obstacle_plan_obstacle_hint_var = tk.StringVar(value="unknown")
        if not hasattr(self, "obstacle_plan_operator_note_var"):
            self.obstacle_plan_operator_note_var = tk.StringVar(value="")
        if not hasattr(self, "obstacle_plan_data"):
            self.obstacle_plan_data = None
        if not hasattr(self, "obstacle_plan_project_combo"):
            self.obstacle_plan_project_combo = None
        if not hasattr(self, "obstacle_plan_environment_combo"):
            self.obstacle_plan_environment_combo = None
        if not hasattr(self, "obstacle_plan_method_combo"):
            self.obstacle_plan_method_combo = None
        if not hasattr(self, "obstacle_plan_episode_tree"):
            self.obstacle_plan_episode_tree = None
        if not hasattr(self, "obstacle_plan_tree_iid_to_episode_id"):
            self.obstacle_plan_tree_iid_to_episode_id = {}
        if not hasattr(self, "obstacle_plan_report_text"):
            self.obstacle_plan_report_text = None
        if not hasattr(self, "obstacle_plan_runner_thread"):
            self.obstacle_plan_runner_thread = None
        if not hasattr(self, "obstacle_plan_runner_process"):
            self.obstacle_plan_runner_process = None

    def obstacle_avoidance_data_root(self) -> Path:
        self.ensure_obstacle_avoidance_state()
        raw = str(self.obstacle_avoidance_data_dir_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else PROJECT_ROOT / "obstacle_avoidance_data"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def obstacle_avoidance_interval_s(self) -> float:
        self.ensure_obstacle_avoidance_state()
        try:
            return max(0.05, float(self.obstacle_avoidance_interval_s_var.get().strip()))
        except Exception:
            return 0.5

    def open_obstacle_avoidance_window(self) -> None:
        self.ensure_obstacle_avoidance_state()
        if self.obstacle_avoidance_window is not None and self.obstacle_avoidance_window.winfo_exists():
            self.obstacle_avoidance_window.lift()
            self.obstacle_avoidance_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("Obstacle Avoidance Data / Plans")
        window.geometry("1220x860")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        self.obstacle_avoidance_window = window

        notebook = ttk.Notebook(window)
        notebook.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        capture_tab = ttk.Frame(notebook)
        plans_tab = ttk.Frame(notebook)
        notebook.add(capture_tab, text="Capture / Train")
        notebook.add(plans_tab, text="Plans / Experiments")

        self.build_obstacle_avoidance_capture_tab(capture_tab)
        self.build_obstacle_avoidance_plans_tab(plans_tab)
        try:
            self.load_obstacle_plans()
        except Exception as exc:
            self.obstacle_plan_runner_status_var.set(f"Plan load failed: {exc}")
            self.obstacle_plan_append_report({"status": "error", "task": "load_plans", "error": str(exc)})

        def close_window() -> None:
            self.obstacle_avoidance_window = None
            self.obstacle_avoidance_report_text = None
            self.obstacle_plan_report_text = None
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)

    def select_obstacle_avoidance_data_dir(self) -> None:
        self.ensure_obstacle_avoidance_state()
        selected = filedialog.askdirectory(
            title="Select obstacle_avoidance_data folder",
            initialdir=str(self.obstacle_avoidance_data_root()),
        )
        if selected:
            self.obstacle_avoidance_data_dir_var.set(selected)

    def build_obstacle_avoidance_capture_tab(self, parent: tk.Widget) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(4, weight=1)

        intro = tk.LabelFrame(parent, text="Instructions")
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        intro.grid_columnconfigure(0, weight=1)
        text = (
            "1. Start Unreal/session, then fly manually or run route movement while this window captures RGB + depth + point cloud.\n"
            "2. Pick collection stage, scenario, method, risk_state, geometry label, and expert_action before capture.\n"
            "3. v0 geometry summary and candidate_action_scores are logged for each tick, but this panel does not auto-control route movement.\n"
            "4. If the UAV is hit/collided, enable Collision or click Mark Collision before collecting the failed sample.\n"
            "5. Build Dataset -> Train Baseline -> Validate Model creates npz/json artifacts under obstacle_avoidance_data."
        )
        tk.Label(intro, text=text, justify="left", anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        config = tk.LabelFrame(parent, text="Capture Labels")
        config.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3, 5):
            config.grid_columnconfigure(col, weight=1)
        tk.Label(config, text="Data Dir").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.obstacle_avoidance_data_dir_var).grid(row=0, column=1, columnspan=5, sticky="ew", padx=6, pady=6)
        tk.Button(config, text="Browse", command=self.select_obstacle_avoidance_data_dir).grid(row=0, column=6, padx=6, pady=6)
        tk.Label(config, text="Session").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.obstacle_avoidance_session_var).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(config, text="Interval s").grid(row=1, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.obstacle_avoidance_interval_s_var, width=8).grid(row=1, column=3, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Stage").grid(row=1, column=4, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_stage_var,
            values=(
                "calibration",
                "static_geometry",
                "manual_expert",
                "rule_v0",
                "route_episode",
                "route_closed_loop",
                "hard_case",
                "dataset_qa",
                "semantic_v1",
                "bc_v1",
                "dagger_rl",
            ),
            state="readonly",
            width=18,
        ).grid(row=1, column=5, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Scenario").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_scenario_var,
            values=("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"),
            width=8,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Method").grid(row=2, column=2, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_method_var,
            values=(
                "manual_keyboard",
                "geometry_rule_v0",
                "distance_rule",
                "no_avoidance",
                "route_follow",
                "vlm_pointcloud_semantic_v1",
                "bc_agent_v1",
                "dagger_or_rl_v2",
            ),
            width=24,
        ).grid(row=2, column=3, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Run").grid(row=2, column=4, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.obstacle_avoidance_run_id_var, width=10).grid(row=2, column=5, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Risk").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_risk_var,
            values=("SAFE", "CAUTION", "BLOCKED", "RECOVERY", "REPLAN", "ABORT_WAYPOINT"),
            state="readonly",
            width=16,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Expert Action").grid(row=3, column=2, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_expert_action_var,
            values=tuple(sorted(ACTION_PAYLOAD_TEMPLATES.keys())),
            state="readonly",
            width=18,
        ).grid(row=3, column=3, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Geometry").grid(row=3, column=4, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_geometry_label_var,
            values=("unknown", "none", "vertical_wall", "overhang_beam", "low_obstacle", "thin_structure"),
            width=18,
        ).grid(row=3, column=5, sticky="w", padx=6, pady=6)
        tk.Checkbutton(config, text="Collision", variable=self.obstacle_avoidance_collision_var).grid(row=4, column=0, sticky="w", padx=6, pady=6)
        tk.Button(config, text="Mark Collision", command=self.mark_obstacle_avoidance_collision).grid(row=4, column=1, sticky="w", padx=6, pady=6)
        tk.Button(config, text="Clear Collision", command=self.clear_obstacle_avoidance_collision).grid(row=4, column=2, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Note").grid(row=4, column=3, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.obstacle_avoidance_operator_note_var).grid(row=4, column=4, columnspan=2, sticky="ew", padx=6, pady=6)

        actions = tk.LabelFrame(parent, text="Actions")
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(actions, text="Capture One Frame", command=self.on_obstacle_avoidance_capture_once).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Start Timed Capture", command=self.on_start_obstacle_avoidance_capture).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Stop Capture", command=self.on_stop_obstacle_avoidance_capture).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Build Dataset", command=self.on_build_obstacle_avoidance_dataset).pack(side="left", padx=(18, 6), pady=6)
        tk.Button(actions, text="Train Baseline", command=self.on_train_obstacle_avoidance_model).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Validate Model", command=self.on_validate_obstacle_avoidance_model).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="QA Report", command=self.on_obstacle_avoidance_quality_report).pack(side="left", padx=6, pady=6)

        status = tk.LabelFrame(parent, text="Status")
        status.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        status.grid_columnconfigure(0, weight=1)
        tk.Label(status, textvariable=self.obstacle_avoidance_status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        report_frame = tk.LabelFrame(parent, text="Report")
        report_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        report_frame.grid_columnconfigure(0, weight=1)
        report_frame.grid_rowconfigure(0, weight=1)
        report = tk.Text(report_frame, height=22, wrap="none", font=("Consolas", 9))
        y_scroll = tk.Scrollbar(report_frame, orient="vertical", command=report.yview)
        x_scroll = tk.Scrollbar(report_frame, orient="horizontal", command=report.xview)
        report.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        report.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.obstacle_avoidance_report_text = report
        self.obstacle_avoidance_append_report({"status": "ready", "data_root": str(self.obstacle_avoidance_data_root())})

    def build_obstacle_avoidance_plans_tab(self, parent: tk.Widget) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)

        path_frame = tk.LabelFrame(parent, text="Plan JSON")
        path_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        path_frame.grid_columnconfigure(1, weight=1)
        tk.Label(path_frame, text="Path").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(path_frame, textvariable=self.obstacle_plan_json_path_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(path_frame, text="Browse", command=self.select_obstacle_plan_json_path).grid(row=0, column=2, padx=6, pady=6)
        tk.Button(path_frame, text="Reload Plans", command=self.load_obstacle_plans).grid(row=0, column=3, padx=6, pady=6)
        tk.Button(path_frame, text="Save Plans", command=self.save_obstacle_plans).grid(row=0, column=4, padx=6, pady=6)
        tk.Button(path_frame, text="Reset Default 10 Episodes", command=self.reset_default_obstacle_plans).grid(row=0, column=5, padx=6, pady=6)

        project_frame = tk.LabelFrame(parent, text="Project / Experiment Defaults")
        project_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for col in (1, 3, 5):
            project_frame.grid_columnconfigure(col, weight=1)
        tk.Label(project_frame, text="Project").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.obstacle_plan_project_combo = ttk.Combobox(
            project_frame,
            textvariable=self.obstacle_plan_project_var,
            state="readonly",
            width=28,
        )
        self.obstacle_plan_project_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        self.obstacle_plan_project_combo.bind("<<ComboboxSelected>>", lambda _event: self.select_obstacle_plan_project())
        tk.Label(project_frame, text="Name").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(project_frame, textvariable=self.obstacle_plan_project_name_var).grid(row=0, column=3, sticky="ew", padx=6, pady=6)
        tk.Button(project_frame, text="Apply Project", command=self.apply_obstacle_plan_project_edits).grid(row=0, column=4, padx=6, pady=6)
        tk.Button(project_frame, text="New", command=self.new_obstacle_plan_project).grid(row=0, column=5, padx=6, pady=6)
        tk.Button(project_frame, text="Copy", command=self.copy_obstacle_plan_project).grid(row=0, column=6, padx=6, pady=6)
        tk.Button(project_frame, text="Delete", command=self.delete_obstacle_plan_project).grid(row=0, column=7, padx=6, pady=6)
        tk.Label(project_frame, text="Environment").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.obstacle_plan_environment_combo = ttk.Combobox(
            project_frame,
            textvariable=self.obstacle_plan_environment_var,
            width=28,
        )
        self.obstacle_plan_environment_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(project_frame, text="Method").grid(row=1, column=2, sticky="w", padx=6, pady=6)
        self.obstacle_plan_method_combo = ttk.Combobox(
            project_frame,
            textvariable=self.obstacle_plan_method_var,
            width=28,
        )
        self.obstacle_plan_method_combo.grid(row=1, column=3, sticky="ew", padx=6, pady=6)
        tk.Label(project_frame, text="Runner").grid(row=1, column=4, sticky="w", padx=6, pady=6)
        tk.Label(project_frame, textvariable=self.obstacle_plan_runner_status_var, anchor="w").grid(row=1, column=5, columnspan=3, sticky="ew", padx=6, pady=6)

        body = tk.Frame(parent)
        body.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        list_frame = tk.LabelFrame(body, text="Episodes")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        columns = ("enabled", "episode_id", "start", "goal", "environment", "method", "note")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="extended", height=16)
        headings = {
            "enabled": "On",
            "episode_id": "Episode",
            "start": "Start [x,y,z,yaw]",
            "goal": "Goal [x,y,z,yaw]",
            "environment": "Environment",
            "method": "Method",
            "note": "Note",
        }
        widths = {"enabled": 48, "episode_id": 78, "start": 170, "goal": 170, "environment": 145, "method": 180, "note": 180}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w", stretch=True)
        y_scroll = tk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        x_scroll = tk.Scrollbar(list_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        tree.bind("<<TreeviewSelect>>", lambda _event: self.on_obstacle_plan_episode_selected())
        self.obstacle_plan_episode_tree = tree

        editor = tk.LabelFrame(body, text="Episode Editor")
        editor.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
        editor.grid_columnconfigure(1, weight=1)
        tk.Checkbutton(editor, text="Enabled", variable=self.obstacle_plan_episode_enabled_var).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Label(editor, textvariable=self.obstacle_plan_selected_episode_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(editor, text="Start Pose").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(editor, textvariable=self.obstacle_plan_start_pose_var).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(editor, text="Goal Pose").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(editor, textvariable=self.obstacle_plan_goal_pose_var).grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(editor, text="Scenario").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(editor, textvariable=self.obstacle_plan_scenario_var).grid(row=3, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(editor, text="Obstacle Hint").grid(row=4, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            editor,
            textvariable=self.obstacle_plan_obstacle_hint_var,
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
        ).grid(row=4, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(editor, text="Note").grid(row=5, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(editor, textvariable=self.obstacle_plan_operator_note_var).grid(row=5, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(editor, text="Apply Episode Edits", command=self.apply_obstacle_plan_episode_edits).grid(row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=(12, 6))
        tk.Button(editor, text="Add Episode", command=self.add_obstacle_plan_episode).grid(row=7, column=0, sticky="ew", padx=6, pady=6)
        tk.Button(editor, text="Delete Episode", command=self.delete_obstacle_plan_episode).grid(row=7, column=1, sticky="ew", padx=6, pady=6)

        runner = tk.LabelFrame(parent, text="Run")
        runner.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(runner, text="Run Selected Episodes", command=self.run_selected_obstacle_plan_episodes).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Stop Plan Run", command=self.stop_obstacle_plan_runner).pack(side="left", padx=6, pady=6)
        tk.Button(runner, text="Export Selected JSON", command=self.export_selected_obstacle_plan_json).pack(side="left", padx=(18, 6), pady=6)

        report_frame = tk.LabelFrame(parent, text="Plan Report")
        report_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        report_frame.grid_columnconfigure(0, weight=1)
        report_frame.grid_rowconfigure(0, weight=1)
        report = tk.Text(report_frame, height=10, wrap="none", font=("Consolas", 9))
        report_y = tk.Scrollbar(report_frame, orient="vertical", command=report.yview)
        report_x = tk.Scrollbar(report_frame, orient="horizontal", command=report.xview)
        report.configure(yscrollcommand=report_y.set, xscrollcommand=report_x.set)
        report.grid(row=0, column=0, sticky="nsew")
        report_y.grid(row=0, column=1, sticky="ns")
        report_x.grid(row=1, column=0, sticky="ew")
        self.obstacle_plan_report_text = report

    def obstacle_avoidance_append_report(self, payload: Dict[str, Any]) -> None:
        self.ensure_obstacle_avoidance_state()
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        widget = self.obstacle_avoidance_report_text
        if widget is not None:
            try:
                widget.insert("end", text + "\n\n")
                widget.see("end")
            except tk.TclError:
                pass

    def obstacle_plan_path(self) -> Path:
        self.ensure_obstacle_avoidance_state()
        raw = str(self.obstacle_plan_json_path_var.get() or "").strip()
        path = Path(raw).expanduser() if raw else self.obstacle_avoidance_data_root() / "plans" / DEFAULT_PLAN_FILENAME
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def obstacle_plan_append_report(self, payload: Any) -> None:
        self.ensure_obstacle_avoidance_state()
        if isinstance(payload, str):
            text = payload.rstrip()
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        widget = self.obstacle_plan_report_text
        if widget is not None:
            try:
                widget.insert("end", text + "\n\n")
                widget.see("end")
            except tk.TclError:
                pass

    def select_obstacle_plan_json_path(self) -> None:
        self.ensure_obstacle_avoidance_state()
        selected = filedialog.asksaveasfilename(
            title="Select obstacle avoidance plan JSON",
            initialdir=str(self.obstacle_plan_path().parent),
            initialfile=self.obstacle_plan_path().name,
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if selected:
            self.obstacle_plan_json_path_var.set(selected)
            self.load_obstacle_plans()

    def load_obstacle_plans(self) -> None:
        self.ensure_obstacle_avoidance_state()
        plan_path = self.obstacle_plan_path()
        created = False
        if not plan_path.exists():
            data = make_default_plans()
            save_plans(plan_path, data)
            created = True
        data = load_plans(plan_path)
        self.obstacle_plan_data = data
        active = str(data.get("active_project_id", DEFAULT_PROJECT_ID))
        self.obstacle_plan_project_var.set(active)
        self.refresh_obstacle_plan_options()
        self.select_obstacle_plan_project()
        self.obstacle_plan_runner_status_var.set(f"Plans loaded: {plan_path}")
        self.obstacle_plan_append_report({"status": "created_default" if created else "loaded", "plan_path": str(plan_path)})

    def save_obstacle_plans(self) -> None:
        self.ensure_obstacle_avoidance_state()
        if self.obstacle_plan_data is None:
            self.obstacle_plan_data = make_default_plans()
        self.apply_obstacle_plan_project_edits(silent=True)
        self.apply_obstacle_plan_episode_edits(silent=True)
        save_plans(self.obstacle_plan_path(), self.obstacle_plan_data)
        self.obstacle_plan_runner_status_var.set("Plans saved")
        self.obstacle_plan_append_report({"status": "saved", "plan_path": str(self.obstacle_plan_path())})

    def reset_default_obstacle_plans(self) -> None:
        self.ensure_obstacle_avoidance_state()
        if not messagebox.askyesno("Reset Plans", "Reset the plan JSON to the default 10 route episodes?"):
            return
        self.obstacle_plan_data = make_default_plans()
        save_plans(self.obstacle_plan_path(), self.obstacle_plan_data)
        self.refresh_obstacle_plan_options()
        self.select_obstacle_plan_project()
        self.obstacle_plan_runner_status_var.set("Plans reset to default 10 episodes")
        self.obstacle_plan_append_report({"status": "reset_default", "plan_path": str(self.obstacle_plan_path())})

    def refresh_obstacle_plan_options(self) -> None:
        data = self.obstacle_plan_data if isinstance(self.obstacle_plan_data, dict) else make_default_plans()
        project_ids = [str(project.get("project_id", "")) for project in data.get("projects", []) if str(project.get("project_id", ""))]
        environment_ids = [str(item.get("environment_id", "")) for item in data.get("environments", []) if str(item.get("environment_id", ""))]
        method_ids = [str(item.get("method_id", "")) for item in data.get("methods", []) if str(item.get("method_id", ""))]
        if self.obstacle_plan_project_combo is not None:
            self.obstacle_plan_project_combo.configure(values=tuple(project_ids))
        if self.obstacle_plan_environment_combo is not None:
            self.obstacle_plan_environment_combo.configure(values=tuple(environment_ids))
        if self.obstacle_plan_method_combo is not None:
            self.obstacle_plan_method_combo.configure(values=tuple(method_ids))

    def current_obstacle_plan_project(self) -> Dict[str, Any]:
        if self.obstacle_plan_data is None:
            self.load_obstacle_plans()
        data = self.obstacle_plan_data if isinstance(self.obstacle_plan_data, dict) else make_default_plans()
        project_id = str(self.obstacle_plan_project_var.get() or data.get("active_project_id", DEFAULT_PROJECT_ID))
        try:
            return project_by_id(data, project_id)
        except Exception:
            projects = data.get("projects", [])
            if not projects:
                data["projects"] = make_default_plans()["projects"]
                projects = data["projects"]
            project = projects[0]
            self.obstacle_plan_project_var.set(str(project.get("project_id", DEFAULT_PROJECT_ID)))
            return project

    def select_obstacle_plan_project(self) -> None:
        data = self.obstacle_plan_data if isinstance(self.obstacle_plan_data, dict) else make_default_plans()
        project = self.current_obstacle_plan_project()
        data["active_project_id"] = str(project.get("project_id", DEFAULT_PROJECT_ID))
        self.obstacle_plan_project_name_var.set(str(project.get("name", project.get("project_id", ""))))
        self.obstacle_plan_environment_var.set(str(project.get("environment_id", DEFAULT_ENVIRONMENT_ID)))
        self.obstacle_plan_method_var.set(str(project.get("default_method", DEFAULT_METHOD_ID)))
        self.refresh_obstacle_plan_episode_tree()

    def apply_obstacle_plan_project_edits(self, *, silent: bool = False) -> None:
        project = self.current_obstacle_plan_project()
        project["name"] = str(self.obstacle_plan_project_name_var.get() or project.get("name", "")).strip() or str(project.get("project_id"))
        project["environment_id"] = str(self.obstacle_plan_environment_var.get() or DEFAULT_ENVIRONMENT_ID).strip()
        project["default_method"] = str(self.obstacle_plan_method_var.get() or DEFAULT_METHOD_ID).strip()
        if isinstance(self.obstacle_plan_data, dict):
            self.obstacle_plan_data["active_project_id"] = str(project.get("project_id", DEFAULT_PROJECT_ID))
        if not silent:
            self.refresh_obstacle_plan_episode_tree()
            self.obstacle_plan_runner_status_var.set("Project edits applied")

    def unique_obstacle_plan_project_id(self, base: str) -> str:
        data = self.obstacle_plan_data if isinstance(self.obstacle_plan_data, dict) else make_default_plans()
        existing = {str(project.get("project_id", "")) for project in data.get("projects", [])}
        root = sanitize_id(base, "project")
        candidate = root
        index = 2
        while candidate in existing:
            candidate = f"{root}_{index}"
            index += 1
        return candidate

    def new_obstacle_plan_project(self) -> None:
        if self.obstacle_plan_data is None:
            self.load_obstacle_plans()
        name = str(self.obstacle_plan_project_name_var.get() or "New obstacle plan").strip()
        project_id = self.unique_obstacle_plan_project_id(name)
        project = {
            "project_id": project_id,
            "name": name,
            "environment_id": str(self.obstacle_plan_environment_var.get() or DEFAULT_ENVIRONMENT_ID),
            "default_method": str(self.obstacle_plan_method_var.get() or DEFAULT_METHOD_ID),
            "episodes": [],
            "experiment_defaults": {
                "stage": "route_episode",
                "launch_sleep": 5,
                "interval_s": 5.0,
                "width": 320,
                "height": 240,
                "reach_tol_cm": 180,
                "max_ticks_per_episode": 220,
                "continue_on_failure": True,
            },
        }
        self.obstacle_plan_data.setdefault("projects", []).append(project)
        self.obstacle_plan_project_var.set(project_id)
        self.refresh_obstacle_plan_options()
        self.select_obstacle_plan_project()
        self.obstacle_plan_runner_status_var.set(f"Created project {project_id}")

    def copy_obstacle_plan_project(self) -> None:
        project = deepcopy(self.current_obstacle_plan_project())
        source_id = str(project.get("project_id", "project"))
        project["project_id"] = self.unique_obstacle_plan_project_id(f"{source_id}_copy")
        project["name"] = f"{project.get('name', source_id)} copy"
        self.obstacle_plan_data.setdefault("projects", []).append(project)
        self.obstacle_plan_project_var.set(str(project["project_id"]))
        self.refresh_obstacle_plan_options()
        self.select_obstacle_plan_project()
        self.obstacle_plan_runner_status_var.set(f"Copied project {source_id}")

    def delete_obstacle_plan_project(self) -> None:
        if self.obstacle_plan_data is None:
            return
        projects = self.obstacle_plan_data.get("projects", [])
        if len(projects) <= 1:
            self.obstacle_plan_runner_status_var.set("Cannot delete the last project")
            return
        project = self.current_obstacle_plan_project()
        project_id = str(project.get("project_id", ""))
        if not messagebox.askyesno("Delete Project", f"Delete project {project_id}?"):
            return
        self.obstacle_plan_data["projects"] = [item for item in projects if str(item.get("project_id", "")) != project_id]
        self.obstacle_plan_project_var.set(str(self.obstacle_plan_data["projects"][0].get("project_id", DEFAULT_PROJECT_ID)))
        self.refresh_obstacle_plan_options()
        self.select_obstacle_plan_project()
        self.obstacle_plan_runner_status_var.set(f"Deleted project {project_id}")

    def obstacle_plan_format_pose(self, pose: Any) -> str:
        try:
            return ", ".join(f"{value:g}" for value in coerce_pose(pose))
        except Exception:
            return ""

    def refresh_obstacle_plan_episode_tree(self) -> None:
        tree = self.obstacle_plan_episode_tree
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        self.obstacle_plan_tree_iid_to_episode_id = {}
        project = self.current_obstacle_plan_project()
        for index, episode in enumerate(project.get("episodes", []), start=1):
            episode_id = str(episode.get("episode_id", f"E{index:02d}"))
            iid = f"{episode_id}::{index}"
            self.obstacle_plan_tree_iid_to_episode_id[iid] = episode_id
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    "yes" if bool(episode.get("enabled", True)) else "no",
                    episode_id,
                    self.obstacle_plan_format_pose(episode.get("start_pose", [])),
                    self.obstacle_plan_format_pose(episode.get("goal_pose", [])),
                    str(episode.get("environment_id", project.get("environment_id", DEFAULT_ENVIRONMENT_ID))),
                    str(episode.get("method", project.get("default_method", DEFAULT_METHOD_ID))),
                    str(episode.get("operator_note", "")),
                ),
            )

    def selected_obstacle_plan_episode_ids(self) -> List[str]:
        tree = self.obstacle_plan_episode_tree
        if tree is None:
            return []
        ids: List[str] = []
        for iid in tree.selection():
            episode_id = self.obstacle_plan_tree_iid_to_episode_id.get(str(iid), "")
            if episode_id:
                ids.append(episode_id)
        return ids

    def find_obstacle_plan_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        project = self.current_obstacle_plan_project()
        for episode in project.get("episodes", []):
            if str(episode.get("episode_id", "")) == str(episode_id):
                return episode
        return None

    def on_obstacle_plan_episode_selected(self) -> None:
        selected = self.selected_obstacle_plan_episode_ids()
        if not selected:
            return
        episode = self.find_obstacle_plan_episode(selected[0])
        if not isinstance(episode, dict):
            return
        self.obstacle_plan_selected_episode_var.set(str(episode.get("episode_id", "")))
        self.obstacle_plan_episode_enabled_var.set(bool(episode.get("enabled", True)))
        self.obstacle_plan_start_pose_var.set(self.obstacle_plan_format_pose(episode.get("start_pose", [])))
        self.obstacle_plan_goal_pose_var.set(self.obstacle_plan_format_pose(episode.get("goal_pose", [])))
        self.obstacle_plan_scenario_var.set(str(episode.get("scenario_id", "")))
        self.obstacle_plan_environment_var.set(str(episode.get("environment_id", self.obstacle_plan_environment_var.get())))
        self.obstacle_plan_method_var.set(str(episode.get("method", self.obstacle_plan_method_var.get())))
        self.obstacle_plan_obstacle_hint_var.set(str(episode.get("obstacle_hint", "unknown")))
        self.obstacle_plan_operator_note_var.set(str(episode.get("operator_note", "")))

    def apply_obstacle_plan_episode_edits(self, *, silent: bool = False) -> None:
        episode_id = str(self.obstacle_plan_selected_episode_var.get() or "").strip()
        if not episode_id:
            if not silent:
                self.obstacle_plan_runner_status_var.set("Select an episode before applying edits")
            return
        episode = self.find_obstacle_plan_episode(episode_id)
        if not isinstance(episode, dict):
            self.obstacle_plan_runner_status_var.set(f"Episode not found: {episode_id}")
            return
        try:
            episode["start_pose"] = coerce_pose(self.obstacle_plan_start_pose_var.get())
            episode["goal_pose"] = coerce_pose(self.obstacle_plan_goal_pose_var.get())
        except Exception as exc:
            self.obstacle_plan_runner_status_var.set(f"Invalid pose: {exc}")
            return
        episode["enabled"] = bool(self.obstacle_plan_episode_enabled_var.get())
        episode["scenario_id"] = str(self.obstacle_plan_scenario_var.get() or f"route_episode_{episode_id}").strip()
        episode["environment_id"] = str(self.obstacle_plan_environment_var.get() or DEFAULT_ENVIRONMENT_ID).strip()
        episode["method"] = str(self.obstacle_plan_method_var.get() or DEFAULT_METHOD_ID).strip()
        episode["obstacle_hint"] = str(self.obstacle_plan_obstacle_hint_var.get() or "unknown").strip()
        episode["operator_note"] = str(self.obstacle_plan_operator_note_var.get() or "").strip()
        errors = validate_plan_episode(episode)
        if errors:
            self.obstacle_plan_runner_status_var.set("; ".join(errors))
            return
        self.refresh_obstacle_plan_episode_tree()
        if not silent:
            self.obstacle_plan_runner_status_var.set(f"Episode {episode_id} edits applied")

    def next_obstacle_plan_episode_id(self) -> str:
        project = self.current_obstacle_plan_project()
        existing = {str(episode.get("episode_id", "")) for episode in project.get("episodes", [])}
        index = len(existing) + 1
        while True:
            candidate = f"E{index:02d}"
            if candidate not in existing:
                return candidate
            index += 1

    def add_obstacle_plan_episode(self) -> None:
        project = self.current_obstacle_plan_project()
        episode_id = self.next_obstacle_plan_episode_id()
        try:
            start_pose = coerce_pose(self.obstacle_plan_start_pose_var.get())
            goal_pose = coerce_pose(self.obstacle_plan_goal_pose_var.get())
        except Exception:
            start_pose = [0.0, 0.0, 200.0, 0.0]
            goal_pose = [300.0, 0.0, 200.0, 0.0]
        episode = {
            "episode_id": episode_id,
            "enabled": True,
            "start_pose": start_pose,
            "goal_pose": goal_pose,
            "scenario_id": f"route_episode_{episode_id}",
            "environment_id": str(self.obstacle_plan_environment_var.get() or project.get("environment_id", DEFAULT_ENVIRONMENT_ID)),
            "method": str(self.obstacle_plan_method_var.get() or project.get("default_method", DEFAULT_METHOD_ID)),
            "obstacle_hint": str(self.obstacle_plan_obstacle_hint_var.get() or "unknown"),
            "operator_note": str(self.obstacle_plan_operator_note_var.get() or ""),
        }
        project.setdefault("episodes", []).append(episode)
        self.refresh_obstacle_plan_episode_tree()
        self.obstacle_plan_selected_episode_var.set(episode_id)
        self.obstacle_plan_runner_status_var.set(f"Added episode {episode_id}")

    def delete_obstacle_plan_episode(self) -> None:
        selected = self.selected_obstacle_plan_episode_ids()
        if not selected:
            self.obstacle_plan_runner_status_var.set("Select episode(s) to delete")
            return
        if not messagebox.askyesno("Delete Episodes", f"Delete {len(selected)} selected episode(s)?"):
            return
        selected_set = set(selected)
        project = self.current_obstacle_plan_project()
        project["episodes"] = [episode for episode in project.get("episodes", []) if str(episode.get("episode_id", "")) not in selected_set]
        self.obstacle_plan_selected_episode_var.set("")
        self.refresh_obstacle_plan_episode_tree()
        self.obstacle_plan_runner_status_var.set(f"Deleted {len(selected)} episode(s)")

    def obstacle_plan_export_path(self, *, purpose: str) -> Path:
        project = self.current_obstacle_plan_project()
        project_id = sanitize_id(project.get("project_id", DEFAULT_PROJECT_ID), DEFAULT_PROJECT_ID)
        method = sanitize_id(self.obstacle_plan_method_var.get() or DEFAULT_METHOD_ID, DEFAULT_METHOD_ID)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.obstacle_avoidance_data_root() / "plans" / purpose / f"{timestamp}_{project_id}_{method}_episodes.json"

    def export_selected_obstacle_plan_json(self) -> Optional[Path]:
        self.ensure_obstacle_avoidance_state()
        self.apply_obstacle_plan_project_edits(silent=True)
        self.apply_obstacle_plan_episode_edits(silent=True)
        project = self.current_obstacle_plan_project()
        selected_ids = self.selected_obstacle_plan_episode_ids()
        try:
            export_path = export_selected_episodes(project, selected_ids, self.obstacle_plan_export_path(purpose="exports"))
        except Exception as exc:
            self.obstacle_plan_runner_status_var.set(f"Export failed: {exc}")
            self.obstacle_plan_append_report({"status": "error", "task": "export_selected_episodes", "error": str(exc)})
            return None
        self.obstacle_plan_runner_status_var.set(f"Exported episodes: {export_path}")
        self.obstacle_plan_append_report({"status": "exported", "episodes_json": str(export_path), "selected_episode_ids": selected_ids})
        return export_path

    def obstacle_plan_runner_command(self, episodes_json: Path, method: str, run_id: str, note: str) -> List[str]:
        env_var = getattr(self, "env_platform_var", None)
        env_platform = str(env_var.get() if env_var is not None else getattr(self.args, "env_platform", "win") or "win")
        return [
            sys.executable,
            "-m",
            "obstacle_avoidance.collect_route_episodes",
            "--episodes-json",
            str(episodes_json),
            "--data-root",
            str(self.obstacle_avoidance_data_root()),
            "--stage",
            "route_episode",
            "--method",
            method,
            "--run-id",
            run_id,
            "--note",
            note,
            "--env-platform",
            env_platform,
            "--launch-sleep",
            "5",
            "--interval-s",
            "5",
            "--width",
            "320",
            "--height",
            "240",
            "--reach-tol-cm",
            "180",
            "--max-ticks-per-episode",
            "220",
            "--continue-on-failure",
        ]

    def run_selected_obstacle_plan_episodes(self) -> None:
        self.ensure_obstacle_avoidance_state()
        if self.obstacle_plan_runner_thread is not None and self.obstacle_plan_runner_thread.is_alive():
            self.obstacle_plan_runner_status_var.set("Plan runner is already running")
            return
        if self.obstacle_plan_data is None:
            self.load_obstacle_plans()
        self.apply_obstacle_plan_project_edits(silent=True)
        self.apply_obstacle_plan_episode_edits(silent=True)
        method = str(self.obstacle_plan_method_var.get() or DEFAULT_METHOD_ID).strip()
        if not method_is_runnable(self.obstacle_plan_data, method):
            message = f"Method {method} is registered but not executable yet"
            self.obstacle_plan_runner_status_var.set(message)
            self.obstacle_plan_append_report(
                {
                    "status": "blocked",
                    "method": method,
                    "reason": "runner_not_implemented",
                    "hint": "Use geometry_rule_v0 for the current executable route-episode baseline.",
                }
            )
            return
        project = self.current_obstacle_plan_project()
        selected_ids = self.selected_obstacle_plan_episode_ids()
        try:
            episodes_json = export_selected_episodes(project, selected_ids, self.obstacle_plan_export_path(purpose="tmp"))
        except Exception as exc:
            self.obstacle_plan_runner_status_var.set(f"Run export failed: {exc}")
            self.obstacle_plan_append_report({"status": "error", "task": "run_export", "error": str(exc)})
            return
        run_id = sanitize_id(
            f"{project.get('project_id', DEFAULT_PROJECT_ID)}_{method}_{datetime.now().strftime('%H%M%S')}",
            "plan_run",
        )
        note = (
            f"plan_project={project.get('project_id', '')}; "
            f"environment={self.obstacle_plan_environment_var.get()}; "
            f"selected={','.join(selected_ids) if selected_ids else 'enabled'}"
        )
        cmd = self.obstacle_plan_runner_command(episodes_json, method, run_id, note)
        self.obstacle_plan_runner_status_var.set("Plan runner starting...")
        self.obstacle_plan_append_report({"status": "starting", "command": cmd, "episodes_json": str(episodes_json)})

        def worker() -> None:
            batch_dir = ""
            return_code = -1
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self.obstacle_plan_runner_process = process
                self.root.after(0, lambda: self.obstacle_plan_runner_status_var.set(f"Plan runner pid={process.pid}"))
                if process.stdout is not None:
                    for line in process.stdout:
                        text = line.rstrip()
                        if text.startswith("ROUTE_EPISODE_BATCH_DONE "):
                            batch_dir = text.split(" ", 1)[1].strip()
                        self.root.after(0, lambda value=text: self.obstacle_plan_append_report(value))
                return_code = process.wait()
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_plan_runner_status_var.set(f"Plan runner failed: {e}"),
                        self.obstacle_plan_append_report({"status": "error", "task": "plan_runner", "error": str(e)}),
                    ),
                )
                return
            finally:
                self.obstacle_plan_runner_process = None

            summary: Dict[str, Any] = {"status": "finished", "return_code": return_code, "batch_dir": batch_dir}
            if batch_dir:
                summary_path = Path(batch_dir) / "route_episode_batch_summary.json"
                summary["summary_path"] = str(summary_path)
                if summary_path.exists():
                    try:
                        summary["batch_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        summary["summary_read_error"] = str(exc)
            status_text = "Plan runner done" if return_code == 0 else f"Plan runner exited with code {return_code}"
            self.root.after(
                0,
                lambda s=summary, text=status_text: (
                    self.obstacle_plan_runner_status_var.set(text),
                    self.obstacle_plan_append_report(s),
                ),
            )

        self.obstacle_plan_runner_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_plan_runner_thread.start()

    def stop_obstacle_plan_runner(self) -> None:
        process = self.obstacle_plan_runner_process
        if process is None or process.poll() is not None:
            self.obstacle_plan_runner_status_var.set("No active plan runner process")
            return
        try:
            process.terminate()
            self.obstacle_plan_runner_status_var.set("Plan runner terminate requested")
        except Exception as exc:
            self.obstacle_plan_runner_status_var.set(f"Plan runner stop failed: {exc}")

    def mark_obstacle_avoidance_collision(self) -> None:
        self.ensure_obstacle_avoidance_state()
        self.obstacle_avoidance_collision_var.set(True)
        self.obstacle_avoidance_risk_var.set("BLOCKED")
        self.obstacle_avoidance_status_var.set("Obstacle Avoidance: collision label enabled")

    def clear_obstacle_avoidance_collision(self) -> None:
        self.ensure_obstacle_avoidance_state()
        self.obstacle_avoidance_collision_var.set(False)
        self.obstacle_avoidance_status_var.set("Obstacle Avoidance: collision label cleared")

    def make_obstacle_avoidance_session_dir(self) -> Path:
        root = self.obstacle_avoidance_data_root() / "sessions"
        title = flight.sanitize_capture_task_title(self.obstacle_avoidance_session_var.get(), default_value="obstacle_avoidance")
        stage = flight.sanitize_capture_task_title(self.obstacle_avoidance_stage_var.get(), default_value="manual_expert")
        scenario = flight.sanitize_capture_task_title(self.obstacle_avoidance_scenario_var.get(), default_value="S0")
        method = flight.sanitize_capture_task_title(self.obstacle_avoidance_method_var.get(), default_value="manual_keyboard")
        run_id = flight.sanitize_capture_task_title(self.obstacle_avoidance_run_id_var.get(), default_value=title)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return flight.make_unique_child_dir(root, f"{timestamp}_{stage}_{scenario}_{method}_{run_id}")

    def obstacle_avoidance_current_target_waypoint(self) -> Dict[str, Any]:
        try:
            if isinstance(self.llm_route3_state, dict):
                exploration = self.llm_route3_state.get("current_exploration_status")
                if isinstance(exploration, dict):
                    value = exploration.get("target_pose")
                    if isinstance(value, dict) and value:
                        return dict(value)
                for key in ("target_pose", "current_target_pose", "current_target", "active_scan_point"):
                    value = self.llm_route3_state.get(key)
                    if isinstance(value, dict) and value:
                        return dict(value)
            if isinstance(self.llm_route_plan, dict):
                points = self.llm_route_plan.get("route_points", [])
                index = int(self.llm_route_plan.get("active_waypoint_index", 0) or 0)
                if isinstance(points, list) and 0 <= index < len(points) and isinstance(points[index], dict):
                    return dict(points[index])
        except Exception:
            pass
        return {}

    def obstacle_avoidance_current_mission_phase(self) -> str:
        try:
            if isinstance(self.llm_route3_state, dict):
                exploration = self.llm_route3_state.get("current_exploration_status")
                if isinstance(exploration, dict):
                    stage = str(exploration.get("stage", "") or "").strip().upper()
                    if stage:
                        return stage
                stage = str(self.llm_route3_state.get("stage", "") or "").strip().upper()
                if stage:
                    return stage
        except Exception:
            pass
        return "MANUAL"

    def obstacle_avoidance_relative_target(self, pose: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(pose, dict) or not isinstance(target, dict) or not target:
            return {"distance_cm": 0.0, "bearing_deg_body": 0.0, "dz_cm": 0.0}
        x = float(pose.get("x", 0.0) or 0.0)
        y = float(pose.get("y", 0.0) or 0.0)
        z = float(pose.get("z", 0.0) or 0.0)
        yaw = float(pose.get("yaw", pose.get("task_yaw", pose.get("yaw_deg", 0.0))) or 0.0)
        dx = float(target.get("x", 0.0) or 0.0) - x
        dy = float(target.get("y", 0.0) or 0.0) - y
        dz = float(target.get("z", z) or z) - z
        absolute_bearing = math.degrees(math.atan2(dy, dx)) if dx or dy else yaw
        bearing = (absolute_bearing - yaw + 180.0) % 360.0 - 180.0
        if bearing == -180.0:
            bearing = 180.0
        return {"distance_cm": float(math.hypot(dx, dy)), "bearing_deg_body": float(bearing), "dz_cm": float(dz)}

    def obstacle_avoidance_action_payload(self) -> Dict[str, float]:
        label = self.obstacle_avoidance_expert_action_var.get().strip().lower()
        return dict(ACTION_PAYLOAD_TEMPLATES.get(label, ACTION_PAYLOAD_TEMPLATES["hold"]))

    def obstacle_avoidance_capture_action_detail(self) -> Dict[str, Any]:
        latest_pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state, dict) and isinstance(self.latest_state.get("pose"), dict) else {}
        stage = self.obstacle_avoidance_stage_var.get().strip() or "manual_expert"
        scenario = self.obstacle_avoidance_scenario_var.get().strip() or "S0"
        method = self.obstacle_avoidance_method_var.get().strip() or "manual_keyboard"
        run_id = self.obstacle_avoidance_run_id_var.get().strip() or "001"
        return {
            "source": "obstacle_avoidance_capture",
            "collection_stage": stage,
            "scenario_id": scenario,
            "method": method,
            "run_id": run_id,
            "mission_phase": self.obstacle_avoidance_current_mission_phase(),
            "risk_state": self.obstacle_avoidance_risk_var.get().strip().upper() or "SAFE",
            "expert_action": self.obstacle_avoidance_expert_action_var.get().strip().lower() or "hold",
            "expert_action_payload": self.obstacle_avoidance_action_payload(),
            "obstacle_geometry_label": self.obstacle_avoidance_geometry_label_var.get().strip().lower() or "unknown",
            "operator_note": self.obstacle_avoidance_operator_note_var.get().strip(),
            "collision_state": bool(self.obstacle_avoidance_collision_var.get()),
            "collision_source": "manual_gui",
            "avoidance_failed": bool(self.obstacle_avoidance_collision_var.get()),
            "target_waypoint": self.obstacle_avoidance_current_target_waypoint(),
            "latest_pose": latest_pose,
            "last_action": self.latest_state.get("last_action", "") if isinstance(self.latest_state, dict) else "",
        }

    def obstacle_avoidance_enrich_capture_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        capture_dir = Path(str(result.get("capture_dir", "") or ""))
        if not capture_dir.exists():
            return result
        if not str(result.get("point_cloud_world_standard_m_npy_path", "") or ""):
            ensured = flight.ensure_standard_world_cloud_for_capture(
                capture_dir,
                capture_payload=result,
                lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
                min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
                max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            )
            if isinstance(ensured, dict):
                result.update(ensured)
        return result

    def obstacle_avoidance_build_event(self, result: Dict[str, Any], action_detail: Dict[str, Any]) -> Dict[str, Any]:
        collision_state = bool(action_detail.get("collision_state", False))
        target_waypoint = action_detail.get("target_waypoint", {})
        current_pose = result.get("pose", {})
        relative_target = self.obstacle_avoidance_relative_target(current_pose, target_waypoint if isinstance(target_waypoint, dict) else {})
        session_dir = Path(str(result.get("stream_dir", "") or "")).name or str(action_detail.get("session_id", "") or "")
        event = {
            "frame_id": int(result.get("frame_index", self.obstacle_avoidance_frame_index) or self.obstacle_avoidance_frame_index),
            "timestamp": result.get("capture_time", datetime.now().isoformat(timespec="milliseconds")),
            "session_id": session_dir,
            "collection_stage": action_detail.get("collection_stage", "manual_expert"),
            "scenario_id": action_detail.get("scenario_id", "S0"),
            "method": action_detail.get("method", "manual_keyboard"),
            "run_id": action_detail.get("run_id", "001"),
            "mission_phase": action_detail.get("mission_phase", "MANUAL"),
            "current_pose": current_pose,
            "pose": current_pose,
            "target_waypoint": target_waypoint,
            "relative_target": relative_target,
            "rgb_path": result.get("rgb_path", ""),
            "depth_npy_path": result.get("depth_npy_path", ""),
            "depth_cm_path": result.get("depth_cm_path", ""),
            "pointcloud_path": result.get("point_cloud_world_standard_m_npy_path", ""),
            "point_cloud_world_standard_m_npy_path": result.get("point_cloud_world_standard_m_npy_path", ""),
            "capture_dir": result.get("capture_dir", ""),
            "pose_json_path": result.get("pose_json_path", ""),
            "action_json_path": result.get("action_json_path", ""),
            "depth_summary": result.get("depth_summary", {}),
            "risk_state": action_detail.get("risk_state", "SAFE"),
            "expert_action": action_detail.get("expert_action", "hold"),
            "expert_action_payload": action_detail.get("expert_action_payload", {}),
            "nominal_action": action_detail.get("expert_action_payload", {}),
            "agent_action": action_detail.get("expert_action_payload", {}),
            "executed_action": action_detail.get("expert_action_payload", {}),
            "last_action": action_detail.get("expert_action_payload", {}),
            "obstacle_geometry_label": action_detail.get("obstacle_geometry_label", "unknown"),
            "operator_note": action_detail.get("operator_note", ""),
            "collision_state": collision_state,
            "collision_source": action_detail.get("collision_source", "manual_gui"),
            "avoidance_failed": collision_state,
            "movement_mode": result.get("movement_mode", ""),
            "movement_enabled": bool(result.get("movement_enabled", False)),
            "point_count": int(result.get("point_count", 0) or 0),
            "coordinate_frame": result.get("coordinate_frame", ""),
            "coordinate_units": result.get("coordinate_units", ""),
        }
        try:
            base_dir = Path(str(result.get("capture_dir", "") or ".")).parent
            pointcloud_summary = summarize_geometry_v0(event, base_dir=base_dir)
            event["pointcloud_summary"] = pointcloud_summary
            scores = score_candidate_actions(pointcloud_summary, relative_target, event.get("last_action", {}))
            event["candidate_action_scores"] = scores
            best_action, best_score = best_candidate_action(scores)
            event["selected_action_reason"] = selected_action_reason(pointcloud_summary, scores)
            event["v0_selected_action"] = {"action": best_action, "score": best_score}
            event["shield_state"] = "NOT_APPLIED_CAPTURE"
            event["oscillation_risk"] = "LOW"
        except Exception as exc:
            event["pointcloud_summary_error"] = str(exc)
            event["candidate_action_scores"] = {}
            event["selected_action_reason"] = "v0 geometry summary failed"
            event["shield_state"] = "NOT_APPLIED_CAPTURE"
            event["oscillation_risk"] = "UNKNOWN"
        try:
            _, metadata = extract_event_features(event, base_dir=Path(str(result.get("capture_dir", "") or ".")).parent)
            event["feature_schema_status"] = "ok" if isinstance(metadata, dict) else "unknown"
        except Exception as exc:
            event["feature_schema_status"] = "error"
            event["feature_schema_error"] = str(exc)
        return event

    def obstacle_avoidance_resolve_event_path(self, event: Dict[str, Any], key: str, session_dir: Path) -> Path:
        raw = str(event.get(key, "") or "").strip()
        if not raw:
            return Path("__missing_obstacle_avoidance_path__")
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        capture_dir = Path(str(event.get("capture_dir", "") or "")).expanduser()
        if capture_dir and not capture_dir.is_absolute():
            capture_dir = session_dir / capture_dir
        if capture_dir and capture_dir.exists():
            candidate = capture_dir / path.name
            if candidate.exists():
                return candidate
        return session_dir / path

    def obstacle_avoidance_quality_report(self, session_dir: Path, *, last_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        events_path = session_dir / "avoidance_events.jsonl"
        rows: List[Dict[str, Any]] = []
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        missing_rgb = 0
        missing_depth = 0
        missing_pointcloud = 0
        missing_pose = 0
        missing_action = 0
        danger_forward = 0
        unsafe_down = 0
        collision_count = 0
        bbox_violation_count = 0
        for event in rows:
            if not self.obstacle_avoidance_resolve_event_path(event, "rgb_path", session_dir).exists():
                missing_rgb += 1
            depth_path = self.obstacle_avoidance_resolve_event_path(event, "depth_npy_path", session_dir)
            if not depth_path.exists():
                depth_path = self.obstacle_avoidance_resolve_event_path(event, "depth_cm_path", session_dir)
            if not depth_path.exists():
                missing_depth += 1
            if not self.obstacle_avoidance_resolve_event_path(event, "pointcloud_path", session_dir).exists():
                missing_pointcloud += 1
            if not isinstance(event.get("current_pose"), dict) or not event.get("current_pose"):
                missing_pose += 1
            action = event.get("executed_action")
            if not isinstance(action, dict):
                missing_action += 1
                action = {}
            summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
            front_min = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
            forward_cm = float(action.get("forward_cm", 0.0) or 0.0)
            up_cm = float(action.get("up_cm", 0.0) or 0.0)
            if front_min > 0.0 and front_min < 250.0 and forward_cm > 1.0:
                danger_forward += 1
            pose_z = 0.0
            pose = event.get("current_pose") if isinstance(event.get("current_pose"), dict) else {}
            try:
                pose_z = float(pose.get("z", 0.0) or 0.0)
            except Exception:
                pose_z = 0.0
            down_clear = bool(summary.get("down_swept_clear", False))
            if up_cm < -1.0 and (not down_clear or pose_z <= 100.0):
                unsafe_down += 1
            if bool(event.get("collision_state", event.get("collision", False))):
                collision_count += 1
            if bool(event.get("bbox_violation", event.get("bbox_violation_state", False))):
                bbox_violation_count += 1

        report = {
            "status": "ok",
            "session_dir": str(session_dir),
            "event_count": len(rows),
            "valid_frame_count": max(0, len(rows) - max(missing_rgb, missing_pointcloud, missing_pose, missing_action)),
            "missing_rgb_count": missing_rgb,
            "missing_depth_count": missing_depth,
            "missing_pointcloud_count": missing_pointcloud,
            "missing_pose_count": missing_pose,
            "missing_action_count": missing_action,
            "danger_forward_violation_count": danger_forward,
            "unsafe_down_action_count": unsafe_down,
            "collision_count": collision_count,
            "bbox_violation_count": bbox_violation_count,
            "last_frame_id": last_event.get("frame_id") if isinstance(last_event, dict) else (rows[-1].get("frame_id") if rows else None),
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        report_path = session_dir / "collection_quality_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        qa_root = self.obstacle_avoidance_data_root() / "qa"
        try:
            qa_root.mkdir(parents=True, exist_ok=True)
            (qa_root / f"{session_dir.name}_collection_quality_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        return report

    def obstacle_avoidance_write_session_summary(self, session_dir: Path, event: Dict[str, Any]) -> Dict[str, Any]:
        quality = self.obstacle_avoidance_quality_report(session_dir, last_event=event)
        summary = {
            "session_id": session_dir.name,
            "session_dir": str(session_dir),
            "collection_stage": event.get("collection_stage", self.obstacle_avoidance_stage_var.get()),
            "scenario_id": event.get("scenario_id", self.obstacle_avoidance_scenario_var.get()),
            "method": event.get("method", self.obstacle_avoidance_method_var.get()),
            "run_id": event.get("run_id", self.obstacle_avoidance_run_id_var.get()),
            "frame_count": self.obstacle_avoidance_frame_index,
            "valid_frame_count": quality.get("valid_frame_count", 0),
            "collision_count": quality.get("collision_count", 0),
            "bbox_violation_count": quality.get("bbox_violation_count", 0),
            "quality_report_path": str(session_dir / "collection_quality_report.json"),
            "last_event": event,
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        (session_dir / "avoidance_session_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def capture_obstacle_avoidance_frame(self, session: flight.DroneFlightSession, session_dir: Path) -> Dict[str, Any]:
        self.obstacle_avoidance_frame_index += 1
        action_detail = self.obstacle_avoidance_capture_action_detail()
        action_detail["session_id"] = session_dir.name
        result = self.safe(
            "Obstacle Avoidance Capture",
            lambda idx=self.obstacle_avoidance_frame_index, action=action_detail: session.capture_lidar_stream_frame(
                session_dir,
                idx,
                action_detail=action,
            ),
        )
        if not isinstance(result, dict):
            raise RuntimeError("capture_lidar_stream_frame failed")
        result = self.obstacle_avoidance_enrich_capture_result(result)
        event = self.obstacle_avoidance_build_event(result, action_detail)
        events_path = session_dir / "avoidance_events.jsonl"
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.obstacle_avoidance_write_session_summary(session_dir, event)
        return event

    def on_obstacle_avoidance_capture_once(self) -> None:
        self.ensure_obstacle_avoidance_state()
        session = self.active_session()
        if session is None:
            return
        if self.obstacle_avoidance_capture_thread is not None and self.obstacle_avoidance_capture_thread.is_alive():
            self.obstacle_avoidance_status_var.set("Obstacle Avoidance: timed capture is running")
            return
        self.sync_capture_options_to_session(session)
        if self.obstacle_avoidance_session_dir is None:
            self.obstacle_avoidance_session_dir = self.make_obstacle_avoidance_session_dir()
            (self.obstacle_avoidance_session_dir / "frames").mkdir(parents=True, exist_ok=True)

        def worker() -> None:
            try:
                event = self.capture_obstacle_avoidance_frame(session, self.obstacle_avoidance_session_dir)
                self.root.after(
                    0,
                    lambda e=event: (
                        self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance: captured frame {e.get('frame_id')}"),
                        self.obstacle_avoidance_append_report({"captured": e}),
                    ),
                )
            except Exception as exc:
                self.root.after(0, lambda e=exc: self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance capture failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def on_start_obstacle_avoidance_capture(self) -> None:
        self.ensure_obstacle_avoidance_state()
        session = self.active_session()
        if session is None:
            return
        if self.obstacle_avoidance_capture_thread is not None and self.obstacle_avoidance_capture_thread.is_alive():
            self.obstacle_avoidance_status_var.set("Obstacle Avoidance: already capturing")
            return
        if self.stream_capture_thread is not None and self.stream_capture_thread.is_alive():
            self.obstacle_avoidance_status_var.set("Obstacle Avoidance: stop Stream Capture first")
            return
        if self.lidar_stream_capture_thread is not None and self.lidar_stream_capture_thread.is_alive():
            self.obstacle_avoidance_status_var.set("Obstacle Avoidance: stop Lidar Stream first")
            return
        self.sync_capture_options_to_session(session)
        session_dir = self.make_obstacle_avoidance_session_dir()
        (session_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.obstacle_avoidance_session_dir = session_dir
        self.obstacle_avoidance_frame_index = 0
        self.obstacle_avoidance_stop_event.clear()
        interval_s = self.obstacle_avoidance_interval_s()
        self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance: capturing -> {session_dir}")

        def worker() -> None:
            try:
                while not self.obstacle_avoidance_stop_event.is_set():
                    started = time.monotonic()
                    try:
                        event = self.capture_obstacle_avoidance_frame(session, session_dir)
                        self.root.after(
                            0,
                            lambda e=event: self.obstacle_avoidance_status_var.set(
                                f"Obstacle Avoidance: captured {e.get('frame_id')} collision={e.get('collision_state')}"
                            ),
                        )
                    except Exception as exc:
                        self.root.after(0, lambda e=exc: self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance capture failed: {e}"))
                        break
                    elapsed = time.monotonic() - started
                    if self.obstacle_avoidance_stop_event.wait(max(0.0, interval_s - elapsed)):
                        break
            finally:
                quality: Dict[str, Any] = {}
                try:
                    quality = self.obstacle_avoidance_quality_report(session_dir)
                except Exception as exc:
                    quality = {"status": "error", "error": str(exc)}
                self.root.after(
                    0,
                    lambda d=session_dir, c=self.obstacle_avoidance_frame_index, q=quality: (
                        self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance: stopped, {c} frames -> {d}"),
                        self.obstacle_avoidance_append_report({"capture_stopped": str(d), "frame_count": c, "quality": q}),
                    ),
                )

        self.obstacle_avoidance_capture_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_capture_thread.start()

    def on_stop_obstacle_avoidance_capture(self) -> None:
        self.ensure_obstacle_avoidance_state()
        self.obstacle_avoidance_stop_event.set()
        self.obstacle_avoidance_status_var.set("Obstacle Avoidance: stopping capture...")

    def on_obstacle_avoidance_quality_report(self) -> None:
        self.ensure_obstacle_avoidance_state()
        session_dir = self.obstacle_avoidance_session_dir
        if session_dir is None:
            self.obstacle_avoidance_status_var.set("Obstacle Avoidance: no active session dir for QA")
            return
        self.run_obstacle_avoidance_task(
            "quality report",
            lambda: self.obstacle_avoidance_quality_report(session_dir),
        )

    def run_obstacle_avoidance_task(self, desc: str, fn) -> None:
        self.ensure_obstacle_avoidance_state()
        if self.obstacle_avoidance_task_thread is not None and self.obstacle_avoidance_task_thread.is_alive():
            self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance: wait for current task before {desc}")
            return

        def worker() -> None:
            self.root.after(0, lambda: self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance: {desc}..."))
            try:
                result = fn()
                self.root.after(
                    0,
                    lambda r=result: (
                        self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance: {desc} done"),
                        self.obstacle_avoidance_append_report(r if isinstance(r, dict) else {"result": str(r)}),
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: (
                        self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance {desc} failed: {e}"),
                        self.obstacle_avoidance_append_report({"status": "error", "task": desc, "error": str(e)}),
                    ),
                )

        self.obstacle_avoidance_task_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_task_thread.start()

    def on_build_obstacle_avoidance_dataset(self) -> None:
        self.run_obstacle_avoidance_task(
            "build dataset",
            lambda: build_dataset(self.obstacle_avoidance_data_root()),
        )

    def on_train_obstacle_avoidance_model(self) -> None:
        root = self.obstacle_avoidance_data_root()
        self.run_obstacle_avoidance_task(
            "train baseline",
            lambda: train_baseline(root / "datasets" / "dataset_latest.npz", model_path=root / "models" / "avoidance_agent_v0.npz"),
        )

    def on_validate_obstacle_avoidance_model(self) -> None:
        root = self.obstacle_avoidance_data_root()
        self.run_obstacle_avoidance_task(
            "validate model",
            lambda: validate_model(
                root / "datasets" / "dataset_latest.npz",
                root / "models" / "avoidance_agent_v0.npz",
                report_path=root / "validation" / "validation_report.json",
            ),
        )
