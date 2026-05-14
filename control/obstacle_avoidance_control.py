from __future__ import annotations

from .common import *

from obstacle_avoidance.dataset import build_dataset
from obstacle_avoidance.features import ACTION_PAYLOAD_TEMPLATES, extract_event_features
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
        window.title("Obstacle Avoidance Data / Train / Validate")
        window.geometry("1120x780")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(4, weight=1)
        self.obstacle_avoidance_window = window

        intro = tk.LabelFrame(window, text="Instructions")
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        intro.grid_columnconfigure(0, weight=1)
        text = (
            "1. Start Unreal/session, then fly manually or run route movement while this window captures RGB + depth + point cloud.\n"
            "2. Pick risk_state and expert_action before each capture or timed capture segment.\n"
            "3. If the UAV is hit/collided, enable Collision / 碰撞 or click Mark Collision before collecting the failed sample.\n"
            "4. In this v0, avoidance_failed is always equal to collision_state; real simulator collision signals can replace the manual label later.\n"
            "5. Build Dataset -> Train Baseline -> Validate Model creates npz/json artifacts under obstacle_avoidance_data."
        )
        tk.Label(intro, text=text, justify="left", anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        config = tk.LabelFrame(window, text="Capture Labels")
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
        tk.Label(config, text="Risk").grid(row=1, column=4, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_risk_var,
            values=("SAFE", "CAUTION", "BLOCKED", "RECOVERY", "REPLAN", "ABORT_WAYPOINT"),
            state="readonly",
            width=16,
        ).grid(row=1, column=5, sticky="w", padx=6, pady=6)
        tk.Label(config, text="Expert Action").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            config,
            textvariable=self.obstacle_avoidance_expert_action_var,
            values=tuple(sorted(ACTION_PAYLOAD_TEMPLATES.keys())),
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=6)
        tk.Checkbutton(config, text="Collision / 碰撞", variable=self.obstacle_avoidance_collision_var).grid(row=2, column=2, sticky="w", padx=6, pady=6)
        tk.Button(config, text="Mark Collision", command=self.mark_obstacle_avoidance_collision).grid(row=2, column=3, sticky="w", padx=6, pady=6)
        tk.Button(config, text="Clear Collision", command=self.clear_obstacle_avoidance_collision).grid(row=2, column=4, sticky="w", padx=6, pady=6)

        actions = tk.LabelFrame(window, text="Actions")
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(actions, text="Capture One Frame", command=self.on_obstacle_avoidance_capture_once).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Start Timed Capture", command=self.on_start_obstacle_avoidance_capture).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Stop Capture", command=self.on_stop_obstacle_avoidance_capture).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Build Dataset", command=self.on_build_obstacle_avoidance_dataset).pack(side="left", padx=(18, 6), pady=6)
        tk.Button(actions, text="Train Baseline", command=self.on_train_obstacle_avoidance_model).pack(side="left", padx=6, pady=6)
        tk.Button(actions, text="Validate Model", command=self.on_validate_obstacle_avoidance_model).pack(side="left", padx=6, pady=6)

        status = tk.LabelFrame(window, text="Status")
        status.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        status.grid_columnconfigure(0, weight=1)
        tk.Label(status, textvariable=self.obstacle_avoidance_status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        report_frame = tk.LabelFrame(window, text="Report")
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

        def close_window() -> None:
            self.obstacle_avoidance_window = None
            self.obstacle_avoidance_report_text = None
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
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return flight.make_unique_child_dir(root, f"{timestamp}_{title}")

    def obstacle_avoidance_current_target_waypoint(self) -> Dict[str, Any]:
        try:
            if isinstance(self.llm_route3_state, dict):
                for key in ("target_pose", "current_target", "active_scan_point"):
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

    def obstacle_avoidance_action_payload(self) -> Dict[str, float]:
        label = self.obstacle_avoidance_expert_action_var.get().strip().lower()
        return dict(ACTION_PAYLOAD_TEMPLATES.get(label, ACTION_PAYLOAD_TEMPLATES["hold"]))

    def obstacle_avoidance_capture_action_detail(self) -> Dict[str, Any]:
        latest_pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state, dict) and isinstance(self.latest_state.get("pose"), dict) else {}
        return {
            "source": "obstacle_avoidance_capture",
            "mission_phase": "MANUAL",
            "risk_state": self.obstacle_avoidance_risk_var.get().strip().upper() or "SAFE",
            "expert_action": self.obstacle_avoidance_expert_action_var.get().strip().lower() or "hold",
            "expert_action_payload": self.obstacle_avoidance_action_payload(),
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
        event = {
            "frame_id": int(result.get("frame_index", self.obstacle_avoidance_frame_index) or self.obstacle_avoidance_frame_index),
            "timestamp": result.get("capture_time", datetime.now().isoformat(timespec="milliseconds")),
            "mission_phase": action_detail.get("mission_phase", "MANUAL"),
            "current_pose": result.get("pose", {}),
            "pose": result.get("pose", {}),
            "target_waypoint": action_detail.get("target_waypoint", {}),
            "relative_target": {},
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
            _, metadata = extract_event_features(event, base_dir=Path(str(result.get("capture_dir", "") or ".")).parent)
            values = metadata.get("feature_values", {}) if isinstance(metadata, dict) else {}
            event["pointcloud_summary"] = {
                "available": bool(values.get("pc_available", 0.0)),
                "point_count": int(max(0.0, math.exp(float(values.get("pc_point_count_log", 0.0))) - 1.0)),
                "front_min_depth_cm": float(values.get("pc_front_min_cm", 0.0)),
                "front_mean_depth_cm": float(values.get("pc_front_mean_cm", 0.0)),
                "corridor_count": int(max(0.0, math.exp(float(values.get("pc_corridor_count_log", 0.0))) - 1.0)),
                "left_min_depth_cm": float(values.get("pc_left_min_cm", 0.0)),
                "right_min_depth_cm": float(values.get("pc_right_min_cm", 0.0)),
                "up_min_depth_cm": float(values.get("pc_up_min_cm", 0.0)),
                "nearest_distance_cm": float(values.get("pc_nearest_cm", 0.0)),
            }
        except Exception as exc:
            event["pointcloud_summary_error"] = str(exc)
        return event

    def capture_obstacle_avoidance_frame(self, session: flight.DroneFlightSession, session_dir: Path) -> Dict[str, Any]:
        self.obstacle_avoidance_frame_index += 1
        action_detail = self.obstacle_avoidance_capture_action_detail()
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
        summary = {
            "session_dir": str(session_dir),
            "frame_count": self.obstacle_avoidance_frame_index,
            "last_event": event,
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        (session_dir / "avoidance_session_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
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
                self.root.after(
                    0,
                    lambda d=session_dir, c=self.obstacle_avoidance_frame_index: (
                        self.obstacle_avoidance_status_var.set(f"Obstacle Avoidance: stopped, {c} frames -> {d}"),
                        self.obstacle_avoidance_append_report({"capture_stopped": str(d), "frame_count": c}),
                    ),
                )

        self.obstacle_avoidance_capture_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_avoidance_capture_thread.start()

    def on_stop_obstacle_avoidance_capture(self) -> None:
        self.ensure_obstacle_avoidance_state()
        self.obstacle_avoidance_stop_event.set()
        self.obstacle_avoidance_status_var.set("Obstacle Avoidance: stopping capture...")

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
