from __future__ import annotations

from .common import *


class ObstacleRepresentationControlMixin:
    def default_obstacle_representation_model_path(self) -> Path:
        plus = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_plus_model.pt"
        legacy = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_model.pt"
        return plus if plus.is_file() else legacy

    def default_obstacle_representation_2_model_path(self) -> Path:
        return PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"

    def ensure_obstacle_representation_state(self) -> None:
        if not hasattr(self, "obstacle_representation_model_var"):
            self.obstacle_representation_model_var = tk.StringVar(value=str(self.default_obstacle_representation_model_path()))
        if not hasattr(self, "obstacle_representation_2_model_var"):
            self.obstacle_representation_2_model_var = tk.StringVar(value=str(self.default_obstacle_representation_2_model_path()))
        if not hasattr(self, "obstacle_representation_status_var"):
            self.obstacle_representation_status_var = tk.StringVar(value="Obstacle Representation: idle")
        if not hasattr(self, "obstacle_representation_result_var"):
            self.obstacle_representation_result_var = tk.StringVar(value="Result: --")
        if not hasattr(self, "obstacle_representation_capture_dir_var"):
            self.obstacle_representation_capture_dir_var = tk.StringVar(value="Capture: --")
        if not hasattr(self, "obstacle_representation_window"):
            self.obstacle_representation_window = None
        if not hasattr(self, "obstacle_representation_rgb_label"):
            self.obstacle_representation_rgb_label = None
        if not hasattr(self, "obstacle_representation_mask_label"):
            self.obstacle_representation_mask_label = None
        if not hasattr(self, "obstacle_representation_rgb_photo"):
            self.obstacle_representation_rgb_photo = None
        if not hasattr(self, "obstacle_representation_mask_photo"):
            self.obstacle_representation_mask_photo = None
        if not hasattr(self, "obstacle_representation_report_text"):
            self.obstacle_representation_report_text = None
        if not hasattr(self, "obstacle_representation_thread"):
            self.obstacle_representation_thread = None
        if not hasattr(self, "obstacle_representation_frame_index"):
            self.obstacle_representation_frame_index = 0
        if not hasattr(self, "or2_monitor_status_var"):
            self.or2_monitor_status_var = tk.StringVar(value="OR2 Monitor: idle")
        if not hasattr(self, "or2_monitor_state_var"):
            self.or2_monitor_state_var = tk.StringVar(value="State: --")
        if not hasattr(self, "or2_monitor_capture_dir_var"):
            self.or2_monitor_capture_dir_var = tk.StringVar(value="Capture: --")
        if not hasattr(self, "or2_monitor_interval_s_var"):
            self.or2_monitor_interval_s_var = tk.StringVar(value="1.0")
        if not hasattr(self, "or2_monitor_frame_count_var"):
            self.or2_monitor_frame_count_var = tk.StringVar(value="Frames: 0")
        if not hasattr(self, "or2_monitor_front_depth_var"):
            self.or2_monitor_front_depth_var = tk.StringVar(value="Front depth: --")
        if not hasattr(self, "or2_monitor_can_forward_var"):
            self.or2_monitor_can_forward_var = tk.StringVar(value="Can forward: --")
        if not hasattr(self, "or2_monitor_window"):
            self.or2_monitor_window = None
        if not hasattr(self, "or2_monitor_state_label"):
            self.or2_monitor_state_label = None
        if not hasattr(self, "or2_monitor_rgb_label"):
            self.or2_monitor_rgb_label = None
        if not hasattr(self, "or2_monitor_mask_label"):
            self.or2_monitor_mask_label = None
        if not hasattr(self, "or2_monitor_rgb_photo"):
            self.or2_monitor_rgb_photo = None
        if not hasattr(self, "or2_monitor_mask_photo"):
            self.or2_monitor_mask_photo = None
        if not hasattr(self, "or2_monitor_report_text"):
            self.or2_monitor_report_text = None
        if not hasattr(self, "or2_monitor_thread"):
            self.or2_monitor_thread = None
        if not hasattr(self, "or2_monitor_stop_event"):
            self.or2_monitor_stop_event = threading.Event()

    def open_obstacle_representation_window(self) -> None:
        self.ensure_obstacle_representation_state()
        if self.obstacle_representation_window is not None and self.obstacle_representation_window.winfo_exists():
            self.obstacle_representation_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.obstacle_representation_window = window
        window.title("Obstacle Representation Demo")
        window.geometry("1120x760")
        window.protocol("WM_DELETE_WINDOW", self.close_obstacle_representation_window)

        storage = tk.LabelFrame(window, text="Scheme A / A+ Class Model")
        storage.pack(fill="x", padx=8, pady=6)
        storage.grid_columnconfigure(1, weight=1)
        tk.Label(storage, text="Model").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(storage, textvariable=self.obstacle_representation_model_var).grid(
            row=0, column=1, sticky="ew", padx=6, pady=6
        )
        tk.Button(storage, text="Browse", command=self.select_obstacle_representation_model).grid(
            row=0, column=2, padx=6, pady=6
        )
        tk.Button(storage, text="Default", command=self.use_default_obstacle_representation_model).grid(
            row=0, column=3, padx=6, pady=6
        )

        storage2 = tk.LabelFrame(window, text="Scheme A+2 Risk Model")
        storage2.pack(fill="x", padx=8, pady=4)
        storage2.grid_columnconfigure(1, weight=1)
        tk.Label(storage2, text="Model").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(storage2, textvariable=self.obstacle_representation_2_model_var).grid(
            row=0, column=1, sticky="ew", padx=6, pady=6
        )
        tk.Button(storage2, text="Browse", command=self.select_obstacle_representation_2_model).grid(
            row=0, column=2, padx=6, pady=6
        )
        tk.Button(storage2, text="Default", command=self.use_default_obstacle_representation_2_model).grid(
            row=0, column=3, padx=6, pady=6
        )

        actions = tk.LabelFrame(window, text="Demo Inference")
        actions.pack(fill="x", padx=8, pady=4)
        tk.Button(actions, text="Analyze Current Frame", command=self.run_obstacle_representation_demo).pack(
            side="left", padx=6, pady=6
        )
        tk.Button(actions, text="Analyze A+2 Risk", command=self.run_obstacle_representation_2_demo).pack(
            side="left", padx=6, pady=6
        )
        tk.Label(actions, textvariable=self.obstacle_representation_status_var, anchor="w").pack(
            side="left", fill="x", expand=True, padx=12, pady=6
        )

        result = tk.LabelFrame(window, text="Prediction")
        result.pack(fill="x", padx=8, pady=4)
        tk.Label(result, textvariable=self.obstacle_representation_result_var, anchor="w").pack(
            fill="x", padx=6, pady=(6, 2)
        )
        tk.Label(result, textvariable=self.obstacle_representation_capture_dir_var, anchor="w").pack(
            fill="x", padx=6, pady=(2, 6)
        )

        previews = tk.Frame(window)
        previews.pack(fill="both", expand=True, padx=8, pady=4)
        previews.grid_columnconfigure(0, weight=1)
        previews.grid_columnconfigure(1, weight=1)
        previews.grid_rowconfigure(0, weight=1)

        rgb_frame = tk.LabelFrame(previews, text="RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        mask_frame = tk.LabelFrame(previews, text="Mask / Risk Demo")
        mask_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)
        self.obstacle_representation_rgb_label = tk.Label(rgb_frame, text="No image")
        self.obstacle_representation_rgb_label.pack(fill="both", expand=True, padx=6, pady=6)
        self.obstacle_representation_mask_label = tk.Label(mask_frame, text="No mask")
        self.obstacle_representation_mask_label.pack(fill="both", expand=True, padx=6, pady=6)
        tk.Label(
            mask_frame,
            text="A+: depth obstacle mask + class color. A+2: yellow=250-450cm, light red=100-250cm, dark red<=100cm, top band=risk.",
            anchor="w",
        ).pack(fill="x", padx=6, pady=(0, 6))

        report = tk.LabelFrame(window, text="Report")
        report.pack(fill="both", padx=8, pady=(4, 8))
        self.obstacle_representation_report_text = tk.Text(report, height=7, wrap="word")
        self.obstacle_representation_report_text.pack(fill="both", expand=True, padx=6, pady=6)

    def close_obstacle_representation_window(self) -> None:
        if self.obstacle_representation_window is not None and self.obstacle_representation_window.winfo_exists():
            self.obstacle_representation_window.destroy()
        self.obstacle_representation_window = None
        self.obstacle_representation_rgb_label = None
        self.obstacle_representation_mask_label = None
        self.obstacle_representation_rgb_photo = None
        self.obstacle_representation_mask_photo = None
        self.obstacle_representation_report_text = None

    def open_obstacle_representation_2_monitor_window(self) -> None:
        self.ensure_obstacle_representation_state()
        if self.or2_monitor_window is not None and self.or2_monitor_window.winfo_exists():
            self.or2_monitor_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.or2_monitor_window = window
        window.title("Obstacle Representation 2 - Live Monitor")
        window.geometry("1120x780")
        window.protocol("WM_DELETE_WINDOW", self.close_obstacle_representation_2_monitor_window)

        storage = tk.LabelFrame(window, text="Model / Capture")
        storage.pack(fill="x", padx=8, pady=6)
        storage.grid_columnconfigure(1, weight=1)
        tk.Label(storage, text="Model").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(storage, textvariable=self.obstacle_representation_2_model_var).grid(
            row=0, column=1, sticky="ew", padx=6, pady=6
        )
        tk.Button(storage, text="Browse", command=self.select_obstacle_representation_2_model).grid(
            row=0, column=2, padx=6, pady=6
        )
        tk.Button(storage, text="Default", command=self.use_default_obstacle_representation_2_model).grid(
            row=0, column=3, padx=6, pady=6
        )
        tk.Label(storage, text="Interval s").grid(row=1, column=0, sticky="w", padx=6, pady=(0, 6))
        tk.Entry(storage, textvariable=self.or2_monitor_interval_s_var, width=10).grid(
            row=1, column=1, sticky="w", padx=6, pady=(0, 6)
        )
        tk.Label(storage, textvariable=self.or2_monitor_capture_dir_var, anchor="w").grid(
            row=1, column=2, columnspan=2, sticky="ew", padx=6, pady=(0, 6)
        )

        controls = tk.LabelFrame(window, text="Live Risk Monitor")
        controls.pack(fill="x", padx=8, pady=4)
        tk.Button(controls, text="Start Monitor", command=self.start_obstacle_representation_2_monitor).pack(
            side="left", padx=6, pady=6
        )
        tk.Button(controls, text="Stop Monitor", command=self.stop_obstacle_representation_2_monitor).pack(
            side="left", padx=6, pady=6
        )
        self.or2_monitor_state_label = tk.Label(
            controls,
            textvariable=self.or2_monitor_state_var,
            width=28,
            anchor="center",
            bg="#d9d9d9",
            fg="black",
        )
        self.or2_monitor_state_label.pack(side="left", padx=(18, 8), pady=6)
        for var in (
            self.or2_monitor_frame_count_var,
            self.or2_monitor_front_depth_var,
            self.or2_monitor_can_forward_var,
            self.or2_monitor_status_var,
        ):
            tk.Label(controls, textvariable=var, anchor="w").pack(side="left", padx=8, pady=6)

        legend = tk.LabelFrame(window, text="Risk Color Legend")
        legend.pack(fill="x", padx=8, pady=4)
        for label, color, text in (
            ("clear", "#3cbe5a", "safe"),
            ("clearance_warning", "#f5c82d", "250-450cm"),
            ("obstacle_warning", "#f5825f", "100-250cm"),
            ("must_stop", "#be1423", "<=100cm"),
        ):
            tk.Label(legend, text=f"{label}: {text}", bg=color, fg="white" if label == "must_stop" else "black").pack(
                side="left", padx=6, pady=6, ipadx=8, ipady=3
            )

        previews = tk.Frame(window)
        previews.pack(fill="both", expand=True, padx=8, pady=4)
        previews.grid_columnconfigure(0, weight=1)
        previews.grid_columnconfigure(1, weight=1)
        previews.grid_rowconfigure(0, weight=1)
        rgb_frame = tk.LabelFrame(previews, text="RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        mask_frame = tk.LabelFrame(previews, text="A+2 Risk Overlay")
        mask_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)
        self.or2_monitor_rgb_label = tk.Label(rgb_frame, text="No image")
        self.or2_monitor_rgb_label.pack(fill="both", expand=True, padx=6, pady=6)
        self.or2_monitor_mask_label = tk.Label(mask_frame, text="No overlay")
        self.or2_monitor_mask_label.pack(fill="both", expand=True, padx=6, pady=6)

        report = tk.LabelFrame(window, text="Latest Event")
        report.pack(fill="both", padx=8, pady=(4, 8))
        self.or2_monitor_report_text = tk.Text(report, height=8, wrap="word")
        self.or2_monitor_report_text.pack(fill="both", expand=True, padx=6, pady=6)

    def close_obstacle_representation_2_monitor_window(self) -> None:
        self.stop_obstacle_representation_2_monitor()
        if self.or2_monitor_window is not None and self.or2_monitor_window.winfo_exists():
            self.or2_monitor_window.destroy()
        self.or2_monitor_window = None
        self.or2_monitor_state_label = None
        self.or2_monitor_rgb_label = None
        self.or2_monitor_mask_label = None
        self.or2_monitor_rgb_photo = None
        self.or2_monitor_mask_photo = None
        self.or2_monitor_report_text = None

    def select_obstacle_representation_model(self) -> None:
        self.ensure_obstacle_representation_state()
        path = filedialog.askopenfilename(
            title="Select Scheme A model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT / "obstacle_representation_data" / "models"),
        )
        if path:
            self.obstacle_representation_model_var.set(path)

    def use_default_obstacle_representation_model(self) -> None:
        self.ensure_obstacle_representation_state()
        self.obstacle_representation_model_var.set(str(self.default_obstacle_representation_model_path()))

    def select_obstacle_representation_2_model(self) -> None:
        self.ensure_obstacle_representation_state()
        path = filedialog.askopenfilename(
            title="Select Scheme A+2 risk model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT / "obstacle_representation_2_data" / "models"),
        )
        if path:
            self.obstacle_representation_2_model_var.set(path)

    def use_default_obstacle_representation_2_model(self) -> None:
        self.ensure_obstacle_representation_state()
        self.obstacle_representation_2_model_var.set(str(self.default_obstacle_representation_2_model_path()))

    def run_obstacle_representation_demo(self) -> None:
        self.ensure_obstacle_representation_state()
        session = self.active_session()
        if session is None:
            return
        thread = getattr(self, "obstacle_representation_thread", None)
        if thread is not None and thread.is_alive():
            self.obstacle_representation_status_var.set("Obstacle Representation: analysis already running")
            return
        model_path = Path(self.obstacle_representation_model_var.get().strip()).expanduser()
        if not model_path.is_file():
            self.obstacle_representation_status_var.set(f"Obstacle Representation: model not found: {model_path}")
            return

        def worker() -> None:
            self.root.after(
                0,
                lambda: self.obstacle_representation_status_var.set(
                    "Obstacle Representation: capturing current frame..."
                ),
            )
            try:
                result = self._capture_and_predict_obstacle_representation(session, model_path)
                self.root.after(0, lambda r=result: self.apply_obstacle_representation_result(r))
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: self.obstacle_representation_status_var.set(
                        f"Obstacle Representation failed: {e}"
                    ),
                )

        self.obstacle_representation_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_representation_thread.start()

    def run_obstacle_representation_2_demo(self) -> None:
        self.ensure_obstacle_representation_state()
        session = self.active_session()
        if session is None:
            return
        thread = getattr(self, "obstacle_representation_thread", None)
        if thread is not None and thread.is_alive():
            self.obstacle_representation_status_var.set("Obstacle Representation: analysis already running")
            return
        model_path = Path(self.obstacle_representation_2_model_var.get().strip()).expanduser()
        if not model_path.is_file():
            self.obstacle_representation_status_var.set(f"Obstacle Representation 2: model not found: {model_path}")
            return

        def worker() -> None:
            self.root.after(
                0,
                lambda: self.obstacle_representation_status_var.set(
                    "Obstacle Representation 2: capturing current frame..."
                ),
            )
            try:
                result = self._capture_and_predict_obstacle_representation_2(session, model_path)
                self.root.after(0, lambda r=result: self.apply_obstacle_representation_2_result(r))
            except Exception as exc:
                self.root.after(
                    0,
                    lambda e=exc: self.obstacle_representation_status_var.set(
                        f"Obstacle Representation 2 failed: {e}"
                    ),
                )

        self.obstacle_representation_thread = threading.Thread(target=worker, daemon=True)
        self.obstacle_representation_thread.start()

    def parse_or2_monitor_interval_s(self) -> float:
        try:
            value = float(self.or2_monitor_interval_s_var.get().strip())
        except Exception:
            value = 1.0
        if not math.isfinite(value):
            value = 1.0
        return max(0.2, min(30.0, value))

    def start_obstacle_representation_2_monitor(self) -> None:
        self.ensure_obstacle_representation_state()
        session = self.active_session()
        if session is None:
            self.or2_monitor_status_var.set("OR2 Monitor: start UE session first")
            return
        thread = getattr(self, "or2_monitor_thread", None)
        if thread is not None and thread.is_alive():
            self.or2_monitor_status_var.set("OR2 Monitor: already running")
            return
        model_path = Path(self.obstacle_representation_2_model_var.get().strip()).expanduser()
        if not model_path.is_file():
            self.or2_monitor_status_var.set(f"OR2 Monitor: model not found: {model_path}")
            return
        interval_s = self.parse_or2_monitor_interval_s()
        self.or2_monitor_interval_s_var.set(f"{interval_s:g}")
        run_dir = (
            PROJECT_ROOT
            / "obstacle_representation_2_data"
            / "live_monitor"
            / datetime.now().strftime("%Y%m%d-%H%M%S_or2_risk_monitor")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        self.or2_monitor_stop_event.clear()
        self.or2_monitor_capture_dir_var.set(f"Capture: {run_dir}")
        self.or2_monitor_frame_count_var.set("Frames: 0")
        self.or2_monitor_front_depth_var.set("Front depth: --")
        self.or2_monitor_can_forward_var.set("Can forward: --")
        self.or2_monitor_state_var.set("State: starting")
        self.or2_monitor_status_var.set("OR2 Monitor: starting")
        self.update_or2_monitor_state_color("starting")

        def worker() -> None:
            try:
                self._run_obstacle_representation_2_monitor(session, model_path, run_dir, interval_s)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self.or2_monitor_status_var.set(f"OR2 Monitor failed: {e}"))
                self.root.after(0, lambda: self.update_or2_monitor_state_color("failed"))

        self.or2_monitor_thread = threading.Thread(target=worker, daemon=True)
        self.or2_monitor_thread.start()

    def stop_obstacle_representation_2_monitor(self) -> None:
        self.ensure_obstacle_representation_state()
        self.or2_monitor_stop_event.set()
        thread = getattr(self, "or2_monitor_thread", None)
        if thread is not None and thread.is_alive():
            self.or2_monitor_status_var.set("OR2 Monitor: stopping")
        else:
            self.or2_monitor_status_var.set("OR2 Monitor: stopped")

    def _run_obstacle_representation_2_monitor(
        self,
        session: flight.DroneFlightSession,
        model_path: Path,
        run_dir: Path,
        interval_s: float,
    ) -> None:
        from obstacle_representation_2.demo import ObstacleRepresentation2Predictor, render_affordance_overlay

        started_at = datetime.now().isoformat(timespec="milliseconds")
        events_path = run_dir / "risk_events.jsonl"
        summary_path = run_dir / "risk_monitor_summary.json"
        predictor = ObstacleRepresentation2Predictor(model_path)
        state_counts: Dict[str, int] = {}
        frame_count = 0
        must_stop_count = 0
        warning_count = 0
        min_front_depth_cm: Optional[float] = None
        old_mode = str(getattr(session.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING))
        self.sync_capture_options_to_session(session)
        old_mode = str(getattr(session.args, "lidar_capture_processing", old_mode))
        try:
            session.args.lidar_capture_processing = "minimal"
            self.root.after(0, lambda: self.or2_monitor_status_var.set("OR2 Monitor: running"))
            frame_index = 1
            while not self.or2_monitor_stop_event.is_set():
                tick_start = time.time()
                capture = session.capture_lidar_stream_frame(
                    str(run_dir),
                    frame_index,
                    {
                        "source": "obstacle_representation_2_live_monitor",
                        "action_name": "or2_risk_monitor",
                    },
                )
                summary = capture.get("pointcloud_summary")
                if not isinstance(summary, dict):
                    summary = (
                        capture.get("depth_obstacle_summary")
                        if isinstance(capture.get("depth_obstacle_summary"), dict)
                        else {}
                    )
                event = {
                    "rgb_path": capture.get("rgb_path", ""),
                    "depth_npy_path": capture.get("depth_npy_path", ""),
                    "capture_dir": capture.get("capture_dir", ""),
                    "pose": capture.get("pose", {}),
                    "pointcloud_summary": summary,
                    "depth_obstacle_summary": capture.get("depth_obstacle_summary", {}),
                    "relative_target": {},
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
                prediction_payload = self.or2_monitor_serializable_prediction(prediction)
                prediction_payload.update(
                    {
                        "prediction_latency_ms": prediction_latency_ms,
                        "risk_overlay_path": str(overlay_path),
                    }
                )
                prediction_path.write_text(
                    json.dumps(prediction_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                risk = str(prediction.get("front_risk_state", "clear"))
                front_depth = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
                if front_depth > 0.0:
                    min_front_depth_cm = (
                        front_depth if min_front_depth_cm is None else min(min_front_depth_cm, front_depth)
                    )
                state_counts[risk] = state_counts.get(risk, 0) + 1
                frame_count += 1
                if risk == "must_stop":
                    must_stop_count += 1
                if risk in {"clearance_warning", "obstacle_warning"}:
                    warning_count += 1
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
                    "front_risk_state": risk,
                    "can_forward": bool(prediction.get("can_forward", False)),
                    "must_stop": bool(prediction.get("must_stop", False)),
                    "front_min_depth_cm": front_depth,
                    "risk_fractions": {
                        "clearance_warning": float(prediction.get("front_clearance_fraction", 0.0) or 0.0),
                        "obstacle_warning": float(prediction.get("front_warning_fraction", 0.0) or 0.0),
                        "must_stop": float(prediction.get("front_stop_fraction", 0.0) or 0.0),
                    },
                    "pointcloud_summary": summary,
                    "prediction_latency_ms": prediction_latency_ms,
                    "session_pose": capture.get("pose", {}),
                }
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event_record, ensure_ascii=False) + "\n")
                result = {
                    "capture": capture,
                    "event": event,
                    "event_record": event_record,
                    "prediction": prediction,
                    "rgb_image": rgb_image,
                    "mask_image": mask_image,
                    "frame_count": frame_count,
                }
                self.root.after(0, lambda r=result: self.apply_obstacle_representation_2_monitor_result(r))
                frame_index += 1
                elapsed = time.time() - tick_start
                remaining = max(0.0, interval_s - elapsed)
                if self.or2_monitor_stop_event.wait(remaining):
                    break
        finally:
            session.args.lidar_capture_processing = old_mode
            summary_payload = {
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="milliseconds"),
                "capture_dir": str(run_dir),
                "frame_count": int(frame_count),
                "state_counts": state_counts,
                "must_stop_count": int(must_stop_count),
                "warning_count": int(warning_count),
                "min_front_depth_cm": min_front_depth_cm,
                "model_path": str(model_path),
                "events_path": str(events_path),
            }
            summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self.root.after(
                0,
                lambda p=summary_path, n=frame_count: self.or2_monitor_status_var.set(
                    f"OR2 Monitor: stopped, frames={n}, summary={p}"
                ),
            )

    def or2_monitor_serializable_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in prediction.items()
            if key not in {"clearance_warning_mask", "obstacle_warning_mask", "must_stop_mask"}
        }

    def update_or2_monitor_state_color(self, risk: str) -> None:
        if self.or2_monitor_state_label is None:
            return
        colors = {
            "clear": ("#3cbe5a", "black"),
            "clearance_warning": ("#f5c82d", "black"),
            "obstacle_warning": ("#f5825f", "black"),
            "must_stop": ("#be1423", "white"),
            "starting": ("#d9d9d9", "black"),
            "failed": ("#7a1f1f", "white"),
        }
        bg, fg = colors.get(str(risk), ("#d9d9d9", "black"))
        self.or2_monitor_state_label.configure(bg=bg, fg=fg)

    def _capture_and_predict_obstacle_representation(
        self,
        session: flight.DroneFlightSession,
        model_path: Path,
    ) -> Dict[str, Any]:
        self.sync_capture_options_to_session(session)
        demo_root = PROJECT_ROOT / "obstacle_representation_data" / "demo_captures"
        run_dir = demo_root / datetime.now().strftime("%Y%m%d-%H%M%S_obstacle_representation")
        old_mode = str(getattr(session.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING))
        try:
            session.args.lidar_capture_processing = "minimal"
            frame_index = int(getattr(self, "obstacle_representation_frame_index", 0))
            self.obstacle_representation_frame_index = frame_index + 1
            capture = session.capture_lidar_stream_frame(
                str(run_dir),
                frame_index,
                {
                    "source": "obstacle_representation_demo",
                    "action_name": "analyze_current_frame",
                },
            )
        finally:
            session.args.lidar_capture_processing = old_mode
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
            "relative_target": {},
        }
        from obstacle_representation.demo import predict_obstacle_representation, render_prediction_mask

        prediction = predict_obstacle_representation(model_path, event["rgb_path"], event)
        event["is_manual_hard_case"] = self.obstacle_representation_is_manual_hard_case(event)
        rgb_image = np.asarray(Image.open(event["rgb_path"]).convert("RGB"), dtype=np.uint8)
        depth_path = Path(str(event.get("depth_npy_path", "") or ""))
        depth_image = np.load(depth_path) if depth_path.is_file() else None
        mask_image = render_prediction_mask(rgb_image, depth_image, prediction)
        return {
            "capture": capture,
            "event": event,
            "prediction": prediction,
            "rgb_image": rgb_image,
            "mask_image": mask_image,
        }

    def _capture_and_predict_obstacle_representation_2(
        self,
        session: flight.DroneFlightSession,
        model_path: Path,
    ) -> Dict[str, Any]:
        self.sync_capture_options_to_session(session)
        demo_root = PROJECT_ROOT / "obstacle_representation_2_data" / "demo_captures"
        run_dir = demo_root / datetime.now().strftime("%Y%m%d-%H%M%S_obstacle_representation_2")
        old_mode = str(getattr(session.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING))
        try:
            session.args.lidar_capture_processing = "minimal"
            frame_index = int(getattr(self, "obstacle_representation_frame_index", 0))
            self.obstacle_representation_frame_index = frame_index + 1
            capture = session.capture_lidar_stream_frame(
                str(run_dir),
                frame_index,
                {
                    "source": "obstacle_representation_2_demo",
                    "action_name": "analyze_a_plus_2_direction",
                },
            )
        finally:
            session.args.lidar_capture_processing = old_mode
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
            "relative_target": {},
        }
        from obstacle_representation_2.demo import predict_obstacle_representation_2, render_affordance_overlay

        prediction = predict_obstacle_representation_2(model_path, event["rgb_path"], event)
        rgb_image = np.asarray(Image.open(event["rgb_path"]).convert("RGB"), dtype=np.uint8)
        mask_image = render_affordance_overlay(rgb_image, prediction)
        return {
            "capture": capture,
            "event": event,
            "prediction": prediction,
            "rgb_image": rgb_image,
            "mask_image": mask_image,
        }

    def obstacle_representation_array_to_photo(
        self,
        image: np.ndarray,
        *,
        max_width: int = 520,
        max_height: int = 360,
    ) -> ImageTk.PhotoImage:
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        scale = min(max_width / max(1, pil.width), max_height / max(1, pil.height), 1.0)
        if scale < 1.0:
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def apply_obstacle_representation_result(self, result: Dict[str, Any]) -> None:
        prediction = result.get("prediction") if isinstance(result.get("prediction"), dict) else {}
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        label = str(prediction.get("predicted_label", "unknown"))
        confidence = float(prediction.get("confidence", 0.0) or 0.0)
        flyover_probability = float(prediction.get("flyover_probability", 0.0) or 0.0)
        self.obstacle_representation_status_var.set("Obstacle Representation: done")
        self.obstacle_representation_result_var.set(
            f"Result: label={label}, confidence={confidence:.3f}, "
            f"flyover={flyover_probability:.3f}, front={float(summary.get('front_min_depth_cm', 0.0) or 0.0):.1f} cm, "
            f"model={prediction.get('model_version', 'unknown')}"
        )
        self.obstacle_representation_capture_dir_var.set(f"Capture: {event.get('capture_dir', '--')}")
        if self.obstacle_representation_rgb_label is not None and isinstance(result.get("rgb_image"), np.ndarray):
            self.obstacle_representation_rgb_photo = self.obstacle_representation_array_to_photo(result["rgb_image"])
            self.obstacle_representation_rgb_label.configure(image=self.obstacle_representation_rgb_photo, text="")
        if self.obstacle_representation_mask_label is not None and isinstance(result.get("mask_image"), np.ndarray):
            self.obstacle_representation_mask_photo = self.obstacle_representation_array_to_photo(result["mask_image"])
            self.obstacle_representation_mask_label.configure(image=self.obstacle_representation_mask_photo, text="")
        if self.obstacle_representation_report_text is not None:
            payload = {
                "prediction": prediction,
                "pointcloud_summary": summary,
                "is_manual_hard_case": bool(event.get("is_manual_hard_case", False)),
                "capture_dir": event.get("capture_dir", ""),
            }
            self.obstacle_representation_report_text.delete("1.0", "end")
            self.obstacle_representation_report_text.insert("end", json.dumps(payload, indent=2, ensure_ascii=False))

    def apply_obstacle_representation_2_result(self, result: Dict[str, Any]) -> None:
        prediction = result.get("prediction") if isinstance(result.get("prediction"), dict) else {}
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        risk = str(prediction.get("front_risk_state", "clear"))
        clearance = float(prediction.get("front_clearance_fraction", 0.0) or 0.0)
        warning = float(prediction.get("front_warning_fraction", 0.0) or 0.0)
        stop = float(prediction.get("front_stop_fraction", 0.0) or 0.0)
        self.obstacle_representation_status_var.set("Obstacle Representation 2: done")
        self.obstacle_representation_result_var.set(
            f"Result: risk={risk}, can_forward={bool(prediction.get('can_forward', False))}, "
            f"must_stop={bool(prediction.get('must_stop', False))}, "
            f"front_yellow={clearance:.3f}, front_light_red={warning:.3f}, front_dark_red={stop:.3f}, "
            f"model={prediction.get('model_version', 'unknown')}"
        )
        self.obstacle_representation_capture_dir_var.set(f"Capture: {event.get('capture_dir', '--')}")
        if self.obstacle_representation_rgb_label is not None and isinstance(result.get("rgb_image"), np.ndarray):
            self.obstacle_representation_rgb_photo = self.obstacle_representation_array_to_photo(result["rgb_image"])
            self.obstacle_representation_rgb_label.configure(image=self.obstacle_representation_rgb_photo, text="")
        if self.obstacle_representation_mask_label is not None and isinstance(result.get("mask_image"), np.ndarray):
            self.obstacle_representation_mask_photo = self.obstacle_representation_array_to_photo(result["mask_image"])
            self.obstacle_representation_mask_label.configure(image=self.obstacle_representation_mask_photo, text="")
        if self.obstacle_representation_report_text is not None:
            payload = {
                "prediction": {
                    key: value
                    for key, value in prediction.items()
                    if key not in {"clearance_warning_mask", "obstacle_warning_mask", "must_stop_mask"}
                },
                "pointcloud_summary": summary,
                "capture_dir": event.get("capture_dir", ""),
            }
            self.obstacle_representation_report_text.delete("1.0", "end")
            self.obstacle_representation_report_text.insert("end", json.dumps(payload, indent=2, ensure_ascii=False))

    def apply_obstacle_representation_2_monitor_result(self, result: Dict[str, Any]) -> None:
        prediction = result.get("prediction") if isinstance(result.get("prediction"), dict) else {}
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        event_record = result.get("event_record") if isinstance(result.get("event_record"), dict) else {}
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        risk = str(prediction.get("front_risk_state", "clear"))
        can_forward = bool(prediction.get("can_forward", False))
        must_stop = bool(prediction.get("must_stop", False))
        front_depth = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
        frame_count = int(result.get("frame_count", 0) or 0)
        clearance = float(prediction.get("front_clearance_fraction", 0.0) or 0.0)
        warning = float(prediction.get("front_warning_fraction", 0.0) or 0.0)
        stop = float(prediction.get("front_stop_fraction", 0.0) or 0.0)
        self.or2_monitor_state_var.set(f"State: {risk}")
        self.or2_monitor_frame_count_var.set(f"Frames: {frame_count}")
        self.or2_monitor_front_depth_var.set(f"Front depth: {front_depth:.1f} cm")
        self.or2_monitor_can_forward_var.set(
            f"Can forward: {'yes' if can_forward else 'no'} / must stop: {'yes' if must_stop else 'no'}"
        )
        self.or2_monitor_status_var.set(
            f"OR2 Monitor: running, yellow={clearance:.3f}, light_red={warning:.3f}, dark_red={stop:.3f}"
        )
        self.update_or2_monitor_state_color(risk)
        if self.or2_monitor_rgb_label is not None and isinstance(result.get("rgb_image"), np.ndarray):
            self.or2_monitor_rgb_photo = self.obstacle_representation_array_to_photo(
                result["rgb_image"],
                max_width=520,
                max_height=340,
            )
            self.or2_monitor_rgb_label.configure(image=self.or2_monitor_rgb_photo, text="")
        if self.or2_monitor_mask_label is not None and isinstance(result.get("mask_image"), np.ndarray):
            self.or2_monitor_mask_photo = self.obstacle_representation_array_to_photo(
                result["mask_image"],
                max_width=520,
                max_height=340,
            )
            self.or2_monitor_mask_label.configure(image=self.or2_monitor_mask_photo, text="")
        if self.or2_monitor_report_text is not None:
            payload = {
                "event": event_record,
                "prediction": self.or2_monitor_serializable_prediction(prediction),
                "pointcloud_summary": summary,
            }
            self.or2_monitor_report_text.delete("1.0", "end")
            self.or2_monitor_report_text.insert("end", json.dumps(payload, indent=2, ensure_ascii=False))

    def obstacle_representation_is_manual_hard_case(self, event: Dict[str, Any]) -> bool:
        manual_path = PROJECT_ROOT / "obstacle_representation_data" / "manual_labels" / "hard_cases.jsonl"
        if not manual_path.is_file():
            return False
        rgb_path = str(event.get("rgb_path", "") or "")
        capture_dir = str(event.get("capture_dir", "") or "")
        try:
            for line in manual_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                row_rgb = str(row.get("rgb_path", "") or "")
                row_capture = str(row.get("capture_json_path", "") or "")
                if row_rgb and (row_rgb in rgb_path or Path(row_rgb).name == Path(rgb_path).name and Path(row_rgb).parent.name == Path(rgb_path).parent.name):
                    return True
                if row_capture and capture_dir and Path(row_capture).parent.name == Path(capture_dir).name:
                    return True
        except Exception:
            return False
        return False
