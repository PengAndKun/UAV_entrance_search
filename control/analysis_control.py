from __future__ import annotations

from .common import *
from .stream_analysis import (
    find_default_weights,
    run_stream_analysis,
    scan_stream_frames,
)


class AnalysisControlMixin:
    def ensure_stream_analysis_state(self) -> None:
        if not hasattr(self, "stream_analysis_window"):
            self.stream_analysis_window = None
        if not hasattr(self, "stream_analysis_thread"):
            self.stream_analysis_thread = None
        if not hasattr(self, "stream_analysis_stop_event"):
            self.stream_analysis_stop_event = threading.Event()
        if not hasattr(self, "stream_analysis_rows"):
            self.stream_analysis_rows = []
        if not hasattr(self, "stream_analysis_result"):
            self.stream_analysis_result = {}
        if not hasattr(self, "stream_analysis_preview_photo"):
            self.stream_analysis_preview_photo = None

    def open_stream_analysis_window(self) -> None:
        self.ensure_stream_analysis_state()
        if self.stream_analysis_window is not None and self.stream_analysis_window.winfo_exists():
            self.stream_analysis_window.lift()
            return

        if not self.stream_analysis_stream_dir_var.get().strip():
            default_dir = self.stream_player_dir or self.stream_capture_dir or self.find_latest_stream_capture_dir()
            if default_dir is not None:
                self.stream_analysis_stream_dir_var.set(str(default_dir))
        if not self.stream_analysis_weights_var.get().strip():
            try:
                weights = find_default_weights()
                self.stream_analysis_weights_var.set(str(weights))
            except Exception as exc:
                self.stream_analysis_status_var.set(f"Analysis: weights auto-discovery failed: {exc}")

        self.stream_analysis_window = tk.Toplevel(self.root)
        self.stream_analysis_window.title("Stream Capture Analysis")
        self.stream_analysis_window.geometry("1320x860")
        self.stream_analysis_window.protocol("WM_DELETE_WINDOW", self.close_stream_analysis_window)
        self.stream_analysis_window.grid_columnconfigure(0, weight=1)
        self.stream_analysis_window.grid_rowconfigure(2, weight=1)

        top = tk.Frame(self.stream_analysis_window)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(5, weight=1)

        tk.Label(top, text="Stream Folder").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=3)
        tk.Entry(top, textvariable=self.stream_analysis_stream_dir_var).grid(row=0, column=1, columnspan=3, sticky="ew", padx=4, pady=3)
        tk.Button(top, text="Browse", command=self.select_stream_analysis_folder).grid(row=0, column=4, padx=4, pady=3)
        tk.Button(top, text="Latest", command=self.use_latest_stream_analysis_folder).grid(row=0, column=5, sticky="w", padx=4, pady=3)

        tk.Label(top, text="Weights").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=3)
        tk.Entry(top, textvariable=self.stream_analysis_weights_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=3)
        tk.Button(top, text="Browse", command=self.select_stream_analysis_weights).grid(row=1, column=4, padx=4, pady=3)
        tk.Button(top, text="Auto", command=self.auto_stream_analysis_weights).grid(row=1, column=5, sticky="w", padx=4, pady=3)

        options = tk.Frame(self.stream_analysis_window)
        options.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for idx in range(14):
            options.grid_columnconfigure(idx, weight=0)
        options.grid_columnconfigure(13, weight=1)
        tk.Label(options, text="Device").grid(row=0, column=0, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=self.stream_analysis_device_var, width=8).grid(row=0, column=1, padx=(0, 8), pady=3)
        tk.Label(options, text="Conf").grid(row=0, column=2, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=self.stream_analysis_conf_var, width=7).grid(row=0, column=3, padx=(0, 8), pady=3)
        tk.Label(options, text="ImgSz").grid(row=0, column=4, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=self.stream_analysis_imgsz_var, width=7).grid(row=0, column=5, padx=(0, 8), pady=3)
        tk.Label(options, text="Stride").grid(row=0, column=6, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=self.stream_analysis_stride_var, width=7).grid(row=0, column=7, padx=(0, 8), pady=3)
        tk.Label(options, text="Max Frames").grid(row=0, column=8, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=self.stream_analysis_max_frames_var, width=8).grid(row=0, column=9, padx=(0, 8), pady=3)
        tk.Button(options, text="Start", command=self.start_stream_analysis).grid(row=0, column=10, padx=4, pady=3)
        tk.Button(options, text="Stop", command=self.stop_stream_analysis).grid(row=0, column=11, padx=4, pady=3)
        tk.Button(options, text="Load Last Results", command=self.load_latest_stream_analysis_results).grid(row=0, column=12, padx=4, pady=3)

        self.stream_analysis_progressbar = ttk.Progressbar(
            options,
            variable=self.stream_analysis_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.stream_analysis_progressbar.grid(row=1, column=0, columnspan=13, sticky="ew", pady=(4, 0))

        body = tk.PanedWindow(self.stream_analysis_window, orient="horizontal", sashrelief="raised")
        body.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = tk.Frame(body)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.stream_analysis_listbox = tk.Listbox(left, width=42, exportselection=False)
        list_scroll = tk.Scrollbar(left, orient="vertical", command=self.stream_analysis_listbox.yview)
        self.stream_analysis_listbox.configure(yscrollcommand=list_scroll.set)
        self.stream_analysis_listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.stream_analysis_listbox.bind("<<ListboxSelect>>", self.on_stream_analysis_frame_selected)
        body.add(left, minsize=320)

        right = tk.Frame(body)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=0)
        right.grid_rowconfigure(2, weight=1)

        top_view = tk.PanedWindow(right, orient="horizontal", sashrelief="raised")
        top_view.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        preview_frame = tk.Frame(top_view)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        self.stream_analysis_preview_label = tk.Label(preview_frame, text="Run analysis or load existing results.", anchor="center")
        self.stream_analysis_preview_label.grid(row=0, column=0, sticky="nsew")
        top_view.add(preview_frame, minsize=560)

        map_frame = tk.LabelFrame(top_view, text="Map Pose")
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(1, weight=1)
        tk.Label(map_frame, textvariable=self.stream_analysis_map_pose_var, anchor="w", justify="left").grid(
            row=0,
            column=0,
            sticky="ew",
            padx=6,
            pady=(4, 2),
        )
        self.load_map_resources(force=not bool(getattr(self, "map_config", None)))
        self.stream_analysis_map_widget = OverheadMapWidget(
            map_frame,
            world_bounds=self.map_world_bounds,
            canvas_w=360,
            canvas_h=260,
        )
        self.stream_analysis_map_widget.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        controls = tk.Frame(map_frame)
        controls.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 4))
        for col in range(6):
            controls.grid_columnconfigure(col, weight=0)
        controls.grid_columnconfigure(5, weight=1)
        tk.Checkbutton(
            controls,
            text="Live Pose",
            variable=self.stream_analysis_live_pose_var,
            command=self.on_toggle_stream_analysis_live_pose,
        ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        tk.Label(controls, text="Step px").grid(row=0, column=1, sticky="e", padx=(0, 2), pady=2)
        tk.Entry(controls, textvariable=self.stream_analysis_map_shift_step_var, width=6).grid(row=0, column=2, sticky="w", padx=(0, 6), pady=2)
        tk.Button(controls, text="Reset Shift", command=self.reset_stream_analysis_map_shift).grid(row=0, column=3, padx=3, pady=2)
        tk.Button(controls, text="Save Shifted Config", command=self.save_stream_analysis_shifted_config).grid(row=0, column=4, padx=3, pady=2)

        arrows = tk.Frame(map_frame)
        arrows.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        tk.Button(arrows, text="Map Left", command=lambda: self.shift_stream_analysis_map(-1.0, 0.0)).pack(side="left", padx=3)
        tk.Button(arrows, text="Map Right", command=lambda: self.shift_stream_analysis_map(1.0, 0.0)).pack(side="left", padx=3)
        tk.Button(arrows, text="Map Up", command=lambda: self.shift_stream_analysis_map(0.0, -1.0)).pack(side="left", padx=3)
        tk.Button(arrows, text="Map Down", command=lambda: self.shift_stream_analysis_map(0.0, 1.0)).pack(side="left", padx=3)
        top_view.add(map_frame, minsize=380)

        self.stream_analysis_summary_text = tk.Text(right, height=7, wrap="word")
        self.stream_analysis_summary_text.grid(row=1, column=0, sticky="ew", padx=(8, 0), pady=(0, 8))
        json_frame = tk.Frame(right)
        json_frame.grid(row=2, column=0, sticky="nsew", padx=(8, 0))
        json_frame.grid_rowconfigure(0, weight=1)
        json_frame.grid_columnconfigure(0, weight=1)
        self.stream_analysis_json_text = tk.Text(json_frame, height=12, wrap="none")
        json_scroll = tk.Scrollbar(json_frame, orient="vertical", command=self.stream_analysis_json_text.yview)
        self.stream_analysis_json_text.configure(yscrollcommand=json_scroll.set)
        self.stream_analysis_json_text.grid(row=0, column=0, sticky="nsew")
        json_scroll.grid(row=0, column=1, sticky="ns")
        body.add(right, minsize=720)

        tk.Label(self.stream_analysis_window, textvariable=self.stream_analysis_status_var, anchor="w").grid(
            row=3,
            column=0,
            sticky="ew",
            padx=8,
            pady=(0, 8),
        )
        self.populate_stream_analysis_results(self.stream_analysis_rows)

    def close_stream_analysis_window(self) -> None:
        self.stop_stream_analysis()
        if self.stream_analysis_window is not None and self.stream_analysis_window.winfo_exists():
            self.stream_analysis_window.destroy()
        self.stream_analysis_window = None
        self.stream_analysis_listbox = None
        self.stream_analysis_preview_label = None
        self.stream_analysis_preview_photo = None
        self.stream_analysis_summary_text = None
        self.stream_analysis_json_text = None
        self.stream_analysis_progressbar = None
        self.stream_analysis_map_widget = None
        if self.stream_analysis_live_after_id is not None:
            try:
                self.root.after_cancel(self.stream_analysis_live_after_id)
            except Exception:
                pass
        self.stream_analysis_live_after_id = None

    def select_stream_analysis_folder(self) -> None:
        initial_dir = self.stream_capture_root_path()
        selected = filedialog.askdirectory(title="Select stream_capture task folder", initialdir=str(initial_dir))
        if selected:
            self.stream_analysis_stream_dir_var.set(str(Path(selected)))
            self.preview_stream_analysis_folder()

    def use_latest_stream_analysis_folder(self) -> None:
        latest_dir = self.find_latest_stream_capture_dir()
        if latest_dir is None:
            self.stream_analysis_status_var.set(f"Analysis: no stream folders under {self.stream_capture_root_path()}")
            return
        self.stream_analysis_stream_dir_var.set(str(latest_dir))
        self.preview_stream_analysis_folder()

    def select_stream_analysis_weights(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select YOLO weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.stream_analysis_weights_var.set(str(Path(selected)))

    def auto_stream_analysis_weights(self) -> None:
        try:
            weights = find_default_weights()
            self.stream_analysis_weights_var.set(str(weights))
            self.stream_analysis_status_var.set(f"Analysis: auto weights {weights}")
        except Exception as exc:
            self.stream_analysis_status_var.set(f"Analysis: auto weights failed: {exc}")

    def preview_stream_analysis_folder(self) -> None:
        stream_dir = Path(self.stream_analysis_stream_dir_var.get().strip())
        try:
            frames = scan_stream_frames(stream_dir, stride=1, max_frames=0)
            self.stream_analysis_status_var.set(f"Analysis: {len(frames)} frames found in {stream_dir}")
        except Exception as exc:
            self.stream_analysis_status_var.set(f"Analysis: folder scan failed: {exc}")

    def parse_stream_analysis_options(self) -> Tuple[float, int, str, int, int]:
        conf = max(0.01, min(1.0, float(self.stream_analysis_conf_var.get().strip() or "0.25")))
        imgsz = max(64, int(float(self.stream_analysis_imgsz_var.get().strip() or "640")))
        device = self.stream_analysis_device_var.get().strip() or "0"
        stride = max(1, int(float(self.stream_analysis_stride_var.get().strip() or "1")))
        max_frames = max(0, int(float(self.stream_analysis_max_frames_var.get().strip() or "0")))
        self.stream_analysis_conf_var.set(f"{conf:g}")
        self.stream_analysis_imgsz_var.set(str(imgsz))
        self.stream_analysis_stride_var.set(str(stride))
        self.stream_analysis_max_frames_var.set(str(max_frames))
        return conf, imgsz, device, stride, max_frames

    def start_stream_analysis(self) -> None:
        self.ensure_stream_analysis_state()
        if self.stream_analysis_thread is not None and self.stream_analysis_thread.is_alive():
            self.stream_analysis_status_var.set("Analysis: already running")
            return
        stream_text = self.stream_analysis_stream_dir_var.get().strip()
        if not stream_text:
            self.use_latest_stream_analysis_folder()
            stream_text = self.stream_analysis_stream_dir_var.get().strip()
        if not stream_text:
            return
        stream_dir = Path(stream_text)
        weights_text = self.stream_analysis_weights_var.get().strip()
        try:
            conf, imgsz, device, stride, max_frames = self.parse_stream_analysis_options()
        except Exception as exc:
            self.stream_analysis_status_var.set(f"Analysis: invalid options: {exc}")
            return

        self.stream_analysis_stop_event.clear()
        self.stream_analysis_rows = []
        self.stream_analysis_result = {}
        self.stream_analysis_progress_var.set(0.0)
        self.populate_stream_analysis_results([])
        self.stream_analysis_status_var.set("Analysis: starting...")

        houses_config = self.map_config if isinstance(getattr(self, "map_config", None), dict) else {}
        houses_config_path = self.map_config_path or self.resolve_project_path(getattr(self.args, "map_config", ""))

        def progress(payload: Dict[str, Any]) -> None:
            self.root.after(0, lambda p=payload: self.apply_stream_analysis_progress(p))

        def worker() -> None:
            try:
                result = run_stream_analysis(
                    stream_dir=stream_dir,
                    weights_path=weights_text or None,
                    houses_config_path=houses_config_path,
                    houses_config=houses_config,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    stride=stride,
                    max_frames=max_frames,
                    stop_event=self.stream_analysis_stop_event,
                    progress_callback=progress,
                )
                self.root.after(0, lambda r=result: self.apply_stream_analysis_result(r))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self.apply_stream_analysis_error(e))

        self.stream_analysis_thread = threading.Thread(target=worker, daemon=True)
        self.stream_analysis_thread.start()

    def stop_stream_analysis(self) -> None:
        self.ensure_stream_analysis_state()
        self.stream_analysis_stop_event.set()
        if self.stream_analysis_thread is not None and self.stream_analysis_thread.is_alive():
            self.stream_analysis_status_var.set("Analysis: stopping after current frame...")

    def apply_stream_analysis_error(self, exc: BaseException) -> None:
        self.stream_analysis_status_var.set(f"Analysis failed: {exc}")
        self.status_var.set(f"Stream analysis failed: {exc}")

    def apply_stream_analysis_progress(self, payload: Dict[str, Any]) -> None:
        total = max(1, int(payload.get("total") or 1))
        processed = max(0, int(payload.get("processed") or 0))
        self.stream_analysis_progress_var.set(min(100.0, processed * 100.0 / total))
        message = str(payload.get("message") or "")
        if message:
            self.stream_analysis_status_var.set(f"Analysis: {message}")
        row = payload.get("row")
        if isinstance(row, dict):
            self.stream_analysis_rows.append(row)
            self.populate_stream_analysis_results(self.stream_analysis_rows)
            if self.stream_analysis_listbox is not None:
                self.stream_analysis_listbox.selection_clear(0, "end")
                self.stream_analysis_listbox.selection_set("end")
                self.stream_analysis_listbox.see("end")
                self.show_stream_analysis_row(row)

    def apply_stream_analysis_result(self, result: Dict[str, Any]) -> None:
        self.stream_analysis_result = result
        rows = result.get("frames", []) if isinstance(result, dict) else []
        if isinstance(rows, list):
            self.stream_analysis_rows = [row for row in rows if isinstance(row, dict)]
            self.populate_stream_analysis_results(self.stream_analysis_rows)
        total = max(1, int(result.get("total_selected_frames") or len(self.stream_analysis_rows) or 1))
        processed = int(result.get("processed_frames") or len(self.stream_analysis_rows))
        self.stream_analysis_progress_var.set(min(100.0, processed * 100.0 / total))
        self.stream_analysis_status_var.set(
            f"Analysis: {result.get('status', 'ok')} {processed}/{total} -> {result.get('analysis_dir', '')}"
        )
        self.status_var.set(f"Stream analysis saved: {result.get('analysis_dir', '')}")
        if self.stream_analysis_rows and self.stream_analysis_listbox is not None:
            self.stream_analysis_listbox.selection_clear(0, "end")
            self.stream_analysis_listbox.selection_set(0)
            self.show_stream_analysis_row(self.stream_analysis_rows[0])

    def populate_stream_analysis_results(self, rows: List[Dict[str, Any]]) -> None:
        if self.stream_analysis_listbox is None:
            return
        self.stream_analysis_listbox.delete(0, "end")
        for row in rows:
            status = str(row.get("status") or "?")
            dets = int(row.get("num_detections") or 0)
            entry_class = str(row.get("entry_class") or row.get("top_class") or "none")
            dist = row.get("entry_distance_cm")
            dist_text = "n/a" if dist is None else f"{float(dist):.0f}cm"
            trav = "T" if row.get("traversable") else "-"
            self.stream_analysis_listbox.insert(
                "end",
                f"{int(row.get('frame_index') or 0):06d} {status} det={dets} {entry_class} {dist_text} {trav}",
            )

    def on_stream_analysis_frame_selected(self, _event=None) -> None:
        if self.stream_analysis_listbox is None:
            return
        selection = self.stream_analysis_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self.stream_analysis_rows):
            self.show_stream_analysis_row(self.stream_analysis_rows[index])

    def show_stream_analysis_row(self, row: Dict[str, Any]) -> None:
        self.stream_analysis_current_row = row
        self.show_stream_analysis_preview(row)
        if self.stream_analysis_live_pose_var.get():
            self.update_stream_analysis_live_map()
        else:
            self.update_stream_analysis_map(row)
        self.show_stream_analysis_summary(row)
        self.show_stream_analysis_json(row)

    def parse_stream_analysis_shift_step(self) -> float:
        try:
            step = float(self.stream_analysis_map_shift_step_var.get().strip() or "5")
        except Exception:
            step = 5.0
        step = max(0.1, min(1000.0, step))
        self.stream_analysis_map_shift_step_var.set(f"{step:g}")
        return step

    def shift_stream_analysis_map(self, direction_x: float, direction_y: float) -> None:
        step = self.parse_stream_analysis_shift_step()
        dx, dy = self.map_display_offset_px
        self.map_display_offset_px = (float(dx) + float(direction_x) * step, float(dy) + float(direction_y) * step)
        self.refresh_stream_analysis_map_after_shift()

    def reset_stream_analysis_map_shift(self) -> None:
        self.map_display_offset_px = (0.0, 0.0)
        self.refresh_stream_analysis_map_after_shift()

    def refresh_stream_analysis_map_after_shift(self) -> None:
        if self.stream_analysis_live_pose_var.get():
            self.update_stream_analysis_live_map()
        elif isinstance(self.stream_analysis_current_row, dict):
            self.update_stream_analysis_map(self.stream_analysis_current_row)
        else:
            self.stream_analysis_map_pose_var.set(
                f"Map pose: shift=({self.map_display_offset_px[0]:.1f}, {self.map_display_offset_px[1]:.1f})"
            )
        if self.map_widget is not None and self.map_window is not None and self.map_window.winfo_exists():
            self.refresh_map_once()

    def on_toggle_stream_analysis_live_pose(self) -> None:
        if self.stream_analysis_live_pose_var.get():
            self.schedule_stream_analysis_live_pose(immediate=True)
        else:
            if self.stream_analysis_live_after_id is not None:
                try:
                    self.root.after_cancel(self.stream_analysis_live_after_id)
                except Exception:
                    pass
                self.stream_analysis_live_after_id = None
            if isinstance(self.stream_analysis_current_row, dict):
                self.show_stream_analysis_row(self.stream_analysis_current_row)

    def schedule_stream_analysis_live_pose(self, *, immediate: bool = False) -> None:
        if self.stream_analysis_live_after_id is not None:
            try:
                self.root.after_cancel(self.stream_analysis_live_after_id)
            except Exception:
                pass
            self.stream_analysis_live_after_id = None
        if not self.stream_analysis_live_pose_var.get():
            return
        delay_ms = 1 if immediate else max(100, int(getattr(self.args, "map_interval_ms", 1000)))
        self.stream_analysis_live_after_id = self.root.after(delay_ms, self.update_stream_analysis_live_map)

    def update_stream_analysis_live_map(self) -> None:
        self.stream_analysis_live_after_id = None
        if self.stream_analysis_map_widget is None or not self.stream_analysis_live_pose_var.get():
            return
        pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        if not pose:
            self.stream_analysis_map_pose_var.set(
                f"Live pose: waiting for session state shift=({self.map_display_offset_px[0]:.1f}, {self.map_display_offset_px[1]:.1f})"
            )
            self.schedule_stream_analysis_live_pose()
            return
        live_row = {
            "frame_name": "live",
            "status": "live",
            "pose": pose,
            "capture_time": datetime.now().isoformat(timespec="seconds"),
        }
        self.update_stream_analysis_map(live_row, live=True)
        self.schedule_stream_analysis_live_pose()

    def save_stream_analysis_shifted_config(self) -> None:
        if not self.map_config:
            self.load_map_resources(force=True)
        if not self.map_config:
            self.stream_analysis_status_var.set("Analysis: map config is not loaded")
            return
        try:
            shifted = json.loads(json.dumps(self.map_config))
            overhead = shifted.setdefault("overhead_map", {})
            overhead["display_offset_px"] = {
                "x": float(self.map_display_offset_px[0]),
                "y": float(self.map_display_offset_px[1]),
            }
            base_dir = PROJECT_ROOT / "assets" / "overhead_map"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = (base_dir / DEFAULT_MANUAL_SHIFT_MAP_CONFIG_NAME).resolve()
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(shifted, fh, indent=2, ensure_ascii=False)
            self.args.map_config = str(output_path)
            self.map_config = {}
            self.map_config_path = None
            self.map_image_path = None
            self.map_image = None
            self.load_map_resources(force=True)
            self.stream_analysis_status_var.set(f"Analysis: saved shifted map config -> {output_path}")
            self.status_var.set(f"Saved shifted map config: {output_path.name}")
            self.refresh_stream_analysis_map_after_shift()
            self.refresh_map_once(force_reload=True)
        except Exception as exc:
            self.stream_analysis_status_var.set(f"Analysis: save shifted config failed: {exc}")

    def stream_analysis_row_key(self, row: Dict[str, Any]) -> str:
        return str(row.get("capture_dir") or row.get("frame_name") or row.get("frame_index") or "")

    def stream_analysis_pose_for_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        pose = row.get("pose") if isinstance(row.get("pose"), dict) else {}
        if pose:
            return pose
        key = self.stream_analysis_row_key(row)
        if key and key in self.stream_analysis_pose_cache:
            return self.stream_analysis_pose_cache[key]
        capture_dir = Path(str(row.get("capture_dir") or ""))
        pose_path = capture_dir / "pose.json"
        parsed_pose: Dict[str, Any] = {}
        try:
            if pose_path.exists():
                payload = json.loads(pose_path.read_text(encoding="utf-8"))
                raw_pose = payload.get("pose") if isinstance(payload, dict) else {}
                if isinstance(raw_pose, dict):
                    parsed_pose = raw_pose
        except Exception:
            parsed_pose = {}
        if key:
            self.stream_analysis_pose_cache[key] = parsed_pose
        return parsed_pose

    def stream_analysis_selected_index(self, row: Dict[str, Any]) -> int:
        if str(row.get("status", "")).lower() == "live":
            return len(self.stream_analysis_rows) - 1
        row_key = self.stream_analysis_row_key(row)
        for index, candidate in enumerate(self.stream_analysis_rows):
            if candidate is row:
                return index
            if row_key and self.stream_analysis_row_key(candidate) == row_key:
                return index
        return max(0, len(self.stream_analysis_rows) - 1)

    def stream_analysis_trajectory_until(self, row: Dict[str, Any], *, live: bool = False) -> List[Dict[str, float]]:
        if live:
            session = self.session
            if session is not None and session.started:
                try:
                    return session.get_trajectory_points(limit=max(1, int(getattr(self.args, "map_trajectory_limit", 500))))
                except Exception:
                    pass
            pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
            try:
                return [{"x": float(pose["x"]), "y": float(pose["y"])}]
            except Exception:
                return []
        selected_index = self.stream_analysis_selected_index(row)
        points: List[Dict[str, float]] = []
        for candidate in self.stream_analysis_rows[: selected_index + 1]:
            pose = self.stream_analysis_pose_for_row(candidate)
            try:
                points.append({"x": float(pose["x"]), "y": float(pose["y"])})
            except Exception:
                continue
        return points

    def update_stream_analysis_map(self, row: Dict[str, Any], *, live: bool = False) -> None:
        if self.stream_analysis_map_widget is None:
            return
        if not self.map_config or self.map_image is None:
            self.load_map_resources(force=True)
        pose = self.stream_analysis_pose_for_row(row)
        if not pose:
            self.stream_analysis_map_pose_var.set(f"Map pose: {row.get('frame_name', '')} no pose.json")
            return
        try:
            pose_x = float(pose.get("x", 0.0))
            pose_y = float(pose.get("y", 0.0))
            pose_z = float(pose.get("z", 0.0))
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0)))
        except Exception:
            self.stream_analysis_map_pose_var.set(f"Map pose: {row.get('frame_name', '')} invalid pose")
            return
        trajectory = self.stream_analysis_trajectory_until(row, live=live)
        image_xy = self.world_to_image_point(pose_x, pose_y)
        if image_xy is None:
            image_text = "image=(n/a)"
        else:
            image_text = f"image=({image_xy[0]:.1f}, {image_xy[1]:.1f})"
        self.stream_analysis_map_pose_var.set(
            f"{row.get('frame_name', '')} world=({pose_x:.1f}, {pose_y:.1f}, {pose_z:.1f}) "
            f"yaw={pose_yaw:.1f} {image_text} "
            f"shift=({self.map_display_offset_px[0]:.1f}, {self.map_display_offset_px[1]:.1f}) "
            f"traj={len(trajectory)}"
        )
        houses, boxes = self.build_map_display(pose)
        calibration = self.map_calibration if isinstance(self.map_calibration, dict) else {}
        anchors = calibration.get("anchors", []) if isinstance(calibration.get("anchors", []), list) else []
        if not self.show_calibration_points_var.get():
            anchors = []
        self.stream_analysis_map_widget.set_background_image(self.map_image)
        self.stream_analysis_map_widget.set_calibration(
            calibration.get("affine_world_to_image"),
            self.map_image_size(),
            anchors,
        )
        self.stream_analysis_map_widget.set_image_layer_offset(*self.map_display_offset_px)
        self.stream_analysis_map_widget.set_house_boxes(boxes)
        self.stream_analysis_map_widget.update_houses([])
        self.stream_analysis_map_widget.set_trajectory(trajectory)
        self.stream_analysis_map_widget.update_uav(pose_x, pose_y, pose_yaw)

    def show_stream_analysis_preview(self, row: Dict[str, Any]) -> None:
        if self.stream_analysis_preview_label is None:
            return
        candidates = [
            row.get("fusion_overlay_path"),
            row.get("yolo_annotated_path"),
            row.get("depth_overlay_path"),
            row.get("rgb_path"),
        ]
        image_path = None
        for raw in candidates:
            if raw and Path(str(raw)).exists():
                image_path = Path(str(raw))
                break
        if image_path is None:
            self.stream_analysis_preview_label.configure(image="", text="No preview image for this frame.")
            self.stream_analysis_preview_photo = None
            return
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((700, 460), Image.Resampling.LANCZOS)
            self.stream_analysis_preview_photo = ImageTk.PhotoImage(image)
            self.stream_analysis_preview_label.configure(image=self.stream_analysis_preview_photo, text="")
        except Exception as exc:
            self.stream_analysis_preview_label.configure(image="", text=f"Preview failed: {exc}")
            self.stream_analysis_preview_photo = None

    def show_stream_analysis_summary(self, row: Dict[str, Any]) -> None:
        if self.stream_analysis_summary_text is None:
            return
        pose = self.stream_analysis_pose_for_row(row)
        if pose:
            try:
                pose_line = (
                    f"Pose: x={float(pose.get('x', 0.0)):.1f} "
                    f"y={float(pose.get('y', 0.0)):.1f} "
                    f"z={float(pose.get('z', 0.0)):.1f} "
                    f"yaw={float(pose.get('task_yaw', pose.get('yaw', 0.0))):.1f}"
                )
            except Exception:
                pose_line = f"Pose: {pose}"
        else:
            pose_line = "Pose: n/a"
        lines = [
            f"Frame: {row.get('frame_name')} ({row.get('status')})",
            pose_line,
            f"YOLO: detections={row.get('num_detections')} top={row.get('top_class')} conf={row.get('top_confidence')}",
            (
                "Fusion: "
                f"class={row.get('entry_class')} "
                f"conf={row.get('entry_semantic_confidence')} "
                f"distance={row.get('entry_distance_cm')}cm "
                f"width={row.get('opening_width_cm')}cm "
                f"traversable={row.get('traversable')} "
                f"crossing_ready={row.get('crossing_ready')}"
            ),
            f"Action: {row.get('recommended_action_hint') or row.get('target_conditioned_action_hint')}",
            f"State: {row.get('target_conditioned_state') or row.get('recommended_subgoal')}",
            f"Reason: {row.get('decision_reason') or row.get('error') or ''}",
        ]
        self.stream_analysis_summary_text.delete("1.0", "end")
        self.stream_analysis_summary_text.insert("1.0", "\n".join(lines))

    def show_stream_analysis_json(self, row: Dict[str, Any]) -> None:
        if self.stream_analysis_json_text is None:
            return
        payload: Dict[str, Any] = {"summary": row}
        fusion_path = row.get("fusion_result_path")
        if fusion_path and Path(str(fusion_path)).exists():
            try:
                payload["fusion_result"] = json.loads(Path(str(fusion_path)).read_text(encoding="utf-8"))
            except Exception as exc:
                payload["fusion_result_error"] = str(exc)
        self.stream_analysis_json_text.delete("1.0", "end")
        self.stream_analysis_json_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))

    def find_latest_analysis_dir(self, stream_dir: Path) -> Optional[Path]:
        analysis_root = stream_dir / "analysis"
        if not analysis_root.exists():
            return None
        candidates = [path for path in analysis_root.iterdir() if path.is_dir()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if (candidate / "analysis_index.json").exists():
                return candidate
        return None

    def load_latest_stream_analysis_results(self) -> None:
        stream_text = self.stream_analysis_stream_dir_var.get().strip()
        if not stream_text:
            self.use_latest_stream_analysis_folder()
            stream_text = self.stream_analysis_stream_dir_var.get().strip()
        if not stream_text:
            return
        stream_dir = Path(stream_text)
        analysis_dir = self.find_latest_analysis_dir(stream_dir)
        if analysis_dir is None:
            self.stream_analysis_status_var.set(f"Analysis: no analysis results under {stream_dir}")
            return
        index_path = analysis_dir / "analysis_index.json"
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            rows = payload.get("frames", [])
            self.stream_analysis_rows = [row for row in rows if isinstance(row, dict)]
            self.stream_analysis_result = payload
            self.populate_stream_analysis_results(self.stream_analysis_rows)
            self.stream_analysis_progress_var.set(100.0 if self.stream_analysis_rows else 0.0)
            self.stream_analysis_status_var.set(f"Analysis: loaded {len(self.stream_analysis_rows)} frames from {analysis_dir}")
            if self.stream_analysis_rows and self.stream_analysis_listbox is not None:
                self.stream_analysis_listbox.selection_clear(0, "end")
                self.stream_analysis_listbox.selection_set(0)
                self.show_stream_analysis_row(self.stream_analysis_rows[0])
        except Exception as exc:
            self.stream_analysis_status_var.set(f"Analysis: failed to load results: {exc}")
