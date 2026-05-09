from __future__ import annotations

from .common import *
from . import lidar_yolo_analysis

import concurrent.futures


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class LidarAnalysisControlMixin:
    def ensure_lidar_analysis_state(self) -> None:
        if not hasattr(self, "lidar_analysis_window"):
            self.lidar_analysis_window = None
        if not hasattr(self, "lidar_analysis_stream_dir_var"):
            self.lidar_analysis_stream_dir_var = tk.StringVar(value="")
        if not hasattr(self, "lidar_analysis_mode_var"):
            self.lidar_analysis_mode_var = tk.StringVar(value="Cumulative")
        if not hasattr(self, "lidar_analysis_max_points_var"):
            self.lidar_analysis_max_points_var = tk.StringVar(value="60000")
        if not hasattr(self, "lidar_analysis_color_mode_var"):
            self.lidar_analysis_color_mode_var = tk.StringVar(value="RGB")
        if not hasattr(self, "lidar_analysis_point_size_var"):
            self.lidar_analysis_point_size_var = tk.StringVar(value="1.0")
        if not hasattr(self, "lidar_analysis_view_preset_var"):
            self.lidar_analysis_view_preset_var = tk.StringVar(value="Perspective")
        if not hasattr(self, "lidar_analysis_rows"):
            self.lidar_analysis_rows = []
        if not hasattr(self, "lidar_analysis_listbox"):
            self.lidar_analysis_listbox = None
        if not hasattr(self, "lidar_analysis_summary_text"):
            self.lidar_analysis_summary_text = None
        if not hasattr(self, "lidar_analysis_json_text"):
            self.lidar_analysis_json_text = None
        if not hasattr(self, "lidar_analysis_fig"):
            self.lidar_analysis_fig = None
        if not hasattr(self, "lidar_analysis_ax"):
            self.lidar_analysis_ax = None
        if not hasattr(self, "lidar_analysis_canvas"):
            self.lidar_analysis_canvas = None
        if not hasattr(self, "lidar_analysis_toolbar"):
            self.lidar_analysis_toolbar = None
        if not hasattr(self, "lidar_analysis_after_id"):
            self.lidar_analysis_after_id = None
        if not hasattr(self, "lidar_analysis_playing"):
            self.lidar_analysis_playing = False
        if not hasattr(self, "lidar_analysis_index"):
            self.lidar_analysis_index = 0
        if not hasattr(self, "lidar_analysis_cached_index"):
            self.lidar_analysis_cached_index = -1
        if not hasattr(self, "lidar_analysis_cached_cloud"):
            self.lidar_analysis_cached_cloud = None
        if not hasattr(self, "lidar_analysis_rebuild_thread"):
            self.lidar_analysis_rebuild_thread = None
        if not hasattr(self, "lidar_analysis_export_thread"):
            self.lidar_analysis_export_thread = None
        if not hasattr(self, "lidar_analysis_open3d_thread"):
            self.lidar_analysis_open3d_thread = None
        if not hasattr(self, "lidar_yolo_analysis_thread"):
            self.lidar_yolo_analysis_thread = None
        if not hasattr(self, "lidar_yolo_stop_event"):
            self.lidar_yolo_stop_event = threading.Event()
        if not hasattr(self, "lidar_yolo_weights_var"):
            self.lidar_yolo_weights_var = tk.StringVar(value=str(lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_WEIGHTS_PATH))
        if not hasattr(self, "lidar_yolo_device_var"):
            self.lidar_yolo_device_var = tk.StringVar(value=lidar_yolo_analysis.default_lidar_yolo_device())
        if not hasattr(self, "lidar_yolo_conf_var"):
            self.lidar_yolo_conf_var = tk.StringVar(value=f"{lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_CONF:g}")
        if not hasattr(self, "lidar_yolo_imgsz_var"):
            self.lidar_yolo_imgsz_var = tk.StringVar(value=str(lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_IMGSZ))
        if not hasattr(self, "lidar_yolo_stride_var"):
            self.lidar_yolo_stride_var = tk.StringVar(value="1")
        if not hasattr(self, "lidar_yolo_max_frames_var"):
            self.lidar_yolo_max_frames_var = tk.StringVar(value="0")
        if not hasattr(self, "lidar_yolo_dedupe_radius_var"):
            self.lidar_yolo_dedupe_radius_var = tk.StringVar(value=f"{lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_DEDUPE_RADIUS_M:g}")
        if not hasattr(self, "lidar_yolo_max_points_var"):
            self.lidar_yolo_max_points_var = tk.StringVar(value=str(lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_MAX_POINTS_PER_DETECTION))

    def open_lidar_analysis_window(self) -> None:
        self.ensure_lidar_analysis_state()
        if self.lidar_analysis_window is not None and self.lidar_analysis_window.winfo_exists():
            self.lidar_analysis_window.lift()
            return

        if not self.lidar_analysis_stream_dir_var.get().strip():
            default_dir = self.lidar_stream_capture_dir or self.find_latest_lidar_stream_capture_dir()
            if default_dir is not None:
                self.lidar_analysis_stream_dir_var.set(str(default_dir))

        self.lidar_analysis_window = tk.Toplevel(self.root)
        self.lidar_analysis_window.title("Lidar Stream Analysis")
        self.lidar_analysis_window.geometry("1320x860")
        self.lidar_analysis_window.protocol("WM_DELETE_WINDOW", self.close_lidar_analysis_window)
        self.lidar_analysis_window.grid_columnconfigure(0, weight=1)
        self.lidar_analysis_window.grid_rowconfigure(2, weight=1)

        top = tk.Frame(self.lidar_analysis_window)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.grid_columnconfigure(1, weight=1)
        tk.Label(top, text="Lidar Folder").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=3)
        tk.Entry(top, textvariable=self.lidar_analysis_stream_dir_var).grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        tk.Button(top, text="Browse", command=self.select_lidar_analysis_folder).grid(row=0, column=2, padx=4, pady=3)
        tk.Button(top, text="Latest", command=self.use_latest_lidar_analysis_folder).grid(row=0, column=3, padx=4, pady=3)
        tk.Button(top, text="Load Frames", command=self.load_lidar_analysis_frames).grid(row=0, column=4, padx=4, pady=3)

        controls = tk.Frame(self.lidar_analysis_window)
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        controls.grid_columnconfigure(19, weight=1)
        tk.Label(controls, text="Mode").grid(row=0, column=0, padx=(0, 3), pady=3)
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.lidar_analysis_mode_var,
            values=("Cumulative", "Current Frame"),
            state="readonly",
            width=15,
        )
        mode_combo.grid(row=0, column=1, padx=(0, 8), pady=3)
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.reload_lidar_analysis_mode())
        tk.Label(controls, text="Max Display Points").grid(row=0, column=2, padx=(0, 3), pady=3)
        tk.Entry(controls, textvariable=self.lidar_analysis_max_points_var, width=9).grid(row=0, column=3, padx=(0, 8), pady=3)
        tk.Label(controls, text="Color Mode").grid(row=0, column=4, padx=(0, 3), pady=3)
        color_combo = ttk.Combobox(
            controls,
            textvariable=self.lidar_analysis_color_mode_var,
            values=("RGB", "Height", "Depth"),
            state="readonly",
            width=9,
        )
        color_combo.grid(row=0, column=5, padx=(0, 8), pady=3)
        color_combo.bind("<<ComboboxSelected>>", lambda _event: self.reload_lidar_analysis_view())
        tk.Label(controls, text="Point Size").grid(row=0, column=6, padx=(0, 3), pady=3)
        tk.Entry(controls, textvariable=self.lidar_analysis_point_size_var, width=6).grid(row=0, column=7, padx=(0, 8), pady=3)
        tk.Label(controls, text="View").grid(row=0, column=8, padx=(0, 3), pady=3)
        view_combo = ttk.Combobox(
            controls,
            textvariable=self.lidar_analysis_view_preset_var,
            values=("Perspective", "Top", "Front", "Side"),
            state="readonly",
            width=12,
        )
        view_combo.grid(row=0, column=9, padx=(0, 8), pady=3)
        view_combo.bind("<<ComboboxSelected>>", lambda _event: self.reload_lidar_analysis_view())
        tk.Button(controls, text="Reset View", command=self.reset_lidar_analysis_view).grid(row=0, column=10, padx=4, pady=3)
        tk.Button(controls, text="Export RViz", command=self.export_lidar_analysis_rviz).grid(row=0, column=11, padx=4, pady=3)
        tk.Button(controls, text="Export Open3D", command=self.export_lidar_analysis_open3d).grid(row=0, column=12, padx=4, pady=3)
        tk.Button(controls, text="YOLO Labels", command=self.open_lidar_yolo_labels_dialog).grid(row=0, column=13, padx=4, pady=3)
        tk.Button(controls, text="Open3D Viewer", command=self.open_lidar_analysis_open3d_viewer).grid(row=0, column=14, padx=4, pady=3)
        tk.Button(controls, text="Rebuild", command=self.rebuild_lidar_analysis_window).grid(row=0, column=15, padx=4, pady=3)
        tk.Button(controls, text="Prev", command=self.show_lidar_analysis_prev).grid(row=0, column=16, padx=4, pady=3)
        tk.Button(controls, text="Play/Pause", command=self.toggle_lidar_analysis_playback).grid(row=0, column=17, padx=4, pady=3)
        tk.Button(controls, text="Stop", command=self.stop_lidar_analysis_playback).grid(row=0, column=18, padx=4, pady=3)
        tk.Button(controls, text="Next", command=self.show_lidar_analysis_next).grid(row=0, column=19, padx=4, pady=3)

        body = tk.PanedWindow(self.lidar_analysis_window, orient="horizontal", sashrelief="raised")
        body.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = tk.Frame(body)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.lidar_analysis_listbox = tk.Listbox(left, width=52, exportselection=False)
        list_scroll = tk.Scrollbar(left, orient="vertical", command=self.lidar_analysis_listbox.yview)
        self.lidar_analysis_listbox.configure(yscrollcommand=list_scroll.set)
        self.lidar_analysis_listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.lidar_analysis_listbox.bind("<<ListboxSelect>>", self.on_lidar_analysis_frame_selected)
        body.add(left, minsize=380)

        right = tk.Frame(body)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=0)
        right.grid_rowconfigure(2, weight=1)

        plot_frame = tk.Frame(right)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self.build_lidar_analysis_plot(plot_frame)

        self.lidar_analysis_summary_text = tk.Text(right, height=7, wrap="word")
        self.lidar_analysis_summary_text.grid(row=1, column=0, sticky="ew", padx=(8, 0), pady=(0, 8))
        json_frame = tk.Frame(right)
        json_frame.grid(row=2, column=0, sticky="nsew", padx=(8, 0))
        json_frame.grid_rowconfigure(0, weight=1)
        json_frame.grid_columnconfigure(0, weight=1)
        self.lidar_analysis_json_text = tk.Text(json_frame, height=12, wrap="none")
        json_scroll = tk.Scrollbar(json_frame, orient="vertical", command=self.lidar_analysis_json_text.yview)
        self.lidar_analysis_json_text.configure(yscrollcommand=json_scroll.set)
        self.lidar_analysis_json_text.grid(row=0, column=0, sticky="nsew")
        json_scroll.grid(row=0, column=1, sticky="ns")
        body.add(right, minsize=820)

        tk.Label(self.lidar_analysis_window, textvariable=self.lidar_stream_analysis_status_var, anchor="w").grid(
            row=3,
            column=0,
            sticky="ew",
            padx=8,
            pady=(0, 8),
        )

        if self.lidar_analysis_stream_dir_var.get().strip():
            self.load_lidar_analysis_frames()

    def close_lidar_analysis_window(self) -> None:
        self.stop_lidar_analysis_playback()
        if getattr(self, "lidar_yolo_stop_event", None) is not None:
            self.lidar_yolo_stop_event.set()
        if getattr(self, "lidar_analysis_window", None) is not None and self.lidar_analysis_window.winfo_exists():
            self.lidar_analysis_window.destroy()
        self.lidar_analysis_window = None
        self.lidar_analysis_listbox = None
        self.lidar_analysis_summary_text = None
        self.lidar_analysis_json_text = None
        self.lidar_analysis_fig = None
        self.lidar_analysis_ax = None
        self.lidar_analysis_canvas = None
        self.lidar_analysis_toolbar = None

    def build_lidar_analysis_plot(self, parent: tk.Widget) -> None:
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as exc:
            tk.Label(parent, text=f"Matplotlib 3D view unavailable: {exc}", anchor="center").grid(row=0, column=0, sticky="nsew")
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: matplotlib unavailable: {exc}")
            return

        fig = Figure(figsize=(8.4, 5.8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left", fill="x", expand=True)
        self.lidar_analysis_fig = fig
        self.lidar_analysis_ax = ax
        self.lidar_analysis_canvas = canvas
        self.lidar_analysis_toolbar = toolbar
        self.plot_lidar_analysis_cloud(np.zeros((0, 6), dtype=np.float32), title="Load a stream_capture_lidar folder.")

    def select_lidar_analysis_folder(self) -> None:
        initial_dir = self.lidar_stream_capture_root_path()
        selected = filedialog.askdirectory(title="Select stream_capture_lidar task folder", initialdir=str(initial_dir))
        if selected:
            self.lidar_analysis_stream_dir_var.set(str(Path(selected)))
            self.load_lidar_analysis_frames()

    def use_latest_lidar_analysis_folder(self) -> None:
        latest_dir = self.find_latest_lidar_stream_capture_dir()
        if latest_dir is None:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: no lidar stream folders under {self.lidar_stream_capture_root_path()}")
            return
        self.lidar_analysis_stream_dir_var.set(str(latest_dir))
        self.load_lidar_analysis_frames()

    def resolve_lidar_analysis_path(self, stream_dir: Path, raw_path: Any) -> Path:
        raw = str(raw_path or "").strip()
        if not raw:
            return Path()
        path = Path(raw)
        if path.is_absolute():
            return path
        return (stream_dir / path).resolve()

    def read_lidar_analysis_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def lidar_analysis_frame_from_dir(
        self,
        stream_dir: Path,
        capture_dir: Path,
        *,
        trajectory_entry: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        entry = trajectory_entry if isinstance(trajectory_entry, dict) else {}
        frame_name = capture_dir.name
        frame_index = int(entry.get("frame_index") or self.parse_lidar_analysis_frame_index(frame_name) or 0)
        capture_path = capture_dir / "capture.json"
        pose_path = capture_dir / "pose.json"
        capture_payload = self.read_lidar_analysis_json(capture_path)
        pose_payload = self.read_lidar_analysis_json(pose_path)
        combined_payload = dict(capture_payload)
        combined_payload.update(entry)
        pending_status = str(
            combined_payload.get("postprocess_status", "")
            or ("pending" if combined_payload.get("raw_capture_only") else "")
        ).lower()
        existing_standard_path = (
            entry.get("point_cloud_world_standard_m_npy_path")
            or capture_payload.get("point_cloud_world_standard_m_npy_path")
            or str(capture_dir / "point_cloud_world_standard_m.npy")
        )
        existing_standard = self.resolve_lidar_analysis_path(stream_dir, existing_standard_path)
        if pending_status in {"pending", "running"} and not existing_standard.exists():
            ensured = {
                "point_cloud_world_standard_m_npy_path": "",
                "depth_projection_selected": combined_payload.get("depth_projection_selected", combined_payload.get("depth_projection", "plane_depth")),
                "projection_corrected": bool(combined_payload.get("projection_corrected", False)),
                "coordinate_frame": "standard_zup",
                "coordinate_units": "m",
                "postprocess_status": pending_status,
            }
        else:
            ensured = flight.ensure_standard_world_cloud_for_capture(
                capture_dir,
                capture_payload=combined_payload,
                lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
                min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
                max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            )
        diagnostics_path = Path(str(ensured.get("projection_diagnostics_path", capture_dir / "projection_diagnostics.json")))
        diagnostics_payload = self.read_lidar_analysis_json(diagnostics_path)
        raw_cloud = (
            entry.get("point_cloud_world_standard_m_npy_path")
            or capture_payload.get("point_cloud_world_standard_m_npy_path")
            or ensured.get("point_cloud_world_standard_m_npy_path")
        )
        cloud_path = self.resolve_lidar_analysis_path(stream_dir, raw_cloud) if raw_cloud else capture_dir / "point_cloud_world_standard_m.npy"
        if not cloud_path.exists() and pending_status not in {"pending", "running"}:
            return None
        point_count = entry.get("point_count", capture_payload.get("point_count", 0))
        if not point_count:
            try:
                point_count = int(np.load(cloud_path, mmap_mode="r").shape[0])
            except Exception:
                point_count = 0
        return {
            "frame_index": frame_index,
            "frame_name": frame_name,
            "capture_time": entry.get("capture_time") or capture_payload.get("capture_time", ""),
            "capture_dir": str(capture_dir),
            "point_cloud_world_npy_path": str(cloud_path),
            "point_cloud_world_standard_m_npy_path": str(cloud_path),
            "rgb_path": entry.get("rgb_path") or capture_payload.get("rgb_path", str(capture_dir / "rgb.png")),
            "point_cloud_preview_path": entry.get("point_cloud_preview_path")
            or capture_payload.get("point_cloud_preview_path", str(capture_dir / "point_cloud_preview.png")),
            "pose_json_path": str(pose_path),
            "capture_json_path": str(capture_path),
            "projection_diagnostics_path": str(diagnostics_path),
            "point_count": int(point_count or 0),
            "invalid_depth_count": int(entry.get("invalid_depth_count", capture_payload.get("invalid_depth_count", 0)) or 0),
            "depth_projection_selected": (
                ensured.get("depth_projection_selected")
                or diagnostics_payload.get("depth_projection_selected")
                or entry.get("depth_projection_selected")
                or capture_payload.get("depth_projection_selected")
                or "plane_depth"
            ),
            "projection_corrected": bool(
                ensured.get(
                    "projection_corrected",
                    diagnostics_payload.get(
                        "projection_corrected",
                        entry.get("projection_corrected", capture_payload.get("projection_corrected", True)),
                    ),
                )
            ),
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "raw_capture_only": bool(combined_payload.get("raw_capture_only", False)),
            "postprocess_status": ensured.get("postprocess_status", combined_payload.get("postprocess_status", "")),
            "postprocess_error": combined_payload.get("postprocess_error", ""),
            "legacy_source_path": ensured.get("legacy_source_path", diagnostics_payload.get("legacy_source_path", "")),
            "trajectory_entry": dict(entry),
            "capture_payload": capture_payload,
            "projection_diagnostics": diagnostics_payload,
            "pose_payload": pose_payload,
        }

    def parse_lidar_analysis_frame_index(self, name: str) -> int:
        match = re.search(r"(\d+)$", str(name or ""))
        if not match:
            return 0
        try:
            return int(match.group(1))
        except Exception:
            return 0

    def scan_lidar_analysis_frames(self, stream_dir: str | Path) -> List[Dict[str, Any]]:
        stream_path = Path(stream_dir).resolve()
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        trajectory = self.read_lidar_analysis_json(stream_path / "trajectory.json")
        entries = trajectory.get("trajectory", []) if isinstance(trajectory, dict) else []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                raw_capture_dir = entry.get("capture_dir")
                if raw_capture_dir:
                    capture_dir = self.resolve_lidar_analysis_path(stream_path, raw_capture_dir)
                elif entry.get("point_cloud_world_npy_path"):
                    capture_dir = self.resolve_lidar_analysis_path(stream_path, entry.get("point_cloud_world_npy_path")).parent
                else:
                    continue
                key = str(capture_dir.resolve())
                if key in seen:
                    continue
                row = self.lidar_analysis_frame_from_dir(stream_path, capture_dir, trajectory_entry=entry)
                if row is not None:
                    rows.append(row)
                    seen.add(key)

        frames_root = stream_path / "frames"
        if frames_root.exists():
            for capture_dir in sorted(path for path in frames_root.glob("frame_*") if path.is_dir()):
                key = str(capture_dir.resolve())
                if key in seen:
                    continue
                row = self.lidar_analysis_frame_from_dir(stream_path, capture_dir)
                if row is not None:
                    rows.append(row)
                    seen.add(key)

        rows.sort(key=lambda item: (int(item.get("frame_index", 0)), str(item.get("frame_name", ""))))
        cumulative_points = 0
        for row in rows:
            cumulative_points += int(row.get("point_count", 0) or 0)
            row["cumulative_point_count"] = cumulative_points
        return rows

    def load_lidar_analysis_frames(self) -> None:
        self.ensure_lidar_analysis_state()
        stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            self.use_latest_lidar_analysis_folder()
            stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            return
        stream_dir = Path(stream_text)
        try:
            rows = self.scan_lidar_analysis_frames(stream_dir)
        except Exception as exc:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: folder scan failed: {exc}")
            return
        self.lidar_analysis_rows = rows
        self.lidar_analysis_index = 0
        self.reset_lidar_analysis_cache()
        self.populate_lidar_analysis_frames(rows)
        total_points = int(rows[-1].get("cumulative_point_count", 0) or 0) if rows else 0
        self.lidar_stream_analysis_status_var.set(
            f"Lidar Analysis: {len(rows)} frames, standard cumulative points={total_points}, units=m -> {stream_dir}"
        )
        if rows and self.lidar_analysis_listbox is not None:
            self.lidar_analysis_listbox.selection_clear(0, "end")
            self.lidar_analysis_listbox.selection_set(0)
            self.lidar_analysis_listbox.see(0)
            self.show_lidar_analysis_row(0)
        elif not rows:
            self.plot_lidar_analysis_cloud(np.zeros((0, 6), dtype=np.float32), title="No point cloud frames found.")

    def populate_lidar_analysis_frames(self, rows: List[Dict[str, Any]]) -> None:
        if self.lidar_analysis_listbox is None:
            return
        self.lidar_analysis_listbox.delete(0, "end")
        for row in rows:
            self.lidar_analysis_listbox.insert(
                "end",
                (
                    f"frame={int(row.get('frame_index', 0)):06d} "
                    f"points={int(row.get('point_count', 0)):7d} "
                    f"cum={int(row.get('cumulative_point_count', 0)):9d} "
                    f"proj={row.get('depth_projection_selected', '')} "
                    f"units={row.get('coordinate_units', 'm')} "
                    f"post={row.get('postprocess_status', '')} "
                    f"{row.get('capture_time', '')}"
                ),
            )

    def on_lidar_analysis_frame_selected(self, _event=None) -> None:
        if self.lidar_analysis_listbox is None:
            return
        selection = self.lidar_analysis_listbox.curselection()
        if not selection:
            return
        self.show_lidar_analysis_row(int(selection[0]))

    def reload_lidar_analysis_mode(self) -> None:
        self.reset_lidar_analysis_cache()
        if self.lidar_analysis_rows:
            self.show_lidar_analysis_row(self.lidar_analysis_index)

    def reload_lidar_analysis_view(self) -> None:
        if self.lidar_analysis_rows:
            self.show_lidar_analysis_row(self.lidar_analysis_index)

    def reset_lidar_analysis_view(self) -> None:
        self.lidar_analysis_view_preset_var.set("Perspective")
        if self.lidar_analysis_rows:
            self.show_lidar_analysis_row(self.lidar_analysis_index)

    def reset_lidar_analysis_cache(self) -> None:
        self.lidar_analysis_cached_index = -1
        self.lidar_analysis_cached_cloud = None

    def parse_lidar_analysis_max_points(self) -> int:
        try:
            value = int(float(self.lidar_analysis_max_points_var.get().strip() or "60000"))
        except Exception:
            value = 60000
        value = max(1000, min(500000, value))
        self.lidar_analysis_max_points_var.set(str(value))
        return value

    def parse_lidar_analysis_point_size(self) -> float:
        try:
            value = float(self.lidar_analysis_point_size_var.get().strip() or "1.0")
        except Exception:
            value = 1.0
        value = max(0.1, min(20.0, value))
        self.lidar_analysis_point_size_var.set(f"{value:g}")
        return value

    def load_lidar_analysis_cloud(self, row: Dict[str, Any]) -> np.ndarray:
        cloud_path = Path(str(row.get("point_cloud_world_npy_path", "")))
        if not cloud_path.exists():
            return np.zeros((0, 6), dtype=np.float32)
        cloud = np.load(cloud_path).astype(np.float32, copy=False)
        if cloud.ndim != 2 or cloud.shape[1] != 6:
            return np.zeros((0, 6), dtype=np.float32)
        return cloud

    def build_lidar_analysis_cloud_for_index(self, index: int) -> np.ndarray:
        rows = self.lidar_analysis_rows
        if not rows:
            return np.zeros((0, 6), dtype=np.float32)
        index = max(0, min(index, len(rows) - 1))
        mode = str(self.lidar_analysis_mode_var.get() or "Cumulative").strip().lower()
        if mode == "current frame":
            return self.load_lidar_analysis_cloud(rows[index])

        cached_cloud = self.lidar_analysis_cached_cloud
        cached_index = int(self.lidar_analysis_cached_index)
        if cached_cloud is not None and cached_index >= 0 and cached_index <= index:
            merged = cached_cloud
            start_index = cached_index + 1
        else:
            merged = np.zeros((0, 6), dtype=np.float32)
            start_index = 0

        for row in rows[start_index : index + 1]:
            cloud = self.load_lidar_analysis_cloud(row)
            if cloud.shape[0] == 0:
                continue
            merged = cloud if merged.shape[0] == 0 else np.vstack((merged, cloud))
            merged = flight.downsample_colored_point_cloud_voxel(
                merged,
                voxel_cm=flight.standard_voxel_size_m(flight.DEFAULT_LIDAR_RECON_VOXEL_CM),
                max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
            )
        self.lidar_analysis_cached_index = index
        self.lidar_analysis_cached_cloud = merged
        return merged

    def sample_lidar_analysis_display_cloud(self, cloud: np.ndarray, max_points: int) -> np.ndarray:
        points = np.asarray(cloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 6 or points.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        finite = np.isfinite(points[:, :3]).all(axis=1)
        if not np.all(finite):
            points = points[finite]
        if points.shape[0] > max_points:
            step = int(math.ceil(points.shape[0] / max_points))
            points = points[::step][:max_points]
        return points

    def show_lidar_analysis_row(self, index: int) -> None:
        if not self.lidar_analysis_rows:
            return
        index = max(0, min(index, len(self.lidar_analysis_rows) - 1))
        self.lidar_analysis_index = index
        if self.lidar_analysis_listbox is not None:
            self.lidar_analysis_listbox.selection_clear(0, "end")
            self.lidar_analysis_listbox.selection_set(index)
            self.lidar_analysis_listbox.see(index)
        row = self.lidar_analysis_rows[index]
        try:
            cloud = self.build_lidar_analysis_cloud_for_index(index)
            display_cloud = self.sample_lidar_analysis_display_cloud(cloud, self.parse_lidar_analysis_max_points())
            mode = str(self.lidar_analysis_mode_var.get() or "Cumulative")
            projection = str(row.get("depth_projection_selected", "plane_depth"))
            title = (
                f"{mode}: frame {int(row.get('frame_index', 0)):06d} "
                f"display={display_cloud.shape[0]} merged={cloud.shape[0]} "
                f"projection={projection} units=m"
            )
            self.plot_lidar_analysis_cloud(display_cloud, title=title)
            self.show_lidar_analysis_summary(row, cloud.shape[0], display_cloud.shape[0])
            self.show_lidar_analysis_json(row)
            self.lidar_stream_analysis_status_var.set(
                f"Lidar Analysis: frame {index + 1}/{len(self.lidar_analysis_rows)}, "
                f"mode={mode}, projection={projection}, units=m, display={display_cloud.shape[0]}, source={cloud.shape[0]}"
            )
        except Exception as exc:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: render failed: {exc}")

    def plot_lidar_analysis_cloud(self, cloud: np.ndarray, *, title: str) -> None:
        ax = getattr(self, "lidar_analysis_ax", None)
        canvas = getattr(self, "lidar_analysis_canvas", None)
        if ax is None or canvas is None:
            return
        ax.clear()
        points = np.asarray(cloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 6 or points.shape[0] == 0:
            ax.set_title(title)
            ax.text2D(0.35, 0.5, "No valid point cloud", transform=ax.transAxes)
            ax.set_xlabel("World X (m)")
            ax.set_ylabel("World Y (m)")
            ax.set_zlabel("World Z (m)")
            canvas.draw_idle()
            return

        xyz = points[:, :3]
        color_mode = str(self.lidar_analysis_color_mode_var.get() or "RGB").strip().lower()
        point_size = self.parse_lidar_analysis_point_size()
        if color_mode == "height":
            values = xyz[:, 2]
            vmin, vmax = self.lidar_analysis_percentile_limits(values)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=point_size,
                depthshade=False,
                linewidths=0,
            )
        elif color_mode == "depth":
            values = np.linalg.norm(xyz - np.nanmean(xyz, axis=0), axis=1)
            vmin, vmax = self.lidar_analysis_percentile_limits(values)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=values,
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
                s=point_size,
                depthshade=False,
                linewidths=0,
            )
        else:
            colors = np.clip(points[:, 3:6] / 255.0, 0.0, 1.0)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size, depthshade=False, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_zlabel("World Z (m)")
        self.apply_lidar_analysis_view_preset(ax)
        self.set_lidar_analysis_equal_axes(ax, xyz)
        canvas.draw_idle()

    def lidar_analysis_percentile_limits(self, values: np.ndarray) -> Tuple[float, float]:
        finite_values = np.asarray(values, dtype=np.float32)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return 0.0, 1.0
        vmin, vmax = np.nanpercentile(finite_values, [2.0, 98.0])
        if not math.isfinite(float(vmin)) or not math.isfinite(float(vmax)) or vmax <= vmin:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    def apply_lidar_analysis_view_preset(self, ax: Any) -> None:
        preset = str(self.lidar_analysis_view_preset_var.get() or "Perspective").strip().lower()
        if preset == "top":
            ax.view_init(elev=90, azim=-90)
        elif preset == "front":
            ax.view_init(elev=0, azim=-90)
        elif preset == "side":
            ax.view_init(elev=0, azim=0)
        else:
            ax.view_init(elev=24, azim=-58)

    def set_lidar_analysis_equal_axes(self, ax: Any, xyz: np.ndarray) -> None:
        mins = np.nanpercentile(xyz, 2.0, axis=0)
        maxs = np.nanpercentile(xyz, 98.0, axis=0)
        if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
            mins = np.nanmin(xyz, axis=0)
            maxs = np.nanmax(xyz, axis=0)
        ranges = np.maximum(maxs - mins, 1.0)
        center = (mins + maxs) * 0.5
        radius = float(np.max(ranges) * 0.55)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def show_lidar_analysis_summary(self, row: Dict[str, Any], source_count: int, display_count: int) -> None:
        if self.lidar_analysis_summary_text is None:
            return
        pose_payload = row.get("pose_payload") if isinstance(row.get("pose_payload"), dict) else {}
        pose = pose_payload.get("pose") if isinstance(pose_payload.get("pose"), dict) else {}
        lines = [
            f"Frame: {int(row.get('frame_index', 0)):06d} ({row.get('frame_name', '')})",
            f"Mode: {self.lidar_analysis_mode_var.get()}",
            f"Color mode: {self.lidar_analysis_color_mode_var.get()}",
            f"Point size: {self.parse_lidar_analysis_point_size():g}",
            f"View preset: {self.lidar_analysis_view_preset_var.get()}",
            f"Projection: {row.get('depth_projection_selected', 'plane_depth')}",
            f"Units: {row.get('coordinate_units', 'm')}",
            f"Coordinate frame: {row.get('coordinate_frame', 'standard_zup')}",
            f"Projection corrected: {bool(row.get('projection_corrected', True))}",
            f"Postprocess: {row.get('postprocess_status', '')}",
            f"Frame points: {int(row.get('point_count', 0) or 0)}",
            f"Raw cumulative points: {int(row.get('cumulative_point_count', 0) or 0)}",
            f"Rendered source points: {int(source_count)}",
            f"Displayed points: {int(display_count)}",
            f"Invalid depth: {int(row.get('invalid_depth_count', 0) or 0)}",
            f"Capture time: {row.get('capture_time', '')}",
            f"Cloud: {row.get('point_cloud_world_npy_path', '')}",
        ]
        if row.get("legacy_source_path"):
            lines.append(f"Legacy source: {row.get('legacy_source_path', '')}")
        if row.get("postprocess_error"):
            lines.append(f"Postprocess error: {row.get('postprocess_error', '')}")
        if pose:
            lines.append(
                "Pose: "
                f"x={float(pose.get('x', 0.0)):.1f}, "
                f"y={float(pose.get('y', 0.0)):.1f}, "
                f"z={float(pose.get('z', 0.0)):.1f}, "
                f"yaw={float(pose.get('task_yaw', pose.get('yaw', 0.0))):.1f}"
            )
        self.lidar_analysis_summary_text.delete("1.0", "end")
        self.lidar_analysis_summary_text.insert("1.0", "\n".join(lines))

    def show_lidar_analysis_json(self, row: Dict[str, Any]) -> None:
        if self.lidar_analysis_json_text is None:
            return
        payload = {
            "frame": row,
            "display": {
                "mode": self.lidar_analysis_mode_var.get(),
                "color_mode": self.lidar_analysis_color_mode_var.get(),
                "point_size": self.parse_lidar_analysis_point_size(),
                "view_preset": self.lidar_analysis_view_preset_var.get(),
                "max_display_points": self.parse_lidar_analysis_max_points(),
            },
        }
        self.lidar_analysis_json_text.delete("1.0", "end")
        self.lidar_analysis_json_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))

    def show_lidar_analysis_next(self) -> None:
        if not self.lidar_analysis_rows:
            return
        self.show_lidar_analysis_row((self.lidar_analysis_index + 1) % len(self.lidar_analysis_rows))

    def show_lidar_analysis_prev(self) -> None:
        if not self.lidar_analysis_rows:
            return
        self.show_lidar_analysis_row((self.lidar_analysis_index - 1) % len(self.lidar_analysis_rows))

    def toggle_lidar_analysis_playback(self) -> None:
        if self.lidar_analysis_playing:
            self.stop_lidar_analysis_playback()
        else:
            self.start_lidar_analysis_playback()

    def start_lidar_analysis_playback(self) -> None:
        if not self.lidar_analysis_rows:
            self.load_lidar_analysis_frames()
        if not self.lidar_analysis_rows:
            return
        self.lidar_analysis_playing = True
        self.cancel_lidar_analysis_after()
        self.lidar_analysis_after_id = self.root.after(self.lidar_analysis_interval_ms(), self.lidar_analysis_tick)
        self.show_lidar_analysis_row(self.lidar_analysis_index)

    def stop_lidar_analysis_playback(self) -> None:
        self.lidar_analysis_playing = False
        self.cancel_lidar_analysis_after()
        if self.lidar_analysis_rows:
            self.lidar_stream_analysis_status_var.set(
                f"Lidar Analysis: paused {self.lidar_analysis_index + 1}/{len(self.lidar_analysis_rows)}"
            )

    def cancel_lidar_analysis_after(self) -> None:
        after_id = getattr(self, "lidar_analysis_after_id", None)
        self.lidar_analysis_after_id = None
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass

    def lidar_analysis_tick(self) -> None:
        self.lidar_analysis_after_id = None
        if not self.lidar_analysis_playing:
            return
        if self.lidar_analysis_window is None or not self.lidar_analysis_window.winfo_exists():
            self.stop_lidar_analysis_playback()
            return
        self.show_lidar_analysis_next()
        self.lidar_analysis_after_id = self.root.after(self.lidar_analysis_interval_ms(), self.lidar_analysis_tick)

    def lidar_analysis_interval_ms(self) -> int:
        stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        interval_s = self.parse_stream_interval_s()
        if stream_text:
            for name in ("trajectory.json", "stream_capture_lidar.json"):
                payload = self.read_lidar_analysis_json(Path(stream_text) / name)
                if payload:
                    try:
                        interval_s = float(payload.get("interval_s", interval_s) or interval_s)
                        break
                    except Exception:
                        pass
        return int(max(80.0, interval_s * 1000.0))

    def lidar_analysis_packed_rgb_float(self, colors: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.rint(colors), 0, 255).astype(np.uint32, copy=False)
        rgb_u32 = (clipped[:, 0] << 16) | (clipped[:, 1] << 8) | clipped[:, 2]
        return rgb_u32.astype(np.uint32, copy=False).view(np.float32)

    def write_lidar_analysis_pcd(self, path: Path, point_cloud: np.ndarray) -> Dict[str, Any]:
        points = np.asarray(point_cloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 6:
            raise ValueError(f"Point cloud must be Nx6, got shape {points.shape}")
        finite = np.isfinite(points[:, :3]).all(axis=1)
        if not np.all(finite):
            points = points[finite]
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb_float = self.lidar_analysis_packed_rgb_float(points[:, 3:6]) if points.shape[0] else np.zeros((0,), dtype=np.float32)
        payload = np.column_stack((points[:, :3], rgb_float)).astype(np.float32, copy=False)
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {payload.shape[0]}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {payload.shape[0]}\n"
            "DATA ascii\n"
        )
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(header)
            if payload.size:
                np.savetxt(handle, payload, fmt=("%.6f", "%.6f", "%.6f", "%.8e"))
        return {"path": str(path), "point_count": int(payload.shape[0])}

    def open_lidar_yolo_labels_dialog(self) -> None:
        self.ensure_lidar_analysis_state()
        if self.lidar_yolo_analysis_thread is not None and self.lidar_yolo_analysis_thread.is_alive():
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: YOLO label analysis already running")
            return
        stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            latest = self.lidar_stream_capture_dir or self.find_latest_lidar_stream_capture_dir()
            if latest is not None:
                self.lidar_analysis_stream_dir_var.set(str(latest))
                stream_text = str(latest)
        if not stream_text:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: no folders under {self.lidar_stream_capture_root_path()}")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Lidar YOLO Labels")
        dialog.geometry("840x390")
        dialog.resizable(True, False)
        dialog.grid_columnconfigure(1, weight=1)
        folder_var = tk.StringVar(value=str(Path(stream_text).resolve()))
        weights_var = self.lidar_yolo_weights_var
        device_var = self.lidar_yolo_device_var
        conf_var = self.lidar_yolo_conf_var
        imgsz_var = self.lidar_yolo_imgsz_var
        stride_var = self.lidar_yolo_stride_var
        max_frames_var = self.lidar_yolo_max_frames_var
        dedupe_var = self.lidar_yolo_dedupe_radius_var
        max_points_var = self.lidar_yolo_max_points_var
        progress_var = tk.DoubleVar(value=0.0)
        status_var = tk.StringVar(value="Choose parameters, then start YOLO semantic projection.")
        eta_var = tk.StringVar(value="ETA: --")

        def set_latest_folder() -> None:
            latest = self.lidar_stream_capture_dir or self.find_latest_lidar_stream_capture_dir()
            if latest is not None:
                folder_var.set(str(Path(latest).resolve()))

        def browse_folder() -> None:
            selected = filedialog.askdirectory(
                title="Select stream_capture_lidar task folder",
                initialdir=str(self.lidar_stream_capture_root_path()),
            )
            if selected:
                folder_var.set(str(Path(selected).resolve()))

        def browse_weights() -> None:
            selected = filedialog.askopenfilename(
                title="Select YOLO weights",
                initialdir=str(lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_WEIGHTS_PATH.parent),
                filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
            )
            if selected:
                weights_var.set(str(Path(selected).resolve()))

        def use_default_weights() -> None:
            weights_var.set(str(lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_WEIGHTS_PATH))

        tk.Label(dialog, text="Folder").grid(row=0, column=0, sticky="e", padx=8, pady=(10, 4))
        tk.Entry(dialog, textvariable=folder_var).grid(row=0, column=1, columnspan=4, sticky="ew", padx=4, pady=(10, 4))
        tk.Button(dialog, text="Browse", command=browse_folder).grid(row=0, column=5, padx=4, pady=(10, 4))
        tk.Button(dialog, text="Latest", command=set_latest_folder).grid(row=0, column=6, padx=8, pady=(10, 4))

        tk.Label(dialog, text="Weights").grid(row=1, column=0, sticky="e", padx=8, pady=4)
        tk.Entry(dialog, textvariable=weights_var).grid(row=1, column=1, columnspan=4, sticky="ew", padx=4, pady=4)
        tk.Button(dialog, text="Browse", command=browse_weights).grid(row=1, column=5, padx=4, pady=4)
        tk.Button(dialog, text="Default", command=use_default_weights).grid(row=1, column=6, padx=8, pady=4)

        options = tk.Frame(dialog)
        options.grid(row=2, column=0, columnspan=7, sticky="ew", padx=8, pady=4)
        tk.Label(options, text="Device").grid(row=0, column=0, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=device_var, width=8).grid(row=0, column=1, padx=(0, 8), pady=3)
        tk.Label(options, text="Conf").grid(row=0, column=2, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=conf_var, width=7).grid(row=0, column=3, padx=(0, 8), pady=3)
        tk.Label(options, text="Imgsz").grid(row=0, column=4, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=imgsz_var, width=7).grid(row=0, column=5, padx=(0, 8), pady=3)
        tk.Label(options, text="Stride").grid(row=0, column=6, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=stride_var, width=7).grid(row=0, column=7, padx=(0, 8), pady=3)
        tk.Label(options, text="Max frames").grid(row=0, column=8, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=max_frames_var, width=8).grid(row=0, column=9, padx=(0, 8), pady=3)
        tk.Label(options, text="Dedupe m").grid(row=0, column=10, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=dedupe_var, width=8).grid(row=0, column=11, padx=(0, 8), pady=3)
        tk.Label(options, text="Pts/det").grid(row=0, column=12, padx=(0, 3), pady=3)
        tk.Entry(options, textvariable=max_points_var, width=8).grid(row=0, column=13, padx=(0, 8), pady=3)

        progress = ttk.Progressbar(dialog, variable=progress_var, maximum=100.0)
        progress.grid(row=3, column=0, columnspan=7, sticky="ew", padx=8, pady=(12, 4))
        tk.Label(dialog, textvariable=status_var, anchor="w").grid(row=4, column=0, columnspan=7, sticky="ew", padx=8, pady=2)
        tk.Label(dialog, textvariable=eta_var, anchor="w").grid(row=5, column=0, columnspan=7, sticky="ew", padx=8, pady=2)

        buttons = tk.Frame(dialog)
        buttons.grid(row=6, column=0, columnspan=7, sticky="e", padx=8, pady=(10, 8))
        start_button = tk.Button(buttons, text="Start YOLO Labels")
        start_button.pack(side="left", padx=4)
        tk.Button(buttons, text="Stop", command=lambda: self.lidar_yolo_stop_event.set()).pack(side="left", padx=4)
        tk.Button(buttons, text="Close", command=dialog.destroy).pack(side="left", padx=4)

        def parse_float(var: tk.StringVar, default: float, *, minimum: float, maximum: float) -> float:
            try:
                value = float(var.get().strip())
            except Exception:
                value = default
            value = max(minimum, min(maximum, value))
            var.set(f"{value:g}")
            return value

        def parse_int(var: tk.StringVar, default: int, *, minimum: int, maximum: int) -> int:
            try:
                value = int(float(var.get().strip()))
            except Exception:
                value = default
            value = max(minimum, min(maximum, value))
            var.set(str(value))
            return value

        def apply_progress(payload: Dict[str, Any], started_at: float) -> None:
            processed = int(payload.get("processed", 0) or 0)
            total = max(1, int(payload.get("total", 1) or 1))
            progress_var.set(min(100.0, processed * 100.0 / total))
            elapsed = max(0.0, time.monotonic() - started_at)
            if processed > 0 and total > processed:
                eta_s = elapsed * (total - processed) / processed
                eta_var.set(f"ETA: {eta_s:,.1f}s remaining | elapsed {elapsed:,.1f}s")
            elif processed >= total:
                eta_var.set(f"ETA: done | elapsed {elapsed:,.1f}s")
            else:
                eta_var.set(f"ETA: estimating... | elapsed {elapsed:,.1f}s")
            status_var.set(str(payload.get("message", "YOLO semantic projection running")))

        def start_analysis() -> None:
            if self.lidar_yolo_analysis_thread is not None and self.lidar_yolo_analysis_thread.is_alive():
                status_var.set("YOLO label analysis is already running.")
                return
            selected_dir = Path(folder_var.get().strip()).resolve()
            selected_weights = Path(weights_var.get().strip()).resolve()
            if not selected_dir.exists():
                status_var.set(f"Folder does not exist: {selected_dir}")
                return
            if not selected_weights.exists():
                status_var.set(f"YOLO weights do not exist: {selected_weights}")
                return
            self.lidar_analysis_stream_dir_var.set(str(selected_dir))
            selected_device = device_var.get().strip() or "0"
            selected_conf = parse_float(conf_var, lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_CONF, minimum=0.01, maximum=1.0)
            selected_imgsz = parse_int(imgsz_var, lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_IMGSZ, minimum=64, maximum=4096)
            selected_stride = parse_int(stride_var, 1, minimum=1, maximum=100000)
            selected_max_frames = parse_int(max_frames_var, 0, minimum=0, maximum=1000000)
            selected_radius = parse_float(
                dedupe_var,
                lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_DEDUPE_RADIUS_M,
                minimum=0.01,
                maximum=100.0,
            )
            selected_max_points = parse_int(
                max_points_var,
                lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_MAX_POINTS_PER_DETECTION,
                minimum=1,
                maximum=200000,
            )
            self.lidar_yolo_stop_event.clear()
            progress_var.set(0.0)
            eta_var.set("ETA: estimating...")
            status_var.set(f"Starting YOLO labels -> {selected_dir / 'open3d_export' / 'semantic'}")
            start_button.configure(state="disabled")
            started_at = time.monotonic()

            def progress_callback(payload: Dict[str, Any]) -> None:
                self.root.after(0, lambda p=payload: apply_progress(p, started_at))

            def worker() -> None:
                try:
                    result = lidar_yolo_analysis.run_lidar_yolo_analysis(
                        stream_dir=selected_dir,
                        weights_path=selected_weights,
                        device=selected_device,
                        conf=selected_conf,
                        imgsz=selected_imgsz,
                        stride=selected_stride,
                        max_frames=selected_max_frames,
                        dedupe_radius_m=selected_radius,
                        max_points_per_detection=selected_max_points,
                        stop_event=self.lidar_yolo_stop_event,
                        progress_callback=progress_callback,
                    )
                except Exception as exc:
                    self.root.after(
                        0,
                        lambda e=exc: (
                            status_var.set(f"YOLO labels failed: {e}"),
                            eta_var.set("ETA: failed"),
                            start_button.configure(state="normal"),
                            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: YOLO labels failed: {e}"),
                        ),
                    )
                    return
                self.root.after(
                    0,
                    lambda r=result: (
                        self.apply_lidar_yolo_analysis_result(r),
                        status_var.set(
                            f"Done: labels={r.get('semantic_label_count', 0)} "
                            f"observations={r.get('semantic_observation_count', 0)} "
                            f"points={r.get('semantic_point_count', 0)}"
                        ),
                        eta_var.set(f"ETA: done | total {time.monotonic() - started_at:,.1f}s"),
                        progress_var.set(100.0),
                        start_button.configure(state="normal"),
                    ),
                )

            self.lidar_yolo_analysis_thread = threading.Thread(target=worker, daemon=True)
            self.lidar_yolo_analysis_thread.start()

        start_button.configure(command=start_analysis)

    def apply_lidar_yolo_analysis_result(self, result: Dict[str, Any]) -> None:
        semantic_export_dir = Path(str(result.get("semantic_export_dir", "")))
        export_dir = semantic_export_dir.parent if semantic_export_dir.name == "semantic" else semantic_export_dir
        try:
            status = flight.open3d_status()
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "README_open3d.md").write_text(self.build_lidar_analysis_open3d_readme(status), encoding="utf-8")
            (export_dir / "open3d_viewer.py").write_text(self.build_lidar_analysis_open3d_viewer_script(), encoding="utf-8")
        except Exception:
            pass
        self.lidar_stream_analysis_status_var.set(
            f"Lidar Analysis: YOLO labels status={result.get('status', '')}, "
            f"labels={result.get('semantic_label_count', 0)}, "
            f"observations={result.get('semantic_observation_count', 0)} -> {semantic_export_dir}"
        )
        self.status_var.set(f"YOLO semantic Open3D labels saved: {semantic_export_dir}")

    def build_lidar_analysis_open3d_readme(self, status: Dict[str, Any]) -> str:
        install_hint = (
            "C:\\Users\\Administrator\\miniconda3\\envs\\unrealcv\\python.exe -m pip install \"open3d>=0.19.0\""
        )
        availability = "available" if status.get("available") else f"unavailable: {status.get('error', '')}"
        return f"""# Open3D export for stream_capture_lidar

This folder contains Open3D-generated point cloud files exported from the UAV controller lidar stream.

Open3D status: {availability}

Files:
- frames/frame_000001_world_standard_m.ply and .pcd: per-frame standard Z-up world point clouds.
- reconstruction_world_standard_m.ply and .pcd: accumulated standard Z-up world point cloud.
- reconstruction_world_standard_m.npy: processed Nx6 point/color array.
- semantic/semantic_points_standard_m.ply and .pcd: YOLO detection regions projected into the same Z-up world frame.
- semantic/semantic_labels.json: deduplicated 3D semantic labels for Open3D overlay.
- open3d_viewer.py: standalone Open3D viewer script.
- open3d_export_summary.json: export metadata.

The coordinates are right-handed standard Z-up world coordinates in meters.

If Open3D is missing, install it in the UnrealCV environment:

```powershell
{install_hint}
```
"""

    def build_lidar_analysis_open3d_viewer_script(self) -> str:
        return r'''from pathlib import Path
import json

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_point_cloud(path):
    if not path.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(path))
    return pcd if pcd.has_points() else None


def load_clouds():
    named = []
    reconstruction = load_point_cloud(ROOT / "reconstruction_world_standard_m.ply")
    if reconstruction is not None:
        named.append(("reconstruction_world_standard_m", reconstruction))
    if not named:
        for path in sorted((ROOT / "frames").glob("*_world_standard_m.ply"))[:8]:
            pcd = load_point_cloud(path)
            if pcd is not None:
                named.append((path.stem, pcd))
    semantic = load_point_cloud(ROOT / "semantic" / "semantic_points_standard_m.ply")
    if semantic is not None:
        named.append(("semantic_yolo_points", semantic))
    return named


def load_labels():
    payload = load_json(ROOT / "semantic" / "semantic_labels.json")
    labels = payload.get("labels", []) if isinstance(payload, dict) else []
    return [label for label in labels if isinstance(label, dict)]


def label_text(label):
    return (
        f"{label.get('class_name', 'object')} #{label.get('label_id', '')} "
        f"{float(label.get('best_confidence') or 0.0):.2f} "
        f"obs={label.get('observation_count', 0)}"
    )


def label_center(label):
    center = label.get("center_world_m", [])
    if not isinstance(center, list) or len(center) < 3:
        return None
    try:
        arr = np.array([float(center[0]), float(center[1]), float(center[2])], dtype=float)
    except Exception:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def label_color(label):
    color = label.get("color_rgb", [255, 255, 255])
    try:
        return [max(0.0, min(1.0, float(color[i]) / 255.0)) for i in range(3)]
    except Exception:
        return [1.0, 1.0, 1.0]


def label_markers(labels):
    markers = []
    for label in labels:
        center = label_center(label)
        if center is None:
            continue
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sphere.translate(center)
        sphere.paint_uniform_color(label_color(label))
        markers.append((f"label_marker_{label.get('label_id', len(markers) + 1)}", sphere))
    return markers


def try_gui_view(named_geometries, labels):
    try:
        from open3d.visualization import gui
        gui.Application.instance.initialize()
        visualizer = o3d.visualization.O3DVisualizer("UAV Open3D Point Cloud + YOLO Labels", 1400, 900)
        for name, geometry in named_geometries:
            visualizer.add_geometry(name, geometry)
        for label in labels:
            center = label_center(label)
            if center is not None:
                visualizer.add_3d_label(center, label_text(label))
        try:
            visualizer.reset_camera_to_default()
        except Exception:
            pass
        gui.Application.instance.add_window(visualizer)
        gui.Application.instance.run()
        return True
    except Exception as exc:
        print(f"O3DVisualizer labels unavailable, falling back to draw_geometries: {exc}")
        return False


def main():
    named_geometries = load_clouds()
    labels = load_labels()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0.0, 0.0, 0.0])
    named_geometries.append(("axis", axis))
    if labels and try_gui_view(named_geometries, labels):
        return
    fallback = [geometry for _name, geometry in named_geometries]
    fallback.extend(geometry for _name, geometry in label_markers(labels))
    if not fallback:
        raise SystemExit(f"No Open3D point clouds found under {ROOT}")
    for label in labels:
        center = label_center(label)
        print(f"{label_text(label)} center={center.tolist() if center is not None else 'n/a'}")
    o3d.visualization.draw_geometries(
        fallback,
        window_name="UAV Open3D Point Cloud",
        width=1400,
        height=900,
    )


if __name__ == "__main__":
    main()
'''

    def build_lidar_analysis_open3d_export(
        self,
        stream_dir: Path,
        rows: Optional[List[Dict[str, Any]]] = None,
        *,
        export_mode: str = "reconstruction_only",
        frame_stride: int = 10,
        max_workers: int = 1,
        estimate_reconstruction_normals: bool = True,
        reuse_existing: bool = True,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        stream_path = Path(stream_dir).resolve()
        export_mode = str(export_mode or "reconstruction_only").strip().lower()
        if export_mode not in {"reconstruction_only", "sampled_frames", "all_frames"}:
            export_mode = "reconstruction_only"
        frame_stride = max(1, int(frame_stride or 1))
        max_workers = max(1, int(max_workers or 1))

        def emit_progress(stage: str, processed: int, total: int, message: str = "") -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(
                    {
                        "stage": stage,
                        "processed": int(processed),
                        "total": int(max(1, total)),
                        "message": str(message),
                        "export_mode": export_mode,
                    }
                )
            except Exception:
                pass

        emit_progress("postprocess_check", 0, 1, "Checking standard point clouds")
        postprocess_result: Dict[str, Any] = {}
        stream_summary = self.read_lidar_analysis_json(stream_path / "stream_capture_lidar.json")
        needs_postprocess = str(stream_summary.get("postprocess_status", "")).lower() in {"pending", "running", ""}
        if not needs_postprocess:
            for capture_dir in sorted(path for path in (stream_path / "frames").glob("frame_*") if path.is_dir()):
                if not (capture_dir / "point_cloud_world_standard_m.npy").exists():
                    needs_postprocess = True
                    break
        if needs_postprocess:
            emit_progress("postprocess", 0, 1, "Postprocessing raw frames")
            postprocess_result = flight.postprocess_lidar_stream_capture(
                stream_path,
                lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
                min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
                max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
                voxel_cm=flight.DEFAULT_LIDAR_RECON_VOXEL_CM,
                max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
            )
            rows = None
            emit_progress("postprocess", 1, 1, "Postprocess complete")
        frame_rows = list(rows) if rows is not None else self.scan_lidar_analysis_frames(stream_path)
        if not frame_rows:
            raise RuntimeError(f"No lidar point cloud frames found in {stream_path}")

        if export_mode == "reconstruction_only":
            export_rows: List[Dict[str, Any]] = []
        elif export_mode == "sampled_frames":
            export_rows = [
                row
                for index, row in enumerate(frame_rows)
                if index == 0 or index == len(frame_rows) - 1 or index % frame_stride == 0
            ]
        else:
            export_rows = list(frame_rows)

        export_dir = stream_path / "open3d_export"
        frames_dir = export_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        status = flight.open3d_status()
        readme_path = export_dir / "README_open3d.md"
        viewer_path = export_dir / "open3d_viewer.py"
        summary_path = export_dir / "open3d_export_summary.json"
        existing_export_summary = self.read_lidar_analysis_json(summary_path)
        semantic_summary = self.read_lidar_analysis_json(export_dir / "semantic" / "semantic_projection_summary.json")
        readme_path.write_text(self.build_lidar_analysis_open3d_readme(status), encoding="utf-8")
        viewer_path.write_text(self.build_lidar_analysis_open3d_viewer_script(), encoding="utf-8")

        summary: Dict[str, Any] = {
            "export_kind": "open3d_point_cloud_export",
            "stream_dir": str(stream_path),
            "export_dir": str(export_dir),
            "frames_dir": str(frames_dir),
            "open3d": status,
            "frame_count": len(frame_rows),
            "exported_frame_count": len(export_rows),
            "source_point_count": 0,
            "reconstruction_point_count": 0,
            "frame_exports": [],
            "readme_path": str(readme_path),
            "viewer_path": str(viewer_path),
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "postprocess_checked": True,
            "postprocess_ran": bool(postprocess_result),
            "postprocess": postprocess_result,
            "skipped_empty_frame_count": 0,
            "skipped_existing_frame_count": 0,
            "export_mode": export_mode,
            "frame_stride": int(frame_stride),
            "max_workers": int(max_workers),
            "estimate_reconstruction_normals": bool(estimate_reconstruction_normals),
            "reuse_existing": bool(reuse_existing),
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        semantic_source = semantic_summary if semantic_summary else existing_export_summary
        if semantic_source:
            summary.update(
                {
                    "semantic_yolo_status": semantic_source.get("status", semantic_source.get("semantic_yolo_status", "")),
                    "semantic_label_count": int(semantic_source.get("semantic_label_count", 0) or 0),
                    "semantic_observation_count": int(semantic_source.get("semantic_observation_count", 0) or 0),
                    "semantic_points_path": semantic_source.get("semantic_points_path", ""),
                    "semantic_labels_path": semantic_source.get("semantic_labels_path", ""),
                    "semantic_analysis_dir": semantic_source.get("analysis_dir", semantic_source.get("semantic_analysis_dir", "")),
                    "semantic_projection_summary_path": semantic_source.get(
                        "summary_path",
                        semantic_source.get("stable_summary_path", semantic_source.get("semantic_projection_summary_path", "")),
                    ),
                    "semantic_selected_frame_count": int(semantic_source.get("selected_frame_count", semantic_source.get("semantic_selected_frame_count", 0)) or 0),
                    "semantic_processed_frame_count": int(semantic_source.get("processed_frame_count", semantic_source.get("semantic_processed_frame_count", 0)) or 0),
                    "semantic_config": semantic_source.get("config", semantic_source.get("semantic_config", {})),
                    "dedupe_radius_m": float(
                        semantic_source.get("dedupe_radius_m", lidar_yolo_analysis.DEFAULT_LIDAR_YOLO_DEDUPE_RADIUS_M)
                    ),
                    "projection_coordinate_frame": "standard_zup",
                    "projection_coordinate_units": "m",
                }
            )
        if not status.get("available"):
            summary["status"] = "open3d_unavailable"
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
            summary["summary_path"] = str(summary_path)
            return summary

        source_point_count = int(
            stream_summary.get("source_point_count", 0)
            or postprocess_result.get("source_point_count", 0)
            or 0
        )
        if source_point_count <= 0:
            source_point_count = sum(int(row.get("point_count", 0) or 0) for row in frame_rows)
        skipped_empty_frame_count = 0
        skipped_existing_frame_count = 0
        frame_exports: List[Dict[str, Any]] = []
        frame_tasks: List[Dict[str, Any]] = []
        for row in export_rows:
            frame_index = int(row.get("frame_index", len(frame_tasks) + 1) or len(frame_tasks) + 1)
            raw_path = str(row.get("point_cloud_world_standard_m_npy_path") or row.get("point_cloud_world_npy_path") or "")
            if not raw_path or not Path(raw_path).exists() or int(row.get("point_count", 0) or 0) <= 0:
                skipped_empty_frame_count += 1
                frame_exports.append(
                    {
                        "frame_index": frame_index,
                        "backend": "open3d",
                        "available": bool(status.get("available")),
                        "version": status.get("version", ""),
                        "basename": f"frame_{frame_index:06d}_world_standard_m",
                        "source_point_count": 0,
                        "processed_point_count": 0,
                        "skipped_empty": True,
                        "coordinate_units": "m",
                        "ply_path": "",
                        "pcd_path": "",
                        "npy_path": "",
                    }
                )
                continue
            frame_tasks.append(
                {
                    "frame_index": frame_index,
                    "npy_path": raw_path,
                    "output_dir": str(frames_dir),
                    "basename": f"frame_{frame_index:06d}_world_standard_m",
                    "voxel_cm": 0.0,
                    "voxel_size": 0.0,
                    "max_points": 0,
                    "estimate_normals": False,
                    "coordinate_units": "m",
                    "reuse_existing": bool(reuse_existing),
                }
            )

        total_units = len(frame_tasks) + 1
        completed_units = 0
        emit_progress("frame_export", completed_units, total_units, f"Exporting {len(frame_tasks)} frame clouds")
        if frame_tasks and max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(flight.save_open3d_point_cloud_from_npy_task, task): task
                    for task in frame_tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    frame_export = future.result()
                    if frame_export.get("skipped_empty"):
                        skipped_empty_frame_count += 1
                    if frame_export.get("skipped_existing"):
                        skipped_existing_frame_count += 1
                    frame_exports.append(frame_export)
                    completed_units += 1
                    emit_progress(
                        "frame_export",
                        completed_units,
                        total_units,
                        f"Exported frame {frame_export.get('frame_index', '')}",
                    )
        else:
            for task in frame_tasks:
                frame_export = flight.save_open3d_point_cloud_from_npy_task(task)
                if frame_export.get("skipped_empty"):
                    skipped_empty_frame_count += 1
                if frame_export.get("skipped_existing"):
                    skipped_existing_frame_count += 1
                frame_exports.append(frame_export)
                completed_units += 1
                emit_progress(
                    "frame_export",
                    completed_units,
                    total_units,
                    f"Exported frame {frame_export.get('frame_index', '')}",
                )
        frame_exports.sort(key=lambda item: int(item.get("frame_index", 0) or 0))

        reconstruction_npy = ""
        for candidate in (
            stream_path / "reconstruction" / "merged_point_cloud_world_standard_m.npy",
            stream_path / "reconstruction" / "reconstruction_world_standard_m.npy",
        ):
            if candidate.exists():
                reconstruction_npy = str(candidate)
                break
        emit_progress("reconstruction_export", completed_units, total_units, "Exporting reconstruction")
        if reconstruction_npy:
            reconstruction_export = flight.save_open3d_point_cloud_from_npy_task(
                {
                    "frame_index": 0,
                    "npy_path": reconstruction_npy,
                    "output_dir": str(export_dir),
                    "basename": "reconstruction_world_standard_m",
                    "voxel_cm": flight.DEFAULT_OPEN3D_VOXEL_CM,
                    "voxel_size": flight.standard_voxel_size_m(flight.DEFAULT_OPEN3D_VOXEL_CM),
                    "max_points": flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
                    "estimate_normals": bool(estimate_reconstruction_normals),
                    "normal_radius_cm": flight.DEFAULT_OPEN3D_NORMAL_RADIUS_CM,
                    "normal_radius": flight.DEFAULT_OPEN3D_NORMAL_RADIUS_CM / 100.0,
                    "coordinate_units": "m",
                    "reuse_existing": bool(reuse_existing),
                }
            )
        else:
            merged = np.zeros((0, 6), dtype=np.float32)
            for row in frame_rows:
                cloud = self.load_lidar_analysis_cloud(row)
                if cloud.shape[0] == 0:
                    continue
                merged = cloud if merged.shape[0] == 0 else np.vstack((merged, cloud))
                merged = flight.downsample_colored_point_cloud_voxel(
                    merged,
                    voxel_cm=flight.standard_voxel_size_m(flight.DEFAULT_LIDAR_RECON_VOXEL_CM),
                    max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
                )
            reconstruction_export = flight.save_open3d_point_cloud_outputs(
                merged,
                export_dir,
                basename="reconstruction_world_standard_m",
                voxel_cm=flight.DEFAULT_OPEN3D_VOXEL_CM,
                voxel_size=flight.standard_voxel_size_m(flight.DEFAULT_OPEN3D_VOXEL_CM),
                max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
                estimate_normals=bool(estimate_reconstruction_normals),
                normal_radius_cm=flight.DEFAULT_OPEN3D_NORMAL_RADIUS_CM,
                normal_radius=flight.DEFAULT_OPEN3D_NORMAL_RADIUS_CM / 100.0,
                coordinate_units="m",
            )
        if reconstruction_export.get("skipped_existing"):
            reconstruction_export["reused_existing"] = True
        completed_units = total_units
        emit_progress("done", completed_units, total_units, "Open3D export complete")
        summary.update(
            {
                "status": "ok",
                "source_point_count": int(source_point_count),
                "reconstruction_point_count": int(reconstruction_export.get("processed_point_count", 0) or 0),
                "skipped_empty_frame_count": int(skipped_empty_frame_count),
                "skipped_existing_frame_count": int(skipped_existing_frame_count),
                "frame_exports": frame_exports,
                "reconstruction": reconstruction_export,
                "reconstruction_world_standard_m_ply_path": reconstruction_export.get("ply_path", ""),
                "reconstruction_world_standard_m_pcd_path": reconstruction_export.get("pcd_path", ""),
                "reconstruction_world_standard_m_npy_path": reconstruction_export.get("npy_path", ""),
                "reconstruction_world_ply_path": reconstruction_export.get("ply_path", ""),
                "reconstruction_world_pcd_path": reconstruction_export.get("pcd_path", ""),
                "reconstruction_world_npy_path": reconstruction_export.get("npy_path", ""),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    def export_lidar_analysis_open3d(self) -> None:
        self.ensure_lidar_analysis_state()
        if self.lidar_analysis_open3d_thread is not None and self.lidar_analysis_open3d_thread.is_alive():
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: Open3D export already running")
            return
        stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            latest = self.lidar_stream_capture_dir or self.find_latest_lidar_stream_capture_dir()
            if latest is not None:
                self.lidar_analysis_stream_dir_var.set(str(latest))
                stream_text = str(latest)
        if not stream_text:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: no folders under {self.lidar_stream_capture_root_path()}")
            return
        self.open_lidar_analysis_open3d_export_dialog(Path(stream_text))

    def open_lidar_analysis_open3d_export_dialog(self, stream_dir: Path) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Open3D Export Options")
        dialog.geometry("720x310")
        dialog.resizable(True, False)
        dialog.grid_columnconfigure(1, weight=1)

        folder_var = tk.StringVar(value=str(Path(stream_dir).resolve()))
        mode_var = tk.StringVar(value="reconstruction_only")
        stride_var = tk.StringVar(value="10")
        workers_default = max(1, min(4, (os.cpu_count() or 2) - 1))
        workers_var = tk.StringVar(value=str(workers_default))
        normals_var = tk.BooleanVar(value=True)
        reuse_var = tk.BooleanVar(value=True)
        progress_var = tk.DoubleVar(value=0.0)
        status_var = tk.StringVar(value="Choose export parameters, then start.")
        eta_var = tk.StringVar(value="ETA: --")

        def set_latest_folder() -> None:
            latest = self.lidar_stream_capture_dir or self.find_latest_lidar_stream_capture_dir()
            if latest is not None:
                folder_var.set(str(Path(latest).resolve()))

        def browse_folder() -> None:
            selected = filedialog.askdirectory(
                title="Select stream_capture_lidar task folder",
                initialdir=str(self.lidar_stream_capture_root_path()),
            )
            if selected:
                folder_var.set(str(Path(selected).resolve()))

        tk.Label(dialog, text="Folder").grid(row=0, column=0, sticky="e", padx=8, pady=(10, 4))
        tk.Entry(dialog, textvariable=folder_var).grid(row=0, column=1, sticky="ew", padx=4, pady=(10, 4))
        tk.Button(dialog, text="Browse", command=browse_folder).grid(row=0, column=2, padx=4, pady=(10, 4))
        tk.Button(dialog, text="Latest", command=set_latest_folder).grid(row=0, column=3, padx=8, pady=(10, 4))

        tk.Label(dialog, text="Mode").grid(row=1, column=0, sticky="e", padx=8, pady=4)
        ttk.Combobox(
            dialog,
            textvariable=mode_var,
            values=("reconstruction_only", "sampled_frames", "all_frames"),
            state="readonly",
            width=22,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        tk.Label(dialog, text="Frame stride").grid(row=2, column=0, sticky="e", padx=8, pady=4)
        tk.Entry(dialog, textvariable=stride_var, width=8).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        tk.Label(dialog, text="CPU workers").grid(row=2, column=2, sticky="e", padx=8, pady=4)
        tk.Entry(dialog, textvariable=workers_var, width=8).grid(row=2, column=3, sticky="w", padx=8, pady=4)

        tk.Checkbutton(dialog, text="Estimate reconstruction normals", variable=normals_var).grid(
            row=3, column=1, sticky="w", padx=4, pady=4
        )
        tk.Checkbutton(dialog, text="Reuse existing exports when present", variable=reuse_var).grid(
            row=4, column=1, sticky="w", padx=4, pady=4
        )

        progress = ttk.Progressbar(dialog, variable=progress_var, maximum=100.0)
        progress.grid(row=5, column=0, columnspan=4, sticky="ew", padx=8, pady=(12, 4))
        tk.Label(dialog, textvariable=status_var, anchor="w").grid(row=6, column=0, columnspan=4, sticky="ew", padx=8, pady=2)
        tk.Label(dialog, textvariable=eta_var, anchor="w").grid(row=7, column=0, columnspan=4, sticky="ew", padx=8, pady=2)

        buttons = tk.Frame(dialog)
        buttons.grid(row=8, column=0, columnspan=4, sticky="e", padx=8, pady=(10, 8))
        start_button = tk.Button(buttons, text="Start Export")
        start_button.pack(side="left", padx=4)
        tk.Button(buttons, text="Close", command=dialog.destroy).pack(side="left", padx=4)

        def parse_positive_int(var: tk.StringVar, default: int, *, minimum: int = 1, maximum: int = 9999) -> int:
            try:
                value = int(float(var.get().strip()))
            except Exception:
                value = default
            value = max(minimum, min(maximum, value))
            var.set(str(value))
            return value

        def apply_progress(payload: Dict[str, Any], started_at: float) -> None:
            processed = int(payload.get("processed", 0) or 0)
            total = max(1, int(payload.get("total", 1) or 1))
            progress_var.set(min(100.0, processed * 100.0 / total))
            elapsed = max(0.0, time.monotonic() - started_at)
            if processed > 0 and total > processed:
                eta_s = elapsed * (total - processed) / processed
                eta_var.set(f"ETA: {eta_s:,.1f}s remaining | elapsed {elapsed:,.1f}s")
            elif processed >= total:
                eta_var.set(f"ETA: done | elapsed {elapsed:,.1f}s")
            else:
                eta_var.set(f"ETA: estimating... | elapsed {elapsed:,.1f}s")
            stage = str(payload.get("stage", "export"))
            message = str(payload.get("message", ""))
            status_var.set(f"{stage}: {processed}/{total} {message}".strip())

        def start_export() -> None:
            if self.lidar_analysis_open3d_thread is not None and self.lidar_analysis_open3d_thread.is_alive():
                status_var.set("Open3D export is already running.")
                return
            selected_dir = Path(folder_var.get().strip()).resolve()
            if not selected_dir.exists():
                status_var.set(f"Folder does not exist: {selected_dir}")
                return
            self.lidar_analysis_stream_dir_var.set(str(selected_dir))
            stride = parse_positive_int(stride_var, 10, minimum=1, maximum=100000)
            workers = parse_positive_int(workers_var, workers_default, minimum=1, maximum=max(1, os.cpu_count() or 1))
            selected_mode = mode_var.get()
            selected_normals = bool(normals_var.get())
            selected_reuse = bool(reuse_var.get())
            progress_var.set(0.0)
            eta_var.set("ETA: estimating...")
            status_var.set(f"Starting Open3D export -> {selected_dir / 'open3d_export'}")
            start_button.configure(state="disabled")
            started_at = time.monotonic()

            def progress_callback(payload: Dict[str, Any]) -> None:
                self.root.after(0, lambda p=payload: apply_progress(p, started_at))

            def worker() -> None:
                try:
                    result = self.build_lidar_analysis_open3d_export(
                        selected_dir,
                        rows=None,
                        export_mode=selected_mode,
                        frame_stride=stride,
                        max_workers=workers,
                        estimate_reconstruction_normals=selected_normals,
                        reuse_existing=selected_reuse,
                        progress_callback=progress_callback,
                    )
                except Exception as exc:
                    self.root.after(
                        0,
                        lambda e=exc: (
                            status_var.set(f"Open3D export failed: {e}"),
                            eta_var.set("ETA: failed"),
                            start_button.configure(state="normal"),
                            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: Open3D export failed: {e}"),
                        ),
                    )
                    return
                self.root.after(
                    0,
                    lambda r=result: (
                        self.apply_lidar_analysis_open3d_export_result(r),
                        status_var.set(
                            f"Done: mode={r.get('export_mode')} frames={r.get('exported_frame_count', 0)} "
                            f"points={r.get('reconstruction_point_count', 0)}"
                        ),
                        eta_var.set(f"ETA: done | total {time.monotonic() - started_at:,.1f}s"),
                        progress_var.set(100.0),
                        start_button.configure(state="normal"),
                    ),
                )

            self.lidar_analysis_open3d_thread = threading.Thread(target=worker, daemon=True)
            self.lidar_analysis_open3d_thread.start()

        start_button.configure(command=start_export)

    def apply_lidar_analysis_open3d_export_result(self, result: Dict[str, Any]) -> None:
        status = str(result.get("status", ""))
        if status == "open3d_unavailable":
            error_text = result.get("open3d", {}).get("error", "") if isinstance(result.get("open3d"), dict) else ""
            self.lidar_stream_analysis_status_var.set(
                f"Lidar Analysis: Open3D unavailable: {error_text} -> {result.get('export_dir', '')}"
            )
            self.status_var.set("Open3D export needs the open3d package")
            return
        self.lidar_stream_analysis_status_var.set(
            f"Lidar Analysis: Open3D export mode={result.get('export_mode', '')}, "
            f"frames={result.get('exported_frame_count', 0)}/{result.get('frame_count', 0)}, "
            f"reused={result.get('skipped_existing_frame_count', 0)}, "
            f"points={result.get('reconstruction_point_count', 0)} -> {result.get('export_dir', '')}"
        )
        self.status_var.set(f"Open3D export saved: {result.get('export_dir', '')}")

    def open_lidar_analysis_open3d_viewer(self) -> None:
        stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            latest = self.lidar_stream_capture_dir or self.find_latest_lidar_stream_capture_dir()
            if latest is not None:
                stream_text = str(latest)
                self.lidar_analysis_stream_dir_var.set(stream_text)
        if not stream_text:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: no folders under {self.lidar_stream_capture_root_path()}")
            return
        export_dir = Path(stream_text) / "open3d_export"
        viewer_path = export_dir / "open3d_viewer.py"
        if not viewer_path.exists():
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: export Open3D files before opening viewer")
            return
        status = flight.open3d_status()
        if not status.get("available"):
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: Open3D unavailable: {status.get('error', '')}")
            return
        try:
            import subprocess
            import sys

            subprocess.Popen([sys.executable, str(viewer_path)], cwd=str(export_dir))
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: Open3D viewer launched -> {viewer_path}")
        except Exception as exc:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: Open3D viewer launch failed: {exc}")

    def build_lidar_analysis_rviz_config(self) -> str:
        return """Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Enabled: true
      Name: Grid
      Value: true
    - Class: rviz_default_plugins/PointCloud2
      Enabled: true
      Name: Lidar Reconstruction
      Topic:
        Value: /uav/lidar/reconstruction
      Style: Points
      Size (Pixels): 2
      Color Transformer: RGB8
    - Class: rviz_default_plugins/PointCloud2
      Enabled: true
      Name: Lidar Current Frame
      Topic:
        Value: /uav/lidar/frame
      Style: Points
      Size (Pixels): 2
      Color Transformer: RGB8
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
  Value: true
Window Geometry:
  Height: 900
  Width: 1400
"""

    def build_lidar_analysis_ros2_publisher_script(self) -> str:
        return r'''#!/usr/bin/env python3
"""Publish exported stream_capture_lidar PCD files for RViz2.

Usage:
  source /opt/ros/<distro>/setup.bash
  python3 publish_lidar_stream_ros2.py
  rviz2 -d rviz_config.rviz
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


EXPORT_DIR = Path(__file__).resolve().parent
FRAME_ID = "map"


def read_pcd_ascii(path: Path):
    fields = []
    data_started = False
    rows = []
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        upper = text.upper()
        if upper.startswith("FIELDS "):
            fields = text.split()[1:]
            continue
        if upper.startswith("DATA "):
            data_started = True
            continue
        if not data_started:
            continue
        parts = text.split()
        if len(parts) < 4:
            continue
        try:
            values = [float(part) for part in parts]
            x = values[fields.index("x")] if "x" in fields else values[0]
            y = values[fields.index("y")] if "y" in fields else values[1]
            z = values[fields.index("z")] if "z" in fields else values[2]
            rgb = values[fields.index("rgb")] if "rgb" in fields else values[3]
            rows.append((x, y, z, rgb))
        except Exception:
            continue
    return rows


def make_cloud(points, frame_id: str) -> PointCloud2:
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = b"".join(struct.pack("<ffff", float(x), float(y), float(z), float(rgb)) for x, y, z, rgb in points)
    return msg


class LidarStreamPublisher(Node):
    def __init__(self):
        super().__init__("uav_lidar_stream_publisher")
        self.reconstruction_pub = self.create_publisher(PointCloud2, "/uav/lidar/reconstruction", 1)
        self.frame_pub = self.create_publisher(PointCloud2, "/uav/lidar/frame", 1)
        self.frame_paths = sorted((EXPORT_DIR / "frames").glob("frame_*_world.pcd"))
        self.frame_index = 0
        reconstruction_path = EXPORT_DIR / "reconstruction_world.pcd"
        self.reconstruction_points = read_pcd_ascii(reconstruction_path) if reconstruction_path.exists() else []
        self.timer = self.create_timer(0.5, self.publish_once)
        self.get_logger().info(f"Loaded {len(self.frame_paths)} frame PCD files from {EXPORT_DIR}")

    def publish_once(self):
        now = self.get_clock().now().to_msg()
        if self.reconstruction_points:
            msg = make_cloud(self.reconstruction_points, FRAME_ID)
            msg.header.stamp = now
            self.reconstruction_pub.publish(msg)
        if self.frame_paths:
            path = self.frame_paths[self.frame_index % len(self.frame_paths)]
            msg = make_cloud(read_pcd_ascii(path), FRAME_ID)
            msg.header.stamp = now
            self.frame_pub.publish(msg)
            self.frame_index += 1


def main():
    rclpy.init()
    node = LidarStreamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
'''

    def build_lidar_analysis_rviz_readme(self) -> str:
        return """# RViz export for stream_capture_lidar

This folder contains standard Z-up meter PCD files exported from the UAV controller lidar stream.

Files:
- frames/frame_000001_world.pcd: per-frame world point clouds.
- reconstruction_world.pcd: accumulated world point cloud.
- rviz_config.rviz: RViz2 display config for /uav/lidar/reconstruction and /uav/lidar/frame.
- publish_lidar_stream_ros2.py: optional ROS2 publisher for RViz2.

The coordinates are right-handed standard Z-up world coordinates in meters.

Example ROS2 usage:

```bash
source /opt/ros/<distro>/setup.bash
cd /path/to/rviz_export
python3 publish_lidar_stream_ros2.py
rviz2 -d rviz_config.rviz
```

If RViz opens but the cloud is not visible, check that Fixed Frame is `map` and that the two PointCloud2 topics are active.
"""

    def build_lidar_analysis_rviz_export(self, stream_dir: Path, rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        stream_path = Path(stream_dir).resolve()
        frame_rows = list(rows) if rows is not None else self.scan_lidar_analysis_frames(stream_path)
        if not frame_rows:
            raise RuntimeError(f"No lidar point cloud frames found in {stream_path}")
        export_dir = stream_path / "rviz_export"
        frames_dir = export_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        merged = np.zeros((0, 6), dtype=np.float32)
        source_point_count = 0
        frame_exports: List[Dict[str, Any]] = []
        for row in frame_rows:
            cloud = self.load_lidar_analysis_cloud(row)
            source_point_count += int(cloud.shape[0])
            frame_index = int(row.get("frame_index", len(frame_exports) + 1) or len(frame_exports) + 1)
            frame_path = frames_dir / f"frame_{frame_index:06d}_world.pcd"
            frame_exports.append(
                {
                    "frame_index": frame_index,
                    **self.write_lidar_analysis_pcd(frame_path, cloud),
                }
            )
            if cloud.shape[0]:
                merged = cloud if merged.shape[0] == 0 else np.vstack((merged, cloud))
                merged = flight.downsample_colored_point_cloud_voxel(
                    merged,
                    voxel_cm=flight.standard_voxel_size_m(flight.DEFAULT_LIDAR_RECON_VOXEL_CM),
                    max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
                )

        reconstruction_pcd = export_dir / "reconstruction_world.pcd"
        reconstruction_export = self.write_lidar_analysis_pcd(reconstruction_pcd, merged)
        rviz_config_path = export_dir / "rviz_config.rviz"
        publisher_path = export_dir / "publish_lidar_stream_ros2.py"
        readme_path = export_dir / "README_rviz.md"
        summary_path = export_dir / "rviz_export_summary.json"
        rviz_config_path.write_text(self.build_lidar_analysis_rviz_config(), encoding="utf-8")
        publisher_path.write_text(self.build_lidar_analysis_ros2_publisher_script(), encoding="utf-8")
        readme_path.write_text(self.build_lidar_analysis_rviz_readme(), encoding="utf-8")
        summary = {
            "export_kind": "rviz_point_cloud_export",
            "stream_dir": str(stream_path),
            "export_dir": str(export_dir),
            "frames_dir": str(frames_dir),
            "frame_count": len(frame_exports),
            "source_point_count": int(source_point_count),
            "reconstruction_point_count": int(reconstruction_export["point_count"]),
            "frame_pcd_paths": [item["path"] for item in frame_exports],
            "reconstruction_world_pcd_path": str(reconstruction_pcd),
            "rviz_config_path": str(rviz_config_path),
            "ros2_publisher_path": str(publisher_path),
            "readme_path": str(readme_path),
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "topics": {
                "reconstruction": "/uav/lidar/reconstruction",
                "frame": "/uav/lidar/frame",
            },
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    def export_lidar_analysis_rviz(self) -> None:
        self.ensure_lidar_analysis_state()
        if self.lidar_analysis_export_thread is not None and self.lidar_analysis_export_thread.is_alive():
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: RViz export already running")
            return
        stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            self.use_latest_lidar_analysis_folder()
            stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            return
        stream_dir = Path(stream_text)
        rows = list(self.lidar_analysis_rows)
        if not rows:
            try:
                rows = self.scan_lidar_analysis_frames(stream_dir)
            except Exception as exc:
                self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: scan before export failed: {exc}")
                return
        self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: exporting RViz files -> {stream_dir / 'rviz_export'}")

        def worker() -> None:
            result = self.safe("Export RViz", lambda: self.build_lidar_analysis_rviz_export(stream_dir, rows))
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_lidar_analysis_rviz_export_result(r))

        self.lidar_analysis_export_thread = threading.Thread(target=worker, daemon=True)
        self.lidar_analysis_export_thread.start()

    def apply_lidar_analysis_rviz_export_result(self, result: Dict[str, Any]) -> None:
        self.lidar_stream_analysis_status_var.set(
            f"Lidar Analysis: RViz export frames={result.get('frame_count', 0)}, "
            f"points={result.get('reconstruction_point_count', 0)} -> {result.get('export_dir', '')}"
        )
        self.status_var.set(f"RViz export saved: {result.get('export_dir', '')}")

    def rebuild_lidar_analysis_window(self) -> None:
        self.ensure_lidar_analysis_state()
        if self.lidar_analysis_rebuild_thread is not None and self.lidar_analysis_rebuild_thread.is_alive():
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: rebuild already running")
            return
        stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            self.use_latest_lidar_analysis_folder()
            stream_text = self.lidar_analysis_stream_dir_var.get().strip()
        if not stream_text:
            return
        stream_dir = Path(stream_text)
        self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: rebuilding -> {stream_dir}")

        def worker() -> None:
            result = self.safe("Analyze Lidar", lambda: self.rebuild_lidar_stream_reconstruction(stream_dir))
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_lidar_analysis_rebuild_result(r))

        self.lidar_analysis_rebuild_thread = threading.Thread(target=worker, daemon=True)
        self.lidar_analysis_rebuild_thread.start()

    def apply_lidar_analysis_rebuild_result(self, result: Dict[str, Any]) -> None:
        stream_dir = result.get("stream_dir", "")
        merged_count = int(result.get("merged_point_count", 0) or 0)
        frame_count = int(result.get("source_frame_count", 0) or 0)
        self.lidar_stream_last_reconstruction = result
        ply_path = result.get("merged_point_cloud_world_standard_m_ply_path", result.get("merged_point_cloud_world_ply_path", ""))
        self.lidar_stream_analysis_status_var.set(
            f"Lidar Analysis: rebuilt frames={frame_count}, merged={merged_count}, units=m -> {ply_path}"
        )
        if stream_dir:
            self.lidar_analysis_stream_dir_var.set(str(stream_dir))
        self.reset_lidar_analysis_cache()
        if self.lidar_analysis_rows:
            self.show_lidar_analysis_row(self.lidar_analysis_index)
