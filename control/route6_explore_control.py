from __future__ import annotations

from .common import *


ROUTE6_DESIGN_DOC = "overleaf/route6_nearest_house_pointcloud_map_design.md"


class Route6ExploreControlMixin:
    def ensure_route6_state(self) -> None:
        if not hasattr(self, "llm_route6_window"):
            self.llm_route6_window = None
        if not hasattr(self, "llm_route6_summary_text"):
            self.llm_route6_summary_text = None
        if not hasattr(self, "llm_route6_state"):
            self.llm_route6_state = {}
        if not hasattr(self, "llm_route6_stop_event"):
            self.llm_route6_stop_event = threading.Event()
        if not hasattr(self, "llm_route6_status_var"):
            self.llm_route6_status_var = tk.StringVar(value="LLM Route V6: idle")
        if not hasattr(self, "llm_route6_stage_var"):
            self.llm_route6_stage_var = tk.StringVar(value="Stage: idle")
        if not hasattr(self, "llm_route6_current_house_var"):
            self.llm_route6_current_house_var = tk.StringVar(value="Current house: n/a")
        if not hasattr(self, "llm_route6_queue_var"):
            self.llm_route6_queue_var = tk.StringVar(value="House queue: n/a")
        if not hasattr(self, "llm_route6_map_status_var"):
            self.llm_route6_map_status_var = tk.StringVar(value="Map: idle")
        if not hasattr(self, "llm_route6_output_dir_var"):
            self.llm_route6_output_dir_var = tk.StringVar(value="Output: n/a")
        if not hasattr(self, "llm_route6_max_houses_var"):
            self.llm_route6_max_houses_var = tk.StringVar(value="3")
        if not hasattr(self, "llm_route6_runtime_min_var"):
            self.llm_route6_runtime_min_var = tk.StringVar(value="30")
        if not hasattr(self, "llm_route6_standoff_cm_var"):
            default = getattr(self, "llm_route_standoff_cm_var", None)
            self.llm_route6_standoff_cm_var = tk.StringVar(value=str(default.get() if default is not None else "850"))
        if not hasattr(self, "llm_route6_scan_z_cm_var"):
            self.llm_route6_scan_z_cm_var = tk.StringVar(value="450")
        if not hasattr(self, "llm_route6_occupancy_resolution_m_var"):
            self.llm_route6_occupancy_resolution_m_var = tk.StringVar(value="0.25")
        if not hasattr(self, "llm_route6_coverage_threshold_var"):
            self.llm_route6_coverage_threshold_var = tk.StringVar(value="0.75")
        if not hasattr(self, "llm_route6_allow_save_corrected_var"):
            self.llm_route6_allow_save_corrected_var = tk.BooleanVar(value=False)

    def route6_design_doc_path(self) -> Path:
        return PROJECT_ROOT / ROUTE6_DESIGN_DOC

    def route6_update_summary_text(self) -> None:
        text_widget = getattr(self, "llm_route6_summary_text", None)
        if text_widget is None:
            return
        design_path = self.route6_design_doc_path()
        payload = {
            "mode": "route6_nearest_house_pointcloud_map",
            "design_doc": str(design_path),
            "current_stage": self.llm_route6_stage_var.get(),
            "planned_pipeline": [
                "rank nearest reachable houses",
                "scan selected house facades",
                "merge valid point clouds",
                "build 2D occupancy grid",
                "extract building/obstacle polygons",
                "write route6 corrected map artifact",
                "run entrance search after local map confidence is sufficient",
            ],
            "implementation_status": "button_and_window_ready; worker/map-builder implementation pending",
        }
        try:
            text_widget.configure(state="normal")
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, json.dumps(payload, indent=2, ensure_ascii=False))
            text_widget.configure(state="disabled")
        except tk.TclError:
            pass

    def route6_set_stage(self, stage: str, message: str = "") -> None:
        self.ensure_route6_state()
        self.llm_route6_stage_var.set(f"Stage: {stage}")
        if message:
            self.llm_route6_status_var.set(f"LLM Route V6: {message}")
        self.llm_route6_state["stage"] = str(stage)
        self.llm_route6_state["message"] = str(message or "")
        self.route6_update_summary_text()

    def open_llm_route_window6(self) -> None:
        self.ensure_route6_state()
        if self.llm_route6_window is not None and self.llm_route6_window.winfo_exists():
            self.llm_route6_window.lift()
            self.llm_route6_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route 6")
        window.geometry("1120x760")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(3, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close_llm_route_window6)

        header = tk.LabelFrame(window, text="Route 6 Nearest House Pointcloud Map")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        for col in (1, 3, 5, 7):
            header.grid_columnconfigure(col, weight=1)

        tk.Label(header, text="Max houses").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_max_houses_var, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Runtime min").grid(row=0, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_runtime_min_var, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Standoff cm").grid(row=0, column=4, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_standoff_cm_var, width=8).grid(row=0, column=5, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Scan z cm").grid(row=0, column=6, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_scan_z_cm_var, width=8).grid(row=0, column=7, sticky="w", padx=6, pady=5)

        tk.Label(header, text="Occupancy m").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_occupancy_resolution_m_var, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=5)
        tk.Label(header, text="Coverage").grid(row=1, column=2, sticky="w", padx=6, pady=5)
        tk.Entry(header, textvariable=self.llm_route6_coverage_threshold_var, width=8).grid(row=1, column=3, sticky="w", padx=6, pady=5)
        tk.Checkbutton(
            header,
            text="Allow save corrected config",
            variable=self.llm_route6_allow_save_corrected_var,
        ).grid(row=1, column=4, columnspan=3, sticky="w", padx=6, pady=5)

        actions = tk.Frame(window)
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        tk.Button(actions, text="Start Nearest House Map Search", command=self.on_route6_start_nearest_map_search).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=self.on_route6_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_route6_clear).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Save Corrected Map Config", command=self.on_route6_save_corrected_map_config).pack(side="left", padx=6, pady=4)

        status = tk.LabelFrame(window, text="Status")
        status.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        status.grid_columnconfigure(0, weight=1)
        status.grid_columnconfigure(1, weight=1)
        tk.Label(status, textvariable=self.llm_route6_stage_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_current_house_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_queue_var, anchor="w").grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_map_status_var, anchor="w").grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_output_dir_var, anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=3)
        tk.Label(status, textvariable=self.llm_route6_status_var, anchor="w", wraplength=1040, justify="left").grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=3)

        summary = tk.LabelFrame(window, text="Route 6 Implementation Contract")
        summary.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 8))
        summary.grid_columnconfigure(0, weight=1)
        summary.grid_rowconfigure(0, weight=1)
        text = tk.Text(summary, height=18, wrap="none", font=("Consolas", 9))
        text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        scroll = tk.Scrollbar(summary, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns", pady=6)
        text.configure(yscrollcommand=scroll.set, state="disabled")

        self.llm_route6_window = window
        self.llm_route6_summary_text = text
        self.route6_update_summary_text()

    def close_llm_route_window6(self) -> None:
        self.ensure_route6_state()
        if self.llm_route6_window is not None:
            try:
                self.llm_route6_window.destroy()
            except Exception:
                pass
        self.llm_route6_window = None
        self.llm_route6_summary_text = None

    def on_route6_start_nearest_map_search(self) -> None:
        self.ensure_route6_state()
        self.llm_route6_stop_event.clear()
        target = ""
        try:
            target = self.selected_route_target_house_id()
        except Exception:
            target = ""
        self.llm_route6_state = {
            "mode": "route6_nearest_house_pointcloud_map",
            "stage": "DESIGN_READY",
            "selected_target_hint": str(target or ""),
            "design_doc": str(self.route6_design_doc_path()),
            "max_houses": self.llm_route6_max_houses_var.get(),
            "runtime_minutes": self.llm_route6_runtime_min_var.get(),
            "standoff_cm": self.llm_route6_standoff_cm_var.get(),
            "scan_z_cm": self.llm_route6_scan_z_cm_var.get(),
            "occupancy_resolution_m": self.llm_route6_occupancy_resolution_m_var.get(),
            "coverage_threshold": self.llm_route6_coverage_threshold_var.get(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.llm_route6_current_house_var.set(f"Current house: {target or 'nearest reachable pending'}")
        self.llm_route6_queue_var.set("House queue: pending ranking")
        self.llm_route6_map_status_var.set("Map: waiting for route6 map builder")
        self.llm_route6_output_dir_var.set("Output: route6_explore_runs/<next run>")
        self.route6_set_stage("DESIGN_READY", "button ready; Route 6 worker/map builder will be implemented next.")

    def on_route6_stop(self) -> None:
        self.ensure_route6_state()
        self.llm_route6_stop_event.set()
        self.route6_set_stage("STOPPED", "stop requested.")

    def on_route6_clear(self) -> None:
        self.ensure_route6_state()
        self.llm_route6_stop_event.clear()
        self.llm_route6_state = {}
        self.llm_route6_stage_var.set("Stage: idle")
        self.llm_route6_current_house_var.set("Current house: n/a")
        self.llm_route6_queue_var.set("House queue: n/a")
        self.llm_route6_map_status_var.set("Map: idle")
        self.llm_route6_output_dir_var.set("Output: n/a")
        self.llm_route6_status_var.set("LLM Route V6: cleared.")
        self.route6_update_summary_text()

    def on_route6_save_corrected_map_config(self) -> None:
        self.ensure_route6_state()
        if not bool(self.llm_route6_allow_save_corrected_var.get()):
            self.llm_route6_status_var.set("LLM Route V6: enable corrected-config save before writing global map config.")
            return
        self.llm_route6_status_var.set("LLM Route V6: no Route 6 corrected config artifact is available yet.")
