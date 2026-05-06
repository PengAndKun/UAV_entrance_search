from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageTk

import run_drone_flight as flight


LOGGER = logging.getLogger(__name__)


MOVE_COMMANDS: Dict[str, Dict[str, Any]] = {
    "w": {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "forward"},
    "s": {"forward_cm": -20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "backward"},
    "a": {"forward_cm": 0.0, "right_cm": -20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "left"},
    "d": {"forward_cm": 0.0, "right_cm": 20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "right"},
    "r": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 20.0, "yaw_delta_deg": 0.0, "action_name": "up"},
    "f": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": -20.0, "yaw_delta_deg": 0.0, "action_name": "down"},
    "q": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": -30.0, "action_name": "yaw_left"},
    "e": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 30.0, "action_name": "yaw_right"},
    "x": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "hold"},
}
YAW_GRID_STEP_DEG = 30.0
YAW_SNAP_SYMBOLS = {"q", "e"}


class RunDroneFlightPanel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session: Optional[flight.DroneFlightSession] = None
        self.latest_state: Dict[str, Any] = {}
        self.manual_request_inflight = False
        self.move_request_inflight = False
        self.state_refresh_inflight = False
        self.preview_refresh_inflight = False
        self.sequence_thread: Optional[threading.Thread] = None
        self.sequence_stop_event = threading.Event()

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
        self.initial_pose_var = tk.StringVar(value=" ".join(str(value) for value in args.initial_pos))

        self.sequence_var = tk.StringVar(value="")
        self.sequence_delay_var = tk.StringVar(value="150")
        self.auto_rgb_var = tk.BooleanVar(value=False)

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
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._bind_hotkeys()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root)
        outer.grid(row=0, column=0, sticky="nsew")
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
        tk.Button(preview, text="Refresh State", command=self.refresh_state_once).pack(side="left", padx=6, pady=6)

        orbit = tk.LabelFrame(outer, text="Orbit Plan")
        orbit.grid(row=6, column=0, sticky="ew", padx=8, pady=(4, 8))
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
        for symbol in MOVE_COMMANDS:
            self.root.bind(symbol, lambda event, s=symbol: self._on_hotkey(event, s))

    def _on_hotkey(self, event: tk.Event, symbol: str) -> None:
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox)):
            return
        self.send_move_symbol(symbol)

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
            save_every=max(0, int(float(self.save_every_var.get().strip()))),
            movement_mode=self.movement_mode_var.get().strip() or flight.DEFAULT_MOVEMENT_MODE,
            initial_pos=initial_pos,
            mode="keyboard",
            auto_action="none",
            max_steps=0,
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

    def on_start_session(self) -> None:
        if self.session is not None and self.session.started:
            self.status_var.set("Session already started.")
            return

        def worker() -> Dict[str, Any]:
            session_args = self.build_flight_args()
            self.session = flight.DroneFlightSession(session_args)
            return self.session.start()

        self.call_async("Starting run_drone_flight", worker)

    def on_stop_session(self) -> None:
        session = self.session
        if session is None:
            self.status_var.set("No active session.")
            return

        def worker() -> Dict[str, Any]:
            session.close()
            return {"status": "ok", "message": "Session stopped", "started": False}

        self.call_async("Stopping session", worker)

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

    def _fmt_float(self, value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return str(value)

    def refresh_state_once(self) -> None:
        session = self.session
        if session is None or not session.started:
            return
        if self.state_refresh_inflight:
            return

        def worker() -> None:
            self.state_refresh_inflight = True
            try:
                state = self.safe("Refreshing state", session.get_state)
                if isinstance(state, dict):
                    self.root.after(0, lambda r=state: self.apply_state(r))
            finally:
                self.state_refresh_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def schedule_state_refresh(self) -> None:
        if not self.manual_request_inflight and not self.move_request_inflight:
            session = self.session
            if session is not None and session.started:
                self.refresh_state_once()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)

    def on_toggle_movement(self) -> None:
        session = self.active_session()
        if session is None:
            return
        self.call_async(
            "Toggling movement",
            lambda: session.set_movement_enabled(not self.movement_enabled_state),
        )

    def on_movement_mode_selected(self, _event: Optional[tk.Event] = None) -> None:
        self.movement_mode_state = self.movement_mode_var.get().strip() or flight.DEFAULT_MOVEMENT_MODE
        session = self.session
        if session is None or not session.started:
            self.status_var.set(f"Movement mode will apply on start: {self.movement_mode_state}")
            return
        self.call_async(
            "Changing movement mode",
            lambda: session.set_movement_mode(self.movement_mode_state),
        )

    def latest_yaw_deg(self) -> Optional[float]:
        pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        if not pose:
            return None
        try:
            return float(pose.get("task_yaw", pose.get("yaw")))
        except (TypeError, ValueError):
            return None

    def snap_yaw_target_for_symbol(self, symbol: str, current_yaw_deg: float) -> float:
        normalized = flight.normalize_angle_deg(float(current_yaw_deg))
        nearest_grid = round(normalized / YAW_GRID_STEP_DEG) * YAW_GRID_STEP_DEG
        direction = -1.0 if symbol.lower() == "q" else 1.0
        return flight.normalize_angle_deg(nearest_grid + direction * YAW_GRID_STEP_DEG)

    def move_payload_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        payload = MOVE_COMMANDS.get(symbol.lower())
        if payload is None:
            return None
        return dict(payload)

    def _execute_move(self, symbol: str, *, from_sequence: bool = False) -> bool:
        session = self.active_session()
        if session is None:
            return False
        payload = self.move_payload_for_symbol(symbol)
        if payload is None:
            return False
        self.move_request_inflight = True
        try:
            response = self.safe(f"Move {symbol}", lambda: session.move_relative(payload))
            if isinstance(response, dict):
                self.root.after(0, lambda r=response: self.apply_state(r))
                if str(response.get("status", "")).lower() in {"error", "disabled"}:
                    return False
                if from_sequence:
                    self.root.after(0, lambda s=symbol: self.status_var.set(f"Sequence sent: {s}"))
                return True
            return False
        finally:
            self.move_request_inflight = False

    def send_move_symbol(self, symbol: str) -> None:
        if self.move_request_inflight:
            self.status_var.set(f"Move {symbol} ignored while another move is in flight.")
            return
        threading.Thread(target=lambda: self._execute_move(symbol), daemon=True).start()

    def on_execute_sequence(self) -> None:
        symbols = [symbol for symbol in self.sequence_var.get().strip().lower() if symbol in MOVE_COMMANDS]
        if not symbols:
            self.status_var.set("No valid sequence symbols.")
            return
        try:
            delay_s = max(0.0, float(self.sequence_delay_var.get().strip()) / 1000.0)
        except ValueError:
            self.status_var.set("Invalid sequence delay.")
            return
        if self.sequence_thread is not None and self.sequence_thread.is_alive():
            self.status_var.set("Sequence already running.")
            return

        def worker() -> None:
            self.sequence_stop_event.clear()
            total = len(symbols)
            for index, symbol in enumerate(symbols, start=1):
                if self.sequence_stop_event.is_set():
                    self.root.after(0, lambda i=index, t=total: self.status_var.set(f"Sequence stopped at {i - 1}/{t}."))
                    return
                self.root.after(0, lambda i=index, t=total, s=symbol: self.status_var.set(f"Sequence {i}/{t}: {s}"))
                if not self._execute_move(symbol, from_sequence=True):
                    self.root.after(0, lambda i=index, t=total: self.status_var.set(f"Sequence failed at {i}/{t}."))
                    return
                time.sleep(delay_s)
            self.root.after(0, lambda: self.status_var.set(f"Sequence completed: {total}/{total}"))

        self.sequence_thread = threading.Thread(target=worker, daemon=True)
        self.sequence_thread.start()

    def on_stop_sequence(self) -> None:
        self.sequence_stop_event.set()
        self.status_var.set("Stopping sequence...")

    def on_set_pose(self) -> None:
        try:
            payload = json.loads(self.pose_text.get("1.0", "end").strip())
        except json.JSONDecodeError as exc:
            self.status_var.set(f"Invalid pose JSON: {exc}")
            return
        session = self.active_session()
        if session is None:
            return
        self.call_async("Setting pose", lambda: session.set_pose(payload))

    def on_save_frame(self) -> None:
        session = self.active_session()
        if session is None:
            return
        self.call_async("Saving RGB frame", lambda: (session.capture_observation(save=True, label="manual"), session.get_state())[1])

    def image_to_photo(self, image: np.ndarray, max_width: int = 900, max_height: int = 620) -> ImageTk.PhotoImage:
        array = np.asarray(image)
        if array.ndim == 4:
            array = array[0]
        array = np.clip(array[:, :, :3], 0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        scale = min(max_width / max(1, pil.width), max_height / max(1, pil.height), 1.0)
        if scale < 1.0:
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def toggle_preview_window(self) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()
            self.preview_window = None
            self.preview_label = None
            self.preview_photo = None
            return
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("UAV RGB Preview")
        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack(fill="both", expand=True)
        self.refresh_preview_window()

    def refresh_preview_window(self) -> None:
        if self.preview_refresh_inflight:
            return
        session = self.session
        if session is None or not session.started:
            return

        def worker() -> None:
            self.preview_refresh_inflight = True
            try:
                image = self.safe("Refresh RGB", session.capture_observation)
                if isinstance(image, np.ndarray):
                    self.root.after(0, lambda img=image: self.apply_preview_image(img))
            finally:
                self.preview_refresh_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def apply_preview_image(self, image: np.ndarray) -> None:
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.toggle_preview_window()
        if self.preview_label is None:
            return
        self.preview_photo = self.image_to_photo(image)
        self.preview_label.configure(image=self.preview_photo)

    def schedule_preview_refresh(self) -> None:
        if self.auto_rgb_var.get() and self.preview_window is not None and self.preview_window.winfo_exists():
            self.refresh_preview_window()
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

    def on_run_orbit(self) -> None:
        session = self.active_session()
        if session is None:
            return

        def worker() -> Dict[str, Any]:
            return session.run_orbit(
                center=[float(self.orbit_center_x_var.get().strip()), float(self.orbit_center_y_var.get().strip())],
                radius=float(self.orbit_radius_var.get().strip()),
                altitude=float(self.orbit_altitude_var.get().strip()),
                steps=int(float(self.orbit_steps_var.get().strip())),
                start_angle=float(self.orbit_start_angle_var.get().strip()),
                clockwise=bool(self.orbit_clockwise_var.get()),
            )

        self.call_async("Running orbit", worker)

    def on_run_scripted(self) -> None:
        session = self.active_session()
        if session is None:
            return
        self.call_async("Running scripted smoke plan", session.run_scripted)

    def on_close(self) -> None:
        self.sequence_stop_event.set()
        session = self.session
        if session is not None and session.started:
            try:
                session.close()
            except Exception as exc:
                LOGGER.warning("Failed to close session: %s", exc)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tk controller for run_drone_flight.py")
    parser.add_argument("--env_platform", "--platform", choices=["auto", "win", "mac", "linux"], default="auto")
    parser.add_argument("--env_root", default=None)
    parser.add_argument("--env_bin", default=None)
    parser.add_argument("--output_dir", default="results/drone_flight_controller")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--launch_sleep", type=int, default=15)
    parser.add_argument("--time_dilation", type=int, default=0)
    parser.add_argument("--step_delay", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--movement_mode", choices=["pose_lock", "physics"], default=flight.DEFAULT_MOVEMENT_MODE)
    parser.add_argument("--initial_pos", nargs="+", type=float, default=flight.DEFAULT_INITIAL_POS)
    parser.add_argument("--orbit_center", nargs=2, type=float, default=flight.DEFAULT_ORBIT_CENTER)
    parser.add_argument("--orbit_radius", type=float, default=flight.DEFAULT_ORBIT_RADIUS)
    parser.add_argument("--orbit_altitude", type=float, default=flight.DEFAULT_ORBIT_ALTITUDE)
    parser.add_argument("--orbit_steps", type=int, default=flight.DEFAULT_ORBIT_STEPS)
    parser.add_argument("--orbit_start_angle", type=float, default=flight.DEFAULT_ORBIT_START_ANGLE)
    parser.add_argument("--orbit_clockwise", action="store_true")
    parser.add_argument("--state_interval_ms", type=int, default=1500)
    parser.add_argument("--preview_interval_ms", type=int, default=1500)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    RunDroneFlightPanel(args).run()


if __name__ == "__main__":
    main()
