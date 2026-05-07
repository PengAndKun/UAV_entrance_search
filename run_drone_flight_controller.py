from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from map_overhead_widget import OverheadMapWidget
import run_drone_flight as flight


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAP_CONFIG_PATH = "assets/overhead_map/houses_config.json"
DEFAULT_MAP_BOUNDS = (1000.0, -500.0, 5000.0, 3000.0)
DEFAULT_CORRECTED_MAP_CONFIG_NAME = "corrected_houses_config.json"


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


def solve_affine_from_anchor_points(anchors: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
    if len(anchors) < 3:
        return None
    try:
        world = np.asarray(
            [[float(anchor["world_x"]), float(anchor["world_y"]), 1.0] for anchor in anchors],
            dtype=np.float64,
        )
        image_x = np.asarray([float(anchor["image_x"]) for anchor in anchors], dtype=np.float64)
        image_y = np.asarray([float(anchor["image_y"]) for anchor in anchors], dtype=np.float64)
        row_x, *_ = np.linalg.lstsq(world, image_x, rcond=None)
        row_y, *_ = np.linalg.lstsq(world, image_y, rcond=None)
        return [
            [float(row_x[0]), float(row_x[1]), float(row_x[2])],
            [float(row_y[0]), float(row_y[1]), float(row_y[2])],
        ]
    except Exception:
        return None


def world_to_image_with_affine(world_x: float, world_y: float, affine: List[List[float]]) -> Tuple[float, float]:
    matrix = np.asarray(affine, dtype=np.float64)
    image_x = float(matrix[0, 0]) * float(world_x) + float(matrix[0, 1]) * float(world_y) + float(matrix[0, 2])
    image_y = float(matrix[1, 0]) * float(world_x) + float(matrix[1, 1]) * float(world_y) + float(matrix[1, 2])
    return image_x, image_y


def image_to_world_with_affine(image_x: float, image_y: float, affine: List[List[float]]) -> Tuple[float, float]:
    matrix = np.asarray(affine, dtype=np.float64)
    linear = matrix[:, :2]
    offset = matrix[:, 2]
    world = np.linalg.solve(linear, np.asarray([float(image_x), float(image_y)], dtype=np.float64) - offset)
    return float(world[0]), float(world[1])


def affine_rmse_px(anchors: List[Dict[str, Any]], affine: List[List[float]]) -> float:
    errors: List[float] = []
    for anchor in anchors:
        predicted_x, predicted_y = world_to_image_with_affine(
            float(anchor["world_x"]),
            float(anchor["world_y"]),
            affine,
        )
        dx = predicted_x - float(anchor["image_x"])
        dy = predicted_y - float(anchor["image_y"])
        errors.append(dx * dx + dy * dy)
    if not errors:
        return 0.0
    return float(math.sqrt(sum(errors) / len(errors)))


def corrected_anchors_from_touch_state(
    original_anchors: List[Dict[str, Any]],
    touch_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    points = touch_state.get("points", []) if isinstance(touch_state.get("points"), list) else []
    done_by_label = {
        str(point.get("label", "")): point
        for point in points
        if isinstance(point, dict) and point.get("status") == "done"
    }
    corrected: List[Dict[str, Any]] = []
    for index, anchor in enumerate(original_anchors[:5], start=1):
        label = str(anchor.get("label", f"P{index}") or f"P{index}")
        point = done_by_label.get(label)
        if point is None:
            raise ValueError(f"Calibration point {label} is not completed")
        contact_x = point.get("contact_world_x")
        contact_y = point.get("contact_world_y")
        if contact_x is None or contact_y is None:
            pose = point.get("contact_pose", [])
            if not isinstance(pose, list) or len(pose) < 2:
                raise ValueError(f"Calibration point {label} has no contact pose")
            contact_x, contact_y = pose[0], pose[1]
        corrected.append({
            "index": int(float(anchor.get("index", index))),
            "label": label,
            "world_x": float(contact_x),
            "world_y": float(contact_y),
            "image_x": float(anchor["image_x"]),
            "image_y": float(anchor["image_y"]),
            "source_world_x": float(anchor["world_x"]),
            "source_world_y": float(anchor["world_y"]),
        })
    if len(corrected) < 5:
        raise ValueError("P1-P5 calibration requires five completed points")
    return corrected


def rebuild_houses_for_corrected_affine(config: Dict[str, Any], affine: List[List[float]]) -> List[Dict[str, Any]]:
    rebuilt: List[Dict[str, Any]] = []
    houses = config.get("houses", []) if isinstance(config.get("houses"), list) else []
    for house in houses:
        if not isinstance(house, dict):
            continue
        new_house = json.loads(json.dumps(house))
        bbox = new_house.get("map_bbox_image")
        if isinstance(bbox, dict):
            try:
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
                cx_img = (x1 + x2) * 0.5
                cy_img = (y1 + y2) * 0.5
                half_w = abs(x2 - x1) * 0.5
                half_h = abs(y2 - y1) * 0.5
                center_x, center_y = image_to_world_with_affine(cx_img, cy_img, affine)
                right_x, right_y = image_to_world_with_affine(cx_img + half_w, cy_img, affine)
                down_x, down_y = image_to_world_with_affine(cx_img, cy_img + half_h, affine)
                radius = max(
                    float(np.hypot(right_x - center_x, right_y - center_y)),
                    float(np.hypot(down_x - center_x, down_y - center_y)),
                    300.0,
                )
                new_house["center_x"] = center_x
                new_house["center_y"] = center_y
                new_house["radius_cm"] = radius
            except Exception as exc:
                new_house["map_rebuild_error"] = str(exc)
        rebuilt.append(new_house)
    return rebuilt


def build_corrected_map_config(config: Dict[str, Any], touch_state: Dict[str, Any]) -> Dict[str, Any]:
    corrected = json.loads(json.dumps(config))
    overhead = corrected.setdefault("overhead_map", {})
    calibration = overhead.setdefault("calibration", {})
    original_anchors = calibration.get("anchors", [])
    if not isinstance(original_anchors, list):
        raise ValueError("Map config has no calibration anchors")
    corrected_anchors = corrected_anchors_from_touch_state(original_anchors, touch_state)
    affine = solve_affine_from_anchor_points(corrected_anchors)
    if affine is None:
        raise ValueError("Failed to solve corrected affine")
    calibration["anchors"] = corrected_anchors
    calibration["affine_world_to_image"] = affine
    calibration["rmse_px"] = affine_rmse_px(corrected_anchors, affine)
    calibration["updated_at"] = time.time()
    calibration["touch_calibration_status"] = str(touch_state.get("status", ""))
    if "image_width" not in calibration:
        calibration["image_width"] = int(overhead.get("image_width", 0) or 0)
    if "image_height" not in calibration:
        calibration["image_height"] = int(overhead.get("image_height", 0) or 0)
    corrected["houses"] = rebuild_houses_for_corrected_affine(corrected, affine)
    corrected["map_touch_calibration"] = {
        "completed_at": touch_state.get("completed_at"),
        "completed_count": touch_state.get("completed_count"),
        "marker_z": touch_state.get("marker_z"),
        "points": touch_state.get("points", []),
    }
    return corrected


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
        self.map_status_var = tk.StringVar(value="Map: closed")
        self.map_pose_var = tk.StringVar(value="Map pose: n/a")
        self.map_calibration_var = tk.StringVar(value="Calibration: idle")
        self.xy_tolerance_var = tk.StringVar(value="60")
        self.z_tolerance_var = tk.StringVar(value="80")
        self.marker_class_var = tk.StringVar(value=flight.DEFAULT_CALIBRATION_MARKER_CLASS)
        self.marker_scale_var = tk.StringVar(
            value=" ".join(str(value) for value in flight.DEFAULT_CALIBRATION_MARKER_SCALE)
        )

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
        self.enhance_rgb_var = tk.BooleanVar(value=bool(args.enhance_rgb))
        self.show_houses_var = tk.BooleanVar(value=True)
        self.show_trajectory_var = tk.BooleanVar(value=True)

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
        self.map_window: Optional[tk.Toplevel] = None
        self.map_widget: Optional[OverheadMapWidget] = None
        self.map_refresh_inflight = False
        self.map_config: Dict[str, Any] = {}
        self.map_config_path: Optional[Path] = None
        self.map_image_path: Optional[Path] = None
        self.map_image: Optional[np.ndarray] = None
        self.map_calibration: Dict[str, Any] = {}
        self.map_world_bounds: Tuple[float, float, float, float] = DEFAULT_MAP_BOUNDS
        self.map_touch_state: Dict[str, Any] = {}
        self.map_touch_poll_inflight = False
        self.map_touch_auto_saved = False
        self.main_canvas: Optional[tk.Canvas] = None
        self.content_frame: Optional[tk.Frame] = None
        self.content_window: Optional[int] = None

        self._build_ui()
        self._bind_hotkeys()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)
        self.root.after(self.args.map_interval_ms, self.schedule_map_refresh)

    def _on_content_frame_configure(self, _event: tk.Event) -> None:
        if self.main_canvas is None:
            return
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event: tk.Event) -> None:
        if self.main_canvas is None or self.content_window is None or self.content_frame is None:
            return
        requested_width = max(self.content_frame.winfo_reqwidth(), int(event.width))
        self.main_canvas.itemconfigure(self.content_window, width=requested_width)
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.main_canvas is None:
            return
        delta = -1 if event.delta > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            self.main_canvas.xview_scroll(delta, "units")
        else:
            self.main_canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        if self.main_canvas is None:
            return
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        self.main_canvas.yview_scroll(direction, "units")

    def _build_ui(self) -> None:
        self.main_canvas = tk.Canvas(self.root, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        h_scrollbar = tk.Scrollbar(self.root, orient="horizontal", command=self.main_canvas.xview)
        self.main_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        outer = tk.Frame(self.main_canvas)
        self.content_frame = outer
        self.content_window = self.main_canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", self._on_content_frame_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self._on_mousewheel_linux)
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
        tk.Checkbutton(preview, text="Enhance RGB", variable=self.enhance_rgb_var).pack(side="left", padx=6, pady=6)
        tk.Button(preview, text="Refresh State", command=self.refresh_state_once).pack(side="left", padx=6, pady=6)

        map_frame = tk.LabelFrame(outer, text="Map")
        map_frame.grid(row=6, column=0, sticky="ew", padx=8, pady=4)
        map_frame.grid_columnconfigure(10, weight=1)
        tk.Button(map_frame, text="Open Map", command=self.toggle_map_window).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(map_frame, text="Refresh Map", command=lambda: self.refresh_map_once(force_reload=True)).grid(row=0, column=1, padx=6, pady=6)
        tk.Checkbutton(map_frame, text="Show Houses", variable=self.show_houses_var, command=self.refresh_map_once).grid(row=0, column=2, padx=6, pady=6)
        tk.Checkbutton(map_frame, text="Show Trajectory", variable=self.show_trajectory_var, command=self.refresh_map_once).grid(row=0, column=3, padx=6, pady=6)
        tk.Label(map_frame, textvariable=self.map_status_var, anchor="w").grid(row=0, column=4, columnspan=7, sticky="ew", padx=6, pady=6)
        tk.Label(map_frame, textvariable=self.map_pose_var, anchor="w").grid(row=1, column=0, columnspan=11, sticky="ew", padx=6, pady=(0, 4))
        tk.Button(map_frame, text="Start P Calibration", command=self.on_start_map_touch_calibration).grid(row=2, column=0, padx=6, pady=6)
        tk.Button(map_frame, text="Stop/Clear", command=self.on_stop_map_touch_calibration).grid(row=2, column=1, padx=6, pady=6)
        tk.Button(map_frame, text="Reset Marker", command=self.on_reset_map_touch_markers).grid(row=2, column=2, padx=6, pady=6)
        tk.Button(map_frame, text="Save Corrected Config", command=self.on_save_corrected_map_config).grid(row=2, column=3, padx=6, pady=6)
        tk.Label(map_frame, text="XY Tol cm").grid(row=2, column=4, padx=(12, 2), pady=6)
        tk.Entry(map_frame, textvariable=self.xy_tolerance_var, width=7).grid(row=2, column=5, padx=(0, 8), pady=6)
        tk.Label(map_frame, text="Z Tol cm").grid(row=2, column=6, padx=(6, 2), pady=6)
        tk.Entry(map_frame, textvariable=self.z_tolerance_var, width=7).grid(row=2, column=7, padx=(0, 8), pady=6)
        tk.Label(map_frame, text="Marker Class").grid(row=3, column=0, padx=(6, 2), pady=(0, 6))
        ttk.Combobox(
            map_frame,
            textvariable=self.marker_class_var,
            values=("BP_drone01_C", "target_C", "bp_character_C", "Cube_C"),
            state="readonly",
            width=14,
        ).grid(row=3, column=1, padx=(0, 8), pady=(0, 6), sticky="w")
        tk.Label(map_frame, text="Marker Scale").grid(row=3, column=2, padx=(6, 2), pady=(0, 6))
        tk.Entry(map_frame, textvariable=self.marker_scale_var, width=7).grid(row=3, column=3, padx=(0, 8), pady=(0, 6))
        tk.Label(map_frame, textvariable=self.map_calibration_var, anchor="w").grid(row=3, column=4, columnspan=7, sticky="ew", padx=6, pady=(0, 6))

        orbit = tk.LabelFrame(outer, text="Orbit Plan")
        orbit.grid(row=7, column=0, sticky="ew", padx=8, pady=(4, 8))
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
            enhance_rgb=bool(self.enhance_rgb_var.get()),
            rgb_enhance_gamma=float(self.args.rgb_enhance_gamma),
            rgb_enhance_gain=float(self.args.rgb_enhance_gain),
            force_kill_unreal_on_stop=bool(self.args.force_kill_unreal_on_stop),
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
            try:
                kill_args = self.build_flight_args()
                flight.configure_local_unreal_env(kill_args)
                flight.force_kill_unreal_processes(kill_args)
                self.status_var.set("No active session; Unreal processes killed.")
            except Exception as exc:
                self.status_var.set(f"No active session; kill failed: {exc}")
            return

        def worker() -> Dict[str, Any]:
            session.close(force_kill_unreal=True)
            self.session = None
            return {"status": "ok", "message": "Session stopped; Unreal processes killed", "started": False}

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
        touch_state = self.latest_state.get("map_touch_calibration")
        if isinstance(touch_state, dict):
            self.apply_map_touch_state(touch_state, refresh=False)
        self.refresh_map_once()

    def _fmt_float(self, value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return str(value)

    def resolve_project_path(self, value: str, *, base_dir: Optional[Path] = None) -> Path:
        raw = str(value or "").strip()
        if not raw:
            return Path()
        path = Path(raw)
        if path.is_absolute():
            return path
        base = base_dir if base_dir is not None else PROJECT_ROOT
        return (base / path).resolve()

    def coerce_map_anchors(self, anchors: Any) -> List[Dict[str, float]]:
        if not isinstance(anchors, list):
            return []
        restored: List[Dict[str, float]] = []
        for index, anchor in enumerate(anchors[:5], start=1):
            if not isinstance(anchor, dict):
                continue
            try:
                restored.append({
                    "index": float(anchor.get("index", index)),
                    "label": str(anchor.get("label", f"P{index}")),
                    "world_x": float(anchor["world_x"]),
                    "world_y": float(anchor["world_y"]),
                    "image_x": float(anchor["image_x"]),
                    "image_y": float(anchor["image_y"]),
                })
            except Exception:
                continue
        return restored

    def solve_affine_from_anchors(self, anchors: List[Dict[str, float]]) -> Optional[List[List[float]]]:
        return solve_affine_from_anchor_points(anchors)

    def normalize_map_calibration(self, payload: Any) -> Dict[str, Any]:
        calibration = payload if isinstance(payload, dict) else {}
        anchors = self.coerce_map_anchors(calibration.get("anchors", []))
        affine = calibration.get("affine_world_to_image")
        if not (isinstance(affine, list) and len(affine) == 2):
            affine = self.solve_affine_from_anchors(anchors)
        normalized: Dict[str, Any] = {"anchors": anchors}
        if isinstance(affine, list) and len(affine) == 2:
            normalized["affine_world_to_image"] = affine
        for key in ("image_width", "image_height"):
            if calibration.get(key) is not None:
                try:
                    normalized[key] = int(calibration.get(key))
                except Exception:
                    pass
        if calibration.get("rmse_px") is not None:
            try:
                normalized["rmse_px"] = float(calibration.get("rmse_px"))
            except Exception:
                pass
        return normalized

    def load_map_resources(self, *, force: bool = False) -> bool:
        config_path = self.resolve_project_path(str(self.args.map_config or DEFAULT_MAP_CONFIG_PATH))
        if not force and self.map_config and self.map_config_path == config_path and self.map_image is not None:
            return True
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except Exception as exc:
            self.map_status_var.set(f"Map: failed to load config ({exc})")
            return False

        world_bounds = config.get("world_bounds", {}) if isinstance(config.get("world_bounds"), dict) else {}
        self.map_world_bounds = (
            float(world_bounds.get("min_x", DEFAULT_MAP_BOUNDS[0])),
            float(world_bounds.get("min_y", DEFAULT_MAP_BOUNDS[1])),
            float(world_bounds.get("max_x", DEFAULT_MAP_BOUNDS[2])),
            float(world_bounds.get("max_y", DEFAULT_MAP_BOUNDS[3])),
        )
        overhead = config.get("overhead_map", {}) if isinstance(config.get("overhead_map"), dict) else {}
        image_value = str(self.args.map_image or "").strip() or str(overhead.get("image_path", "") or "qq.png")
        image_path = self.resolve_project_path(image_value, base_dir=config_path.parent)
        image = cv2.imread(str(image_path))
        if image is None:
            self.map_status_var.set(f"Map: failed to load image {image_path}")
            return False

        self.map_config = config
        self.map_config_path = config_path
        self.map_image_path = image_path
        self.map_image = image
        self.map_calibration = self.normalize_map_calibration(overhead.get("calibration", {}))
        self.map_status_var.set(f"Map: loaded {image_path.name}")
        return True

    def map_image_size(self) -> Optional[Tuple[int, int]]:
        if self.map_image is not None:
            return int(self.map_image.shape[1]), int(self.map_image.shape[0])
        width = self.map_calibration.get("image_width")
        height = self.map_calibration.get("image_height")
        if width and height:
            return int(width), int(height)
        return None

    def world_to_image_point(self, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        affine = self.map_calibration.get("affine_world_to_image")
        if not isinstance(affine, list) or len(affine) != 2:
            return None
        try:
            return world_to_image_with_affine(world_x, world_y, affine)
        except Exception:
            return None

    def find_containing_house_id(self, x: float, y: float, houses: List[Dict[str, Any]]) -> str:
        for house in houses:
            try:
                cx = float(house.get("center_x", 0.0))
                cy = float(house.get("center_y", 0.0))
                radius = float(house.get("radius_cm", 0.0))
            except Exception:
                continue
            if radius > 0.0 and float(np.hypot(float(x) - cx, float(y) - cy)) <= radius:
                return str(house.get("id", "") or "")
        return ""

    def build_map_display(self, pose: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        houses_raw = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        target_id = str(self.map_config.get("current_target_id", "") or "")
        try:
            pose_x = float(pose.get("x", 0.0))
            pose_y = float(pose.get("y", 0.0))
        except Exception:
            pose_x = 0.0
            pose_y = 0.0
        current_id = self.find_containing_house_id(pose_x, pose_y, houses_raw)
        houses: List[Dict[str, Any]] = []
        boxes: List[Dict[str, Any]] = []
        for house in houses_raw:
            if not isinstance(house, dict):
                continue
            hid = str(house.get("id", "") or "")
            if not hid:
                continue
            name = str(house.get("name", hid) or hid)
            is_target = hid == target_id
            is_current = hid == current_id
            try:
                houses.append({
                    "id": hid,
                    "name": f"{name} (UAV)" if is_current else name,
                    "center_x": float(house.get("center_x", 0.0)),
                    "center_y": float(house.get("center_y", 0.0)),
                    "radius_cm": float(house.get("radius_cm", 600.0)),
                    "status": str(house.get("status", "UNSEARCHED") or "UNSEARCHED"),
                    "is_target": is_target,
                    "is_current": is_current,
                })
            except Exception:
                continue
            bbox = house.get("map_bbox_image")
            if isinstance(bbox, dict):
                try:
                    boxes.append({
                        "id": hid,
                        "name": name,
                        "status": str(house.get("status", "UNSEARCHED") or "UNSEARCHED"),
                        "is_target": is_target,
                        "is_current": is_current,
                        "map_bbox_image": {
                            "x1": float(bbox["x1"]),
                            "y1": float(bbox["y1"]),
                            "x2": float(bbox["x2"]),
                            "y2": float(bbox["y2"]),
                        },
                    })
                except Exception:
                    continue
        return houses, boxes

    def map_touch_anchors_for_session(self) -> List[Dict[str, Any]]:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return []
        anchors = self.map_calibration.get("anchors", [])
        if not isinstance(anchors, list):
            self.status_var.set("Map calibration anchors are missing.")
            return []
        normalized = sorted(
            [dict(anchor) for anchor in anchors if isinstance(anchor, dict)],
            key=lambda anchor: float(anchor.get("index", 9999)),
        )[:5]
        if len(normalized) < 5:
            self.status_var.set("P1-P5 anchors are required before calibration.")
            return []
        return normalized

    def map_touch_tolerances(self) -> Tuple[float, float]:
        try:
            xy_tol = max(0.0, float(self.xy_tolerance_var.get().strip()))
            z_tol = max(0.0, float(self.z_tolerance_var.get().strip()))
            return xy_tol, z_tol
        except ValueError:
            self.status_var.set("Invalid map calibration tolerance.")
            return 60.0, 80.0

    def call_map_touch_async(self, desc: str, fn) -> None:
        if self.manual_request_inflight:
            self.status_var.set(f"{desc} skipped while another request is running.")
            return

        def worker() -> None:
            self.manual_request_inflight = True
            self.root.after(0, lambda: self.status_var.set(f"{desc}..."))
            try:
                result = self.safe(desc, fn)
                if isinstance(result, dict):
                    self.root.after(0, lambda r=result: self.apply_map_touch_state(r))
            finally:
                self.manual_request_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def format_map_touch_state(self, state: Dict[str, Any]) -> str:
        status = str(state.get("status", "idle") or "idle")
        active = str(state.get("active_point", "") or "")
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        completed = int(state.get("completed_count", 0) or 0)
        total = len(points) if points else 5
        dx = state.get("distance_xy_cm")
        dz = state.get("distance_z_cm")
        parts = [f"Calibration: {status}", f"completed={completed}/{total}"]
        if active:
            parts.append(f"active={active}")
            active_point = next(
                (point for point in points if isinstance(point, dict) and str(point.get("label", "")) == active),
                None,
            )
            if isinstance(active_point, dict):
                try:
                    parts.append(
                        "target="
                        f"({float(active_point.get('target_world_x', 0.0)):.1f},"
                        f"{float(active_point.get('target_world_y', 0.0)):.1f},"
                        f"{float(active_point.get('target_world_z', 0.0)):.1f})"
                    )
                    marker_class = str(active_point.get("marker_class", "") or state.get("marker_class_preference", ""))
                    marker_scale = active_point.get("marker_scale", state.get("marker_scale"))
                    if marker_class:
                        parts.append(f"marker={marker_class}")
                    if isinstance(marker_scale, list):
                        parts.append("scale=" + ",".join(self._fmt_float(value) for value in marker_scale[:3]))
                except Exception:
                    pass
        if dx is not None:
            parts.append(f"dx={self._fmt_float(dx)}cm")
        if dz is not None:
            parts.append(f"dz={self._fmt_float(dz)}cm")
        saved = state.get("saved_corrected_config")
        if saved:
            parts.append(f"saved={Path(str(saved)).name}")
        return " ".join(parts)

    def apply_map_touch_state(self, state: Dict[str, Any], *, refresh: bool = True) -> None:
        previous_saved = self.map_touch_state.get("saved_corrected_config") if isinstance(self.map_touch_state, dict) else None
        self.map_touch_state = state if isinstance(state, dict) else {}
        if previous_saved and not self.map_touch_state.get("saved_corrected_config"):
            self.map_touch_state["saved_corrected_config"] = previous_saved
        self.map_calibration_var.set(self.format_map_touch_state(self.map_touch_state))
        if self.map_touch_state.get("status") == "running":
            self.map_touch_auto_saved = False
        if self.map_touch_state.get("status") == "complete" and not self.map_touch_auto_saved:
            self.map_touch_auto_saved = True
            self.save_corrected_map_config(auto=True)
            return
        if refresh:
            self.refresh_map_once()

    def anchors_with_touch_status(self, anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        state = self.map_touch_state if isinstance(self.map_touch_state, dict) else {}
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        status_by_label = {
            str(point.get("label", "")): str(point.get("status", "") or "")
            for point in points
            if isinstance(point, dict)
        }
        enriched: List[Dict[str, Any]] = []
        for anchor in anchors:
            item = dict(anchor)
            label = str(item.get("label", item.get("index", "")))
            if label in status_by_label:
                item["status"] = status_by_label[label]
            enriched.append(item)
        return enriched

    def poll_map_touch_calibration_once(self) -> None:
        session = self.session
        if session is None or not session.started or self.map_touch_poll_inflight:
            return
        state = self.map_touch_state if isinstance(self.map_touch_state, dict) else {}
        if not bool(state.get("running", False)):
            return
        xy_tol, z_tol = self.map_touch_tolerances()

        def worker() -> None:
            self.map_touch_poll_inflight = True
            try:
                result = self.safe(
                    "Polling map calibration",
                    lambda: session.poll_map_touch_calibration(xy_tolerance_cm=xy_tol, z_tolerance_cm=z_tol),
                )
                if isinstance(result, dict):
                    self.root.after(0, lambda r=result: self.apply_map_touch_state(r))
            finally:
                self.map_touch_poll_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def on_start_map_touch_calibration(self) -> None:
        session = self.active_session()
        if session is None:
            return
        anchors = self.map_touch_anchors_for_session()
        if not anchors:
            return
        marker_class = self.marker_class_var.get().strip() or flight.DEFAULT_CALIBRATION_MARKER_CLASS
        marker_scale = self.marker_scale_var.get().strip() or str(flight.DEFAULT_CALIBRATION_MARKER_SCALE[0])
        self.map_touch_auto_saved = False
        self.call_map_touch_async(
            "Starting P calibration",
            lambda: session.start_map_touch_calibration(
                anchors,
                marker_class=marker_class,
                marker_scale=marker_scale,
            ),
        )

    def on_stop_map_touch_calibration(self) -> None:
        session = self.active_session()
        if session is None:
            self.map_touch_state = {}
            self.map_calibration_var.set("Calibration: idle")
            self.refresh_map_once()
            return
        self.map_touch_auto_saved = False
        self.call_map_touch_async(
            "Stopping P calibration",
            lambda: session.stop_map_touch_calibration(cleanup=True),
        )

    def on_reset_map_touch_markers(self) -> None:
        session = self.active_session()
        if session is None:
            self.map_touch_state = {}
            self.map_calibration_var.set("Calibration: idle")
            self.refresh_map_once()
            return
        self.map_touch_auto_saved = False
        self.call_map_touch_async(
            "Resetting collision marker",
            session.reset_map_touch_calibration_markers,
        )

    def on_save_corrected_map_config(self) -> None:
        self.save_corrected_map_config(auto=False)

    def save_corrected_map_config(self, *, auto: bool = False) -> None:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return
        state = self.map_touch_state if isinstance(self.map_touch_state, dict) else {}
        session = self.session
        if (not state or state.get("status") not in {"complete", "stopped"}) and session is not None:
            try:
                state = session.get_map_touch_calibration_state()
            except Exception:
                pass
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        completed = len([point for point in points if isinstance(point, dict) and point.get("status") == "done"])
        if completed < 5:
            self.status_var.set("Need completed P1-P5 contacts before saving corrected config.")
            return
        try:
            corrected = build_corrected_map_config(self.map_config, state)
            base_dir = self.map_config_path.parent if self.map_config_path is not None else PROJECT_ROOT / "assets" / "overhead_map"
            output_path = (base_dir / DEFAULT_CORRECTED_MAP_CONFIG_NAME).resolve()
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(corrected, fh, indent=2, ensure_ascii=False)

            log_path = None
            run_dir_value = str(self.latest_state.get("run_dir", "") or "")
            if not run_dir_value and session is not None and session.run_dir is not None:
                run_dir_value = str(session.run_dir)
            if run_dir_value:
                run_dir = Path(run_dir_value)
                if run_dir.exists():
                    log_path = run_dir / "map_touch_calibration.json"
                    with open(log_path, "w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "touch_state": state,
                                "corrected_config": str(output_path),
                                "rmse_px": corrected.get("overhead_map", {}).get("calibration", {}).get("rmse_px"),
                                "saved_at": time.time(),
                            },
                            fh,
                            indent=2,
                            ensure_ascii=False,
                        )

            self.args.map_config = str(output_path)
            self.map_config = {}
            self.map_config_path = None
            self.map_image_path = None
            self.map_image = None
            self.load_map_resources(force=True)
            self.map_touch_state = dict(state)
            self.map_touch_state["saved_corrected_config"] = str(output_path)
            self.map_calibration_var.set(self.format_map_touch_state(self.map_touch_state))
            prefix = "Auto saved" if auto else "Saved"
            suffix = f"; log={log_path.name}" if log_path is not None else ""
            self.status_var.set(f"{prefix} corrected map config: {output_path.name}{suffix}")
            self.refresh_map_once(force_reload=True)
        except Exception as exc:
            LOGGER.warning("Save corrected map config failed: %s", exc)
            self.status_var.set(f"Save corrected map config failed: {exc}")

    def toggle_map_window(self) -> None:
        if self.map_window is not None and self.map_window.winfo_exists():
            self.close_map_window()
            return
        self.load_map_resources(force=True)
        self.map_window = tk.Toplevel(self.root)
        self.map_window.title("Overhead Map - UAV Pose")
        self.map_window.resizable(False, False)
        self.map_window.protocol("WM_DELETE_WINDOW", self.close_map_window)
        toolbar = tk.Frame(self.map_window)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(toolbar, textvariable=self.map_pose_var, anchor="w").pack(side="left", padx=(0, 12))
        tk.Label(toolbar, textvariable=self.map_status_var, anchor="w").pack(side="left")
        self.map_widget = OverheadMapWidget(self.map_window, world_bounds=self.map_world_bounds, canvas_w=900, canvas_h=480)
        self.map_widget.canvas.pack(padx=8, pady=8)
        self.refresh_map_once(force_reload=False)

    def close_map_window(self) -> None:
        try:
            if self.map_window is not None and self.map_window.winfo_exists():
                self.map_window.destroy()
        except Exception:
            pass
        self.map_window = None
        self.map_widget = None
        self.map_status_var.set("Map: closed")

    def refresh_map_once(self, force_reload: bool = False) -> None:
        if self.map_refresh_inflight:
            return
        if self.map_widget is None or self.map_window is None or not self.map_window.winfo_exists():
            if force_reload:
                self.map_status_var.set("Map: open map first")
            return
        self.map_refresh_inflight = True
        try:
            if not self.load_map_resources(force=force_reload):
                return
            pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
            pose_x = float(pose.get("x", 0.0)) if pose else 0.0
            pose_y = float(pose.get("y", 0.0)) if pose else 0.0
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0))) if pose else 0.0
            image_point = self.world_to_image_point(pose_x, pose_y)
            if image_point is None:
                self.map_pose_var.set(f"Map pose: world=({pose_x:.1f}, {pose_y:.1f}) yaw={pose_yaw:.1f} image=n/a")
            else:
                self.map_pose_var.set(
                    f"Map pose: world=({pose_x:.1f}, {pose_y:.1f}) "
                    f"image=({image_point[0]:.1f}, {image_point[1]:.1f}) yaw={pose_yaw:.1f}"
                )

            houses, boxes = self.build_map_display(pose)
            calibration = self.map_calibration
            affine = calibration.get("affine_world_to_image")
            anchors = calibration.get("anchors", []) if isinstance(calibration.get("anchors", []), list) else []
            anchors = self.anchors_with_touch_status(anchors)
            self.map_widget.set_background_image(self.map_image)
            self.map_widget.set_calibration(affine, self.map_image_size(), anchors)
            self.map_widget.set_house_boxes(boxes if self.show_houses_var.get() else [])
            self.map_widget.update_houses([])
            self.map_widget.update_uav(pose_x, pose_y, pose_yaw)
            trajectory: List[Dict[str, float]] = []
            session = self.session
            if self.show_trajectory_var.get() and session is not None:
                trajectory = session.get_trajectory_points(limit=max(1, int(self.args.map_trajectory_limit)))
            self.map_widget.set_trajectory(trajectory)
        finally:
            self.map_refresh_inflight = False

    def schedule_map_refresh(self) -> None:
        self.poll_map_touch_calibration_once()
        if self.map_widget is not None and self.map_window is not None and self.map_window.winfo_exists():
            self.refresh_map_once()
        self.root.after(self.args.map_interval_ms, self.schedule_map_refresh)

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
        session.args.enhance_rgb = bool(self.enhance_rgb_var.get())
        session.args.rgb_enhance_gamma = float(self.args.rgb_enhance_gamma)
        session.args.rgb_enhance_gain = float(self.args.rgb_enhance_gain)
        self.call_async("Saving RGB frame", lambda: (session.capture_observation(save=True, label="manual"), session.get_state())[1])

    def image_to_photo(self, image: np.ndarray, max_width: int = 900, max_height: int = 620) -> ImageTk.PhotoImage:
        array = flight.prepare_observation_rgb(
            image,
            enhance=bool(self.enhance_rgb_var.get()),
            gamma=float(self.args.rgb_enhance_gamma),
            gain=float(self.args.rgb_enhance_gain),
        )
        if array is None:
            array = np.zeros((1, 1, 3), dtype=np.uint8)
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
                session.close(force_kill_unreal=True)
            except Exception as exc:
                LOGGER.warning("Failed to close session: %s", exc)
        self.close_map_window()
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
    parser.add_argument("--enhance_rgb", dest="enhance_rgb", action="store_true", default=flight.DEFAULT_RGB_ENHANCE_ENABLED)
    parser.add_argument("--no_enhance_rgb", dest="enhance_rgb", action="store_false")
    parser.add_argument("--rgb_enhance_gamma", type=float, default=flight.DEFAULT_RGB_ENHANCE_GAMMA)
    parser.add_argument("--rgb_enhance_gain", type=float, default=flight.DEFAULT_RGB_ENHANCE_GAIN)
    parser.add_argument("--force_kill_unreal_on_stop", dest="force_kill_unreal_on_stop", action="store_true",
                        default=flight.DEFAULT_FORCE_KILL_UNREAL_ON_STOP)
    parser.add_argument("--no_force_kill_unreal_on_stop", dest="force_kill_unreal_on_stop", action="store_false")
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
    parser.add_argument("--map_config", default=DEFAULT_MAP_CONFIG_PATH)
    parser.add_argument("--map_image", default="")
    parser.add_argument("--map_interval_ms", type=int, default=1000)
    parser.add_argument("--map_trajectory_limit", type=int, default=500)
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
