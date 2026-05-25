from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_drone_flight as flight

from control.constants import DEFAULT_BASE_MAP_CONFIG_PATH, DEFAULT_KEYBOARD_INTERVAL_MS, DEFAULT_MAP_CONFIG_PATH, DEFAULT_SETTING_MAP_CONFIG_NAME
from control.utils import default_llm_api_style
from control.route6_explore_control import Route6ExploreControlMixin
from control import route6_map_builder


class ValueVar:
    def __init__(self, value: Any = "") -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_default_map_config() -> Dict[str, Any]:
    candidates = [
        PROJECT_ROOT / DEFAULT_MAP_CONFIG_PATH,
        PROJECT_ROOT / "assets" / "overhead_map" / DEFAULT_SETTING_MAP_CONFIG_NAME,
        PROJECT_ROOT / DEFAULT_BASE_MAP_CONFIG_PATH,
    ]
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    raise FileNotFoundError(f"No Route 6 map config found in: {', '.join(str(path) for path in candidates)}")


def route6_smoke_state_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    house_states = state.get("house_states", {}) if isinstance(state.get("house_states", {}), dict) else {}
    mapped_statuses = {"mapped_complete", "mapped_partial", "searched", "searched_no_entry"}
    searched_statuses = {"searched", "searched_no_entry"}
    mapped_house_ids: List[str] = []
    searched_house_ids: List[str] = []
    needs_capture_house_ids: List[str] = []
    blocked_house_ids: List[str] = []
    valid_scan_capture_count_total = 0
    merged_point_count_total = 0
    for hid, raw_state in house_states.items():
        item = raw_state if isinstance(raw_state, dict) else {}
        status = str(item.get("status", "") or "")
        if status in mapped_statuses:
            mapped_house_ids.append(str(hid))
        if status in searched_statuses:
            searched_house_ids.append(str(hid))
        if status in {"needs_capture", "needs_rescan"}:
            needs_capture_house_ids.append(str(hid))
        if status in {"blocked", "terminal_blocked", "map_conflict"}:
            blocked_house_ids.append(str(hid))
        try:
            valid_scan_capture_count_total += int(item.get("valid_scan_capture_count", 0) or 0)
        except Exception:
            pass
        try:
            merged_point_count_total += int(item.get("merged_point_count", 0) or 0)
        except Exception:
            pass
    return {
        "mapped_house_ids": mapped_house_ids,
        "mapped_house_count": len(mapped_house_ids),
        "searched_house_ids": searched_house_ids,
        "searched_house_count": len(searched_house_ids),
        "needs_capture_house_ids": needs_capture_house_ids,
        "needs_capture_house_count": len(needs_capture_house_ids),
        "blocked_house_ids": blocked_house_ids,
        "blocked_house_count": len(blocked_house_ids),
        "valid_scan_capture_count_total": int(valid_scan_capture_count_total),
        "merged_point_count_total": int(merged_point_count_total),
    }


def set_panel_route6_var(panel: Any, name: str, value: Any) -> None:
    variable = getattr(panel, name, None)
    if hasattr(variable, "set"):
        variable.set(value)
        return
    setattr(panel, name, ValueVar(value))


def route6_optional_float_text(value: Optional[float], default_text: str) -> str:
    if value is None:
        return default_text
    return str(float(value))


class Route6SmokeHarness(Route6ExploreControlMixin):
    def __init__(
        self,
        *,
        output_root: Path,
        map_config: Dict[str, Any],
        max_houses: int,
        runtime_minutes: float,
        standoff_cm: Optional[float] = None,
        scan_z_cm: Optional[float] = None,
        initial_pose: Optional[Dict[str, float]] = None,
    ) -> None:
        self.args = SimpleNamespace(lidar_depth_max_cm=1200.0)
        self.map_config = map_config
        self.latest_state = {
            "pose": [
                float((initial_pose or {}).get("x", 0.0)),
                float((initial_pose or {}).get("y", 0.0)),
                float((initial_pose or {}).get("z", 450.0)),
                float((initial_pose or {}).get("yaw", 0.0)),
            ]
        }
        self.llm_route6_state: Dict[str, Any] = {}
        self.llm_route6_window = None
        self.llm_route6_summary_text = None
        self.llm_route6_map_widget = None
        self.llm_route6_map_frame = None
        self.llm_route6_thread = None
        self.llm_route6_stop_event = threading.Event()
        self.llm_route6_pause_event = threading.Event()
        self.llm_route6_force_next_event = threading.Event()
        self.llm_route6_status_var = ValueVar("LLM Route V6: idle")
        self.llm_route6_stage_var = ValueVar("Stage: idle")
        self.llm_route6_current_house_var = ValueVar("Current house: n/a")
        self.llm_route6_queue_var = ValueVar("House queue: n/a")
        self.llm_route6_map_status_var = ValueVar("Map: idle")
        self.llm_route6_output_dir_var = ValueVar("Output: n/a")
        self.llm_route6_metrics_var = ValueVar("Metrics: mapped=0 searched=0 blocked=0 confidence=n/a corrected=n/a")
        self.llm_route6_max_houses_var = ValueVar(str(max(1, int(max_houses))))
        self.llm_route6_runtime_min_var = ValueVar(str(float(runtime_minutes)))
        self.llm_route6_standoff_cm_var = ValueVar(route6_optional_float_text(standoff_cm, "850"))
        self.llm_route6_scan_z_cm_var = ValueVar(route6_optional_float_text(scan_z_cm, "450"))
        self.llm_route6_occupancy_resolution_m_var = ValueVar("0.25")
        self.llm_route6_coverage_threshold_var = ValueVar("0.75")
        self.llm_route6_allow_save_corrected_var = ValueVar(False)
        self.llm_route6_output_root_override = Path(output_root)
        self.active_nbv_execute_calls: List[Dict[str, Any]] = []

    def resolve_project_path(self, value: str, *, base_dir: Optional[Path] = None) -> Path:
        path = Path(str(value or ""))
        if path.is_absolute():
            return path
        return ((base_dir or PROJECT_ROOT) / path).resolve()

    def route3_current_pose(self, _session: Any = None) -> Dict[str, float]:
        pose = self.latest_state.get("pose", [0.0, 0.0, 450.0, 0.0])
        return {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "yaw": float(pose[3] if len(pose) > 3 else 0.0),
        }

    def write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_jsonl_artifact(self, path: Path) -> List[Dict[str, Any]]:
        return read_jsonl(Path(path))

    def active_nbv_initial_scan_points(self, house_id: str) -> List[Dict[str, Any]]:
        hid = str(house_id or "").strip()
        bbox = self._house_bbox(hid)
        pose = route6_map_builder.facade_scan_pose_for_bbox(
            bbox,
            "west",
            current_pose=self.route6_current_pose(),
            standoff_cm=float(self.llm_route6_standoff_cm_var.get()),
            scan_z_cm=float(self.llm_route6_scan_z_cm_var.get()),
        )
        return [
            {
                **pose,
                "scan_id": f"{hid}_west_route6_smoke_000",
                "house_id": hid,
                "facade": "west",
                "view_type": "route6_offline_smoke_scan",
                "capture_trigger": "offline_mock_pointcloud",
                "status": "planned",
            }
        ]

    def active_nbv_execute_scan_points(
        self,
        _session: Any,
        output_dir: Path,
        target_house_id: str,
        points: List[Dict[str, Any]],
        *,
        round_index: int,
        all_points: List[Dict[str, Any]],
    ) -> None:
        hid = str(target_house_id or "").strip()
        scan_id = str((points[0] if points else {}).get("scan_id", f"{hid}_west_route6_smoke_000"))
        cloud = self._mock_house_cloud(hid)
        cloud_path = Path(output_dir) / f"route6_smoke_house_{hid}_round_{round_index}.npy"
        np.save(cloud_path, cloud.astype(np.float32, copy=False))
        for point in points:
            if isinstance(point, dict):
                point["status"] = "captured"
                point["point_count"] = int(cloud.shape[0])
        row = {
            "scan_id": scan_id,
            "house_id": hid,
            "facade": "west",
            "capture_guard_passed": True,
            "point_count": int(cloud.shape[0]),
            "point_cloud_world_standard_m_npy_path": str(cloud_path),
            "capture_kind": "route6_offline_smoke",
            "capture_status": "ok",
            "view_type": "route6_offline_smoke_scan",
        }
        self.append_jsonl(Path(output_dir) / "lidar_capture_log.jsonl", row)
        self.active_nbv_execute_calls.append(
            {
                "target_house_id": hid,
                "round_index": int(round_index),
                "scan_ids": [str(point.get("scan_id", "")) for point in points if isinstance(point, dict)],
                "point_count": int(cloud.shape[0]),
                "cloud_path": str(cloud_path),
            }
        )

    def _house_bbox(self, house_id: str) -> Dict[str, float]:
        houses = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        house = next(
            (item for item in houses if isinstance(item, dict) and str(item.get("id", item.get("house_id", ""))) == str(house_id)),
            None,
        )
        bbox = route6_map_builder.house_world_bbox(self.map_config, house) if isinstance(house, dict) else {}
        if bbox:
            return bbox
        return {"min_x": 100.0, "max_x": 250.0, "min_y": -100.0, "max_y": 100.0}

    def _mock_house_cloud(self, house_id: str) -> np.ndarray:
        bbox = self._house_bbox(house_id)
        min_x = float(bbox.get("min_x", 100.0)) / 100.0
        max_x = float(bbox.get("max_x", 250.0)) / 100.0
        min_y = float(bbox.get("min_y", -100.0)) / 100.0
        max_y = float(bbox.get("max_y", 100.0)) / 100.0
        xs = np.linspace(min_x, max_x, 12)
        ys = np.linspace(min_y, max_y, 12)
        points: List[List[float]] = []
        for x in xs:
            for y in ys:
                points.append([float(x), float(y), 1.2, 120.0, 160.0, 220.0])
        return np.asarray(points, dtype=np.float32)


def run_route6_smoke(
    *,
    mode: str = "offline_mock",
    output_root: Optional[Path] = None,
    max_houses: int = 1,
    runtime_minutes: float = 5.0,
    max_scan_points: int = 0,
    standoff_cm: Optional[float] = None,
    scan_z_cm: Optional[float] = None,
    map_config: Optional[Dict[str, Any]] = None,
    initial_pose: Optional[Dict[str, float]] = None,
    panel_factory: Optional[Any] = None,
    start_session: bool = True,
    controller_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    if mode == "live_controller":
        return run_route6_live_controller_smoke(
            output_root=output_root,
            max_houses=max_houses,
            runtime_minutes=runtime_minutes,
            max_scan_points=max_scan_points,
            standoff_cm=standoff_cm,
            scan_z_cm=scan_z_cm,
            initial_pose=initial_pose,
            panel_factory=panel_factory,
            start_session=start_session,
            controller_args=controller_args,
        )
    if mode != "offline_mock":
        raise ValueError("Route 6 smoke runner supports mode='offline_mock' or mode='live_controller'.")
    root = Path(output_root or PROJECT_ROOT / "route6_smoke_runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = map_config if isinstance(map_config, dict) else load_default_map_config()
    harness = Route6SmokeHarness(
        output_root=root,
        map_config=config,
        max_houses=max_houses,
        runtime_minutes=runtime_minutes,
        standoff_cm=standoff_cm,
        scan_z_cm=scan_z_cm,
        initial_pose=initial_pose,
    )
    harness.route6_full_explore_worker(session=object(), force_new=True)
    state = harness.llm_route6_state if isinstance(harness.llm_route6_state, dict) else {}
    run_dir = Path(str(state.get("output_dir", root)))
    worker_stage = str(state.get("stage", "") or "")
    processed = state.get("processed_house_ids", []) if isinstance(state.get("processed_house_ids", []), list) else []
    expected_houses = max(1, int(max_houses))
    metrics = route6_smoke_state_metrics(state)
    smoke_ok = (
        worker_stage == "DONE"
        and len(processed) >= expected_houses
        and int(metrics["mapped_house_count"]) > 0
        and int(metrics["valid_scan_capture_count_total"]) > 0
        and int(metrics["merged_point_count_total"]) > 0
    )
    route6_state_path = run_dir / "route6_state.json"
    summary_path = Path(str(state.get("exploration_summary_path", run_dir / "route6_exploration_summary.csv")))
    quality_path = run_dir / "map" / "route6_map_quality_report.json"
    result_path = run_dir / "route6_smoke_result.json"
    payload = {
        "schema": "route6_smoke_result_v1",
        "mode": mode,
        "status": "ok" if smoke_ok else "check",
        "worker_stage": worker_stage,
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "route6_state_path": str(route6_state_path),
        "route6_exploration_summary_path": str(summary_path),
        "route6_map_quality_report_path": str(quality_path),
        "processed_house_ids": [str(item) for item in processed],
        "processed_house_count": len(processed),
        "active_nbv_execute_call_count": len(harness.active_nbv_execute_calls),
        "active_nbv_execute_calls": harness.active_nbv_execute_calls,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(metrics)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def default_live_controller_args(
    *,
    output_root: Path,
    initial_pose: Optional[Dict[str, float]] = None,
    env_platform: str = "auto",
    env_root: Optional[str] = None,
    env_bin: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    launch_sleep: int = 15,
    time_dilation: int = 0,
    step_delay: float = 0.1,
    map_config_path: str = DEFAULT_MAP_CONFIG_PATH,
    lidar_capture_processing: str = "full",
) -> argparse.Namespace:
    pose = initial_pose or {}
    initial_pos = [
        float(pose.get("x", 0.0)),
        float(pose.get("y", 0.0)),
        float(pose.get("z", 100.0)),
        float(pose.get("yaw", 0.0)),
    ]
    args = flight.default_session_args(
        env_platform=env_platform,
        env_root=env_root,
        env_bin=env_bin,
        output_dir=str(Path(output_root) / "drone_session"),
        width=int(width),
        height=int(height),
        launch_sleep=int(launch_sleep),
        time_dilation=int(time_dilation),
        step_delay=float(step_delay),
        save_every=0,
        movement_mode=flight.DEFAULT_MOVEMENT_MODE,
        initial_pos=initial_pos,
        mode="keyboard",
        auto_action="none",
        max_steps=0,
        lidar_capture_processing=flight.normalize_lidar_capture_processing(lidar_capture_processing),
    )
    extras = {
        "keyboard_interval_ms": DEFAULT_KEYBOARD_INTERVAL_MS,
        "state_interval_ms": 1500,
        "preview_interval_ms": 1500,
        "map_config": map_config_path,
        "map_image": "",
        "map_interval_ms": 1000,
        "map_trajectory_limit": 500,
        "llm_api_style": default_llm_api_style(),
        "llm_base_url": "",
        "llm_api_key": "",
        "llm_model": "",
        "llm_route_timeout_s": 60.0,
        "route_step_cm": 120.0,
        "route_delay_ms": 100.0,
    }
    for key, value in extras.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def run_route6_live_controller_smoke(
    *,
    output_root: Optional[Path] = None,
    max_houses: int = 1,
    runtime_minutes: float = 5.0,
    max_scan_points: int = 0,
    standoff_cm: Optional[float] = None,
    scan_z_cm: Optional[float] = None,
    initial_pose: Optional[Dict[str, float]] = None,
    panel_factory: Optional[Any] = None,
    start_session: bool = True,
    controller_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    from control.panel import RunDroneFlightPanel

    root = Path(output_root or PROJECT_ROOT / "route6_live_smoke_runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    args = controller_args or default_live_controller_args(output_root=root, initial_pose=initial_pose)
    factory = panel_factory or RunDroneFlightPanel
    result_run_dir = root / f"route6_live_controller_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}"
    panel = None
    worker_called = False
    session_started = False
    start_state: Dict[str, Any] = {}
    error = ""
    try:
        panel = factory(args)
        if callable(getattr(panel, "ensure_route6_state", None)):
            panel.ensure_route6_state()
        setattr(panel, "llm_route6_output_root_override", root)
        set_panel_route6_var(panel, "llm_route6_max_houses_var", str(max(1, int(max_houses))))
        set_panel_route6_var(panel, "llm_route6_runtime_min_var", str(float(runtime_minutes)))
        if standoff_cm is not None:
            set_panel_route6_var(panel, "llm_route6_standoff_cm_var", str(float(standoff_cm)))
        if scan_z_cm is not None:
            set_panel_route6_var(panel, "llm_route6_scan_z_cm_var", str(float(scan_z_cm)))
        setattr(panel, "llm_route6_scan_point_limit_override", max(0, int(max_scan_points)))
        session = getattr(panel, "session", None)
        if start_session:
            session_args = panel.build_flight_args() if callable(getattr(panel, "build_flight_args", None)) else args
            session = flight.DroneFlightSession(session_args)
            panel.session = session
            start_state = session.start()
            session_started = bool(getattr(session, "started", False))
            if callable(getattr(panel, "apply_state", None)) and isinstance(start_state, dict):
                panel.apply_state(start_state)
        if session is None:
            raise RuntimeError("live_controller mode requires a started or injected controller session")
        worker_called = True
        panel.route6_full_explore_worker(session, force_new=True)
    except Exception as exc:
        error = str(exc)
    state = getattr(panel, "llm_route6_state", {}) if panel is not None and isinstance(getattr(panel, "llm_route6_state", {}), dict) else {}
    if state.get("output_dir"):
        result_run_dir = Path(str(state.get("output_dir")))
    result_run_dir.mkdir(parents=True, exist_ok=True)
    worker_stage = str(state.get("stage", "") or ("FAILED" if error else ""))
    processed = state.get("processed_house_ids", []) if isinstance(state.get("processed_house_ids", []), list) else []
    expected_houses = max(1, int(max_houses))
    metrics = route6_smoke_state_metrics(state)
    smoke_ok = (
        not error
        and worker_stage == "DONE"
        and worker_called
        and len(processed) >= expected_houses
        and int(metrics["mapped_house_count"]) > 0
        and int(metrics["valid_scan_capture_count_total"]) > 0
        and int(metrics["merged_point_count_total"]) > 0
    )
    result_path = result_run_dir / "route6_smoke_result.json"
    quality_path = result_run_dir / "map" / "route6_map_quality_report.json"
    summary_path = Path(str(state.get("exploration_summary_path", result_run_dir / "route6_exploration_summary.csv")))
    payload = {
        "schema": "route6_smoke_result_v1",
        "mode": "live_controller",
        "status": "ok" if smoke_ok else "check",
        "worker_stage": worker_stage,
        "error": error,
        "run_dir": str(result_run_dir),
        "result_path": str(result_path),
        "route6_state_path": str(result_run_dir / "route6_state.json"),
        "route6_exploration_summary_path": str(summary_path),
        "route6_map_quality_report_path": str(quality_path),
        "processed_house_ids": [str(item) for item in processed],
        "processed_house_count": len(processed),
        "session_started": session_started,
        "session_start_status": str(start_state.get("status", "")) if isinstance(start_state, dict) else "",
        "controller_worker_called": worker_called,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(metrics)
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        if panel is not None and callable(getattr(panel, "on_close", None)):
            panel.on_close()
        elif panel is not None:
            session = getattr(panel, "session", None)
            if session is not None and callable(getattr(session, "close", None)):
                session.close(force_kill_unreal=True)
    except Exception as exc:
        payload["close_error"] = str(exc)
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_initial_pose(values: Optional[List[float]]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    padded = list(values) + [0.0] * (4 - len(values))
    return {"x": float(padded[0]), "y": float(padded[1]), "z": float(padded[2]), "yaw": float(padded[3])}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an auditable Route 6 smoke pass on the existing worker path.")
    parser.add_argument("--mode", choices=["offline_mock", "live_controller"], default="offline_mock")
    parser.add_argument("--output_root", default="route6_smoke_runs")
    parser.add_argument("--map_config", default="", help="Optional map config JSON. Defaults to the controller map config.")
    parser.add_argument("--max_houses", type=int, default=1)
    parser.add_argument("--runtime_minutes", type=float, default=5.0)
    parser.add_argument("--max_scan_points", type=int, default=0, help="Optional Route 6 live smoke limit; 0 means use the full scan plan.")
    parser.add_argument("--standoff_cm", type=float, default=None, help="Optional Route 6 scan standoff override in Unreal cm.")
    parser.add_argument("--scan_z_cm", type=float, default=None, help="Optional Route 6 scan altitude override in Unreal cm.")
    parser.add_argument("--initial_pose", nargs="+", type=float, default=None, metavar="POSE", help="X Y Z YAW in cm/degrees.")
    parser.add_argument("--env_platform", "--platform", choices=["auto", "win", "mac", "linux"], default="auto")
    parser.add_argument("--env_root", default=None)
    parser.add_argument("--env_bin", default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--launch_sleep", type=int, default=15)
    parser.add_argument("--time_dilation", type=int, default=0)
    parser.add_argument("--step_delay", type=float, default=0.1)
    parser.add_argument("--lidar_capture_processing", choices=sorted(flight.LIDAR_CAPTURE_PROCESSING_MODES), default="full")
    parser.add_argument("--no_start_session", action="store_true", help="Create the controller but skip DroneFlightSession.start(); useful only for injected tests.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    map_config = None
    if args.map_config:
        map_path = Path(args.map_config)
        if not map_path.is_absolute():
            map_path = PROJECT_ROOT / map_path
        map_config = json.loads(map_path.read_text(encoding="utf-8"))
    controller_args = None
    if args.mode == "live_controller":
        controller_args = default_live_controller_args(
            output_root=Path(args.output_root),
            initial_pose=parse_initial_pose(args.initial_pose),
            env_platform=args.env_platform,
            env_root=args.env_root,
            env_bin=args.env_bin,
            width=args.width,
            height=args.height,
            launch_sleep=args.launch_sleep,
            time_dilation=args.time_dilation,
            step_delay=args.step_delay,
            map_config_path=args.map_config or DEFAULT_MAP_CONFIG_PATH,
            lidar_capture_processing=args.lidar_capture_processing,
        )
    result = run_route6_smoke(
        mode=args.mode,
        output_root=Path(args.output_root),
        max_houses=args.max_houses,
        runtime_minutes=args.runtime_minutes,
        max_scan_points=args.max_scan_points,
        standoff_cm=args.standoff_cm,
        scan_z_cm=args.scan_z_cm,
        map_config=map_config,
        initial_pose=parse_initial_pose(args.initial_pose),
        start_session=not bool(args.no_start_session),
        controller_args=controller_args,
    )
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
