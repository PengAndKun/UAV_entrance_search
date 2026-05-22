from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from control.common import PROJECT_ROOT
import control.route5_fusion_control as route5_module
from control.route5_fusion_control import Route5FusionControlMixin


class _Var:
    def __init__(self, value: Any = "") -> None:
        self._value = value

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value


class _Root:
    def after(self, _delay_ms: int, callback=None, *args):
        if callback is not None:
            callback(*args)
        return "after-id"

    def after_cancel(self, _job: Any) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.started = True
        self.calls = []

    def move_relative(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("move_relative", dict(payload)))
        return {"status": "ok", "movement_enabled": True, "movement_mode": "physics", "payload": payload}

    def set_movement_enabled(self, enabled: bool) -> Dict[str, Any]:
        self.calls.append(("set_movement_enabled", bool(enabled)))
        return {"status": "ok", "movement_enabled": bool(enabled), "movement_mode": "physics"}


class _Route5Harness(Route5FusionControlMixin):
    def __init__(self) -> None:
        self.root = _Root()
        self.llm_route5_status_var = _Var("LLM Route V5: idle")
        self.llm_route5_map_status_var = _Var("Route V5 Map: idle")
        self.llm_route5_stage_var = _Var("Stage: idle")
        self.llm_route5_active_var = _Var("Active: n/a")
        self.llm_route5_target_var = _Var("Target: n/a")
        self.llm_route5_error_var = _Var("Error: n/a")
        self.llm_route5_payload_var = _Var("Payload: hold")
        self.llm_route5_progress_text_var = _Var("Fusion: 0%")
        self.llm_route5_progress_var = _Var(0.0)
        self.llm_route5_current_status_var = _Var("Current: idle")
        self.llm_route5_next_status_var = _Var("Next: n/a")
        self.llm_route5_avoidance_status_var = _Var("Avoidance: idle")
        self.llm_route5_representation_status_var = _Var("OR2: idle")
        self.llm_route5_thinking_status_var = _Var("Thinking: idle")
        self.llm_route5_paused_var = _Var(False)
        self.llm_route5_auto_refresh_var = _Var(False)
        self.llm_route5_show_all_obs_points_var = _Var(False)
        self.llm_route5_show_facade_captures_var = _Var(False)
        self.llm_route5_move_tick_ms_var = _Var(175)
        self.llm_route5_nav_step_cm_var = _Var(25)
        self.llm_route5_reach_tol_cm_var = _Var(70)
        self.llm_route5_z_tol_cm_var = _Var(45)
        self.llm_route5_yaw_tol_deg_var = _Var(12)
        self.llm_route5_max_stage_s_var = _Var(95)
        self.llm_route5_sensing_interval_s_var = _Var(1.0)
        self.llm_route5_representation_model_var = _Var(str(PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"))
        self.llm_route5_oa3_plan_var = _Var(str(PROJECT_ROOT / "obstacle_avoidance_3_data" / "plans" / "obstacle_avoidance_3_plans.json"))
        self.route5_or2_state_var = _Var("State: --")
        self.route5_or2_frame_count_var = _Var("Frames: 0")
        self.route5_or2_front_depth_var = _Var("Front depth: --")
        self.route5_or2_can_forward_var = _Var("Can forward: --")
        self.route5_or2_selected_direction_var = _Var("Selected direction: --")
        self.route5_or2_corridor_var = _Var("Corridors: --")
        self.route5_or2_capture_dir_var = _Var("OR2 capture: --")
        self.route5_or2_interval_s_var = _Var("1.0")
        self.llm_route5_state: Dict[str, Any] = {}
        self.llm_route5_completed_facades = set()
        self.llm_route5_blocked_facades = set()
        self.llm_route5_thread = None
        self.llm_route5_stop_event = threading.Event()
        self.llm_route5_pause_event = threading.Event()
        self.llm_route5_control_locked = False
        self.llm_route3_pause_event = threading.Event()
        self.llm_route4_pause_event = threading.Event()
        self.llm_route3_control_locked = False
        self.llm_route4_control_locked = False
        self.llm_route5_auto_refresh_job = None
        self.llm_route3_completed_facades = {"west"}
        self.llm_route3_blocked_facades = {"north"}
        self.llm_route4_state = {"mode": "route4"}
        self.llm_route4_completed_facades = {"west"}
        self.llm_route4_blocked_facades = {"north"}
        self.map_world_bounds = (-1.0, -1.0, 1.0, 1.0)
        self.latest_state = {"pose": {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0, "task_yaw": 0.0}}
        self.llm_timeout_s_var = _Var("60.0")
        self.fake_llm_api_key = ""
        self.fake_llm_model = "verify-model"
        self.llm_call_payloads = []

    def selected_route_target_house_id(self) -> str:
        return "001"

    def effective_llm_api_key(self) -> str:
        return self.fake_llm_api_key

    def effective_llm_model(self) -> str:
        return self.fake_llm_model

    def call_configured_llm_text(self, **kwargs):
        self.llm_call_payloads.append(dict(kwargs))
        return {"raw_text": "{\"repair_action\":\"backoff_then_descend\",\"reason\":\"verify_llm_repair\"}"}

    def active_session(self):
        return getattr(self, "fake_session", None)

    def stop_keyboard_control(self, *, send_hold: bool = False, force_hold: bool = False) -> None:
        self.keyboard_stopped = {"send_hold": bool(send_hold), "force_hold": bool(force_hold)}

    def update_keyboard_status(self, message: str = "") -> None:
        self.keyboard_status = str(message or "")

    def apply_state(self, state: Dict[str, Any]) -> None:
        self.latest_state = state

    def safe(self, _label: str, fn):
        return fn()

    def write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_jsonl_artifact(self, path: Path):
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def route3_safety_report_for_pose(self, _target_house_id: str, pose: Dict[str, Any]) -> Dict[str, Any]:
        if getattr(self, "force_unsafe_reset", False):
            return {"safe": False, "reason": "verify_forced_unsafe"}
        if float(pose.get("x", 0.0) or 0.0) > 500.0:
            return {"safe": False, "reason": "verify_out_of_bounds"}
        if getattr(self, "force_negative_y_unsafe", False) and float(pose.get("y", 0.0) or 0.0) < -10.0:
            return {"safe": False, "reason": "verify_out_of_bounds"}
        return {"safe": True, "reason": "verify_safe"}

    def route3_predict_next_pose(self, current: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, float]:
        yaw = math.radians(float(current.get("yaw", 0.0) or 0.0))
        forward = float(payload.get("forward_cm", 0.0) or 0.0)
        right = float(payload.get("right_cm", 0.0) or 0.0)
        return {
            "x": float(current.get("x", 0.0) or 0.0) + forward * math.cos(yaw) - right * math.sin(yaw),
            "y": float(current.get("y", 0.0) or 0.0) + forward * math.sin(yaw) + right * math.cos(yaw),
            "z": float(current.get("z", 0.0) or 0.0) + float(payload.get("up_cm", 0.0) or 0.0),
            "yaw": float(current.get("yaw", 0.0) or 0.0) + float(payload.get("yaw_delta_deg", 0.0) or 0.0),
        }

    def _normalize_angle_deg(self, value: float) -> float:
        return ((float(value) + 180.0) % 360.0) - 180.0

    def route3_movement_payload_for_target(self, current: Dict[str, Any], target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        yaw = math.radians(float(current.get("yaw", 0.0) or 0.0))
        dx = float(target.get("x", 0.0) or 0.0) - float(current.get("x", 0.0) or 0.0)
        dy = float(target.get("y", 0.0) or 0.0) - float(current.get("y", 0.0) or 0.0)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        right = -dx * math.sin(yaw) + dy * math.cos(yaw)
        distance = math.hypot(forward, right)
        step = float(config.get("nav_step_cm", 20.0) or 20.0)
        if distance > step:
            scale = step / distance
            forward *= scale
            right *= scale
        return {
            "forward_cm": round(forward, 3),
            "right_cm": round(right, 3),
            "up_cm": 0.0,
            "yaw_delta_deg": 0.0,
            "action_name": "route3_nav",
        }

    def route3_observation_attempts_for_facade(self, _target_house_id: str, facade: str, base: Dict[str, Any]):
        attempts = getattr(self, "fake_observation_attempts", {})
        if facade in attempts:
            return attempts[facade]
        return [base] if base else []

    def route3_target_pose_from_point(self, point: Dict[str, Any]) -> Dict[str, float]:
        return {
            "x": float(point.get("x", 0.0) or 0.0),
            "y": float(point.get("y", 0.0) or 0.0),
            "z": float(point.get("z", 100.0) or 100.0),
            "yaw": float(point.get("yaw", point.get("yaw_deg", 0.0)) or 0.0),
        }

    def house_world_bbox_for_id(self, _house_id: str):
        return {"min_x": -337.67, "max_x": 1726.0, "min_y": 1420.65, "max_y": 2177.44}

    def point_inside_open_bbox(self, x: float, y: float, bbox: Dict[str, Any]) -> bool:
        return float(bbox["min_x"]) < float(x) < float(bbox["max_x"]) and float(bbox["min_y"]) < float(y) < float(bbox["max_y"])

    def route2_facade_axis_range(self, bbox: Dict[str, Any], facade: str):
        if facade in {"north", "south"}:
            return float(bbox["min_x"]), float(bbox["max_x"])
        return float(bbox["min_y"]), float(bbox["max_y"])

    def route2_facade_center_axis(self, bbox: Dict[str, Any], facade: str) -> float:
        lo, hi = self.route2_facade_axis_range(bbox, facade)
        return (lo + hi) / 2.0

    def route2_observation_map_bounds_report(self, _x: float, _y: float):
        return {"in_bounds": True, "source": "verify"}

    def route2_observation_blocking_house(self, *_args):
        return {}

    def route2_facade_id(self, house_id: str, facade: str) -> str:
        return f"{house_id}_{facade}"

    def _as_float_or_none(self, value: Any):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def route3_plan_navigation_waypoints(self, current: Dict[str, Any], target: Dict[str, Any], _target_house_id: str, *, grid_cm: float):
        return {
            "status": "ok",
            "reason": "verify_direct",
            "grid_cm": grid_cm,
            "waypoints": [target],
            "raw_waypoints": [current, target],
        }

    def route3_navigation_plan_cost_cm(self, current: Dict[str, Any], plan: Dict[str, Any]) -> float:
        target = (plan.get("waypoints") or [{}])[-1]
        return math.hypot(
            float(target.get("x", 0.0) or 0.0) - float(current.get("x", 0.0) or 0.0),
            float(target.get("y", 0.0) or 0.0) - float(current.get("y", 0.0) or 0.0),
        )

    def route2_all_facade_observation_candidates(self, _target_house_id: str, skip_completed: bool = False):
        return list(getattr(self, "fake_all_observation_candidates", []))

    def route5_state_output_dir(self):
        output_dir = self.llm_route5_state.get("output_dir")
        return Path(output_dir) if output_dir else None

    def route2_facade_paths(self):
        output_dir = self.route5_state_output_dir()
        house_id = str(self.llm_route5_state.get("target_house_id", "001") or "001")
        facade = str(self.llm_route5_state.get("active_facade", "west") or "west")
        return output_dir, output_dir / "facade_observations" / f"{house_id}_{facade}", house_id, facade


def _stop_mask_prediction(kind: str) -> Dict[str, Any]:
    stop = np.zeros((96, 96), dtype=np.float32)
    if kind == "left_clear":
        stop[25:80, 55:96] = 1.0
    elif kind == "all_blocked":
        stop[:, :] = 1.0
    front = stop[25:75, 34:66]
    state = "must_stop" if float(np.mean(front >= 0.5)) > 0.01 else "clear"
    return {
        "status": "ok",
        "front_risk_state": state,
        "can_forward": state == "clear",
        "must_stop": state == "must_stop",
        "must_stop_mask": stop,
        "obstacle_warning_mask": np.zeros_like(stop),
        "clearance_warning_mask": np.zeros_like(stop),
        "risk_overlay_path": "frame_000001/or2_risk_overlay.png",
        "prediction_json_path": "frame_000001/or2_risk_prediction.json",
        "reason": "verify_route5_fake_prediction",
    }


def main() -> None:
    harness = _Route5Harness()
    model_path = harness.default_route5_or2_model_path()
    plan_path = harness.default_route5_oa3_plan_path()
    assert model_path == PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"
    assert plan_path == PROJECT_ROOT / "obstacle_avoidance_3_data" / "plans" / "obstacle_avoidance_3_plans.json"
    assert (PROJECT_ROOT / "obstacle_representation_2_data" / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"
    route5_source = (PROJECT_DIR / "control" / "route5_fusion_control.py").read_text(encoding="utf-8")
    assert "Global Pause All" in route5_source
    assert "Release Movement" in route5_source
    for removed in (
        "Start OR2 Monitor",
        "Stop OR2 Monitor",
        "Capture OR2 Once",
        "def start_route5_or2_monitor",
        "def capture_route5_or2_once",
        "def run_route5_or2_monitor",
        "route5_or2_monitor_thread = threading.Thread",
    ):
        assert removed not in route5_source, removed

    with tempfile.TemporaryDirectory() as temp_dir:
        output_root = Path(temp_dir)
        harness.llm_route5_output_root_override = output_root
        output_dir = harness.route5_initialize_run("001", force_new=True)
        assert output_dir.parent == output_root
        assert output_dir.name.startswith("house_001_autosearch_v5_or2_fused_")
        assert harness.llm_route5_state["mode"] == "route5_llm_route_oa3_or2_fusion"
        assert harness.llm_route5_state["oa3_config"]["or2_model_path"].endswith("a_plus_2_model.pt")
        assert harness.llm_route5_state["oa3_config"]["oa3_plan_path"].endswith("obstacle_avoidance_3_plans.json")
        for artifact in (
            "route5_fusion_state.json",
            "route5_fusion_events.jsonl",
            "route5_navigation_plan.jsonl",
            "route5_movement_trace.jsonl",
            "route5_frame_decisions.jsonl",
            "route5_or2_risk_events.jsonl",
            "route5_target_resets.jsonl",
            "route5_plan_deviation_repairs.jsonl",
            "route5_scan_postprocess_events.jsonl",
            "route5_or2_monitor_summary.json",
            "avoidance_events.jsonl",
            "avoidance_session_summary.json",
            "house_exploration_memory_events.jsonl",
        ):
            assert (output_dir / artifact).exists(), artifact
        assert harness.llm_route4_state == {"mode": "route4"}
        assert harness.llm_route4_completed_facades == {"west"}
        assert harness.llm_route4_blocked_facades == {"north"}

        fake_session = _FakeSession()
        harness.fake_session = fake_session
        harness.llm_route3_control_locked = True
        harness.llm_route4_control_locked = True
        harness.llm_route5_control_locked = True
        pause_result = harness.on_route5_global_pause_all()
        assert pause_result["paused"] == ["route3", "route4", "route5"], pause_result
        assert harness.llm_route3_pause_event.is_set()
        assert harness.llm_route4_pause_event.is_set()
        assert harness.llm_route5_pause_event.is_set()
        assert any(call[0] == "move_relative" for call in fake_session.calls), fake_session.calls
        release_result = harness.on_route5_release_movement()
        assert release_result["released"] is True, release_result
        assert harness.llm_route3_control_locked is False
        assert harness.llm_route4_control_locked is False
        assert harness.llm_route5_control_locked is False
        assert any(call == ("set_movement_enabled", True) for call in fake_session.calls), fake_session.calls

        harness.llm_route5_state.update(
            current_exploration_status={
                "stage": "NAV_TO_SCAN_POINT",
                "facade": "east",
                "target_id": "002_scan_0016_east_single_300cm",
                "scan_id": "002_scan_0016_east_single_300cm",
                "target_pose": {"x": 2125.6, "y": 1420.65, "z": 300.0, "yaw": 180.0},
            },
            current_navigation_plan={
                "stage": "NAV_TO_SCAN_POINT",
                "facade": "east",
                "target_id": "002_scan_0016_east_single_300cm",
                "plan": {"waypoints": [{"x": 1800.0, "y": 1380.0, "z": 300.0, "yaw": 10.0}, {"x": 2125.6, "y": 1420.65, "z": 300.0, "yaw": 180.0}]},
            },
            reset_route_points=[{"x": -99.0, "y": -99.0, "z": 300.0, "label": "old_reset"}],
            last_target_reset={
                "stage": "NAV_TO_SCAN_POINT",
                "facade": "east",
                "target_id": "002_scan_0016_east_single_300cm",
                "reset_target_id": "002_scan_0016_east_single_300cm_reset_1",
                "reset_target_pose": {"x": 2200.0, "y": 1311.0, "z": 323.0, "yaw": 180.0},
            },
        )
        harness.llm_route2_state = {
            "facade_scan_points": [
                {"scan_id": "old_south_scan", "facade": "south", "x": 0.0, "y": 0.0, "z": 300.0},
                {"scan_id": "future_east_scan", "facade": "east", "x": 9999.0, "y": 9999.0, "z": 300.0},
            ]
        }
        visible = harness.route5_active_map_route_points()
        labels = [item.get("label") for item in visible]
        assert "002_scan_0016_east_single_300cm" in labels, visible
        assert "002_scan_0016_east_single_300cm_wp_1" in labels, visible
        assert "002_scan_0016_east_single_300cm_reset_1" in labels, visible
        assert "old_south_scan" not in labels and "future_east_scan" not in labels and "old_reset" not in labels, visible

        size = harness.route5_map_canvas_size_for_width(1450, image_size=(2226, 1188))
        assert size["width"] >= 1200 and 300 <= size["height"] <= 620, size

        harness.fake_all_observation_candidates = [
            {"label": "002_south_obs_1", "facade": "south", "x": 10.0, "y": 10.0, "z": 280.0, "status": "planned"},
            {"label": "002_east_obs_1", "facade": "east", "x": 20.0, "y": 20.0, "z": 280.0, "status": "planned"},
            {"label": "002_west_obs_1", "facade": "west", "x": 30.0, "y": 30.0, "z": 280.0, "status": "blocked"},
        ]
        harness.llm_route5_completed_facades = {"south"}
        harness.llm_route5_blocked_facades = {"west"}
        harness.llm_route5_show_all_obs_points_var.set(True)
        visible_obs = harness.route5_active_map_route_points()
        obs_by_label = {item["label"]: item for item in visible_obs if item.get("route_point_type") == "observation_point"}
        assert obs_by_label["002_south_obs_1"]["status"] == "captured", obs_by_label
        assert obs_by_label["002_east_obs_1"]["status"] == "pending", obs_by_label
        assert obs_by_label["002_west_obs_1"]["status"] == "blocked", obs_by_label
        harness.llm_route5_show_all_obs_points_var.set(False)

        harness.llm_route5_state["facade_scan_points"] = [
            {"scan_id": "002_scan_0016_east_single_300cm", "facade": "east", "x": 101.0, "y": 101.0, "z": 300.0},
            {"scan_id": "002_scan_0017_east_single_300cm", "facade": "east", "x": 102.0, "y": 102.0, "z": 300.0},
            {"scan_id": "002_scan_0001_south_single_300cm", "facade": "south", "x": 103.0, "y": 103.0, "z": 300.0},
        ]
        harness.llm_route5_state["captured_scan_ids"] = ["002_scan_0016_east_single_300cm"]
        harness.llm_route5_show_facade_captures_var.set(True)
        visible_scan = harness.route5_active_map_route_points()
        scan_by_id = {item["scan_id"]: item for item in visible_scan if item.get("route_point_type") == "scan_point"}
        assert set(scan_by_id) == {"002_scan_0016_east_single_300cm", "002_scan_0017_east_single_300cm"}, scan_by_id
        assert scan_by_id["002_scan_0016_east_single_300cm"]["status"] == "active", scan_by_id
        assert scan_by_id["002_scan_0017_east_single_300cm"]["status"] == "pending", scan_by_id
        harness.llm_route5_show_facade_captures_var.set(False)

        harness.llm_route5_state["task_plan"] = {"facade_priority": ["east", "north", "west", "south"]}
        harness.fake_observation_attempts = {
            "east": [{"facade": "east", "x": 1000.0, "y": 0.0, "z": 100.0, "yaw": 180.0, "observation_attempt_index": 1}],
            "north": [{"facade": "north", "x": 120.0, "y": 0.0, "z": 100.0, "yaw": 90.0, "observation_attempt_index": 1}],
            "west": [{"facade": "west", "x": 600.0, "y": 0.0, "z": 100.0, "yaw": 0.0, "observation_attempt_index": 1}],
        }
        ranked = harness.route5_rank_observation_candidates(
            "001",
            [{"facade": name} for name in ("east", "north", "west")],
            completed={"south"},
            blocked=set(),
            start_pose={"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
        )
        assert [item["facade"] for item in ranked[:3]] == ["north", "west", "east"], ranked
        assert ranked[0]["route5_selection_policy"] == "nearest_from_current_uav_pose", ranked[0]
        next_decision = harness.route5_decide_next_facade("001", ranked, completed={"south"}, blocked=set())
        assert next_decision["target_facade"] == "north", next_decision
        assert next_decision["reason"] == "nearest_from_current_uav_pose", next_decision

        incomplete_gate = harness.route5_facade_completion_gate(
            {"overall_passed": False, "coverage_report": {"captured_scan_count": 0}},
            observation={},
            rgb_result={"status": "ok"},
            scan_capture_count=0,
        )
        assert incomplete_gate["completion_status"] == "scan_incomplete", incomplete_gate
        assert incomplete_gate["complete"] is False, incomplete_gate
        stopped_gate = harness.route5_facade_completion_gate(
            {"overall_passed": True, "coverage_report": {"captured_scan_count": 1}},
            observation={},
            rgb_result={"status": "ok"},
            scan_capture_count=1,
            scan_loop_stopped=True,
        )
        assert stopped_gate["completion_status"] == "scan_incomplete", stopped_gate
        assert stopped_gate["complete"] is False, stopped_gate
        full_gate = harness.route5_facade_completion_gate(
            {"overall_passed": True, "coverage_report": {"captured_scan_count": 2}},
            observation={},
            rgb_result={"status": "ok"},
            scan_capture_count=2,
        )
        assert full_gate["completion_status"] == "full_completed", full_gate
        assert full_gate["complete"] is True, full_gate
        degraded_gate = harness.route5_facade_completion_gate(
            {"overall_passed": False, "coverage_report": {"captured_scan_count": 0}},
            observation={"route5_degraded_observation": True},
            rgb_result={"status": "ok", "rgb_path": "coarse_rgb.png"},
            scan_capture_count=0,
        )
        assert degraded_gate["completion_status"] == "scan_incomplete", degraded_gate
        assert degraded_gate["complete"] is False, degraded_gate
        assert degraded_gate["reason"] == "degraded_observation_without_scan_capture", degraded_gate

        near_arrival = harness.route5_near_obstacle_arrival_state(
            {"reached": False, "dist_xy_cm": 140.0, "dist_3d_cm": 145.0},
            {"front_risk_state": "obstacle_warning", "front_min_depth_cm": 220.0, "avoidance_active": True},
            {"pointcloud_summary": {"front_min_depth_cm": 220.0, "forward_swept_clear": False}},
        )
        assert near_arrival["near_obstacle_reached"] is True, near_arrival
        assert near_arrival["arrival_policy"] == "near_obstacle_reached", near_arrival
        far_arrival = harness.route5_near_obstacle_arrival_state(
            {"reached": False, "dist_xy_cm": 220.0, "dist_3d_cm": 225.0},
            {"front_risk_state": "obstacle_warning", "front_min_depth_cm": 220.0, "avoidance_active": True},
            {"pointcloud_summary": {"front_min_depth_cm": 220.0, "forward_swept_clear": False}},
        )
        assert far_arrival["near_obstacle_reached"] is False, far_arrival
        assert far_arrival["arrival_policy"] == "approach_with_caution", far_arrival
        clear_near_arrival = harness.route5_near_obstacle_arrival_state(
            {"reached": False, "dist_xy_cm": 100.0, "dist_3d_cm": 100.0},
            {"front_risk_state": "clear", "front_min_depth_cm": 600.0, "avoidance_active": False},
            {"pointcloud_summary": {"front_min_depth_cm": 600.0, "forward_swept_clear": True}},
        )
        assert clear_near_arrival["near_obstacle_reached"] is False, clear_near_arrival

        event = {
            "frame_id": 1,
            "route5_stage": "NAV_TO_SCAN_POINT",
            "facade": "west",
            "target_id": "001_scan_0001_west_single_300cm",
            "capture_dir": str(output_dir / "frames" / "frame_000001"),
            "current_pose": {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
            "target_waypoint": {"x": 100.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
            "pointcloud_summary": {"available": True, "front_min_depth_cm": 80.0, "forward_swept_clear": False},
            "relative_target": {"bearing_deg_body": -5.0, "dz_cm": 0.0},
            "or2_prediction": _stop_mask_prediction("left_clear"),
            "collision_state": False,
            "avoidance_failed": False,
        }
        decision = harness.route5_or2_decision_for_event(
            event,
            nominal_payload={"forward_cm": 25.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav"},
            config={"nav_step_cm": 25.0, "reach_tol_cm": 70.0},
            current_pose=event["current_pose"],
            start_pose=event["current_pose"],
            target_pose=event["target_waypoint"],
            last_action={"action_name": "hold", "forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
        )
        assert decision["gate"]["avoidance_active"] is True, decision
        assert decision["selected_direction"] == "left", decision
        assert decision["payload"]["right_cm"] < 0.0, decision

        clear_event = dict(event)
        clear_event["pointcloud_summary"] = {"available": True, "front_min_depth_cm": 600.0, "forward_swept_clear": True}
        clear_event["or2_prediction"] = _stop_mask_prediction("clear")
        clear_decision = harness.route5_or2_decision_for_event(
            clear_event,
            nominal_payload={"forward_cm": 25.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav"},
            config={"nav_step_cm": 25.0, "reach_tol_cm": 70.0},
            current_pose=clear_event["current_pose"],
            start_pose=clear_event["current_pose"],
            target_pose=clear_event["target_waypoint"],
            last_action={"action_name": "hold"},
        )
        assert clear_decision["gate"]["avoidance_active"] is False, clear_decision
        assert clear_decision["selected_direction"] in {"forward", "slow_forward"}, clear_decision
        assert clear_decision["payload"]["forward_cm"] == 25.0, clear_decision

        fallback_event = dict(event)
        fallback_event["or2_prediction"] = {"status": "error", "reason": "model unavailable"}
        fallback = harness.route5_or2_decision_for_event(
            fallback_event,
            nominal_payload={"forward_cm": 25.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav"},
            config={"nav_step_cm": 25.0, "reach_tol_cm": 70.0},
            current_pose=fallback_event["current_pose"],
            start_pose=fallback_event["current_pose"],
            target_pose=fallback_event["target_waypoint"],
            last_action={"action_name": "hold"},
        )
        assert fallback["gate"]["source"] == "route5_depth_pointcloud_fallback", fallback
        assert fallback["gate"]["avoidance_active"] is True, fallback

        harness.force_unsafe_reset = False
        harness.force_negative_y_unsafe = True
        unsafe_left_event = dict(event)
        unsafe_left_event["target_house_id"] = "001"
        unsafe_left_event["current_pose"] = {"x": 490.0, "y": 0.0, "z": 100.0, "yaw": 0.0}
        unsafe_left_event["target_waypoint"] = {"x": 100.0, "y": 0.0, "z": 100.0, "yaw": 0.0}
        unsafe_left_event["or2_prediction"] = _stop_mask_prediction("left_clear")
        safe_alternative = harness.route5_or2_decision_for_event(
            unsafe_left_event,
            nominal_payload={"forward_cm": 25.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav"},
            config={"nav_step_cm": 25.0, "reach_tol_cm": 70.0},
            current_pose=unsafe_left_event["current_pose"],
            start_pose=unsafe_left_event["current_pose"],
            target_pose=unsafe_left_event["target_waypoint"],
            last_action={"action_name": "hold"},
        )
        assert safe_alternative["selected_direction"] != "left", safe_alternative
        assert safe_alternative["event_updates"]["or2_selected_action_rejected_reason"] == "verify_out_of_bounds", safe_alternative
        assert safe_alternative["event_updates"]["safe_alternative_action"], safe_alternative
        assert safe_alternative["event_updates"]["candidate_safety_scores"], safe_alternative
        harness.force_negative_y_unsafe = False

        nav_config = {
            "nav_step_cm": 20.0,
            "reach_tol_cm": 60.0,
            "z_tol_cm": 40.0,
            "yaw_tol_deg": 10.0,
        }
        yaw_first = harness.route5_movement_payload_for_target_with_lookahead(
            {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
            {"x": 0.0, "y": 1000.0, "z": 100.0, "yaw": 90.0},
            nav_config,
            stage="NAV_TO_OBS",
        )
        assert yaw_first["yaw_policy"] == "face_waypoint_then_forward", yaw_first
        assert yaw_first["forward_cm"] == 0.0, yaw_first
        assert yaw_first["right_cm"] == 0.0, yaw_first
        assert yaw_first["yaw_delta_deg"] == 30.0, yaw_first
        forward_after_yaw = harness.route5_movement_payload_for_target_with_lookahead(
            {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 90.0},
            {"x": 0.0, "y": 1000.0, "z": 100.0, "yaw": 90.0},
            nav_config,
            stage="NAV_TO_OBS",
        )
        assert forward_after_yaw["yaw_policy"] == "face_waypoint_then_forward", forward_after_yaw
        assert forward_after_yaw["forward_cm"] == 20.0, forward_after_yaw
        assert forward_after_yaw["right_cm"] == 0.0, forward_after_yaw
        assert forward_after_yaw["yaw_delta_deg"] == 0.0, forward_after_yaw

        tracker = harness.route5_new_target_reset_tracker("001_scan_0012_west_single_300cm", {"x": 0.0, "y": 400.0, "z": 100.0, "yaw": 0.0})
        reset_event = {
            "frame_id": 50,
            "or2_selected_direction": "right",
            "or2_rule": {"selected_direction": "right", "candidate_action_scores": {"right": 0.95, "left": 0.1, "up": 0.2, "backoff": 0.05}},
            "or2_prediction": {"front_risk_state": "obstacle_warning", "can_forward": True, "must_stop": False},
            "pointcloud_summary": {"front_min_depth_cm": 180.0},
        }
        reset_gate = {"avoidance_active": True, "front_risk_state": "obstacle_warning", "selected_direction": "right", "front_min_depth_cm": 180.0}
        reset_error = {"reached": False, "dist_xy_cm": 420.0}
        for _ in range(8):
            tracker = harness.route5_record_target_reset_tick(
                tracker,
                reset_event,
                reset_gate,
                reset_error,
                current_pose={"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
                target_pose={"x": 0.0, "y": 400.0, "z": 100.0, "yaw": 0.0},
            )
        trigger = harness.route5_should_reset_target(tracker, reset_error)
        assert trigger["should_reset"] is True, trigger
        candidate = harness.route5_build_target_reset_candidate(
            target_house_id="001",
            stage="NAV_TO_SCAN_POINT",
            facade="west",
            target_id="001_scan_0012_west_single_300cm",
            current_pose={"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
            target_pose={"x": 0.0, "y": 400.0, "z": 100.0, "yaw": 0.0},
            tracker=tracker,
            reset_reason=trigger["reason"],
            reset_index=1,
        )
        assert candidate["status"] == "ok", candidate
        assert candidate["reset_direction"] == "right", candidate
        assert candidate["reset_target_pose"]["y"] > 0.0, candidate
        assert candidate["reset_target_pose"]["yaw"] == 0.0, candidate
        harness.route5_record_target_reset(output_dir, candidate)
        reset_rows = harness.read_jsonl_artifact(output_dir / "route5_target_resets.jsonl")
        assert reset_rows and reset_rows[-1]["reset_target_id"].endswith("_reset_1"), reset_rows
        harness.force_unsafe_reset = True
        failed_candidate = harness.route5_build_target_reset_candidate(
            target_house_id="001",
            stage="NAV_TO_SCAN_POINT",
            facade="west",
            target_id="001_scan_0012_west_single_300cm",
            current_pose={"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
            target_pose={"x": 0.0, "y": 400.0, "z": 100.0, "yaw": 0.0},
            tracker=tracker,
            reset_reason=trigger["reason"],
            reset_index=2,
        )
        assert failed_candidate["status"] == "failed", failed_candidate
        assert failed_candidate["rejected_candidates"], failed_candidate
        harness.force_unsafe_reset = False

        high_tracker = harness.route5_new_target_reset_tracker(
            "001_scan_0013_west_single_300cm",
            {"x": 0.0, "y": 400.0, "z": 300.0, "yaw": 0.0},
        )
        high_tracker["last_direction"] = "up"
        high_tracker["direction_counts"] = {"up": 8}
        high_candidate = harness.route5_build_target_reset_candidate(
            target_house_id="001",
            stage="NAV_TO_SCAN_POINT",
            facade="west",
            target_id="001_scan_0013_west_single_300cm",
            current_pose={"x": 0.0, "y": 0.0, "z": 760.0, "yaw": 0.0},
            target_pose={"x": 0.0, "y": 400.0, "z": 300.0, "yaw": 0.0},
            tracker=high_tracker,
            reset_reason="verify_high_altitude",
            reset_index=1,
        )
        assert high_candidate["status"] == "ok", high_candidate
        assert high_candidate["reset_direction"] != "up", high_candidate
        assert high_candidate["reset_target_pose"]["z"] <= 380.0, high_candidate
        assert high_candidate["height_corridor"]["max_z_cm"] == 380.0, high_candidate

        deviation_tracker = harness.route5_new_target_reset_tracker(
            "001_scan_0014_west_single_300cm",
            {"x": 0.0, "y": 500.0, "z": 300.0, "yaw": 0.0},
        )
        deviation_event = {
            "frame_id": 70,
            "or2_selected_direction": "up",
            "or2_rule": {"selected_direction": "up", "candidate_action_scores": {"up": 0.99}},
            "or2_prediction": {"front_risk_state": "clearance_warning"},
            "pointcloud_summary": {"front_min_depth_cm": 260.0},
        }
        for _ in range(3):
            deviation_tracker = harness.route5_record_target_reset_tick(
                deviation_tracker,
                deviation_event,
                {"avoidance_active": True, "front_risk_state": "clearance_warning", "selected_direction": "up", "front_min_depth_cm": 260.0},
                {"reached": False, "dist_xy_cm": 410.0},
                current_pose={"x": 0.0, "y": 0.0, "z": 760.0, "yaw": 0.0},
                target_pose={"x": 0.0, "y": 500.0, "z": 300.0, "yaw": 0.0},
            )
        deviation = harness.route5_plan_deviation_state(
            "NAV_TO_SCAN_POINT",
            {"x": 0.0, "y": 0.0, "z": 760.0, "yaw": 0.0},
            {"x": 0.0, "y": 500.0, "z": 300.0, "yaw": 0.0},
            deviation_tracker,
            {"reached": False, "dist_xy_cm": 410.0},
            deviation_event,
        )
        assert deviation["should_repair"] is True, deviation
        repair = harness.route5_plan_deviation_repair_decision(
            output_dir,
            target_house_id="001",
            stage="NAV_TO_SCAN_POINT",
            facade="west",
            target_id="001_scan_0014_west_single_300cm",
            current_pose={"x": 0.0, "y": 0.0, "z": 760.0, "yaw": 0.0},
            target_pose={"x": 0.0, "y": 500.0, "z": 300.0, "yaw": 0.0},
            tracker=deviation_tracker,
            error={"reached": False, "dist_xy_cm": 410.0},
            event=deviation_event,
            repair_index=1,
        )
        assert repair["status"] == "ok", repair
        assert repair["repair_action"] in {"descend_to_target_band", "backoff_then_descend", "lateral_reset_then_descend"}, repair
        assert repair["repair_target_pose"]["z"] <= 380.0, repair
        assert (output_dir / "route5_plan_deviation_repairs.jsonl").read_text(encoding="utf-8").strip(), repair
        harness.fake_llm_api_key = "verify-key"
        llm_repair = harness.route5_plan_deviation_repair_decision(
            output_dir,
            target_house_id="001",
            stage="NAV_TO_SCAN_POINT",
            facade="west",
            target_id="001_scan_0014_west_single_300cm_llm",
            current_pose={"x": 0.0, "y": 0.0, "z": 760.0, "yaw": 0.0},
            target_pose={"x": 0.0, "y": 500.0, "z": 300.0, "yaw": 0.0},
            tracker=deviation_tracker,
            error={"reached": False, "dist_xy_cm": 410.0},
            event=deviation_event,
            repair_index=2,
        )
        assert llm_repair["repair_action"] == "backoff_then_descend", llm_repair
        assert harness.llm_call_payloads and "timeout_s" not in harness.llm_call_payloads[-1], harness.llm_call_payloads[-1]
        assert "system_prompt" in harness.llm_call_payloads[-1] and "user_prompt" in harness.llm_call_payloads[-1], harness.llm_call_payloads[-1]
        harness.fake_llm_api_key = ""

        far_blocked_nav = {
            "status": "blocked",
            "reason": "non_target_house_clearance",
            "target_id": "002_west_obs_attempt_4",
            "current_pose": {"x": 707.365, "y": 2510.527, "z": 309.855, "yaw": -120.845},
            "pose_error": {"dist_xy_cm": 1130.03, "reached": False},
            "safety": {"safe": False, "reason": "non_target_house_clearance", "blocking_house_id": "house_11"},
        }
        rescue_candidate = harness.route5_observation_rescue_candidate(
            target_house_id="002",
            facade="west",
            original_observation={"x": -377.835, "y": 2195.394, "z": 309.855, "yaw_deg": -120.845},
            nav_result=far_blocked_nav,
        )
        assert rescue_candidate == {}, rescue_candidate
        assert harness.route5_observation_failure_retry_status(far_blocked_nav)["terminal"] is False

        harness.llm_route5_state["active_facade"] = "west"
        harness.llm_route5_state["target_house_id"] = "001"
        facade_dir = output_dir / "facade_observations" / "001_west"
        capture_dir = output_dir / "frames" / "frame_009000"
        facade_dir.mkdir(parents=True, exist_ok=True)
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "capture.json").write_text(json.dumps({"point_count": 0, "capture_dir": str(capture_dir)}, ensure_ascii=False), encoding="utf-8")
        capture_row = {
            "frame_index": 9000,
            "scan_id": "001_scan_0001_west_single_300cm",
            "facade": "west",
            "capture_dir": str(capture_dir),
            "point_count": 0,
        }
        execution_row = {
            "scan_id": "001_scan_0001_west_single_300cm",
            "facade": "west",
            "capture_status": "ok",
            "capture_dirs": [str(capture_dir)],
            "point_count": 0,
        }
        for path in (output_dir / "lidar_capture_log.jsonl", facade_dir / "lidar_capture_log.jsonl"):
            harness.append_jsonl(path, capture_row)
        for path in (output_dir / "scan_execution_log.jsonl", facade_dir / "scan_execution_log.jsonl"):
            harness.append_jsonl(path, execution_row)
        original_ensure = route5_module.flight.ensure_standard_world_cloud_for_capture
        try:
            def fake_ensure(path, **_kwargs):
                return {
                    "point_count": 42,
                    "point_cloud_world_standard_m_npy_path": str(Path(path) / "point_cloud_world_standard_m.npy"),
                    "point_cloud_world_standard_m_ply_path": str(Path(path) / "point_cloud_world_standard_m.ply"),
                    "postprocess_status": "done",
                }
            route5_module.flight.ensure_standard_world_cloud_for_capture = fake_ensure
            refresh = harness.route5_refresh_facade_scan_pointcloud_rows(output_dir, facade_dir, "west")
        finally:
            route5_module.flight.ensure_standard_world_cloud_for_capture = original_ensure
        assert refresh["source_point_count"] == 42, refresh
        refreshed_capture = harness.read_jsonl_artifact(facade_dir / "lidar_capture_log.jsonl")[0]
        refreshed_execution = harness.read_jsonl_artifact(facade_dir / "scan_execution_log.jsonl")[0]
        assert refreshed_capture["point_count"] == 42, refreshed_capture
        assert refreshed_execution["point_count"] == 42, refreshed_execution
        assert harness.read_jsonl_artifact(output_dir / "route5_scan_postprocess_events.jsonl"), refresh

        postprocess_gate = harness.route5_facade_completion_gate(
            {"overall_passed": False, "coverage_report": {"captured_scan_count": 1}},
            observation={},
            rgb_result={"status": "ok"},
            scan_capture_count=1,
            postprocess_result={"processed_frame_count": 1, "source_point_count": 0, "failed_count": 1},
        )
        assert postprocess_gate["completion_status"] == "postprocess_failed", postprocess_gate
        assert postprocess_gate["terminal"] is False, postprocess_gate
        assert harness.route5_facade_status_is_terminal("scan_incomplete") is False
        assert harness.route5_facade_status_is_terminal("postprocess_failed") is False

        blocked_event = {
            "frame_id": 2,
            "route5_stage": "NAV_TO_SCAN_POINT",
            "facade": "west",
            "target_id": "001_scan_blocked",
            "capture_dir": str(output_dir / "frames" / "frame_000002"),
            "current_pose": {"x": 0.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
            "target_waypoint": {"x": 100.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
            "avoidance_active": True,
            "collision_state": False,
            "avoidance_failed": False,
        }
        blocked_decision = harness.route5_write_safety_blocked_frame_decision(
            output_dir,
            blocked_event,
            {"forward_cm": 25.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav"},
            {"safe": False, "reason": "non_target_house_clearance"},
        )
        assert blocked_decision["decision_reason"] == "safety_blocked_before_move", blocked_decision
        assert json.loads((output_dir / "frames" / "frame_000002" / "decision.json").read_text(encoding="utf-8"))["decision_reason"] == "safety_blocked_before_move"

        event.update(decision["event_updates"])
        event["selected_action_payload"] = decision["payload"]
        event["nominal_action"] = {"forward_cm": 25.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav"}
        event["target_reset_candidate"] = candidate
        event["target_reset_applied"] = True
        event["reset_reason"] = candidate["reset_reason"]
        event["plan_deviation"] = deviation
        event["plan_repair_applied"] = True
        event["plan_repair"] = repair
        event["repair_action"] = repair["repair_action"]
        event["repair_reason"] = repair["repair_reason"]
        event.update(near_arrival)
        doc = harness.route5_write_frame_decision(output_dir, event, final_payload=decision["payload"])
        saved = json.loads((output_dir / "frames" / "frame_000001" / "decision.json").read_text(encoding="utf-8"))
        assert saved["or2"]["front_risk_state"] == "must_stop", saved
        assert saved["or2"]["selected_direction"] == "left", saved
        assert saved["or2"]["risk_overlay_path"].endswith("or2_risk_overlay.png"), saved
        assert saved["arrival_policy"] == "near_obstacle_reached", saved
        assert saved["near_obstacle_reached"] is True, saved
        assert saved["front_depth_cm"] == 220.0, saved
        assert saved["distance_to_goal_cm"] == 145.0, saved
        assert saved["target_reset_applied"] is True, saved
        assert saved["target_reset_candidate"]["reset_direction"] == "right", saved
        assert saved["reset_reason"] == candidate["reset_reason"], saved
        assert saved["plan_repair_applied"] is True, saved
        assert saved["repair_action"] == repair["repair_action"], saved
        assert saved["plan_deviation"]["should_repair"] is True, saved
        assert saved["avoidance_failed"] is False
        assert doc["avoidance_failed"] == doc["collision_state"]
        assert (output_dir / "route5_frame_decisions.jsonl").read_text(encoding="utf-8").strip()

    print("OK route5 fusion verification: state isolation, OR2/OA3 defaults, readonly monitor, global pause/release movement, target reset, map overlays, responsive map sizing, dynamic facade rank, completion gate, plan deviation repair, near-obstacle arrival, frame decisions, fallback, and collision contract passed")


if __name__ == "__main__":
    main()
