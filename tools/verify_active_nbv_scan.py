from __future__ import annotations

import ast
import importlib
import json
import math
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ValueVar:
    def __init__(self, value: Any = "") -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_module() -> Any:
    return importlib.import_module("control.active_nbv_scan_control")


def panel_source() -> str:
    return (PROJECT_ROOT / "control" / "panel.py").read_text(encoding="utf-8")


def test_panel_button_wiring() -> None:
    source = panel_source()
    assert_true(
        "ActiveNBVScanControlMixin" in source,
        "RunDroneFlightPanel must import and inherit ActiveNBVScanControlMixin",
    )
    assert_true(
        'text="Open Active NBV Scan"' in source or "text='Open Active NBV Scan'" in source,
        "main Route6_entrance_search actions row must expose the Open Active NBV Scan button",
    )
    assert_true(
        "command=self.open_active_nbv_scan_window" in source,
        "Open Active NBV Scan button must call open_active_nbv_scan_window",
    )


class Harness:
    def __init__(self, mixin_cls: type) -> None:
        self.args = SimpleNamespace(lidar_depth_max_cm=1200.0)
        self.llm_route_standoff_cm_var = SimpleNamespace(get=lambda: "850")
        self.llm_route_scan_spacing_cm_var = SimpleNamespace(get=lambda: "200")
        self.active_nbv_view_count_var = SimpleNamespace(get=lambda: "3")
        self.active_nbv_max_rescan_rounds_var = SimpleNamespace(get=lambda: "2")
        self.active_nbv_facade_threshold_var = SimpleNamespace(get=lambda: "0.75")
        self.active_nbv_mean_threshold_var = SimpleNamespace(get=lambda: "0.85")
        self.active_nbv_standoff_cm_var = SimpleNamespace(get=lambda: "850")
        self.active_nbv_initial_view_count_var = SimpleNamespace(get=lambda: "3")
        self.active_nbv_scan_z_cm_var = SimpleNamespace(get=lambda: "450")
        self.active_nbv_status_var = ValueVar("Active NBV: idle")
        self.active_nbv_stage_var = ValueVar("Stage: idle")
        self.active_nbv_target_var = ValueVar("Target: n/a")
        self.active_nbv_progress_var = ValueVar(0.0)
        self.active_nbv_progress_text_var = ValueVar("Progress: 0%")
        self.active_nbv_window = None
        self.active_nbv_thread = None
        self.active_nbv_stop_event = threading.Event()
        self.active_nbv_pause_event = threading.Event()
        self.active_nbv_state: Dict[str, Any] = {}
        self.active_nbv_output_dir = None
        self.active_nbv_preview_text = None
        self.active_nbv_map_text = None
        self.active_nbv_map_widget = None
        self.active_nbv_map_frame = None
        self.active_nbv_map_status_var = ValueVar("Active NBV Map: idle")
        self.llm_route5_representation_model_var = SimpleNamespace(get=lambda: str(PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"))
        self.llm_route5_oa3_plan_var = SimpleNamespace(get=lambda: str(PROJECT_ROOT / "obstacle_avoidance_3_data" / "plans" / "obstacle_avoidance_3_plans.json"))
        self.mixin_cls = mixin_cls

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.mixin_cls, name, None)
        if callable(attr):
            return lambda *args, **kwargs: attr(self, *args, **kwargs)
        raise AttributeError(name)

    def ensure_route5_state(self) -> None:
        return None

    def default_route5_or2_model_path(self) -> Path:
        return PROJECT_ROOT / "obstacle_representation_2_data" / "models" / "a_plus_2_model.pt"

    def default_route5_oa3_plan_path(self) -> Path:
        return PROJECT_ROOT / "obstacle_avoidance_3_data" / "plans" / "obstacle_avoidance_3_plans.json"

    def active_session(self) -> Any:
        return None

    def read_jsonl_artifact(self, path: Path) -> List[Dict[str, Any]]:
        if not Path(path).is_file():
            return []
        rows: List[Dict[str, Any]] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def route5_map_status_style(self, status: str) -> Dict[str, str]:
        colors = {
            "active": "#22d3ee",
            "captured": "#22c55e",
            "blocked": "#ef4444",
            "failed": "#ef4444",
            "planned": "#facc15",
            "pending": "#facc15",
        }
        return {"color": colors.get(str(status), "#ffd166"), "outline_color": "#111827"}

    def route5_json_safe(self, value: Any) -> Any:
        return json.loads(json.dumps(value, default=str))

    def house_world_bbox_for_id(self, house_id: str) -> Dict[str, float]:
        return {
            "min_x": 1000.0,
            "max_x": 1600.0,
            "min_y": 2000.0,
            "max_y": 2600.0,
            "center_x": 1300.0,
            "center_y": 2300.0,
        }

    def route_scan_standoff_cm(self) -> float:
        return 850.0

    def route_scan_spacing_cm(self) -> float:
        return 200.0

    def route2_facade_id(self, house_id: str, facade: str) -> str:
        return f"{house_id}_{facade}"

    def route2_facade_pose_from_axis(self, bbox: Dict[str, float], facade: str, axis_value: float, standoff: float, z_cm: float) -> Dict[str, float]:
        center_x = 0.5 * (float(bbox["min_x"]) + float(bbox["max_x"]))
        center_y = 0.5 * (float(bbox["min_y"]) + float(bbox["max_y"]))
        if facade == "south":
            x, y = float(axis_value), float(bbox["min_y"]) - standoff
        elif facade == "north":
            x, y = float(axis_value), float(bbox["max_y"]) + standoff
        elif facade == "east":
            x, y = float(bbox["max_x"]) + standoff, float(axis_value)
        else:
            x, y = float(bbox["min_x"]) - standoff, float(axis_value)
        yaw = math.degrees(math.atan2(center_y - y, center_x - x))
        return {"x": x, "y": y, "z": float(z_cm), "yaw_deg": yaw}

    def route3_target_pose_from_point(self, point: Dict[str, Any]) -> Dict[str, float]:
        return {
            "x": float(point.get("x", 0.0) or 0.0),
            "y": float(point.get("y", 0.0) or 0.0),
            "z": float(point.get("z", 0.0) or 0.0),
            "yaw": float(point.get("yaw_deg", point.get("yaw", 0.0)) or 0.0),
        }

    def active_nbv_route5_oa3_config(self) -> Dict[str, Any]:
        return self.mixin_cls.active_nbv_route5_oa3_config(self)

    def active_nbv_initial_scan_points(self, house_id: str) -> List[Dict[str, Any]]:
        return self.mixin_cls.active_nbv_initial_scan_points(self, house_id)

    def active_nbv_build_rescan_plan(self, house_id: str, coverage_report: Dict[str, Any], existing_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.mixin_cls.active_nbv_build_rescan_plan(self, house_id, coverage_report, existing_points)

    def active_nbv_score_view_candidate(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return self.mixin_cls.active_nbv_score_view_candidate(self, candidate, context)


def test_helper_contracts() -> None:
    module = load_module()
    mixin_cls = module.ActiveNBVScanControlMixin
    harness = Harness(mixin_cls)
    points = harness.active_nbv_initial_scan_points("002")
    assert_true(len(points) == 12, f"expected 12 sparse initial views, got {len(points)}")
    view_types = {str(item.get("view_type", "")) for item in points}
    assert_true({"face_center", "left_oblique", "right_oblique"}.issubset(view_types), f"missing sparse view types: {view_types}")
    assert_true(all(str(item.get("status", "")) == "planned" for item in points), "initial views must be planned")

    coverage_report = {
        "facades": {
            "south": {"point_cloud_coverage": 0.82, "scan_completion_ratio": 1.0},
            "east": {"point_cloud_coverage": 0.52, "scan_completion_ratio": 0.5},
            "north": {"point_cloud_coverage": 0.76, "scan_completion_ratio": 1.0},
            "west": {"point_cloud_coverage": 0.40, "scan_completion_ratio": 0.33},
        }
    }
    rescan_plan = harness.active_nbv_build_rescan_plan("002", coverage_report, points)
    rescan_points = rescan_plan.get("rescan_points", [])
    rescan_facades = {str(item.get("facade", "")) for item in rescan_points}
    assert_true(rescan_facades == {"east", "west"}, f"expected east/west rescan only, got {rescan_facades}")
    assert_true(all(str(item.get("view_type", "")) in {"hole_center_face_view", "oblique_edge_view", "high_center"} for item in rescan_points), "rescan views must use NBV view types")

    score = harness.active_nbv_score_view_candidate(
        {"scan_id": "002_east_rescan_000", "facade": "east"},
        {
            "coverage_gain": 0.12,
            "view_quality": 0.8,
            "safety_score": 0.9,
            "redundant_view_penalty": 0.1,
            "path_length_cost": 0.2,
            "obstacle_risk": 0.05,
        },
    )
    for key in ("coverage_gain", "view_quality", "safety_score", "redundant_view_penalty", "path_length_cost", "obstacle_risk", "reward"):
        assert_true(key in score, f"reward score missing {key}")

    config = harness.active_nbv_route5_oa3_config()
    assert_true(str(config.get("method", "")) == "active_nbv_or2_rule_v1", "active NBV must declare its OR2 method id")
    assert_true(str(config.get("or2_model_path", "")).endswith("a_plus_2_model.pt"), "default OR2 model path mismatch")
    assert_true(str(config.get("oa3_plan_path", "")).endswith("obstacle_avoidance_3_plans.json"), "default OA3 plan path mismatch")


def test_start_without_session_is_safe() -> None:
    module = load_module()
    harness = Harness(module.ActiveNBVScanControlMixin)
    harness.on_active_nbv_start()
    assert_true(
        "start a session first" in str(harness.active_nbv_status_var.get()),
        "Start without active session should show a status message instead of launching a worker",
    )


def test_active_nbv_window_has_scroll_wheel_support() -> None:
    source = (PROJECT_ROOT / "control" / "active_nbv_scan_control.py").read_text(encoding="utf-8")
    assert_true("active_nbv_scroll_canvas" in source, "Active NBV window should keep a scroll canvas reference")
    assert_true("ttk.Scrollbar" in source and "yscrollcommand" in source, "Active NBV window should have a vertical scrollbar")
    assert_true("<MouseWheel>" in source, "Active NBV window should bind Windows mouse wheel scrolling")
    assert_true("<Button-4>" in source and "<Button-5>" in source, "Active NBV window should bind Linux wheel scrolling")


def test_active_nbv_run_map_payload_helpers() -> None:
    module = load_module()
    harness = Harness(module.ActiveNBVScanControlMixin)
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        (output_dir / "scan_points.json").write_text(
            json.dumps(
                {
                    "scan_points": [
                        {"scan_id": "scan_a", "facade": "south", "x": 10, "y": 20, "status": "captured"},
                        {"scan_id": "scan_b", "facade": "east", "x": 30, "y": 40, "status": "blocked"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "trajectory.json").write_text(
            json.dumps({"trajectory": [{"pose": {"x": 1, "y": 2}}, {"actual_pose": {"x": 3, "y": 4}}]}),
            encoding="utf-8",
        )
        (output_dir / "route5_movement_trace.jsonl").write_text(
            json.dumps({"current_pose": {"x": 5, "y": 6}}) + "\n",
            encoding="utf-8",
        )
        (output_dir / "route5_navigation_plan.jsonl").write_text(
            json.dumps({"target_id": "scan_b", "plan": {"waypoints": [{"x": 7, "y": 8}]}}) + "\n",
            encoding="utf-8",
        )
        (output_dir / "route5_target_resets.jsonl").write_text(
            json.dumps({"reset_target_id": "reset_1", "reset_target_pose": {"x": 9, "y": 10}}) + "\n",
            encoding="utf-8",
        )
        (output_dir / "local_obstacle_summary.json").write_text(
            json.dumps(
                {
                    "occupied_voxel_count": 2,
                    "preview_points": [
                        {"x": 100.0, "y": 120.0, "z": 180.0, "confidence": 0.9, "semantic_hint": "tree_canopy_or_cluster"},
                        {"x": 125.0, "y": 120.0, "z": 180.0, "confidence": 0.7, "semantic_hint": "tree_canopy_or_cluster"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        points = harness.active_nbv_map_route_points(output_dir)
        labels = {str(item.get("label", "")) for item in points}
        assert_true({"scan_a", "scan_b", "scan_b_wp_1", "reset_1"}.issubset(labels), f"map labels missing: {labels}")
        by_label = {str(item.get("label", "")): item for item in points}
        assert_true(by_label["scan_a"]["status"] == "captured", by_label["scan_a"])
        assert_true(by_label["scan_b"]["status"] == "blocked", by_label["scan_b"])
        trajectory = harness.active_nbv_map_trajectory(output_dir)
        assert_true(len(trajectory) == 3, f"expected 3 trajectory points, got {trajectory}")
        obstacle_points = harness.active_nbv_map_obstacle_points(output_dir)
        assert_true(len(obstacle_points) == 2, f"expected local obstacle overlay points, got {obstacle_points}")
        assert_true(all(item.get("route_point_type") == "local_3d_obstacle" for item in obstacle_points), obstacle_points)


def test_required_public_methods_present() -> None:
    module = load_module()
    cls = module.ActiveNBVScanControlMixin
    for name in (
        "open_active_nbv_scan_window",
        "on_active_nbv_start",
        "on_active_nbv_stop",
        "on_active_nbv_clear",
        "on_active_nbv_validate",
        "active_nbv_initial_scan_points",
        "active_nbv_build_coverage_report",
        "active_nbv_build_rescan_plan",
        "active_nbv_score_view_candidate",
        "active_nbv_map_route_points",
        "active_nbv_map_trajectory",
        "active_nbv_map_obstacle_points",
        "refresh_active_nbv_map",
    ):
        assert_true(hasattr(cls, name), f"missing public method {name}")


def main() -> None:
    test_panel_button_wiring()
    test_required_public_methods_present()
    test_helper_contracts()
    test_start_without_session_is_safe()
    test_active_nbv_window_has_scroll_wheel_support()
    test_active_nbv_run_map_payload_helpers()
    print("verify_active_nbv_scan: OK")


if __name__ == "__main__":
    main()
