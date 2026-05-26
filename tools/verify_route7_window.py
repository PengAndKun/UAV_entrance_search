from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.route_control import RouteControlMixin
from control.route5_fusion_control import Route5FusionControlMixin


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class HarnessVar:
    def __init__(self, value: Any = "") -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


class Route2ZHarness(RouteControlMixin):
    def __init__(self, *, current_z_cm: float, override_z_cm: float | None = None) -> None:
        self.current_z_cm = float(current_z_cm)
        self.route_observation_z_override_cm = override_z_cm
        self.map_config = {"houses": []}
        self.captured_z_values: List[float] = []

    def load_map_resources(self, force: bool = False) -> bool:
        return True

    def house_world_bbox_for_id(self, _house_id: str) -> Dict[str, float]:
        return {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 800.0}

    def current_route_pose(self) -> Dict[str, float]:
        return {"x": -500.0, "y": -500.0, "z": self.current_z_cm}

    def scan_corridor_standoff_by_facade(self, _house_id: str, _bbox: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def route2_observation_standoff_cm(
        self,
        *,
        facade_length_cm: float,
        facade_depth_cm: float,
        facade_info: Dict[str, Any],
    ) -> tuple[float, Dict[str, Any]]:
        return 850.0, {"observation_standoff_mode": "verify"}

    def route2_projected_observation_choice(
        self,
        target_house_id: str,
        bbox: Dict[str, Any],
        facade: str,
        axis_value: float,
        z_cm: float,
        desired_standoff_cm: float,
        observation_meta: Dict[str, Any],
        facade_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.captured_z_values.append(float(z_cm))
        return {
            "pose": {"x": float(axis_value), "y": float(axis_value), "z": float(z_cm), "yaw_deg": 0.0},
            "standoff_cm": float(desired_standoff_cm),
            "axis_value": float(axis_value),
            "meta": dict(observation_meta),
        }

    def route2_facade_id(self, house_id: str, facade: str) -> str:
        return f"{house_id}_{facade}"

    def _as_float_or_none(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None


class Route7RunHarness(Route5FusionControlMixin):
    def __init__(self, root: Path) -> None:
        self.root_dir = root
        self.llm_route5_state: Dict[str, Any] = {}
        self.llm_route6_state: Dict[str, Any] = {}
        self.llm_route5_completed_facades = set()
        self.llm_route5_blocked_facades = set()
        self.llm_route7_map_layer_var = HarnessVar("z_300")
        self.llm_route7_map_status_var = HarnessVar("Route V7 Map: idle")

    def ensure_route5_state(self) -> None:
        if not isinstance(getattr(self, "llm_route5_state", None), dict):
            self.llm_route5_state = {}

    def resolve_project_path(self, value: str) -> Path:
        return self.root_dir / value

    def route5_nav_config(self) -> Dict[str, Any]:
        return {}

    def route5_sensing_config(self) -> Dict[str, Any]:
        return {}

    def route5_oa3_config(self) -> Dict[str, Any]:
        return {}

    def route5_write_state_artifact(self) -> None:
        return None

    def route5_log_event(self, output_dir: Path | None, event_type: str, payload: Dict[str, Any]) -> None:
        return None

    def route6_record_update_map_output_dir(self, output_dir: Path, *, source: str = "") -> bool:
        self.llm_route6_state["route6_update_map_output_dir"] = str(output_dir)
        self.llm_route6_state["output_dir"] = str(output_dir)
        self.llm_route6_state["route6_update_map_output_source"] = source
        return False


class Route7MapLayerScanHarness(Route7RunHarness, RouteControlMixin):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.args = type("Args", (), {"lidar_depth_max_cm": 1200.0, "lidar_depth_min_cm": 60.0})()
        self.map_config = {"houses": []}
        self.llm_route7_map_layer_var = HarnessVar("z_300")
        self.latest_state = {"pose": {"x": 500.0, "y": -700.0, "z": 300.0, "yaw": 90.0}}
        self.llm_route2_state = {
            "target_house_id": "002",
            "output_dir": str(root / "llm_route_7_fusion_runs" / "house_002_test"),
            "facade": "south",
            "observation_point": {"x": 500.0, "y": -700.0, "z": 300.0, "facade": "south", "standoff_cm": 650.0},
            "facade_analysis": {"floor_count_estimate": 1, "target_score": "medium"},
        }
        Path(self.llm_route2_state["output_dir"]).mkdir(parents=True, exist_ok=True)

    def house_world_bbox_for_id(self, _house_id: str) -> Dict[str, float]:
        return {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 800.0, "center_x": 500.0, "center_y": 400.0}

    def _as_float_or_none(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def route6_update_map_load_manifest(self, *, build_if_missing: bool = False, output_dir: Path | None = None) -> Dict[str, Any]:
        return {
            "schema": "route6_layered_occupancy_manifest_v1",
            "layers": [
                {
                    "schema": "route6_layered_occupancy_layer_artifact_v1",
                    "z_cm": 300.0,
                    "point_count": 120,
                    "occupied_cell_count": 30,
                    "occupancy_metadata_path": "",
                    "occupancy_grid_path": "",
                    "occupancy_preview_path": "",
                }
            ],
        }

    def route2_write_state_artifact(self) -> None:
        return None

    def write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def scan_point_validation_report(self, target_house_id: str, scan_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"valid": bool(scan_points), "scan_point_count": len(scan_points)}


class Route7MapRouteHarness(Route7MapLayerScanHarness):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.output_dir = Path(self.llm_route2_state["output_dir"])
        self.layer_dir = self.output_dir / "map" / "layered_occupancy" / "z_300"
        self.layer_dir.mkdir(parents=True, exist_ok=True)
        self.grid_path = self.layer_dir / "occupancy_grid.npy"
        self.metadata_path = self.layer_dir / "occupancy_grid.json"
        grid = np.zeros((21, 21), dtype=np.int16)
        grid[10, 4:17] = 100
        grid[10, 10] = 0
        np.save(self.grid_path, grid)
        metadata = {
            "width": 21,
            "height": 21,
            "resolution_m": 0.25,
            "origin_standard_m": [0.0, 0.0],
            "occupied_cell_count": int(np.sum(grid >= 100)),
        }
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def route6_update_map_load_manifest(self, *, build_if_missing: bool = False, output_dir: Path | None = None) -> Dict[str, Any]:
        return {
            "schema": "route6_layered_occupancy_manifest_v1",
            "layers": [
                {
                    "schema": "route6_layered_occupancy_layer_artifact_v1",
                    "z_cm": 300.0,
                    "point_count": 120,
                    "occupied_cell_count": 12,
                    "occupancy_metadata_path": str(self.metadata_path),
                    "occupancy_grid_path": str(self.grid_path),
                    "occupancy_preview_path": "",
                }
            ],
        }

    def route6_update_map_load_layer_metadata(self, layer_record: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(str(layer_record.get("occupancy_metadata_path", "") or ""))
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def route3_safety_report_for_pose(self, target_house_id: str, pose: Dict[str, Any]) -> Dict[str, Any]:
        return {"safe": True, "reason": "verify_route7_map_safe"}

    def selected_route_target_house_id(self) -> str:
        return "002"

    def _normalize_angle_deg(self, value: float) -> float:
        angle = (float(value) + 180.0) % 360.0 - 180.0
        return angle

    def query_local_3d_safety(
        self,
        current_pose: Dict[str, Any],
        payload: Dict[str, Any],
        config: Dict[str, Any] | None = None,
        *,
        output_dir: Path | None = None,
        stage: str = "",
        target_id: str = "",
        candidate_direction: str = "",
    ) -> Dict[str, Any]:
        direction = str(candidate_direction or payload.get("action_name", "") or "")
        if "forward" in direction:
            return {
                "checked": True,
                "safe": False,
                "reason": f"local_3d_occupancy_blocked:{candidate_direction or 'forward'}",
                "blocked_directions": ["forward"],
            }
        return {"checked": True, "safe": True, "reason": "local_3d_clear", "blocked_directions": []}

    def append_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_route2_observation_z_override_for_route7() -> None:
    baseline = Route2ZHarness(current_z_cm=450.0)
    baseline_candidates = baseline.route2_safe_observation_candidates("001", skip_completed=False)
    assert_true(baseline_candidates, "baseline should produce facade candidates")
    assert_true(set(round(value, 3) for value in baseline.captured_z_values) == {450.0}, f"baseline should use current z: {baseline.captured_z_values}")

    override = Route2ZHarness(current_z_cm=450.0, override_z_cm=300.0)
    override_candidates = override.route2_safe_observation_candidates("001", skip_completed=False)
    assert_true(override_candidates, "override should still produce facade candidates")
    assert_true(set(round(value, 3) for value in override.captured_z_values) == {300.0}, f"Window 7 should force 300cm candidates: {override.captured_z_values}")


def test_route7_run_directories_are_dedicated_and_fresh(tmp_dir: Path) -> None:
    harness = Route7RunHarness(tmp_dir)
    first = harness.make_route7_fused_output_dir("002")
    second = harness.make_route7_fused_output_dir("002")
    root = tmp_dir / "llm_route_7_fusion_runs"
    assert_true(first.parent == root, f"V7 run should live under llm_route_7_fusion_runs: {first}")
    assert_true(first != second, f"each V7 start needs a fresh run directory: first={first} second={second}")
    assert_true(first.name.startswith("house_002_autosearch_v7_or2_fused_"), f"V7 run name should identify v7: {first.name}")
    for subdir in ("frames", "reconstruction", "facade_observations", "map"):
        assert_true((first / subdir).is_dir(), f"V7 run should create {subdir}: {first}")

    selected = harness.route7_prepare_new_map_output_dir("002")
    assert_true(selected.parent == root, f"prepared V7 map dir should use dedicated root: {selected}")
    assert_true(harness.llm_route5_state.get("output_dir") == str(selected), harness.llm_route5_state)
    assert_true(harness.llm_route6_state.get("route6_update_map_output_dir") == str(selected), harness.llm_route6_state)
    assert_true(harness.llm_route6_state.get("output_dir") == str(selected), harness.llm_route6_state)


def test_route7_map_layer_edge_scan_uses_two_house_edge_points(tmp_dir: Path) -> None:
    harness = Route7MapLayerScanHarness(tmp_dir)
    plan = harness.route7_plan_facade_map_layer_scan_current()
    points = [point for point in plan.get("points", []) if isinstance(point, dict)]
    assert_true(len(points) == 2, f"V7 should plan two map-layer edge capture points per facade: {points}")
    assert_true({point.get("route_point_type") for point in points} == {"map_layer_edge_capture"}, points)
    assert_true({point.get("view_type") for point in points} == {"route7_map_layer_edge_capture"}, points)
    assert_true({round(float(point.get("z", 0.0)), 3) for point in points} == {300.0}, points)
    assert_true({point.get("route7_map_layer_key") for point in points} == {"z_300"}, points)
    axis_values = sorted(round(float(point.get("axis_value", 0.0)), 2) for point in points)
    assert_true(axis_values[0] < 250.0 and axis_values[1] > 750.0, f"V7 points should sit near both ends of the house side: {axis_values}")
    assert_true(all(str(point.get("yaw_source", "")) == "route7_face_house_edge" for point in points), points)


def test_route7_map_route_planner_uses_layered_occupancy(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    start = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 450.0, "y": -450.0, "z": 300.0, "yaw": 90.0}
    plan = harness.route7_plan_navigation_waypoints_from_map(
        start,
        target,
        output_dir=harness.output_dir,
        stage="NAV_TO_SCAN_POINT",
        target_id="002_south_z_300_edge_001",
        target_house_id="002",
    )
    assert_true(plan.get("status") == "ok", f"V7 map planner should find a route through z_300 grid: {plan}")
    assert_true(plan.get("planner_source") == "route7_layered_occupancy_astar", plan)
    assert_true(plan.get("route7_map_route_replan_policy") == "map_first_runtime_revalidate", plan)
    assert_true(len([item for item in plan.get("waypoints", []) if isinstance(item, dict)]) >= 1, plan)
    raw_cells = [tuple(item) for item in plan.get("raw_cells", []) if isinstance(item, list) and len(item) == 2]
    occupied_wall_cells = {(10, col) for col in range(4, 17) if col != 10}
    assert_true(not any(cell in occupied_wall_cells for cell in raw_cells), f"planned route should avoid occupied wall cells: {raw_cells}")
    assert_true(any(row == 10 and col >= 17 for row, col in raw_cells), f"planned route should detour around the map wall: {raw_cells}")


def test_route7_local_3d_soft_block_requests_map_replan(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 450.0, "y": -450.0, "z": 300.0, "yaw": 90.0}
    local_3d = {"safe": False, "reason": "local_3d_occupancy_blocked:forward", "blocked_directions": ["forward"]}
    gate = {
        "front_risk_state": "obstacle_warning",
        "front_min_depth_cm": 281.75,
        "can_forward": True,
        "must_stop": False,
    }
    decision = harness.route7_local_3d_replan_decision(
        current,
        target,
        {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_or2_forward"},
        local_3d,
        gate,
        output_dir=harness.output_dir,
        stage="NAV_TO_SCAN_POINT",
        target_id="002_south_z_300_edge_001",
    )
    assert_true(decision.get("action") == "replan", f"soft local 3D block should request map-route replan, not stop: {decision}")
    assert_true(decision.get("route7_map_route_replan_policy") == "map_first_runtime_revalidate", decision)
    assert_true(float(decision.get("front_min_depth_cm", 0.0) or 0.0) > 250.0, decision)


def test_route7_soft_warning_does_not_select_backoff(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 150.0, "y": -150.0, "z": 300.0, "yaw": 90.0}
    nominal = {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav_lookahead"}
    gate = {
        "front_risk_state": "clearance_warning",
        "front_min_depth_cm": 301.0,
        "can_forward": True,
        "must_stop": False,
        "selected_direction": "forward",
        "reason": "verify_soft_warning",
    }
    alternative = harness.route5_apply_or2_safe_alternative(
        event={
            "route5_stage": "NAV_TO_SCAN_POINT",
            "target_id": "002_south_z_300_edge_001",
            "target_house_id": "002",
            "route5_output_dir": str(harness.output_dir),
            "pointcloud_summary": {"available": True, "front_min_depth_cm": 301.0, "forward_swept_clear": False},
        },
        gate=gate,
        rule={"selected_direction": "forward", "candidate_action_scores": {"forward": 1.0, "slow_forward": 0.9, "backoff": 0.2}},
        selected_direction="forward",
        selected_payload={"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_or2_forward"},
        nominal_payload=nominal,
        config={"nav_step_cm": 20.0},
        current_pose=current,
        target_pose=target,
    )
    assert_true(alternative.get("direction") in {"forward", "slow_forward"}, f"V7 soft warning should continue the map route instead of backoff: {alternative}")
    assert_true(alternative.get("safe_alternative_action") != "backoff", alternative)
    assert_true((alternative.get("route7_soft_obstacle_policy") or {}).get("mode") == "continue_planned_route", alternative)


def test_route7_frame_decision_log_and_trajectory_visualization(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    start = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 450.0, "y": -450.0, "z": 300.0, "yaw": 90.0}
    plan = harness.route7_plan_navigation_waypoints_from_map(
        start,
        target,
        output_dir=harness.output_dir,
        stage="NAV_TO_SCAN_POINT",
        target_id="002_south_z_300_edge_001",
        target_house_id="002",
    )
    visual = harness.route7_write_navigation_plan_visualization(
        harness.output_dir,
        plan,
        current_pose=start,
        target_pose=target,
        target_id="002_south_z_300_edge_001",
    )
    assert_true(Path(str(visual.get("visualization_path", ""))).is_file(), f"V7 should write a planned trajectory preview: {visual}")

    harness.route5_write_frame_decision(
        harness.output_dir,
        {
            "frame_id": 1,
            "route_window_label": "V7",
            "route5_stage": "NAV_TO_SCAN_POINT",
            "target_id": "002_south_z_300_edge_001",
            "current_pose": start,
            "target_waypoint": target,
            "or2_prediction": {"front_risk_state": "clearance_warning", "can_forward": True, "must_stop": False},
            "or2_rule": {"selected_direction": "forward", "reason": "verify"},
            "avoidance_gate": {"front_min_depth_cm": 301.0, "front_risk_state": "clearance_warning", "reason": "verify"},
            "selected_action": "route5_or2_slow_forward",
            "selected_action_reason": "route7_soft_obstacle_not_deep_red_continue_planned_route",
            "selected_action_payload": {"forward_cm": 7.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
            "route7_soft_obstacle_policy": {"mode": "continue_planned_route"},
        },
        final_payload={"forward_cm": 7.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0},
    )
    log_path = harness.output_dir / "route7_frame_decision_log.jsonl"
    assert_true(log_path.is_file(), "V7 should write a concise per-frame decision log")
    logged = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert_true(logged.get("selected_action") == "route5_or2_slow_forward", logged)
    assert_true("reason" in logged and logged["reason"], logged)


def test_window7_source_contract() -> None:
    panel_source = (PROJECT_ROOT / "control" / "panel.py").read_text(encoding="utf-8")
    route5_source = (PROJECT_ROOT / "control" / "route5_fusion_control.py").read_text(encoding="utf-8")

    assert_true("Open LLM Route Window 7" in panel_source, "main panel should expose Open LLM Route Window 7")
    assert_true("command=self.open_llm_route_window7" in panel_source, "Window 7 button should call open_llm_route_window7")

    for token in (
        "def ensure_route7_state",
        "def route7_default_exploration_z_cm",
        "def refresh_llm_route7_update_map",
        "def make_route7_fused_output_dir",
        "def route7_prepare_new_map_output_dir",
        "def route7_start_update_map_realtime",
        "def route7_plan_facade_map_layer_scan_current",
        "def route7_build_update_map_after_observation",
        "def route7_plan_navigation_waypoints_from_map",
        "def route7_local_3d_replan_decision",
        "def on_route7_stop",
        "def open_llm_route_window7",
        "def on_route7_start_fused_search",
        "def on_route7_step_facade",
    ):
        assert_true(token in route5_source, f"Route 5 mixin should define Window 7 contract token: {token}")

    refresh_start = route5_source.find("def refresh_llm_route7_update_map")
    refresh_end = route5_source.find("def open_llm_route_window7")
    refresh_block = route5_source[refresh_start:refresh_end]
    assert_true("route6_update_map_load_manifest" in refresh_block, "Window 7 map should read Route 6 Update Map layered manifest")
    assert_true("route6_update_map_layer_preview_path" in refresh_block, "Window 7 map should use Route 6 Update Map layer previews")
    assert_true("route6_draw_update_map_uav_overlay" in refresh_block, "Window 7 map should reuse Route 6 UAV overlay drawing")

    window7_start = route5_source.find("def open_llm_route_window7")
    window7_end = route5_source.find("def on_route7_start_fused_search")
    window7_block = route5_source[window7_start:window7_end]
    assert_true("LLM House Entrance Route V7" in window7_block, "Window 7 title should identify V7")
    assert_true("Route 6 Update Map" in window7_block, "Window 7 map frame should be Route 6 Update Map based")
    assert_true("z_300" in window7_block or "route7_default_layer_key" in window7_block, "Window 7 should default its map layer to 300cm")

    start_block = route5_source[route5_source.find("def on_route7_start_fused_search"):route5_source.find("def on_route7_step_facade")]
    assert_true("observation_z_cm=self.route7_default_exploration_z_cm()" in start_block, "Window 7 start should force 300cm exploration height")
    assert_true("route7_prepare_new_map_output_dir" in start_block, "Window 7 start should prepare a fresh dedicated map run")
    assert_true("route7_start_update_map_realtime" in start_block, "Window 7 start should start Route 6 Update Map realtime")

    worker_block = route5_source[route5_source.find("def route5_full_search_worker"):route5_source.find("def refresh_route5_preview")]
    assert_true("route7_build_update_map_after_observation" in worker_block, "Window 7 should build/update the obstacle map after observation")
    assert_true("route7_plan_facade_map_layer_scan_current" in worker_block, "Window 7 should plan scan captures from the selected map layer")

    navigate_block = route5_source[route5_source.find("def route5_navigate_to_pose_with_fusion"):route5_source.find("def route5_run_summary")]
    assert_true("route7_should_use_map_route_planner" in navigate_block, "V7 navigation should choose the map-route planner")
    assert_true("route7_plan_navigation_waypoints_from_map" in navigate_block, "V7 navigation should plan from Route 6 layered occupancy")
    follow_block = route5_source[route5_source.find("def route5_follow_navigation_waypoint_with_fusion"):route5_source.find("def route5_navigate_to_pose_with_fusion")]
    assert_true("route7_local_3d_replan_decision" in follow_block, "V7 local 3D blocks should be revalidated against map-route distance")
    assert_true("route7_map_route_replan_required" in follow_block, "V7 should replan instead of hard-stopping on soft local 3D blocks")
    assert_true("route7_frame_decision_log.jsonl" in route5_source, "V7 should write a concise per-frame decision log")
    assert_true("route7_write_navigation_plan_visualization" in route5_source, "V7 should write planned trajectory visualization artifacts")

    ui_block = route5_source[route5_source.find("def _build_llm_route5_section"):route5_source.find("def route7_update_map_layer_values")]
    assert_true("route5_fixed_status_label" in ui_block, "Route status labels should use fixed layout containers")
    assert_true("grid_propagate(False)" in ui_block, "Route status/preview frames should not resize as text changes")

    stop_block = route5_source[route5_source.find("def on_route7_stop"):route5_source.find("def open_llm_route_window5")]
    assert_true("route6_update_map_realtime_stop_event.set()" in stop_block, "Window 7 stop should stop realtime map updates")
    assert_true("route6_update_map_capture_stop_event.set()" in stop_block, "Window 7 stop should stop update-map capture")
    assert_true("on_route7_stop" in window7_block, "Window 7 stop button should call V7 stop-all handler")


def main() -> None:
    import tempfile

    test_route2_observation_z_override_for_route7()
    with tempfile.TemporaryDirectory(prefix="route7_verify_") as raw:
        test_route7_run_directories_are_dedicated_and_fresh(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_scan_verify_") as raw:
        test_route7_map_layer_edge_scan_uses_two_house_edge_points(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_map_route_verify_") as raw:
        test_route7_map_route_planner_uses_layered_occupancy(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_soft_block_verify_") as raw:
        test_route7_local_3d_soft_block_requests_map_replan(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_soft_warning_verify_") as raw:
        test_route7_soft_warning_does_not_select_backoff(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_frame_log_verify_") as raw:
        test_route7_frame_decision_log_and_trajectory_visualization(Path(raw))
    test_window7_source_contract()
    print("PASS route7 window verification")


if __name__ == "__main__":
    main()
