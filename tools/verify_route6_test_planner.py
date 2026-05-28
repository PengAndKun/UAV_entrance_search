from __future__ import annotations

import importlib
import json
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


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


def panel_source() -> str:
    return (PROJECT_ROOT / "control" / "panel.py").read_text(encoding="utf-8")


def load_route6_module() -> Any:
    return importlib.import_module("control.route6_explore_control")


def load_builder() -> Any:
    return importlib.import_module("control.route6_map_builder")


def sample_layered_cloud() -> np.ndarray:
    points: List[List[float]] = []
    for x in np.linspace(9.8, 16.2, 20):
        for y in np.linspace(-5.2, 5.2, 20):
            points.append([float(x), float(-y / 100.0), 0.5, 20.0, 20.0, 20.0])
    return np.asarray(points, dtype=np.float32)


class PlannerHarness:
    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = tmp_dir
        self.latest_state = {"pose": [0.0, 0.0, 450.0, 0.0]}
        self.map_config = {
            "houses": [
                {
                    "id": "001",
                    "name": "House_001",
                    "bbox": {"min_x": 1000.0, "max_x": 1600.0, "min_y": -500.0, "max_y": 500.0},
                },
                {
                    "id": "002",
                    "name": "House_002",
                    "bbox": {"min_x": 2600.0, "max_x": 3200.0, "min_y": -400.0, "max_y": 400.0},
                },
            ]
        }
        self.llm_route6_state: Dict[str, Any] = {}
        self.llm_route6_status_var = ValueVar("LLM Route V6: idle")
        self.llm_route6_stage_var = ValueVar("Stage: idle")
        self.llm_route6_current_house_var = ValueVar("Current house: n/a")
        self.llm_route6_queue_var = ValueVar("House queue: n/a")
        self.llm_route6_map_status_var = ValueVar("Map: idle")
        self.llm_route6_output_dir_var = ValueVar("Output: n/a")
        self.llm_route6_metrics_var = ValueVar("Metrics: mapped=0 searched=0 blocked=0 confidence=n/a corrected=n/a")
        self.llm_route6_max_houses_var = ValueVar("3")
        self.llm_route6_runtime_min_var = ValueVar("30")
        self.llm_route6_standoff_cm_var = ValueVar("850")
        self.llm_route6_scan_z_cm_var = ValueVar("450")
        self.llm_route6_occupancy_resolution_m_var = ValueVar("0.25")
        self.llm_route6_coverage_threshold_var = ValueVar("0.75")
        self.llm_route6_allow_save_corrected_var = ValueVar(False)
        self.llm_route6_task_prompt_var = ValueVar("Search selected house entrance.")
        self.llm_route6_selected_target_var = ValueVar("Selected target: n/a")
        self.llm_route6_realtime_map_status_var = ValueVar("Realtime map: idle")
        self.llm_route6_map_analysis_status_var = ValueVar("LLM Map Analysis: idle")
        self.llm_route6_map_analysis_detail_var = ValueVar("Semantic target: n/a")
        self.llm_route6_navigation_target_var = ValueVar("Navigation target: n/a")
        self.llm_route6_visual_status_var = ValueVar("LLM Visual Direction Analysis: idle")
        self.llm_route6_visual_detail_var = ValueVar("Visual target: n/a")
        self.llm_route6_conflict_status_var = ValueVar("Height Conflict / Replan: idle")
        self.llm_route6_conflict_detail_var = ValueVar("Conflict detail: n/a")
        self.llm_route6_or_status_var = ValueVar("OR Avoidance: idle")
        self.llm_route6_or_detail_var = ValueVar("OR detail: n/a")
        self.route6_update_map_layer_var = ValueVar("z_050")
        self.route6_update_map_status_var = ValueVar("Route 6 Update Map: idle")
        self.route6_update_map_pose_var = ValueVar("UAV x=n/a y=n/a z=n/a yaw=n/a")
        self.route6_update_map_capture_interval_s_var = ValueVar("1.0")
        self.route6_update_map_min_move_cm_var = ValueVar("50")
        self.route6_update_map_min_yaw_deg_var = ValueVar("5")
        self.route6_test_planner_radar_distance_cm_var = ValueVar("300")
        self.route6_test_planner_scan_z_cm_var = ValueVar("450")
        self.route6_test_planner_edge_var = ValueVar("auto nearest")
        self.route6_test_planner_algorithm_var = ValueVar("nearest edge")
        self.route6_test_planner_scan_mode_var = ValueVar("single best point")
        self.route6_test_planner_fov_deg_var = ValueVar("60")
        self.route6_test_planner_overlap_var = ValueVar("0.30")
        self.route6_test_planner_coverage_threshold_var = ValueVar("0.90")
        self.route6_test_planner_reset_step_cm_var = ValueVar("150")
        self.route6_test_planner_status_var = ValueVar("Route 6 Test Planner: idle")
        self.route6_test_planner_house_listbox = None
        self.route6_test_planner_result_text = None
        self.route6_test_planner_preview_label = None
        self.route6_test_planner_preview_photo = None
        self.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
        self.llm_route6_thread = None
        self.llm_route6_stop_event = threading.Event()
        self.llm_route6_pause_event = threading.Event()
        self.llm_route6_force_next_event = threading.Event()

    def __getattr__(self, name: str):
        module = load_route6_module()
        attr = getattr(module.Route6ExploreControlMixin, name, None)
        if callable(attr):
            return lambda *args, **kwargs: attr(self, *args, **kwargs)
        raise AttributeError(name)

    def route6_output_root(self) -> Path:
        return Path(self.llm_route6_output_root_override)

    def route6_current_pose(self) -> Dict[str, float]:
        return {"x": 0.0, "y": 0.0, "z": 450.0, "yaw": 0.0}

    def route6_known_house_polygon_records(self) -> List[Dict[str, Any]]:
        return []

    def route6_get_runtime_map_config(self) -> Dict[str, Any]:
        return self.map_config

    def route6_write_state_artifact(self) -> None:
        return None

    def route6_json_safe(self, payload: Any) -> Any:
        module = load_route6_module()
        return module.Route6ExploreControlMixin.route6_json_safe(self, payload)

    def route6_write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.route6_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_map(harness: PlannerHarness) -> Path:
    builder = load_builder()
    output_dir = harness.route6_output_root() / "route6_update_map_test_planner"
    result = builder.write_route6_layered_occupancy_artifacts(output_dir, sample_layered_cloud(), resolution_m=0.25)
    harness.llm_route6_state["route6_update_map_output_dir"] = str(output_dir)
    harness.llm_route6_state["route6_update_map"] = {
        "output_dir": str(output_dir),
        "manifest_path": str(result["manifest_path"]),
    }
    harness.llm_route6_state["output_dir"] = str(output_dir)
    return output_dir


def add_layer_blocking_column(harness: PlannerHarness, output_dir: Path, layer_key: str, x_cm: float, y_min_cm: float, y_max_cm: float) -> None:
    manifest = harness.route6_update_map_load_manifest(build_if_missing=False, output_dir=output_dir)
    layer = next(
        (
            item
            for item in manifest.get("layers", [])
            if isinstance(item, dict) and harness._route6_update_map_layer_key(item) == layer_key
        ),
        {},
    )
    assert_true(bool(layer), f"test setup should find layer {layer_key}: {manifest}")
    metadata = harness.route6_update_map_load_layer_metadata(layer)
    grid_path = Path(str(layer.get("occupancy_grid_path", "") or ""))
    grid = np.asarray(np.load(grid_path), dtype=np.int16)
    for y_cm in np.linspace(float(y_min_cm), float(y_max_cm), 120):
        index = harness.route6_layer_grid_index_for_point(metadata, float(x_cm), float(y_cm))
        if not bool(index.get("in_bounds", False)):
            continue
        row = int(index.get("row", 0) or 0)
        col = int(index.get("col", 0) or 0)
        for dc in (-1, 0, 1):
            if 0 <= row < grid.shape[0] and 0 <= col + dc < grid.shape[1]:
                grid[row, col + dc] = 100
    np.save(grid_path, grid)


def test_panel_route6_test_planner_button_wiring() -> None:
    source = panel_source()
    assert_true(
        "Route 6 Test Planner" in source,
        "main Route6_entrance_search row should expose the Route 6 Test Planner button",
    )
    assert_true(
        "command=self.open_route6_test_planner_window" in source,
        "Route 6 Test Planner button must call open_route6_test_planner_window",
    )


def test_route6_test_planner_mixin_contract() -> None:
    module = load_route6_module()
    mixin = module.Route6ExploreControlMixin
    for method_name in (
        "open_route6_test_planner_window",
        "close_route6_test_planner_window",
        "open_route6_test_planner_formula_window",
        "route6_test_planner_formula_payload",
        "route6_build_offline_test_plan",
        "route6_draw_offline_test_plan_overlay",
        "route6_test_planner_visual_calculation_records",
        "refresh_route6_test_planner_preview",
        "on_route6_test_planner_show_formula",
        "on_route6_test_planner_analyze",
        "route6_reset_current_observation_point",
        "on_route6_test_planner_reset_current_observation_point",
    ):
        assert_true(callable(getattr(mixin, method_name, None)), f"Route 6 planner must provide {method_name}()")


def test_route6_test_planner_analysis_package_contract() -> None:
    analysis_module = importlib.import_module("control.route6_test_planner_analysis")
    for function_name in (
        "active_reset_target",
        "anchor_for_point",
        "reset_summary",
        "visual_calculation_records",
        "visual_formula",
    ):
        assert_true(callable(getattr(analysis_module, function_name, None)), f"analysis package must export {function_name}()")


def test_route6_test_planner_formula_button_and_payload() -> None:
    source = (PROJECT_ROOT / "control" / "route6_explore_control.py").read_text(encoding="utf-8")
    assert_true("Show Formula" in source, "Route 6 planner window should expose a Show Formula button")
    assert_true("route6_test_planner_edge_var" in source, "Route 6 planner should expose an edge selector variable")
    assert_true("auto nearest" in source and "south" in source and "west" in source, "edge selector should include auto/south/east/north/west")
    assert_true("route6_test_planner_algorithm_var" in source, "Route 6 planner should expose an algorithm selector variable")
    assert_true(
        "nearest edge" in source and "frontier-based" in source and "surface edge explorer" in source,
        "algorithm selector should include the paper-inspired planner options",
    )
    assert_true("route6_test_planner_scan_mode_var" in source, "Route 6 planner should expose a scan mode selector variable")
    assert_true("Scan Mode" in source and "FOV deg" in source and "Overlap" in source and "Coverage" in source, "scan coverage controls should be visible")
    assert_true("Reset Current Observation Point" in source, "Route 6 planner window should expose an observation reset button")
    assert_true("route6_test_planner_scroll_canvas" in source, "Route 6 planner window should be scrollable")
    assert_true("route6_test_planner_analysis" in source, "Route 6 planner analysis should live in a separate package")
    assert_true(
        "command=self.on_route6_test_planner_show_formula" in source,
        "Show Formula button should render the observation-point formula payload",
    )
    assert_true(
        "Route 6 Formula / Code Sources" in source,
        "Show Formula should open a dedicated formula/code-source window",
    )
    module = load_route6_module()
    with tempfile.TemporaryDirectory(prefix="route6_formula_") as raw:
        harness = PlannerHarness(Path(raw))
        payload = module.Route6ExploreControlMixin.route6_test_planner_formula_payload(harness)
    assert_true(payload["schema"] == "route6_offline_test_planner_formula_v1", f"unexpected formula schema: {payload}")
    text = "\n".join(payload.get("formulas", []))
    for item in ("t = clamp", "P_edge", "P_obs", "yaw_deg"):
        assert_true(item in text, f"formula payload should include {item}: {payload}")
    for source_ref in payload.get("code_sources", []):
        assert_true("file" in source_ref and "symbol" in source_ref, f"code source should include file and symbol: {source_ref}")


def test_route6_offline_planner_selects_nearest_edge_and_writes_artifact(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)

    assert_true(plan["schema"] == "route6_offline_test_plan_v1", f"unexpected schema: {plan}")
    assert_true(plan["selected_house_ids"] == ["001"], f"selected house should be preserved: {plan}")
    assert_true(plan["selected_observation_point"]["house_id"] == "001", f"house 001 should be selected: {plan}")
    assert_true(plan["selected_observation_point"]["edge"] == "west", f"west edge should be nearest to pose: {plan}")
    assert_true(abs(float(plan["selected_observation_point"]["x"]) - 700.0) < 1e-6, f"observation point should be radar distance outside west edge: {plan}")
    assert_true(abs(float(plan["selected_observation_point"]["y"]) - 0.0) < 1e-6, f"observation point should align with nearest edge point: {plan}")
    assert_true(Path(plan["artifact_path"]).is_file(), f"offline plan artifact should be written: {plan}")
    payload = json.loads(Path(plan["artifact_path"]).read_text(encoding="utf-8"))
    assert_true(payload["selected_observation_point"]["edge"] == "west", f"artifact should contain selected edge: {payload}")
    analysis_dir = output_dir / "route6_test_planner_analysis"
    visual_path = analysis_dir / "visual_calculation_records.json"
    assert_true(visual_path.is_file(), f"visual calculation records should be split into analysis folder: {visual_path}")
    visual_payload = json.loads(visual_path.read_text(encoding="utf-8"))
    assert_true(isinstance(visual_payload.get("records", []), list), f"visual analysis artifact should expose records: {visual_payload}")


def test_route6_offline_planner_uses_operator_selected_edge(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_test_planner_edge_var.set("north")

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)

    point = plan["selected_observation_point"]
    assert_true(point["edge"] == "north", f"operator-selected north edge should override nearest-edge mode: {plan}")
    assert_true(abs(float(point["x"]) - 1000.0) < 1e-6, f"north observation should keep nearest point x on selected edge: {plan}")
    assert_true(abs(float(point["y"]) - 800.0) < 1e-6, f"north observation should offset from bbox.max_y by radar distance: {plan}")
    assert_true(plan["edge_selection_mode"] == "operator_selected_edge", f"plan should report manual edge mode: {plan}")
    assert_true(plan["requested_edge"] == "north", f"plan should preserve requested edge: {plan}")


def test_route6_offline_planner_uses_surface_edge_explorer_center_anchor(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_test_planner_edge_var.set("north")
    harness.route6_test_planner_algorithm_var.set("surface edge explorer")

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)

    point = plan["selected_observation_point"]
    selected_edge = plan["house_plans"][0]["selected_edge"]
    assert_true(plan["planning_algorithm"] == "surface_edge_explorer", f"plan should preserve selected algorithm: {plan}")
    assert_true(point["edge"] == "north", f"manual edge should still be respected: {plan}")
    assert_true(abs(float(point["x"]) - 1300.0) < 1e-6, f"surface-edge explorer should use the edge center anchor: {plan}")
    assert_true(abs(float(point["y"]) - 800.0) < 1e-6, f"surface-edge explorer should keep radar standoff from selected edge: {plan}")
    assert_true(point["algorithm"] == "surface_edge_explorer", f"observation point should include algorithm: {plan}")
    assert_true(
        selected_edge["algorithm_components"]["anchor_policy"] == "edge_center_surface_standoff",
        f"selected edge should report the surface-edge calculation: {selected_edge}",
    )


def test_route6_offline_planner_reports_nbv_information_gain_components(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_test_planner_algorithm_var.set("nbv information gain")

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)

    selected_edge = plan["house_plans"][0]["selected_edge"]
    assert_true(plan["planning_algorithm"] == "nbv_information_gain", f"plan should preserve NBV algorithm: {plan}")
    assert_true("algorithm_score" in selected_edge, f"selected edge should expose an algorithm score: {selected_edge}")
    assert_true(
        "expected_information_gain_cm" in selected_edge.get("algorithm_components", {}),
        f"NBV edge should expose information-gain components: {selected_edge}",
    )
    assert_true(
        plan["selected_observation_point"]["algorithm"] == "nbv_information_gain",
        f"selected observation point should include algorithm: {plan}",
    )


def test_route6_offline_planner_generates_multi_point_scan_coverage(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_test_planner_edge_var.set("north")
    harness.route6_test_planner_scan_mode_var.set("multi-point edge coverage")
    harness.route6_test_planner_fov_deg_var.set("60")
    harness.route6_test_planner_overlap_var.set("0.30")
    harness.route6_test_planner_coverage_threshold_var.set("0.90")

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)

    assert_true(plan["scan_mode"] == "multi_point_edge_coverage", f"plan should preserve scan mode: {plan}")
    assert_true(plan["scan_satisfied"] is True, f"coverage should satisfy the threshold: {plan}")
    selected_points = plan["selected_scan_observation_points"]
    assert_true(len(selected_points) >= 2, f"long selected edge should need multiple scan points: {plan}")
    assert_true(all(item["edge"] == "north" for item in selected_points), f"scan points should target the selected edge: {selected_points}")
    assert_true(abs(float(selected_points[0]["y"]) - 800.0) < 1e-6, f"scan points should keep radar standoff outside north edge: {selected_points}")
    assert_true(
        plan["scan_coverage_summary"]["coverage_width_cm"] > 300.0,
        f"coverage summary should include FOV-derived width: {plan}",
    )
    assert_true(
        plan["scan_coverage_summary"]["coverage_ratio"] >= 0.90,
        f"coverage ratio should meet threshold: {plan}",
    )
    house_scan = plan["house_plans"][0]["scan_plan"]
    assert_true(house_scan["scan_satisfied"] is True, f"house scan plan should report satisfied coverage: {house_scan}")
    assert_true(len(house_scan["scan_observation_points"]) == len(selected_points), f"selected scan points should match house scan plan: {plan}")


def test_route6_north_scan_points_are_centered_inside_edge_span(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    harness.map_config["houses"][1]["bbox"] = {
        "min_x": -250.0,
        "max_x": 1680.0,
        "min_y": 1450.0,
        "max_y": 2200.0,
    }
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_test_planner_edge_var.set("north")
    harness.route6_test_planner_scan_mode_var.set("multi-point edge coverage")
    harness.route6_test_planner_fov_deg_var.set("60")
    harness.route6_test_planner_overlap_var.set("0.30")
    harness.route6_test_planner_coverage_threshold_var.set("0.90")

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["002"], radar_distance_cm=850.0)

    bbox = plan["house_plans"][0]["bbox"]
    points = plan["selected_scan_observation_points"]
    assert_true(len(points) == 3, f"house 002 north should use three centered scan points with these parameters: {plan}")
    for point in points:
        anchor = point["anchor_point_cm"]
        assert_true(float(bbox["min_x"]) < float(anchor["x"]) < float(bbox["max_x"]), f"anchor should be inside edge span, not at endpoint: {point}")
        assert_true(float(bbox["min_x"]) < float(point["x"]) < float(bbox["max_x"]), f"north scan point should stay inside the edge x span: {point}")
        assert_true(abs(float(point["y"]) - (float(bbox["max_y"]) + 850.0)) < 1e-6, f"north scan point should keep constant standoff: {point}")
    progresses = [float(point["edge_progress_cm"]) for point in points]
    assert_true(min(progresses) > 0.0 and max(progresses) < float(plan["house_plans"][0]["scan_plan"]["edge_length_cm"]), f"scan progress should avoid endpoints: {progresses}")


def test_route6_reset_reports_non_blocking_problem_without_moving_point(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    harness.map_config["houses"].append(
        {
            "id": "003",
            "name": "Blocking_Room",
            "bbox": {"min_x": 620.0, "max_x": 760.0, "min_y": -120.0, "max_y": 120.0},
        }
    )
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)
    old_point = plan["selected_observation_point"]
    assert_true(float(old_point["x"]) == 700.0, f"test setup should place initial point inside blocking room: {plan}")

    reset = harness.route6_reset_current_observation_point(output_dir=output_dir)

    assert_true(reset["reset_status"] == "no_reset_needed", f"non-blocking problems should be reported without moving: {reset}")
    assert_true("inside_house_bbox" in reset["problems"], f"reset should explain the point is inside a house bbox: {reset}")
    assert_true("front_blocked" not in reset["problems"], f"test setup should not trigger line-of-sight reset: {reset}")
    assert_true(float(reset["new_point"]["x"]) == float(old_point["x"]), f"west point should remain unchanged without front blockage: {reset}")
    assert_true(float(reset["new_point"]["y"]) == float(old_point["y"]), f"west point should remain unchanged without front blockage: {reset}")
    updated = harness.llm_route6_state["route6_offline_test_plan"]
    assert_true(updated["selected_observation_point"]["x"] == old_point["x"], f"state plan should keep the original point: {updated}")


def test_route6_reset_distance_candidates_move_toward_edge_when_front_blocked(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    harness.ensure_route6_state()

    blocked = harness.route6_test_planner_reset_distance_candidates(
        base_distance_cm=300.0,
        reset_step_cm=150.0,
        max_attempts=5,
        problems=["front_blocked"],
    )
    inside = harness.route6_test_planner_reset_distance_candidates(
        base_distance_cm=300.0,
        reset_step_cm=150.0,
        max_attempts=3,
        problems=["inside_house_bbox"],
    )

    assert_true(blocked[0] < 300.0, f"front-blocked reset should first move toward the exploration edge: {blocked}")
    assert_true(blocked[:2] == [150.0, 50.0], f"front-blocked reset should try closer standoffs before outward fallback: {blocked}")
    assert_true(inside == [], f"non-front-blocked problems should not generate reset distances: {inside}")


def test_route6_reset_reports_failed_scan_point_without_committing_candidate(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    harness.map_config["houses"][1]["bbox"] = {
        "min_x": -250.0,
        "max_x": 1680.0,
        "min_y": 1450.0,
        "max_y": 2200.0,
    }
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_update_map_layer_var.set("z_300")
    harness.route6_test_planner_edge_var.set("auto nearest")
    harness.route6_test_planner_scan_mode_var.set("multi-point edge coverage")
    harness.route6_test_planner_fov_deg_var.set("60")
    harness.route6_test_planner_overlap_var.set("0.30")
    harness.route6_test_planner_coverage_threshold_var.set("0.90")
    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["002"], radar_distance_cm=850.0)
    before_points = json.loads(json.dumps(plan["selected_scan_observation_points"]))
    s2 = next(item for item in before_points if int(item.get("scan_index", 0) or 0) == 2)
    add_layer_blocking_column(
        harness,
        output_dir,
        "z_300",
        float(s2["anchor_point_cm"]["x"]),
        float(s2["y"]) + 50.0,
        float(s2["anchor_point_cm"]["y"]) - 30.0,
    )

    reset = harness.route6_reset_current_observation_point(output_dir=output_dir)

    failed_scan_resets = [
        item
        for item in reset.get("scan_point_resets", [])
        if isinstance(item, dict) and str(item.get("reset_status", "")) == "failed"
    ]
    assert_true(failed_scan_resets, f"test setup should force at least one scan point reset failure: {reset}")
    assert_true(reset["reset_status"] == "scan_reset_failed", f"top-level reset status should report scan point failure: {reset}")
    updated = harness.llm_route6_state["route6_offline_test_plan"]["selected_scan_observation_points"]
    assert_true(updated == before_points, f"failed scan reset candidates must not be committed to the plan: {updated} != {before_points}")


def test_route6_reset_ignores_target_edge_adjacent_scan_samples(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    harness.map_config["houses"][1]["bbox"] = {
        "min_x": -250.0,
        "max_x": 1680.0,
        "min_y": 1450.0,
        "max_y": 2200.0,
    }
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_update_map_layer_var.set("z_300")
    harness.route6_test_planner_edge_var.set("auto nearest")
    harness.route6_test_planner_scan_mode_var.set("multi-point edge coverage")
    harness.route6_test_planner_fov_deg_var.set("60")
    harness.route6_test_planner_overlap_var.set("0.30")
    harness.route6_test_planner_coverage_threshold_var.set("0.90")
    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["002"], radar_distance_cm=850.0)
    before_points = json.loads(json.dumps(plan["selected_scan_observation_points"]))
    s2 = next(item for item in before_points if int(item.get("scan_index", 0) or 0) == 2)
    anchor = s2["anchor_point_cm"]
    add_layer_blocking_column(
        harness,
        output_dir,
        "z_300",
        float(anchor["x"]),
        float(anchor["y"]) - 140.0,
        float(anchor["y"]) - 140.0,
    )

    safety = harness.route6_test_planner_observation_safety_report(
        s2,
        anchor=anchor,
        target_house_id="002",
        output_dir=output_dir,
        selected_layer_key="z_300",
    )
    reset = harness.route6_reset_current_observation_point(output_dir=output_dir)

    assert_true("front_blocked" not in safety["problems"], f"facade-adjacent samples should not block the scan ray: {safety}")
    updated = harness.llm_route6_state["route6_offline_test_plan"]["selected_scan_observation_points"]
    updated_s2 = next(item for item in updated if int(item.get("scan_index", 0) or 0) == 2)
    assert_true(updated_s2 == s2, f"S2 should remain unchanged when only target-edge-adjacent cells are occupied: {reset}")


def test_route6_visual_calculation_records_use_actual_anchor(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    harness.map_config["houses"][1]["bbox"] = {
        "min_x": -250.0,
        "max_x": 1680.0,
        "min_y": 1450.0,
        "max_y": 2200.0,
    }
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_update_map_layer_var.set("z_300")
    harness.route6_test_planner_edge_var.set("auto nearest")
    harness.route6_test_planner_scan_mode_var.set("multi-point edge coverage")
    harness.route6_test_planner_fov_deg_var.set("60")
    harness.route6_test_planner_overlap_var.set("0.30")
    harness.route6_test_planner_coverage_threshold_var.set("0.90")

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["002"], radar_distance_cm=850.0)
    records = harness.route6_test_planner_visual_calculation_records(plan, output_dir=output_dir, selected_layer_key="z_300")

    main = next(item for item in records if item["kind"] == "selected_observation")
    selected = plan["selected_observation_point"]
    nearest = selected["nearest_edge_point_cm"]
    center = selected["edge_center_cm"]
    assert_true(main["anchor_point_cm"] == nearest, f"main observation visual ray must target nearest/algorithm anchor: {main}")
    assert_true(main["anchor_point_cm"] != center, f"main visual ray must not use edge center as a fake ray target: {main}")
    assert_true(main["display_on_map"] is False, f"multi-point mode should not draw the base selected observation as a navigation point: {main}")
    scan_records = [item for item in records if item["kind"] == "scan_observation"]
    assert_true(len(scan_records) == 3, f"visual calculation should include S1/S2/S3 records: {records}")
    assert_true(all(item["display_on_map"] is True for item in scan_records), f"multi-point mode should draw S1/S2/S3 as active navigation points: {scan_records}")
    assert_true(all(item["coverage_segment_cm"] for item in scan_records), f"scan records should expose coverage intervals: {scan_records}")
    assert_true(all("front_blocked" in item["safety_report"]["details"] for item in records), f"visual records should include line-of-sight safety reports: {records}")


def test_route6_reset_multiscan_treats_base_observation_as_reference(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    harness.map_config["houses"][1]["bbox"] = {
        "min_x": -250.0,
        "max_x": 1680.0,
        "min_y": 1450.0,
        "max_y": 2200.0,
    }
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    harness.route6_update_map_layer_var.set("z_300")
    harness.route6_test_planner_edge_var.set("auto nearest")
    harness.route6_test_planner_scan_mode_var.set("multi-point edge coverage")
    harness.route6_test_planner_fov_deg_var.set("60")
    harness.route6_test_planner_overlap_var.set("0.30")
    harness.route6_test_planner_coverage_threshold_var.set("0.90")
    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["002"], radar_distance_cm=850.0)
    before_selected = json.loads(json.dumps(plan["selected_observation_point"]))
    before_scan_points = json.loads(json.dumps(plan["selected_scan_observation_points"]))

    reset = harness.route6_reset_current_observation_point(output_dir=output_dir)

    assert_true(reset["active_reset_target"] == "scan_observation_points", f"multi-point reset should target S points, not the base observation: {reset}")
    assert_true(reset["main_reset_status"] == "reference_not_reset", f"base selected observation should be reference-only in multi-point reset: {reset}")
    assert_true(len(reset["scan_point_resets"]) == 3, f"reset should audit all selected scan points: {reset}")
    updated = harness.llm_route6_state["route6_offline_test_plan"]
    assert_true(updated["selected_observation_point"] == before_selected, f"base selected observation should remain unchanged: {updated}")
    assert_true(updated["selected_scan_observation_points"] == before_scan_points, f"safe scan points should remain unchanged: {updated}")


def test_route6_offline_planner_sorts_multi_house_by_nearest_edge(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()

    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["002", "001"], radar_distance_cm=300.0)

    ordered = [item["house_id"] for item in plan["observation_points"]]
    assert_true(ordered[:2] == ["001", "002"], f"nearest house edge should sort first: {ordered}")
    assert_true(plan["selected_observation_point"]["house_id"] == "001", f"nearest edge across selected houses should be first: {plan}")


def test_route6_offline_planner_loads_manifest_without_rebuilding(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()

    def fail_build(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("offline planner must load an existing Route 6 Update Map without rebuilding")

    harness.route6_update_map_build_from_pointcloud = fail_build
    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)

    assert_true(plan["manifest_available"] is True, f"manifest should be loaded: {plan}")
    assert_true(Path(plan["manifest_path"]).is_file(), f"manifest path should exist: {plan}")


def test_route6_offline_planner_overlay_draws_result(tmp_dir: Path) -> None:
    harness = PlannerHarness(tmp_dir)
    output_dir = prepare_map(harness)
    harness.ensure_route6_state()
    plan = harness.route6_build_offline_test_plan(output_dir=output_dir, selected_house_ids=["001"], radar_distance_cm=300.0)
    manifest = harness.route6_update_map_load_manifest(build_if_missing=False, output_dir=output_dir)
    layer = manifest["layers"][0]
    source = Image.open(layer["occupancy_preview_path"]).convert("RGB")

    overlay = harness.route6_draw_offline_test_plan_overlay(source, layer, plan, scale=1)

    assert_true(overlay.size == source.size, "overlay should preserve preview dimensions")
    diff = np.asarray(overlay.convert("RGB"), dtype=np.int16) - np.asarray(source.convert("RGB"), dtype=np.int16)
    changed = int(np.sum(np.any(diff != 0, axis=2)))
    assert_true(changed > 0, "overlay should draw selected edge and observation point")


def run_with_tmp(test_fn) -> None:
    with tempfile.TemporaryDirectory(prefix="route6_test_planner_") as raw:
        test_fn(Path(raw))


def main() -> None:
    tests = [
        test_panel_route6_test_planner_button_wiring,
        test_route6_test_planner_mixin_contract,
        test_route6_test_planner_analysis_package_contract,
        test_route6_test_planner_formula_button_and_payload,
        lambda: run_with_tmp(test_route6_offline_planner_selects_nearest_edge_and_writes_artifact),
        lambda: run_with_tmp(test_route6_offline_planner_uses_operator_selected_edge),
        lambda: run_with_tmp(test_route6_offline_planner_uses_surface_edge_explorer_center_anchor),
        lambda: run_with_tmp(test_route6_offline_planner_reports_nbv_information_gain_components),
        lambda: run_with_tmp(test_route6_offline_planner_generates_multi_point_scan_coverage),
        lambda: run_with_tmp(test_route6_north_scan_points_are_centered_inside_edge_span),
        lambda: run_with_tmp(test_route6_reset_reports_non_blocking_problem_without_moving_point),
        lambda: run_with_tmp(test_route6_reset_distance_candidates_move_toward_edge_when_front_blocked),
        lambda: run_with_tmp(test_route6_reset_reports_failed_scan_point_without_committing_candidate),
        lambda: run_with_tmp(test_route6_reset_ignores_target_edge_adjacent_scan_samples),
        lambda: run_with_tmp(test_route6_visual_calculation_records_use_actual_anchor),
        lambda: run_with_tmp(test_route6_reset_multiscan_treats_base_observation_as_reference),
        lambda: run_with_tmp(test_route6_offline_planner_sorts_multi_house_by_nearest_edge),
        lambda: run_with_tmp(test_route6_offline_planner_loads_manifest_without_rebuilding),
        lambda: run_with_tmp(test_route6_offline_planner_overlay_draws_result),
    ]
    for test in tests:
        test()
        print(f"PASS {getattr(test, '__name__', '<lambda>')}")
    print("PASS route6 test planner verification")


if __name__ == "__main__":
    main()
