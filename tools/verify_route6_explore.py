from __future__ import annotations

import copy
import importlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np


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


def load_builder() -> Any:
    return importlib.import_module("control.route6_map_builder")


def sample_map_config() -> Dict[str, Any]:
    return {
        "overhead_map": {
            "calibration": {
                "affine_world_to_image": [[0.1, 0.0, 0.0], [0.0, -0.1, 1000.0]],
                "image_width": 1200,
                "image_height": 1000,
            }
        },
        "houses": [
            {
                "id": "001",
                "name": "House_1",
                "center_x": 1000.0,
                "center_y": 0.0,
                "radius_cm": 300.0,
                "map_bbox_image": {"x1": 70.0, "y1": 970.0, "x2": 130.0, "y2": 1030.0},
            },
            {
                "id": "002",
                "name": "House_2",
                "center_x": 300.0,
                "center_y": 0.0,
                "radius_cm": 200.0,
                "map_bbox_image": {"x1": 20.0, "y1": 980.0, "x2": 40.0, "y2": 1020.0},
            },
            {
                "id": "003",
                "name": "House_3",
                "center_x": 600.0,
                "center_y": 800.0,
                "radius_cm": 250.0,
            },
        ],
    }


def test_rank_house_candidates_prefers_nearest_reachable_boundary() -> None:
    builder = load_builder()
    pose = {"x": 0.0, "y": 0.0, "z": 450.0, "yaw": 0.0}
    states = {"002": {"status": "searched"}, "003": {"status": "blocked", "cooldown_active": True}}
    candidates = builder.rank_house_candidates(sample_map_config(), pose, house_states=states, standoff_cm=850.0, scan_z_cm=450.0)
    assert_true(candidates, "expected at least one reachable Route 6 house candidate")
    assert_true(candidates[0]["house_id"] == "001", f"searched/cooldown houses should be skipped; got {candidates[0]}")
    assert_true(candidates[0]["nearest_boundary_distance_cm"] < 1000.0, "candidate should use nearest bbox boundary, not house center only")
    assert_true(candidates[0]["nearest_scan_pose"]["facade"] in {"south", "east", "north", "west"}, "candidate should include nearest facade scan pose")
    assert_true(candidates[0]["score"] >= 0.0, "candidate score should be numeric and non-negative")


def test_valid_pointcloud_rows_filter_capture_guard_and_point_count(tmp_dir: Path) -> None:
    builder = load_builder()
    good_path = tmp_dir / "good.npy"
    bad_path = tmp_dir / "bad.npy"
    np.save(good_path, np.zeros((4, 6), dtype=np.float32))
    np.save(bad_path, np.zeros((0, 6), dtype=np.float32))
    rows = [
        {"scan_id": "good", "facade": "west", "capture_guard_passed": True, "point_count": 4, "point_cloud_world_standard_m_npy_path": str(good_path)},
        {"scan_id": "bad_guard", "facade": "west", "capture_guard_passed": False, "point_count": 4, "point_cloud_world_standard_m_npy_path": str(good_path)},
        {"scan_id": "bad_count", "facade": "east", "capture_guard_passed": True, "point_count": 0, "point_cloud_world_standard_m_npy_path": str(good_path)},
        {"scan_id": "bad_facade", "facade": "roof", "capture_guard_passed": True, "point_count": 4, "point_cloud_world_standard_m_npy_path": str(good_path)},
        {"scan_id": "missing", "facade": "north", "capture_guard_passed": True, "point_count": 4, "point_cloud_world_standard_m_npy_path": str(tmp_dir / "missing.npy")},
        {"scan_id": "bad_empty", "facade": "south", "capture_guard_passed": True, "point_count": 4, "point_cloud_world_standard_m_npy_path": str(bad_path)},
    ]
    selected = builder.filter_valid_pointcloud_rows(rows)
    assert_true([row["scan_id"] for row in selected] == ["good"], f"unexpected valid rows: {selected}")


def test_occupancy_polygon_and_corrected_config_artifacts(tmp_dir: Path) -> None:
    builder = load_builder()
    points = []
    for x in np.linspace(1.0, 2.0, 8):
        for y in np.linspace(-1.5, -0.5, 8):
            points.append([x, y, 1.2, 255.0, 0.0, 0.0])
    cloud = np.asarray(points, dtype=np.float32)
    occupancy = builder.build_occupancy_grid(cloud, resolution_m=0.25, occupied_threshold=1)
    assert_true(occupancy["grid"].shape[0] > 0 and occupancy["grid"].shape[1] > 0, "occupancy grid should be non-empty")
    assert_true(int(np.sum(occupancy["grid"] == 100)) > 0, "occupancy grid should contain occupied cells")
    polygon = builder.extract_building_polygon(occupancy, house_id="002")
    assert_true(polygon["quality"]["occupied_cell_count"] > 0, "polygon should report occupied cells")
    assert_true(polygon["quality"]["confidence"] > 0.0, "polygon should have positive confidence")
    corrected = builder.build_corrected_map_config_from_polygon(sample_map_config(), "002", polygon)
    house = next(item for item in corrected["houses"] if str(item.get("id")) == "002")
    assert_true(house["route6_map_status"] in {"corrected", "candidate_only"}, f"unexpected map status: {house}")
    assert_true("route6_candidate_bbox_world" in house, "corrected house should record Route 6 candidate bbox")
    output = builder.write_route6_map_artifacts(tmp_dir, sample_map_config(), "002", cloud, resolution_m=0.25)
    for key in ("occupancy_grid_path", "occupancy_preview_path", "polygons_path", "corrected_config_path", "quality_report_path"):
        assert_true(Path(output[key]).is_file(), f"missing Route 6 artifact {key}: {output}")
    quality = json.loads(Path(output["quality_report_path"]).read_text(encoding="utf-8"))
    assert_true(quality["house_count_total"] == 3, "quality report should include total house count")
    assert_true(quality["house_count_mapped"] >= 1, "quality report should count the mapped house")


def test_polygon_component_selection_prefers_target_bbox_over_largest_component(tmp_dir: Path) -> None:
    builder = load_builder()
    near_points = []
    for x in np.linspace(2.6, 3.2, 5):
        for y in np.linspace(-0.3, 0.3, 5):
            near_points.append([x, y, 1.1, 80.0, 180.0, 255.0])
    far_points = []
    for x in np.linspace(12.0, 15.0, 16):
        for y in np.linspace(-8.0, -5.0, 16):
            far_points.append([x, y, 1.1, 255.0, 80.0, 80.0])
    cloud = np.asarray(near_points + far_points, dtype=np.float32)
    bbox = builder.house_world_bbox(sample_map_config(), sample_map_config()["houses"][1])
    occupancy = builder.build_occupancy_grid(cloud, resolution_m=0.25, occupied_threshold=1)
    polygon = builder.extract_building_polygon(occupancy, house_id="002", target_bbox_unreal_cm=bbox)
    assert_true(180.0 <= polygon["bbox"]["min_x"] <= 330.0, f"Route 6 should select the target-near component, not the far larger component: {polygon}")
    assert_true(polygon["quality"]["selected_component_reason"] == "nearest_target_bbox", f"polygon should record target-bbox component selection: {polygon}")
    output = builder.write_route6_map_artifacts(tmp_dir, sample_map_config(), "002", cloud, resolution_m=0.25)
    assert_true(output["polygon"]["bbox"]["max_x"] < 500.0, f"artifact polygon should also use the target-near component: {output['polygon']}")


def test_corrected_config_rejects_low_point_quality_even_with_high_confidence() -> None:
    builder = load_builder()
    polygon = {
        "schema": "route6_building_polygon_v1",
        "house_id": "002",
        "bbox": {"min_x": 220.0, "max_x": 380.0, "min_y": -80.0, "max_y": 80.0},
        "quality": {
            "confidence": 0.95,
            "point_count": 42,
            "occupied_cell_count": 60,
            "component_area_m2": 20.0,
        },
    }
    corrected = builder.build_corrected_map_config_from_polygon(sample_map_config(), "002", polygon)
    house = next(item for item in corrected["houses"] if str(item.get("id")) == "002")
    assert_true(house["route6_map_status"] == "candidate_only", f"low point-count polygon must not update formal map fields: {house}")
    assert_true(house["route6_correction_rejected_reason"] == "point_count_below_threshold", f"rejection reason should be explicit: {house}")
    assert_true(house["center_x"] == 300.0 and house["center_y"] == 0.0, f"formal center should remain unchanged: {house}")


def test_corrected_config_marks_large_shift_as_map_conflict() -> None:
    builder = load_builder()
    polygon = {
        "schema": "route6_building_polygon_v1",
        "house_id": "002",
        "bbox": {"min_x": 5200.0, "max_x": 5400.0, "min_y": 5200.0, "max_y": 5400.0},
        "quality": {
            "confidence": 0.95,
            "point_count": 8000,
            "occupied_cell_count": 80,
            "component_area_m2": 80.0,
        },
    }
    corrected = builder.build_corrected_map_config_from_polygon(sample_map_config(), "002", polygon)
    house = next(item for item in corrected["houses"] if str(item.get("id")) == "002")
    assert_true(house["route6_map_status"] == "map_conflict", f"large-shift polygon should require review: {house}")
    assert_true(house["route6_correction_rejected_reason"] == "center_shift_exceeds_threshold", f"large-shift rejection reason should be explicit: {house}")
    assert_true(house["center_x"] == 300.0 and house["center_y"] == 0.0, f"formal center should remain unchanged on map conflict: {house}")
    assert_true("route6_candidate_bbox_world" in house, f"map conflict should preserve candidate bbox for review: {house}")


class Harness:
    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = tmp_dir
        self.map_config = sample_map_config()
        self.latest_state = {"pose": [0.0, 0.0, 450.0, 0.0]}
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
        self.llm_route6_window = None
        self.llm_route6_summary_text = None
        self.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
        self.generated_scan_points: List[Dict[str, Any]] = []
        self.active_nbv_execute_calls: List[Dict[str, Any]] = []
        self.generate_capture_assets = False

    def ensure_route6_state(self) -> None:
        module = importlib.import_module("control.route6_explore_control")
        module.Route6ExploreControlMixin.ensure_route6_state(self)

    def __getattr__(self, name: str):
        module = importlib.import_module("control.route6_explore_control")
        attr = getattr(module.Route6ExploreControlMixin, name, None)
        if callable(attr):
            return lambda *args, **kwargs: attr(self, *args, **kwargs)
        raise AttributeError(name)

    def resolve_project_path(self, value: str, *, base_dir: Path | None = None) -> Path:
        path = Path(str(value or ""))
        if path.is_absolute():
            return path
        return (self.tmp_dir / path).resolve()

    def selected_route_target_house_id(self) -> str:
        return "002"

    def route3_current_pose(self, _session: Any = None) -> Dict[str, float]:
        return {"x": 0.0, "y": 0.0, "z": 450.0, "yaw": 0.0}

    def write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_jsonl_artifact(self, path: Path) -> List[Dict[str, Any]]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def active_nbv_initial_scan_points(self, house_id: str) -> List[Dict[str, Any]]:
        self.generated_scan_points = [
            {
                "scan_id": f"{house_id}_west_route6_initial_000",
                "house_id": str(house_id),
                "facade": "west",
                "x": -750.0,
                "y": 0.0,
                "z": 450.0,
                "yaw_deg": 0.0,
                "status": "planned",
            }
        ]
        return list(self.generated_scan_points)

    def active_nbv_build_coverage_report(self, house_id: str, points: List[Dict[str, Any]], *, output_dir: Path | None = None) -> Dict[str, Any]:
        return {
            "schema": "active_nbv_coverage_report_v1",
            "target_house_id": str(house_id),
            "valid_scan_capture_count": 1,
            "merged_point_count": 36,
            "mean_facade_coverage": 1.0,
            "complete": True,
            "facades": {
                "west": {
                    "facade": "west",
                    "planned_scan_count": 1,
                    "captured_scan_count": 1,
                    "point_cloud_coverage": 1.0,
                    "needs_rescan": False,
                }
            },
        }

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
        self.active_nbv_execute_calls.append({
            "target_house_id": str(target_house_id),
            "round_index": int(round_index),
            "scan_ids": [str(point.get("scan_id", "")) for point in points if isinstance(point, dict)],
            "view_types": [str(point.get("view_type", "")) for point in points if isinstance(point, dict)],
        })
        cloud_points = []
        base_x_m = 1.0 + 0.2 * float(int(str(target_house_id)) if str(target_house_id).isdigit() else 1)
        for x in np.linspace(base_x_m, base_x_m + 0.8, 6):
            for y in np.linspace(-0.9, -0.2, 6):
                cloud_points.append([x, y, 1.0, 120.0, 120.0, 120.0])
        cloud_path = Path(output_dir) / f"route6_mock_house_{target_house_id}_round_{round_index}.npy"
        np.save(cloud_path, np.asarray(cloud_points, dtype=np.float32))
        scan_id = str(points[0].get("scan_id", f"{target_house_id}_west_route6_initial_000") if points else f"{target_house_id}_west_route6_initial_000")
        capture_dir = Path(output_dir) / "facade_observations" / "west" / scan_id
        capture_payload: Dict[str, Any] = {}
        if self.generate_capture_assets:
            capture_dir.mkdir(parents=True, exist_ok=True)
            (capture_dir / "rgb.png").write_bytes(b"route6-test-rgb")
            np.save(capture_dir / "depth.npy", np.ones((3, 3), dtype=np.float32))
            (capture_dir / "camera_info.json").write_text(json.dumps({"width": 3, "height": 3}), encoding="utf-8")
            capture_payload = {
                "capture_dir": str(capture_dir),
                "rgb_path": str(capture_dir / "rgb.png"),
                "depth_npy_path": str(capture_dir / "depth.npy"),
                "camera_info_path": str(capture_dir / "camera_info.json"),
            }
        for point in points:
            point["status"] = "captured"
            point["point_count"] = len(cloud_points)
        row = {
            "scan_id": scan_id,
            "house_id": str(target_house_id),
            "facade": "west",
            "capture_guard_passed": True,
            "point_count": len(cloud_points),
            "point_cloud_world_standard_m_npy_path": str(cloud_path),
            "capture_kind": "active_nbv_scan",
            "capture_status": "ok",
            "view_type": str(points[0].get("view_type", "") if points else ""),
        }
        row.update(capture_payload)
        self.append_jsonl(Path(output_dir) / "lidar_capture_log.jsonl", row)


class AnalysisHarness(Harness):
    def __init__(self, tmp_dir: Path) -> None:
        super().__init__(tmp_dir)
        self.generate_capture_assets = True
        self.route5_analysis_calls: List[str] = []

    def route5_run_capture_analysis(
        self,
        run_dir: Path,
        *,
        weights_path: Path | None = None,
        stop_event: Any = None,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        self.route5_analysis_calls.append(str(run_dir))
        analysis_dir = Path(run_dir) / "route5_capture_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        candidates = {
            "candidate_count": 1,
            "candidates": [
                {
                    "class_name": "door",
                    "confidence": 0.91,
                    "frame_name": "route6_mock_frame",
                }
            ],
        }
        (analysis_dir / "entrance_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        summary = {
            "status": "ok",
            "run_dir": str(run_dir),
            "analysis_dir": str(analysis_dir),
            "included_count": 1,
            "semantic_label_count": 1,
            "semantic_point_count": 12,
            "created_at": "2026-05-25T00:00:00.000",
        }
        (analysis_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def test_route6_initialize_run_writes_state_and_queue(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = Harness(tmp_dir)
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    run_dir = module.Route6ExploreControlMixin.route6_initialize_run(harness, force_new=True)
    assert_true(run_dir.is_dir(), "Route 6 initialize should create a run directory")
    assert_true((run_dir / "route6_state.json").is_file(), "Route 6 initialize should write route6_state.json")
    assert_true((run_dir / "route6_house_queue.json").is_file(), "Route 6 initialize should write route6_house_queue.json")
    state = json.loads((run_dir / "route6_state.json").read_text(encoding="utf-8"))
    assert_true(state["stage"] == "SELECT_HOUSE", f"unexpected Route 6 stage: {state}")
    queue = json.loads((run_dir / "route6_house_queue.json").read_text(encoding="utf-8"))
    assert_true(queue["selected_house_id"] == "002", f"nearest reachable house should be selected: {queue}")


def test_route6_cli_builds_artifacts_from_lidar_log(tmp_dir: Path) -> None:
    module = importlib.import_module("tools.build_route6_occupancy_map")
    run_dir = tmp_dir / "route6_source_run"
    run_dir.mkdir(parents=True)
    points = []
    for x in np.linspace(1.0, 2.0, 6):
        for y in np.linspace(-1.0, -0.25, 6):
            points.append([x, y, 1.0, 100.0, 100.0, 100.0])
    cloud_path = run_dir / "frame_001.npy"
    np.save(cloud_path, np.asarray(points, dtype=np.float32))
    with (run_dir / "lidar_capture_log.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "scan_id": "002_west_001",
            "facade": "west",
            "capture_guard_passed": True,
            "point_count": len(points),
            "point_cloud_world_standard_m_npy_path": str(cloud_path),
        }) + "\n")
    map_path = tmp_dir / "houses_config.json"
    map_path.write_text(json.dumps(sample_map_config(), indent=2), encoding="utf-8")
    output_dir = tmp_dir / "route6_build_out"
    result = module.build_route6_occupancy_from_run(run_dir, map_path, output_dir, house_id="002", resolution_m=0.25)
    assert_true(Path(result["corrected_config_path"]).is_file(), f"missing corrected config from CLI builder: {result}")
    assert_true(Path(result["quality_report_path"]).is_file(), f"missing quality report from CLI builder: {result}")


def test_route6_worker_postprocesses_existing_valid_lidar_log(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = Harness(tmp_dir)
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    run_dir = module.Route6ExploreControlMixin.route6_initialize_run(harness, force_new=True)
    selected = harness.llm_route6_state["selected_house_id"]
    points = module.Route6ExploreControlMixin.route6_plan_selected_house_scan_points(harness, run_dir, selected)
    assert_true(points and (run_dir / "houses" / f"house_{selected}" / "scan_points.json").is_file(), "Route 6 should write house scan_points.json")

    cloud_points = []
    for x in np.linspace(1.0, 1.8, 6):
        for y in np.linspace(-0.9, -0.2, 6):
            cloud_points.append([x, y, 1.0, 120.0, 120.0, 120.0])
    cloud_path = run_dir / "frame_001.npy"
    np.save(cloud_path, np.asarray(cloud_points, dtype=np.float32))
    harness.append_jsonl(
        run_dir / "lidar_capture_log.jsonl",
        {
            "scan_id": points[0]["scan_id"],
            "house_id": selected,
            "facade": "west",
            "capture_guard_passed": True,
            "point_count": len(cloud_points),
            "point_cloud_world_standard_m_npy_path": str(cloud_path),
        },
    )
    result = module.Route6ExploreControlMixin.route6_postprocess_selected_house(harness, run_dir, selected)
    assert_true(result["status"] == "mapped_complete", f"Route 6 postprocess should map the house: {result}")
    assert_true((run_dir / "houses" / f"house_{selected}" / "coverage_report.json").is_file(), "Route 6 should write house coverage_report.json")
    assert_true((run_dir / "houses" / f"house_{selected}" / "house_state.json").is_file(), "Route 6 should write house_state.json")
    assert_true((run_dir / "houses" / f"house_{selected}" / "pointcloud" / "merged_point_cloud_world_standard_m.ply").is_file(), "Route 6 should write a replayable house PLY point cloud")
    house_map_dir = run_dir / "houses" / f"house_{selected}" / "map"
    assert_true((house_map_dir / "occupancy_grid.npy").is_file(), "Route 6 should mirror the house occupancy grid")
    assert_true((house_map_dir / "occupancy_grid.png").is_file(), "Route 6 should mirror the house occupancy preview")
    occupancy_meta = json.loads((house_map_dir / "occupancy_grid.json").read_text(encoding="utf-8"))
    assert_true(occupancy_meta["schema"] == "route6_occupancy_grid_v1", f"house occupancy metadata should be inspectable JSON: {occupancy_meta}")
    assert_true(occupancy_meta["coordinate_frame"]["pointcloud"] == "standard_m", f"occupancy metadata should document pointcloud frame: {occupancy_meta}")
    assert_true(occupancy_meta["coordinate_frame"]["map_config"] == "unreal_cm", f"occupancy metadata should document map frame: {occupancy_meta}")
    building_polygon = json.loads((house_map_dir / "building_polygon.json").read_text(encoding="utf-8"))
    assert_true(building_polygon["schema"] == "route6_building_polygon_v1", f"house building_polygon.json should contain the selected polygon: {building_polygon}")
    corrected_bbox = json.loads((house_map_dir / "corrected_bbox.json").read_text(encoding="utf-8"))
    assert_true(corrected_bbox["schema"] == "route6_corrected_bbox_record_v1", f"house corrected_bbox.json should be diagnostic and explicit: {corrected_bbox}")
    assert_true(corrected_bbox["house_id"] == str(selected), f"corrected bbox should identify its house: {corrected_bbox}")
    assert_true("candidate_bbox_world" in corrected_bbox, f"corrected bbox should preserve candidate bbox: {corrected_bbox}")
    assert_true("route6_map_status" in corrected_bbox, f"corrected bbox should record accepted/rejected correction status: {corrected_bbox}")
    assert_true(Path(result["map_artifacts"]["building_polygon_path"]).is_file(), f"house state should link building polygon artifact: {result}")
    assert_true(Path(result["map_artifacts"]["corrected_bbox_path"]).is_file(), f"house state should link corrected bbox artifact: {result}")
    assert_true(Path(result["map_artifacts"]["occupancy_metadata_path"]).is_file(), f"house state should link occupancy metadata: {result}")
    assert_true(Path(result["merged_pointcloud_ply_path"]).is_file(), f"house state should link PLY point cloud: {result}")
    assert_true((run_dir / "map" / "route6_corrected_houses_config.json").is_file(), "Route 6 should write corrected map artifact")
    state = json.loads((run_dir / "route6_state.json").read_text(encoding="utf-8"))
    assert_true(state["house_states"][selected]["status"] == "mapped_complete", f"Route 6 state should record mapped house: {state}")
    house_state = state["house_states"][selected]
    assert_true("facades" in house_state and "west" in house_state["facades"], f"house_state should summarize facade coverage: {house_state}")
    assert_true("map_confidence" in house_state, f"house_state should expose map_confidence: {house_state}")
    assert_true(house_state["entrance_status"] == "not_started", f"house_state should initialize entrance_status: {house_state}")
    assert_true(house_state["blocked_reason"] == "", f"house_state should initialize blocked_reason: {house_state}")


def test_route6_scan_plan_prepends_nearest_scout_before_active_nbv_points(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")

    class MultiPointHarness(Harness):
        def active_nbv_initial_scan_points(self, house_id: str) -> List[Dict[str, Any]]:
            return [
                {
                    "scan_id": f"{house_id}_north_active_nbv_initial_000",
                    "house_id": str(house_id),
                    "facade": "north",
                    "x": 9000.0,
                    "y": 9000.0,
                    "z": 450.0,
                    "yaw_deg": -90.0,
                    "status": "planned",
                },
                {
                    "scan_id": f"{house_id}_west_active_nbv_initial_001",
                    "house_id": str(house_id),
                    "facade": "west",
                    "x": -9000.0,
                    "y": 9000.0,
                    "z": 450.0,
                    "yaw_deg": 0.0,
                    "status": "planned",
                },
            ]

    harness = MultiPointHarness(tmp_dir)
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    run_dir = module.Route6ExploreControlMixin.route6_initialize_run(harness, force_new=True)
    harness.llm_route6_state["selected_candidate"] = {
        "house_id": "002",
        "nearest_facade": "south",
        "nearest_scan_pose": {
            "x": 300.0,
            "y": -850.0,
            "z": 450.0,
            "yaw_deg": 90.0,
            "facade": "south",
            "standoff_cm": 850.0,
        },
    }
    points = module.Route6ExploreControlMixin.route6_plan_selected_house_scan_points(harness, run_dir, "002")
    assert_true(points[0]["view_type"] == "route6_nearest_facade_scout", f"Route 6 should visit the nearest reachable facade scout before wider NBV points: {points}")
    assert_true(points[0]["scan_id"] == "002_south_route6_nearest_scout_000", f"nearest scout scan id should be deterministic: {points[0]}")
    assert_true(points[0]["x"] == 300.0 and points[0]["y"] == -850.0, f"nearest scout should use selected candidate nearest_scan_pose: {points[0]}")
    assert_true(len(points) == 3, f"nearest scout should be prepended without dropping active NBV points: {points}")
    harness.llm_route6_scan_point_limit_override = 1
    limited_points = module.Route6ExploreControlMixin.route6_plan_selected_house_scan_points(harness, run_dir, "002")
    assert_true(len(limited_points) == 1 and limited_points[0]["scan_id"] == "002_south_route6_nearest_scout_000", f"Route 6 should honor explicit scan point limit overrides for smoke tests: {limited_points}")


def test_route6_worker_maps_two_houses_and_writes_entrance_reports(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = Harness(tmp_dir)
    harness.llm_route6_max_houses_var.set("2")
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_full_explore_worker(harness, session=object(), force_new=True)
    run_dir = Path(harness.llm_route6_state["output_dir"])
    processed = harness.llm_route6_state.get("processed_house_ids", [])
    assert_true(len(processed) == 2, f"Route 6 should continue to the next nearest house after mapping one: {harness.llm_route6_state}")
    assert_true(len(set(processed)) == 2, f"Route 6 should not remap the same house twice: {processed}")
    for house_id in processed:
        entrance_dir = run_dir / "houses" / f"house_{house_id}" / "entrance"
        report_path = entrance_dir / "entrance_validation_report.json"
        yolo_manifest_path = entrance_dir / "yolo_manifest.json"
        assert_true(report_path.is_file(), f"Route 6 should write entrance validation report for house {house_id}")
        assert_true(yolo_manifest_path.is_file(), f"Route 6 should write entrance YOLO/capture manifest for house {house_id}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert_true(report["schema"] == "route6_entrance_validation_report_v1", f"unexpected entrance report schema: {report}")
        assert_true(report["entry_search_complete"] is False, f"Route 6 should not fake a completed entrance search: {report}")
        assert_true(Path(report["corrected_config_path"]).is_file(), f"entrance report should link corrected config: {report}")
    final_state = json.loads((run_dir / "route6_state.json").read_text(encoding="utf-8"))
    assert_true(final_state["stage"] == "DONE", f"Route 6 worker should finish cleanly after max_houses: {final_state}")
    corrected = json.loads((run_dir / "map" / "route6_corrected_houses_config.json").read_text(encoding="utf-8"))
    corrected_by_id = {
        str(item.get("id", item.get("house_id", ""))): item
        for item in corrected.get("houses", [])
        if isinstance(item, dict)
    }
    for house_id in processed:
        house_record = corrected_by_id.get(str(house_id), {})
        assert_true("route6_candidate_bbox_world" in house_record, f"run-level corrected config should retain cumulative Route 6 record for house {house_id}: {corrected}")
    polygons = json.loads((run_dir / "map" / "route6_polygons.json").read_text(encoding="utf-8"))
    polygon_ids = {str(item.get("house_id", "")) for item in polygons.get("polygons", []) if isinstance(item, dict)}
    assert_true(set(map(str, processed)).issubset(polygon_ids), f"run-level polygon artifact should retain every processed house polygon: {polygons}")
    global_occupancy_meta_path = run_dir / "map" / "route6_occupancy_grid.json"
    assert_true(global_occupancy_meta_path.is_file(), "Route 6 should write a run-level cumulative occupancy metadata file")
    global_occupancy = json.loads(global_occupancy_meta_path.read_text(encoding="utf-8"))
    assert_true(global_occupancy["schema"] == "route6_global_occupancy_grid_v1", f"run-level occupancy should be explicitly global: {global_occupancy}")
    assert_true(global_occupancy["scope"] == "run_global", f"run-level occupancy should not be a single-house artifact: {global_occupancy}")
    assert_true(set(map(str, processed)).issubset(set(map(str, global_occupancy.get("house_ids", [])))), f"global occupancy should include every processed house: {global_occupancy}")
    assert_true(int(global_occupancy.get("source_pointcloud_count", 0)) >= len(processed), f"global occupancy should merge one pointcloud per processed house: {global_occupancy}")
    global_grid = np.load(run_dir / "map" / "route6_occupancy_grid.npy")
    assert_true(global_grid.ndim == 2 and int(np.sum(global_grid == 100)) > 0, "run-level cumulative occupancy grid should contain occupied cells")
    quality = json.loads((run_dir / "map" / "route6_map_quality_report.json").read_text(encoding="utf-8"))
    assert_true(quality["global_occupancy_grid_path"] == str(run_dir / "map" / "route6_occupancy_grid.npy"), f"quality report should link cumulative occupancy grid: {quality}")
    assert_true(set(map(str, processed)).issubset(set(map(str, quality.get("global_occupancy_house_ids", [])))), f"quality report should list houses in cumulative occupancy: {quality}")
    events = harness.read_jsonl_artifact(run_dir / "route6_events.jsonl")
    stage_values = [str(item.get("stage", "")) for item in events if str(item.get("event_type", "")) == "stage"]
    for required_stage in ("INIT_RUN", "LOAD_MAP", "RANK_HOUSES", "SELECT_HOUSE", "EXTRACT_POLYGONS", "DONE"):
        assert_true(required_stage in stage_values, f"Route 6 event log should include stage={required_stage}: {stage_values}")


def test_route6_worker_runs_route5_analysis_and_writes_run_summary(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = AnalysisHarness(tmp_dir)
    harness.llm_route6_max_houses_var.set("1")
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_full_explore_worker(harness, session=object(), force_new=True)
    run_dir = Path(harness.llm_route6_state["output_dir"])
    processed = harness.llm_route6_state.get("processed_house_ids", [])
    assert_true(len(processed) == 1, f"Route 6 should process one house in this test: {harness.llm_route6_state}")
    selected = processed[0]
    assert_true(harness.route5_analysis_calls == [str(run_dir), str(run_dir)], f"Route 6 should run Route 5 analysis before and after close-confirm scan: {harness.route5_analysis_calls}")
    report_path = run_dir / "houses" / f"house_{selected}" / "entrance" / "entrance_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert_true(report["entry_search_complete"] is True, f"Route 6 should complete entrance search after close-confirm analysis: {report}")
    assert_true(report["candidate_count"] == 1, f"Route 6 should carry Route 5 entrance candidates into its report: {report}")
    assert_true(report["route5_analysis_status"] == "ok", f"Route 6 should record Route 5 analysis status: {report}")
    assert_true(report["status"] == "entrance_confirmed", f"candidate-positive close-confirm analysis should confirm entrance: {report}")
    confirm_plan_path = Path(report["close_confirm_scan_plan_path"])
    assert_true(confirm_plan_path.is_file(), f"Route 6 should write close-confirm scan plan: {report}")
    confirm_plan = json.loads(confirm_plan_path.read_text(encoding="utf-8"))
    assert_true(confirm_plan["schema"] == "route6_close_confirm_scan_plan_v1", f"unexpected confirm plan schema: {confirm_plan}")
    assert_true(confirm_plan["candidate_count"] == 1, f"confirm plan should include candidate count: {confirm_plan}")
    assert_true(confirm_plan["scan_points"], f"confirm plan should include executable scan points: {confirm_plan}")
    assert_true(confirm_plan["scan_points"][0]["view_type"] == "route6_close_confirm_scan", f"confirm scan point should be typed: {confirm_plan}")
    assert_true(
        any("route6_close_confirm_scan" in call["view_types"] for call in harness.active_nbv_execute_calls),
        f"Route 6 should execute close-confirm scan points when a live session is available: {harness.active_nbv_execute_calls}",
    )
    execution_report_path = run_dir / "houses" / f"house_{selected}" / "entrance" / "close_confirm_execution_report.json"
    assert_true(execution_report_path.is_file(), "Route 6 should write close-confirm execution report")
    execution_report = json.loads(execution_report_path.read_text(encoding="utf-8"))
    assert_true(execution_report["status"] == "executed", f"close-confirm execution report should record execution: {execution_report}")
    assert_true(execution_report["executed_scan_count"] == 1, f"execution report should count confirm scans: {execution_report}")
    confirm_analysis_path = run_dir / "houses" / f"house_{selected}" / "entrance" / "close_confirm_analysis_report.json"
    assert_true(confirm_analysis_path.is_file(), "Route 6 should write close-confirm analysis report")
    confirm_analysis = json.loads(confirm_analysis_path.read_text(encoding="utf-8"))
    assert_true(confirm_analysis["schema"] == "route6_close_confirm_analysis_report_v1", f"unexpected close-confirm analysis schema: {confirm_analysis}")
    assert_true(confirm_analysis["status"] == "confirmed", f"close-confirm analysis should confirm the candidate when Route 5 still sees it: {confirm_analysis}")
    assert_true(confirm_analysis["included_confirm_capture_count"] == 1, f"confirm analysis should filter confirm captures: {confirm_analysis}")
    assert_true(confirm_analysis["candidate_count"] == 1, f"confirm analysis should preserve candidate count: {confirm_analysis}")
    obstacle_report_path = run_dir / "houses" / f"house_{selected}" / "entrance" / "obstacle_validation_report.json"
    assert_true(obstacle_report_path.is_file(), "Route 6 should write obstacle_validation_report.json after close confirm")
    obstacle_report = json.loads(obstacle_report_path.read_text(encoding="utf-8"))
    assert_true(obstacle_report["schema"] == "route6_obstacle_validation_report_v1", f"unexpected obstacle validation schema: {obstacle_report}")
    assert_true(obstacle_report["status"] == "clear", f"clear confirm captures should pass obstacle validation: {obstacle_report}")
    assert_true(obstacle_report["passable_space_confirmed"] is True, f"clear confirm captures should confirm passable space: {obstacle_report}")
    assert_true(obstacle_report["blocking_capture_count"] == 0, f"clear confirm captures should not include blockers: {obstacle_report}")
    final_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert_true(final_report["entry_search_complete"] is True, f"close-confirm analysis should complete the entrance search: {final_report}")
    assert_true(final_report["status"] == "entrance_confirmed", f"final entrance report should record confirmed status: {final_report}")
    assert_true(final_report["obstacle_validation_report_path"] == str(obstacle_report_path), f"final report should link obstacle validation: {final_report}")
    assert_true(final_report["obstacle_validation_status"] == "clear", f"final report should record clear obstacle validation: {final_report}")
    house_state = json.loads((run_dir / "houses" / f"house_{selected}" / "house_state.json").read_text(encoding="utf-8"))
    assert_true(house_state["status"] == "searched", f"candidate-positive close confirm should mark house searched: {house_state}")
    assert_true(house_state["search_status"] == "entrance_confirmed", f"close-confirm analysis should confirm entrance: {house_state}")
    assert_true(house_state["close_confirm_status"] == "confirmed", f"house state should record close-confirm analysis: {house_state}")
    assert_true(house_state["obstacle_validation_status"] == "clear", f"house state should record clear obstacle validation: {house_state}")
    summary_path = run_dir / "route6_exploration_summary.csv"
    assert_true(summary_path.is_file(), "Route 6 should write route6_exploration_summary.csv")
    summary_text = summary_path.read_text(encoding="utf-8")
    assert_true("house_id,status,search_status,entrance_status" in summary_text, f"summary header is missing expected fields: {summary_text}")
    assert_true(",searched,entrance_confirmed,confirmed_by_close_confirm_scan" in summary_text, f"summary should record confirmed entrance status: {summary_text}")
    quality = json.loads((run_dir / "map" / "route6_map_quality_report.json").read_text(encoding="utf-8"))
    assert_true(quality["run_dir"] == str(run_dir), f"quality report should identify the run dir: {quality}")
    assert_true(quality["house_count_searched"] == 1, f"quality report should count close-confirmed houses as searched: {quality}")


def test_route6_worker_honors_runtime_stop_condition(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = Harness(tmp_dir)
    harness.llm_route6_runtime_min_var.set("0")
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_full_explore_worker(harness, session=object(), force_new=True)
    state = harness.llm_route6_state
    assert_true(state["stage"] == "STOPPED", f"Route 6 should stop when runtime budget is exhausted: {state}")
    assert_true(state.get("processed_house_ids", []) == [], f"runtime stop should happen before processing houses: {state}")


def test_route6_smoke_runner_offline_mock_writes_auditable_result(tmp_dir: Path) -> None:
    module = importlib.import_module("tools.run_route6_smoke")
    result = module.run_route6_smoke(
        mode="offline_mock",
        output_root=tmp_dir / "route6_smoke_runs",
        max_houses=1,
        runtime_minutes=5.0,
        map_config=sample_map_config(),
    )
    result_path = Path(result["result_path"])
    assert_true(result_path.is_file(), f"Route 6 smoke runner should write a result JSON: {result}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert_true(payload["schema"] == "route6_smoke_result_v1", f"unexpected smoke result schema: {payload}")
    assert_true(payload["mode"] == "offline_mock", f"smoke result should record mode: {payload}")
    assert_true(payload["status"] == "ok", f"offline mock should finish Route 6 worker successfully: {payload}")
    assert_true(payload["worker_stage"] == "DONE", f"worker should finish after one mocked house: {payload}")
    assert_true(payload["processed_house_count"] == 1, f"smoke result should count processed houses: {payload}")
    assert_true(payload["active_nbv_execute_call_count"] >= 1, f"smoke result should prove scan executor was used: {payload}")
    state_path = Path(payload["route6_state_path"])
    assert_true(state_path.is_file(), f"smoke result should link route6_state.json: {payload}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert_true(state["stage"] == payload["worker_stage"], f"smoke result should match worker state: {payload} vs {state}")
    for key in ("route6_exploration_summary_path", "route6_map_quality_report_path"):
        assert_true(Path(payload[key]).is_file(), f"smoke result should link artifact {key}: {payload}")


def test_route6_smoke_live_controller_defaults_to_pointcloud_capture(tmp_dir: Path) -> None:
    module = importlib.import_module("tools.run_route6_smoke")
    args = module.default_live_controller_args(output_root=tmp_dir / "route6_live_smoke_runs")
    assert_true(
        str(getattr(args, "lidar_capture_processing", "")) == "full",
        f"Route 6 live smoke should default to pointcloud-producing capture mode: {args}",
    )


def test_route6_smoke_runner_live_controller_uses_worker_contract(tmp_dir: Path) -> None:
    module = importlib.import_module("tools.run_route6_smoke")
    created_panels: List[Any] = []

    class FakePanel:
        def __init__(self, args: Any) -> None:
            self.args = args
            self.session = object()
            self.worker_calls: List[Dict[str, Any]] = []
            self.closed = False
            self.llm_route6_state: Dict[str, Any] = {}
            self.llm_route6_max_houses_var = ValueVar("0")
            self.llm_route6_runtime_min_var = ValueVar("0")
            created_panels.append(self)

        def route6_full_explore_worker(self, session: Any, *, force_new: bool = False) -> None:
            self.worker_calls.append({"session_is_panel_session": session is self.session, "force_new": force_new})
            run_dir = tmp_dir / "live_controller_run"
            (run_dir / "map").mkdir(parents=True, exist_ok=True)
            (run_dir / "route6_state.json").write_text(json.dumps({"stage": "DONE"}, indent=2), encoding="utf-8")
            (run_dir / "route6_exploration_summary.csv").write_text("house_id,status\n002,mapped_complete\n", encoding="utf-8")
            (run_dir / "map" / "route6_map_quality_report.json").write_text(
                json.dumps({"schema": "route6_map_quality_report_v1", "run_dir": str(run_dir)}, indent=2),
                encoding="utf-8",
            )
            self.llm_route6_state = {
                "stage": "DONE",
                "output_dir": str(run_dir),
                "processed_house_ids": ["002"],
                "house_states": {
                    "002": {
                        "status": "mapped_complete",
                        "valid_scan_capture_count": 1,
                        "merged_point_count": 36,
                    }
                },
                "exploration_summary_path": str(run_dir / "route6_exploration_summary.csv"),
            }

        def on_close(self) -> None:
            self.closed = True

    result = module.run_route6_smoke(
        mode="live_controller",
        output_root=tmp_dir / "route6_live_smoke_runs",
        max_houses=1,
        runtime_minutes=5.0,
        panel_factory=FakePanel,
        start_session=False,
    )
    assert_true(created_panels, "live_controller smoke should instantiate a controller panel")
    panel = created_panels[0]
    assert_true(panel.worker_calls == [{"session_is_panel_session": True, "force_new": True}], f"live_controller smoke should call Route 6 worker once: {panel.worker_calls}")
    assert_true(panel.closed is True, "live_controller smoke should close the controller panel")
    assert_true(result["schema"] == "route6_smoke_result_v1", f"unexpected live smoke schema: {result}")
    assert_true(result["mode"] == "live_controller", f"live smoke result should record mode: {result}")
    assert_true(result["status"] == "ok", f"fake live controller should finish as ok: {result}")
    assert_true(result["controller_worker_called"] is True, f"live result should prove the Route 6 worker was used: {result}")
    assert_true(Path(result["result_path"]).is_file(), f"live smoke result JSON should be written: {result}")
    assert_true(result["mapped_house_count"] == 1, f"live smoke should count mapped houses: {result}")
    assert_true(result["valid_scan_capture_count_total"] == 1, f"live smoke should count valid scan captures: {result}")


def test_route6_smoke_runner_live_controller_initializes_route6_vars_and_rejects_empty_captures(tmp_dir: Path) -> None:
    module = importlib.import_module("tools.run_route6_smoke")
    created_panels: List[Any] = []

    class FakePanel:
        def __init__(self, args: Any) -> None:
            self.args = args
            self.session = object()
            self.closed = False
            self.ensure_called = False
            self.worker_max_houses = ""
            self.worker_runtime_minutes = ""
            self.llm_route6_state: Dict[str, Any] = {}
            created_panels.append(self)

        def ensure_route6_state(self) -> None:
            self.ensure_called = True
            self.llm_route6_max_houses_var = ValueVar("3")
            self.llm_route6_runtime_min_var = ValueVar("30")
            self.llm_route6_standoff_cm_var = ValueVar("850")
            self.llm_route6_scan_z_cm_var = ValueVar("450")

        def route6_full_explore_worker(self, session: Any, *, force_new: bool = False) -> None:
            self.worker_max_houses = self.llm_route6_max_houses_var.get() if hasattr(self, "llm_route6_max_houses_var") else "missing"
            self.worker_runtime_minutes = self.llm_route6_runtime_min_var.get() if hasattr(self, "llm_route6_runtime_min_var") else "missing"
            self.worker_standoff_cm = self.llm_route6_standoff_cm_var.get() if hasattr(self, "llm_route6_standoff_cm_var") else "missing"
            self.worker_scan_z_cm = self.llm_route6_scan_z_cm_var.get() if hasattr(self, "llm_route6_scan_z_cm_var") else "missing"
            run_dir = tmp_dir / "live_controller_empty_capture_run"
            (run_dir / "map").mkdir(parents=True, exist_ok=True)
            (run_dir / "route6_state.json").write_text(json.dumps({"stage": "DONE"}, indent=2), encoding="utf-8")
            (run_dir / "route6_exploration_summary.csv").write_text("house_id,status\n003,needs_capture\n", encoding="utf-8")
            (run_dir / "map" / "route6_map_quality_report.json").write_text(
                json.dumps({"schema": "route6_map_quality_report_v1", "run_dir": str(run_dir)}, indent=2),
                encoding="utf-8",
            )
            self.llm_route6_state = {
                "stage": "DONE",
                "output_dir": str(run_dir),
                "processed_house_ids": ["003"],
                "house_states": {
                    "003": {
                        "status": "needs_capture",
                        "valid_scan_capture_count": 0,
                        "merged_point_count": 0,
                    }
                },
                "exploration_summary_path": str(run_dir / "route6_exploration_summary.csv"),
            }

        def on_close(self) -> None:
            self.closed = True

    result = module.run_route6_smoke(
        mode="live_controller",
        output_root=tmp_dir / "route6_live_smoke_runs",
        max_houses=1,
        runtime_minutes=2.5,
        max_scan_points=1,
        standoff_cm=650.0,
        scan_z_cm=700.0,
        panel_factory=FakePanel,
        start_session=False,
    )
    panel = created_panels[0]
    assert_true(panel.ensure_called is True, "live_controller smoke should initialize Route 6 vars before setting overrides")
    assert_true(panel.worker_max_houses == "1", f"live_controller smoke should pass max_houses into the worker: {panel.worker_max_houses}")
    assert_true(panel.worker_runtime_minutes == "2.5", f"live_controller smoke should pass runtime_minutes into the worker: {panel.worker_runtime_minutes}")
    assert_true(panel.worker_standoff_cm == "650.0", f"live_controller smoke should pass standoff_cm into the worker: {panel.worker_standoff_cm}")
    assert_true(panel.worker_scan_z_cm == "700.0", f"live_controller smoke should pass scan_z_cm into the worker: {panel.worker_scan_z_cm}")
    assert_true(getattr(panel, "llm_route6_scan_point_limit_override", None) == 1, "live_controller smoke should pass max_scan_points as a Route 6 override")
    assert_true(result["status"] == "check", f"empty live captures must not be reported as ok: {result}")
    assert_true(result["mapped_house_count"] == 0, f"empty live captures should not count mapped houses: {result}")
    assert_true(result["valid_scan_capture_count_total"] == 0, f"empty live captures should report zero valid captures: {result}")


def run_with_tmp(test_fn) -> None:
    with tempfile.TemporaryDirectory(prefix="route6_verify_") as raw:
        test_fn(Path(raw))


def main() -> None:
    tests = [
        lambda: test_rank_house_candidates_prefers_nearest_reachable_boundary(),
        lambda: run_with_tmp(test_valid_pointcloud_rows_filter_capture_guard_and_point_count),
        lambda: run_with_tmp(test_occupancy_polygon_and_corrected_config_artifacts),
        lambda: run_with_tmp(test_polygon_component_selection_prefers_target_bbox_over_largest_component),
        lambda: test_corrected_config_rejects_low_point_quality_even_with_high_confidence(),
        lambda: test_corrected_config_marks_large_shift_as_map_conflict(),
        lambda: run_with_tmp(test_route6_initialize_run_writes_state_and_queue),
        lambda: run_with_tmp(test_route6_cli_builds_artifacts_from_lidar_log),
        lambda: run_with_tmp(test_route6_worker_postprocesses_existing_valid_lidar_log),
        lambda: run_with_tmp(test_route6_scan_plan_prepends_nearest_scout_before_active_nbv_points),
        lambda: run_with_tmp(test_route6_worker_maps_two_houses_and_writes_entrance_reports),
        lambda: run_with_tmp(test_route6_worker_runs_route5_analysis_and_writes_run_summary),
        lambda: run_with_tmp(test_route6_worker_honors_runtime_stop_condition),
        lambda: run_with_tmp(test_route6_smoke_runner_offline_mock_writes_auditable_result),
        lambda: run_with_tmp(test_route6_smoke_live_controller_defaults_to_pointcloud_capture),
        lambda: run_with_tmp(test_route6_smoke_runner_live_controller_uses_worker_contract),
        lambda: run_with_tmp(test_route6_smoke_runner_live_controller_initializes_route6_vars_and_rejects_empty_captures),
    ]
    for test in tests:
        test()
        print(f"PASS {getattr(test, '__name__', 'route6_test')}")
    print("PASS route6 explore verification")


if __name__ == "__main__":
    main()
