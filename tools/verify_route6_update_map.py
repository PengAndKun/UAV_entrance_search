from __future__ import annotations

import importlib
import json
import os
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


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_builder() -> Any:
    return importlib.import_module("control.route6_map_builder")


class ValueVar:
    def __init__(self, value: Any = "") -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


class FakeSession:
    def __init__(self) -> None:
        self.capture_calls: List[Dict[str, Any]] = []

    def capture_lidar_stream_frame(self, stream_dir: Path, frame_index: int, action_detail: Dict[str, Any] | None = None) -> Dict[str, Any]:
        frame_dir = Path(stream_dir) / "frames" / f"frame_{int(frame_index):06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        cloud = sample_layered_cloud()
        cloud_path = frame_dir / "point_cloud_world_standard_m.npy"
        np.save(cloud_path, cloud)
        self.capture_calls.append({"stream_dir": str(stream_dir), "frame_index": int(frame_index), "action_detail": action_detail or {}})
        return {
            "status": "ok",
            "capture_status": "ok",
            "capture_kind": "route6_update_map_capture",
            "frame_index": int(frame_index),
            "capture_dir": str(frame_dir),
            "point_cloud_world_standard_m_npy_path": str(cloud_path),
            "point_count": int(cloud.shape[0]),
        }


class FakeRawDepthSession:
    def __init__(self) -> None:
        self.capture_calls: List[Dict[str, Any]] = []

    def capture_lidar_stream_frame(self, stream_dir: Path, frame_index: int, action_detail: Dict[str, Any] | None = None) -> Dict[str, Any]:
        from PIL import Image

        frame_dir = Path(stream_dir) / "frames" / f"frame_{int(frame_index):06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        depth = np.full((16, 16), 300.0 + float(frame_index), dtype=np.float32)
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[..., 1] = 180
        np.save(frame_dir / "depth.npy", depth)
        Image.fromarray(rgb).save(frame_dir / "rgb.png")
        camera_info = {
            "location": {"x": 0.0, "y": 0.0, "z": 200.0},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "horizontal_fov_deg": 90.0,
            "image_width": 16,
            "image_height": 16,
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
        }
        (frame_dir / "camera_info.json").write_text(json.dumps(camera_info, indent=2), encoding="utf-8")
        capture = {
            "status": "ok",
            "capture_status": "ok",
            "capture_kind": "route6_update_map_capture",
            "frame_index": int(frame_index),
            "capture_dir": str(frame_dir),
            "depth_npy_path": str(frame_dir / "depth.npy"),
            "rgb_path": str(frame_dir / "rgb.png"),
            "camera_info_path": str(frame_dir / "camera_info.json"),
            "point_cloud_world_standard_m_npy_path": "",
            "point_count": 0,
            "raw_capture_only": True,
            "postprocess_status": "pending",
        }
        (frame_dir / "capture.json").write_text(json.dumps(capture, indent=2), encoding="utf-8")
        self.capture_calls.append({"stream_dir": str(stream_dir), "frame_index": int(frame_index), "action_detail": action_detail or {}})
        return capture


class CaptureHarness:
    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = tmp_dir
        self.session = FakeSession()
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
        self.llm_route6_max_houses_var = ValueVar("1")
        self.llm_route6_runtime_min_var = ValueVar("5")
        self.llm_route6_standoff_cm_var = ValueVar("850")
        self.llm_route6_scan_z_cm_var = ValueVar("450")
        self.llm_route6_occupancy_resolution_m_var = ValueVar("0.25")
        self.llm_route6_coverage_threshold_var = ValueVar("0.75")
        self.llm_route6_allow_save_corrected_var = ValueVar(False)
        self.llm_route6_task_prompt_var = ValueVar("Explore the first house north of current UAV.")
        self.llm_route6_selected_target_var = ValueVar("Selected target: n/a")
        self.llm_route6_realtime_map_status_var = ValueVar("Realtime map: idle")
        self.route6_update_map_layer_var = ValueVar("z_050")
        self.route6_update_map_status_var = ValueVar("Route 6 Update Map: idle")
        self.route6_update_map_pose_var = ValueVar("UAV x=n/a y=n/a z=n/a yaw=n/a")
        self.route6_update_map_window = None
        self.route6_update_map_scroll_canvas = None
        self.route6_update_map_content_frame = None
        self.route6_update_map_preview_label = None
        self.route6_update_map_preview_photo = None
        self.route6_update_map_layer_combo = None
        self.llm_route6_realtime_map_preview_label = None
        self.llm_route6_realtime_map_preview_photo = None
        self.llm_route6_realtime_map_layer_combo = None
        self.route6_update_map_capture_thread = None
        self.route6_update_map_realtime_thread = None
        self.route6_update_map_min_move_cm_var = ValueVar("0")
        self.route6_update_map_min_yaw_deg_var = ValueVar("0")

    def __getattr__(self, name: str):
        module = importlib.import_module("control.route6_explore_control")
        attr = getattr(module.Route6ExploreControlMixin, name, None)
        if callable(attr):
            return lambda *args, **kwargs: attr(self, *args, **kwargs)
        raise AttributeError(name)

    def route6_output_root(self) -> Path:
        return Path(getattr(self, "llm_route6_output_root_override", self.tmp_dir / "route6_update_map_runs"))

    def route6_float_param(self, variable: Any, default: float, *, min_value: float = 0.0, max_value: float = 1e9) -> float:
        try:
            value = variable.get() if hasattr(variable, "get") else variable
            return max(float(min_value), min(float(max_value), float(value)))
        except Exception:
            return float(default)

    def write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sample_layered_cloud() -> np.ndarray:
    points: List[List[float]] = []
    for index, layer_z_cm in enumerate(range(50, 651, 50)):
        offset_x = float(index) * 0.7
        z_m = layer_z_cm / 100.0
        for x in np.linspace(offset_x, offset_x + 0.3, 4):
            for y in np.linspace(0.0, 0.3, 4):
                points.append([float(x), float(y), z_m, 220.0, 220.0, 220.0])
    points.append([12.0, 12.0, 9.0, 255.0, 0.0, 0.0])
    return np.asarray(points, dtype=np.float32)


def test_layered_occupancy_projects_pointcloud_by_z_layers() -> None:
    builder = load_builder()
    layered = builder.build_route6_layered_occupancy_maps(
        sample_layered_cloud(),
        layer_z_cm=[50, 200, 350, 500, 650],
        layer_band_cm=40.0,
        resolution_m=0.25,
        occupied_threshold=1,
    )
    assert_true(layered["schema"] == "route6_layered_occupancy_v1", f"unexpected schema: {layered}")
    assert_true(layered["layer_z_cm"] == [50, 200, 350, 500, 650], f"unexpected layers: {layered}")
    assert_true(len(layered["layers"]) == 5, f"expected five layer maps: {layered}")
    for layer in layered["layers"]:
        grid = np.asarray(layer["occupancy"]["grid"])
        assert_true(int(layer["point_count"]) == 16, f"each synthetic z layer should receive exactly its own points: {layer}")
        assert_true(int(np.sum(grid >= 100)) > 0, f"layer should have occupied cells: {layer}")
        assert_true(layer["z_min_cm"] <= layer["z_cm"] <= layer["z_max_cm"], f"layer band should bracket z: {layer}")
    assert_true(
        sum(int(layer["point_count"]) for layer in layered["layers"]) == 80,
        f"out-of-band 900cm point should not be assigned to fixed Route 6 layers: {layered}",
    )


def test_default_route6_layers_are_every_50cm_from_50_to_650() -> None:
    builder = load_builder()
    expected_layers = list(range(50, 651, 50))
    assert_true(list(builder.DEFAULT_ROUTE6_LAYER_Z_CM) == expected_layers, f"default layers should be every 50cm: {builder.DEFAULT_ROUTE6_LAYER_Z_CM}")
    assert_true(float(builder.DEFAULT_ROUTE6_LAYER_BAND_CM) == 25.0, f"default 50cm layers should use +/-25cm bands: {builder.DEFAULT_ROUTE6_LAYER_BAND_CM}")
    assert_true(int(builder.DEFAULT_ROUTE6_LAYER_OCCUPIED_THRESHOLD) == 2, f"default layer threshold should suppress single-hit noise: {builder.DEFAULT_ROUTE6_LAYER_OCCUPIED_THRESHOLD}")
    layered = builder.build_route6_layered_occupancy_maps(
        sample_layered_cloud(),
        resolution_m=0.25,
    )
    assert_true(layered["layer_z_cm"] == expected_layers, f"default layered map should use 13 z layers: {layered['layer_z_cm']}")
    assert_true(layered["layer_band_cm"] == 25.0, f"default layered map should use +/-25cm layer band: {layered}")
    assert_true(layered["occupied_threshold"] == 2, f"default layered map should use occupancy threshold=2: {layered}")
    assert_true(len(layered["layers"]) == 13, f"expected 13 layers: {layered}")
    assert_true(all(int(layer["point_count"]) == 16 for layer in layered["layers"]), f"each synthetic layer should keep its own points: {layered}")


def test_default_route6_layer_band_does_not_overlap_adjacent_50cm_layers() -> None:
    builder = load_builder()
    cloud = np.asarray(
        [
            [1.0, 1.0, 4.50, 0.0, 0.0, 0.0],
            [2.0, 2.0, 5.00, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    layered = builder.build_route6_layered_occupancy_maps(cloud, resolution_m=0.25, occupied_threshold=1)
    counts = {int(layer["z_cm"]): int(layer["point_count"]) for layer in layered["layers"]}
    assert_true(counts[450] == 1 and counts[500] == 1, f"adjacent layers should not share points: {counts}")
    assert_true(sum(counts.values()) == 2, f"default +/-25cm bands should avoid overlap: {counts}")


def test_route6_voxel_downsample_reduces_duplicate_points_before_layering() -> None:
    builder = load_builder()
    duplicate_a = np.repeat(np.asarray([[0.01, 0.01, 0.50, 10.0, 20.0, 30.0]], dtype=np.float32), 100, axis=0)
    duplicate_b = np.repeat(np.asarray([[0.30, 0.30, 0.50, 40.0, 50.0, 60.0]], dtype=np.float32), 50, axis=0)
    out_of_bounds = np.asarray([[65.0, 0.0, 0.50, 0.0, 0.0, 0.0]], dtype=np.float32)
    cloud = np.vstack([duplicate_a, duplicate_b, out_of_bounds])
    reduced = builder.voxel_downsample_point_cloud(
        cloud,
        voxel_size_m=0.25,
        fixed_world_bounds_cm=builder.DEFAULT_ROUTE6_FIXED_WORLD_BOUNDS_CM,
    )
    assert_true(reduced.shape[0] == 2, f"duplicates should merge into two in-bounds voxels: {reduced.shape}")
    assert_true(reduced.shape[0] < cloud.shape[0], f"downsample should reduce point count: {cloud.shape} -> {reduced.shape}")


def test_layered_occupancy_artifacts_are_written(tmp_dir: Path) -> None:
    builder = load_builder()
    result = builder.write_route6_layered_occupancy_artifacts(
        tmp_dir,
        sample_layered_cloud(),
        layer_z_cm=[50, 200, 350, 500, 650],
        layer_band_cm=40.0,
        resolution_m=0.25,
        occupied_threshold=1,
    )
    root = Path(result["layered_occupancy_dir"])
    assert_true(root.is_dir(), f"layered occupancy directory should exist: {result}")
    manifest_path = Path(result["manifest_path"])
    assert_true(manifest_path.is_file(), f"manifest should be written: {result}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_true(manifest["schema"] == "route6_layered_occupancy_manifest_v1", f"unexpected manifest: {manifest}")
    assert_true(manifest["layer_count"] == 5, f"manifest should include five layers: {manifest}")
    for layer in manifest["layers"]:
        for key in ("occupancy_grid_path", "occupancy_metadata_path", "occupancy_preview_path"):
            assert_true(Path(layer[key]).is_file(), f"missing layer artifact {key}: {layer}")
    assert_true((root / "z_050" / "occupancy_grid.npy").is_file(), f"z_050 layer artifact should exist under stable folder: {result}")
    assert_true((root / "z_650" / "occupancy_grid.png").is_file(), f"z_650 preview should exist under stable folder: {result}")


def test_route6_known_house_polygons_overlay_all_layers_and_plan_nav(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    builder = load_builder()
    harness = CaptureHarness(tmp_dir)
    run_dir = tmp_dir / "route6_update_map_known_houses"
    result = builder.write_route6_layered_occupancy_artifacts(run_dir, sample_layered_cloud(), resolution_m=0.25)

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_record_update_map_output_dir(harness, run_dir, source="test_known_houses")
    manifest = module.Route6ExploreControlMixin.route6_apply_known_house_polygons_to_update_map(harness, run_dir)
    plan = module.Route6ExploreControlMixin.route6_build_known_house_navigation_plan(harness, run_dir, target_house_id="001")

    polygon_path = run_dir / "map" / "known_house_polygons.json"
    assert_true(polygon_path.is_file(), "known house polygon artifact should be written")
    polygons = json.loads(polygon_path.read_text(encoding="utf-8"))
    assert_true(polygons["house_count"] == 5, f"five operator-provided houses should be recorded: {polygons}")
    records_by_id = {record["house_id"]: record for record in polygons["houses"]}
    assert_true(set(records_by_id) == {"001", "002", "003", "004", "005"}, f"known house ids should be stable: {records_by_id}")
    expected_bboxes = {
        "003": {"min_x": -3000.0, "max_x": -800.0, "min_y": 1450.0, "max_y": 2450.0},
        "004": {"min_x": -3600.0, "max_x": -1100.0, "min_y": -2400.0, "max_y": -1550.0},
        "005": {"min_x": -300.0, "max_x": 1700.0, "min_y": -2350.0, "max_y": -1400.0},
    }
    for house_id, expected_bbox in expected_bboxes.items():
        assert_true(records_by_id[house_id]["bbox"] == expected_bbox, f"house {house_id} bbox should match operator coordinates: {records_by_id[house_id]}")
    assert_true(manifest["known_house_overlay"]["house_count"] == 5, f"manifest should summarize known houses: {manifest}")
    assert_true(len(manifest["layers"]) == int(result["layer_count"]), f"all layers should remain present: {manifest}")
    for layer in manifest["layers"]:
        overlay_path = Path(layer["known_house_overlay_preview_path"])
        assert_true(overlay_path.is_file(), f"each layer should have a known-house overlay: {layer}")
        image = np.asarray(Image.open(overlay_path).convert("RGB"))
        colored = np.sum(np.any((image > 0) & (image < 245), axis=2) & ~np.all(image < 20, axis=2))
        assert_true(colored > 0, f"known house overlay should draw pale house outlines: {overlay_path}")
    assert_true(plan["schema"] == "route6_known_house_navigation_plan_v1", f"unexpected nav plan schema: {plan}")
    assert_true(plan["target_house_id"] == "001", f"house 1 should be planned by known coordinates: {plan}")
    assert_true(plan["target_pose_cm"]["x"] < 2800.0, f"navigation point should be outside house 1 bbox, not its center: {plan}")
    assert_true((run_dir / "route6_known_house_navigation_plan.json").is_file(), "known-house navigation plan should be written")
    scan_points = module.Route6ExploreControlMixin.route6_plan_selected_house_scan_points(harness, run_dir, "001")
    assert_true(scan_points and scan_points[0]["x"] == plan["target_pose_cm"]["x"], f"known nav point should seed the movement scan plan: {scan_points}")
    assert_true(scan_points[0]["view_type"] == "route6_nearest_facade_scout", f"known coordinate mode should use the Route 5-style scout movement point: {scan_points}")


def test_route6_known_house_overlay_is_bottom_light_reference_layer(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    records = module.Route6ExploreControlMixin.route6_enable_known_house_coordinate_mode(harness)
    metadata = {
        "width": 100,
        "height": 100,
        "resolution_m": 1.0,
        "origin_standard_m": [-50.0, -50.0],
    }
    image = Image.new("RGB", (100, 100), "white")
    obstacle_px = module.Route6ExploreControlMixin.route6_unreal_cm_to_layer_pixel(harness, metadata, 2800.0, -1400.0)
    image.putpixel(obstacle_px, (0, 0, 0))

    overlay = module.Route6ExploreControlMixin.route6_draw_known_house_overlay_on_image(harness, image, metadata, records)
    arr = np.asarray(overlay.convert("RGB"))
    obstacle_color = tuple(int(value) for value in arr[obstacle_px[1], obstacle_px[0], :])
    colored_mask = np.any((arr > 0) & (arr < 245), axis=2) & ~np.all(arr < 20, axis=2)
    colored_pixels = arr[colored_mask]

    assert_true(obstacle_color == (0, 0, 0), f"black obstacle pixels should stay above the known-house layer: {obstacle_color}")
    assert_true(colored_pixels.size > 0, "known house reference should still draw visible pixels")
    assert_true(int(np.min(colored_pixels)) >= 120, f"known house layer should use pale reference colors: min={int(np.min(colored_pixels))}")


def test_route6_layered_map_uses_fixed_bounds_and_black_obstacle_preview() -> None:
    builder = load_builder()
    cloud = np.asarray(
        [
            [0.0, 0.0, 0.50, 0.0, 0.0, 0.0],
            [49.9, 49.9, 0.50, 0.0, 0.0, 0.0],
            [65.0, 0.0, 0.50, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    layered = builder.build_route6_layered_occupancy_maps(
        cloud,
        layer_z_cm=[50],
        layer_band_cm=40.0,
        resolution_m=0.25,
        occupied_threshold=1,
    )
    occupancy = layered["layers"][0]["occupancy"]
    grid = np.asarray(occupancy["grid"])
    assert_true(occupancy["fixed_world_bounds_cm"] == {"min_x": -6000, "max_x": 6000, "min_y": -6000, "max_y": 6000}, f"fixed cm bounds should be recorded: {occupancy}")
    assert_true(occupancy["origin_standard_m"] == [-60.0, -60.0], f"origin should map -6000cm to -60m: {occupancy}")
    assert_true(occupancy["width"] == 480 and occupancy["height"] == 480, f"120m at 0.25m should produce 480x480 grid: {occupancy}")
    assert_true(occupancy["in_bounds_point_count"] == 2, f"out-of-bounds points should be ignored for fixed map: {occupancy}")
    assert_true(occupancy["out_of_bounds_point_count"] == 1, f"out-of-bounds count should be visible: {occupancy}")
    assert_true(int(np.sum(grid >= 100)) == 2, f"two small occupied cells should be marked: {occupancy}")
    preview = builder.occupancy_preview_image(occupancy)
    assert_true(preview.dtype == np.uint8, f"preview should be uint8: {preview.dtype}")
    assert_true(int(np.min(preview)) == 0, f"occupied cells should render black: min={int(np.min(preview))}")
    assert_true(int(np.max(preview)) == 255, f"free/no-obstacle cells should render white: max={int(np.max(preview))}")


def test_route6_update_map_window_contract() -> None:
    source = (PROJECT_ROOT / "control" / "route6_explore_control.py").read_text(encoding="utf-8")
    panel_source = (PROJECT_ROOT / "control" / "panel.py").read_text(encoding="utf-8")
    for text in (
        "Route 6 Update Map",
        "open_route6_update_map_window",
        "route6_update_map_layer_var",
        "route6_update_map_scroll_canvas",
        "route6_update_map_layer_combo",
        "route6_update_map_pose_var",
        "refresh_route6_update_map_window",
        "route6_update_map_uav_pose_text",
        "route6_draw_update_map_uav_overlay",
        "_on_route6_update_map_mousewheel",
        "<MouseWheel>",
        "<Button-4>",
        "<Button-5>",
    ):
        assert_true(text in source or text in panel_source, f"Route 6 Update Map UI should declare {text}")
    assert_true("bind_all" not in source[source.find("open_route6_update_map_window"):source.find("def close_route6_update_map_window")], "Route 6 Update Map wheel handling should not bind globally to the main window")


def test_route6_uav_overlay_draws_heading_arrow_and_compass(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.latest_state = {"pose": [0.0, 0.0, 300.0, -90.0]}
    metadata_path = tmp_dir / "occupancy_grid.json"
    metadata_path.write_text(
        json.dumps(
            {
                "width": 100,
                "height": 100,
                "resolution_m": 1.0,
                "origin_standard_m": [-50.0, -50.0],
            }
        ),
        encoding="utf-8",
    )
    image = Image.new("RGB", (100, 100), "white")
    layer_record = {"occupancy_metadata_path": str(metadata_path)}

    overlay = module.Route6ExploreControlMixin.route6_draw_update_map_uav_overlay(harness, image, layer_record, scale=1)
    arr = np.asarray(overlay)
    center_x = 50
    center_y = 49
    red_pixels_above = np.sum(
        (arr[: center_y - 8, center_x - 3 : center_x + 4, 0] > 180)
        & (arr[: center_y - 8, center_x - 3 : center_x + 4, 1] < 80)
        & (arr[: center_y - 8, center_x - 3 : center_x + 4, 2] < 80)
    )
    compass_nonwhite = np.sum(np.any(arr[4:44, 4:44] < 245, axis=2))
    source = (PROJECT_ROOT / "control" / "route6_explore_control.py").read_text(encoding="utf-8")
    compass_source = source[source.find("def route6_draw_update_map_compass_overlay"):source.find("def route6_draw_update_map_uav_overlay")]

    assert_true(red_pixels_above > 0, "UAV overlay should draw a yaw direction arrow beyond the old crosshair radius")
    assert_true(compass_nonwhite > 0, "UAV overlay should draw a compass/heading marker in the top-left corner")
    for label in ('"N"', '"E"', '"S"', '"W"'):
        assert_true(label in compass_source, f"top-left compass should include cardinal label {label}")


def test_route6_update_map_capture_buttons_and_handlers_contract() -> None:
    source = (PROJECT_ROOT / "control" / "route6_explore_control.py").read_text(encoding="utf-8")
    for text in (
        "Start Capture",
        "Stop Capture",
        "Generate Map",
        "Start Realtime Update",
        "Stop Realtime Update",
        "Stop Map + Lock Movement",
        "Min move cm",
        "Min yaw deg",
        "on_route6_update_map_start_capture",
        "on_route6_update_map_stop_capture",
        "on_route6_update_map_generate_map",
        "on_route6_update_map_start_realtime",
        "on_route6_update_map_stop_realtime",
        "on_route6_update_map_stop_and_lock_movement",
        "route6_stop_map_capture_and_lock_movement",
        "route6_update_map_capture_worker",
        "route6_update_map_realtime_worker",
        "route6_update_map_capture_once",
        "route6_update_map_capture_stop_event",
        "route6_update_map_realtime_stop_event",
    ):
        assert_true(text in source, f"Route 6 Update Map capture UI should declare {text}")


def test_route6_update_map_capture_once_writes_lidar_log_and_generates_map(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    output_dir = module.Route6ExploreControlMixin.make_route6_update_map_output_dir(harness)
    capture = module.Route6ExploreControlMixin.route6_update_map_capture_once(harness, harness.session, output_dir)
    assert_true(capture["capture_status"] == "ok", f"capture should report ok: {capture}")
    assert_true(Path(capture["point_cloud_world_standard_m_npy_path"]).is_file(), f"capture should write a point cloud: {capture}")
    log_path = output_dir / "lidar_capture_log.jsonl"
    assert_true(log_path.is_file(), f"capture should append lidar_capture_log.jsonl under output dir: {output_dir}")
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_true(rows and rows[-1]["capture_kind"] == "route6_update_map_capture", f"unexpected capture log rows: {rows}")
    result = module.Route6ExploreControlMixin.on_route6_update_map_generate_map(harness)
    assert_true(result and Path(result["manifest_path"]).is_file(), f"Generate Map should create layered manifest: {result}")
    assert_true(result["layer_count"] == 13, f"Generate Map should create 13 layers: {result}")
    assert_true(harness.route6_update_map_status_var.get().startswith("Route 6 Update Map:"), "status var should be updated after Generate Map")


def test_route6_update_map_realtime_worker_captures_and_rebuilds(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    harness.route6_update_map_capture_interval_s_var = ValueVar("0.01")
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    output_dir = module.Route6ExploreControlMixin.make_route6_update_map_output_dir(harness)

    result = module.Route6ExploreControlMixin.route6_update_map_realtime_worker(
        harness,
        harness.session,
        output_dir,
        max_iterations=2,
    )

    assert_true(result["capture_count"] == 2, f"realtime worker should capture two frames: {result}")
    assert_true(result["map_count"] == 2, f"realtime worker should rebuild after each capture: {result}")
    assert_true(len(harness.session.capture_calls) == 2, f"session should be captured twice: {harness.session.capture_calls}")
    assert_true(Path(result["last_manifest_path"]).is_file(), f"realtime worker should write a manifest: {result}")
    realtime_state = harness.llm_route6_state.get("route6_update_map_realtime", {})
    assert_true(realtime_state.get("running") is False, f"realtime state should stop cleanly: {realtime_state}")
    assert_true(realtime_state.get("map_count") == 2, f"realtime state should record map count: {realtime_state}")


def test_route6_update_map_realtime_preserves_active_search_output_dir(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    harness.route6_update_map_capture_interval_s_var = ValueVar("0.01")
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    root = harness.route6_output_root()
    search_dir = root / "route6_nearest_map_20260525-210527-423"
    search_dir.mkdir(parents=True, exist_ok=True)
    harness.llm_route6_state["output_dir"] = str(search_dir)
    update_dir = module.Route6ExploreControlMixin.make_route6_update_map_output_dir(harness)

    result = module.Route6ExploreControlMixin.route6_update_map_realtime_worker(
        harness,
        harness.session,
        update_dir,
        max_iterations=1,
    )

    assert_true(result["capture_count"] == 1, f"realtime worker should still capture map data: {result}")
    assert_true(
        harness.llm_route6_state.get("output_dir") == str(search_dir),
        f"active LLM search output_dir must stay on the movement run: {harness.llm_route6_state}",
    )
    assert_true(
        harness.llm_route6_state.get("route6_update_map_output_dir") == str(update_dir),
        f"update-map output should be tracked separately: {harness.llm_route6_state}",
    )
    realtime_state = harness.llm_route6_state.get("route6_update_map_realtime", {})
    update_state = harness.llm_route6_state.get("route6_update_map", {})
    assert_true(realtime_state.get("output_dir") == str(update_dir), f"realtime state should keep update-map dir: {realtime_state}")
    assert_true(update_state.get("output_dir") == str(update_dir), f"map state should keep update-map dir: {update_state}")
    latest = module.Route6ExploreControlMixin.route6_update_map_latest_output_dir(harness)
    assert_true(latest == update_dir, f"latest update-map lookup should prefer realtime map dir, not search dir: {latest}")


def test_route6_update_map_realtime_worker_postprocesses_raw_depth_before_rebuild(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.session = FakeRawDepthSession()
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    harness.route6_update_map_capture_interval_s_var = ValueVar("0.01")
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    output_dir = module.Route6ExploreControlMixin.make_route6_update_map_output_dir(harness)

    result = module.Route6ExploreControlMixin.route6_update_map_realtime_worker(
        harness,
        harness.session,
        output_dir,
        max_iterations=1,
    )

    standard_path = output_dir / "frames" / "frame_000001" / "point_cloud_world_standard_m.npy"
    merged_path = output_dir / "map" / "route6_update_map_merged_point_cloud_world_standard_m.npy"
    cleanup_path = output_dir / "map" / "route6_update_map_pointcloud_cleanup.json"
    assert_true(not standard_path.exists(), f"raw realtime frame pointcloud should be deleted after map rebuild: {result}")
    assert_true(merged_path.is_file(), f"map-level merged pointcloud should preserve accumulated map data: {result}")
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    assert_true(str(standard_path) in cleanup["deleted_paths"], f"cleanup should record deleted postprocessed frame cloud: {cleanup}")
    assert_true(result["capture_count"] == 1, f"realtime raw-depth worker should capture once: {result}")
    assert_true(result["map_count"] == 1, f"realtime raw-depth worker should build a map after postprocess: {result}")
    assert_true(Path(result["last_manifest_path"]).is_file(), f"raw-depth realtime should write manifest: {result}")


def test_route6_update_map_ignores_generated_frame_pointcloud_artifacts(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    run_dir = harness.route6_output_root() / "route6_update_map_20260525-193055-446"
    capture_dir = run_dir / "frames" / "frame_000001"
    capture_dir.mkdir(parents=True, exist_ok=True)
    np.save(capture_dir / "point_cloud_world_standard_m.npy", sample_layered_cloud())
    np.save(capture_dir / "point_cloud_camera.npy", sample_layered_cloud())
    open3d_dir = capture_dir / "open3d"
    open3d_dir.mkdir(parents=True, exist_ok=True)
    np.save(open3d_dir / "point_cloud_camera_open3d.npy", sample_layered_cloud())
    duplicate_dir = run_dir / "frames" / "frame_000500"
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    np.save(duplicate_dir / "point_cloud_world_standard_m.npy", sample_layered_cloud())
    log_row = {
        "capture_kind": "route6_update_map_capture",
        "frame_index": 1,
        "capture_dir": str(capture_dir),
        "capture_status": "ok",
    }
    harness.append_jsonl(run_dir / "route6_update_map_capture_log.jsonl", log_row)

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    candidates = module.Route6ExploreControlMixin.route6_capture_folder_candidate_pointcloud_paths(harness, run_dir)
    paths = module.Route6ExploreControlMixin.route6_capture_folder_pointcloud_paths(harness, run_dir)

    assert_true(candidates == [], f"generated frame pointcloud artifacts should not be re-imported as candidates: {candidates}")
    assert_true(paths == [capture_dir / "point_cloud_world_standard_m.npy"], f"capture log should keep only real capture frame pointclouds: {paths}")


def test_route6_update_map_realtime_skips_stationary_pose(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    harness.route6_update_map_capture_interval_s_var = ValueVar("0.01")
    harness.route6_update_map_min_move_cm_var = ValueVar("50")
    harness.route6_update_map_min_yaw_deg_var = ValueVar("5")
    harness.latest_state = {"pose": [0.0, 0.0, 100.0, 0.0]}
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    output_dir = module.Route6ExploreControlMixin.make_route6_update_map_output_dir(harness)

    result = module.Route6ExploreControlMixin.route6_update_map_realtime_worker(
        harness,
        harness.session,
        output_dir,
        max_iterations=3,
    )

    assert_true(result["capture_count"] == 1, f"stationary realtime update should keep only the first capture: {result}")
    assert_true(result["skipped_stationary_count"] == 2, f"stationary loop should report skipped frames: {result}")
    assert_true(len(harness.session.capture_calls) == 1, f"session should not collect useless stationary frames: {harness.session.capture_calls}")
    skip_path = output_dir / "route6_update_map_skip_events.jsonl"
    assert_true(skip_path.is_file(), "stationary realtime update should write lightweight skip events instead of pointcloud frames")
    skip_events = [json.loads(line) for line in skip_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert_true(skip_events[-1]["schema"] == "route6_update_map_skip_event_v1", f"unexpected skip event schema: {skip_events}")
    assert_true(skip_events[-1]["reason"] == "stationary_pose", f"stationary skip reason should be explicit: {skip_events}")


def test_route6_update_map_generation_records_premerge_voxel_reduction(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    root = harness.route6_output_root()
    run_dir = root / "route6_update_map_20260525-160000-000"
    frame_dir = run_dir / "frames" / "frame_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    duplicate_cloud = np.repeat(np.asarray([[0.01, 0.01, 0.50, 10.0, 20.0, 30.0]], dtype=np.float32), 1000, axis=0)
    np.save(frame_dir / "point_cloud_world_standard_m.npy", duplicate_cloud)

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_set_selected_capture_folder(harness, run_dir)
    result = module.Route6ExploreControlMixin.on_route6_capture_folder_generate_map(harness)
    assert_true(result["raw_point_count"] == 1000, f"raw point count should be tracked: {result}")
    assert_true(result["merged_point_count"] == 1, f"duplicate points should voxel merge before map build: {result}")
    assert_true(result["pointcloud_reduction_ratio"] < 0.01, f"reduction ratio should show strong compression: {result}")


def test_route6_update_map_deletes_frame_pointclouds_after_successful_build(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    run_dir = tmp_dir / "route6_update_map_cleanup"
    frame_dir = run_dir / "frames" / "frame_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    source_path = frame_dir / "point_cloud_world_standard_m.npy"
    extra_cloud_path = frame_dir / "point_cloud_camera.npy"
    nested_cloud_path = frame_dir / "open3d" / "point_cloud_world_standard_m.ply"
    nested_cloud_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(source_path, sample_layered_cloud())
    np.save(extra_cloud_path, sample_layered_cloud())
    nested_cloud_path.write_text("ply\n", encoding="utf-8")
    module.Route6ExploreControlMixin.ensure_route6_state(harness)

    result = module.Route6ExploreControlMixin.route6_update_map_build_from_pointcloud(harness, run_dir)

    assert_true(result and Path(result["manifest_path"]).is_file(), f"map build should succeed before cleanup: {result}")
    assert_true(not source_path.exists(), "frame pointcloud should be removed after it has been merged into the map")
    assert_true(not extra_cloud_path.exists(), "extra frame pointcloud arrays should also be removed after map update")
    assert_true(not nested_cloud_path.exists(), "nested frame pointcloud files should also be removed after map update")
    cleanup_path = run_dir / "map" / "route6_update_map_pointcloud_cleanup.json"
    assert_true(cleanup_path.is_file(), "cleanup artifact should record deleted frame pointclouds")
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    assert_true(cleanup["deleted_count"] >= 3, f"cleanup should record all deleted frame pointclouds: {cleanup}")
    assert_true(str(source_path) in cleanup["deleted_paths"], f"cleanup should identify the merged source: {cleanup}")


def test_route6_update_map_generate_map_guides_empty_runs_to_folder_reader(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    result = module.Route6ExploreControlMixin.on_route6_update_map_generate_map(harness)
    assert_true(result == {}, f"empty run should not create a map: {result}")
    status = str(harness.route6_update_map_status_var.get())
    assert_true("Open Capture Folders" in status, f"empty Generate Map should point operator to folder reader: {status}")


def test_route6_capture_folder_reader_window_contract() -> None:
    source = (PROJECT_ROOT / "control" / "route6_explore_control.py").read_text(encoding="utf-8")
    for text in (
        "Route 6 Capture Folder Reader",
        "Open Capture Folders",
        "Process Pointcloud Data",
        "Pointcloud Report",
        "open_route6_capture_folder_reader_window",
        "refresh_route6_capture_folder_list",
        "route6_capture_folder_listbox",
        "route6_pointcloud_report_text",
        "route6_process_capture_folder_pointclouds",
        "on_route6_capture_folder_process_pointcloud",
        "on_route6_capture_folder_generate_map",
        "on_route6_capture_folder_load_map",
        "Load Map",
    ):
        assert_true(text in source, f"Route 6 capture folder reader should declare {text}")
    start = source.find("def open_route6_capture_folder_reader_window")
    end = source.find("def close_route6_capture_folder_reader_window")
    assert_true(start >= 0 and end > start, "capture folder reader window should have an open/close method slice")
    assert_true("bind_all" not in source[start:end], "capture folder reader wheel/list handling should not bind globally")


def test_route6_pointcloud_processing_reports_empty_folder(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    run_dir = write_capture_folder(harness.route6_output_root(), "route6_update_map_20260525-140000-000", with_cloud=False, mtime=3000)

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_set_selected_capture_folder(harness, run_dir)
    report = module.Route6ExploreControlMixin.on_route6_capture_folder_process_pointcloud(harness)

    assert_true(report.get("status") == "missing_pointcloud", f"empty folder should report missing pointcloud: {report}")
    assert_true(report.get("candidate_file_count") == 0, f"empty folder should have no candidate files: {report}")
    assert_true(Path(report["report_path"]).is_file(), f"process report should be written for operator debugging: {report}")
    assert_true("no pointcloud" in str(harness.route6_capture_folder_status_var.get()).lower(), f"status should explain empty folder: {harness.route6_capture_folder_status_var.get()}")


def test_route6_pointcloud_processing_standardizes_raw_npy_and_generates_map(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    run_dir = harness.route6_output_root() / "route6_update_map_20260525-150000-000"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "lidar_points.npy", sample_layered_cloud()[:, :3])

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_set_selected_capture_folder(harness, run_dir)
    report = module.Route6ExploreControlMixin.on_route6_capture_folder_process_pointcloud(harness)

    standard_path = run_dir / "frames" / "frame_000001" / "point_cloud_world_standard_m.npy"
    assert_true(report.get("status") == "processed", f"raw npy should be processed: {report}")
    assert_true(Path(report["report_path"]).is_file(), f"process report should be written: {report}")
    assert_true(standard_path.is_file(), f"raw npy should be standardized into Route 6 frame path: {report}")
    cloud = np.asarray(np.load(standard_path))
    assert_true(cloud.ndim == 2 and cloud.shape[1] == 6, f"standard pointcloud should be Nx6: {cloud.shape}")

    build = module.Route6ExploreControlMixin.on_route6_capture_folder_generate_map(harness)
    assert_true(build and Path(build["manifest_path"]).is_file(), f"processed pointcloud should generate map: {build}")


def test_route6_pointcloud_processing_postprocesses_raw_depth_frames(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    run_dir = harness.route6_output_root() / "route6_update_map_20260525-153000-000"
    frame_dir = run_dir / "frames" / "frame_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    depth = np.full((16, 16), 300.0, dtype=np.float32)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 1] = 180
    np.save(frame_dir / "depth.npy", depth)
    from PIL import Image

    Image.fromarray(rgb).save(frame_dir / "rgb.png")
    camera_info = {
        "location": {"x": 0.0, "y": 0.0, "z": 200.0},
        "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "horizontal_fov_deg": 90.0,
        "image_width": 16,
        "image_height": 16,
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
    }
    (frame_dir / "camera_info.json").write_text(json.dumps(camera_info, indent=2), encoding="utf-8")
    capture = {
        "capture_dir": str(frame_dir),
        "depth_npy_path": str(frame_dir / "depth.npy"),
        "rgb_path": str(frame_dir / "rgb.png"),
        "camera_info_path": str(frame_dir / "camera_info.json"),
        "lidar_depth_min_cm": 20.0,
        "lidar_depth_max_cm": 1200.0,
        "raw_capture_only": True,
        "postprocess_status": "pending",
    }
    (frame_dir / "capture.json").write_text(json.dumps(capture, indent=2), encoding="utf-8")

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    module.Route6ExploreControlMixin.route6_set_selected_capture_folder(harness, run_dir)
    report = module.Route6ExploreControlMixin.on_route6_capture_folder_process_pointcloud(harness)

    standard_path = frame_dir / "point_cloud_world_standard_m.npy"
    assert_true(report.get("status") == "processed", f"raw depth frame should be postprocessed: {report}")
    assert_true(report.get("raw_depth_frame_count") == 1, f"raw depth frame count should be reported: {report}")
    assert_true(report.get("postprocessed_frame_count") == 1, f"postprocessed frame count should be reported: {report}")
    assert_true(standard_path.is_file(), f"raw depth should produce standard pointcloud: {report}")
    cloud = np.asarray(np.load(standard_path))
    assert_true(cloud.ndim == 2 and cloud.shape[1] == 6 and cloud.shape[0] > 0, f"standard pointcloud should be non-empty Nx6: {cloud.shape}")


def write_capture_folder(root: Path, name: str, *, with_cloud: bool, mtime: int) -> Path:
    run_dir = root / name
    frame_dir = run_dir / "frames" / "frame_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if with_cloud:
        np.save(frame_dir / "point_cloud_world_standard_m.npy", sample_layered_cloud())
    os.utime(frame_dir, (mtime, mtime))
    os.utime(run_dir, (mtime, mtime))
    return run_dir


def test_route6_capture_folder_reader_selects_generates_and_loads_map(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    root = harness.route6_output_root()
    empty_dir = write_capture_folder(root, "route6_update_map_20260525-120000-000", with_cloud=False, mtime=1000)
    cloud_dir = write_capture_folder(root, "route6_update_map_20260525-130000-000", with_cloud=True, mtime=2000)

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    records = module.Route6ExploreControlMixin.route6_list_capture_folders(harness)
    assert_true(records and Path(records[0]["path"]) == cloud_dir, f"newest capture folder should be first: {records}")
    assert_true(records[0]["pointcloud_count"] == 1, f"cloud folder should expose pointcloud_count: {records}")
    assert_true(any(Path(record["path"]) == empty_dir for record in records), f"empty capture folder should still be listed: {records}")

    module.Route6ExploreControlMixin.route6_set_selected_capture_folder(harness, cloud_dir)
    build = module.Route6ExploreControlMixin.on_route6_capture_folder_generate_map(harness)
    assert_true(build and Path(build["manifest_path"]).is_file(), f"Generate Map should use selected folder: {build}")

    harness.llm_route6_state.pop("route6_update_map", None)
    load = module.Route6ExploreControlMixin.on_route6_capture_folder_load_map(harness)
    assert_true(load and load.get("schema") == "route6_layered_occupancy_manifest_v1", f"Load Map should read selected manifest: {load}")
    assert_true(harness.llm_route6_state.get("output_dir") == str(cloud_dir), f"selected output dir should be persisted: {harness.llm_route6_state}")
    assert_true("loaded" in str(harness.route6_update_map_status_var.get()).lower(), f"load should update status: {harness.route6_update_map_status_var.get()}")


def test_route6_window6_realtime_refresh_reads_manifest_without_rebuild(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    run_dir = harness.route6_output_root() / "route6_update_map_20260525-210000-000"
    frame_dir = run_dir / "frames" / "frame_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.save(frame_dir / "point_cloud_world_standard_m.npy", sample_layered_cloud())

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    build = module.Route6ExploreControlMixin.route6_update_map_build_from_pointcloud(harness, run_dir)
    assert_true(build and Path(build["manifest_path"]).is_file(), f"test setup should create a manifest: {build}")
    harness.llm_route6_state["output_dir"] = str(run_dir)

    def fail_build(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("Window 6 realtime refresh must not rebuild the map")

    harness.route6_update_map_build_from_pointcloud = fail_build
    manifest = module.Route6ExploreControlMixin.refresh_llm_route6_realtime_map(harness)

    assert_true(manifest.get("schema") == "route6_layered_occupancy_manifest_v1", f"refresh should read manifest: {manifest}")
    assert_true("loaded" in str(harness.llm_route6_realtime_map_status_var.get()).lower(), f"status should report loaded map: {harness.llm_route6_realtime_map_status_var.get()}")
    assert_true(str(harness.route6_update_map_layer_var.get()).startswith("z_"), f"refresh should keep/select a layer: {harness.route6_update_map_layer_var.get()}")


def test_route6_llm_analysis_reads_manifest_without_rebuild(tmp_dir: Path) -> None:
    module = importlib.import_module("control.route6_explore_control")
    harness = CaptureHarness(tmp_dir)
    harness.llm_route6_output_root_override = tmp_dir / "route6_explore_runs"
    run_dir = harness.route6_output_root() / "route6_update_map_20260525-230000-000"
    frame_dir = run_dir / "frames" / "frame_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.save(frame_dir / "point_cloud_world_standard_m.npy", sample_layered_cloud())

    module.Route6ExploreControlMixin.ensure_route6_state(harness)
    build = module.Route6ExploreControlMixin.route6_update_map_build_from_pointcloud(harness, run_dir)
    assert_true(build and Path(build["manifest_path"]).is_file(), f"test setup should create a manifest: {build}")
    harness.llm_route6_state["route6_update_map_output_dir"] = str(run_dir)

    def fail_build(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("LLM map analysis must read the existing manifest without rebuilding")

    harness.route6_update_map_build_from_pointcloud = fail_build
    context = module.Route6ExploreControlMixin.route6_build_semantic_map_context(harness)

    assert_true(context.get("schema") == "route6_semantic_map_context_v1", f"semantic context should be built: {context}")
    assert_true(context.get("manifest_path") == str(build["manifest_path"]), f"semantic context should use existing manifest: {context}")
    assert_true(context.get("layer_count") == 13, f"semantic analysis should see all 13 layers: {context}")


def run_with_tmp(test_fn) -> None:
    with tempfile.TemporaryDirectory(prefix="route6_update_map_") as raw:
        test_fn(Path(raw))


def main() -> None:
    tests = [
        test_layered_occupancy_projects_pointcloud_by_z_layers,
        test_default_route6_layers_are_every_50cm_from_50_to_650,
        test_default_route6_layer_band_does_not_overlap_adjacent_50cm_layers,
        test_route6_voxel_downsample_reduces_duplicate_points_before_layering,
        lambda: run_with_tmp(test_layered_occupancy_artifacts_are_written),
        lambda: run_with_tmp(test_route6_known_house_polygons_overlay_all_layers_and_plan_nav),
        lambda: run_with_tmp(test_route6_known_house_overlay_is_bottom_light_reference_layer),
        test_route6_layered_map_uses_fixed_bounds_and_black_obstacle_preview,
        test_route6_update_map_window_contract,
        lambda: run_with_tmp(test_route6_uav_overlay_draws_heading_arrow_and_compass),
        test_route6_update_map_capture_buttons_and_handlers_contract,
        lambda: run_with_tmp(test_route6_update_map_capture_once_writes_lidar_log_and_generates_map),
        lambda: run_with_tmp(test_route6_update_map_realtime_worker_captures_and_rebuilds),
        lambda: run_with_tmp(test_route6_update_map_realtime_preserves_active_search_output_dir),
        lambda: run_with_tmp(test_route6_update_map_realtime_worker_postprocesses_raw_depth_before_rebuild),
        lambda: run_with_tmp(test_route6_update_map_ignores_generated_frame_pointcloud_artifacts),
        lambda: run_with_tmp(test_route6_update_map_realtime_skips_stationary_pose),
        lambda: run_with_tmp(test_route6_update_map_generation_records_premerge_voxel_reduction),
        lambda: run_with_tmp(test_route6_update_map_deletes_frame_pointclouds_after_successful_build),
        lambda: run_with_tmp(test_route6_update_map_generate_map_guides_empty_runs_to_folder_reader),
        test_route6_capture_folder_reader_window_contract,
        lambda: run_with_tmp(test_route6_pointcloud_processing_reports_empty_folder),
        lambda: run_with_tmp(test_route6_pointcloud_processing_standardizes_raw_npy_and_generates_map),
        lambda: run_with_tmp(test_route6_pointcloud_processing_postprocesses_raw_depth_frames),
        lambda: run_with_tmp(test_route6_capture_folder_reader_selects_generates_and_loads_map),
        lambda: run_with_tmp(test_route6_window6_realtime_refresh_reads_manifest_without_rebuild),
        lambda: run_with_tmp(test_route6_llm_analysis_reads_manifest_without_rebuild),
    ]
    for test in tests:
        test()
        print(f"PASS {getattr(test, '__name__', '<lambda>')}")
    print("PASS route6 update map verification")


if __name__ == "__main__":
    main()
