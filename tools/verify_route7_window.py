from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


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


class PreviewLabelHarness:
    def __init__(self) -> None:
        self.configs: List[Dict[str, Any]] = []

    def configure(self, **kwargs: Any) -> None:
        self.configs.append(dict(kwargs))


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


class Route7StaticBaseHarness(Route7RunHarness, RouteControlMixin):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.map_config = {
            "houses": [
                {"id": "002", "center_x": 500.0, "center_y": -400.0, "radius_cm": 500.0},
                {"id": "003", "center_x": -300.0, "center_y": 250.0, "radius_cm": 420.0},
                {"id": "006", "center_x": -7600.0, "center_y": -2000.0, "radius_cm": 500.0},
                {"id": "007", "center_x": -7350.0, "center_y": 200.0, "radius_cm": 500.0},
                {"id": "008", "center_x": -6800.0, "center_y": 2360.0, "radius_cm": 500.0},
                {"id": "house_13", "center_x": 4170.0, "center_y": 2410.0, "radius_cm": 500.0},
            ]
        }

    def load_map_resources(self, force: bool = False) -> bool:
        return True

    def _as_float_or_none(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def route6_known_house_polygon_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for index in range(1, 6):
            house_id = f"{index:03d}"
            center_x = float(index * 1000)
            center_y = float(index * -500)
            records.append(
                {
                    "house_id": house_id,
                    "name": f"house_{index}",
                    "bbox": {
                        "min_x": center_x - 100.0,
                        "max_x": center_x + 100.0,
                        "min_y": center_y - 100.0,
                        "max_y": center_y + 100.0,
                    },
                    "source": "operator_known_coordinates",
                }
            )
        return records


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

    def write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class Route7MultiLayerRouteHarness(Route7MapRouteHarness):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.layer_records: List[Dict[str, Any]] = []
        for z_cm, blocked in ((300.0, True), (350.0, False)):
            key = f"z_{int(z_cm):03d}"
            layer_dir = self.output_dir / "map" / "layered_occupancy" / key
            layer_dir.mkdir(parents=True, exist_ok=True)
            grid = np.zeros((21, 21), dtype=np.int16)
            if blocked:
                grid[10, 0:21] = 100
            grid_path = layer_dir / "occupancy_grid.npy"
            metadata_path = layer_dir / "occupancy_grid.json"
            np.save(grid_path, grid)
            metadata = {
                "width": 21,
                "height": 21,
                "resolution_m": 0.25,
                "origin_standard_m": [0.0, 0.0],
                "occupied_cell_count": int(np.sum(grid >= 100)),
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.layer_records.append(
                {
                    "schema": "route6_layered_occupancy_layer_artifact_v1",
                    "z_cm": z_cm,
                    "point_count": 120,
                    "occupied_cell_count": int(np.sum(grid >= 100)),
                    "occupancy_metadata_path": str(metadata_path),
                    "occupancy_grid_path": str(grid_path),
                    "occupancy_preview_path": "",
                }
            )

    def route6_update_map_load_manifest(self, *, build_if_missing: bool = False, output_dir: Path | None = None) -> Dict[str, Any]:
        return {
            "schema": "route6_layered_occupancy_manifest_v1",
            "layers": list(self.layer_records),
        }


class Route7TargetHouseBoundaryHarness(Route7MapRouteHarness):
    def house_world_bbox_for_id(self, _house_id: str) -> Dict[str, float]:
        return {"min_x": 0.0, "max_x": 1000.0, "min_y": -800.0, "max_y": 0.0, "center_x": 500.0, "center_y": -400.0}

    def route3_safety_report_for_pose(self, target_house_id: str, pose: Dict[str, Any]) -> Dict[str, Any]:
        bbox = self.house_world_bbox_for_id(target_house_id)
        x = float(pose.get("x", 0.0) or 0.0)
        y = float(pose.get("y", 0.0) or 0.0)
        if bbox and bbox["min_x"] < x < bbox["max_x"] and bbox["min_y"] < y < bbox["max_y"]:
            return {
                "safe": False,
                "reason": "target_house_bbox",
                "blocking_house_id": target_house_id,
                "obstacle": bbox,
            }
        return {"safe": True, "reason": "verify_route7_map_safe"}


class Route7EdgeObservationHarness(Route7MapRouteHarness):
    def __init__(self, root: Path, *, block_first_point: bool = False) -> None:
        super().__init__(root)
        self.args = type("Args", (), {"lidar_depth_max_cm": 400.0, "lidar_depth_min_cm": 60.0})()
        grid = np.zeros((80, 80), dtype=np.int16)
        if block_first_point:
            grid[10, 5] = 100
        np.save(self.grid_path, grid)
        metadata = {
            "width": 80,
            "height": 80,
            "resolution_m": 0.25,
            "origin_standard_m": [0.0, 0.0],
            "occupied_cell_count": int(np.sum(grid >= 100)),
        }
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.llm_route5_state.update(
            {
                "route_window_label": "V7",
                "target_house_id": "002",
                "output_dir": str(self.output_dir),
                "route7_map_output_dir": str(self.output_dir),
            }
        )

    def house_world_bbox_for_id(self, _house_id: str) -> Dict[str, float]:
        return {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 800.0, "center_x": 500.0, "center_y": 400.0}


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
    assert_true(first.name.startswith("house_002_autosearch_v7_or3_1_fused_"), f"V7 run name should identify OR3_1 v7: {first.name}")
    for subdir in ("frames", "open3d_frames", "reconstruction", "facade_observations", "map"):
        assert_true((first / subdir).is_dir(), f"V7 run should create {subdir}: {first}")

    selected = harness.route7_prepare_new_map_output_dir("002")
    assert_true(selected.parent == root, f"prepared V7 map dir should use dedicated root: {selected}")
    assert_true(harness.llm_route5_state.get("output_dir") == str(selected), harness.llm_route5_state)
    assert_true(harness.llm_route6_state.get("route6_update_map_output_dir") == str(selected), harness.llm_route6_state)
    assert_true(harness.llm_route6_state.get("output_dir") == str(selected), harness.llm_route6_state)
    assert_true(harness.llm_route5_state.get("mode") == "route7_llm_route_oa3_or3_1_fusion", harness.llm_route5_state)
    assert_true(str(harness.llm_route5_state.get("or3_1_model_path", "")).endswith("a_plus_3_1_model.pt"), harness.llm_route5_state)


def test_route7_uses_or3_1_as_primary_representation(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    harness.llm_route5_representation_model_var = HarnessVar(str(tmp_dir / "missing_or2_model.pt"))

    def fake_or3_1(event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "ok",
            "front_risk_state": "clearance_warning",
            "can_forward": True,
            "must_stop": False,
            "model_path": str(harness.default_route7_or31_model_path()),
            "prediction_json_path": str(harness.output_dir / "frame_000001" / "or3_1_risk_prediction.json"),
            "risk_overlay_path": str(harness.output_dir / "frame_000001" / "or3_1_risk_overlay.png"),
        }

    harness.route5_predict_obstacle_representation_3 = fake_or3_1
    prediction = harness.route5_predict_obstacle_representation(
        {
            "route_window_label": "V7",
            "route5_output_dir": str(harness.output_dir),
            "rgb_path": str(harness.output_dir / "missing_rgb.png"),
            "capture_dir": str(harness.output_dir / "frame_000001"),
        }
    )
    assert_true(prediction.get("status") == "ok", f"Route7 should not fall through to missing OR2 model: {prediction}")
    assert_true(prediction.get("route7_primary_representation") == "or3_1", prediction)
    assert_true(prediction.get("or3_variant") == "or3_1", prediction)
    assert_true(str(prediction.get("model_path", "")).endswith("a_plus_3_1_model.pt"), prediction)


def test_route7_static_house_base_is_frozen_at_run_start(tmp_dir: Path) -> None:
    harness = Route7StaticBaseHarness(tmp_dir)
    selected = harness.route7_prepare_new_map_output_dir("002")
    snapshot = harness.llm_route5_state.get("route7_static_house_base", {})
    houses = snapshot.get("houses", []) if isinstance(snapshot, dict) else []
    assert_true(len(houses) == 5, f"V7 static base should only include the current five house coordinates: {houses}")
    assert_true({item.get("house_id") for item in houses} == {"001", "002", "003", "004", "005"}, houses)
    house_002 = next((item for item in houses if item.get("house_id") == "002"), {})
    assert_true(snapshot.get("schema") == "route7_static_house_base_v1", snapshot)
    assert_true(house_002.get("bbox_world"), f"V7 should freeze house bbox records at run start: {snapshot}")
    original_bbox = dict(house_002.get("bbox_world", {}))
    harness.map_config["houses"][0]["center_x"] = 9999.0
    harness.map_config["houses"][0]["center_y"] = 9999.0
    frozen_bbox = next(
        item.get("bbox_world", {})
        for item in harness.llm_route5_state.get("route7_static_house_base", {}).get("houses", [])
        if item.get("house_id") == "002"
    )
    assert_true(frozen_bbox == original_bbox, f"static house base should not update after map_config changes: {frozen_bbox} vs {original_bbox}")
    artifact = selected / "map" / "route7_static_house_base.json"
    assert_true(artifact.is_file(), f"Route7 should write static house base artifact: {artifact}")


def test_route7_static_house_base_filters_legacy_full_map_snapshots(tmp_dir: Path) -> None:
    harness = Route7StaticBaseHarness(tmp_dir)
    harness.llm_route5_state["route7_static_house_base"] = {
        "schema": "route7_static_house_base_v1",
        "houses": [
            {"house_id": f"{index:03d}", "bbox_world": {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1}}
            for index in range(1, 9)
        ],
    }
    filtered = harness.route7_static_house_base()
    houses = filtered.get("houses", []) if isinstance(filtered.get("houses", []), list) else []
    assert_true(len(houses) == 5, f"legacy full-map snapshot should be filtered before drawing: {filtered}")
    assert_true({item.get("house_id") for item in houses} == {"001", "002", "003", "004", "005"}, houses)


def test_route7_open3d_frame_stream_indexes_are_separate(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    output_dir = harness.output_dir
    (output_dir / "frames" / "frame_000002").mkdir(parents=True)
    (output_dir / "open3d_frames" / "frame_000007").mkdir(parents=True)
    harness.append_jsonl(output_dir / "lidar_capture_log.jsonl", {"frame_index": 2, "frame_stream": "frames"})
    harness.append_jsonl(output_dir / "lidar_capture_log.jsonl", {"frame_index": 7, "frame_stream": "open3d_frames"})
    assert_true(harness.route2_next_frame_index(output_dir, frames_subdir="frames") == 3, "ordinary OR2 frames should keep their own index space")
    assert_true(harness.route2_next_frame_index(output_dir, frames_subdir="open3d_frames") == 8, "Open3D frames should keep their own index space")


def test_route7_edge_observation_minimal_points_use_lidar_formula(tmp_dir: Path) -> None:
    harness = Route7EdgeObservationHarness(tmp_dir)
    attempts = harness.route7_edge_observation_attempts_for_facade("002", "south", output_dir=harness.output_dir)
    expected_count = max(1, int(math.ceil(1000.0 / (400.0 * 0.8))))
    assert_true(len(attempts) == expected_count, f"Route7 should generate the minimum lidar edge points: {attempts}")
    forbidden_sources = {"center_projection", "left_third_projection", "right_third_projection", "far_center_projection", "elevated_center_projection"}
    sources = {str(item.get("observation_attempt_source", "")) for item in attempts}
    assert_true(sources == {"route7_lidar_edge_minimal"}, f"Route7 should not reuse old observation sources: {sources}")
    assert_true(not (sources & forbidden_sources), f"Old Route3 observation sources should be absent: {sources}")
    formula = attempts[0].get("route7_observation_formula", {})
    assert_true(formula.get("point_count") == expected_count, formula)
    assert_true(round(float(formula.get("effective_lidar_radius_cm", 0.0)), 3) == 320.0, formula)
    assert_true({round(float(item.get("z", 0.0)), 3) for item in attempts} == {300.0}, attempts)


def test_route7_edge_observation_replans_points_inside_obstacles(tmp_dir: Path) -> None:
    harness = Route7EdgeObservationHarness(tmp_dir, block_first_point=True)
    attempts = harness.route7_edge_observation_attempts_for_facade("002", "south", output_dir=harness.output_dir)
    replanned = [item for item in attempts if item.get("route7_observation_replanned_from_blocked")]
    assert_true(replanned, f"Route7 should redesign an edge observation point that lands inside an occupied cell: {attempts}")
    first = replanned[0]
    assert_true(first.get("route7_original_map_cell") == [10, 5], first)
    assert_true(first.get("route7_map_cell") != [10, 5], first)
    layer_record, layer_key, _layer_z, _manifest = harness.route7_selected_map_layer_record(harness.output_dir)
    block_report = harness.route7_layer_pose_block_report(layer_record, first, layer_key=layer_key, inflation_cells=1)
    assert_true(block_report.get("blocked") is False, f"Redesigned observation point should be free on the selected layer: {block_report}")


def test_route7_observation_navigation_uses_spatial_astar(tmp_dir: Path) -> None:
    harness = Route7EdgeObservationHarness(tmp_dir)
    attempts = harness.route7_edge_observation_attempts_for_facade("002", "south", output_dir=harness.output_dir)
    target = attempts[0]
    plan = harness.route7_plan_spatial_navigation_path(
        {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0},
        target,
        output_dir=harness.output_dir,
        stage="NAV_TO_OBS",
        target_id=str(target.get("target_id", "")),
        target_house_id="002",
    )
    assert_true(plan.get("status") == "ok", f"Route7 spatial A* should plan to the edge observation point: {plan}")
    assert_true(plan.get("planner_source") == "route7_spatial_occupancy_astar", plan)
    assert_true(plan.get("route7_space_astar") is True, plan)
    assert_true("z_layer" in plan.get("route7_astar_dimensions", []), plan)


def test_route7_ranked_observation_candidates_discard_old_attempts(tmp_dir: Path) -> None:
    harness = Route7EdgeObservationHarness(tmp_dir)
    candidates = harness.route7_edge_observation_candidates_for_house("002", output_dir=harness.output_dir, facades=["south"])
    ranked = harness.route5_rank_observation_candidates(
        "002",
        candidates,
        completed=set(),
        blocked={"east", "north", "west"},
        start_pose={"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0},
    )
    assert_true(ranked, f"Route7 should rank edge observation candidates: {ranked}")
    attempts = ranked[0].get("observation_attempts", [])
    assert_true(attempts, ranked[0])
    assert_true({item.get("observation_attempt_source") for item in attempts} == {"route7_lidar_edge_minimal"}, attempts)
    planners = {item.get("route5_navigation_plan", {}).get("planner_source") for item in attempts if isinstance(item, dict)}
    assert_true(planners == {"route7_spatial_occupancy_astar"}, f"Route7 observation navigation should use spatial A*: {planners}")


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


def test_route7_realtime_multilayer_route_uses_alternate_layer(tmp_dir: Path) -> None:
    harness = Route7MultiLayerRouteHarness(tmp_dir)
    start = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 450.0, "y": -450.0, "z": 300.0, "yaw": 90.0}
    plan = harness.route7_update_realtime_navigation_route(
        start,
        target,
        output_dir=harness.output_dir,
        stage="NAV_TO_SCAN_POINT",
        target_id="002_south_z_300_edge_001",
        target_house_id="002",
        update_reason="verify_multilayer_route",
    )
    assert_true(plan.get("status") == "ok", f"V7 realtime route should find an alternate map layer: {plan}")
    assert_true(plan.get("planner_source") == "route7_realtime_multilayer_occupancy_astar", plan)
    assert_true(plan.get("layer_key") == "z_350", f"blocked z_300 should route through the clear z_350 layer: {plan}")
    assert_true(bool(plan.get("route7_multilayer_route", False)), plan)
    assert_true(int(plan.get("route7_layer_transition_count", 0) or 0) >= 2, plan)
    assert_true(any(str(item.get("route_point_type", "")) == "route7_layer_transition" for item in plan.get("waypoints", [])), plan)
    segments = [item for item in plan.get("route7_route_segments", []) if isinstance(item, dict)]
    assert_true(any(str(item.get("layer_key", "")) == "z_350" for item in segments), f"route segments should carry layer colors/keys: {segments}")
    state_plan = harness.llm_route5_state.get("route7_realtime_route_plan", {})
    assert_true(state_plan.get("target_id") == "002_south_z_300_edge_001", state_plan)
    artifact = harness.output_dir / "map" / "route7_realtime_route_plan.json"
    assert_true(artifact.is_file(), f"realtime route plan JSON should be written: {artifact}")
    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert_true(written.get("route7_route_segments"), written)


def test_route7_local_3d_soft_block_continues_when_map_route_clear(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 150.0, "y": -150.0, "z": 300.0, "yaw": 90.0}
    local_3d = {"safe": False, "reason": "local_3d_occupancy_blocked:forward", "blocked_directions": ["forward"]}
    gate = {
        "front_risk_state": "obstacle_warning",
        "front_min_depth_cm": 158.75,
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
    assert_true(decision.get("action") == "continue_cautious", f"soft local 3D block should continue the clear V7 map route, not skip the facade: {decision}")
    assert_true(decision.get("route7_map_route_replan_policy") == "map_first_runtime_revalidate", decision)
    assert_true(float(decision.get("front_min_depth_cm", 0.0) or 0.0) < 180.0, decision)


def test_route7_target_house_boundary_waypoint_allowed_when_map_route_clear(tmp_dir: Path) -> None:
    harness = Route7TargetHouseBoundaryHarness(tmp_dir)
    harness.llm_route5_state["route_window_label"] = "V7"
    boundary_pose = {"x": 2.0, "y": -400.0, "z": 300.0, "yaw": 0.0}
    center_pose = {"x": 500.0, "y": -400.0, "z": 300.0, "yaw": 0.0}

    boundary_safety = harness.route7_navigation_safety_report(
        "002",
        boundary_pose,
        stage="NAV_TO_OBS",
        facade="west",
        target_id="002_west_obs_attempt_1",
        output_dir=harness.output_dir,
    )
    assert_true(boundary_safety.get("safe") is True, f"V7 facade boundary waypoint should be allowed through target bbox guard: {boundary_safety}")
    assert_true(boundary_safety.get("reason") == "route7_target_house_boundary_waypoint_allowed", boundary_safety)

    center_safety = harness.route7_navigation_safety_report(
        "002",
        center_pose,
        stage="NAV_TO_OBS",
        facade="west",
        target_id="002_west_obs_attempt_1",
        output_dir=harness.output_dir,
    )
    assert_true(center_safety.get("safe") is False, f"V7 should not allow arbitrary flight through target house interior: {center_safety}")
    assert_true(center_safety.get("reason") == "target_house_bbox", center_safety)


def test_route7_observation_replan_failure_is_retryable_and_task_locked(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    harness.llm_route5_state["route_window_label"] = "V7"
    harness.llm_route5_completed_facades = {"south"}
    harness.llm_route5_blocked_facades = {"west", "east", "north"}

    retry = harness.route5_observation_failure_retry_status(
        {"reason": "route7_map_route_replan_required", "pose_error": {"dist_xy_cm": 720.0}}
    )
    assert_true(retry.get("terminal") is False, f"V7 map-route replan should not terminal-block a facade after one observation cycle: {retry}")

    status = harness.route5_final_status_for_task_lock("done")
    assert_true(status == "blocked_selected_house_incomplete", f"V7 should keep the selected-house task locked when facades are blocked: {status}")


def test_route7_soft_warning_does_not_select_backoff(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 90.0}
    target = {"x": 150.0, "y": -50.0, "z": 300.0, "yaw": 90.0}
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
    assert_true(alternative.get("direction") == "route7_yaw_to_nav_point_q", f"V7 should stop only forward and turn q toward the navigation point: {alternative}")
    assert_true(float(alternative.get("payload", {}).get("forward_cm", 0.0) or 0.0) == 0.0, alternative)
    assert_true(float(alternative.get("payload", {}).get("yaw_delta_deg", 0.0) or 0.0) < 0.0, alternative)
    assert_true(alternative.get("safe_alternative_action") != "backoff", alternative)
    assert_true((alternative.get("route7_soft_obstacle_policy") or {}).get("mode") == "continue_planned_route", alternative)


def test_route7_yaw_aligned_forward_block_requests_navigation_point_reset(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 150.0, "y": -50.0, "z": 300.0, "yaw": 90.0}
    gate = {
        "front_risk_state": "obstacle_warning",
        "front_min_depth_cm": 160.0,
        "can_forward": True,
        "must_stop": False,
        "selected_direction": "forward",
    }
    alternative = harness.route5_apply_or2_safe_alternative(
        event={
            "route5_stage": "NAV_TO_SCAN_POINT",
            "target_id": "002_south_z_300_edge_001",
            "target_house_id": "002",
            "route5_output_dir": str(harness.output_dir),
            "pointcloud_summary": {"available": True, "front_min_depth_cm": 160.0, "forward_swept_clear": False},
        },
        gate=gate,
        rule={"selected_direction": "forward", "candidate_action_scores": {"forward": 1.0}},
        selected_direction="forward",
        selected_payload={"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_or2_forward"},
        nominal_payload={"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav_lookahead"},
        config={"nav_step_cm": 20.0, "yaw_tol_deg": 10.0},
        current_pose=current,
        target_pose=target,
    )
    assert_true(bool(alternative.get("route7_navigation_point_reset_request", False)), f"yaw-aligned front block should reset the navigation point: {alternative}")
    tracker = harness.route5_new_target_reset_tracker("002_south_z_300_edge_001", target, target)
    tracker = harness.route5_record_target_reset_tick(
        tracker,
        {"route7_navigation_point_reset_request": True, "route7_soft_obstacle_policy": {"mode": "continue_planned_route"}},
        {"route7_navigation_point_reset_request": True, "route7_soft_obstacle_policy": {"mode": "continue_planned_route"}},
        {"reached": False, "dist_xy_cm": 100.0},
        current_pose=current,
        target_pose=target,
    )
    decision = harness.route5_should_reset_target(tracker, {"reached": False, "dist_xy_cm": 100.0})
    assert_true(decision.get("should_reset") is True and decision.get("reason") == "route7_navigation_point_reset_request", decision)


def test_route7_front_square_deep_red_is_only_hard_avoidance(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 150.0, "y": -150.0, "z": 300.0, "yaw": 90.0}
    event = {
        "route5_stage": "NAV_TO_SCAN_POINT",
        "target_id": "002_south_z_300_edge_001",
        "route5_output_dir": str(harness.output_dir),
        "pointcloud_summary": {"front_min_depth_cm": 90.0},
    }
    soft_gate = {
        "front_risk_state": "obstacle_warning",
        "front_min_depth_cm": 90.0,
        "can_forward": True,
        "must_stop": False,
    }
    soft_rule = {"corridor_risks": {"front_center": {"stop_fraction": 0.0, "warning_fraction": 0.2, "clearance_fraction": 0.0}}}
    soft_policy = harness.route7_or2_soft_obstacle_policy(event, soft_gate, current, target, output_dir=harness.output_dir, rule=soft_rule)
    assert_true(soft_policy.get("mode") == "continue_planned_route", f"front depth alone should not trigger V7 hard avoidance without front-square deep red: {soft_policy}")

    deep_rule = {"corridor_risks": {"front_center": {"stop_fraction": 0.2, "warning_fraction": 0.0, "clearance_fraction": 0.0}}}
    deep_policy = harness.route7_or2_soft_obstacle_policy(event, soft_gate, current, target, output_dir=harness.output_dir, rule=deep_rule)
    assert_true(deep_policy.get("mode") == "hard_avoidance", f"front-square deep red should trigger OR takeover: {deep_policy}")
    takeover = deep_policy.get("route7_front_square_deep_red_takeover", {})
    assert_true(float(takeover.get("front_stop_fraction", 0.0) or 0.0) > 0.01, takeover)


def test_route7_or3_projection_box_deep_red_is_only_hard_avoidance(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 150.0, "y": -150.0, "z": 300.0, "yaw": 90.0}
    event = {
        "route5_stage": "NAV_TO_SCAN_POINT",
        "target_id": "002_south_z_300_edge_001",
        "route5_output_dir": str(harness.output_dir),
        "pointcloud_summary": {"front_min_depth_cm": 90.0},
    }
    soft_gate = {
        "front_risk_state": "must_stop",
        "must_stop": True,
        "front_box_stop_fraction": 0.0,
        "projection_box_stop_threshold": 0.01,
    }
    soft_policy = harness.route7_or2_soft_obstacle_policy(event, soft_gate, current, target, output_dir=harness.output_dir)
    assert_true(
        soft_policy.get("mode") == "continue_planned_route",
        f"OR3 must_stop without projection-box deep red should continue planned route: {soft_policy}",
    )

    hard_gate = dict(soft_gate)
    hard_gate["front_box_stop_fraction"] = 0.02
    hard_policy = harness.route7_or2_soft_obstacle_policy(event, hard_gate, current, target, output_dir=harness.output_dir)
    assert_true(hard_policy.get("mode") == "hard_avoidance", f"OR3 projection-box deep red should trigger hard avoidance: {hard_policy}")
    takeover = hard_policy.get("route7_front_square_deep_red_takeover", {})
    assert_true(float(takeover.get("front_stop_fraction", 0.0) or 0.0) == 0.02, takeover)


def test_route7_or3_hard_gate_marks_route_decision_active(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 150.0, "y": -150.0, "z": 300.0, "yaw": 90.0}
    nominal = {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "route5_nav"}
    event = {
        "route_window_label": "V7",
        "route5_stage": "NAV_TO_SCAN_POINT",
        "target_id": "002_south_z_300_edge_001",
        "route5_output_dir": str(harness.output_dir),
        "pointcloud_summary": {"front_min_depth_cm": 500.0, "forward_swept_clear": True},
        "relative_target": {"distance_cm": 250.0, "bearing_deg_body": 0.0},
        "or2_prediction": {"status": "ok", "front_risk_state": "clear", "can_forward": True, "must_stop": False},
        "or3_prediction": {
            "status": "ok",
            "front_risk_state": "must_stop",
            "can_forward": False,
            "must_stop": True,
            "front_box_stop_fraction": 0.02,
            "front_box_warning_fraction": 0.0,
            "front_box_clearance_fraction": 0.0,
            "projection_box_stop_threshold": 0.01,
            "projection_box": {"x0": 0.42, "x1": 0.58, "y0": 0.38, "y1": 0.72, "stop_fraction_threshold": 0.01},
        },
    }
    decision = harness.route5_or2_decision_for_event(
        event,
        nominal_payload=nominal,
        config={"nav_step_cm": 20.0, "reach_tol_cm": 60.0, "yaw_tol_deg": 10.0},
        current_pose=current,
        start_pose=current,
        target_pose=target,
        last_action={"action_name": "hold"},
    )
    gate = decision.get("gate", {})
    updates = decision.get("event_updates", {})
    policy = gate.get("route7_soft_obstacle_policy", {}) if isinstance(gate.get("route7_soft_obstacle_policy", {}), dict) else {}
    assert_true(policy.get("mode") == "hard_avoidance", f"OR3 projection-box deep red should reach hard policy: {decision}")
    assert_true(gate.get("source") == "route7_or3_1_a_plus_3_1", f"Route7 should use OR3_1 gate source: {decision}")
    assert_true(bool(updates.get("or3_1_primary", False)), f"Route7 decision updates should mark OR3_1 primary: {decision}")
    assert_true(bool(gate.get("avoidance_active", False)), f"hard policy should mark gate active: {decision}")
    assert_true(bool(updates.get("avoidance_active", False)), f"hard policy should mark event update active: {decision}")
    assert_true(decision.get("mission_phase") == "OR3_1_AVOIDANCE", f"hard policy should leave route navigation phase through OR3_1: {decision}")


def test_route7_deep_red_policy_applies_to_observation_navigation(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 238.513, "y": 473.645, "z": 333.298, "yaw": 47.821}
    target = {"x": 350.22, "y": 570.65, "z": 300.0, "yaw": 90.0}
    event = {
        "route5_stage": "NAV_TO_OBS",
        "target_id": "002_south_obs_attempt_1",
        "route5_output_dir": str(harness.output_dir),
        "pointcloud_summary": {"front_min_depth_cm": 315.25, "forward_swept_clear": False},
    }
    gate = {
        "front_risk_state": "clearance_warning",
        "front_min_depth_cm": 315.25,
        "can_forward": True,
        "must_stop": False,
    }
    rule = {"corridor_risks": {"front_center": {"stop_fraction": 0.0, "warning_fraction": 0.0, "clearance_fraction": 0.0}}}
    assert_true(
        harness.route7_should_use_map_route_planner("NAV_TO_OBS", "002_south_obs_attempt_1", output_dir=harness.output_dir),
        "V7 deep-red-only route priority must also apply while flying to observation points",
    )
    policy = harness.route7_or2_soft_obstacle_policy(event, gate, current, target, output_dir=harness.output_dir, rule=rule)
    assert_true(policy.get("mode") == "continue_planned_route", f"NAV_TO_OBS should not backoff without front-square deep red: {policy}")


def test_route5_lookahead_keeps_translation_near_observation_goal(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    current = {"x": 297.33, "y": 538.559, "z": 333.298, "yaw": 46.408}
    target = {"x": 350.22, "y": 570.65, "z": 300.0, "yaw": 90.0}
    payload = harness.route5_movement_payload_for_target_with_lookahead(
        current,
        target,
        {"nav_step_cm": 20.0, "reach_tol_cm": 60.0, "z_tol_cm": 40.0, "yaw_tol_deg": 10.0},
        stage="NAV_TO_OBS",
    )
    translation = abs(float(payload.get("forward_cm", 0.0) or 0.0)) + abs(float(payload.get("right_cm", 0.0) or 0.0)) + abs(float(payload.get("up_cm", 0.0) or 0.0))
    assert_true(translation > 0.5, f"close-range observation approach should not get stuck in yaw-only lookahead: {payload}")
    assert_true(str(payload.get("action_name", "")) != "route5_nav_lookahead" or float(payload.get("forward_cm", 0.0) or 0.0) > 0.0, payload)


def test_route7_map_route_lookahead_translates_while_yawing(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    harness.llm_route5_state["route_window_label"] = "V7"
    current = {"x": 0.0, "y": 0.0, "z": 300.0, "yaw": 8.632}
    target = {
        "x": 350.22,
        "y": 570.65,
        "z": 300.0,
        "yaw": 90.0,
        "route_point_type": "navigation_waypoint",
        "route7_map_layer_key": "z_300",
        "route7_map_route_replan_policy": "map_first_runtime_revalidate",
    }
    payload = harness.route5_movement_payload_for_target_with_lookahead(
        current,
        target,
        {"nav_step_cm": 20.0, "reach_tol_cm": 60.0, "z_tol_cm": 40.0, "yaw_tol_deg": 10.0},
        stage="NAV_TO_OBS",
    )
    translation = abs(float(payload.get("forward_cm", 0.0) or 0.0)) + abs(float(payload.get("right_cm", 0.0) or 0.0))
    assert_true(translation > 0.5, f"Route7 map route should not spend map-clear ticks on pure rotation: {payload}")
    assert_true(abs(float(payload.get("yaw_delta_deg", 0.0) or 0.0)) > 0.5, f"Route7 should still yaw toward the waypoint while translating: {payload}")
    assert_true(str(payload.get("yaw_policy", "")) == "route7_translate_while_yawing_map_route", payload)


def test_route7_continue_planned_route_does_not_trigger_target_reset(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    original_target = {"x": 1478.36, "y": 1021.05, "z": 300.0, "yaw": 90.0}
    current = {"x": 474.906, "y": 655.587, "z": 304.884, "yaw": 29.831}
    tracker = harness.route5_new_target_reset_tracker("002_south_z_300_edge_002", original_target, original_target)
    gate = {
        "avoidance_active": True,
        "front_risk_state": "obstacle_warning",
        "selected_direction": "slow_forward",
        "front_min_depth_cm": 261.5,
        "route7_soft_obstacle_policy": {"mode": "continue_planned_route"},
    }
    error = {"reached": False, "dist_xy_cm": 1059.78, "dist_3d_cm": 1059.791, "dz": 4.884}
    for frame_id in range(83, 91):
        event = {
            "frame_id": frame_id,
            "route5_stage": "NAV_TO_SCAN_POINT",
            "target_id": "002_south_z_300_edge_002",
            "route5_output_dir": str(harness.output_dir),
            "distance_to_goal_cm": 1059.78,
            "route_cross_track_cm": 42.718,
            "or2_front_risk_state": "obstacle_warning",
            "or2_selected_direction": "slow_forward",
            "route7_soft_obstacle_policy": {"mode": "continue_planned_route"},
            "or2_rule": {
                "selected_direction": "forward",
                "candidate_action_scores": {"forward": 1.11, "slow_forward": 1.03, "right": 0.99},
                "corridor_risks": {"front_center": {"stop_fraction": 0.0, "warning_fraction": 0.01}},
            },
        }
        tracker = harness.route5_record_target_reset_tick(tracker, event, gate, error, current_pose=current, target_pose=original_target)
    decision = harness.route5_should_reset_target(tracker, error)
    assert_true(not decision.get("should_reset"), f"V7 continue-planned-route frames must not reset a far target: tracker={tracker} decision={decision}")
    assert_true(int(tracker.get("consecutive_avoidance_ticks", 0) or 0) == 0, tracker)


def test_route7_continue_planned_route_does_not_near_obstacle_arrive(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    state = harness.route5_near_obstacle_arrival_state(
        {"reached": False, "dist_xy_cm": 121.491, "dist_3d_cm": 121.589},
        {
            "avoidance_active": True,
            "front_risk_state": "obstacle_warning",
            "front_min_depth_cm": 261.5,
            "must_stop": False,
            "route7_soft_obstacle_policy": {"mode": "continue_planned_route"},
        },
        {
            "route5_stage": "NAV_TO_SCAN_POINT",
            "target_id": "002_south_z_300_edge_002",
            "distance_to_goal_cm": 121.491,
            "or2_front_risk_state": "obstacle_warning",
            "route7_soft_obstacle_policy": {"mode": "continue_planned_route"},
            "pointcloud_summary": {"front_min_depth_cm": 261.5, "forward_swept_clear": False},
        },
    )
    assert_true(not state.get("near_obstacle_reached"), f"V7 route-continuation policy should not mark soft warning as arrival: {state}")
    assert_true(state.get("arrival_policy") == "continue_navigation", state)


def test_route7_overlay_preserves_original_target_after_reset(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    original_target = {"x": 1478.36, "y": 1021.05, "z": 300.0, "yaw": 90.0}
    reset_target = {"x": 415.213, "y": 759.687, "z": 304.884, "yaw": 90.0}
    harness.llm_route5_state = {
        "route_window_label": "V7",
        "target_house_id": "002",
        "current_exploration_status": {
            "stage": "NAV_TO_SCAN_POINT",
            "facade": "south",
            "target_id": "002_south_z_300_edge_002_reset_1",
            "target_pose": reset_target,
        },
        "last_target_reset": {
            "stage": "NAV_TO_SCAN_POINT",
            "facade": "south",
            "target_id": "002_south_z_300_edge_002",
            "reset_target_id": "002_south_z_300_edge_002_reset_1",
            "original_target_pose": original_target,
            "reset_target_pose": reset_target,
        },
    }
    points = harness.route7_layered_route_points()
    original_points = [point for point in points if point.get("route_point_type") == "original_navigation_target"]
    assert_true(original_points, f"V7 map overlay should keep the original far navigation target visible after a reset: {points}")
    assert_true(original_points[0].get("target_id") == "002_south_z_300_edge_002", original_points)


def test_route7_overlay_uses_current_facade_edge_observation_points_only(tmp_dir: Path) -> None:
    harness = Route7EdgeObservationHarness(tmp_dir)
    harness.llm_route5_state.update(
        {
            "route_window_label": "V7",
            "target_house_id": "002",
            "current_facade": "south",
            "current_exploration_status": {"facade": "south", "stage": "NAV_TO_OBS", "target_id": "002_south_edge_obs_001"},
        }
    )
    points = harness.route7_layered_route_points()
    observations = [point for point in points if point.get("route_point_type") in {"observation_point", "route7_edge_observation_point"}]
    assert_true(observations, f"Route7 map should expose current facade edge observation points: {points}")
    assert_true({point.get("facade") for point in observations} == {"south"}, observations)
    assert_true({point.get("observation_attempt_source") for point in observations} == {"route7_lidar_edge_minimal"}, observations)
    assert_true(not any(point.get("overlay") == "all_observation_points" for point in observations), observations)


def test_route7_task_switch_visualizes_active_subtasks(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    harness.llm_route5_state.update(
        {
            "route_window_label": "V7",
            "target_house_id": "002",
            "current_facade": "south",
            "stage": "NAV_TO_SCAN_POINT",
            "current_exploration_status": {
                "facade": "south",
                "stage": "NAV_TO_SCAN_POINT",
                "target_id": "002_south_z_300_edge_001",
            },
            "task_plan": {
                "subtasks": [
                    {"order": 1, "facade": "south", "status": "pending"},
                    {"order": 2, "facade": "east", "status": "pending"},
                ]
            },
            "current_facade_scan_points": [
                {"scan_id": "002_south_z_300_edge_001", "facade": "south"},
                {"scan_id": "002_south_z_300_edge_002", "facade": "south"},
            ],
        }
    )
    records = harness.route7_subtask_switch_records()
    labels = [record.get("label") for record in records]
    assert_true("observe" in labels and "scan 1" in labels and "scan 2" in labels, records)
    active = [record for record in records if record.get("status") == "active"]
    assert_true(active and active[0].get("label") == "scan 1", records)
    text = harness.route7_task_switch_visualization_text()
    assert_true("Subtasks:" in text and "scan 1:active" in text, text)


def test_route7_map_refresh_keeps_last_preview_on_missing_frame(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    preview = PreviewLabelHarness()
    harness.llm_route7_update_map_preview_label = preview
    harness.refresh_llm_route7_update_map(build_if_missing=False)
    assert_true(preview.configs, "refresh should update the preview label state")
    cleared = [config for config in preview.configs if config.get("image", None) == ""]
    assert_true(not cleared, f"Route7 map refresh should keep the last good preview image on transient missing files: {preview.configs}")


def test_route7_draw_realtime_route_plan_uses_last_drawable_when_current_empty(tmp_dir: Path) -> None:
    harness = Route7MapRouteHarness(tmp_dir)
    layer_record, layer_key, _layer_z, _manifest = harness.route7_selected_map_layer_record(harness.output_dir)
    last_plan = {
        "schema": "route7_realtime_route_plan_v1",
        "status": "ok",
        "route7_route_segments": [
            {
                "kind": "horizontal_route",
                "layer_key": layer_key,
                "from_pose": {"x": 50.0, "y": -50.0, "z": 300.0},
                "to_pose": {"x": 450.0, "y": -450.0, "z": 300.0},
            }
        ],
    }
    harness.llm_route5_state["route7_realtime_route_plan"] = {"status": "blocked", "route7_route_segments": []}
    harness.llm_route5_state["route7_last_drawable_route_plan"] = last_plan
    image = Image.new("RGB", (21 * 8, 21 * 8), "white")
    drawn = harness.route7_draw_realtime_route_plan(image, layer_record, selected_layer_key=layer_key, scale=8)
    pixels = np.asarray(drawn)
    nonwhite = int(np.sum(np.any(pixels != 255, axis=2)))
    assert_true(nonwhite > 0, "Route7 should keep drawing the last valid path when current realtime plan is transiently empty")


def test_route7_spatial_astar_visualization_is_written(tmp_dir: Path) -> None:
    harness = Route7EdgeObservationHarness(tmp_dir)
    start = {"x": 50.0, "y": -50.0, "z": 300.0, "yaw": 0.0}
    target = {"x": 450.0, "y": -450.0, "z": 300.0, "yaw": 90.0}
    plan = harness.route7_plan_spatial_navigation_path(
        start,
        target,
        output_dir=harness.output_dir,
        stage="NAV_TO_OBS",
        target_id="002_south_edge_obs_001",
        target_house_id="002",
    )
    visual = harness.route7_write_navigation_plan_visualization(
        harness.output_dir,
        plan,
        current_pose=start,
        target_pose=target,
        target_id="002_south_edge_obs_001",
    )
    assert_true(Path(str(visual.get("visualization_path", ""))).is_file(), f"Spatial A* should write trajectory preview: {visual}")


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
    assert_true("yaw_error_deg" in logged, f"V7 per-frame log should include yaw diagnostics: {logged}")
    assert_true("forward_blocked" in logged, f"V7 per-frame log should include whether forward was blocked: {logged}")


def test_window7_source_contract() -> None:
    panel_source = (PROJECT_ROOT / "control" / "panel.py").read_text(encoding="utf-8")
    route5_source = (PROJECT_ROOT / "control" / "route5_fusion_control.py").read_text(encoding="utf-8")
    or3_demo_source = (PROJECT_ROOT / "obstacle_representation_3" / "demo.py").read_text(encoding="utf-8")

    assert_true("Open LLM Route Window 7" in panel_source, "main panel should expose Open LLM Route Window 7")
    assert_true("command=self.open_llm_route_window7" in panel_source, "Window 7 button should call open_llm_route_window7")
    assert_true("ObstacleRepresentation3Predictor" in or3_demo_source, "OR3 demo should expose ObstacleRepresentation3Predictor")
    assert_true("front_box_stop_fraction" in or3_demo_source, "OR3 demo should expose projection-box stop fraction")
    assert_true(
        "obstacle_representation_3_data" in route5_source or "ObstacleRepresentation3Predictor" in route5_source,
        "Route 5 mixin should expose OR3 integration source tokens",
    )
    assert_true("a_plus_3_1_model.pt" in route5_source, "Route7 should default to the OR3_1 checkpoint")
    assert_true("route7_llm_route_oa3_or3_1_fusion" in route5_source, "Route7 mode should identify OR3_1 fusion")
    assert_true("route7_primary_representation" in route5_source, "Route7 predictions should mark OR3_1 as the primary representation")

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
        "def route7_update_realtime_navigation_route",
        "def route7_lidar_edge_observation_formula",
        "def route7_edge_observation_attempts_for_facade",
        "def route7_edge_observation_candidates_for_house",
        "def route7_plan_spatial_navigation_path",
        "def route7_observation_formula_text",
        "def open_route7_observation_formula_window",
        "def route7_draw_realtime_route_plan",
        "def route7_static_house_base_snapshot",
        "def route7_draw_static_house_base",
        "def route7_front_square_deep_red_takeover",
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
    prepare_block = route5_source[route5_source.find("def route7_prepare_new_map_output_dir"):route5_source.find("def route7_current_map_output_dir")]
    assert_true("route7_static_house_base" in prepare_block, "Window 7 run preparation should freeze house coordinates as the map base")

    worker_block = route5_source[route5_source.find("def route5_full_search_worker"):route5_source.find("def refresh_route5_preview")]
    assert_true("route7_edge_observation_candidates_for_house" in worker_block, "Window 7 should use direct lidar edge observation candidates")
    assert_true("route7_build_update_map_after_observation" in worker_block, "Window 7 should build/update the obstacle map after observation")
    assert_true("route7_plan_facade_map_layer_scan_current" in worker_block, "Window 7 should plan scan captures from the selected map layer")
    high_level_log_index = worker_block.find('self.route5_log_event(output_dir, "high_level_decision", decision)')
    select_facade_index = worker_block.find('facade = str(decision.get("target_facade"', high_level_log_index)
    assert_true(high_level_log_index >= 0 and select_facade_index > high_level_log_index, "worker source should expose the high-level decision-to-select-facade span")
    assert_true(
        "self.llm_route5_stop_event.is_set()" in worker_block[high_level_log_index:select_facade_index],
        "V7 worker should honor stop requests that arrive during next-facade decision before marking a facade in progress",
    )
    attempts_index = worker_block.find("for attempt_index, attempt in enumerate(observation_attempts, start=1):")
    obs_failure_index = worker_block.find('if nav_result.get("status") != "ok":', attempts_index)
    assert_true(attempts_index >= 0 and obs_failure_index > attempts_index, "worker source should expose the observation attempt failure span")
    assert_true(
        "self.llm_route5_stop_event.is_set()" in worker_block[attempts_index:obs_failure_index],
        "V7 worker should stop cleanly before treating an interrupted observation as terminal_blocked",
    )

    navigate_block = route5_source[route5_source.find("def route5_navigate_to_pose_with_fusion"):route5_source.find("def route5_run_summary")]
    assert_true("route7_should_use_map_route_planner" in navigate_block, "V7 navigation should choose the map-route planner")
    assert_true("route7_plan_navigation_waypoints_from_map" in navigate_block, "V7 navigation should plan from Route 6 layered occupancy")
    assert_true("route7_update_realtime_navigation_route" in navigate_block, "V7 navigation should keep a realtime route plan updated")
    assert_true("route7_route_segments" in route5_source, "V7 realtime route should expose layer-colored route segments")
    assert_true("route7_front_square_deep_red_takeover" in route5_source, "V7 OR should only take over on front-square deep red")
    follow_block = route5_source[route5_source.find("def route5_follow_navigation_waypoint_with_fusion"):route5_source.find("def route5_navigate_to_pose_with_fusion")]
    assert_true("route7_local_3d_replan_decision" in follow_block, "V7 local 3D blocks should be revalidated against map-route distance")
    assert_true("route7_map_route_replan_required" in follow_block, "V7 should replan instead of hard-stopping on soft local 3D blocks")
    assert_true("route7_frame_decision_log.jsonl" in route5_source, "V7 should write a concise per-frame decision log")
    assert_true("route7_write_navigation_plan_visualization" in route5_source, "V7 should write planned trajectory visualization artifacts")
    assert_true("route7_yaw_to_nav_point" in route5_source, "V7 forward obstacle handling should yaw q/e toward the navigation point instead of slow-forwarding")
    route_control_source = (PROJECT_ROOT / "control" / "route_control.py").read_text(encoding="utf-8")
    flight_source = (PROJECT_ROOT / "run_drone_flight.py").read_text(encoding="utf-8")
    assert_true("frames_subdir" in flight_source, "LiDAR stream capture should accept a separate frame stream directory")
    assert_true("open3d_frames" in route_control_source, "Route7 Open3D captures should be routed to open3d_frames")

    ui_block = route5_source[route5_source.find("def _build_llm_route5_section"):route5_source.find("def route7_update_map_layer_values")]
    assert_true("route5_fixed_status_label" in ui_block, "Route status labels should use fixed layout containers")
    assert_true("grid_propagate(False)" in ui_block, "Route status/preview frames should not resize as text changes")
    assert_true("route7_task_switch_visualization_text" in route5_source, "Window 7 should expose a task-switch visualization below the map")
    assert_true("llm_route7_task_switch_text" in route5_source, "Window 7 should maintain a task-switch visualization widget")
    assert_true("route7_subtask_switch_records" in route5_source, "Window 7 should expose active-facade subtask visualization records")
    assert_true("route7_last_drawable_route_plan" in route5_source, "Window 7 should preserve the last drawable path across transient empty route plans")

    stop_block = route5_source[route5_source.find("def on_route7_stop"):route5_source.find("def open_llm_route_window5")]
    assert_true("route6_update_map_realtime_stop_event.set()" in stop_block, "Window 7 stop should stop realtime map updates")
    assert_true("route6_update_map_capture_stop_event.set()" in stop_block, "Window 7 stop should stop update-map capture")
    assert_true("on_route7_stop" in window7_block, "Window 7 stop button should call V7 stop-all handler")


def main() -> None:
    import tempfile

    test_route2_observation_z_override_for_route7()
    with tempfile.TemporaryDirectory(prefix="route7_verify_") as raw:
        test_route7_run_directories_are_dedicated_and_fresh(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_or31_primary_verify_") as raw:
        test_route7_uses_or3_1_as_primary_representation(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_static_house_base_verify_") as raw:
        test_route7_static_house_base_is_frozen_at_run_start(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_static_house_legacy_filter_verify_") as raw:
        test_route7_static_house_base_filters_legacy_full_map_snapshots(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_open3d_frames_verify_") as raw:
        test_route7_open3d_frame_stream_indexes_are_separate(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_edge_observation_formula_verify_") as raw:
        test_route7_edge_observation_minimal_points_use_lidar_formula(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_edge_observation_blocked_verify_") as raw:
        test_route7_edge_observation_replans_points_inside_obstacles(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_spatial_astar_verify_") as raw:
        test_route7_observation_navigation_uses_spatial_astar(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_edge_rank_verify_") as raw:
        test_route7_ranked_observation_candidates_discard_old_attempts(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_scan_verify_") as raw:
        test_route7_map_layer_edge_scan_uses_two_house_edge_points(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_map_route_verify_") as raw:
        test_route7_map_route_planner_uses_layered_occupancy(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_multilayer_route_verify_") as raw:
        test_route7_realtime_multilayer_route_uses_alternate_layer(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_soft_block_verify_") as raw:
        test_route7_local_3d_soft_block_continues_when_map_route_clear(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_target_boundary_verify_") as raw:
        test_route7_target_house_boundary_waypoint_allowed_when_map_route_clear(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_task_lock_verify_") as raw:
        test_route7_observation_replan_failure_is_retryable_and_task_locked(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_soft_warning_verify_") as raw:
        test_route7_soft_warning_does_not_select_backoff(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_nav_reset_verify_") as raw:
        test_route7_yaw_aligned_forward_block_requests_navigation_point_reset(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_deep_red_verify_") as raw:
        test_route7_front_square_deep_red_is_only_hard_avoidance(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_or3_deep_red_verify_") as raw:
        test_route7_or3_projection_box_deep_red_is_only_hard_avoidance(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_or3_decision_verify_") as raw:
        test_route7_or3_hard_gate_marks_route_decision_active(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_nav_to_obs_verify_") as raw:
        test_route7_deep_red_policy_applies_to_observation_navigation(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_lookahead_verify_") as raw:
        test_route5_lookahead_keeps_translation_near_observation_goal(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_translate_yaw_verify_") as raw:
        test_route7_map_route_lookahead_translates_while_yawing(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_target_reset_verify_") as raw:
        test_route7_continue_planned_route_does_not_trigger_target_reset(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_arrival_policy_verify_") as raw:
        test_route7_continue_planned_route_does_not_near_obstacle_arrive(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_overlay_reset_verify_") as raw:
        test_route7_overlay_preserves_original_target_after_reset(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_current_facade_overlay_verify_") as raw:
        test_route7_overlay_uses_current_facade_edge_observation_points_only(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_subtask_switch_verify_") as raw:
        test_route7_task_switch_visualizes_active_subtasks(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_map_flicker_verify_") as raw:
        test_route7_map_refresh_keeps_last_preview_on_missing_frame(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_route_line_fallback_verify_") as raw:
        test_route7_draw_realtime_route_plan_uses_last_drawable_when_current_empty(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_spatial_visual_verify_") as raw:
        test_route7_spatial_astar_visualization_is_written(Path(raw))
    with tempfile.TemporaryDirectory(prefix="route7_frame_log_verify_") as raw:
        test_route7_frame_decision_log_and_trajectory_visualization(Path(raw))
    test_window7_source_contract()
    print("PASS route7 window verification")


if __name__ == "__main__":
    main()
