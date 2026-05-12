from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.route_control import RouteControlMixin


class _Var:
    def __init__(self, value: object) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value

    def set(self, value: object) -> None:
        self.value = str(value)


class _Route2Harness(RouteControlMixin):
    def __init__(self) -> None:
        self.args = SimpleNamespace(lidar_depth_min_cm=20.0, lidar_depth_max_cm=1200.0)
        self.map_config = {
            "houses": [
                {"id": "001", "center_x": 0.0, "center_y": 0.0, "radius_cm": 1000.0, "approach_z": 650.0},
                {"id": "002", "center_x": 2500.0, "center_y": 0.0, "radius_cm": 1000.0, "approach_z": 650.0},
                {"id": "003", "center_x": 0.0, "center_y": 1800.0, "radius_cm": 900.0, "approach_z": 650.0},
            ]
        }
        self.map_calibration = {}
        self.latest_state = {"pose": {"x": -1500.0, "y": 0.0, "z": 600.0, "task_yaw": 0.0}}
        self.llm_route_standoff_cm_var = _Var(850.0)
        self.llm_route_scan_spacing_cm_var = _Var(150.0)
        self.llm_route_capture_count_var = _Var(1)
        self.route_capture_interval_s_var = _Var(0.5)
        self.llm_task_text_var = _Var("route2_verify")
        self.llm_route2_floor_height_m_var = _Var(3.0)
        self.llm_route2_default_floors_var = _Var(2)
        self.llm_route2_low_z_cm_var = _Var(250.0)
        self.llm_route2_z_step_cm_var = _Var(350.0)
        self.llm_route2_density_mode_var = _Var("auto")
        self.llm_route2_selected_facade_var = _Var("auto")
        self.llm_route2_state = {}
        self.llm_route2_completed_facades = set()
        self.custom_bboxes = {}

    def load_map_resources(self, force: bool = False) -> bool:
        return True

    def _as_float_or_none(self, value):
        try:
            return float(value)
        except Exception:
            return None

    def _normalize_angle_deg(self, angle_deg: float) -> float:
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    def selected_route_target_house_id(self) -> str:
        return "001"

    def lidar_capture_processing_mode(self) -> str:
        return "full"

    def house_records_for_route_planning(self):
        if self.custom_bboxes:
            records = []
            for house_id, bbox in self.custom_bboxes.items():
                center_x = 0.5 * (float(bbox["min_x"]) + float(bbox["max_x"]))
                center_y = 0.5 * (float(bbox["min_y"]) + float(bbox["max_y"]))
                radius = max(abs(float(bbox["max_x"]) - float(bbox["min_x"])), abs(float(bbox["max_y"]) - float(bbox["min_y"])))
                records.append({"id": house_id, "x": center_x, "y": center_y, "radius_cm": radius})
            return records
        return super().house_records_for_route_planning()

    def house_world_bbox_for_id(self, house_id: str):
        bbox = self.custom_bboxes.get(str(house_id or "").strip())
        if bbox:
            return dict(bbox)
        return super().house_world_bbox_for_id(house_id)


def main() -> None:
    harness = _Route2Harness()

    candidates = harness.route2_safe_observation_candidates("001")
    assert candidates, "expected safe observation candidates"
    observation = candidates[0]
    assert observation["facade"] == "west", f"expected nearest west facade, got {observation['facade']}"
    all_facade_candidates = harness.route2_all_facade_observation_candidates("001", skip_completed=False)
    assert len(all_facade_candidates) == 4, f"expected four facade candidates, got {len(all_facade_candidates)}"
    assert {candidate["facade"] for candidate in all_facade_candidates} == {"south", "east", "north", "west"}
    standoff, meta = harness.route2_observation_standoff_cm(
        facade_length_cm=1800.0,
        facade_depth_cm=3700.0,
        facade_info={"standoff_cm": 850.0, "clearance_cm": 180.0},
    )
    assert standoff < 1000.0, f"short facade should use closer panorama standoff, got {standoff}"
    assert meta["observation_panorama_span_cm"] == 1800.0
    east_candidates = harness.route2_safe_observation_candidates("001", skip_completed=False, facade_filter="east")
    assert east_candidates, "expected manually selected east facade candidate"
    assert {candidate["facade"] for candidate in east_candidates} == {"east"}
    auto_candidates = harness.route2_safe_observation_candidates("001", skip_completed=False, facade_filter="auto")
    assert len(auto_candidates) == len(harness.route2_safe_observation_candidates("001", skip_completed=False))
    for centered_facade in ("south", "north"):
        centered = harness.route2_safe_observation_candidates("001", skip_completed=False, facade_filter=centered_facade)[0]
        assert abs(float(centered["axis_value"]) - float(centered["axis_center_cm"])) < 1e-6, centered

    harness.custom_bboxes = {
        "001": {"min_x": 200.0, "max_x": 1500.0, "min_y": 0.0, "max_y": 1000.0, "center_x": 850.0, "center_y": 500.0},
        "002": {"min_x": 1600.0, "max_x": 2600.0, "min_y": 0.0, "max_y": 1000.0, "center_x": 2100.0, "center_y": 500.0},
    }
    blocked_candidates = harness.route2_all_facade_observation_candidates("002", skip_completed=False)
    west_blocked = next(candidate for candidate in blocked_candidates if candidate["facade"] == "west")
    assert west_blocked["status"] == "blocked", west_blocked
    assert west_blocked["observation_blocking_house_id"] == "001", west_blocked
    harness.custom_bboxes = {
        "003": {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 1000.0, "center_x": 500.0, "center_y": 500.0},
        "002": {"min_x": 1320.0, "max_x": 2320.0, "min_y": 0.0, "max_y": 1000.0, "center_x": 1820.0, "center_y": 500.0},
    }
    alley_candidates = harness.route2_all_facade_observation_candidates("003", skip_completed=False)
    east_alley = next(candidate for candidate in alley_candidates if candidate["facade"] == "east")
    assert east_alley["status"] == "planned", east_alley
    assert east_alley["observation_standoff_mode"] == "facade_center_projection_boundary_adjusted", east_alley
    assert 1000.0 < float(east_alley["x"]) < 1140.0, east_alley
    assert not east_alley.get("observation_blocking_house_id"), east_alley
    assert east_alley["safe_interval_source"] == "facade_center_projection", east_alley
    harness.custom_bboxes = {
        "004": {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 1000.0, "center_x": 500.0, "center_y": 500.0},
        "002": {"min_x": 1320.0, "max_x": 2320.0, "min_y": 2000.0, "max_y": 3000.0, "center_x": 1820.0, "center_y": 2500.0},
    }
    projected_north = next(
        candidate for candidate in harness.route2_all_facade_observation_candidates("004", skip_completed=False)
        if candidate["facade"] == "north"
    )
    assert projected_north["observation_standoff_mode"] == "facade_center_projection", projected_north
    assert not projected_north.get("observation_boundary_adjustment"), projected_north
    adjusted = harness.route2_adjust_observation_to_blocking_boundary(
        {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 1000.0},
        "east",
        500.0,
        300.0,
        850.0,
        {"house_id": "002", "min_x": 1140.0, "max_x": 2400.0, "min_y": -200.0, "max_y": 1200.0, "clearance_cm": 180.0},
    )
    assert adjusted and adjusted["pose"]["x"] == 1138.0, adjusted
    assert adjusted["adjustment"]["blocking_house_id"] == "002", adjusted
    harness.custom_bboxes = {}
    harness.map_calibration = {
        "affine_world_to_image": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "image_width": 100,
        "image_height": 100,
    }
    assert harness.route2_observation_map_bounds_report(50.0, 50.0)["in_bounds"]
    assert not harness.route2_observation_map_bounds_report(-5.0, 50.0)["in_bounds"]
    harness.map_calibration = {}

    facade_points = {}
    for facade in ("west", "south", "north", "east"):
        facade_observation = harness.route2_safe_observation_candidates("001", skip_completed=False, facade_filter=facade)[0]
        harness.llm_route2_state = {
            "target_house_id": "001",
            "facade": facade,
            "facade_id": harness.route2_facade_id("001", facade),
            "observation_point": facade_observation,
        }
        fallback = harness.route2_fallback_facade_analysis("test fallback")
        points = harness.route2_generate_facade_scan_points(fallback)
        facade_points[facade] = points
        assert points, f"expected fallback scan points for {facade}"
        assert {point["height_band"] for point in points} == {"single_300cm"}
        assert {point["z"] for point in points} == {300.0}
        assert harness.scan_point_validation_report("001", points)["valid"]
    assert len(facade_points["west"]) > len(facade_points["south"]) > len(facade_points["north"])
    assert len(facade_points["south"]) > len(facade_points["east"])

    harness.llm_route2_state = {
        "target_house_id": "001",
        "facade": "west",
        "facade_id": harness.route2_facade_id("001", "west"),
        "observation_point": harness.route2_safe_observation_candidates("001", skip_completed=False, facade_filter="west")[0],
    }
    fallback = harness.route2_fallback_facade_analysis("test fallback")
    assert fallback["floor_count_estimate"] == 2
    assert fallback["recommended_height_bands"] == ["single_300cm"]
    medium_points = facade_points["west"]

    three_floor = dict(fallback, floor_count_estimate=3)
    three_floor_points = harness.route2_generate_facade_scan_points(three_floor)
    assert {point["height_band"] for point in three_floor_points} == {"low_ground_250cm", "upper_floor_600cm"}
    assert {point["z"] for point in three_floor_points} == {250.0, 600.0}

    with_cue = dict(
        fallback,
        detected_cues=[{"type": "door_candidate", "region": "low-center", "confidence": 0.8}],
    )
    cue_points = harness.route2_generate_facade_scan_points(with_cue)
    assert any("confirm" in point["scan_id"] for point in cue_points), "expected confirm scan point"

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        for facade in ("west", "south"):
            raw_points = facade_points[facade]
            assigned = harness.route2_assign_global_scan_ids(output_dir, "001", facade, raw_points)
            facade_dir = harness.route2_facade_dir(output_dir, "001", facade)
            harness.write_json_artifact(
                facade_dir / "facade_search_plan.json",
                {"house_id": "001", "facade": facade, "scan_points": assigned},
            )
        merged = harness.route2_write_merged_scan_points(output_dir, "001")
        orders = [point["global_scan_order"] for point in merged]
        assert orders == list(range(1, len(orders) + 1)), orders
        assert merged[0]["scan_id"].startswith("001_scan_0001_west_")
        payload = json.loads((output_dir / "scan_points.json").read_text(encoding="utf-8"))
        assert [point["global_scan_order"] for point in payload["scan_points"]] == orders
        (output_dir / "frames" / "frame_000001").mkdir(parents=True)
        (output_dir / "frames" / "frame_000001-01").mkdir(parents=True)
        harness.append_jsonl(output_dir / "scan_execution_log.jsonl", {"frame_indices": [2]})
        assert harness.route2_next_frame_index(output_dir) == 3
        harness.append_jsonl(output_dir / "lidar_capture_log.jsonl", {"frame_index": 2, "capture_dir": "frames/frame_000002"})
        harness.route2_write_lidar_summary(output_dir, running=False)
        trajectory = json.loads((output_dir / "trajectory.json").read_text(encoding="utf-8"))
        assert trajectory["frame_count"] == 1

    print(
        "OK route2 facade verification: "
        f"nearest={observation['facade']} fallback_points={len(medium_points)} cue_points={len(cue_points)}"
    )


if __name__ == "__main__":
    main()
