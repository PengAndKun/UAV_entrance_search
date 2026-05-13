from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from verify_route2_facade import _Route2Harness, _Var


class _Route3Harness(_Route2Harness):
    def __init__(self) -> None:
        super().__init__()
        self.llm_route3_move_tick_ms_var = _Var(150)
        self.llm_route3_nav_step_cm_var = _Var(20)
        self.llm_route3_reach_tol_cm_var = _Var(60)
        self.llm_route3_z_tol_cm_var = _Var(40)
        self.llm_route3_yaw_tol_deg_var = _Var(10)
        self.llm_route3_max_stage_s_var = _Var(90)
        self.llm_route3_state = {}
        self.llm_route3_completed_facades = set()
        self.llm_route3_blocked_facades = set()
        self.map_world_bounds = (-5000.0, -5000.0, 5000.0, 5000.0)


def assert_close(value: float, expected: float, eps: float = 1e-6) -> None:
    assert abs(float(value) - float(expected)) <= eps, f"{value} != {expected}"


def main() -> None:
    harness = _Route3Harness()
    cfg = harness.route3_nav_config()
    assert cfg["nav_step_cm"] == 20.0

    current = {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0}
    payload = harness.route3_movement_payload_for_target(current, {"x": 200.0, "y": 0.0, "z": 100.0, "yaw": 0.0}, cfg)
    assert payload["forward_cm"] > 0.0 and payload["right_cm"] == 0.0, payload
    assert_close(payload["forward_cm"], cfg["nav_step_cm"])

    payload = harness.route3_movement_payload_for_target(current, {"x": 0.0, "y": 200.0, "z": 100.0, "yaw": 0.0}, cfg)
    assert payload["right_cm"] > 0.0 and payload["forward_cm"] == 0.0, payload

    current_yaw90 = {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 90.0}
    payload = harness.route3_movement_payload_for_target(current_yaw90, {"x": 0.0, "y": 200.0, "z": 100.0, "yaw": 90.0}, cfg)
    assert payload["forward_cm"] > 0.0 and abs(payload["right_cm"]) <= 1e-6, payload

    payload = harness.route3_movement_payload_for_target(current, {"x": 0.0, "y": 0.0, "z": 250.0, "yaw": 45.0}, cfg)
    assert payload["up_cm"] > 0.0 and payload["yaw_delta_deg"] > 0.0, payload
    assert payload["yaw_delta_deg"] == 30.0, payload

    reached = harness.route3_pose_error(
        {"x": 10.0, "y": -10.0, "z": 115.0, "yaw": 5.0},
        {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
        cfg,
    )
    assert reached["reached"], reached
    not_reached = harness.route3_pose_error(
        {"x": 100.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
        {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0},
        cfg,
    )
    assert not not_reached["reached"], not_reached

    predicted = harness.route3_predict_next_pose(
        {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 90.0},
        {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 10.0, "yaw_delta_deg": -15.0},
    )
    assert abs(predicted["x"]) <= 1e-6, predicted
    assert_close(predicted["y"], 20.0)
    assert_close(predicted["z"], 110.0)
    assert_close(predicted["yaw"], 75.0)

    harness.custom_bboxes = {
        "001": {"min_x": -100.0, "max_x": 100.0, "min_y": -100.0, "max_y": 100.0, "center_x": 0.0, "center_y": 0.0},
        "002": {"min_x": 500.0, "max_x": 700.0, "min_y": -100.0, "max_y": 100.0, "center_x": 600.0, "center_y": 0.0},
    }
    safe = harness.route3_safety_report_for_pose("001", {"x": 300.0, "y": 300.0, "z": 100.0, "yaw": 0.0})
    assert safe["safe"], safe
    blocked = harness.route3_safety_report_for_pose("001", {"x": 600.0, "y": 0.0, "z": 100.0, "yaw": 0.0})
    assert not blocked["safe"] and blocked["reason"] == "non_target_house_clearance", blocked

    out_of_bounds = harness.route3_safety_report_for_pose("001", {"x": 6000.0, "y": 0.0, "z": 100.0, "yaw": 0.0})
    assert not out_of_bounds["safe"] and out_of_bounds["reason"] == "map_boundary", out_of_bounds

    harness.custom_bboxes = {
        "001": {"min_x": 100.0, "max_x": 300.0, "min_y": -120.0, "max_y": 120.0, "center_x": 200.0, "center_y": 0.0},
        "002": {"min_x": 450.0, "max_x": 650.0, "min_y": -120.0, "max_y": 120.0, "center_x": 550.0, "center_y": 0.0},
    }
    plan = harness.route3_plan_navigation_waypoints(
        {"x": -300.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
        {"x": 900.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
        "003",
        grid_cm=100.0,
    )
    assert plan["status"] == "ok", plan
    assert plan["reason"] == "astar_path", plan
    waypoints = plan["waypoints"]
    assert len(waypoints) >= 2, waypoints
    obstacles = harness.route3_navigation_obstacles("003")
    previous = {"x": -300.0, "y": 0.0}
    for waypoint in waypoints:
        assert not harness.route3_segment_blocked_by_obstacles(previous, waypoint, obstacles), (previous, waypoint, plan)
        previous = waypoint
    blocked_target = harness.route3_safety_report_for_pose("003", {"x": 200.0, "y": 0.0, "z": 300.0, "yaw": 0.0})
    assert not blocked_target["safe"] and blocked_target["reason"] in {"target_house_bbox", "non_target_house_clearance"}, blocked_target

    harness.map_world_bounds = (1000.0, -500.0, 5000.0, 3000.0)
    harness.map_calibration = {
        "affine_world_to_image": [[0.05, 0.0, 500.0], [0.0, 0.05, 500.0]],
        "image_width": 1000,
        "image_height": 1000,
    }
    harness.custom_bboxes = {
        "002": {"min_x": -337.67, "max_x": 1726.0, "min_y": 1420.65, "max_y": 2177.44, "center_x": 694.17, "center_y": 1799.04},
        "003": {"min_x": -1400.0, "max_x": -1200.0, "min_y": 1450.0, "max_y": 2150.0, "center_x": -1300.0, "center_y": 1800.0},
    }
    west_plan = harness.route3_plan_navigation_waypoints(
        {"x": 2125.22, "y": 1463.25, "z": 340.0, "yaw": 180.0},
        {"x": -512.21, "y": 1799.04, "z": 340.0, "yaw": 0.0},
        "002",
        grid_cm=100.0,
    )
    assert west_plan["status"] == "ok", west_plan
    assert float(west_plan["bounds"]["min_x"]) < -512.21, west_plan
    harness.map_calibration = {}
    harness.map_world_bounds = (-5000.0, -5000.0, 5000.0, 5000.0)

    harness.custom_bboxes = {
        "002": {"min_x": -337.67, "max_x": 1726.0, "min_y": 1420.65, "max_y": 2177.44, "center_x": 694.17, "center_y": 1799.04},
    }
    north_corner_start = {"x": 1732.196, "y": 2169.938, "z": 298.0, "yaw": 171.743}
    bad_north_segment_end = {"x": 1622.165, "y": 2195.394, "z": 298.0, "yaw": 171.743}
    target_obstacles = harness.route3_navigation_obstacles("002")
    bad_segment = harness.route3_segment_blocked_by_obstacles(north_corner_start, bad_north_segment_end, target_obstacles)
    assert bad_segment and bad_segment.get("house_id") == "002", bad_segment
    north_plan = harness.route3_plan_navigation_waypoints(
        north_corner_start,
        {"x": 694.17, "y": 2413.99, "z": 298.0, "yaw": -90.0},
        "002",
        grid_cm=100.0,
    )
    assert north_plan["status"] == "ok", north_plan
    previous = north_corner_start
    for waypoint in north_plan["waypoints"]:
        assert not harness.route3_segment_blocked_by_obstacles(previous, waypoint, target_obstacles), (previous, waypoint, north_plan)
        previous = waypoint

    harness.custom_bboxes = {
        "001": {"min_x": 0.0, "max_x": 200.0, "min_y": 0.0, "max_y": 200.0, "center_x": 100.0, "center_y": 100.0},
        "002": {"min_x": 1000.0, "max_x": 1200.0, "min_y": 0.0, "max_y": 200.0, "center_x": 1100.0, "center_y": 100.0},
    }
    inside_start = {"x": 100.0, "y": 250.0, "z": 280.0, "yaw": 0.0}
    outside_target = {"x": 600.0, "y": 600.0, "z": 280.0, "yaw": 0.0}
    escape_plan = harness.route3_plan_navigation_waypoints(inside_start, outside_target, "002", grid_cm=100.0)
    assert escape_plan["status"] == "ok", escape_plan
    assert escape_plan["reason"].startswith("start_escape_then_"), escape_plan
    first_waypoint = escape_plan["waypoints"][0]
    assert first_waypoint.get("escape_from_obstacle_house_id") == "001", escape_plan
    assert first_waypoint.get("waypoint_role") == "escape", first_waypoint
    strict_cfg = dict(cfg)
    strict_cfg["reach_tol_cm"] = float(first_waypoint["strict_reach_tol_cm"])
    strict_cfg["yaw_tol_deg"] = float(first_waypoint["strict_yaw_tol_deg"])
    assert not harness.route3_pose_error(inside_start, first_waypoint, strict_cfg)["reached"], first_waypoint
    assert not harness.route3_point_blocked_by_obstacles(
        float(first_waypoint["x"]),
        float(first_waypoint["y"]),
        harness.route3_navigation_obstacles("002"),
    ), first_waypoint
    escape_obstacle = first_waypoint["escape_from_obstacle"]
    dx = float(first_waypoint["x"]) - float(inside_start["x"])
    dy = float(first_waypoint["y"]) - float(inside_start["y"])
    length = max(1.0, float(np.hypot(dx, dy)))
    predicted_escape_step = {
        "x": float(inside_start["x"]) + 20.0 * dx / length,
        "y": float(inside_start["y"]) + 20.0 * dy / length,
        "z": float(inside_start["z"]),
        "yaw": 0.0,
    }
    escape_safety = harness.route3_safety_report_for_pose("002", predicted_escape_step)
    assert not escape_safety["safe"], escape_safety
    assert harness.route3_escape_safety_allowed(
        inside_start,
        predicted_escape_step,
        first_waypoint,
        escape_safety,
        escape_obstacle,
    ), (escape_safety, first_waypoint)

    unordered = [
        {"scan_id": "a", "x": 0.0, "y": 0.0, "z": 300.0, "floor_index": 1, "local_scan_index": 0},
        {"scan_id": "b", "x": 1000.0, "y": 0.0, "z": 300.0, "floor_index": 1, "local_scan_index": 1},
        {"scan_id": "c", "x": 0.0, "y": 0.0, "z": 600.0, "floor_index": 2, "local_scan_index": 2},
    ]
    ordered = harness.route2_order_scan_points_continuously(unordered, start_pose={"x": 0.0, "y": 0.0, "z": 300.0})
    assert [point["scan_id"] for point in ordered] == ["a", "c", "b"], ordered
    assert ordered[1]["previous_local_scan_id"] == "a", ordered
    assert ordered[1]["travel_delta_cm"] < ordered[2]["travel_delta_cm"], ordered

    observation = {
        "observation_boundary_adjustment": {"source": "blocking_house_clearance_boundary"},
        "observation_panorama_coverage_ratio": 0.93,
    }
    assert harness.route3_observation_needs_panorama(observation)
    observation = {"observation_panorama_coverage_ratio": 0.5}
    assert harness.route3_observation_needs_panorama(observation)
    observation = {"observation_panorama_coverage_ratio": 0.95}
    assert not harness.route3_observation_needs_panorama(observation)
    harness.custom_bboxes = {
        "001": {"min_x": 0.0, "max_x": 1000.0, "min_y": 0.0, "max_y": 800.0, "center_x": 500.0, "center_y": 400.0},
    }
    poses = harness.route3_panorama_observation_poses("001", "south", {"x": 500.0, "y": -850.0, "z": 300.0, "yaw": 90.0})
    assert [pose["label"] for pose in poses] == ["left", "center", "right"], poses
    left_delta = harness._normalize_angle_deg(float(poses[0]["yaw"]) - float(poses[1]["yaw"]))
    right_delta = harness._normalize_angle_deg(float(poses[2]["yaw"]) - float(poses[1]["yaw"]))
    assert left_delta * right_delta < 0.0, poses
    assert 15.0 <= abs(left_delta) <= 45.0, poses
    assert 15.0 <= abs(right_delta) <= 45.0, poses

    with tempfile.TemporaryDirectory() as temp_dir:
        hedge_path = Path(temp_dir) / "hedge.png"
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:48, :] = (210, 220, 230)
        image[48:, :] = (30, 150, 40)
        assert cv2.imwrite(str(hedge_path), image)
        obstacle = harness.route3_obstacle_fallback_analysis(hedge_path, {"z": 280.0}, "")
        assert obstacle["foreground_obstacle_present"], obstacle
        assert obstacle["recommend_raise"], obstacle
        assert float(obstacle["recommended_observation_z_cm"]) > 280.0, obstacle

    print("OK route3 autonomy verification: movement payload, reach tolerance, prediction, and safety checks passed")


if __name__ == "__main__":
    main()
