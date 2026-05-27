from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from obstacle_representation_2.teacher import depth_masks

from .or2_direction_rule import deep_red_stop_active, red_recovery_clear_enough, select_or2_direction
from .plans import DEFAULT_METHOD_ID, make_default_plans, normalize_plans, sanitize_id, sync_oa2_plan_into_oa3


def prediction_from_stop_mask(stop: np.ndarray, warning: np.ndarray | None = None, clearance: np.ndarray | None = None) -> dict:
    stop = stop.astype(np.float32)
    warning = np.zeros_like(stop) if warning is None else warning.astype(np.float32)
    clearance = np.zeros_like(stop) if clearance is None else clearance.astype(np.float32)
    return {
        "front_risk_state": "must_stop" if float(np.mean(stop[25:75, 34:66] >= 0.5)) > 0.01 else "clear",
        "must_stop_mask": stop,
        "obstacle_warning_mask": warning,
        "clearance_warning_mask": clearance,
    }


class OR2DirectionRuleTest(unittest.TestCase):
    def test_front_red_left_clear_selects_left(self) -> None:
        stop = np.zeros((96, 96), dtype=np.float32)
        stop[25:80, 55:96] = 1.0
        result = select_or2_direction(prediction_from_stop_mask(stop), {"front_min_depth_cm": 80.0}, {"bearing_deg_body": -5.0})
        self.assertEqual(result["selected_direction"], "left")


class OA2PlanSyncTest(unittest.TestCase):
    def test_normalize_empty_project_restores_default_episodes(self) -> None:
        data = {
            "active_project_id": "route_obstacle_or2_collection_v1",
            "projects": [
                {
                    "project_id": "route_obstacle_or2_collection_v1",
                    "name": "empty",
                    "episodes": [],
                }
            ],
        }
        normalized = normalize_plans(data)
        episodes = normalized["projects"][0]["episodes"]
        self.assertGreaterEqual(len(episodes), 10)
        self.assertEqual(episodes[0]["episode_id"], "E01")

    def test_sync_oa2_plan_imports_all_active_project_episodes(self) -> None:
        oa2_data = {
            "active_project_id": "route_obstacle_collection_v2",
            "projects": [
                {
                    "project_id": "route_obstacle_collection_v2",
                    "name": "OA2 points",
                    "environment_id": "tree_or_pole",
                    "default_method": "pointcloud_direction_rule",
                    "episodes": [
                        {
                            "episode_id": "E01",
                            "enabled": True,
                            "start_pose": [1, 2, 3, 4],
                            "goal_pose": [5, 6, 7, 8],
                            "scenario_id": "oa2_route_E01",
                            "environment_id": "tree_or_pole",
                            "method": "pointcloud_direction_rule",
                            "obstacle_hint": "tree",
                            "operator_note": "baseline",
                        },
                        {
                            "episode_id": "E13",
                            "enabled": True,
                            "start_pose": [10, 20, 30, 40],
                            "goal_pose": [50, 60, 70, 80],
                            "scenario_id": "oa2_route_E13",
                            "environment_id": "fence_or_rail",
                            "method": "pointcloud_direction_rule",
                            "obstacle_hint": "fence",
                            "operator_note": "new point",
                        },
                    ],
                }
            ],
        }
        synced = sync_oa2_plan_into_oa3(make_default_plans(), oa2_data)
        project = synced["projects"][0]
        episodes = project["episodes"]
        self.assertEqual([item["episode_id"] for item in episodes], ["E01", "E13"])
        self.assertEqual(episodes[1]["start_pose"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(episodes[1]["goal_pose"], [50.0, 60.0, 70.0, 80.0])
        self.assertEqual(episodes[1]["obstacle_hint"], "fence")

    def test_sync_oa2_plan_forces_oa3_method(self) -> None:
        oa2_data = {
            "projects": [
                {
                    "project_id": "p",
                    "episodes": [
                        {
                            "episode_id": "E11",
                            "start_pose": [0, 0, 100, 0],
                            "goal_pose": [100, 0, 100, 0],
                            "method": "pointcloud_direction_rule",
                        }
                    ],
                }
            ]
        }
        synced = sync_oa2_plan_into_oa3(make_default_plans(), oa2_data)
        episode = synced["projects"][0]["episodes"][0]
        self.assertEqual(episode["method"], DEFAULT_METHOD_ID)
        self.assertEqual(synced["projects"][0]["default_method"], DEFAULT_METHOD_ID)

    def test_sanitize_id_can_cap_length_for_windows_paths(self) -> None:
        value = sanitize_id("route_obstacle_or2_collection_v1_obstacle_representation_direction_rule_v1_e01_163432", "run", max_len=32)
        self.assertLessEqual(len(value), 32)
        self.assertTrue(value.startswith("route_obstacle"))

    def test_front_red_right_clear_selects_right(self) -> None:
        stop = np.zeros((96, 96), dtype=np.float32)
        stop[25:80, 0:45] = 1.0
        result = select_or2_direction(prediction_from_stop_mask(stop), {"front_min_depth_cm": 80.0}, {"bearing_deg_body": 5.0})
        self.assertEqual(result["selected_direction"], "right")

    def test_sides_red_up_clear_selects_up(self) -> None:
        stop = np.zeros((96, 96), dtype=np.float32)
        stop[25:80, :] = 1.0
        stop[0:24, 24:72] = 0.0
        result = select_or2_direction(prediction_from_stop_mask(stop), {"front_min_depth_cm": 80.0}, {"dz_cm": 120.0})
        self.assertEqual(result["selected_direction"], "up")

    def test_all_escape_corridors_red_selects_backoff(self) -> None:
        stop = np.ones((96, 96), dtype=np.float32)
        result = select_or2_direction(prediction_from_stop_mask(stop), {"front_min_depth_cm": 80.0}, {})
        self.assertEqual(result["selected_direction"], "backoff")

    def test_clear_front_selects_forward(self) -> None:
        stop = np.zeros((96, 96), dtype=np.float32)
        result = select_or2_direction(prediction_from_stop_mask(stop), {"front_min_depth_cm": 600.0}, {"bearing_deg_body": 0.0})
        self.assertIn(result["selected_direction"], {"forward", "slow_forward"})

    def test_manual_dark_red_sample_selects_left(self) -> None:
        sample_dir = Path("obstacle_representation_2_data/manual_review_samples/dark_red_stop_le_100cm/025_0b01c440")
        depth_path = sample_dir / "depth.npy"
        if not depth_path.is_file():
            self.skipTest(f"manual sample missing: {depth_path}")
        masks = depth_masks(np.load(depth_path), image_size=96)
        prediction = {
            "front_risk_state": "must_stop",
            "clearance_warning_mask": masks[0],
            "obstacle_warning_mask": masks[1],
            "must_stop_mask": masks[2],
        }
        result = select_or2_direction(
            prediction,
            {"front_min_depth_cm": 143.875},
            {"bearing_deg_body": -110.01, "dz_cm": -466.4},
        )
        self.assertEqual(result["selected_direction"], "left")

    def test_red_recovery_releases_on_clearance_when_deep_red_left_view(self) -> None:
        prediction = prediction_from_stop_mask(np.zeros((96, 96), dtype=np.float32))
        prediction["front_risk_state"] = "clearance_warning"
        self.assertTrue(red_recovery_clear_enough(prediction, {"front_min_depth_cm": 327.0}, {}))

    def test_red_recovery_does_not_release_when_too_close(self) -> None:
        prediction = prediction_from_stop_mask(np.zeros((96, 96), dtype=np.float32))
        prediction["front_risk_state"] = "clearance_warning"
        self.assertFalse(red_recovery_clear_enough(prediction, {"front_min_depth_cm": 260.0}, {}))

    def test_red_recovery_releases_only_when_front_far_clear(self) -> None:
        prediction = prediction_from_stop_mask(np.zeros((96, 96), dtype=np.float32))
        prediction["front_risk_state"] = "clear"
        self.assertTrue(red_recovery_clear_enough(prediction, {"front_min_depth_cm": 520.0}, {}))

    def test_deep_red_stop_detects_low_front_depth(self) -> None:
        prediction = prediction_from_stop_mask(np.zeros((96, 96), dtype=np.float32))
        prediction["front_risk_state"] = "clear"
        self.assertTrue(deep_red_stop_active(prediction, {"front_min_depth_cm": 80.0}, {}))


if __name__ == "__main__":
    unittest.main()
