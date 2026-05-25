from __future__ import annotations

import unittest

import numpy as np

from .plans import (
    DEFAULT_PLAN_FILENAME,
    LLM_DIRECT_METHOD_ID,
    LLM_STRATEGY_METHOD_ID,
    make_default_plans,
    method_is_runnable,
)
from .height_estimator import estimate_pointcloud_flyover_height
from .policy import (
    apply_strategy_to_episode_metadata,
    normalize_direct_decision,
    normalize_strategy_decision,
    refine_strategy_with_pointcloud_context,
    shield_direct_payload,
    strategy_from_episode_metadata,
)


class ObstacleAvoidanceLLMPolicyTests(unittest.TestCase):
    def test_direct_action_forward_is_blocked_by_safety_shield(self) -> None:
        payload, meta = normalize_direct_decision(
            {
                "action_name": "forward",
                "forward_cm": 80,
                "right_cm": 0,
                "up_cm": 0,
                "yaw_delta_deg": 0,
                "reason": "LLM thinks the path is clear.",
                "confidence": 0.9,
            }
        )
        shielded, shield = shield_direct_payload(
            payload,
            {
                "front_min_depth_cm": 120,
                "forward_swept_clear": False,
                "backoff_swept_clear": True,
                "down_swept_clear": True,
            },
            {"z": 300},
        )
        self.assertGreater(payload["forward_cm"], 0)
        self.assertLessEqual(shielded["forward_cm"], 0)
        self.assertIn(shielded["action_name"], {"backoff", "hold"})
        self.assertEqual(shield["state"], "BLOCKED_FORWARD")
        self.assertEqual(meta["action_name"], "forward")

    def test_direct_action_down_is_blocked_when_down_not_clear(self) -> None:
        payload, _meta = normalize_direct_decision({"action_name": "down", "up_cm": -35, "reason": "descend"})
        shielded, shield = shield_direct_payload(
            payload,
            {"front_min_depth_cm": 900, "forward_swept_clear": True, "down_swept_clear": False},
            {"z": 250},
        )
        self.assertEqual(shielded["action_name"], "hold")
        self.assertEqual(shielded["up_cm"], 0.0)
        self.assertEqual(shield["state"], "BLOCKED_DOWN")

    def test_strategy_response_normalizes_to_pointcloud_rule(self) -> None:
        strategy = normalize_strategy_decision(
            {
                "environment_id": "building_or_roof",
                "obstacle_hint": "building",
                "recommended_method": "anything_else",
                "flyover_z_cm": 1200,
                "lateral_preference": "left",
                "vertical_policy": "flyover_then_descend_near_xy",
                "strategy_reason": "Large obstacle ahead.",
            }
        )
        self.assertEqual(strategy["recommended_method"], "pointcloud_direction_rule")
        self.assertEqual(strategy["environment_id"], "building_or_roof")
        self.assertEqual(strategy["obstacle_hint"], "building")
        self.assertEqual(strategy["flyover_z_cm"], 1200.0)

    def test_strategy_updates_episode_environment_and_hint(self) -> None:
        episode = {
            "episode_id": "E01",
            "environment_id": "default_unreal_scene",
            "obstacle_hint": "unknown",
        }
        strategy = normalize_strategy_decision(
            {
                "environment_id": "fence_or_rail",
                "obstacle_hint": "fence_or_rail",
                "flyover_z_cm": 850,
                "lateral_preference": "right",
                "vertical_policy": "flyover_then_descend_near_xy",
                "strategy_reason": "Fence-like obstacle across the Route6_entrance_search.",
            }
        )
        updated = apply_strategy_to_episode_metadata(episode, strategy)
        self.assertEqual(updated["environment_id"], "fence_or_rail")
        self.assertEqual(updated["obstacle_hint"], "fence_or_rail")
        self.assertEqual(updated["llm_strategy"]["flyover_z_cm"], 850.0)
        self.assertEqual(updated["llm_strategy"]["recommended_method"], "pointcloud_direction_rule")

    def test_strategy_metadata_preserves_pointcloud_height_estimate(self) -> None:
        episode = {"episode_id": "E01", "environment_id": "default_unreal_scene", "obstacle_hint": "unknown"}
        updated = apply_strategy_to_episode_metadata(
            episode,
            {
                "environment_id": "fence_or_rail",
                "obstacle_hint": "fence_or_rail",
                "flyover_z_cm": 180.0,
                "pointcloud_height_estimate": {
                    "available": True,
                    "obstacle_height_cm": 210.0,
                    "recommended_flyover_z_cm": 530.0,
                    "recommended_vertical_offset_cm": 180.0,
                },
                "pointcloud_recommended_flyover_z_cm": 530.0,
                "pointcloud_recommended_vertical_offset_cm": 180.0,
                "pointcloud_obstacle_height_cm": 210.0,
            },
        )
        self.assertEqual(updated["llm_strategy"]["pointcloud_recommended_flyover_z_cm"], 530.0)
        self.assertEqual(updated["llm_strategy"]["pointcloud_height_estimate"]["obstacle_height_cm"], 210.0)

    def test_low_wide_obstacle_misread_as_building_is_refined_to_fence(self) -> None:
        strategy = normalize_strategy_decision(
            {
                "environment_id": "building_or_roof",
                "obstacle_hint": "low_obstacle_ahead_below",
                "strategy_reason": "Low obstacle across the forward corridor.",
            }
        )
        strategy["pointcloud_height_estimate"] = {
            "available": True,
            "obstacle_height_cm": 242.27,
            "obstacle_top_z_cm": 292.314,
            "recommended_flyover_z_cm": 382.314,
        }
        event = {
            "pointcloud_summary": {
                "obstacle_geometry": "low_obstacle",
                "obstacle_width_cm": 359.49,
                "obstacle_height_cm": 382.986,
                "left_swept_clear": False,
                "right_swept_clear": False,
                "up_swept_clear": True,
            }
        }
        refined = refine_strategy_with_pointcloud_context(strategy, event)
        self.assertEqual(refined["environment_id"], "fence_or_rail")
        self.assertEqual(refined["obstacle_hint"], "fence_or_rail")
        self.assertEqual(refined["semantic_refinement_source"], "pointcloud_low_wide_flyover")

    def test_tall_building_strategy_is_not_refined_to_fence(self) -> None:
        strategy = normalize_strategy_decision(
            {
                "environment_id": "building_or_roof",
                "obstacle_hint": "building facade with roof edge",
                "strategy_reason": "Tall building facade blocks the corridor.",
            }
        )
        strategy["pointcloud_height_estimate"] = {
            "available": True,
            "obstacle_height_cm": 900.0,
            "obstacle_top_z_cm": 1180.0,
            "recommended_flyover_z_cm": 1270.0,
        }
        event = {
            "pointcloud_summary": {
                "obstacle_geometry": "vertical_wall",
                "obstacle_width_cm": 700.0,
                "obstacle_height_cm": 900.0,
                "up_swept_clear": False,
            }
        }
        refined = refine_strategy_with_pointcloud_context(strategy, event)
        self.assertEqual(refined["environment_id"], "building_or_roof")
        self.assertEqual(refined["obstacle_hint"], "building")

    def test_free_text_hint_is_canonicalized_and_moved_to_note(self) -> None:
        episode = {
            "episode_id": "E01",
            "environment_id": "default_unreal_scene",
            "obstacle_hint": "unknown",
            "operator_note": "",
        }
        strategy = normalize_strategy_decision(
            {
                "environment_id": "tree_trunk_or_pole",
                "obstacle_hint": "tree trunk directly ahead",
                "strategy_reason": "The obstacle is narrow, so prefer a side offset rather than fly over.",
            }
        )
        updated = apply_strategy_to_episode_metadata(episode, strategy)
        self.assertEqual(updated["obstacle_hint"], "tree_trunk_or_pole")
        self.assertIn("tree trunk directly ahead", updated["operator_note"])
        self.assertIn("prefer a side offset", updated["operator_note"])

    def test_strategy_execution_uses_episode_metadata_without_llm_call(self) -> None:
        episode = {
            "episode_id": "E01",
            "environment_id": "tree_trunk_or_pole",
            "obstacle_hint": "tree_trunk_or_pole",
            "llm_strategy": {
                "environment_id": "tree_trunk_or_pole",
                "obstacle_hint": "tree trunk directly ahead",
                "flyover_z_cm": 0,
                "lateral_preference": "right",
                "vertical_policy": "level_lateral_avoidance",
                "strategy_reason": "Cached analysis from the LLM Analyze step.",
            },
        }
        strategy = strategy_from_episode_metadata(episode)
        self.assertEqual(strategy["environment_id"], "tree_trunk_or_pole")
        self.assertEqual(strategy["obstacle_hint"], "tree_trunk_or_pole")
        self.assertEqual(strategy["lateral_preference"], "right")
        self.assertEqual(strategy["strategy_source"], "episode_metadata")
        self.assertFalse(strategy["llm_call_required"])

    def test_pointcloud_height_estimator_uses_depth_geometry(self) -> None:
        depth = np.full((100, 100), 1200.0, dtype=np.float32)
        depth[25:80, 40:60] = 300.0
        event = {
            "depth_array_cm": depth,
            "camera_info": {
                "horizontal_fov_deg": 90.0,
                "image_width": 100,
                "image_height": 100,
                "location": {"z": 200.0},
                "rotation": {"pitch": 0.0},
            },
            "current_pose": {"z": 200.0},
            "pointcloud_summary": {
                "front_min_depth_cm": 300.0,
                "obstacle_height_cm": 300.0,
                "image_width": 100,
                "image_height": 100,
            },
        }
        estimate = estimate_pointcloud_flyover_height(
            event,
            {"environment_id": "fence_or_rail", "obstacle_hint": "fence_or_rail"},
            safety_margin_cm=80.0,
        )
        self.assertTrue(estimate["available"])
        self.assertEqual(estimate["height_source"], "depth_projection")
        self.assertGreater(estimate["obstacle_height_cm"], 200.0)
        self.assertGreater(estimate["obstacle_top_z_cm"], 300.0)
        self.assertGreater(estimate["recommended_flyover_z_cm"], estimate["obstacle_top_z_cm"])
        self.assertGreater(estimate["recommended_vertical_offset_cm"], 0.0)

    def test_pointcloud_height_estimator_does_not_force_tree_trunk_flyover(self) -> None:
        depth = np.full((64, 64), 900.0, dtype=np.float32)
        depth[18:52, 30:34] = 260.0
        event = {
            "depth_array_cm": depth,
            "camera_info": {
                "horizontal_fov_deg": 90.0,
                "image_width": 64,
                "image_height": 64,
                "location": {"z": 220.0},
            },
            "current_pose": {"z": 220.0},
            "pointcloud_summary": {"front_min_depth_cm": 260.0},
        }
        estimate = estimate_pointcloud_flyover_height(
            event,
            {"environment_id": "tree_trunk_or_pole", "obstacle_hint": "tree_trunk_or_pole"},
        )
        self.assertTrue(estimate["available"])
        self.assertFalse(estimate["flyover_recommended"])
        self.assertEqual(estimate["recommended_vertical_offset_cm"], 0.0)


class ObstacleAvoidanceLLMPlansTests(unittest.TestCase):
    def test_default_llm_plans_have_ten_episodes_and_llm_methods(self) -> None:
        plans = make_default_plans()
        self.assertEqual(DEFAULT_PLAN_FILENAME, "obstacle_avoidance_llm_plans.json")
        self.assertEqual(len(plans["projects"][0]["episodes"]), 10)
        method_ids = {item["method_id"] for item in plans["methods"]}
        self.assertIn(LLM_DIRECT_METHOD_ID, method_ids)
        self.assertIn(LLM_STRATEGY_METHOD_ID, method_ids)
        self.assertTrue(method_is_runnable(plans, LLM_DIRECT_METHOD_ID))
        self.assertTrue(method_is_runnable(plans, LLM_STRATEGY_METHOD_ID))


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
