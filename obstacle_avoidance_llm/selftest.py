from __future__ import annotations

import unittest

from .plans import (
    DEFAULT_PLAN_FILENAME,
    LLM_DIRECT_METHOD_ID,
    LLM_STRATEGY_METHOD_ID,
    make_default_plans,
    method_is_runnable,
)
from .policy import normalize_direct_decision, normalize_strategy_decision, shield_direct_payload


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
