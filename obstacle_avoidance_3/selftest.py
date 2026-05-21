from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from obstacle_representation_2.teacher import depth_masks

from .or2_direction_rule import select_or2_direction


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


if __name__ == "__main__":
    unittest.main()
