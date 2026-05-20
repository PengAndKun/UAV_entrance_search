from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

from .demo import predict_obstacle_representation_2, render_affordance_overlay
from .model import APlus2AffordanceNet
from .schema import DIRECTION_LABELS, GEOMETRY_FEATURE_NAMES
from .teacher import compute_affordance_teacher


class ObstacleRepresentation2Tests(unittest.TestCase):
    def test_teacher_blocks_forward_and_selects_up_for_wide_obstacle(self) -> None:
        depth = np.full((64, 64), 800.0, dtype=np.float32)
        depth[20:48, 16:48] = 180.0
        event = {
            "pointcloud_summary": {
                "front_min_depth_cm": 180.0,
                "left_min_depth_cm": 180.0,
                "right_min_depth_cm": 180.0,
                "up_min_depth_cm": 700.0,
                "forward_swept_clear": False,
                "left_swept_clear": False,
                "right_swept_clear": False,
                "up_swept_clear": True,
                "obstacle_width_cm": 360.0,
                "obstacle_geometry": "low_obstacle",
            },
            "obstacle_hint": "fence_or_rail",
        }
        teacher = compute_affordance_teacher(event, depth, image_size=32)
        self.assertTrue(teacher["red_front_blocked"])
        self.assertEqual(teacher["direction_label"], "up")
        self.assertEqual(teacher["direction_scores"]["forward"], 0.0)

    def test_model_forward_shapes(self) -> None:
        model = APlus2AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_directions=len(DIRECTION_LABELS))
        out = model(
            torch.zeros(2, 3, 32, 32),
            torch.zeros(2, 1, 32, 32),
            torch.zeros(2, len(GEOMETRY_FEATURE_NAMES)),
        )
        self.assertEqual(tuple(out["mask_logits"].shape), (2, 2, 32, 32))
        self.assertEqual(tuple(out["direction_logits"].shape), (2, len(DIRECTION_LABELS)))
        self.assertEqual(tuple(out["score_logits"].shape), (2, len(DIRECTION_LABELS)))
        self.assertEqual(tuple(out["flyover_delta"].shape), (2,))

    def test_demo_loads_checkpoint_and_renders_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "a_plus_2_model.pt"
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.npy"
            Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8)).save(rgb_path)
            np.save(depth_path, np.full((32, 32), 800.0, dtype=np.float32))
            model = APlus2AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_directions=len(DIRECTION_LABELS))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "model_version": "a_plus_2_v1",
                        "image_size": 32,
                        "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
                        "direction_labels": list(DIRECTION_LABELS),
                    },
                    "geometry_mean": np.zeros(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                    "geometry_std": np.ones(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                },
                model_path,
            )
            prediction = predict_obstacle_representation_2(
                model_path,
                rgb_path,
                {"depth_npy_path": str(depth_path), "pointcloud_summary": {"front_min_depth_cm": 800.0}},
                device_name="cpu",
            )
            overlay = render_affordance_overlay(np.asarray(Image.open(rgb_path).convert("RGB")), prediction)
            self.assertIn(prediction["selected_direction"], DIRECTION_LABELS)
            self.assertEqual(tuple(overlay.shape), (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
