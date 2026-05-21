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
from .schema import GEOMETRY_FEATURE_NAMES, RISK_STATES
from .teacher import compute_affordance_teacher


class ObstacleRepresentation2Tests(unittest.TestCase):
    def test_teacher_assigns_three_depth_risk_layers(self) -> None:
        depth = np.full((64, 64), 800.0, dtype=np.float32)
        depth[0:12, :] = 330.0
        depth[20:32, :] = 180.0
        depth[42:50, :] = 80.0
        event = {"pointcloud_summary": {"front_min_depth_cm": 80.0}}
        teacher = compute_affordance_teacher(event, depth, image_size=32)
        self.assertEqual(tuple(teacher["masks"].shape), (3, 32, 32))
        self.assertEqual(teacher["front_risk_state"], "must_stop")
        self.assertTrue(teacher["must_stop"])
        self.assertFalse(teacher["can_forward"])

    def test_model_forward_shapes(self) -> None:
        model = APlus2AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(RISK_STATES))
        out = model(
            torch.zeros(2, 3, 32, 32),
            torch.zeros(2, 1, 32, 32),
            torch.zeros(2, len(GEOMETRY_FEATURE_NAMES)),
        )
        self.assertEqual(tuple(out["mask_logits"].shape), (2, 3, 32, 32))
        self.assertEqual(tuple(out["risk_logits"].shape), (2, len(RISK_STATES)))
        self.assertEqual(tuple(out["can_forward_logits"].shape), (2,))
        self.assertEqual(tuple(out["must_stop_logits"].shape), (2,))

    def test_demo_loads_checkpoint_and_renders_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "a_plus_2_model.pt"
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.npy"
            Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8)).save(rgb_path)
            np.save(depth_path, np.full((32, 32), 800.0, dtype=np.float32))
            model = APlus2AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(RISK_STATES))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "model_version": "a_plus_2_v1",
                        "image_size": 32,
                        "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
                        "risk_states": list(RISK_STATES),
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
            self.assertIn(prediction["front_risk_state"], RISK_STATES)
            self.assertEqual(tuple(overlay.shape), (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
