from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


class ObstacleRepresentation33FocalTests(unittest.TestCase):
    def test_training_defaults_use_3_3_names(self) -> None:
        from .train import default_training_paths

        paths = default_training_paths(Path("obstacle_representation_3_data"))

        self.assertEqual(paths["dataset"].name, "a_plus_3_1_dataset_latest.npz")
        self.assertEqual(paths["pretrained_model"].name, "a_plus_3_model.pt")
        self.assertEqual(paths["model"].name, "a_plus_3_3_model.pt")
        self.assertEqual(paths["metrics"].name, "a_plus_3_3_training_metrics.json")

    def test_effective_number_weights_raise_rare_classes(self) -> None:
        from .train import effective_number_class_weights

        weights = effective_number_class_weights(np.asarray([1000, 600, 1600, 30], dtype=np.float32), beta=0.999)

        self.assertGreater(float(weights[3]), float(weights[0]))
        self.assertGreater(float(weights[3]), float(weights[2]))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=5)

    def test_focal_loss_downweights_easy_examples(self) -> None:
        torch = self._require_torch()
        from .train import focal_cross_entropy

        easy_logits = torch.tensor([[8.0, -2.0, -2.0, -2.0]], dtype=torch.float32)
        hard_logits = torch.tensor([[0.4, 0.2, 0.1, 0.0]], dtype=torch.float32)
        target = torch.tensor([0], dtype=torch.long)
        weights = torch.ones(4, dtype=torch.float32)

        easy = float(focal_cross_entropy(easy_logits, target, weights, gamma=2.0).item())
        hard = float(focal_cross_entropy(hard_logits, target, weights, gamma=2.0).item())

        self.assertLess(easy, hard)

    def _require_torch(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        return torch


if __name__ == "__main__":
    unittest.main()
