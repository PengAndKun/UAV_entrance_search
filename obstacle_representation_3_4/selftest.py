from __future__ import annotations

import unittest
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class ObstacleRepresentation34TverskyTests(unittest.TestCase):
    def test_training_defaults_use_3_4_names(self) -> None:
        from .train import default_training_paths

        paths = default_training_paths(Path("obstacle_representation_3_data"))

        self.assertEqual(paths["dataset"].name, "a_plus_3_1_dataset_latest.npz")
        self.assertEqual(paths["pretrained_model"].name, "a_plus_3_model.pt")
        self.assertEqual(paths["model"].name, "a_plus_3_4_model.pt")
        self.assertEqual(paths["metrics"].name, "a_plus_3_4_training_metrics.json")

    def test_tversky_config_is_recall_weighted_for_stop_mask(self) -> None:
        from .train import default_tversky_config

        config = default_tversky_config()

        self.assertLess(config["tversky_alpha"], config["tversky_beta"])
        self.assertGreater(config["stop_tversky_loss_weight"], 0.0)
        self.assertGreater(config["must_stop_loss_weight"], 1.0)

    def test_focal_tversky_penalizes_missed_foreground(self) -> None:
        torch = self._require_torch()
        from .train import focal_tversky_loss_from_logits

        target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        target[:, :, 2:6, 2:6] = 1.0
        missed_logits = torch.full((1, 1, 8, 8), -4.0, dtype=torch.float32)
        hit_logits = torch.full((1, 1, 8, 8), -4.0, dtype=torch.float32)
        hit_logits[:, :, 2:6, 2:6] = 4.0

        missed = float(focal_tversky_loss_from_logits(missed_logits, target, alpha=0.3, beta=0.7, gamma=0.75).item())
        hit = float(focal_tversky_loss_from_logits(hit_logits, target, alpha=0.3, beta=0.7, gamma=0.75).item())

        self.assertGreater(missed, hit)

    def _require_torch(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        return torch


if __name__ == "__main__":
    unittest.main()
