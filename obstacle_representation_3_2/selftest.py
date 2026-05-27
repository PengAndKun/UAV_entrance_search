from __future__ import annotations

import unittest
from pathlib import Path


class ObstacleRepresentation32SafeFineTuneTests(unittest.TestCase):
    def test_training_defaults_preserve_prior_models_and_use_3_2_names(self) -> None:
        from .train import default_training_paths

        paths = default_training_paths(Path("obstacle_representation_3_data"))

        self.assertEqual(paths["dataset"].name, "a_plus_3_1_dataset_latest.npz")
        self.assertEqual(paths["pretrained_model"].name, "a_plus_3_model.pt")
        self.assertEqual(paths["model"].name, "a_plus_3_2_model.pt")
        self.assertEqual(paths["metrics"].name, "a_plus_3_2_training_metrics.json")
        self.assertNotEqual(paths["model"].name, "a_plus_3_1_model.pt")

    def test_safe_finetune_config_biases_stop_learning(self) -> None:
        from .train import default_safe_finetune_config

        config = default_safe_finetune_config()

        self.assertLess(config["learning_rate"], 1e-3)
        self.assertGreater(config["stop_mask_loss_weight"], 1.0)
        self.assertGreater(config["must_stop_loss_weight"], 1.0)
        self.assertGreater(config["risk_class_weights"]["must_stop"], config["risk_class_weights"]["clear"])
        self.assertGreater(config["distillation_weight"], 0.0)

    def test_validation_defaults_use_3_2_report_names(self) -> None:
        from .validate import default_validation_paths

        paths = default_validation_paths(Path("obstacle_representation_3_data"))

        self.assertEqual(paths["dataset"].name, "a_plus_3_1_dataset_latest.npz")
        self.assertEqual(paths["model"].name, "a_plus_3_2_model.pt")
        self.assertEqual(paths["report"].name, "a_plus_3_2_validation_report.json")
        self.assertEqual(paths["comparison"].name, "a_plus_3_1_vs_3_2_comparison.json")


if __name__ == "__main__":
    unittest.main()
