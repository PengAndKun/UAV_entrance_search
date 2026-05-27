from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class ObstacleRepresentation35HardReplayTests(unittest.TestCase):
    def test_training_defaults_use_3_5_names_and_3_4_seed(self) -> None:
        from .train import default_training_paths

        paths = default_training_paths(Path("obstacle_representation_3_data"))

        self.assertEqual(paths["dataset"].name, "a_plus_3_1_dataset_latest.npz")
        self.assertEqual(paths["pretrained_model"].name, "a_plus_3_4_model.pt")
        self.assertEqual(paths["model"].name, "a_plus_3_5_model.pt")
        self.assertEqual(paths["metrics"].name, "a_plus_3_5_training_metrics.json")

    def test_hard_replay_weights_emphasize_must_stop_and_review_samples(self) -> None:
        from .train import hard_replay_sample_weights

        risk = np.asarray([0, 1, 2, 3], dtype=np.int64)
        must_stop = np.asarray([False, False, False, True])
        needs_review = np.asarray([False, True, False, False])
        weights = hard_replay_sample_weights(risk, must_stop, needs_review)

        self.assertGreater(float(weights[3]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[0]))


if __name__ == "__main__":
    unittest.main()
