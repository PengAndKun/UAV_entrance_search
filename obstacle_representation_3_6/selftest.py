from __future__ import annotations

import unittest
from pathlib import Path


class ObstacleRepresentation36EnsembleTests(unittest.TestCase):
    def test_default_paths_use_3_3_risk_and_3_4_stop_models(self) -> None:
        from .evaluate import default_ensemble_paths

        paths = default_ensemble_paths(Path("obstacle_representation_3_data"))

        self.assertEqual(paths["risk_model"].name, "a_plus_3_3_model.pt")
        self.assertEqual(paths["stop_model"].name, "a_plus_3_4_model.pt")
        self.assertEqual(paths["report"].name, "a_plus_3_6_rule_ensemble_report.json")

    def test_choose_ensemble_prediction_prefers_stop_model_must_stop(self) -> None:
        from .evaluate import choose_ensemble_prediction

        pred = choose_ensemble_prediction(risk_pred=1, stop_pred=3, stop_idx=3)

        self.assertEqual(pred, 3)


if __name__ == "__main__":
    unittest.main()
