from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .build_dataset import build_dataset
from .model import SchemeAObstacleNet, SchemeAPlusObstacleNet
from .schema import GEOMETRY_FEATURE_NAMES, OBSTACLE_LABELS
from .teacher_labels import canonical_obstacle_label, teacher_label_from_event


class TeacherLabelTests(unittest.TestCase):
    def test_fence_and_rail_normalize_to_fence_or_rail(self) -> None:
        self.assertEqual(canonical_obstacle_label("fence"), "fence_or_rail")
        self.assertEqual(canonical_obstacle_label("rail"), "fence_or_rail")
        self.assertEqual(canonical_obstacle_label("low horizontal railing"), "fence_or_rail")

    def test_free_text_tree_hint_normalizes_to_tree_subtype(self) -> None:
        narrow = {"pointcloud_summary": {"obstacle_width_cm": 55.0}}
        wide = {"pointcloud_summary": {"obstacle_width_cm": 320.0}}
        self.assertEqual(canonical_obstacle_label("tree trunk directly ahead", narrow), "tree_trunk_or_pole")
        self.assertEqual(canonical_obstacle_label("tree canopy cluster", wide), "tree_canopy_or_cluster")

    def test_low_wide_geometry_becomes_fence_teacher_label(self) -> None:
        event = {
            "pointcloud_summary": {
                "obstacle_geometry": "low_obstacle",
                "obstacle_width_cm": 360.0,
                "up_swept_clear": True,
            }
        }
        label = teacher_label_from_event(event)
        self.assertEqual(label["label"], "fence_or_rail")
        self.assertTrue(label["flyover_recommended"])

    def test_manual_hard_label_overrides_llm_and_geometry(self) -> None:
        event = {
            "manual_label": "building",
            "manual_label_source": "hard_case",
            "llm_strategy": {"obstacle_hint": "fence_or_rail"},
            "pointcloud_summary": {
                "front_min_depth_cm": 700.0,
                "obstacle_geometry": "none",
            },
        }
        label = teacher_label_from_event(event)
        self.assertEqual(label["label"], "building")
        self.assertEqual(label["teacher_source"], "manual_hard_label")
        self.assertGreater(label["sample_weight"], 1.0)


class DatasetBuilderTests(unittest.TestCase):
    def test_build_dataset_skips_missing_rgb_and_writes_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "source" / "sessions" / "run_a"
            frame = session / "frames" / "frame_000001"
            frame.mkdir(parents=True)
            rgb_path = frame / "rgb.png"
            Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(rgb_path)
            event = {
                "rgb_path": str(rgb_path),
                "depth_npy_path": str(frame / "depth.npy"),
                "pointcloud_summary": {
                    "front_min_depth_cm": 300.0,
                    "obstacle_geometry": "low_obstacle",
                    "obstacle_width_cm": 350.0,
                    "up_swept_clear": True,
                },
                "relative_target": {"distance_cm": 500.0, "bearing_deg_body": 3.0, "dz_cm": 0.0},
                "llm_strategy": {"obstacle_hint": "fence", "environment_id": "fence_or_rail"},
            }
            np.save(frame / "depth.npy", np.ones((16, 16), dtype=np.float32))
            missing_event = dict(event)
            missing_event["rgb_path"] = str(frame / "missing.png")
            events_path = session / "avoidance_events.jsonl"
            events_path.write_text(
                json.dumps(event, ensure_ascii=False) + "\n" + json.dumps(missing_event, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output_root = root / "out"
            summary = build_dataset([root / "source"], output_root=output_root, seed=7)
            self.assertEqual(summary["total_samples"], 1)
            self.assertEqual(summary["missing_counts"]["missing_rgb"], 1)
            self.assertTrue((output_root / "datasets" / "dataset_latest.npz").is_file())
            self.assertTrue((output_root / "datasets" / "dataset_latest.jsonl").is_file())

    def test_manual_hard_cases_are_loaded_and_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            frame = root / "demo" / "frames" / "frame_000019"
            frame.mkdir(parents=True)
            rgb_path = frame / "rgb.png"
            depth_path = frame / "depth.npy"
            capture_path = frame / "capture.json"
            Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8)).save(rgb_path)
            np.save(depth_path, np.full((16, 16), 650.0, dtype=np.float32))
            capture_path.write_text(
                json.dumps(
                    {
                        "rgb_path": str(rgb_path),
                        "depth_npy_path": str(depth_path),
                        "depth_obstacle_summary": {
                            "front_min_depth_cm": 663.0,
                            "obstacle_geometry": "none",
                            "obstacle_width_cm": 668.0,
                            "obstacle_height_cm": 782.0,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manual_dir = output_root / "manual_labels"
            manual_dir.mkdir(parents=True)
            (manual_dir / "hard_cases.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "hard_building_demo",
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "capture_json_path": str(capture_path),
                        "label": "building",
                        "note": "white building side wall hard case",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = build_dataset([root / "missing_source"], output_root=output_root, seed=7)
            self.assertEqual(summary["total_samples"], 1)
            self.assertEqual(summary["manual_label_counts"]["building"], 1)
            rows = [
                json.loads(line)
                for line in (output_root / "datasets" / "dataset_latest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["label"], "building")
            self.assertEqual(rows[0]["teacher_source"], "manual_hard_label")
            self.assertTrue(rows[0]["is_manual_hard_case"])

    def test_group_split_keeps_event_file_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "sessions"
            for run_idx, label in enumerate(("building", "fence_or_rail", "tree", "unknown"), start=1):
                session = source / f"run_{run_idx}"
                session.mkdir(parents=True)
                lines = []
                for frame_idx in range(3):
                    frame = session / "frames" / f"frame_{frame_idx:06d}"
                    frame.mkdir(parents=True)
                    rgb_path = frame / "rgb.png"
                    depth_path = frame / "depth.npy"
                    Image.fromarray(np.full((16, 16, 3), 60 + run_idx, dtype=np.uint8)).save(rgb_path)
                    np.save(depth_path, np.ones((16, 16), dtype=np.float32))
                    lines.append(
                        json.dumps(
                            {
                                "rgb_path": str(rgb_path),
                                "depth_npy_path": str(depth_path),
                                "obstacle_hint": label,
                                "pointcloud_summary": {
                                    "front_min_depth_cm": 240.0,
                                    "obstacle_geometry": "vertical_wall",
                                    "obstacle_width_cm": 300.0,
                                },
                            },
                            ensure_ascii=False,
                        )
                    )
                (session / "avoidance_events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            output_root = root / "out"
            build_dataset([root / "source"], output_root=output_root, seed=7)
            rows = [
                json.loads(line)
                for line in (output_root / "datasets" / "dataset_latest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            by_group = {}
            for row in rows:
                by_group.setdefault(row["group_id"], set()).add(row["split"])
            self.assertTrue(by_group)
            self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))


class SchemeAModelTests(unittest.TestCase):
    def test_scheme_a_forward_outputs_label_and_flyover_logits(self) -> None:
        import torch

        model = SchemeAObstacleNet(
            num_labels=len(OBSTACLE_LABELS),
            geometry_dim=len(GEOMETRY_FEATURE_NAMES),
        )
        rgb = torch.randn(2, 3, 96, 96)
        geom = torch.randn(2, len(GEOMETRY_FEATURE_NAMES))
        out = model(rgb, geom)
        self.assertEqual(tuple(out["label_logits"].shape), (2, len(OBSTACLE_LABELS)))
        self.assertEqual(tuple(out["flyover_logits"].shape), (2,))

    def test_scheme_a_plus_forward_outputs_label_and_flyover_logits(self) -> None:
        import torch

        model = SchemeAPlusObstacleNet(
            num_labels=len(OBSTACLE_LABELS),
            geometry_dim=len(GEOMETRY_FEATURE_NAMES),
        )
        rgb = torch.randn(2, 3, 96, 96)
        depth = torch.randn(2, 1, 96, 96)
        geom = torch.randn(2, len(GEOMETRY_FEATURE_NAMES))
        out = model(rgb, geom, depth)
        self.assertEqual(tuple(out["label_logits"].shape), (2, len(OBSTACLE_LABELS)))
        self.assertEqual(tuple(out["flyover_logits"].shape), (2,))

    def test_demo_predict_from_event_returns_label_and_probabilities(self) -> None:
        import torch

        from .demo import predict_obstacle_representation

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "rgb.png"
            model_path = root / "scheme_a_model.pt"
            Image.fromarray(np.full((24, 24, 3), 180, dtype=np.uint8)).save(image_path)
            model = SchemeAObstacleNet(
                num_labels=len(OBSTACLE_LABELS),
                geometry_dim=len(GEOMETRY_FEATURE_NAMES),
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "num_labels": len(OBSTACLE_LABELS),
                        "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
                        "image_size": 32,
                        "labels": list(OBSTACLE_LABELS),
                        "geometry_feature_names": GEOMETRY_FEATURE_NAMES,
                    },
                    "geometry_mean": np.zeros(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                    "geometry_std": np.ones(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                },
                model_path,
            )
            result = predict_obstacle_representation(
                model_path,
                image_path,
                {
                    "pointcloud_summary": {
                        "front_min_depth_cm": 180.0,
                        "obstacle_geometry": "low_obstacle",
                        "obstacle_width_cm": 360.0,
                        "up_swept_clear": True,
                    }
                },
            )
            self.assertIn(result["predicted_label"], OBSTACLE_LABELS)
            self.assertEqual(set(result["probabilities"].keys()), set(OBSTACLE_LABELS))
            self.assertIn("flyover_probability", result)

    def test_demo_predict_loads_scheme_a_plus_checkpoint(self) -> None:
        import torch

        from .demo import predict_obstacle_representation

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "rgb.png"
            depth_path = root / "depth.npy"
            model_path = root / "scheme_a_plus_model.pt"
            Image.fromarray(np.full((24, 24, 3), 180, dtype=np.uint8)).save(image_path)
            np.save(depth_path, np.full((24, 24), 450.0, dtype=np.float32))
            model = SchemeAPlusObstacleNet(
                num_labels=len(OBSTACLE_LABELS),
                geometry_dim=len(GEOMETRY_FEATURE_NAMES),
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "model_version": "scheme_a_plus_v1",
                        "use_depth": True,
                        "num_labels": len(OBSTACLE_LABELS),
                        "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
                        "image_size": 32,
                        "labels": list(OBSTACLE_LABELS),
                        "geometry_feature_names": GEOMETRY_FEATURE_NAMES,
                    },
                    "geometry_mean": np.zeros(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                    "geometry_std": np.ones(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                },
                model_path,
            )
            result = predict_obstacle_representation(
                model_path,
                image_path,
                {
                    "depth_npy_path": str(depth_path),
                    "pointcloud_summary": {
                        "front_min_depth_cm": 663.0,
                        "obstacle_geometry": "none",
                        "obstacle_width_cm": 660.0,
                    },
                },
            )
            self.assertEqual(result["model_version"], "scheme_a_plus_v1")
            self.assertIn(result["predicted_label"], OBSTACLE_LABELS)
            self.assertIn("building_vs_fence_margin", result)

    def test_demo_mask_visualization_matches_rgb_size(self) -> None:
        from .demo import render_prediction_mask

        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        depth = np.full((32, 48), 500.0, dtype=np.float32)
        depth[8:24, 18:30] = 120.0
        mask = render_prediction_mask(
            rgb,
            depth,
            {
                "predicted_label": "fence_or_rail",
                "flyover_recommended": True,
                "confidence": 0.8,
            },
        )
        self.assertEqual(mask.shape, rgb.shape)
        self.assertGreater(int(mask[..., 1].max()), 0)


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
