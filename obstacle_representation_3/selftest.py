from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .schema import GEOMETRY_FEATURE_NAMES, MASK_CHANNELS, PROJECTION_BOX, RISK_STATES
from .teacher import compute_affordance_teacher, depth_masks


class ObstacleRepresentation3TeacherTests(unittest.TestCase):
    def test_deep_red_inside_projection_box_triggers_must_stop(self) -> None:
        depth = np.full((100, 100), 800.0, dtype=np.float32)
        depth[45:49, 45:49] = 80.0

        teacher = compute_affordance_teacher({"pointcloud_summary": {"front_min_depth_cm": 80.0}}, depth, image_size=100)

        self.assertEqual(teacher["front_risk_state"], "must_stop")
        self.assertEqual(teacher["front_risk_index"], 3)
        self.assertTrue(teacher["must_stop"])
        self.assertFalse(teacher["can_forward"])
        self.assertGreater(teacher["front_box_stop_fraction"], 0.01)

    def test_deep_red_outside_projection_box_is_obstacle_warning(self) -> None:
        depth = np.full((100, 100), 800.0, dtype=np.float32)
        depth[2:20, 2:20] = 80.0

        teacher = compute_affordance_teacher({"pointcloud_summary": {"front_min_depth_cm": 80.0}}, depth, image_size=100)

        self.assertEqual(teacher["front_risk_state"], "obstacle_warning")
        self.assertEqual(teacher["front_risk_index"], 2)
        self.assertFalse(teacher["must_stop"])
        self.assertTrue(teacher["can_forward"])
        self.assertEqual(teacher["front_box_stop_fraction"], 0.0)
        self.assertGreater(teacher["full_stop_fraction"], 0.0)

    def test_front_min_depth_alone_does_not_force_must_stop(self) -> None:
        depth = np.full((100, 100), 800.0, dtype=np.float32)

        teacher = compute_affordance_teacher({"pointcloud_summary": {"front_min_depth_cm": 80.0}}, depth, image_size=100)

        self.assertEqual(teacher["front_risk_state"], "obstacle_warning")
        self.assertFalse(teacher["must_stop"])
        self.assertTrue(teacher["can_forward"])
        self.assertEqual(teacher["front_box_stop_fraction"], 0.0)

    def test_stop_fraction_below_threshold_is_obstacle_warning(self) -> None:
        depth = np.full((100, 100), 800.0, dtype=np.float32)
        depth[38, 42:47] = 80.0

        teacher = compute_affordance_teacher({"pointcloud_summary": {"front_min_depth_cm": 80.0}}, depth, image_size=100)

        self.assertEqual(teacher["front_box_pixel_count"], 544)
        self.assertAlmostEqual(teacher["front_box_stop_fraction"], 5.0 / 544.0)
        self.assertEqual(teacher["front_risk_state"], "obstacle_warning")
        self.assertFalse(teacher["must_stop"])

    def test_stop_fraction_just_over_threshold_triggers_must_stop(self) -> None:
        depth = np.full((100, 100), 800.0, dtype=np.float32)
        depth[38, 42:48] = 80.0

        teacher = compute_affordance_teacher({"pointcloud_summary": {"front_min_depth_cm": 80.0}}, depth, image_size=100)

        self.assertEqual(teacher["front_box_pixel_count"], 544)
        self.assertAlmostEqual(teacher["front_box_stop_fraction"], 6.0 / 544.0)
        self.assertEqual(teacher["front_risk_state"], "must_stop")
        self.assertTrue(teacher["must_stop"])

    def test_depth_masks_keep_or2_channel_order(self) -> None:
        depth = np.full((6, 6), 800.0, dtype=np.float32)
        depth[0:2, :] = 330.0
        depth[2:4, :] = 180.0
        depth[4:6, :] = 80.0

        masks = depth_masks(depth, image_size=6)

        self.assertEqual(MASK_CHANNELS, ("clearance_warning", "obstacle_warning", "must_stop"))
        self.assertEqual(tuple(masks.shape), (3, 6, 6))
        self.assertTrue(np.all(masks[0, 0:2, :] == 1.0))
        self.assertTrue(np.all(masks[1, 2:4, :] == 1.0))
        self.assertTrue(np.all(masks[2, 4:6, :] == 1.0))


class ObstacleRepresentation3DatasetBuilderTests(unittest.TestCase):
    def test_build_dataset_from_broad_historical_jsonl_scan(self) -> None:
        from .build_dataset import build_dataset

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "historical"
            route_dir = data_root / "route6_explore_runs" / "house_002" / "session_001"
            route_dir.mkdir(parents=True)
            rgb_path = route_dir / "rgb_0001.jpg"
            depth_path = route_dir / "depth_0001.npy"
            rgb_path.write_bytes(b"dummy rgb bytes")

            depth = np.full((100, 100), 800.0, dtype=np.float32)
            depth[45:52, 45:52] = 80.0
            np.save(str(depth_path), depth)

            event = {
                "rgb_path": "rgb_0001.jpg",
                "depth_npy_path": "depth_0001.npy",
                "pointcloud_summary": {
                    "front_min_depth_cm": 80.0,
                    "front_mean_depth_cm": 220.0,
                    "nearest_distance_cm": 80.0,
                    "valid_depth_count": 10000,
                    "invalid_depth_count": 0,
                    "forward_swept_clear": False,
                },
                "relative_target": {
                    "distance_cm": 350.0,
                    "bearing_deg_body": 4.0,
                    "dz_cm": -20.0,
                },
            }
            event_file = route_dir / "route6_teacher_events.jsonl"
            event_file.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")

            output_root = tmp_path / "or3"
            summary = build_dataset([data_root], output_root=output_root, image_size=100, seed=7)

            jsonl_path = output_root / "datasets" / "a_plus_3_dataset_latest.jsonl"
            npz_path = output_root / "datasets" / "a_plus_3_dataset_latest.npz"
            summary_path = output_root / "datasets" / "a_plus_3_dataset_summary.json"
            self.assertTrue(jsonl_path.is_file())
            self.assertTrue(npz_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertEqual(summary["total_samples"], 1)
            self.assertEqual(summary["thresholds"]["stop_depth_cm"], 100.0)
            self.assertEqual(summary["projection_box_stop_support_count"], 1)

            sample = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("projection_box", sample)
            self.assertGreater(sample["front_box_stop_fraction"], 0.01)
            self.assertEqual(sample["front_risk_state"], "must_stop")
            self.assertTrue(sample["must_stop"])
            self.assertEqual(len(sample["geometry_features"]), len(GEOMETRY_FEATURE_NAMES))
            self.assertEqual(sample["source_root"], str(data_root))
            self.assertTrue(sample["group_id"])
            self.assertIn(sample["split"], {"train", "val", "test"})

            with np.load(str(npz_path), allow_pickle=True) as npz:
                for field in (
                    "image_paths",
                    "depth_paths",
                    "geometry",
                    "mask_targets",
                    "risk_indices",
                    "can_forward",
                    "must_stop",
                    "projection_box",
                    "risk_states",
                    "geometry_feature_names",
                    "image_size",
                    "splits",
                    "group_ids",
                    "front_box_stop_fraction",
                    "front_box_warning_fraction",
                    "front_box_clearance_fraction",
                ):
                    self.assertIn(field, npz.files)
                self.assertEqual(tuple(npz["mask_targets"].shape), (1, 3, 100, 100))
                self.assertEqual(int(npz["risk_indices"][0]), 3)
                self.assertTrue(bool(npz["must_stop"][0]))
                self.assertGreater(float(npz["front_box_stop_fraction"][0]), 0.01)


class ObstacleRepresentation3ModelDemoTests(unittest.TestCase):
    def _require_torch(self):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        return torch

    def _require_pillow(self):
        try:
            from PIL import Image
        except Exception as exc:
            self.skipTest(f"Pillow unavailable: {exc}")
        return Image

    def test_a_plus_3_affordance_net_forward_shapes(self) -> None:
        torch = self._require_torch()
        from .model import APlus3AffordanceNet

        model = APlus3AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(RISK_STATES))
        model.eval()
        rgb = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
        depth = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        geometry = torch.zeros((2, len(GEOMETRY_FEATURE_NAMES)), dtype=torch.float32)

        with torch.no_grad():
            output = model(rgb, depth, geometry)

        self.assertEqual(tuple(output["mask_logits"].shape), (2, 3, 32, 32))
        self.assertEqual(tuple(output["risk_logits"].shape), (2, len(RISK_STATES)))
        self.assertEqual(tuple(output["can_forward_logits"].shape), (2,))
        self.assertEqual(tuple(output["must_stop_logits"].shape), (2,))

    def _write_demo_inputs(self, tmp_path: Path) -> tuple[Path, Path]:
        Image = self._require_pillow()
        rgb_path = tmp_path / "rgb.png"
        depth_path = tmp_path / "depth.npy"
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 1] = 96
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        depth = np.full((32, 32), 800.0, dtype=np.float32)
        np.save(str(depth_path), depth)
        return rgb_path, depth_path

    def _write_synthetic_checkpoint(self, tmp_path: Path, *, force_raw_stop: bool = False, force_raw_clear: bool = False) -> Path:
        torch = self._require_torch()
        from .model import APlus3AffordanceNet

        model = APlus3AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(RISK_STATES))
        model.eval()
        if force_raw_stop or force_raw_clear:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.mask_head.bias.fill_(-10.0)
                if force_raw_stop:
                    model.risk_logits.bias[RISK_STATES.index("must_stop")] = 10.0
                    model.must_stop_head.bias.fill_(10.0)
                    model.can_forward_head.bias.fill_(-10.0)
                else:
                    model.risk_logits.bias[RISK_STATES.index("clear")] = 10.0
                    model.must_stop_head.bias.fill_(-10.0)
                    model.can_forward_head.bias.fill_(10.0)
        checkpoint_path = tmp_path / "a_plus_3_model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {
                    "model_version": "a_plus_3_v1",
                    "image_size": 32,
                    "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
                    "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
                    "risk_states": list(RISK_STATES),
                    "mask_channels": list(MASK_CHANNELS),
                    "projection_box": dict(PROJECTION_BOX),
                },
                "geometry_mean": np.zeros(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                "geometry_std": np.ones(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
            },
            checkpoint_path,
        )
        return checkpoint_path

    def test_demo_loads_synthetic_checkpoint_and_renders_overlay(self) -> None:
        Image = self._require_pillow()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rgb_path, depth_path = self._write_demo_inputs(tmp_path)
            checkpoint_path = self._write_synthetic_checkpoint(tmp_path, force_raw_clear=True)
            from .demo import predict_obstacle_representation_3, render_affordance_overlay

            prediction = predict_obstacle_representation_3(
                checkpoint_path,
                rgb_path,
                {"depth_npy_path": str(depth_path), "pointcloud_summary": {"front_min_depth_cm": 800.0}},
                device_name="cpu",
            )
            overlay = render_affordance_overlay(Image.open(rgb_path), prediction)

        self.assertIn(prediction["front_risk_state"], RISK_STATES)
        self.assertIn("projection_box", prediction)
        self.assertIn("front_box_stop_fraction", prediction)
        self.assertEqual(tuple(overlay.shape), (32, 32, 3))

    def test_demo_must_stop_requires_projection_box_stop_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rgb_path, depth_path = self._write_demo_inputs(tmp_path)
            checkpoint_path = self._write_synthetic_checkpoint(tmp_path, force_raw_stop=True)
            from .demo import predict_obstacle_representation_3

            prediction = predict_obstacle_representation_3(
                checkpoint_path,
                rgb_path,
                {"depth_npy_path": str(depth_path), "pointcloud_summary": {"front_min_depth_cm": 80.0}},
                device_name="cpu",
            )

        self.assertEqual(prediction["front_box_stop_fraction"], 0.0)
        self.assertEqual(prediction["front_risk_state"], "obstacle_warning")
        self.assertFalse(prediction["must_stop"])

    def test_demo_depth_projection_box_stop_overrides_model_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rgb_path, depth_path = self._write_demo_inputs(tmp_path)
            depth = np.full((32, 32), 800.0, dtype=np.float32)
            depth[13:18, 14:18] = 80.0
            np.save(str(depth_path), depth)
            checkpoint_path = self._write_synthetic_checkpoint(tmp_path, force_raw_clear=True)
            from .demo import predict_obstacle_representation_3

            prediction = predict_obstacle_representation_3(
                checkpoint_path,
                rgb_path,
                {"depth_npy_path": str(depth_path), "pointcloud_summary": {"front_min_depth_cm": 80.0}},
                device_name="cpu",
            )

        self.assertGreater(prediction["front_box_stop_fraction"], 0.01)
        self.assertEqual(prediction["front_risk_state"], "must_stop")
        self.assertTrue(prediction["must_stop"])
        self.assertIn("depth_projection_box_stop", prediction["reason"])

    def test_demo_depth_stop_outside_projection_box_stays_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rgb_path, depth_path = self._write_demo_inputs(tmp_path)
            depth = np.full((32, 32), 800.0, dtype=np.float32)
            depth[0:4, 0:4] = 80.0
            np.save(str(depth_path), depth)
            checkpoint_path = self._write_synthetic_checkpoint(tmp_path, force_raw_clear=True)
            from .demo import predict_obstacle_representation_3

            prediction = predict_obstacle_representation_3(
                checkpoint_path,
                rgb_path,
                {"depth_npy_path": str(depth_path), "pointcloud_summary": {"front_min_depth_cm": 80.0}},
                device_name="cpu",
            )

        self.assertEqual(prediction["front_box_stop_fraction"], 0.0)
        self.assertEqual(prediction["front_risk_state"], "obstacle_warning")
        self.assertFalse(prediction["must_stop"])


class ObstacleRepresentation3ValidationTests(unittest.TestCase):
    def _require_torch(self):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        return torch

    def _require_pillow(self):
        try:
            from PIL import Image
        except Exception as exc:
            self.skipTest(f"Pillow unavailable: {exc}")
        return Image

    def test_validation_uses_checkpoint_projection_box(self) -> None:
        torch = self._require_torch()
        Image = self._require_pillow()
        from .model import APlus3AffordanceNet
        from .validate import validate_model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.npy"
            Image.fromarray(np.full((32, 32, 3), 80, dtype=np.uint8), mode="RGB").save(rgb_path)
            np.save(str(depth_path), np.full((32, 32), 800.0, dtype=np.float32))

            mask_targets = np.zeros((1, 3, 32, 32), dtype=np.float32)
            mask_targets[0, 2, 7:10, 7:10] = 1.0
            dataset_path = root / "a_plus_3_dataset_latest.npz"
            np.savez_compressed(
                dataset_path,
                image_paths=np.asarray([str(rgb_path)], dtype=object),
                depth_paths=np.asarray([str(depth_path)], dtype=object),
                geometry=np.zeros((1, len(GEOMETRY_FEATURE_NAMES)), dtype=np.float32),
                mask_targets=mask_targets,
                risk_indices=np.asarray([RISK_STATES.index("must_stop")], dtype=np.int64),
                can_forward=np.asarray([False], dtype=bool),
                must_stop=np.asarray([True], dtype=bool),
                splits=np.asarray(["test"], dtype=object),
                group_ids=np.asarray(["g1"], dtype=object),
                risk_states=np.asarray(RISK_STATES, dtype=object),
                geometry_feature_names=np.asarray(GEOMETRY_FEATURE_NAMES, dtype=object),
                projection_box=np.asarray([0.42, 0.58, 0.38, 0.72, 0.01], dtype=np.float32),
                image_size=np.asarray([32], dtype=np.int64),
            )
            checkpoint_box = {"x0": 0.20, "x1": 0.60, "y0": 0.20, "y1": 0.60, "stop_fraction_threshold": 0.01}
            checkpoint_risk_states = ["clear", "clearance_warning", "must_stop", "obstacle_warning"]
            model = APlus3AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(RISK_STATES))
            model_path = root / "a_plus_3_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "model_version": "a_plus_3_v1",
                        "image_size": 32,
                        "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
                        "risk_states": checkpoint_risk_states,
                        "projection_box": checkpoint_box,
                    },
                    "geometry_mean": np.zeros(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                    "geometry_std": np.ones(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32),
                },
                model_path,
            )

            report = validate_model(dataset_path, model_path, report_path=root / "report.json", batch_size=1)

        self.assertEqual(report["projection_box"], checkpoint_box)
        self.assertEqual(report["risk_states"], checkpoint_risk_states)
        self.assertEqual(report["projection_box_stop_support_count"], 1)
        self.assertEqual(report["per_class"]["must_stop"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
