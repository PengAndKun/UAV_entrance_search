from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import APlus2AffordanceNet
from .schema import DIRECTION_LABELS
from .train import APlus2Dataset, load_dataset_arrays, write_json


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _macro_f1(confusion: np.ndarray) -> tuple[float, Dict[str, Dict[str, float]]]:
    per_class: Dict[str, Dict[str, float]] = {}
    values: List[float] = []
    for idx, label in enumerate(DIRECTION_LABELS):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - tp)
        fn = float(confusion[idx, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        support = int(confusion[idx, :].sum())
        if support > 0:
            values.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return (float(np.mean(values)) if values else 0.0), per_class


def validate_model(
    dataset_path: Path,
    model_path: Path,
    *,
    report_path: Path | None = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    report_path = report_path or dataset_path.parents[1] / "validation" / "a_plus_2_validation_report.json"
    arrays = load_dataset_arrays(dataset_path)
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 96))
    geometry_mean = np.asarray(checkpoint.get("geometry_mean"), dtype=np.float32)
    geometry_std = np.asarray(checkpoint.get("geometry_std"), dtype=np.float32)
    geometry_std = np.where(np.abs(geometry_std) < 1e-6, 1.0, geometry_std).astype(np.float32)

    test_indices = np.where(arrays["splits"] == "test")[0]
    if test_indices.size == 0:
        test_indices = np.where(arrays["splits"] == "val")[0]
    if test_indices.size == 0:
        test_indices = np.arange(arrays["image_paths"].shape[0])
    ds = APlus2Dataset(arrays, test_indices, image_size=image_size, geometry_mean=geometry_mean, geometry_std=geometry_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APlus2AffordanceNet(geometry_dim=int(config.get("geometry_dim", geometry_mean.shape[0])), num_directions=len(DIRECTION_LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    confusion = np.zeros((len(DIRECTION_LABELS), len(DIRECTION_LABELS)), dtype=np.int64)
    mask_iou_sum = 0.0
    count = 0
    forward_danger_violations = 0
    raw_forward_danger_violations = 0
    red_blocked_count = 0
    up_needed_count = 0
    up_correct_count = 0
    latency_samples: List[float] = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            geometry = batch["geometry"].to(device)
            target_mask = batch["mask"].to(device) >= 0.5
            target_direction = batch["direction"].cpu().numpy()
            start = time.perf_counter()
            output = model(rgb, depth, geometry)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latency_samples.extend([elapsed_ms / max(1, rgb.shape[0])] * int(rgb.shape[0]))
            raw_pred_direction = torch.argmax(output["direction_logits"], dim=1).detach().cpu().numpy()
            score_prob = torch.sigmoid(output["score_logits"]).detach().cpu().numpy()
            pred_direction = raw_pred_direction.copy()
            pred_mask = (torch.sigmoid(output["mask_logits"]) >= 0.5).detach()
            intersection = torch.logical_and(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
            union = torch.logical_or(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
            iou = torch.where(union > 0, intersection / torch.clamp(union, min=1.0), torch.ones_like(union))
            mask_iou_sum += float(iou.sum().item())
            batch_indices = test_indices[offset : offset + int(rgb.shape[0])]
            offset += int(rgb.shape[0])
            red_blocked = arrays["red_front_blocked"][batch_indices]
            red_blocked_count += int(np.count_nonzero(red_blocked))
            forward_idx = DIRECTION_LABELS.index("forward")
            up_idx = DIRECTION_LABELS.index("up")
            raw_forward_danger_violations += int(np.count_nonzero(red_blocked & (raw_pred_direction == forward_idx)))
            for row_idx, blocked in enumerate(red_blocked):
                if bool(blocked) and int(pred_direction[row_idx]) == forward_idx:
                    fallback_scores = score_prob[row_idx].copy()
                    fallback_scores[forward_idx] = -1.0
                    pred_direction[row_idx] = int(np.argmax(fallback_scores))
            forward_danger_violations += int(np.count_nonzero(red_blocked & (pred_direction == forward_idx)))
            up_teacher = target_direction == up_idx
            up_needed_count += int(np.count_nonzero(up_teacher))
            up_correct_count += int(np.count_nonzero(up_teacher & (pred_direction == up_idx)))
            for true_idx, pred_idx in zip(target_direction, pred_direction):
                confusion[int(true_idx), int(pred_idx)] += 1
            count += int(rgb.shape[0])

    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    macro_f1, per_class = _macro_f1(confusion)
    report = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "model_version": str(config.get("model_version", "a_plus_2_v1")),
        "test_count": int(count),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mask_iou": float(mask_iou_sum / max(1, count)),
        "raw_forward_danger_violation_count": int(raw_forward_danger_violations),
        "raw_forward_danger_violation_rate": float(raw_forward_danger_violations / max(1, red_blocked_count)),
        "forward_danger_violation_count": int(forward_danger_violations),
        "forward_danger_violation_rate": float(forward_danger_violations / max(1, red_blocked_count)),
        "red_front_blocked_count": int(red_blocked_count),
        "up_recall_when_front_blocked": float(up_correct_count / max(1, up_needed_count)),
        "up_needed_count": int(up_needed_count),
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
        "direction_labels": list(DIRECTION_LABELS),
        "latency_ms_mean": float(np.mean(latency_samples)) if latency_samples else 0.0,
        "latency_ms_p95": float(np.percentile(latency_samples, 95)) if latency_samples else 0.0,
        "device": str(device),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Obstacle Representation 2 A+2 model.")
    parser.add_argument("--dataset", default="obstacle_representation_2_data/datasets/a_plus_2_dataset_latest.npz")
    parser.add_argument("--model", default="obstacle_representation_2_data/models/a_plus_2_model.pt")
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = validate_model(
        Path(args.dataset),
        Path(args.model),
        report_path=Path(args.report) if args.report else None,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
