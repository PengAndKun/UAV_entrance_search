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

from .model import APlus3AffordanceNet
from .schema import PROJECTION_BOX, RISK_STATES
from .teacher import projection_box_stats
from .train import APlus3Dataset, load_dataset_arrays, write_json


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _macro_f1(confusion: np.ndarray, risk_states: List[str]) -> tuple[float, Dict[str, Dict[str, float]]]:
    per_class: Dict[str, Dict[str, float]] = {}
    values: List[float] = []
    for idx, label in enumerate(risk_states):
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


def _projection_box_from_array(value: np.ndarray | None) -> Dict[str, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 5:
        return None
    return {
        "x0": float(arr[0]),
        "x1": float(arr[1]),
        "y0": float(arr[2]),
        "y1": float(arr[3]),
        "stop_fraction_threshold": float(arr[4]),
    }


def _projection_boxes_match(left: Dict[str, Any] | None, right: Dict[str, Any] | None) -> bool:
    if not left or not right:
        return False
    for key in ("x0", "x1", "y0", "y1", "stop_fraction_threshold"):
        if abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) > 1e-6:
            return False
    return True


def _labels_from_config(value: Any, fallback: tuple[str, ...] = RISK_STATES) -> List[str]:
    if isinstance(value, np.ndarray):
        labels = [str(item) for item in value.astype(str).tolist()]
    elif isinstance(value, (list, tuple)):
        labels = [str(item) for item in value]
    else:
        labels = []
    return labels if labels else list(fallback)


def _remap_risk_indices(indices: np.ndarray, source_labels: List[str], target_labels: List[str]) -> np.ndarray:
    if source_labels == target_labels:
        return indices.astype(np.int64, copy=False)
    mapped = np.zeros(indices.shape, dtype=np.int64)
    for idx, value in enumerate(indices.astype(np.int64, copy=False)):
        label = source_labels[int(value)] if 0 <= int(value) < len(source_labels) else "clear"
        mapped[idx] = int(target_labels.index(label)) if label in target_labels else 0
    return mapped


def _target_projection_support(
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    threshold: float,
    projection_box: Dict[str, float],
) -> int:
    dataset_box = _projection_box_from_array(arrays.get("projection_box"))
    if "front_box_stop_fraction" in arrays and _projection_boxes_match(dataset_box, projection_box):
        return int(np.count_nonzero(np.asarray(arrays["front_box_stop_fraction"])[indices] > threshold))
    support = 0
    for idx in indices:
        stats = projection_box_stats(np.asarray(arrays["mask_targets"][int(idx)], dtype=np.float32), projection_box)
        support += int(float(stats["front_box_stop_fraction"]) > threshold)
    return int(support)


def _pred_box_stop_fractions(mask_prob: np.ndarray, projection_box: Dict[str, float]) -> np.ndarray:
    values = []
    for masks in mask_prob:
        stats = projection_box_stats(np.asarray(masks, dtype=np.float32), projection_box)
        values.append(float(stats["front_box_stop_fraction"]))
    return np.asarray(values, dtype=np.float32)


def validate_model(
    dataset_path: Path,
    model_path: Path,
    *,
    report_path: Path | None = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    report_path = report_path or dataset_path.parents[1] / "validation" / "a_plus_3_validation_report.json"
    arrays = load_dataset_arrays(dataset_path)
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 96))
    risk_states = _labels_from_config(config.get("risk_states"), RISK_STATES)
    dataset_risk_states = _labels_from_config(arrays.get("risk_states"), RISK_STATES)
    if dataset_risk_states != risk_states:
        arrays = dict(arrays)
        arrays["risk_indices"] = _remap_risk_indices(arrays["risk_indices"], dataset_risk_states, risk_states)
    geometry_mean = np.asarray(checkpoint.get("geometry_mean"), dtype=np.float32)
    geometry_std = np.asarray(checkpoint.get("geometry_std"), dtype=np.float32)
    geometry_std = np.where(np.abs(geometry_std) < 1e-6, 1.0, geometry_std).astype(np.float32)
    projection_box = dict(config.get("projection_box", PROJECTION_BOX))
    stop_threshold = float(projection_box.get("stop_fraction_threshold", PROJECTION_BOX["stop_fraction_threshold"]))

    test_indices = np.where(arrays["splits"] == "test")[0]
    if test_indices.size == 0:
        test_indices = np.where(arrays["splits"] == "val")[0]
    if test_indices.size == 0:
        test_indices = np.arange(arrays["image_paths"].shape[0])
    ds = APlus3Dataset(arrays, test_indices, image_size=image_size, geometry_mean=geometry_mean, geometry_std=geometry_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APlus3AffordanceNet(
        geometry_dim=int(config.get("geometry_dim", geometry_mean.shape[0])),
        num_risk_states=len(risk_states),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    confusion = np.zeros((len(risk_states), len(risk_states)), dtype=np.int64)
    mask_iou_sum = 0.0
    count = 0
    raw_must_stop_miss_count = 0
    shielded_must_stop_miss_count = 0
    must_stop_count = 0
    can_forward_errors = 0
    latency_samples: List[float] = []
    stop_idx = risk_states.index("must_stop")
    warning_idx = risk_states.index("obstacle_warning")
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            geometry = batch["geometry"].to(device)
            target_mask = batch["mask"].to(device) >= 0.5
            target_risk = batch["risk"].cpu().numpy()
            target_must_stop = batch["must_stop"].cpu().numpy().astype(bool)
            target_can_forward = batch["can_forward"].cpu().numpy().astype(bool)
            start = time.perf_counter()
            output = model(rgb, depth, geometry)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latency_samples.extend([elapsed_ms / max(1, rgb.shape[0])] * int(rgb.shape[0]))
            raw_pred_risk = torch.argmax(output["risk_logits"], dim=1).detach().cpu().numpy()
            raw_pred_must_stop = torch.sigmoid(output["must_stop_logits"]).detach().cpu().numpy() >= 0.5
            pred_can_forward = torch.sigmoid(output["can_forward_logits"]).detach().cpu().numpy() >= 0.5
            mask_prob = torch.sigmoid(output["mask_logits"]).detach()
            pred_mask = mask_prob >= 0.5
            intersection = torch.logical_and(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
            union = torch.logical_or(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
            iou = torch.where(union > 0, intersection / torch.clamp(union, min=1.0), torch.ones_like(union))
            mask_iou_sum += float(iou.sum().item())

            pred_box_stop = _pred_box_stop_fractions(mask_prob.cpu().numpy(), projection_box)
            shielded_pred_must_stop = pred_box_stop > stop_threshold
            raw_must_stop_miss_count += int(np.count_nonzero(target_must_stop & (raw_pred_risk != stop_idx) & ~raw_pred_must_stop))
            pred_risk = raw_pred_risk.copy()
            pred_risk[raw_pred_must_stop | (raw_pred_risk == stop_idx)] = stop_idx
            pred_risk[(pred_risk == stop_idx) & ~shielded_pred_must_stop] = warning_idx
            shielded_must_stop_miss_count += int(np.count_nonzero(target_must_stop & (pred_risk != stop_idx)))
            must_stop_count += int(np.count_nonzero(target_must_stop))
            can_forward_errors += int(np.count_nonzero(pred_can_forward != target_can_forward))
            for true_idx, pred_idx in zip(target_risk, pred_risk):
                confusion[int(true_idx), int(pred_idx)] += 1
            count += int(rgb.shape[0])

    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    macro_f1, per_class = _macro_f1(confusion, risk_states)
    report = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "model_version": str(config.get("model_version", "a_plus_3_v1")),
        "test_count": int(count),
        "risk_accuracy": accuracy,
        "macro_f1": macro_f1,
        "mask_iou": float(mask_iou_sum / max(1, count)),
        "raw_must_stop_miss_count": int(raw_must_stop_miss_count),
        "raw_must_stop_miss_rate": float(raw_must_stop_miss_count / max(1, must_stop_count)),
        "shielded_must_stop_miss_count": int(shielded_must_stop_miss_count),
        "shielded_must_stop_miss_rate": float(shielded_must_stop_miss_count / max(1, must_stop_count)),
        "must_stop_count": int(must_stop_count),
        "can_forward_error_count": int(can_forward_errors),
        "can_forward_error_rate": float(can_forward_errors / max(1, count)),
        "projection_box": projection_box,
        "projection_box_stop_support_count": _target_projection_support(arrays, test_indices, stop_threshold, projection_box),
        "projection_box_stop_threshold": stop_threshold,
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
        "risk_states": list(risk_states),
        "latency_ms_mean": float(np.mean(latency_samples)) if latency_samples else 0.0,
        "latency_ms_p95": float(np.percentile(latency_samples, 95)) if latency_samples else 0.0,
        "device": str(device),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Obstacle Representation 3 A+3 risk model.")
    parser.add_argument("--dataset", default="obstacle_representation_3_data/datasets/a_plus_3_dataset_latest.npz")
    parser.add_argument("--model", default="obstacle_representation_3_data/models/a_plus_3_model.pt")
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
