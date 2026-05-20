from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import SchemeAObstacleNet, SchemeAPlusObstacleNet
from .schema import GEOMETRY_FEATURE_NAMES, OBSTACLE_LABELS
from .train import SchemeADataset, load_dataset_arrays, write_json


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> np.ndarray:
    matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= true < num_labels and 0 <= pred < num_labels:
            matrix[true, pred] += 1
    return matrix


def classification_metrics(matrix: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class = {}
    f1_values = []
    for idx, label in enumerate(labels):
        tp = float(matrix[idx, idx])
        support = float(matrix[idx, :].sum())
        predicted = float(matrix[:, idx].sum())
        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted if predicted > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[label] = {
            "support": int(support),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        if support > 0:
            f1_values.append(f1)
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(float(np.mean(f1_values)), 6) if f1_values else 0.0,
        "per_class": per_class,
    }


def validate_model(
    dataset_path: Path,
    model_path: Path,
    *,
    report_path: Path | None = None,
    split: str = "test",
    batch_size: int = 32,
) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]
    labels = list(config.get("labels", OBSTACLE_LABELS))
    image_size = int(config.get("image_size", 96))
    use_depth = bool(config.get("use_depth", False))
    model_version = str(config.get("model_version", "scheme_a_v1"))
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    arrays = load_dataset_arrays(dataset_path)
    indices = np.where(arrays["splits"] == split)[0]
    if indices.size == 0 and split == "test":
        indices = np.where(arrays["splits"] == "val")[0]
        split = "val"
    if indices.size == 0:
        indices = np.arange(arrays["label_indices"].shape[0])
        split = "all"
    dataset = SchemeADataset(
        arrays["image_paths"],
        arrays["geometry"],
        arrays["label_indices"],
        arrays["flyover"],
        indices,
        depth_paths=arrays["depth_paths"],
        sample_weights=arrays["sample_weights"],
        use_depth=use_depth,
        image_size=image_size,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_depth:
        model = SchemeAPlusObstacleNet(num_labels=len(labels), geometry_dim=len(GEOMETRY_FEATURE_NAMES)).to(device)
    else:
        model = SchemeAObstacleNet(num_labels=len(labels), geometry_dim=len(GEOMETRY_FEATURE_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    fly_true: List[float] = []
    fly_pred: List[float] = []
    with torch.no_grad():
        for batch in loader:
            if use_depth:
                out = model(batch["rgb"].to(device), batch["geometry"].to(device), batch["depth"].to(device))
            else:
                out = model(batch["rgb"].to(device), batch["geometry"].to(device))
            y_true.extend(batch["label"].numpy().astype(int).tolist())
            y_pred.extend(torch.argmax(out["label_logits"], dim=1).cpu().numpy().astype(int).tolist())
            fly_true.extend(batch["flyover"].numpy().astype(float).tolist())
            fly_pred.extend((torch.sigmoid(out["flyover_logits"]).cpu().numpy() >= 0.5).astype(float).tolist())

    matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), len(labels))
    metrics = classification_metrics(matrix, labels)
    flyover_accuracy = float(np.mean(np.asarray(fly_true) == np.asarray(fly_pred))) if fly_true else 0.0

    latency_ms_mean = 0.0
    if len(dataset) > 0:
        latency_loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset), 32), shuffle=False, num_workers=0)
        latency_batch = next(iter(latency_loader))
        rgb = latency_batch["rgb"].to(device)
        geometry = latency_batch["geometry"].to(device)
        depth = latency_batch["depth"].to(device) if use_depth else None
        with torch.no_grad():
            for _ in range(3):
                _ = model(rgb, geometry, depth) if use_depth else model(rgb, geometry)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            runs = 20
            for _ in range(runs):
                _ = model(rgb, geometry, depth) if use_depth else model(rgb, geometry)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        latency_ms_mean = elapsed * 1000.0 / (runs * max(1, int(rgb.shape[0])))

    report = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "model_version": model_version,
        "use_depth": bool(use_depth),
        "split": split,
        "sample_count": int(len(dataset)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "flyover_accuracy": round(float(flyover_accuracy), 6),
        "per_class_recall": {label: item["recall"] for label, item in metrics["per_class"].items()},
        "per_class": metrics["per_class"],
        "confusion_matrix": matrix.tolist(),
        "building_fence_confusion": {
            "building_as_fence_or_rail": int(matrix[labels.index("building"), labels.index("fence_or_rail")])
            if "building" in labels and "fence_or_rail" in labels
            else 0,
            "fence_or_rail_as_building": int(matrix[labels.index("fence_or_rail"), labels.index("building")])
            if "building" in labels and "fence_or_rail" in labels
            else 0,
        },
        "labels": labels,
        "latency_ms_mean": round(float(latency_ms_mean), 6),
        "device": str(device),
    }
    report_path = report_path or dataset_path.parents[1] / "validation" / "validation_report.json"
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Scheme A obstacle representation model.")
    parser.add_argument("--dataset", default="obstacle_representation_data/datasets/dataset_latest.npz")
    parser.add_argument("--model", default="obstacle_representation_data/models/scheme_a_model.pt")
    parser.add_argument("--report", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = validate_model(
        Path(args.dataset),
        Path(args.model),
        report_path=Path(args.report) if args.report else None,
        split=args.split,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
