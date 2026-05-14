from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .dataset import label_counts, write_json


def fit_centroids(x: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(sorted({str(item) for item in labels.tolist()}))
    if classes.size == 0:
        return np.asarray(["UNKNOWN"]), np.zeros((1, x.shape[1]), dtype=np.float32)
    centroids = []
    for label in classes:
        mask = labels.astype(str) == str(label)
        centroids.append(np.mean(x[mask], axis=0) if np.any(mask) else np.zeros((x.shape[1],), dtype=np.float32))
    return classes, np.vstack(centroids).astype(np.float32, copy=False)


def predict_centroids(x: np.ndarray, classes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.asarray([], dtype=str)
    diff = x[:, None, :] - centroids[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    return classes[np.argmin(distances, axis=1)]


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    if truth.size == 0:
        return 0.0
    return float(np.mean(pred.astype(str) == truth.astype(str)))


def train_baseline(
    dataset_path: Path,
    *,
    model_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    model_path = model_path or dataset_path.parents[1] / "models" / "avoidance_agent_v0.npz"
    summary_path = summary_path or model_path.with_name("model_summary.json")
    data = np.load(dataset_path, allow_pickle=True)
    x = data["X"].astype(np.float32, copy=False)
    if x.shape[0] <= 0:
        raise ValueError(f"dataset has no samples: {dataset_path}")
    y_risk = data["y_risk"].astype(str)
    y_action = data["y_action"].astype(str)
    y_collision = data["y_collision"].astype(str)
    y_action_vector = data["y_action_vector"].astype(np.float32, copy=False)
    feature_names = data["feature_names"].astype(str)

    order = np.arange(x.shape[0])
    val_mask = (order % 5 == 0) if x.shape[0] >= 5 else np.zeros_like(order, dtype=bool)
    train_mask = ~val_mask
    if not np.any(train_mask):
        train_mask[:] = True
        val_mask[:] = False
    mean = np.mean(x[train_mask], axis=0)
    std = np.std(x[train_mask], axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_norm = (x - mean) / std

    risk_classes, risk_centroids = fit_centroids(x_norm[train_mask], y_risk[train_mask])
    action_classes, action_centroids = fit_centroids(x_norm[train_mask], y_action[train_mask])
    collision_classes, collision_centroids = fit_centroids(x_norm[train_mask], y_collision[train_mask])

    action_templates = []
    for label in action_classes:
        mask = y_action.astype(str) == str(label)
        action_templates.append(
            np.mean(y_action_vector[mask], axis=0) if np.any(mask) else np.zeros((4,), dtype=np.float32)
        )
    action_templates_array = np.vstack(action_templates).astype(np.float32, copy=False)

    eval_mask = val_mask if np.any(val_mask) else train_mask
    risk_pred = predict_centroids(x_norm[eval_mask], risk_classes, risk_centroids)
    action_pred = predict_centroids(x_norm[eval_mask], action_classes, action_centroids)
    collision_pred = predict_centroids(x_norm[eval_mask], collision_classes, collision_centroids)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        model_type=np.asarray(["nearest_centroid_v0"]),
        feature_names=feature_names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        risk_classes=risk_classes,
        risk_centroids=risk_centroids,
        action_classes=action_classes,
        action_centroids=action_centroids,
        collision_classes=collision_classes,
        collision_centroids=collision_centroids,
        action_templates=action_templates_array,
        action_vector_names=np.asarray(["forward_cm", "right_cm", "up_cm", "yaw_delta_deg"]),
        danger_depth_cm=np.asarray([250.0], dtype=np.float32),
        trained_at=np.asarray([datetime.now().isoformat(timespec="seconds")]),
    )

    summary = {
        "status": "ok",
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "dataset_path": str(dataset_path),
        "sample_count": int(x.shape[0]),
        "train_count": int(np.sum(train_mask)),
        "validation_count": int(np.sum(eval_mask)),
        "feature_count": int(x.shape[1]),
        "risk_state_accuracy": round(accuracy(risk_pred, y_risk[eval_mask]), 4),
        "expert_action_accuracy": round(accuracy(action_pred, y_action[eval_mask]), 4),
        "collision_failure_accuracy": round(accuracy(collision_pred, y_collision[eval_mask]), 4),
        "risk_state_counts": label_counts(y_risk),
        "expert_action_counts": label_counts(y_action),
        "collision_counts": label_counts(y_collision),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train obstacle avoidance nearest-centroid baseline.")
    parser.add_argument("--dataset", default="obstacle_avoidance_data/datasets/dataset_latest.npz")
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    summary = train_baseline(Path(args.dataset), model_path=Path(args.model) if args.model else None)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
