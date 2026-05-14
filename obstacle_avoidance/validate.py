from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset import label_counts, write_json
from .train import accuracy, predict_centroids


def confusion_counts(pred: np.ndarray, truth: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p, t in zip(pred.astype(str).tolist(), truth.astype(str).tolist()):
        key = f"{t}->{p}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def validate_model(
    dataset_path: Path,
    model_path: Path,
    *,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    report_path = report_path or dataset_path.parents[1] / "validation" / "validation_report.json"
    data = np.load(dataset_path, allow_pickle=True)
    model = np.load(model_path, allow_pickle=True)
    x = data["X"].astype(np.float32, copy=False)
    y_risk = data["y_risk"].astype(str)
    y_action = data["y_action"].astype(str)
    y_collision = data["y_collision"].astype(str)
    refs = data["event_refs"].astype(str) if "event_refs" in data.files else np.asarray([""] * x.shape[0])
    if x.shape[0] <= 0:
        raise ValueError(f"dataset has no samples: {dataset_path}")

    x_norm = (x - model["mean"].astype(np.float32)) / model["std"].astype(np.float32)
    risk_pred = predict_centroids(x_norm, model["risk_classes"].astype(str), model["risk_centroids"].astype(np.float32))
    action_pred = predict_centroids(x_norm, model["action_classes"].astype(str), model["action_centroids"].astype(np.float32))
    collision_pred = predict_centroids(
        x_norm,
        model["collision_classes"].astype(str),
        model["collision_centroids"].astype(np.float32),
    )

    feature_names = data["feature_names"].astype(str).tolist()
    front_idx = feature_names.index("depth_front_min_cm") if "depth_front_min_cm" in feature_names else -1
    pc_front_idx = feature_names.index("pc_front_min_cm") if "pc_front_min_cm" in feature_names else -1
    action_classes = model["action_classes"].astype(str).tolist()
    templates = model["action_templates"].astype(np.float32)
    danger_depth_cm = float(model["danger_depth_cm"][0]) if "danger_depth_cm" in model.files else 250.0
    danger_forward_violations: List[Dict[str, Any]] = []
    failure_examples: List[Dict[str, Any]] = []
    for idx in range(x.shape[0]):
        depths = []
        if front_idx >= 0 and x[idx, front_idx] > 0:
            depths.append(float(x[idx, front_idx]))
        if pc_front_idx >= 0 and x[idx, pc_front_idx] > 0:
            depths.append(float(x[idx, pc_front_idx]))
        min_depth = min(depths) if depths else 0.0
        try:
            action_index = action_classes.index(str(action_pred[idx]))
            forward_cm = float(templates[action_index, 0])
        except Exception:
            forward_cm = 0.0
        if min_depth > 0.0 and min_depth < danger_depth_cm and forward_cm > 1.0:
            danger_forward_violations.append(
                {
                    "row": idx,
                    "min_depth_cm": round(min_depth, 3),
                    "predicted_action": str(action_pred[idx]),
                    "predicted_forward_cm": round(forward_cm, 3),
                    "event_ref": refs[idx],
                }
            )
        if str(y_collision[idx]) == "1" and str(collision_pred[idx]) != "1":
            failure_examples.append(
                {
                    "row": idx,
                    "reason": "collision_predicted_non_collision",
                    "predicted_collision": str(collision_pred[idx]),
                    "event_ref": refs[idx],
                }
            )

    report = {
        "status": "ok",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "report_path": str(report_path),
        "sample_count": int(x.shape[0]),
        "risk_state_accuracy": round(accuracy(risk_pred, y_risk), 4),
        "expert_action_accuracy": round(accuracy(action_pred, y_action), 4),
        "collision_failure_accuracy": round(accuracy(collision_pred, y_collision), 4),
        "danger_forward_violation_count": len(danger_forward_violations),
        "danger_forward_violations": danger_forward_violations[:25],
        "failure_examples": failure_examples[:25],
        "risk_state_counts": label_counts(y_risk),
        "expert_action_counts": label_counts(y_action),
        "collision_counts": label_counts(y_collision),
        "risk_confusion": confusion_counts(risk_pred, y_risk),
        "action_confusion": confusion_counts(action_pred, y_action),
        "collision_confusion": confusion_counts(collision_pred, y_collision),
        "validated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate obstacle avoidance baseline model.")
    parser.add_argument("--dataset", default="obstacle_avoidance_data/datasets/dataset_latest.npz")
    parser.add_argument("--model", default="obstacle_avoidance_data/models/avoidance_agent_v0.npz")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = validate_model(Path(args.dataset), Path(args.model), report_path=Path(args.report) if args.report else None)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
