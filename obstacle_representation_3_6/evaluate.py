from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from . import MODEL_VERSION


def default_ensemble_paths(output_root: Path | str = Path("obstacle_representation_3_data")) -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "dataset": root / "datasets" / "a_plus_3_1_dataset_latest.npz",
        "risk_model": root / "models" / "a_plus_3_3_model.pt",
        "stop_model": root / "models" / "a_plus_3_4_model.pt",
        "report": root / "validation" / "a_plus_3_6_rule_ensemble_report.json",
    }


def choose_ensemble_prediction(*, risk_pred: int, stop_pred: int, stop_idx: int) -> int:
    return int(stop_idx) if int(stop_pred) == int(stop_idx) else int(risk_pred)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _load_checkpoint(path: Path, device: Any) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate_rule_ensemble(
    dataset_path: Path | str | None = None,
    *,
    risk_model_path: Path | str | None = None,
    stop_model_path: Path | str | None = None,
    output_root: Path | str = Path("obstacle_representation_3_data"),
    report_path: Path | str | None = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from obstacle_representation_3.model import APlus3AffordanceNet
    from obstacle_representation_3.schema import GEOMETRY_FEATURE_NAMES, RISK_STATES
    from obstacle_representation_3.train import APlus3Dataset, load_dataset_arrays
    from obstacle_representation_3.validate import _pred_box_stop_fractions

    paths = default_ensemble_paths(output_root)
    dataset = Path(dataset_path) if dataset_path else paths["dataset"]
    risk_model_file = Path(risk_model_path) if risk_model_path else paths["risk_model"]
    stop_model_file = Path(stop_model_path) if stop_model_path else paths["stop_model"]
    report = Path(report_path) if report_path else paths["report"]
    arrays = load_dataset_arrays(dataset)

    risk_checkpoint = _load_checkpoint(risk_model_file, "cpu")
    stop_checkpoint = _load_checkpoint(stop_model_file, "cpu")
    config = risk_checkpoint.get("config", {})
    image_size = int(config.get("image_size", 96))
    risk_states = list(config.get("risk_states", RISK_STATES))
    projection_box = dict(stop_checkpoint.get("config", {}).get("projection_box", config.get("projection_box", {})))
    stop_threshold = float(projection_box.get("stop_fraction_threshold", 0.01))
    stop_idx = risk_states.index("must_stop")
    warning_idx = risk_states.index("obstacle_warning")
    test_indices = np.where(arrays["splits"] == "test")[0]
    if test_indices.size == 0:
        test_indices = np.arange(arrays["image_paths"].shape[0])
    geometry_mean = np.asarray(risk_checkpoint.get("geometry_mean"), dtype=np.float32)
    geometry_std = np.asarray(risk_checkpoint.get("geometry_std"), dtype=np.float32)
    geometry_std = np.where(np.abs(geometry_std) < 1e-6, 1.0, geometry_std).astype(np.float32)
    ds = APlus3Dataset(arrays, test_indices, image_size=image_size, geometry_mean=geometry_mean, geometry_std=geometry_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    risk_model = APlus3AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(risk_states)).to(device)
    stop_model = APlus3AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(risk_states)).to(device)
    risk_model.load_state_dict(risk_checkpoint["model_state"])
    stop_model.load_state_dict(stop_checkpoint["model_state"])
    risk_model.eval()
    stop_model.eval()

    confusion = np.zeros((len(risk_states), len(risk_states)), dtype=np.int64)
    can_forward_errors = 0
    must_stop_count = 0
    must_stop_miss = 0
    latency_samples: List[float] = []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            geometry = batch["geometry"].to(device)
            target_risk = batch["risk"].cpu().numpy()
            target_must_stop = batch["must_stop"].cpu().numpy().astype(bool)
            target_can_forward = batch["can_forward"].cpu().numpy().astype(bool)
            start = time.perf_counter()
            risk_output = risk_model(rgb, depth, geometry)
            stop_output = stop_model(rgb, depth, geometry)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_samples.extend([(time.perf_counter() - start) * 1000.0 / max(1, rgb.shape[0])] * int(rgb.shape[0]))
            risk_pred = torch.argmax(risk_output["risk_logits"], dim=1).detach().cpu().numpy()
            stop_box = _pred_box_stop_fractions(torch.sigmoid(stop_output["mask_logits"]).detach().cpu().numpy(), projection_box)
            stop_raw = torch.argmax(stop_output["risk_logits"], dim=1).detach().cpu().numpy()
            stop_pred = np.where(stop_box > stop_threshold, stop_idx, np.where(stop_raw == stop_idx, warning_idx, stop_raw))
            pred = np.asarray(
                [choose_ensemble_prediction(risk_pred=int(r), stop_pred=int(s), stop_idx=stop_idx) for r, s in zip(risk_pred, stop_pred)],
                dtype=np.int64,
            )
            pred_can_forward = pred != stop_idx
            can_forward_errors += int(np.count_nonzero(pred_can_forward != target_can_forward))
            must_stop_count += int(np.count_nonzero(target_must_stop))
            must_stop_miss += int(np.count_nonzero(target_must_stop & (pred != stop_idx)))
            for true_idx, pred_idx in zip(target_risk, pred):
                confusion[int(true_idx), int(pred_idx)] += 1

    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    macro_f1, per_class = _macro_f1(confusion, risk_states)
    report_data = {
        "model_version": MODEL_VERSION,
        "dataset_path": str(dataset),
        "risk_model_path": str(risk_model_file),
        "stop_model_path": str(stop_model_file),
        "test_count": int(confusion.sum()),
        "risk_accuracy": accuracy,
        "macro_f1": macro_f1,
        "mask_iou": None,
        "shielded_must_stop_miss_count": int(must_stop_miss),
        "shielded_must_stop_miss_rate": float(must_stop_miss / max(1, must_stop_count)),
        "must_stop_count": int(must_stop_count),
        "can_forward_error_count": int(can_forward_errors),
        "can_forward_error_rate": float(can_forward_errors / max(1, confusion.sum())),
        "per_class": per_class,
        "confusion_matrix": confusion.astype(int).tolist(),
        "risk_states": risk_states,
        "projection_box": projection_box,
        "projection_box_stop_threshold": stop_threshold,
        "latency_ms_mean": float(np.mean(latency_samples)) if latency_samples else 0.0,
        "latency_ms_p95": float(np.percentile(latency_samples, 95)) if latency_samples else 0.0,
        "device": str(device),
    }
    write_json(report, report_data)
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OR3.6 risk/stop rule ensemble.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--risk-model", default=None)
    parser.add_argument("--stop-model", default=None)
    parser.add_argument("--output-root", default="obstacle_representation_3_data")
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = evaluate_rule_ensemble(
        Path(args.dataset) if args.dataset else None,
        risk_model_path=Path(args.risk_model) if args.risk_model else None,
        stop_model_path=Path(args.stop_model) if args.stop_model else None,
        output_root=Path(args.output_root),
        report_path=Path(args.report) if args.report else None,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
