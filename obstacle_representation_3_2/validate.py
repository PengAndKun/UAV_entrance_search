from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def default_validation_paths(output_root: Path | str = Path("obstacle_representation_3_data")) -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "dataset": root / "datasets" / "a_plus_3_1_dataset_latest.npz",
        "model": root / "models" / "a_plus_3_2_model.pt",
        "report": root / "validation" / "a_plus_3_2_validation_report.json",
        "comparison": root / "validation" / "a_plus_3_1_vs_3_2_comparison.json",
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_model_3_2(
    dataset_path: Path | str | None = None,
    model_path: Path | str | None = None,
    *,
    output_root: Path | str = Path("obstacle_representation_3_data"),
    report_path: Path | str | None = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    from obstacle_representation_3.validate import validate_model

    paths = default_validation_paths(output_root)
    dataset = Path(dataset_path) if dataset_path else paths["dataset"]
    model = Path(model_path) if model_path else paths["model"]
    report = Path(report_path) if report_path else paths["report"]
    return validate_model(dataset, model, report_path=report, batch_size=batch_size)


def compare_reports(
    baseline_report_path: Path | str,
    candidate_report_path: Path | str,
    *,
    output_path: Path | str,
    baseline_label: str = "baseline",
    candidate_label: str = "or3_2",
) -> Dict[str, Any]:
    baseline = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_report_path).read_text(encoding="utf-8"))
    metrics = ["risk_accuracy", "macro_f1", "mask_iou", "shielded_must_stop_miss_rate", "can_forward_error_rate"]
    deltas = {metric: float(candidate.get(metric, 0.0)) - float(baseline.get(metric, 0.0)) for metric in metrics}
    comparison = {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_report_path": str(baseline_report_path),
        "candidate_report_path": str(candidate_report_path),
        "baseline_model_path": baseline.get("model_path", ""),
        "candidate_model_path": candidate.get("model_path", ""),
        "metrics_compared": metrics,
        baseline_label: {metric: baseline.get(metric) for metric in metrics},
        candidate_label: {metric: candidate.get(metric) for metric in metrics},
        f"delta_{candidate_label}_minus_{baseline_label}": deltas,
        f"{baseline_label}_must_stop": baseline.get("per_class", {}).get("must_stop", {}),
        f"{candidate_label}_must_stop": candidate.get("per_class", {}).get("must_stop", {}),
        "risk_accuracy_improved": bool(deltas["risk_accuracy"] > 0.0),
        "macro_f1_improved": bool(deltas["macro_f1"] > 0.0),
        "mask_iou_improved": bool(deltas["mask_iou"] > 0.0),
        "must_stop_miss_rate_improved": bool(deltas["shielded_must_stop_miss_rate"] < 0.0),
    }
    write_json(Path(output_path), comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Obstacle Representation 3.2 model.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-root", default="obstacle_representation_3_data")
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = validate_model_3_2(
        Path(args.dataset) if args.dataset else None,
        Path(args.model) if args.model else None,
        output_root=Path(args.output_root),
        report_path=Path(args.report) if args.report else None,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
