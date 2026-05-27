from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def default_validation_paths(output_root: Path | str = Path("obstacle_representation_3_data")) -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "dataset": root / "datasets" / "a_plus_3_1_dataset_latest.npz",
        "model": root / "models" / "a_plus_3_1_model.pt",
        "report": root / "validation" / "a_plus_3_1_validation_report.json",
        "comparison": root / "validation" / "a_plus_3_vs_3_1_comparison.json",
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_model_3_1(
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
    or3_report_path: Path | str = Path("obstacle_representation_3_data/validation/a_plus_3_validation_report.json"),
    or31_report_path: Path | str = Path("obstacle_representation_3_data/validation/a_plus_3_1_validation_report.json"),
    *,
    output_path: Path | str = Path("obstacle_representation_3_data/validation/a_plus_3_vs_3_1_comparison.json"),
) -> Dict[str, Any]:
    or3_report = json.loads(Path(or3_report_path).read_text(encoding="utf-8"))
    or31_report = json.loads(Path(or31_report_path).read_text(encoding="utf-8"))
    metrics = ["risk_accuracy", "macro_f1", "mask_iou", "shielded_must_stop_miss_rate", "can_forward_error_rate"]
    deltas = {}
    for metric in metrics:
        deltas[metric] = float(or31_report.get(metric, 0.0)) - float(or3_report.get(metric, 0.0))
    comparison = {
        "or3_report_path": str(or3_report_path),
        "or3_1_report_path": str(or31_report_path),
        "or3_model_path": or3_report.get("model_path", ""),
        "or3_1_model_path": or31_report.get("model_path", ""),
        "metrics_compared": metrics,
        "or3": {metric: or3_report.get(metric) for metric in metrics},
        "or3_1": {metric: or31_report.get(metric) for metric in metrics},
        "delta_or3_1_minus_or3": deltas,
        "or3_must_stop": or3_report.get("per_class", {}).get("must_stop", {}),
        "or3_1_must_stop": or31_report.get("per_class", {}).get("must_stop", {}),
        "accuracy_improved": bool(deltas["risk_accuracy"] > 0.0),
        "macro_f1_improved": bool(deltas["macro_f1"] > 0.0),
        "mask_iou_improved": bool(deltas["mask_iou"] > 0.0),
    }
    write_json(Path(output_path), comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Obstacle Representation 3.1 model.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-root", default="obstacle_representation_3_data")
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--or3-report", default="obstacle_representation_3_data/validation/a_plus_3_validation_report.json")
    parser.add_argument("--comparison", default=None)
    args = parser.parse_args()
    report = validate_model_3_1(
        Path(args.dataset) if args.dataset else None,
        Path(args.model) if args.model else None,
        output_root=Path(args.output_root),
        report_path=Path(args.report) if args.report else None,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.compare:
        paths = default_validation_paths(Path(args.output_root))
        comparison = compare_reports(
            Path(args.or3_report),
            Path(args.report) if args.report else paths["report"],
            output_path=Path(args.comparison) if args.comparison else paths["comparison"],
        )
        print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
