from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def default_validation_paths(output_root: Path | str = Path("obstacle_representation_3_data")) -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "dataset": root / "datasets" / "a_plus_3_1_dataset_latest.npz",
        "model": root / "models" / "a_plus_3_5_model.pt",
        "report": root / "validation" / "a_plus_3_5_validation_report.json",
    }


def validate_model_3_5(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Obstacle Representation 3.5 model.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-root", default="obstacle_representation_3_data")
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = validate_model_3_5(
        Path(args.dataset) if args.dataset else None,
        Path(args.model) if args.model else None,
        output_root=Path(args.output_root),
        report_path=Path(args.report) if args.report else None,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
