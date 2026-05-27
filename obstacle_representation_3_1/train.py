from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from . import MODEL_VERSION


def default_training_paths(output_root: Path | str = Path("obstacle_representation_3_data")) -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "dataset": root / "datasets" / "a_plus_3_1_dataset_latest.npz",
        "model": root / "models" / "a_plus_3_1_model.pt",
        "metrics": root / "models" / "a_plus_3_1_training_metrics.json",
        "work_root": root / "or3_1_teacher_review" / "training_work",
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def train_model_3_1(
    dataset_path: Path | str | None = None,
    *,
    output_root: Path | str = Path("obstacle_representation_3_data"),
    epochs: int = 12,
    batch_size: int = 32,
    image_size: int = 96,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    import torch
    from obstacle_representation_3.train import train_model

    paths = default_training_paths(output_root)
    dataset = Path(dataset_path) if dataset_path else paths["dataset"]
    work_root = paths["work_root"]
    summary = train_model(
        dataset,
        output_root=work_root,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    source_model = work_root / "models" / "a_plus_3_model.pt"
    source_metrics = work_root / "models" / "a_plus_3_training_metrics.json"
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(source_model, map_location="cpu", weights_only=False)
    checkpoint.setdefault("config", {})
    checkpoint["config"]["model_version"] = MODEL_VERSION
    checkpoint["dataset_path"] = str(dataset)
    torch.save(checkpoint, paths["model"])

    if source_metrics.is_file():
        metrics = json.loads(source_metrics.read_text(encoding="utf-8"))
    else:
        metrics = dict(summary)
    metrics["model_path"] = str(paths["model"])
    metrics["metrics_path"] = str(paths["metrics"])
    metrics["dataset_path"] = str(dataset)
    metrics["model_version"] = MODEL_VERSION
    metrics["source_training_work_root"] = str(work_root)
    write_json(paths["metrics"], metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Obstacle Representation 3.1 A+3.1 model.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-root", default="obstacle_representation_3_data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = train_model_3_1(
        Path(args.dataset) if args.dataset else None,
        output_root=Path(args.output_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
