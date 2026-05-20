from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model import SchemeAObstacleNet, SchemeAPlusObstacleNet
from .schema import GEOMETRY_FEATURE_NAMES, LABEL_TO_INDEX, OBSTACLE_LABELS


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class SchemeADataset(Dataset):
    def __init__(
        self,
        image_paths: np.ndarray,
        geometry: np.ndarray,
        labels: np.ndarray,
        flyover: np.ndarray,
        indices: np.ndarray,
        *,
        depth_paths: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        use_depth: bool = False,
        image_size: int,
        geometry_mean: np.ndarray,
        geometry_std: np.ndarray,
    ) -> None:
        self.image_paths = image_paths.astype(str)
        self.depth_paths = depth_paths.astype(str) if depth_paths is not None else np.asarray([""] * len(image_paths), dtype=str)
        self.geometry = geometry.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.flyover = flyover.astype(np.float32, copy=False)
        self.sample_weights = (
            sample_weights.astype(np.float32, copy=False)
            if sample_weights is not None
            else np.ones_like(self.flyover, dtype=np.float32)
        )
        self.indices = indices.astype(np.int64, copy=False)
        self.use_depth = bool(use_depth)
        self.image_size = int(image_size)
        self.geometry_mean = geometry_mean.astype(np.float32, copy=False)
        self.geometry_std = geometry_std.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        with Image.open(self.image_paths[idx]) as image:
            rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0
        rgb_arr = np.transpose(rgb_arr, (2, 0, 1))
        geom = (self.geometry[idx] - self.geometry_mean) / self.geometry_std
        item = {
            "rgb": torch.from_numpy(rgb_arr),
            "geometry": torch.from_numpy(geom.astype(np.float32, copy=False)),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "flyover": torch.tensor(float(self.flyover[idx]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32),
        }
        if self.use_depth:
            item["depth"] = torch.from_numpy(load_depth_array(self.depth_paths[idx], self.image_size).copy())
        return item


def load_depth_array(depth_path: str, image_size: int) -> np.ndarray:
    try:
        depth = np.load(str(depth_path)).astype(np.float32)
    except Exception:
        depth = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        depth = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 65000.0)
    cleaned = np.zeros(depth.shape, dtype=np.float32)
    cleaned[valid] = np.clip(depth[valid], 0.0, 1200.0) / 1200.0
    pil = Image.fromarray(cleaned.astype(np.float32), mode="F")
    resized = pil.resize((int(image_size), int(image_size)), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, :, :]


def load_dataset_arrays(dataset_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    n = int(data["image_paths"].shape[0])
    return {
        "image_paths": data["image_paths"],
        "depth_paths": data["depth_paths"] if "depth_paths" in data else np.asarray([""] * n, dtype=object),
        "geometry": data["geometry"].astype(np.float32),
        "label_indices": data["label_indices"].astype(np.int64),
        "flyover": data["flyover"].astype(np.float32),
        "sample_weights": data["sample_weights"].astype(np.float32) if "sample_weights" in data else np.ones(n, dtype=np.float32),
        "splits": data["splits"].astype(str),
        "teacher_sources": data["teacher_sources"].astype(str) if "teacher_sources" in data else np.asarray(["unknown"] * n, dtype=str),
        "group_ids": data["group_ids"].astype(str) if "group_ids" in data else np.asarray([""] * n, dtype=str),
        "is_manual_hard_case": data["is_manual_hard_case"].astype(bool) if "is_manual_hard_case" in data else np.zeros(n, dtype=bool),
    }


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((torch.argmax(logits, dim=1) == labels).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor,
    device: torch.device,
    use_depth: bool,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_label_acc = 0.0
    total_flyover_acc = 0.0
    total_count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        geometry = batch["geometry"].to(device)
        labels = batch["label"].to(device)
        flyover = batch["flyover"].to(device)
        weights = batch["sample_weight"].to(device)
        depth = batch.get("depth")
        depth_tensor = depth.to(device) if use_depth and depth is not None else None
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out = model(rgb, geometry, depth_tensor) if use_depth else model(rgb, geometry)
            label_loss = F.cross_entropy(out["label_logits"], labels, weight=class_weights, reduction="none")
            flyover_loss = F.binary_cross_entropy_with_logits(out["flyover_logits"], flyover, reduction="none")
            loss_each = label_loss + 0.3 * flyover_loss
            loss = torch.sum(loss_each * weights) / torch.clamp(torch.sum(weights), min=1.0)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = int(labels.numel())
        total_count += batch_size
        total_loss += float(loss.item()) * batch_size
        total_label_acc += accuracy_from_logits(out["label_logits"].detach(), labels) * batch_size
        flyover_pred = (torch.sigmoid(out["flyover_logits"].detach()) >= 0.5).float()
        total_flyover_acc += float((flyover_pred == flyover).float().mean().item()) * batch_size
    denom = max(1, total_count)
    return {
        "loss": total_loss / denom,
        "label_accuracy": total_label_acc / denom,
        "flyover_accuracy": total_flyover_acc / denom,
        "count": total_count,
    }


def train_model(
    dataset_path: Path,
    *,
    output_root: Path | None = None,
    epochs: int = 15,
    batch_size: int = 32,
    image_size: int = 96,
    learning_rate: float = 1e-3,
    seed: int = 42,
    model_version: str = "scheme_a",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_path = Path(dataset_path)
    output_root = output_root or dataset_path.parents[1]
    model_dir = output_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    use_depth = str(model_version).strip().lower() in {"scheme_a_plus", "scheme_a_plus_v1", "a_plus"}
    normalized_model_version = "scheme_a_plus_v1" if use_depth else "scheme_a_v1"
    model_path = model_dir / ("scheme_a_plus_model.pt" if use_depth else "scheme_a_model.pt")
    metrics_path = model_dir / ("training_metrics_scheme_a_plus.json" if use_depth else "training_metrics.json")
    label_map_path = model_dir / "label_map.json"

    arrays = load_dataset_arrays(dataset_path)
    train_indices = np.where(arrays["splits"] == "train")[0]
    val_indices = np.where(arrays["splits"] == "val")[0]
    if train_indices.size == 0:
        raise ValueError(f"dataset has no train samples: {dataset_path}")
    if val_indices.size == 0:
        val_indices = train_indices

    geometry_mean = arrays["geometry"][train_indices].mean(axis=0)
    geometry_std = arrays["geometry"][train_indices].std(axis=0)
    geometry_std = np.where(geometry_std < 1e-6, 1.0, geometry_std).astype(np.float32)
    geometry_mean = geometry_mean.astype(np.float32)

    train_ds = SchemeADataset(
        arrays["image_paths"],
        arrays["geometry"],
        arrays["label_indices"],
        arrays["flyover"],
        train_indices,
        depth_paths=arrays["depth_paths"],
        sample_weights=arrays["sample_weights"],
        use_depth=use_depth,
        image_size=image_size,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
    )
    val_ds = SchemeADataset(
        arrays["image_paths"],
        arrays["geometry"],
        arrays["label_indices"],
        arrays["flyover"],
        val_indices,
        depth_paths=arrays["depth_paths"],
        sample_weights=arrays["sample_weights"],
        use_depth=use_depth,
        image_size=image_size,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
    )
    sampler_weights = arrays["sample_weights"][train_indices].astype(np.float32, copy=True)
    building_fence = np.isin(arrays["label_indices"][train_indices], [LABEL_TO_INDEX["building"], LABEL_TO_INDEX["fence_or_rail"]])
    sampler_weights[building_fence] *= 1.5
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(np.maximum(sampler_weights, 1e-3), dtype=torch.double),
        num_samples=int(train_indices.size),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_depth:
        model: nn.Module = SchemeAPlusObstacleNet(num_labels=len(OBSTACLE_LABELS), geometry_dim=len(GEOMETRY_FEATURE_NAMES)).to(device)
    else:
        model = SchemeAObstacleNet(num_labels=len(OBSTACLE_LABELS), geometry_dim=len(GEOMETRY_FEATURE_NAMES)).to(device)
    class_counts = np.bincount(arrays["label_indices"][train_indices], minlength=len(OBSTACLE_LABELS)).astype(np.float32)
    class_weights = np.where(class_counts > 0, 1.0 / np.sqrt(np.maximum(class_counts, 1.0)), 0.0)
    if float(class_weights.sum()) > 0.0:
        class_weights = class_weights / class_weights[class_weights > 0].mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    start = time.perf_counter()
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights_tensor,
            use_depth=use_depth,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            class_weights=class_weights_tensor,
            use_depth=use_depth,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['label_accuracy']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "model_version": normalized_model_version,
            "use_depth": bool(use_depth),
            "num_labels": len(OBSTACLE_LABELS),
            "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
            "image_size": int(image_size),
            "labels": list(OBSTACLE_LABELS),
            "geometry_feature_names": GEOMETRY_FEATURE_NAMES,
        },
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "dataset_path": str(dataset_path),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(checkpoint, model_path)
    duration_s = time.perf_counter() - start
    summary = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "dataset_path": str(dataset_path),
        "model_version": normalized_model_version,
        "use_depth": bool(use_depth),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "learning_rate": float(learning_rate),
        "device": str(device),
        "train_count": int(train_indices.size),
        "val_count": int(val_indices.size),
        "duration_s": round(float(duration_s), 3),
        "best_val_loss": round(float(best_val_loss), 6),
        "final_train": history[-1]["train"] if history else {},
        "final_val": history[-1]["val"] if history else {},
        "history": history,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(metrics_path, summary)
    write_json(label_map_path, LABEL_TO_INDEX)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Scheme A obstacle representation model.")
    parser.add_argument("--dataset", default="obstacle_representation_data/datasets/dataset_latest.npz")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-version", choices=["scheme_a", "scheme_a_plus"], default="scheme_a")
    args = parser.parse_args()
    summary = train_model(
        Path(args.dataset),
        output_root=Path(args.output_root) if args.output_root else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        model_version=args.model_version,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
