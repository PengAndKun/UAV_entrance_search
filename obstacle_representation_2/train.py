from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model import APlus2AffordanceNet
from .schema import DIRECTION_LABELS, GEOMETRY_FEATURE_NAMES, MAX_DEPTH_CM, normalize_depth_cm


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_dataset_arrays(dataset_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    return {
        "image_paths": data["image_paths"],
        "depth_paths": data["depth_paths"],
        "geometry": data["geometry"].astype(np.float32),
        "mask_targets": data["mask_targets"].astype(np.float32),
        "direction_indices": data["direction_indices"].astype(np.int64),
        "direction_scores": data["direction_scores"].astype(np.float32),
        "flyover_delta_cm": data["flyover_delta_cm"].astype(np.float32),
        "red_front_blocked": data["red_front_blocked"].astype(bool),
        "splits": data["splits"].astype(str),
        "group_ids": data["group_ids"].astype(str),
    }


def load_rgb(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((int(image_size), int(image_size)), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def load_depth(path: str, image_size: int) -> np.ndarray:
    try:
        depth = np.load(path).astype(np.float32, copy=False)
        depth = np.squeeze(depth)
    except Exception:
        depth = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
    if depth.ndim != 2:
        depth = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
    norm = normalize_depth_cm(depth, max_depth_cm=MAX_DEPTH_CM)
    pil = Image.fromarray(norm.astype(np.float32), mode="F")
    return np.asarray(pil.resize((int(image_size), int(image_size)), Image.BILINEAR), dtype=np.float32)[None, :, :]


class APlus2Dataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        image_size: int,
        geometry_mean: np.ndarray,
        geometry_std: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.indices = indices.astype(np.int64, copy=False)
        self.image_size = int(image_size)
        self.geometry_mean = geometry_mean.astype(np.float32, copy=False)
        self.geometry_std = geometry_std.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        geometry = (self.arrays["geometry"][idx] - self.geometry_mean) / self.geometry_std
        flyover_norm = float(self.arrays["flyover_delta_cm"][idx]) / 400.0
        return {
            "rgb": torch.from_numpy(load_rgb(str(self.arrays["image_paths"][idx]), self.image_size).copy()),
            "depth": torch.from_numpy(load_depth(str(self.arrays["depth_paths"][idx]), self.image_size).copy()),
            "geometry": torch.from_numpy(geometry.astype(np.float32, copy=False)),
            "mask": torch.from_numpy(self.arrays["mask_targets"][idx].astype(np.float32, copy=False)),
            "direction": torch.tensor(int(self.arrays["direction_indices"][idx]), dtype=torch.long),
            "scores": torch.from_numpy(self.arrays["direction_scores"][idx].astype(np.float32, copy=False)),
            "flyover": torch.tensor(flyover_norm, dtype=torch.float32),
        }


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    intersection = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob + target, dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mask_iou = 0.0
    total_dir_acc = 0.0
    total_count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        geometry = batch["geometry"].to(device)
        mask = batch["mask"].to(device)
        direction = batch["direction"].to(device)
        scores = batch["scores"].to(device)
        flyover = batch["flyover"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(rgb, depth, geometry)
            bce = F.binary_cross_entropy_with_logits(output["mask_logits"], mask)
            dice = dice_loss_from_logits(output["mask_logits"], mask)
            loss_mask = bce + dice
            loss_dir = F.cross_entropy(output["direction_logits"], direction)
            loss_score = F.mse_loss(torch.sigmoid(output["score_logits"]), scores)
            loss_flyover = F.smooth_l1_loss(torch.clamp(output["flyover_delta"], min=0.0), flyover)
            loss = loss_mask + loss_dir + 0.2 * loss_score + 0.1 * loss_flyover
            if training:
                loss.backward()
                optimizer.step()
        pred_mask = torch.sigmoid(output["mask_logits"].detach()) >= 0.5
        target_mask = mask >= 0.5
        intersection = torch.logical_and(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
        union = torch.logical_or(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
        mask_iou = torch.where(union > 0, intersection / torch.clamp(union, min=1.0), torch.ones_like(union))
        pred_dir = torch.argmax(output["direction_logits"].detach(), dim=1)
        batch_size = int(direction.numel())
        total_count += batch_size
        total_loss += float(loss.item()) * batch_size
        total_mask_iou += float(mask_iou.mean().item()) * batch_size
        total_dir_acc += float((pred_dir == direction).float().mean().item()) * batch_size
    denom = max(1, total_count)
    return {
        "loss": total_loss / denom,
        "mask_iou": total_mask_iou / denom,
        "direction_accuracy": total_dir_acc / denom,
        "count": total_count,
    }


def train_model(
    dataset_path: Path,
    *,
    output_root: Path | None = None,
    epochs: int = 12,
    batch_size: int = 32,
    image_size: int = 96,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_path = Path(dataset_path)
    output_root = output_root or dataset_path.parents[1]
    model_dir = output_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "a_plus_2_model.pt"
    metrics_path = model_dir / "a_plus_2_training_metrics.json"

    arrays = load_dataset_arrays(dataset_path)
    train_indices = np.where(arrays["splits"] == "train")[0]
    val_indices = np.where(arrays["splits"] == "val")[0]
    if train_indices.size == 0:
        raise ValueError(f"dataset has no train samples: {dataset_path}")
    if val_indices.size == 0:
        val_indices = train_indices
    geometry_mean = arrays["geometry"][train_indices].mean(axis=0).astype(np.float32)
    geometry_std = arrays["geometry"][train_indices].std(axis=0).astype(np.float32)
    geometry_std = np.where(geometry_std < 1e-6, 1.0, geometry_std).astype(np.float32)

    train_ds = APlus2Dataset(arrays, train_indices, image_size=image_size, geometry_mean=geometry_mean, geometry_std=geometry_std)
    val_ds = APlus2Dataset(arrays, val_indices, image_size=image_size, geometry_mean=geometry_mean, geometry_std=geometry_std)
    direction_counts = np.bincount(arrays["direction_indices"][train_indices], minlength=len(DIRECTION_LABELS)).astype(np.float32)
    sampler_weights = np.asarray([1.0 / max(1.0, direction_counts[idx]) for idx in arrays["direction_indices"][train_indices]], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sampler_weights, dtype=torch.double),
        num_samples=int(train_indices.size),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APlus2AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_directions=len(DIRECTION_LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_val_loss = float("inf")
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dir_acc={val_metrics['direction_accuracy']:.4f} "
            f"val_mask_iou={val_metrics['mask_iou']:.4f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "model_version": "a_plus_2_v1",
            "image_size": int(image_size),
            "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
            "geometry_feature_names": GEOMETRY_FEATURE_NAMES,
            "direction_labels": list(DIRECTION_LABELS),
            "mask_channels": ["danger", "insufficient_clearance"],
            "danger_depth_cm": 250.0,
            "clearance_depth_cm": 450.0,
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
        "model_version": "a_plus_2_v1",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "learning_rate": float(learning_rate),
        "device": str(device),
        "train_count": int(train_indices.size),
        "val_count": int(val_indices.size),
        "duration_s": round(float(duration_s), 3),
        "best_val_loss": round(float(best_val_loss), 6),
        "direction_labels": list(DIRECTION_LABELS),
        "final_train": history[-1]["train"] if history else {},
        "final_val": history[-1]["val"] if history else {},
        "history": history,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(metrics_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Obstacle Representation 2 A+2 model.")
    parser.add_argument("--dataset", default="obstacle_representation_2_data/datasets/a_plus_2_dataset_latest.npz")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = train_model(
        Path(args.dataset),
        output_root=Path(args.output_root) if args.output_root else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
