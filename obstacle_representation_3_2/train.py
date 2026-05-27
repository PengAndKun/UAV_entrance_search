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

from . import MODEL_VERSION


def default_training_paths(output_root: Path | str = Path("obstacle_representation_3_data")) -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "dataset": root / "datasets" / "a_plus_3_1_dataset_latest.npz",
        "pretrained_model": root / "models" / "a_plus_3_model.pt",
        "model": root / "models" / "a_plus_3_2_model.pt",
        "metrics": root / "models" / "a_plus_3_2_training_metrics.json",
    }


def default_safe_finetune_config() -> Dict[str, Any]:
    return {
        "learning_rate": 2e-4,
        "stop_mask_loss_weight": 3.0,
        "must_stop_loss_weight": 4.0,
        "distillation_weight": 0.08,
        "risk_class_weights": {
            "clear": 1.0,
            "clearance_warning": 1.2,
            "obstacle_warning": 1.2,
            "must_stop": 3.5,
        },
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint(path: Path, device: Any) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _tensor_metrics(output: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, float]:
    import torch

    risk = batch["risk"]
    mask = batch["mask"]
    must_stop = batch["must_stop"]
    pred_mask = torch.sigmoid(output["mask_logits"].detach()) >= 0.5
    target_mask = mask >= 0.5
    intersection = torch.logical_and(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
    union = torch.logical_or(pred_mask, target_mask).sum(dim=(1, 2, 3)).float()
    mask_iou = torch.where(union > 0, intersection / torch.clamp(union, min=1.0), torch.ones_like(union))
    pred_risk = torch.argmax(output["risk_logits"].detach(), dim=1)
    pred_stop = (torch.sigmoid(output["must_stop_logits"].detach()) >= 0.5).float()
    return {
        "mask_iou": float(mask_iou.mean().item()),
        "risk_accuracy": float((pred_risk == risk).float().mean().item()),
        "must_stop_accuracy": float((pred_stop == must_stop).float().mean().item()),
    }


def _run_safe_epoch(
    model: Any,
    teacher_model: Any,
    loader: Any,
    *,
    optimizer: Any,
    device: Any,
    config: Dict[str, Any],
) -> Dict[str, float]:
    import torch
    from torch.nn import functional as F

    training = optimizer is not None
    model.train(training)
    teacher_model.eval()
    risk_weights = torch.tensor(
        [
            float(config["risk_class_weights"]["clear"]),
            float(config["risk_class_weights"]["clearance_warning"]),
            float(config["risk_class_weights"]["obstacle_warning"]),
            float(config["risk_class_weights"]["must_stop"]),
        ],
        dtype=torch.float32,
        device=device,
    )
    mask_weights = torch.tensor([1.0, 1.0, float(config["stop_mask_loss_weight"])], dtype=torch.float32, device=device).view(
        1, 3, 1, 1
    )
    pos_weight = torch.tensor([float(config["must_stop_loss_weight"])], dtype=torch.float32, device=device)
    total_loss = total_mask_iou = total_risk_acc = total_stop_acc = 0.0
    total_count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        geometry = batch["geometry"].to(device)
        mask = batch["mask"].to(device)
        risk = batch["risk"].to(device)
        can_forward = batch["can_forward"].to(device)
        must_stop = batch["must_stop"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher = teacher_model(rgb, depth, geometry)
        with torch.set_grad_enabled(training):
            output = model(rgb, depth, geometry)
            raw_mask_loss = F.binary_cross_entropy_with_logits(output["mask_logits"], mask, reduction="none")
            loss_mask = (raw_mask_loss * mask_weights).mean()
            loss_risk = F.cross_entropy(output["risk_logits"], risk, weight=risk_weights)
            loss_can_forward = F.binary_cross_entropy_with_logits(output["can_forward_logits"], can_forward)
            loss_must_stop = F.binary_cross_entropy_with_logits(output["must_stop_logits"], must_stop, pos_weight=pos_weight)
            loss_distill = F.mse_loss(output["must_stop_logits"], teacher["must_stop_logits"].detach())
            loss_distill = loss_distill + 0.25 * F.mse_loss(output["mask_logits"][:, 2], teacher["mask_logits"][:, 2].detach())
            loss = loss_mask + loss_risk + 0.2 * loss_can_forward + 0.5 * loss_must_stop + float(config["distillation_weight"]) * loss_distill
            if training:
                loss.backward()
                optimizer.step()
        metrics = _tensor_metrics(output, {"risk": risk, "mask": mask, "must_stop": must_stop})
        batch_size = int(risk.numel())
        total_count += batch_size
        total_loss += float(loss.item()) * batch_size
        total_mask_iou += metrics["mask_iou"] * batch_size
        total_risk_acc += metrics["risk_accuracy"] * batch_size
        total_stop_acc += metrics["must_stop_accuracy"] * batch_size
    denom = max(1, total_count)
    return {
        "loss": total_loss / denom,
        "mask_iou": total_mask_iou / denom,
        "risk_accuracy": total_risk_acc / denom,
        "must_stop_accuracy": total_stop_acc / denom,
        "count": total_count,
    }


def train_model_3_2(
    dataset_path: Path | str | None = None,
    *,
    pretrained_model_path: Path | str | None = None,
    output_root: Path | str = Path("obstacle_representation_3_data"),
    epochs: int = 8,
    batch_size: int = 32,
    image_size: int = 96,
    seed: int = 42,
    learning_rate: float | None = None,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from obstacle_representation_3.model import APlus3AffordanceNet
    from obstacle_representation_3.schema import GEOMETRY_FEATURE_NAMES, MASK_CHANNELS, PROJECTION_BOX, RISK_STATES
    from obstacle_representation_3.train import APlus3Dataset, load_dataset_arrays

    paths = default_training_paths(output_root)
    dataset = Path(dataset_path) if dataset_path else paths["dataset"]
    pretrained = Path(pretrained_model_path) if pretrained_model_path else paths["pretrained_model"]
    config = default_safe_finetune_config()
    if learning_rate is not None:
        config["learning_rate"] = float(learning_rate)

    torch.manual_seed(seed)
    np.random.seed(seed)
    arrays = load_dataset_arrays(dataset)
    train_indices = np.where(arrays["splits"] == "train")[0]
    val_indices = np.where(arrays["splits"] == "val")[0]
    if train_indices.size == 0:
        raise ValueError(f"dataset has no train samples: {dataset}")
    if val_indices.size == 0:
        val_indices = train_indices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_checkpoint = _load_checkpoint(pretrained, "cpu")
    geometry_mean = np.asarray(pretrained_checkpoint.get("geometry_mean"), dtype=np.float32)
    geometry_std = np.asarray(pretrained_checkpoint.get("geometry_std"), dtype=np.float32)
    if geometry_mean.shape != (len(GEOMETRY_FEATURE_NAMES),):
        geometry_mean = arrays["geometry"][train_indices].mean(axis=0).astype(np.float32)
    if geometry_std.shape != (len(GEOMETRY_FEATURE_NAMES),):
        geometry_std = arrays["geometry"][train_indices].std(axis=0).astype(np.float32)
    geometry_std = np.where(np.abs(geometry_std) < 1e-6, 1.0, geometry_std).astype(np.float32)

    train_ds = APlus3Dataset(arrays, train_indices, image_size=image_size, geometry_mean=geometry_mean, geometry_std=geometry_std)
    val_ds = APlus3Dataset(arrays, val_indices, image_size=image_size, geometry_mean=geometry_mean, geometry_std=geometry_std)
    risk_counts = np.bincount(arrays["risk_indices"][train_indices], minlength=len(RISK_STATES)).astype(np.float32)
    sampler_weights = np.asarray(
        [
            float(config["risk_class_weights"][RISK_STATES[int(idx)]]) / max(1.0, risk_counts[int(idx)])
            for idx in arrays["risk_indices"][train_indices]
        ],
        dtype=np.float32,
    )
    sampler = WeightedRandomSampler(torch.as_tensor(sampler_weights, dtype=torch.double), int(train_indices.size), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = APlus3AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(RISK_STATES)).to(device)
    teacher_model = APlus3AffordanceNet(geometry_dim=len(GEOMETRY_FEATURE_NAMES), num_risk_states=len(RISK_STATES)).to(device)
    model.load_state_dict(pretrained_checkpoint["model_state"])
    teacher_model.load_state_dict(pretrained_checkpoint["model_state"])
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    history = []
    best_score = float("-inf")
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_safe_epoch(model, teacher_model, train_loader, optimizer=optimizer, device=device, config=config)
        val_metrics = _run_safe_epoch(model, teacher_model, val_loader, optimizer=None, device=device, config=config)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        score = float(val_metrics["risk_accuracy"]) + 0.35 * float(val_metrics["must_stop_accuracy"]) + 0.15 * float(val_metrics["mask_iou"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_risk_acc={val_metrics['risk_accuracy']:.4f} "
            f"val_mask_iou={val_metrics['mask_iou']:.4f} val_stop_acc={val_metrics['must_stop_accuracy']:.4f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "model_version": MODEL_VERSION,
            "image_size": int(image_size),
            "geometry_dim": len(GEOMETRY_FEATURE_NAMES),
            "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "risk_states": list(RISK_STATES),
            "mask_channels": list(MASK_CHANNELS),
            "projection_box": dict(PROJECTION_BOX),
            "safe_finetune_config": config,
            "pretrained_model_path": str(pretrained),
            "pretrained_model_version": str(pretrained_checkpoint.get("config", {}).get("model_version", "")),
        },
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "dataset_path": str(dataset),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(checkpoint, paths["model"])
    duration_s = time.perf_counter() - start
    summary = {
        "model_path": str(paths["model"]),
        "metrics_path": str(paths["metrics"]),
        "dataset_path": str(dataset),
        "pretrained_model_path": str(pretrained),
        "model_version": MODEL_VERSION,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "learning_rate": float(config["learning_rate"]),
        "safe_finetune_config": config,
        "device": str(device),
        "train_count": int(train_indices.size),
        "val_count": int(val_indices.size),
        "duration_s": round(float(duration_s), 3),
        "best_score": round(float(best_score), 6),
        "final_train": history[-1]["train"] if history else {},
        "final_val": history[-1]["val"] if history else {},
        "history": history,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(paths["metrics"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Obstacle Representation 3.2 safe fine-tuned model.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--output-root", default="obstacle_representation_3_data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = train_model_3_2(
        Path(args.dataset) if args.dataset else None,
        pretrained_model_path=Path(args.pretrained_model) if args.pretrained_model else None,
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
