from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

from .model import APlus2AffordanceNet
from .schema import DIRECTION_LABELS, GEOMETRY_FEATURE_NAMES, event_geometry_vector
from .train import load_depth, load_rgb


DIRECTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "forward": (60, 190, 90),
    "left": (60, 130, 230),
    "right": (80, 210, 230),
    "up": (245, 190, 45),
    "backoff": (230, 80, 70),
    "hold": (145, 145, 155),
}


def _load_checkpoint(model_path: Path, device: torch.device) -> tuple[APlus2AffordanceNet, Dict[str, Any]]:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model = APlus2AffordanceNet(
        geometry_dim=int(config.get("geometry_dim", len(GEOMETRY_FEATURE_NAMES))),
        num_directions=len(config.get("direction_labels", DIRECTION_LABELS)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_obstacle_representation_2(
    model_path: str | Path,
    rgb_path: str | Path,
    event: Dict[str, Any],
    *,
    device_name: str | None = None,
) -> Dict[str, Any]:
    model_path = Path(model_path).expanduser()
    rgb_path = Path(rgb_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not rgb_path.is_file():
        raise FileNotFoundError(f"rgb image not found: {rgb_path}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = _load_checkpoint(model_path, device)
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 96))
    labels = list(config.get("direction_labels", DIRECTION_LABELS))

    geometry = event_geometry_vector(event)
    mean = np.asarray(checkpoint.get("geometry_mean", np.zeros_like(geometry)), dtype=np.float32)
    std = np.asarray(checkpoint.get("geometry_std", np.ones_like(geometry)), dtype=np.float32)
    std = np.where(np.abs(std) < 1e-6, 1.0, std).astype(np.float32)
    rgb = torch.from_numpy(load_rgb(str(rgb_path), image_size).copy()).unsqueeze(0).to(device)
    depth = torch.from_numpy(load_depth(str(event.get("depth_npy_path") or event.get("depth_path") or ""), image_size).copy()).unsqueeze(0).to(device)
    geometry_tensor = torch.from_numpy(((geometry - mean) / std).astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(rgb, depth, geometry_tensor)
        mask_prob = torch.sigmoid(output["mask_logits"])[0].detach().cpu().numpy()
        direction_prob = torch.softmax(output["direction_logits"], dim=1)[0].detach().cpu().numpy()
        score_prob = torch.sigmoid(output["score_logits"])[0].detach().cpu().numpy()
        flyover_delta_cm = float(max(0.0, output["flyover_delta"][0].detach().cpu().item() * 400.0))
    selected_idx = int(np.argmax(direction_prob))
    selected_direction = labels[selected_idx] if 0 <= selected_idx < len(labels) else "hold"
    danger = mask_prob[0] if mask_prob.ndim == 3 else np.zeros((image_size, image_size), dtype=np.float32)
    insufficient = mask_prob[1] if mask_prob.ndim == 3 and mask_prob.shape[0] > 1 else np.zeros_like(danger)
    h, w = danger.shape
    front = danger[int(h * 0.26) : int(h * 0.82), int(w * 0.34) : int(w * 0.66)]
    front_insuff = insufficient[int(h * 0.26) : int(h * 0.82), int(w * 0.34) : int(w * 0.66)]
    front_red_fraction = float(np.mean(front >= 0.5)) if front.size else 0.0
    front_insufficient_fraction = float(np.mean(front_insuff >= 0.5)) if front_insuff.size else 0.0
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    front_min = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
    red_front_blocked = bool(front_red_fraction >= 0.025 or front_insufficient_fraction >= 0.22 or (front_min > 0.0 and front_min < 250.0))
    direction_scores = {label: float(score_prob[idx]) for idx, label in enumerate(labels)}
    if red_front_blocked:
        direction_scores["forward"] = 0.0
        if selected_direction == "forward":
            selected_direction = max((label for label in labels if label != "forward"), key=lambda item: direction_scores.get(item, 0.0))
    reason = (
        f"selected={selected_direction}; front_red={front_red_fraction:.3f}; "
        f"front_insufficient={front_insufficient_fraction:.3f}; front_min={front_min:.1f}cm"
    )
    return {
        "model_path": str(model_path),
        "model_version": str(config.get("model_version", "a_plus_2_v1")),
        "rgb_path": str(rgb_path),
        "danger_mask": danger.astype(np.float32),
        "insufficient_clearance_mask": insufficient.astype(np.float32),
        "direction_probabilities": {label: float(direction_prob[idx]) for idx, label in enumerate(labels)},
        "direction_scores": direction_scores,
        "selected_direction": selected_direction,
        "red_front_blocked": bool(red_front_blocked),
        "front_red_fraction": front_red_fraction,
        "front_insufficient_fraction": front_insufficient_fraction,
        "flyover_delta_cm": flyover_delta_cm,
        "reason": reason,
        "device": str(device),
        "image_size": int(image_size),
    }


def render_affordance_overlay(rgb_image: Any, prediction: Dict[str, Any], *, alpha: float = 0.72) -> np.ndarray:
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb_image must be HxWx3")
    rgb = rgb[..., :3].astype(np.uint8, copy=False)
    h, w = rgb.shape[:2]
    danger = np.asarray(prediction.get("danger_mask", np.zeros((h, w))), dtype=np.float32)
    insufficient = np.asarray(prediction.get("insufficient_clearance_mask", np.zeros_like(danger)), dtype=np.float32)
    if danger.shape != (h, w):
        danger = np.asarray(Image.fromarray(danger.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR), dtype=np.float32)
    if insufficient.shape != (h, w):
        insufficient = np.asarray(Image.fromarray(insufficient.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR), dtype=np.float32)
    canvas = (rgb.astype(np.float32) * 0.35).astype(np.float32)
    yellow = np.asarray((245, 190, 45), dtype=np.float32)
    red = np.asarray((230, 50, 50), dtype=np.float32)
    yellow_mask = insufficient >= 0.5
    red_mask = danger >= 0.5
    canvas[yellow_mask] = canvas[yellow_mask] * (1.0 - alpha) + yellow * alpha
    canvas[red_mask] = canvas[red_mask] * (1.0 - alpha) + red * alpha
    direction = str(prediction.get("selected_direction", "hold"))
    color = np.asarray(DIRECTION_COLORS.get(direction, DIRECTION_COLORS["hold"]), dtype=np.float32)
    band_h = max(10, h // 18)
    canvas[:band_h, :] = color
    return np.clip(canvas, 0, 255).astype(np.uint8)
