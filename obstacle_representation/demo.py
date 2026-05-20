from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image

from .model import SchemeAObstacleNet, SchemeAPlusObstacleNet
from .schema import GEOMETRY_FEATURE_NAMES, OBSTACLE_LABELS, geometry_vector
from .train import load_depth_array


LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "open_path": (60, 190, 90),
    "tree_trunk_or_pole": (170, 110, 45),
    "tree_canopy_or_cluster": (70, 170, 70),
    "fence_or_rail": (245, 190, 45),
    "building": (230, 80, 70),
    "mixed": (165, 95, 210),
    "unknown": (120, 135, 150),
}


def _load_checkpoint(model_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    labels = list(config.get("labels", OBSTACLE_LABELS))
    geometry_dim = int(config.get("geometry_dim", len(GEOMETRY_FEATURE_NAMES)))
    if bool(config.get("use_depth", False)):
        model = SchemeAPlusObstacleNet(num_labels=len(labels), geometry_dim=geometry_dim).to(device)
    else:
        model = SchemeAObstacleNet(num_labels=len(labels), geometry_dim=geometry_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _rgb_tensor_from_path(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _depth_tensor_from_event(event: Dict[str, Any], image_size: int, device: torch.device) -> torch.Tensor:
    depth_path = str(event.get("depth_npy_path") or event.get("depth_path") or "")
    depth = load_depth_array(depth_path, image_size)
    return torch.from_numpy(depth.copy()).unsqueeze(0).to(device)


def predict_obstacle_representation(
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
    labels: List[str] = list(config.get("labels", OBSTACLE_LABELS))
    image_size = int(config.get("image_size", 96))
    model_version = str(config.get("model_version", "scheme_a_v1"))
    use_depth = bool(config.get("use_depth", False))

    geometry = geometry_vector(event)
    mean = np.asarray(checkpoint.get("geometry_mean", np.zeros_like(geometry)), dtype=np.float32)
    std = np.asarray(checkpoint.get("geometry_std", np.ones_like(geometry)), dtype=np.float32)
    std = np.where(np.abs(std) < 1e-6, 1.0, std).astype(np.float32)
    if mean.shape[0] != geometry.shape[0] or std.shape[0] != geometry.shape[0]:
        raise ValueError(f"checkpoint geometry shape mismatch: model={mean.shape[0]} event={geometry.shape[0]}")
    geometry_tensor = torch.from_numpy(((geometry - mean) / std).astype(np.float32)).unsqueeze(0).to(device)
    rgb_tensor = _rgb_tensor_from_path(rgb_path, image_size, device)
    depth_tensor = _depth_tensor_from_event(event, image_size, device) if use_depth else None

    with torch.no_grad():
        output = model(rgb_tensor, geometry_tensor, depth_tensor) if use_depth else model(rgb_tensor, geometry_tensor)
        probabilities_tensor = torch.softmax(output["label_logits"], dim=1)[0].detach().cpu()
        flyover_probability = float(torch.sigmoid(output["flyover_logits"])[0].detach().cpu().item())
    probabilities = {label: float(probabilities_tensor[idx].item()) for idx, label in enumerate(labels)}
    best_index = int(torch.argmax(probabilities_tensor).item())
    top3 = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
    predicted_label = labels[best_index] if 0 <= best_index < len(labels) else "unknown"
    building_vs_fence_margin = float(probabilities.get("building", 0.0) - probabilities.get("fence_or_rail", 0.0))
    return {
        "model_path": str(model_path),
        "model_version": model_version,
        "use_depth": bool(use_depth),
        "rgb_path": str(rgb_path),
        "predicted_label": predicted_label,
        "confidence": float(probabilities.get(predicted_label, 0.0)),
        "probabilities": probabilities,
        "top3": [{"label": label, "probability": probability} for label, probability in top3],
        "building_vs_fence_margin": building_vs_fence_margin,
        "flyover_probability": flyover_probability,
        "flyover_recommended": bool(flyover_probability >= 0.5),
        "device": str(device),
        "image_size": int(image_size),
    }


def _coerce_depth(depth_image: Any) -> np.ndarray | None:
    if depth_image is None:
        return None
    depth = np.asarray(depth_image)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.squeeze(depth).astype(np.float32, copy=False)
    return depth if depth.ndim == 2 else None


def _nearest_depth_mask(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not bool(np.any(valid)):
        return np.zeros(depth.shape, dtype=bool)
    valid_values = depth[valid]
    p15 = float(np.percentile(valid_values, 15))
    p35 = float(np.percentile(valid_values, 35))
    threshold = max(80.0, min(650.0, p35 + max(40.0, 0.35 * max(1.0, p35 - p15))))
    return valid & (depth <= threshold)


def render_prediction_mask(
    rgb_image: Any,
    depth_image: Any,
    prediction: Dict[str, Any],
    *,
    alpha: float = 0.78,
) -> np.ndarray:
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb_image must be HxWx3")
    rgb = rgb[..., :3].astype(np.uint8, copy=False)
    depth = _coerce_depth(depth_image)
    label = str(prediction.get("predicted_label", "unknown") or "unknown")
    color = np.asarray(LABEL_COLORS.get(label, LABEL_COLORS["unknown"]), dtype=np.float32)
    base = (rgb.astype(np.float32) * 0.22).astype(np.uint8)
    canvas = base.astype(np.float32)
    if depth is not None:
        if depth.shape[:2] != rgb.shape[:2]:
            depth_pil = Image.fromarray(depth.astype(np.float32), mode="F")
            depth = np.asarray(depth_pil.resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST), dtype=np.float32)
        mask = _nearest_depth_mask(depth)
    else:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
    if label == "open_path":
        mask[:] = False
    if bool(np.any(mask)):
        canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha
    else:
        band_h = max(6, rgb.shape[0] // 18)
        canvas[:band_h, :] = color
    return np.clip(canvas, 0, 255).astype(np.uint8)
