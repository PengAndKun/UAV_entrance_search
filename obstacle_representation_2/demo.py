from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

from .model import APlus2AffordanceNet
from .schema import GEOMETRY_FEATURE_NAMES, RISK_STATES, event_geometry_vector
from .train import load_depth, load_rgb


RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "clear": (60, 190, 90),
    "clearance_warning": (245, 200, 45),
    "obstacle_warning": (245, 130, 95),
    "must_stop": (190, 20, 35),
}


def _load_checkpoint(model_path: Path, device: torch.device) -> tuple[APlus2AffordanceNet, Dict[str, Any]]:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model = APlus2AffordanceNet(
        geometry_dim=int(config.get("geometry_dim", len(GEOMETRY_FEATURE_NAMES))),
        num_risk_states=len(config.get("risk_states", RISK_STATES)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _front_fraction(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    region = mask[int(h * 0.26) : int(h * 0.82), int(w * 0.34) : int(w * 0.66)]
    return float(np.mean(region >= 0.5)) if region.size else 0.0


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
    labels = list(config.get("risk_states", RISK_STATES))

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
        risk_prob = torch.softmax(output["risk_logits"], dim=1)[0].detach().cpu().numpy()
        can_forward_probability = float(torch.sigmoid(output["can_forward_logits"])[0].detach().cpu().item())
        must_stop_probability = float(torch.sigmoid(output["must_stop_logits"])[0].detach().cpu().item())
    selected_idx = int(np.argmax(risk_prob))
    front_risk_state = labels[selected_idx] if 0 <= selected_idx < len(labels) else "clear"
    clearance = mask_prob[0] if mask_prob.ndim == 3 else np.zeros((image_size, image_size), dtype=np.float32)
    warning = mask_prob[1] if mask_prob.ndim == 3 and mask_prob.shape[0] > 1 else np.zeros_like(clearance)
    stop = mask_prob[2] if mask_prob.ndim == 3 and mask_prob.shape[0] > 2 else np.zeros_like(clearance)
    front_clearance_fraction = _front_fraction(clearance)
    front_warning_fraction = _front_fraction(warning)
    front_stop_fraction = _front_fraction(stop)
    summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
    front_min = float(summary.get("front_min_depth_cm", 0.0) or 0.0)

    # Safety override: the raw model can only become more conservative.
    if front_stop_fraction >= 0.01 or (front_min > 0.0 and front_min <= 100.0) or must_stop_probability >= 0.5:
        front_risk_state = "must_stop"
    elif front_warning_fraction >= 0.025 and labels.index(front_risk_state) < labels.index("obstacle_warning"):
        front_risk_state = "obstacle_warning"
    elif front_clearance_fraction >= 0.10 and labels.index(front_risk_state) < labels.index("clearance_warning"):
        front_risk_state = "clearance_warning"
    must_stop = front_risk_state == "must_stop"
    can_forward = bool((not must_stop) and can_forward_probability >= 0.35)
    reason = (
        f"risk={front_risk_state}; can_forward={can_forward}; must_stop={must_stop}; "
        f"front_clearance={front_clearance_fraction:.3f}; front_warning={front_warning_fraction:.3f}; "
        f"front_stop={front_stop_fraction:.3f}; front_min={front_min:.1f}cm"
    )
    return {
        "model_path": str(model_path),
        "model_version": str(config.get("model_version", "a_plus_2_v1")),
        "rgb_path": str(rgb_path),
        "clearance_warning_mask": clearance.astype(np.float32),
        "obstacle_warning_mask": warning.astype(np.float32),
        "must_stop_mask": stop.astype(np.float32),
        "risk_probabilities": {label: float(risk_prob[idx]) for idx, label in enumerate(labels)},
        "front_risk_state": front_risk_state,
        "can_forward": bool(can_forward),
        "must_stop": bool(must_stop),
        "can_forward_probability": can_forward_probability,
        "must_stop_probability": must_stop_probability,
        "front_clearance_fraction": front_clearance_fraction,
        "front_warning_fraction": front_warning_fraction,
        "front_stop_fraction": front_stop_fraction,
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
    clearance = np.asarray(prediction.get("clearance_warning_mask", np.zeros((h, w))), dtype=np.float32)
    warning = np.asarray(prediction.get("obstacle_warning_mask", np.zeros_like(clearance)), dtype=np.float32)
    stop = np.asarray(prediction.get("must_stop_mask", np.zeros_like(clearance)), dtype=np.float32)
    for name, mask in (("clearance", clearance), ("warning", warning), ("stop", stop)):
        if mask.shape != (h, w):
            resized = np.asarray(Image.fromarray(mask.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR), dtype=np.float32)
            if name == "clearance":
                clearance = resized
            elif name == "warning":
                warning = resized
            else:
                stop = resized
    canvas = (rgb.astype(np.float32) * 0.35).astype(np.float32)
    yellow = np.asarray(RISK_COLORS["clearance_warning"], dtype=np.float32)
    light_red = np.asarray(RISK_COLORS["obstacle_warning"], dtype=np.float32)
    dark_red = np.asarray(RISK_COLORS["must_stop"], dtype=np.float32)
    clearance_mask = clearance >= 0.5
    warning_mask = warning >= 0.5
    stop_mask = stop >= 0.5
    canvas[clearance_mask] = canvas[clearance_mask] * (1.0 - alpha) + yellow * alpha
    canvas[warning_mask] = canvas[warning_mask] * (1.0 - alpha) + light_red * alpha
    canvas[stop_mask] = canvas[stop_mask] * (1.0 - alpha) + dark_red * alpha
    state = str(prediction.get("front_risk_state", "clear"))
    color = np.asarray(RISK_COLORS.get(state, RISK_COLORS["clear"]), dtype=np.float32)
    band_h = max(10, h // 18)
    canvas[:band_h, :] = color
    return np.clip(canvas, 0, 255).astype(np.uint8)
