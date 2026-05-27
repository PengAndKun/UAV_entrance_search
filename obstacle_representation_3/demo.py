from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

from .model import APlus3AffordanceNet
from .schema import GEOMETRY_FEATURE_NAMES, PROJECTION_BOX, RISK_STATES, event_geometry_vector
from .teacher import depth_masks, projection_box_slices, projection_box_stats
from .train import load_depth, load_rgb


RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "clear": (60, 190, 90),
    "clearance_warning": (245, 200, 45),
    "obstacle_warning": (245, 130, 95),
    "must_stop": (190, 20, 35),
}


def _load_checkpoint(model_path: Path, device: torch.device) -> tuple[APlus3AffordanceNet, Dict[str, Any]]:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model = APlus3AffordanceNet(
        geometry_dim=int(config.get("geometry_dim", len(GEOMETRY_FEATURE_NAMES))),
        num_risk_states=len(config.get("risk_states", RISK_STATES)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask > 0.5)) / float(mask.size) if mask.size else 0.0


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(np.float32, copy=False)
    return np.asarray(Image.fromarray(mask.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR), dtype=np.float32)


def _load_depth_cm(path: str) -> np.ndarray | None:
    if not path:
        return None
    try:
        depth = np.load(path).astype(np.float32, copy=False)
        depth = np.squeeze(depth)
    except Exception:
        return None
    if depth.ndim != 2:
        return None
    return depth


class ObstacleRepresentation3Predictor:
    def __init__(self, model_path: str | Path, *, device_name: str | None = None) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.checkpoint = _load_checkpoint(self.model_path, self.device)
        self.config = self.checkpoint.get("config", {})
        self.image_size = int(self.config.get("image_size", 96))
        self.labels = list(self.config.get("risk_states", RISK_STATES))
        self.projection_box = dict(self.config.get("projection_box", PROJECTION_BOX))
        self.stop_threshold = float(self.projection_box.get("stop_fraction_threshold", PROJECTION_BOX["stop_fraction_threshold"]))
        self.geometry_mean = np.asarray(
            self.checkpoint.get("geometry_mean", np.zeros(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32)),
            dtype=np.float32,
        )
        self.geometry_std = np.asarray(
            self.checkpoint.get("geometry_std", np.ones(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32)),
            dtype=np.float32,
        )
        self.geometry_std = np.where(np.abs(self.geometry_std) < 1e-6, 1.0, self.geometry_std).astype(np.float32)

    def predict(self, rgb_path: str | Path, event: Dict[str, Any]) -> Dict[str, Any]:
        rgb_path = Path(rgb_path).expanduser()
        if not rgb_path.is_file():
            raise FileNotFoundError(f"rgb image not found: {rgb_path}")
        geometry = event_geometry_vector(event)
        mean = self.geometry_mean if self.geometry_mean.shape == geometry.shape else np.zeros_like(geometry, dtype=np.float32)
        std = self.geometry_std if self.geometry_std.shape == geometry.shape else np.ones_like(geometry, dtype=np.float32)
        std = np.where(np.abs(std) < 1e-6, 1.0, std).astype(np.float32)
        rgb = torch.from_numpy(load_rgb(str(rgb_path), self.image_size).copy()).unsqueeze(0).to(self.device)
        depth_path = str(event.get("depth_npy_path") or event.get("depth_path") or "")
        depth = torch.from_numpy(load_depth(depth_path, self.image_size).copy()).unsqueeze(0).to(self.device)
        geometry_tensor = torch.from_numpy(((geometry - mean) / std).astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(rgb, depth, geometry_tensor)
            mask_prob = torch.sigmoid(output["mask_logits"])[0].detach().cpu().numpy()
            risk_prob = torch.softmax(output["risk_logits"], dim=1)[0].detach().cpu().numpy()
            can_forward_probability = float(torch.sigmoid(output["can_forward_logits"])[0].detach().cpu().item())
            must_stop_probability = float(torch.sigmoid(output["must_stop_logits"])[0].detach().cpu().item())
        selected_idx = int(np.argmax(risk_prob))
        raw_state = self.labels[selected_idx] if 0 <= selected_idx < len(self.labels) else "clear"
        clearance = mask_prob[0] if mask_prob.ndim == 3 else np.zeros((self.image_size, self.image_size), dtype=np.float32)
        warning = mask_prob[1] if mask_prob.ndim == 3 and mask_prob.shape[0] > 1 else np.zeros_like(clearance)
        stop = mask_prob[2] if mask_prob.ndim == 3 and mask_prob.shape[0] > 2 else np.zeros_like(clearance)
        model_masks = np.stack([clearance, warning, stop]).astype(np.float32, copy=False)
        model_box = projection_box_stats(model_masks, self.projection_box)

        depth_teacher_masks = None
        depth_box = {
            "front_box_clearance_fraction": 0.0,
            "front_box_warning_fraction": 0.0,
            "front_box_stop_fraction": 0.0,
            "front_box_pixel_count": int(model_box.get("front_box_pixel_count", 0)),
        }
        raw_depth = _load_depth_cm(depth_path)
        if raw_depth is not None:
            depth_teacher_masks = depth_masks(raw_depth, self.image_size)
            depth_box = projection_box_stats(depth_teacher_masks, self.projection_box)
            clearance = np.maximum(clearance, depth_teacher_masks[0])
            warning = np.maximum(warning, depth_teacher_masks[1])
            stop = np.maximum(stop, depth_teacher_masks[2])
        masks = np.stack([clearance, warning, stop]).astype(np.float32, copy=False)
        box = projection_box_stats(masks, self.projection_box)
        full_clearance_fraction = _fraction(clearance)
        full_warning_fraction = _fraction(warning)
        full_stop_fraction = _fraction(stop)
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        front_min = float(summary.get("front_min_depth_cm", 0.0) or 0.0)

        raw_rank = self.labels.index(raw_state) if raw_state in self.labels else 0
        clearance_rank = self.labels.index("clearance_warning") if "clearance_warning" in self.labels else 1
        warning_rank = self.labels.index("obstacle_warning") if "obstacle_warning" in self.labels else 2
        front_risk_state = raw_state
        safety_reason = "raw_model"
        depth_box_stop_fraction = float(depth_box["front_box_stop_fraction"])
        depth_projection_box_override = depth_box_stop_fraction > self.stop_threshold
        if depth_projection_box_override:
            front_risk_state = "must_stop"
            safety_reason = "depth_projection_box_stop"
        elif float(model_box["front_box_stop_fraction"]) > self.stop_threshold:
            front_risk_state = "must_stop"
            safety_reason = "projection_box_stop_mask"
        elif (
            full_stop_fraction > 0.0
            or float(box["front_box_warning_fraction"]) > 0.0
            or full_warning_fraction > 0.0
            or (front_min > 0.0 and front_min <= 250.0)
            or raw_state == "must_stop"
            or must_stop_probability >= 0.5
        ):
            front_risk_state = "obstacle_warning"
            safety_reason = "must_stop_shielded_to_warning"
        elif (
            float(box["front_box_clearance_fraction"]) > 0.0
            or full_clearance_fraction > 0.0
            or (front_min > 250.0 and front_min <= 450.0)
            or raw_rank >= clearance_rank
        ):
            front_risk_state = "clearance_warning" if raw_rank < warning_rank else raw_state
            safety_reason = "clearance_or_raw_warning"

        must_stop = front_risk_state == "must_stop"
        can_forward = bool((not must_stop) and can_forward_probability >= 0.35)
        reason = (
            f"risk={front_risk_state}; can_forward={can_forward}; must_stop={must_stop}; "
            f"front_box_clearance={float(box['front_box_clearance_fraction']):.3f}; "
            f"front_box_warning={float(box['front_box_warning_fraction']):.3f}; "
            f"front_box_stop={float(box['front_box_stop_fraction']):.3f}; "
            f"full_stop={full_stop_fraction:.3f}; front_min={front_min:.1f}cm; gate={safety_reason}"
        )
        return {
            "model_path": str(self.model_path),
            "model_version": str(self.config.get("model_version", "a_plus_3_v1")),
            "rgb_path": str(rgb_path),
            "depth_path": depth_path,
            "clearance_warning_mask": clearance.astype(np.float32),
            "obstacle_warning_mask": warning.astype(np.float32),
            "must_stop_mask": stop.astype(np.float32),
            "risk_probabilities": {label: float(risk_prob[idx]) for idx, label in enumerate(self.labels)},
            "raw_front_risk_state": raw_state,
            "front_risk_state": front_risk_state,
            "can_forward": bool(can_forward),
            "must_stop": bool(must_stop),
            "can_forward_probability": can_forward_probability,
            "must_stop_probability": must_stop_probability,
            "projection_box": dict(self.projection_box),
            "projection_box_stop_threshold": self.stop_threshold,
            "front_box_clearance_fraction": float(box["front_box_clearance_fraction"]),
            "front_box_warning_fraction": float(box["front_box_warning_fraction"]),
            "front_box_stop_fraction": float(box["front_box_stop_fraction"]),
            "model_front_box_clearance_fraction": float(model_box["front_box_clearance_fraction"]),
            "model_front_box_warning_fraction": float(model_box["front_box_warning_fraction"]),
            "model_front_box_stop_fraction": float(model_box["front_box_stop_fraction"]),
            "depth_front_box_clearance_fraction": float(depth_box["front_box_clearance_fraction"]),
            "depth_front_box_warning_fraction": float(depth_box["front_box_warning_fraction"]),
            "depth_front_box_stop_fraction": depth_box_stop_fraction,
            "depth_projection_box_override": bool(depth_projection_box_override),
            "depth_available": bool(depth_teacher_masks is not None),
            "front_box_pixel_count": int(box["front_box_pixel_count"]),
            "front_clearance_fraction": float(box["front_box_clearance_fraction"]),
            "front_warning_fraction": float(box["front_box_warning_fraction"]),
            "front_stop_fraction": float(box["front_box_stop_fraction"]),
            "full_clearance_fraction": full_clearance_fraction,
            "full_warning_fraction": full_warning_fraction,
            "full_stop_fraction": full_stop_fraction,
            "reason": reason,
            "device": str(self.device),
            "image_size": int(self.image_size),
        }


def predict_obstacle_representation_3(
    model_path: str | Path,
    rgb_path: str | Path,
    event: Dict[str, Any],
    *,
    device_name: str | None = None,
) -> Dict[str, Any]:
    return ObstacleRepresentation3Predictor(model_path, device_name=device_name).predict(rgb_path, event)


def render_affordance_overlay(rgb_image: Any, prediction: Dict[str, Any], *, alpha: float = 0.72) -> np.ndarray:
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb_image must be HxWx3")
    rgb = rgb[..., :3].astype(np.uint8, copy=False)
    h, w = rgb.shape[:2]
    clearance = _resize_mask(np.asarray(prediction.get("clearance_warning_mask", np.zeros((h, w))), dtype=np.float32), w, h)
    warning = _resize_mask(np.asarray(prediction.get("obstacle_warning_mask", np.zeros((h, w))), dtype=np.float32), w, h)
    stop = _resize_mask(np.asarray(prediction.get("must_stop_mask", np.zeros((h, w))), dtype=np.float32), w, h)
    canvas = (rgb.astype(np.float32) * 0.35).astype(np.float32)
    for mask, label in (
        (clearance > 0.5, "clearance_warning"),
        (warning > 0.5, "obstacle_warning"),
        (stop > 0.5, "must_stop"),
    ):
        color = np.asarray(RISK_COLORS[label], dtype=np.float32)
        canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha
    state = str(prediction.get("front_risk_state", "clear"))
    canvas[: max(10, h // 18), :] = np.asarray(RISK_COLORS.get(state, RISK_COLORS["clear"]), dtype=np.float32)

    box = dict(prediction.get("projection_box", PROJECTION_BOX))
    y_slice, x_slice = projection_box_slices((h, w), box)
    color = np.asarray(RISK_COLORS["must_stop"], dtype=np.float32)
    y0, y1 = y_slice.start or 0, y_slice.stop or h
    x0, x1 = x_slice.start or 0, x_slice.stop or w
    thickness = max(1, min(h, w) // 80)
    canvas[max(0, y0 - thickness) : min(h, y0 + thickness), x0:x1] = color
    canvas[max(0, y1 - thickness) : min(h, y1 + thickness), x0:x1] = color
    canvas[y0:y1, max(0, x0 - thickness) : min(w, x0 + thickness)] = color
    canvas[y0:y1, max(0, x1 - thickness) : min(w, x1 + thickness)] = color
    return np.clip(canvas, 0, 255).astype(np.uint8)


__all__ = ["ObstacleRepresentation3Predictor", "predict_obstacle_representation_3", "render_affordance_overlay"]
