from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from obstacle_representation_3.schema import (
    CLEARANCE_DEPTH_CM,
    GEOMETRY_FEATURE_NAMES,
    PROJECTION_BOX,
    RISK_STATES,
    RISK_TO_INDEX,
    STOP_DEPTH_CM,
    WARNING_DEPTH_CM,
    event_geometry_vector,
)
from obstacle_representation_3.teacher import compute_affordance_teacher, projection_box_slices, projection_box_stats

from . import MODEL_VERSION


DEFAULT_SOURCE_JSONL = Path("obstacle_representation_3_data/datasets/a_plus_3_dataset_latest.jsonl")
DEFAULT_OUTPUT_ROOT = Path("obstacle_representation_3_data")
RGB_OBSTACLE_SCORE_THRESHOLD = 0.55


def write_json(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if isinstance(item, dict):
            samples.append(item)
    return samples


def load_depth_array(path: str | Path) -> np.ndarray:
    depth = np.load(str(path)).astype(np.float32, copy=False)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"expected 2D depth array: {path}")
    return depth


def load_rgb_array(path: str | Path, image_size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((int(image_size), int(image_size)), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def rgb_projection_features(rgb: np.ndarray, projection_box: Dict[str, float] | None = None) -> Dict[str, float]:
    box = projection_box or PROJECTION_BOX
    y_slice, x_slice = projection_box_slices(rgb.shape[:2], box)
    center = np.asarray(rgb[y_slice, x_slice, :], dtype=np.float32)
    full = np.asarray(rgb, dtype=np.float32)
    if center.size == 0:
        return {
            "rgb_center_obstacle_score": 0.0,
            "rgb_center_edge_strength": 0.0,
            "rgb_center_contrast": 0.0,
            "rgb_center_saturation": 0.0,
            "rgb_center_darkness": 0.0,
        }
    center_luma = 0.2126 * center[..., 0] + 0.7152 * center[..., 1] + 0.0722 * center[..., 2]
    full_luma = 0.2126 * full[..., 0] + 0.7152 * full[..., 1] + 0.0722 * full[..., 2]
    dx = np.abs(np.diff(center_luma, axis=1)).mean() if center_luma.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(center_luma, axis=0)).mean() if center_luma.shape[0] > 1 else 0.0
    edge_strength = float(dx + dy)
    contrast = float(abs(float(center_luma.mean()) - float(full_luma.mean())) + float(center_luma.std()))
    saturation = float((center.max(axis=2) - center.min(axis=2)).mean())
    darkness = float(max(0.0, 1.0 - float(center_luma.mean())))
    score = float(np.clip(edge_strength * 2.8 + contrast * 1.3 + saturation * 0.5 + max(0.0, darkness - 0.45) * 0.45, 0.0, 1.0))
    return {
        "rgb_center_obstacle_score": score,
        "rgb_center_edge_strength": edge_strength,
        "rgb_center_contrast": contrast,
        "rgb_center_saturation": saturation,
        "rgb_center_darkness": darkness,
    }


def _front_min_cm(sample: Dict[str, Any]) -> float:
    summary = sample.get("pointcloud_summary") if isinstance(sample.get("pointcloud_summary"), dict) else {}
    try:
        return float(summary.get("front_min_depth_cm", 0.0) or 0.0)
    except Exception:
        return 0.0


def _geometry_features(sample: Dict[str, Any]) -> np.ndarray:
    values = sample.get("geometry_features")
    if isinstance(values, list) and len(values) == len(GEOMETRY_FEATURE_NAMES):
        return np.asarray(values, dtype=np.float32)
    return event_geometry_vector(sample)


def _review_sample_with_masks(sample: Dict[str, Any], *, image_size: int = 96) -> Tuple[Dict[str, Any], np.ndarray]:
    depth = load_depth_array(sample["depth_path"])
    rgb = load_rgb_array(sample["rgb_path"], image_size)
    event = dict(sample)
    if "pointcloud_summary" not in event and isinstance(sample.get("depth_obstacle_summary"), dict):
        event["pointcloud_summary"] = sample["depth_obstacle_summary"]
    teacher = compute_affordance_teacher(event, depth, image_size=int(image_size))
    masks = np.asarray(teacher["masks"], dtype=np.float32).copy()
    rgb_features = rgb_projection_features(rgb, teacher.get("projection_box", PROJECTION_BOX))
    rgb_score = float(rgb_features["rgb_center_obstacle_score"])
    front_min = _front_min_cm(sample)

    base_state = str(teacher["front_risk_state"])
    state = base_state
    confidence = 0.95
    needs_review = False
    reasons: List[str] = []

    if base_state == "must_stop":
        confidence = 1.0
        reasons.append("depth_projection_box_stop")
    elif base_state == "obstacle_warning":
        confidence = 0.9
        if float(teacher["front_box_stop_fraction"]) <= float(PROJECTION_BOX["stop_fraction_threshold"]) and (
            float(teacher["full_stop_fraction"]) > 0.0 or (0.0 < front_min <= STOP_DEPTH_CM)
        ):
            reasons.append("depth_stop_outside_projection_box")
        else:
            reasons.append("depth_obstacle_warning")
    elif base_state == "clearance_warning":
        confidence = 0.85
        reasons.append("depth_clearance_warning")
    else:
        confidence = 0.95
        reasons.append("depth_clear")

    if base_state == "clear" and rgb_score >= RGB_OBSTACLE_SCORE_THRESHOLD:
        y_slice, x_slice = projection_box_slices(masks.shape[-2:], teacher.get("projection_box", PROJECTION_BOX))
        masks[0, y_slice, x_slice] = 1.0
        state = "clearance_warning"
        confidence = 0.58
        needs_review = True
        reasons.append("rgb_center_obstacle_evidence_without_depth_support")
    elif base_state == "clearance_warning" and rgb_score >= 0.72 and 0.0 < front_min <= WARNING_DEPTH_CM:
        y_slice, x_slice = projection_box_slices(masks.shape[-2:], teacher.get("projection_box", PROJECTION_BOX))
        masks[1, y_slice, x_slice] = 1.0
        state = "obstacle_warning"
        confidence = 0.66
        needs_review = True
        reasons.append("rgb_center_obstacle_evidence_reinforces_near_depth")

    stats = projection_box_stats(masks, teacher.get("projection_box", PROJECTION_BOX))
    review = {
        "or3_front_risk_state": str(sample.get("front_risk_state", base_state)),
        "depth_teacher_risk_state": base_state,
        "teacher_3_1_model_version": MODEL_VERSION,
        "teacher_3_1_risk_state": state,
        "teacher_3_1_risk_index": int(RISK_TO_INDEX[state]),
        "teacher_3_1_can_forward": state != "must_stop",
        "teacher_3_1_must_stop": state == "must_stop",
        "teacher_3_1_confidence": float(confidence),
        "teacher_3_1_needs_review": bool(needs_review),
        "teacher_3_1_reason": ";".join(reasons),
        "front_box_clearance_fraction": float(stats["front_box_clearance_fraction"]),
        "front_box_warning_fraction": float(stats["front_box_warning_fraction"]),
        "front_box_stop_fraction": float(stats["front_box_stop_fraction"]),
        "front_box_pixel_count": int(stats["front_box_pixel_count"]),
        "full_clearance_fraction": float(np.count_nonzero(masks[0] > 0.5) / max(1, masks[0].size)),
        "full_warning_fraction": float(np.count_nonzero(masks[1] > 0.5) / max(1, masks[1].size)),
        "full_stop_fraction": float(np.count_nonzero(masks[2] > 0.5) / max(1, masks[2].size)),
        **rgb_features,
    }
    return review, masks


def review_sample(sample: Dict[str, Any], *, image_size: int = 96) -> Dict[str, Any]:
    review, _ = _review_sample_with_masks(sample, image_size=image_size)
    return review


def counts(values: Iterable[Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def projection_box_array() -> np.ndarray:
    return np.asarray(
        [
            float(PROJECTION_BOX["x0"]),
            float(PROJECTION_BOX["x1"]),
            float(PROJECTION_BOX["y0"]),
            float(PROJECTION_BOX["y1"]),
            float(PROJECTION_BOX["stop_fraction_threshold"]),
        ],
        dtype=np.float32,
    )


def build_or3_1_dataset(
    source_jsonl: Path | str = DEFAULT_SOURCE_JSONL,
    *,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    image_size: int = 96,
) -> Dict[str, Any]:
    source_jsonl = Path(source_jsonl)
    output_root = Path(output_root)
    dataset_dir = output_root / "datasets"
    review_dir = output_root / "or3_1_teacher_review"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_dir / "a_plus_3_1_dataset_latest.jsonl"
    npz_path = dataset_dir / "a_plus_3_1_dataset_latest.npz"
    summary_path = dataset_dir / "a_plus_3_1_dataset_summary.json"
    review_summary_path = review_dir / "a_plus_3_1_teacher_review_summary.json"
    changed_jsonl_path = review_dir / "a_plus_3_1_changed_samples.jsonl"
    needs_review_jsonl_path = review_dir / "a_plus_3_1_needs_review_samples.jsonl"

    samples = read_jsonl(source_jsonl)
    reviewed_samples: List[Dict[str, Any]] = []
    changed_samples: List[Dict[str, Any]] = []
    needs_review_samples: List[Dict[str, Any]] = []
    mask_targets: List[np.ndarray] = []
    geometry_rows: List[np.ndarray] = []

    for sample in samples:
        review, masks = _review_sample_with_masks(sample, image_size=int(image_size))
        reviewed = dict(sample)
        reviewed.update(review)
        changed = reviewed["or3_front_risk_state"] != reviewed["teacher_3_1_risk_state"]
        reviewed["teacher_3_1_changed_from_or3"] = bool(changed)
        reviewed["front_risk_state_3_1"] = reviewed["teacher_3_1_risk_state"]
        reviewed["front_risk_index_3_1"] = int(reviewed["teacher_3_1_risk_index"])
        reviewed["can_forward_3_1"] = bool(reviewed["teacher_3_1_can_forward"])
        reviewed["must_stop_3_1"] = bool(reviewed["teacher_3_1_must_stop"])
        reviewed_samples.append(reviewed)
        mask_targets.append(masks.astype(np.float32, copy=False))
        geometry_rows.append(_geometry_features(reviewed))
        if changed:
            changed_samples.append(reviewed)
        if reviewed["teacher_3_1_needs_review"]:
            needs_review_samples.append(reviewed)

    if not reviewed_samples:
        raise ValueError(f"no samples found in {source_jsonl}")

    image_paths = np.asarray([sample["rgb_path"] for sample in reviewed_samples], dtype=object)
    depth_paths = np.asarray([sample["depth_path"] for sample in reviewed_samples], dtype=object)
    splits = np.asarray([sample.get("split", "train") for sample in reviewed_samples], dtype=object)
    group_ids = np.asarray([sample.get("group_id", "") for sample in reviewed_samples], dtype=object)
    risk_indices = np.asarray([sample["teacher_3_1_risk_index"] for sample in reviewed_samples], dtype=np.int64)
    can_forward = np.asarray([sample["teacher_3_1_can_forward"] for sample in reviewed_samples], dtype=bool)
    must_stop = np.asarray([sample["teacher_3_1_must_stop"] for sample in reviewed_samples], dtype=bool)
    front_box_stop_fraction = np.asarray([sample["front_box_stop_fraction"] for sample in reviewed_samples], dtype=np.float32)
    front_box_warning_fraction = np.asarray([sample["front_box_warning_fraction"] for sample in reviewed_samples], dtype=np.float32)
    front_box_clearance_fraction = np.asarray([sample["front_box_clearance_fraction"] for sample in reviewed_samples], dtype=np.float32)
    needs_review = np.asarray([sample["teacher_3_1_needs_review"] for sample in reviewed_samples], dtype=bool)
    confidence = np.asarray([sample["teacher_3_1_confidence"] for sample in reviewed_samples], dtype=np.float32)

    write_jsonl(jsonl_path, reviewed_samples)
    write_jsonl(changed_jsonl_path, changed_samples)
    write_jsonl(needs_review_jsonl_path, needs_review_samples)
    np.savez_compressed(
        npz_path,
        image_paths=image_paths,
        depth_paths=depth_paths,
        geometry=np.asarray(geometry_rows, dtype=np.float32),
        mask_targets=np.asarray(mask_targets, dtype=np.float32),
        risk_indices=risk_indices,
        can_forward=can_forward,
        must_stop=must_stop,
        front_box_stop_fraction=front_box_stop_fraction,
        front_box_warning_fraction=front_box_warning_fraction,
        front_box_clearance_fraction=front_box_clearance_fraction,
        splits=splits,
        group_ids=group_ids,
        risk_states=np.asarray(RISK_STATES, dtype=object),
        geometry_feature_names=np.asarray(GEOMETRY_FEATURE_NAMES, dtype=object),
        projection_box=projection_box_array(),
        image_size=np.asarray([int(image_size)], dtype=np.int64),
        teacher_3_1_needs_review=needs_review,
        teacher_3_1_confidence=confidence,
    )

    labels = [sample["teacher_3_1_risk_state"] for sample in reviewed_samples]
    summary = {
        "dataset_path": str(npz_path),
        "dataset_jsonl_path": str(jsonl_path),
        "source_dataset_jsonl_path": str(source_jsonl),
        "teacher_review_dir": str(review_dir),
        "changed_samples_jsonl_path": str(changed_jsonl_path),
        "needs_review_jsonl_path": str(needs_review_jsonl_path),
        "model_version": MODEL_VERSION,
        "total_samples": len(reviewed_samples),
        "changed_label_count": len(changed_samples),
        "needs_review_count": len(needs_review_samples),
        "risk_state_counts": {label: int(counts(labels).get(label, 0)) for label in RISK_STATES},
        "split_counts": counts(splits.tolist()),
        "source_counts": counts(sample.get("source_root", "") for sample in reviewed_samples),
        "projection_box": dict(PROJECTION_BOX),
        "thresholds": {
            "stop_depth_cm": float(STOP_DEPTH_CM),
            "warning_depth_cm": float(WARNING_DEPTH_CM),
            "clearance_depth_cm": float(CLEARANCE_DEPTH_CM),
            "rgb_center_obstacle_score": float(RGB_OBSTACLE_SCORE_THRESHOLD),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(summary_path, summary)
    write_json(review_summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OR3.1 teacher-reviewed dataset.")
    parser.add_argument("--source-jsonl", default=str(DEFAULT_SOURCE_JSONL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()
    summary = build_or3_1_dataset(Path(args.source_jsonl), output_root=Path(args.output_root), image_size=args.image_size)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
