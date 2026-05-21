from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from obstacle_representation.build_dataset import iter_events, resolve_path, sample_group_id, split_indices_by_group_label, write_json

from .schema import GEOMETRY_FEATURE_NAMES, RISK_STATES, RISK_TO_INDEX, event_geometry_vector
from .teacher import compute_affordance_teacher


EVENT_FILENAMES = {"analysis_events.jsonl", "avoidance_events.jsonl"}


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_depth_array(path: Path) -> np.ndarray | None:
    try:
        depth = np.load(str(path)).astype(np.float32, copy=False)
    except Exception:
        return None
    depth = np.squeeze(depth)
    return depth if depth.ndim == 2 and depth.size else None


def source_root_for_event(event_file: Path, data_roots: Sequence[Path]) -> str:
    for root in data_roots:
        try:
            if root == event_file or root in event_file.parents:
                return str(root)
        except Exception:
            continue
    return str(data_roots[0]) if data_roots else ""


def build_dataset(
    data_roots: Sequence[Path],
    *,
    output_root: Path = Path("obstacle_representation_2_data"),
    image_size: int = 96,
    seed: int = 42,
) -> Dict[str, Any]:
    dataset_dir = output_root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_dir / "a_plus_2_dataset_latest.jsonl"
    npz_path = dataset_dir / "a_plus_2_dataset_latest.npz"
    summary_path = dataset_dir / "a_plus_2_dataset_summary.json"
    if jsonl_path.exists():
        jsonl_path.unlink()

    samples: List[Dict[str, Any]] = []
    missing_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    for event_file, line_no, event in iter_events(data_roots):
        rgb_path = resolve_path(event.get("rgb_path"), event_file=event_file)
        depth_path = resolve_path(event.get("depth_npy_path") or event.get("depth_cm_path"), event_file=event_file)
        if not rgb_path.is_file():
            missing_counts["missing_rgb"] = missing_counts.get("missing_rgb", 0) + 1
            continue
        if not depth_path.is_file():
            missing_counts["missing_depth"] = missing_counts.get("missing_depth", 0) + 1
            continue
        if not isinstance(event.get("pointcloud_summary"), dict):
            missing_counts["missing_pointcloud_summary"] = missing_counts.get("missing_pointcloud_summary", 0) + 1
            continue
        depth = load_depth_array(depth_path)
        if depth is None:
            missing_counts["invalid_depth"] = missing_counts.get("invalid_depth", 0) + 1
            continue
        teacher = compute_affordance_teacher(event, depth, image_size=image_size)
        source_root = source_root_for_event(event_file, data_roots)
        source_counts[source_root] = source_counts.get(source_root, 0) + 1
        group_id = str(event.get("group_id") or sample_group_id(event_file, event))
        sample = {
            "sample_id": f"{event_file.parent.name}:{line_no}",
            "event_file": str(event_file),
            "event_line": int(line_no),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "group_id": group_id,
            "source_root": source_root,
            "direction_label": teacher["direction_label"],
            "direction_index": int(teacher["direction_index"]),
            "direction_scores": teacher["direction_scores"],
            "flyover_delta_cm": float(teacher["flyover_delta_cm"]),
            "red_front_blocked": bool(teacher["red_front_blocked"]),
            "front_red_fraction": float(teacher["front_red_fraction"]),
            "front_insufficient_fraction": float(teacher["front_insufficient_fraction"]),
            "front_risk_state": teacher["front_risk_state"],
            "front_risk_index": int(teacher["front_risk_index"]),
            "can_forward": bool(teacher["can_forward"]),
            "must_stop": bool(teacher["must_stop"]),
            "front_clearance_fraction": float(teacher["front_clearance_fraction"]),
            "front_warning_fraction": float(teacher["front_warning_fraction"]),
            "front_stop_fraction": float(teacher["front_stop_fraction"]),
            "teacher_source": str(teacher["teacher_source"]),
            "geometry_features": event_geometry_vector(event).astype(float).tolist(),
            "pointcloud_summary": event.get("pointcloud_summary", {}),
            "relative_target": event.get("relative_target", {}),
        }
        samples.append(sample)

    if not samples:
        raise ValueError(f"no OR2 samples built from data_roots={data_roots}")

    labels = [sample["front_risk_state"] for sample in samples]
    groups = [sample["group_id"] for sample in samples]
    splits = split_indices_by_group_label(labels, groups, seed=seed)
    masks = []
    for sample, split in zip(samples, splits):
        sample["split"] = split
        append_jsonl(jsonl_path, sample)
        depth = load_depth_array(Path(sample["depth_path"]))
        teacher = compute_affordance_teacher(
            {
                "pointcloud_summary": sample["pointcloud_summary"],
                "relative_target": sample["relative_target"],
                "distance_to_goal_cm": sample.get("distance_to_goal_cm", 0.0),
            },
            depth if depth is not None else np.zeros((image_size, image_size), dtype=np.float32),
            image_size=image_size,
        )
        masks.append(teacher["masks"].astype(np.float32, copy=False))

    image_paths = np.asarray([sample["rgb_path"] for sample in samples], dtype=object)
    depth_paths = np.asarray([sample["depth_path"] for sample in samples], dtype=object)
    geometry = np.asarray([sample["geometry_features"] for sample in samples], dtype=np.float32)
    mask_targets = np.asarray(masks, dtype=np.float32)
    direction_indices = np.asarray([sample["direction_index"] for sample in samples], dtype=np.int64)
    direction_scores = np.asarray(
        [[sample["direction_scores"].get(label, 0.0) for label in ("forward", "left", "right", "up", "backoff", "hold")] for sample in samples],
        dtype=np.float32,
    )
    flyover_delta_cm = np.asarray([sample["flyover_delta_cm"] for sample in samples], dtype=np.float32)
    red_front_blocked = np.asarray([sample["red_front_blocked"] for sample in samples], dtype=bool)
    risk_indices = np.asarray([sample["front_risk_index"] for sample in samples], dtype=np.int64)
    can_forward = np.asarray([sample["can_forward"] for sample in samples], dtype=bool)
    must_stop = np.asarray([sample["must_stop"] for sample in samples], dtype=bool)
    split_array = np.asarray(splits, dtype=object)
    group_ids = np.asarray(groups, dtype=object)
    np.savez_compressed(
        npz_path,
        image_paths=image_paths,
        depth_paths=depth_paths,
        geometry=geometry,
        mask_targets=mask_targets,
        direction_indices=direction_indices,
        direction_scores=direction_scores,
        flyover_delta_cm=flyover_delta_cm,
        red_front_blocked=red_front_blocked,
        risk_indices=risk_indices,
        can_forward=can_forward,
        must_stop=must_stop,
        splits=split_array,
        group_ids=group_ids,
        direction_labels=np.asarray(("forward", "left", "right", "up", "backoff", "hold"), dtype=object),
        risk_states=np.asarray(RISK_STATES, dtype=object),
        geometry_feature_names=np.asarray(GEOMETRY_FEATURE_NAMES, dtype=object),
        image_size=np.asarray([int(image_size)], dtype=np.int64),
    )

    def counts(items: Iterable[Any]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for item in items:
            key = str(item)
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    group_splits = {group: splits[idx] for idx, group in enumerate(groups)}
    summary = {
        "dataset_path": str(npz_path),
        "dataset_jsonl_path": str(jsonl_path),
        "total_samples": len(samples),
        "data_roots": [str(root) for root in data_roots],
        "source_counts": source_counts,
        "risk_state_counts": {label: int(counts(labels).get(label, 0)) for label in RISK_STATES},
        "legacy_direction_label_counts": counts(sample["direction_label"] for sample in samples),
        "split_counts": counts(splits),
        "group_split_counts": counts(group_splits.values()),
        "can_forward_count": int(np.count_nonzero(can_forward)),
        "must_stop_count": int(np.count_nonzero(must_stop)),
        "teacher_source_counts": counts(sample["teacher_source"] for sample in samples),
        "missing_counts": dict(sorted(missing_counts.items())),
        "risk_map": RISK_TO_INDEX,
        "geometry_feature_names": GEOMETRY_FEATURE_NAMES,
        "image_size": int(image_size),
        "stop_depth_cm": 100.0,
        "warning_depth_cm": 250.0,
        "clearance_depth_cm": 450.0,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Obstacle Representation 2 affordance dataset.")
    parser.add_argument("--data-roots", nargs="+", default=["obstacle_avoidance_llm_data", "obstacle_avoidance_2_data"])
    parser.add_argument("--output-root", default="obstacle_representation_2_data")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_dataset(
        [Path(item) for item in args.data_roots],
        output_root=Path(args.output_root),
        image_size=args.image_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
