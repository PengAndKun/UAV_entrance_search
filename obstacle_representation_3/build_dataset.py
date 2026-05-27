from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from .schema import (
    CLEARANCE_DEPTH_CM,
    GEOMETRY_FEATURE_NAMES,
    PROJECTION_BOX,
    RISK_STATES,
    RISK_TO_INDEX,
    STOP_DEPTH_CM,
    WARNING_DEPTH_CM,
    event_geometry_vector,
)
from .teacher import compute_affordance_teacher


DEFAULT_DATA_ROOTS = [
    Path("obstacle_avoidance_llm_data"),
    Path("obstacle_avoidance_2_data"),
    Path("obstacle_avoidance_3_data"),
    Path("obstacle_representation_2_data"),
    Path("obstacle_avoidance_data"),
    Path("route6_explore_runs"),
    Path("llm_route_7_fusion_runs"),
    Path("llm_route5_fusion_runs"),
]


def write_json(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def resolve_path(value: Any, *, event_file: Path) -> Path:
    if value is None:
        return Path()
    path = Path(str(value))
    if path.is_absolute():
        return path
    return event_file.parent / path


def iter_event_files(data_roots: Sequence[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in data_roots:
        root_path = Path(root)
        if root_path.is_file() and root_path.suffix.lower() == ".jsonl":
            candidates = [root_path]
        elif root_path.is_dir():
            candidates = sorted(root_path.rglob("*.jsonl"))
        else:
            candidates = []
        for candidate in candidates:
            try:
                key = candidate.resolve()
            except Exception:
                key = candidate
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def iter_events(data_roots: Sequence[Path]) -> Iterator[Tuple[Path, int, Dict[str, Any]]]:
    for event_file in iter_event_files(data_roots):
        try:
            lines = event_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except Exception:
                continue
            if isinstance(event, dict):
                yield event_file, line_no, event


def sample_group_id(event_file: Path, event: Dict[str, Any]) -> str:
    for key in ("group_id", "session_id", "run_id", "route_id", "trajectory_id"):
        value = event.get(key)
        if value not in (None, ""):
            return str(value)
    parent = event_file.parent
    if parent.name:
        return str(parent)
    return str(event_file)


def split_indices_by_group_label(labels: Sequence[str], groups: Sequence[str], *, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    group_to_label: Dict[str, str] = {}
    group_order: List[str] = []
    for label, group in zip(labels, groups):
        group_key = str(group)
        if group_key not in group_to_label:
            group_to_label[group_key] = str(label)
            group_order.append(group_key)

    label_to_groups: Dict[str, List[str]] = {}
    for group in group_order:
        label_to_groups.setdefault(group_to_label[group], []).append(group)

    group_to_split: Dict[str, str] = {}
    for label in sorted(label_to_groups):
        label_groups = label_to_groups[label][:]
        rng.shuffle(label_groups)
        total = len(label_groups)
        if total == 1:
            train_count, val_count = 1, 0
        elif total == 2:
            train_count, val_count = 1, 1
        else:
            val_count = max(1, int(round(total * 0.1)))
            test_count = max(1, int(round(total * 0.1)))
            train_count = max(1, total - val_count - test_count)
            while train_count + val_count + test_count > total:
                if val_count >= test_count and val_count > 1:
                    val_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    train_count -= 1
        for idx, group in enumerate(label_groups):
            if idx < train_count:
                split = "train"
            elif idx < train_count + val_count:
                split = "val"
            else:
                split = "test"
            group_to_split[group] = split

    return [group_to_split.get(str(group), "train") for group in groups]


def counts(items: Iterable[Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in items:
        key = str(item)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def load_depth_array(path: Path) -> np.ndarray | None:
    try:
        depth = np.load(str(path)).astype(np.float32, copy=False)
    except Exception:
        return None
    depth = np.squeeze(depth)
    if depth.ndim != 2 or depth.size == 0:
        return None
    return depth


def source_root_for_event(event_file: Path, data_roots: Sequence[Path]) -> str:
    for root in data_roots:
        root_path = Path(root)
        try:
            if root_path == event_file or root_path in event_file.parents:
                return str(root_path)
        except Exception:
            continue
    return str(data_roots[0]) if data_roots else ""


def _projection_box_array() -> np.ndarray:
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


def build_dataset(
    data_roots: Sequence[Path | str],
    *,
    output_root: Path | str = Path("obstacle_representation_3_data"),
    image_size: int = 96,
    seed: int = 42,
) -> Dict[str, Any]:
    data_root_paths = [Path(root) for root in data_roots]
    output_root_path = Path(output_root)
    dataset_dir = output_root_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_dir / "a_plus_3_dataset_latest.jsonl"
    npz_path = dataset_dir / "a_plus_3_dataset_latest.npz"
    summary_path = dataset_dir / "a_plus_3_dataset_summary.json"
    if jsonl_path.exists():
        jsonl_path.unlink()

    samples: List[Dict[str, Any]] = []
    mask_targets: List[np.ndarray] = []
    missing_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    for event_file, line_no, raw_event in iter_events(data_root_paths):
        event = dict(raw_event)
        rgb_path = resolve_path(event.get("rgb_path") or event.get("image_path"), event_file=event_file)
        depth_path = resolve_path(
            event.get("depth_npy_path") or event.get("depth_cm_path") or event.get("depth_path"),
            event_file=event_file,
        )
        if not rgb_path.is_file():
            missing_counts["missing_rgb"] = missing_counts.get("missing_rgb", 0) + 1
            continue
        if not depth_path.is_file():
            missing_counts["missing_depth"] = missing_counts.get("missing_depth", 0) + 1
            continue
        pointcloud_summary = event.get("pointcloud_summary")
        if not isinstance(pointcloud_summary, dict):
            pointcloud_summary = event.get("depth_obstacle_summary")
            if isinstance(pointcloud_summary, dict):
                event["pointcloud_summary"] = pointcloud_summary
        if not isinstance(pointcloud_summary, dict):
            missing_counts["missing_pointcloud_summary"] = missing_counts.get("missing_pointcloud_summary", 0) + 1
            continue
        depth = load_depth_array(depth_path)
        if depth is None:
            missing_counts["invalid_depth"] = missing_counts.get("invalid_depth", 0) + 1
            continue

        teacher = compute_affordance_teacher(event, depth, image_size=int(image_size))
        source_root = source_root_for_event(event_file, data_root_paths)
        source_counts[source_root] = source_counts.get(source_root, 0) + 1
        group_id = sample_group_id(event_file, event)
        sample = {
            "sample_id": f"{event_file.parent.name}:{line_no}",
            "event_file": str(event_file),
            "event_line": int(line_no),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "group_id": group_id,
            "source_root": source_root,
            "front_risk_state": teacher["front_risk_state"],
            "front_risk_index": int(teacher["front_risk_index"]),
            "can_forward": bool(teacher["can_forward"]),
            "must_stop": bool(teacher["must_stop"]),
            "projection_box": teacher["projection_box"],
            "front_box_clearance_fraction": float(teacher["front_box_clearance_fraction"]),
            "front_box_warning_fraction": float(teacher["front_box_warning_fraction"]),
            "front_box_stop_fraction": float(teacher["front_box_stop_fraction"]),
            "front_box_pixel_count": int(teacher["front_box_pixel_count"]),
            "front_clearance_fraction": float(teacher["front_clearance_fraction"]),
            "front_warning_fraction": float(teacher["front_warning_fraction"]),
            "front_stop_fraction": float(teacher["front_stop_fraction"]),
            "full_stop_fraction": float(teacher["full_stop_fraction"]),
            "full_warning_fraction": float(teacher["full_warning_fraction"]),
            "full_clearance_fraction": float(teacher["full_clearance_fraction"]),
            "teacher_source": str(teacher["teacher_source"]),
            "geometry_features": event_geometry_vector(event).astype(float).tolist(),
            "pointcloud_summary": pointcloud_summary,
            "relative_target": event.get("relative_target", {}),
        }
        samples.append(sample)
        mask_targets.append(teacher["masks"].astype(np.float32, copy=False))

    if not samples:
        raise ValueError(
            "no OR3 samples built; expected events with existing rgb_path, depth .npy path, "
            f"and pointcloud_summary under data_roots={[str(root) for root in data_root_paths]}; "
            f"missing_counts={dict(sorted(missing_counts.items()))}"
        )

    labels = [sample["front_risk_state"] for sample in samples]
    groups = [sample["group_id"] for sample in samples]
    splits = split_indices_by_group_label(labels, groups, seed=int(seed))
    for sample, split in zip(samples, splits):
        sample["split"] = split
        append_jsonl(jsonl_path, sample)

    image_paths = np.asarray([sample["rgb_path"] for sample in samples], dtype=object)
    depth_paths = np.asarray([sample["depth_path"] for sample in samples], dtype=object)
    geometry = np.asarray([sample["geometry_features"] for sample in samples], dtype=np.float32)
    risk_indices = np.asarray([sample["front_risk_index"] for sample in samples], dtype=np.int64)
    can_forward = np.asarray([sample["can_forward"] for sample in samples], dtype=bool)
    must_stop = np.asarray([sample["must_stop"] for sample in samples], dtype=bool)
    front_box_stop_fraction = np.asarray([sample["front_box_stop_fraction"] for sample in samples], dtype=np.float32)
    front_box_warning_fraction = np.asarray([sample["front_box_warning_fraction"] for sample in samples], dtype=np.float32)
    front_box_clearance_fraction = np.asarray([sample["front_box_clearance_fraction"] for sample in samples], dtype=np.float32)
    split_array = np.asarray(splits, dtype=object)
    group_ids = np.asarray(groups, dtype=object)

    np.savez_compressed(
        npz_path,
        image_paths=image_paths,
        depth_paths=depth_paths,
        geometry=geometry,
        mask_targets=np.asarray(mask_targets, dtype=np.float32),
        risk_indices=risk_indices,
        can_forward=can_forward,
        must_stop=must_stop,
        front_box_stop_fraction=front_box_stop_fraction,
        front_box_warning_fraction=front_box_warning_fraction,
        front_box_clearance_fraction=front_box_clearance_fraction,
        splits=split_array,
        group_ids=group_ids,
        risk_states=np.asarray(RISK_STATES, dtype=object),
        geometry_feature_names=np.asarray(GEOMETRY_FEATURE_NAMES, dtype=object),
        projection_box=_projection_box_array(),
        image_size=np.asarray([int(image_size)], dtype=np.int64),
    )

    group_to_split = {group: split for group, split in zip(groups, splits)}
    summary = {
        "dataset_path": str(npz_path),
        "dataset_jsonl_path": str(jsonl_path),
        "total_samples": len(samples),
        "data_roots": [str(root) for root in data_root_paths],
        "source_counts": dict(sorted(source_counts.items())),
        "risk_state_counts": {label: int(counts(labels).get(label, 0)) for label in RISK_STATES},
        "split_counts": counts(splits),
        "group_split_counts": counts(group_to_split.values()),
        "can_forward_count": int(np.count_nonzero(can_forward)),
        "must_stop_count": int(np.count_nonzero(must_stop)),
        "projection_box_stop_support_count": int(
            np.count_nonzero(front_box_stop_fraction > float(PROJECTION_BOX["stop_fraction_threshold"]))
        ),
        "teacher_source_counts": counts(sample["teacher_source"] for sample in samples),
        "missing_counts": dict(sorted(missing_counts.items())),
        "risk_map": dict(RISK_TO_INDEX),
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "projection_box": {key: float(value) for key, value in PROJECTION_BOX.items()},
        "image_size": int(image_size),
        "thresholds": {
            "stop_depth_cm": float(STOP_DEPTH_CM),
            "warning_depth_cm": float(WARNING_DEPTH_CM),
            "clearance_depth_cm": float(CLEARANCE_DEPTH_CM),
        },
        "stop_depth_cm": float(STOP_DEPTH_CM),
        "warning_depth_cm": float(WARNING_DEPTH_CM),
        "clearance_depth_cm": float(CLEARANCE_DEPTH_CM),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Obstacle Representation 3 affordance dataset.")
    parser.add_argument("--data-roots", nargs="+", default=[str(root) for root in DEFAULT_DATA_ROOTS])
    parser.add_argument("--output-root", default="obstacle_representation_3_data")
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
