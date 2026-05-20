from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .schema import GEOMETRY_FEATURE_NAMES, LABEL_TO_INDEX, OBSTACLE_LABELS, counts, geometry_vector
from .teacher_labels import teacher_label_from_event


EVENT_FILENAMES = {"analysis_events.jsonl", "avoidance_events.jsonl"}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def resolve_path(value: Any, *, event_file: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("__missing_obstacle_representation_path__")
    path = Path(text)
    if path.is_absolute():
        return path
    return (event_file.parent / path).resolve()


def iter_event_files(data_roots: Sequence[Path]) -> Iterable[Path]:
    for root in data_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            if path.name in EVENT_FILENAMES:
                yield path


def iter_events(data_roots: Sequence[Path]) -> Iterable[tuple[Path, int, Dict[str, Any]]]:
    for path in iter_event_files(data_roots):
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict):
                yield path, line_no, event


def split_indices_by_label(labels: List[str], *, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)
    splits = ["train"] * len(labels)
    for _label, indices in by_label.items():
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            train_count, val_count = 1, 0
        elif n == 2:
            train_count, val_count = 1, 1
        else:
            train_count = max(1, int(round(n * 0.70)))
            val_count = max(1, int(round(n * 0.15)))
            if train_count + val_count >= n:
                train_count = max(1, n - 2)
                val_count = 1
        for pos, sample_idx in enumerate(indices):
            if pos < train_count:
                splits[sample_idx] = "train"
            elif pos < train_count + val_count:
                splits[sample_idx] = "val"
            else:
                splits[sample_idx] = "test"
    return splits


def build_dataset(
    data_roots: Sequence[Path],
    *,
    output_root: Path = Path("obstacle_representation_data"),
    seed: int = 42,
) -> Dict[str, Any]:
    dataset_dir = output_root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_dir / "dataset_latest.jsonl"
    npz_path = dataset_dir / "dataset_latest.npz"
    summary_path = dataset_dir / "dataset_summary.json"
    if jsonl_path.exists():
        jsonl_path.unlink()

    samples: List[Dict[str, Any]] = []
    missing_counts: Dict[str, int] = {}
    raw_labels: List[str] = []
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
        teacher = teacher_label_from_event(event)
        label = str(teacher["label"])
        raw_labels.append(str(teacher.get("raw_label", "")))
        source_root = str(next((root for root in data_roots if root in event_file.parents or root == event_file), data_roots[0]))
        source_counts[source_root] = source_counts.get(source_root, 0) + 1
        sample = {
            "sample_id": f"{event_file.parent.name}:{line_no}",
            "event_file": str(event_file),
            "event_line": line_no,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "label": label,
            "label_index": int(teacher["label_index"]),
            "raw_label": str(teacher.get("raw_label", "")),
            "teacher_source": str(teacher.get("teacher_source", "")),
            "flyover_recommended": bool(teacher.get("flyover_recommended", False)),
            "geometry_features": geometry_vector(event).astype(float).tolist(),
            "pointcloud_summary": event.get("pointcloud_summary", {}),
            "relative_target": event.get("relative_target", {}),
            "source_root": source_root,
        }
        samples.append(sample)

    labels = [sample["label"] for sample in samples]
    splits = split_indices_by_label(labels, seed=seed)
    for sample, split in zip(samples, splits):
        sample["split"] = split
        append_jsonl(jsonl_path, sample)

    image_paths = np.asarray([sample["rgb_path"] for sample in samples], dtype=object)
    depth_paths = np.asarray([sample["depth_path"] for sample in samples], dtype=object)
    geometry = np.asarray([sample["geometry_features"] for sample in samples], dtype=np.float32)
    label_indices = np.asarray([sample["label_index"] for sample in samples], dtype=np.int64)
    flyover = np.asarray([1.0 if sample["flyover_recommended"] else 0.0 for sample in samples], dtype=np.float32)
    split_array = np.asarray([sample["split"] for sample in samples], dtype=object)
    np.savez_compressed(
        npz_path,
        image_paths=image_paths,
        depth_paths=depth_paths,
        geometry=geometry,
        label_indices=label_indices,
        flyover=flyover,
        splits=split_array,
        labels=np.asarray(OBSTACLE_LABELS, dtype=object),
        geometry_feature_names=np.asarray(GEOMETRY_FEATURE_NAMES, dtype=object),
    )
    label_counts = counts(labels)
    label_counts_all = {label: int(label_counts.get(label, 0)) for label in OBSTACLE_LABELS}
    summary = {
        "dataset_path": str(npz_path),
        "dataset_jsonl_path": str(jsonl_path),
        "total_samples": len(samples),
        "data_roots": [str(root) for root in data_roots],
        "source_counts": source_counts,
        "label_counts": label_counts_all,
        "raw_label_counts_top": sorted(counts(raw_labels).items(), key=lambda kv: (-kv[1], kv[0]))[:30],
        "split_counts": counts(splits),
        "teacher_source_counts": counts(sample["teacher_source"] for sample in samples),
        "missing_counts": dict(sorted(missing_counts.items())),
        "low_sample_labels": [label for label, count in label_counts_all.items() if count < 20],
        "label_map": LABEL_TO_INDEX,
        "geometry_feature_names": GEOMETRY_FEATURE_NAMES,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scheme A obstacle representation dataset.")
    parser.add_argument("--data-roots", nargs="+", default=["obstacle_avoidance_llm_data", "obstacle_avoidance_2_data"])
    parser.add_argument("--output-root", default="obstacle_representation_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_dataset([Path(item) for item in args.data_roots], output_root=Path(args.output_root), seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
