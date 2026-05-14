from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .features import FEATURE_NAMES, extract_event_features


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_counts(labels: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def event_files(data_root: Path) -> List[Path]:
    sessions = data_root / "sessions"
    if not sessions.exists():
        return []
    return sorted(sessions.glob("*/avoidance_events.jsonl"))


def build_dataset(
    data_root: Path,
    *,
    output_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    data_root = Path(data_root)
    output_path = output_path or data_root / "datasets" / "dataset_latest.npz"
    index_path = index_path or data_root / "datasets" / "dataset_index_latest.jsonl"
    summary_path = summary_path or data_root / "datasets" / "dataset_summary.json"

    vectors: List[np.ndarray] = []
    risk_labels: List[str] = []
    action_labels: List[str] = []
    collision_labels: List[int] = []
    failed_labels: List[int] = []
    action_vectors: List[List[float]] = []
    index_rows: List[Dict[str, Any]] = []

    for events_path in event_files(data_root):
        base_dir = events_path.parent
        for row_index, event in enumerate(read_jsonl(events_path)):
            vector, metadata = extract_event_features(event, base_dir=base_dir)
            vectors.append(vector)
            risk_labels.append(str(metadata["risk_state"]))
            action_labels.append(str(metadata["expert_action"]))
            collision = 1 if bool(metadata["collision_state"]) else 0
            collision_labels.append(collision)
            failed_labels.append(collision)
            action_vectors.append(list(metadata["action_vector"]))
            index_rows.append(
                {
                    "dataset_row": len(index_rows),
                    "event_path": str(events_path),
                    "event_row": row_index,
                    "capture_dir": event.get("capture_dir", ""),
                    "rgb_path": event.get("rgb_path", ""),
                    "pointcloud_path": event.get("pointcloud_path", event.get("point_cloud_world_standard_m_npy_path", "")),
                    "risk_state": metadata["risk_state"],
                    "expert_action": metadata["expert_action"],
                    "collision_state": bool(collision),
                    "avoidance_failed": bool(collision),
                }
            )

    if vectors:
        x = np.vstack(vectors).astype(np.float32, copy=False)
        y_action_vector = np.asarray(action_vectors, dtype=np.float32)
    else:
        x = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        y_action_vector = np.zeros((0, 4), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=x,
        feature_names=np.asarray(FEATURE_NAMES),
        y_risk=np.asarray(risk_labels),
        y_action=np.asarray(action_labels),
        y_collision=np.asarray(collision_labels, dtype=np.int64),
        y_failed=np.asarray(failed_labels, dtype=np.int64),
        y_action_vector=y_action_vector,
        event_refs=np.asarray([json.dumps(row, ensure_ascii=False) for row in index_rows]),
        created_at=np.asarray([datetime.now().isoformat(timespec="seconds")]),
    )
    write_jsonl(index_path, index_rows)
    summary = {
        "status": "ok",
        "dataset_path": str(output_path),
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "data_root": str(data_root),
        "sample_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "event_file_count": len(event_files(data_root)),
        "risk_state_counts": label_counts(risk_labels),
        "expert_action_counts": label_counts(action_labels),
        "collision_counts": label_counts(collision_labels),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build obstacle avoidance dataset from avoidance_events.jsonl files.")
    parser.add_argument("--data-root", default="obstacle_avoidance_data")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    summary = build_dataset(Path(args.data_root), output_path=Path(args.output) if args.output else None)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
