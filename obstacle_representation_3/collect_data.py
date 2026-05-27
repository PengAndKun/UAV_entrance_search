import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_DATASET_JSONL = Path("obstacle_representation_3_data/datasets/a_plus_3_dataset_latest.jsonl")
DEFAULT_DATASET_NPZ = Path("obstacle_representation_3_data/datasets/a_plus_3_dataset_latest.npz")
DEFAULT_OUTPUT_ROOT = Path("obstacle_representation_3_data")


def write_json(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, samples: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


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


def sanitize_segment(value: str) -> str:
    text = str(value).replace("\\", "/").strip("/")
    parts = [part for part in text.split("/") if part not in {"", ".", ".."}]
    return "__".join(parts) if parts else "unknown"


def safe_file_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:16]
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)
    stem = stem[:64] if stem else "file"
    suffix = path.suffix if len(path.suffix) <= 12 else ""
    return f"{digest}_{stem}{suffix}"


def best_relative_path(path: Path, source_root: str) -> Path:
    candidates = []
    if source_root:
        candidates.append(Path(source_root))
    candidates.append(Path.cwd())
    for root in candidates:
        try:
            return path.resolve().relative_to(root.resolve())
        except Exception:
            continue
    return Path(path.name)


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def copy_if_needed(source: Path, target: Path) -> Tuple[bool, int]:
    if not source.is_file():
        return False, 0
    target.parent.mkdir(parents=True, exist_ok=True)
    size = int(source.stat().st_size)
    if target.is_file() and int(target.stat().st_size) == size:
        return False, size
    shutil.copy2(source, target)
    return True, size


def collected_path(source: Path, source_root: str, collected_root: Path, kind: str) -> Path:
    root_name = sanitize_segment(source_root or source.parent.name)
    return collected_root / "files" / kind / root_name / safe_file_name(source)


def backup_latest(dataset_jsonl: Path, dataset_npz: Path) -> Dict[str, str]:
    backups: Dict[str, str] = {}
    if dataset_jsonl.is_file():
        backup = dataset_jsonl.with_name(dataset_jsonl.stem + "_original_paths" + dataset_jsonl.suffix)
        if not backup.is_file():
            shutil.copy2(dataset_jsonl, backup)
        backups["jsonl_original_paths"] = str(backup)
    if dataset_npz.is_file():
        backup = dataset_npz.with_name(dataset_npz.stem + "_original_paths" + dataset_npz.suffix)
        if not backup.is_file():
            shutil.copy2(dataset_npz, backup)
        backups["npz_original_paths"] = str(backup)
    summary = dataset_jsonl.with_name("a_plus_3_dataset_summary.json")
    if summary.is_file():
        backup = summary.with_name("a_plus_3_dataset_summary_original_paths.json")
        if not backup.is_file():
            shutil.copy2(summary, backup)
        backups["summary_original_paths"] = str(backup)
    return backups


def update_latest_summary(dataset_jsonl: Path, manifest: Dict[str, Any]) -> None:
    summary_path = dataset_jsonl.with_name("a_plus_3_dataset_summary.json")
    original_summary_path = dataset_jsonl.with_name("a_plus_3_dataset_summary_original_paths.json")
    if original_summary_path.is_file():
        try:
            summary = json.loads(original_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    elif summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    else:
        summary = {}
    summary["dataset_path"] = str(dataset_jsonl.with_suffix(".npz"))
    summary["dataset_jsonl_path"] = str(dataset_jsonl)
    summary["paths_collected_under_obstacle_representation_3_data"] = True
    summary["collection_root"] = manifest.get("collection_root", "")
    summary["collection_manifest_path"] = str(Path(manifest.get("collection_root", "")) / "a_plus_3_collection_manifest.json")
    summary["collection_created_at"] = manifest.get("created_at", "")
    summary["unique_collected_targets"] = manifest.get("unique_collected_targets", 0)
    summary["collected_copied_this_run_gb"] = manifest.get("copied_this_run_gb", 0.0)
    summary["total_collected_gb"] = manifest.get("total_collected_gb", 0.0)
    summary["original_path_backups"] = manifest.get("backups", {})
    if "data_roots" in summary:
        summary["original_data_roots"] = summary.get("data_roots", [])
    summary["data_roots"] = [str(manifest.get("collection_root", ""))]
    write_json(summary_path, summary)


def rewrite_npz(dataset_npz: Path, output_npz: Path, image_paths: List[str], depth_paths: List[str]) -> None:
    source = np.load(dataset_npz, allow_pickle=True)
    arrays = {name: source[name] for name in source.files}
    arrays["image_paths"] = np.asarray(image_paths, dtype=object)
    arrays["depth_paths"] = np.asarray(depth_paths, dtype=object)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def collect_dataset(
    dataset_jsonl: Path,
    dataset_npz: Path,
    output_root: Path,
    *,
    update_latest: bool = True,
) -> Dict[str, Any]:
    samples = read_jsonl(dataset_jsonl)
    collected_root = output_root / "collected"
    copied_files = 0
    skipped_files = 0
    missing_files = 0
    copied_bytes = 0
    total_bytes = 0
    rewritten_samples: List[Dict[str, Any]] = []
    seen_targets: set[Path] = set()
    source_counts: Dict[str, int] = {}

    for sample in samples:
        source_root = str(sample.get("source_root", ""))
        source_counts[source_root] = source_counts.get(source_root, 0) + 1
        rewritten = dict(sample)
        for key, kind in (("rgb_path", "rgb"), ("depth_path", "depth")):
            source = Path(str(sample.get(f"original_{key}") or sample.get(key, "")))
            target = collected_path(source, source_root, collected_root, kind)
            rewritten[f"original_{key}"] = str(source)
            rewritten[key] = repo_relative(target)
            if target not in seen_targets:
                copied, size = copy_if_needed(source, target)
                if copied:
                    copied_files += 1
                    copied_bytes += size
                elif source.is_file():
                    skipped_files += 1
                else:
                    missing_files += 1
                total_bytes += size
                seen_targets.add(target)

        event_file = Path(str(sample.get("original_event_file") or sample.get("event_file", "")))
        event_target = collected_path(event_file, source_root, collected_root, "events")
        rewritten["original_event_file"] = str(event_file)
        rewritten["event_file"] = repo_relative(event_target)
        if event_target not in seen_targets:
            copied, size = copy_if_needed(event_file, event_target)
            if copied:
                copied_files += 1
                copied_bytes += size
            elif event_file.is_file():
                skipped_files += 1
            else:
                missing_files += 1
            total_bytes += size
            seen_targets.add(event_target)
        rewritten_samples.append(rewritten)

    collected_jsonl = collected_root / "a_plus_3_dataset_collected.jsonl"
    collected_npz = collected_root / "a_plus_3_dataset_collected.npz"
    write_jsonl(collected_jsonl, rewritten_samples)
    rewrite_npz(
        dataset_npz,
        collected_npz,
        [sample["rgb_path"] for sample in rewritten_samples],
        [sample["depth_path"] for sample in rewritten_samples],
    )

    backups: Dict[str, str] = {}
    latest_jsonl = dataset_jsonl
    latest_npz = dataset_npz
    if update_latest:
        backups = backup_latest(dataset_jsonl, dataset_npz)
        write_jsonl(latest_jsonl, rewritten_samples)
        rewrite_npz(
            dataset_npz if not backups.get("npz_original_paths") else Path(backups["npz_original_paths"]),
            latest_npz,
            [sample["rgb_path"] for sample in rewritten_samples],
            [sample["depth_path"] for sample in rewritten_samples],
        )

    manifest = {
        "collection_root": str(collected_root),
        "collected_dataset_jsonl": str(collected_jsonl),
        "collected_dataset_npz": str(collected_npz),
        "latest_dataset_jsonl": str(latest_jsonl) if update_latest else "",
        "latest_dataset_npz": str(latest_npz) if update_latest else "",
        "sample_count": len(rewritten_samples),
        "unique_collected_targets": len(seen_targets),
        "copied_files": copied_files,
        "skipped_existing_files": skipped_files,
        "missing_files": missing_files,
        "copied_this_run_gb": round(float(copied_bytes) / 1024.0 / 1024.0 / 1024.0, 6),
        "total_collected_gb": round(float(total_bytes) / 1024.0 / 1024.0 / 1024.0, 6),
        "source_counts": dict(sorted(source_counts.items())),
        "backups": backups,
        "updated_latest_paths": bool(update_latest),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(collected_root / "a_plus_3_collection_manifest.json", manifest)
    if update_latest:
        update_latest_summary(dataset_jsonl, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OR3 dataset files under obstacle_representation_3_data.")
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET_JSONL))
    parser.add_argument("--dataset-npz", default=str(DEFAULT_DATASET_NPZ))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()
    manifest = collect_dataset(
        Path(args.dataset_jsonl),
        Path(args.dataset_npz),
        Path(args.output_root),
        update_latest=not bool(args.no_update_latest),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
