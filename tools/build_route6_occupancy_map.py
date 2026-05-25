from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control import route6_map_builder  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def build_route6_occupancy_from_run(
    run_dir: Path,
    map_config_path: Path,
    output_dir: Path,
    *,
    house_id: str,
    resolution_m: float = 0.25,
    z_min_m: float = 0.2,
    z_max_m: float = 8.0,
    crop_margin_m: float = 2.0,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    map_config_path = Path(map_config_path)
    output_dir = Path(output_dir)
    map_config = json.loads(map_config_path.read_text(encoding="utf-8"))
    rows = read_jsonl(run_dir / "lidar_capture_log.jsonl")
    valid_rows = route6_map_builder.filter_valid_pointcloud_rows(rows)
    if not valid_rows:
        raise ValueError(f"no valid Route 6 pointcloud rows in {run_dir / 'lidar_capture_log.jsonl'}")
    cloud = route6_map_builder.merge_pointcloud_rows(valid_rows)
    houses = map_config.get("houses", []) if isinstance(map_config.get("houses"), list) else []
    house = next((item for item in houses if isinstance(item, dict) and str(item.get("id", "")) == str(house_id)), None)
    bbox = route6_map_builder.house_world_bbox(map_config, house) if isinstance(house, dict) else None
    filtered = route6_map_builder.filter_pointcloud_for_mapping(
        cloud,
        z_min_m=float(z_min_m),
        z_max_m=float(z_max_m),
        bbox_unreal_cm=bbox,
        crop_margin_m=float(crop_margin_m),
    )
    if filtered.shape[0] == 0:
        filtered = cloud
    pointcloud_dir = output_dir / "houses" / f"house_{house_id}" / "pointcloud"
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    merged_path = pointcloud_dir / "merged_point_cloud_world_standard_m.npy"
    np.save(merged_path, filtered.astype(np.float32, copy=False))
    merged_ply_path = pointcloud_dir / "merged_point_cloud_world_standard_m.ply"
    route6_map_builder.write_pointcloud_ply(merged_ply_path, filtered)
    result = route6_map_builder.write_route6_map_artifacts(output_dir, map_config, house_id, filtered, resolution_m=float(resolution_m))
    result.update({
        "schema": "route6_occupancy_build_result_v1",
        "run_dir": str(run_dir),
        "map_config_path": str(map_config_path),
        "house_id": str(house_id),
        "valid_scan_capture_count": len(valid_rows),
        "merged_point_count": int(filtered.shape[0]),
        "merged_pointcloud_path": str(merged_path),
        "merged_pointcloud_ply_path": str(merged_ply_path),
    })
    route6_map_builder.write_json(output_dir / "route6_occupancy_build_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Route 6 occupancy/polygon/corrected-map artifacts from a lidar capture run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--map_config", type=Path, default=PROJECT_ROOT / "assets" / "overhead_map" / "manual_shift_houses_config.json")
    parser.add_argument("--output_dir", type=Path, default=Path(""))
    parser.add_argument("--house_id", default="")
    parser.add_argument("--resolution_m", type=float, default=0.25)
    parser.add_argument("--z_min_m", type=float, default=0.2)
    parser.add_argument("--z_max_m", type=float, default=8.0)
    parser.add_argument("--crop_margin_m", type=float, default=2.0)
    args = parser.parse_args()
    output_dir = args.output_dir if str(args.output_dir) else args.run_dir / "route6_occupancy"
    house_id = str(args.house_id or "").strip()
    if not house_id:
        raise SystemExit("--house_id is required")
    result = build_route6_occupancy_from_run(
        args.run_dir,
        args.map_config,
        output_dir,
        house_id=house_id,
        resolution_m=args.resolution_m,
        z_min_m=args.z_min_m,
        z_max_m=args.z_max_m,
        crop_margin_m=args.crop_margin_m,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
