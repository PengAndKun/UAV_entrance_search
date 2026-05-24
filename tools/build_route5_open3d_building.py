from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d


FACADE_COLORS = {
    "south": (230, 88, 80),
    "east": (80, 160, 230),
    "north": (94, 185, 94),
    "west": (230, 184, 72),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def affine_image_to_world(image_x: float, image_y: float, affine: List[List[float]]) -> Tuple[float, float]:
    matrix = np.asarray(affine, dtype=np.float64)
    linear = matrix[:, :2]
    offset = matrix[:, 2]
    world = np.linalg.solve(linear, np.asarray([float(image_x), float(image_y)], dtype=np.float64) - offset)
    return float(world[0]), float(world[1])


def homography_image_to_world(image_x: float, image_y: float, homography: List[List[float]]) -> Tuple[float, float]:
    inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    point = inverse @ np.asarray([float(image_x), float(image_y), 1.0], dtype=np.float64)
    if abs(float(point[2])) <= 1e-12:
        raise ValueError("inverse homography projection has zero scale")
    return float(point[0] / point[2]), float(point[1] / point[2])


def image_to_world(image_x: float, image_y: float, calibration: Dict[str, Any]) -> Tuple[float, float]:
    homography = calibration.get("homography_world_to_image") if isinstance(calibration, dict) else None
    if isinstance(homography, list) and len(homography) == 3:
        return homography_image_to_world(image_x, image_y, homography)
    affine = calibration.get("affine_world_to_image") if isinstance(calibration, dict) else None
    if isinstance(affine, list) and len(affine) == 2:
        return affine_image_to_world(image_x, image_y, affine)
    raise ValueError("map calibration has no affine or homography transform")


def load_house_bbox_standard_m(map_config_path: Path, house_id: str, margin_m: float, z_min_m: float, z_max_m: float) -> Dict[str, Any]:
    payload = json.loads(map_config_path.read_text(encoding="utf-8"))
    houses = payload.get("houses", []) if isinstance(payload.get("houses"), list) else []
    house = next((item for item in houses if isinstance(item, dict) and str(item.get("id", "") or "") == str(house_id)), None)
    if not isinstance(house, dict):
        raise ValueError(f"house id {house_id!r} not found in {map_config_path}")
    calibration = payload.get("overhead_map", {}).get("calibration", {}) if isinstance(payload.get("overhead_map"), dict) else {}
    bbox_image = house.get("map_bbox_image", {}) if isinstance(house.get("map_bbox_image"), dict) else {}
    world_points: List[Tuple[float, float]] = []
    if bbox_image and calibration:
        x1 = float(bbox_image["x1"])
        x2 = float(bbox_image["x2"])
        y1 = float(bbox_image["y1"])
        y2 = float(bbox_image["y2"])
        for image_x, image_y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
            world_points.append(image_to_world(image_x, image_y, calibration))
    if not world_points:
        center_x = float(house["center_x"])
        center_y = float(house["center_y"])
        half = max(150.0, 0.45 * float(house.get("radius_cm", 1000.0)))
        world_points = [
            (center_x - half, center_y - half),
            (center_x - half, center_y + half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
        ]
    xs = [point[0] for point in world_points]
    ys = [point[1] for point in world_points]
    unreal_bbox_cm = {
        "min_x": float(min(xs)),
        "max_x": float(max(xs)),
        "min_y": float(min(ys)),
        "max_y": float(max(ys)),
        "center_x": float(0.5 * (min(xs) + max(xs))),
        "center_y": float(0.5 * (min(ys) + max(ys))),
    }
    standard_bbox_m = {
        "min_x": unreal_bbox_cm["min_x"] / 100.0 - margin_m,
        "max_x": unreal_bbox_cm["max_x"] / 100.0 + margin_m,
        "min_y": -unreal_bbox_cm["max_y"] / 100.0 - margin_m,
        "max_y": -unreal_bbox_cm["min_y"] / 100.0 + margin_m,
        "min_z": float(z_min_m),
        "max_z": float(z_max_m),
    }
    return {
        "house_id": str(house_id),
        "map_config_path": str(map_config_path),
        "unreal_bbox_cm": unreal_bbox_cm,
        "standard_bbox_m": standard_bbox_m,
        "margin_m": float(margin_m),
        "z_min_m": float(z_min_m),
        "z_max_m": float(z_max_m),
    }


def selected_scan_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    seen: set[Tuple[str, str]] = set()
    for row in read_jsonl(run_dir / "lidar_capture_log.jsonl"):
        scan_id = str(row.get("scan_id", "") or "").strip()
        facade = str(row.get("facade", "") or "").strip().lower()
        status = str(row.get("capture_status", row.get("status", "")) or "").strip().lower()
        raw_path = Path(str(row.get("point_cloud_world_standard_m_npy_path", "") or ""))
        if not scan_id or facade not in FACADE_COLORS:
            continue
        if status not in {"ok", "captured", "done"}:
            continue
        if row.get("capture_guard_passed") is not True:
            continue
        if int(row.get("point_count", 0) or 0) <= 0:
            continue
        if not raw_path.is_file():
            continue
        key = (scan_id, str(raw_path))
        if key in seen:
            continue
        seen.add(key)
        rows.append({**row, "facade": facade, "point_cloud_world_standard_m_npy_path": str(raw_path)})
    rows.sort(key=lambda item: (str(item.get("facade", "")), int(item.get("frame_index", 0) or 0), str(item.get("scan_id", ""))))
    return rows


def filter_cloud(cloud: np.ndarray, bbox: Dict[str, float]) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6 or arr.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyz = arr[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    mask = (
        finite
        & (xyz[:, 0] >= float(bbox["min_x"]))
        & (xyz[:, 0] <= float(bbox["max_x"]))
        & (xyz[:, 1] >= float(bbox["min_y"]))
        & (xyz[:, 1] <= float(bbox["max_y"]))
        & (xyz[:, 2] >= float(bbox["min_z"]))
        & (xyz[:, 2] <= float(bbox["max_z"]))
    )
    if not np.any(mask):
        return np.zeros((0, 6), dtype=np.float32)
    return arr[mask, :6].astype(np.float32, copy=False)


def limit_points(cloud: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or cloud.shape[0] <= max_points:
        return cloud
    indices = np.linspace(0, cloud.shape[0] - 1, num=max_points, dtype=np.int64)
    return cloud[indices]


def voxel_downsample_numpy(cloud: np.ndarray, voxel_m: float, max_points: int = 0) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6 or arr.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if not math.isfinite(float(voxel_m)) or float(voxel_m) <= 0.0:
        return limit_points(arr[:, :6].astype(np.float32, copy=False), int(max_points))
    xyz = arr[:, :3]
    keys = np.floor(xyz / float(voxel_m)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    out = np.zeros((counts.shape[0], 6), dtype=np.float32)
    for column in range(6):
        sums = np.bincount(inverse, weights=arr[:, column].astype(np.float64), minlength=counts.shape[0])
        out[:, column] = (sums / np.maximum(counts, 1.0)).astype(np.float32)
    return limit_points(out, int(max_points))


def merge_incremental(existing: np.ndarray, new_cloud: np.ndarray, voxel_m: float, max_points: int) -> np.ndarray:
    if new_cloud.shape[0] == 0:
        return existing
    merged = new_cloud if existing.shape[0] == 0 else np.vstack((existing, new_cloud))
    return voxel_downsample_numpy(merged, voxel_m, max_points=max_points)


def to_open3d(cloud: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.shape[0] == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(arr[:, 3:6], 0, 255).astype(np.float64) / 255.0)
    return pcd


def from_open3d(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    if not pcd.has_points():
        return np.zeros((0, 6), dtype=np.float32)
    points = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32) * 255.0
    else:
        colors = np.full((points.shape[0], 3), 255.0, dtype=np.float32)
    return np.column_stack((points, colors)).astype(np.float32)


def remove_outliers(cloud: np.ndarray, nb_neighbors: int, std_ratio: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    if cloud.shape[0] < max(10, nb_neighbors):
        return cloud, {"enabled": False, "reason": "too_few_points"}
    pcd = to_open3d(cloud)
    cleaned, indices = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return from_open3d(cleaned), {
        "enabled": True,
        "nb_neighbors": int(nb_neighbors),
        "std_ratio": float(std_ratio),
        "input_point_count": int(cloud.shape[0]),
        "kept_point_count": int(len(indices)),
        "removed_point_count": int(cloud.shape[0] - len(indices)),
    }


def cloud_stats(cloud: np.ndarray) -> Dict[str, Any]:
    if cloud.shape[0] == 0:
        return {"point_count": 0}
    xyz = cloud[:, :3]
    return {
        "point_count": int(cloud.shape[0]),
        "min_xyz": [round(float(v), 5) for v in np.min(xyz, axis=0)],
        "max_xyz": [round(float(v), 5) for v in np.max(xyz, axis=0)],
        "mean_xyz": [round(float(v), 5) for v in np.mean(xyz, axis=0)],
    }


def save_cloud_bundle(cloud: np.ndarray, output_dir: Path, basename: str, *, estimate_normals: bool = False) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{basename}.npy"
    ply_path = output_dir / f"{basename}.ply"
    pcd_path = output_dir / f"{basename}.pcd"
    np.save(npy_path, cloud.astype(np.float32, copy=False))
    pcd = to_open3d(cloud)
    if estimate_normals and pcd.has_points():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False, print_progress=False)
    o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False, print_progress=False)
    return {
        **cloud_stats(cloud),
        "npy_path": str(npy_path),
        "ply_path": str(ply_path),
        "pcd_path": str(pcd_path),
        "estimate_normals": bool(estimate_normals),
    }


def facade_semantic_cloud(facade_clouds: Dict[str, np.ndarray]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for facade, cloud in facade_clouds.items():
        if cloud.shape[0] == 0:
            continue
        colored = cloud.copy()
        colored[:, 3:6] = np.asarray(FACADE_COLORS.get(facade, (255, 255, 255)), dtype=np.float32)
        chunks.append(colored)
    return np.vstack(chunks) if chunks else np.zeros((0, 6), dtype=np.float32)


def write_viewer_script(output_dir: Path, summary_path: Path) -> str:
    viewer_path = output_dir / "open3d_viewer.py"
    script = f'''from __future__ import annotations

import json
from pathlib import Path

import open3d as o3d

SUMMARY_PATH = Path(r"{summary_path}")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    paths = [
        summary["outputs"]["merged_rgb"]["ply_path"],
        summary["outputs"]["merged_facade_semantic"]["ply_path"],
    ]
    geometries = []
    for path in paths:
        p = Path(path)
        if p.is_file():
            cloud = o3d.io.read_point_cloud(str(p))
            cloud.translate((0, 0, 0), relative=True)
            geometries.append(cloud)
    if not geometries:
        raise SystemExit("No Open3D point clouds found. Rebuild the export first.")
    o3d.visualization.draw_geometries(geometries, window_name="Route5 House 002 Building Open3D")


if __name__ == "__main__":
    main()
'''
    viewer_path.write_text(script, encoding="utf-8")
    return str(viewer_path)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    house_id = str(args.house_id).zfill(3) if str(args.house_id).isdigit() else str(args.house_id)
    map_config = Path(args.map_config).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / f"open3d_building_house_{house_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_payload = load_house_bbox_standard_m(map_config, house_id, args.crop_margin_m, args.z_min_m, args.z_max_m)
    bbox = bbox_payload["standard_bbox_m"]
    rows = selected_scan_rows(run_dir)
    facade_clouds: Dict[str, np.ndarray] = {facade: np.zeros((0, 6), dtype=np.float32) for facade in FACADE_COLORS}
    facade_sources: Dict[str, List[Dict[str, Any]]] = {facade: [] for facade in FACADE_COLORS}
    skipped: List[Dict[str, Any]] = []

    for row in rows:
        facade = str(row.get("facade", "") or "").lower()
        raw_path = Path(str(row.get("point_cloud_world_standard_m_npy_path", "") or ""))
        raw_count = int(row.get("point_count", 0) or 0)
        try:
            raw = np.load(raw_path, mmap_mode="r")
            cropped = filter_cloud(raw, bbox)
            frame = voxel_downsample_numpy(cropped, args.per_frame_voxel_m, max_points=args.max_points_per_frame)
            facade_clouds[facade] = merge_incremental(
                facade_clouds[facade],
                frame,
                args.facade_voxel_m,
                max_points=args.max_points_per_facade,
            )
            facade_sources[facade].append(
                {
                    "frame_index": int(row.get("frame_index", 0) or 0),
                    "scan_id": str(row.get("scan_id", "") or ""),
                    "raw_point_count": raw_count,
                    "cropped_point_count": int(cropped.shape[0]),
                    "frame_point_count_after_voxel": int(frame.shape[0]),
                    "source_path": str(raw_path),
                }
            )
        except Exception as exc:
            skipped.append(
                {
                    "frame_index": row.get("frame_index"),
                    "scan_id": row.get("scan_id"),
                    "facade": facade,
                    "source_path": str(raw_path),
                    "reason": str(exc),
                }
            )

    outlier_reports: Dict[str, Any] = {}
    if not args.no_outlier_removal:
        for facade, cloud in list(facade_clouds.items()):
            cleaned, report = remove_outliers(cloud, args.outlier_neighbors, args.outlier_std_ratio)
            facade_clouds[facade] = cleaned
            outlier_reports[facade] = report

    facade_outputs: Dict[str, Any] = {}
    for facade, cloud in facade_clouds.items():
        facade_outputs[facade] = {
            **save_cloud_bundle(cloud, output_dir / "facades", f"house_{house_id}_{facade}_rgb"),
            "source_captures": facade_sources[facade],
            "outlier_removal": outlier_reports.get(facade, {}),
        }

    merged = np.zeros((0, 6), dtype=np.float32)
    for facade in ("south", "west", "east", "north"):
        merged = merge_incremental(merged, facade_clouds.get(facade, np.zeros((0, 6), dtype=np.float32)), args.merge_voxel_m, args.max_points_merged)
    semantic = voxel_downsample_numpy(facade_semantic_cloud(facade_clouds), args.merge_voxel_m, max_points=args.max_points_merged)

    outputs = {
        "facades": facade_outputs,
        "merged_rgb": save_cloud_bundle(merged, output_dir, f"house_{house_id}_building_rgb", estimate_normals=args.estimate_normals),
        "merged_facade_semantic": save_cloud_bundle(semantic, output_dir, f"house_{house_id}_building_facade_semantic", estimate_normals=False),
    }
    summary = {
        "schema": "route5_open3d_building_export_v1",
        "status": "ok" if merged.shape[0] > 0 else "empty",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "house_id": house_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_policy": {
            "source_log": str(run_dir / "lidar_capture_log.jsonl"),
            "include_only_capture_status_ok": True,
            "include_only_capture_guard_passed": True,
            "include_only_positive_point_count": True,
            "selected_capture_count": len(rows),
            "skipped_processing_errors": skipped,
        },
        "crop": bbox_payload,
        "voxel_policy": {
            "per_frame_voxel_m": float(args.per_frame_voxel_m),
            "facade_voxel_m": float(args.facade_voxel_m),
            "merge_voxel_m": float(args.merge_voxel_m),
            "max_points_per_frame": int(args.max_points_per_frame),
            "max_points_per_facade": int(args.max_points_per_facade),
            "max_points_merged": int(args.max_points_merged),
            "outlier_removal": not bool(args.no_outlier_removal),
        },
        "outputs": outputs,
    }
    summary_path = output_dir / "open3d_building_summary.json"
    summary["viewer_script_path"] = write_viewer_script(output_dir, summary_path)
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact Open3D building point cloud from a Route 5 run.")
    parser.add_argument("run_dir", help="Route 5 run directory")
    parser.add_argument("--house_id", default="002")
    parser.add_argument("--map_config", default="assets/overhead_map/manual_shift_houses_config.json")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--crop_margin_m", type=float, default=1.8)
    parser.add_argument("--z_min_m", type=float, default=0.05)
    parser.add_argument("--z_max_m", type=float, default=8.0)
    parser.add_argument("--per_frame_voxel_m", type=float, default=0.04)
    parser.add_argument("--facade_voxel_m", type=float, default=0.06)
    parser.add_argument("--merge_voxel_m", type=float, default=0.08)
    parser.add_argument("--max_points_per_frame", type=int, default=35000)
    parser.add_argument("--max_points_per_facade", type=int, default=160000)
    parser.add_argument("--max_points_merged", type=int, default=320000)
    parser.add_argument("--no_outlier_removal", action="store_true")
    parser.add_argument("--outlier_neighbors", type=int, default=20)
    parser.add_argument("--outlier_std_ratio", type=float, default=2.5)
    parser.add_argument("--estimate_normals", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    merged = summary.get("outputs", {}).get("merged_rgb", {})
    print(
        "Open3D building export",
        summary.get("status"),
        "selected_captures=",
        summary.get("selection_policy", {}).get("selected_capture_count"),
        "merged_points=",
        merged.get("point_count"),
        "output_dir=",
        summary.get("output_dir"),
    )


if __name__ == "__main__":
    main()
