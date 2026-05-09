from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

import run_drone_flight as flight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIDAR_YOLO_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "assets"
    / "yolo_model"
    / "models"
    / "phase2_entry_detector"
    / "yolo26n_phase2_entry_v1"
    / "weights"
    / "best.pt"
)
DEFAULT_LIDAR_YOLO_CONF = 0.25
DEFAULT_LIDAR_YOLO_IMGSZ = 640
DEFAULT_LIDAR_YOLO_DEDUPE_RADIUS_M = 0.75
DEFAULT_LIDAR_YOLO_MAX_POINTS_PER_DETECTION = 2000

ProgressCallback = Callable[[Dict[str, Any]], None]

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "open door": (40, 230, 80),
    "open_door": (40, 230, 80),
    "door": (255, 245, 80),
    "close door": (255, 145, 30),
    "closed door": (255, 145, 30),
    "close_door": (255, 145, 30),
    "window": (50, 170, 255),
}


@dataclass(frozen=True)
class LidarYoloFrame:
    frame_index: int
    frame_name: str
    capture_dir: Path
    rgb_path: Path
    depth_npy_path: Path
    camera_info_path: Path
    capture_json_path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_frame_index(name: str) -> int:
    match = re.search(r"(\d+)$", str(name or ""))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def resolve_stream_path(stream_dir: Path, raw_path: Any) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        return Path()
    path = Path(raw)
    if path.is_absolute():
        return path
    return (stream_dir / path).resolve()


def normalize_class_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").replace("-", " ").strip().lower())


def class_color_rgb(class_name: Any) -> Tuple[int, int, int]:
    normalized = normalize_class_name(class_name)
    if normalized in CLASS_COLORS:
        return CLASS_COLORS[normalized]
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(normalized or "object"))
    return (
        80 + seed % 150,
        80 + (seed * 7) % 150,
        80 + (seed * 13) % 150,
    )


def default_lidar_yolo_device() -> str:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def normalize_lidar_yolo_device(device: Any) -> str:
    raw = str(device or "").strip()
    if not raw or raw.lower() == "auto":
        return default_lidar_yolo_device()
    if raw.lower() == "cpu":
        return "cpu"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch

        if torch.cuda.is_available():
            return raw
    except Exception:
        pass
    return "cpu"


def scan_lidar_yolo_frames(stream_dir: str | Path, *, stride: int = 1, max_frames: int = 0) -> List[LidarYoloFrame]:
    stream_path = Path(stream_dir).resolve()
    stride = max(1, int(stride or 1))
    max_frames = max(0, int(max_frames or 0))
    frames_root = stream_path / "frames"
    frames: List[LidarYoloFrame] = []
    if not frames_root.exists():
        return frames

    for capture_dir in sorted(path for path in frames_root.glob("frame_*") if path.is_dir()):
        capture_payload = load_json(capture_dir / "capture.json", {})
        frame_index = int(capture_payload.get("frame_index") or parse_frame_index(capture_dir.name) or 0)
        rgb_path = resolve_stream_path(stream_path, capture_payload.get("rgb_path")) if capture_payload.get("rgb_path") else capture_dir / "rgb.png"
        depth_path = (
            resolve_stream_path(stream_path, capture_payload.get("depth_npy_path"))
            if capture_payload.get("depth_npy_path")
            else capture_dir / "depth.npy"
        )
        camera_info_path = (
            resolve_stream_path(stream_path, capture_payload.get("camera_info_path"))
            if capture_payload.get("camera_info_path")
            else capture_dir / "camera_info.json"
        )
        if rgb_path.exists() and depth_path.exists() and camera_info_path.exists():
            frames.append(
                LidarYoloFrame(
                    frame_index=frame_index,
                    frame_name=capture_dir.name,
                    capture_dir=capture_dir,
                    rgb_path=rgb_path,
                    depth_npy_path=depth_path,
                    camera_info_path=camera_info_path,
                    capture_json_path=capture_dir / "capture.json",
                )
            )
    selected = frames[::stride]
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def clamp_bbox_xyxy(box: List[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if len(box) < 4:
        return None
    x1 = max(0, min(width - 1, int(math.floor(float(box[0])))))
    y1 = max(0, min(height - 1, int(math.floor(float(box[1])))))
    x2 = max(0, min(width, int(math.ceil(float(box[2])))))
    y2 = max(0, min(height, int(math.ceil(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def scale_bbox_xyxy(
    box: List[float],
    *,
    source_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    if source_w <= 0 or source_h <= 0:
        return None
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)
    return clamp_bbox_xyxy(
        [float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy],
        target_w,
        target_h,
    )


def deterministic_sample_indices(size: int, max_count: int) -> np.ndarray:
    count = int(size)
    limit = int(max_count)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=limit, dtype=np.int64)


def project_bbox_depth_to_world_standard_m(
    *,
    depth_image: Any,
    camera_info: Dict[str, Any],
    rgb_shape: Tuple[int, int],
    rgb_bbox_xyxy: List[float],
    class_name: str,
    max_points: int = DEFAULT_LIDAR_YOLO_MAX_POINTS_PER_DETECTION,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    depth_cm = flight.coerce_depth_planar_image(depth_image).astype(np.float32, copy=False)
    depth_h, depth_w = depth_cm.shape[:2]
    depth_bbox = scale_bbox_xyxy(rgb_bbox_xyxy, source_shape=rgb_shape, target_shape=(depth_h, depth_w))
    if depth_bbox is None:
        return np.zeros((0, 6), dtype=np.float32), {
            "status": "empty_bbox",
            "point_count": 0,
            "rgb_bbox_xyxy": rgb_bbox_xyxy,
            "depth_bbox_xyxy": [],
        }
    x1, y1, x2, y2 = depth_bbox
    min_depth = float(camera_info.get("min_depth_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM) or flight.DEFAULT_LIDAR_DEPTH_MIN_CM)
    max_depth = float(camera_info.get("max_depth_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM) or flight.DEFAULT_LIDAR_DEPTH_MAX_CM)
    if not math.isfinite(max_depth) or max_depth <= min_depth:
        max_depth = float(np.nanmax(depth_cm)) if np.isfinite(depth_cm).any() else min_depth + 1.0

    region = depth_cm[y1:y2, x1:x2]
    valid = np.isfinite(region) & (region >= min_depth) & (region <= max_depth)
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return np.zeros((0, 6), dtype=np.float32), {
            "status": "no_valid_depth",
            "point_count": 0,
            "rgb_bbox_xyxy": rgb_bbox_xyxy,
            "depth_bbox_xyxy": [x1, y1, x2, y2],
            "valid_depth_count": 0,
        }

    ys_local, xs_local = np.nonzero(valid)
    sample_indices = deterministic_sample_indices(valid_count, int(max_points))
    ys = ys_local[sample_indices].astype(np.float32, copy=False) + float(y1)
    xs = xs_local[sample_indices].astype(np.float32, copy=False) + float(x1)
    selected_depth = region[valid][sample_indices].astype(np.float32, copy=False)

    fov, _fov_source = flight.parse_camera_fov_degrees(camera_info.get("horizontal_fov_deg"))
    intrinsics = camera_info.get("intrinsics") if isinstance(camera_info.get("intrinsics"), dict) else {}
    if not intrinsics:
        intrinsics = flight.camera_intrinsics_from_fov(depth_w, depth_h, fov)
    fx = float(intrinsics.get("fx", 1.0) or 1.0)
    fy = float(intrinsics.get("fy", fx) or fx)
    cx = float(intrinsics.get("cx", (depth_w - 1.0) / 2.0) or 0.0)
    cy = float(intrinsics.get("cy", (depth_h - 1.0) / 2.0) or 0.0)

    x_norm = (xs - cx) / fx
    y_norm = (ys - cy) / fy
    projection = str(
        camera_info.get("depth_projection_selected")
        or camera_info.get("depth_projection")
        or "plane_depth"
    ).strip().lower()
    if projection in {"ray", "ray_depth"}:
        z = selected_depth / np.sqrt(1.0 + np.square(x_norm) + np.square(y_norm)).astype(np.float32, copy=False)
        projection = "ray_depth"
    else:
        z = selected_depth
        projection = "plane_depth"
    x_right = x_norm * z
    y_down = y_norm * z
    local_unreal_cm = np.column_stack((z, x_right, -y_down)).astype(np.float32, copy=False)

    location = camera_info.get("location", {}) if isinstance(camera_info.get("location"), dict) else {}
    camera_location = np.array(
        [
            float(location.get("x", 0.0) or 0.0),
            float(location.get("y", 0.0) or 0.0),
            float(location.get("z", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )
    rotation = camera_info.get("rotation", {}) if isinstance(camera_info.get("rotation"), dict) else {}
    world_unreal_cm = local_unreal_cm @ flight.unreal_rotation_matrix_from_camera(rotation).T + camera_location
    world_xyz_m = np.column_stack(
        (
            world_unreal_cm[:, 0],
            -world_unreal_cm[:, 1],
            world_unreal_cm[:, 2],
        )
    ).astype(np.float32, copy=False) / 100.0
    color = np.array(class_color_rgb(class_name), dtype=np.float32)
    colors = np.repeat(color.reshape(1, 3), world_xyz_m.shape[0], axis=0)
    cloud = np.column_stack((world_xyz_m, colors)).astype(np.float32, copy=False)
    bbox_min = np.nanmin(world_xyz_m, axis=0)
    bbox_max = np.nanmax(world_xyz_m, axis=0)
    center = np.nanmedian(world_xyz_m, axis=0)
    return cloud, {
        "status": "ok",
        "point_count": int(cloud.shape[0]),
        "valid_depth_count": valid_count,
        "rgb_bbox_xyxy": [float(v) for v in rgb_bbox_xyxy],
        "depth_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "center_world_m": [float(v) for v in center],
        "bbox_world_m": {
            "min": [float(v) for v in bbox_min],
            "max": [float(v) for v in bbox_max],
        },
        "depth_projection_selected": projection,
        "projection_coordinate_frame": "standard_zup",
        "projection_coordinate_units": "m",
    }


def yolo_detections_from_result(result: Any) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    annotated = result.plot()
    detections: List[Dict[str, Any]] = []
    names = result.names if isinstance(result.names, dict) else {}
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            class_name = str(names.get(cls_id, str(cls_id)))
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "class_name_normalized": normalize_class_name(class_name),
                    "confidence": confidence,
                    "xyxy": xyxy,
                }
            )
    detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return detections, annotated


def bbox_union(existing: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    old_min = np.array(existing.get("min", [0.0, 0.0, 0.0]), dtype=np.float32)
    old_max = np.array(existing.get("max", [0.0, 0.0, 0.0]), dtype=np.float32)
    new_min = np.array(candidate.get("min", [0.0, 0.0, 0.0]), dtype=np.float32)
    new_max = np.array(candidate.get("max", [0.0, 0.0, 0.0]), dtype=np.float32)
    return {
        "min": [float(v) for v in np.minimum(old_min, new_min)],
        "max": [float(v) for v in np.maximum(old_max, new_max)],
    }


def dedupe_semantic_observations(
    observations: List[Dict[str, Any]],
    *,
    radius_m: float = DEFAULT_LIDAR_YOLO_DEDUPE_RADIUS_M,
) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    threshold = max(0.0, float(radius_m))
    for obs in observations:
        center = np.array(obs.get("center_world_m", []), dtype=np.float32)
        if center.shape[0] != 3 or not np.isfinite(center).all():
            continue
        normalized = normalize_class_name(obs.get("class_name_normalized") or obs.get("class_name"))
        best_index = -1
        best_distance = float("inf")
        for idx, label in enumerate(labels):
            if normalize_class_name(label.get("class_name_normalized") or label.get("class_name")) != normalized:
                continue
            label_center = np.array(label.get("center_world_m", []), dtype=np.float32)
            if label_center.shape[0] != 3:
                continue
            distance = float(np.linalg.norm(center - label_center))
            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_index = idx
        if best_index < 0:
            labels.append(
                {
                    "label_id": len(labels) + 1,
                    "class_name": obs.get("class_name", normalized),
                    "class_name_normalized": normalized,
                    "best_confidence": float(obs.get("confidence", 0.0) or 0.0),
                    "observation_count": 1,
                    "frame_indices": [int(obs.get("frame_index", 0) or 0)],
                    "center_world_m": [float(v) for v in center],
                    "bbox_world_m": obs.get("bbox_world_m", {}),
                    "representative_observation": obs,
                    "observation_ids": [obs.get("observation_id", "")],
                    "color_rgb": list(class_color_rgb(normalized)),
                }
            )
            continue
        label = labels[best_index]
        previous_count = int(label.get("observation_count", 1) or 1)
        new_count = previous_count + 1
        previous_center = np.array(label.get("center_world_m", [0.0, 0.0, 0.0]), dtype=np.float32)
        merged_center = (previous_center * previous_count + center) / float(new_count)
        label["center_world_m"] = [float(v) for v in merged_center]
        label["observation_count"] = new_count
        frame_indices = set(int(v) for v in label.get("frame_indices", []) if v is not None)
        frame_indices.add(int(obs.get("frame_index", 0) or 0))
        label["frame_indices"] = sorted(frame_indices)
        if isinstance(obs.get("bbox_world_m"), dict):
            current_bbox = label.get("bbox_world_m") if isinstance(label.get("bbox_world_m"), dict) else obs.get("bbox_world_m")
            label["bbox_world_m"] = bbox_union(current_bbox, obs.get("bbox_world_m", {}))
        label.setdefault("observation_ids", []).append(obs.get("observation_id", ""))
        confidence = float(obs.get("confidence", 0.0) or 0.0)
        if confidence > float(label.get("best_confidence", 0.0) or 0.0):
            label["best_confidence"] = confidence
            label["class_name"] = obs.get("class_name", label.get("class_name", normalized))
            label["representative_observation"] = obs
    return labels


def update_open3d_summary_with_semantics(stream_dir: Path, semantic_summary: Dict[str, Any]) -> None:
    export_dir = stream_dir / "open3d_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    summary_path = export_dir / "open3d_export_summary.json"
    summary = load_json(summary_path, {})
    if not isinstance(summary, dict):
        summary = {}
    summary.update(
        {
            "semantic_yolo_status": semantic_summary.get("status", "ok"),
            "semantic_label_count": int(semantic_summary.get("semantic_label_count", 0) or 0),
            "semantic_observation_count": int(semantic_summary.get("semantic_observation_count", 0) or 0),
            "semantic_points_path": semantic_summary.get("semantic_points_path", ""),
            "semantic_labels_path": semantic_summary.get("semantic_labels_path", ""),
            "semantic_analysis_dir": semantic_summary.get("analysis_dir", ""),
            "semantic_projection_summary_path": semantic_summary.get("summary_path", semantic_summary.get("stable_summary_path", "")),
            "semantic_selected_frame_count": int(semantic_summary.get("selected_frame_count", 0) or 0),
            "semantic_processed_frame_count": int(semantic_summary.get("processed_frame_count", 0) or 0),
            "semantic_config": semantic_summary.get("config", {}),
            "dedupe_radius_m": float(semantic_summary.get("dedupe_radius_m", DEFAULT_LIDAR_YOLO_DEDUPE_RADIUS_M)),
            "projection_coordinate_frame": "standard_zup",
            "projection_coordinate_units": "m",
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
    )
    write_json(summary_path, summary)


def run_lidar_yolo_analysis(
    *,
    stream_dir: str | Path,
    weights_path: str | Path = DEFAULT_LIDAR_YOLO_WEIGHTS_PATH,
    device: str = "0",
    conf: float = DEFAULT_LIDAR_YOLO_CONF,
    imgsz: int = DEFAULT_LIDAR_YOLO_IMGSZ,
    stride: int = 1,
    max_frames: int = 0,
    dedupe_radius_m: float = DEFAULT_LIDAR_YOLO_DEDUPE_RADIUS_M,
    max_points_per_detection: int = DEFAULT_LIDAR_YOLO_MAX_POINTS_PER_DETECTION,
    stop_event: Optional[threading.Event] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
    (PROJECT_ROOT / ".ultralytics").mkdir(parents=True, exist_ok=True)

    stream_path = Path(stream_dir).resolve()
    if not stream_path.exists():
        raise RuntimeError(f"Lidar stream folder not found: {stream_path}")
    selected_weights = Path(weights_path).expanduser().resolve()
    if not selected_weights.exists():
        raise RuntimeError(f"YOLO weights not found: {selected_weights}")
    selected_frames = scan_lidar_yolo_frames(stream_path, stride=stride, max_frames=max_frames)
    if not selected_frames:
        raise RuntimeError(f"No lidar frames with rgb.png, depth.npy, and camera_info.json found in {stream_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Lidar YOLO labels require ultralytics. Install it with: pip install -U ultralytics") from exc

    selected_device = normalize_lidar_yolo_device(device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    analysis_dir = stream_path / "lidar_yolo_analysis" / timestamp
    semantic_export_dir = stream_path / "open3d_export" / "semantic"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    semantic_export_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="milliseconds")

    config = {
        "weights_path": str(selected_weights),
        "device": selected_device,
        "conf": float(conf),
        "imgsz": int(imgsz),
        "stride": int(max(1, stride)),
        "max_frames": int(max(0, max_frames)),
        "dedupe_radius_m": float(dedupe_radius_m),
        "max_points_per_detection": int(max_points_per_detection),
        "projection_coordinate_frame": "standard_zup",
        "projection_coordinate_units": "m",
    }
    progress_total = len(selected_frames)
    if progress_callback:
        progress_callback(
            {
                "type": "start",
                "analysis_dir": str(analysis_dir),
                "processed": 0,
                "total": progress_total,
                "message": f"Loading YOLO model for {progress_total} lidar frames",
            }
        )

    model = YOLO(str(selected_weights))
    frame_rows: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    semantic_clouds: List[np.ndarray] = []
    stopped = False
    errors = 0
    started_monotonic = time.monotonic()

    for position, frame in enumerate(selected_frames, start=1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        frame_dir = analysis_dir / frame.frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_started = time.monotonic()
        try:
            rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise RuntimeError(f"Failed to read RGB image: {frame.rgb_path}")
            depth_raw = np.load(frame.depth_npy_path)
            camera_info = load_json(frame.camera_info_path, {})
            if not isinstance(camera_info, dict):
                raise RuntimeError(f"Invalid camera_info.json: {frame.camera_info_path}")

            result_list = model.predict(
                source=rgb_bgr,
                conf=float(conf),
                imgsz=int(imgsz),
                device=selected_device,
                verbose=False,
            )
            if not result_list:
                raise RuntimeError("YOLO model returned no results")
            detections, annotated = yolo_detections_from_result(result_list[0])
            annotated_path = frame_dir / "yolo_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated)

            projected_detections: List[Dict[str, Any]] = []
            for det_index, detection in enumerate(detections):
                cloud, projection = project_bbox_depth_to_world_standard_m(
                    depth_image=depth_raw,
                    camera_info=camera_info,
                    rgb_shape=rgb_bgr.shape[:2],
                    rgb_bbox_xyxy=[float(v) for v in detection.get("xyxy", [])],
                    class_name=str(detection.get("class_name", "")),
                    max_points=int(max_points_per_detection),
                )
                detection_payload = {
                    **detection,
                    "frame_index": frame.frame_index,
                    "frame_name": frame.frame_name,
                    "detection_index": det_index,
                    "observation_id": f"{frame.frame_name}:{det_index}",
                    "projection": projection,
                }
                if cloud.shape[0] > 0 and projection.get("status") == "ok":
                    detection_payload.update(
                        {
                            "point_count": int(cloud.shape[0]),
                            "center_world_m": projection.get("center_world_m", []),
                            "bbox_world_m": projection.get("bbox_world_m", {}),
                        }
                    )
                    observations.append(detection_payload)
                    semantic_clouds.append(cloud)
                else:
                    detection_payload["point_count"] = 0
                projected_detections.append(detection_payload)

            yolo_payload = {
                "weights_path": str(selected_weights),
                "annotated_path": str(annotated_path),
                "num_detections": len(detections),
                "detections": detections,
            }
            projection_payload = {
                "frame_index": frame.frame_index,
                "frame_name": frame.frame_name,
                "capture_dir": str(frame.capture_dir),
                "rgb_path": str(frame.rgb_path),
                "depth_npy_path": str(frame.depth_npy_path),
                "camera_info_path": str(frame.camera_info_path),
                "projection_coordinate_frame": "standard_zup",
                "projection_coordinate_units": "m",
                "detections": projected_detections,
            }
            write_json(frame_dir / "yolo_result.json", yolo_payload)
            write_json(frame_dir / "semantic_projection.json", projection_payload)
            frame_row = {
                "frame_index": frame.frame_index,
                "frame_name": frame.frame_name,
                "status": "ok",
                "capture_dir": str(frame.capture_dir),
                "analysis_frame_dir": str(frame_dir),
                "rgb_path": str(frame.rgb_path),
                "yolo_annotated_path": str(annotated_path),
                "yolo_result_path": str(frame_dir / "yolo_result.json"),
                "semantic_projection_path": str(frame_dir / "semantic_projection.json"),
                "num_detections": len(detections),
                "projected_detection_count": sum(1 for item in projected_detections if int(item.get("point_count", 0) or 0) > 0),
                "semantic_point_count": int(sum(int(item.get("point_count", 0) or 0) for item in projected_detections)),
                "elapsed_s": float(time.monotonic() - frame_started),
            }
        except Exception as exc:
            errors += 1
            frame_row = {
                "frame_index": frame.frame_index,
                "frame_name": frame.frame_name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "capture_dir": str(frame.capture_dir),
                "analysis_frame_dir": str(frame_dir),
                "num_detections": 0,
                "projected_detection_count": 0,
                "semantic_point_count": 0,
                "elapsed_s": float(time.monotonic() - frame_started),
            }
            write_json(frame_dir / "semantic_projection.json", frame_row)
        frame_rows.append(frame_row)
        if progress_callback:
            progress_callback(
                {
                    "type": "frame",
                    "analysis_dir": str(analysis_dir),
                    "processed": len(frame_rows),
                    "total": progress_total,
                    "row": frame_row,
                    "message": f"{len(frame_rows)}/{progress_total} {frame_row.get('status')} {frame.frame_name}",
                }
            )

    labels = dedupe_semantic_observations(observations, radius_m=float(dedupe_radius_m))
    semantic_points = (
        np.vstack(semantic_clouds).astype(np.float32, copy=False)
        if semantic_clouds
        else np.zeros((0, 6), dtype=np.float32)
    )
    semantic_npy_path = semantic_export_dir / "semantic_points_standard_m.npy"
    np.save(semantic_npy_path, semantic_points)
    open3d_outputs = flight.save_open3d_point_cloud_outputs(
        semantic_points,
        semantic_export_dir,
        basename="semantic_points_standard_m",
        voxel_cm=0.0,
        voxel_size=0.0,
        max_points=0,
        estimate_normals=False,
        coordinate_units="m",
    )
    labels_path = semantic_export_dir / "semantic_labels.json"
    write_json(
        labels_path,
        {
            "labels": labels,
            "label_count": len(labels),
            "observation_count": len(observations),
            "dedupe_radius_m": float(dedupe_radius_m),
            "projection_coordinate_frame": "standard_zup",
            "projection_coordinate_units": "m",
        },
    )
    finished_at = datetime.now().isoformat(timespec="milliseconds")
    summary = {
        "status": "stopped" if stopped else ("partial_error" if errors else "ok"),
        "stream_dir": str(stream_path),
        "analysis_dir": str(analysis_dir),
        "semantic_export_dir": str(semantic_export_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": float(time.monotonic() - started_monotonic),
        "config": config,
        "selected_frame_count": len(selected_frames),
        "processed_frame_count": len(frame_rows),
        "error_frame_count": int(errors),
        "semantic_observation_count": len(observations),
        "semantic_label_count": len(labels),
        "semantic_point_count": int(semantic_points.shape[0]),
        "dedupe_radius_m": float(dedupe_radius_m),
        "semantic_points_path": str(semantic_npy_path),
        "semantic_points_ply_path": open3d_outputs.get("ply_path", ""),
        "semantic_points_pcd_path": open3d_outputs.get("pcd_path", ""),
        "semantic_labels_path": str(labels_path),
        "open3d": open3d_outputs,
        "projection_coordinate_frame": "standard_zup",
        "projection_coordinate_units": "m",
        "frames": frame_rows,
    }
    summary_path = analysis_dir / "semantic_projection_summary.json"
    stable_summary_path = semantic_export_dir / "semantic_projection_summary.json"
    summary["summary_path"] = str(summary_path)
    summary["stable_summary_path"] = str(stable_summary_path)
    write_json(summary_path, summary)
    write_json(stable_summary_path, summary)
    update_open3d_summary_with_semantics(stream_path, summary)
    if progress_callback:
        progress_callback(
            {
                "type": "done",
                "analysis_dir": str(analysis_dir),
                "processed": len(frame_rows),
                "total": progress_total,
                "message": f"YOLO labels done: labels={len(labels)} observations={len(observations)}",
            }
        )
    return {
        **summary,
    }
