from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class LocalObstacleMapConfig:
    voxel_cm: float = 25.0
    radius_cm: float = 1200.0
    ttl_frames: int = 150
    ttl_seconds: float = 60.0
    swept_radius_cm: float = 100.0
    swept_vertical_radius_cm: float = 90.0
    max_points_per_update: int = 1600
    max_preview_points: int = 300
    synthetic_depth_cm: float = 120.0
    close_depth_cm: float = 180.0


class LocalObstacleMap:
    def __init__(self, config: Optional[LocalObstacleMapConfig] = None) -> None:
        self.config = config or LocalObstacleMapConfig()
        self._voxels: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._last_frame_id = 0

    def clear(self) -> None:
        self._voxels.clear()
        self._last_frame_id = 0

    def occupied_voxel_count(self) -> int:
        return len(self._voxels)

    def update_from_event(self, event: Dict[str, Any], current_pose: Dict[str, Any]) -> Dict[str, Any]:
        event_dict = event if isinstance(event, dict) else {}
        pose = current_pose if isinstance(current_pose, dict) else {}
        frame_id = self._frame_id(event_dict)
        self._last_frame_id = max(self._last_frame_id, frame_id)
        self._decay(frame_id)
        semantic_hint = self._semantic_hint(event_dict)
        points: List[Tuple[float, float, float, str, float, str]] = []
        points.extend(self._points_from_standard_cloud(event_dict, pose, semantic_hint))
        if not points:
            points.extend(self._points_from_depth_summary(event_dict, pose, semantic_hint))
        added = 0
        touched: List[Dict[str, Any]] = []
        now = time.time()
        for x_cm, y_cm, z_cm, source, confidence, hint in points:
            key = self._voxel_key(x_cm, y_cm, z_cm)
            existed = key in self._voxels
            center = self._voxel_center(key)
            item = self._voxels.setdefault(
                key,
                {
                    "count": 0,
                    "first_seen_frame": frame_id,
                    "first_seen_time": now,
                    "center": {"x": center[0], "y": center[1], "z": center[2]},
                },
            )
            item["count"] = int(item.get("count", 0) or 0) + 1
            item["last_seen_frame"] = frame_id
            item["last_seen_time"] = now
            item["source"] = source
            item["confidence"] = max(float(item.get("confidence", 0.0) or 0.0), float(confidence))
            item["semantic_hint"] = hint
            if not existed:
                added += 1
            if len(touched) < self.config.max_preview_points:
                touched.append(self._voxel_record(key, item))
        summary = self.summary()
        return {
            "schema": "local_obstacle_map_update_v1",
            "frame_id": frame_id,
            "source": "pointcloud" if any(point[3] == "pointcloud_standard_m" for point in points) else "depth_or_or2",
            "input_point_count": len(points),
            "added_voxel_count": added,
            "touched_voxel_count": len(touched),
            "occupied_voxel_count": summary["occupied_voxel_count"],
            "semantic_hint": semantic_hint,
            "preview_points": touched,
            "summary": summary,
        }

    def query_safety(
        self,
        current_pose: Dict[str, Any],
        payload: Dict[str, Any],
        *,
        safety_radius_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        pose = current_pose if isinstance(current_pose, dict) else {}
        action = payload if isinstance(payload, dict) else {}
        labels = self._payload_direction_labels(action)
        start = (
            self._as_float(pose.get("x"), 0.0),
            self._as_float(pose.get("y"), 0.0),
            self._as_float(pose.get("z"), 0.0),
        )
        end = self._predict_pose_xyz(pose, action)
        distance = math.dist(start, end)
        if distance <= 0.5:
            return {
                "checked": True,
                "safe": True,
                "reason": "no_translation",
                "blocked_directions": [],
                "occupied_voxel_count": self.occupied_voxel_count(),
                "intersecting_voxels": [],
            }
        radius = float(safety_radius_cm if safety_radius_cm is not None else self.config.swept_radius_cm)
        blocked: List[Dict[str, Any]] = []
        for key, item in self._voxels.items():
            center = item.get("center", {}) if isinstance(item.get("center"), dict) else {}
            point = (
                self._as_float(center.get("x"), 0.0),
                self._as_float(center.get("y"), 0.0),
                self._as_float(center.get("z"), 0.0),
            )
            horizontal_dist, vertical_dist, along = self._point_segment_distance_components(point, start, end)
            if 0.0 <= along <= 1.0 and horizontal_dist <= radius and vertical_dist <= self.config.swept_vertical_radius_cm:
                blocked.append(
                    {
                        **self._voxel_record(key, item),
                        "horizontal_distance_cm": round(horizontal_dist, 3),
                        "vertical_distance_cm": round(vertical_dist, 3),
                        "segment_t": round(along, 4),
                    }
                )
                if len(blocked) >= 80:
                    break
        safe = not blocked
        return {
            "checked": True,
            "safe": safe,
            "reason": "local_3d_clear" if safe else f"local_3d_occupancy_blocked:{','.join(labels) if labels else 'path'}",
            "blocked_directions": [] if safe else labels,
            "occupied_voxel_count": self.occupied_voxel_count(),
            "intersecting_voxels": blocked,
            "payload": self._json_safe(action),
            "current_pose": self._json_safe(pose),
            "predicted_xyz": {"x": round(end[0], 3), "y": round(end[1], 3), "z": round(end[2], 3)},
        }

    def blocked_directions(self, current_pose: Dict[str, Any], candidate_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for name, payload in (candidate_payloads if isinstance(candidate_payloads, dict) else {}).items():
            result[str(name)] = self.query_safety(current_pose, payload)
        return result

    def summary(self) -> Dict[str, Any]:
        preview = self.preview_points()
        return {
            "schema": "local_obstacle_summary_v1",
            "occupied_voxel_count": self.occupied_voxel_count(),
            "voxel_cm": float(self.config.voxel_cm),
            "radius_cm": float(self.config.radius_cm),
            "ttl_frames": int(self.config.ttl_frames),
            "ttl_seconds": float(self.config.ttl_seconds),
            "preview_points": preview,
            "last_frame_id": int(self._last_frame_id),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }

    def preview_points(self) -> List[Dict[str, Any]]:
        rows = [self._voxel_record(key, item) for key, item in self._voxels.items()]
        rows.sort(key=lambda item: (-float(item.get("confidence", 0.0) or 0.0), -int(item.get("last_seen_frame", 0) or 0)))
        return rows[: max(0, int(self.config.max_preview_points))]

    def write_artifacts(self, output_dir: Path) -> Dict[str, Any]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        (target / "local_obstacle_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self.write_preview_ply(target / "local_obstacle_preview.ply")
        return summary

    def write_preview_ply(self, path: Path) -> None:
        points = self.preview_points()
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        for item in points:
            confidence = float(item.get("confidence", 0.0) or 0.0)
            r = 255
            g = int(max(64, min(180, 80 + 100 * confidence)))
            b = 40
            lines.append(f"{float(item.get('x', 0.0)):.3f} {float(item.get('y', 0.0)):.3f} {float(item.get('z', 0.0)):.3f} {r} {g} {b}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _points_from_standard_cloud(
        self,
        event: Dict[str, Any],
        pose: Dict[str, Any],
        semantic_hint: str,
    ) -> List[Tuple[float, float, float, str, float, str]]:
        raw_path = (
            event.get("point_cloud_world_standard_m_npy_path")
            or event.get("pointcloud_path")
            or event.get("point_cloud_world_npy_path")
        )
        if not raw_path:
            return []
        path = Path(str(raw_path))
        if not path.is_file():
            return []
        try:
            arr = np.load(path)
        except Exception:
            return []
        data = np.asarray(arr, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] == 0:
            return []
        xyz = data[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if xyz.shape[0] == 0:
            return []
        uses_standard_m = "standard_m" in str(path).lower() or str(event.get("coordinate_units", "")).lower() == "m"
        if uses_standard_m:
            world_cm = np.column_stack((xyz[:, 0] * 100.0, -xyz[:, 1] * 100.0, xyz[:, 2] * 100.0))
        else:
            world_cm = xyz
        current = np.asarray(
            [
                self._as_float(pose.get("x"), 0.0),
                self._as_float(pose.get("y"), 0.0),
                self._as_float(pose.get("z"), 0.0),
            ],
            dtype=np.float32,
        )
        dist = np.linalg.norm(world_cm - current.reshape(1, 3), axis=1)
        selected = world_cm[dist <= float(self.config.radius_cm)]
        if selected.shape[0] == 0:
            return []
        max_count = max(1, int(self.config.max_points_per_update))
        if selected.shape[0] > max_count:
            indices = np.linspace(0, selected.shape[0] - 1, num=max_count, dtype=np.int64)
            selected = selected[indices]
        return [(float(row[0]), float(row[1]), float(row[2]), "pointcloud_standard_m", 0.95, semantic_hint) for row in selected]

    def _points_from_depth_summary(
        self,
        event: Dict[str, Any],
        pose: Dict[str, Any],
        semantic_hint: str,
    ) -> List[Tuple[float, float, float, str, float, str]]:
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        if not summary and isinstance(event.get("depth_obstacle_summary"), dict):
            summary = event.get("depth_obstacle_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        directions = [
            ("forward", "front_min_depth_cm", "forward_swept_clear"),
            ("right", "right_min_depth_cm", "right_swept_clear"),
            ("left", "left_min_depth_cm", "left_swept_clear"),
            ("up", "up_min_depth_cm", "up_swept_clear"),
            ("down", "down_min_depth_cm", "down_swept_clear"),
        ]
        prediction = event.get("or2_prediction", {}) if isinstance(event.get("or2_prediction"), dict) else {}
        or2_blocked = str(prediction.get("front_risk_state", "") or "").lower() in {"must_stop", "obstacle_warning", "clearance_warning"}
        points: List[Tuple[float, float, float, str, float, str]] = []
        for direction, depth_key, clear_key in directions:
            depth = self._as_float(summary.get(depth_key), 0.0)
            blocked = summary.get(clear_key) is False or (0.0 < depth <= float(self.config.close_depth_cm))
            if direction == "forward" and or2_blocked:
                blocked = True
            if not blocked:
                continue
            distance = depth if depth > 0.0 else float(self.config.synthetic_depth_cm)
            for offset in (-0.5, 0.0, 0.5):
                x_cm, y_cm, z_cm = self._synthetic_point_for_direction(pose, direction, distance, lateral_offset=offset)
                points.append((x_cm, y_cm, z_cm, f"depth_sector_{direction}", 0.7 if depth > 0.0 else 0.45, semantic_hint))
        return points

    def _synthetic_point_for_direction(
        self,
        pose: Dict[str, Any],
        direction: str,
        distance_cm: float,
        *,
        lateral_offset: float,
    ) -> Tuple[float, float, float]:
        x0 = self._as_float(pose.get("x"), 0.0)
        y0 = self._as_float(pose.get("y"), 0.0)
        z0 = self._as_float(pose.get("z"), 0.0)
        yaw = math.radians(self._as_float(pose.get("yaw", pose.get("task_yaw")), 0.0))
        forward = (math.cos(yaw), math.sin(yaw), 0.0)
        right = (-math.sin(yaw), math.cos(yaw), 0.0)
        up = (0.0, 0.0, 1.0)
        distance = max(20.0, min(float(self.config.radius_cm), float(distance_cm)))
        spread = float(self.config.voxel_cm) * float(lateral_offset)
        if direction == "forward":
            vector = forward
            side = right
        elif direction == "backoff":
            vector = (-forward[0], -forward[1], 0.0)
            side = right
        elif direction == "right":
            vector = right
            side = forward
        elif direction == "left":
            vector = (-right[0], -right[1], 0.0)
            side = forward
        elif direction == "up":
            vector = up
            side = forward
        elif direction == "down":
            vector = (0.0, 0.0, -1.0)
            side = forward
        else:
            vector = forward
            side = right
        return (
            x0 + vector[0] * distance + side[0] * spread,
            y0 + vector[1] * distance + side[1] * spread,
            z0 + vector[2] * distance + side[2] * spread,
        )

    def _decay(self, frame_id: int) -> None:
        now = time.time()
        ttl_frames = max(1, int(self.config.ttl_frames))
        ttl_seconds = max(0.1, float(self.config.ttl_seconds))
        stale = [
            key
            for key, item in self._voxels.items()
            if int(item.get("last_seen_frame", frame_id) or frame_id) < frame_id - ttl_frames
            or float(item.get("last_seen_time", now) or now) < now - ttl_seconds
        ]
        for key in stale:
            self._voxels.pop(key, None)

    def _voxel_key(self, x_cm: float, y_cm: float, z_cm: float) -> Tuple[int, int, int]:
        voxel = max(1.0, float(self.config.voxel_cm))
        return (int(math.floor(float(x_cm) / voxel)), int(math.floor(float(y_cm) / voxel)), int(math.floor(float(z_cm) / voxel)))

    def _voxel_center(self, key: Tuple[int, int, int]) -> Tuple[float, float, float]:
        voxel = max(1.0, float(self.config.voxel_cm))
        return ((key[0] + 0.5) * voxel, (key[1] + 0.5) * voxel, (key[2] + 0.5) * voxel)

    def _voxel_record(self, key: Tuple[int, int, int], item: Dict[str, Any]) -> Dict[str, Any]:
        center = item.get("center", {}) if isinstance(item.get("center"), dict) else {}
        return {
            "voxel": [int(key[0]), int(key[1]), int(key[2])],
            "x": round(self._as_float(center.get("x"), 0.0), 3),
            "y": round(self._as_float(center.get("y"), 0.0), 3),
            "z": round(self._as_float(center.get("z"), 0.0), 3),
            "count": int(item.get("count", 0) or 0),
            "last_seen_frame": int(item.get("last_seen_frame", 0) or 0),
            "source": str(item.get("source", "") or ""),
            "confidence": round(float(item.get("confidence", 0.0) or 0.0), 4),
            "semantic_hint": str(item.get("semantic_hint", "") or ""),
        }

    def _point_segment_distance_components(
        self,
        point: Tuple[float, float, float],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        sx, sy, sz = start
        ex, ey, ez = end
        px, py, pz = point
        vx, vy, vz = ex - sx, ey - sy, ez - sz
        length_sq = vx * vx + vy * vy + vz * vz
        if length_sq <= 1e-9:
            return math.hypot(px - sx, py - sy), abs(pz - sz), 0.0
        t = ((px - sx) * vx + (py - sy) * vy + (pz - sz) * vz) / length_sq
        t_clamped = max(0.0, min(1.0, t))
        nearest = (sx + t_clamped * vx, sy + t_clamped * vy, sz + t_clamped * vz)
        return math.hypot(px - nearest[0], py - nearest[1]), abs(pz - nearest[2]), float(t_clamped)

    def _predict_pose_xyz(self, pose: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[float, float, float]:
        yaw = math.radians(self._as_float(pose.get("yaw", pose.get("task_yaw")), 0.0))
        forward = self._as_float(payload.get("forward_cm"), 0.0)
        right = self._as_float(payload.get("right_cm"), 0.0)
        up = self._as_float(payload.get("up_cm"), 0.0)
        dx = forward * math.cos(yaw) - right * math.sin(yaw)
        dy = forward * math.sin(yaw) + right * math.cos(yaw)
        return (
            self._as_float(pose.get("x"), 0.0) + dx,
            self._as_float(pose.get("y"), 0.0) + dy,
            self._as_float(pose.get("z"), 0.0) + up,
        )

    def _payload_direction_labels(self, payload: Dict[str, Any]) -> List[str]:
        labels: List[str] = []
        forward = self._as_float(payload.get("forward_cm"), 0.0)
        right = self._as_float(payload.get("right_cm"), 0.0)
        up = self._as_float(payload.get("up_cm"), 0.0)
        if forward > 0.5:
            labels.append("forward")
        elif forward < -0.5:
            labels.append("backoff")
        if right > 0.5:
            labels.append("right")
        elif right < -0.5:
            labels.append("left")
        if up > 0.5:
            labels.append("up")
        elif up < -0.5:
            labels.append("down")
        return labels or ["path"]

    def _semantic_hint(self, event: Dict[str, Any]) -> str:
        prediction = event.get("or2_prediction") if isinstance(event.get("or2_prediction"), dict) else {}
        for key in ("semantic_hint", "obstacle_hint", "representation_obstacle_hint"):
            value = str(event.get(key, "") or "").strip()
            if value:
                return value
        state = str(prediction.get("front_risk_state", "") or "").strip()
        if state:
            return f"or2_{state}"
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        geometry = str(summary.get("obstacle_geometry", "") or "").strip()
        return geometry or "unknown"

    def _frame_id(self, event: Dict[str, Any]) -> int:
        try:
            return max(0, int(float(event.get("frame_id", self._last_frame_id + 1) or self._last_frame_id + 1)))
        except Exception:
            return self._last_frame_id + 1

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except Exception:
            result = float(default)
        return result if math.isfinite(result) else float(default)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
