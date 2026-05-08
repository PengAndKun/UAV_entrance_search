from __future__ import annotations

import importlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UAV_FLOW_ROOT = Path(r"E:\github\UAV-Flow")
DEFAULT_HOUSES_CONFIG_PATH = PROJECT_ROOT / "assets" / "overhead_map" / "houses_config.json"

ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass(frozen=True)
class StreamFrame:
    frame_index: int
    frame_name: str
    capture_dir: Path
    rgb_path: Path
    depth_cm_path: Path
    pose_path: Path
    capture_path: Path
    trajectory_entry: Dict[str, Any]


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


def resolve_uav_flow_root(explicit_root: Optional[str | Path] = None) -> Path:
    raw = str(explicit_root or os.environ.get("UAV_FLOW_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_UAV_FLOW_ROOT


def load_houses_config(config_path: Optional[str | Path] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(config, dict) and config:
        return json.loads(json.dumps(config, default=_json_default))
    path = Path(config_path) if config_path else DEFAULT_HOUSES_CONFIG_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    payload = load_json(path, {})
    return payload if isinstance(payload, dict) else {}


def import_phase2_module(uav_flow_root: Optional[str | Path] = None):
    root = resolve_uav_flow_root(uav_flow_root)
    if not root.exists():
        raise RuntimeError(f"UAV-Flow root not found: {root}")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ultralytics_config_dir = PROJECT_ROOT / ".ultralytics"
    ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(ultralytics_config_dir))
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module("phase2_multimodal_fusion_analysis.fusion_entry_analysis")


def find_default_weights(uav_flow_root: Optional[str | Path] = None) -> Path:
    module = import_phase2_module(uav_flow_root)
    weights = module.find_latest_phase2_weights()
    return Path(weights).resolve()


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


def frame_from_capture_dir(
    stream_dir: Path,
    capture_dir: Path,
    *,
    trajectory_entry: Optional[Dict[str, Any]] = None,
) -> Optional[StreamFrame]:
    entry = trajectory_entry if isinstance(trajectory_entry, dict) else {}
    frame_name = capture_dir.name
    frame_index = int(entry.get("frame_index") or parse_frame_index(frame_name) or 0)

    rgb_path = resolve_stream_path(stream_dir, entry.get("rgb_path")) if entry.get("rgb_path") else capture_dir / "rgb.png"
    depth_cm_path = (
        resolve_stream_path(stream_dir, entry.get("depth_cm_path"))
        if entry.get("depth_cm_path")
        else capture_dir / "depth_cm.png"
    )
    pose_path = capture_dir / "pose.json"
    capture_path = capture_dir / "capture.json"
    if not rgb_path.exists() or not depth_cm_path.exists():
        return None
    return StreamFrame(
        frame_index=frame_index,
        frame_name=frame_name,
        capture_dir=capture_dir,
        rgb_path=rgb_path,
        depth_cm_path=depth_cm_path,
        pose_path=pose_path,
        capture_path=capture_path,
        trajectory_entry=dict(entry),
    )


def scan_stream_frames(stream_dir: str | Path, *, stride: int = 1, max_frames: int = 0) -> List[StreamFrame]:
    stream_path = Path(stream_dir).resolve()
    stride = max(1, int(stride or 1))
    max_frames = max(0, int(max_frames or 0))
    frames: List[StreamFrame] = []
    seen: set[str] = set()

    trajectory = load_json(stream_path / "trajectory.json", {})
    entries = trajectory.get("trajectory", []) if isinstance(trajectory, dict) else []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_capture_dir = entry.get("capture_dir")
            if raw_capture_dir:
                capture_dir = resolve_stream_path(stream_path, raw_capture_dir)
            elif entry.get("rgb_path"):
                capture_dir = resolve_stream_path(stream_path, entry.get("rgb_path")).parent
            else:
                continue
            key = str(capture_dir.resolve())
            if key in seen:
                continue
            frame = frame_from_capture_dir(stream_path, capture_dir, trajectory_entry=entry)
            if frame is not None:
                frames.append(frame)
                seen.add(key)

    frames_root = stream_path / "frames"
    if frames_root.exists():
        for capture_dir in sorted(path for path in frames_root.glob("frame_*") if path.is_dir()):
            key = str(capture_dir.resolve())
            if key in seen:
                continue
            frame = frame_from_capture_dir(stream_path, capture_dir)
            if frame is not None:
                frames.append(frame)
                seen.add(key)

    frames.sort(key=lambda item: (item.frame_index, item.frame_name))
    selected = frames[::stride]
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def build_phase2_state(frame: StreamFrame, stream_dir: Path, stream_summary: Dict[str, Any]) -> Dict[str, Any]:
    pose_payload = load_json(frame.pose_path, {})
    capture_payload = load_json(frame.capture_path, {})
    pose = pose_payload.get("pose") if isinstance(pose_payload, dict) else {}
    state: Dict[str, Any] = {
        "pose": pose if isinstance(pose, dict) else {},
        "frame_index": frame.frame_index,
        "capture_time": (
            pose_payload.get("capture_time")
            if isinstance(pose_payload, dict)
            else frame.trajectory_entry.get("capture_time", "")
        ),
        "task_label": stream_summary.get("task_title") or stream_summary.get("safe_task_title") or stream_dir.name,
        "stream_capture": {
            "stream_dir": str(stream_dir),
            "frame_name": frame.frame_name,
            "capture_dir": str(frame.capture_dir),
            "rgb_path": str(frame.rgb_path),
            "depth_cm_path": str(frame.depth_cm_path),
            "trajectory_entry": frame.trajectory_entry,
        },
    }
    if isinstance(pose_payload, dict):
        state["pose_payload"] = pose_payload
    if isinstance(capture_payload, dict):
        state["capture_payload"] = capture_payload
        if "depth_summary" in capture_payload:
            state["depth_summary"] = capture_payload.get("depth_summary")
    return state


def no_memory_write_update(**_kwargs: Any) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "reason": "stream_offline_analysis_no_memory_write",
        "memory_write_disabled": True,
    }


def run_phase2_no_memory(module: Any, **kwargs: Any) -> Dict[str, Any]:
    original_update = getattr(module, "update_entry_search_memory_from_fusion", None)
    if original_update is not None:
        setattr(module, "update_entry_search_memory_from_fusion", no_memory_write_update)
    try:
        return module.run_phase2_fusion_analysis(**kwargs)
    finally:
        if original_update is not None:
            setattr(module, "update_entry_search_memory_from_fusion", original_update)


def fusion_overlay_path_for_run(run_dir: Path, result: Optional[Dict[str, Any]] = None) -> str:
    candidates: List[Path] = []
    if isinstance(result, dict):
        labeling_dir = result.get("labeling_dir")
        if labeling_dir:
            candidates.append(Path(str(labeling_dir)) / "fusion_overlay.png")
        for key in ("fusion_overlay_path",):
            value = result.get(key)
            if value:
                candidates.append(Path(str(value)))
    candidates.extend(
        [
            run_dir / "labeling" / "fusion_overlay.png",
            run_dir / "labeling" / "yolo_annotated.jpg",
            run_dir / "labeling" / "depth_overlay.png",
        ]
    )
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def summarize_phase2_result(frame: StreamFrame, run_dir: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    yolo = result.get("yolo", {}) if isinstance(result.get("yolo"), dict) else {}
    fusion = result.get("fusion", {}) if isinstance(result.get("fusion"), dict) else {}
    detections = yolo.get("detections", []) if isinstance(yolo.get("detections"), list) else []
    top_detection = detections[0] if detections else {}
    panel_summary = result.get("panel_summary")
    row = {
        "frame_index": frame.frame_index,
        "frame_name": frame.frame_name,
        "status": result.get("status", "ok"),
        "capture_dir": str(frame.capture_dir),
        "rgb_path": str(frame.rgb_path),
        "depth_cm_path": str(frame.depth_cm_path),
        "run_dir": str(run_dir),
        "labeling_dir": str(run_dir / "labeling"),
        "fusion_overlay_path": fusion_overlay_path_for_run(run_dir, result),
        "yolo_annotated_path": str(run_dir / "labeling" / "yolo_annotated.jpg"),
        "depth_overlay_path": str(run_dir / "labeling" / "depth_overlay.png"),
        "fusion_result_path": str(run_dir / "labeling" / "fusion_result.json"),
        "yolo_result_path": str(run_dir / "labeling" / "yolo_result.json"),
        "depth_result_path": str(run_dir / "labeling" / "depth_result.json"),
        "num_detections": int(yolo.get("num_detections") or 0),
        "top_class": top_detection.get("class_name"),
        "top_confidence": top_detection.get("confidence"),
        "entry_class": fusion.get("entry_class"),
        "entry_semantic_confidence": fusion.get("entry_semantic_confidence"),
        "entry_distance_cm": fusion.get("entry_distance_cm"),
        "opening_width_cm": fusion.get("opening_width_cm"),
        "traversable": bool(fusion.get("traversable", False)),
        "crossing_ready": bool(fusion.get("crossing_ready", False)),
        "recommended_subgoal": fusion.get("recommended_subgoal"),
        "recommended_action_hint": fusion.get("recommended_action_hint"),
        "target_conditioned_state": fusion.get("target_conditioned_state"),
        "target_conditioned_action_hint": fusion.get("target_conditioned_action_hint"),
        "decision_reason": fusion.get("decision_reason"),
        "decision_text": fusion.get("decision_text"),
        "panel_summary": panel_summary,
    }
    return row


def summarize_error_frame(frame: StreamFrame, run_dir: Path, exc: BaseException) -> Dict[str, Any]:
    return {
        "frame_index": frame.frame_index,
        "frame_name": frame.frame_name,
        "status": "error",
        "error": str(exc),
        "capture_dir": str(frame.capture_dir),
        "rgb_path": str(frame.rgb_path),
        "depth_cm_path": str(frame.depth_cm_path),
        "run_dir": str(run_dir),
        "labeling_dir": str(run_dir / "labeling"),
        "fusion_overlay_path": "",
        "num_detections": 0,
    }


def pick_best_entry_candidate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [row for row in rows if row.get("status") == "ok" and row.get("entry_class")]
    if not candidates:
        return {}

    def score(row: Dict[str, Any]) -> Tuple[float, float, float]:
        conf = float(row.get("entry_semantic_confidence") or row.get("top_confidence") or 0.0)
        traversable = 1.0 if row.get("traversable") else 0.0
        distance = float(row.get("entry_distance_cm") or 999999.0)
        return (traversable, conf, -distance)

    return max(candidates, key=score)


def write_analysis_outputs(
    analysis_dir: Path,
    *,
    stream_dir: Path,
    rows: List[Dict[str, Any]],
    total_frames: int,
    started_at: str,
    completed: bool,
    stopped: bool,
    config: Dict[str, Any],
    log_lines: List[str],
) -> None:
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    error_count = sum(1 for row in rows if row.get("status") == "error")
    summary = {
        "status": "complete" if completed else ("stopped" if stopped else "running"),
        "stream_dir": str(stream_dir),
        "analysis_dir": str(analysis_dir),
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        "total_selected_frames": int(total_frames),
        "processed_frames": len(rows),
        "ok_frames": ok_count,
        "error_frames": error_count,
        "config": config,
        "best_entry_candidate": pick_best_entry_candidate(rows),
    }
    index = {
        **summary,
        "frames": rows,
    }
    write_json(analysis_dir / "analysis_summary.json", summary)
    write_json(analysis_dir / "analysis_index.json", index)
    (analysis_dir / "analysis_log.txt").write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")


def run_stream_analysis(
    *,
    stream_dir: str | Path,
    weights_path: Optional[str | Path] = None,
    uav_flow_root: Optional[str | Path] = None,
    houses_config_path: Optional[str | Path] = None,
    houses_config: Optional[Dict[str, Any]] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
    stride: int = 1,
    max_frames: int = 0,
    stop_event: Optional[threading.Event] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    stream_path = Path(stream_dir).resolve()
    if not stream_path.exists():
        raise RuntimeError(f"Stream folder not found: {stream_path}")

    phase2_module = import_phase2_module(uav_flow_root)
    selected_weights = Path(weights_path).expanduser().resolve() if weights_path else Path(phase2_module.find_latest_phase2_weights()).resolve()
    if not selected_weights.exists():
        raise RuntimeError(f"YOLO weights not found: {selected_weights}")

    selected_frames = scan_stream_frames(stream_path, stride=stride, max_frames=max_frames)
    if not selected_frames:
        raise RuntimeError(f"No stream frames with rgb.png and depth_cm.png found in {stream_path}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    analysis_dir = stream_path / "analysis" / timestamp
    analysis_dir.mkdir(parents=True, exist_ok=True)

    stream_summary = load_json(stream_path / "trajectory.json", {})
    if not isinstance(stream_summary, dict):
        stream_summary = load_json(stream_path / "stream_capture.json", {})
    if not isinstance(stream_summary, dict):
        stream_summary = {}
    houses_cfg = load_houses_config(houses_config_path, houses_config)
    config = {
        "weights_path": str(selected_weights),
        "uav_flow_root": str(resolve_uav_flow_root(uav_flow_root)),
        "houses_config_path": str(houses_config_path or DEFAULT_HOUSES_CONFIG_PATH),
        "conf": float(conf),
        "imgsz": int(imgsz),
        "device": str(device),
        "stride": int(max(1, stride)),
        "max_frames": int(max(0, max_frames)),
        "memory_write_disabled": True,
    }
    started_at = datetime.now().isoformat(timespec="milliseconds")
    rows: List[Dict[str, Any]] = []
    log_lines: List[str] = [
        f"[{started_at}] stream analysis started",
        f"stream_dir={stream_path}",
        f"analysis_dir={analysis_dir}",
        f"weights={selected_weights}",
        f"selected_frames={len(selected_frames)}",
    ]
    write_analysis_outputs(
        analysis_dir,
        stream_dir=stream_path,
        rows=rows,
        total_frames=len(selected_frames),
        started_at=started_at,
        completed=False,
        stopped=False,
        config=config,
        log_lines=log_lines,
    )

    if progress_callback:
        progress_callback(
            {
                "type": "start",
                "analysis_dir": str(analysis_dir),
                "total": len(selected_frames),
                "processed": 0,
                "message": f"Analyzing {len(selected_frames)} frames",
            }
        )

    stopped = False
    for position, frame in enumerate(selected_frames, start=1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            log_lines.append(f"[{datetime.now().isoformat(timespec='milliseconds')}] stopped before {frame.frame_name}")
            break
        run_dir = analysis_dir / frame.frame_name
        run_dir.mkdir(parents=True, exist_ok=True)
        frame_started = time.monotonic()
        try:
            rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise RuntimeError(f"Failed to read rgb image: {frame.rgb_path}")
            depth_raw = cv2.imread(str(frame.depth_cm_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise RuntimeError(f"Failed to read depth image: {frame.depth_cm_path}")
            state = build_phase2_state(frame, stream_path, stream_summary)
            result = run_phase2_no_memory(
                phase2_module,
                rgb_bgr=rgb_bgr,
                depth_raw=depth_raw,
                existing_run_dir=run_dir,
                weights=selected_weights,
                label=frame.frame_name,
                camera_info={
                    "source": "stream_capture_offline",
                    "rgb_shape": list(rgb_bgr.shape),
                    "depth_shape": list(depth_raw.shape),
                },
                state=state,
                houses_config=houses_cfg,
                conf=float(conf),
                imgsz=int(imgsz),
                device=str(device),
            )
            row = summarize_phase2_result(frame, run_dir, result)
            elapsed = time.monotonic() - frame_started
            log_lines.append(f"[{datetime.now().isoformat(timespec='milliseconds')}] ok {frame.frame_name} {elapsed:.2f}s")
        except Exception as exc:
            row = summarize_error_frame(frame, run_dir, exc)
            log_lines.append(f"[{datetime.now().isoformat(timespec='milliseconds')}] error {frame.frame_name}: {exc}")
        rows.append(row)
        write_analysis_outputs(
            analysis_dir,
            stream_dir=stream_path,
            rows=rows,
            total_frames=len(selected_frames),
            started_at=started_at,
            completed=False,
            stopped=stopped,
            config=config,
            log_lines=log_lines,
        )
        if progress_callback:
            progress_callback(
                {
                    "type": "frame",
                    "analysis_dir": str(analysis_dir),
                    "total": len(selected_frames),
                    "processed": len(rows),
                    "row": row,
                    "message": f"{len(rows)}/{len(selected_frames)} {row.get('status')} {frame.frame_name}",
                }
            )

    completed = not stopped and len(rows) >= len(selected_frames)
    write_analysis_outputs(
        analysis_dir,
        stream_dir=stream_path,
        rows=rows,
        total_frames=len(selected_frames),
        started_at=started_at,
        completed=completed,
        stopped=stopped,
        config=config,
        log_lines=log_lines,
    )
    result = {
        "status": "stopped" if stopped else "ok",
        "stream_dir": str(stream_path),
        "analysis_dir": str(analysis_dir),
        "summary_path": str(analysis_dir / "analysis_summary.json"),
        "index_path": str(analysis_dir / "analysis_index.json"),
        "log_path": str(analysis_dir / "analysis_log.txt"),
        "frames": rows,
        "processed_frames": len(rows),
        "total_selected_frames": len(selected_frames),
        "best_entry_candidate": pick_best_entry_candidate(rows),
    }
    if progress_callback:
        progress_callback(
            {
                "type": "complete",
                "analysis_dir": str(analysis_dir),
                "total": len(selected_frames),
                "processed": len(rows),
                "result": result,
                "message": f"Analysis {'stopped' if stopped else 'complete'}: {len(rows)}/{len(selected_frames)} frames",
            }
        )
    return result
