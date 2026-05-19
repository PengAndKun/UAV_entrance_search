from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import run_drone_flight as flight

from .geometry_v0 import best_candidate_action, score_candidate_actions, selected_action_reason, summarize_geometry_v0


ACTION_PAYLOADS: Dict[str, Dict[str, float]] = {
    "hold": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "hold"},
    "slow_forward": {"forward_cm": 5.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "slow_forward"},
    "forward": {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "forward"},
    "backoff": {"forward_cm": -20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "backoff"},
    "yaw_left": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": -30.0, "action_name": "yaw_left"},
    "yaw_right": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 30.0, "action_name": "yaw_right"},
    "up": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 20.0, "yaw_delta_deg": 0.0, "action_name": "up"},
    "down": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": -20.0, "yaw_delta_deg": 0.0, "action_name": "down"},
    "left": {"forward_cm": 0.0, "right_cm": -20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "side_step_left"},
    "right": {"forward_cm": 0.0, "right_cm": 20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "side_step_right"},
}


def sanitize(value: str, default: str) -> str:
    text = str(value or "").strip() or default
    return flight.sanitize_capture_task_title(text, default_value=default)


def pose_yaw(pose: Dict[str, Any]) -> float:
    return float(pose.get("yaw", pose.get("task_yaw", pose.get("yaw_deg", 0.0))) or 0.0)


def relative_target(pose: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, float]:
    if not target:
        return {"distance_cm": 0.0, "bearing_deg_body": 0.0, "dz_cm": 0.0}
    x = float(pose.get("x", 0.0) or 0.0)
    y = float(pose.get("y", 0.0) or 0.0)
    z = float(pose.get("z", 0.0) or 0.0)
    yaw = pose_yaw(pose)
    dx = float(target.get("x", x) or x) - x
    dy = float(target.get("y", y) or y) - y
    dz = float(target.get("z", z) or z) - z
    absolute = math.degrees(math.atan2(dy, dx)) if dx or dy else yaw
    bearing = (absolute - yaw + 180.0) % 360.0 - 180.0
    if bearing == -180.0:
        bearing = 180.0
    return {"distance_cm": float(math.hypot(dx, dy)), "bearing_deg_body": float(bearing), "dz_cm": float(dz)}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def qa_report(session_dir: Path, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_rgb = 0
    missing_depth = 0
    missing_pointcloud = 0
    missing_pose = 0
    missing_action = 0
    danger_forward = 0
    unsafe_down = 0
    for event in events:
        if not Path(str(event.get("rgb_path", ""))).exists():
            missing_rgb += 1
        depth_path = Path(str(event.get("depth_npy_path", "") or event.get("depth_cm_path", "")))
        if not depth_path.exists():
            missing_depth += 1
        if not Path(str(event.get("pointcloud_path", ""))).exists():
            missing_pointcloud += 1
        if not isinstance(event.get("current_pose"), dict) or not event.get("current_pose"):
            missing_pose += 1
        action = event.get("executed_action")
        if not isinstance(action, dict):
            missing_action += 1
            action = {}
        summary = event.get("pointcloud_summary") if isinstance(event.get("pointcloud_summary"), dict) else {}
        front_min = float(summary.get("front_min_depth_cm", 0.0) or 0.0)
        if front_min > 0.0 and front_min < 250.0 and float(action.get("forward_cm", 0.0) or 0.0) > 1.0:
            danger_forward += 1
        pose = event.get("current_pose") if isinstance(event.get("current_pose"), dict) else {}
        pose_z = float(pose.get("z", 0.0) or 0.0)
        if float(action.get("up_cm", 0.0) or 0.0) < -1.0 and (not summary.get("down_swept_clear", False) or pose_z <= 100.0):
            unsafe_down += 1
    report = {
        "status": "ok",
        "session_dir": str(session_dir),
        "event_count": len(events),
        "valid_frame_count": max(0, len(events) - max(missing_rgb, missing_pointcloud, missing_pose, missing_action)),
        "missing_rgb_count": missing_rgb,
        "missing_depth_count": missing_depth,
        "missing_pointcloud_count": missing_pointcloud,
        "missing_pose_count": missing_pose,
        "missing_action_count": missing_action,
        "danger_forward_violation_count": danger_forward,
        "unsafe_down_action_count": unsafe_down,
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    write_json(session_dir / "collection_quality_report.json", report)
    return report


def build_event(
    result: Dict[str, Any],
    *,
    session_dir: Path,
    frame_id: int,
    args: argparse.Namespace,
    executed_action: Dict[str, float],
    target_waypoint: Dict[str, Any],
) -> Dict[str, Any]:
    pose = result.get("pose", {}) if isinstance(result.get("pose"), dict) else {}
    rel = relative_target(pose, target_waypoint)
    event = {
        "frame_id": frame_id,
        "timestamp": result.get("capture_time", datetime.now().isoformat(timespec="milliseconds")),
        "session_id": session_dir.name,
        "collection_stage": args.stage,
        "scenario_id": args.scenario,
        "method": args.method,
        "run_id": args.run_id,
        "mission_phase": "MANUAL",
        "current_pose": pose,
        "pose": pose,
        "target_waypoint": target_waypoint,
        "relative_target": rel,
        "rgb_path": result.get("rgb_path", ""),
        "depth_npy_path": result.get("depth_npy_path", ""),
        "depth_cm_path": result.get("depth_cm_path", ""),
        "pointcloud_path": result.get("point_cloud_world_standard_m_npy_path", ""),
        "point_cloud_world_standard_m_npy_path": result.get("point_cloud_world_standard_m_npy_path", ""),
        "capture_dir": result.get("capture_dir", ""),
        "pose_json_path": result.get("pose_json_path", ""),
        "action_json_path": result.get("action_json_path", ""),
        "depth_summary": result.get("depth_summary", {}),
        "risk_state": args.risk,
        "expert_action": str(executed_action.get("action_name", "hold")),
        "expert_action_payload": executed_action,
        "nominal_action": executed_action,
        "agent_action": executed_action,
        "executed_action": executed_action,
        "last_action": executed_action,
        "obstacle_geometry_label": args.geometry_label,
        "operator_note": args.note,
        "collision_state": False,
        "collision_source": "auto_collect",
        "avoidance_failed": False,
        "movement_mode": result.get("movement_mode", ""),
        "movement_enabled": bool(result.get("movement_enabled", False)),
        "point_count": int(result.get("point_count", 0) or 0),
        "coordinate_frame": result.get("coordinate_frame", ""),
        "coordinate_units": result.get("coordinate_units", ""),
    }
    summary = summarize_geometry_v0(event, base_dir=Path(str(result.get("capture_dir", ".") or ".")).parent)
    event["pointcloud_summary"] = summary
    scores = score_candidate_actions(summary, rel, executed_action)
    event["candidate_action_scores"] = scores
    action, score = best_candidate_action(scores)
    event["v0_selected_action"] = {"action": action, "score": score}
    event["selected_action_reason"] = selected_action_reason(summary, scores)
    event["shield_state"] = "NOT_APPLIED_AUTO_COLLECT"
    event["oscillation_risk"] = "LOW"
    return event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect obstacle avoidance RGB/depth/pointcloud data.")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--launch-sleep", type=int, default=5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--env-platform", default="win", choices=["auto", "win", "mac", "linux"])
    parser.add_argument("--data-root", default="obstacle_avoidance_data")
    parser.add_argument("--stage", default="rule_v0")
    parser.add_argument("--scenario", default="S0")
    parser.add_argument("--method", default="geometry_rule_v0")
    parser.add_argument("--run-id", default="auto")
    parser.add_argument("--risk", default="SAFE")
    parser.add_argument("--geometry-label", default="unknown")
    parser.add_argument("--note", default="auto collection")
    parser.add_argument("--action-cycle", default="hold,yaw_right,yaw_right,yaw_right,yaw_right,yaw_right,yaw_right,yaw_right,yaw_right,yaw_right,yaw_right,yaw_right")
    parser.add_argument("--time-dilation", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage = sanitize(args.stage, "rule_v0")
    scenario = sanitize(args.scenario, "S0")
    method = sanitize(args.method, "geometry_rule_v0")
    run_id = sanitize(args.run_id, "auto")
    session_dir = Path(args.data_root).resolve() / "sessions" / f"{timestamp}_{stage}_{scenario}_{method}_{run_id}"
    session_dir.mkdir(parents=True, exist_ok=False)
    events_path = session_dir / "avoidance_events.jsonl"
    actions = [item.strip().lower() for item in args.action_cycle.split(",") if item.strip()]
    if not actions:
        actions = ["hold"]

    session_args = flight.default_session_args(
        env_platform=args.env_platform,
        output_dir=str(session_dir / "run"),
        launch_sleep=args.launch_sleep,
        width=args.width,
        height=args.height,
        time_dilation=args.time_dilation,
        step_delay=0.05,
        save_every=0,
        lidar_capture_processing="full",
        movement_mode="physics",
        force_kill_unreal_on_stop=True,
        log_level=args.log_level,
    )
    session = flight.DroneFlightSession(session_args)
    events: List[Dict[str, Any]] = []
    write_json(
        session_dir / "collection_config.json",
        {
            "session_dir": str(session_dir),
            "args": vars(args),
            "started_at": datetime.now().isoformat(timespec="milliseconds"),
        },
    )
    try:
        start_result = session.start()
        write_json(session_dir / "start_result.json", start_result if isinstance(start_result, dict) else {"result": str(start_result)})
        session.set_movement_mode("physics")
        session.set_movement_enabled(True)
        target_waypoint: Dict[str, Any] = {}
        last_action = ACTION_PAYLOADS["hold"]
        for frame_id in range(1, max(1, args.frames) + 1):
            action_name = actions[(frame_id - 1) % len(actions)]
            payload = dict(ACTION_PAYLOADS.get(action_name, ACTION_PAYLOADS["hold"]))
            if frame_id > 1 and any(abs(float(payload.get(key, 0.0) or 0.0)) > 1e-9 for key in ("forward_cm", "right_cm", "up_cm", "yaw_delta_deg")):
                session.move_relative(payload)
                last_action = payload
            state = session.get_state()
            pose = state.get("pose", {}) if isinstance(state, dict) and isinstance(state.get("pose"), dict) else {}
            if not target_waypoint and pose:
                yaw = pose_yaw(pose)
                target_waypoint = {
                    "x": float(pose.get("x", 0.0) or 0.0) + 600.0 * math.cos(math.radians(yaw)),
                    "y": float(pose.get("y", 0.0) or 0.0) + 600.0 * math.sin(math.radians(yaw)),
                    "z": float(pose.get("z", 0.0) or 0.0),
                    "source": "auto_collect_forward_probe",
                }
            action_detail = {
                "source": "obstacle_avoidance_auto_collect",
                "collection_stage": args.stage,
                "scenario_id": args.scenario,
                "method": args.method,
                "run_id": args.run_id,
                "mission_phase": "MANUAL",
                "risk_state": args.risk,
                "expert_action": str(last_action.get("action_name", "hold")),
                "expert_action_payload": last_action,
                "target_waypoint": target_waypoint,
            }
            result = session.capture_lidar_stream_frame(session_dir, frame_id, action_detail=action_detail)
            event = build_event(result, session_dir=session_dir, frame_id=frame_id, args=args, executed_action=last_action, target_waypoint=target_waypoint)
            append_jsonl(events_path, event)
            events.append(event)
            summary = {
                "session_id": session_dir.name,
                "session_dir": str(session_dir),
                "frame_count": frame_id,
                "last_event": event,
                "updated_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            write_json(session_dir / "avoidance_session_summary.json", summary)
            print(
                f"[{frame_id}/{args.frames}] geometry={event['pointcloud_summary'].get('obstacle_geometry')} "
                f"front={event['pointcloud_summary'].get('front_min_depth_cm')} "
                f"best={event['v0_selected_action']['action']} -> {result.get('capture_dir')}",
                flush=True,
            )
            if frame_id < args.frames:
                time.sleep(max(0.0, args.interval_s))
        quality = qa_report(session_dir, events)
        write_json(
            session_dir / "avoidance_session_summary.json",
            {
                "session_id": session_dir.name,
                "session_dir": str(session_dir),
                "collection_stage": args.stage,
                "scenario_id": args.scenario,
                "method": args.method,
                "run_id": args.run_id,
                "frame_count": len(events),
                "valid_frame_count": quality.get("valid_frame_count", 0),
                "quality_report_path": str(session_dir / "collection_quality_report.json"),
                "finished_at": datetime.now().isoformat(timespec="milliseconds"),
            },
        )
        print(f"COLLECTION_DONE {session_dir}", flush=True)
    finally:
        session.close(force_kill_unreal=True)


if __name__ == "__main__":
    main()
