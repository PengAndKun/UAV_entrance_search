from __future__ import annotations

from .common import *
from obstacle_avoidance.collect_route_episodes import action_payload, distance_3d_cm


class Route5CaptureGuardMixin:
    def route5_validation_captured_scan_count(self, validation: Dict[str, Any], *, scan_capture_count: int = 0) -> int:
        validation = validation if isinstance(validation, dict) else {}
        captured = int(scan_capture_count or 0)
        coverage = validation.get("coverage_report", {}) if isinstance(validation.get("coverage_report"), dict) else {}
        for key in ("captured_scan_count", "capture_count", "captured_count"):
            value = coverage.get(key, validation.get(key))
            try:
                captured = max(captured, int(float(value)))
            except Exception:
                pass
        for check in validation.get("checks", []) if isinstance(validation.get("checks"), list) else []:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name", "") or "")
            detail = check.get("detail")
            if name == "lidar_rows_exist" and isinstance(detail, str):
                match = re.search(r"(\d+)", detail)
                if match:
                    captured = max(captured, int(match.group(1)))
        return int(captured)

    def route5_rgb_capture_ok(self, rgb_result: Dict[str, Any]) -> bool:
        result = rgb_result if isinstance(rgb_result, dict) else {}
        status = str(result.get("status", result.get("capture_status", "")) or "").strip().lower()
        if status == "ok":
            return True
        for key in ("rgb_path", "image_path", "copied_rgb_path", "capture_dir"):
            if str(result.get(key, "") or "").strip():
                return True
        return False

    def route5_pose_facade_corridor_check(
        self,
        target_house_id: str,
        facade: str,
        pose: Dict[str, Any],
        *,
        axis_margin_cm: float = 250.0,
        side_margin_cm: float = 150.0,
    ) -> Dict[str, Any]:
        facade_name = str(facade or "").strip().lower()
        pose = pose if isinstance(pose, dict) else {}
        try:
            bbox = {}
            route7_bbox_fn = getattr(self, "route7_house_bbox_for_id", None)
            if callable(route7_bbox_fn):
                bbox = route7_bbox_fn(str(target_house_id or ""))
            if not bbox:
                bbox = self.house_world_bbox_for_id(str(target_house_id or ""))
        except Exception:
            bbox = {}
        if facade_name not in {"east", "west", "north", "south"} or not isinstance(bbox, dict) or not bbox:
            return {
                "same_facade_corridor": True,
                "enforced": False,
                "reason": "facade_bbox_unavailable",
                "facade": facade_name,
                "pose": self.route5_json_safe(pose),
            }
        try:
            x = float(pose.get("x", 0.0) or 0.0)
            y = float(pose.get("y", 0.0) or 0.0)
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            return {
                "same_facade_corridor": True,
                "enforced": False,
                "reason": "pose_or_bbox_invalid",
                "facade": facade_name,
                "pose": self.route5_json_safe(pose),
                "bbox": self.route5_json_safe(bbox),
            }
        axis_margin = max(0.0, float(axis_margin_cm))
        side_margin = max(0.0, float(side_margin_cm))
        if facade_name == "east":
            side_ok = x >= max_x - side_margin
            axis_ok = min_y - axis_margin <= y <= max_y + axis_margin
            side_distance_cm = x - max_x
            axis_value = y
            axis_min, axis_max = min_y, max_y
        elif facade_name == "west":
            side_ok = x <= min_x + side_margin
            axis_ok = min_y - axis_margin <= y <= max_y + axis_margin
            side_distance_cm = min_x - x
            axis_value = y
            axis_min, axis_max = min_y, max_y
        elif facade_name == "north":
            side_ok = y >= max_y - side_margin
            axis_ok = min_x - axis_margin <= x <= max_x + axis_margin
            side_distance_cm = y - max_y
            axis_value = x
            axis_min, axis_max = min_x, max_x
        else:
            side_ok = y <= min_y + side_margin
            axis_ok = min_x - axis_margin <= x <= max_x + axis_margin
            side_distance_cm = min_y - y
            axis_value = x
            axis_min, axis_max = min_x, max_x
        return {
            "same_facade_corridor": bool(side_ok and axis_ok),
            "enforced": True,
            "reason": "ok" if bool(side_ok and axis_ok) else ("wrong_facade_side" if not side_ok else "axis_outside_facade_bounds"),
            "facade": facade_name,
            "pose": self.route5_json_safe(pose),
            "bbox": self.route5_json_safe(bbox),
            "side_ok": bool(side_ok),
            "axis_ok": bool(axis_ok),
            "side_distance_cm": round(float(side_distance_cm), 3),
            "axis_value_cm": round(float(axis_value), 3),
            "axis_min_cm": round(float(axis_min), 3),
            "axis_max_cm": round(float(axis_max), 3),
            "axis_margin_cm": float(axis_margin),
            "side_margin_cm": float(side_margin),
        }

    def route5_facade_corridor_check(
        self,
        target_house_id: str,
        facade: str,
        pose: Dict[str, Any],
        *,
        original_target_pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        original = original_target_pose if isinstance(original_target_pose, dict) and original_target_pose else {}
        original_check = self.route5_pose_facade_corridor_check(target_house_id, facade, original) if original else {"same_facade_corridor": False, "enforced": False, "reason": "no_original_pose"}
        pose_check = self.route5_pose_facade_corridor_check(target_house_id, facade, pose)
        enforce = bool(original_check.get("enforced", False)) and bool(original_check.get("same_facade_corridor", False))
        same = bool(pose_check.get("same_facade_corridor", True)) if enforce else True
        return {
            "same_facade_corridor": bool(same),
            "enforced": bool(enforce),
            "reason": str(pose_check.get("reason", "ok") if enforce else "original_reference_not_corridor_enforced"),
            "facade": str(facade or "").strip().lower(),
            "original_reference_valid": bool(original_check.get("same_facade_corridor", False)),
            "original_check": self.route5_json_safe(original_check),
            "pose_check": self.route5_json_safe(pose_check),
        }

    def route5_capture_guard_state(
        self,
        *,
        target_house_id: str,
        stage: str,
        facade: str,
        target_id: str,
        original_target_pose: Dict[str, Any],
        runtime_target_pose: Dict[str, Any],
        capture_pose: Dict[str, Any],
        pose_error: Optional[Dict[str, Any]] = None,
        arrival_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        capture_kind: str = "scan",
    ) -> Dict[str, Any]:
        original = dict(original_target_pose if isinstance(original_target_pose, dict) else {})
        runtime = dict(runtime_target_pose if isinstance(runtime_target_pose, dict) else {})
        capture = dict(capture_pose if isinstance(capture_pose, dict) else {})
        pose_error = pose_error if isinstance(pose_error, dict) else {}
        arrival_state = arrival_state if isinstance(arrival_state, dict) else {}
        config = config if isinstance(config, dict) else {}
        reach_tol = self.route5_event_float(config.get("reach_tol_cm", 60.0), default=60.0)
        near_tol = 150.0
        if not original:
            original = dict(runtime)
        if not capture:
            capture = dict(runtime)
        distance_original = distance_3d_cm(capture, original) if original and capture else float("inf")
        distance_runtime = distance_3d_cm(capture, runtime) if runtime and capture else float("inf")
        corridor_check = self.route5_facade_corridor_check(target_house_id, facade, capture, original_target_pose=original)
        if not bool(corridor_check.get("same_facade_corridor", True)):
            passed = False
            reason = "facade_corridor_mismatch"
            policy = "blocked_wrong_facade_corridor"
        else:
            standard_original_reached = bool(distance_original <= max(5.0, reach_tol))
            near_obstacle_confirmed = bool(arrival_state.get("near_obstacle_confirmed", False) or arrival_state.get("near_obstacle_reached", False))
            near_obstacle_reached_original = bool(near_obstacle_confirmed and distance_original <= near_tol)
            if standard_original_reached:
                passed = True
                reason = "standard_reach_original_target"
                policy = "standard_reach_tolerance_original_target"
            elif near_obstacle_reached_original:
                passed = True
                reason = "near_obstacle_reached_original_target"
                policy = "near_obstacle_original_target_150cm"
            else:
                passed = False
                policy = "blocked_not_near_original_target"
                reason = "original_target_distance_exceeded" if distance_original > near_tol else "original_target_not_reached"
        return {
            "schema": "route5_capture_guard_v1",
            "capture_guard_passed": bool(passed),
            "status": "ok" if passed else "rejected",
            "reason": reason,
            "capture_policy": policy,
            "capture_kind": str(capture_kind or ""),
            "stage": str(stage or ""),
            "facade": str(facade or "").strip().lower(),
            "target_id": str(target_id or ""),
            "target_house_id": str(target_house_id or ""),
            "original_target_pose": self.route5_json_safe(original),
            "runtime_target_pose": self.route5_json_safe(runtime),
            "capture_pose": self.route5_json_safe(capture),
            "pose_error": self.route5_json_safe(pose_error),
            "arrival_state": self.route5_json_safe(arrival_state),
            "facade_corridor_check": self.route5_json_safe(corridor_check),
            "distance_to_original_target_cm": round(float(distance_original), 3) if math.isfinite(distance_original) else 0.0,
            "distance_to_runtime_target_cm": round(float(distance_runtime), 3) if math.isfinite(distance_runtime) else 0.0,
            "reach_tol_cm": round(float(reach_tol), 3),
            "near_obstacle_capture_tol_cm": near_tol,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }

    def route5_capture_guard_event_matches(
        self,
        event: Dict[str, Any],
        *,
        stage: str,
        facade: str,
        target_id: str,
    ) -> bool:
        event = event if isinstance(event, dict) else {}
        expected_stage = str(stage or "").strip().lower()
        expected_facade = str(facade or "").strip().lower()
        expected_target = str(target_id or "").strip()
        event_stage = str(event.get("route5_stage", event.get("stage", "")) or "").strip().lower()
        event_facade = str(event.get("facade", "") or "").strip().lower()
        event_target = str(event.get("target_id", "") or "").strip()
        if event_stage and expected_stage and event_stage != expected_stage:
            return False
        if event_facade and expected_facade and event_facade != expected_facade:
            return False
        if event_target and expected_target and event_target != expected_target:
            return False
        return bool(event)

    def route5_capture_guard_arrival_state(
        self,
        *,
        nav_result: Dict[str, Any],
        stage: str,
        facade: str,
        target_id: str,
        original_target_pose: Dict[str, Any],
        runtime_target_pose: Dict[str, Any],
        capture_pose: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nav_result = nav_result if isinstance(nav_result, dict) else {}
        existing = dict(nav_result.get("arrival_state", {}) if isinstance(nav_result.get("arrival_state"), dict) else {})
        has_navigation_arrival = bool(
            existing.get("arrival_policy")
            or existing.get("near_obstacle_reached", False)
            or existing.get("near_obstacle_confirmed", False)
        )
        if has_navigation_arrival:
            existing.setdefault("capture_guard_arrival_source", "navigation_result")
            existing.setdefault("capture_guard_recomputed_arrival", False)
            return existing

        original = dict(original_target_pose if isinstance(original_target_pose, dict) else {})
        runtime = dict(runtime_target_pose if isinstance(runtime_target_pose, dict) else {})
        capture = dict(capture_pose if isinstance(capture_pose, dict) else {})
        if not original:
            original = dict(runtime)
        if not capture:
            capture = dict(runtime)
        config = config if isinstance(config, dict) else {}
        reach_tol = self.route5_event_float(config.get("reach_tol_cm", 60.0), default=60.0)
        if original and capture:
            distance_original = distance_3d_cm(capture, original)
            dist_xy = math.hypot(
                float(capture.get("x", 0.0) or 0.0) - float(original.get("x", 0.0) or 0.0),
                float(capture.get("y", 0.0) or 0.0) - float(original.get("y", 0.0) or 0.0),
            )
        else:
            distance_original = self.route5_event_float((nav_result.get("pose_error", {}) if isinstance(nav_result.get("pose_error"), dict) else {}).get("dist_3d_cm", 0.0), default=0.0)
            dist_xy = self.route5_event_float((nav_result.get("pose_error", {}) if isinstance(nav_result.get("pose_error"), dict) else {}).get("dist_xy_cm", distance_original), default=distance_original)
        error = dict(nav_result.get("pose_error", {}) if isinstance(nav_result.get("pose_error"), dict) else {})
        error.update(
            {
                "reached": bool(distance_original <= max(5.0, reach_tol)),
                "dist_xy_cm": round(float(dist_xy), 3),
                "dist_3d_cm": round(float(distance_original), 3),
            }
        )

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for name in ("final_obstacle_event", "obstacle_event", "event"):
            if isinstance(nav_result.get(name), dict):
                candidates.append((f"navigation_{name}", nav_result.get(name, {})))
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        for name in ("last_obstacle_event", "last_or2_event"):
            if isinstance(state.get(name), dict):
                candidates.append((name, state.get(name, {})))

        for source, event in candidates:
            if not self.route5_capture_guard_event_matches(event, stage=stage, facade=facade, target_id=target_id):
                continue
            summary = event.get("pointcloud_summary", event.get("depth_obstacle_summary", {}))
            summary = summary if isinstance(summary, dict) else {}
            prediction = event.get("or2_prediction", {}) if isinstance(event.get("or2_prediction"), dict) else {}
            gate = dict(event.get("avoidance_gate", {}) if isinstance(event.get("avoidance_gate"), dict) else {})
            gate.setdefault("front_risk_state", prediction.get("front_risk_state", event.get("or2_front_risk_state", "")))
            gate.setdefault("front_min_depth_cm", summary.get("front_min_depth_cm", event.get("front_depth_cm", 0.0)))
            gate.setdefault("must_stop", bool(prediction.get("must_stop", event.get("or2_must_stop", False))))
            risk = str(gate.get("front_risk_state", "") or "").strip().lower()
            gate.setdefault("avoidance_active", bool(risk in {"obstacle_warning", "must_stop"} or gate.get("must_stop", False)))
            arrival_event = dict(event)
            arrival_event["distance_to_goal_cm"] = round(float(distance_original), 3)
            arrival_event["pointcloud_summary"] = summary
            recomputed = self.route5_near_obstacle_arrival_state(error, gate, arrival_event)
            recomputed.update(
                {
                    "capture_guard_arrival_source": source,
                    "capture_guard_recomputed_arrival": True,
                    "capture_guard_recomputed_reason": "navigation_arrival_state_missing",
                }
            )
            return recomputed

        existing.setdefault("capture_guard_arrival_source", "missing")
        existing.setdefault("capture_guard_recomputed_arrival", False)
        return existing

    def route5_update_capture_guard_repeat_state(self, guard: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(guard if isinstance(guard, dict) else {})
        if bool(record.get("capture_guard_passed", False)):
            record["capture_guard_repeat_count"] = 0
            record["next_retry_action"] = "capture_allowed"
            return record
        key = "|".join(
            [
                str(record.get("capture_kind", "") or ""),
                str(record.get("stage", "") or ""),
                str(record.get("facade", "") or ""),
                str(record.get("target_id", "") or ""),
                str(record.get("reason", "") or ""),
            ]
        )
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        counts = dict(state.get("capture_guard_repeat_counts", {}) if isinstance(state.get("capture_guard_repeat_counts"), dict) else {})
        count = int(counts.get(key, 0) or 0) + 1
        counts[key] = count
        distance_original = self.route5_event_float(record.get("distance_to_original_target_cm", 0.0), default=0.0)
        reason = str(record.get("reason", "") or "")
        capture_kind = str(record.get("capture_kind", "") or "").strip().lower()
        if capture_kind == "observation" and reason == "original_target_not_reached" and distance_original <= 150.0 and count >= 2:
            next_action = "try_next_observation_or_rescue"
            skip_targets = dict(state.get("capture_guard_skip_targets", {}) if isinstance(state.get("capture_guard_skip_targets"), dict) else {})
            target_key = str(record.get("target_id", "") or "")
            if target_key:
                skip_targets[target_key] = {
                    "reason": reason,
                    "facade": str(record.get("facade", "") or ""),
                    "repeat_count": count,
                    "next_retry_action": next_action,
                    "updated_at": datetime.now().isoformat(timespec="milliseconds"),
                }
                self.route5_update_state(capture_guard_repeat_counts=counts, capture_guard_skip_targets=skip_targets)
            else:
                self.route5_update_state(capture_guard_repeat_counts=counts)
        else:
            next_action = "retry_same_target"
            self.route5_update_state(capture_guard_repeat_counts=counts)
        record["capture_guard_repeat_count"] = int(count)
        record["next_retry_action"] = next_action
        return record

    def route5_record_capture_guard(self, output_dir: Optional[Path], guard: Dict[str, Any]) -> Dict[str, Any]:
        record = self.route5_update_capture_guard_repeat_state(guard if isinstance(guard, dict) else {})
        record = self.route5_json_safe(record)
        if output_dir is not None:
            self.append_jsonl(output_dir / "route5_capture_guard_events.jsonl", record)
            self.route5_log_event(output_dir, "capture_guard", record)
        return record

    def route5_write_capture_guard_blocked_decision(
        self,
        output_dir: Path,
        guard: Dict[str, Any],
        *,
        current_pose: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            frame_id = int(self.route2_next_frame_index(output_dir))
        except Exception:
            frame_id = int(time.time() * 1000) % 1000000
        event = {
            "frame_id": frame_id,
            "route5_stage": str(guard.get("stage", "") or ""),
            "facade": str(guard.get("facade", "") or ""),
            "target_id": str(guard.get("target_id", "") or ""),
            "capture_dir": str(output_dir / "frames" / f"frame_{frame_id:06d}"),
            "current_pose": self.route5_json_safe(current_pose),
            "target_waypoint": self.route5_json_safe(guard.get("runtime_target_pose", {})),
            "original_target_pose": self.route5_json_safe(guard.get("original_target_pose", {})),
            "runtime_target_pose": self.route5_json_safe(guard.get("runtime_target_pose", {})),
            "capture_pose": self.route5_json_safe(guard.get("capture_pose", current_pose)),
            "capture_guard": self.route5_json_safe(guard),
            "capture_guard_blocked": True,
            "capture_guard_passed": False,
            "selected_action": "route5_capture_guard_hold",
            "selected_action_reason": str(guard.get("reason", "capture_guard_blocked") or "capture_guard_blocked"),
            "selected_action_payload": action_payload("hold"),
            "collision_state": False,
            "avoidance_failed": False,
        }
        return self.route5_write_frame_decision(output_dir, self.route5_normalize_avoidance_event(event), final_payload=action_payload("hold"))

    def route5_facade_completion_gate(
        self,
        validation: Dict[str, Any],
        *,
        observation: Dict[str, Any],
        rgb_result: Dict[str, Any],
        scan_capture_count: int = 0,
        valid_scan_capture_count: Optional[int] = None,
        scan_loop_stopped: bool = False,
        postprocess_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        validation = validation if isinstance(validation, dict) else {}
        observation = observation if isinstance(observation, dict) else {}
        postprocess_result = postprocess_result if isinstance(postprocess_result, dict) else {}
        raw_captured_scan_count = self.route5_validation_captured_scan_count(validation, scan_capture_count=scan_capture_count)
        if valid_scan_capture_count is None:
            captured_scan_count = raw_captured_scan_count
        else:
            captured_scan_count = max(0, int(valid_scan_capture_count or 0))
        validation_passed = bool(validation.get("overall_passed", False))
        degraded_observation = bool(observation.get("route5_degraded_observation", False))
        rgb_ok = self.route5_rgb_capture_ok(rgb_result)
        if scan_loop_stopped:
            status = "scan_incomplete"
            complete = False
            reason = "scan_loop_stopped_before_completion"
        elif valid_scan_capture_count is not None and raw_captured_scan_count > 0 and captured_scan_count <= 0:
            status = "invalid_capture_pose_retryable"
            complete = False
            reason = "no_capture_guard_passed_scan_captures"
        elif (
            captured_scan_count > 0
            and not validation_passed
            and int(postprocess_result.get("processed_frame_count", 0) or 0) > 0
            and int(postprocess_result.get("source_point_count", 0) or 0) <= 0
        ):
            status = "postprocess_failed"
            complete = False
            reason = "postprocess_point_count_zero"
        elif validation_passed and captured_scan_count > 0:
            status = "degraded_completed" if degraded_observation else "full_completed"
            complete = True
            reason = status
        elif degraded_observation and rgb_ok and captured_scan_count <= 0:
            status = "scan_incomplete"
            complete = False
            reason = "degraded_observation_without_scan_capture"
        else:
            status = "scan_incomplete"
            complete = False
            if captured_scan_count <= 0:
                reason = "no_scan_captures"
            elif not validation_passed:
                reason = "validation_failed"
            else:
                reason = "scan_completion_gate_failed"
        return {
            "complete": bool(complete),
            "completion_status": status,
            "reason": reason,
            "validation_passed": validation_passed,
            "captured_scan_count": int(captured_scan_count),
            "raw_captured_scan_count": int(raw_captured_scan_count),
            "valid_scan_capture_count": int(captured_scan_count),
            "degraded_observation": degraded_observation,
            "rgb_capture_ok": rgb_ok,
            "scan_loop_stopped": bool(scan_loop_stopped),
            "postprocess_result": self.route5_json_safe(postprocess_result),
            "terminal": self.route5_facade_status_is_terminal(status),
        }

    def route5_facade_status_is_terminal(self, status: str) -> bool:
        return str(status or "").strip().lower() in {
            "full_completed",
            "degraded_completed",
            "terminal_blocked",
            "terminal_failed",
            "manual_stopped",
            "collision",
        }

    def route5_max_facade_retry_count(self) -> int:
        return 3

    def route5_final_status_for_task_lock(self, status: str) -> str:
        final_status = str(status or "").strip() or "done"
        if final_status != "done":
            return final_status
        terminal_failed = set(self.llm_route5_blocked_facades)
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        terminal_failed.update(
            str(item).strip().lower()
            for item in (state.get("terminal_failed_facades", []) if isinstance(state.get("terminal_failed_facades", []), list) else [])
            if str(item).strip().lower()
        )
        if not terminal_failed:
            return final_status
        route_label = str(state.get("route_window_label", "") or "").strip().upper()
        if route_label == "V7":
            return "blocked_selected_house_incomplete"
        return "done_with_blocked"

