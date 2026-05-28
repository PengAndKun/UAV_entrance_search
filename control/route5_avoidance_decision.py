from __future__ import annotations

from .common import *
from .local_obstacle_map import LocalObstacleMap, LocalObstacleMapConfig

from obstacle_avoidance.collect_route_episodes import (
    DEFAULT_ROUTE_SIDE_CORRECTION_CM,
    DEFAULT_ROUTE_VERTICAL_STEP_CM,
    action_payload,
    build_route_event,
    distance_3d_cm,
    pose_from_state,
    risk_state_from_summary,
    select_route_action,
)
from obstacle_avoidance_llm.policy import (
    LLM_STRATEGY_METHOD_ID,
    refine_strategy_with_pointcloud_context,
    strategy_from_episode_metadata,
)
from obstacle_avoidance_3.control_utils import serializable_or2_prediction
from obstacle_avoidance_3.or2_direction_rule import select_or2_direction
from obstacle_representation_2.demo import ObstacleRepresentation2Predictor, render_affordance_overlay

try:
    from obstacle_representation_3.demo import (
        ObstacleRepresentation3Predictor,
        render_affordance_overlay as render_or3_affordance_overlay,
    )
except Exception:
    ObstacleRepresentation3Predictor = None
    render_or3_affordance_overlay = None


class Route5AvoidanceDecisionMixin:
    def route5_local_obstacle_config(self) -> LocalObstacleMapConfig:
        return LocalObstacleMapConfig(voxel_cm=25.0, radius_cm=1200.0, ttl_frames=150, ttl_seconds=60.0)

    def route5_ensure_local_obstacle_map(self, output_dir: Optional[Path] = None) -> LocalObstacleMap:
        self.ensure_route5_state()
        target_dir = Path(output_dir) if output_dir is not None else self.route5_state_output_dir()
        current_dir = getattr(self, "route5_local_obstacle_output_dir", None)
        if (
            getattr(self, "route5_local_obstacle_map", None) is None
            or (target_dir is not None and current_dir is not None and Path(current_dir) != target_dir)
        ):
            self.route5_local_obstacle_map = LocalObstacleMap(self.route5_local_obstacle_config())
        if target_dir is not None:
            self.route5_local_obstacle_output_dir = target_dir
        return self.route5_local_obstacle_map

    def write_local_obstacle_artifacts(self, output_dir: Path) -> Dict[str, Any]:
        obstacle_map = self.route5_ensure_local_obstacle_map(output_dir)
        return obstacle_map.write_artifacts(Path(output_dir))

    def update_local_obstacle_map_from_event(
        self,
        event: Dict[str, Any],
        current_pose: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        obstacle_map = self.route5_ensure_local_obstacle_map(output_dir)
        update = obstacle_map.update_from_event(event if isinstance(event, dict) else {}, current_pose if isinstance(current_pose, dict) else {})
        record = {
            "event_type": "local_obstacle_map_update",
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            **self.route5_json_safe(update),
        }
        self.append_jsonl(Path(output_dir) / "local_obstacle_map.jsonl", record)
        self.write_local_obstacle_artifacts(Path(output_dir))
        return record

    def query_local_3d_safety(
        self,
        current_pose: Dict[str, Any],
        payload: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        *,
        output_dir: Optional[Path] = None,
        stage: str = "",
        target_id: str = "",
        candidate_direction: str = "",
    ) -> Dict[str, Any]:
        obstacle_map = self.route5_ensure_local_obstacle_map(output_dir)
        safety_radius = None
        if isinstance(config, dict) and config.get("local_3d_safety_radius_cm") is not None:
            try:
                safety_radius = float(config.get("local_3d_safety_radius_cm"))
            except Exception:
                safety_radius = None
        safety = obstacle_map.query_safety(current_pose if isinstance(current_pose, dict) else {}, payload if isinstance(payload, dict) else {}, safety_radius_cm=safety_radius)
        safety.update(
            {
                "stage": str(stage or ""),
                "target_id": str(target_id or ""),
                "candidate_direction": str(candidate_direction or ""),
            }
        )
        target_dir = Path(output_dir) if output_dir is not None else getattr(self, "route5_local_obstacle_output_dir", None)
        if target_dir is not None:
            self.append_jsonl(
                Path(target_dir) / "local_3d_safety_events.jsonl",
                {
                    "event_type": "local_3d_safety_check",
                    "created_at": datetime.now().isoformat(timespec="milliseconds"),
                    "stage": str(stage or ""),
                    "target_id": str(target_id or ""),
                    "candidate_direction": str(candidate_direction or ""),
                    "local_3d_safety": self.route5_json_safe(safety),
                },
            )
        return self.route5_json_safe(safety)

    def local_3d_blocked_directions(
        self,
        current_pose: Dict[str, Any],
        candidate_payloads: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        obstacle_map = self.route5_ensure_local_obstacle_map()
        return self.route5_json_safe(obstacle_map.blocked_directions(current_pose, candidate_payloads))

    def route5_set_control_lock(self, locked: bool) -> None:
        self.ensure_route5_state()
        self.llm_route5_control_locked = bool(locked)
        try:
            if locked:
                self.root.after(0, lambda: self.stop_keyboard_control(send_hold=False))
                self.root.after(0, lambda: self.update_keyboard_status("locked by LLM Route V5"))
            else:
                self.root.after(0, self.update_keyboard_status)
        except Exception:
            pass

    def route5_enable_physics_movement(self, session: flight.DroneFlightSession) -> None:
        self.route5_set_control_lock(True)
        result = self.safe("Route V5 movement mode physics", lambda: session.set_movement_mode("physics"))
        self.safe("Route V5 enable movement", lambda: session.set_movement_enabled(True))
        self.movement_mode_state = "physics"
        try:
            self.root.after(0, lambda: self.movement_mode_var.set("physics"))
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_state(r))
        except Exception:
            pass

    def route5_hold(self, session: flight.DroneFlightSession, *, output_dir: Optional[Path] = None, reason: str = "hold") -> Dict[str, Any]:
        payload = action_payload("hold")
        result = self.safe("Route V5 hold", lambda: session.move_relative(payload))
        self.route5_log_event(output_dir, "hold", {"reason": reason, "payload": payload})
        return result if isinstance(result, dict) else {}

    def route5_route6_full_stop_requested(self) -> bool:
        event = getattr(self, "route6_full_stop_event", None)
        return bool(hasattr(event, "is_set") and event.is_set())

    def route5_wait_if_paused(self, session: flight.DroneFlightSession, output_dir: Optional[Path]) -> bool:
        self.ensure_route5_state()
        held = False
        while self.llm_route5_pause_event.is_set() and not self.llm_route5_stop_event.is_set():
            if not held:
                self.route5_hold(session, output_dir=output_dir, reason="paused")
                held = True
            self.root.after(0, lambda: self.llm_route5_status_var.set("LLM Route V5: paused."))
            time.sleep(0.2)
        return bool(self.llm_route5_stop_event.is_set())

    def route5_semantic_hint_from_prediction(self, prediction: Dict[str, Any]) -> str:
        label = str((prediction if isinstance(prediction, dict) else {}).get("predicted_label", "") or "").strip().lower()
        mapping = {
            "open_path": "unknown",
            "none": "unknown",
            "unknown": "unknown",
            "tree_trunk_or_pole": "tree_trunk_or_pole",
            "tree_canopy_or_cluster": "tree_canopy_or_cluster",
            "fence_or_rail": "fence_or_rail",
            "building": "building",
            "mixed": "mixed",
        }
        return mapping.get(label, "unknown")

    def route5_event_float(self, value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if math.isfinite(number) else float(default)
        except Exception:
            return float(default)

    def route5_avoidance_trigger_distance_cm(self, semantic_hint: str) -> float:
        hint = str(semantic_hint or "unknown").strip().lower()
        if hint == "building":
            return 500.0
        if hint in {"tree_trunk_or_pole", "tree_canopy_or_cluster", "fence_or_rail", "mixed", "unknown"}:
            return 120.0
        return 120.0

    def route5_avoidance_gate_semantic_hint(self, event: Dict[str, Any], strategy: Dict[str, Any]) -> Tuple[str, str]:
        prediction = event.get("representation_prediction", {}) if isinstance(event.get("representation_prediction"), dict) else {}
        representation_hint = self.route5_semantic_hint_from_prediction(prediction)
        if representation_hint != "unknown":
            return representation_hint, "representation_prediction"
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        geometry = str(summary.get("obstacle_geometry", "") or "").strip().lower()
        if geometry in {"vertical_wall", "overhang_beam", "building", "wall", "facade"}:
            return "building", "pointcloud_geometry"
        if geometry in {"low_obstacle", "thin_vertical", "fence", "rail"}:
            return "fence_or_rail", "pointcloud_geometry"
        strategy_hint = self.route5_semantic_hint_from_prediction({"predicted_label": str(strategy.get("obstacle_hint", "") or "")})
        if strategy_hint != "unknown":
            return strategy_hint, "llm_strategy"
        return "unknown", "default"

    def route5_should_apply_avoidance(self, event: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        semantic_hint, semantic_source = self.route5_avoidance_gate_semantic_hint(event, strategy if isinstance(strategy, dict) else {})
        trigger_distance_cm = self.route5_avoidance_trigger_distance_cm(semantic_hint)
        front_min_depth_cm = self.route5_event_float(summary.get("front_min_depth_cm"), default=0.0)
        forward_swept_clear = bool(summary.get("forward_swept_clear", True))
        pointcloud_available = bool(summary.get("available", bool(summary)))
        geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown")
        has_front_depth = front_min_depth_cm > 0.0
        distance_triggered = has_front_depth and front_min_depth_cm <= trigger_distance_cm
        blocked_without_depth = (not forward_swept_clear) and not has_front_depth
        avoidance_active = bool(pointcloud_available and (distance_triggered or blocked_without_depth))
        if not pointcloud_available:
            reason = "pointcloud_unavailable"
        elif distance_triggered:
            reason = f"front_depth_within_{trigger_distance_cm:.0f}cm"
        elif blocked_without_depth:
            reason = "forward_swept_blocked_without_depth"
        elif not forward_swept_clear:
            reason = f"forward_swept_blocked_but_front_depth_{front_min_depth_cm:.1f}cm_exceeds_{trigger_distance_cm:.0f}cm"
        elif has_front_depth:
            reason = f"front_clear_depth_{front_min_depth_cm:.1f}cm_exceeds_{trigger_distance_cm:.0f}cm"
        else:
            reason = "no_front_obstacle_depth"
        return {
            "avoidance_active": avoidance_active,
            "semantic_hint": semantic_hint,
            "semantic_source": semantic_source,
            "trigger_distance_cm": float(trigger_distance_cm),
            "front_min_depth_cm": float(front_min_depth_cm),
            "forward_swept_clear": forward_swept_clear,
            "pointcloud_available": pointcloud_available,
            "obstacle_geometry": geometry,
            "reason": reason,
        }

    def route5_depth_lookahead_stages(self) -> set:
        return {"NAV_TO_OBS", "NAV_TO_SCAN_POINT", "ACTIVE_NBV_NAV_TO_SCAN_POINT"}

    def route7_should_translate_while_yawing(self, target: Dict[str, Any], stage: str) -> bool:
        if not self.route5_event_is_route7({}):
            return False
        stage_name = str(stage or "").strip().upper()
        if stage_name not in {"NAV_TO_OBS", "NAV_TO_SCAN_POINT", "ACTIVE_NBV_NAV_TO_SCAN_POINT"}:
            return False
        if not isinstance(target, dict):
            return False
        route_type = str(target.get("route_point_type", "") or "")
        return bool(
            target.get("route7_map_layer_key")
            or target.get("route7_map_cell")
            or target.get("route7_map_route_replan_policy")
            or route_type in {"navigation_waypoint", "route7_layer_transition", "map_layer_edge_capture"}
        )

    def route5_payload_blocked_directions_from_depth(
        self,
        summary: Dict[str, Any],
        nominal_payload: Dict[str, Any],
    ) -> List[str]:
        if not isinstance(summary, dict) or not isinstance(nominal_payload, dict):
            return []
        deadband_cm = 0.5
        blocked: List[str] = []
        forward_cm = self.route5_event_float(nominal_payload.get("forward_cm"), default=0.0)
        right_cm = self.route5_event_float(nominal_payload.get("right_cm"), default=0.0)
        up_cm = self.route5_event_float(nominal_payload.get("up_cm"), default=0.0)
        if forward_cm > deadband_cm and summary.get("forward_swept_clear") is False:
            blocked.append("forward")
        if forward_cm < -deadband_cm and summary.get("backoff_swept_clear") is False:
            blocked.append("backoff")
        if right_cm > deadband_cm and summary.get("right_swept_clear") is False:
            blocked.append("right")
        if right_cm < -deadband_cm and summary.get("left_swept_clear") is False:
            blocked.append("left")
        if up_cm > deadband_cm and summary.get("up_swept_clear") is False:
            blocked.append("up")
        if up_cm < -deadband_cm and summary.get("down_swept_clear") is False:
            blocked.append("down")
        return blocked

    def route5_should_precheck_depth_before_payload(self, stage: str, nominal_payload: Dict[str, Any]) -> bool:
        stage_name = str(stage or "").strip().upper()
        if stage_name not in self.route5_depth_lookahead_stages() or not isinstance(nominal_payload, dict):
            return False
        deadband_cm = 0.5
        return any(
            abs(self.route5_event_float(nominal_payload.get(key), default=0.0)) > deadband_cm
            for key in ("forward_cm", "right_cm", "up_cm")
        )

    def route5_navigation_travel_yaw_deg(self, current: Dict[str, Any], target: Dict[str, Any]) -> float:
        dx = self.route5_event_float((target if isinstance(target, dict) else {}).get("x"), default=0.0) - self.route5_event_float((current if isinstance(current, dict) else {}).get("x"), default=0.0)
        dy = self.route5_event_float((target if isinstance(target, dict) else {}).get("y"), default=0.0) - self.route5_event_float((current if isinstance(current, dict) else {}).get("y"), default=0.0)
        if math.hypot(dx, dy) <= 1e-6:
            return self._normalize_angle_deg(self.route5_event_float((current if isinstance(current, dict) else {}).get("yaw"), default=0.0))
        return self._normalize_angle_deg(math.degrees(math.atan2(dy, dx)))

    def route5_depth_lookahead_plan(
        self,
        current: Dict[str, Any],
        target: Dict[str, Any],
        *,
        stage: str,
        facade: str,
        target_id: str,
    ) -> Dict[str, Any]:
        current_dict = current if isinstance(current, dict) else {}
        target_dict = target if isinstance(target, dict) else {}
        stage_name = str(stage or "").strip().upper()
        current_yaw = self.route5_event_float(current_dict.get("yaw"), default=0.0)
        capture_yaw = self.route5_event_float(target_dict.get("yaw", target_dict.get("yaw_deg")), default=current_yaw)
        dx = self.route5_event_float(target_dict.get("x"), default=0.0) - self.route5_event_float(current_dict.get("x"), default=0.0)
        dy = self.route5_event_float(target_dict.get("y"), default=0.0) - self.route5_event_float(current_dict.get("y"), default=0.0)
        horizontal_distance_cm = float(math.hypot(dx, dy))
        enabled = bool(stage_name in self.route5_depth_lookahead_stages() and horizontal_distance_cm > 1.0)
        look_yaw = self.route5_navigation_travel_yaw_deg(current_dict, target_dict) if enabled else current_yaw
        yaw_error = self._normalize_angle_deg(float(look_yaw) - float(current_yaw))
        status = "ALIGN_LOOK_YAW" if enabled and abs(yaw_error) > 1.0 else ("SENSE_LOOK_YAW" if enabled else "NO_LOOKAHEAD")
        thinking = (
            f"Thinking: {stage_name or 'NAV'} {status} look waypoint {target_id} "
            f"yaw={float(look_yaw):.1f} current={float(current_yaw):.1f} "
            f"capture={float(capture_yaw):.1f} dist={horizontal_distance_cm:.1f}cm"
        )
        return {
            "enabled": enabled,
            "stage": stage_name,
            "facade": facade,
            "target_id": target_id,
            "look_direction": "waypoint_bearing" if enabled else "current_yaw",
            "look_yaw_deg": round(float(look_yaw), 3),
            "current_yaw_deg": round(float(current_yaw), 3),
            "capture_yaw_deg": round(float(capture_yaw), 3),
            "yaw_error_deg": round(float(yaw_error), 3),
            "horizontal_distance_cm": round(horizontal_distance_cm, 3),
            "segment_start": {
                "x": self.route5_event_float(current_dict.get("x"), default=0.0),
                "y": self.route5_event_float(current_dict.get("y"), default=0.0),
                "z": self.route5_event_float(current_dict.get("z"), default=0.0),
            },
            "segment_goal": {
                "x": self.route5_event_float(target_dict.get("x"), default=0.0),
                "y": self.route5_event_float(target_dict.get("y"), default=0.0),
                "z": self.route5_event_float(target_dict.get("z"), default=0.0),
            },
            "decision_state": status,
            "thinking_status": thinking,
            "policy": "face_waypoint_bearing_for_depth_then_capture_house_yaw",
        }

    def route5_set_thinking_status(
        self,
        decision: Any,
        *,
        output_dir: Optional[Path] = None,
        write_artifact: bool = False,
    ) -> None:
        self.ensure_route5_state()
        if isinstance(decision, dict):
            payload = dict(decision)
            text = str(payload.get("thinking_status", "") or payload.get("status", "") or "Thinking: active")
        else:
            payload = {"thinking_status": str(decision)}
            text = str(decision)
        self.route5_update_state(thinking_status=text, last_navigation_decision=payload)
        if write_artifact:
            self.route5_write_state_artifact()
        if output_dir is not None:
            self.route5_log_event(output_dir, "thinking_status", payload)
        try:
            self.root.after(0, lambda t=text: self.llm_route5_thinking_status_var.set(t))
        except Exception:
            try:
                self.llm_route5_thinking_status_var.set(text)
            except Exception:
                pass

    def route5_movement_payload_for_target_with_lookahead(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        config: Dict[str, float],
        *,
        stage: str,
    ) -> Dict[str, Any]:
        payload = self.route3_movement_payload_for_target(current, target, config)
        stage_name = str(stage or "").strip().upper()
        if stage_name not in self.route5_depth_lookahead_stages():
            return payload
        dx = float(target["x"]) - float(current["x"])
        dy = float(target["y"]) - float(current["y"])
        dist_xy = float(math.hypot(dx, dy))
        if dist_xy <= float(config["reach_tol_cm"]):
            return payload
        final_approach_radius = float(config["reach_tol_cm"]) + max(5.0, float(config["nav_step_cm"]))
        if dist_xy <= final_approach_radius:
            payload["yaw_policy"] = "close_range_final_approach"
            payload["lookahead_suppressed_reason"] = "within_final_approach_radius"
            payload["look_yaw_deg"] = round(float(self.route5_navigation_travel_yaw_deg(current, target)), 3)
            payload["capture_yaw_deg"] = round(float(target.get("yaw", target.get("yaw_deg", 0.0)) or 0.0), 3)
            return payload
        look_yaw = self.route5_navigation_travel_yaw_deg(current, target)
        yaw_error = self._normalize_angle_deg(float(look_yaw) - float(current.get("yaw", 0.0)))
        yaw_delta = 0.0 if abs(yaw_error) <= float(config["yaw_tol_deg"]) else max(-30.0, min(30.0, float(yaw_error)))
        dz = float(target.get("z", current.get("z", 0.0)) or 0.0) - float(current.get("z", 0.0) or 0.0)
        step = float(config["nav_step_cm"])
        up = 0.0 if abs(dz) <= float(config["z_tol_cm"]) else max(-step, min(step, dz))
        right = 0.0
        route7_translate_while_yawing = bool(abs(yaw_delta) > 1e-6 and self.route7_should_translate_while_yawing(target, stage_name))
        if abs(yaw_delta) > 1e-6:
            if route7_translate_while_yawing:
                scale = 0.5 if abs(yaw_error) >= 45.0 else 0.75
                forward = float(payload.get("forward_cm", 0.0) or 0.0) * scale
                right = float(payload.get("right_cm", 0.0) or 0.0) * scale
            else:
                forward = 0.0
        else:
            forward = min(step, dist_xy)
        payload = {
            "forward_cm": round(float(forward), 3),
            "right_cm": round(float(right), 3),
            "up_cm": round(float(up), 3),
            "yaw_delta_deg": round(float(yaw_delta), 3),
            "nominal_body_payload": {
                "forward_cm": payload.get("forward_cm", 0.0),
                "right_cm": payload.get("right_cm", 0.0),
                "up_cm": payload.get("up_cm", 0.0),
                "yaw_delta_deg": payload.get("yaw_delta_deg", 0.0),
            },
        }
        payload["action_name"] = "route5_nav_lookahead" if any(abs(float(payload.get(key, 0.0) or 0.0)) > 1e-6 for key in ("forward_cm", "right_cm", "up_cm", "yaw_delta_deg")) else "hold"
        payload["yaw_policy"] = "route7_translate_while_yawing_map_route" if route7_translate_while_yawing else "face_waypoint_then_forward"
        if route7_translate_while_yawing:
            payload["route7_translation_while_yawing"] = True
            payload["route7_yaw_translation_scale"] = round(float(scale), 3)
        payload["look_yaw_deg"] = round(float(look_yaw), 3)
        payload["capture_yaw_deg"] = round(float(target.get("yaw", target.get("yaw_deg", 0.0)) or 0.0), 3)
        return payload

    def route5_align_to_navigation_depth_yaw(
        self,
        session: flight.DroneFlightSession,
        current: Dict[str, float],
        target_pose: Dict[str, float],
        *,
        output_dir: Path,
        stage: str,
        facade: str,
        target_id: str,
        config: Dict[str, float],
    ) -> Dict[str, Any]:
        plan = self.route5_depth_lookahead_plan(current, target_pose, stage=stage, facade=facade, target_id=target_id)
        if not bool(plan.get("enabled", False)):
            self.route5_set_thinking_status(plan, output_dir=output_dir)
            return {"status": "skipped", "reason": "lookahead_not_required", "current_pose": current, "lookahead_plan": plan}
        attempts = 0
        max_attempts = max(1, min(8, int(math.ceil(abs(float(plan.get("yaw_error_deg", 0.0))) / 30.0)) + 2))
        updated_current = dict(current)
        last_payload = action_payload("hold")
        while (
            attempts < max_attempts
            and abs(float(plan.get("yaw_error_deg", 0.0))) > float(config["yaw_tol_deg"])
            and not self.llm_route5_stop_event.is_set()
        ):
            yaw_delta = max(-30.0, min(30.0, float(plan.get("yaw_error_deg", 0.0))))
            yaw_payload = {
                "forward_cm": 0.0,
                "right_cm": 0.0,
                "up_cm": 0.0,
                "yaw_delta_deg": round(float(yaw_delta), 3),
                "action_name": "route5_depth_lookahead_yaw",
                "yaw_policy": "align_to_waypoint_bearing_before_depth",
                "look_yaw_deg": plan.get("look_yaw_deg"),
                "capture_yaw_deg": plan.get("capture_yaw_deg"),
            }
            tick_plan = dict(plan, decision_state="ALIGN_LOOK_YAW", thinking_status=f"{plan['thinking_status']} attempt={attempts + 1}/{max_attempts}")
            self.route5_set_thinking_status(tick_plan, output_dir=output_dir)
            self.route5_log_event(
                output_dir,
                "depth_lookahead_yaw_align",
                {
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "attempt": attempts + 1,
                    "payload": yaw_payload,
                    "lookahead_plan": plan,
                },
            )
            result = self.safe("Route V5 depth lookahead yaw", lambda p=yaw_payload: session.move_relative(p))
            if not isinstance(result, dict):
                return {"status": "failed", "reason": "lookahead_yaw_failed", "current_pose": updated_current, "lookahead_plan": plan}
            try:
                self.root.after(0, lambda r=result: self.apply_state(r))
            except Exception:
                pass
            updated_current = self.route3_pose_from_payload(result) or self.route3_current_pose(session) or self.route3_predict_next_pose(updated_current, yaw_payload)
            last_payload = yaw_payload
            attempts += 1
            plan = self.route5_depth_lookahead_plan(updated_current, target_pose, stage=stage, facade=facade, target_id=target_id)
        done_plan = dict(plan, decision_state="SENSE_LOOK_YAW", thinking_status=f"{plan['thinking_status']} -> sensing depth")
        self.route5_set_thinking_status(done_plan, output_dir=output_dir)
        return {
            "status": "ok",
            "reason": "lookahead_yaw_ready",
            "current_pose": updated_current,
            "lookahead_plan": done_plan,
            "attempts": attempts,
            "last_action": last_payload,
        }

    def route5_depth_lookahead_semantic_hint(self, summary: Dict[str, Any], gate: Dict[str, Any]) -> str:
        geometry = str(summary.get("obstacle_geometry", "") or "").strip().lower() if isinstance(summary, dict) else ""
        if geometry in {"vertical_wall", "overhang_beam", "building", "wall", "facade"}:
            return "building"
        if geometry in {"low_obstacle", "thin_vertical", "fence", "rail"}:
            return "fence_or_rail"
        hint = str(gate.get("semantic_hint", "unknown") or "unknown").strip().lower() if isinstance(gate, dict) else "unknown"
        return hint if hint else "unknown"

    def route5_depth_lookahead_gate(
        self,
        event: Dict[str, Any],
        gate: Dict[str, Any],
        nominal_payload: Dict[str, Any],
        *,
        stage: str = "",
    ) -> Dict[str, Any]:
        result = dict(gate) if isinstance(gate, dict) else {}
        result["depth_lookahead_checked"] = True
        result["lookahead_blocked_directions"] = []
        event_dict = event if isinstance(event, dict) else {}
        stage_name = str(stage or event_dict.get("stage") or event_dict.get("navigation_stage") or "").strip().upper()
        if stage_name not in self.route5_depth_lookahead_stages():
            result["depth_lookahead_reason"] = "stage_not_collection_navigation"
            return result
        summary = event_dict.get("pointcloud_summary", {}) if isinstance(event_dict.get("pointcloud_summary"), dict) else {}
        pointcloud_available = bool(summary.get("available", bool(summary)))
        if not pointcloud_available:
            result["depth_lookahead_reason"] = "pointcloud_unavailable"
            return result
        blocked_directions = self.route5_payload_blocked_directions_from_depth(summary, nominal_payload if isinstance(nominal_payload, dict) else {})
        if not blocked_directions:
            if bool(result.get("avoidance_active", False)):
                result["depth_lookahead_reason"] = "base_gate_already_active"
                return result
            result["depth_lookahead_reason"] = "nominal_direction_clear"
            return result
        semantic_hint = self.route5_depth_lookahead_semantic_hint(summary, result)
        trigger_distance_cm = self.route5_avoidance_trigger_distance_cm(semantic_hint)
        result.update(
            {
                "avoidance_active": True,
                "semantic_hint": semantic_hint,
                "semantic_source": "depth_lookahead_pointcloud",
                "trigger_distance_cm": float(trigger_distance_cm),
                "front_min_depth_cm": float(self.route5_event_float(summary.get("front_min_depth_cm"), default=0.0)),
                "forward_swept_clear": bool(summary.get("forward_swept_clear", True)),
                "pointcloud_available": pointcloud_available,
                "obstacle_geometry": str(summary.get("obstacle_geometry", "unknown") or "unknown"),
                "reason": "depth_lookahead_blocked_before_collection",
                "depth_lookahead_reason": "nominal_payload_intersects_blocked_swept_direction",
                "lookahead_blocked_directions": blocked_directions,
                "lookahead_stage": stage_name,
                "lookahead_nominal_payload": dict(nominal_payload) if isinstance(nominal_payload, dict) else {},
                "lookahead_policy": "observe_depth_before_advancing_to_capture_point",
            }
        )
        return result

    def route5_strategy_cache_key(self, event: Dict[str, Any]) -> str:
        prediction = event.get("representation_prediction", {}) if isinstance(event, dict) else {}
        hint = self.route5_semantic_hint_from_prediction(prediction if isinstance(prediction, dict) else {})
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown").strip().lower()
        waypoint = str(event.get("target_id", "") or event.get("waypoint_id", "") or "unknown_waypoint").strip()
        return f"{waypoint}|{hint}|{geometry}"

    def route5_strategy_for_event(self, event: Dict[str, Any], *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        self.ensure_route5_state()
        cache = self.llm_route5_state.get("obstacle_strategy_cache", {}) if isinstance(self.llm_route5_state.get("obstacle_strategy_cache"), dict) else {}
        key = self.route5_strategy_cache_key(event)
        if key in cache and isinstance(cache[key], dict):
            cached = dict(cache[key])
            cached["strategy_source"] = "representation_prediction_cached"
            cached["strategy_cache_key"] = key
            cached["strategy_cache_hit"] = True
            cached["strategy_cache_reason"] = "reused cached strategy for waypoint/semantic/geometry"
            return cached
        prediction = event.get("representation_prediction", {}) if isinstance(event.get("representation_prediction"), dict) else {}
        hint = self.route5_semantic_hint_from_prediction(prediction)
        environment = hint if hint != "building" else "building_or_roof"
        if environment == "unknown":
            environment = "default_unreal_scene"
        episode = {
            "episode_id": str(event.get("episode_id", "route5_fusion") or "route5_fusion"),
            "environment_id": environment,
            "obstacle_hint": hint,
            "method": LLM_STRATEGY_METHOD_ID,
            "operator_note": "Route5 fused navigation strategy.",
        }
        strategy: Dict[str, Any]
        if self.effective_llm_api_key() and hasattr(self, "obstacle_avoidance_llm_strategy_decision"):
            try:
                strategy = self.obstacle_avoidance_llm_strategy_decision(event, episode)
                strategy["strategy_source"] = "route5_oa_llm_strategy"
                llm_raw = strategy.get("llm_raw", {}) if isinstance(strategy.get("llm_raw"), dict) else {}
                if output_dir is None:
                    output_dir = self.route5_state_output_dir()
                if output_dir is not None:
                    ref = self.route5_log_llm_call(
                        output_dir,
                        "oa_strategy",
                        {
                            "frame_id": event.get("frame_id"),
                            "facade": event.get("facade"),
                            "target_id": event.get("target_id"),
                            "episode": episode,
                            "representation_prediction": event.get("representation_prediction", {}),
                            "pointcloud_summary": event.get("pointcloud_summary", {}),
                        },
                        llm_raw or {"raw_text": json.dumps(strategy, ensure_ascii=False)},
                        frame_id=int(event.get("frame_id", 0) or 0) if event.get("frame_id") is not None else None,
                        facade=str(event.get("facade", "") or ""),
                        target_id=str(event.get("target_id", "") or ""),
                        decision=strategy,
                    )
                    event.setdefault("llm_call_refs", []).append(ref)
            except Exception as exc:
                strategy = strategy_from_episode_metadata(episode)
                strategy["strategy_source"] = "representation_prediction_no_api"
                strategy["llm_error"] = str(exc)
        else:
            strategy = strategy_from_episode_metadata(episode)
            strategy["strategy_source"] = "representation_prediction_no_api"
        strategy_source = str(strategy.get("strategy_source", "representation_prediction_no_api") or "representation_prediction_no_api")
        llm_error = str(strategy.get("llm_error", "") or "")
        strategy = refine_strategy_with_pointcloud_context(strategy, event)
        strategy["strategy_source"] = strategy_source
        if llm_error:
            strategy["llm_error"] = llm_error
        strategy["strategy_cache_key"] = key
        strategy["strategy_cache_hit"] = False
        strategy["strategy_cache_reason"] = "new strategy computed and cached"
        cache[key] = dict(strategy)
        self.route5_update_state(obstacle_strategy_cache=cache)
        return strategy

    def route5_obstacle_args(self, config: Dict[str, float]) -> argparse.Namespace:
        return argparse.Namespace(
            stage="route5_fused_navigation",
            method=LLM_STRATEGY_METHOD_ID,
            run_id="route5",
            geometry_label="auto",
            note="Route5 fused Route6_entrance_search and obstacle avoidance.",
            reach_tol_cm=float(config.get("reach_tol_cm", 60.0)),
            route_step_cm=float(config.get("nav_step_cm", 20.0)),
            side_correction_cm=max(float(DEFAULT_ROUTE_SIDE_CORRECTION_CM), float(config.get("nav_step_cm", 20.0))),
            vertical_step_cm=max(float(DEFAULT_ROUTE_VERTICAL_STEP_CM), float(config.get("nav_step_cm", 20.0))),
        )

    def route5_predict_obstacle_representation_3(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_route5_state()
        if ObstacleRepresentation3Predictor is None:
            return {"status": "skipped", "reason": "or3_import_unavailable"}
        route7_primary = self.route5_event_is_route7(event)
        model_path = self.route5_or3_model_path_for_event(event)
        variant = "or3_1" if route7_primary else "or3"
        rgb_path = Path(str(event.get("rgb_path", "") or "")).expanduser()
        if not model_path.is_file():
            result = {"status": "skipped", "reason": "model_not_found", "model_path": str(model_path), "or3_variant": variant}
            if route7_primary:
                result["route7_primary_representation"] = "or3_1"
            return result
        if not rgb_path.is_file():
            result = {"status": "skipped", "reason": "rgb_not_found", "rgb_path": str(rgb_path), "or3_variant": variant, "model_path": str(model_path)}
            if route7_primary:
                result["route7_primary_representation"] = "or3_1"
            return result
        try:
            predictor_path = str(model_path)
            if getattr(self, "route5_or3_predictor", None) is None or str(getattr(self, "route5_or3_predictor_path", "")) != predictor_path:
                self.route5_or3_predictor = ObstacleRepresentation3Predictor(model_path)
                self.route5_or3_predictor_path = predictor_path
            predictor = self.route5_or3_predictor
            started = time.perf_counter()
            prediction = predictor.predict(str(rgb_path), event)
            prediction["prediction_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            prediction["status"] = "ok"
            prediction["model_path"] = str(model_path)
            prediction["or3_variant"] = variant
            if route7_primary:
                prediction["route7_primary_representation"] = "or3_1"
            capture_dir = Path(str(event.get("capture_dir", ""))) if str(event.get("capture_dir", "") or "") else rgb_path.parent
            capture_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = capture_dir / f"{variant}_risk_overlay.png"
            prediction_path = capture_dir / f"{variant}_risk_prediction.json"
            if render_or3_affordance_overlay is not None:
                try:
                    rgb_image = np.asarray(Image.open(str(rgb_path)).convert("RGB"), dtype=np.uint8)
                    overlay = render_or3_affordance_overlay(rgb_image, prediction)
                    Image.fromarray(overlay).save(overlay_path)
                    prediction["risk_overlay_path"] = str(overlay_path)
                except Exception as exc:
                    prediction["overlay_error"] = str(exc)
                    prediction["risk_overlay_path"] = ""
            else:
                prediction["risk_overlay_path"] = ""
            prediction["prediction_json_path"] = str(prediction_path)
            self.write_json_artifact(
                prediction_path,
                {
                    **serializable_or2_prediction(prediction),
                    "prediction_latency_ms": prediction.get("prediction_latency_ms"),
                    "risk_overlay_path": prediction.get("risk_overlay_path", ""),
                    "model_path": str(model_path),
                    "or3_variant": variant,
                    "route7_primary_representation": prediction.get("route7_primary_representation", ""),
                },
            )
            return prediction
        except Exception as exc:
            result = {"status": "error", "reason": str(exc), "model_path": str(model_path), "rgb_path": str(rgb_path), "or3_variant": variant}
            if route7_primary:
                result["route7_primary_representation"] = "or3_1"
            return result

    def route5_predict_obstacle_representation(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_route5_state()
        route7_primary = self.route5_event_is_route7(event)
        rgb_path = Path(str(event.get("rgb_path", "") or "")).expanduser()
        or3_prediction = self.route5_predict_obstacle_representation_3(event)
        if isinstance(or3_prediction, dict) and or3_prediction.get("status") == "ok":
            event["or3_prediction"] = or3_prediction
            if route7_primary:
                event["or3_1_prediction"] = or3_prediction
                return {
                    **or3_prediction,
                    "route7_primary_representation": "or3_1",
                    "or3_variant": "or3_1",
                    "or2_replaced_by": "or3_1",
                }
        if route7_primary:
            return {
                **(or3_prediction if isinstance(or3_prediction, dict) else {}),
                "status": str((or3_prediction if isinstance(or3_prediction, dict) else {}).get("status", "skipped") or "skipped"),
                "route7_primary_representation": "or3_1",
                "or3_variant": "or3_1",
                "or2_replaced_by": "or3_1",
            }
        model_path = Path(str(self.llm_route5_representation_model_var.get() or "")).expanduser()
        if not model_path.is_file():
            return {"status": "skipped", "reason": "model_not_found", "model_path": str(model_path)}
        if not rgb_path.is_file():
            return {"status": "skipped", "reason": "rgb_not_found", "rgb_path": str(rgb_path)}
        try:
            predictor_path = str(model_path)
            if getattr(self, "route5_or2_predictor", None) is None or str(getattr(self, "route5_or2_predictor_path", "")) != predictor_path:
                self.route5_or2_predictor = ObstacleRepresentation2Predictor(model_path)
                self.route5_or2_predictor_path = predictor_path
            predictor = self.route5_or2_predictor
            started = time.perf_counter()
            prediction = predictor.predict(str(rgb_path), event)
            prediction["prediction_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            prediction["status"] = "ok"
            prediction["model_path"] = str(model_path)
            if isinstance(or3_prediction, dict) and or3_prediction.get("status") == "ok":
                prediction["or3_prediction"] = or3_prediction
            capture_dir = Path(str(event.get("capture_dir", ""))) if str(event.get("capture_dir", "") or "") else rgb_path.parent
            capture_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = capture_dir / "or2_risk_overlay.png"
            prediction_path = capture_dir / "or2_risk_prediction.json"
            try:
                rgb_image = np.asarray(Image.open(str(rgb_path)).convert("RGB"), dtype=np.uint8)
                overlay = render_affordance_overlay(rgb_image, prediction)
                Image.fromarray(overlay).save(overlay_path)
                prediction["risk_overlay_path"] = str(overlay_path)
            except Exception as exc:
                prediction["overlay_error"] = str(exc)
                prediction["risk_overlay_path"] = ""
            prediction["prediction_json_path"] = str(prediction_path)
            self.write_json_artifact(
                prediction_path,
                {
                    **serializable_or2_prediction(prediction),
                    "or3_prediction": serializable_or2_prediction(or3_prediction) if isinstance(or3_prediction, dict) else {},
                    "prediction_latency_ms": prediction.get("prediction_latency_ms"),
                    "risk_overlay_path": prediction.get("risk_overlay_path", ""),
                    "model_path": str(model_path),
                },
            )
            return prediction
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "model_path": str(model_path), "rgb_path": str(rgb_path)}

    def route5_scale_nominal_payload(self, nominal_payload: Dict[str, Any], factor: float, *, action_name: str) -> Dict[str, Any]:
        payload = dict(nominal_payload if isinstance(nominal_payload, dict) else {})
        for key in ("forward_cm", "right_cm", "up_cm", "yaw_delta_deg"):
            payload[key] = round(float(payload.get(key, 0.0) or 0.0) * float(factor), 3)
        payload["action_name"] = action_name
        return payload

    def route7_yaw_to_nav_point_payload(
        self,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        look_yaw = self.route5_navigation_travel_yaw_deg(current_pose, target_pose)
        current_yaw = self.route5_event_float((current_pose if isinstance(current_pose, dict) else {}).get("yaw"), default=0.0)
        yaw_error = self._normalize_angle_deg(float(look_yaw) - float(current_yaw))
        yaw_tol = max(1.0, self.route5_event_float((config if isinstance(config, dict) else {}).get("yaw_tol_deg"), default=10.0))
        yaw_delta = 0.0 if abs(yaw_error) <= yaw_tol else max(-30.0, min(30.0, float(yaw_error)))
        if yaw_delta < -1e-6:
            action_name = "route7_yaw_to_nav_point_q"
            key_hint = "q"
        elif yaw_delta > 1e-6:
            action_name = "route7_yaw_to_nav_point_e"
            key_hint = "e"
        else:
            action_name = "route7_navigation_point_reset_required"
            key_hint = "aligned"
        return {
            "forward_cm": 0.0,
            "right_cm": 0.0,
            "up_cm": 0.0,
            "yaw_delta_deg": round(float(yaw_delta), 3),
            "action_name": action_name,
            "route7_avoidance_policy": "forward_block_only_yaw_to_navigation_point",
            "route7_keyboard_hint": key_hint,
            "look_yaw_deg": round(float(look_yaw), 3),
            "current_yaw_deg": round(float(current_yaw), 3),
            "yaw_error_deg": round(float(yaw_error), 3),
            "yaw_tolerance_deg": round(float(yaw_tol), 3),
        }

    def route5_or2_direction_payload(self, selected_direction: str, nominal_payload: Dict[str, Any], config: Dict[str, float]) -> Dict[str, Any]:
        direction = str(selected_direction or "hold").strip().lower()
        if direction == "forward":
            payload = dict(nominal_payload if isinstance(nominal_payload, dict) else action_payload("forward"))
            payload["action_name"] = "route5_or2_forward"
            return payload
        if direction == "slow_forward":
            return self.route5_scale_nominal_payload(nominal_payload, 0.35, action_name="route5_or2_slow_forward")
        payload = action_payload(direction)
        step = max(5.0, float((config if isinstance(config, dict) else {}).get("nav_step_cm", 20.0) or 20.0))
        if direction in {"left", "right"}:
            payload["right_cm"] = -step if direction == "left" else step
        elif direction == "up":
            payload["up_cm"] = step
        elif direction == "backoff":
            payload["forward_cm"] = -step
        payload["action_name"] = {
            "left": "route5_or2_side_step_left",
            "right": "route5_or2_side_step_right",
            "up": "route5_or2_up",
            "backoff": "route5_or2_backoff",
            "hold": "route5_or2_hold",
        }.get(direction, payload.get("action_name", direction))
        return payload

    def route5_or2_direction_risk_acceptable(
        self,
        direction: str,
        *,
        risk_state: str,
        gate: Dict[str, Any],
        rule: Dict[str, Any],
    ) -> bool:
        direction = str(direction or "").strip().lower()
        risk = str(risk_state or "").strip().lower()
        scores = rule.get("candidate_action_scores", {}) if isinstance(rule.get("candidate_action_scores"), dict) else {}
        score = self.route5_event_float(scores.get(direction), default=1.0 if direction == "hold" else 0.0)
        if direction == "hold":
            return True
        if direction in {"forward", "slow_forward"}:
            if bool(gate.get("must_stop", False)) or risk == "must_stop":
                return False
            return bool(gate.get("can_forward", True)) and score >= 0.0
        if risk == "must_stop":
            return score > 0.05
        return score >= 0.0

    def route5_or2_candidate_order(self, selected_direction: str) -> List[str]:
        selected = str(selected_direction or "").strip().lower()
        base = ["slow_forward", "forward", "right", "left", "backoff", "up", "hold"]
        ordered: List[str] = []
        if selected:
            ordered.append(selected)
        for direction in base:
            if direction not in ordered:
                ordered.append(direction)
        return ordered

    def route5_apply_or2_safe_alternative(
        self,
        *,
        event: Dict[str, Any],
        gate: Dict[str, Any],
        rule: Dict[str, Any],
        selected_direction: str,
        selected_payload: Dict[str, Any],
        nominal_payload: Dict[str, Any],
        config: Dict[str, float],
        current_pose: Dict[str, float],
        target_pose: Dict[str, float],
    ) -> Dict[str, Any]:
        target_house_id = str(event.get("target_house_id", event.get("house_id", self.selected_route_target_house_id() if hasattr(self, "selected_route_target_house_id") else "")) or "")
        risk_state = str(gate.get("front_risk_state", event.get("or2_front_risk_state", "")) or "")
        candidate_scores: Dict[str, Any] = {}
        selected_safety: Dict[str, Any] = {}
        selected = str(selected_direction or "").strip().lower()
        selected_rejected_reason = ""
        depth_summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        output_dir_value = event.get("route5_output_dir") if isinstance(event, dict) else None
        output_dir = Path(str(output_dir_value)) if output_dir_value else getattr(self, "route5_local_obstacle_output_dir", None)
        route7_policy = gate.get("route7_soft_obstacle_policy", {}) if isinstance(gate.get("route7_soft_obstacle_policy", {}), dict) else {}
        if not route7_policy or str(route7_policy.get("mode", "") or "") == "not_route7_map_route":
            route7_policy = self.route7_or2_soft_obstacle_policy(event, gate, current_pose, target_pose, output_dir=output_dir, rule=rule)
        route7_policy_mode = str(route7_policy.get("mode", "") or "")
        if route7_policy_mode == "replan":
            return {
                "direction": "route7_replan",
                "payload": action_payload("hold"),
                "selected_action": "route7_map_route_replan_required",
                "reason_suffix": f"; {route7_policy.get('reason', 'route7_map_route_replan_required')}",
                "candidate_safety_scores": {},
                "safe_alternative_action": "",
                "or2_selected_action_rejected_reason": "route7_map_route_replan_required",
                "route7_soft_obstacle_policy": self.route5_json_safe(route7_policy),
                "route7_map_route_replan_request": True,
            }
        for direction in self.route5_or2_candidate_order(selected):
            payload = self.route5_or2_direction_payload(direction, nominal_payload, config)
            predicted = self.route3_predict_next_pose(current_pose, payload)
            safety = self.route3_safety_report_for_pose(target_house_id, predicted)
            risk_ok = self.route5_or2_direction_risk_acceptable(direction, risk_state=risk_state, gate=gate, rule=rule)
            depth_blocked = self.route5_payload_blocked_directions_from_depth(depth_summary, payload)
            depth_clear = not bool(depth_blocked)
            local_3d_safety = self.query_local_3d_safety(
                current_pose,
                payload,
                config,
                output_dir=output_dir,
                stage=str(event.get("route5_stage", event.get("stage", "")) or ""),
                target_id=str(event.get("target_id", "") or ""),
                candidate_direction=direction,
            )
            local_3d_clear = bool(local_3d_safety.get("safe", True))
            local_3d_blocked = list(local_3d_safety.get("blocked_directions", [])) if isinstance(local_3d_safety.get("blocked_directions"), list) else []
            map_safe = bool(safety.get("safe", False))
            safety_reason = str(safety.get("reason", "") or "")
            if not depth_clear:
                safety_reason = f"depth_swept_direction_blocked:{','.join(depth_blocked)}"
            if not local_3d_clear:
                safety_reason = str(local_3d_safety.get("reason", "") or f"local_3d_occupancy_blocked:{direction}")
            safe = bool(map_safe and depth_clear and local_3d_clear)
            candidate_scores[direction] = {
                "safe": safe,
                "map_safe": map_safe,
                "depth_swept_clear": depth_clear,
                "depth_blocked_directions": list(depth_blocked),
                "local_3d_clear": local_3d_clear,
                "local_3d_blocked_directions": local_3d_blocked,
                "local_3d_rejected_reason": "" if local_3d_clear else str(local_3d_safety.get("reason", "") or f"local_3d_occupancy_blocked:{direction}"),
                "local_3d_safety": self.route5_json_safe(local_3d_safety),
                "risk_acceptable": bool(risk_ok),
                "safety_reason": safety_reason,
                "payload": self.route5_json_safe(payload),
                "predicted_pose": self.route5_json_safe(predicted),
            }
            if direction == selected:
                selected_safety = safety
                if not map_safe:
                    selected_rejected_reason = str(safety.get("reason", "unsafe_selected_or2_action") or "unsafe_selected_or2_action")
                elif not depth_clear:
                    selected_rejected_reason = safety_reason
                elif not local_3d_clear:
                    selected_rejected_reason = safety_reason
                elif not risk_ok:
                    selected_rejected_reason = "or2_risk_not_acceptable"
                if (
                    route7_policy_mode == "continue_planned_route"
                    and direction in {"forward", "slow_forward"}
                    and selected_rejected_reason
                ):
                    continue_payload = self.route7_yaw_to_nav_point_payload(current_pose, target_pose, config)
                    yaw_error = self.route5_event_float(continue_payload.get("yaw_error_deg"), default=0.0)
                    yaw_aligned = abs(float(yaw_error)) <= self.route5_event_float(continue_payload.get("yaw_tolerance_deg"), default=10.0)
                    continue_direction = str(continue_payload.get("action_name", "route7_yaw_to_nav_point") or "route7_yaw_to_nav_point")
                    reset_request = bool(yaw_aligned)
                    reason_suffix = (
                        f"; {route7_policy.get('reason', 'route7_soft_obstacle_not_deep_red_continue_planned_route')} "
                        f"selected_or2_action={selected or 'unknown'} soft_rejected={selected_rejected_reason} "
                        f"forward_blocked=True continue={continue_direction}"
                    )
                    if reset_request:
                        reason_suffix += " route7_navigation_point_reset_request=True"
                    return {
                        "direction": continue_direction,
                        "payload": continue_payload,
                        "selected_action": str(continue_payload.get("action_name", continue_direction)),
                        "reason_suffix": reason_suffix,
                        "candidate_safety_scores": candidate_scores,
                        "safe_alternative_action": "" if continue_direction == selected else continue_direction,
                        "or2_selected_action_rejected_reason": selected_rejected_reason,
                        "route7_soft_obstacle_policy": self.route5_json_safe(route7_policy),
                        "route7_forward_blocked": True,
                        "route7_yaw_error_deg": round(float(yaw_error), 3),
                        "route7_navigation_point_reset_request": reset_request,
                        "avoidance_active": True,
                    }
            if safe and risk_ok:
                if route7_policy_mode == "continue_planned_route" and direction not in {"forward", "slow_forward", selected}:
                    continue
                if direction == selected:
                    return {
                        "direction": selected_direction,
                        "payload": selected_payload,
                        "selected_action": str(selected_payload.get("action_name", selected_direction)),
                        "reason_suffix": "",
                        "candidate_safety_scores": candidate_scores,
                        "safe_alternative_action": "",
                        "or2_selected_action_rejected_reason": "",
                        "route7_soft_obstacle_policy": self.route5_json_safe(route7_policy),
                    }
                reason_suffix = (
                    f"; selected_or2_action={selected or 'unknown'} rejected="
                    f"{selected_rejected_reason or str(selected_safety.get('reason', 'unsafe_or2_action') or 'unsafe_or2_action')} "
                    f"safe_alternative={direction}"
                )
                return {
                    "direction": direction,
                    "payload": payload,
                    "selected_action": str(payload.get("action_name", direction)),
                    "reason_suffix": reason_suffix,
                    "candidate_safety_scores": candidate_scores,
                    "safe_alternative_action": direction,
                    "or2_selected_action_rejected_reason": selected_rejected_reason or str(selected_safety.get("reason", "unsafe_or2_action") or "unsafe_or2_action"),
                    "route7_soft_obstacle_policy": self.route5_json_safe(route7_policy),
                }
        return {
            "direction": selected_direction,
            "payload": selected_payload,
            "selected_action": str(selected_payload.get("action_name", selected_direction)),
            "reason_suffix": "; no_safe_or2_alternative",
            "candidate_safety_scores": candidate_scores,
            "safe_alternative_action": "",
            "or2_selected_action_rejected_reason": selected_rejected_reason,
            "route7_soft_obstacle_policy": self.route5_json_safe(route7_policy),
        }

    def route5_depth_pointcloud_fallback_decision(
        self,
        event: Dict[str, Any],
        nominal_payload: Dict[str, Any],
        config: Dict[str, float],
        current_pose: Dict[str, float],
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        last_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        gate = self.route5_should_apply_avoidance(event, {})
        gate = self.route5_depth_lookahead_gate(event, gate, nominal_payload, stage=str(event.get("route5_stage", event.get("stage", "")) or ""))
        gate["source"] = "route5_depth_pointcloud_fallback"
        if bool(gate.get("avoidance_active", False)):
            obstacle_hint = str(gate.get("semantic_hint", event.get("representation_obstacle_hint", "unknown")) or "unknown")
            selected_action, payload, reason, phase = select_route_action(
                event.get("pointcloud_summary", {}),
                event.get("candidate_action_scores", {}),
                event.get("relative_target", {}),
                method="pointcloud_direction_rule",
                distance_to_goal_cm=float(event.get("distance_to_goal_cm", distance_3d_cm(current_pose, target_pose)) or 0.0),
                reach_tol_cm=float(config.get("reach_tol_cm", 60.0)),
                last_action=last_action,
                current_pose=current_pose,
                start=start_pose,
                goal=target_pose,
                route_step_cm=float(config.get("nav_step_cm", 20.0)),
                side_correction_cm=max(float(DEFAULT_ROUTE_SIDE_CORRECTION_CM), float(config.get("nav_step_cm", 20.0))),
                vertical_step_cm=max(float(DEFAULT_ROUTE_VERTICAL_STEP_CM), float(config.get("nav_step_cm", 20.0))),
                obstacle_hint=obstacle_hint,
                environment_id="building_or_roof" if obstacle_hint == "building" else "default_unreal_scene",
                building_state={},
            )
            risk_state = risk_state_from_summary(event.get("pointcloud_summary", {}), phase)
        else:
            selected_action = "route5_nav"
            payload = dict(nominal_payload)
            reason = str(gate.get("reason", "fallback_gate_inactive"))
            phase = "ROUTE_NAVIGATION"
            risk_state = "CLEAR"
        return {
            "gate": gate,
            "payload": payload,
            "selected_action": selected_action,
            "selected_direction": selected_action,
            "selected_action_reason": reason,
            "mission_phase": phase,
            "risk_state": risk_state,
            "rule": {"selected_direction": selected_action, "reason": reason, "candidate_action_scores": {}, "corridor_risks": {}},
            "event_updates": {
                "avoidance_gate": gate,
                "avoidance_active": bool(gate.get("avoidance_active", False)),
                "selected_action": selected_action,
                "selected_action_payload": payload,
                "selected_action_reason": reason,
                "mission_phase": phase,
                "risk_state": risk_state,
                "or2_fallback_reason": str((event.get("or2_prediction", {}) if isinstance(event.get("or2_prediction"), dict) else {}).get("reason", "")),
            },
        }

    def route5_or2_decision_for_event(
        self,
        event: Dict[str, Any],
        *,
        nominal_payload: Dict[str, Any],
        config: Dict[str, float],
        current_pose: Dict[str, float],
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        last_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        route7_primary_event = self.route5_event_is_route7(event)
        or2_prediction = event.get("or2_prediction", {}) if isinstance(event.get("or2_prediction"), dict) else {}
        or3_1_prediction = event.get("or3_1_prediction", {}) if isinstance(event.get("or3_1_prediction"), dict) else {}
        or3_prediction = event.get("or3_prediction", {}) if isinstance(event.get("or3_prediction"), dict) else {}
        prediction = or2_prediction
        route7_or3_1_primary = False
        if route7_primary_event:
            if or3_1_prediction.get("front_risk_state"):
                prediction = or3_1_prediction
                route7_or3_1_primary = True
            elif or3_prediction.get("front_risk_state"):
                prediction = or3_prediction
                route7_or3_1_primary = True
            elif str(or2_prediction.get("route7_primary_representation", "") or "") == "or3_1" or str(or2_prediction.get("or3_variant", "") or "") == "or3_1":
                prediction = or2_prediction
                route7_or3_1_primary = True
        if str(prediction.get("status", "ok") or "ok") != "ok" or "front_risk_state" not in prediction:
            return self.route5_depth_pointcloud_fallback_decision(event, nominal_payload, config, current_pose, start_pose, target_pose, last_action)
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        rule = select_or2_direction(prediction, summary, event.get("relative_target", {}), last_action=last_action)
        selected_direction = str(rule.get("selected_direction", "hold") or "hold")
        payload = self.route5_or2_direction_payload(selected_direction, nominal_payload, config)
        front_depth = self.route5_event_float(summary.get("front_min_depth_cm"), default=0.0)
        risk_state = str(prediction.get("front_risk_state", "clear") or "clear")
        front_blocked = bool(rule.get("front_blocked", False))
        active = bool(selected_direction not in {"forward", "slow_forward"} or front_blocked or risk_state in {"must_stop", "obstacle_warning"})
        reason = str(rule.get("reason", "or2_direction_rule"))
        if route7_or3_1_primary and reason == "or2_direction_rule":
            reason = "or3_1_direction_rule"
        gate = {
            "source": "route7_or3_1_a_plus_3_1" if route7_or3_1_primary else "route5_or2_a_plus_2",
            "avoidance_active": active,
            "front_risk_state": risk_state,
            "can_forward": bool(prediction.get("can_forward", selected_direction in {"forward", "slow_forward"})),
            "must_stop": bool(prediction.get("must_stop", risk_state == "must_stop")),
            "front_min_depth_cm": float(front_depth),
            "selected_direction": selected_direction,
            "reason": reason,
            "route7_primary_representation": "or3_1" if route7_or3_1_primary else "",
        }
        route7_policy = self.route7_or2_soft_obstacle_policy(event, gate, current_pose, target_pose, rule=rule)
        if str(route7_policy.get("mode", "") or "") == "continue_planned_route":
            active = False
            selected_direction = "forward"
            payload = dict(nominal_payload if isinstance(nominal_payload, dict) else payload)
            payload["action_name"] = str(payload.get("action_name", "route7_planned_route_forward") or "route7_planned_route_forward")
            gate["avoidance_active"] = False
            gate["route7_deep_red_only_avoidance"] = True
            gate["route7_soft_obstacle_policy"] = self.route5_json_safe(route7_policy)
            gate["reason"] = f"{reason}; {route7_policy.get('reason', 'route7_soft_obstacle_not_deep_red_continue_planned_route')}"
        elif str(route7_policy.get("mode", "") or "") == "replan":
            active = True
            gate["avoidance_active"] = True
            gate["route7_map_route_replan_request"] = True
            gate["route7_soft_obstacle_policy"] = self.route5_json_safe(route7_policy)
            gate["reason"] = f"{reason}; {route7_policy.get('reason', 'route7_map_route_replan_required')}"
        elif str(route7_policy.get("mode", "") or "") == "hard_avoidance":
            active = True
            gate["avoidance_active"] = True
            gate["can_forward"] = False
            gate["route7_deep_red_only_avoidance"] = True
            gate["route7_hard_avoidance_request"] = True
            gate["route7_soft_obstacle_policy"] = self.route5_json_safe(route7_policy)
            gate["reason"] = f"{reason}; {route7_policy.get('reason', 'route7_front_square_deep_red_takeover')}"
        alternative = self.route5_apply_or2_safe_alternative(
            event=event,
            gate=gate,
            rule=rule,
            selected_direction=selected_direction,
            selected_payload=payload,
            nominal_payload=nominal_payload,
            config=config,
            current_pose=current_pose,
            target_pose=target_pose,
        )
        selected_direction = str(alternative.get("direction", selected_direction) or selected_direction)
        payload = dict(alternative.get("payload", payload) if isinstance(alternative.get("payload", payload), dict) else payload)
        if "avoidance_active" in alternative:
            active = bool(alternative.get("avoidance_active", active))
        if alternative.get("reason_suffix"):
            reason = f"{reason}{alternative.get('reason_suffix')}"
            gate["reason"] = reason
        gate["selected_direction"] = selected_direction
        if alternative.get("safe_alternative_action"):
            gate["safe_alternative_action"] = str(alternative.get("safe_alternative_action", "") or "")
            gate["or2_selected_action_rejected_reason"] = str(alternative.get("or2_selected_action_rejected_reason", "") or "")
            gate["candidate_safety_scores"] = self.route5_json_safe(alternative.get("candidate_safety_scores", {}))
        if alternative.get("route7_soft_obstacle_policy"):
            gate["route7_soft_obstacle_policy"] = self.route5_json_safe(alternative.get("route7_soft_obstacle_policy", {}))
        if alternative.get("route7_map_route_replan_request"):
            gate["route7_map_route_replan_request"] = True
        if alternative.get("route7_navigation_point_reset_request"):
            gate["route7_navigation_point_reset_request"] = True
        if alternative.get("route7_forward_blocked"):
            gate["route7_forward_blocked"] = True
        if alternative.get("route7_yaw_error_deg") is not None:
            gate["route7_yaw_error_deg"] = alternative.get("route7_yaw_error_deg")
        selected_action = str(payload.get("action_name", selected_direction))
        phase = "OR3_1_AVOIDANCE" if active and route7_or3_1_primary else ("OR2_AVOIDANCE" if active else "ROUTE_NAVIGATION")
        event_updates = {
            "or2_prediction": prediction,
            "or3_prediction": prediction if route7_or3_1_primary else event.get("or3_prediction", {}),
            "or3_1_prediction": prediction if route7_or3_1_primary else event.get("or3_1_prediction", {}),
            "or3_1_primary": bool(route7_or3_1_primary),
            "route7_primary_representation": "or3_1" if route7_or3_1_primary else "",
            "or2_rule": rule,
            "or2_front_risk_state": risk_state,
            "or2_can_forward": gate["can_forward"],
            "or2_must_stop": gate["must_stop"],
            "or2_corridor_risks": rule.get("corridor_risks", {}),
            "or2_candidate_action_scores": rule.get("candidate_action_scores", {}),
            "or2_selected_direction": selected_direction,
            "or2_rule_reason": reason,
            "or2_selected_action_rejected_reason": str(alternative.get("or2_selected_action_rejected_reason", "") or ""),
            "safe_alternative_action": str(alternative.get("safe_alternative_action", "") or ""),
            "candidate_safety_scores": self.route5_json_safe(alternative.get("candidate_safety_scores", {})),
            "route7_soft_obstacle_policy": self.route5_json_safe(alternative.get("route7_soft_obstacle_policy", gate.get("route7_soft_obstacle_policy", {}))),
            "route7_map_route_replan_request": bool(alternative.get("route7_map_route_replan_request", gate.get("route7_map_route_replan_request", False))),
            "route7_navigation_point_reset_request": bool(alternative.get("route7_navigation_point_reset_request", gate.get("route7_navigation_point_reset_request", False))),
            "route7_forward_blocked": bool(alternative.get("route7_forward_blocked", gate.get("route7_forward_blocked", False))),
            "route7_yaw_error_deg": alternative.get("route7_yaw_error_deg", gate.get("route7_yaw_error_deg")),
            "or2_risk_overlay_path": str(prediction.get("risk_overlay_path", "") or ""),
            "or2_prediction_json_path": str(prediction.get("prediction_json_path", "") or ""),
            "avoidance_gate": gate,
            "avoidance_active": active,
            "selected_action": selected_action,
            "selected_action_payload": payload,
            "selected_action_reason": reason,
            "mission_phase": phase,
            "risk_state": risk_state.upper(),
            "expert_action": selected_action,
            "expert_action_payload": payload,
            "agent_action": payload,
            "nominal_action": nominal_payload,
        }
        return {
            "gate": gate,
            "payload": payload,
            "selected_action": selected_action,
            "selected_direction": selected_direction,
            "selected_action_reason": reason,
            "mission_phase": phase,
            "risk_state": risk_state.upper(),
            "rule": rule,
            "event_updates": event_updates,
        }

