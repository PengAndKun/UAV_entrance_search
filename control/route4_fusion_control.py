from __future__ import annotations

from copy import deepcopy

from .common import *

from obstacle_avoidance.collect_route_episodes import (
    DEFAULT_ROUTE_SIDE_CORRECTION_CM,
    DEFAULT_ROUTE_VERTICAL_STEP_CM,
    action_payload,
    annotate_collision_state,
    build_route_event,
    distance_3d_cm,
    pose_from_state,
    risk_state_from_summary,
    route_completion_state,
    select_route_action,
)
from obstacle_avoidance_llm.policy import (
    LLM_STRATEGY_METHOD_ID,
    refine_strategy_with_pointcloud_context,
    strategy_from_episode_metadata,
)


class Route4FusionControlMixin:
    def default_route4_representation_model_path(self) -> Path:
        plus = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_plus_model.pt"
        legacy = PROJECT_ROOT / "obstacle_representation_data" / "models" / "scheme_a_model.pt"
        return plus if plus.is_file() else legacy

    def ensure_route4_state(self) -> None:
        if not hasattr(self, "llm_route4_status_var"):
            self.llm_route4_status_var = tk.StringVar(value="LLM Route V4: idle")
        if not hasattr(self, "llm_route4_map_status_var"):
            self.llm_route4_map_status_var = tk.StringVar(value="Route V4 Map: idle")
        if not hasattr(self, "llm_route4_stage_var"):
            self.llm_route4_stage_var = tk.StringVar(value="Stage: idle")
        if not hasattr(self, "llm_route4_active_var"):
            self.llm_route4_active_var = tk.StringVar(value="Active: n/a")
        if not hasattr(self, "llm_route4_target_var"):
            self.llm_route4_target_var = tk.StringVar(value="Target: n/a")
        if not hasattr(self, "llm_route4_error_var"):
            self.llm_route4_error_var = tk.StringVar(value="Error: n/a")
        if not hasattr(self, "llm_route4_payload_var"):
            self.llm_route4_payload_var = tk.StringVar(value="Payload: hold")
        if not hasattr(self, "llm_route4_progress_text_var"):
            self.llm_route4_progress_text_var = tk.StringVar(value="Fusion: 0%")
        if not hasattr(self, "llm_route4_progress_var"):
            self.llm_route4_progress_var = tk.DoubleVar(value=0.0)
        if not hasattr(self, "llm_route4_current_status_var"):
            self.llm_route4_current_status_var = tk.StringVar(value="Current: idle")
        if not hasattr(self, "llm_route4_next_status_var"):
            self.llm_route4_next_status_var = tk.StringVar(value="Next: n/a")
        if not hasattr(self, "llm_route4_avoidance_status_var"):
            self.llm_route4_avoidance_status_var = tk.StringVar(value="Avoidance: idle")
        if not hasattr(self, "llm_route4_representation_status_var"):
            self.llm_route4_representation_status_var = tk.StringVar(value="Representation: idle")
        if not hasattr(self, "llm_route4_paused_var"):
            self.llm_route4_paused_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route4_auto_refresh_var"):
            self.llm_route4_auto_refresh_var = tk.BooleanVar(value=False)
        if not hasattr(self, "llm_route4_move_tick_ms_var"):
            self.llm_route4_move_tick_ms_var = tk.StringVar(value="150")
        if not hasattr(self, "llm_route4_nav_step_cm_var"):
            self.llm_route4_nav_step_cm_var = tk.StringVar(value="20")
        if not hasattr(self, "llm_route4_reach_tol_cm_var"):
            self.llm_route4_reach_tol_cm_var = tk.StringVar(value="60")
        if not hasattr(self, "llm_route4_z_tol_cm_var"):
            self.llm_route4_z_tol_cm_var = tk.StringVar(value="40")
        if not hasattr(self, "llm_route4_yaw_tol_deg_var"):
            self.llm_route4_yaw_tol_deg_var = tk.StringVar(value="10")
        if not hasattr(self, "llm_route4_max_stage_s_var"):
            self.llm_route4_max_stage_s_var = tk.StringVar(value="90")
        if not hasattr(self, "llm_route4_sensing_interval_s_var"):
            self.llm_route4_sensing_interval_s_var = tk.StringVar(value="1.0")
        if not hasattr(self, "llm_route4_representation_model_var"):
            self.llm_route4_representation_model_var = tk.StringVar(value=str(self.default_route4_representation_model_path()))
        if not hasattr(self, "llm_route4_window"):
            self.llm_route4_window = None
        if not hasattr(self, "llm_route4_map_widget"):
            self.llm_route4_map_widget = None
        if not hasattr(self, "llm_route4_preview_text"):
            self.llm_route4_preview_text = None
        if not hasattr(self, "llm_route4_analysis_text"):
            self.llm_route4_analysis_text = None
        if not hasattr(self, "llm_route4_rgb_label"):
            self.llm_route4_rgb_label = None
        if not hasattr(self, "llm_route4_rgb_photo"):
            self.llm_route4_rgb_photo = None
        if not hasattr(self, "llm_route4_state"):
            self.llm_route4_state = {}
        if not hasattr(self, "llm_route4_completed_facades"):
            self.llm_route4_completed_facades = set()
        if not hasattr(self, "llm_route4_blocked_facades"):
            self.llm_route4_blocked_facades = set()
        if not hasattr(self, "llm_route4_thread"):
            self.llm_route4_thread = None
        if not hasattr(self, "llm_route4_stop_event"):
            self.llm_route4_stop_event = threading.Event()
        if not hasattr(self, "llm_route4_pause_event"):
            self.llm_route4_pause_event = threading.Event()
        if not hasattr(self, "llm_route4_control_locked"):
            self.llm_route4_control_locked = False
        if not hasattr(self, "llm_route4_auto_refresh_job"):
            self.llm_route4_auto_refresh_job = None

    def route4_float_param(self, var: tk.StringVar, default: float, *, min_value: float, max_value: float) -> float:
        try:
            value = float(var.get().strip())
        except Exception:
            value = float(default)
        return max(float(min_value), min(float(max_value), float(value)))

    def route4_nav_config(self) -> Dict[str, float]:
        self.ensure_route4_state()
        return {
            "move_tick_ms": self.route4_float_param(self.llm_route4_move_tick_ms_var, 150.0, min_value=50.0, max_value=2000.0),
            "nav_step_cm": self.route4_float_param(self.llm_route4_nav_step_cm_var, 20.0, min_value=5.0, max_value=200.0),
            "reach_tol_cm": self.route4_float_param(self.llm_route4_reach_tol_cm_var, 60.0, min_value=5.0, max_value=500.0),
            "z_tol_cm": self.route4_float_param(self.llm_route4_z_tol_cm_var, 40.0, min_value=5.0, max_value=500.0),
            "yaw_tol_deg": self.route4_float_param(self.llm_route4_yaw_tol_deg_var, 10.0, min_value=1.0, max_value=90.0),
            "max_stage_s": self.route4_float_param(self.llm_route4_max_stage_s_var, 90.0, min_value=5.0, max_value=900.0),
        }

    def route4_sensing_config(self) -> Dict[str, Any]:
        self.ensure_route4_state()
        interval_s = self.route4_float_param(self.llm_route4_sensing_interval_s_var, 1.0, min_value=0.2, max_value=30.0)
        return {
            "sensing_interval_s": float(interval_s),
            "representation_model_path": str(self.llm_route4_representation_model_var.get() or ""),
            "llm_strategy_mode": LLM_STRATEGY_METHOD_ID,
            "capture_processing": "minimal",
        }

    def make_route4_fused_output_dir(self, target_house_id: str) -> Path:
        root = self.resolve_project_path("route_capture_lidar")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_autosearch_v4_fused_{timestamp}"
        root.mkdir(parents=True, exist_ok=True)
        candidate = root / base_name
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = root / f"{base_name}_{suffix}"
        (candidate / "frames").mkdir(parents=True, exist_ok=True)
        (candidate / "reconstruction").mkdir(parents=True, exist_ok=True)
        (candidate / "facade_observations").mkdir(parents=True, exist_ok=True)
        return candidate

    def route4_state_output_dir(self) -> Optional[Path]:
        self.ensure_route4_state()
        state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        raw = str(state.get("output_dir", "") or "")
        if not raw:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        (path / "reconstruction").mkdir(parents=True, exist_ok=True)
        (path / "facade_observations").mkdir(parents=True, exist_ok=True)
        return path

    def route4_update_state(self, **updates: Any) -> Dict[str, Any]:
        self.ensure_route4_state()
        state = dict(self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {})
        state.update(updates)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route4_state = state
        return state

    def route4_write_state_artifact(self) -> None:
        output_dir = self.route4_state_output_dir()
        if output_dir is None:
            return
        self.write_json_artifact(output_dir / "route4_fusion_state.json", self.llm_route4_state)

    def route4_log_event(self, output_dir: Optional[Path], event_type: str, payload: Dict[str, Any]) -> None:
        if output_dir is None:
            return
        row = {
            "event_type": str(event_type),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "payload": payload,
        }
        self.append_jsonl(output_dir / "route4_fusion_events.jsonl", row)

    def route4_set_stage(
        self,
        stage: str,
        *,
        output_dir: Optional[Path] = None,
        facade: str = "",
        target: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> None:
        self.ensure_route4_state()
        updates: Dict[str, Any] = {"stage": stage}
        if facade:
            updates["current_facade"] = facade
        if target:
            updates["target_pose"] = target
        if error:
            updates["last_error"] = error
        self.route4_update_state(**updates)
        self.route4_write_state_artifact()
        self.route4_log_event(output_dir or self.route4_state_output_dir(), "stage", {"stage": stage, "facade": facade, "message": message})
        try:
            self.root.after(0, lambda s=stage: self.llm_route4_stage_var.set(f"Stage: {s}"))
            if facade:
                self.root.after(0, lambda f=facade: self.llm_route4_active_var.set(f"Active: facade={f}"))
            if target:
                self.root.after(
                    0,
                    lambda t=target: self.llm_route4_target_var.set(
                        f"Target: x={float(t.get('x', 0.0)):.1f} y={float(t.get('y', 0.0)):.1f} z={float(t.get('z', 0.0)):.1f}"
                    ),
                )
            if error:
                self.root.after(0, lambda e=error: self.llm_route4_error_var.set(f"Error: {e.get('reason', e.get('status', 'CHECK'))}"))
            if message:
                self.root.after(0, lambda m=message: self.llm_route4_status_var.set(f"LLM Route V4: {m}"))
        except Exception:
            pass

    def route4_initialize_run(self, target_house_id: str, *, force_new: bool = False) -> Path:
        self.ensure_route4_state()
        current = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        current_target = str(current.get("target_house_id", "") or "")
        output_dir = self.route4_state_output_dir()
        created_new_run = False
        if force_new or output_dir is None or current_target != str(target_house_id):
            created_new_run = True
            output_dir = self.make_route4_fused_output_dir(target_house_id)
            self.llm_route4_completed_facades = set()
            self.llm_route4_blocked_facades = set()
            self.llm_route4_state = {
                "mode": "route4_llm_route_avoidance_fusion",
                "target_house_id": target_house_id,
                "output_dir": str(output_dir),
                "stage": "INIT_RUN",
                "nav_config": self.route4_nav_config(),
                "sensing_config": self.route4_sensing_config(),
                "last_obstacle_event": {},
                "obstacle_strategy_cache": {},
                "completed_facades": [],
                "blocked_facades": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            for artifact_name in (
                "route4_fusion_events.jsonl",
                "route4_navigation_plan.jsonl",
                "route4_movement_trace.jsonl",
                "avoidance_events.jsonl",
            ):
                (output_dir / artifact_name).touch(exist_ok=True)
        self.llm_route2_state = {
            "mode": "facade_by_facade_vlm_v4_fused",
            "target_house_id": target_house_id,
            "output_dir": str(output_dir),
            "facade": "",
            "facade_id": "",
            "completed_facades": sorted(self.llm_route4_completed_facades),
        }
        self.llm_route2_completed_facades = set(self.llm_route4_completed_facades)
        self.route4_write_state_artifact()
        if created_new_run:
            self.route4_write_avoidance_summary(output_dir, status="initialized")
        return output_dir

    def route4_task_facade_priority(self) -> List[str]:
        state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        plan = state.get("task_plan", {}) if isinstance(state.get("task_plan"), dict) else {}
        raw = plan.get("facade_priority", []) if isinstance(plan, dict) else []
        result = [str(item).strip().lower() for item in raw if str(item).strip().lower() in {"south", "east", "north", "west"}]
        for facade in ("south", "east", "north", "west"):
            if facade not in result:
                result.append(facade)
        return result

    def route4_analyze_task_plan(self, target_house_id: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        task_text = self.llm_task_text_var.get().strip()
        registry = self.route3_house_registry_for_task_plan()
        parsed: Dict[str, Any] = {}
        response_payload: Dict[str, Any] = {}
        if self.effective_llm_api_key() and self.effective_llm_model():
            try:
                context = {
                    "selected_target_house_id": str(target_house_id or ""),
                    "task_text": task_text,
                    "available_facades": ["south", "east", "north", "west"],
                    "house_registry": registry,
                    "fusion_mode": "route4_llm_route_avoidance_fusion",
                }
                response_payload = self.call_configured_llm_text(
                    system_prompt=(
                        "You are the high-level task planner for a UAV house search with local obstacle avoidance. "
                        "Return strict compact JSON only. Do not output low-level movement commands."
                    ),
                    user_prompt=(
                        "Plan ordered target houses and facade priority for the fused route + obstacle-avoidance run.\n"
                        f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}\n"
                        f"Expected JSON:\n{json.dumps(LLM_ROUTE3_TASK_PLAN_SCHEMA, indent=2, ensure_ascii=False)}"
                    ),
                    max_output_tokens=700,
                    json_schema=LLM_ROUTE3_TASK_PLAN_SCHEMA,
                )
                parsed = extract_json_object(str(response_payload.get("raw_text", "") or ""))
            except Exception as exc:
                response_payload = {"error": str(exc), "fallback_used": True}
        if not parsed:
            explicit_targets = self.route3_explicit_house_sequence_from_text(task_text, registry)
            fallback_target = explicit_targets[0] if explicit_targets else str(target_house_id or "")
            local_targets = explicit_targets or ([fallback_target] if fallback_target else [])
            parsed = {
                "target_house_id": str(fallback_target or ""),
                "ordered_targets": self.route3_task_target_entries_from_ids(
                    local_targets,
                    source="local_fallback_task_text" if explicit_targets else "selected_target_house",
                ),
                "major_task": task_text or "Search selected house entrance with obstacle avoidance.",
                "facade_priority": self.route3_default_facade_priority(task_text),
                "preferred_start_facade": "",
                "completion_criteria": "Reachable facades have RGB/VLM analysis, scan captures, validation, and avoidance logs.",
                "reason": "Local fallback task analysis.",
                "planner_source": "route4_local_fallback_no_or_failed_api",
            }
        plan = self.route3_normalize_task_plan(parsed, target_house_id, task_text)
        self.route4_update_state(task_plan=plan)
        if output_dir is not None:
            self.write_json_artifact(output_dir / "route4_task_plan.json", {"plan": plan, "llm_response": response_payload})
            self.route4_log_event(output_dir, "task_plan_analysis", {"task_plan": plan, "llm_response": response_payload})
            self.route4_write_state_artifact()
        return plan

    def route4_rank_observation_candidates(
        self,
        target_house_id: str,
        candidates: List[Dict[str, Any]],
        completed: set[str],
        blocked: set[str],
        *,
        start_pose: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        current = dict(start_pose or self.route3_current_pose() or self.current_route_pose() or {})
        priority = self.route4_task_facade_priority()
        priority_index = {facade: idx for idx, facade in enumerate(priority)}
        base_by_facade: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            if isinstance(candidate, dict):
                facade = str(candidate.get("facade", "") or "").strip().lower()
                if facade in {"south", "east", "north", "west"} and facade not in base_by_facade:
                    base_by_facade[facade] = candidate
        ranked: List[Dict[str, Any]] = []
        for facade in ("south", "east", "north", "west"):
            if facade in completed or facade in blocked:
                continue
            base = base_by_facade.get(facade, {})
            attempts = self.route3_observation_attempts_for_facade(target_house_id, facade, base)
            ranked_attempts: List[Dict[str, Any]] = []
            for attempt in attempts:
                item = dict(attempt)
                pose = self.route3_target_pose_from_point(item)
                if not current:
                    item["route4_navigation_status"] = "unknown_no_current_pose"
                    item["route4_navigation_cost_cm"] = float(item.get("observation_selection_score", item.get("distance_to_uav_cm", 0.0)) or 0.0)
                    ranked_attempts.append(item)
                    continue
                plan = self.route3_plan_navigation_waypoints(current, pose, target_house_id, grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM))
                item["route4_navigation_plan"] = plan
                item["route4_navigation_status"] = str(plan.get("status", "blocked") or "blocked")
                item["route4_navigation_reason"] = str(plan.get("reason", "") or "")
                item["route3_navigation_status"] = item["route4_navigation_status"]
                if plan.get("status") == "ok":
                    item["route4_navigation_cost_cm"] = round(self.route3_navigation_plan_cost_cm(current, plan), 2)
                    item["status"] = "planned"
                else:
                    item["route4_navigation_cost_cm"] = float("inf")
                    item["status"] = "blocked"
                    item["observation_block_reason"] = str(plan.get("reason", item.get("observation_block_reason", "navigation_blocked")) or "navigation_blocked")
                ranked_attempts.append(item)
            ranked_attempts.sort(
                key=lambda item: (
                    0 if item.get("route4_navigation_status") == "ok" else 1,
                    float(item.get("route4_navigation_cost_cm", float("inf"))),
                    int(item.get("observation_attempt_index", 999) or 999),
                )
            )
            feasible = next((item for item in ranked_attempts if item.get("route4_navigation_status") == "ok"), None)
            selected = dict(feasible or (ranked_attempts[0] if ranked_attempts else base))
            nav_cost = float(selected.get("route4_navigation_cost_cm", selected.get("distance_to_uav_cm", 0.0)) or 0.0)
            if not math.isfinite(nav_cost):
                nav_cost = 1_000_000.0
            selected["route4_observation_rank_score"] = round(float(nav_cost + 25.0 * float(priority_index.get(facade, len(priority)))), 2)
            selected["route4_facade_priority_index"] = priority_index.get(facade, len(priority))
            selected["selected_observation_attempt"] = dict(feasible or selected)
            selected["observation_attempts"] = ranked_attempts
            selected["observation_attempt_count"] = len(ranked_attempts)
            if feasible is None:
                selected["status"] = "blocked"
                selected["route4_navigation_status"] = str(selected.get("route4_navigation_status", "blocked") or "blocked")
            ranked.append(selected)
        ranked.sort(
            key=lambda item: (
                0 if item.get("route4_navigation_status") == "ok" and item.get("status") != "blocked" else 1,
                float(item.get("route4_observation_rank_score", 1_000_000.0)),
                int(item.get("route4_facade_priority_index", 99) or 99),
            )
        )
        return ranked

    def route4_decide_next_facade(
        self,
        target_house_id: str,
        candidates: List[Dict[str, Any]],
        completed: set[str],
        blocked: set[str],
    ) -> Dict[str, Any]:
        available = [
            candidate for candidate in candidates
            if isinstance(candidate, dict)
            and str(candidate.get("facade", "") or "") not in completed
            and str(candidate.get("facade", "") or "") not in blocked
            and str(candidate.get("status", "") or "") != "blocked"
            and not str(candidate.get("observation_blocking_house_id", "") or "")
            and str(candidate.get("route4_navigation_status", candidate.get("route3_navigation_status", "ok")) or "ok") == "ok"
        ]
        fallback = {
            "next_action": "select_facade" if available else "done",
            "target_facade": str(available[0].get("facade", "") or "") if available else "",
            "reason": "fallback nearest feasible uncompleted facade",
            "rescan_required": False,
            "stop_condition_met": not bool(available),
            "planner_source": "route4_rule_fallback",
            "ranked_candidates_considered": len(candidates),
        }
        if not available or not self.effective_llm_api_key():
            return fallback
        try:
            context = {
                "target_house_id": target_house_id,
                "completed_facades": sorted(completed),
                "blocked_facades": sorted(blocked),
                "candidate_observation_points": available,
                "current_pose": self.route3_current_pose(),
                "task": self.llm_task_text_var.get().strip(),
                "fusion_mode": "route4_llm_route_avoidance_fusion",
            }
            response = self.call_configured_llm_text(
                system_prompt=(
                    "You are a high-level UAV house facade search supervisor for a fused route and obstacle-avoidance run. "
                    "Choose the next facade only; do not output low-level movement commands. Return compact JSON."
                ),
                user_prompt=(
                    "Choose the next safe facade to search. Return JSON with keys "
                    "next_action(select_facade|done), target_facade, reason, rescan_required, stop_condition_met.\n"
                    f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
                ),
                max_output_tokens=400,
                json_schema={
                    "next_action": "select_facade",
                    "target_facade": "west",
                    "reason": "",
                    "rescan_required": False,
                    "stop_condition_met": False,
                },
            )
            parsed = extract_json_object(str(response.get("raw_text", "") or ""))
            chosen = str(parsed.get("target_facade", "") or "").strip().lower()
            valid_facades = {str(item.get("facade", "") or "") for item in available}
            if chosen in valid_facades:
                nearest = str(available[0].get("facade", "") or "") if available else chosen
                if chosen != nearest:
                    parsed["llm_requested_facade"] = chosen
                    parsed["target_facade"] = nearest
                    parsed["rank_correction_reason"] = "nearest feasible observation point has priority over LLM facade preference"
                    parsed["planner_source"] = "route4_llm_high_level_rank_corrected"
                else:
                    parsed["planner_source"] = "route4_llm_high_level"
                return parsed
        except Exception as exc:
            return {**fallback, "llm_error": str(exc)}
        return fallback

    def route4_set_control_lock(self, locked: bool) -> None:
        self.ensure_route4_state()
        self.llm_route4_control_locked = bool(locked)
        try:
            if locked:
                self.root.after(0, lambda: self.stop_keyboard_control(send_hold=False))
                self.root.after(0, lambda: self.update_keyboard_status("locked by LLM Route V4"))
            else:
                self.root.after(0, self.update_keyboard_status)
        except Exception:
            pass

    def route4_enable_physics_movement(self, session: flight.DroneFlightSession) -> None:
        self.route4_set_control_lock(True)
        result = self.safe("Route V4 movement mode physics", lambda: session.set_movement_mode("physics"))
        self.safe("Route V4 enable movement", lambda: session.set_movement_enabled(True))
        self.movement_mode_state = "physics"
        try:
            self.root.after(0, lambda: self.movement_mode_var.set("physics"))
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_state(r))
        except Exception:
            pass

    def route4_hold(self, session: flight.DroneFlightSession, *, output_dir: Optional[Path] = None, reason: str = "hold") -> Dict[str, Any]:
        payload = action_payload("hold")
        result = self.safe("Route V4 hold", lambda: session.move_relative(payload))
        self.route4_log_event(output_dir, "hold", {"reason": reason, "payload": payload})
        return result if isinstance(result, dict) else {}

    def route4_wait_if_paused(self, session: flight.DroneFlightSession, output_dir: Optional[Path]) -> bool:
        self.ensure_route4_state()
        held = False
        while self.llm_route4_pause_event.is_set() and not self.llm_route4_stop_event.is_set():
            if not held:
                self.route4_hold(session, output_dir=output_dir, reason="paused")
                held = True
            self.root.after(0, lambda: self.llm_route4_status_var.set("LLM Route V4: paused."))
            time.sleep(0.2)
        return bool(self.llm_route4_stop_event.is_set())

    def route4_semantic_hint_from_prediction(self, prediction: Dict[str, Any]) -> str:
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

    def route4_strategy_cache_key(self, event: Dict[str, Any]) -> str:
        prediction = event.get("representation_prediction", {}) if isinstance(event, dict) else {}
        hint = self.route4_semantic_hint_from_prediction(prediction if isinstance(prediction, dict) else {})
        summary = event.get("pointcloud_summary", {}) if isinstance(event.get("pointcloud_summary"), dict) else {}
        geometry = str(summary.get("obstacle_geometry", "unknown") or "unknown").strip().lower()
        waypoint = str(event.get("target_id", "") or event.get("waypoint_id", "") or "unknown_waypoint").strip()
        return f"{waypoint}|{hint}|{geometry}"

    def route4_strategy_for_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_route4_state()
        cache = self.llm_route4_state.get("obstacle_strategy_cache", {}) if isinstance(self.llm_route4_state.get("obstacle_strategy_cache"), dict) else {}
        key = self.route4_strategy_cache_key(event)
        if key in cache and isinstance(cache[key], dict):
            cached = dict(cache[key])
            cached["strategy_source"] = "representation_prediction_cached"
            return cached
        prediction = event.get("representation_prediction", {}) if isinstance(event.get("representation_prediction"), dict) else {}
        hint = self.route4_semantic_hint_from_prediction(prediction)
        environment = hint if hint != "building" else "building_or_roof"
        if environment == "unknown":
            environment = "default_unreal_scene"
        episode = {
            "episode_id": str(event.get("episode_id", "route4_fusion") or "route4_fusion"),
            "environment_id": environment,
            "obstacle_hint": hint,
            "method": LLM_STRATEGY_METHOD_ID,
            "operator_note": "Route4 fused navigation strategy.",
        }
        strategy: Dict[str, Any]
        if self.effective_llm_api_key() and hasattr(self, "obstacle_avoidance_llm_strategy_decision"):
            try:
                strategy = self.obstacle_avoidance_llm_strategy_decision(event, episode)
                strategy["strategy_source"] = "route4_oa_llm_strategy"
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
        cache[key] = dict(strategy)
        self.route4_update_state(obstacle_strategy_cache=cache)
        return strategy

    def route4_obstacle_args(self, config: Dict[str, float]) -> argparse.Namespace:
        return argparse.Namespace(
            stage="route4_fused_navigation",
            method=LLM_STRATEGY_METHOD_ID,
            run_id="route4",
            geometry_label="auto",
            note="Route4 fused route and obstacle avoidance.",
            reach_tol_cm=float(config.get("reach_tol_cm", 60.0)),
            route_step_cm=float(config.get("nav_step_cm", 20.0)),
            side_correction_cm=max(float(DEFAULT_ROUTE_SIDE_CORRECTION_CM), float(config.get("nav_step_cm", 20.0))),
            vertical_step_cm=max(float(DEFAULT_ROUTE_VERTICAL_STEP_CM), float(config.get("nav_step_cm", 20.0))),
        )

    def route4_predict_obstacle_representation(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_route4_state()
        model_path = Path(str(self.llm_route4_representation_model_var.get() or "")).expanduser()
        rgb_path = Path(str(event.get("rgb_path", "") or "")).expanduser()
        if not model_path.is_file():
            return {"status": "skipped", "reason": "model_not_found", "model_path": str(model_path)}
        if not rgb_path.is_file():
            return {"status": "skipped", "reason": "rgb_not_found", "rgb_path": str(rgb_path)}
        try:
            from obstacle_representation.demo import predict_obstacle_representation

            prediction = predict_obstacle_representation(model_path, rgb_path, event)
            prediction["status"] = "ok"
            return prediction
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "model_path": str(model_path), "rgb_path": str(rgb_path)}

    def route4_capture_obstacle_event(
        self,
        session: flight.DroneFlightSession,
        *,
        output_dir: Path,
        target_pose: Dict[str, float],
        start_pose: Dict[str, float],
        last_action: Dict[str, Any],
        stage: str,
        facade: str,
        target_id: str,
        frame_id: int,
        config: Dict[str, float],
    ) -> Dict[str, Any]:
        action_detail = {
            "source": "llm_route_v4_fused_navigation",
            "stage": stage,
            "facade": facade,
            "target_id": target_id,
            "target_waypoint": target_pose,
            "last_action": last_action,
        }
        old_mode = str(getattr(session.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING))
        try:
            session.args.lidar_capture_processing = "minimal"
            capture_result = session.capture_lidar_stream_frame(output_dir, frame_id, action_detail=action_detail)
        finally:
            try:
                session.args.lidar_capture_processing = old_mode
            except Exception:
                pass
        if not isinstance(capture_result, dict):
            raise RuntimeError("capture_lidar_stream_frame returned non-dict")
        episode = {
            "episode_id": f"route4_{target_id}_{frame_id}",
            "scenario_id": f"route4_{target_id}",
            "environment_id": "default_unreal_scene",
            "method": LLM_STRATEGY_METHOD_ID,
            "obstacle_hint": "unknown",
        }
        event = build_route_event(
            capture_result,
            session_dir=output_dir,
            frame_id=frame_id,
            args=self.route4_obstacle_args(config),
            episode=episode,
            episode_index=1,
            start=start_pose,
            goal=target_pose,
            last_action=last_action,
        )
        event["source"] = "llm_route_v4_fused_navigation"
        event["route4_stage"] = stage
        event["facade"] = facade
        event["target_id"] = target_id
        prediction = self.route4_predict_obstacle_representation(event)
        event["representation_prediction"] = prediction
        hint = self.route4_semantic_hint_from_prediction(prediction)
        event["representation_obstacle_hint"] = hint
        self.route4_update_state(last_obstacle_event=event)
        self.route4_log_event(output_dir, "obstacle_sensing", event)
        try:
            label = str(prediction.get("predicted_label", prediction.get("status", "unknown")) or "unknown")
            confidence = float(prediction.get("confidence", 0.0) or 0.0)
            self.root.after(0, lambda l=label, c=confidence: self.llm_route4_representation_status_var.set(f"Representation: {l} conf={c:.2f}"))
        except Exception:
            pass
        return event

    def route4_normalize_avoidance_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(event if isinstance(event, dict) else {})
        collision = bool(normalized.get("collision_state", False))
        normalized["collision_state"] = collision
        normalized["avoidance_failed"] = collision
        return normalized

    def route4_write_avoidance_summary(self, output_dir: Path, *, status: str = "running") -> Dict[str, Any]:
        events = self.read_jsonl_artifact(output_dir / "avoidance_events.jsonl")
        collision_count = sum(1 for item in events if bool(item.get("collision_state", False)))
        summary = {
            "source": "llm_route_v4_fused_navigation",
            "status": status,
            "event_count": len(events),
            "collision_count": collision_count,
            "avoidance_failed_count": sum(1 for item in events if bool(item.get("avoidance_failed", False))),
            "last_event": events[-1] if events else {},
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(output_dir / "avoidance_session_summary.json", summary)
        return summary

    def route4_follow_navigation_waypoint_with_fusion(
        self,
        session: flight.DroneFlightSession,
        target_pose: Dict[str, float],
        *,
        output_dir: Path,
        stage: str,
        facade: str,
        target_id: str,
        target_house_id: str,
        waypoint_index: int,
        waypoint_count: int,
        config: Dict[str, float],
        escape_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = self.route3_current_pose(session)
        if not current:
            return {"status": "failed", "reason": "missing_current_pose"}
        started_at = time.time()
        tick_s = float(config["move_tick_ms"]) / 1000.0
        sensing_interval_s = float(self.route4_sensing_config()["sensing_interval_s"])
        step_index = 0
        last_error: Dict[str, Any] = {}
        last_sense_at = 0.0
        last_action = action_payload("hold")
        start_pose = dict(current)
        target_safety = self.route3_safety_report_for_pose(target_house_id, target_pose)
        if not bool(target_safety.get("safe", False)):
            self.route4_hold(session, output_dir=output_dir, reason="unsafe_waypoint")
            return {"status": "blocked", "reason": "unsafe_waypoint", "safety": target_safety}
        while not self.llm_route4_stop_event.is_set():
            if self.route4_wait_if_paused(session, output_dir):
                break
            error = self.route3_pose_error(current, target_pose, config)
            last_error = error
            self.root.after(
                0,
                lambda e=error: self.llm_route4_error_var.set(
                    f"Error: xy={float(e['dist_xy_cm']):.1f} z={float(e['dz']):.1f} yaw={float(e['yaw_error_deg']):.1f}"
                ),
            )
            if bool(error.get("reached", False)):
                self.route4_hold(session, output_dir=output_dir, reason="target_reached")
                return {
                    "status": "ok",
                    "reason": "target_reached",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "waypoint_index": waypoint_index,
                    "waypoint_count": waypoint_count,
                    "pose_error": error,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "current_pose": current,
                }
            if time.time() - started_at > float(config["max_stage_s"]):
                self.route4_hold(session, output_dir=output_dir, reason="nav_timeout")
                return {
                    "status": "timeout",
                    "reason": "nav_timeout",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "pose_error": error,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "current_pose": current,
                }
            payload = self.route3_movement_payload_for_target(current, target_pose, config)
            event: Dict[str, Any] = {}
            should_sense = step_index == 0 or (time.time() - last_sense_at) >= sensing_interval_s
            if should_sense:
                try:
                    frame_id = self.route2_next_frame_index(output_dir)
                    event = self.route4_capture_obstacle_event(
                        session,
                        output_dir=output_dir,
                        target_pose=target_pose,
                        start_pose=start_pose,
                        last_action=last_action,
                        stage=stage,
                        facade=facade,
                        target_id=target_id,
                        frame_id=frame_id,
                        config=config,
                    )
                    last_sense_at = time.time()
                    strategy = self.route4_strategy_for_event(event)
                    event["llm_strategy"] = deepcopy(strategy)
                    obstacle_hint = str(strategy.get("obstacle_hint", event.get("representation_obstacle_hint", "unknown")) or "unknown")
                    environment_id = str(strategy.get("environment_id", "default_unreal_scene") or "default_unreal_scene")
                    selected_action, avoidance_payload, reason, phase = select_route_action(
                        event["pointcloud_summary"],
                        event["candidate_action_scores"],
                        event.get("relative_target", {}),
                        method="pointcloud_direction_rule",
                        distance_to_goal_cm=float(event.get("distance_to_goal_cm", distance_3d_cm(current, target_pose)) or 0.0),
                        reach_tol_cm=float(config["reach_tol_cm"]),
                        last_action=last_action,
                        current_pose=current,
                        start=start_pose,
                        goal=target_pose,
                        route_step_cm=float(config["nav_step_cm"]),
                        side_correction_cm=max(float(DEFAULT_ROUTE_SIDE_CORRECTION_CM), float(config["nav_step_cm"])),
                        vertical_step_cm=max(float(DEFAULT_ROUTE_VERTICAL_STEP_CM), float(config["nav_step_cm"])),
                        obstacle_hint=obstacle_hint,
                        environment_id=environment_id,
                        building_state={},
                    )
                    risk_state = risk_state_from_summary(event["pointcloud_summary"], phase)
                    event.update(
                        {
                            "mission_phase": phase,
                            "risk_state": risk_state,
                            "selected_action": selected_action,
                            "selected_action_payload": avoidance_payload,
                            "selected_action_reason": reason,
                            "expert_action": str(avoidance_payload.get("action_name", selected_action)),
                            "expert_action_payload": avoidance_payload,
                            "agent_action": avoidance_payload,
                            "nominal_action": payload,
                        }
                    )
                    payload = avoidance_payload
                    self.root.after(
                        0,
                        lambda h=obstacle_hint, a=selected_action, r=risk_state: self.llm_route4_avoidance_status_var.set(
                            f"Avoidance: hint={h} action={a} risk={r}"
                        ),
                    )
                except Exception as exc:
                    self.route4_log_event(output_dir, "obstacle_sensing_failed", {"error": str(exc), "stage": stage, "target_id": target_id})
                    self.root.after(0, lambda e=exc: self.llm_route4_avoidance_status_var.set(f"Avoidance: fallback navigation ({e})"))
            predicted = self.route3_predict_next_pose(current, payload)
            safety = self.route3_safety_report_for_pose(target_house_id, predicted)
            escape_safety_allowed = self.route3_escape_safety_allowed(current, predicted, target_pose, safety, escape_obstacle)
            trace = {
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "waypoint_index": waypoint_index,
                "waypoint_count": waypoint_count,
                "step_index": step_index,
                "current_pose": current,
                "target_pose": target_pose,
                "pose_error": error,
                "payload": payload,
                "predicted_pose": predicted,
                "safety": safety,
                "obstacle_event_frame_id": event.get("frame_id") if event else None,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            self.append_jsonl(output_dir / "route4_movement_trace.jsonl", trace)
            self.root.after(
                0,
                lambda p=payload: self.llm_route4_payload_var.set(
                    f"Payload: f={p['forward_cm']} r={p['right_cm']} u={p['up_cm']} yaw={p['yaw_delta_deg']}"
                ),
            )
            if not bool(safety.get("safe", False)) and not escape_safety_allowed:
                self.route4_hold(session, output_dir=output_dir, reason=str(safety.get("reason", "unsafe_next_step")))
                self.route4_log_event(output_dir, "navigation_blocked", trace)
                return {
                    "status": "blocked",
                    "reason": str(safety.get("reason", "unsafe_next_step")),
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "pose_error": error,
                    "safety": safety,
                    "current_pose": current,
                }
            result = self.safe("Route V4 fused movement tick", lambda p=payload: session.move_relative(p))
            if not isinstance(result, dict):
                self.route4_hold(session, output_dir=output_dir, reason="movement_failed")
                return {"status": "failed", "reason": "movement_failed", "pose_error": error, "current_pose": current}
            self.root.after(0, lambda r=result: self.apply_state(r))
            post_pose = self.route3_pose_from_payload(result) or predicted
            if event:
                hit_state = self.read_obstacle_avoidance_2_hit_state(session) if hasattr(self, "read_obstacle_avoidance_2_hit_state") else {}
                post_reached, completion_reason, post_raw_progress, post_cross_track_cm = route_completion_state(
                    post_pose,
                    start_pose,
                    target_pose,
                    reach_tol_cm=float(config["reach_tol_cm"]),
                )
                event["goal_completion_reason"] = completion_reason
                event["goal_reached"] = bool(post_reached)
                event["post_action_pose"] = post_pose
                event["post_distance_to_goal_cm"] = round(distance_3d_cm(post_pose, target_pose), 3)
                event["post_raw_route_progress"] = round(post_raw_progress, 5)
                event["post_route_cross_track_cm"] = round(post_cross_track_cm, 3)
                annotate_collision_state(event, pre_pose=current, post_pose=post_pose, payload=payload, explicit_collision=hit_state)
                event = self.route4_normalize_avoidance_event(event)
                self.append_jsonl(output_dir / "avoidance_events.jsonl", event)
                self.route4_write_avoidance_summary(output_dir, status="running")
                if bool(event.get("collision_state")):
                    self.route4_hold(session, output_dir=output_dir, reason="collision")
                    return {"status": "collision", "reason": "collision", "event": event, "current_pose": post_pose}
            current = post_pose
            last_action = payload
            step_index += 1
            if self.llm_route4_stop_event.wait(max(0.01, tick_s)):
                break
        self.route4_hold(session, output_dir=output_dir, reason="stopped")
        return {"status": "stopped", "reason": "stopped", "pose_error": last_error, "current_pose": current}

    def route4_navigate_to_pose_with_fusion(
        self,
        session: flight.DroneFlightSession,
        target_pose: Dict[str, float],
        *,
        output_dir: Path,
        stage: str,
        facade: str,
        target_id: str,
        target_house_id: str,
    ) -> Dict[str, Any]:
        base_config = self.route4_nav_config()
        self.route4_enable_physics_movement(session)
        current = self.route3_current_pose(session)
        if not current:
            return {"status": "failed", "reason": "missing_current_pose"}
        max_replans = int(LLM_ROUTE3_ASTAR_MAX_REPLANS)
        replan_count = 0
        started_at = time.time()
        last_result: Dict[str, Any] = {}
        while replan_count <= max_replans and not self.llm_route4_stop_event.is_set():
            plan = self.route3_plan_navigation_waypoints(current, target_pose, target_house_id, grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM))
            plan_log = {
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "target_pose": target_pose,
                "current_pose": current,
                "replan_count": replan_count,
                "plan": plan,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            self.append_jsonl(output_dir / "route4_navigation_plan.jsonl", plan_log)
            self.route4_log_event(output_dir, "navigation_plan", plan_log)
            if plan.get("status") != "ok":
                self.route4_hold(session, output_dir=output_dir, reason=str(plan.get("reason", "navigation_plan_failed")))
                return {
                    "status": "blocked",
                    "reason": str(plan.get("reason", "navigation_plan_failed")),
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "navigation_plan": plan,
                    "replan_count": replan_count,
                    "elapsed_s": round(time.time() - started_at, 3),
                }
            waypoints = [dict(item) for item in plan.get("waypoints", []) if isinstance(item, dict)] or [dict(target_pose)]
            waypoint_count = len(waypoints)
            blocked_for_replan = False
            for idx, waypoint in enumerate(waypoints, start=1):
                waypoint_pose = {
                    "x": float(waypoint.get("x", target_pose["x"])),
                    "y": float(waypoint.get("y", target_pose["y"])),
                    "z": float(waypoint.get("z", target_pose["z"])),
                    "yaw": float(waypoint.get("yaw", target_pose["yaw"])),
                }
                segment_config = dict(base_config)
                is_escape_waypoint = isinstance(waypoint.get("escape_from_obstacle"), dict)
                if is_escape_waypoint:
                    segment_config["reach_tol_cm"] = min(
                        float(segment_config["reach_tol_cm"]),
                        float(waypoint.get("strict_reach_tol_cm", LLM_ROUTE3_ESCAPE_REACH_TOL_CM) or LLM_ROUTE3_ESCAPE_REACH_TOL_CM),
                    )
                    segment_config["yaw_tol_deg"] = float(waypoint.get("strict_yaw_tol_deg", 180.0) or 180.0)
                elif idx < waypoint_count:
                    segment_config["reach_tol_cm"] = max(float(segment_config["reach_tol_cm"]), float(LLM_ROUTE3_NAV_SEGMENT_REACH_TOL_CM))
                    segment_config["yaw_tol_deg"] = 180.0
                result = self.route4_follow_navigation_waypoint_with_fusion(
                    session,
                    waypoint_pose,
                    output_dir=output_dir,
                    stage=stage,
                    facade=facade,
                    target_id=target_id,
                    target_house_id=target_house_id,
                    waypoint_index=idx,
                    waypoint_count=waypoint_count,
                    config=segment_config,
                    escape_obstacle=waypoint.get("escape_from_obstacle") if isinstance(waypoint.get("escape_from_obstacle"), dict) else None,
                )
                last_result = result
                if result.get("status") == "ok":
                    current = result.get("current_pose", waypoint_pose) if isinstance(result.get("current_pose"), dict) else waypoint_pose
                    continue
                if result.get("status") == "blocked" and replan_count < max_replans:
                    fresh = self.route3_current_pose(session)
                    current = fresh or (result.get("current_pose", current) if isinstance(result.get("current_pose"), dict) else current)
                    replan_count += 1
                    blocked_for_replan = True
                    break
                result["navigation_plan"] = plan
                result["replan_count"] = replan_count
                result["elapsed_s"] = round(time.time() - started_at, 3)
                return result
            if blocked_for_replan:
                continue
            return {
                "status": "ok",
                "reason": "target_reached",
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "navigation_plan": plan,
                "replan_count": replan_count,
                "waypoint_count": waypoint_count,
                "pose_error": last_result.get("pose_error", {}),
                "elapsed_s": round(time.time() - started_at, 3),
            }
        self.route4_hold(session, output_dir=output_dir, reason="replan_exhausted")
        return {"status": "blocked", "reason": "replan_exhausted", "last_result": last_result, "replan_count": replan_count}

    def route4_run_summary(self, output_dir: Path, *, status: str) -> Dict[str, Any]:
        capture_rows = self.read_jsonl_artifact(output_dir / "lidar_capture_log.jsonl")
        avoidance_rows = self.read_jsonl_artifact(output_dir / "avoidance_events.jsonl")
        attempted_facades = set(self.llm_route4_completed_facades) | set(self.llm_route4_blocked_facades)
        summary = {
            "mode": "route4_llm_route_avoidance_fusion",
            "status": status,
            "target_house_id": str(self.llm_route4_state.get("target_house_id", "") or ""),
            "output_dir": str(output_dir),
            "completed_facades": sorted(self.llm_route4_completed_facades),
            "blocked_facades": sorted(self.llm_route4_blocked_facades),
            "attempted_facades": sorted(attempted_facades),
            "capture_count": len(capture_rows),
            "avoidance_event_count": len(avoidance_rows),
            "collision_count": sum(1 for item in avoidance_rows if bool(item.get("collision_state", False))),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(output_dir / "route4_fusion_summary.json", summary)
        self.route4_write_avoidance_summary(output_dir, status=status)
        return summary

    def route4_full_search_worker(self, session: flight.DroneFlightSession, *, single_facade: bool = False, force_new: bool = False) -> None:
        self.ensure_route4_state()
        selected_target_house_id = self.selected_route_target_house_id()
        if not selected_target_house_id:
            self.root.after(0, lambda: self.llm_route4_status_var.set("LLM Route V4: select a target house first."))
            return
        output_dir: Optional[Path] = None
        status = "done"
        try:
            output_dir = self.route4_initialize_run(selected_target_house_id, force_new=force_new)
            self.route4_set_stage("TASK_ANALYSIS", output_dir=output_dir, message=f"analyzing task for selected house={selected_target_house_id}")
            task_plan = self.route4_analyze_task_plan(selected_target_house_id, output_dir=output_dir)
            target_house_id = str(task_plan.get("target_house_id", selected_target_house_id) or selected_target_house_id)
            self.route4_update_state(output_dir=str(output_dir), target_house_id=target_house_id)
            self.route4_set_stage("PLAN_4_FACADES", output_dir=output_dir, message=f"planning facade candidates for house={target_house_id}")
            self.route4_set_control_lock(True)
            while not self.llm_route4_stop_event.is_set():
                completed = set(self.llm_route4_completed_facades)
                blocked = set(self.llm_route4_blocked_facades)
                if len(completed | blocked) >= 4:
                    break
                facade_candidates = self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False)
                ranked_candidates = self.route4_rank_observation_candidates(target_house_id, facade_candidates, completed, blocked, start_pose=self.route3_current_pose(session))
                self.route4_update_state(ranked_facade_candidates=ranked_candidates)
                decision = self.route4_decide_next_facade(target_house_id, ranked_candidates, completed, blocked)
                self.route4_log_event(output_dir, "high_level_decision", decision)
                if decision.get("next_action") == "done" or decision.get("stop_condition_met"):
                    break
                facade = str(decision.get("target_facade", "") or "").strip().lower()
                selected = next((item for item in ranked_candidates if str(item.get("facade", "") or "") == facade), {})
                if not facade or not selected:
                    status = "blocked_no_facade"
                    break
                self.route4_set_stage("SELECT_NEXT_FACADE", output_dir=output_dir, facade=facade, message=f"selected facade={facade}")
                self.llm_route2_state = {"target_house_id": target_house_id, "output_dir": str(output_dir)}
                facade_dir = self.route2_facade_dir(output_dir, target_house_id, facade)
                observation_attempts = self.route3_ordered_observation_attempts(selected)
                observation: Dict[str, Any] = {}
                obs_pose: Dict[str, float] = {}
                nav_result: Dict[str, Any] = {}
                for attempt_index, attempt in enumerate(observation_attempts, start=1):
                    if self.llm_route4_stop_event.is_set():
                        break
                    attempt_status = str(attempt.get("status", "") or "")
                    if attempt_status == "blocked" and str(attempt.get("route4_navigation_status", attempt.get("route3_navigation_status", "")) or "") != "ok":
                        self.route4_log_event(output_dir, "observation_attempt_skipped", {"facade": facade, "attempt_index": attempt_index, "attempt": attempt})
                        continue
                    self.apply_route2_observation_plan(target_house_id, attempt, ranked_candidates, status_label=f"v4 selected attempt {attempt_index}/{len(observation_attempts)}")
                    observation = self.route2_selected_state().get("observation_point", attempt)
                    obs_pose = self.route3_target_pose_from_point(observation)
                    self.route4_update_state(
                        selected_observation_attempt=observation,
                        current_exploration_status={
                            "stage": "NAV_TO_OBS",
                            "facade": facade,
                            "observation_attempt_index": attempt_index,
                            "observation_attempt_count": len(observation_attempts),
                            "target_pose": obs_pose,
                        },
                    )
                    self.route4_set_stage("NAV_TO_OBS", output_dir=output_dir, facade=facade, target=obs_pose, message=f"navigating to {facade} observation attempt {attempt_index}/{len(observation_attempts)}")
                    nav_result = self.route4_navigate_to_pose_with_fusion(
                        session,
                        obs_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_OBS",
                        facade=facade,
                        target_id=f"{target_house_id}_{facade}_obs_attempt_{attempt_index}",
                        target_house_id=target_house_id,
                    )
                    self.route4_log_event(output_dir, "observation_attempt_navigation_result", {"facade": facade, "attempt_index": attempt_index, "navigation": nav_result})
                    if nav_result.get("status") == "ok":
                        break
                if nav_result.get("status") != "ok":
                    self.llm_route4_blocked_facades.add(facade)
                    blocked_reasons = dict(self.llm_route4_state.get("blocked_facade_reasons", {}) if isinstance(self.llm_route4_state.get("blocked_facade_reasons"), dict) else {})
                    blocked_reasons[facade] = {"reason": str(nav_result.get("reason", "observation_navigation_failed") or "observation_navigation_failed"), "last_navigation_result": nav_result}
                    self.route4_update_state(blocked_facades=sorted(self.llm_route4_blocked_facades), blocked_facade_reasons=blocked_reasons)
                    self.route4_write_state_artifact()
                    if single_facade:
                        break
                    continue
                self.route4_set_stage("CAPTURE_RGB", output_dir=output_dir, facade=facade, target=obs_pose, message=f"capturing {facade} RGB")
                rgb_result = self.route3_capture_facade_rgb_current(session, output_dir=output_dir, facade_dir=facade_dir, house_id=target_house_id, facade=facade, planned_pose=obs_pose)
                self.route4_log_event(output_dir, "facade_rgb_capture", rgb_result)
                self.root.after(0, self.refresh_route4_support_views)
                self.route4_set_stage("ANALYZE_VLM", output_dir=output_dir, facade=facade, message=f"analyzing {facade} facade")
                self.route2_analyze_facade_vlm_worker()
                self.root.after(0, self.refresh_route4_support_views)
                self.route4_set_stage("PLAN_SCAN", output_dir=output_dir, facade=facade, message=f"planning {facade} scan")
                plan_result = self.route3_plan_facade_scan_current()
                points = [point for point in plan_result.get("points", []) if isinstance(point, dict)]
                self.route4_log_event(output_dir, "facade_scan_plan", {"facade": facade, "point_count": len(points), "validation": plan_result.get("validation", {})})
                if not points:
                    self.llm_route4_blocked_facades.add(facade)
                    self.route4_update_state(blocked_facades=sorted(self.llm_route4_blocked_facades))
                    if single_facade:
                        break
                    continue
                total = len(points)
                for idx, point in enumerate(points, start=1):
                    if self.llm_route4_stop_event.is_set():
                        break
                    scan_id = str(point.get("scan_id", "") or f"{facade}_{idx}")
                    target_pose = self.route3_target_pose_from_point(point)
                    self.route4_update_state(
                        current_exploration_status={
                            "stage": "NAV_TO_SCAN_POINT",
                            "facade": facade,
                            "point_index": idx,
                            "point_total": total,
                            "scan_id": scan_id,
                            "target_pose": target_pose,
                        }
                    )
                    self.route4_set_stage("NAV_TO_SCAN_POINT", output_dir=output_dir, facade=facade, target=target_pose, message=f"scan {idx}/{total} {scan_id}")
                    nav_scan = self.route4_navigate_to_pose_with_fusion(
                        session,
                        target_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_SCAN_POINT",
                        facade=facade,
                        target_id=scan_id,
                        target_house_id=target_house_id,
                    )
                    self.route4_log_event(output_dir, "scan_navigation_result", nav_scan)
                    if nav_scan.get("status") != "ok":
                        point["status"] = "blocked"
                        point["block_reason"] = nav_scan.get("reason", "navigation_failed")
                        continue
                    self.route4_set_stage("CAPTURE_SCAN", output_dir=output_dir, facade=facade, target=target_pose, message=f"capturing {scan_id}")
                    capture = self.route3_capture_scan_point_current(session, output_dir=output_dir, facade_dir=facade_dir, point=point, planned_pose=target_pose)
                    self.route4_log_event(output_dir, "scan_capture", {"scan_id": scan_id, **capture})
                    progress = 100.0 * (len(self.llm_route4_completed_facades) + (idx / max(1, total))) / 4.0
                    self.root.after(0, lambda v=progress: self.llm_route4_progress_var.set(max(0.0, min(100.0, v))))
                    self.root.after(0, lambda i=idx, t=total, f=facade: self.llm_route4_progress_text_var.set(f"Fusion: {f} {i}/{t}"))
                    self.root.after(0, self.refresh_route4_support_views)
                self.route2_write_lidar_summary(output_dir, running=False)
                self.route4_set_stage("VALIDATE_FACADE", output_dir=output_dir, facade=facade, message=f"validating {facade}")
                validation = self.route2_validate_facade()
                self.route4_log_event(output_dir, "facade_validation", validation)
                self.llm_route4_completed_facades.add(facade)
                self.route4_update_state(completed_facades=sorted(self.llm_route4_completed_facades), blocked_facades=sorted(self.llm_route4_blocked_facades))
                self.route4_write_state_artifact()
                self.root.after(0, self.refresh_route4_support_views)
                if single_facade:
                    status = "single_facade_complete"
                    break
                self.route4_set_stage("DECIDE_NEXT", output_dir=output_dir, facade=facade, message=f"{facade} complete; deciding next facade")
            if self.llm_route4_stop_event.is_set():
                status = "stopped"
            if status == "done" and self.llm_route4_blocked_facades:
                status = "done_with_blocked"
            final_stage = "DONE" if status == "done" else ("DONE_WITH_BLOCKED" if status == "done_with_blocked" else "DECIDE_NEXT")
            self.route4_set_stage(final_stage, output_dir=output_dir, message=f"fused autosearch {status}")
            summary = self.route4_run_summary(output_dir, status=status)
            self.route4_log_event(output_dir, "summary", summary)
            self.root.after(0, lambda s=status, d=output_dir: self.llm_route4_status_var.set(f"LLM Route V4: {s} -> {d}"))
            self.root.after(0, self.refresh_route4_support_views)
        except Exception as exc:
            LOGGER.warning("Route V4 fused autosearch failed: %s", exc)
            if output_dir is not None:
                self.route4_log_event(output_dir, "error", {"reason": str(exc)})
                self.route4_run_summary(output_dir, status="failed")
            self.root.after(0, lambda e=exc: self.llm_route4_status_var.set(f"LLM Route V4: failed: {e}"))
        finally:
            self.route4_set_control_lock(False)

    def refresh_route4_preview(self) -> None:
        preview_text = getattr(self, "llm_route4_preview_text", None)
        if preview_text is None:
            return
        payload = {
            "fusion_state": self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {},
            "active_facade_state": self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {},
        }
        try:
            if not preview_text.winfo_exists():
                self.llm_route4_preview_text = None
                return
            preview_text.configure(state="normal")
            preview_text.delete("1.0", "end")
            preview_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            preview_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route4_preview_text = None

    def refresh_route4_analysis_view(self) -> None:
        analysis_text = getattr(self, "llm_route4_analysis_text", None)
        if analysis_text is None:
            return
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        obstacle_event = self.llm_route4_state.get("last_obstacle_event", {}) if isinstance(self.llm_route4_state, dict) else {}
        payload = {"facade_analysis": analysis or {"status": "No facade analysis yet."}, "last_obstacle_event": obstacle_event}
        try:
            if not analysis_text.winfo_exists():
                self.llm_route4_analysis_text = None
                return
            analysis_text.configure(state="normal")
            analysis_text.delete("1.0", "end")
            analysis_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            analysis_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route4_analysis_text = None

    def refresh_route4_rgb_display(self) -> None:
        widget = getattr(self, "llm_route4_rgb_label", None)
        if widget is None:
            return
        image_path = self.route2_current_rgb_path()
        if image_path is None:
            try:
                self.route2_draw_rgb_preview_message(widget, "No facade RGB")
                self.llm_route4_rgb_photo = None
            except tk.TclError:
                self.llm_route4_rgb_label = None
            return
        try:
            image = Image.open(image_path).convert("RGB")
            photo = ImageTk.PhotoImage(self.route2_rgb_preview_image(image, widget))
            self.route2_draw_rgb_preview_photo(widget, photo)
            self.llm_route4_rgb_photo = photo
        except Exception as exc:
            LOGGER.warning("Refresh route v4 facade RGB failed: %s", exc)
            try:
                self.route2_draw_rgb_preview_message(widget, f"RGB preview failed: {exc}")
                self.llm_route4_rgb_photo = None
            except tk.TclError:
                self.llm_route4_rgb_label = None

    def refresh_llm_route4_map(self) -> None:
        self.ensure_route4_state()
        widget = getattr(self, "llm_route4_map_widget", None)
        if widget is None:
            return
        try:
            if not self.load_map_resources(force=not bool(self.map_config)):
                self.llm_route4_map_status_var.set("Route V4 Map: map unavailable")
                return
            pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
            pose_x = float(pose.get("x", 0.0)) if pose else 0.0
            pose_y = float(pose.get("y", 0.0)) if pose else 0.0
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0))) if pose else 0.0
            houses, boxes = self.build_map_display(pose)
            widget.set_background_image(self.map_image)
            widget.set_calibration(
                self.map_calibration.get("affine_world_to_image"),
                self.map_image_size(),
                [],
                self.map_calibration.get("homography_world_to_image"),
            )
            widget.set_image_layer_offset(*self.map_display_offset_px)
            widget.set_house_boxes(boxes)
            widget.update_houses([])
            widget.update_uav(pose_x, pose_y, pose_yaw)
            route_points: List[Dict[str, Any]] = []
            state = self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {}
            route4_state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
            candidates_for_map = route4_state.get("ranked_facade_candidates", [])
            for candidate in candidates_for_map if isinstance(candidates_for_map, list) else []:
                if isinstance(candidate, dict):
                    selected_attempt = candidate.get("selected_observation_attempt", {}) if isinstance(candidate.get("selected_observation_attempt"), dict) else {}
                    item = dict(selected_attempt or candidate)
                    item["label"] = str(item.get("label", "") or f"{item.get('facade', '')}_obs")
                    item["route_point_type"] = "observation_point"
                    route_points.append(item)
            for point in state.get("facade_scan_points", []) if isinstance(state.get("facade_scan_points"), list) else []:
                if isinstance(point, dict):
                    item = dict(point)
                    item["label"] = str(item.get("scan_id", "") or f"scan_{len(route_points)}")
                    item["route_point_type"] = "scan_point"
                    route_points.append(item)
            widget.set_route_plan({"route_points": route_points})
            output_dir = self.route4_state_output_dir()
            if output_dir is not None:
                trace_rows = self.read_jsonl_artifact(output_dir / "route4_movement_trace.jsonl")
                trajectory = [row.get("current_pose", {}) for row in trace_rows[-400:] if isinstance(row.get("current_pose"), dict)]
                widget.set_trajectory(trajectory)
            self.llm_route4_map_status_var.set(
                f"Route V4 Map: houses={len(houses)} route_points={len(route_points)} completed={len(self.llm_route4_completed_facades)}/4"
            )
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM route v4 map failed: %s", exc)
            self.llm_route4_map_status_var.set(f"Route V4 Map: failed: {exc}")

    def refresh_route4_support_views(self) -> None:
        self.ensure_route4_state()
        completed = len(getattr(self, "llm_route4_completed_facades", set()) or set())
        blocked = len(getattr(self, "llm_route4_blocked_facades", set()) or set())
        progress = 100.0 * float(min(4, completed + blocked)) / 4.0
        state = self.llm_route4_state if isinstance(getattr(self, "llm_route4_state", None), dict) else {}
        current_status = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status"), dict) else {}
        try:
            self.llm_route4_progress_var.set(max(0.0, min(100.0, progress)))
            self.llm_route4_progress_text_var.set(f"Fusion: completed={completed} blocked={blocked}")
            if current_status:
                facade = str(current_status.get("facade", state.get("current_facade", "")) or state.get("current_facade", "") or "-")
                stage = str(current_status.get("stage", state.get("stage", "")) or state.get("stage", "") or "-")
                point_index = current_status.get("point_index")
                point_total = current_status.get("point_total")
                suffix = f" {point_index}/{point_total}" if point_index is not None and point_total is not None else ""
                self.llm_route4_current_status_var.set(f"Current: {stage} {facade}{suffix}")
            else:
                self.llm_route4_current_status_var.set(f"Current: {state.get('stage', 'idle')} {state.get('current_facade', '')}")
            next_status = state.get("next_exploration_status", {}) if isinstance(state.get("next_exploration_status"), dict) else {}
            next_facade = str(next_status.get("facade", next_status.get("target_facade", "")) or "")
            self.llm_route4_next_status_var.set(f"Next: {next_facade}" if next_facade else "Next: n/a")
        except tk.TclError:
            pass
        self.refresh_route4_preview()
        self.refresh_route4_analysis_view()
        self.refresh_route4_rgb_display()
        self.refresh_llm_route4_map()

    def _build_llm_route4_section(self, parent: tk.Misc) -> tk.LabelFrame:
        self.ensure_route4_state()
        route = tk.LabelFrame(parent, text="LLM House Entrance Route V4 Fused Route + Avoidance")
        for col in (1, 3, 5):
            route.grid_columnconfigure(col, weight=1)
        route.grid_rowconfigure(9, weight=1)
        tk.Label(route, text="Target House").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        combo = ttk.Combobox(route, textvariable=self.llm_route_target_var, values=list(self.house_choice_map.keys()), state="readonly", width=22)
        combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        if combo not in self.house_target_combos:
            self.house_target_combos.append(combo)
        tk.Label(route, text="Task").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_task_text_var).grid(row=0, column=3, columnspan=3, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="API").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        api_combo = ttk.Combobox(route, textvariable=self.llm_api_style_var, values=LLM_API_STYLE_OPTIONS, state="readonly", width=17)
        api_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        api_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_llm_api_defaults(force=False))
        tk.Label(route, text="Base URL").grid(row=1, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_base_url_var).grid(row=1, column=3, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="Model").grid(row=1, column=4, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_model_var, width=18).grid(row=1, column=5, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="API Key").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_api_key_var, show="*").grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(route, text="Timeout s").grid(row=2, column=2, sticky="w", padx=6, pady=6)
        tk.Entry(route, textvariable=self.llm_timeout_s_var, width=8).grid(row=2, column=3, sticky="w", padx=6, pady=6)
        tk.Label(route, textvariable=self.llm_route4_active_var, anchor="w").grid(row=2, column=4, columnspan=2, sticky="ew", padx=6, pady=6)

        rep = tk.Frame(route)
        rep.grid(row=3, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(rep, text="Representation Model").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(rep, textvariable=self.llm_route4_representation_model_var, width=54).pack(side="left", fill="x", expand=True, padx=(0, 6), pady=4)
        tk.Button(rep, text="Default", command=lambda: self.llm_route4_representation_model_var.set(str(self.default_route4_representation_model_path()))).pack(side="left", padx=6, pady=4)
        tk.Label(rep, text="Sense interval s").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(rep, textvariable=self.llm_route4_sensing_interval_s_var, width=7).pack(side="left", padx=(0, 8), pady=4)

        config = tk.Frame(route)
        config.grid(row=4, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        tk.Label(config, text="Floor height m").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_floor_height_m_var, width=6).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Default floors").pack(side="left", padx=(6, 2), pady=4)
        tk.Entry(config, textvariable=self.llm_route2_default_floors_var, width=5).pack(side="left", padx=(0, 8), pady=4)
        tk.Label(config, text="Density mode").pack(side="left", padx=(6, 2), pady=4)
        ttk.Combobox(config, textvariable=self.llm_route2_density_mode_var, values=("auto", "high", "medium", "low"), state="readonly", width=8).pack(side="left", padx=(0, 8), pady=4)

        nav = tk.Frame(route)
        nav.grid(row=5, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 2))
        for label, var, width in (
            ("Move tick ms", self.llm_route4_move_tick_ms_var, 6),
            ("Nav step cm", self.llm_route4_nav_step_cm_var, 6),
            ("Reach tol cm", self.llm_route4_reach_tol_cm_var, 6),
            ("Z tol cm", self.llm_route4_z_tol_cm_var, 6),
            ("Yaw tol deg", self.llm_route4_yaw_tol_deg_var, 6),
            ("Max stage s", self.llm_route4_max_stage_s_var, 6),
        ):
            tk.Label(nav, text=label).pack(side="left", padx=(6, 2), pady=4)
            tk.Entry(nav, textvariable=var, width=width).pack(side="left", padx=(0, 6), pady=4)

        actions = tk.Frame(route)
        actions.grid(row=6, column=0, columnspan=6, sticky="ew", padx=0, pady=(0, 4))
        tk.Button(actions, text="Start Fused Route+Avoidance", command=self.on_route4_start_fused_search).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Step Facade", command=self.on_route4_step_facade).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Pause/Resume", command=self.on_route4_toggle_pause).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Stop", command=self.on_route4_stop).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Clear", command=self.on_route4_clear).pack(side="left", padx=6, pady=4)
        tk.Button(actions, text="Validate Run", command=self.on_route4_validate_run).pack(side="left", padx=6, pady=4)

        status = tk.Frame(route)
        status.grid(row=7, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        status.grid_columnconfigure(1, weight=1)
        tk.Label(status, textvariable=self.llm_route4_stage_var, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 12))
        tk.Label(status, textvariable=self.llm_route4_target_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=(0, 12))
        tk.Label(status, textvariable=self.llm_route4_error_var, anchor="w").grid(row=0, column=2, sticky="w")
        tk.Label(status, textvariable=self.llm_route4_payload_var, anchor="w").grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(status, textvariable=self.llm_route4_avoidance_status_var, anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(status, textvariable=self.llm_route4_representation_status_var, anchor="w").grid(row=3, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        tk.Label(route, textvariable=self.llm_route4_status_var, anchor="w").grid(row=8, column=0, columnspan=6, sticky="ew", padx=6, pady=(0, 4))
        preview_frame = tk.Frame(route)
        preview_frame.grid(row=9, column=0, columnspan=6, sticky="nsew", padx=6, pady=(0, 6))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_text = tk.Text(preview_frame, height=7, width=96, wrap="none", font=("Consolas", 9))
        preview_y = tk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
        preview_x = tk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
        preview_text.configure(yscrollcommand=preview_y.set, xscrollcommand=preview_x.set)
        preview_text.grid(row=0, column=0, sticky="nsew")
        preview_y.grid(row=0, column=1, sticky="ns")
        preview_x.grid(row=1, column=0, sticky="ew")
        preview_text.configure(state="disabled")
        self.llm_route4_preview_text = preview_text
        setattr(route, "_llm_route4_combo", combo)
        return route

    def open_llm_route_window4(self) -> None:
        self.ensure_route4_state()
        if self.llm_route4_window is not None and self.llm_route4_window.winfo_exists():
            self.llm_route4_window.lift()
            self.llm_route4_window.focus_force()
            return
        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route V4 Fused Route + Avoidance")
        window.geometry("980x760")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        content = tk.Frame(window)
        content.grid(row=0, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(2, weight=1)

        route = self._build_llm_route4_section(content)
        route.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        support = tk.Frame(content)
        support.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 4))
        support.grid_columnconfigure(0, weight=0)
        support.grid_columnconfigure(1, weight=1)
        rgb_frame = tk.LabelFrame(support, text="Facade RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        self.llm_route4_rgb_label = tk.Canvas(rgb_frame, width=330, height=220, bg="#202020", highlightthickness=0)
        self.llm_route4_rgb_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route4_rgb_label.bind("<Configure>", lambda _event: self.refresh_route4_rgb_display(), add="+")
        analysis_frame = tk.LabelFrame(support, text="Fusion Analysis")
        analysis_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_rowconfigure(0, weight=1)
        analysis_text = tk.Text(analysis_frame, height=10, width=64, wrap="none", font=("Consolas", 9))
        analysis_y = tk.Scrollbar(analysis_frame, orient="vertical", command=analysis_text.yview)
        analysis_x = tk.Scrollbar(analysis_frame, orient="horizontal", command=analysis_text.xview)
        analysis_text.configure(yscrollcommand=analysis_y.set, xscrollcommand=analysis_x.set)
        analysis_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        analysis_y.grid(row=0, column=1, sticky="ns", pady=(6, 0))
        analysis_x.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
        analysis_text.configure(state="disabled")
        self.llm_route4_analysis_text = analysis_text

        map_frame = tk.LabelFrame(content, text="Map Facade V4 Fusion")
        map_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(1, weight=1)
        map_toolbar = tk.Frame(map_frame)
        map_toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        tk.Label(map_toolbar, textvariable=self.llm_route4_map_status_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(map_toolbar, textvariable=self.llm_route4_current_status_var, anchor="w").pack(side="left", padx=(8, 2))
        tk.Label(map_toolbar, textvariable=self.llm_route4_next_status_var, anchor="w").pack(side="left", padx=(8, 2))
        tk.Label(map_toolbar, textvariable=self.llm_route4_progress_text_var, anchor="e").pack(side="left", padx=(8, 2))
        ttk.Progressbar(map_toolbar, variable=self.llm_route4_progress_var, maximum=100.0, length=150, mode="determinate").pack(side="left", padx=(0, 8))
        tk.Button(map_toolbar, text="Refresh Map", command=self.refresh_llm_route4_map).pack(side="right", padx=6)
        self.load_map_resources(force=True)
        self.llm_route4_map_widget = OverheadMapWidget(map_frame, world_bounds=self.map_world_bounds, canvas_w=800, canvas_h=320)
        self.llm_route4_map_widget.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route4_window = window

        def close_window() -> None:
            combo = getattr(route, "_llm_route4_combo", None)
            if combo is not None and combo in self.house_target_combos:
                self.house_target_combos.remove(combo)
            self.llm_route4_window = None
            self.llm_route4_map_widget = None
            self.llm_route4_preview_text = None
            self.llm_route4_analysis_text = None
            self.llm_route4_rgb_label = None
            self.llm_route4_rgb_photo = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.refresh_route4_support_views()

    def on_route4_start_fused_search(self) -> None:
        self.ensure_route4_state()
        session = self.active_session()
        if session is None:
            return
        if self.llm_route4_thread is not None and self.llm_route4_thread.is_alive():
            self.llm_route4_status_var.set("LLM Route V4: already running.")
            return
        self.llm_route4_stop_event.clear()
        self.llm_route4_pause_event.clear()
        self.llm_route4_paused_var.set(False)
        self.llm_route4_thread = threading.Thread(target=lambda: self.route4_full_search_worker(session, single_facade=False, force_new=True), daemon=True)
        self.llm_route4_thread.start()

    def on_route4_step_facade(self) -> None:
        self.ensure_route4_state()
        session = self.active_session()
        if session is None:
            return
        if self.llm_route4_thread is not None and self.llm_route4_thread.is_alive():
            self.llm_route4_status_var.set("LLM Route V4: wait for current worker.")
            return
        self.llm_route4_stop_event.clear()
        self.llm_route4_pause_event.clear()
        self.llm_route4_thread = threading.Thread(target=lambda: self.route4_full_search_worker(session, single_facade=True, force_new=False), daemon=True)
        self.llm_route4_thread.start()

    def on_route4_toggle_pause(self) -> None:
        self.ensure_route4_state()
        if self.llm_route4_pause_event.is_set():
            self.llm_route4_pause_event.clear()
            self.llm_route4_paused_var.set(False)
            self.llm_route4_status_var.set("LLM Route V4: resumed.")
        else:
            self.llm_route4_pause_event.set()
            self.llm_route4_paused_var.set(True)
            session = self.active_session()
            if session is not None:
                self.route4_hold(session, output_dir=self.route4_state_output_dir(), reason="pause_button")
            self.llm_route4_status_var.set("LLM Route V4: paused.")

    def on_route4_stop(self) -> None:
        self.ensure_route4_state()
        self.llm_route4_stop_event.set()
        self.llm_route4_pause_event.clear()
        session = self.active_session()
        if session is not None:
            self.route4_hold(session, output_dir=self.route4_state_output_dir(), reason="stop_button")
        self.llm_route4_status_var.set("LLM Route V4: stop requested.")

    def on_route4_clear(self) -> None:
        self.ensure_route4_state()
        if self.llm_route4_thread is not None and self.llm_route4_thread.is_alive():
            self.llm_route4_status_var.set("LLM Route V4: stop before clearing.")
            return
        self.llm_route4_state = {}
        self.llm_route4_completed_facades = set()
        self.llm_route4_blocked_facades = set()
        self.llm_route4_stage_var.set("Stage: idle")
        self.llm_route4_active_var.set("Active: n/a")
        self.llm_route4_target_var.set("Target: n/a")
        self.llm_route4_error_var.set("Error: n/a")
        self.llm_route4_payload_var.set("Payload: hold")
        self.llm_route4_current_status_var.set("Current: idle")
        self.llm_route4_next_status_var.set("Next: n/a")
        self.llm_route4_avoidance_status_var.set("Avoidance: idle")
        self.llm_route4_representation_status_var.set("Representation: idle")
        self.llm_route4_status_var.set("LLM Route V4: cleared.")
        self.refresh_route4_support_views()

    def on_route4_validate_run(self) -> None:
        self.ensure_route4_state()
        output_dir = self.route4_state_output_dir()
        if output_dir is None:
            self.llm_route4_status_var.set("LLM Route V4: no run to validate.")
            return
        summary = self.route4_run_summary(output_dir, status=str(self.llm_route4_state.get("stage", "manual_validate") or "manual_validate"))
        self.llm_route4_status_var.set(f"LLM Route V4: run summary -> {output_dir / 'route4_fusion_summary.json'}")
        self.route4_log_event(output_dir, "manual_validate_run", summary)
