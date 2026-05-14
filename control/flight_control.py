from __future__ import annotations

from .common import *


class FlightControlMixin:
    def on_start_session(self) -> None:
        if self.session is not None and self.session.started:
            self.status_var.set("Session already started.")
            return

        def worker() -> Dict[str, Any]:
            session_args = self.build_flight_args()
            self.session = flight.DroneFlightSession(session_args)
            return self.session.start()

        self.call_async("Starting run_drone_flight", worker)

    def on_stop_session(self) -> None:
        self.stop_keyboard_control(send_hold=False)
        self.stream_capture_stop_event.set()
        self.lidar_stream_capture_stop_event.set()
        obstacle_stop_event = getattr(self, "obstacle_avoidance_stop_event", None)
        if obstacle_stop_event is not None:
            obstacle_stop_event.set()
        self.route_stop_event.set()
        route3_stop_event = getattr(self, "llm_route3_stop_event", None)
        if route3_stop_event is not None:
            route3_stop_event.set()
        route3_pause_event = getattr(self, "llm_route3_pause_event", None)
        if route3_pause_event is not None:
            route3_pause_event.clear()
        session = self.session
        if session is None:
            try:
                kill_args = self.build_flight_args()
                flight.configure_local_unreal_env(kill_args)
                flight.force_kill_unreal_processes(kill_args)
                self.status_var.set("No active session; Unreal processes killed.")
            except Exception as exc:
                self.status_var.set(f"No active session; kill failed: {exc}")
            return

        def worker() -> Dict[str, Any]:
            session.close(force_kill_unreal=True)
            self.session = None
            return {"status": "ok", "message": "Session stopped; Unreal processes killed", "started": False}

        self.call_async("Stopping session", worker)

    def refresh_state_once(self) -> None:
        session = self.session
        if session is None or not session.started:
            return
        if self.state_refresh_inflight:
            return

        def worker() -> None:
            self.state_refresh_inflight = True
            try:
                state = self.safe("Refreshing state", session.get_state)
                if isinstance(state, dict):
                    self.root.after(0, lambda r=state: self.apply_state(r))
            finally:
                self.state_refresh_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def schedule_state_refresh(self) -> None:
        if not self.manual_request_inflight and not self.move_request_inflight:
            session = self.session
            if session is not None and session.started:
                self.refresh_state_once()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)

    def on_toggle_movement(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.movement_enabled_state:
            self.stop_keyboard_control(send_hold=True)
        self.call_async(
            "Toggling movement",
            lambda: session.set_movement_enabled(not self.movement_enabled_state),
        )

    def on_movement_mode_selected(self, _event: Optional[tk.Event] = None) -> None:
        self.movement_mode_state = self.movement_mode_var.get().strip() or flight.DEFAULT_MOVEMENT_MODE
        session = self.session
        if session is None or not session.started:
            self.status_var.set(f"Movement mode will apply on start: {self.movement_mode_state}")
            return
        self.call_async(
            "Changing movement mode",
            lambda: session.set_movement_mode(self.movement_mode_state),
        )

    def latest_yaw_deg(self) -> Optional[float]:
        pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        if not pose:
            return None
        try:
            return float(pose.get("task_yaw", pose.get("yaw")))
        except (TypeError, ValueError):
            return None

    def snap_yaw_target_for_symbol(self, symbol: str, current_yaw_deg: float) -> float:
        normalized = flight.normalize_angle_deg(float(current_yaw_deg))
        nearest_grid = round(normalized / YAW_GRID_STEP_DEG) * YAW_GRID_STEP_DEG
        direction = -1.0 if symbol.lower() == "q" else 1.0
        return flight.normalize_angle_deg(nearest_grid + direction * YAW_GRID_STEP_DEG)

    def move_payload_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        payload = MOVE_COMMANDS.get(symbol.lower())
        if payload is None:
            return None
        return dict(payload)

    def combined_move_payload_for_symbols(self, symbols: set[str]) -> Dict[str, Any]:
        active = {symbol.lower() for symbol in symbols if symbol.lower() in MOVE_COMMANDS}
        if "x" in active:
            return dict(MOVE_COMMANDS["x"])

        forward_axis = float("w" in active) - float("s" in active)
        right_axis = float("d" in active) - float("a" in active)
        up_axis = float("r" in active) - float("f" in active)
        yaw_axis = float("e" in active) - float("q" in active)

        labels: List[str] = []
        if forward_axis > 0:
            labels.append("forward")
        elif forward_axis < 0:
            labels.append("backward")
        if right_axis > 0:
            labels.append("right")
        elif right_axis < 0:
            labels.append("left")
        if up_axis > 0:
            labels.append("up")
        elif up_axis < 0:
            labels.append("down")
        if yaw_axis > 0:
            labels.append("yaw_right")
        elif yaw_axis < 0:
            labels.append("yaw_left")

        return {
            "forward_cm": 20.0 * forward_axis,
            "right_cm": 20.0 * right_axis,
            "up_cm": 20.0 * up_axis,
            "yaw_delta_deg": 30.0 * yaw_axis,
            "action_name": "keys_" + "_".join(labels) if labels else "hold",
        }

    def ordered_keyboard_symbols(self) -> List[str]:
        ordered = [symbol for symbol in KEYBOARD_SYMBOL_ORDER if symbol in self.keyboard_pressed_symbols]
        extra = sorted(symbol for symbol in self.keyboard_pressed_symbols if symbol not in KEYBOARD_SYMBOL_ORDER)
        return ordered + extra

    def build_stream_action_detail(self) -> Dict[str, Any]:
        pressed = self.ordered_keyboard_symbols()
        if pressed:
            source = "keyboard"
            combined_payload = self.combined_move_payload_for_symbols(set(pressed))
        elif self.sequence_thread is not None and self.sequence_thread.is_alive():
            source = "sequence"
            combined_payload = {}
        elif self.route_thread is not None and self.route_thread.is_alive():
            source = "route"
            combined_payload = {}
        else:
            source = "session"
            combined_payload = {}
        return {
            "source": source,
            "pressed_keys": pressed,
            "combined_move_payload": combined_payload,
            "keyboard_enabled": bool(self.keyboard_enabled_var.get()),
            "keyboard_interval_ms": self.keyboard_interval_ms(),
            "movement_mode": self.movement_mode_state,
            "movement_enabled": bool(self.movement_enabled_state),
            "move_request_inflight": bool(self.move_request_inflight),
            "keyboard_request_inflight": bool(self.keyboard_request_inflight),
            "sequence_running": bool(self.sequence_thread is not None and self.sequence_thread.is_alive()),
            "route_running": bool(self.route_thread is not None and self.route_thread.is_alive()),
            "latest_state_action": str(self.latest_state.get("last_action", "")),
            "latest_state_step": int(self.latest_state.get("step_count", 0) or 0),
        }

    def keyboard_interval_ms(self) -> int:
        try:
            value = int(float(self.keyboard_interval_ms_var.get().strip()))
        except Exception:
            value = DEFAULT_KEYBOARD_INTERVAL_MS
        value = max(30, min(1000, value))
        if self.keyboard_interval_ms_var.get().strip() != str(value):
            self.root.after(0, lambda v=value: self.keyboard_interval_ms_var.set(str(v)))
        return value

    def update_keyboard_status(self, message: str = "") -> None:
        if message:
            self.keyboard_status_var.set(f"Keyboard: {message}")
            return
        ordered = [symbol for symbol in KEYBOARD_SYMBOL_ORDER if symbol in self.keyboard_pressed_symbols]
        extra = sorted(symbol for symbol in self.keyboard_pressed_symbols if symbol not in KEYBOARD_SYMBOL_ORDER)
        pressed = "+".join(ordered + extra)
        if pressed:
            self.keyboard_status_var.set(f"Keyboard: holding {pressed}")
        else:
            self.keyboard_status_var.set("Keyboard: idle")

    def cancel_keyboard_loop(self) -> None:
        after_id = self.keyboard_loop_after_id
        self.keyboard_loop_after_id = None
        if after_id is None:
            return
        try:
            self.root.after_cancel(after_id)
        except Exception:
            pass

    def stop_keyboard_control(self, *, send_hold: bool = False, force_hold: bool = False) -> None:
        had_pressed = bool(self.keyboard_pressed_symbols)
        self.keyboard_pressed_symbols.clear()
        self.cancel_keyboard_loop()
        self.update_keyboard_status()
        if send_hold and (had_pressed or force_hold) and not self.move_request_inflight and not self.keyboard_request_inflight:
            self.send_move_symbol("x")

    def on_keyboard_enabled_changed(self) -> None:
        if not self.keyboard_enabled_var.get():
            self.stop_keyboard_control(send_hold=True)
            self.update_keyboard_status("single key mode")
        else:
            self.update_keyboard_status()

    def start_keyboard_loop(self) -> None:
        if self.keyboard_loop_after_id is not None or self.keyboard_request_inflight:
            return
        self.keyboard_loop_after_id = self.root.after(0, self.keyboard_control_tick)

    def keyboard_control_tick(self) -> None:
        self.keyboard_loop_after_id = None
        if getattr(self, "llm_route3_control_locked", False):
            self.keyboard_pressed_symbols.clear()
            self.update_keyboard_status("locked by LLM Route V3")
            return
        if not self.keyboard_enabled_var.get():
            self.stop_keyboard_control(send_hold=True)
            return
        if not self.keyboard_pressed_symbols:
            self.update_keyboard_status()
            return
        session = self.session
        if session is None or not session.started:
            self.keyboard_pressed_symbols.clear()
            self.update_keyboard_status("start session first")
            return
        if not self.movement_enabled_state:
            self.keyboard_pressed_symbols.clear()
            self.update_keyboard_status("enable Basic Movement first")
            return
        if self.manual_request_inflight or self.move_request_inflight:
            self.keyboard_loop_after_id = self.root.after(self.keyboard_interval_ms(), self.keyboard_control_tick)
            return

        payload = self.combined_move_payload_for_symbols(self.keyboard_pressed_symbols)
        interval_ms = self.keyboard_interval_ms()
        self.keyboard_request_inflight = True
        self.move_request_inflight = True
        self.update_keyboard_status()

        def worker() -> None:
            try:
                try:
                    session.args.control_dt = max(0.03, min(1.0, interval_ms / 1000.0))
                except Exception:
                    pass
                response = self.safe("Keyboard move", lambda: session.move_relative(payload))
                if isinstance(response, dict):
                    self.root.after(0, lambda r=response: self.apply_state(r))
                    status = str(response.get("status", "")).lower()
                    if status in {"error", "disabled"}:
                        self.root.after(0, self.stop_keyboard_control)
            finally:
                self.move_request_inflight = False
                self.keyboard_request_inflight = False
                self.root.after(0, self._after_keyboard_tick)

        threading.Thread(target=worker, daemon=True).start()

    def _after_keyboard_tick(self) -> None:
        if self.keyboard_pressed_symbols and self.keyboard_enabled_var.get():
            self.keyboard_loop_after_id = self.root.after(0, self.keyboard_control_tick)
        else:
            self.update_keyboard_status()

    def _execute_move_payload(
        self,
        payload: Dict[str, Any],
        label: str,
        *,
        from_sequence: bool = False,
        sequence_symbol: str = "",
    ) -> bool:
        session = self.active_session()
        if session is None:
            return False
        self.move_request_inflight = True
        try:
            response = self.safe(label, lambda: session.move_relative(payload))
            if isinstance(response, dict):
                self.root.after(0, lambda r=response: self.apply_state(r))
                if str(response.get("status", "")).lower() in {"error", "disabled"}:
                    return False
                if from_sequence:
                    self.root.after(0, lambda s=sequence_symbol or label: self.status_var.set(f"Sequence sent: {s}"))
                return True
            return False
        finally:
            self.move_request_inflight = False

    def _execute_move(self, symbol: str, *, from_sequence: bool = False) -> bool:
        payload = self.move_payload_for_symbol(symbol)
        if payload is None:
            return False
        return self._execute_move_payload(
            payload,
            f"Move {symbol}",
            from_sequence=from_sequence,
            sequence_symbol=symbol,
        )

    def send_move_symbol(self, symbol: str) -> None:
        if getattr(self, "llm_route3_control_locked", False):
            self.status_var.set(f"Move {symbol} ignored while LLM Route V3 controls movement.")
            return
        if self.move_request_inflight or self.keyboard_request_inflight:
            self.status_var.set(f"Move {symbol} ignored while another move is in flight.")
            return
        threading.Thread(target=lambda: self._execute_move(symbol), daemon=True).start()

    def on_execute_sequence(self) -> None:
        symbols = [symbol for symbol in self.sequence_var.get().strip().lower() if symbol in MOVE_COMMANDS]
        if not symbols:
            self.status_var.set("No valid sequence symbols.")
            return
        try:
            delay_s = max(0.0, float(self.sequence_delay_var.get().strip()) / 1000.0)
        except ValueError:
            self.status_var.set("Invalid sequence delay.")
            return
        if self.sequence_thread is not None and self.sequence_thread.is_alive():
            self.status_var.set("Sequence already running.")
            return

        def worker() -> None:
            self.sequence_stop_event.clear()
            total = len(symbols)
            for index, symbol in enumerate(symbols, start=1):
                if self.sequence_stop_event.is_set():
                    self.root.after(0, lambda i=index, t=total: self.status_var.set(f"Sequence stopped at {i - 1}/{t}."))
                    return
                self.root.after(0, lambda i=index, t=total, s=symbol: self.status_var.set(f"Sequence {i}/{t}: {s}"))
                if not self._execute_move(symbol, from_sequence=True):
                    self.root.after(0, lambda i=index, t=total: self.status_var.set(f"Sequence failed at {i}/{t}."))
                    return
                time.sleep(delay_s)
            self.root.after(0, lambda: self.status_var.set(f"Sequence completed: {total}/{total}"))

        self.sequence_thread = threading.Thread(target=worker, daemon=True)
        self.sequence_thread.start()

    def on_stop_sequence(self) -> None:
        self.sequence_stop_event.set()
        self.status_var.set("Stopping sequence...")

    def on_set_pose(self) -> None:
        try:
            payload = json.loads(self.pose_text.get("1.0", "end").strip())
        except json.JSONDecodeError as exc:
            self.status_var.set(f"Invalid pose JSON: {exc}")
            return
        session = self.active_session()
        if session is None:
            return
        self.call_async("Setting pose", lambda: session.set_pose(payload))

    def sync_capture_options_to_session(self, session: flight.DroneFlightSession) -> None:
        lidar_min_cm, lidar_max_cm = self.parse_lidar_depth_range()
        session.args.enhance_rgb = bool(self.enhance_rgb_var.get())
        session.args.rgb_enhance_gamma = float(self.args.rgb_enhance_gamma)
        session.args.rgb_enhance_gain = float(self.args.rgb_enhance_gain)
        session.args.rgb_source_order = self.rgb_source_order_var.get().strip() or flight.DEFAULT_RGB_SOURCE_ORDER
        session.args.temp_capture_dir = str(getattr(self.args, "temp_capture_dir", flight.DEFAULT_TEMP_CAPTURE_DIR))
        session.args.temp_capture_lidar_dir = str(getattr(self.args, "temp_capture_lidar_dir", flight.DEFAULT_TEMP_CAPTURE_LIDAR_DIR))
        session.args.stream_capture_dir = str(getattr(self.args, "stream_capture_dir", flight.DEFAULT_STREAM_CAPTURE_DIR))
        session.args.stream_capture_lidar_dir = str(getattr(self.args, "stream_capture_lidar_dir", flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR))
        session.args.depth_min_cm = float(getattr(self.args, "depth_min_cm", flight.DEFAULT_DEPTH_MIN_CM))
        session.args.depth_max_cm = float(getattr(self.args, "depth_max_cm", flight.DEFAULT_DEPTH_MAX_CM))
        session.args.lidar_depth_min_cm = lidar_min_cm
        session.args.lidar_depth_max_cm = lidar_max_cm
        session.args.lidar_depth_projection = str(
            getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)
        )
        session.args.lidar_capture_processing = flight.normalize_lidar_capture_processing(
            getattr(self, "lidar_capture_processing_var", None).get()
            if getattr(self, "lidar_capture_processing_var", None) is not None
            else getattr(self.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING)
        )

    def parse_lidar_depth_range(self) -> Tuple[float, float]:
        try:
            min_cm = float(self.lidar_depth_min_cm_var.get().strip())
        except Exception:
            min_cm = float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
        try:
            max_cm = float(self.lidar_depth_max_cm_var.get().strip())
        except Exception:
            max_cm = float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
        if not math.isfinite(min_cm):
            min_cm = flight.DEFAULT_LIDAR_DEPTH_MIN_CM
        if not math.isfinite(max_cm):
            max_cm = flight.DEFAULT_LIDAR_DEPTH_MAX_CM
        min_cm = max(0.0, min(65535.0, min_cm))
        max_cm = max(0.0, min(65535.0, max_cm))
        if max_cm < min_cm:
            min_cm, max_cm = max_cm, min_cm
        if max_cm <= min_cm:
            max_cm = min(65535.0, min_cm + 1.0)
        min_text = f"{min_cm:g}"
        max_text = f"{max_cm:g}"
        if getattr(self, "lidar_depth_min_cm_var", None) is not None and self.lidar_depth_min_cm_var.get().strip() != min_text:
            self.root.after(0, lambda value=min_text: self.lidar_depth_min_cm_var.set(value))
        if getattr(self, "lidar_depth_max_cm_var", None) is not None and self.lidar_depth_max_cm_var.get().strip() != max_text:
            self.root.after(0, lambda value=max_text: self.lidar_depth_max_cm_var.set(value))
        return min_cm, max_cm

    def on_save_frame(self) -> None:
        session = self.active_session()
        if session is None:
            return
        self.sync_capture_options_to_session(session)
        self.call_async("Saving RGB frame", lambda: (session.capture_observation(save=True, label="manual"), session.get_state())[1])

    def on_temp_capture(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.temp_capture_inflight:
            self.status_var.set("Temp Capture is already running.")
            return

        self.sync_capture_options_to_session(session)

        def worker() -> None:
            self.temp_capture_inflight = True
            self.root.after(0, lambda: self.status_var.set("Temp Capture..."))
            try:
                result = self.safe(
                    "Temp Capture",
                    lambda: session.capture_temp_bundle(output_root=session.args.temp_capture_dir),
                )
                if isinstance(result, dict):
                    self.root.after(0, lambda r=result: self.apply_temp_capture_result(r))
            finally:
                self.temp_capture_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def apply_temp_capture_result(self, result: Dict[str, Any]) -> None:
        message = str(result.get("message", "") or result.get("status", "Temp Capture finished"))
        self.status_var.set(message)
        self.latest_state.setdefault("last_temp_capture", result)
        self.latest_state["last_temp_capture"] = result
        pose = result.get("pose") if isinstance(result.get("pose"), dict) else {}
        if pose:
            self.latest_state["pose"] = pose
            self.pose_var.set(
                "Pose "
                f"x={float(pose.get('x', 0.0)):.1f} "
                f"y={float(pose.get('y', 0.0)):.1f} "
                f"z={float(pose.get('z', 0.0)):.1f} "
                f"yaw={float(pose.get('task_yaw', pose.get('yaw', 0.0))):.1f} "
                "action=temp_capture"
            )
        self.refresh_map_once()

    def on_temp_capture_lidar(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.temp_capture_lidar_inflight:
            self.status_var.set("Temp Capture Lidar is already running.")
            return

        self.sync_capture_options_to_session(session)

        def worker() -> None:
            self.temp_capture_lidar_inflight = True
            self.root.after(0, lambda: self.status_var.set("Temp Capture Lidar..."))
            try:
                result = self.safe(
                    "Temp Capture Lidar",
                    lambda: session.capture_temp_lidar_bundle(output_root=session.args.temp_capture_lidar_dir),
                )
                if isinstance(result, dict):
                    self.root.after(0, lambda r=result: self.apply_temp_capture_lidar_result(r))
            finally:
                self.temp_capture_lidar_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def apply_temp_capture_lidar_result(self, result: Dict[str, Any]) -> None:
        message = str(result.get("message", "") or result.get("status", "Temp Capture Lidar finished"))
        point_count = int(result.get("point_count", 0) or 0)
        min_cm = result.get("min_depth_cm", result.get("lidar_depth_min_cm", ""))
        max_cm = result.get("max_depth_cm", result.get("lidar_depth_max_cm", ""))
        if point_count:
            message = f"{message} | points={point_count}"
        if min_cm != "" and max_cm != "":
            message = f"{message} | lidar={self._fmt_float(min_cm)}-{self._fmt_float(max_cm)}cm"
        self.status_var.set(message)
        self.latest_state.setdefault("last_temp_capture_lidar", result)
        self.latest_state["last_temp_capture_lidar"] = result
        pose = result.get("pose") if isinstance(result.get("pose"), dict) else {}
        if pose:
            self.latest_state["pose"] = pose
            self.pose_var.set(
                "Pose "
                f"x={float(pose.get('x', 0.0)):.1f} "
                f"y={float(pose.get('y', 0.0)):.1f} "
                f"z={float(pose.get('z', 0.0)):.1f} "
                f"yaw={float(pose.get('task_yaw', pose.get('yaw', 0.0))):.1f} "
                "action=temp_capture_lidar"
            )
        self.refresh_map_once()

    def parse_stream_interval_s(self) -> float:
        try:
            interval_s = float(self.stream_interval_s_var.get().strip())
        except Exception:
            interval_s = flight.DEFAULT_STREAM_CAPTURE_INTERVAL_S
        interval_s = max(0.1, min(3600.0, interval_s))
        normalized = f"{interval_s:g}"
        if self.stream_interval_s_var.get().strip() != normalized:
            self.root.after(0, lambda value=normalized: self.stream_interval_s_var.set(value))
        return interval_s

    def set_stream_task_entry_locked(self, locked: bool) -> None:
        entry = getattr(self, "stream_task_entry", None)
        if entry is None:
            return
        try:
            entry.configure(state="readonly" if locked else "normal")
        except tk.TclError:
            pass

    def make_stream_capture_dir(self, task_title: str) -> Path:
        root_value = str(getattr(self.args, "stream_capture_dir", flight.DEFAULT_STREAM_CAPTURE_DIR) or flight.DEFAULT_STREAM_CAPTURE_DIR)
        root_path = flight.resolve_project_output_path(root_value, flight.DEFAULT_STREAM_CAPTURE_DIR)
        safe_task = flight.sanitize_capture_task_title(task_title)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        return flight.make_unique_child_dir(root_path, f"{safe_task}_{timestamp}")

    def write_stream_capture_summary(
        self,
        stream_dir: Path,
        *,
        task_title: str,
        interval_s: float,
        started_at: str,
        stopped_at: str = "",
        running: bool = True,
    ) -> None:
        summary = {
            "task_title": task_title,
            "safe_task_title": flight.sanitize_capture_task_title(task_title),
            "interval_s": float(interval_s),
            "stream_dir": str(stream_dir),
            "frames_dir": str(stream_dir / "frames"),
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
            "stopped_at": stopped_at,
            "running": bool(running),
            "frame_count": len(self.stream_capture_trajectory),
        }
        (stream_dir / "stream_capture.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        trajectory_payload = dict(summary)
        trajectory_payload["trajectory"] = self.stream_capture_trajectory
        (stream_dir / "trajectory.json").write_text(json.dumps(trajectory_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def make_lidar_stream_capture_dir(self, task_title: str) -> Path:
        root_value = str(
            getattr(self.args, "stream_capture_lidar_dir", flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR)
            or flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR
        )
        root_path = flight.resolve_project_output_path(root_value, flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR)
        safe_task = flight.sanitize_capture_task_title(task_title)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        return flight.make_unique_child_dir(root_path, f"{safe_task}_{timestamp}")

    def lidar_stream_capture_root_path(self) -> Path:
        root_value = str(
            getattr(self.args, "stream_capture_lidar_dir", flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR)
            or flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR
        )
        return flight.resolve_project_output_path(root_value, flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR)

    def lidar_capture_processing_mode(self) -> str:
        return flight.normalize_lidar_capture_processing(
            getattr(self, "lidar_capture_processing_var", None).get()
            if getattr(self, "lidar_capture_processing_var", None) is not None
            else getattr(self.args, "lidar_capture_processing", flight.DEFAULT_LIDAR_CAPTURE_PROCESSING)
        )

    def write_lidar_stream_capture_summary(
        self,
        stream_dir: Path,
        *,
        task_title: str,
        interval_s: float,
        started_at: str,
        stopped_at: str = "",
        running: bool = True,
    ) -> None:
        summary = {
            "capture_kind": "stream_capture_lidar",
            "task_title": task_title,
            "safe_task_title": flight.sanitize_capture_task_title(task_title),
            "interval_s": float(interval_s),
            "stream_dir": str(stream_dir),
            "frames_dir": str(stream_dir / "frames"),
            "reconstruction_dir": str(stream_dir / "reconstruction"),
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
            "stopped_at": stopped_at,
            "running": bool(running),
            "frame_count": len(self.lidar_stream_capture_trajectory),
            "source_mode": "depth_backprojected_lidar",
            "lidar_capture_processing": self.lidar_capture_processing_mode(),
            "lidar_depth_min_cm": float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
            "lidar_depth_max_cm": float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            "lidar_depth_projection": str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
            "depth_projection_selected": flight.select_lidar_depth_projection(
                getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)
            ),
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "postprocess_status": (
                self.lidar_stream_last_reconstruction.get("postprocess_status")
                if isinstance(self.lidar_stream_last_reconstruction, dict) and self.lidar_stream_last_reconstruction.get("postprocess_status")
                else ("pending" if self.lidar_capture_processing_mode() == "smooth" else "done")
            ),
            "source_point_count": int(self.lidar_stream_source_point_count),
            "reconstruction": self.lidar_stream_last_reconstruction,
        }
        (stream_dir / "stream_capture_lidar.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        trajectory_payload = dict(summary)
        trajectory_payload["trajectory"] = self.lidar_stream_capture_trajectory
        (stream_dir / "trajectory.json").write_text(json.dumps(trajectory_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_lidar_stream_reconstruction(
        self,
        stream_dir: Path,
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        cloud = self.lidar_stream_reconstruction_cloud
        if cloud is None:
            cloud = np.zeros((0, 6), dtype=np.float32)
        if not force and int(len(self.lidar_stream_capture_trajectory)) % flight.DEFAULT_LIDAR_RECON_WRITE_EVERY != 0:
            return self.lidar_stream_last_reconstruction
        reconstruction = flight.save_lidar_reconstruction_outputs(
            cloud,
            stream_dir / "reconstruction",
            source_frame_count=len(self.lidar_stream_capture_trajectory),
            source_point_count=int(self.lidar_stream_source_point_count),
            voxel_cm=flight.DEFAULT_LIDAR_RECON_VOXEL_CM,
            max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
            coordinate_frame="standard_zup",
            coordinate_units="m",
        )
        merged_path = reconstruction.get("merged_point_cloud_world_standard_m_npy_path") or reconstruction["merged_point_cloud_world_npy_path"]
        self.lidar_stream_reconstruction_cloud = np.load(merged_path).astype(np.float32, copy=False)
        self.lidar_stream_last_reconstruction = reconstruction
        return reconstruction

    def postprocess_lidar_stream_capture(self, stream_dir: Path) -> Dict[str, Any]:
        result = flight.postprocess_lidar_stream_capture(
            Path(stream_dir),
            lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
            min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
            max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            voxel_cm=flight.DEFAULT_LIDAR_RECON_VOXEL_CM,
            max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
        )
        reconstruction = result.get("reconstruction", {}) if isinstance(result.get("reconstruction"), dict) else {}
        self.lidar_stream_last_reconstruction = reconstruction
        self.lidar_stream_source_point_count = int(result.get("source_point_count", self.lidar_stream_source_point_count) or 0)
        if isinstance(result.get("frame_results"), list):
            by_dir = {
                str(Path(str(item.get("capture_dir", ""))).resolve()): item
                for item in result["frame_results"]
                if isinstance(item, dict) and item.get("capture_dir")
            }
            for entry in self.lidar_stream_capture_trajectory:
                raw_capture_dir = str(entry.get("capture_dir", "") or "")
                if not raw_capture_dir:
                    continue
                key = str(Path(raw_capture_dir).resolve())
                if key in by_dir:
                    entry.update(by_dir[key])
        merged_path = reconstruction.get("merged_point_cloud_world_standard_m_npy_path") or reconstruction.get("merged_point_cloud_world_npy_path", "")
        if merged_path and Path(str(merged_path)).exists():
            self.lidar_stream_reconstruction_cloud = np.load(str(merged_path)).astype(np.float32, copy=False)
        return result

    def update_lidar_stream_reconstruction_from_result(self, result: Dict[str, Any]) -> int:
        raw_path = str(result.get("point_cloud_world_standard_m_npy_path", "") or "")
        if not raw_path and result.get("capture_dir"):
            ensured = flight.ensure_standard_world_cloud_for_capture(
                Path(str(result.get("capture_dir"))),
                capture_payload=result,
                lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
                min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
                max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            )
            raw_path = str(ensured.get("point_cloud_world_standard_m_npy_path", "") or "")
        if not raw_path:
            raw_path = str(result.get("point_cloud_world_npy_path", "") or "")
        frame_count = int(result.get("point_count", 0) or 0)
        self.lidar_stream_source_point_count += max(0, frame_count)
        if not raw_path:
            return int(self.lidar_stream_reconstruction_cloud.shape[0]) if self.lidar_stream_reconstruction_cloud is not None else 0
        world_path = Path(raw_path)
        if not world_path.exists():
            return int(self.lidar_stream_reconstruction_cloud.shape[0]) if self.lidar_stream_reconstruction_cloud is not None else 0
        frame_cloud = np.load(world_path).astype(np.float32, copy=False)
        if frame_cloud.ndim != 2 or frame_cloud.shape[1] != 6 or frame_cloud.shape[0] == 0:
            return int(self.lidar_stream_reconstruction_cloud.shape[0]) if self.lidar_stream_reconstruction_cloud is not None else 0
        if self.lidar_stream_reconstruction_cloud is None or self.lidar_stream_reconstruction_cloud.shape[0] == 0:
            combined = frame_cloud
        else:
            combined = np.vstack((self.lidar_stream_reconstruction_cloud, frame_cloud))
        self.lidar_stream_reconstruction_cloud = flight.downsample_colored_point_cloud_voxel(
            combined,
            voxel_cm=flight.standard_voxel_size_m(flight.DEFAULT_LIDAR_RECON_VOXEL_CM),
            max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
        )
        return int(self.lidar_stream_reconstruction_cloud.shape[0])

    def on_start_stream_capture(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.stream_capture_thread is not None and self.stream_capture_thread.is_alive():
            self.stream_status_var.set("Stream Capture: already running")
            return
        if self.lidar_stream_capture_thread is not None and self.lidar_stream_capture_thread.is_alive():
            self.stream_status_var.set("Stream Capture: stop lidar capture first")
            return

        self.sync_capture_options_to_session(session)
        task_title = self.stream_task_title_var.get().strip() or "stream_task"
        interval_s = self.parse_stream_interval_s()
        stream_dir = self.make_stream_capture_dir(task_title)
        (stream_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.stream_capture_dir = stream_dir
        self.stream_capture_trajectory = []
        self.stream_capture_stop_event.clear()
        started_at = datetime.now().isoformat(timespec="milliseconds")
        self.write_stream_capture_summary(stream_dir, task_title=task_title, interval_s=interval_s, started_at=started_at)
        self.set_stream_task_entry_locked(True)
        self.stream_status_var.set(f"Stream Capture: 0 frames -> {stream_dir}")
        self.status_var.set(f"Stream Capture started: {stream_dir}")

        def worker() -> None:
            frame_index = 0
            try:
                while not self.stream_capture_stop_event.is_set():
                    frame_index += 1
                    frame_started = time.monotonic()
                    result = self.safe(
                        "Stream Capture",
                        lambda idx=frame_index, action=self.build_stream_action_detail(): session.capture_stream_frame(
                            stream_dir,
                            idx,
                            action_detail=action,
                        ),
                    )
                    if not isinstance(result, dict):
                        break
                    action_detail = result.get("action_detail", {}) if isinstance(result.get("action_detail"), dict) else {}
                    entry = {
                        "frame_index": int(result.get("frame_index", frame_index)),
                        "capture_time": result.get("capture_time", ""),
                        "pose": result.get("pose", {}),
                        "commanded_pose": result.get("commanded_pose", {}),
                        "actual_pose": result.get("actual_pose", {}),
                        "pose_error": result.get("pose_error", {}),
                        "action_detail": action_detail,
                        "last_action": action_detail.get("last_action", result.get("last_action", "")),
                        "movement_mode": action_detail.get("movement_mode", result.get("movement_mode", "")),
                        "movement_enabled": bool(action_detail.get("movement_enabled", False)),
                        "capture_dir": result.get("capture_dir", ""),
                        "rgb_path": result.get("rgb_path", ""),
                        "depth_cm_path": result.get("depth_cm_path", ""),
                        "depth_preview_path": result.get("depth_preview_path", ""),
                        "depth_npy_path": result.get("depth_npy_path", ""),
                        "pose_json_path": result.get("pose_json_path", ""),
                        "action_json_path": result.get("action_json_path", ""),
                    }
                    self.stream_capture_trajectory.append(entry)
                    self.write_stream_capture_summary(
                        stream_dir,
                        task_title=task_title,
                        interval_s=interval_s,
                        started_at=started_at,
                    )
                    self.root.after(
                        0,
                        lambda r=result, count=len(self.stream_capture_trajectory), d=stream_dir: self.apply_stream_capture_result(r, count, d),
                    )
                    elapsed_s = time.monotonic() - frame_started
                    if self.stream_capture_stop_event.wait(max(0.0, interval_s - elapsed_s)):
                        break
            finally:
                stopped_at = datetime.now().isoformat(timespec="milliseconds")
                try:
                    self.write_stream_capture_summary(
                        stream_dir,
                        task_title=task_title,
                        interval_s=interval_s,
                        started_at=started_at,
                        stopped_at=stopped_at,
                        running=False,
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to write stream capture summary: %s", exc)
                self.root.after(
                    0,
                    lambda count=len(self.stream_capture_trajectory), d=stream_dir: (
                        self.stream_status_var.set(f"Stream Capture: stopped, {count} frames -> {d}"),
                        self.set_stream_task_entry_locked(False),
                    ),
                )

        self.stream_capture_thread = threading.Thread(target=worker, daemon=True)
        self.stream_capture_thread.start()

    def on_stop_stream_capture(self) -> None:
        self.stream_capture_stop_event.set()
        self.set_stream_task_entry_locked(False)
        if self.stream_capture_thread is not None and self.stream_capture_thread.is_alive():
            self.stream_status_var.set("Stream Capture: stopping...")
            self.status_var.set("Stopping Stream Capture...")
        else:
            self.stream_status_var.set("Stream Capture: idle")

    def on_start_lidar_stream_capture(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.stream_capture_thread is not None and self.stream_capture_thread.is_alive():
            self.lidar_stream_status_var.set("Lidar Stream: stop Stream Capture first")
            return
        if self.lidar_stream_capture_thread is not None and self.lidar_stream_capture_thread.is_alive():
            self.lidar_stream_status_var.set("Lidar Stream: already running")
            return

        self.sync_capture_options_to_session(session)
        processing_mode = self.lidar_capture_processing_mode()
        task_title = self.stream_task_title_var.get().strip() or "stream_task"
        interval_s = self.parse_stream_interval_s()
        stream_dir = self.make_lidar_stream_capture_dir(task_title)
        (stream_dir / "frames").mkdir(parents=True, exist_ok=True)
        (stream_dir / "reconstruction").mkdir(parents=True, exist_ok=True)
        self.lidar_stream_capture_dir = stream_dir
        self.lidar_stream_capture_trajectory = []
        self.lidar_stream_reconstruction_cloud = np.zeros((0, 6), dtype=np.float32)
        self.lidar_stream_source_point_count = 0
        self.lidar_stream_last_reconstruction = {}
        self.lidar_stream_capture_stop_event.clear()
        started_at = datetime.now().isoformat(timespec="milliseconds")
        self.write_lidar_stream_capture_summary(stream_dir, task_title=task_title, interval_s=interval_s, started_at=started_at)
        self.set_stream_task_entry_locked(True)
        if processing_mode == "smooth":
            self.lidar_stream_status_var.set(f"Lidar Stream: 0 raw frames, mode=smooth -> {stream_dir}")
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: pending until capture stops")
            self.status_var.set("Capturing raw frames...")
        else:
            self.lidar_stream_status_var.set(f"Lidar Stream: 0 frames, merged=0, mode=full -> {stream_dir}")
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: idle")
            self.status_var.set(f"Lidar Stream started: {stream_dir}")

        def worker() -> None:
            frame_index = 0
            try:
                while not self.lidar_stream_capture_stop_event.is_set():
                    frame_index += 1
                    frame_started = time.monotonic()
                    result = self.safe(
                        "Lidar Stream Capture",
                        lambda idx=frame_index, action=self.build_stream_action_detail(): session.capture_lidar_stream_frame(
                            stream_dir,
                            idx,
                            action_detail=action,
                        ),
                    )
                    if not isinstance(result, dict):
                        break
                    action_detail = result.get("action_detail", {}) if isinstance(result.get("action_detail"), dict) else {}
                    if processing_mode == "smooth":
                        merged_count = int(self.lidar_stream_reconstruction_cloud.shape[0]) if self.lidar_stream_reconstruction_cloud is not None else 0
                    else:
                        merged_count = self.update_lidar_stream_reconstruction_from_result(result)
                    entry = {
                        "frame_index": int(result.get("frame_index", frame_index)),
                        "capture_time": result.get("capture_time", ""),
                        "pose": result.get("pose", {}),
                        "commanded_pose": result.get("commanded_pose", {}),
                        "actual_pose": result.get("actual_pose", {}),
                        "pose_error": result.get("pose_error", {}),
                        "action_detail": action_detail,
                        "last_action": action_detail.get("last_action", result.get("last_action", "")),
                        "movement_mode": action_detail.get("movement_mode", result.get("movement_mode", "")),
                        "movement_enabled": bool(action_detail.get("movement_enabled", False)),
                        "capture_dir": result.get("capture_dir", ""),
                        "rgb_path": result.get("rgb_path", ""),
                        "depth_npy_path": result.get("depth_npy_path", ""),
                        "depth_cm_path": result.get("depth_cm_path", ""),
                        "depth_preview_path": result.get("depth_preview_path", ""),
                        "point_cloud_camera_npy_path": result.get("point_cloud_camera_npy_path", ""),
                        "point_cloud_world_npy_path": result.get("point_cloud_world_npy_path", ""),
                        "point_cloud_camera_ply_path": result.get("point_cloud_camera_ply_path", ""),
                        "point_cloud_world_ply_path": result.get("point_cloud_world_ply_path", ""),
                        "point_cloud_camera_standard_m_npy_path": result.get("point_cloud_camera_standard_m_npy_path", ""),
                        "point_cloud_world_standard_m_npy_path": result.get("point_cloud_world_standard_m_npy_path", ""),
                        "point_cloud_camera_standard_m_ply_path": result.get("point_cloud_camera_standard_m_ply_path", ""),
                        "point_cloud_world_standard_m_ply_path": result.get("point_cloud_world_standard_m_ply_path", ""),
                        "point_cloud_preview_path": result.get("point_cloud_preview_path", ""),
                        "camera_info_path": result.get("camera_info_path", ""),
                        "projection_diagnostics_path": result.get("projection_diagnostics_path", ""),
                        "open3d_camera": result.get("open3d_camera", {}),
                        "open3d_world": result.get("open3d_world", {}),
                        "open3d_camera_standard_m": result.get("open3d_camera_standard_m", {}),
                        "open3d_world_standard_m": result.get("open3d_world_standard_m", {}),
                        "pose_json_path": result.get("pose_json_path", ""),
                        "action_json_path": result.get("action_json_path", ""),
                        "point_count": int(result.get("point_count", 0) or 0),
                        "invalid_depth_count": int(result.get("invalid_depth_count", 0) or 0),
                        "depth_projection_selected": result.get("depth_projection_selected", result.get("depth_projection", "")),
                        "projection_corrected": bool(result.get("projection_corrected", True)),
                        "coordinate_frame": result.get("coordinate_frame", "standard_zup"),
                        "coordinate_units": result.get("coordinate_units", "m"),
                        "raw_capture_only": bool(result.get("raw_capture_only", processing_mode == "smooth")),
                        "postprocess_status": result.get("postprocess_status", "pending" if processing_mode == "smooth" else "done"),
                        "postprocess_started_at": result.get("postprocess_started_at", ""),
                        "postprocess_finished_at": result.get("postprocess_finished_at", ""),
                        "postprocess_error": result.get("postprocess_error", ""),
                        "lidar_capture_processing": processing_mode,
                        "merged_point_count": int(merged_count),
                    }
                    self.lidar_stream_capture_trajectory.append(entry)
                    if processing_mode == "full":
                        reconstruction = self.save_lidar_stream_reconstruction(stream_dir, force=False)
                        if reconstruction:
                            self.lidar_stream_capture_trajectory[-1]["reconstruction"] = reconstruction
                    self.write_lidar_stream_capture_summary(
                        stream_dir,
                        task_title=task_title,
                        interval_s=interval_s,
                        started_at=started_at,
                    )
                    self.root.after(
                        0,
                        lambda r=result, count=len(self.lidar_stream_capture_trajectory), merged=merged_count, d=stream_dir: self.apply_lidar_stream_capture_result(
                            r,
                            count,
                            merged,
                            d,
                        ),
                    )
                    elapsed_s = time.monotonic() - frame_started
                    if self.lidar_stream_capture_stop_event.wait(max(0.0, interval_s - elapsed_s)):
                        break
            finally:
                stopped_at = datetime.now().isoformat(timespec="milliseconds")
                try:
                    self.write_lidar_stream_capture_summary(
                        stream_dir,
                        task_title=task_title,
                        interval_s=interval_s,
                        started_at=started_at,
                        stopped_at=stopped_at,
                        running=False,
                    )
                    if processing_mode == "smooth":
                        self.root.after(
                            0,
                            lambda d=stream_dir: (
                                self.lidar_stream_status_var.set(f"Lidar Stream: stopped, postprocessing -> {d}"),
                                self.lidar_stream_analysis_status_var.set("Lidar Analysis: postprocessing point clouds..."),
                                self.status_var.set("Postprocessing point clouds..."),
                            ),
                        )
                        postprocess_result = self.postprocess_lidar_stream_capture(stream_dir)
                        self.root.after(
                            0,
                            lambda r=postprocess_result: self.apply_lidar_stream_postprocess_result(r),
                        )
                    else:
                        self.save_lidar_stream_reconstruction(stream_dir, force=True)
                        self.write_lidar_stream_capture_summary(
                            stream_dir,
                            task_title=task_title,
                            interval_s=interval_s,
                            started_at=started_at,
                            stopped_at=stopped_at,
                            running=False,
                        )
                except Exception as exc:
                    LOGGER.warning("Failed to finalize lidar stream capture: %s", exc)
                self.root.after(
                    0,
                    lambda count=len(self.lidar_stream_capture_trajectory), d=stream_dir: (
                        self.lidar_stream_status_var.set(f"Lidar Stream: stopped, {count} frames -> {d}"),
                        self.lidar_stream_analysis_status_var.set(
                            f"Lidar Analysis: merged={self.lidar_stream_last_reconstruction.get('merged_point_count', 0)} "
                            f"units=m -> {self.lidar_stream_last_reconstruction.get('merged_point_cloud_world_standard_m_ply_path', self.lidar_stream_last_reconstruction.get('merged_point_cloud_world_ply_path', ''))}"
                        ),
                        self.set_stream_task_entry_locked(False),
                    ),
                )

        self.lidar_stream_capture_thread = threading.Thread(target=worker, daemon=True)
        self.lidar_stream_capture_thread.start()

    def on_stop_lidar_stream_capture(self) -> None:
        self.lidar_stream_capture_stop_event.set()
        self.set_stream_task_entry_locked(False)
        if self.lidar_stream_capture_thread is not None and self.lidar_stream_capture_thread.is_alive():
            self.lidar_stream_status_var.set("Lidar Stream: stopping...")
            self.status_var.set("Stopping Lidar Stream...")
        else:
            self.lidar_stream_status_var.set("Lidar Stream: idle")

    def apply_lidar_stream_capture_result(
        self,
        result: Dict[str, Any],
        frame_count: int,
        merged_point_count: int,
        stream_dir: Path,
    ) -> None:
        self.latest_state.setdefault("last_lidar_stream_capture", result)
        self.latest_state["last_lidar_stream_capture"] = result
        self.stream_player_dir = stream_dir
        raw_only = bool(result.get("raw_capture_only", False))
        self.stream_player_image_mode_var.set("rgb" if raw_only else "point_cloud_preview")
        pose = result.get("pose") if isinstance(result.get("pose"), dict) else {}
        if pose:
            self.latest_state["pose"] = pose
            self.pose_var.set(
                "Pose "
                f"x={float(pose.get('x', 0.0)):.1f} "
                f"y={float(pose.get('y', 0.0)):.1f} "
                f"z={float(pose.get('z', 0.0)):.1f} "
                f"yaw={float(pose.get('task_yaw', pose.get('yaw', 0.0))):.1f} "
                f"action=lidar_stream_{frame_count}"
            )
        point_count = int(result.get("point_count", 0) or 0)
        invalid_count = int(result.get("invalid_depth_count", 0) or 0)
        if raw_only:
            self.lidar_stream_status_var.set(
                f"Lidar Stream: {frame_count} raw frames, postprocess=pending -> {stream_dir}"
            )
            self.status_var.set(f"Capturing raw frames... frame {frame_count}")
        else:
            self.lidar_stream_status_var.set(
                f"Lidar Stream: {frame_count} frames, last={point_count}, invalid={invalid_count}, "
                f"merged={merged_point_count} -> {stream_dir}"
            )
            self.status_var.set(f"Lidar Stream frame {frame_count} saved")
        now = time.monotonic()
        if now - float(getattr(self, "lidar_stream_last_map_refresh", 0.0) or 0.0) >= 1.5:
            self.lidar_stream_last_map_refresh = now
            self.refresh_map_once()

    def apply_lidar_stream_postprocess_result(self, result: Dict[str, Any]) -> None:
        stream_dir = result.get("stream_dir", "")
        reconstruction = result.get("reconstruction", {}) if isinstance(result.get("reconstruction"), dict) else {}
        if reconstruction:
            self.lidar_stream_last_reconstruction = reconstruction
        merged_count = int(result.get("merged_point_count", reconstruction.get("merged_point_count", 0)) or 0)
        frame_count = int(result.get("frame_count", 0) or 0)
        status = str(result.get("postprocess_status", "done") or "done")
        ply_path = reconstruction.get(
            "merged_point_cloud_world_standard_m_ply_path",
            reconstruction.get("merged_point_cloud_world_ply_path", ""),
        )
        self.lidar_stream_status_var.set(
            f"Lidar Stream: postprocess {status}, frames={frame_count}, merged={merged_count} -> {stream_dir}"
        )
        self.lidar_stream_analysis_status_var.set(
            f"Lidar Analysis: postprocess {status}, units=m -> {ply_path}"
        )
        self.status_var.set(f"Lidar postprocess {status}: {merged_count} points")

    def collect_lidar_stream_world_cloud_paths(self, stream_dir: Path) -> List[Path]:
        stream_path = Path(stream_dir)
        paths: List[Path] = []
        seen: set[str] = set()

        def add_path(cloud_path: Path) -> None:
            key = str(cloud_path.resolve())
            if cloud_path.exists() and key not in seen:
                paths.append(cloud_path)
                seen.add(key)

        def add_capture_standard_path(capture_dir: Path, payload: Optional[Dict[str, Any]] = None) -> None:
            try:
                ensured = flight.ensure_standard_world_cloud_for_capture(
                    capture_dir,
                    capture_payload=payload,
                    lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
                    min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
                    max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
                )
            except Exception as exc:
                LOGGER.warning("Failed to ensure standard lidar frame %s: %s", capture_dir, exc)
                ensured = {}
            raw = ensured.get("point_cloud_world_standard_m_npy_path") if isinstance(ensured, dict) else ""
            if raw:
                add_path(Path(str(raw)))

        trajectory_path = stream_path / "trajectory.json"
        if trajectory_path.exists():
            try:
                payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
                trajectory = payload.get("trajectory", [])
                if isinstance(trajectory, list):
                    for entry in trajectory:
                        if not isinstance(entry, dict):
                            continue
                        raw_capture_dir = entry.get("capture_dir")
                        capture_dir = self.resolve_lidar_analysis_path(stream_path, raw_capture_dir) if hasattr(self, "resolve_lidar_analysis_path") and raw_capture_dir else Path(str(raw_capture_dir or ""))
                        if raw_capture_dir and not capture_dir.is_absolute():
                            capture_dir = (stream_path / capture_dir).resolve()
                        if raw_capture_dir and capture_dir.exists():
                            add_capture_standard_path(capture_dir, entry)
                            continue
                        raw_path = entry.get("point_cloud_world_standard_m_npy_path") or entry.get("point_cloud_world_npy_path")
                        if not raw_path:
                            continue
                        cloud_path = Path(str(raw_path))
                        if not cloud_path.is_absolute():
                            cloud_path = stream_path / cloud_path
                        if cloud_path.name == "point_cloud_world.npy":
                            add_capture_standard_path(cloud_path.parent, entry)
                        else:
                            add_path(cloud_path)
            except Exception as exc:
                LOGGER.warning("Failed to read lidar stream trajectory %s: %s", trajectory_path, exc)
        for capture_dir in sorted(path for path in (stream_path / "frames").glob("frame_*") if path.is_dir()):
            add_capture_standard_path(capture_dir)
        return paths

    def update_lidar_stream_reconstruction_metadata(self, stream_dir: Path, reconstruction: Dict[str, Any]) -> None:
        stream_path = Path(stream_dir)
        summary_path = stream_path / "stream_capture_lidar.json"
        summary: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
            except Exception as exc:
                LOGGER.warning("Failed to update lidar stream summary %s: %s", summary_path, exc)
        summary.update(
            {
                "capture_kind": "stream_capture_lidar",
                "stream_dir": str(stream_path),
                "frames_dir": str(stream_path / "frames"),
                "reconstruction_dir": str(stream_path / "reconstruction"),
                "updated_at": datetime.now().isoformat(timespec="milliseconds"),
                "source_mode": "depth_backprojected_lidar",
                "coordinate_frame": "standard_zup",
                "coordinate_units": "m",
                "depth_projection_selected": reconstruction.get("depth_projection_selected", "standard_zup_merge"),
                "source_point_count": int(reconstruction.get("source_point_count", 0) or 0),
                "reconstruction": reconstruction,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        trajectory_path = stream_path / "trajectory.json"
        trajectory_payload: Dict[str, Any] = {}
        if trajectory_path.exists():
            try:
                loaded = json.loads(trajectory_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    trajectory_payload = loaded
            except Exception as exc:
                LOGGER.warning("Failed to update lidar stream trajectory %s: %s", trajectory_path, exc)
        trajectory = trajectory_payload.get("trajectory", [])
        if not isinstance(trajectory, list):
            trajectory = []
        trajectory_payload.update(summary)
        trajectory_payload["trajectory"] = trajectory
        trajectory_path.write_text(json.dumps(trajectory_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def rebuild_lidar_stream_reconstruction(self, stream_dir: Path) -> Dict[str, Any]:
        result = self.postprocess_lidar_stream_capture(Path(stream_dir))
        reconstruction = result.get("reconstruction", {}) if isinstance(result.get("reconstruction"), dict) else {}
        merged_result = dict(reconstruction)
        merged_result.update(
            {
                "stream_dir": str(stream_dir),
                "input_cloud_file_count": int(result.get("input_cloud_file_count", 0) or 0),
                "source_frame_count": int(result.get("source_frame_count", reconstruction.get("source_frame_count", 0)) or 0),
                "source_point_count": int(result.get("source_point_count", reconstruction.get("source_point_count", 0)) or 0),
                "merged_point_count": int(result.get("merged_point_count", reconstruction.get("merged_point_count", 0)) or 0),
                "postprocess_status": result.get("postprocess_status", "done"),
                "postprocess_started_at": result.get("postprocess_started_at", ""),
                "postprocess_finished_at": result.get("postprocess_finished_at", ""),
                "postprocess_error": result.get("postprocess_error", ""),
            }
        )
        return merged_result

    def find_latest_lidar_stream_capture_dir(self) -> Optional[Path]:
        root = self.lidar_stream_capture_root_path()
        if not root.exists():
            return None
        candidates = [path for path in root.iterdir() if path.is_dir()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in candidates:
            frames_root = candidate / "frames"
            if frames_root.exists() and any(path.is_dir() for path in frames_root.glob("frame_*")):
                return candidate
        return None

    def on_analyze_lidar_stream(self) -> None:
        if self.lidar_stream_capture_thread is not None and self.lidar_stream_capture_thread.is_alive():
            self.lidar_stream_analysis_status_var.set("Lidar Analysis: stop lidar capture before analyzing")
            return
        stream_dir = self.lidar_stream_capture_dir
        if stream_dir is None or not Path(stream_dir).exists():
            stream_dir = self.find_latest_lidar_stream_capture_dir()
        if stream_dir is None:
            self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: no folders under {self.lidar_stream_capture_root_path()}")
            return
        self.lidar_stream_analysis_status_var.set(f"Lidar Analysis: rebuilding -> {stream_dir}")

        def worker() -> None:
            result = self.safe("Analyze Lidar", lambda: self.rebuild_lidar_stream_reconstruction(Path(stream_dir)))
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_lidar_stream_analysis_result(r))

        threading.Thread(target=worker, daemon=True).start()

    def apply_lidar_stream_analysis_result(self, result: Dict[str, Any]) -> None:
        stream_dir = result.get("stream_dir", "")
        merged_count = int(result.get("merged_point_count", 0) or 0)
        frame_count = int(result.get("source_frame_count", 0) or 0)
        self.lidar_stream_last_reconstruction = result
        ply_path = result.get("merged_point_cloud_world_standard_m_ply_path", result.get("merged_point_cloud_world_ply_path", ""))
        self.lidar_stream_analysis_status_var.set(
            f"Lidar Analysis: frames={frame_count}, merged={merged_count}, units=m -> {ply_path}"
        )
        if stream_dir:
            self.stream_player_dir = Path(stream_dir)
            self.stream_player_image_mode_var.set("point_cloud_preview")
        self.status_var.set(f"Lidar analysis complete: {merged_count} points")

    def apply_stream_capture_result(self, result: Dict[str, Any], frame_count: int, stream_dir: Path) -> None:
        self.latest_state.setdefault("last_stream_capture", result)
        self.latest_state["last_stream_capture"] = result
        pose = result.get("pose") if isinstance(result.get("pose"), dict) else {}
        if pose:
            self.latest_state["pose"] = pose
            self.pose_var.set(
                "Pose "
                f"x={float(pose.get('x', 0.0)):.1f} "
                f"y={float(pose.get('y', 0.0)):.1f} "
                f"z={float(pose.get('z', 0.0)):.1f} "
                f"yaw={float(pose.get('task_yaw', pose.get('yaw', 0.0))):.1f} "
                f"action=stream_{frame_count}"
            )
        self.stream_status_var.set(f"Stream Capture: {frame_count} frames -> {stream_dir}")
        self.status_var.set(f"Stream Capture frame {frame_count} saved")
        self.refresh_map_once()

    def stream_capture_root_path(self) -> Path:
        root_value = str(getattr(self.args, "stream_capture_dir", flight.DEFAULT_STREAM_CAPTURE_DIR) or flight.DEFAULT_STREAM_CAPTURE_DIR)
        return flight.resolve_project_output_path(root_value, flight.DEFAULT_STREAM_CAPTURE_DIR)

    def stream_player_path_key(self) -> Tuple[str, str]:
        mode = str(self.stream_player_image_mode_var.get() or "rgb").strip().lower()
        if mode == "depth_preview":
            return "depth_preview_path", "depth_preview.png"
        if mode == "point_cloud_preview":
            return "point_cloud_preview_path", "point_cloud_preview.png"
        return "rgb_path", "rgb.png"

    def collect_stream_player_frames(self, stream_dir: Path) -> Tuple[List[Path], float]:
        stream_path = Path(stream_dir)
        path_key, fallback_name = self.stream_player_path_key()
        interval_s = self.parse_stream_interval_s()
        frames: List[Path] = []
        seen: set[str] = set()

        trajectory_path = stream_path / "trajectory.json"
        if trajectory_path.exists():
            try:
                payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
                interval_s = float(payload.get("interval_s", interval_s) or interval_s)
                trajectory = payload.get("trajectory", [])
                if isinstance(trajectory, list):
                    for entry in trajectory:
                        if not isinstance(entry, dict):
                            continue
                        raw_path = entry.get(path_key)
                        if not raw_path and entry.get("capture_dir"):
                            raw_path = str(Path(str(entry.get("capture_dir"))) / fallback_name)
                        if not raw_path:
                            continue
                        frame_path = Path(str(raw_path))
                        if not frame_path.is_absolute():
                            frame_path = stream_path / frame_path
                        frame_key = str(frame_path.resolve())
                        if frame_path.exists() and frame_key not in seen:
                            frames.append(frame_path)
                            seen.add(frame_key)
            except Exception as exc:
                LOGGER.warning("Failed to read stream trajectory %s: %s", trajectory_path, exc)

        for frame_path in sorted((stream_path / "frames").glob(f"frame_*/*{fallback_name}")):
            frame_key = str(frame_path.resolve())
            if frame_key not in seen and frame_path.exists():
                frames.append(frame_path)
                seen.add(frame_key)
        return frames, max(0.05, interval_s)

    def refresh_stream_player_frames(self) -> None:
        if self.stream_player_dir is None:
            return
        current_path = None
        if self.stream_player_frames:
            self.stream_player_index = max(0, min(self.stream_player_index, len(self.stream_player_frames) - 1))
            current_path = self.stream_player_frames[self.stream_player_index]
        frames, interval_s = self.collect_stream_player_frames(self.stream_player_dir)
        self.stream_player_frames = frames
        self.stream_player_interval_ms = int(max(50.0, interval_s * 1000.0))
        if not frames:
            self.stream_player_index = 0
            return
        if current_path in frames:
            self.stream_player_index = frames.index(current_path)
        else:
            self.stream_player_index = max(0, min(self.stream_player_index, len(frames) - 1))

    def load_stream_player_dir(self, stream_dir: Path, *, autoplay: bool = False) -> None:
        self.stop_stream_player()
        self.stream_player_dir = Path(stream_dir)
        self.stream_player_index = 0
        self.refresh_stream_player_frames()
        if not self.stream_player_frames:
            self.stream_player_status_var.set(f"Stream Player: no frames in {self.stream_player_dir}")
            return
        self.display_stream_player_frame()
        if autoplay:
            self.start_stream_player()

    def find_latest_stream_capture_dir(self) -> Optional[Path]:
        candidates: List[Path] = []
        for root in (self.stream_capture_root_path(), self.lidar_stream_capture_root_path()):
            if root.exists():
                candidates.extend(path for path in root.iterdir() if path.is_dir())
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in candidates:
            frames, _interval_s = self.collect_stream_player_frames(candidate)
            if frames:
                return candidate
        return None

    def open_stream_player_window(self) -> None:
        if self.stream_player_window is not None and self.stream_player_window.winfo_exists():
            self.stream_player_window.lift()
            return
        self.stream_player_window = tk.Toplevel(self.root)
        self.stream_player_window.title("Stream Capture Player")
        self.stream_player_window.geometry("760x560")
        self.stream_player_window.protocol("WM_DELETE_WINDOW", self.close_stream_player_window)

        toolbar = tk.Frame(self.stream_player_window)
        toolbar.pack(fill="x", padx=6, pady=6)
        tk.Button(toolbar, text="Load Folder", command=self.select_stream_player_folder).pack(side="left", padx=(0, 6))
        tk.Button(toolbar, text="Latest", command=lambda: self.load_latest_stream_player(autoplay=False)).pack(side="left", padx=6)
        tk.Button(toolbar, text="Prev", command=self.show_stream_player_prev).pack(side="left", padx=6)
        tk.Button(toolbar, text="Play/Pause", command=self.toggle_stream_player_playback).pack(side="left", padx=6)
        tk.Button(toolbar, text="Stop", command=self.stop_stream_player).pack(side="left", padx=6)
        tk.Button(toolbar, text="Next", command=self.show_stream_player_next).pack(side="left", padx=6)
        tk.Label(toolbar, text="Image").pack(side="left", padx=(12, 2))
        mode_combo = ttk.Combobox(
            toolbar,
            textvariable=self.stream_player_image_mode_var,
            values=("rgb", "depth_preview", "point_cloud_preview"),
            state="readonly",
            width=20,
        )
        mode_combo.pack(side="left", padx=(0, 6))
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.reload_stream_player_mode())

        self.stream_player_label = tk.Label(self.stream_player_window, text="Load a stream_capture task folder.", anchor="center")
        self.stream_player_label.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        tk.Label(self.stream_player_window, textvariable=self.stream_player_status_var, anchor="w").pack(fill="x", padx=6, pady=(0, 6))

        if self.stream_player_dir is not None:
            self.load_stream_player_dir(self.stream_player_dir, autoplay=False)
        elif self.stream_capture_dir is not None:
            self.load_stream_player_dir(self.stream_capture_dir, autoplay=False)
        elif self.lidar_stream_capture_dir is not None:
            self.load_stream_player_dir(self.lidar_stream_capture_dir, autoplay=False)

    def close_stream_player_window(self) -> None:
        self.stop_stream_player()
        if self.stream_player_window is not None and self.stream_player_window.winfo_exists():
            self.stream_player_window.destroy()
        self.stream_player_window = None
        self.stream_player_label = None
        self.stream_player_photo = None

    def select_stream_player_folder(self) -> None:
        self.open_stream_player_window()
        initial_dir = self.stream_capture_root_path()
        selected = filedialog.askdirectory(title="Select stream_capture task folder", initialdir=str(initial_dir))
        if selected:
            self.load_stream_player_dir(Path(selected), autoplay=False)

    def load_latest_stream_player(self, *, autoplay: bool = False) -> None:
        self.open_stream_player_window()
        latest_dir = self.find_latest_stream_capture_dir()
        if latest_dir is None:
            self.stream_player_status_var.set(
                f"Stream Player: no stream folders under {self.stream_capture_root_path()} or {self.lidar_stream_capture_root_path()}"
            )
            return
        self.load_stream_player_dir(latest_dir, autoplay=autoplay)

    def play_latest_stream_capture(self) -> None:
        self.load_latest_stream_player(autoplay=True)

    def reload_stream_player_mode(self) -> None:
        if self.stream_player_dir is None:
            return
        autoplay = bool(self.stream_player_playing)
        self.load_stream_player_dir(self.stream_player_dir, autoplay=autoplay)

    def display_stream_player_frame(self) -> None:
        if self.stream_player_label is None:
            return
        if not self.stream_player_frames:
            self.stream_player_label.configure(image="", text="No stream frames loaded.")
            self.stream_player_status_var.set("Stream Player: no frames loaded")
            return
        self.stream_player_index = max(0, min(self.stream_player_index, len(self.stream_player_frames) - 1))
        frame_path = self.stream_player_frames[self.stream_player_index]
        try:
            image = Image.open(frame_path).convert("RGB")
            image.thumbnail((940, 700), Image.Resampling.LANCZOS)
            self.stream_player_photo = ImageTk.PhotoImage(image)
            self.stream_player_label.configure(image=self.stream_player_photo, text="")
            self.stream_player_status_var.set(
                f"Stream Player: {self.stream_player_index + 1}/{len(self.stream_player_frames)} "
                f"{self.stream_player_image_mode_var.get()} {frame_path.parent.name}"
            )
        except Exception as exc:
            self.stream_player_label.configure(image="", text=f"Failed to load {frame_path}")
            self.stream_player_status_var.set(f"Stream Player failed: {exc}")

    def show_stream_player_next(self) -> None:
        self.refresh_stream_player_frames()
        if not self.stream_player_frames:
            self.display_stream_player_frame()
            return
        self.stream_player_index = (self.stream_player_index + 1) % len(self.stream_player_frames)
        self.display_stream_player_frame()

    def show_stream_player_prev(self) -> None:
        self.refresh_stream_player_frames()
        if not self.stream_player_frames:
            self.display_stream_player_frame()
            return
        self.stream_player_index = (self.stream_player_index - 1) % len(self.stream_player_frames)
        self.display_stream_player_frame()

    def start_stream_player(self) -> None:
        if not self.stream_player_frames:
            self.refresh_stream_player_frames()
        if not self.stream_player_frames:
            self.display_stream_player_frame()
            return
        self.stream_player_playing = True
        self.cancel_stream_player_after()
        self.stream_player_after_id = self.root.after(self.stream_player_interval_ms, self.stream_player_tick)
        self.display_stream_player_frame()

    def toggle_stream_player_playback(self) -> None:
        if self.stream_player_playing:
            self.stop_stream_player()
        else:
            self.start_stream_player()

    def stream_player_tick(self) -> None:
        self.stream_player_after_id = None
        if not self.stream_player_playing:
            return
        if self.stream_player_window is None or not self.stream_player_window.winfo_exists():
            self.stop_stream_player()
            return
        self.show_stream_player_next()
        self.stream_player_after_id = self.root.after(self.stream_player_interval_ms, self.stream_player_tick)

    def cancel_stream_player_after(self) -> None:
        after_id = self.stream_player_after_id
        self.stream_player_after_id = None
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass

    def stop_stream_player(self) -> None:
        self.stream_player_playing = False
        self.cancel_stream_player_after()
        if self.stream_player_frames:
            self.stream_player_status_var.set(
                f"Stream Player: paused {self.stream_player_index + 1}/{len(self.stream_player_frames)}"
            )
        else:
            self.stream_player_status_var.set("Stream Player: idle")

    def image_to_photo(self, image: np.ndarray, max_width: int = 900, max_height: int = 620) -> ImageTk.PhotoImage:
        array = flight.prepare_observation_rgb(
            image,
            enhance=bool(self.enhance_rgb_var.get()),
            gamma=float(self.args.rgb_enhance_gamma),
            gain=float(self.args.rgb_enhance_gain),
            source_order=self.rgb_source_order_var.get().strip() or flight.DEFAULT_RGB_SOURCE_ORDER,
        )
        if array is None:
            array = np.zeros((1, 1, 3), dtype=np.uint8)
        pil = Image.fromarray(array)
        scale = min(max_width / max(1, pil.width), max_height / max(1, pil.height), 1.0)
        if scale < 1.0:
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def toggle_preview_window(self) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()
            self.preview_window = None
            self.preview_label = None
            self.preview_photo = None
            return
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("UAV RGB Preview")
        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack(fill="both", expand=True)
        self.refresh_preview_window()

    def refresh_preview_window(self) -> None:
        if self.preview_refresh_inflight:
            return
        session = self.session
        if session is None or not session.started:
            return

        def worker() -> None:
            self.preview_refresh_inflight = True
            try:
                image = self.safe("Refresh RGB", session.capture_observation)
                if isinstance(image, np.ndarray):
                    self.root.after(0, lambda img=image: self.apply_preview_image(img))
            finally:
                self.preview_refresh_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def apply_preview_image(self, image: np.ndarray) -> None:
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.toggle_preview_window()
        if self.preview_label is None:
            return
        self.preview_photo = self.image_to_photo(image)
        self.preview_label.configure(image=self.preview_photo)

    def schedule_preview_refresh(self) -> None:
        if self.auto_rgb_var.get() and self.preview_window is not None and self.preview_window.winfo_exists():
            self.refresh_preview_window()
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

    def on_run_orbit(self) -> None:
        session = self.active_session()
        if session is None:
            return

        def worker() -> Dict[str, Any]:
            return session.run_orbit(
                center=[float(self.orbit_center_x_var.get().strip()), float(self.orbit_center_y_var.get().strip())],
                radius=float(self.orbit_radius_var.get().strip()),
                altitude=float(self.orbit_altitude_var.get().strip()),
                steps=int(float(self.orbit_steps_var.get().strip())),
                start_angle=float(self.orbit_start_angle_var.get().strip()),
                clockwise=bool(self.orbit_clockwise_var.get()),
            )

        self.call_async("Running orbit", worker)

    def on_run_scripted(self) -> None:
        session = self.active_session()
        if session is None:
            return
        self.call_async("Running scripted smoke plan", session.run_scripted)

