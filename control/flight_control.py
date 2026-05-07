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
        self.route_stop_event.set()
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
        session.args.enhance_rgb = bool(self.enhance_rgb_var.get())
        session.args.rgb_enhance_gamma = float(self.args.rgb_enhance_gamma)
        session.args.rgb_enhance_gain = float(self.args.rgb_enhance_gain)
        session.args.rgb_source_order = self.rgb_source_order_var.get().strip() or flight.DEFAULT_RGB_SOURCE_ORDER
        session.args.temp_capture_dir = str(getattr(self.args, "temp_capture_dir", flight.DEFAULT_TEMP_CAPTURE_DIR))
        session.args.stream_capture_dir = str(getattr(self.args, "stream_capture_dir", flight.DEFAULT_STREAM_CAPTURE_DIR))
        session.args.depth_min_cm = float(getattr(self.args, "depth_min_cm", flight.DEFAULT_DEPTH_MIN_CM))
        session.args.depth_max_cm = float(getattr(self.args, "depth_max_cm", flight.DEFAULT_DEPTH_MAX_CM))

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

    def on_start_stream_capture(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.stream_capture_thread is not None and self.stream_capture_thread.is_alive():
            self.stream_status_var.set("Stream Capture: already running")
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
                        lambda idx=frame_index: session.capture_stream_frame(stream_dir, idx),
                    )
                    if not isinstance(result, dict):
                        break
                    entry = {
                        "frame_index": int(result.get("frame_index", frame_index)),
                        "capture_time": result.get("capture_time", ""),
                        "pose": result.get("pose", {}),
                        "capture_dir": result.get("capture_dir", ""),
                        "rgb_path": result.get("rgb_path", ""),
                        "depth_cm_path": result.get("depth_cm_path", ""),
                        "depth_preview_path": result.get("depth_preview_path", ""),
                        "depth_npy_path": result.get("depth_npy_path", ""),
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
                    lambda count=len(self.stream_capture_trajectory), d=stream_dir: self.stream_status_var.set(
                        f"Stream Capture: stopped, {count} frames -> {d}"
                    ),
                )

        self.stream_capture_thread = threading.Thread(target=worker, daemon=True)
        self.stream_capture_thread.start()

    def on_stop_stream_capture(self) -> None:
        self.stream_capture_stop_event.set()
        if self.stream_capture_thread is not None and self.stream_capture_thread.is_alive():
            self.stream_status_var.set("Stream Capture: stopping...")
            self.status_var.set("Stopping Stream Capture...")
        else:
            self.stream_status_var.set("Stream Capture: idle")

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
        root = self.stream_capture_root_path()
        if not root.exists():
            return None
        candidates = [path for path in root.iterdir() if path.is_dir()]
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
            values=("rgb", "depth_preview"),
            state="readonly",
            width=13,
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
            self.stream_player_status_var.set(f"Stream Player: no stream folders under {self.stream_capture_root_path()}")
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

