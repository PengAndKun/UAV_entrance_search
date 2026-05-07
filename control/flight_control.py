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

    def on_save_frame(self) -> None:
        session = self.active_session()
        if session is None:
            return
        session.args.enhance_rgb = bool(self.enhance_rgb_var.get())
        session.args.rgb_enhance_gamma = float(self.args.rgb_enhance_gamma)
        session.args.rgb_enhance_gain = float(self.args.rgb_enhance_gain)
        session.args.rgb_source_order = self.rgb_source_order_var.get().strip() or flight.DEFAULT_RGB_SOURCE_ORDER
        self.call_async("Saving RGB frame", lambda: (session.capture_observation(save=True, label="manual"), session.get_state())[1])

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

