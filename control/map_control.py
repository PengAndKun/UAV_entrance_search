from __future__ import annotations

from .common import *


class MapControlMixin:
    def coerce_map_anchors(self, anchors: Any) -> List[Dict[str, float]]:
        if not isinstance(anchors, list):
            return []
        restored: List[Dict[str, float]] = []
        for index, anchor in enumerate(anchors[:5], start=1):
            if not isinstance(anchor, dict):
                continue
            try:
                restored.append({
                    "index": float(anchor.get("index", index)),
                    "label": str(anchor.get("label", f"P{index}")),
                    "world_x": float(anchor["world_x"]),
                    "world_y": float(anchor["world_y"]),
                    "image_x": float(anchor["image_x"]),
                    "image_y": float(anchor["image_y"]),
                })
            except Exception:
                continue
        return restored

    def solve_affine_from_anchors(self, anchors: List[Dict[str, float]]) -> Optional[List[List[float]]]:
        return solve_affine_from_anchor_points(anchors)

    def solve_homography_from_anchors(self, anchors: List[Dict[str, float]]) -> Optional[List[List[float]]]:
        return solve_homography_from_anchor_points(anchors)

    def normalize_map_calibration(self, payload: Any) -> Dict[str, Any]:
        calibration = payload if isinstance(payload, dict) else {}
        anchors = self.coerce_map_anchors(calibration.get("anchors", []))
        affine = calibration.get("affine_world_to_image")
        if not (isinstance(affine, list) and len(affine) == 2):
            affine = self.solve_affine_from_anchors(anchors)
        homography = calibration.get("homography_world_to_image")
        if not (isinstance(homography, list) and len(homography) == 3):
            homography = self.solve_homography_from_anchors(anchors)
        normalized: Dict[str, Any] = {"anchors": anchors}
        if isinstance(affine, list) and len(affine) == 2:
            normalized["affine_world_to_image"] = affine
        if isinstance(homography, list) and len(homography) == 3:
            normalized["homography_world_to_image"] = homography
            normalized["transform_mode"] = "homography"
        else:
            normalized["transform_mode"] = "affine" if "affine_world_to_image" in normalized else ""
        for key in ("image_width", "image_height"):
            if calibration.get(key) is not None:
                try:
                    normalized[key] = int(calibration.get(key))
                except Exception:
                    pass
        for key in ("affine_rmse_px", "homography_rmse_px"):
            if calibration.get(key) is not None:
                try:
                    normalized[key] = float(calibration.get(key))
                except Exception:
                    pass
        if calibration.get("rmse_px") is not None:
            try:
                normalized["rmse_px"] = float(calibration.get("rmse_px"))
            except Exception:
                pass
        return normalized

    def load_map_resources(self, *, force: bool = False) -> bool:
        config_path = self.resolve_project_path(str(self.args.map_config or DEFAULT_MAP_CONFIG_PATH))
        if config_path.name == DEFAULT_MANUAL_SHIFT_MAP_CONFIG_NAME:
            setting_path = config_path.parent / DEFAULT_SETTING_MAP_CONFIG_NAME
            if setting_path.exists():
                config_path = setting_path
        if not config_path.exists() and config_path.name == DEFAULT_MANUAL_SHIFT_MAP_CONFIG_NAME:
            fallback_path = self.resolve_project_path(DEFAULT_BASE_MAP_CONFIG_PATH)
            if fallback_path.exists():
                config_path = fallback_path
        if not force and self.map_config and self.map_config_path == config_path and self.map_image is not None:
            return True
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except Exception as exc:
            self.map_status_var.set(f"Map: failed to load config ({exc})")
            return False

        world_bounds = config.get("world_bounds", {}) if isinstance(config.get("world_bounds"), dict) else {}
        self.map_world_bounds = (
            float(world_bounds.get("min_x", DEFAULT_MAP_BOUNDS[0])),
            float(world_bounds.get("min_y", DEFAULT_MAP_BOUNDS[1])),
            float(world_bounds.get("max_x", DEFAULT_MAP_BOUNDS[2])),
            float(world_bounds.get("max_y", DEFAULT_MAP_BOUNDS[3])),
        )
        overhead = config.get("overhead_map", {}) if isinstance(config.get("overhead_map"), dict) else {}
        display_offset = overhead.get("display_offset_px", {}) if isinstance(overhead.get("display_offset_px"), dict) else {}
        try:
            self.map_display_offset_px = (
                float(display_offset.get("x", 0.0)),
                float(display_offset.get("y", 0.0)),
            )
        except Exception:
            self.map_display_offset_px = (0.0, 0.0)
        image_value = str(self.args.map_image or "").strip() or str(overhead.get("image_path", "") or "qq.png")
        image_path = self.resolve_project_path(image_value, base_dir=config_path.parent)
        image = cv2.imread(str(image_path))
        if image is None:
            self.map_status_var.set(f"Map: failed to load image {image_path}")
            return False

        self.map_config = config
        self.map_config_path = config_path
        self.map_image_path = image_path
        self.map_image = image
        self.map_calibration = self.normalize_map_calibration(overhead.get("calibration", {}))
        self.map_status_var.set(f"Map: loaded {image_path.name} config={config_path.name}")
        self.refresh_house_target_choices()
        return True

    def refresh_house_target_choices(self) -> None:
        houses = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        choice_map: Dict[str, str] = {}
        display_by_id: Dict[str, str] = {}
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("id", "") or "").strip()
            if not house_id:
                continue
            name = str(house.get("name", house_id) or house_id).strip()
            status = str(house.get("status", "") or "").strip()
            display = f"{house_id} - {name}"
            if status:
                display += f" [{status}]"
            choice_map[display] = house_id
            display_by_id[house_id] = display
        self.house_choice_map = choice_map
        self.house_display_by_id = display_by_id
        values = list(choice_map.keys())
        live_combos: List[ttk.Combobox] = []
        for combo in list(getattr(self, "house_target_combos", [])):
            try:
                if combo.winfo_exists():
                    combo["values"] = values
                    live_combos.append(combo)
            except tk.TclError:
                pass
        self.house_target_combos = live_combos
        if self.house_target_combo is not None and self.house_target_combo not in live_combos:
            self.house_target_combo = live_combos[0] if live_combos else None
        if self.house_target_combo is not None and not live_combos:
            try:
                self.house_target_combo["values"] = values
            except tk.TclError:
                self.house_target_combo = None
        current_display = self.llm_route_target_var.get().strip()
        current_id = choice_map.get(current_display, "")
        preferred_id = (
            str(self.llm_route_plan.get("target_house_id", "") or "")
            if isinstance(self.llm_route_plan, dict)
            else ""
        )
        if not preferred_id:
            preferred_id = str(self.map_config.get("current_target_id", "") or "")
        if preferred_id in display_by_id and (not current_id or current_id != preferred_id):
            self.llm_route_target_var.set(display_by_id[preferred_id])
        elif not current_display and values:
            self.llm_route_target_var.set(values[0])

    def selected_route_target_house_id(self) -> str:
        display = self.llm_route_target_var.get().strip()
        house_id = self.house_choice_map.get(display, "")
        if house_id:
            return house_id
        if display in self.house_display_by_id:
            return display
        text = display.lower().replace("_", " ")
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            padded = digits.zfill(3)
            for candidate in (padded, digits):
                if candidate in self.house_display_by_id:
                    return candidate
        return str(self.map_config.get("current_target_id", "") or "").strip()

    def set_selected_route_target_house(self, house_id: str) -> None:
        hid = str(house_id or "").strip()
        if hid and hid in self.house_display_by_id:
            self.llm_route_target_var.set(self.house_display_by_id[hid])

    def house_registry_for_llm_plan(self) -> Dict[str, Any]:
        if not self.map_config:
            self.load_map_resources(force=True)
        houses = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        available: List[Dict[str, Any]] = []
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("id", "") or "").strip()
            if not house_id:
                continue
            name = str(house.get("name", house_id) or house_id).strip()
            numeric = str(int(house_id)) if house_id.isdigit() else house_id
            aliases = sorted(
                {
                    house_id,
                    numeric,
                    name,
                    name.lower(),
                    f"house {numeric}",
                    f"house_{numeric}",
                    f"House_{numeric}",
                    f"house {house_id}",
                }
            )
            available.append({
                "house_id": house_id,
                "house_name": name,
                "aliases": aliases,
                "status": str(house.get("status", "") or ""),
                "center_x": house.get("center_x"),
                "center_y": house.get("center_y"),
                "radius_cm": house.get("radius_cm"),
                "map_bbox_image": house.get("map_bbox_image", {}),
            })
        return {
            "current_target_id": str(self.map_config.get("current_target_id", "") or ""),
            "selected_target_id": self.selected_route_target_house_id(),
            "available_houses": available,
        }

    def map_image_size(self) -> Optional[Tuple[int, int]]:
        if self.map_image is not None:
            return int(self.map_image.shape[1]), int(self.map_image.shape[0])
        width = self.map_calibration.get("image_width")
        height = self.map_calibration.get("image_height")
        if width and height:
            return int(width), int(height)
        return None

    def world_to_image_point(self, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        try:
            return world_to_image_with_calibration(world_x, world_y, self.map_calibration)
        except Exception:
            return None

    def find_containing_house_id(self, x: float, y: float, houses: List[Dict[str, Any]]) -> str:
        for house in houses:
            try:
                cx = float(house.get("center_x", 0.0))
                cy = float(house.get("center_y", 0.0))
                radius = float(house.get("radius_cm", 0.0))
            except Exception:
                continue
            if radius > 0.0 and float(np.hypot(float(x) - cx, float(y) - cy)) <= radius:
                return str(house.get("id", "") or "")
        return ""

    def build_map_display(self, pose: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        houses_raw = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        target_id = ""
        if isinstance(self.llm_route_plan, dict):
            target_id = str(self.llm_route_plan.get("target_house_id", "") or "")
        if not target_id:
            target_id = self.selected_route_target_house_id()
        if not target_id:
            target_id = str(self.map_config.get("current_target_id", "") or "")
        try:
            pose_x = float(pose.get("x", 0.0))
            pose_y = float(pose.get("y", 0.0))
        except Exception:
            pose_x = 0.0
            pose_y = 0.0
        current_id = self.find_containing_house_id(pose_x, pose_y, houses_raw)
        houses: List[Dict[str, Any]] = []
        boxes: List[Dict[str, Any]] = []
        for house in houses_raw:
            if not isinstance(house, dict):
                continue
            hid = str(house.get("id", "") or "")
            if not hid:
                continue
            name = str(house.get("name", hid) or hid)
            is_target = hid == target_id
            is_current = hid == current_id
            try:
                houses.append({
                    "id": hid,
                    "name": f"{name} (UAV)" if is_current else name,
                    "center_x": float(house.get("center_x", 0.0)),
                    "center_y": float(house.get("center_y", 0.0)),
                    "radius_cm": float(house.get("radius_cm", 600.0)),
                    "status": str(house.get("status", "UNSEARCHED") or "UNSEARCHED"),
                    "is_target": is_target,
                    "is_current": is_current,
                })
            except Exception:
                continue
            bbox = house.get("map_bbox_image")
            if isinstance(bbox, dict):
                try:
                    boxes.append({
                        "id": hid,
                        "name": name,
                        "status": str(house.get("status", "UNSEARCHED") or "UNSEARCHED"),
                        "is_target": is_target,
                        "is_current": is_current,
                        "map_bbox_image": {
                            "x1": float(bbox["x1"]),
                            "y1": float(bbox["y1"]),
                            "x2": float(bbox["x2"]),
                            "y2": float(bbox["y2"]),
                        },
                    })
                except Exception:
                    continue
        return houses, boxes

    def map_touch_anchors_for_session(self) -> List[Dict[str, Any]]:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return []
        anchors = self.map_calibration.get("anchors", [])
        if not isinstance(anchors, list):
            self.status_var.set("Map calibration anchors are missing.")
            return []
        normalized = sorted(
            [dict(anchor) for anchor in anchors if isinstance(anchor, dict)],
            key=lambda anchor: float(anchor.get("index", 9999)),
        )[:5]
        if len(normalized) < 5:
            self.status_var.set("P1-P5 anchors are required before calibration.")
            return []
        return normalized

    def map_touch_tolerances(self) -> Tuple[float, float]:
        try:
            xy_tol = max(0.0, float(self.xy_tolerance_var.get().strip()))
            z_tol = max(0.0, float(self.z_tolerance_var.get().strip()))
            return xy_tol, z_tol
        except ValueError:
            self.status_var.set("Invalid map calibration tolerance.")
            return 60.0, 80.0

    def call_map_touch_async(self, desc: str, fn) -> None:
        if self.manual_request_inflight:
            self.status_var.set(f"{desc} skipped while another request is running.")
            return

        def worker() -> None:
            self.manual_request_inflight = True
            self.root.after(0, lambda: self.status_var.set(f"{desc}..."))
            try:
                result = self.safe(desc, fn)
                if isinstance(result, dict):
                    self.root.after(0, lambda r=result: self.apply_map_touch_state(r))
            finally:
                self.manual_request_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def format_map_touch_state(self, state: Dict[str, Any]) -> str:
        status = str(state.get("status", "idle") or "idle")
        active = str(state.get("active_point", "") or "")
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        completed = int(state.get("completed_count", 0) or 0)
        total = len(points) if points else 5
        dx = state.get("distance_xy_cm")
        dz = state.get("distance_z_cm")
        parts = [f"Calibration: {status}", f"completed={completed}/{total}"]
        if active:
            parts.append(f"active={active}")
            active_point = next(
                (point for point in points if isinstance(point, dict) and str(point.get("label", "")) == active),
                None,
            )
            if isinstance(active_point, dict):
                try:
                    parts.append(
                        "target="
                        f"({float(active_point.get('target_world_x', 0.0)):.1f},"
                        f"{float(active_point.get('target_world_y', 0.0)):.1f},"
                        f"{float(active_point.get('target_world_z', 0.0)):.1f})"
                    )
                    marker_class = str(active_point.get("marker_class", "") or state.get("marker_class_preference", ""))
                    marker_scale = active_point.get("marker_scale", state.get("marker_scale"))
                    if marker_class:
                        parts.append(f"marker={marker_class}")
                    if isinstance(marker_scale, list):
                        parts.append("scale=" + ",".join(self._fmt_float(value) for value in marker_scale[:3]))
                except Exception:
                    pass
        if dx is not None:
            parts.append(f"dx={self._fmt_float(dx)}cm")
        if dz is not None:
            parts.append(f"dz={self._fmt_float(dz)}cm")
        saved = state.get("saved_corrected_config")
        if saved:
            parts.append(f"saved={Path(str(saved)).name}")
        return " ".join(parts)

    def apply_map_touch_state(self, state: Dict[str, Any], *, refresh: bool = True) -> None:
        previous_saved = self.map_touch_state.get("saved_corrected_config") if isinstance(self.map_touch_state, dict) else None
        self.map_touch_state = state if isinstance(state, dict) else {}
        if previous_saved and not self.map_touch_state.get("saved_corrected_config"):
            self.map_touch_state["saved_corrected_config"] = previous_saved
        self.map_calibration_var.set(self.format_map_touch_state(self.map_touch_state))
        if self.map_touch_state.get("status") == "running":
            self.map_touch_auto_saved = False
        if self.map_touch_state.get("status") == "complete" and not self.map_touch_auto_saved:
            self.map_touch_auto_saved = True
            self.save_corrected_map_config(auto=True)
            return
        if refresh:
            self.refresh_map_once()

    def anchors_with_touch_status(self, anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        state = self.map_touch_state if isinstance(self.map_touch_state, dict) else {}
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        status_by_label = {
            str(point.get("label", "")): str(point.get("status", "") or "")
            for point in points
            if isinstance(point, dict)
        }
        enriched: List[Dict[str, Any]] = []
        for anchor in anchors:
            item = dict(anchor)
            label = str(item.get("label", item.get("index", "")))
            if label in status_by_label:
                item["status"] = status_by_label[label]
            enriched.append(item)
        return enriched

    def poll_map_touch_calibration_once(self) -> None:
        session = self.session
        if session is None or not session.started or self.map_touch_poll_inflight:
            return
        state = self.map_touch_state if isinstance(self.map_touch_state, dict) else {}
        if not bool(state.get("running", False)):
            return
        xy_tol, z_tol = self.map_touch_tolerances()

        def worker() -> None:
            self.map_touch_poll_inflight = True
            try:
                result = self.safe(
                    "Polling map calibration",
                    lambda: session.poll_map_touch_calibration(xy_tolerance_cm=xy_tol, z_tolerance_cm=z_tol),
                )
                if isinstance(result, dict):
                    self.root.after(0, lambda r=result: self.apply_map_touch_state(r))
            finally:
                self.map_touch_poll_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def on_start_map_touch_calibration(self) -> None:
        session = self.active_session()
        if session is None:
            return
        anchors = self.map_touch_anchors_for_session()
        if not anchors:
            return
        marker_class = self.marker_class_var.get().strip() or flight.DEFAULT_CALIBRATION_MARKER_CLASS
        marker_scale = self.marker_scale_var.get().strip() or str(flight.DEFAULT_CALIBRATION_MARKER_SCALE[0])
        self.map_touch_auto_saved = False
        self.call_map_touch_async(
            "Starting P calibration",
            lambda: session.start_map_touch_calibration(
                anchors,
                marker_class=marker_class,
                marker_scale=marker_scale,
            ),
        )

    def on_stop_map_touch_calibration(self) -> None:
        session = self.active_session()
        if session is None:
            self.map_touch_state = {}
            self.map_calibration_var.set("Calibration: idle")
            self.refresh_map_once()
            return
        self.map_touch_auto_saved = False
        self.call_map_touch_async(
            "Stopping P calibration",
            lambda: session.stop_map_touch_calibration(cleanup=True),
        )

    def on_reset_map_touch_markers(self) -> None:
        session = self.active_session()
        if session is None:
            self.map_touch_state = {}
            self.map_calibration_var.set("Calibration: idle")
            self.refresh_map_once()
            return
        self.map_touch_auto_saved = False
        self.call_map_touch_async(
            "Resetting collision marker",
            session.reset_map_touch_calibration_markers,
        )

    def on_save_corrected_map_config(self) -> None:
        self.save_corrected_map_config(auto=False)

    def save_corrected_map_config(self, *, auto: bool = False) -> None:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return
        state = self.map_touch_state if isinstance(self.map_touch_state, dict) else {}
        session = self.session
        if (not state or state.get("status") not in {"complete", "stopped"}) and session is not None:
            try:
                state = session.get_map_touch_calibration_state()
            except Exception:
                pass
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        completed = len([point for point in points if isinstance(point, dict) and point.get("status") == "done"])
        if completed < 5:
            self.status_var.set("Need completed P1-P5 contacts before saving corrected config.")
            return
        try:
            corrected = build_corrected_map_config(self.map_config, state)
            base_dir = self.map_config_path.parent if self.map_config_path is not None else PROJECT_ROOT / "assets" / "overhead_map"
            output_path = (base_dir / DEFAULT_CORRECTED_MAP_CONFIG_NAME).resolve()
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(corrected, fh, indent=2, ensure_ascii=False)

            log_path = None
            run_dir_value = str(self.latest_state.get("run_dir", "") or "")
            if not run_dir_value and session is not None and session.run_dir is not None:
                run_dir_value = str(session.run_dir)
            if run_dir_value:
                run_dir = Path(run_dir_value)
                if run_dir.exists():
                    log_path = run_dir / "map_touch_calibration.json"
                    with open(log_path, "w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "touch_state": state,
                                "corrected_config": str(output_path),
                                "rmse_px": corrected.get("overhead_map", {}).get("calibration", {}).get("rmse_px"),
                                "saved_at": time.time(),
                            },
                            fh,
                            indent=2,
                            ensure_ascii=False,
                        )

            self.args.map_config = str(output_path)
            self.map_config = {}
            self.map_config_path = None
            self.map_image_path = None
            self.map_image = None
            self.load_map_resources(force=True)
            self.map_touch_state = dict(state)
            self.map_touch_state["saved_corrected_config"] = str(output_path)
            self.map_calibration_var.set(self.format_map_touch_state(self.map_touch_state))
            prefix = "Auto saved" if auto else "Saved"
            suffix = f"; log={log_path.name}" if log_path is not None else ""
            self.status_var.set(f"{prefix} corrected map config: {output_path.name}{suffix}")
            self.refresh_map_once(force_reload=True)
        except Exception as exc:
            LOGGER.warning("Save corrected map config failed: %s", exc)
            self.status_var.set(f"Save corrected map config failed: {exc}")

    def default_map_setting_anchors(self) -> List[Dict[str, Any]]:
        image_size = self.map_image_size() or (1000, 1000)
        image_w, image_h = float(image_size[0]), float(image_size[1])
        presets = [
            ("P1", 0.18, 0.20),
            ("P2", 0.82, 0.20),
            ("P3", 0.82, 0.80),
            ("P4", 0.18, 0.80),
            ("P5", 0.50, 0.50),
        ]
        bounds = self.map_world_bounds if isinstance(self.map_world_bounds, tuple) and len(self.map_world_bounds) == 4 else DEFAULT_MAP_BOUNDS
        min_x, min_y, max_x, max_y = [float(value) for value in bounds]
        anchors: List[Dict[str, Any]] = []
        for index, (label, fx, fy) in enumerate(presets, start=1):
            image_x = image_w * fx
            image_y = image_h * fy
            try:
                world_x, world_y = image_to_world_with_calibration(image_x, image_y, self.map_calibration)
            except Exception:
                world_x = min_x + (max_x - min_x) * fx
                world_y = min_y + (max_y - min_y) * fy
            anchors.append(
                {
                    "index": index,
                    "label": label,
                    "world_x": round(float(world_x), 3),
                    "world_y": round(float(world_y), 3),
                    "image_x": round(float(image_x), 3),
                    "image_y": round(float(image_y), 3),
                }
            )
        return anchors

    def map_setting_base_anchors(self) -> List[Dict[str, Any]]:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return []
        anchors = self.map_calibration.get("anchors", []) if isinstance(self.map_calibration, dict) else []
        normalized = sorted(
            [dict(anchor) for anchor in anchors if isinstance(anchor, dict)],
            key=lambda anchor: float(anchor.get("index", 9999)),
        )[:5]
        return normalized if len(normalized) >= 3 else self.default_map_setting_anchors()

    def map_setting_anchor_payload(self) -> List[Dict[str, Any]]:
        rows = getattr(self, "map_setting_anchor_vars", [])
        anchors: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            try:
                label = str(row["label"].get() or f"P{index}").strip() or f"P{index}"
                selected = str(
                    self.map_setting_selected_anchor_var.get()
                    if getattr(self, "map_setting_selected_anchor_var", None) is not None
                    else ""
                )
                anchors.append(
                    {
                        "index": index,
                        "label": label,
                        "world_x": float(row["world_x"].get()),
                        "world_y": float(row["world_y"].get()),
                        "image_x": float(row["image_x"].get()),
                        "image_y": float(row["image_y"].get()),
                        "status": "active" if label == selected else "pending",
                    }
                )
            except Exception:
                continue
        return anchors

    def setting_map_preview_houses(self, calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
        preview_config = json.loads(json.dumps(self.map_config if isinstance(self.map_config, dict) else {}))
        preview_config["houses"] = rebuild_houses_for_corrected_transform(preview_config, calibration)
        houses: List[Dict[str, Any]] = []
        target_id = self.selected_route_target_house_id() or str(preview_config.get("current_target_id", "") or "")
        for house in preview_config.get("houses", []) if isinstance(preview_config.get("houses"), list) else []:
            if not isinstance(house, dict):
                continue
            try:
                hid = str(house.get("id", "") or "")
                houses.append(
                    {
                        "id": hid,
                        "name": str(house.get("name", hid) or hid),
                        "center_x": float(house.get("center_x", 0.0)),
                        "center_y": float(house.get("center_y", 0.0)),
                        "radius_cm": float(house.get("radius_cm", 600.0)),
                        "status": str(house.get("status", "UNSEARCHED") or "UNSEARCHED"),
                        "is_target": hid == target_id,
                        "is_current": False,
                    }
                )
            except Exception:
                continue
        return houses

    def open_setting_map_window(self) -> None:
        if getattr(self, "map_setting_window", None) is not None and self.map_setting_window.winfo_exists():
            self.map_setting_window.lift()
            self.map_setting_window.focus_force()
            return
        if not self.load_map_resources(force=True):
            return
        window = tk.Toplevel(self.root)
        window.title("Setting Map Calibration")
        window.geometry("1120x620")
        window.grid_columnconfigure(1, weight=1)
        window.grid_rowconfigure(1, weight=1)
        self.map_setting_window = window
        self.map_setting_status_var = tk.StringVar(value="Setting Map: edit 5 anchors or click map to set selected anchor.")
        self.map_setting_selected_anchor_var = tk.StringVar(value="P1")
        self.map_setting_click_coord_var = tk.StringVar(value="Last image coord: n/a")
        self.map_setting_anchor_vars = []

        header = tk.Frame(window)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        tk.Label(header, textvariable=self.map_setting_status_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(header, text="Apply Preview", command=self.apply_setting_map_preview).pack(side="right", padx=4)
        tk.Button(header, text="Save Setting Config", command=self.save_setting_map_config).pack(side="right", padx=4)
        tk.Button(header, text="Reload Anchors", command=self.reload_setting_map_anchors).pack(side="right", padx=4)

        table = tk.LabelFrame(window, text="Calibration Points")
        table.grid(row=1, column=0, sticky="nsw", padx=(8, 4), pady=4)
        for col, title in enumerate(("Use", "Label", "World X", "World Y", "Image X", "Image Y")):
            tk.Label(table, text=title).grid(row=0, column=col, sticky="w", padx=4, pady=4)
        for row_index, anchor in enumerate(self.map_setting_base_anchors()[:5], start=1):
            label = str(anchor.get("label", f"P{row_index}") or f"P{row_index}")
            vars_row = {
                "label": tk.StringVar(value=label),
                "world_x": tk.StringVar(value=self._fmt_float(anchor.get("world_x", 0.0))),
                "world_y": tk.StringVar(value=self._fmt_float(anchor.get("world_y", 0.0))),
                "image_x": tk.StringVar(value=self._fmt_float(anchor.get("image_x", 0.0))),
                "image_y": tk.StringVar(value=self._fmt_float(anchor.get("image_y", 0.0))),
            }
            self.map_setting_anchor_vars.append(vars_row)
            ttk.Radiobutton(
                table,
                variable=self.map_setting_selected_anchor_var,
                value=label,
            ).grid(row=row_index, column=0, padx=4, pady=3)
            for col, key in enumerate(("label", "world_x", "world_y", "image_x", "image_y"), start=1):
                width = 7 if key == "label" else 10
                tk.Entry(table, textvariable=vars_row[key], width=width).grid(row=row_index, column=col, padx=4, pady=3)

        tk.Button(table, text="Use Default 5 Points", command=self.use_default_setting_map_anchors).grid(
            row=7, column=0, columnspan=3, sticky="ew", padx=4, pady=(8, 4)
        )
        tk.Button(table, text="Set Selected To UAV Image", command=self.set_setting_anchor_to_current_uav_image).grid(
            row=7, column=3, columnspan=3, sticky="ew", padx=4, pady=(8, 4)
        )
        tk.Label(table, textvariable=self.map_setting_click_coord_var, anchor="w").grid(
            row=8, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 4)
        )
        tk.Button(table, text="Copy Image Coord", command=self.copy_setting_map_image_coord).grid(
            row=8, column=4, columnspan=2, sticky="ew", padx=4, pady=(4, 4)
        )

        map_frame = tk.LabelFrame(window, text="Preview")
        map_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(0, weight=1)
        self.map_setting_widget = OverheadMapWidget(map_frame, world_bounds=self.map_world_bounds, canvas_w=760, canvas_h=500)
        self.map_setting_widget.canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.map_setting_widget.set_map_click_callback(self.on_setting_map_click)
        self.map_setting_widget.set_calibration_anchor_callbacks(
            self.on_setting_anchor_select,
            self.on_setting_anchor_drag,
        )

        def close_window() -> None:
            self.map_setting_window = None
            self.map_setting_widget = None
            self.map_setting_anchor_vars = []
            try:
                window.destroy()
            except tk.TclError:
                pass

        window.protocol("WM_DELETE_WINDOW", close_window)
        self.apply_setting_map_preview()

    def reload_setting_map_anchors(self) -> None:
        if getattr(self, "map_setting_anchor_vars", None) is None:
            return
        anchors = self.map_setting_base_anchors()[:5]
        for row, anchor in zip(self.map_setting_anchor_vars, anchors):
            row["label"].set(str(anchor.get("label", row["label"].get()) or row["label"].get()))
            row["world_x"].set(self._fmt_float(anchor.get("world_x", 0.0)))
            row["world_y"].set(self._fmt_float(anchor.get("world_y", 0.0)))
            row["image_x"].set(self._fmt_float(anchor.get("image_x", 0.0)))
            row["image_y"].set(self._fmt_float(anchor.get("image_y", 0.0)))
        self.apply_setting_map_preview()

    def use_default_setting_map_anchors(self) -> None:
        anchors = self.default_map_setting_anchors()[:5]
        for row, anchor in zip(getattr(self, "map_setting_anchor_vars", []), anchors):
            row["label"].set(str(anchor.get("label", row["label"].get()) or row["label"].get()))
            row["world_x"].set(self._fmt_float(anchor.get("world_x", 0.0)))
            row["world_y"].set(self._fmt_float(anchor.get("world_y", 0.0)))
            row["image_x"].set(self._fmt_float(anchor.get("image_x", 0.0)))
            row["image_y"].set(self._fmt_float(anchor.get("image_y", 0.0)))
        self.apply_setting_map_preview()

    def on_setting_map_click(self, image_x: float, image_y: float) -> None:
        self.map_setting_last_image_coord = (float(image_x), float(image_y))
        if getattr(self, "map_setting_click_coord_var", None) is not None:
            self.map_setting_click_coord_var.set(
                f"Last image coord: x={float(image_x):.1f}, y={float(image_y):.1f}"
            )
        selected = str(self.map_setting_selected_anchor_var.get() if getattr(self, "map_setting_selected_anchor_var", None) is not None else "")
        for row in getattr(self, "map_setting_anchor_vars", []):
            if str(row["label"].get()) == selected:
                row["image_x"].set(self._fmt_float(image_x))
                row["image_y"].set(self._fmt_float(image_y))
                self.apply_setting_map_preview()
                return

    def on_setting_anchor_select(self, label: str, image_x: float, image_y: float) -> None:
        label = str(label or "").strip()
        if not label:
            return
        self.map_setting_last_image_coord = (float(image_x), float(image_y))
        if getattr(self, "map_setting_selected_anchor_var", None) is not None:
            self.map_setting_selected_anchor_var.set(label)
        if getattr(self, "map_setting_click_coord_var", None) is not None:
            self.map_setting_click_coord_var.set(
                f"Last image coord: x={float(image_x):.1f}, y={float(image_y):.1f}"
            )
        self.apply_setting_map_preview()

    def on_setting_anchor_drag(self, label: str, image_x: float, image_y: float) -> None:
        label = str(label or "").strip()
        if not label:
            return
        self.map_setting_last_image_coord = (float(image_x), float(image_y))
        if getattr(self, "map_setting_selected_anchor_var", None) is not None:
            self.map_setting_selected_anchor_var.set(label)
        if getattr(self, "map_setting_click_coord_var", None) is not None:
            self.map_setting_click_coord_var.set(
                f"Last image coord: x={float(image_x):.1f}, y={float(image_y):.1f}"
            )
        for row in getattr(self, "map_setting_anchor_vars", []):
            if str(row["label"].get()) == label:
                row["image_x"].set(self._fmt_float(image_x))
                row["image_y"].set(self._fmt_float(image_y))
                self.apply_setting_map_preview()
                return

    def copy_setting_map_image_coord(self) -> None:
        coord = getattr(self, "map_setting_last_image_coord", None)
        if coord is None:
            if getattr(self, "map_setting_status_var", None) is not None:
                self.map_setting_status_var.set("Setting Map: click map first to generate image coord.")
            return
        text = f"{float(coord[0]):.1f}, {float(coord[1]):.1f}"
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.map_setting_status_var.set(f"Setting Map: copied image coord {text}")
        except Exception as exc:
            self.map_setting_status_var.set(f"Setting Map: copy failed: {exc}")

    def set_setting_anchor_to_current_uav_image(self) -> None:
        pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        x = self._as_float_or_none(pose.get("x"))
        y = self._as_float_or_none(pose.get("y"))
        if x is None or y is None:
            self.map_setting_status_var.set("Setting Map: no current UAV pose.")
            return
        image_point = self.world_to_image_point(float(x), float(y))
        if image_point is None:
            self.map_setting_status_var.set("Setting Map: current map affine unavailable.")
            return
        self.on_setting_map_click(float(image_point[0]), float(image_point[1]))

    def apply_setting_map_preview(self) -> None:
        widget = getattr(self, "map_setting_widget", None)
        if widget is None:
            return
        anchors = self.map_setting_anchor_payload()
        if len(anchors) < 3:
            self.map_setting_status_var.set("Setting Map: need at least 3 valid points.")
            return
        affine = solve_affine_from_anchor_points(anchors)
        if affine is None:
            self.map_setting_status_var.set("Setting Map: failed to solve affine.")
            return
        homography = solve_homography_from_anchor_points(anchors)
        preview_calibration: Dict[str, Any] = {"anchors": anchors, "affine_world_to_image": affine}
        if homography is not None:
            preview_calibration["homography_world_to_image"] = homography
            preview_calibration["transform_mode"] = "homography"
            rmse = homography_rmse_px(anchors, homography)
            mode = "homography"
        else:
            preview_calibration["transform_mode"] = "affine"
            rmse = affine_rmse_px(anchors, affine)
            mode = "affine"
        self.map_setting_preview_affine = affine
        self.map_setting_preview_homography = homography
        pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        pose_x = float(pose.get("x", 0.0)) if pose else 0.0
        pose_y = float(pose.get("y", 0.0)) if pose else 0.0
        pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0))) if pose else 0.0
        _houses, boxes = self.build_map_display(pose)
        widget.set_background_image(self.map_image)
        widget.set_calibration(affine, self.map_image_size(), anchors, homography)
        widget.set_image_layer_offset(*self.map_display_offset_px)
        widget.set_house_boxes(boxes if self.show_houses_var.get() else [])
        widget.update_houses([])
        widget.update_uav(pose_x, pose_y, pose_yaw)
        widget.set_trajectory([])
        widget.set_route_plan({})
        self.map_setting_status_var.set(
            f"Setting Map: preview {mode} solved, points={len(anchors)} rmse={rmse:.2f}px. Click map to move selected point."
        )

    def save_setting_map_config(self) -> None:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return
        anchors = self.map_setting_anchor_payload()
        if len(anchors) < 3:
            self.map_setting_status_var.set("Setting Map: need at least 3 valid points before saving.")
            return
        affine = solve_affine_from_anchor_points(anchors)
        if affine is None:
            self.map_setting_status_var.set("Setting Map: failed to solve affine.")
            return
        homography = solve_homography_from_anchor_points(anchors)
        try:
            corrected = json.loads(json.dumps(self.map_config))
            overhead = corrected.setdefault("overhead_map", {})
            calibration = overhead.setdefault("calibration", {})
            calibration["anchors"] = anchors
            calibration["affine_world_to_image"] = affine
            calibration["affine_rmse_px"] = affine_rmse_px(anchors, affine)
            if homography is not None:
                calibration["homography_world_to_image"] = homography
                calibration["homography_rmse_px"] = homography_rmse_px(anchors, homography)
                calibration["rmse_px"] = calibration["homography_rmse_px"]
                calibration["transform_mode"] = "homography"
            else:
                calibration.pop("homography_world_to_image", None)
                calibration.pop("homography_rmse_px", None)
                calibration["rmse_px"] = calibration["affine_rmse_px"]
                calibration["transform_mode"] = "affine"
            calibration["updated_at"] = time.time()
            calibration["setting_map_status"] = "manual_anchor_adjusted"
            image_size = self.map_image_size()
            if image_size is not None:
                calibration["image_width"] = int(image_size[0])
                calibration["image_height"] = int(image_size[1])
            corrected["houses"] = rebuild_houses_for_corrected_transform(corrected, calibration)
            corrected["map_setting_adjustment"] = {
                "saved_at": time.time(),
                "anchor_count": len(anchors),
                "rmse_px": calibration["rmse_px"],
                "transform_mode": calibration["transform_mode"],
                "source_config": str(self.map_config_path or ""),
            }
            base_dir = self.map_config_path.parent if self.map_config_path is not None else PROJECT_ROOT / "assets" / "overhead_map"
            output_path = (base_dir / DEFAULT_SETTING_MAP_CONFIG_NAME).resolve()
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(corrected, fh, indent=2, ensure_ascii=False)
            self.args.map_config = str(output_path)
            self.map_config = {}
            self.map_config_path = None
            self.map_image_path = None
            self.map_image = None
            self.load_map_resources(force=True)
            self.refresh_house_target_choices()
            self.map_setting_status_var.set(f"Setting Map: saved {output_path.name}, rmse={calibration['rmse_px']:.2f}px")
            self.status_var.set(f"Saved setting map config: {output_path.name}")
            self.refresh_map_once(force_reload=True)
            self.refresh_llm_route_map()
            self.refresh_llm_route2_map()
            self.refresh_llm_route3_map()
            if hasattr(self, "refresh_llm_route4_map"):
                self.refresh_llm_route4_map()
        except Exception as exc:
            LOGGER.warning("Save setting map config failed: %s", exc)
            self.map_setting_status_var.set(f"Setting Map: save failed: {exc}")
            self.status_var.set(f"Save setting map config failed: {exc}")

    def toggle_map_window(self) -> None:
        if self.map_window is not None and self.map_window.winfo_exists():
            self.close_map_window()
            return
        self.load_map_resources(force=True)
        self.map_window = tk.Toplevel(self.root)
        self.map_window.title("Overhead Map - UAV Pose")
        self.map_window.resizable(False, False)
        self.map_window.protocol("WM_DELETE_WINDOW", self.close_map_window)
        toolbar = tk.Frame(self.map_window)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(toolbar, textvariable=self.map_pose_var, anchor="w").pack(side="left", padx=(0, 12))
        tk.Label(toolbar, textvariable=self.map_status_var, anchor="w").pack(side="left")
        self.map_widget = OverheadMapWidget(self.map_window, world_bounds=self.map_world_bounds, canvas_w=900, canvas_h=480)
        self.map_widget.canvas.pack(padx=8, pady=8)
        self.refresh_map_once(force_reload=False)

    def close_map_window(self) -> None:
        try:
            if self.map_window is not None and self.map_window.winfo_exists():
                self.map_window.destroy()
        except Exception:
            pass
        self.map_window = None
        self.map_widget = None
        self.map_status_var.set("Map: closed")

    def refresh_map_once(self, force_reload: bool = False) -> None:
        if self.map_refresh_inflight:
            return
        if self.map_widget is None or self.map_window is None or not self.map_window.winfo_exists():
            if force_reload:
                self.map_status_var.set("Map: open map first")
            return
        self.map_refresh_inflight = True
        try:
            if not self.load_map_resources(force=force_reload):
                return
            pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
            pose_x = float(pose.get("x", 0.0)) if pose else 0.0
            pose_y = float(pose.get("y", 0.0)) if pose else 0.0
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0))) if pose else 0.0
            image_point = self.world_to_image_point(pose_x, pose_y)
            if image_point is None:
                self.map_pose_var.set(f"Map pose: world=({pose_x:.1f}, {pose_y:.1f}) yaw={pose_yaw:.1f} image=n/a")
            else:
                self.map_pose_var.set(
                    f"Map pose: world=({pose_x:.1f}, {pose_y:.1f}) "
                    f"image=({image_point[0]:.1f}, {image_point[1]:.1f}) yaw={pose_yaw:.1f}"
                )

            houses, boxes = self.build_map_display(pose)
            calibration = self.map_calibration
            affine = calibration.get("affine_world_to_image")
            anchors = calibration.get("anchors", []) if isinstance(calibration.get("anchors", []), list) else []
            anchors = self.anchors_with_touch_status(anchors) if self.show_calibration_points_var.get() else []
            self.map_widget.set_background_image(self.map_image)
            self.map_widget.set_calibration(
                affine,
                self.map_image_size(),
                anchors,
                calibration.get("homography_world_to_image"),
            )
            self.map_widget.set_image_layer_offset(*self.map_display_offset_px)
            self.map_widget.set_house_boxes(boxes if self.show_houses_var.get() else [])
            self.map_widget.update_houses([])
            self.map_widget.update_uav(pose_x, pose_y, pose_yaw)
            trajectory: List[Dict[str, float]] = []
            session = self.session
            if self.show_trajectory_var.get() and session is not None:
                trajectory = session.get_trajectory_points(limit=max(1, int(self.args.map_trajectory_limit)))
            self.map_widget.set_trajectory(trajectory)
            route_plan = self.refresh_active_route_plan_for_pose(pose)
            self.map_widget.set_route_plan(route_plan if route_plan else {})
        finally:
            self.map_refresh_inflight = False

    def schedule_map_refresh(self) -> None:
        self.poll_map_touch_calibration_once()
        if self.map_widget is not None and self.map_window is not None and self.map_window.winfo_exists():
            self.refresh_map_once()
        self.root.after(self.args.map_interval_ms, self.schedule_map_refresh)

