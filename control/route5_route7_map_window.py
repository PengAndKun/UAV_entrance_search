from __future__ import annotations

from PIL import ImageDraw

from .common import *
from . import route6_map_builder


class Route7MapWindowMixin:
    def route7_update_map_layer_values(self, layers: List[Dict[str, Any]]) -> List[str]:
        key_fn = getattr(self, "_route6_update_map_layer_key", None)
        values: List[str] = []
        for layer in (layers if isinstance(layers, list) else []):
            if not isinstance(layer, dict):
                continue
            if callable(key_fn):
                values.append(str(key_fn(layer)))
            else:
                values.append(f"z_{int(float(layer.get('z_cm', 0) or 0)):03d}")
        return values

    def route7_select_update_map_layer_key(self, layers: List[Dict[str, Any]]) -> str:
        values = self.route7_update_map_layer_values(layers)
        selected = str(self.llm_route7_map_layer_var.get() or "").strip()
        default_key = self.route7_default_layer_key()
        if selected in values:
            return selected
        if default_key in values:
            return default_key
        return values[0] if values else default_key

    def _on_llm_route7_window_mousewheel(self, event: tk.Event):
        canvas = getattr(self, "llm_route7_window_canvas", None)
        if canvas is None:
            return "break"
        delta = -1 if int(getattr(event, "delta", 0)) > 0 else 1
        if int(getattr(event, "state", 0)) & 0x0001:
            canvas.xview_scroll(delta, "units")
        else:
            canvas.yview_scroll(delta, "units")
        return "break"

    def _on_llm_route7_window_mousewheel_linux(self, event: tk.Event):
        canvas = getattr(self, "llm_route7_window_canvas", None)
        if canvas is None:
            return "break"
        direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
        canvas.yview_scroll(direction, "units")
        return "break"

    def _bind_llm_route7_window_mousewheel_tree(self, widget: tk.Widget) -> None:
        try:
            if isinstance(widget, tk.Text):
                self._bind_llm_route5_text_mousewheel(widget)
                return
            widget.bind("<MouseWheel>", self._on_llm_route7_window_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_llm_route7_window_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_llm_route7_window_mousewheel_linux, add="+")
            children = widget.winfo_children()
        except tk.TclError:
            return
        for child in children:
            self._bind_llm_route7_window_mousewheel_tree(child)

    def route7_layered_route_points(self) -> List[Dict[str, Any]]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        active = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status"), dict) else {}
        target_house_id = str(state.get("target_house_id", "") or self.selected_route_target_house_id() or "")
        points: List[Dict[str, Any]] = []
        if target_house_id:
            points.extend(self.route7_observation_overlay_points(target_house_id, active))
        for point in self.route5_active_map_route_points():
            item = dict(point if isinstance(point, dict) else {})
            route_type = str(item.get("route_point_type", "") or "")
            if route_type in {"current_target", "navigation_waypoint", "target_reset", "original_navigation_target"}:
                if route_type in {"navigation_waypoint", "original_navigation_target"} and not item.get("z"):
                    item["z"] = self.route7_default_exploration_z_cm()
                points.append(item)
        seen: set[Tuple[str, str, str]] = set()
        deduped: List[Dict[str, Any]] = []
        for item in points:
            try:
                float(item.get("x"))
                float(item.get("y"))
            except Exception:
                continue
            key = (
                str(item.get("route_point_type", "")),
                str(item.get("label", item.get("target_id", item.get("scan_id", "")))),
                f"{float(item.get('x', 0.0) or 0.0):.2f},{float(item.get('y', 0.0) or 0.0):.2f}",
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def route7_route_point_color(self, point: Dict[str, Any], selected_layer_key: str) -> Tuple[Tuple[int, int, int], bool]:
        route_type = str((point if isinstance(point, dict) else {}).get("route_point_type", "") or "")
        z_cm = float((point if isinstance(point, dict) else {}).get("z", self.route7_default_exploration_z_cm()) or self.route7_default_exploration_z_cm())
        selected_z = float(self.route6_layer_z_from_key(selected_layer_key) if callable(getattr(self, "route6_layer_z_from_key", None)) else self.route7_map_layer_z_cm_from_key(selected_layer_key))
        strong = abs(z_cm - selected_z) <= 25.0
        palette = {
            "observation_point": ((21, 101, 192), (170, 205, 235)),
            "route7_edge_observation_point": ((21, 101, 192), (170, 205, 235)),
            "current_target": ((194, 24, 91), (238, 183, 207)),
            "navigation_waypoint": ((46, 125, 50), (184, 220, 186)),
            "original_navigation_target": ((2, 132, 199), (186, 230, 253)),
            "target_reset": ((239, 108, 0), (247, 199, 150)),
            "scan_point": ((106, 27, 154), (205, 180, 219)),
        }
        strong_color, pale_color = palette.get(route_type, ((55, 65, 81), (190, 195, 205)))
        return (strong_color if strong else pale_color), strong

    def route7_layer_route_color(self, layer_key: str, selected_layer_key: str) -> Tuple[Tuple[int, int, int], bool]:
        key = str(layer_key or selected_layer_key or self.route7_default_layer_key())
        selected = str(selected_layer_key or self.route7_default_layer_key())
        palette = {
            "z_250": (37, 99, 235),
            "z_300": (22, 101, 52),
            "z_350": (194, 65, 12),
            "z_400": (126, 34, 206),
            "z_450": (8, 145, 178),
        }
        base = palette.get(key)
        if base is None:
            try:
                z_cm = self.route7_map_layer_z_cm_from_key(key)
            except Exception:
                z_cm = self.route7_default_exploration_z_cm()
            hue = int(abs(z_cm) // 50) % 4
            base = ((22, 101, 52), (37, 99, 235), (194, 65, 12), (126, 34, 206))[hue]
        strong = key == selected
        if strong:
            return base, True
        pale = tuple(int(round(float(channel) * 0.35 + 255.0 * 0.65)) for channel in base)
        return pale, False

    def route7_current_route_visualization_context(self) -> Dict[str, str]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        current = state.get("route7_realtime_route_plan", {}) if isinstance(state.get("route7_realtime_route_plan", {}), dict) else {}
        active = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status"), dict) else {}
        return {
            "stage": str(current.get("stage") or active.get("stage") or state.get("stage") or "").strip(),
            "target_id": str(current.get("target_id") or active.get("target_id") or state.get("target_id") or "").strip(),
            "target_house_id": str(current.get("target_house_id") or state.get("target_house_id") or active.get("target_house_id") or "").strip(),
        }

    def route7_route_plan_context_mismatches(self, plan: Dict[str, Any], context: Dict[str, str]) -> List[str]:
        if not isinstance(plan, dict):
            return ["plan:not_dict"]
        mismatches: List[str] = []
        for key in ("target_id", "target_house_id"):
            expected = str((context or {}).get(key, "") or "").strip()
            if not expected:
                continue
            actual = str(plan.get(key, "") or "").strip()
            if not actual:
                mismatches.append(f"{key}:missing")
            elif actual != expected:
                mismatches.append(f"{key}:{actual}!={expected}")
        expected_stage = str((context or {}).get("stage", "") or "").strip()
        actual_stage = str(plan.get("stage", "") or "").strip()
        if expected_stage and actual_stage and actual_stage != expected_stage:
            mismatches.append(f"stage:{actual_stage}!={expected_stage}")
        return mismatches

    def route7_set_route_visualization_status(
        self,
        status: str,
        *,
        context: Optional[Dict[str, str]] = None,
        current: Optional[Dict[str, Any]] = None,
        fallback: Optional[Dict[str, Any]] = None,
        mismatches: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": str(status or ""),
            "context": dict(context or {}),
            "mismatches": list(mismatches or []),
        }
        if isinstance(current, dict):
            payload["current_status"] = str(current.get("status", "") or "")
            payload["current_reason"] = str(current.get("reason", "") or "")
            payload["current_target_id"] = str(current.get("target_id", "") or "")
            payload["current_segment_count"] = len(current.get("route7_route_segments", []) if isinstance(current.get("route7_route_segments", []), list) else [])
        if isinstance(fallback, dict):
            payload["fallback_status"] = str(fallback.get("status", "") or "")
            payload["fallback_reason"] = str(fallback.get("reason", "") or "")
            payload["fallback_target_id"] = str(fallback.get("target_id", "") or "")
            payload["fallback_segment_count"] = len(fallback.get("route7_route_segments", []) if isinstance(fallback.get("route7_route_segments", []), list) else [])
        try:
            if isinstance(getattr(self, "llm_route5_state", None), dict):
                self.llm_route5_state["route7_route_visualization_status"] = payload
        except Exception:
            pass
        return payload

    def route7_route_visualization_status_suffix(self) -> str:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        payload = state.get("route7_route_visualization_status", {}) if isinstance(state.get("route7_route_visualization_status", {}), dict) else {}
        status = str(payload.get("status", "") or "")
        if not status:
            return ""
        if status == "current_route_segments_ok":
            return " route=current"
        if status == "using_matching_last_drawable_route_plan":
            return " route=fallback"
        if status == "last_drawable_target_mismatch":
            return " route=hidden target_mismatch"
        if status == "current_empty_no_matching_fallback":
            current_status = str(payload.get("current_status", "") or "")
            current_reason = str(payload.get("current_reason", "") or "")
            if current_status or current_reason:
                detail = ":".join(part for part in (current_status, current_reason) if part)
                return f" route=hidden {detail[:48]}"
            return " route=hidden no_plan"
        return f" route={status[:48]}"

    def route7_drawable_realtime_route_plan(self) -> Dict[str, Any]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        context = self.route7_current_route_visualization_context()
        current = state.get("route7_realtime_route_plan", {}) if isinstance(state.get("route7_realtime_route_plan", {}), dict) else {}
        current_segments = current.get("route7_route_segments", []) if isinstance(current.get("route7_route_segments", []), list) else []
        if current_segments:
            self.route7_set_route_visualization_status("current_route_segments_ok", context=context, current=current)
            return current
        fallback = state.get("route7_last_drawable_route_plan", {}) if isinstance(state.get("route7_last_drawable_route_plan", {}), dict) else {}
        fallback_segments = fallback.get("route7_route_segments", []) if isinstance(fallback.get("route7_route_segments", []), list) else []
        if fallback_segments:
            mismatches = self.route7_route_plan_context_mismatches(fallback, context)
            if mismatches:
                self.route7_set_route_visualization_status(
                    "last_drawable_target_mismatch",
                    context=context,
                    current=current,
                    fallback=fallback,
                    mismatches=mismatches,
                )
                return current
            self.route7_set_route_visualization_status(
                "using_matching_last_drawable_route_plan",
                context=context,
                current=current,
                fallback=fallback,
            )
            return fallback
        self.route7_set_route_visualization_status("current_empty_no_matching_fallback", context=context, current=current)
        return current

    def route7_static_house_base(self) -> Dict[str, Any]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        base = state.get("route7_static_house_base", {}) if isinstance(state.get("route7_static_house_base", {}), dict) else {}
        if base:
            return self.route7_filter_static_house_base(base)
        output_dir = self.route7_current_map_output_dir()
        if output_dir is not None:
            try:
                path = Path(output_dir) / "map" / "route7_static_house_base.json"
                payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                if isinstance(payload, dict):
                    return self.route7_filter_static_house_base(payload)
            except Exception:
                return {}
        return {}

    def route7_current_house_base_ids(self) -> List[str]:
        known_records_fn = getattr(self, "route6_known_house_polygon_records", None)
        known_records = known_records_fn() if callable(known_records_fn) else []
        ids: List[str] = []
        if isinstance(known_records, list):
            for record in known_records[:5]:
                if not isinstance(record, dict):
                    continue
                house_id = str(record.get("house_id", record.get("id", "")) or "").strip()
                if house_id and house_id not in ids:
                    ids.append(house_id)
        return ids or ["001", "002", "003", "004", "005"]

    def route7_filter_static_house_base(self, base: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(base, dict):
            return {}
        houses = base.get("houses", []) if isinstance(base.get("houses", []), list) else []
        if not houses:
            return base
        allowed = set(self.route7_current_house_base_ids())
        filtered = [
            house
            for house in houses
            if isinstance(house, dict) and str(house.get("house_id", house.get("id", "")) or "").strip() in allowed
        ]
        if not filtered:
            return base
        if len(filtered) == len(houses):
            return base
        payload = dict(base)
        payload["houses"] = self.route5_json_safe(filtered)
        payload["house_count"] = len(filtered)
        payload["route7_static_house_base_filter"] = "current_five_operator_house_coordinates"
        return payload

    def route7_draw_static_house_base(
        self,
        image: Image.Image,
        layer_record: Dict[str, Any],
        *,
        selected_layer_key: str,
        scale: int = 1,
    ) -> Image.Image:
        metadata = self.route6_update_map_load_layer_metadata(layer_record) if callable(getattr(self, "route6_update_map_load_layer_metadata", None)) else {}
        if not metadata:
            return image
        base = self.route7_static_house_base()
        houses = base.get("houses", []) if isinstance(base.get("houses", []), list) else []
        if not houses:
            return image
        draw = ImageDraw.Draw(image)
        factor = max(1, int(scale))
        for house in houses:
            if not isinstance(house, dict):
                continue
            bbox = house.get("bbox_world", {}) if isinstance(house.get("bbox_world", {}), dict) else {}
            try:
                min_x = float(bbox["min_x"])
                max_x = float(bbox["max_x"])
                min_y = float(bbox["min_y"])
                max_y = float(bbox["max_y"])
            except Exception:
                continue
            corners = [
                self.route7_layer_point_to_pixel({"x": min_x, "y": min_y}, metadata, scale=factor),
                self.route7_layer_point_to_pixel({"x": max_x, "y": min_y}, metadata, scale=factor),
                self.route7_layer_point_to_pixel({"x": max_x, "y": max_y}, metadata, scale=factor),
                self.route7_layer_point_to_pixel({"x": min_x, "y": max_y}, metadata, scale=factor),
            ]
            if any(point is None for point in corners):
                continue
            is_target = bool(house.get("is_target_house", False))
            outline = (89, 155, 235) if is_target else (180, 190, 202)
            fill = (89, 155, 235, 20) if is_target else (180, 190, 202, 12)
            points = [(int(point[0]), int(point[1])) for point in corners if point is not None]
            if len(points) >= 4:
                draw.polygon(points, outline=outline, fill=fill if image.mode == "RGBA" else None)
                draw.line(points + [points[0]], fill=outline, width=max(1, 2 * factor if is_target else factor))
                label = str(house.get("label", house.get("house_id", "")) or "")
                if label:
                    cx = sum(point[0] for point in points) / len(points)
                    cy = sum(point[1] for point in points) / len(points)
                    draw.text((cx - 16 * factor, cy - 6 * factor), label[:10], fill=outline)
        return image

    def route7_draw_realtime_route_plan(
        self,
        image: Image.Image,
        layer_record: Dict[str, Any],
        *,
        selected_layer_key: str,
        scale: int = 1,
    ) -> Image.Image:
        metadata = self.route6_update_map_load_layer_metadata(layer_record) if callable(getattr(self, "route6_update_map_load_layer_metadata", None)) else {}
        if not metadata:
            return image
        route_plan = self.route7_drawable_realtime_route_plan()
        segments = route_plan.get("route7_route_segments", []) if isinstance(route_plan.get("route7_route_segments", []), list) else []
        if not segments:
            return image
        draw = ImageDraw.Draw(image)
        factor = max(1, int(scale))
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            from_pose = segment.get("from_pose", {}) if isinstance(segment.get("from_pose", {}), dict) else {}
            to_pose = segment.get("to_pose", {}) if isinstance(segment.get("to_pose", {}), dict) else {}
            start = self.route7_layer_point_to_pixel(from_pose, metadata, scale=factor)
            end = self.route7_layer_point_to_pixel(to_pose, metadata, scale=factor)
            if start is None or end is None:
                continue
            color, strong = self.route7_layer_route_color(str(segment.get("layer_key", "") or ""), selected_layer_key)
            width = max(2, 4 * factor if strong else 2 * factor)
            kind = str(segment.get("kind", "") or "")
            if kind == "vertical_transition":
                radius = max(4, 3 * factor)
                draw.ellipse((end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius), outline=color, width=max(1, factor))
                draw.line((start[0], start[1], end[0], end[1]), fill=color, width=max(1, factor))
            else:
                draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
        return image

    def route7_layer_point_to_pixel(self, point: Dict[str, Any], metadata: Dict[str, Any], *, scale: int = 1) -> Optional[Tuple[int, int]]:
        try:
            width = int(metadata.get("width", 0) or 0)
            height = int(metadata.get("height", 0) or 0)
            resolution = float(metadata.get("resolution_m", 0.25) or 0.25)
            origin_x, origin_y = [float(value) for value in metadata.get("origin_standard_m", [0.0, 0.0])]
            standard_x_m = float(point.get("x", 0.0) or 0.0) / 100.0
            standard_y_m = -float(point.get("y", 0.0) or 0.0) / 100.0
        except Exception:
            return None
        if width <= 0 or height <= 0 or resolution <= 0:
            return None
        col = int(math.floor((standard_x_m - origin_x) / resolution))
        row = int(math.floor((standard_y_m - origin_y) / resolution))
        if col < 0 or col >= width or row < 0 or row >= height:
            return None
        factor = max(1, int(scale))
        return int(col * factor + factor / 2), int((height - 1 - row) * factor + factor / 2)

    def route7_draw_layered_route_points(
        self,
        image: Image.Image,
        layer_record: Dict[str, Any],
        *,
        selected_layer_key: str,
        scale: int = 1,
    ) -> Image.Image:
        metadata = self.route6_update_map_load_layer_metadata(layer_record) if callable(getattr(self, "route6_update_map_load_layer_metadata", None)) else {}
        if not metadata:
            return image
        image = self.route7_draw_realtime_route_plan(image, layer_record, selected_layer_key=selected_layer_key, scale=scale)
        draw = ImageDraw.Draw(image)
        factor = max(1, int(scale))
        for point in self.route7_layered_route_points():
            pixel = self.route7_layer_point_to_pixel(point, metadata, scale=factor)
            if pixel is None:
                continue
            color, strong = self.route7_route_point_color(point, selected_layer_key)
            route_type = str(point.get("route_point_type", "") or "")
            radius = max(3, 4 * factor if strong else 3 * factor)
            x_px, y_px = pixel
            draw.ellipse(
                (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                fill=color,
                outline=(20, 20, 20) if strong else color,
                width=max(1, factor if strong else 1),
            )
            label = str(point.get("label", point.get("target_id", point.get("scan_id", ""))) or "")
            if label and strong:
                draw.text((x_px + radius + 2, y_px - radius), label[:18], fill=color)
        return image

    def route7_observation_formula_text(self, target_house_id: str = "", facade: str = "") -> str:
        hid = str(target_house_id or self.selected_route_target_house_id() or "").strip()
        selected_facade = str(facade or "").strip().lower()
        bbox: Dict[str, Any] = {}
        if hid:
            try:
                if callable(getattr(self, "route7_house_bbox_for_id", None)):
                    bbox = self.route7_house_bbox_for_id(hid, output_dir=self.route7_current_map_output_dir())
                if not bbox:
                    bbox = self.house_world_bbox_for_id(hid)
            except Exception:
                bbox = {}
        formula: Dict[str, Any] = {}
        if bbox and selected_facade in {"south", "east", "north", "west"}:
            formula = self.route7_lidar_edge_observation_formula(bbox, selected_facade)
        elif bbox:
            selected_facade = "south"
            formula = self.route7_lidar_edge_observation_formula(bbox, selected_facade)
        lidar_radius = self.route7_lidar_radius_cm()
        effective_radius = 0.8 * float(lidar_radius)
        lines = [
            "Route7 Edge Observation Formula",
            "",
            "Minimum facade-edge points:",
            "  R_lidar = lidar_depth_max_cm",
            "  R_eff = 0.8 * R_lidar",
            "  L_edge = abs(axis_max - axis_min)",
            "  N_edge = max(1, ceil(L_edge / R_eff))",
            "  spacing = L_edge / N_edge",
            "  axis_i = axis_min + (i + 0.5) * spacing",
            "  D_obs = clamp(0.65 * R_eff, 250 cm, 0.95 * R_eff)",
            "  P_i = facade_pose(axis_i, D_obs, z=300 cm)",
            "",
            "Obstacle redesign:",
            "  c_i = map_cell(P_i, selected_layer)",
            "  if occupied(inflate(map), c_i):",
            "      c_i' = nearest_free_cell(inflate(map), c_i)",
            "      P_i' = cell_center_pose(c_i', z=300 cm, yaw=face_facade)",
            "",
            "Spatial A* navigation:",
            "  route7_spatial_occupancy_astar searches x_cell, y_cell, and z_layer.",
            "  z_300 is preferred first; alternate layers add vertical transition waypoints.",
            "",
            f"Current R_lidar: {lidar_radius:.1f} cm",
            f"Current R_eff: {effective_radius:.1f} cm",
        ]
        if formula:
            lines.extend(
                [
                    f"Current house: {hid}",
                    f"Current facade: {selected_facade}",
                    f"L_edge: {float(formula.get('edge_length_cm', 0.0) or 0.0):.1f} cm",
                    f"N_edge: {int(formula.get('point_count', 0) or 0)}",
                    f"spacing: {float(formula.get('spacing_cm', 0.0) or 0.0):.1f} cm",
                    f"D_obs: {float(formula.get('observation_standoff_cm', 0.0) or 0.0):.1f} cm",
                ]
            )
        return "\n".join(lines)

    def open_route7_observation_formula_window(self) -> None:
        self.ensure_route7_state()
        try:
            window = tk.Toplevel(getattr(self, "root", None))
            window.title("Route7 Observation Formula")
            window.geometry("760x520")
            frame = tk.Frame(window)
            frame.pack(fill="both", expand=True, padx=8, pady=8)
            text = tk.Text(frame, wrap="word", font=("Consolas", 10))
            scroll = tk.Scrollbar(frame, orient="vertical", command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            text.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            text.insert("1.0", self.route7_observation_formula_text())
            text.configure(state="disabled")
        except Exception as exc:
            try:
                self.llm_route7_map_status_var.set(f"Route V7 Formula: failed: {exc}")
            except Exception:
                pass

    def route7_task_switch_records(self) -> List[Dict[str, Any]]:
        self.ensure_route7_state()
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        plan = state.get("task_plan", {}) if isinstance(state.get("task_plan", {}), dict) else {}
        if not plan:
            output_dir = self.route7_current_map_output_dir()
            try:
                path = Path(output_dir) / "route5_task_plan.json" if output_dir is not None else None
                payload = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.is_file() else {}
                plan = payload.get("plan", {}) if isinstance(payload.get("plan", {}), dict) else {}
            except Exception:
                plan = {}
        subtasks = plan.get("subtasks", []) if isinstance(plan.get("subtasks", []), list) else []
        ordered: List[str] = []
        for item in subtasks:
            if not isinstance(item, dict):
                continue
            facade = str(item.get("facade", "") or "").strip().lower()
            if facade and facade not in ordered:
                ordered.append(facade)
        if not ordered:
            ordered = list(plan.get("facade_priority", [])) if isinstance(plan.get("facade_priority", []), list) else []
            ordered = [str(item).strip().lower() for item in ordered if str(item).strip()]
        if not ordered:
            ordered = ["south", "east", "west", "north"]
        completed = {str(item).strip().lower() for item in (getattr(self, "llm_route5_completed_facades", set()) or set())}
        completed.update(str(item).strip().lower() for item in state.get("completed_facades", []) if str(item).strip())
        blocked = {str(item).strip().lower() for item in (getattr(self, "llm_route5_blocked_facades", set()) or set())}
        blocked.update(str(item).strip().lower() for item in state.get("blocked_facades", []) if str(item).strip())
        current_facade = str(state.get("current_facade", "") or "").strip().lower()
        stage = str(state.get("stage", "") or "").strip()
        records: List[Dict[str, Any]] = []
        for index, facade in enumerate(ordered, start=1):
            status = "pending"
            if facade in completed:
                status = "completed"
            elif facade in blocked:
                status = "blocked"
            elif facade and facade == current_facade:
                status = "active"
            records.append(
                {
                    "order": index,
                    "facade": facade,
                    "status": status,
                    "stage": stage if status == "active" else "",
                    "is_current": bool(facade and facade == current_facade),
                }
            )
        return records

    def route7_subtask_switch_records(self) -> List[Dict[str, Any]]:
        self.ensure_route7_state()
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        active = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status", {}), dict) else {}
        facade = str(active.get("facade", state.get("current_facade", "")) or "").strip().lower()
        stage = str(active.get("stage", state.get("stage", "")) or "").strip().upper()
        completed_facades = {str(item).strip().lower() for item in (getattr(self, "llm_route5_completed_facades", set()) or set())}
        completed_facades.update(str(item).strip().lower() for item in state.get("completed_facades", []) if str(item).strip())
        raw_scan_points: List[Dict[str, Any]] = []
        for source in (state.get("current_facade_scan_points", []), state.get("facade_scan_points", [])):
            if isinstance(source, list):
                raw_scan_points.extend([dict(item) for item in source if isinstance(item, dict)])
        if not raw_scan_points:
            route2_state = getattr(self, "llm_route2_state", {}) if isinstance(getattr(self, "llm_route2_state", {}), dict) else {}
            source = route2_state.get("facade_scan_points", [])
            if isinstance(source, list):
                raw_scan_points.extend([dict(item) for item in source if isinstance(item, dict)])
        scan_points = [
            item for item in raw_scan_points
            if not facade or str(item.get("facade", facade) or "").strip().lower() == facade
        ]
        completed_scan_ids = self.route5_completed_scan_ids()
        active_target_id = str(active.get("target_id", active.get("scan_id", "")) or "")

        def phase_status(label: str) -> str:
            if facade and facade in completed_facades:
                return "completed"
            if label == "observe":
                if stage in {"NAV_TO_OBS", "OBSERVE", "OBSERVATION_CAPTURE"}:
                    return "active"
                if stage in {"BUILD_MAP", "NAV_TO_SCAN_POINT", "SCAN", "VALIDATE", "ANALYZE", "COMPLETE_FACADE"} or scan_points:
                    return "completed"
            if label == "map":
                if stage in {"BUILD_MAP", "ROUTE7_BUILD_MAP", "MAP_UPDATE"}:
                    return "active"
                if stage in {"NAV_TO_SCAN_POINT", "SCAN", "VALIDATE", "ANALYZE", "COMPLETE_FACADE"} or scan_points:
                    return "completed"
            if label == "validate" and stage in {"VALIDATE", "ANALYZE", "COMPLETE_FACADE"}:
                return "active" if stage != "COMPLETE_FACADE" else "completed"
            return "pending"

        records: List[Dict[str, Any]] = [
            {"order": 1, "label": "observe", "status": phase_status("observe"), "facade": facade},
            {"order": 2, "label": "map", "status": phase_status("map"), "facade": facade},
        ]
        scan_start = len(records) + 1
        for index, point in enumerate(scan_points, start=1):
            scan_id = str(point.get("scan_id", point.get("target_id", point.get("label", ""))) or "")
            status = "completed" if scan_id and scan_id in completed_scan_ids else "pending"
            if stage == "NAV_TO_SCAN_POINT" and active_target_id and scan_id == active_target_id:
                status = "active"
            elif stage == "NAV_TO_SCAN_POINT" and not active_target_id and index == 1:
                status = "active"
            records.append(
                {
                    "order": scan_start + index - 1,
                    "label": f"scan {index}",
                    "status": status,
                    "facade": facade,
                    "target_id": scan_id,
                }
            )
        records.append({"order": len(records) + 1, "label": "validate", "status": phase_status("validate"), "facade": facade})
        return records

    def route7_task_switch_visualization_text(self) -> str:
        records = self.route7_task_switch_records()
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        house_id = str(state.get("target_house_id", "") or "-")
        parts: List[str] = []
        for record in records:
            label = str(record.get("facade", "") or "-")
            status = str(record.get("status", "") or "pending")
            stage = str(record.get("stage", "") or "")
            parts.append(f"{label}:{status}{('/' + stage) if stage else ''}")
        subparts = [
            f"{str(record.get('label', '-') or '-')}:{str(record.get('status', 'pending') or 'pending')}"
            for record in self.route7_subtask_switch_records()
        ]
        suffix = (" | Subtasks: " + " -> ".join(subparts)) if subparts else ""
        return f"Task switch house={house_id} | " + " -> ".join(parts) + suffix

    def refresh_route7_task_switch_visualization(self) -> None:
        self.ensure_route7_state()
        text = self.route7_task_switch_visualization_text()
        try:
            self.llm_route7_task_switch_text.set(text)
        except Exception:
            pass
        canvas = getattr(self, "llm_route7_task_switch_canvas", None)
        if canvas is None:
            return
        try:
            canvas.delete("all")
            width = max(640, int(canvas.winfo_width() or 1000))
            height = max(68, int(canvas.winfo_height() or 76))
            records = self.route7_task_switch_records()
            if not records:
                canvas.create_text(12, height // 2, text=text, anchor="w", fill="#4a4a4a")
                return
            subtask_records = self.route7_subtask_switch_records()
            colors = {
                "active": ("#2f6fed", "white"),
                "completed": ("#3c9b52", "white"),
                "blocked": ("#b83a3a", "white"),
                "pending": ("#d8dde6", "#20242a"),
            }
            margin = 12
            gap = 10

            def draw_row(row_records: List[Dict[str, Any]], y0: int, y1: int, *, kind: str) -> None:
                if not row_records:
                    return
                chip_w = max(78 if kind == "subtask" else 96, int((width - margin * 2 - gap * (len(row_records) - 1)) / max(1, len(row_records))))
                for index, record in enumerate(row_records):
                    x0 = margin + index * (chip_w + gap)
                    x1 = x0 + chip_w
                    status = str(record.get("status", "pending") or "pending")
                    fill, fg = colors.get(status, colors["pending"])
                    canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#6b7280", width=1)
                    if kind == "subtask":
                        label = str(record.get("label", "") or "-").upper()
                        bottom = status
                    else:
                        facade = str(record.get("facade", "") or "-").upper()
                        stage = str(record.get("stage", "") or "")
                        label = f"{int(record.get('order', index + 1))}. {facade}"
                        bottom = stage or status
                    canvas.create_text((x0 + x1) / 2, y0 + 13, text=label, fill=fg, font=("Arial", 9, "bold"))
                    canvas.create_text((x0 + x1) / 2, y0 + 30, text=bottom, fill=fg, font=("Arial", 8))
                    if index < len(row_records) - 1:
                        ax = x1 + 2
                        ay = (y0 + y1) / 2
                        canvas.create_line(ax, ay, ax + gap - 4, ay, fill="#6b7280", width=2, arrow="last")

            draw_row(records, 10, 50, kind="facade")
            if subtask_records:
                canvas.create_text(margin, 66, text="Subtasks", anchor="w", fill="#4b5563", font=("Arial", 8, "bold"))
                draw_row(subtask_records, 78, min(height - 8, 118), kind="subtask")
        except tk.TclError:
            pass

    def refresh_llm_route7_update_map(self, *, build_if_missing: bool = False) -> Dict[str, Any]:
        self.ensure_route7_state()
        if callable(getattr(self, "ensure_route6_state", None)):
            self.ensure_route6_state()
        try:
            if callable(getattr(self, "route6_update_map_uav_pose_text", None)):
                self.route6_update_map_uav_pose_text()
        except Exception:
            pass
        output_dir = self.route7_current_map_output_dir()
        load_manifest = getattr(self, "route6_update_map_load_manifest", None)
        manifest = load_manifest(build_if_missing=bool(build_if_missing), output_dir=output_dir) if callable(load_manifest) else {}
        combo = getattr(self, "llm_route7_update_map_layer_combo", None)
        preview = getattr(self, "llm_route7_update_map_preview_label", None)
        try:
            self.refresh_route7_task_switch_visualization()
        except Exception:
            pass
        if not manifest:
            self.llm_route7_map_status_var.set("Route V7 Map: no Route 6 Update Map layered artifact yet.")
            if preview is not None:
                try:
                    preview.configure(text="No Route 6 Update Map layered occupancy map available.")
                except tk.TclError:
                    pass
            return {}
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        values = self.route7_update_map_layer_values(layers)
        if combo is not None:
            try:
                combo.configure(values=values)
            except tk.TclError:
                pass
        selected = self.route7_select_update_map_layer_key(layers)
        if selected:
            self.llm_route7_map_layer_var.set(selected)
        key_fn = getattr(self, "_route6_update_map_layer_key", None)
        layer_record = next(
            (
                layer
                for layer in layers
                if isinstance(layer, dict)
                and ((str(key_fn(layer)) if callable(key_fn) else f"z_{int(float(layer.get('z_cm', 0) or 0)):03d}") == selected)
            ),
            {},
        )
        point_count = int((layer_record or {}).get("point_count", 0) or 0)
        occupied_count = int((layer_record or {}).get("occupied_cell_count", 0) or 0)
        status_prefix = f"Route V7 Map: Route 6 Update Map {selected or 'n/a'} points={point_count} occupied={occupied_count}"
        self.llm_route7_map_status_var.set(status_prefix + self.route7_route_visualization_status_suffix())
        if preview is None:
            return manifest
        preview_path_fn = getattr(self, "route6_update_map_layer_preview_path", None)
        preview_path = preview_path_fn(layer_record) if callable(preview_path_fn) else Path(str((layer_record or {}).get("occupancy_preview_path", "") or ""))
        if not Path(preview_path).is_file():
            try:
                preview.configure(text=f"Route 6 Update Map preview missing: {preview_path}")
            except tk.TclError:
                pass
            return manifest
        try:
            image = Image.open(preview_path).convert("RGB")
            width, height = image.size
            frame = getattr(self, "llm_route7_update_map_frame", None)
            try:
                available_w = int(frame.winfo_width() or 1040) if frame is not None else 1040
            except Exception:
                available_w = 1040
            scale = max(1, min(8, int((available_w - 40) / max(1, max(width, height)))))
            if scale > 1:
                image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
            image = self.route7_draw_static_house_base(image, layer_record, selected_layer_key=selected, scale=scale)
            overlay_fn = getattr(self, "route6_draw_update_map_uav_overlay", None)
            if callable(overlay_fn):
                image = overlay_fn(image, layer_record, scale=scale)
            image = self.route7_draw_layered_route_points(image, layer_record, selected_layer_key=selected, scale=scale)
            self.llm_route7_map_status_var.set(status_prefix + self.route7_route_visualization_status_suffix())
            photo = ImageTk.PhotoImage(image)
            self.llm_route7_update_map_preview_photo = photo
            preview.configure(image=photo, text="")
        except Exception as exc:
            try:
                preview.configure(text=f"Route V7 Route 6 Update Map preview failed: {exc}")
            except tk.TclError:
                pass
        return manifest

    def route7_schedule_update_map_refresh(self) -> None:
        window = getattr(self, "llm_route7_window", None)
        if window is None:
            self.llm_route7_update_map_after_id = None
            return
        try:
            self.refresh_llm_route7_update_map(build_if_missing=False)
            self.llm_route7_update_map_after_id = window.after(1000, self.route7_schedule_update_map_refresh)
        except tk.TclError:
            self.llm_route7_update_map_after_id = None

    def close_llm_route_window7(self) -> None:
        self.ensure_route7_state()
        after_id = getattr(self, "llm_route7_update_map_after_id", None)
        if after_id is not None and self.llm_route7_window is not None:
            try:
                self.llm_route7_window.after_cancel(after_id)
            except Exception:
                pass
        self.llm_route7_update_map_after_id = None
        if self.llm_route7_window is not None:
            try:
                self.llm_route7_window.destroy()
            except Exception:
                pass
        self.llm_route7_window = None
        self.llm_route7_window_canvas = None
        self.llm_route7_window_content = None
        self.llm_route7_window_content_window = None
        self.llm_route7_update_map_frame = None
        self.llm_route7_update_map_preview_label = None
        self.llm_route7_update_map_preview_photo = None
        self.llm_route7_update_map_layer_combo = None
        self.llm_route7_task_switch_canvas = None
        self.llm_route5_preview_text = None
        self.llm_route5_analysis_text = None
        self.llm_route5_rgb_label = None
        self.llm_route5_rgb_photo = None
        self.route5_or2_state_label = None
        self.route5_or2_rgb_label = None
        self.route5_or2_mask_label = None
        self.route5_or2_rgb_photo = None
        self.route5_or2_mask_photo = None
        self.route5_or2_report_text = None
        self.stop_route5_or2_monitor()
        self.cancel_route5_auto_refresh()

    def open_llm_route_window7(self) -> None:
        self.ensure_route7_state()
        if self.llm_route7_window is not None and self.llm_route7_window.winfo_exists():
            self.llm_route7_window.lift()
            self.llm_route7_window.focus_force()
            return
        self.llm_route7_map_layer_var.set(self.route7_default_layer_key())
        window = tk.Toplevel(self.root)
        window.title("LLM House Entrance Route V7 Fused Route + Avoidance + Route 6 Update Map")
        window.geometry("1120x860")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close_llm_route_window7)

        window_canvas = tk.Canvas(window, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(window, orient="vertical", command=window_canvas.yview)
        h_scrollbar = tk.Scrollbar(window, orient="horizontal", command=window_canvas.xview)
        window_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        window_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        content = tk.Frame(window_canvas)
        content_window = window_canvas.create_window((0, 0), window=content, anchor="nw")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(3, weight=1)

        def _sync_scrollregion(_event: tk.Event) -> None:
            try:
                window_canvas.configure(scrollregion=window_canvas.bbox("all"))
            except tk.TclError:
                pass

        def _sync_content_width(event: tk.Event) -> None:
            try:
                window_canvas.itemconfigure(content_window, width=max(1060, int(event.width)))
            except tk.TclError:
                pass

        content.bind("<Configure>", _sync_scrollregion)
        window_canvas.bind("<Configure>", _sync_content_width)

        route = self._build_llm_route5_section(
            content,
            route_label="V7",
            start_command=self.on_route7_start_fused_search,
            step_command=self.on_route7_step_facade,
            stop_command=self.on_route7_stop,
        )
        route.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        support = tk.Frame(content)
        support.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 4))
        support.grid_columnconfigure(0, weight=0)
        support.grid_columnconfigure(1, weight=1)
        rgb_frame = tk.LabelFrame(support, text="Facade RGB")
        rgb_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        self.llm_route5_rgb_label = tk.Canvas(rgb_frame, width=330, height=220, bg="#202020", highlightthickness=0)
        self.llm_route5_rgb_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.llm_route5_rgb_label.bind("<Configure>", lambda _event: self.refresh_route5_rgb_display(), add="+")
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
        self.llm_route5_analysis_text = analysis_text

        monitor = self.build_route5_or2_monitor_panel(content, representation_label="OR3_1")
        monitor.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 4))

        map_frame = tk.LabelFrame(content, text="Route 6 Update Map Layered Occupancy")
        map_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 8))
        map_frame.grid_columnconfigure(0, weight=1)
        map_frame.grid_rowconfigure(2, weight=1)
        toolbar = tk.Frame(map_frame)
        toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        tk.Label(toolbar, text="Layer").pack(side="left", padx=(0, 4))
        layer_combo = ttk.Combobox(
            toolbar,
            textvariable=self.llm_route7_map_layer_var,
            values=[f"z_{int(value):03d}" for value in route6_map_builder.DEFAULT_ROUTE6_LAYER_Z_CM],
            state="readonly",
            width=10,
        )
        layer_combo.pack(side="left", padx=4)
        layer_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_llm_route7_update_map(build_if_missing=False))
        tk.Button(toolbar, text="Load Latest Map", command=lambda: self.refresh_llm_route7_update_map(build_if_missing=False)).pack(side="left", padx=6)
        tk.Button(toolbar, text="Route 6 Update Map", command=self.open_route6_update_map_window).pack(side="left", padx=6)
        tk.Button(toolbar, text="Edge Formula", command=self.open_route7_observation_formula_window).pack(side="left", padx=6)
        tk.Label(toolbar, textvariable=self.llm_route7_map_status_var, anchor="w").pack(side="left", fill="x", expand=True, padx=8)
        map_status = tk.Frame(map_frame)
        map_status.grid(row=1, column=0, sticky="ew", padx=6, pady=(6, 0))
        map_status.grid_columnconfigure(0, weight=1)
        tk.Label(map_status, textvariable=self.llm_route5_current_status_var, anchor="w", wraplength=360, justify="left").grid(row=0, column=0, sticky="ew", padx=(0, 8))
        tk.Label(map_status, textvariable=self.llm_route5_next_status_var, anchor="w", wraplength=300, justify="left").grid(row=0, column=1, sticky="ew", padx=(0, 8))
        tk.Label(map_status, textvariable=self.llm_route5_progress_text_var, anchor="e", wraplength=220, justify="left").grid(row=0, column=2, sticky="e", padx=(8, 2))
        ttk.Progressbar(map_status, variable=self.llm_route5_progress_var, maximum=100.0, length=150, mode="determinate").grid(row=0, column=3, sticky="e", padx=(0, 8))
        preview = tk.Label(map_frame, text="Loading Route 6 Update Map layered occupancy map...", anchor="center", justify="center")
        preview.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        task_switch_frame = tk.LabelFrame(map_frame, text="Route7 Task Switch")
        task_switch_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        task_switch_frame.grid_columnconfigure(0, weight=1)
        tk.Label(task_switch_frame, textvariable=self.llm_route7_task_switch_text, anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 2))
        task_switch_canvas = tk.Canvas(task_switch_frame, height=128, bg="#f4f6f8", highlightthickness=1, highlightbackground="#c9ced6")
        task_switch_canvas.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        task_switch_canvas.bind("<Configure>", lambda _event: self.refresh_route7_task_switch_visualization(), add="+")

        self.llm_route7_window = window
        self.llm_route7_window_canvas = window_canvas
        self.llm_route7_window_content = content
        self.llm_route7_window_content_window = content_window
        self.llm_route7_update_map_frame = map_frame
        self.llm_route7_update_map_layer_combo = layer_combo
        self.llm_route7_update_map_preview_label = preview
        self.llm_route7_task_switch_canvas = task_switch_canvas
        self._bind_llm_route7_window_mousewheel_tree(content)
        window_canvas.bind("<MouseWheel>", self._on_llm_route7_window_mousewheel, add="+")
        window_canvas.bind("<Button-4>", self._on_llm_route7_window_mousewheel_linux, add="+")
        window_canvas.bind("<Button-5>", self._on_llm_route7_window_mousewheel_linux, add="+")
        self.refresh_route5_support_views()
        self.refresh_llm_route7_update_map(build_if_missing=False)
        self.route7_schedule_update_map_refresh()

    def on_route7_start_fused_search(self) -> None:
        self.ensure_route7_state()
        session = self.active_session()
        if session is None:
            return
        selected_target_house_id = self.selected_route_target_house_id()
        if not selected_target_house_id:
            self.llm_route5_status_var.set("LLM Route V7: select a target house first.")
            return
        if self.llm_route5_thread is not None and self.llm_route5_thread.is_alive():
            self.llm_route5_status_var.set("LLM Route V7: already running.")
            return
        try:
            self.llm_route5_representation_model_var.set(str(self.default_route7_or31_model_path()))
        except Exception:
            pass
        output_dir = self.route7_prepare_new_map_output_dir(selected_target_house_id)
        if not self.route7_start_update_map_realtime(session, output_dir):
            self.llm_route5_status_var.set("LLM Route V7: waiting for previous map update to stop.")
            return
        self.llm_route5_stop_event.clear()
        self.llm_route5_pause_event.clear()
        self.llm_route5_paused_var.set(False)
        self.llm_route5_thread = threading.Thread(
            target=lambda: self.route5_full_search_worker(
                session,
                single_facade=False,
                force_new=True,
                observation_z_cm=self.route7_default_exploration_z_cm(),
                route_window_label="V7",
                output_dir_override=output_dir,
            ),
            daemon=True,
        )
        self.llm_route5_thread.start()

    def on_route7_step_facade(self) -> None:
        self.ensure_route7_state()
        self.route5_update_state(route_window_label="V7", observation_z_override_cm=self.route7_default_exploration_z_cm())
        session = self.active_session()
        if session is None:
            return
        if self.llm_route5_thread is not None and self.llm_route5_thread.is_alive():
            self.llm_route5_status_var.set("LLM Route V7: wait for current worker.")
            return
        self.llm_route5_stop_event.clear()
        self.llm_route5_pause_event.clear()
        self.llm_route5_thread = threading.Thread(
            target=lambda: self.route5_full_search_worker(
                session,
                single_facade=True,
                force_new=False,
                observation_z_cm=self.route7_default_exploration_z_cm(),
                route_window_label="V7",
            ),
            daemon=True,
        )
        self.llm_route5_thread.start()

    def on_route7_stop(self) -> None:
        self.ensure_route7_state()
        if callable(getattr(self, "ensure_route6_state", None)):
            self.ensure_route6_state()
        self.llm_route5_stop_event.set()
        self.llm_route5_pause_event.clear()
        try:
            self.llm_route5_paused_var.set(False)
        except Exception:
            pass
        if hasattr(self, "route6_update_map_realtime_stop_event"):
            self.route6_update_map_realtime_stop_event.set()
        if hasattr(self, "route6_update_map_capture_stop_event"):
            self.route6_update_map_capture_stop_event.set()
        session = self.active_session()
        if session is not None:
            self.route5_hold(session, output_dir=self.route5_state_output_dir(), reason="route7_stop_button")
        self.llm_route5_status_var.set("LLM Route V7: stop requested.")
        try:
            self.route6_update_map_status_var.set("Route 6 Update Map: V7 stop requested.")
            self.llm_route7_map_status_var.set("Route V7 Map: stop requested.")
        except Exception:
            pass

