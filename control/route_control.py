from __future__ import annotations

from .common import *
from .route_scan import generate_rule_scan_points, normalize_facade_order


class RouteControlMixin:
    def apply_llm_api_defaults(self, force: bool = False) -> None:
        style = normalize_llm_api_style(self.llm_api_style_var.get())
        if style != self.llm_api_style_var.get().strip():
            self.llm_api_style_var.set(style)
        base_url = self.llm_base_url_var.get().strip()
        api_key = self.llm_api_key_var.get().strip()
        model = self.llm_model_var.get().strip()
        if style.startswith("openai"):
            if force or not base_url or "anthropic" in base_url.lower():
                self.llm_base_url_var.set(
                    os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("OPENAI_API_BASE")
                    or LLM_OPENAI_DEFAULT_BASE_URL
                )
            if force or not api_key:
                self.llm_api_key_var.set(os.environ.get("OPENAI_API_KEY", api_key))
            if force or not model or model.lower().startswith("claude"):
                self.llm_model_var.set(os.environ.get("OPENAI_MODEL", model or "gpt-5.5"))
        elif style == "anthropic_sdk":
            if force or not base_url or "openai" in base_url.lower():
                self.llm_base_url_var.set(os.environ.get("ANTHROPIC_BASE_URL") or LLM_ANTHROPIC_DEFAULT_BASE_URL)
            if force or not api_key:
                self.llm_api_key_var.set(os.environ.get("ANTHROPIC_AUTH_TOKEN", api_key))
            if force or not model or model.lower().startswith("gpt-"):
                self.llm_model_var.set(os.environ.get("ANTHROPIC_MODEL", model or "claude-3-5-sonnet-latest"))

    def current_llm_api_style(self) -> str:
        style = normalize_llm_api_style(self.llm_api_style_var.get())
        if self.llm_api_style_var.get() != style:
            self.llm_api_style_var.set(style)
        return style

    def effective_llm_base_url(self) -> str:
        self.apply_llm_api_defaults(force=False)
        return self.llm_base_url_var.get().strip()

    def effective_llm_api_key(self) -> str:
        self.apply_llm_api_defaults(force=False)
        return self.llm_api_key_var.get().strip()

    def effective_llm_model(self) -> str:
        self.apply_llm_api_defaults(force=False)
        return self.llm_model_var.get().strip()

    def llm_route_timeout_s(self) -> float:
        try:
            return max(5.0, float(self.llm_timeout_s_var.get().strip()))
        except Exception:
            return 60.0

    def image_to_world_point(self, image_x: float, image_y: float) -> Optional[Tuple[float, float]]:
        affine = self.map_calibration.get("affine_world_to_image")
        if not isinstance(affine, list) or len(affine) != 2:
            return None
        try:
            return image_to_world_with_affine(float(image_x), float(image_y), affine)
        except Exception:
            return None

    def current_route_pose(self) -> Dict[str, float]:
        pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        x = self._as_float_or_none(pose.get("x"))
        y = self._as_float_or_none(pose.get("y"))
        z = self._as_float_or_none(pose.get("z"))
        yaw = self._as_float_or_none(pose.get("task_yaw", pose.get("yaw")))
        if x is None or y is None or yaw is None:
            return {}
        return {"x": float(x), "y": float(y), "z": float(z if z is not None else 100.0), "yaw": float(yaw)}

    def house_records_for_route_planning(self) -> List[Dict[str, Any]]:
        if not self.map_config:
            self.load_map_resources(force=True)
        records: List[Dict[str, Any]] = []
        houses = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("id", "") or "").strip()
            x = self._as_float_or_none(house.get("center_x"))
            y = self._as_float_or_none(house.get("center_y"))
            radius = self._as_float_or_none(house.get("radius_cm"))
            if house_id and x is not None and y is not None and radius is not None:
                records.append({"id": house_id, "x": float(x), "y": float(y), "radius_cm": float(radius)})
        return records

    def house_radius_cm_for_id(self, house_id: str) -> Optional[float]:
        hid = str(house_id or "").strip()
        for house in self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []:
            if isinstance(house, dict) and str(house.get("id", "") or "").strip() == hid:
                return self._as_float_or_none(house.get("radius_cm"))
        return None

    def house_world_bbox_for_id(self, house_id: str) -> Dict[str, float]:
        hid = str(house_id or "").strip()
        if not hid:
            return {}
        if not self.map_config:
            self.load_map_resources(force=True)
        houses = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        for house in houses:
            if not isinstance(house, dict) or str(house.get("id", "") or "").strip() != hid:
                continue
            points: List[Tuple[float, float]] = []
            bbox_image = house.get("map_bbox_image", {}) if isinstance(house.get("map_bbox_image"), dict) else {}
            x1 = self._as_float_or_none(bbox_image.get("x1"))
            y1 = self._as_float_or_none(bbox_image.get("y1"))
            x2 = self._as_float_or_none(bbox_image.get("x2"))
            y2 = self._as_float_or_none(bbox_image.get("y2"))
            if None not in (x1, y1, x2, y2):
                for image_x, image_y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
                    world_point = self.image_to_world_point(float(image_x), float(image_y))
                    if world_point:
                        points.append(world_point)
            if points:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                return {
                    "min_x": min(xs),
                    "max_x": max(xs),
                    "min_y": min(ys),
                    "max_y": max(ys),
                    "center_x": 0.5 * (min(xs) + max(xs)),
                    "center_y": 0.5 * (min(ys) + max(ys)),
                    "source": "map_bbox_image_affine",
                }
            center_x = self._as_float_or_none(house.get("center_x"))
            center_y = self._as_float_or_none(house.get("center_y"))
            radius = self._as_float_or_none(house.get("radius_cm"))
            if center_x is not None and center_y is not None and radius is not None:
                half = max(150.0, float(radius) * 0.45)
                return {
                    "min_x": float(center_x) - half,
                    "max_x": float(center_x) + half,
                    "min_y": float(center_y) - half,
                    "max_y": float(center_y) + half,
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "source": "center_radius_fallback",
                }
        return {}

    def route_forbidden_house_bboxes(
        self,
        *,
        target_house_id: str,
        current_house_id: str = "",
        clearance_cm: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        target_id = str(target_house_id or "").strip()
        current_id = str(current_house_id or "").strip()
        ignored = {target_id}
        if current_id:
            ignored.add(current_id)
        clearance = float(LLM_ROUTE_HOUSE_CLEARANCE_CM if clearance_cm is None else clearance_cm)
        obstacles: List[Dict[str, Any]] = []
        for house in self.house_records_for_route_planning():
            house_id = str(house.get("id", "") or "").strip()
            if not house_id or house_id in ignored:
                continue
            bbox = self.house_world_bbox_for_id(house_id)
            if not bbox:
                continue
            try:
                min_x = float(bbox["min_x"]) - clearance
                max_x = float(bbox["max_x"]) + clearance
                min_y = float(bbox["min_y"]) - clearance
                max_y = float(bbox["max_y"]) + clearance
            except Exception:
                continue
            obstacles.append({
                "house_id": house_id,
                "clearance_cm": clearance,
                "min_x": round(min_x, 2),
                "max_x": round(max_x, 2),
                "min_y": round(min_y, 2),
                "max_y": round(max_y, 2),
            })
        return obstacles

    def point_inside_open_bbox(self, x: float, y: float, bbox: Dict[str, Any], *, eps: float = 1e-6) -> bool:
        try:
            return bool(
                float(bbox["min_x"]) + eps < float(x) < float(bbox["max_x"]) - eps
                and float(bbox["min_y"]) + eps < float(y) < float(bbox["max_y"]) - eps
            )
        except Exception:
            return False

    def segment_intersects_open_bbox(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        bbox: Dict[str, Any],
        *,
        eps: float = 1e-6,
    ) -> bool:
        if self.point_inside_open_bbox(ax, ay, bbox, eps=eps) or self.point_inside_open_bbox(bx, by, bbox, eps=eps):
            return True
        try:
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            return False
        dx = float(bx) - float(ax)
        dy = float(by) - float(ay)
        t0 = 0.0
        t1 = 1.0
        for p, q in (
            (-dx, float(ax) - min_x),
            (dx, max_x - float(ax)),
            (-dy, float(ay) - min_y),
            (dy, max_y - float(ay)),
        ):
            if abs(p) <= eps:
                if q < 0.0:
                    return False
                continue
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
        if t1 < t0:
            return False
        mid_t = max(0.0, min(1.0, 0.5 * (t0 + t1)))
        mid_x = float(ax) + mid_t * dx
        mid_y = float(ay) + mid_t * dy
        return self.point_inside_open_bbox(mid_x, mid_y, bbox, eps=eps)

    def route_house_violation_report(
        self,
        route_points: List[Dict[str, Any]],
        *,
        target_house_id: str,
        current_house_id: str = "",
        clearance_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        points: List[Dict[str, float]] = []
        for point in route_points if isinstance(route_points, list) else []:
            if not isinstance(point, dict):
                continue
            x = self._as_float_or_none(point.get("x", point.get("world_x")))
            y = self._as_float_or_none(point.get("y", point.get("world_y")))
            if x is not None and y is not None:
                points.append({"x": float(x), "y": float(y)})
        obstacles = self.route_forbidden_house_bboxes(
            target_house_id=target_house_id,
            current_house_id=current_house_id,
            clearance_cm=clearance_cm,
        )
        violations: List[Dict[str, Any]] = []
        for segment_index in range(max(0, len(points) - 1)):
            start = points[segment_index]
            end = points[segment_index + 1]
            for obstacle in obstacles:
                if self.segment_intersects_open_bbox(
                    start["x"], start["y"], end["x"], end["y"], obstacle
                ):
                    violations.append({
                        "segment_index": segment_index,
                        "house_id": str(obstacle.get("house_id", "") or ""),
                        "segment_start": {"x": round(start["x"], 2), "y": round(start["y"], 2)},
                        "segment_end": {"x": round(end["x"], 2), "y": round(end["y"], 2)},
                    })
        return {
            "valid": bool(len(points) >= 2 and not violations),
            "point_count": len(points),
            "checked_house_count": len(obstacles),
            "clearance_cm": float(LLM_ROUTE_HOUSE_CLEARANCE_CM if clearance_cm is None else clearance_cm),
            "violation_count": len(violations),
            "violations": violations[:12],
        }

    def visibility_graph_route_points(
        self,
        *,
        start: Dict[str, Any],
        final: Dict[str, Any],
        target_house_id: str,
        current_house_id: str = "",
    ) -> List[Dict[str, Any]]:
        direct_report = self.route_house_violation_report(
            [start, final],
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        if bool(direct_report.get("valid", False)):
            return [dict(start), dict(final)]
        obstacles = self.route_forbidden_house_bboxes(
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        if not obstacles:
            return [dict(start), dict(final)]
        candidates: List[Dict[str, Any]] = []
        key_to_index: Dict[Tuple[float, float], int] = {}

        def add_candidate(x: float, y: float, label: str) -> Optional[int]:
            point = {"x": round(float(x), 2), "y": round(float(y), 2), "label": label, "status": "planned"}
            for obstacle in obstacles:
                if self.point_inside_open_bbox(float(point["x"]), float(point["y"]), obstacle):
                    return None
            key = (round(float(point["x"]), 2), round(float(point["y"]), 2))
            if key in key_to_index:
                return key_to_index[key]
            key_to_index[key] = len(candidates)
            candidates.append(point)
            return key_to_index[key]

        start_idx = add_candidate(float(start["x"]), float(start["y"]), str(start.get("label", "start") or "start"))
        final_idx = add_candidate(float(final["x"]), float(final["y"]), str(final.get("label", "target") or "target"))
        if start_idx is None or final_idx is None:
            return []
        for obstacle in obstacles:
            house_id = str(obstacle.get("house_id", "") or "")
            for x, y, suffix in (
                (obstacle["min_x"], obstacle["min_y"], "sw"),
                (obstacle["min_x"], obstacle["max_y"], "nw"),
                (obstacle["max_x"], obstacle["min_y"], "se"),
                (obstacle["max_x"], obstacle["max_y"], "ne"),
            ):
                add_candidate(float(x), float(y), f"avoid:{house_id}:{suffix}")

        def clear_edge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            report = self.route_house_violation_report(
                [a, b],
                target_house_id=target_house_id,
                current_house_id=current_house_id,
            )
            return bool(report.get("valid", False))

        n = len(candidates)
        distances = [float("inf")] * n
        previous: List[Optional[int]] = [None] * n
        distances[start_idx] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, start_idx)]
        visited: set[int] = set()
        while heap:
            current_dist, idx = heapq.heappop(heap)
            if idx in visited:
                continue
            visited.add(idx)
            if idx == final_idx:
                break
            a = candidates[idx]
            for next_idx, b in enumerate(candidates):
                if next_idx == idx or next_idx in visited or not clear_edge(a, b):
                    continue
                weight = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
                new_dist = current_dist + weight
                if new_dist < distances[next_idx]:
                    distances[next_idx] = new_dist
                    previous[next_idx] = idx
                    heapq.heappush(heap, (new_dist, next_idx))
        if not math.isfinite(distances[final_idx]):
            return []
        path_indices: List[int] = []
        idx: Optional[int] = final_idx
        while idx is not None:
            path_indices.append(idx)
            idx = previous[idx]
        path_indices.reverse()
        points: List[Dict[str, Any]] = []
        for order, idx_value in enumerate(path_indices):
            point = dict(candidates[idx_value])
            if order == 0:
                point["label"] = str(start.get("label", "start") or "start")
                point["status"] = "visited"
            elif order == len(path_indices) - 1:
                point["label"] = str(final.get("label", "target") or "target")
                point["status"] = "target"
            points.append(point)
        return points

    def point_to_segment_distance_cm(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:
        abx = float(bx) - float(ax)
        aby = float(by) - float(ay)
        apx = float(px) - float(ax)
        apy = float(py) - float(ay)
        denom = abx * abx + aby * aby
        if denom <= 1e-6:
            return math.hypot(apx, apy)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx = float(ax) + t * abx
        cy = float(ay) + t * aby
        return math.hypot(float(px) - cx, float(py) - cy)

    def axis_route_clearance(
        self,
        *,
        pose: Dict[str, float],
        target: Dict[str, float],
        first_axis: str,
        target_house_id: str,
        current_house_id: str = "",
    ) -> Dict[str, Any]:
        cx = float(pose["x"])
        cy = float(pose["y"])
        tx = float(target["x"])
        ty = float(target["y"])
        waypoint = {"x": tx, "y": cy} if first_axis == "x" else {"x": cx, "y": ty}
        segments = [
            {"from": {"x": cx, "y": cy}, "to": waypoint},
            {"from": waypoint, "to": {"x": tx, "y": ty}},
        ]
        ignored = {str(target_house_id or "").strip(), str(current_house_id or "").strip()}
        blockers: List[Dict[str, Any]] = []
        min_clearance: Optional[float] = None
        for house in self.house_records_for_route_planning():
            house_id = str(house.get("id", "") or "").strip()
            if not house_id or house_id in ignored:
                continue
            radius = float(house.get("radius_cm", 0.0) or 0.0) + float(LLM_ROUTE_HOUSE_CLEARANCE_CM)
            for idx, segment in enumerate(segments):
                dist = self.point_to_segment_distance_cm(
                    float(house["x"]),
                    float(house["y"]),
                    float(segment["from"]["x"]),
                    float(segment["from"]["y"]),
                    float(segment["to"]["x"]),
                    float(segment["to"]["y"]),
                )
                clearance = dist - radius
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                if clearance < 0.0:
                    blockers.append({
                        "house_id": house_id,
                        "segment_index": idx,
                        "clearance_cm": round(float(clearance), 2),
                    })
        return {
            "first_axis": first_axis,
            "segments": segments,
            "blocker_count": len(blockers),
            "min_clearance_cm": round(float(min_clearance), 2) if min_clearance is not None else None,
            "blockers": blockers[:8],
        }

    def choose_axis_route_order(
        self,
        *,
        pose: Dict[str, float],
        target: Dict[str, float],
        target_house_id: str,
        current_house_id: str = "",
    ) -> Dict[str, Any]:
        x_route = self.axis_route_clearance(
            pose=pose,
            target=target,
            first_axis="x",
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        y_route = self.axis_route_clearance(
            pose=pose,
            target=target,
            first_axis="y",
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        dx = abs(float(target["x"]) - float(pose["x"]))
        dy = abs(float(target["y"]) - float(pose["y"]))

        def route_key(route: Dict[str, Any]) -> Tuple[int, float, float]:
            min_clearance = self._as_float_or_none(route.get("min_clearance_cm"))
            return (
                -int(route.get("blocker_count", 0) or 0),
                min_clearance if min_clearance is not None else 1e9,
                dx if route.get("first_axis") == "x" else dy,
            )

        selected = x_route if route_key(x_route) >= route_key(y_route) else y_route
        fallback_axis = "x" if dx >= dy else "y"
        if int(x_route.get("blocker_count", 0) or 0) == int(y_route.get("blocker_count", 0) or 0):
            x_clearance = self._as_float_or_none(x_route.get("min_clearance_cm"))
            y_clearance = self._as_float_or_none(y_route.get("min_clearance_cm"))
            if x_clearance is not None and y_clearance is not None and abs(x_clearance - y_clearance) < 100.0:
                selected = x_route if fallback_axis == "x" else y_route
        return {
            "selected_first_axis": str(selected.get("first_axis", fallback_axis)),
            "fallback_axis": fallback_axis,
            "x_first": x_route,
            "y_first": y_route,
            "reason": "choose fewer house intersections, then larger clearance",
        }

    def target_house_standoff_waypoint(self, *, pose: Dict[str, float], house_id: str) -> Dict[str, Any]:
        bbox = self.house_world_bbox_for_id(house_id)
        if not pose or not bbox:
            return {}
        try:
            px = float(pose["x"])
            py = float(pose["y"])
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            return {}

        def clamp(value: float, lower: float, upper: float) -> float:
            return max(float(lower), min(float(upper), float(value)))

        if py < min_y:
            face_id = "south"
            boundary_x = clamp(px, min_x, max_x)
            boundary_y = min_y
            waypoint_x = boundary_x
            waypoint_y = min_y - float(LLM_ROUTE_STANDOFF_CM)
        elif py > max_y:
            face_id = "north"
            boundary_x = clamp(px, min_x, max_x)
            boundary_y = max_y
            waypoint_x = boundary_x
            waypoint_y = max_y + float(LLM_ROUTE_STANDOFF_CM)
        elif px < min_x:
            face_id = "west"
            boundary_x = min_x
            boundary_y = clamp(py, min_y, max_y)
            waypoint_x = min_x - float(LLM_ROUTE_STANDOFF_CM)
            waypoint_y = boundary_y
        elif px > max_x:
            face_id = "east"
            boundary_x = max_x
            boundary_y = clamp(py, min_y, max_y)
            waypoint_x = max_x + float(LLM_ROUTE_STANDOFF_CM)
            waypoint_y = boundary_y
        else:
            distances = {
                "south": abs(py - min_y),
                "north": abs(py - max_y),
                "west": abs(px - min_x),
                "east": abs(px - max_x),
            }
            face_id = min(distances, key=distances.get)
            if face_id == "south":
                boundary_x = clamp(px, min_x, max_x)
                boundary_y = min_y
                waypoint_x = boundary_x
                waypoint_y = min_y - float(LLM_ROUTE_STANDOFF_CM)
            elif face_id == "north":
                boundary_x = clamp(px, min_x, max_x)
                boundary_y = max_y
                waypoint_x = boundary_x
                waypoint_y = max_y + float(LLM_ROUTE_STANDOFF_CM)
            elif face_id == "west":
                boundary_x = min_x
                boundary_y = clamp(py, min_y, max_y)
                waypoint_x = min_x - float(LLM_ROUTE_STANDOFF_CM)
                waypoint_y = boundary_y
            else:
                boundary_x = max_x
                boundary_y = clamp(py, min_y, max_y)
                waypoint_x = max_x + float(LLM_ROUTE_STANDOFF_CM)
                waypoint_y = boundary_y
        return {
            "house_id": str(house_id or ""),
            "face_id": face_id,
            "x": float(waypoint_x),
            "y": float(waypoint_y),
            "boundary_point_world": {"x": round(float(boundary_x), 2), "y": round(float(boundary_y), 2)},
            "bbox_world": {
                "min_x": round(float(min_x), 2),
                "max_x": round(float(max_x), 2),
                "min_y": round(float(min_y), 2),
                "max_y": round(float(max_y), 2),
                "center_x": round(float(bbox.get("center_x", 0.5 * (min_x + max_x))), 2),
                "center_y": round(float(bbox.get("center_y", 0.5 * (min_y + max_y))), 2),
                "source": str(bbox.get("source", "") or ""),
            },
            "target_standoff_cm": float(LLM_ROUTE_STANDOFF_CM),
            "reason": "nearest target-house face standoff waypoint from map bbox and UAV pose",
        }

    def target_house_perimeter_route_points(self, waypoint: Dict[str, Any]) -> List[Dict[str, Any]]:
        bbox = waypoint.get("bbox_world", {}) if isinstance(waypoint.get("bbox_world"), dict) else {}
        try:
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            return []
        front_face = str(waypoint.get("face_id", "") or "south")
        if front_face not in {"south", "east", "north", "west"}:
            front_face = "south"
        house_id = str(waypoint.get("house_id", "") or "")

        def adaptive_standoff(face_id: str) -> Dict[str, Any]:
            default_standoff = float(waypoint.get("target_standoff_cm", LLM_ROUTE_STANDOFF_CM) or LLM_ROUTE_STANDOFF_CM)
            clearance = float(LLM_ROUTE_HOUSE_CLEARANCE_CM)
            buffer_cm = float(LLM_ROUTE_FACE_STANDOFF_BUFFER_CM)
            min_standoff = float(LLM_ROUTE_MIN_PERIMETER_STANDOFF_CM)
            limiting_gap: Optional[float] = None
            limiting_house_id = ""
            for obstacle in self.route_forbidden_house_bboxes(target_house_id=house_id, clearance_cm=0.0):
                try:
                    obs_min_x = float(obstacle["min_x"])
                    obs_max_x = float(obstacle["max_x"])
                    obs_min_y = float(obstacle["min_y"])
                    obs_max_y = float(obstacle["max_y"])
                except Exception:
                    continue
                gap: Optional[float] = None
                overlap_x = min(max_x, obs_max_x) >= max(min_x, obs_min_x)
                overlap_y = min(max_y, obs_max_y) >= max(min_y, obs_min_y)
                if face_id == "south" and obs_max_y <= min_y and overlap_x:
                    gap = min_y - obs_max_y
                elif face_id == "north" and obs_min_y >= max_y and overlap_x:
                    gap = obs_min_y - max_y
                elif face_id == "west" and obs_max_x <= min_x and overlap_y:
                    gap = min_x - obs_max_x
                elif face_id == "east" and obs_min_x >= max_x and overlap_y:
                    gap = obs_min_x - max_x
                if gap is None or gap <= 0.0:
                    continue
                if limiting_gap is None or gap < limiting_gap:
                    limiting_gap = float(gap)
                    limiting_house_id = str(obstacle.get("house_id", "") or "")
            standoff = default_standoff
            mode = "default"
            if limiting_gap is not None:
                max_safe = float(limiting_gap) - clearance - buffer_cm
                if max_safe >= min_standoff:
                    standoff = min(default_standoff, max_safe)
                    mode = "gap_limited"
                else:
                    standoff = max(20.0, min(default_standoff, float(limiting_gap) * 0.45))
                    mode = "tight_corridor"
            return {
                "face_id": face_id,
                "standoff_cm": round(float(standoff), 2),
                "mode": mode,
                "limiting_house_id": limiting_house_id,
                "limiting_gap_cm": round(float(limiting_gap), 2) if limiting_gap is not None else None,
            }

        standoff_by_face = {face_id: adaptive_standoff(face_id) for face_id in ("south", "east", "north", "west")}

        def face_points(face_id: str) -> List[Dict[str, Any]]:
            info = standoff_by_face[face_id]
            standoff = float(info.get("standoff_cm", LLM_ROUTE_STANDOFF_CM) or LLM_ROUTE_STANDOFF_CM)
            base = {
                "face_id": face_id,
                "adaptive_standoff_cm": round(float(standoff), 2),
                "standoff_mode": str(info.get("mode", "")),
                "limiting_house_id": str(info.get("limiting_house_id", "") or ""),
                "limiting_gap_cm": info.get("limiting_gap_cm"),
            }
            if face_id == "south":
                return [{**base, "x": min_x, "y": min_y - standoff}, {**base, "x": max_x, "y": min_y - standoff}]
            if face_id == "east":
                return [{**base, "x": max_x + standoff, "y": min_y}, {**base, "x": max_x + standoff, "y": max_y}]
            if face_id == "north":
                return [{**base, "x": max_x, "y": max_y + standoff}, {**base, "x": min_x, "y": max_y + standoff}]
            return [{**base, "x": min_x - standoff, "y": max_y}, {**base, "x": min_x - standoff, "y": min_y}]

        clockwise_faces = ["south", "east", "north", "west"]
        front_index = clockwise_faces.index(front_face)
        front_points = face_points(front_face)
        try:
            wx = float(waypoint["x"])
            wy = float(waypoint["y"])
            nearest_front_endpoint = min(
                range(2),
                key=lambda idx: math.hypot(float(front_points[idx]["x"]) - wx, float(front_points[idx]["y"]) - wy),
            )
        except Exception:
            nearest_front_endpoint = 0
        direction = 1 if nearest_front_endpoint == 0 else -1
        face_roles = [
            ("front", front_face),
            ("side", clockwise_faces[(front_index + direction) % 4]),
            ("back", clockwise_faces[(front_index + 2) % 4]),
            ("other_side", clockwise_faces[(front_index - direction) % 4]),
        ]
        ordered: List[Dict[str, Any]] = []
        for scan_order, (face_role, face_id) in enumerate(face_roles, start=1):
            points_for_face = face_points(face_id)
            if direction < 0:
                points_for_face = list(reversed(points_for_face))
            for endpoint_index, point in enumerate(points_for_face):
                label_suffix = "scan_start" if endpoint_index == 0 else "scan_end"
                ordered.append({
                    "x": round(float(point["x"]), 2),
                    "y": round(float(point["y"]), 2),
                    "label": f"{face_role}:{face_id}_{label_suffix}",
                    "status": "planned",
                    "face_role": face_role,
                    "face_id": face_id,
                    "scan_order": scan_order,
                    "adaptive_standoff_cm": point.get("adaptive_standoff_cm"),
                    "standoff_mode": point.get("standoff_mode"),
                    "limiting_house_id": point.get("limiting_house_id"),
                    "limiting_gap_cm": point.get("limiting_gap_cm"),
                })
        return ordered

    def target_boundary_context(self, house_id: str, pose: Dict[str, float]) -> Dict[str, Any]:
        bbox = self.house_world_bbox_for_id(house_id)
        nearest_distance = None
        nearest_point: Dict[str, float] = {}
        if pose and bbox:
            try:
                px = float(pose["x"])
                py = float(pose["y"])
                nearest_x = min(max(px, float(bbox["min_x"])), float(bbox["max_x"]))
                nearest_y = min(max(py, float(bbox["min_y"])), float(bbox["max_y"]))
                nearest_distance = math.hypot(nearest_x - px, nearest_y - py)
                nearest_point = {"x": round(nearest_x, 2), "y": round(nearest_y, 2)}
            except Exception:
                nearest_distance = None
        outside = True if nearest_distance is None else nearest_distance > float(LLM_ROUTE_STANDOFF_CM)
        return {
            "target_house_id": str(house_id or ""),
            "nearest_boundary_distance_cm": round(float(nearest_distance), 2) if nearest_distance is not None else None,
            "nearest_boundary_point_world": nearest_point,
            "target_search_standoff_cm": float(LLM_ROUTE_STANDOFF_CM),
            "outside_search_boundary": bool(outside),
        }

    def build_map_route_plan_for_target(
        self,
        *,
        house_id: str,
        pose: Dict[str, float],
        current_house_id: str = "",
        boundary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hid = str(house_id or "").strip()
        if not hid:
            return {"available": False, "active": False, "reason": "no_target_house_id"}
        if not pose:
            return {"available": False, "active": False, "target_house_id": hid, "reason": "missing_uav_pose"}
        waypoint = self.target_house_standoff_waypoint(pose=pose, house_id=hid)
        if not waypoint:
            return {"available": False, "active": False, "target_house_id": hid, "reason": "missing_target_house_bbox"}
        try:
            px = float(pose["x"])
            py = float(pose["y"])
            yaw = float(pose.get("yaw", 0.0))
            tx = float(waypoint["x"])
            ty = float(waypoint["y"])
        except Exception:
            return {"available": False, "active": False, "target_house_id": hid, "reason": "invalid_pose_or_waypoint"}

        route_choice = self.choose_axis_route_order(
            pose={"x": px, "y": py},
            target={"x": tx, "y": ty},
            target_house_id=hid,
            current_house_id=current_house_id,
        )
        first_axis = str(route_choice.get("selected_first_axis", "") or "x")
        via = {"x": tx, "y": py, "axis": "x"} if first_axis == "x" else {"x": px, "y": ty, "axis": "y"}

        def distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
            return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

        start = {"x": px, "y": py, "label": "start", "status": "visited"}
        final = {"x": tx, "y": ty, "label": f"{hid}:{waypoint.get('face_id', 'face')}", "status": "target"}
        axis_points: List[Dict[str, Any]] = [start]
        if (
            distance(start, via) > float(LLM_ROUTE_WAYPOINT_REACHED_CM)
            and distance(via, final) > float(LLM_ROUTE_WAYPOINT_REACHED_CM)
        ):
            axis_points.append({"x": float(via["x"]), "y": float(via["y"]), "label": f"via-{first_axis}", "status": "planned"})
        axis_points.append(final)
        visibility_points = self.visibility_graph_route_points(
            start=start,
            final=final,
            target_house_id=hid,
            current_house_id=current_house_id,
        )
        points: List[Dict[str, Any]] = visibility_points if visibility_points else axis_points
        route_solver = "visibility_graph" if visibility_points else "axis_aligned"
        safe_perimeter_points: List[Dict[str, Any]] = []
        for perimeter_point in self.target_house_perimeter_route_points(waypoint):
            if distance(points[-1], perimeter_point) <= float(LLM_ROUTE_WAYPOINT_REACHED_CM):
                continue
            candidate_points = points + [dict(perimeter_point)]
            report = self.route_house_violation_report(
                candidate_points,
                target_house_id=hid,
                current_house_id=current_house_id,
            )
            if not bool(report.get("valid", False)):
                break
            points.append(dict(perimeter_point))
            safe_perimeter_points.append(dict(perimeter_point))
        route_safety_report = self.route_house_violation_report(
            points,
            target_house_id=hid,
            current_house_id=current_house_id,
        )
        plan = {
            "available": True,
            "active": True,
            "mode": "map_visibility_route_to_target_house_standoff" if route_solver == "visibility_graph" else "map_axis_aligned_route_to_target_house_standoff",
            "planner_source": "deterministic_visibility_fallback",
            "route_solver": route_solver,
            "target_house_id": hid,
            "current_house_id": str(current_house_id or ""),
            "target_face_id": str(waypoint.get("face_id", "") or ""),
            "route_stage": "transit_to_standoff",
            "exploration_policy": "unknown_entry_front_then_side_then_back_then_other_side",
            "route_points": [
                {
                    "x": round(float(point["x"]), 2),
                    "y": round(float(point["y"]), 2),
                    "label": str(point.get("label", "") or ""),
                    "status": str(point.get("status", "") or ""),
                }
                for point in points
            ],
            "perimeter_route_points": safe_perimeter_points,
            "route_choice": route_choice,
            "route_safety_report": route_safety_report,
            "forbidden_house_bboxes": self.route_forbidden_house_bboxes(target_house_id=hid, current_house_id=current_house_id),
            "target_standoff_waypoint": waypoint,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return self.refresh_map_route_plan_progress(plan, pose=pose, boundary=boundary or {})

    def refresh_map_route_plan_progress(
        self,
        plan: Dict[str, Any],
        *,
        pose: Dict[str, float],
        boundary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(plan, dict) or not isinstance(pose, dict):
            return {}
        points = plan.get("route_points", []) if isinstance(plan.get("route_points"), list) else []
        valid_points: List[Dict[str, Any]] = []
        for idx, point in enumerate(points):
            if not isinstance(point, dict):
                continue
            x = self._as_float_or_none(point.get("x", point.get("world_x")))
            y = self._as_float_or_none(point.get("y", point.get("world_y")))
            if x is None or y is None:
                continue
            point_payload = dict(point)
            point_payload.update({
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "label": str(point.get("label", "") or f"wp_{idx}"),
                "status": str(point.get("status", "") or "planned"),
            })
            valid_points.append(point_payload)
        if not valid_points:
            return {}
        try:
            px = float(pose["x"])
            py = float(pose["y"])
            yaw = float(pose.get("yaw", 0.0))
        except Exception:
            return dict(plan)
        active_index = len(valid_points) - 1
        for idx, point in enumerate(valid_points):
            distance_cm = math.hypot(float(point["x"]) - px, float(point["y"]) - py)
            if distance_cm > float(LLM_ROUTE_WAYPOINT_REACHED_CM):
                active_index = idx
                break
        for idx, point in enumerate(valid_points):
            if idx < active_index:
                point["status"] = "visited"
            elif idx == active_index:
                point["status"] = "active"
            elif point.get("status") in {"visited", "active"}:
                point["status"] = "planned"
        current_waypoint = valid_points[active_index]
        dx = float(current_waypoint["x"]) - px
        dy = float(current_waypoint["y"]) - py
        distance_to_waypoint = math.hypot(dx, dy)
        desired_yaw = math.degrees(math.atan2(dy, dx)) if distance_to_waypoint > 1e-6 else yaw
        yaw_delta = self._normalize_angle_deg(desired_yaw - yaw)
        if abs(yaw_delta) > float(LLM_ROUTE_ALIGN_TOLERANCE_DEG):
            action_symbol = "e" if yaw_delta > 0.0 else "q"
            repeat = 1
            phase = "align_to_map_waypoint"
        else:
            action_symbol = "w"
            repeat = LLM_ROUTE_DEFAULT_REPEAT_CAP if distance_to_waypoint > 420.0 else 3
            phase = "advance_to_map_waypoint"
        outside_boundary = bool(boundary.get("outside_search_boundary", True)) if isinstance(boundary, dict) else True
        refreshed = dict(plan)
        refreshed["route_points"] = valid_points
        refreshed["active_waypoint_index"] = int(active_index)
        refreshed["current_waypoint"] = {
            "x": round(float(current_waypoint["x"]), 2),
            "y": round(float(current_waypoint["y"]), 2),
            "label": str(current_waypoint.get("label", "") or ""),
        }
        refreshed["distance_to_waypoint_cm"] = round(float(distance_to_waypoint), 2)
        refreshed["desired_yaw_deg"] = round(float(desired_yaw), 2)
        refreshed["current_yaw_deg"] = round(float(yaw), 2)
        refreshed["yaw_delta_deg"] = round(float(yaw_delta), 2)
        refreshed["action_symbol"] = action_symbol
        refreshed["repeat"] = int(max(1, repeat))
        refreshed["phase"] = phase
        refreshed["active"] = bool(outside_boundary and distance_to_waypoint > float(LLM_ROUTE_WAYPOINT_REACHED_CM))
        refreshed["route_stage"] = "transit_to_standoff" if refreshed["active"] else "perimeter_search_ready"
        refreshed["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return refreshed

    def active_map_route_plan_for_target(self, house_id: str, pose: Dict[str, float]) -> Dict[str, Any]:
        hid = str(house_id or "").strip()
        if not hid:
            return {}
        houses = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        current_house_id = self.find_containing_house_id(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), houses) if pose else ""
        boundary = self.target_boundary_context(hid, pose) if pose else {}
        if (
            isinstance(self.llm_route_plan, dict)
            and str(self.llm_route_plan.get("target_house_id", "") or "") == hid
            and isinstance(self.llm_route_plan.get("route_points"), list)
            and self.llm_route_plan.get("route_points")
        ):
            refreshed = self.refresh_map_route_plan_progress(self.llm_route_plan, pose=pose, boundary=boundary)
            if refreshed:
                self.llm_route_plan = refreshed
                return refreshed
        return self.build_map_route_plan_for_target(
            house_id=hid,
            pose=pose,
            current_house_id=current_house_id,
            boundary=boundary,
        )

    def refresh_active_route_plan_for_pose(self, raw_pose: Dict[str, Any]) -> Dict[str, Any]:
        pose = self.current_route_pose()
        if not pose and isinstance(raw_pose, dict):
            x = self._as_float_or_none(raw_pose.get("x"))
            y = self._as_float_or_none(raw_pose.get("y"))
            z = self._as_float_or_none(raw_pose.get("z"))
            yaw = self._as_float_or_none(raw_pose.get("task_yaw", raw_pose.get("yaw")))
            if x is not None and y is not None and yaw is not None:
                pose = {"x": x, "y": y, "z": z if z is not None else 100.0, "yaw": yaw}
        if not pose or not isinstance(self.llm_route_plan, dict) or not self.llm_route_plan.get("route_points"):
            return {}
        target_id = str(self.llm_route_plan.get("target_house_id", "") or self.selected_route_target_house_id())
        boundary = self.target_boundary_context(target_id, pose)
        refreshed = self.refresh_map_route_plan_progress(self.llm_route_plan, pose=pose, boundary=boundary)
        if refreshed:
            self.llm_route_plan = refreshed
            self.refresh_route_preview()
        return refreshed

    def route_artifact_dir(self) -> Path:
        run_dir_value = str(self.latest_state.get("run_dir", "") or "")
        session = self.session
        if not run_dir_value and session is not None and session.run_dir is not None:
            run_dir_value = str(session.run_dir)
        if run_dir_value:
            base = Path(run_dir_value)
        else:
            output_dir = self.resolve_project_path(self.output_dir_var.get().strip() or self.args.output_dir)
            base = output_dir / "route_planning"
        path = base / "llm_route"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_llm_route_vlm_image(self, context: Dict[str, Any], output_path: Path) -> Dict[str, str]:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return {"image_path": "", "image_b64": ""}
        image = None if self.map_image is None else self.map_image.copy()
        if image is None:
            return {"image_path": "", "image_b64": ""}
        target_house_id = str(context.get("target_house_id", "") or "")
        current_house_id = str(context.get("current_house_id", "") or "")
        houses = context.get("houses", []) if isinstance(context.get("houses"), list) else []
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("house_id", "") or "")
            bbox = house.get("map_bbox_image", {}) if isinstance(house.get("map_bbox_image"), dict) else {}
            x1 = self._as_float_or_none(bbox.get("x1"))
            y1 = self._as_float_or_none(bbox.get("y1"))
            x2 = self._as_float_or_none(bbox.get("x2"))
            y2 = self._as_float_or_none(bbox.get("y2"))
            if None in (x1, y1, x2, y2):
                continue
            if house_id == target_house_id:
                color = (0, 40, 255)
                width = 5
            elif house_id == current_house_id:
                color = (255, 220, 40)
                width = 4
            else:
                color = (0, 165, 255)
                width = 3
            p1 = (int(round(min(float(x1), float(x2)))), int(round(min(float(y1), float(y2)))))
            p2 = (int(round(max(float(x1), float(x2)))), int(round(max(float(y1), float(y2)))))
            cv2.rectangle(image, p1, p2, color, width)
            cv2.putText(
                image,
                f"H{house_id}",
                (p1[0], max(18, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
        pose = context.get("uav_pose", {}) if isinstance(context.get("uav_pose"), dict) else {}
        px = self._as_float_or_none(pose.get("x"))
        py = self._as_float_or_none(pose.get("y"))
        if px is not None and py is not None:
            image_point = self.world_to_image_point(float(px), float(py))
            if image_point is not None:
                ix, iy = int(round(image_point[0])), int(round(image_point[1]))
                cv2.circle(image, (ix, iy), 10, (0, 0, 255), -1)
                cv2.circle(image, (ix, iy), 15, (255, 255, 255), 2)
                cv2.putText(image, "UAV_START", (ix + 16, iy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        fallback = context.get("deterministic_fallback_route", {}) if isinstance(context.get("deterministic_fallback_route"), dict) else {}
        fallback_points = fallback.get("route_points", []) if isinstance(fallback.get("route_points"), list) else []
        fallback_image_points: List[Tuple[int, int]] = []
        for point in fallback_points:
            if not isinstance(point, dict):
                continue
            wx = self._as_float_or_none(point.get("x"))
            wy = self._as_float_or_none(point.get("y"))
            if wx is None or wy is None:
                continue
            image_point = self.world_to_image_point(float(wx), float(wy))
            if image_point is not None:
                fallback_image_points.append((int(round(image_point[0])), int(round(image_point[1]))))
        if len(fallback_image_points) >= 2:
            for idx in range(len(fallback_image_points) - 1):
                cv2.line(image, fallback_image_points[idx], fallback_image_points[idx + 1], (255, 210, 40), 4, cv2.LINE_AA)
            for idx, point in enumerate(fallback_image_points):
                cv2.circle(image, point, 8, (255, 210, 40), -1)
                cv2.putText(image, f"F{idx}", (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "Plan route through open street/yard space. Avoid non-target house boxes.",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Target house is RED. Other houses are ORANGE. Yellow path is deterministic fallback.",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return {"image_path": str(output_path), "image_b64": ""}
        return {"image_path": str(output_path), "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii")}

    def build_llm_map_route_planning_context(self, target_house_id: str) -> Dict[str, Any]:
        if not self.load_map_resources(force=not bool(self.map_config)):
            raise RuntimeError("map config/image unavailable")
        pose = self.current_route_pose()
        if not pose:
            pose = {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0}
        houses_raw = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        current_house_id = self.find_containing_house_id(float(pose["x"]), float(pose["y"]), houses_raw)
        house_records: List[Dict[str, Any]] = []
        for house in houses_raw:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("id", "") or "").strip()
            if not house_id:
                continue
            house_records.append({
                "house_id": house_id,
                "house_name": str(house.get("name", house_id) or house_id),
                "status": str(house.get("status", "") or ""),
                "center_x": self._as_float_or_none(house.get("center_x")),
                "center_y": self._as_float_or_none(house.get("center_y")),
                "radius_cm": self._as_float_or_none(house.get("radius_cm")),
                "bbox_world": self.house_world_bbox_for_id(house_id),
                "map_bbox_image": house.get("map_bbox_image", {}) if isinstance(house.get("map_bbox_image"), dict) else {},
                "is_target": house_id == str(target_house_id),
                "is_current": house_id == current_house_id,
            })
        boundary = self.target_boundary_context(str(target_house_id), pose)
        fallback_route = self.build_map_route_plan_for_target(
            house_id=str(target_house_id),
            pose=pose,
            current_house_id=current_house_id,
            boundary=boundary,
        )
        return {
            "version": "llm_map_route_planning_context_v1",
            "coordinate_system": "Unreal world centimeters. x/y are horizontal map coordinates; z is altitude and not used for waypoints.",
            "target_house_id": str(target_house_id or ""),
            "current_house_id": current_house_id,
            "uav_pose": pose,
            "target_boundary_context": boundary,
            "houses": house_records,
            "forbidden_non_target_house_bboxes": self.route_forbidden_house_bboxes(
                target_house_id=str(target_house_id),
                current_house_id=current_house_id,
            ),
            "deterministic_fallback_route": fallback_route,
            "planning_rules": [
                "Plan 3-10 high-level world-coordinate waypoints to reach a standoff point outside the target house.",
                "Do not enter or cross forbidden_non_target_house_bboxes.",
                "Prefer open streets, yards, and simple axis-aligned or gently bending paths.",
                "Because the entrance is unknown, after arrival include route points that scan the target front, side, back, and other side when safe.",
                "Return strict JSON only.",
            ],
        }

    def build_llm_route_system_prompt(self) -> str:
        return (
            "You are an overhead-map VLM route planner for a UAV searching for a house entrance. "
            "Use the annotated map image and structured world-coordinate data. "
            "Plan only high-level map waypoints, not keyboard controls. "
            "Never enter or cross non-target house boxes. Return strict JSON only."
        )

    def build_llm_route_user_prompt(self, context: Dict[str, Any]) -> str:
        return "\n".join([
            "Planning context:",
            json.dumps(context, indent=2, ensure_ascii=False),
            "",
            "Return route_points as numeric world-coordinate waypoints.",
            "Each point must include label, x, y, and optional status.",
            "The first point should be near the UAV pose; the last transit point should be outside the target house bbox.",
            "Use the attached annotated map image to choose open-space corridors.",
            "Hard constraint: no segment may enter forbidden_non_target_house_bboxes.",
            "",
            "Expected JSON shape:",
            json.dumps(LLM_ROUTE_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
        ])

    def llm_endpoint_url(self, base_url: str, suffix: str) -> str:
        base = str(base_url or "").rstrip("/")
        if not base:
            base = LLM_OPENAI_DEFAULT_BASE_URL
        if base.endswith(suffix):
            return base
        if base.endswith("/v1") and suffix.startswith("/v1/"):
            return base + suffix[3:]
        return base + suffix

    def http_post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_s) as response:
                text = response.read().decode("utf-8", errors="replace")
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc

    def call_configured_llm_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        json_schema: Optional[Dict[str, Any]] = None,
        image_b64: str = "",
    ) -> Dict[str, Any]:
        style = self.current_llm_api_style()
        base_url = self.effective_llm_base_url()
        api_key = self.effective_llm_api_key()
        model = self.effective_llm_model()
        if not base_url or not api_key or not model:
            raise RuntimeError("missing LLM API base/key/model")
        timeout_s = self.llm_route_timeout_s()
        start_time = time.time()
        if style == "openai_chat":
            endpoint = self.llm_endpoint_url(base_url, "/v1/chat/completions")
            user_content: Any = user_prompt
            if image_b64:
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "max_tokens": int(max_output_tokens),
                "response_format": {"type": "json_object"} if json_schema is not None else None,
            }
            payload = {key: value for key, value in payload.items() if value is not None}
            raw = self.http_post_json(endpoint, {"Authorization": f"Bearer {api_key}"}, payload, timeout_s)
            text = str(raw.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(raw.get("choices"), list) else "")
        elif style == "openai_responses":
            endpoint = self.llm_endpoint_url(base_url, "/v1/responses")
            content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
            if image_b64:
                content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"})
            payload = {
                "model": model,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_output_tokens": int(max_output_tokens),
            }
            raw = self.http_post_json(endpoint, {"Authorization": f"Bearer {api_key}"}, payload, timeout_s)
            text = str(raw.get("output_text", "") or "")
            if not text:
                chunks: List[str] = []
                for item in raw.get("output", []) if isinstance(raw.get("output"), list) else []:
                    for content_item in item.get("content", []) if isinstance(item.get("content"), list) else []:
                        if content_item.get("type") in {"output_text", "text"}:
                            chunks.append(str(content_item.get("text", "") or ""))
                text = "\n".join(chunks)
        else:
            endpoint = self.llm_endpoint_url(base_url or LLM_ANTHROPIC_DEFAULT_BASE_URL, "/v1/messages")
            content = [{"type": "text", "text": user_prompt}]
            if image_b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                })
            payload = {
                "model": model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": int(max_output_tokens),
                "temperature": 0,
            }
            raw = self.http_post_json(
                endpoint,
                {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                payload,
                timeout_s,
            )
            chunks = []
            for item in raw.get("content", []) if isinstance(raw.get("content"), list) else []:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "") or ""))
            text = "\n".join(chunks)
        return {
            "api_style": style,
            "model_name": model,
            "base_url": base_url,
            "latency_ms": round((time.time() - start_time) * 1000.0, 3),
            "raw_text": text,
            "raw_response": raw,
        }

    def normalize_llm_map_route_plan(self, parsed: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}
        target_house_id = str(context.get("target_house_id", "") or "").strip()
        pose = context.get("uav_pose", {}) if isinstance(context.get("uav_pose"), dict) else {}
        fallback = context.get("deterministic_fallback_route", {}) if isinstance(context.get("deterministic_fallback_route"), dict) else {}
        raw_points = parsed.get("route_points") or parsed.get("waypoints") or []
        points: List[Dict[str, Any]] = []
        if isinstance(raw_points, list):
            for idx, point in enumerate(raw_points):
                if not isinstance(point, dict):
                    continue
                x = self._as_float_or_none(point.get("x", point.get("world_x")))
                y = self._as_float_or_none(point.get("y", point.get("world_y")))
                if x is None or y is None:
                    continue
                points.append({
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "label": str(point.get("label", "") or f"llm_wp_{idx}"),
                    "status": str(point.get("status", "") or "planned"),
                })
        if pose and points:
            try:
                px = float(pose["x"])
                py = float(pose["y"])
                if math.hypot(float(points[0]["x"]) - px, float(points[0]["y"]) - py) > 120.0:
                    points.insert(0, {"x": round(px, 2), "y": round(py, 2), "label": "start", "status": "visited"})
            except Exception:
                pass
        if len(points) < 2:
            fallback_plan = dict(fallback)
            fallback_plan["planner_source"] = "deterministic_fallback_after_invalid_llm_route"
            fallback_plan["llm_route_error"] = "LLM route had fewer than two valid numeric waypoints"
            return fallback_plan
        current_house_id = str(context.get("current_house_id", "") or "")
        safety_report = self.route_house_violation_report(
            points[:16],
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        if not bool(safety_report.get("valid", False)):
            fallback_plan = dict(fallback)
            fallback_plan["planner_source"] = "deterministic_fallback_after_llm_route_crossed_house"
            fallback_plan["llm_route_safety_report"] = safety_report
            fallback_plan["llm_route_error"] = "LLM route violated forbidden non-target house bbox clearance"
            return fallback_plan
        plan = {
            "available": True,
            "active": True,
            "mode": "llm_map_route_plan",
            "planner_source": "llm",
            "target_house_id": target_house_id,
            "current_house_id": current_house_id,
            "route_name": str(parsed.get("route_name", "") or f"llm_route_to_{target_house_id}"),
            "route_stage": "transit_to_standoff",
            "route_points": points[:16],
            "perimeter_search_order": [
                str(item)
                for item in (parsed.get("perimeter_search_order", []) if isinstance(parsed.get("perimeter_search_order"), list) else [])
                if str(item or "").strip()
            ][:16],
            "preferred_facade_order": normalize_facade_order(
                parsed.get("preferred_facade_order", [])
                if isinstance(parsed.get("preferred_facade_order"), list)
                else []
            ),
            "avoid_house_ids": [
                str(item)
                for item in (parsed.get("avoid_house_ids", []) if isinstance(parsed.get("avoid_house_ids"), list) else [])
                if str(item or "").strip()
            ][:16],
            "replan_triggers": [
                str(item)
                for item in (parsed.get("replan_triggers", []) if isinstance(parsed.get("replan_triggers"), list) else [])
                if str(item or "").strip()
            ][:16],
            "reason": str(parsed.get("reason", "") or "LLM-generated overhead-map route."),
            "route_safety_report": safety_report,
            "forbidden_house_bboxes": context.get("forbidden_non_target_house_bboxes", []),
            "deterministic_fallback_route": fallback,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return self.refresh_map_route_plan_progress(
            plan,
            pose=pose,
            boundary=context.get("target_boundary_context", {}) if isinstance(context.get("target_boundary_context"), dict) else {},
        )

    def generate_llm_route_plan(self, target_house_id: str) -> Dict[str, Any]:
        artifact_dir = self.route_artifact_dir()
        context = self.build_llm_map_route_planning_context(target_house_id)
        system_prompt = self.build_llm_route_system_prompt()
        user_prompt = self.build_llm_route_user_prompt(context)
        image_payload = self.write_llm_route_vlm_image(context, artifact_dir / "map_route_vlm.jpg")
        prompt_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "planning_context": context,
            "output_schema": LLM_ROUTE_OUTPUT_SCHEMA,
            "api_style": self.current_llm_api_style(),
            "model_name": self.effective_llm_model(),
            "base_url": self.effective_llm_base_url(),
            "vlm_image_path": image_payload.get("image_path", ""),
            "vlm_image_included": bool(image_payload.get("image_b64")),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(artifact_dir / "map_route_prompt.json", "w", encoding="utf-8") as fh:
            json.dump(prompt_payload, fh, indent=2, ensure_ascii=False)
        result = self.call_configured_llm_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1300,
            json_schema=LLM_ROUTE_OUTPUT_SCHEMA,
            image_b64=str(image_payload.get("image_b64", "") or ""),
        )
        raw_text = str(result.get("raw_text", "") or "")
        parsed = extract_json_object(raw_text)
        route_plan = self.normalize_llm_map_route_plan(parsed, context)
        route_plan["vlm_image_path"] = image_payload.get("image_path", "")
        route_plan["vlm_image_used"] = bool(image_payload.get("image_b64"))
        with open(artifact_dir / "map_route_response.json", "w", encoding="utf-8") as fh:
            json.dump({**result, "parsed": parsed, "normalized_map_route_plan": route_plan}, fh, indent=2, ensure_ascii=False)
        with open(artifact_dir / "map_route_plan.json", "w", encoding="utf-8") as fh:
            json.dump(route_plan, fh, indent=2, ensure_ascii=False)
        return route_plan

    def build_task_alias_map(self, registry: Dict[str, Any]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        houses = registry.get("available_houses", []) if isinstance(registry.get("available_houses"), list) else []
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("house_id", "") or "").strip()
            if not house_id:
                continue
            aliases = house.get("aliases", []) if isinstance(house.get("aliases"), list) else []
            for alias in aliases + [house_id, str(house.get("house_name", "") or "")]:
                text = str(alias or "").strip().lower().replace("_", " ")
                if text:
                    alias_map[text] = house_id
            if house_id.isdigit():
                alias_map[str(int(house_id))] = house_id
        return alias_map

    def resolve_plan_house_id(self, value: Any, alias_map: Dict[str, str]) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        direct = raw.lower().replace("_", " ")
        if direct in alias_map:
            return alias_map[direct]
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            for candidate in (digits.zfill(3), digits):
                if candidate.lower() in alias_map:
                    return alias_map[candidate.lower()]
        return ""

    def local_task_plan_from_text(self, task_text: str, registry: Dict[str, Any]) -> Dict[str, Any]:
        alias_map = self.build_task_alias_map(registry)
        ordered: List[str] = []
        text = str(task_text or "")
        for match in re.finditer(r"(?:house|House|房屋|房子|楼|house_)?\s*0*(\d{1,3})", text):
            house_id = self.resolve_plan_house_id(match.group(1), alias_map)
            if house_id and house_id not in ordered:
                ordered.append(house_id)
        if not ordered:
            selected = self.resolve_plan_house_id(self.selected_route_target_house_id(), alias_map)
            if selected:
                ordered.append(selected)
        return {
            "version": "local_house_task_plan_v1",
            "status": "ok" if ordered else "needs_user_review",
            "plan_id": f"local_task_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "task_text": text,
            "ordered_targets": [
                {
                    "order": index,
                    "house_id": house_id,
                    "house_alias": house_id,
                    "goal": "search_entry",
                    "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
                    "status": "pending",
                }
                for index, house_id in enumerate(ordered, start=1)
            ],
            "reason": "Local fallback parsed house ids from task text.",
        }

    def build_llm_task_system_prompt(self) -> str:
        return (
            "You are a UAV task planner. Parse the user's natural language instruction "
            "into an ordered list of house entrance-search targets. Only use house ids "
            "from the provided registry. Return strict JSON only."
        )

    def build_llm_task_user_prompt(self, task_text: str, registry: Dict[str, Any]) -> str:
        return "\n".join([
            "User task:",
            str(task_text or ""),
            "",
            "House registry:",
            json.dumps(registry, indent=2, ensure_ascii=False),
            "",
            "Return a multi-house search plan. Each goal should be search_entry.",
            "If a requested house cannot be matched, return status=needs_user_review and unmatched_targets.",
            "",
            "Expected JSON shape:",
            json.dumps(LLM_TASK_PLAN_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
        ])

    def normalize_llm_task_plan(self, parsed: Dict[str, Any], task_text: str, registry: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}
        alias_map = self.build_task_alias_map(registry)
        raw_targets = parsed.get("ordered_targets", []) if isinstance(parsed.get("ordered_targets"), list) else []
        normalized_targets: List[Dict[str, Any]] = []
        unmatched = [
            str(item)
            for item in (parsed.get("unmatched_targets", []) if isinstance(parsed.get("unmatched_targets"), list) else [])
            if str(item or "").strip()
        ]
        seen: set[str] = set()
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            raw_house = item.get("house_id") or item.get("house_alias") or item.get("target") or item.get("name")
            house_id = self.resolve_plan_house_id(raw_house, alias_map)
            if not house_id:
                unmatched.append(str(raw_house or ""))
                continue
            if house_id in seen:
                continue
            seen.add(house_id)
            normalized_targets.append({
                "order": len(normalized_targets) + 1,
                "house_id": house_id,
                "house_alias": str(item.get("house_alias", raw_house) or raw_house or house_id),
                "goal": "search_entry",
                "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
                "status": "pending",
            })
        if not normalized_targets and not unmatched:
            return self.local_task_plan_from_text(task_text, registry)
        return {
            "version": "llm_multi_house_task_plan_v1",
            "status": "ok" if normalized_targets and not unmatched else "needs_user_review",
            "plan_id": str(parsed.get("plan_id", "") or f"llm_task_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "task_text": str(task_text or ""),
            "ordered_targets": normalized_targets,
            "unmatched_targets": unmatched,
            "reason": str(parsed.get("reason", "") or "Normalized from LLM task planner output."),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def first_task_house_id(self, plan: Dict[str, Any]) -> str:
        targets = plan.get("ordered_targets", []) if isinstance(plan.get("ordered_targets"), list) else []
        for item in targets:
            if isinstance(item, dict) and str(item.get("house_id", "") or "").strip():
                return str(item.get("house_id", "") or "").strip()
        return ""

    def run_llm_task_analyze(self, task_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        registry = self.house_registry_for_llm_plan()
        artifact_dir = self.route_artifact_dir()
        plan: Dict[str, Any]
        route_plan: Dict[str, Any] = {}
        if self.effective_llm_api_key() and self.effective_llm_model():
            system_prompt = self.build_llm_task_system_prompt()
            user_prompt = self.build_llm_task_user_prompt(task_text, registry)
            with open(artifact_dir / "task_plan_prompt.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "registry": registry,
                        "output_schema": LLM_TASK_PLAN_OUTPUT_SCHEMA,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )
            result = self.call_configured_llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=900,
                json_schema=LLM_TASK_PLAN_OUTPUT_SCHEMA,
            )
            parsed = extract_json_object(str(result.get("raw_text", "") or ""))
            plan = self.normalize_llm_task_plan(parsed, task_text, registry)
            with open(artifact_dir / "task_plan_response.json", "w", encoding="utf-8") as fh:
                json.dump({**result, "parsed": parsed, "normalized_task_plan": plan}, fh, indent=2, ensure_ascii=False)
        else:
            plan = self.local_task_plan_from_text(task_text, registry)
            plan["planner_source"] = "local_fallback_no_api_key"
        first_house = self.first_task_house_id(plan)
        if first_house:
            try:
                route_plan = self.generate_llm_route_plan(first_house) if self.effective_llm_api_key() else self.fallback_route_plan(first_house)
            except Exception as exc:
                route_plan = self.fallback_route_plan(first_house)
                route_plan["llm_route_generation_error"] = str(exc)
        return plan, route_plan

    def fallback_route_plan(self, target_house_id: str) -> Dict[str, Any]:
        if not self.load_map_resources(force=not bool(self.map_config)):
            return {"available": False, "active": False, "target_house_id": str(target_house_id), "reason": "map unavailable"}
        pose = self.current_route_pose() or {"x": 0.0, "y": 0.0, "z": 100.0, "yaw": 0.0}
        houses = self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []
        current_house_id = self.find_containing_house_id(float(pose["x"]), float(pose["y"]), houses)
        return self.build_map_route_plan_for_target(
            house_id=target_house_id,
            pose=pose,
            current_house_id=current_house_id,
            boundary=self.target_boundary_context(target_house_id, pose),
        )

    def route_scan_standoff_cm(self) -> float:
        try:
            return max(20.0, float(self.llm_route_standoff_cm_var.get().strip()))
        except Exception:
            return float(LLM_ROUTE_SCAN_STANDOFF_CM)

    def route_scan_spacing_cm(self) -> float:
        try:
            return max(1.0, float(self.llm_route_scan_spacing_cm_var.get().strip()))
        except Exception:
            return float(LLM_ROUTE_SCAN_SPACING_CM)

    def route_capture_count(self) -> int:
        try:
            return max(1, int(float(self.llm_route_capture_count_var.get().strip())))
        except Exception:
            return int(LLM_ROUTE_CAPTURE_COUNT)

    def target_house_capture_altitude_cm(self, house_id: str) -> float:
        hid = str(house_id or "").strip()
        if not self.map_config:
            self.load_map_resources(force=True)
        for house in self.map_config.get("houses", []) if isinstance(self.map_config.get("houses"), list) else []:
            if isinstance(house, dict) and str(house.get("id", "") or "").strip() == hid:
                altitude = self._as_float_or_none(house.get("approach_z"))
                return max(500.0, float(altitude if altitude is not None else 500.0))
        return 500.0

    def preferred_facade_order_for_route_plan(self, route_plan: Dict[str, Any]) -> List[str]:
        if not isinstance(route_plan, dict):
            return normalize_facade_order([])
        raw = route_plan.get("preferred_facade_order")
        if not isinstance(raw, list) or not raw:
            raw = route_plan.get("facade_order")
        raw_items = raw if isinstance(raw, list) else []
        has_explicit_facade = any(str(item or "").strip().lower() in {"south", "east", "north", "west"} for item in raw_items)
        if has_explicit_facade:
            return normalize_facade_order(raw_items)
        face_id = str(route_plan.get("target_face_id", "") or "").strip().lower()
        if face_id not in {"south", "east", "north", "west"}:
            waypoint = route_plan.get("target_standoff_waypoint", {}) if isinstance(route_plan.get("target_standoff_waypoint"), dict) else {}
            face_id = str(waypoint.get("face_id", "") or "").strip().lower()
        if face_id in {"south", "east", "north", "west"}:
            base = ["south", "east", "north", "west"]
            start = base.index(face_id)
            return base[start:] + base[:start]
        return normalize_facade_order([])

    def scan_corridor_standoff_by_facade(self, target_house_id: str, bbox: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        try:
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            return {}
        default_standoff = self.route_scan_standoff_cm()
        min_standoff = max(float(LLM_ROUTE_MIN_PERIMETER_STANDOFF_CM), float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)))
        clearance = min(220.0, max(120.0, default_standoff * 0.22))
        corridor_info: Dict[str, Dict[str, Any]] = {}
        obstacles = self.route_forbidden_house_bboxes(target_house_id=target_house_id, clearance_cm=0.0)

        def overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
            return min(a_max, b_max) >= max(a_min, b_min)

        for facade in ("south", "east", "north", "west"):
            best_gap: Optional[float] = None
            best_house_id = ""
            for obstacle in obstacles:
                try:
                    obs_min_x = float(obstacle["min_x"])
                    obs_max_x = float(obstacle["max_x"])
                    obs_min_y = float(obstacle["min_y"])
                    obs_max_y = float(obstacle["max_y"])
                except Exception:
                    continue
                gap: Optional[float] = None
                if facade == "south" and obs_max_y <= min_y and overlap(min_x, max_x, obs_min_x, obs_max_x):
                    gap = min_y - obs_max_y
                elif facade == "north" and obs_min_y >= max_y and overlap(min_x, max_x, obs_min_x, obs_max_x):
                    gap = obs_min_y - max_y
                elif facade == "west" and obs_max_x <= min_x and overlap(min_y, max_y, obs_min_y, obs_max_y):
                    gap = min_x - obs_max_x
                elif facade == "east" and obs_min_x >= max_x and overlap(min_y, max_y, obs_min_y, obs_max_y):
                    gap = obs_min_x - max_x
                if gap is None or gap <= 0.0:
                    continue
                if best_gap is None or float(gap) < best_gap:
                    best_gap = float(gap)
                    best_house_id = str(obstacle.get("house_id", "") or "")
            standoff = default_standoff
            mode = "open_default"
            margin = None
            safe = True
            if best_gap is not None:
                centerline = max(20.0, best_gap * 0.5)
                center_margin = max(0.0, min(centerline, best_gap - centerline))
                if best_gap > (2.0 * default_standoff + clearance):
                    standoff = default_standoff
                    mode = "wide_corridor_default_standoff"
                    margin = best_gap - default_standoff
                elif best_gap > min_standoff + clearance:
                    standoff = max(min_standoff, centerline)
                    mode = "alley_midline"
                    margin = center_margin
                else:
                    standoff = centerline
                    mode = "tight_alley_midline"
                    margin = center_margin
                safe = margin is None or margin >= clearance
            corridor_info[facade] = {
                "face_id": facade,
                "standoff_cm": round(float(standoff), 2),
                "mode": mode,
                "gap_cm": round(float(best_gap), 2) if best_gap is not None else None,
                "side_margin_cm": round(float(margin), 2) if margin is not None else None,
                "safe": bool(safe),
                "blocking_house_id": best_house_id,
                "clearance_cm": round(float(clearance), 2),
                "default_standoff_cm": round(float(default_standoff), 2),
                "min_standoff_cm": round(float(min_standoff), 2),
            }
        return corridor_info

    def build_rule_scan_points_for_route_plan(self, route_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        hid = str(route_plan.get("target_house_id", "") or self.selected_route_target_house_id()).strip()
        if not hid:
            return []
        bbox = self.house_world_bbox_for_id(hid)
        if not bbox:
            return []
        corridor_info = self.scan_corridor_standoff_by_facade(hid, bbox)
        route_plan["scan_corridor_standoff_by_facade"] = corridor_info
        return generate_rule_scan_points(
            house_id=hid,
            bbox_world=bbox,
            facade_order=self.preferred_facade_order_for_route_plan(route_plan),
            standoff_cm=self.route_scan_standoff_cm(),
            scan_spacing_cm=self.route_scan_spacing_cm(),
            altitude_cm=self.target_house_capture_altitude_cm(hid),
            facade_standoff_info=corridor_info,
            lidar_range_cm=[
                float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
                float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            ],
        )

    def route_to_standoff_points(self, route_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_points = route_plan.get("route_points", []) if isinstance(route_plan.get("route_points"), list) else []
        perimeter = route_plan.get("perimeter_route_points", []) if isinstance(route_plan.get("perimeter_route_points"), list) else []
        perimeter_labels = {
            str(point.get("label", "") or "")
            for point in perimeter
            if isinstance(point, dict) and str(point.get("label", "") or "")
        }
        transit: List[Dict[str, Any]] = []
        for idx, point in enumerate(raw_points):
            if not isinstance(point, dict):
                continue
            label = str(point.get("label", "") or "")
            if point.get("route_point_type") == "scan_point" or point.get("scan_id"):
                continue
            if label in perimeter_labels or ("_scan_" in label and ":" in label):
                continue
            item = dict(point)
            item["route_point_type"] = str(item.get("route_point_type", "") or "transit")
            item["label"] = label or f"transit_{idx}"
            transit.append(item)
        return transit

    def scan_points_as_route_points(self, scan_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        route_points: List[Dict[str, Any]] = []
        for point in scan_points:
            if not isinstance(point, dict):
                continue
            item = dict(point)
            item["label"] = str(point.get("scan_id", "") or f"scan_{len(route_points):03d}")
            item["route_point_type"] = "scan_point"
            item["status"] = str(point.get("status", "") or "planned")
            route_points.append(item)
        return route_points

    def scan_point_validation_report(self, target_house_id: str, scan_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        bbox = self.house_world_bbox_for_id(target_house_id)
        invalid: List[Dict[str, Any]] = []
        min_range = float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
        max_range = float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
        for point in scan_points:
            if not isinstance(point, dict):
                continue
            x = self._as_float_or_none(point.get("x"))
            y = self._as_float_or_none(point.get("y"))
            standoff = self._as_float_or_none(point.get("standoff_cm"))
            scan_id = str(point.get("scan_id", "") or "")
            if x is None or y is None:
                invalid.append({"scan_id": scan_id, "reason": "missing_xy"})
                continue
            if bbox and self.point_inside_open_bbox(float(x), float(y), bbox):
                invalid.append({"scan_id": scan_id, "reason": "inside_target_bbox"})
            if standoff is not None and not (min_range <= float(standoff) <= max_range):
                invalid.append({"scan_id": scan_id, "reason": "standoff_outside_lidar_range", "standoff_cm": standoff})
            if point.get("corridor_safe") is False:
                invalid.append(
                    {
                        "scan_id": scan_id,
                        "reason": "corridor_side_margin_below_clearance",
                        "corridor_side_margin_cm": point.get("corridor_side_margin_cm"),
                        "corridor_clearance_cm": point.get("corridor_clearance_cm"),
                        "blocking_house_id": point.get("corridor_blocking_house_id", ""),
                    }
                )
        return {
            "valid": not invalid and bool(scan_points),
            "scan_point_count": len(scan_points),
            "lidar_range_cm": [min_range, max_range],
            "invalid_count": len(invalid),
            "invalid": invalid[:16],
        }

    def should_validate_route_segment(self, start: Dict[str, Any], end: Dict[str, Any]) -> bool:
        start_scan = start.get("route_point_type") == "scan_point" or bool(start.get("scan_id"))
        end_scan = end.get("route_point_type") == "scan_point" or bool(end.get("scan_id"))
        if start_scan and end_scan and str(start.get("facade", "") or "") != str(end.get("facade", "") or ""):
            return False
        return True

    def route_house_violation_report_for_scan_plan(
        self,
        route_points: List[Dict[str, Any]],
        *,
        target_house_id: str,
        current_house_id: str = "",
        clearance_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        points: List[Dict[str, Any]] = []
        for point in route_points if isinstance(route_points, list) else []:
            if not isinstance(point, dict):
                continue
            x = self._as_float_or_none(point.get("x", point.get("world_x")))
            y = self._as_float_or_none(point.get("y", point.get("world_y")))
            if x is None or y is None:
                continue
            item = dict(point)
            item["x"] = float(x)
            item["y"] = float(y)
            points.append(item)
        obstacles = self.route_forbidden_house_bboxes(
            target_house_id=target_house_id,
            current_house_id=current_house_id,
            clearance_cm=clearance_cm,
        )
        violations: List[Dict[str, Any]] = []
        skipped_segments = 0
        for segment_index in range(max(0, len(points) - 1)):
            start = points[segment_index]
            end = points[segment_index + 1]
            if not self.should_validate_route_segment(start, end):
                skipped_segments += 1
                continue
            for obstacle in obstacles:
                if self.segment_intersects_open_bbox(start["x"], start["y"], end["x"], end["y"], obstacle):
                    violations.append(
                        {
                            "segment_index": segment_index,
                            "house_id": str(obstacle.get("house_id", "") or ""),
                            "segment_start": {"x": round(float(start["x"]), 2), "y": round(float(start["y"]), 2)},
                            "segment_end": {"x": round(float(end["x"]), 2), "y": round(float(end["y"]), 2)},
                        }
                    )
        return {
            "valid": bool(len(points) >= 2 and not violations),
            "point_count": len(points),
            "checked_house_count": len(obstacles),
            "clearance_cm": float(LLM_ROUTE_HOUSE_CLEARANCE_CM if clearance_cm is None else clearance_cm),
            "skipped_scan_transition_segments": skipped_segments,
            "violation_count": len(violations),
            "violations": violations[:12],
        }

    def target_house_route_violation_report(self, target_house_id: str, route_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        bbox = self.house_world_bbox_for_id(target_house_id)
        if not bbox:
            return {"valid": False, "reason": "missing_target_bbox", "violation_count": 0, "violations": []}
        points: List[Dict[str, Any]] = []
        for point in route_points if isinstance(route_points, list) else []:
            if not isinstance(point, dict):
                continue
            x = self._as_float_or_none(point.get("x", point.get("world_x")))
            y = self._as_float_or_none(point.get("y", point.get("world_y")))
            if x is not None and y is not None:
                item = dict(point)
                item["x"] = float(x)
                item["y"] = float(y)
                points.append(item)
        violations: List[Dict[str, Any]] = []
        skipped_segments = 0
        for idx in range(max(0, len(points) - 1)):
            start = points[idx]
            end = points[idx + 1]
            if not self.should_validate_route_segment(start, end):
                skipped_segments += 1
                continue
            if self.segment_intersects_open_bbox(start["x"], start["y"], end["x"], end["y"], bbox):
                violations.append(
                    {
                        "segment_index": idx,
                        "segment_start": {"x": round(start["x"], 2), "y": round(start["y"], 2)},
                        "segment_end": {"x": round(end["x"], 2), "y": round(end["y"], 2)},
                    }
                )
        return {
            "valid": bool(len(points) >= 2 and not violations),
            "point_count": len(points),
            "skipped_scan_transition_segments": skipped_segments,
            "violation_count": len(violations),
            "violations": violations[:12],
        }

    def make_house_search_output_dir(self, target_house_id: str) -> Path:
        session = self.session
        if session is not None and session.run_dir is not None:
            root = Path(session.run_dir) / "house_search"
        else:
            root = self.resolve_project_path("results/house_search")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_run_{timestamp}"
        root.mkdir(parents=True, exist_ok=True)
        candidate = root / base_name
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = root / f"{base_name}_{suffix}"
        (candidate / "frames").mkdir(parents=True, exist_ok=True)
        (candidate / "reconstruction").mkdir(parents=True, exist_ok=True)
        return candidate

    def write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def build_llm_route_strategy_payload(
        self,
        route_plan: Dict[str, Any],
        scan_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        target_house_id = str(route_plan.get("target_house_id", "") or "")
        return {
            "target_house_order": [target_house_id] if target_house_id else [],
            "route_strategy": str(route_plan.get("mode", "") or route_plan.get("route_strategy", "") or "map_visibility_route_to_target_house_standoff"),
            "preferred_facade_order": self.preferred_facade_order_for_route_plan(route_plan),
            "scan_strategy": "perimeter_scan_with_optional_rescan",
            "capture_priority": ["front_face_view", "side_oblique_view", "back_face_view", "coverage_gap_rescan"],
            "avoid_house_ids": route_plan.get("avoid_house_ids", []),
            "replan_triggers": route_plan.get("replan_triggers", []),
            "planner_source": str(route_plan.get("planner_source", "") or "unknown"),
            "scan_point_count": len(scan_points),
            "reason": str(route_plan.get("reason", "") or "Rule-generated facade scan route from target house bbox."),
        }

    def prepare_route_plan_with_scan_points(self, route_plan: Dict[str, Any]) -> Dict[str, Any]:
        plan = dict(route_plan if isinstance(route_plan, dict) else {})
        target_house_id = str(plan.get("target_house_id", "") or self.selected_route_target_house_id()).strip()
        if not target_house_id:
            return plan
        plan["target_house_id"] = target_house_id
        scan_points = self.build_rule_scan_points_for_route_plan(plan)
        transit_points = self.route_to_standoff_points(plan)
        scan_route_points = self.scan_points_as_route_points(scan_points)
        route_points = transit_points + scan_route_points
        current_house_id = str(plan.get("current_house_id", "") or "")
        route_safety_report = self.route_house_violation_report_for_scan_plan(
            route_points,
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        target_bbox_report = self.target_house_route_violation_report(target_house_id, route_points)
        scan_validation = self.scan_point_validation_report(target_house_id, scan_points)
        blocked = (
            not bool(route_safety_report.get("valid", False))
            or not bool(target_bbox_report.get("valid", False))
            or not bool(scan_validation.get("valid", False))
        )
        house_search_dir = self.make_house_search_output_dir(target_house_id)
        execution_summary = {
            "house_search_dir": str(house_search_dir),
            "target_house_id": target_house_id,
            "planned_scan_count": len(scan_points),
            "completed_scan_count": 0,
            "capture_success_count": 0,
            "capture_failure_count": 0,
            "merged_point_count": 0,
            "route_blocked_by_safety": bool(blocked),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        plan.update(
            {
                "house_search_dir": str(house_search_dir),
                "route_to_standoff_points": transit_points,
                "route_points": route_points,
                "scan_points": scan_points,
                "scan_point_count": len(scan_points),
                "scan_standoff_cm": self.route_scan_standoff_cm(),
                "scan_spacing_cm": self.route_scan_spacing_cm(),
                "scan_corridor_standoff_by_facade": plan.get("scan_corridor_standoff_by_facade", {}),
                "scan_corridor_policy": "use alley centerline between target and nearest overlapping non-target house; mark unsafe if side margin is below clearance",
                "capture_count_per_scan_point": self.route_capture_count(),
                "preferred_facade_order": self.preferred_facade_order_for_route_plan(plan),
                "route_safety_report": route_safety_report,
                "target_bbox_route_report": target_bbox_report,
                "scan_point_validation_report": scan_validation,
                "route_blocked_by_safety": bool(blocked),
                "active": bool(plan.get("active", True)) and not blocked,
            }
        )
        strategy_payload = self.build_llm_route_strategy_payload(plan, scan_points)
        self.write_json_artifact(house_search_dir / "llm_route_strategy.json", strategy_payload)
        self.write_json_artifact(house_search_dir / "normalized_route_plan.json", plan)
        self.write_json_artifact(
            house_search_dir / "scan_points.json",
            {
                "schema": LLM_ROUTE_SCAN_POINTS_SCHEMA,
                "target_house_id": target_house_id,
                "scan_points": scan_points,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self.write_json_artifact(house_search_dir / "execution_summary.json", execution_summary)
        self.llm_route_scan_points = scan_points
        self.llm_route_lidar_trajectory = []
        self.llm_route_execution_summary = execution_summary
        self.llm_route_validation_report = {}
        self.house_search_dir = house_search_dir
        return plan

    def refresh_llm_route_map(self) -> None:
        widget = getattr(self, "llm_route_map_widget", None)
        if widget is None:
            return
        try:
            if not self.load_map_resources(force=not bool(self.map_config)):
                self.llm_route_map_status_var.set("Route Map: map unavailable")
                return
            pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
            pose_x = float(pose.get("x", 0.0)) if pose else 0.0
            pose_y = float(pose.get("y", 0.0)) if pose else 0.0
            pose_yaw = float(pose.get("task_yaw", pose.get("yaw", 0.0))) if pose else 0.0
            houses, boxes = self.build_map_display(pose)
            widget.set_background_image(self.map_image)
            widget.set_calibration(self.map_calibration.get("affine_world_to_image"), self.map_image_size(), [])
            widget.set_image_layer_offset(*self.map_display_offset_px)
            widget.set_house_boxes(boxes)
            widget.update_houses([])
            widget.update_uav(pose_x, pose_y, pose_yaw)
            route_plan = self.llm_route_plan if isinstance(self.llm_route_plan, dict) else {}
            widget.set_route_plan(route_plan)
            scan_count = len(self.llm_route_scan_points) if isinstance(self.llm_route_scan_points, list) else 0
            route_count = len(route_plan.get("route_points", [])) if isinstance(route_plan.get("route_points"), list) else 0
            self.llm_route_map_status_var.set(f"Route Map: houses={len(houses)} route_points={route_count} scan_points={scan_count}")
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM route map failed: %s", exc)
            self.llm_route_map_status_var.set(f"Route Map: failed: {exc}")

    def apply_route_plan(self, route_plan: Dict[str, Any], *, status_prefix: str = "LLM Route") -> None:
        self.llm_route_plan = self.prepare_route_plan_with_scan_points(route_plan) if isinstance(route_plan, dict) else {}
        target_house_id = str(self.llm_route_plan.get("target_house_id", "") or "")
        if target_house_id:
            self.set_selected_route_target_house(target_house_id)
        self.refresh_route_preview()
        self.refresh_map_once()
        point_count = len(self.llm_route_plan.get("route_points", [])) if isinstance(self.llm_route_plan.get("route_points"), list) else 0
        source = str(self.llm_route_plan.get("planner_source", "") or "unknown")
        if self.llm_route_plan.get("route_blocked_by_safety"):
            message = "route blocked by safety validation"
        else:
            message = str(self.llm_route_plan.get("llm_route_error", "") or self.llm_route_plan.get("reason", "") or "")
        suffix = f" | {message}" if message else ""
        self.llm_route_status_var.set(f"{status_prefix}: house={target_house_id or '-'} source={source} points={point_count}{suffix}")
        self.refresh_llm_route_map()

    def refresh_route_preview(self) -> None:
        payload = {
            "task_plan": self.llm_task_plan if isinstance(self.llm_task_plan, dict) else {},
            "route_plan": self.llm_route_plan if isinstance(self.llm_route_plan, dict) else {},
            "scan_points": self.llm_route_scan_points if isinstance(self.llm_route_scan_points, list) else [],
            "execution_summary": self.llm_route_execution_summary if isinstance(self.llm_route_execution_summary, dict) else {},
            "validation_report": self.llm_route_validation_report if isinstance(self.llm_route_validation_report, dict) else {},
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        live_texts: List[tk.Text] = []
        for preview_text in list(getattr(self, "llm_route_preview_texts", [])):
            try:
                if not preview_text.winfo_exists():
                    continue
                preview_text.configure(state="normal")
                preview_text.delete("1.0", "end")
                preview_text.insert("1.0", text)
                preview_text.configure(state="disabled")
                live_texts.append(preview_text)
            except tk.TclError:
                pass
        self.llm_route_preview_texts = live_texts
        self.llm_route_preview_text = live_texts[0] if live_texts else None

    def on_llm_task_analyze(self) -> None:
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route_status_var.set("LLM Route: wait for current route worker.")
            return
        task_text = self.llm_task_text_var.get().strip()
        if not task_text:
            self.llm_route_status_var.set("LLM Route: task text is empty.")
            return

        def worker() -> None:
            self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: analyzing task..."))
            try:
                plan, route_plan = self.run_llm_task_analyze(task_text)
                self.root.after(0, lambda p=plan, r=route_plan: self.apply_task_and_route_plan(p, r))
            except Exception as exc:
                LOGGER.warning("LLM task analyze failed: %s", exc)
                self.root.after(0, lambda e=exc: self.llm_route_status_var.set(f"LLM Route: task analyze failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def apply_task_and_route_plan(self, plan: Dict[str, Any], route_plan: Dict[str, Any]) -> None:
        self.llm_task_plan = plan if isinstance(plan, dict) else {}
        if isinstance(route_plan, dict) and route_plan:
            self.apply_route_plan(route_plan, status_prefix="LLM Task+Route")
        else:
            self.refresh_route_preview()
        first_house = self.first_task_house_id(self.llm_task_plan)
        if first_house:
            self.set_selected_route_target_house(first_house)
        targets = self.llm_task_plan.get("ordered_targets", []) if isinstance(self.llm_task_plan.get("ordered_targets"), list) else []
        ids = [str(item.get("house_id", "") or "") for item in targets if isinstance(item, dict)]
        if not route_plan:
            self.llm_route_status_var.set(f"LLM Task: {' -> '.join(ids) if ids else 'no target'}")

    def on_llm_route_plan(self) -> None:
        target_house_id = self.selected_route_target_house_id()
        if not target_house_id:
            self.llm_route_status_var.set("LLM Route: no target house selected.")
            return

        def worker() -> None:
            self.root.after(0, lambda hid=target_house_id: self.llm_route_status_var.set(f"LLM Route: planning house {hid}..."))
            try:
                if not self.effective_llm_api_key():
                    plan = self.fallback_route_plan(target_house_id)
                    plan["planner_source"] = "deterministic_fallback_no_api_key"
                    self.root.after(0, lambda p=plan: self.apply_route_plan(p, status_prefix="Fallback Route"))
                    return
                plan = self.generate_llm_route_plan(target_house_id)
                self.root.after(0, lambda p=plan: self.apply_route_plan(p, status_prefix="LLM Route"))
            except Exception as exc:
                LOGGER.warning("LLM route plan failed: %s", exc)
                try:
                    fallback = self.fallback_route_plan(target_house_id)
                    fallback["llm_route_generation_error"] = str(exc)
                    self.root.after(0, lambda p=fallback: self.apply_route_plan(p, status_prefix="Fallback Route"))
                except Exception as fallback_exc:
                    self.root.after(0, lambda e=fallback_exc: self.llm_route_status_var.set(f"LLM Route failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def on_fallback_route_plan(self) -> None:
        target_house_id = self.selected_route_target_house_id()
        if not target_house_id:
            self.llm_route_status_var.set("LLM Route: no target house selected.")
            return
        try:
            self.apply_route_plan(self.fallback_route_plan(target_house_id), status_prefix="Fallback Route")
        except Exception as exc:
            self.llm_route_status_var.set(f"Fallback Route failed: {exc}")

    def on_clear_route_plan(self) -> None:
        self.route_stop_event.set()
        self.llm_route_plan = {}
        self.llm_task_plan = {}
        self.llm_route_scan_points = []
        self.llm_route_lidar_trajectory = []
        self.llm_route_execution_summary = {}
        self.llm_route_validation_report = {}
        self.house_search_dir = None
        self.refresh_route_preview()
        self.refresh_llm_route_map()
        self.refresh_map_once()
        self.llm_route_status_var.set("LLM Route: cleared.")

    def route_step_cm(self) -> float:
        try:
            return max(20.0, float(self.route_step_cm_var.get().strip()))
        except Exception:
            return 120.0

    def route_delay_s(self) -> float:
        try:
            return max(0.0, float(self.route_delay_ms_var.get().strip()) / 1000.0)
        except Exception:
            return 0.1

    def route_points_to_follow(self, *, auto: bool) -> List[Dict[str, Any]]:
        pose = self.current_route_pose()
        if pose and isinstance(self.llm_route_plan, dict):
            self.llm_route_plan = self.refresh_map_route_plan_progress(
                self.llm_route_plan,
                pose=pose,
                boundary=self.target_boundary_context(str(self.llm_route_plan.get("target_house_id", "")), pose),
            )
        points = self.llm_route_plan.get("route_points", []) if isinstance(self.llm_route_plan.get("route_points"), list) else []
        if not points:
            return []
        active_index = int(self.llm_route_plan.get("active_waypoint_index", 0) or 0)
        active_index = max(0, min(active_index, len(points) - 1))
        selected = points[active_index:] if auto else points[active_index:active_index + 1]
        result: List[Dict[str, Any]] = []
        for offset, point in enumerate(selected):
            if not isinstance(point, dict):
                continue
            item = dict(point)
            item["_route_point_index"] = active_index + offset
            result.append(item)
        return result

    def on_follow_route_next(self) -> None:
        self.start_route_follow(auto=False)

    def on_follow_route_auto(self) -> None:
        self.start_route_follow(auto=True)

    def on_stop_route_follow(self) -> None:
        self.route_stop_event.set()
        self.llm_route_status_var.set("LLM Route: stopping follow...")

    def on_direct_scan_capture_test(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route_status_var.set("LLM Route: worker already running.")
            return
        if not self.llm_route_scan_points:
            target_house_id = self.selected_route_target_house_id()
            if not target_house_id:
                self.llm_route_status_var.set("LLM Route: no target house selected.")
                return
            try:
                plan = self.fallback_route_plan(target_house_id)
                plan["planner_source"] = "direct_capture_fallback_no_route"
                self.apply_route_plan(plan, status_prefix="Direct Capture Setup")
            except Exception as exc:
                self.llm_route_status_var.set(f"Direct Capture setup failed: {exc}")
                return
        target_house_id = str(self.llm_route_plan.get("target_house_id", "") or self.selected_route_target_house_id())
        scan_validation = self.scan_point_validation_report(target_house_id, self.llm_route_scan_points)
        if not bool(scan_validation.get("valid", False)):
            self.llm_route_status_var.set("Direct Capture: scan point validation failed.")
            self.llm_route_validation_report = {"scan_point_validation_report": scan_validation, "overall_passed": False}
            self.refresh_route_preview()
            return
        self.sync_capture_options_to_session(session)
        self.route_stop_event.clear()
        self.route_thread = threading.Thread(
            target=lambda: self.direct_scan_capture_worker(session),
            daemon=True,
        )
        self.route_thread.start()

    def direct_scan_capture_worker(self, session: flight.DroneFlightSession) -> None:
        scan_points = self.scan_points_as_route_points(self.llm_route_scan_points)
        route_points = self.llm_route_plan.get("route_points", []) if isinstance(self.llm_route_plan.get("route_points"), list) else []
        index_by_scan_id = {
            str(point.get("scan_id", "") or ""): idx
            for idx, point in enumerate(route_points)
            if isinstance(point, dict) and str(point.get("scan_id", "") or "")
        }
        total = len(scan_points)
        captured_any = False
        self.root.after(0, lambda: self.llm_route_status_var.set(f"Direct Capture: starting {total} scan points..."))
        for point_index, point in enumerate(scan_points, start=1):
            if self.route_stop_event.is_set():
                self.root.after(0, lambda: self.llm_route_status_var.set("Direct Capture: stopped."))
                return
            scan_id = str(point.get("scan_id", "") or "")
            if scan_id in index_by_scan_id:
                point["_route_point_index"] = index_by_scan_id[scan_id]
            capture_entry = self.capture_at_scan_route_point(
                session,
                point,
                point_index=point_index,
                total=total,
            )
            captured_any = captured_any or capture_entry.get("capture_status") == "ok"
            self.root.after(0, self.refresh_llm_route_map)
        if self.route_stop_event.is_set():
            self.root.after(0, lambda: self.llm_route_status_var.set("Direct Capture: stopped."))
            return
        if captured_any:
            try:
                finalize_result = self.finalize_house_search_outputs()
                validation = self.validate_house_search_data(finalize_result=finalize_result)
                self.root.after(
                    0,
                    lambda v=validation: self.llm_route_status_var.set(
                        f"Direct Capture: complete, validation={'PASS' if v.get('overall_passed') else 'CHECK'}"
                    ),
                )
            except Exception as exc:
                LOGGER.warning("Direct capture finalize/validate failed: %s", exc)
                self.root.after(0, lambda e=exc: self.llm_route_status_var.set(f"Direct Capture: finalize failed: {e}"))
        else:
            self.root.after(0, lambda: self.llm_route_status_var.set("Direct Capture: no captures completed."))

    def on_validate_house_search_data(self) -> None:
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route_status_var.set("Validate Data: wait for current worker.")
            return

        def worker() -> None:
            self.root.after(0, lambda: self.llm_route_status_var.set("Validate Data: running..."))
            try:
                validation = self.validate_house_search_data()
                self.root.after(
                    0,
                    lambda v=validation: self.llm_route_status_var.set(
                        f"Validate Data: {'PASS' if v.get('overall_passed') else 'CHECK'} -> {v.get('validation_report_path', '')}"
                    ),
                )
                self.root.after(0, self.refresh_route_preview)
                self.root.after(0, self.refresh_llm_route_map)
            except Exception as exc:
                LOGGER.warning("Validate house search data failed: %s", exc)
                self.root.after(0, lambda e=exc: self.llm_route_status_var.set(f"Validate Data failed: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def start_route_follow(self, *, auto: bool) -> None:
        session = self.active_session()
        if session is None:
            return
        if not isinstance(self.llm_route_plan, dict) or not self.llm_route_plan.get("route_points"):
            self.on_fallback_route_plan()
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route_status_var.set("LLM Route: follow already running.")
            return
        if isinstance(self.llm_route_plan, dict) and self.llm_route_plan.get("route_blocked_by_safety"):
            self.llm_route_status_var.set("LLM Route: blocked by route/scan safety validation.")
            self.refresh_route_preview()
            return
        self.sync_capture_options_to_session(session)
        points = self.route_points_to_follow(auto=auto)
        if not points:
            self.llm_route_status_var.set("LLM Route: no route point to follow.")
            return
        self.route_stop_event.clear()
        self.route_thread = threading.Thread(
            target=lambda: self.follow_route_worker(session, points, auto=auto),
            daemon=True,
        )
        self.route_thread.start()

    def is_scan_route_point(self, point: Dict[str, Any]) -> bool:
        return bool(
            isinstance(point, dict)
            and (point.get("route_point_type") == "scan_point" or str(point.get("scan_id", "") or ""))
        )

    def append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def house_search_output_dir_for_active_plan(self) -> Optional[Path]:
        raw = ""
        if isinstance(self.llm_route_plan, dict):
            raw = str(self.llm_route_plan.get("house_search_dir", "") or "")
        if not raw and self.house_search_dir is not None:
            raw = str(self.house_search_dir)
        if not raw:
            target_house_id = self.selected_route_target_house_id()
            if not target_house_id:
                return None
            self.house_search_dir = self.make_house_search_output_dir(target_house_id)
            raw = str(self.house_search_dir)
            if isinstance(self.llm_route_plan, dict):
                self.llm_route_plan["house_search_dir"] = raw
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        (path / "reconstruction").mkdir(parents=True, exist_ok=True)
        return path

    def update_route_point_runtime_status(self, point: Dict[str, Any], status: str) -> None:
        index = point.get("_route_point_index")
        try:
            idx = int(index)
        except Exception:
            idx = -1
        if idx >= 0 and isinstance(self.llm_route_plan, dict):
            points = self.llm_route_plan.get("route_points", [])
            if isinstance(points, list) and idx < len(points) and isinstance(points[idx], dict):
                points[idx]["status"] = status
        scan_id = str(point.get("scan_id", "") or "")
        if scan_id:
            for scan_point in self.llm_route_scan_points:
                if isinstance(scan_point, dict) and str(scan_point.get("scan_id", "") or "") == scan_id:
                    scan_point["status"] = status

    def write_house_search_lidar_summary(self, search_dir: Path, *, running: bool) -> None:
        summary = {
            "capture_kind": "house_search_lidar",
            "task_title": self.llm_task_text_var.get().strip() or "house_search",
            "stream_dir": str(search_dir),
            "frames_dir": str(search_dir / "frames"),
            "reconstruction_dir": str(search_dir / "reconstruction"),
            "running": bool(running),
            "frame_count": len(self.llm_route_lidar_trajectory),
            "lidar_depth_min_cm": float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
            "lidar_depth_max_cm": float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            "lidar_depth_projection": str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
            "lidar_capture_processing": self.lidar_capture_processing_mode(),
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        (search_dir / "stream_capture_lidar.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        trajectory_payload = dict(summary)
        trajectory_payload["trajectory"] = self.llm_route_lidar_trajectory
        (search_dir / "trajectory.json").write_text(json.dumps(trajectory_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def capture_at_scan_route_point(
        self,
        session: flight.DroneFlightSession,
        point: Dict[str, Any],
        *,
        point_index: int,
        total: int,
    ) -> Dict[str, Any]:
        search_dir = self.house_search_output_dir_for_active_plan()
        if search_dir is None:
            return {"capture_status": "failed", "error": "missing_house_search_dir"}
        scan_id = str(point.get("scan_id", "") or point.get("label", f"scan_{point_index:03d}"))
        planned_pose = {
            "x": float(point.get("x", 0.0)),
            "y": float(point.get("y", 0.0)),
            "z": float(point.get("z", self.target_house_capture_altitude_cm(str(point.get("house_id", ""))))),
            "yaw": float(point.get("yaw_deg", point.get("yaw", 0.0))),
        }
        self.root.after(
            0,
            lambda i=point_index, t=total, sid=scan_id: self.llm_route_status_var.set(f"LLM Route: align/capture {i}/{t} {sid}"),
        )
        set_result = self.safe("Route scan point align", lambda: session.set_pose(planned_pose))
        if isinstance(set_result, dict):
            self.root.after(0, lambda r=set_result: self.apply_state(r))
        if self.route_stop_event.wait(0.3):
            return {"capture_status": "stopped", "scan_id": scan_id}
        capture_results: List[Dict[str, Any]] = []
        capture_count = self.route_capture_count()
        for capture_index in range(capture_count):
            if self.route_stop_event.is_set():
                break
            frame_index = len(self.llm_route_lidar_trajectory) + 1
            action_detail = dict(self.build_stream_action_detail())
            action_detail.update(
                {
                    "source": "llm_house_search_route",
                    "scan_id": scan_id,
                    "house_id": str(point.get("house_id", "") or self.selected_route_target_house_id()),
                    "facade": str(point.get("facade", "") or ""),
                    "capture_index": capture_index + 1,
                    "capture_count": capture_count,
                    "planned_pose": planned_pose,
                }
            )
            result = self.safe(
                "Route scan lidar capture",
                lambda idx=frame_index, action=action_detail: session.capture_lidar_stream_frame(
                    search_dir,
                    idx,
                    action_detail=action,
                ),
            )
            if not isinstance(result, dict):
                continue
            capture_results.append(result)
            trajectory_entry = {
                "frame_index": int(result.get("frame_index", frame_index)),
                "capture_time": result.get("capture_time", ""),
                "scan_id": scan_id,
                "house_id": str(point.get("house_id", "") or ""),
                "facade": str(point.get("facade", "") or ""),
                "planned_pose": planned_pose,
                "pose": result.get("pose", {}),
                "commanded_pose": result.get("commanded_pose", {}),
                "actual_pose": result.get("actual_pose", {}),
                "pose_error": result.get("pose_error", {}),
                "action_detail": action_detail,
                "capture_dir": result.get("capture_dir", ""),
                "rgb_path": result.get("rgb_path", ""),
                "point_cloud_world_standard_m_npy_path": result.get("point_cloud_world_standard_m_npy_path", ""),
                "point_cloud_world_standard_m_ply_path": result.get("point_cloud_world_standard_m_ply_path", ""),
                "point_cloud_preview_path": result.get("point_cloud_preview_path", ""),
                "point_count": int(result.get("point_count", 0) or 0),
                "raw_capture_only": bool(result.get("raw_capture_only", self.lidar_capture_processing_mode() == "smooth")),
                "postprocess_status": result.get("postprocess_status", "pending"),
            }
            self.llm_route_lidar_trajectory.append(trajectory_entry)
            self.append_jsonl(search_dir / "lidar_capture_log.jsonl", trajectory_entry)
            self.write_house_search_lidar_summary(search_dir, running=True)
            self.root.after(
                0,
                lambda sid=scan_id, c=len(self.llm_route_lidar_trajectory): self.llm_route_status_var.set(
                    f"LLM Route: captured {sid} frames={c}"
                ),
            )
        state = self.safe("Route scan final state", session.get_state)
        actual_pose = state.get("pose", {}) if isinstance(state, dict) and isinstance(state.get("pose"), dict) else {}
        ax = self._as_float_or_none(actual_pose.get("x"))
        ay = self._as_float_or_none(actual_pose.get("y"))
        ayaw = self._as_float_or_none(actual_pose.get("task_yaw", actual_pose.get("yaw")))
        position_error = (
            math.hypot(float(ax) - planned_pose["x"], float(ay) - planned_pose["y"])
            if ax is not None and ay is not None
            else None
        )
        yaw_error = self._normalize_angle_deg(float(ayaw) - planned_pose["yaw"]) if ayaw is not None else None
        capture_status = "ok" if capture_results else "failed"
        execution_entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "house_id": str(point.get("house_id", "") or ""),
            "scan_id": scan_id,
            "facade": str(point.get("facade", "") or ""),
            "planned_pose": planned_pose,
            "actual_pose": actual_pose,
            "position_error_cm": round(float(position_error), 3) if position_error is not None else None,
            "yaw_error_deg": round(float(yaw_error), 3) if yaw_error is not None else None,
            "safety_state": "SAFE",
            "capture_status": capture_status,
            "capture_count": len(capture_results),
            "capture_dirs": [str(item.get("capture_dir", "") or "") for item in capture_results],
            "point_cloud_paths": [
                str(item.get("point_cloud_world_standard_m_ply_path", "") or item.get("point_cloud_world_ply_path", "") or "")
                for item in capture_results
            ],
        }
        self.append_jsonl(search_dir / "scan_execution_log.jsonl", execution_entry)
        if capture_status == "ok":
            self.update_route_point_runtime_status(point, "captured")
        else:
            self.update_route_point_runtime_status(point, "capture_failed")
        completed = sum(1 for item in self.llm_route_scan_points if isinstance(item, dict) and item.get("status") in {"captured", "visited"})
        self.llm_route_execution_summary.update(
            {
                "completed_scan_count": int(completed),
                "capture_success_count": int(self.llm_route_execution_summary.get("capture_success_count", 0) or 0) + (1 if capture_status == "ok" else 0),
                "capture_failure_count": int(self.llm_route_execution_summary.get("capture_failure_count", 0) or 0) + (0 if capture_status == "ok" else 1),
                "last_scan_id": scan_id,
                "last_capture_status": capture_status,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.write_json_artifact(search_dir / "execution_summary.json", self.llm_route_execution_summary)
        self.root.after(0, self.refresh_route_preview)
        return execution_entry

    def build_house_search_coverage_report(self, postprocess_result: Dict[str, Any]) -> Dict[str, Any]:
        target_house_id = str(self.llm_route_plan.get("target_house_id", "") or self.selected_route_target_house_id())
        bbox = self.house_world_bbox_for_id(target_house_id)
        spacing = self.route_scan_spacing_cm()
        standoff = self.route_scan_standoff_cm()
        reconstruction = postprocess_result.get("reconstruction", {}) if isinstance(postprocess_result.get("reconstruction"), dict) else {}
        merged_count = int(postprocess_result.get("merged_point_count", reconstruction.get("merged_point_count", 0)) or 0)
        cloud_path = str(
            reconstruction.get("merged_point_cloud_world_standard_m_npy_path")
            or postprocess_result.get("merged_point_cloud_world_standard_m_npy_path", "")
            or ""
        )
        facades: Dict[str, Dict[str, Any]] = {}
        for facade in ("south", "east", "north", "west"):
            planned = [p for p in self.llm_route_scan_points if isinstance(p, dict) and p.get("facade") == facade]
            captured = [p for p in planned if p.get("status") in {"captured", "visited"}]
            facades[facade] = {
                "planned_scan_count": len(planned),
                "captured_scan_count": len(captured),
                "scan_completion_ratio": round(len(captured) / max(1, len(planned)), 4),
                "point_cloud_coverage": None,
                "covered_cells": 0,
                "total_cells": 0,
            }
        if bbox and cloud_path and Path(cloud_path).exists():
            try:
                cloud = np.load(cloud_path, mmap_mode="r")
                if getattr(cloud, "ndim", 0) == 2 and cloud.shape[1] >= 3 and cloud.shape[0] > 0:
                    x_cm = np.asarray(cloud[:, 0], dtype=np.float32) * 100.0
                    y_cm = -np.asarray(cloud[:, 1], dtype=np.float32) * 100.0
                    min_x = float(bbox["min_x"])
                    max_x = float(bbox["max_x"])
                    min_y = float(bbox["min_y"])
                    max_y = float(bbox["max_y"])
                    band_cm = max(250.0, min(float(standoff) * 0.4, 500.0))
                    facade_specs = {
                        "south": (x_cm, (x_cm >= min_x) & (x_cm <= max_x) & (y_cm >= min_y - band_cm) & (y_cm <= min_y + band_cm), min_x, max_x),
                        "north": (x_cm, (x_cm >= min_x) & (x_cm <= max_x) & (y_cm >= max_y - band_cm) & (y_cm <= max_y + band_cm), min_x, max_x),
                        "west": (y_cm, (y_cm >= min_y) & (y_cm <= max_y) & (x_cm >= min_x - band_cm) & (x_cm <= min_x + band_cm), min_y, max_y),
                        "east": (y_cm, (y_cm >= min_y) & (y_cm <= max_y) & (x_cm >= max_x - band_cm) & (x_cm <= max_x + band_cm), min_y, max_y),
                    }
                    for facade, (axis_values, mask, start, end) in facade_specs.items():
                        total_cells = max(1, int(math.ceil(abs(float(end) - float(start)) / max(1.0, spacing))))
                        selected = np.asarray(axis_values[mask], dtype=np.float32)
                        covered_cells = 0
                        if selected.size:
                            cell_indices = np.floor((selected - min(float(start), float(end))) / max(1.0, spacing)).astype(np.int32)
                            cell_indices = cell_indices[(cell_indices >= 0) & (cell_indices < total_cells)]
                            covered_cells = int(np.unique(cell_indices).size)
                        facades[facade].update(
                            {
                                "point_cloud_coverage": round(float(covered_cells) / float(total_cells), 4),
                                "covered_cells": covered_cells,
                                "total_cells": total_cells,
                            }
                        )
            except Exception as exc:
                LOGGER.warning("House search coverage estimate failed: %s", exc)
        coverage_values = [
            float(item["point_cloud_coverage"])
            for item in facades.values()
            if item.get("point_cloud_coverage") is not None
        ]
        if not coverage_values:
            coverage_values = [float(item["scan_completion_ratio"]) for item in facades.values()]
        mean_coverage = float(sum(coverage_values) / max(1, len(coverage_values)))
        complete = bool(
            all(float(item.get("point_cloud_coverage") if item.get("point_cloud_coverage") is not None else item["scan_completion_ratio"]) >= 0.75 for item in facades.values())
            and mean_coverage >= 0.85
            and merged_count > 0
        )
        return {
            "target_house_id": target_house_id,
            "house_search_dir": str(self.house_search_output_dir_for_active_plan() or ""),
            "facades": facades,
            "mean_facade_coverage": round(mean_coverage, 4),
            "merged_point_count": merged_count,
            "source_frame_count": int(postprocess_result.get("source_frame_count", 0) or 0),
            "complete": complete,
            "coverage_mode": "point_cloud_band_grid" if cloud_path and Path(cloud_path).exists() else "scan_completion_fallback",
            "cloud_path": cloud_path,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def build_house_search_rescan_plan(self, coverage_report: Dict[str, Any]) -> Dict[str, Any]:
        target_house_id = str(coverage_report.get("target_house_id", "") or "")
        rescan_points: List[Dict[str, Any]] = []
        facades = coverage_report.get("facades", {}) if isinstance(coverage_report.get("facades"), dict) else {}
        for facade, report in facades.items():
            if not isinstance(report, dict):
                continue
            coverage = report.get("point_cloud_coverage")
            score = float(coverage if coverage is not None else report.get("scan_completion_ratio", 0.0) or 0.0)
            if score >= 0.75:
                continue
            candidates = [
                point for point in self.llm_route_scan_points
                if isinstance(point, dict) and str(point.get("facade", "") or "") == str(facade)
            ]
            if not candidates:
                continue
            point = dict(candidates[len(candidates) // 2])
            point["scan_id"] = f"{target_house_id}_{facade}_rescan_000"
            point["status"] = "planned"
            point["view_type"] = "hole_center_face_view"
            point["rescan_reason"] = f"{facade} coverage below threshold"
            rescan_points.append(point)
        return {
            "house_id": target_house_id,
            "rescan_reason": "coverage below threshold" if rescan_points else "",
            "rescan_points": rescan_points,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def write_house_search_summary_csv(self, search_dir: Path, coverage_report: Dict[str, Any]) -> None:
        headers = [
            "target_house_id",
            "planned_scan_count",
            "completed_scan_count",
            "capture_success_count",
            "capture_failure_count",
            "merged_point_count",
            "mean_facade_coverage",
            "complete",
        ]
        row = {
            "target_house_id": str(coverage_report.get("target_house_id", "") or ""),
            "planned_scan_count": str(self.llm_route_execution_summary.get("planned_scan_count", 0)),
            "completed_scan_count": str(self.llm_route_execution_summary.get("completed_scan_count", 0)),
            "capture_success_count": str(self.llm_route_execution_summary.get("capture_success_count", 0)),
            "capture_failure_count": str(self.llm_route_execution_summary.get("capture_failure_count", 0)),
            "merged_point_count": str(coverage_report.get("merged_point_count", 0)),
            "mean_facade_coverage": str(coverage_report.get("mean_facade_coverage", 0.0)),
            "complete": str(bool(coverage_report.get("complete", False))),
        }
        lines = [",".join(headers), ",".join(row.get(header, "") for header in headers)]
        (search_dir / "house_search_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def finalize_house_search_outputs(self) -> Dict[str, Any]:
        search_dir = self.house_search_output_dir_for_active_plan()
        if search_dir is None:
            return {}
        self.write_house_search_lidar_summary(search_dir, running=False)
        postprocess_result = flight.postprocess_lidar_stream_capture(
            search_dir,
            lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
            min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
            max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            voxel_cm=flight.DEFAULT_LIDAR_RECON_VOXEL_CM,
            max_points=flight.DEFAULT_LIDAR_RECON_MAX_POINTS,
        )
        coverage_report = self.build_house_search_coverage_report(postprocess_result)
        rescan_plan = self.build_house_search_rescan_plan(coverage_report)
        self.write_json_artifact(search_dir / "coverage_report.json", coverage_report)
        self.write_json_artifact(search_dir / "rescan_plan.json", rescan_plan)
        self.write_house_search_summary_csv(search_dir, coverage_report)
        reconstruction = postprocess_result.get("reconstruction", {}) if isinstance(postprocess_result.get("reconstruction"), dict) else {}
        self.llm_route_execution_summary.update(
            {
                "merged_point_count": int(coverage_report.get("merged_point_count", 0) or 0),
                "mean_facade_coverage": coverage_report.get("mean_facade_coverage", 0.0),
                "complete": bool(coverage_report.get("complete", False)),
                "coverage_report_path": str(search_dir / "coverage_report.json"),
                "rescan_plan_path": str(search_dir / "rescan_plan.json"),
                "merged_point_cloud_world_standard_m_ply_path": str(
                    reconstruction.get("merged_point_cloud_world_standard_m_ply_path", "")
                ),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.write_json_artifact(search_dir / "execution_summary.json", self.llm_route_execution_summary)
        self.root.after(0, self.refresh_route_preview)
        return {
            "postprocess_result": postprocess_result,
            "coverage_report": coverage_report,
            "rescan_plan": rescan_plan,
        }

    def read_jsonl_artifact(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def validate_house_search_data(self, *, finalize_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        search_dir = self.house_search_output_dir_for_active_plan()
        if search_dir is None:
            raise RuntimeError("missing house search output directory")
        frames_dir = search_dir / "frames"
        frame_dirs = sorted(path for path in frames_dir.glob("frame_*") if path.is_dir()) if frames_dir.exists() else []
        if finalize_result is None and frame_dirs:
            finalize_result = self.finalize_house_search_outputs()
        coverage_report = {}
        rescan_plan = {}
        if isinstance(finalize_result, dict):
            coverage_report = finalize_result.get("coverage_report", {}) if isinstance(finalize_result.get("coverage_report"), dict) else {}
            rescan_plan = finalize_result.get("rescan_plan", {}) if isinstance(finalize_result.get("rescan_plan"), dict) else {}
        if not coverage_report:
            coverage_report = flight.read_json_object(search_dir / "coverage_report.json")
        if not rescan_plan:
            rescan_plan = flight.read_json_object(search_dir / "rescan_plan.json")
        scan_points_payload = flight.read_json_object(search_dir / "scan_points.json")
        file_scan_points = scan_points_payload.get("scan_points", []) if isinstance(scan_points_payload.get("scan_points"), list) else []
        scan_points = self.llm_route_scan_points if self.llm_route_scan_points else file_scan_points
        target_house_id = str(
            coverage_report.get("target_house_id", "")
            or scan_points_payload.get("target_house_id", "")
            or self.llm_route_plan.get("target_house_id", "")
            or self.selected_route_target_house_id()
        )
        execution_rows = self.read_jsonl_artifact(search_dir / "scan_execution_log.jsonl")
        capture_rows = self.read_jsonl_artifact(search_dir / "lidar_capture_log.jsonl")
        successful_scan_ids = {
            str(row.get("scan_id", "") or "")
            for row in execution_rows
            if str(row.get("capture_status", "") or "") == "ok"
        }
        facade_set = {
            str(point.get("facade", "") or "")
            for point in scan_points
            if isinstance(point, dict) and str(point.get("facade", "") or "")
        }
        scan_validation = self.scan_point_validation_report(target_house_id, scan_points)
        planned_count = len(scan_points)
        capture_success_rate = len(successful_scan_ids) / max(1, planned_count)
        merged_count = int(coverage_report.get("merged_point_count", 0) or 0)
        mean_coverage = float(coverage_report.get("mean_facade_coverage", 0.0) or 0.0)
        rescan_points = rescan_plan.get("rescan_points", []) if isinstance(rescan_plan.get("rescan_points"), list) else []
        reconstruction_ply = ""
        if isinstance(self.llm_route_execution_summary, dict):
            reconstruction_ply = str(self.llm_route_execution_summary.get("merged_point_cloud_world_standard_m_ply_path", "") or "")
        if not reconstruction_ply:
            reconstruction_ply = str(
                coverage_report.get("cloud_path", "") or search_dir / "reconstruction" / "merged_point_cloud_world_standard_m.ply"
            )
        checks = [
            {
                "name": "scan_points_exist",
                "passed": planned_count > 0,
                "detail": f"{planned_count} planned scan points",
            },
            {
                "name": "four_facades_planned",
                "passed": {"south", "east", "north", "west"}.issubset(facade_set),
                "detail": ",".join(sorted(facade_set)),
            },
            {
                "name": "scan_point_geometry_valid",
                "passed": bool(scan_validation.get("valid", False)),
                "detail": scan_validation,
            },
            {
                "name": "capture_success_rate",
                "passed": capture_success_rate >= 0.90,
                "detail": f"{len(successful_scan_ids)}/{planned_count} scan points captured",
            },
            {
                "name": "lidar_capture_rows_exist",
                "passed": len(capture_rows) > 0,
                "detail": f"{len(capture_rows)} lidar capture rows",
            },
            {
                "name": "reconstruction_points_exist",
                "passed": merged_count > 0,
                "detail": f"merged_point_count={merged_count}",
            },
            {
                "name": "coverage_complete_or_rescan_available",
                "passed": mean_coverage >= 0.85 or len(rescan_points) > 0,
                "detail": f"mean_coverage={mean_coverage:.4f}, rescan_points={len(rescan_points)}",
            },
        ]
        validation = {
            "target_house_id": target_house_id,
            "house_search_dir": str(search_dir),
            "overall_passed": all(bool(check.get("passed", False)) for check in checks),
            "checks": checks,
            "planned_scan_count": planned_count,
            "captured_scan_count": len(successful_scan_ids),
            "capture_success_rate": round(float(capture_success_rate), 4),
            "lidar_capture_row_count": len(capture_rows),
            "frame_dir_count": len(frame_dirs),
            "merged_point_count": merged_count,
            "mean_facade_coverage": round(mean_coverage, 4),
            "rescan_point_count": len(rescan_points),
            "coverage_report_path": str(search_dir / "coverage_report.json"),
            "rescan_plan_path": str(search_dir / "rescan_plan.json"),
            "reconstruction_ply_path": reconstruction_ply,
            "validated_at": datetime.now().isoformat(timespec="seconds"),
        }
        validation_path = search_dir / "validation_report.json"
        validation["validation_report_path"] = str(validation_path)
        self.write_json_artifact(validation_path, validation)
        self.llm_route_validation_report = validation
        self.llm_route_execution_summary.update(
            {
                "validation_report_path": str(validation_path),
                "validation_passed": bool(validation["overall_passed"]),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.write_json_artifact(search_dir / "execution_summary.json", self.llm_route_execution_summary)
        return validation

    def follow_route_worker(self, session: flight.DroneFlightSession, points: List[Dict[str, Any]], *, auto: bool) -> None:
        step_cm = self.route_step_cm()
        delay_s = self.route_delay_s()
        total = len(points)
        captured_any = False
        for point_index, point in enumerate(points, start=1):
            if self.route_stop_event.is_set():
                self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: follow stopped."))
                return
            tx = self._as_float_or_none(point.get("x"))
            ty = self._as_float_or_none(point.get("y"))
            if tx is None or ty is None:
                continue
            is_scan_point = self.is_scan_route_point(point)
            target_z = self._as_float_or_none(point.get("z")) if is_scan_point else None
            label = str(point.get("label", f"wp{point_index}") or f"wp{point_index}")
            while not self.route_stop_event.is_set():
                state = self.safe("Route follow state", session.get_state)
                pose = state.get("pose", {}) if isinstance(state, dict) and isinstance(state.get("pose"), dict) else {}
                px = self._as_float_or_none(pose.get("x"))
                py = self._as_float_or_none(pose.get("y"))
                pz = self._as_float_or_none(pose.get("z"))
                if px is None or py is None:
                    self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: missing UAV pose during follow."))
                    return
                dx = float(tx) - float(px)
                dy = float(ty) - float(py)
                distance_cm = math.hypot(dx, dy)
                if distance_cm <= float(LLM_ROUTE_WAYPOINT_REACHED_CM):
                    self.root.after(
                        0,
                        lambda i=point_index, t=total, l=label: self.llm_route_status_var.set(f"LLM Route: reached {i}/{t} {l}"),
                    )
                    break
                travel = min(step_cm, distance_cm)
                ratio = travel / max(1e-6, distance_cm)
                nx = float(px) + dx * ratio
                ny = float(py) + dy * ratio
                yaw = math.degrees(math.atan2(dy, dx))
                self.root.after(
                    0,
                    lambda i=point_index, t=total, l=label, d=distance_cm: self.llm_route_status_var.set(
                        f"LLM Route: following {i}/{t} {l} dist={d:.1f}cm"
                    ),
                )
                response = self.safe(
                    "Route follow set pose",
                    lambda nx=nx, ny=ny, pz=pz, yaw=yaw, target_z=target_z: session.set_pose(
                        {"x": nx, "y": ny, "z": float(target_z if target_z is not None else (pz if pz is not None else 100.0)), "yaw": yaw}
                    ),
                )
                if isinstance(response, dict):
                    self.root.after(0, lambda r=response: self.apply_state(r))
                else:
                    return
                time.sleep(delay_s)
            if is_scan_point and not self.route_stop_event.is_set():
                capture_entry = self.capture_at_scan_route_point(
                    session,
                    point,
                    point_index=point_index,
                    total=total,
                )
                captured_any = captured_any or capture_entry.get("capture_status") == "ok"
            if not auto:
                break
        if self.route_stop_event.is_set():
            self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: follow stopped."))
        else:
            if captured_any:
                try:
                    finalize_result = self.finalize_house_search_outputs()
                    coverage = finalize_result.get("coverage_report", {}) if isinstance(finalize_result, dict) else {}
                    self.root.after(
                        0,
                        lambda c=coverage: self.llm_route_status_var.set(
                            f"LLM Route: follow complete, coverage={float(c.get('mean_facade_coverage', 0.0) or 0.0):.2f}"
                        ),
                    )
                except Exception as exc:
                    LOGGER.warning("House search finalize failed: %s", exc)
                    self.root.after(0, lambda e=exc: self.llm_route_status_var.set(f"LLM Route: finalize failed: {e}"))
            else:
                self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: follow complete."))

