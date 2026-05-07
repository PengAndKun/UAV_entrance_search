from __future__ import annotations

from .common import *


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
            valid_points.append({
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "label": str(point.get("label", "") or f"wp_{idx}"),
                "status": str(point.get("status", "") or "planned"),
            })
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

    def apply_route_plan(self, route_plan: Dict[str, Any], *, status_prefix: str = "LLM Route") -> None:
        self.llm_route_plan = route_plan if isinstance(route_plan, dict) else {}
        target_house_id = str(self.llm_route_plan.get("target_house_id", "") or "")
        if target_house_id:
            self.set_selected_route_target_house(target_house_id)
        self.refresh_route_preview()
        self.refresh_map_once()
        point_count = len(self.llm_route_plan.get("route_points", [])) if isinstance(self.llm_route_plan.get("route_points"), list) else 0
        source = str(self.llm_route_plan.get("planner_source", "") or "unknown")
        message = str(self.llm_route_plan.get("llm_route_error", "") or self.llm_route_plan.get("reason", "") or "")
        suffix = f" | {message}" if message else ""
        self.llm_route_status_var.set(f"{status_prefix}: house={target_house_id or '-'} source={source} points={point_count}{suffix}")

    def refresh_route_preview(self) -> None:
        if self.llm_route_preview_text is None:
            return
        payload = {
            "task_plan": self.llm_task_plan if isinstance(self.llm_task_plan, dict) else {},
            "route_plan": self.llm_route_plan if isinstance(self.llm_route_plan, dict) else {},
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        try:
            self.llm_route_preview_text.configure(state="normal")
            self.llm_route_preview_text.delete("1.0", "end")
            self.llm_route_preview_text.insert("1.0", text)
            self.llm_route_preview_text.configure(state="disabled")
        except tk.TclError:
            pass

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
        self.refresh_route_preview()
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
        return list(points[active_index:] if auto else points[active_index:active_index + 1])

    def on_follow_route_next(self) -> None:
        self.start_route_follow(auto=False)

    def on_follow_route_auto(self) -> None:
        self.start_route_follow(auto=True)

    def on_stop_route_follow(self) -> None:
        self.route_stop_event.set()
        self.llm_route_status_var.set("LLM Route: stopping follow...")

    def start_route_follow(self, *, auto: bool) -> None:
        session = self.active_session()
        if session is None:
            return
        if not isinstance(self.llm_route_plan, dict) or not self.llm_route_plan.get("route_points"):
            self.on_fallback_route_plan()
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route_status_var.set("LLM Route: follow already running.")
            return
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

    def follow_route_worker(self, session: flight.DroneFlightSession, points: List[Dict[str, Any]], *, auto: bool) -> None:
        step_cm = self.route_step_cm()
        delay_s = self.route_delay_s()
        total = len(points)
        for point_index, point in enumerate(points, start=1):
            if self.route_stop_event.is_set():
                self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: follow stopped."))
                return
            tx = self._as_float_or_none(point.get("x"))
            ty = self._as_float_or_none(point.get("y"))
            if tx is None or ty is None:
                continue
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
                    lambda nx=nx, ny=ny, pz=pz, yaw=yaw: session.set_pose({"x": nx, "y": ny, "z": float(pz if pz is not None else 100.0), "yaw": yaw}),
                )
                if isinstance(response, dict):
                    self.root.after(0, lambda r=response: self.apply_state(r))
                else:
                    return
                time.sleep(delay_s)
            if not auto:
                break
        if self.route_stop_event.is_set():
            self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: follow stopped."))
        else:
            self.root.after(0, lambda: self.llm_route_status_var.set("LLM Route: follow complete."))

