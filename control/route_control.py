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
        try:
            return image_to_world_with_calibration(float(image_x), float(image_y), self.map_calibration)
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
            "Plan Route6_entrance_search through open street/yard space. Avoid non-target house boxes.",
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
                "Because the entrance is unknown, after arrival include Route6_entrance_search points that scan the target front, side, back, and other side when safe.",
                "Return strict JSON only.",
            ],
        }

    def build_llm_route_system_prompt(self) -> str:
        return (
            "You are an overhead-map VLM Route6_entrance_search planner for a UAV searching for a house entrance. "
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
            fallback_plan["llm_route_error"] = "LLM Route6_entrance_search had fewer than two valid numeric waypoints"
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
            fallback_plan["llm_route_error"] = "LLM Route6_entrance_search violated forbidden non-target house bbox clearance"
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
            "reason": str(parsed.get("reason", "") or "LLM-generated overhead-map Route6_entrance_search."),
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

    def route_capture_interval_s(self) -> float:
        try:
            value = float(self.route_capture_interval_s_var.get().strip())
        except Exception:
            value = float(LLM_ROUTE_PATH_CAPTURE_INTERVAL_S)
        return max(0.05, min(10.0, float(value)))

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

    def scan_safe_intervals_by_facade(
        self,
        target_house_id: str,
        bbox: Dict[str, Any],
        corridor_info: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        try:
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            return {}
        spacing = self.route_scan_spacing_cm()
        min_segment_len = max(60.0, min(180.0, float(spacing) * 0.5))
        obstacles = self.route_forbidden_house_bboxes(
            target_house_id=target_house_id,
            clearance_cm=float(LLM_ROUTE_HOUSE_CLEARANCE_CM),
        )

        def standoff_for_facade(facade: str) -> float:
            info = corridor_info.get(facade, {}) if isinstance(corridor_info.get(facade), dict) else {}
            try:
                return max(20.0, float(info.get("standoff_cm", self.route_scan_standoff_cm())))
            except Exception:
                return self.route_scan_standoff_cm()

        def subtract_interval(intervals: List[Tuple[float, float]], cut_min: float, cut_max: float) -> List[Tuple[float, float]]:
            lo = float(min(cut_min, cut_max))
            hi = float(max(cut_min, cut_max))
            remaining: List[Tuple[float, float]] = []
            for start, end in intervals:
                start = float(start)
                end = float(end)
                if hi <= start or lo >= end:
                    remaining.append((start, end))
                    continue
                if lo > start:
                    remaining.append((start, max(start, lo)))
                if hi < end:
                    remaining.append((min(end, hi), end))
            return [(start, end) for start, end in remaining if end - start >= min_segment_len]

        safe: Dict[str, List[Dict[str, Any]]] = {}
        for facade in ("south", "east", "north", "west"):
            if facade in {"south", "north"}:
                intervals: List[Tuple[float, float]] = [(min_x, max_x)]
                line_value = min_y - standoff_for_facade(facade) if facade == "south" else max_y + standoff_for_facade(facade)
                for obstacle in obstacles:
                    try:
                        obs_min_y = float(obstacle["min_y"])
                        obs_max_y = float(obstacle["max_y"])
                        obs_min_x = float(obstacle["min_x"])
                        obs_max_x = float(obstacle["max_x"])
                    except Exception:
                        continue
                    if obs_min_y <= line_value <= obs_max_y:
                        intervals = subtract_interval(intervals, obs_min_x, obs_max_x)
            else:
                intervals = [(min_y, max_y)]
                line_value = max_x + standoff_for_facade(facade) if facade == "east" else min_x - standoff_for_facade(facade)
                for obstacle in obstacles:
                    try:
                        obs_min_x = float(obstacle["min_x"])
                        obs_max_x = float(obstacle["max_x"])
                        obs_min_y = float(obstacle["min_y"])
                        obs_max_y = float(obstacle["max_y"])
                    except Exception:
                        continue
                    if obs_min_x <= line_value <= obs_max_x:
                        intervals = subtract_interval(intervals, obs_min_y, obs_max_y)
            safe[facade] = [
                {
                    "index": idx,
                    "min": round(float(start), 2),
                    "max": round(float(end), 2),
                    "source": "bbox_clearance_clipped",
                    "min_segment_len_cm": round(float(min_segment_len), 2),
                }
                for idx, (start, end) in enumerate(intervals)
            ]
        return safe

    def build_rule_scan_points_for_route_plan(self, route_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        hid = str(route_plan.get("target_house_id", "") or self.selected_route_target_house_id()).strip()
        if not hid:
            return []
        bbox = self.house_world_bbox_for_id(hid)
        if not bbox:
            return []
        corridor_info = self.scan_corridor_standoff_by_facade(hid, bbox)
        route_plan["scan_corridor_standoff_by_facade"] = corridor_info
        safe_intervals = self.scan_safe_intervals_by_facade(hid, bbox, corridor_info)
        route_plan["scan_safe_intervals_by_facade"] = safe_intervals
        return generate_rule_scan_points(
            house_id=hid,
            bbox_world=bbox,
            facade_order=self.preferred_facade_order_for_route_plan(route_plan),
            standoff_cm=self.route_scan_standoff_cm(),
            scan_spacing_cm=self.route_scan_spacing_cm(),
            altitude_cm=self.target_house_capture_altitude_cm(hid),
            facade_standoff_info=corridor_info,
            facade_axis_intervals=safe_intervals,
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
        raw_obstacles = self.route_forbidden_house_bboxes(target_house_id=target_house_id, clearance_cm=0.0)
        clearance_obstacles = self.route_forbidden_house_bboxes(
            target_house_id=target_house_id,
            clearance_cm=float(LLM_ROUTE_HOUSE_CLEARANCE_CM),
        )
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
            for obstacle in raw_obstacles:
                if self.point_inside_open_bbox(float(x), float(y), obstacle):
                    invalid.append(
                        {
                            "scan_id": scan_id,
                            "reason": "inside_non_target_bbox",
                            "house_id": str(obstacle.get("house_id", "") or ""),
                        }
                    )
                    break
            for obstacle in clearance_obstacles:
                if self.point_inside_open_bbox(float(x), float(y), obstacle):
                    invalid.append(
                        {
                            "scan_id": scan_id,
                            "reason": "inside_non_target_clearance",
                            "house_id": str(obstacle.get("house_id", "") or ""),
                            "clearance_cm": obstacle.get("clearance_cm", LLM_ROUTE_HOUSE_CLEARANCE_CM),
                        }
                    )
                    break
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
        if start_scan and end_scan and str(start.get("facade", "") or "") == str(end.get("facade", "") or ""):
            start_interval = start.get("safe_interval_index")
            end_interval = end.get("safe_interval_index")
            if start_interval is not None and end_interval is not None and str(start_interval) != str(end_interval):
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

    def make_route_capture_output_dir(self, target_house_id: str) -> Path:
        root = self.resolve_project_path("route_capture_lidar")
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

    def make_route2_facade_output_dir(self, target_house_id: str) -> Path:
        root = self.resolve_project_path("route_capture_lidar")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_facade_v2_{timestamp}"
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

    def route2_floor_height_m(self) -> float:
        try:
            return max(1.0, min(8.0, float(self.llm_route2_floor_height_m_var.get().strip())))
        except Exception:
            return float(LLM_ROUTE2_DEFAULT_FLOOR_HEIGHT_M)

    def route2_default_floors(self) -> int:
        try:
            return max(1, min(int(LLM_ROUTE2_MAX_FLOORS), int(float(self.llm_route2_default_floors_var.get().strip()))))
        except Exception:
            return int(LLM_ROUTE2_DEFAULT_FLOORS)

    def route2_low_z_cm(self) -> float:
        try:
            return max(50.0, min(1500.0, float(self.llm_route2_low_z_cm_var.get().strip())))
        except Exception:
            return float(LLM_ROUTE2_LOW_Z_CM)

    def route2_z_step_cm(self) -> float:
        try:
            return max(100.0, min(800.0, float(self.llm_route2_z_step_cm_var.get().strip())))
        except Exception:
            return float(LLM_ROUTE2_Z_STEP_CM)

    def route2_density_mode(self) -> str:
        value = str(self.llm_route2_density_mode_var.get() or "auto").strip().lower()
        return value if value in {"auto", "high", "medium", "low"} else "auto"

    def route2_facade_dir(self, output_dir: Path, house_id: str, facade: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{house_id}_{facade}").strip("_")
        path = output_dir / "facade_observations" / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def route2_state_output_dir(self) -> Optional[Path]:
        raw = ""
        state = self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {}
        raw = str(state.get("output_dir", "") or "")
        if not raw:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        (path / "reconstruction").mkdir(parents=True, exist_ok=True)
        (path / "facade_observations").mkdir(parents=True, exist_ok=True)
        return path

    def route2_facade_id(self, house_id: str, facade: str) -> str:
        return f"{str(house_id or '').strip()}_{str(facade or '').strip()}"

    def prepare_active_plan_for_route_capture(self, target_house_id: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frames").mkdir(parents=True, exist_ok=True)
        (output_dir / "reconstruction").mkdir(parents=True, exist_ok=True)
        self.house_search_dir = output_dir
        if isinstance(self.llm_route_plan, dict):
            self.llm_route_plan["house_search_dir"] = str(output_dir)
            self.llm_route_plan["route_capture_lidar_dir"] = str(output_dir)
            self.llm_route_plan["route_capture_mode"] = "scan_point_path_lidar_qe_yaw"
            self.llm_route_plan["route_capture_interval_s"] = self.route_capture_interval_s()
        self.llm_route_lidar_trajectory = []
        self.llm_route_execution_summary.update(
            {
                "house_search_dir": str(output_dir),
                "route_capture_lidar_dir": str(output_dir),
                "route_capture_mode": "scan_point_path_lidar_qe_yaw",
                "route_capture_interval_s": self.route_capture_interval_s(),
                "target_house_id": target_house_id,
                "planned_scan_count": len(self.llm_route_scan_points),
                "completed_scan_count": 0,
                "capture_success_count": 0,
                "capture_failure_count": 0,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.write_json_artifact(output_dir / "normalized_route_plan.json", self.llm_route_plan if isinstance(self.llm_route_plan, dict) else {})
        self.write_json_artifact(
            output_dir / "scan_points.json",
            {
                "schema": LLM_ROUTE_SCAN_POINTS_SCHEMA,
                "target_house_id": target_house_id,
                "scan_points": self.llm_route_scan_points,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self.write_json_artifact(output_dir / "execution_summary.json", self.llm_route_execution_summary)

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
            "reason": str(route_plan.get("reason", "") or "Rule-generated facade scan Route6_entrance_search from target house bbox."),
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
                "scan_safe_intervals_by_facade": plan.get("scan_safe_intervals_by_facade", {}),
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
            route_plan = self.llm_route_plan if isinstance(self.llm_route_plan, dict) else {}
            widget.set_route_plan(route_plan)
            scan_count = len(self.llm_route_scan_points) if isinstance(self.llm_route_scan_points, list) else 0
            route_count = len(route_plan.get("route_points", [])) if isinstance(route_plan.get("route_points"), list) else 0
            self.llm_route_map_status_var.set(f"Route Map: houses={len(houses)} route_points={route_count} scan_points={scan_count}")
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM Route6_entrance_search map failed: %s", exc)
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
            message = "Route6_entrance_search blocked by safety validation"
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

    def refresh_route2_preview(self) -> None:
        preview_text = getattr(self, "llm_route2_preview_text", None)
        payload = self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {}
        if preview_text is not None:
            try:
                if not preview_text.winfo_exists():
                    self.llm_route2_preview_text = None
                else:
                    preview_text.configure(state="normal")
                    preview_text.delete("1.0", "end")
                    preview_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
                    preview_text.configure(state="disabled")
            except tk.TclError:
                self.llm_route2_preview_text = None
        self.refresh_route2_support_views()

    def refresh_route2_support_views(self) -> None:
        self.refresh_route2_progress()
        self.refresh_route2_facade_status()
        self.refresh_route2_analysis_view()
        self.refresh_route2_rgb_display()

    def refresh_route2_facade_status(self) -> None:
        status_var = getattr(self, "llm_route2_facade_status_var", None)
        if status_var is None:
            return
        state = self.route2_selected_state()
        facade = str(state.get("facade", "") or "")
        if not facade:
            status = "Processed: no facade"
        else:
            validation = state.get("validation_report", {}) if isinstance(state.get("validation_report"), dict) else {}
            points = state.get("facade_scan_points", []) if isinstance(state.get("facade_scan_points"), list) else []
            captured = sum(1 for point in points if isinstance(point, dict) and str(point.get("status", "") or "") == "captured")
            if validation:
                status = f"Processed: validated {'PASS' if validation.get('overall_passed') else 'CHECK'}"
            elif points and captured >= len(points):
                status = f"Processed: captured {captured}/{len(points)}"
            elif points:
                status = f"Processed: in progress {captured}/{len(points)}"
            elif state.get("coarse_rgb_path") or state.get("coarse_capture"):
                status = "Processed: RGB captured"
            elif state.get("observation_point"):
                status = "Processed: planned"
            else:
                status = "Processed: no"
        try:
            status_var.set(status)
        except tk.TclError:
            pass

    def refresh_route2_progress(self) -> None:
        progress_var = getattr(self, "llm_route2_progress_var", None)
        progress_text_var = getattr(self, "llm_route2_progress_text_var", None)
        if progress_var is None or progress_text_var is None:
            return
        state = self.route2_selected_state()
        points = state.get("facade_scan_points", []) if isinstance(state.get("facade_scan_points"), list) else []
        captured_ids = {
            str(row.get("scan_id", "") or "")
            for row in state.get("facade_capture_rows", [])
            if isinstance(row, dict) and str(row.get("scan_id", "") or "")
        } if isinstance(state.get("facade_capture_rows"), list) else set()
        captured_count = 0
        for point in points:
            if not isinstance(point, dict):
                continue
            if point.get("status") == "captured" or str(point.get("scan_id", "") or "") in captured_ids:
                captured_count += 1
        if points:
            value = 100.0 * float(captured_count) / float(max(1, len(points)))
            text = f"Explore: {captured_count}/{len(points)}"
        else:
            stage_count = 0
            if isinstance(state.get("observation_point"), dict) and state.get("observation_point"):
                stage_count += 1
            if state.get("coarse_rgb_path") or state.get("coarse_capture"):
                stage_count += 1
            if isinstance(state.get("facade_analysis"), dict) and state.get("facade_analysis"):
                stage_count += 1
            value = 100.0 * float(stage_count) / 3.0
            text = f"Explore: stage {stage_count}/3"
        try:
            progress_var.set(max(0.0, min(100.0, float(value))))
            progress_text_var.set(text)
        except tk.TclError:
            pass

    def refresh_route2_analysis_view(self) -> None:
        analysis_text = getattr(self, "llm_route2_analysis_text", None)
        if analysis_text is None:
            return
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        if not analysis:
            payload: Dict[str, Any] = {
                "facade": state.get("facade", ""),
                "status": "No facade analysis yet.",
            }
        else:
            payload = analysis
        try:
            if not analysis_text.winfo_exists():
                self.llm_route2_analysis_text = None
                return
            analysis_text.configure(state="normal")
            analysis_text.delete("1.0", "end")
            analysis_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            analysis_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route2_analysis_text = None

    def refresh_route2_rgb_display(self) -> None:
        widget = getattr(self, "llm_route2_rgb_label", None)
        if widget is None:
            return
        image_path = self.route2_current_rgb_path()
        if image_path is None:
            try:
                self.route2_draw_rgb_preview_message(widget, "No facade RGB")
                self.llm_route2_rgb_photo = None
                self.llm_route2_rgb_status_var.set("Facade RGB: none")
            except tk.TclError:
                self.llm_route2_rgb_label = None
            return
        try:
            image = Image.open(image_path).convert("RGB")
            photo = ImageTk.PhotoImage(self.route2_rgb_preview_image(image, widget))
            self.route2_draw_rgb_preview_photo(widget, photo)
            self.llm_route2_rgb_photo = photo
            suffix = " (panorama used)" if image_path.name == "coarse_rgb_panorama.png" else ""
            self.llm_route2_rgb_status_var.set(f"Facade RGB: {image_path.name}{suffix}")
        except Exception as exc:
            LOGGER.warning("Refresh Route6_entrance_search v2 facade RGB failed: %s", exc)
            try:
                self.route2_draw_rgb_preview_message(widget, f"RGB load failed:\n{exc}")
                self.llm_route2_rgb_photo = None
                self.llm_route2_rgb_status_var.set("Facade RGB: load failed")
            except tk.TclError:
                self.llm_route2_rgb_label = None

    def route2_draw_rgb_preview_message(self, widget: tk.Widget, message: str) -> None:
        if isinstance(widget, tk.Canvas):
            widget.delete("all")
            try:
                w = max(1, int(widget.winfo_width()))
                h = max(1, int(widget.winfo_height()))
            except Exception:
                w, h = 330, 230
            widget.create_rectangle(0, 0, w, h, fill="#202020", outline="")
            widget.create_text(
                w / 2,
                h / 2,
                text=str(message),
                fill="#dddddd",
                font=("Consolas", 9),
                justify="center",
                width=max(120, w - 24),
            )
        else:
            widget.configure(image="", text=str(message))

    def route2_draw_rgb_preview_photo(self, widget: tk.Widget, photo: ImageTk.PhotoImage) -> None:
        if isinstance(widget, tk.Canvas):
            widget.delete("all")
            try:
                w = max(1, int(widget.winfo_width()))
                h = max(1, int(widget.winfo_height()))
            except Exception:
                w, h = 330, 230
            widget.create_rectangle(0, 0, w, h, fill="#202020", outline="")
            widget.create_image(w / 2, h / 2, image=photo, anchor="center")
        else:
            widget.configure(image=photo, text="")

    def route2_rgb_preview_image(self, image: Image.Image, widget: tk.Widget) -> Image.Image:
        try:
            target_w = int(widget.winfo_width())
            target_h = int(widget.winfo_height())
        except Exception:
            target_w, target_h = 330, 230
        if target_w <= 8:
            try:
                target_w = int(float(widget.cget("width")))
            except Exception:
                target_w = 330
        if target_h <= 8:
            try:
                target_h = int(float(widget.cget("height")))
            except Exception:
                target_h = 230
        target_w = min(520, max(240, target_w))
        target_h = min(360, max(180, target_h))
        source_w, source_h = image.size
        if source_w <= 0 or source_h <= 0:
            return Image.new("RGB", (target_w, target_h), (32, 32, 32))
        scale = min(float(target_w) / float(source_w), float(target_h) / float(source_h))
        resize_w = max(1, int(round(float(source_w) * scale)))
        resize_h = max(1, int(round(float(source_h) * scale)))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        resized = image.resize((resize_w, resize_h), resample)
        preview = Image.new("RGB", (target_w, target_h), (32, 32, 32))
        left = max(0, (target_w - resize_w) // 2)
        top = max(0, (target_h - resize_h) // 2)
        preview.paste(resized, (left, top))
        return preview

    def route2_candidate_path(self, raw: Any) -> Optional[Path]:
        text = str(raw or "").strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            path = self.resolve_project_path(text)
        try:
            path = path.resolve()
        except Exception:
            pass
        return path

    def route2_current_rgb_path(self) -> Optional[Path]:
        state = self.route2_selected_state()
        candidates: List[Any] = [state.get("coarse_rgb_panorama_path")]
        coarse_capture = state.get("coarse_capture", {}) if isinstance(state.get("coarse_capture"), dict) else {}
        candidates.append(coarse_capture.get("coarse_rgb_panorama_path"))
        panorama_capture = state.get("panorama_capture", {}) if isinstance(state.get("panorama_capture"), dict) else {}
        candidates.append(panorama_capture.get("coarse_rgb_panorama_path"))
        candidates.append(state.get("coarse_rgb_path"))
        candidates.append(coarse_capture.get("coarse_rgb_path"))
        capture_result = coarse_capture.get("capture_result", {}) if isinstance(coarse_capture.get("capture_result"), dict) else {}
        candidates.append(capture_result.get("rgb_path"))
        capture_dir = self.route2_candidate_path(capture_result.get("capture_dir"))
        if capture_dir is not None:
            candidates.append(capture_dir / "rgb.png")
        _, facade_dir, _, _ = self.route2_facade_paths()
        if facade_dir is not None:
            candidates.append(facade_dir / "coarse_rgb_panorama.png")
            candidates.append(facade_dir / "coarse_rgb_center.png")
            candidates.append(facade_dir / "coarse_rgb.png")
        for candidate in candidates:
            path = candidate if isinstance(candidate, Path) else self.route2_candidate_path(candidate)
            if path is not None and path.exists() and path.is_file():
                return path
        return None

    def cancel_route2_auto_refresh(self) -> None:
        job = getattr(self, "llm_route2_auto_refresh_job", None)
        if job:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        self.llm_route2_auto_refresh_job = None

    def on_route2_auto_refresh_toggle(self) -> None:
        auto_var = getattr(self, "llm_route2_auto_refresh_var", None)
        enabled = bool(auto_var.get()) if auto_var is not None else False
        if enabled:
            self.route2_auto_refresh_tick()
        else:
            self.cancel_route2_auto_refresh()

    def route2_auto_refresh_tick(self) -> None:
        self.cancel_route2_auto_refresh()
        try:
            if not bool(self.llm_route2_auto_refresh_var.get()):
                return
        except tk.TclError:
            return
        self.refresh_llm_route2_map()
        self.refresh_route2_preview()
        self.llm_route2_auto_refresh_job = self.root.after(1500, self.route2_auto_refresh_tick)

    def refresh_llm_route2_map(self) -> None:
        widget = getattr(self, "llm_route2_map_widget", None)
        if widget is None:
            return
        try:
            if not self.load_map_resources(force=not bool(self.map_config)):
                self.llm_route2_map_status_var.set("Route V2 Map: map unavailable")
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
            state = self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {}
            route_points: List[Dict[str, Any]] = []
            observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
            candidates = state.get("candidate_observation_points", [])
            current_facade = str(observation.get("facade", "") or "")
            current_label = str(observation.get("label", "") or "")
            rendered_candidate_keys: set[Tuple[str, str]] = set()
            if isinstance(candidates, list) and candidates:
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    item = dict(candidate)
                    label = str(item.get("label", "") or f"{item.get('house_id', '')}_{item.get('facade', '')}_obs")
                    facade = str(item.get("facade", "") or "")
                    item["label"] = label
                    item["route_point_type"] = "observation_point" if (
                        facade == current_facade or label == current_label
                    ) else "observation_candidate"
                    route_points.append(item)
                    rendered_candidate_keys.add((facade, label))
            if observation and (current_facade, current_label) not in rendered_candidate_keys:
                item = dict(observation)
                item["label"] = str(item.get("label", "") or f"{item.get('facade', '')}_obs")
                item["route_point_type"] = "observation_point"
                route_points.append(item)
            for point in state.get("facade_scan_points", []) if isinstance(state.get("facade_scan_points"), list) else []:
                if not isinstance(point, dict):
                    continue
                item = dict(point)
                band = str(item.get("height_band", "") or "")
                item["label"] = str(item.get("scan_id", "") or band or f"v2_{len(route_points)}")
                item["route_point_type"] = "scan_point"
                route_points.append(item)
            widget.set_route_plan({"route_points": route_points})
            self.llm_route2_map_status_var.set(
                f"Route V2 Map: houses={len(houses)} facade={state.get('facade', '-')} points={len(route_points)}"
            )
            self.refresh_route2_progress()
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM Route6_entrance_search v2 map failed: %s", exc)
            self.llm_route2_map_status_var.set(f"Route V2 Map: failed: {exc}")

    def route2_facade_axis_range(self, bbox: Dict[str, Any], facade: str) -> Tuple[float, float]:
        facade = str(facade or "").strip().lower()
        if facade in {"south", "north"}:
            return float(bbox["min_x"]), float(bbox["max_x"])
        return float(bbox["min_y"]), float(bbox["max_y"])

    def route2_facade_pose_from_axis(
        self,
        bbox: Dict[str, Any],
        facade: str,
        axis_value: float,
        standoff_cm: float,
        z_cm: float,
    ) -> Dict[str, float]:
        facade = str(facade or "").strip().lower()
        min_x = float(bbox["min_x"])
        max_x = float(bbox["max_x"])
        min_y = float(bbox["min_y"])
        max_y = float(bbox["max_y"])
        center_x = float(bbox.get("center_x", 0.5 * (min_x + max_x)))
        center_y = float(bbox.get("center_y", 0.5 * (min_y + max_y)))
        standoff = max(20.0, float(standoff_cm))
        axis = float(axis_value)
        if facade == "south":
            x = axis
            y = min_y - standoff
            target_x, target_y = axis, min_y
        elif facade == "north":
            x = axis
            y = max_y + standoff
            target_x, target_y = axis, max_y
        elif facade == "east":
            x = max_x + standoff
            y = axis
            target_x, target_y = max_x, axis
        else:
            x = min_x - standoff
            y = axis
            target_x, target_y = min_x, axis
        yaw = math.degrees(math.atan2(float(target_y) - float(y), float(target_x) - float(x)))
        return {
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "z": round(float(z_cm), 2),
            "yaw_deg": round(float(yaw), 2),
            "target_x": round(float(target_x), 2),
            "target_y": round(float(target_y), 2),
        }

    def route2_facade_center_axis(self, bbox: Dict[str, Any], facade: str) -> float:
        facade = str(facade or "").strip().lower()
        if facade in {"south", "north"}:
            return float(bbox.get("center_x", 0.5 * (float(bbox["min_x"]) + float(bbox["max_x"]))))
        return float(bbox.get("center_y", 0.5 * (float(bbox["min_y"]) + float(bbox["max_y"]))))

    def route2_observation_blocking_house_id(
        self,
        target_house_id: str,
        x: float,
        y: float,
        *,
        clearance_cm: Optional[float] = None,
    ) -> str:
        for obstacle in self.route_forbidden_house_bboxes(
            target_house_id=str(target_house_id or "").strip(),
            clearance_cm=float(LLM_ROUTE_HOUSE_CLEARANCE_CM if clearance_cm is None else clearance_cm),
        ):
            if self.point_inside_open_bbox(float(x), float(y), obstacle):
                return str(obstacle.get("house_id", "") or "")
        return ""

    def route2_observation_blocking_house(
        self,
        target_house_id: str,
        x: float,
        y: float,
        *,
        clearance_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        for obstacle in self.route_forbidden_house_bboxes(
            target_house_id=str(target_house_id or "").strip(),
            clearance_cm=float(LLM_ROUTE_HOUSE_CLEARANCE_CM if clearance_cm is None else clearance_cm),
        ):
            if self.point_inside_open_bbox(float(x), float(y), obstacle):
                return dict(obstacle)
        return {}

    def route2_adjust_observation_to_blocking_boundary(
        self,
        bbox: Dict[str, Any],
        facade: str,
        axis_value: float,
        z_cm: float,
        original_standoff_cm: float,
        blocking_obstacle: Dict[str, Any],
    ) -> Dict[str, Any]:
        facade = str(facade or "").strip().lower()
        if facade not in {"south", "east", "north", "west"} or not blocking_obstacle:
            return {}
        boundary_gap = float(LLM_ROUTE3_OBSERVATION_BOUNDARY_GAP_CM)
        try:
            if facade == "east":
                new_standoff = float(blocking_obstacle["min_x"]) - boundary_gap - float(bbox["max_x"])
            elif facade == "west":
                new_standoff = float(bbox["min_x"]) - (float(blocking_obstacle["max_x"]) + boundary_gap)
            elif facade == "north":
                new_standoff = float(blocking_obstacle["min_y"]) - boundary_gap - float(bbox["max_y"])
            else:
                new_standoff = float(bbox["min_y"]) - (float(blocking_obstacle["max_y"]) + boundary_gap)
        except Exception:
            return {}
        if new_standoff < float(LLM_ROUTE_MIN_PERIMETER_STANDOFF_CM):
            return {}
        pose = self.route2_facade_pose_from_axis(bbox, facade, axis_value, new_standoff, z_cm)
        return {
            "pose": pose,
            "standoff_cm": float(new_standoff),
            "adjustment": {
                "source": "blocking_house_clearance_boundary",
                "blocking_house_id": str(blocking_obstacle.get("house_id", "") or ""),
                "blocking_clearance_cm": blocking_obstacle.get("clearance_cm", LLM_ROUTE_HOUSE_CLEARANCE_CM),
                "original_standoff_cm": round(float(original_standoff_cm), 2),
                "adjusted_standoff_cm": round(float(new_standoff), 2),
                "boundary_gap_cm": boundary_gap,
            },
        }

    def route2_observation_map_bounds_report(self, x: float, y: float) -> Dict[str, Any]:
        calibration = self.map_calibration if isinstance(getattr(self, "map_calibration", None), dict) else {}
        image_w = calibration.get("image_width")
        image_h = calibration.get("image_height")
        if (image_w is None or image_h is None) and getattr(self, "map_image", None) is not None:
            try:
                image_h, image_w = self.map_image.shape[:2]
            except Exception:
                image_w = image_h = None
        if image_w is not None and image_h is not None:
            try:
                image_x, image_y = world_to_image_with_calibration(float(x), float(y), calibration)
                margin_px = 2.0
                in_bounds = (
                    margin_px <= float(image_x) <= float(image_w) - margin_px
                    and margin_px <= float(image_y) <= float(image_h) - margin_px
                )
                return {
                    "in_bounds": bool(in_bounds),
                    "source": f"{calibration.get('transform_mode', 'calibration')}_image_bounds",
                    "image_x": round(float(image_x), 2),
                    "image_y": round(float(image_y), 2),
                    "image_width": int(float(image_w)),
                    "image_height": int(float(image_h)),
                }
            except Exception:
                pass
        bounds = getattr(self, "map_world_bounds", None)
        if isinstance(bounds, tuple) and len(bounds) == 4:
            try:
                min_x, min_y, max_x, max_y = [float(value) for value in bounds]
                return {
                    "in_bounds": bool(min_x <= float(x) <= max_x and min_y <= float(y) <= max_y),
                    "source": "world_bounds",
                    "min_x": round(min_x, 2),
                    "min_y": round(min_y, 2),
                    "max_x": round(max_x, 2),
                    "max_y": round(max_y, 2),
                }
            except Exception:
                pass
        return {"in_bounds": True, "source": "unavailable"}

    def route2_observation_standoff_options(
        self,
        desired_standoff_cm: float,
        facade_info: Dict[str, Any],
        observation_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        clearance = self._as_float_or_none(observation_meta.get("observation_clearance_cm"))
        clearance = float(clearance if clearance is not None else LLM_ROUTE_HOUSE_CLEARANCE_CM)
        min_standoff = self._as_float_or_none(facade_info.get("min_standoff_cm"))
        min_standoff = float(min_standoff if min_standoff is not None else LLM_ROUTE_MIN_PERIMETER_STANDOFF_CM)
        gap = self._as_float_or_none(facade_info.get("gap_cm"))
        blocking_house_id = str(facade_info.get("blocking_house_id", "") or "")
        options: List[Dict[str, Any]] = []
        seen: set[float] = set()

        def add(value: Any, source: str) -> None:
            numeric = self._as_float_or_none(value)
            if numeric is None:
                return
            value_f = max(20.0, float(numeric))
            key = round(value_f, 2)
            if key in seen:
                return
            seen.add(key)
            options.append({"standoff_cm": value_f, "source": source})

        if gap is not None and float(gap) > 0.0 and blocking_house_id:
            max_before_neighbor_clearance = float(gap) - clearance - 2.0
            if max_before_neighbor_clearance > 20.0:
                add(max_before_neighbor_clearance, "neighbor_clearance_clamped")
            add(float(gap) * 0.5, "alley_midline")
            add(float(gap) - clearance * 0.5, "neighbor_soft_clearance_clamped")
        add(desired_standoff_cm, "facade_panorama_desired")
        add(facade_info.get("standoff_cm"), "corridor_scan_standoff")
        add(self.route_scan_standoff_cm(), "scan_default_standoff")
        add(min_standoff, "minimum_perimeter_standoff")
        return options

    def route2_observation_meta_for_standoff(
        self,
        base_meta: Dict[str, Any],
        facade_info: Dict[str, Any],
        standoff_cm: float,
        source: str,
    ) -> Dict[str, Any]:
        meta = dict(base_meta)
        desired = self._as_float_or_none(meta.get("observation_required_standoff_cm"))
        desired = float(desired if desired is not None else standoff_cm)
        gap = self._as_float_or_none(facade_info.get("gap_cm"))
        clearance = self._as_float_or_none(meta.get("observation_clearance_cm"))
        clearance = float(clearance if clearance is not None else LLM_ROUTE_HOUSE_CLEARANCE_CM)
        standoff = float(standoff_cm)
        meta["observation_standoff_mode"] = str(source or meta.get("observation_standoff_mode", "candidate"))
        meta["observation_panorama_coverage_ratio"] = round(max(0.0, min(1.0, standoff / max(1.0, desired))), 3)
        meta["observation_actual_standoff_cm"] = round(standoff, 2)
        if gap is not None:
            neighbor_margin = max(0.0, float(gap) - standoff)
            meta["observation_neighbor_gap_cm"] = round(float(gap), 2)
            meta["observation_neighbor_margin_cm"] = round(float(neighbor_margin), 2)
            meta["observation_neighbor_clearance_ok"] = bool(neighbor_margin >= clearance)
        return meta

    def route2_projected_observation_choice(
        self,
        target_house_id: str,
        bbox: Dict[str, Any],
        facade: str,
        axis_value: float,
        z_cm: float,
        desired_standoff_cm: float,
        observation_meta: Dict[str, Any],
        facade_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        standoff = float(desired_standoff_cm)
        meta = self.route2_observation_meta_for_standoff(
            observation_meta,
            facade_info,
            standoff,
            "facade_center_projection",
        )
        pose_payload = self.route2_facade_pose_from_axis(bbox, facade, axis_value, standoff, z_cm)
        boundary_adjustment: Dict[str, Any] = {}
        blocking_obstacle = self.route2_observation_blocking_house(
            target_house_id,
            float(pose_payload["x"]),
            float(pose_payload["y"]),
        )
        if blocking_obstacle:
            adjusted = self.route2_adjust_observation_to_blocking_boundary(
                bbox,
                facade,
                axis_value,
                z_cm,
                standoff,
                blocking_obstacle,
            )
            if not adjusted:
                return {}
            pose_payload = adjusted["pose"] if isinstance(adjusted.get("pose"), dict) else {}
            standoff = float(adjusted.get("standoff_cm", standoff))
            meta = self.route2_observation_meta_for_standoff(
                observation_meta,
                facade_info,
                standoff,
                "facade_center_projection_boundary_adjusted",
            )
            boundary_adjustment = adjusted.get("adjustment", {}) if isinstance(adjusted.get("adjustment"), dict) else {}
            blocking_obstacle = self.route2_observation_blocking_house(
                target_house_id,
                float(pose_payload.get("x", 0.0)),
                float(pose_payload.get("y", 0.0)),
            )
            if blocking_obstacle:
                return {}
        bounds_report = self.route2_observation_map_bounds_report(
            float(pose_payload["x"]),
            float(pose_payload["y"]),
        )
        if not bool(bounds_report.get("in_bounds", True)):
            return {}
        return {
            "pose": pose_payload,
            "axis_value": float(axis_value),
            "standoff_cm": float(standoff),
            "meta": meta,
            "bounds_report": bounds_report,
            "boundary_adjustment": boundary_adjustment,
        }

    def route2_observation_standoff_cm(
        self,
        *,
        facade_length_cm: float,
        facade_depth_cm: float,
        facade_info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        base_scan_standoff = self._as_float_or_none(facade_info.get("standoff_cm"))
        base_scan_standoff = float(base_scan_standoff if base_scan_standoff is not None else self.route_scan_standoff_cm())
        panorama_span = min(abs(float(facade_length_cm)), abs(float(facade_depth_cm)))
        desired = max(
            float(LLM_ROUTE2_OBSERVATION_MIN_STANDOFF_CM),
            float(panorama_span) * float(LLM_ROUTE2_OBSERVATION_DEPTH_FACTOR),
            base_scan_standoff,
        )
        desired = min(float(LLM_ROUTE2_OBSERVATION_MAX_STANDOFF_CM), desired)
        clearance = self._as_float_or_none(facade_info.get("clearance_cm"))
        clearance = float(clearance if clearance is not None else LLM_ROUTE_HOUSE_CLEARANCE_CM)
        standoff = desired
        mode = "facade_short_side_panorama_standoff"
        coverage_ratio = max(0.0, min(1.0, float(standoff) / max(1.0, float(desired))))
        return (
            float(standoff),
            {
                "observation_required_standoff_cm": round(float(desired), 2),
                "observation_standoff_mode": mode,
                "observation_panorama_coverage_ratio": round(float(coverage_ratio), 3),
                "observation_available_side_margin_cm": None,
                "observation_clearance_cm": round(float(clearance), 2),
                "observation_facade_length_cm": round(float(facade_length_cm), 2),
                "observation_facade_depth_cm": round(float(facade_depth_cm), 2),
                "observation_panorama_span_cm": round(float(panorama_span), 2),
                "observation_depth_factor": float(LLM_ROUTE2_OBSERVATION_DEPTH_FACTOR),
                "observation_length_factor": float(LLM_ROUTE2_OBSERVATION_LENGTH_FACTOR),
            },
        )

    def route2_safe_observation_candidates(
        self,
        target_house_id: str,
        *,
        skip_completed: bool = True,
        facade_filter: str = "",
    ) -> List[Dict[str, Any]]:
        hid = str(target_house_id or "").strip()
        if not hid:
            return []
        selected_facade = str(facade_filter or "").strip().lower()
        if selected_facade == "auto":
            selected_facade = ""
        if selected_facade and selected_facade not in {"south", "east", "north", "west"}:
            return []
        if not self.load_map_resources(force=not bool(self.map_config)):
            return []
        bbox = self.house_world_bbox_for_id(hid)
        if not bbox:
            return []
        pose = self.current_route_pose()
        px = self._as_float_or_none(pose.get("x")) if pose else None
        py = self._as_float_or_none(pose.get("y")) if pose else None
        pose_yaw = self._as_float_or_none(pose.get("yaw")) if pose else None
        corridor_info = self.scan_corridor_standoff_by_facade(hid, bbox)
        completed = set(getattr(self, "llm_route2_completed_facades", set()) or set()) if skip_completed else set()
        pose_z = self._as_float_or_none(pose.get("z")) if pose else None
        override_z = self._as_float_or_none(getattr(self, "route_observation_z_override_cm", None))
        if override_z is not None and 80.0 <= float(override_z) <= 900.0:
            z_cm = float(override_z)
        elif pose_z is not None and 80.0 <= float(pose_z) <= 900.0:
            z_cm = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), float(pose_z))
        else:
            z_cm = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), min(max(self.route2_low_z_cm(), 150.0), 300.0))
        candidates: List[Dict[str, Any]] = []
        for facade in ("south", "east", "north", "west"):
            if selected_facade and facade != selected_facade:
                continue
            if facade in completed:
                continue
            facade_info = corridor_info.get(facade, {}) if isinstance(corridor_info.get(facade), dict) else {}
            axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
            facade_length = abs(float(axis_max) - float(axis_min))
            facade_depth = abs(float(bbox["max_y"]) - float(bbox["min_y"])) if facade in {"south", "north"} else abs(float(bbox["max_x"]) - float(bbox["min_x"]))
            desired_standoff, observation_meta = self.route2_observation_standoff_cm(
                facade_length_cm=facade_length,
                facade_depth_cm=facade_depth,
                facade_info=facade_info,
            )
            facade_center_axis = self.route2_facade_center_axis(bbox, facade)
            projected_choice = self.route2_projected_observation_choice(
                hid,
                bbox,
                facade,
                facade_center_axis,
                z_cm,
                desired_standoff,
                observation_meta,
                facade_info,
            )
            if projected_choice:
                pose_payload = dict(projected_choice.get("pose", {}))
                if pose_yaw is not None:
                    yaw_delta = abs(self._normalize_angle_deg(float(pose_yaw) - float(pose_payload.get("yaw_deg", 0.0))))
                    if yaw_delta <= 15.0:
                        pose_payload["yaw_deg"] = round(float(pose_yaw), 2)
                if px is not None and py is not None:
                    distance = math.hypot(float(pose_payload["x"]) - float(px), float(pose_payload["y"]) - float(py))
                else:
                    distance = 0.0
                projected_meta = projected_choice.get("meta", {}) if isinstance(projected_choice.get("meta"), dict) else {}
                candidates.append(
                    {
                        **pose_payload,
                        "label": f"{hid}_{facade}_obs",
                        "route_point_type": "observation_point",
                        "house_id": hid,
                        "facade": facade,
                        "facade_id": self.route2_facade_id(hid, facade),
                        "axis_value": round(float(projected_choice.get("axis_value", facade_center_axis)), 2),
                        "axis_center_cm": round(float(facade_center_axis), 2),
                        "axis_center_error_cm": round(abs(float(projected_choice.get("axis_value", facade_center_axis)) - float(facade_center_axis)), 2),
                        "standoff_cm": round(float(projected_choice.get("standoff_cm", desired_standoff)), 2),
                        **projected_meta,
                        "observation_fallback_used": bool(projected_choice.get("boundary_adjustment")),
                        "safe_interval_index": -1,
                        "safe_interval_count": 0,
                        "safe_axis_min": round(float(axis_min), 2),
                        "safe_axis_max": round(float(axis_max), 2),
                        "safe_interval_source": "facade_center_projection",
                        "corridor_mode": str(facade_info.get("mode", "default") or "default"),
                        "corridor_gap_cm": facade_info.get("gap_cm"),
                        "corridor_side_margin_cm": facade_info.get("side_margin_cm"),
                        "corridor_blocking_house_id": str(facade_info.get("blocking_house_id", "") or ""),
                        "corridor_clearance_cm": facade_info.get("clearance_cm"),
                        "observation_blocking_house_id": "",
                        "observation_map_bounds": projected_choice.get("bounds_report", {}),
                        "observation_boundary_adjustment": projected_choice.get("boundary_adjustment", {}),
                        "status": "planned",
                        "distance_to_uav_cm": round(float(distance), 2),
                        "observation_selection_score": round(float(distance), 2),
                    }
                )
                continue
            best_interval: Optional[Dict[str, Any]] = None
            best_axis: Optional[float] = None
            best_distance: Optional[float] = None
            best_score: Optional[float] = None
            best_standoff: Optional[float] = None
            best_meta: Dict[str, Any] = {}
            best_bounds_report: Dict[str, Any] = {}
            best_interval_count = 0
            best_boundary_adjustment: Dict[str, Any] = {}
            for standoff_option in self.route2_observation_standoff_options(desired_standoff, facade_info, observation_meta):
                if not isinstance(standoff_option, dict):
                    continue
                standoff = self._as_float_or_none(standoff_option.get("standoff_cm"))
                if standoff is None:
                    continue
                option_source = str(standoff_option.get("source", "") or "candidate")
                option_meta = self.route2_observation_meta_for_standoff(
                    observation_meta,
                    facade_info,
                    float(standoff),
                    option_source,
                )
                observation_corridor_info = dict(corridor_info)
                observation_corridor_info[facade] = {**facade_info, "standoff_cm": float(standoff)}
                safe_intervals = self.scan_safe_intervals_by_facade(hid, bbox, observation_corridor_info)
                intervals = safe_intervals.get(facade, []) if isinstance(safe_intervals.get(facade), list) else []
                projection_interval = {
                    "index": -1,
                    "min": float(axis_min),
                    "max": float(axis_max),
                    "source": "facade_center_projection",
                    "min_segment_len_cm": 0.0,
                }
                candidate_intervals = [projection_interval] + intervals
                for interval in candidate_intervals:
                    if not isinstance(interval, dict):
                        continue
                    lo = self._as_float_or_none(interval.get("min", interval.get("axis_min")))
                    hi = self._as_float_or_none(interval.get("max", interval.get("axis_max")))
                    if lo is None or hi is None:
                        continue
                    lo = max(float(axis_min), min(float(lo), float(hi)))
                    hi = min(float(axis_max), max(float(lo), float(hi)))
                    if hi < lo:
                        continue
                    axis_value = min(max(float(facade_center_axis), lo), hi)
                    candidate_standoff = float(standoff)
                    candidate_meta = dict(option_meta)
                    boundary_adjustment: Dict[str, Any] = {}
                    pose_payload = self.route2_facade_pose_from_axis(bbox, facade, axis_value, candidate_standoff, z_cm)
                    blocking_obstacle = self.route2_observation_blocking_house(
                        hid,
                        float(pose_payload["x"]),
                        float(pose_payload["y"]),
                    )
                    if blocking_obstacle:
                        adjusted = self.route2_adjust_observation_to_blocking_boundary(
                            bbox,
                            facade,
                            axis_value,
                            z_cm,
                            candidate_standoff,
                            blocking_obstacle,
                        )
                        if not adjusted:
                            continue
                        pose_payload = adjusted["pose"] if isinstance(adjusted.get("pose"), dict) else {}
                        candidate_standoff = float(adjusted.get("standoff_cm", candidate_standoff))
                        candidate_meta = self.route2_observation_meta_for_standoff(
                            observation_meta,
                            facade_info,
                            candidate_standoff,
                            f"{option_source}_boundary_adjusted",
                        )
                        boundary_adjustment = adjusted.get("adjustment", {}) if isinstance(adjusted.get("adjustment"), dict) else {}
                        blocking_obstacle = self.route2_observation_blocking_house(
                            hid,
                            float(pose_payload.get("x", 0.0)),
                            float(pose_payload.get("y", 0.0)),
                        )
                        if blocking_obstacle:
                            continue
                    bounds_report = self.route2_observation_map_bounds_report(
                        float(pose_payload["x"]),
                        float(pose_payload["y"]),
                    )
                    if not bool(bounds_report.get("in_bounds", True)):
                        continue
                    if px is not None and py is not None:
                        distance = math.hypot(float(pose_payload["x"]) - float(px), float(pose_payload["y"]) - float(py))
                    else:
                        distance = 0.0
                    center_error = abs(float(axis_value) - float(facade_center_axis))
                    coverage_penalty = (1.0 - float(candidate_meta["observation_panorama_coverage_ratio"])) * 1500.0
                    standoff_penalty = abs(float(candidate_standoff) - float(desired_standoff)) * 0.05
                    neighbor_penalty = 0.0
                    gap = self._as_float_or_none(facade_info.get("gap_cm"))
                    blocking_neighbor = str(facade_info.get("blocking_house_id", "") or "")
                    clearance = self._as_float_or_none(candidate_meta.get("observation_clearance_cm"))
                    clearance = float(clearance if clearance is not None else LLM_ROUTE_HOUSE_CLEARANCE_CM)
                    if gap is not None and blocking_neighbor:
                        max_corridor_standoff = max(20.0, float(gap) - clearance - 2.0)
                        if float(candidate_standoff) > max_corridor_standoff:
                            neighbor_penalty += 5000.0
                    projection_bonus = -250.0 if str(interval.get("source", "") or "") == "facade_center_projection" else 0.0
                    score = float(distance) + center_error * 0.35 + coverage_penalty + standoff_penalty + neighbor_penalty + projection_bonus
                    if best_score is None or score < best_score:
                        best_interval = interval
                        best_axis = axis_value
                        best_distance = distance
                        best_score = score
                        best_standoff = float(candidate_standoff)
                        best_meta = dict(candidate_meta)
                        best_bounds_report = dict(bounds_report)
                        best_interval_count = len(intervals)
                        best_boundary_adjustment = dict(boundary_adjustment)
            if best_interval is None or best_axis is None or best_standoff is None:
                continue
            pose_payload = self.route2_facade_pose_from_axis(bbox, facade, best_axis, best_standoff, z_cm)
            if pose_yaw is not None:
                yaw_delta = abs(self._normalize_angle_deg(float(pose_yaw) - float(pose_payload.get("yaw_deg", 0.0))))
                if yaw_delta <= 15.0:
                    pose_payload["yaw_deg"] = round(float(pose_yaw), 2)
            candidates.append(
                {
                    **pose_payload,
                    "label": f"{hid}_{facade}_obs",
                    "route_point_type": "observation_point",
                    "house_id": hid,
                    "facade": facade,
                    "facade_id": self.route2_facade_id(hid, facade),
                    "axis_value": round(float(best_axis), 2),
                    "axis_center_cm": round(float(facade_center_axis), 2),
                    "axis_center_error_cm": round(abs(float(best_axis) - float(facade_center_axis)), 2),
                    "standoff_cm": round(float(best_standoff), 2),
                    **best_meta,
                    "observation_fallback_used": str(best_meta.get("observation_standoff_mode", "")) != "facade_panorama_desired",
                    "safe_interval_index": best_interval.get("index", 0),
                    "safe_interval_count": best_interval_count,
                    "safe_axis_min": round(float(best_interval.get("min", best_interval.get("axis_min", axis_min))), 2),
                    "safe_axis_max": round(float(best_interval.get("max", best_interval.get("axis_max", axis_max))), 2),
                    "safe_interval_source": str(best_interval.get("source", "") or ""),
                    "corridor_mode": str(facade_info.get("mode", "default") or "default"),
                    "corridor_gap_cm": facade_info.get("gap_cm"),
                    "corridor_side_margin_cm": facade_info.get("side_margin_cm"),
                    "corridor_blocking_house_id": str(facade_info.get("blocking_house_id", "") or ""),
                    "corridor_clearance_cm": facade_info.get("clearance_cm"),
                    "observation_blocking_house_id": "",
                    "observation_map_bounds": best_bounds_report,
                    "observation_boundary_adjustment": best_boundary_adjustment,
                    "status": "planned",
                    "distance_to_uav_cm": round(float(best_distance or 0.0), 2),
                    "observation_selection_score": round(float(best_score or 0.0), 2),
                }
            )
        candidates.sort(key=lambda item: float(item.get("observation_selection_score", item.get("distance_to_uav_cm", 0.0))))
        return candidates

    def route2_direct_observation_candidate_for_facade(self, target_house_id: str, facade: str) -> Dict[str, Any]:
        hid = str(target_house_id or "").strip()
        facade = str(facade or "").strip().lower()
        if not hid or facade not in {"south", "east", "north", "west"}:
            return {}
        bbox = self.house_world_bbox_for_id(hid)
        if not bbox:
            return {}
        pose = self.current_route_pose()
        px = self._as_float_or_none(pose.get("x")) if pose else None
        py = self._as_float_or_none(pose.get("y")) if pose else None
        pose_z = self._as_float_or_none(pose.get("z")) if pose else None
        if pose_z is not None and 80.0 <= float(pose_z) <= 900.0:
            z_cm = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), float(pose_z))
        else:
            z_cm = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), min(max(self.route2_low_z_cm(), 150.0), 300.0))
        axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
        axis_value = self.route2_facade_center_axis(bbox, facade)
        facade_length = abs(float(axis_max) - float(axis_min))
        facade_depth = (
            abs(float(bbox["max_y"]) - float(bbox["min_y"]))
            if facade in {"south", "north"}
            else abs(float(bbox["max_x"]) - float(bbox["min_x"]))
        )
        corridor_info = self.scan_corridor_standoff_by_facade(hid, bbox)
        facade_info = corridor_info.get(facade, {}) if isinstance(corridor_info.get(facade), dict) else {}
        desired_standoff, observation_meta = self.route2_observation_standoff_cm(
            facade_length_cm=facade_length,
            facade_depth_cm=facade_depth,
            facade_info=facade_info,
        )
        selected_standoff = float(desired_standoff)
        selected_meta = dict(observation_meta)
        pose_payload: Dict[str, Any] = {}
        blocking_house_id = ""
        bounds_report: Dict[str, Any] = {}
        boundary_adjustment: Dict[str, Any] = {}
        projected_choice = self.route2_projected_observation_choice(
            hid,
            bbox,
            facade,
            axis_value,
            z_cm,
            desired_standoff,
            observation_meta,
            facade_info,
        )
        if projected_choice:
            pose_payload = dict(projected_choice.get("pose", {}))
            selected_standoff = float(projected_choice.get("standoff_cm", selected_standoff))
            selected_meta = projected_choice.get("meta", {}) if isinstance(projected_choice.get("meta"), dict) else selected_meta
            bounds_report = projected_choice.get("bounds_report", {}) if isinstance(projected_choice.get("bounds_report"), dict) else {}
            boundary_adjustment = projected_choice.get("boundary_adjustment", {}) if isinstance(projected_choice.get("boundary_adjustment"), dict) else {}
        for standoff_option in self.route2_observation_standoff_options(desired_standoff, facade_info, observation_meta):
            if pose_payload:
                break
            standoff = self._as_float_or_none(standoff_option.get("standoff_cm"))
            if standoff is None:
                continue
            option_meta = self.route2_observation_meta_for_standoff(
                observation_meta,
                facade_info,
                float(standoff),
                str(standoff_option.get("source", "") or "candidate"),
            )
            option_pose = self.route2_facade_pose_from_axis(bbox, facade, axis_value, float(standoff), z_cm)
            option_blocking_obstacle = self.route2_observation_blocking_house(
                hid,
                float(option_pose["x"]),
                float(option_pose["y"]),
            )
            option_blocking_house_id = str(option_blocking_obstacle.get("house_id", "") or "")
            option_standoff = float(standoff)
            option_adjustment: Dict[str, Any] = {}
            if option_blocking_obstacle:
                adjusted = self.route2_adjust_observation_to_blocking_boundary(
                    bbox,
                    facade,
                    axis_value,
                    z_cm,
                    option_standoff,
                    option_blocking_obstacle,
                )
                if adjusted:
                    option_pose = adjusted["pose"] if isinstance(adjusted.get("pose"), dict) else option_pose
                    option_standoff = float(adjusted.get("standoff_cm", option_standoff))
                    option_meta = self.route2_observation_meta_for_standoff(
                        observation_meta,
                        facade_info,
                        option_standoff,
                        f"{standoff_option.get('source', '') or 'candidate'}_boundary_adjusted",
                    )
                    option_adjustment = adjusted.get("adjustment", {}) if isinstance(adjusted.get("adjustment"), dict) else {}
                    option_blocking_obstacle = self.route2_observation_blocking_house(
                        hid,
                        float(option_pose["x"]),
                        float(option_pose["y"]),
                    )
                    option_blocking_house_id = str(option_blocking_obstacle.get("house_id", "") or "")
            option_bounds = self.route2_observation_map_bounds_report(float(option_pose["x"]), float(option_pose["y"]))
            if not pose_payload:
                selected_standoff = float(option_standoff)
                selected_meta = dict(option_meta)
                pose_payload = dict(option_pose)
                blocking_house_id = option_blocking_house_id
                bounds_report = dict(option_bounds)
                boundary_adjustment = dict(option_adjustment)
            if not option_blocking_house_id and bool(option_bounds.get("in_bounds", True)):
                selected_standoff = float(option_standoff)
                selected_meta = dict(option_meta)
                pose_payload = dict(option_pose)
                blocking_house_id = ""
                bounds_report = dict(option_bounds)
                boundary_adjustment = dict(option_adjustment)
                break
        if not pose_payload:
            pose_payload = self.route2_facade_pose_from_axis(bbox, facade, axis_value, desired_standoff, z_cm)
            blocking_house_id = self.route2_observation_blocking_house_id(
                hid,
                float(pose_payload["x"]),
                float(pose_payload["y"]),
            )
            bounds_report = self.route2_observation_map_bounds_report(float(pose_payload["x"]), float(pose_payload["y"]))
        status = "planned" if not blocking_house_id and bool(bounds_report.get("in_bounds", True)) else "blocked"
        if px is not None and py is not None:
            distance = math.hypot(float(pose_payload["x"]) - float(px), float(pose_payload["y"]) - float(py))
        else:
            distance = 0.0
        return {
            **pose_payload,
            "label": f"{hid}_{facade}_obs",
            "route_point_type": "observation_candidate",
            "house_id": hid,
            "facade": facade,
            "facade_id": self.route2_facade_id(hid, facade),
            "axis_value": round(float(axis_value), 2),
            "axis_center_cm": round(float(axis_value), 2),
            "axis_center_error_cm": 0.0,
            "standoff_cm": round(float(selected_standoff), 2),
            **selected_meta,
            "observation_fallback_used": str(selected_meta.get("observation_standoff_mode", "")) != "facade_panorama_desired",
            "observation_candidate_source": "direct_facade_center_fallback",
            "safe_interval_index": 0,
            "safe_interval_count": 0,
            "safe_axis_min": round(float(axis_min), 2),
            "safe_axis_max": round(float(axis_max), 2),
            "safe_interval_source": "direct_bbox_center_no_safe_interval",
            "corridor_mode": str(facade_info.get("mode", "default") or "default"),
            "corridor_gap_cm": facade_info.get("gap_cm"),
            "corridor_side_margin_cm": facade_info.get("side_margin_cm"),
            "corridor_blocking_house_id": str(facade_info.get("blocking_house_id", "") or ""),
            "corridor_clearance_cm": facade_info.get("clearance_cm"),
            "observation_blocking_house_id": blocking_house_id,
            "observation_map_bounds": bounds_report,
            "observation_boundary_adjustment": boundary_adjustment,
            "observation_block_reason": "non_target_house" if blocking_house_id else ("" if bounds_report.get("in_bounds", True) else "map_boundary"),
            "status": status,
            "distance_to_uav_cm": round(float(distance), 2),
            "observation_selection_score": round(float(distance) + (20000.0 if status == "blocked" else 0.0), 2),
        }

    def route2_all_facade_observation_candidates(
        self,
        target_house_id: str,
        *,
        skip_completed: bool = False,
    ) -> List[Dict[str, Any]]:
        raw_candidates = self.route2_safe_observation_candidates(
            target_house_id,
            skip_completed=skip_completed,
            facade_filter="auto",
        )
        by_facade: Dict[str, Dict[str, Any]] = {}
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            facade = str(candidate.get("facade", "") or "").strip().lower()
            if facade in {"south", "east", "north", "west"} and facade not in by_facade:
                by_facade[facade] = candidate
        for facade in ("south", "east", "north", "west"):
            if facade in by_facade:
                continue
            fallback = self.route2_direct_observation_candidate_for_facade(target_house_id, facade)
            if fallback:
                by_facade[facade] = fallback
        candidates = [by_facade[facade] for facade in ("south", "east", "north", "west") if facade in by_facade]
        candidates.sort(key=lambda item: float(item.get("observation_selection_score", item.get("distance_to_uav_cm", 0.0))))
        return candidates

    def route2_selected_state(self) -> Dict[str, Any]:
        state = self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {}
        return state

    def route2_update_state(self, **updates: Any) -> Dict[str, Any]:
        state = dict(self.route2_selected_state())
        state.update(updates)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route2_state = state
        return state

    def route2_write_state_artifact(self) -> None:
        output_dir = self.route2_state_output_dir()
        if output_dir is None:
            return
        self.write_json_artifact(output_dir / "facade_v2_state.json", self.route2_selected_state())

    def apply_route2_observation_plan(
        self,
        target_house_id: str,
        observation: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        *,
        status_label: str,
    ) -> None:
        previous_state = self.route2_selected_state()
        previous_target = str(previous_state.get("target_house_id", "") or "")
        current_output_dir = self.route2_state_output_dir() if previous_target == str(target_house_id or "") else None
        output_dir = current_output_dir or self.make_route2_facade_output_dir(target_house_id)
        facade = str(observation.get("facade", "") or "")
        facade_dir = self.route2_facade_dir(output_dir, target_house_id, facade)
        state = {
            "mode": "facade_by_facade_vlm_v2",
            "target_house_id": target_house_id,
            "output_dir": str(output_dir),
            "facade": facade,
            "facade_id": self.route2_facade_id(target_house_id, facade),
            "floor_height_m": self.route2_floor_height_m(),
            "default_floors": self.route2_default_floors(),
            "low_z_cm": self.route2_low_z_cm(),
            "z_step_cm": self.route2_z_step_cm(),
            "density_mode": self.route2_density_mode(),
            "observation_point": observation,
            "candidate_observation_points": candidates,
            "facade_analysis": {},
            "facade_search_plan": {},
            "facade_scan_points": [],
            "facade_capture_rows": [],
            "validation_report": {},
            "completed_facades": sorted(getattr(self, "llm_route2_completed_facades", set()) or []),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.llm_route2_state = state
        self.write_json_artifact(facade_dir / "facade_observation_point.json", observation)
        self.route2_write_state_artifact()

        def ui_update() -> None:
            self.llm_route2_facade_var.set(f"Facade: {facade or 'n/a'}")
            selected_var = getattr(self, "llm_route2_selected_facade_var", None)
            if selected_var is not None and facade:
                try:
                    selected_var.set(facade)
                except tk.TclError:
                    pass
            self.llm_route2_status_var.set(
                f"LLM Route V2: {status_label} facade={facade} obs=({observation.get('x')},{observation.get('y')},{observation.get('z')})"
            )
            self.refresh_route2_preview()
            self.refresh_llm_route2_map()
            self.refresh_route3_support_views()

        self.root.after(0, ui_update)

    def on_route2_plan_nearest_facade(self) -> None:
        target_house_id = self.selected_route_target_house_id()
        if not target_house_id:
            self.llm_route2_status_var.set("LLM Route V2: select a target house first.")
            return
        candidates = self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False)
        if not candidates:
            self.llm_route2_status_var.set("LLM Route V2: no safe facade observation point.")
            return
        movable_candidates = [
            candidate for candidate in candidates
            if (
                isinstance(candidate, dict)
                and str(candidate.get("status", "") or "") != "blocked"
                and not str(candidate.get("observation_blocking_house_id", "") or "")
            )
        ]
        if not movable_candidates:
            self.route2_update_state(
                mode="facade_by_facade_vlm_v2",
                target_house_id=target_house_id,
                candidate_observation_points=candidates,
                facade="",
                observation_point={},
            )
            self.llm_route2_status_var.set("LLM Route V2: all facade observation points are blocked by house safety validation.")
            self.refresh_route2_preview()
            self.refresh_llm_route2_map()
            return
        selected_observation = dict(movable_candidates[0] if movable_candidates else candidates[0])
        self.apply_route2_observation_plan(
            target_house_id,
            selected_observation,
            candidates,
            status_label=f"4-facade candidates={len(candidates)} nearest",
        )

    def on_route2_plan_selected_facade(self) -> None:
        target_house_id = self.selected_route_target_house_id()
        if not target_house_id:
            self.llm_route2_status_var.set("LLM Route V2: select a target house first.")
            return
        selected_var = getattr(self, "llm_route2_selected_facade_var", None)
        selected = str(selected_var.get() if selected_var is not None else "").strip().lower()
        if not selected or selected == "auto":
            self.on_route2_plan_nearest_facade()
            return
        all_candidates = self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False)
        selected_candidates = [
            candidate for candidate in all_candidates
            if isinstance(candidate, dict) and str(candidate.get("facade", "") or "").strip().lower() == selected
        ]
        if not selected_candidates:
            self.llm_route2_status_var.set(f"LLM Route V2: no safe observation point for {selected}.")
            return
        selected_observation = dict(selected_candidates[0])
        blocking_house_id = str(selected_observation.get("observation_blocking_house_id", "") or "")
        if str(selected_observation.get("status", "") or "") == "blocked" or blocking_house_id:
            self.route2_update_state(
                mode="facade_by_facade_vlm_v2",
                target_house_id=target_house_id,
                candidate_observation_points=all_candidates,
                facade=selected,
                observation_point={},
            )
            blocked_by = f" by {blocking_house_id}" if blocking_house_id else ""
            self.llm_route2_status_var.set(f"LLM Route V2: {selected} observation blocked{blocked_by}; not selected.")
            self.refresh_route2_preview()
            self.refresh_llm_route2_map()
            return
        self.apply_route2_observation_plan(
            target_house_id,
            selected_observation,
            all_candidates,
            status_label="selected",
        )

    def on_route2_clear(self) -> None:
        self.llm_route2_state = {}
        self.llm_route2_completed_facades = set()
        self.llm_route2_facade_var.set("Facade: n/a")
        selected_var = getattr(self, "llm_route2_selected_facade_var", None)
        if selected_var is not None:
            try:
                selected_var.set("auto")
            except tk.TclError:
                pass
        self.llm_route2_facade_status_var.set("Processed: no facade")
        self.llm_route2_status_var.set("LLM Route V2: cleared.")
        self.refresh_route2_preview()
        self.refresh_llm_route2_map()

    def on_route2_stop(self) -> None:
        self.route_stop_event.set()
        self.llm_route2_status_var.set("LLM Route V2: stop requested.")

    def on_route2_next_facade(self) -> None:
        state = self.route2_selected_state()
        facade = str(state.get("facade", "") or "")
        if facade:
            self.llm_route2_completed_facades.add(facade)
        self.on_route2_plan_nearest_facade()

    def route2_read_image_b64(self, image_path: Path) -> str:
        if not image_path.exists():
            return ""
        return base64.b64encode(image_path.read_bytes()).decode("ascii")

    def route2_fallback_facade_analysis(self, reason: str = "") -> Dict[str, Any]:
        state = self.route2_selected_state()
        facade = str(state.get("facade", "") or "")
        floors = self.route2_default_floors()
        bands = ["single_300cm"] if floors <= 2 else ["low_ground_250cm", "upper_floor_600cm"]
        return {
            "facade_id": str(state.get("facade_id", "") or ""),
            "house_id": str(state.get("target_house_id", "") or ""),
            "facade": facade,
            "floor_count_estimate": floors,
            "semantic_complexity": "medium",
            "terrace_or_awning_risk": "unknown",
            "target_score": "medium",
            "detected_cues": [],
            "recommended_translation_span": "medium",
            "recommended_scan_density": "medium",
            "recommended_height_bands": bands,
            "reason": reason or "Fallback facade analysis: no usable VLM result.",
            "planner_source": "fallback_no_or_failed_vlm",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def route2_normalize_target_score(self, value: Any, fallback: str = "medium") -> str:
        number = self._as_float_or_none(value)
        if number is not None:
            if float(number) >= 0.67:
                return "high"
            if float(number) >= 0.34:
                return "medium"
            return "low"
        text = str(value or fallback).strip().lower()
        aliases = {
            "small": "high",
            "dense": "high",
            "important": "high",
            "large": "low",
            "sparse": "low",
            "minor": "low",
        }
        text = aliases.get(text, text)
        return text if text in {"low", "medium", "high"} else fallback

    def route2_normalize_translation_span(self, value: Any, *, target_score: str, complexity: str) -> str:
        text = str(value or "").strip().lower()
        aliases = {
            "high": "small",
            "dense": "small",
            "low": "large",
            "sparse": "large",
            "normal": "medium",
            "mid": "medium",
        }
        text = aliases.get(text, text)
        if text in {"small", "medium", "large"}:
            return text
        if str(target_score) == "high" or str(complexity) == "high":
            return "small"
        if str(target_score) == "low" and str(complexity) == "low":
            return "large"
        return "medium"

    def route2_normalize_facade_analysis(self, parsed: Dict[str, Any], fallback_reason: str = "") -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}
        fallback = self.route2_fallback_facade_analysis(fallback_reason)
        floors_raw = self._as_float_or_none(parsed.get("floor_count_estimate", parsed.get("floors")))
        floors = int(round(float(floors_raw))) if floors_raw is not None else int(fallback["floor_count_estimate"])
        floors = max(1, min(int(LLM_ROUTE2_MAX_FLOORS), floors))
        complexity = str(parsed.get("semantic_complexity", fallback["semantic_complexity"]) or "").strip().lower()
        if complexity not in {"low", "medium", "high"}:
            complexity = "medium"
        density = str(parsed.get("recommended_scan_density", complexity) or "").strip().lower()
        if density not in {"low", "medium", "high"}:
            density = complexity
        target_score = self.route2_normalize_target_score(parsed.get("target_score", parsed.get("target_priority")), "medium")
        span = self.route2_normalize_translation_span(
            parsed.get("recommended_translation_span", parsed.get("translation_span", parsed.get("scan_spacing_class"))),
            target_score=target_score,
            complexity=complexity,
        )
        terrace = str(parsed.get("terrace_or_awning_risk", parsed.get("terrace_awning_risk", "unknown")) or "unknown").strip().lower()
        if terrace not in {"none", "low", "medium", "high", "unknown", "yes", "no"}:
            terrace = "unknown"
        raw_cues = parsed.get("detected_cues", [])
        cues: List[Dict[str, Any]] = []
        if isinstance(raw_cues, list):
            for item in raw_cues[:16]:
                if not isinstance(item, dict):
                    continue
                region = str(item.get("region", "") or item.get("semantic_region", "") or "").strip().lower()
                horizontal = str(item.get("horizontal", "") or "").strip().lower()
                vertical = str(item.get("vertical", "") or "").strip().lower()
                if not region and horizontal and vertical:
                    region = f"{vertical}-{horizontal}"
                if not region:
                    region = "mid-center"
                cues.append(
                    {
                        "type": str(item.get("type", item.get("cue_type", "cue")) or "cue"),
                        "region": region,
                        "confidence": self._as_float_or_none(item.get("confidence")) if self._as_float_or_none(item.get("confidence")) is not None else None,
                        "reason": str(item.get("reason", "") or ""),
                    }
                )
        bands = parsed.get("recommended_height_bands", [])
        if not isinstance(bands, list) or not bands:
            bands = ["single_300cm"] if floors <= 2 else ["low_ground_250cm", "upper_floor_600cm"]
        normalized = dict(fallback)
        normalized.update(
            {
                "floor_count_estimate": floors,
                "semantic_complexity": complexity,
                "terrace_or_awning_risk": terrace,
                "target_score": target_score,
                "detected_cues": cues,
                "recommended_scan_density": density,
                "recommended_translation_span": span,
                "recommended_height_bands": [str(item) for item in bands[: int(LLM_ROUTE2_MAX_FLOORS)]],
                "reason": str(parsed.get("reason", fallback["reason"]) or fallback["reason"]),
                "planner_source": "vlm" if parsed else fallback["planner_source"],
            }
        )
        return normalized

    def route2_build_facade_vlm_prompt(self) -> Tuple[str, str]:
        state = self.route2_selected_state()
        house_id = str(state.get("target_house_id", "") or "")
        facade = str(state.get("facade", "") or "")
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        bbox = self.house_world_bbox_for_id(house_id)
        context = {
            "house_id": house_id,
            "facade": facade,
            "bbox_world_cm": bbox,
            "observation_pose": observation,
            "observation_obstacle_analysis": state.get("observation_obstacle_analysis", {}),
            "floor_height_m": self.route2_floor_height_m(),
            "default_floors": self.route2_default_floors(),
            "expected_schema": LLM_ROUTE2_FACADE_ANALYSIS_SCHEMA,
        }
        system_prompt = (
            "You analyze one facade RGB image for a UAV house entrance search planner. "
            "Return only compact JSON. Estimate visible floor count and whether terrace/awning/porch roof "
            "may block the view. Score facade entrance-search target richness as low/medium/high, then choose "
            "small/medium/large lateral translation span for later scan points."
        )
        user_prompt = (
            "Analyze this facade image and return JSON with keys: "
            "floor_count_estimate, semantic_complexity(low|medium|high), terrace_or_awning_risk, "
            "target_score(low|medium|high or 0-1), detected_cues, recommended_translation_span(small|medium|large), "
            "recommended_height_bands, reason. For ordinary one/two-floor houses prefer one search band near 300cm; "
            "for three-floor/tall facades prefer two bands near 250cm and 600cm.\n"
            f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
        )
        return system_prompt, user_prompt

    def route2_facade_paths(self) -> Tuple[Optional[Path], Optional[Path], str, str]:
        state = self.route2_selected_state()
        house_id = str(state.get("target_house_id", "") or "")
        facade = str(state.get("facade", "") or "")
        output_dir = self.route2_state_output_dir()
        if not house_id or not facade or output_dir is None:
            return None, None, house_id, facade
        facade_dir = self.route2_facade_dir(output_dir, house_id, facade)
        return output_dir, facade_dir, house_id, facade

    def on_route2_move_to_observation(self) -> None:
        session = self.active_session()
        if session is None:
            return
        state = self.route2_selected_state()
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        if not observation:
            self.on_route2_plan_nearest_facade()
            state = self.route2_selected_state()
            observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        if not observation:
            self.llm_route2_status_var.set("LLM Route V2: plan nearest facade first.")
            return
        if str(observation.get("status", "") or "") == "blocked":
            self.llm_route2_status_var.set("LLM Route V2: selected observation is blocked; choose another facade.")
            return
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route2_status_var.set("LLM Route V2: wait for current worker.")
            return
        self.route_stop_event.clear()
        self.route_thread = threading.Thread(target=lambda: self.route2_move_to_observation_worker(session), daemon=True)
        self.route_thread.start()

    def route2_move_to_observation_worker(self, session: flight.DroneFlightSession) -> None:
        state = self.route2_selected_state()
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        house_id = str(state.get("target_house_id", "") or observation.get("house_id", "") or "")
        facade = str(state.get("facade", "") or observation.get("facade", "") or "")
        if not observation:
            self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: missing observation point."))
            return
        planned_pose = {
            "x": float(observation.get("x", 0.0)),
            "y": float(observation.get("y", 0.0)),
            "z": float(observation.get("z", 300.0)),
            "yaw": float(observation.get("yaw_deg", observation.get("yaw", 0.0))),
        }
        self.root.after(
            0,
            lambda: self.llm_route2_status_var.set(
                f"LLM Route V2: moving to obs {house_id}_{facade} ({planned_pose['x']:.1f},{planned_pose['y']:.1f},{planned_pose['z']:.1f})"
            ),
        )
        result = self.safe("Route V2 move to observation", lambda: session.set_pose(planned_pose))
        if isinstance(result, dict):
            self.root.after(0, lambda r=result: self.apply_state(r))
            move_payload = {
                "house_id": house_id,
                "facade": facade,
                "planned_pose": planned_pose,
                "result_pose": result.get("pose", {}) if isinstance(result.get("pose"), dict) else {},
                "moved_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            self.route2_update_state(last_observation_move=move_payload)
            self.route2_write_state_artifact()
            self.root.after(0, self.refresh_route2_preview)
            self.root.after(0, self.refresh_llm_route2_map)
            self.root.after(0, lambda: self.llm_route2_status_var.set(f"LLM Route V2: reached obs {house_id}_{facade}."))
        else:
            self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: move to obs failed."))

    def on_route2_capture_facade_rgb(self) -> None:
        session = self.active_session()
        if session is None:
            return
        state = self.route2_selected_state()
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        if not observation:
            self.on_route2_plan_nearest_facade()
            state = self.route2_selected_state()
            observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        if not observation:
            self.llm_route2_status_var.set("LLM Route V2: plan nearest facade first.")
            return
        if str(observation.get("status", "") or "") == "blocked":
            self.llm_route2_status_var.set("LLM Route V2: selected observation is blocked; choose another facade.")
            return
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route2_status_var.set("LLM Route V2: wait for current worker.")
            return
        self.sync_capture_options_to_session(session)
        self.route_stop_event.clear()
        self.route_thread = threading.Thread(target=lambda: self.route2_capture_facade_rgb_worker(session), daemon=True)
        self.route_thread.start()

    def route2_capture_facade_rgb_worker(self, session: flight.DroneFlightSession) -> None:
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: missing facade output directory."))
            return
        state = self.route2_selected_state()
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        planned_pose = {
            "x": float(observation.get("x", 0.0)),
            "y": float(observation.get("y", 0.0)),
            "z": float(observation.get("z", 600.0)),
            "yaw": float(observation.get("yaw_deg", observation.get("yaw", 0.0))),
        }
        self.root.after(0, lambda: self.llm_route2_status_var.set(f"LLM Route V2: moving to {house_id}_{facade} observation."))
        result = self.safe("Route V2 observation set pose", lambda: session.set_pose(planned_pose))
        if isinstance(result, dict):
            self.root.after(0, lambda r=result: self.apply_state(r))
        if self.route_stop_event.wait(0.3):
            self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: stopped before RGB capture."))
            return
        action_detail = dict(self.build_stream_action_detail())
        action_detail.update(
            {
                "source": "llm_route_v2_facade_observation",
                "house_id": house_id,
                "facade": facade,
                "facade_id": self.route2_facade_id(house_id, facade),
                "planned_pose": planned_pose,
            }
        )
        frame_index = self.route2_next_frame_index(output_dir)
        capture_result = self.safe(
            "Route V2 coarse RGB/LiDAR capture",
            lambda: session.capture_lidar_stream_frame(output_dir, frame_index, action_detail=action_detail),
        )
        if not isinstance(capture_result, dict):
            self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: coarse RGB capture failed."))
            return
        coarse_rgb_path = facade_dir / "coarse_rgb.png"
        rgb_path = self.route2_candidate_path(capture_result.get("rgb_path"))
        if rgb_path is not None and rgb_path.exists() and rgb_path.is_file():
            try:
                if rgb_path.resolve() != coarse_rgb_path.resolve():
                    coarse_rgb_path.write_bytes(rgb_path.read_bytes())
            except Exception:
                coarse_rgb_path.write_bytes(rgb_path.read_bytes())
        coarse_rgb_value = str(coarse_rgb_path if coarse_rgb_path.exists() else (rgb_path or ""))
        point_cloud_path = self.route2_candidate_path(capture_result.get("point_cloud_world_standard_m_ply_path"))
        if point_cloud_path is not None and point_cloud_path.exists() and point_cloud_path.is_file():
            (facade_dir / "coarse_lidar_preview.ply").write_bytes(point_cloud_path.read_bytes())
        capture_payload = {
            "capture_kind": "facade_coarse_observation",
            "house_id": house_id,
            "facade": facade,
            "planned_pose": planned_pose,
            "capture_result": capture_result,
            "coarse_rgb_path": coarse_rgb_value,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(facade_dir / "coarse_capture.json", capture_payload)
        capture_row = {
            "frame_index": int(capture_result.get("frame_index", frame_index)),
            "capture_time": capture_result.get("capture_time", ""),
            "capture_kind": "facade_coarse_observation",
            "scan_id": f"{house_id}_{facade}_coarse_observation",
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "planned_pose": planned_pose,
            "pose": capture_result.get("pose", {}),
            "commanded_pose": capture_result.get("commanded_pose", {}),
            "actual_pose": capture_result.get("actual_pose", {}),
            "pose_error": capture_result.get("pose_error", {}),
            "capture_dir": capture_result.get("capture_dir", ""),
            "rgb_path": capture_result.get("rgb_path", ""),
            "point_cloud_world_standard_m_npy_path": capture_result.get("point_cloud_world_standard_m_npy_path", ""),
            "point_cloud_world_standard_m_ply_path": capture_result.get("point_cloud_world_standard_m_ply_path", ""),
            "point_cloud_preview_path": capture_result.get("point_cloud_preview_path", ""),
            "point_count": int(capture_result.get("point_count", 0) or 0),
            "action_detail": action_detail,
        }
        self.append_jsonl(output_dir / "lidar_capture_log.jsonl", capture_row)
        self.append_jsonl(facade_dir / "coarse_lidar_capture_log.jsonl", capture_row)
        self.route2_write_lidar_summary(output_dir, running=True)
        self.route2_update_state(coarse_capture=capture_payload, coarse_rgb_path=coarse_rgb_value)
        self.route2_write_state_artifact()
        self.root.after(0, self.refresh_route2_preview)
        self.root.after(0, self.refresh_llm_route2_map)
        self.root.after(0, lambda: self.llm_route2_status_var.set(f"LLM Route V2: coarse RGB saved -> {coarse_rgb_path}"))

    def on_route2_analyze_facade_vlm(self) -> None:
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route2_status_var.set("LLM Route V2: wait for current worker.")
            return
        if not self.route2_selected_state().get("observation_point"):
            self.llm_route2_status_var.set("LLM Route V2: plan/capture facade first.")
            return
        self.route_thread = threading.Thread(target=self.route2_analyze_facade_vlm_worker, daemon=True)
        self.route_thread.start()

    def route2_analyze_facade_vlm_worker(self) -> None:
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: missing facade output directory."))
            return
        image_path = self.route2_current_rgb_path()
        analysis: Dict[str, Any]
        response_payload: Dict[str, Any] = {}
        self.root.after(0, lambda: self.llm_route2_status_var.set(f"LLM Route V2: analyzing {house_id}_{facade} with VLM..."))
        try:
            if image_path is None:
                raise RuntimeError("missing coarse RGB image")
            image_b64 = self.route2_read_image_b64(image_path)
            if not image_b64:
                raise RuntimeError("missing coarse RGB image")
            if not self.effective_llm_api_key():
                raise RuntimeError("missing API key")
            system_prompt, user_prompt = self.route2_build_facade_vlm_prompt()
            response_payload = self.call_configured_llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=900,
                json_schema=LLM_ROUTE2_FACADE_ANALYSIS_SCHEMA,
                image_b64=image_b64,
            )
            parsed = extract_json_object(str(response_payload.get("raw_text", "") or ""))
            analysis = self.route2_normalize_facade_analysis(parsed, "")
            response_payload["parsed"] = parsed
        except Exception as exc:
            LOGGER.warning("Route V2 facade VLM analysis fallback: %s", exc)
            analysis = self.route2_fallback_facade_analysis(str(exc))
            response_payload = {"error": str(exc), "fallback_used": True}
        self.write_json_artifact(facade_dir / "llm_facade_response.json", response_payload)
        self.write_json_artifact(facade_dir / "llm_facade_analysis.json", analysis)
        self.route2_update_state(facade_analysis=analysis)
        self.route2_write_state_artifact()
        self.root.after(0, self.refresh_route2_preview)
        self.root.after(
            0,
            lambda a=analysis: self.llm_route2_status_var.set(
                f"LLM Route V2: analysis {a.get('planner_source')} floors={a.get('floor_count_estimate')} complexity={a.get('semantic_complexity')}"
            ),
        )

    def route2_spacing_from_analysis(self, analysis: Dict[str, Any]) -> float:
        state = self.route2_selected_state()
        house_id = str(state.get("target_house_id", "") or analysis.get("house_id", "") or "")
        facade = str(state.get("facade", "") or analysis.get("facade", "") or "")
        density = self.route2_density_rule(house_id, facade, analysis).get("density", "medium")
        return float(LLM_ROUTE2_RULE_DENSITY_SPACING_CM.get(str(density), LLM_ROUTE2_RULE_DENSITY_SPACING_CM["medium"]))

    def route2_translation_span_from_analysis(self, analysis: Dict[str, Any]) -> str:
        density = self.route2_density_rule(
            str(analysis.get("house_id", "") or self.route2_selected_state().get("target_house_id", "") or ""),
            str(analysis.get("facade", "") or self.route2_selected_state().get("facade", "") or ""),
            analysis,
        ).get("density", "medium")
        return {"dense": "small", "medium": "medium", "sparse": "large"}.get(str(density), "medium")

    def route2_density_rank(self, density: str) -> int:
        return {"sparse": 0, "medium": 1, "dense": 2}.get(str(density or "").strip().lower(), 1)

    def route2_density_from_rank(self, rank: int) -> str:
        if int(rank) >= 2:
            return "dense"
        if int(rank) <= 0:
            return "sparse"
        return "medium"

    def route2_default_facade_density(self, house_id: str, facade: str) -> str:
        hid = str(house_id or "").strip()
        facade_key = str(facade or "").strip().lower()
        mapping = LLM_ROUTE2_HOUSE_FACADE_FALLBACK_DENSITY.get(hid, {})
        density = str(mapping.get(facade_key, "") or "").strip().lower()
        return density if density in {"dense", "medium", "sparse"} else "medium"

    def route2_is_entry_cue(self, cue: Dict[str, Any]) -> bool:
        text = " ".join(
            str(cue.get(key, "") or "").lower()
            for key in ("type", "cue_type", "region", "semantic_region", "reason")
            if isinstance(cue, dict)
        )
        return any(keyword in text for keyword in ("door", "entrance", "entry", "gate", "porch", "awning"))

    def route2_has_entry_cue(self, analysis: Dict[str, Any]) -> bool:
        cues = analysis.get("detected_cues", []) if isinstance(analysis.get("detected_cues"), list) else []
        if any(isinstance(cue, dict) and self.route2_is_entry_cue(cue) for cue in cues):
            return True
        text = " ".join(
            str(analysis.get(key, "") or "").lower()
            for key in ("reason", "terrace_or_awning_risk")
        )
        return any(keyword in text for keyword in ("door", "entrance", "entry", "gate", "porch", "awning"))

    def route2_density_rule(self, house_id: str, facade: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.route2_density_mode()
        if mode != "auto":
            density = {"high": "dense", "medium": "medium", "low": "sparse"}.get(mode, "medium")
            return {
                "density": density,
                "source": "manual_density_mode",
                "reason": f"Density mode override: {mode}",
            }
        default_density = self.route2_default_facade_density(house_id, facade)
        planner_source = str(analysis.get("planner_source", "") or "").lower()
        if planner_source.startswith("fallback") or not analysis:
            return {
                "density": default_density,
                "source": "house_facade_fallback",
                "reason": f"Fallback density for house={house_id} facade={facade}",
            }
        target_score = self.route2_normalize_target_score(analysis.get("target_score"), "medium")
        recommended_density = str(analysis.get("recommended_scan_density", "") or "").strip().lower()
        if recommended_density not in {"low", "medium", "high"}:
            recommended_density = str(analysis.get("semantic_complexity", "medium") or "medium").strip().lower()
        recommended_density = recommended_density if recommended_density in {"low", "medium", "high"} else "medium"
        has_entry_cue = self.route2_has_entry_cue(analysis)
        density = default_density
        if recommended_density == "high" or target_score == "high" or (has_entry_cue and target_score in {"medium", "high"}):
            density = "dense"
        elif recommended_density == "low" and target_score == "low" and not has_entry_cue:
            density = "sparse"
        elif recommended_density == "medium" or target_score == "medium":
            density = self.route2_density_from_rank(max(self.route2_density_rank(default_density), 1))
        return {
            "density": density,
            "source": "vlm_semantics_rule_fusion",
            "default_density": default_density,
            "recommended_scan_density": recommended_density,
            "target_score": target_score,
            "entry_cue": bool(has_entry_cue),
            "reason": "Final scan density is rule-clamped from facade role, VLM density, target score, and entry cues.",
        }

    def route2_detail_scan_preferred_standoff_cm(self, analysis: Dict[str, Any]) -> Tuple[float, str]:
        span = self.route2_translation_span_from_analysis(analysis)
        max_range = float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
        min_range = float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
        ratio = float(LLM_ROUTE2_SCAN_STANDOFF_RATIO.get(span, LLM_ROUTE2_SCAN_STANDOFF_RATIO["medium"]))
        upper = min(float(LLM_ROUTE2_SCAN_STANDOFF_MAX_CM), max(min_range + 80.0, max_range - 100.0))
        lower = max(float(LLM_ROUTE2_SCAN_STANDOFF_MIN_CM), min_range + 80.0)
        preferred = min(max(lower, max_range * ratio), upper)
        return float(preferred), span

    def route2_scan_standoff_candidates(self, preferred: float, span: str) -> List[float]:
        if span == "large":
            raw = [preferred, preferred + 180.0, preferred + 320.0, preferred - 140.0, preferred - 280.0]
        elif span == "small":
            raw = [preferred, preferred - 100.0, preferred + 120.0, preferred - 200.0, preferred + 240.0]
        else:
            raw = [preferred, preferred + 140.0, preferred - 140.0, preferred + 280.0, preferred - 260.0]
        min_range = float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
        max_range = float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
        lower = max(float(LLM_ROUTE2_SCAN_STANDOFF_MIN_CM), min_range + 80.0)
        upper = min(float(LLM_ROUTE2_SCAN_STANDOFF_MAX_CM), max(lower, max_range - 100.0))
        values: List[float] = []
        for value in raw:
            clipped = round(min(max(lower, float(value)), upper), 2)
            if all(abs(clipped - existing) > 1.0 for existing in values):
                values.append(clipped)
        return values

    def route2_safe_scan_intervals_for_standoff(
        self,
        house_id: str,
        facade: str,
        bbox: Dict[str, Any],
        standoff: float,
    ) -> List[Dict[str, Any]]:
        corridor_info = self.scan_corridor_standoff_by_facade(house_id, bbox)
        facade_info = corridor_info.get(facade, {}) if isinstance(corridor_info.get(facade), dict) else {}
        corridor_info[facade] = {**facade_info, "standoff_cm": float(standoff), "safe": True}
        safe = self.scan_safe_intervals_by_facade(house_id, bbox, corridor_info)
        return safe.get(facade, []) if isinstance(safe.get(facade), list) else []

    def route2_select_scan_geometry(
        self,
        house_id: str,
        facade: str,
        bbox: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        preferred, span = self.route2_detail_scan_preferred_standoff_cm(analysis)
        best: Dict[str, Any] = {}
        candidates = self.route2_scan_standoff_candidates(preferred, span)
        observation = self.route2_selected_state().get("observation_point", {})
        if isinstance(observation, dict) and str(observation.get("facade", "") or "").strip().lower() == str(facade).strip().lower():
            observed_standoff = self._as_float_or_none(observation.get("standoff_cm"))
            if observed_standoff is not None:
                min_range = float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM))
                max_range = float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM))
                if min_range + 40.0 <= float(observed_standoff) <= max_range - 40.0:
                    if all(abs(float(observed_standoff) - value) > 1.0 for value in candidates):
                        candidates.append(round(float(observed_standoff), 2))
        for standoff in candidates:
            intervals = self.route2_safe_scan_intervals_for_standoff(house_id, facade, bbox, standoff)
            if not intervals:
                continue
            safe_length = sum(
                max(0.0, float(item.get("max", 0.0)) - float(item.get("min", 0.0)))
                for item in intervals
                if isinstance(item, dict)
            )
            best = {
                "standoff_cm": round(float(standoff), 2),
                "preferred_standoff_cm": round(float(preferred), 2),
                "translation_span": span,
                "safe_intervals": intervals,
                "safe_interval_count": len(intervals),
                "safe_axis_total_cm": round(float(safe_length), 2),
                "scan_standoff_mode": (
                    "observation_tight_corridor_detail_scan"
                    if isinstance(observation, dict)
                    and abs(float(standoff) - float(observation.get("standoff_cm", float("nan")))) <= 1.0
                    else "lidar_range_detail_scan"
                ),
            }
            break
        if not best:
            best = {
                "standoff_cm": round(float(preferred), 2),
                "preferred_standoff_cm": round(float(preferred), 2),
                "translation_span": span,
                "safe_intervals": [],
                "safe_interval_count": 0,
                "safe_axis_total_cm": 0.0,
                "scan_standoff_mode": "no_safe_interval_for_lidar_range_detail_scan",
            }
        return best

    def route2_height_bands_from_analysis(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        floors = self._as_float_or_none(analysis.get("floor_count_estimate"))
        floor_count = int(round(float(floors))) if floors is not None else self.route2_default_floors()
        floor_count = max(1, min(int(LLM_ROUTE2_MAX_FLOORS), floor_count))
        low_z = self.route2_low_z_cm()
        z_step = self.route2_z_step_cm()
        bands: List[Dict[str, Any]] = []
        if floor_count <= 2:
            z_cm = low_z + float(LLM_ROUTE2_SINGLE_BAND_Z_OFFSET_CM)
            return [
                {
                    "height_band": "single_300cm",
                    "floor_index": 1,
                    "z_cm": round(float(z_cm), 2),
                    "floor_height_m": self.route2_floor_height_m(),
                    "floor_count_estimate": floor_count,
                    "search_layer_role": "one_pass_for_one_or_two_floor_facade",
                }
            ]
        band_specs = [
            ("low_ground_250cm", 1, low_z, "low_ground_floor_band"),
            ("upper_floor_600cm", 2, low_z + z_step, "upper_floor_band"),
        ]
        for name, floor_index, z_cm, role in band_specs:
            bands.append(
                {
                    "height_band": name,
                    "floor_index": floor_index,
                    "z_cm": round(float(z_cm), 2),
                    "floor_height_m": self.route2_floor_height_m(),
                    "floor_count_estimate": floor_count,
                    "search_layer_role": role,
                }
            )
        return bands

    def route2_axis_value_for_region(self, observation: Dict[str, Any], region: str) -> float:
        axis_min = float(observation.get("safe_axis_min", observation.get("axis_value", 0.0)))
        axis_max = float(observation.get("safe_axis_max", observation.get("axis_value", 0.0)))
        region_l = str(region or "").lower()
        if "left" in region_l:
            ratio = 1.0 / 6.0
        elif "right" in region_l:
            ratio = 5.0 / 6.0
        else:
            ratio = 0.5
        return axis_min + (axis_max - axis_min) * ratio

    def route2_axis_value_for_region_intervals(
        self,
        intervals: List[Dict[str, Any]],
        region: str,
    ) -> Tuple[float, Dict[str, Any]]:
        valid = [
            item for item in intervals
            if isinstance(item, dict)
            and self._as_float_or_none(item.get("min")) is not None
            and self._as_float_or_none(item.get("max")) is not None
        ]
        if not valid:
            return 0.0, {"min": 0.0, "max": 0.0, "index": 0}
        axis_min = min(float(item.get("min")) for item in valid)
        axis_max = max(float(item.get("max")) for item in valid)
        region_l = str(region or "").lower()
        if "left" in region_l:
            ratio = 1.0 / 6.0
        elif "right" in region_l:
            ratio = 5.0 / 6.0
        else:
            ratio = 0.5
        raw = axis_min + (axis_max - axis_min) * ratio
        for item in valid:
            lo = float(item.get("min"))
            hi = float(item.get("max"))
            if lo <= raw <= hi:
                return raw, item
        best_item = max(valid, key=lambda item: float(item.get("max")) - float(item.get("min")))
        lo = float(best_item.get("min"))
        hi = float(best_item.get("max"))
        return min(max(raw, lo), hi), best_item

    def route2_band_for_region(self, bands: List[Dict[str, Any]], region: str) -> Dict[str, Any]:
        if not bands:
            return {"height_band": "floor_1", "floor_index": 1, "z_cm": self.route2_low_z_cm()}
        region_l = str(region or "").lower()
        if "low" in region_l:
            return bands[0]
        if "high" in region_l:
            return bands[-1]
        return bands[min(len(bands) - 1, len(bands) // 2)]

    def route2_interval_sample_counts(
        self,
        intervals: List[Dict[str, Any]],
        *,
        spacing: float,
        density: str,
    ) -> List[int]:
        limits = LLM_ROUTE2_RULE_DENSITY_LIMITS.get(
            str(density),
            LLM_ROUTE2_RULE_DENSITY_LIMITS["medium"],
        )
        min_count = int(limits.get("min", 4))
        max_count = int(limits.get("max", 12))
        lengths = [
            max(0.0, abs(float(interval.get("max", 0.0)) - float(interval.get("min", 0.0))))
            for interval in intervals
            if isinstance(interval, dict)
        ]
        if not lengths:
            return []
        total_length = sum(lengths)
        desired_total = int(math.ceil(total_length / max(1.0, float(spacing)))) + 1
        desired_total = max(min_count, min(max_count, desired_total))
        desired_total = max(desired_total, min(max_count, 2 * len(lengths)))
        if total_length <= 0.0:
            return [max(1, desired_total // max(1, len(lengths)))] * len(lengths)
        counts = [max(2, int(round(desired_total * length / total_length))) for length in lengths]
        while sum(counts) > desired_total:
            candidates = [idx for idx, value in enumerate(counts) if value > 2]
            if not candidates:
                break
            idx = max(candidates, key=lambda item: counts[item])
            counts[idx] -= 1
        while sum(counts) < desired_total:
            idx = max(range(len(counts)), key=lambda item: lengths[item])
            counts[idx] += 1
        return counts

    def route2_generate_facade_scan_points(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        state = self.route2_selected_state()
        house_id = str(state.get("target_house_id", "") or "")
        facade = str(state.get("facade", "") or "")
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        bbox = self.house_world_bbox_for_id(house_id)
        if not house_id or not facade or not observation or not bbox:
            return []
        spacing = self.route2_spacing_from_analysis(analysis)
        density_rule = self.route2_density_rule(house_id, facade, analysis)
        density = str(density_rule.get("density", "medium") or "medium")
        scan_geometry = self.route2_select_scan_geometry(house_id, facade, bbox, analysis)
        translation_span = str(scan_geometry.get("translation_span", "medium") or "medium")
        bands = self.route2_height_bands_from_analysis(analysis)
        standoff = float(scan_geometry.get("standoff_cm", self.route_scan_standoff_cm()))
        intervals = [
            item for item in scan_geometry.get("safe_intervals", [])
            if isinstance(item, dict) and self._as_float_or_none(item.get("min")) is not None and self._as_float_or_none(item.get("max")) is not None
        ]
        if not intervals:
            return []
        points: List[Dict[str, Any]] = []
        scan_index = 0
        interval_counts = self.route2_interval_sample_counts(intervals, spacing=spacing, density=density)
        planned_facade_sample_count = sum(interval_counts)
        for band in bands:
            for interval_position, interval in enumerate(intervals):
                axis_min = float(interval.get("min"))
                axis_max = float(interval.get("max"))
                count = interval_counts[interval_position] if interval_position < len(interval_counts) else 2
                for local_idx in range(count):
                    ratio = float(local_idx) / float(max(1, count - 1))
                    axis_value = axis_min + (axis_max - axis_min) * ratio
                    pose = self.route2_facade_pose_from_axis(bbox, facade, axis_value, standoff, float(band["z_cm"]))
                    points.append(
                        {
                            "scan_id": f"{house_id}_{facade}_{band['height_band']}_{scan_index:03d}",
                            "local_scan_index": scan_index,
                            "house_id": house_id,
                            "facade": facade,
                            "facade_id": self.route2_facade_id(house_id, facade),
                            "height_band": str(band["height_band"]),
                            "floor_index": int(band["floor_index"]),
                            "semantic_region": "full_facade",
                            "target_score": str(analysis.get("target_score", "medium") or "medium"),
                            "translation_span": translation_span,
                            "rule_density": density,
                            "density_rule": density_rule,
                            "planned_facade_sample_count": planned_facade_sample_count,
                            "x": pose["x"],
                            "y": pose["y"],
                            "z": pose["z"],
                            "yaw_deg": pose["yaw_deg"],
                            "standoff_cm": round(float(standoff), 2),
                            "preferred_standoff_cm": scan_geometry.get("preferred_standoff_cm"),
                            "scan_standoff_mode": scan_geometry.get("scan_standoff_mode"),
                            "lidar_max_range_cm": float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
                            "scan_spacing_cm": round(float(spacing), 2),
                            "view_type": "facade_floor_band_scan",
                            "capture_trigger": "arrive_align_hover_capture",
                            "safe_interval_index": interval.get("index", interval_position),
                            "safe_interval_count": len(intervals),
                            "safe_axis_min": round(axis_min, 2),
                            "safe_axis_max": round(axis_max, 2),
                            "status": "planned",
                        }
                    )
                    scan_index += 1
        cue_index = 0
        for cue in analysis.get("detected_cues", []) if isinstance(analysis.get("detected_cues"), list) else []:
            if not isinstance(cue, dict):
                continue
            if density == "sparse" and not self.route2_is_entry_cue(cue):
                continue
            region = str(cue.get("region", "") or "mid-center")
            axis_value, cue_interval = self.route2_axis_value_for_region_intervals(intervals, region)
            band = self.route2_band_for_region(bands, region)
            pose = self.route2_facade_pose_from_axis(bbox, facade, axis_value, standoff, float(band["z_cm"]))
            points.append(
                {
                    "scan_id": f"{house_id}_{facade}_{band['height_band']}_confirm_{cue_index:03d}",
                    "local_scan_index": scan_index,
                    "house_id": house_id,
                    "facade": facade,
                    "facade_id": self.route2_facade_id(house_id, facade),
                    "height_band": str(band["height_band"]),
                    "floor_index": int(band["floor_index"]),
                    "semantic_region": region,
                    "cue_type": str(cue.get("type", "cue") or "cue"),
                    "cue_confidence": cue.get("confidence"),
                    "target_score": str(analysis.get("target_score", "medium") or "medium"),
                    "translation_span": translation_span,
                    "rule_density": density,
                    "density_rule": density_rule,
                    "planned_facade_sample_count": planned_facade_sample_count,
                    "x": pose["x"],
                    "y": pose["y"],
                    "z": pose["z"],
                    "yaw_deg": pose["yaw_deg"],
                    "standoff_cm": round(float(standoff), 2),
                    "preferred_standoff_cm": scan_geometry.get("preferred_standoff_cm"),
                    "scan_standoff_mode": scan_geometry.get("scan_standoff_mode"),
                    "lidar_max_range_cm": float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
                    "scan_spacing_cm": round(float(spacing), 2),
                    "view_type": "semantic_candidate_confirm_scan",
                    "capture_trigger": "arrive_align_hover_capture",
                    "safe_interval_index": cue_interval.get("index", 0),
                    "safe_interval_count": len(intervals),
                    "safe_axis_min": round(float(cue_interval.get("min", axis_value)), 2),
                    "safe_axis_max": round(float(cue_interval.get("max", axis_value)), 2),
                    "status": "planned",
                }
            )
            cue_index += 1
            scan_index += 1
        return points

    def route2_scan_continuity_start_pose(self) -> Dict[str, float]:
        pose = self.current_route_pose() or {}
        x = self._as_float_or_none(pose.get("x"))
        y = self._as_float_or_none(pose.get("y"))
        z = self._as_float_or_none(pose.get("z"))
        if x is not None and y is not None:
            return {"x": float(x), "y": float(y), "z": float(z if z is not None else 0.0)}
        observation = self.route2_selected_state().get("observation_point", {})
        if isinstance(observation, dict):
            ox = self._as_float_or_none(observation.get("x"))
            oy = self._as_float_or_none(observation.get("y"))
            oz = self._as_float_or_none(observation.get("z"))
            if ox is not None and oy is not None:
                return {"x": float(ox), "y": float(oy), "z": float(oz if oz is not None else 0.0)}
        return {}

    def route2_scan_continuity_cost(self, current: Dict[str, Any], point: Dict[str, Any]) -> float:
        dx = float(point.get("x", 0.0)) - float(current.get("x", 0.0))
        dy = float(point.get("y", 0.0)) - float(current.get("y", 0.0))
        dz = float(point.get("z", 0.0)) - float(current.get("z", 0.0))
        return float(math.hypot(dx, dy) + 0.35 * abs(dz))

    def route2_order_scan_points_continuously(
        self,
        points: List[Dict[str, Any]],
        *,
        start_pose: Optional[Dict[str, Any]] = None,
        next_observation_pose: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        remaining = [dict(point) for point in points if isinstance(point, dict)]
        if len(remaining) <= 1:
            for idx, item in enumerate(remaining):
                item["continuous_order_index"] = idx
                item["continuous_sort_source"] = "single_or_empty"
                item["travel_delta_cm"] = 0.0
                if isinstance(next_observation_pose, dict) and next_observation_pose:
                    item["next_facade_hint"] = str(next_observation_pose.get("facade", "") or "")
                    item["distance_to_next_observation_cm"] = round(
                        math.hypot(
                            float(item.get("x", 0.0)) - float(next_observation_pose.get("x", 0.0)),
                            float(item.get("y", 0.0)) - float(next_observation_pose.get("y", 0.0)),
                        ),
                        2,
                    )
            return remaining
        if isinstance(next_observation_pose, dict) and next_observation_pose:
            next_x = self._as_float_or_none(next_observation_pose.get("x"))
            next_y = self._as_float_or_none(next_observation_pose.get("y"))
            if next_x is not None and next_y is not None:
                groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
                for item in remaining:
                    key = (int(round(float(item.get("x", 0.0)) * 10.0)), int(round(float(item.get("y", 0.0)) * 10.0)))
                    groups.setdefault(key, []).append(item)
                ordered_groups = sorted(
                    groups.values(),
                    key=lambda group: (
                        -math.hypot(
                            float(group[0].get("x", 0.0)) - float(next_x),
                            float(group[0].get("y", 0.0)) - float(next_y),
                        ),
                        min(int(item.get("local_scan_index", 0) or 0) for item in group),
                    ),
                )
                ordered = []
                for group in ordered_groups:
                    ordered.extend(
                        sorted(
                            group,
                            key=lambda item: (
                                int(item.get("floor_index", 0) or 0),
                                float(item.get("z", 0.0) or 0.0),
                                int(item.get("local_scan_index", 0) or 0),
                            ),
                        )
                    )
                current = dict(start_pose or self.route2_scan_continuity_start_pose() or {})
                if not current and ordered:
                    current = dict(ordered[0])
                previous_local_id = ""
                next_facade = str(next_observation_pose.get("facade", "") or "")
                for idx, item in enumerate(ordered):
                    travel_delta = self.route2_scan_continuity_cost(current, item) if current else 0.0
                    item["continuous_order_index"] = idx
                    item["previous_local_scan_id"] = previous_local_id
                    item["travel_delta_cm"] = round(float(travel_delta), 2)
                    item["continuous_sort_source"] = "end_near_next_facade"
                    item["sequence_goal"] = "end_near_next_facade"
                    item["next_facade_hint"] = next_facade
                    item["distance_to_next_observation_cm"] = round(
                        math.hypot(float(item.get("x", 0.0)) - float(next_x), float(item.get("y", 0.0)) - float(next_y)),
                        2,
                    )
                    previous_local_id = str(item.get("scan_id", "") or "")
                    current = item
                return ordered
        current = dict(start_pose or self.route2_scan_continuity_start_pose() or {})
        if not current:
            current = dict(remaining[0])
        ordered: List[Dict[str, Any]] = []
        previous_local_id = ""
        while remaining:
            best_idx = min(
                range(len(remaining)),
                key=lambda idx: (
                    self.route2_scan_continuity_cost(current, remaining[idx]),
                    int(remaining[idx].get("floor_index", 0) or 0),
                    int(remaining[idx].get("local_scan_index", idx) or idx),
                ),
            )
            item = remaining.pop(best_idx)
            travel_delta = self.route2_scan_continuity_cost(current, item)
            item["continuous_order_index"] = len(ordered)
            item["previous_local_scan_id"] = previous_local_id
            item["travel_delta_cm"] = round(float(travel_delta), 2)
            item["continuous_sort_source"] = "nearest_neighbor_xy_plus_z"
            ordered.append(item)
            previous_local_id = str(item.get("scan_id", "") or "")
            current = item
        return ordered

    def route2_scan_order_from_point(self, point: Dict[str, Any]) -> int:
        value = self._as_float_or_none(point.get("global_scan_order"))
        if value is not None:
            return int(value)
        scan_id = str(point.get("scan_id", "") or "")
        match = re.search(r"_scan_(\d+)", scan_id)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return 0
        return 0

    def route2_existing_facade_scan_points(self, output_dir: Path, *, exclude_facade: str = "") -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        root = output_dir / "facade_observations"
        if not root.exists():
            return points
        excluded = str(exclude_facade or "").strip().lower()
        for path in sorted(root.glob("*/facade_search_plan.json")):
            payload = flight.read_json_object(path)
            if not isinstance(payload, dict):
                continue
            facade = str(payload.get("facade", "") or "").strip().lower()
            if excluded and facade == excluded:
                continue
            for point in payload.get("scan_points", []) if isinstance(payload.get("scan_points"), list) else []:
                if isinstance(point, dict):
                    points.append(dict(point))
        return points

    def route2_assign_global_scan_ids(
        self,
        output_dir: Path,
        house_id: str,
        facade: str,
        points: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        existing = self.route2_existing_facade_scan_points(output_dir, exclude_facade=facade)
        next_order = max([self.route2_scan_order_from_point(point) for point in existing] + [0]) + 1
        assigned: List[Dict[str, Any]] = []
        for offset, point in enumerate(points):
            item = dict(point)
            order = next_order + offset
            band = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("height_band", "band") or "band")).strip("_")
            suffix = f"{facade}_{band}"
            if str(item.get("view_type", "") or "") == "semantic_candidate_confirm_scan":
                suffix = f"{suffix}_confirm"
            item["global_scan_order"] = int(order)
            item["scan_id"] = f"{house_id}_scan_{order:04d}_{suffix}"
            assigned.append(item)
        previous_scan_id = ""
        for idx, item in enumerate(assigned):
            item["continuous_order_index"] = int(idx)
            item["previous_scan_id"] = previous_scan_id
            previous_scan_id = str(item.get("scan_id", "") or "")
        return assigned

    def route2_write_merged_scan_points(self, output_dir: Path, house_id: str) -> List[Dict[str, Any]]:
        points = self.route2_existing_facade_scan_points(output_dir)
        points.sort(key=lambda item: (self.route2_scan_order_from_point(item), str(item.get("scan_id", "") or "")))
        facade_counts: Dict[str, int] = {}
        for point in points:
            facade = str(point.get("facade", "") or "")
            facade_counts[facade] = facade_counts.get(facade, 0) + 1
        self.write_json_artifact(
            output_dir / "scan_points.json",
            {
                "schema": "facade_v2_global_scan_points",
                "target_house_id": house_id,
                "scan_points": points,
                "facade_counts": facade_counts,
                "total_scan_count": len(points),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return points

    def on_route2_plan_facade_scan(self) -> None:
        state = self.route2_selected_state()
        if not state.get("observation_point"):
            self.llm_route2_status_var.set("LLM Route V2: plan nearest facade first.")
            return
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        if not analysis:
            analysis = self.route2_fallback_facade_analysis("Plan Facade Scan used fallback before VLM analysis.")
        points = self.route2_generate_facade_scan_points(analysis)
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            self.llm_route2_status_var.set("LLM Route V2: missing facade output directory.")
            return
        points = self.route2_order_scan_points_continuously(points)
        points = self.route2_assign_global_scan_ids(output_dir, house_id, facade, points)
        validation = self.scan_point_validation_report(house_id, points)
        search_plan = {
            "schema": "facade_v2_scan_plan",
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "observation_point": state.get("observation_point", {}),
            "facade_analysis": analysis,
            "scan_points": points,
            "scan_point_validation_report": validation,
            "route_blocked_by_safety": not bool(validation.get("valid", False)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(facade_dir / "facade_search_plan.json", search_plan)
        merged_points = self.route2_write_merged_scan_points(output_dir, house_id)
        self.route2_update_state(facade_analysis=analysis, facade_search_plan=search_plan, facade_scan_points=points, validation_report=validation)
        self.route2_write_state_artifact()
        self.refresh_route2_preview()
        self.refresh_llm_route2_map()
        suffix = " | blocked by safety validation" if not bool(validation.get("valid", False)) else ""
        self.llm_route2_status_var.set(f"LLM Route V2: planned facade scan {facade} points={len(points)} global_total={len(merged_points)}{suffix}")

    def on_route2_capture_facade_scan(self) -> None:
        session = self.active_session()
        if session is None:
            return
        state = self.route2_selected_state()
        points = state.get("facade_scan_points", []) if isinstance(state.get("facade_scan_points"), list) else []
        if not points:
            self.on_route2_plan_facade_scan()
            state = self.route2_selected_state()
            points = state.get("facade_scan_points", []) if isinstance(state.get("facade_scan_points"), list) else []
        if not points:
            self.llm_route2_status_var.set("LLM Route V2: no facade scan points.")
            return
        validation = self.scan_point_validation_report(str(state.get("target_house_id", "") or ""), points)
        if not bool(validation.get("valid", False)):
            self.route2_update_state(validation_report=validation)
            self.refresh_route2_preview()
            self.llm_route2_status_var.set("LLM Route V2: scan blocked by safety validation.")
            return
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route2_status_var.set("LLM Route V2: wait for current worker.")
            return
        self.sync_capture_options_to_session(session)
        self.route_stop_event.clear()
        self.route_thread = threading.Thread(target=lambda: self.route2_capture_facade_scan_worker(session), daemon=True)
        self.route_thread.start()

    def route2_next_frame_index(self, output_dir: Path) -> int:
        frame_indices: List[int] = []
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            for path in frames_dir.glob("frame_*"):
                if not path.is_dir():
                    continue
                match = re.match(r"frame_(\d+)", path.name)
                if match:
                    try:
                        frame_indices.append(int(match.group(1)))
                    except Exception:
                        pass
        for log_name in ("lidar_capture_log.jsonl", "scan_execution_log.jsonl"):
            for row in self.read_jsonl_artifact(output_dir / log_name):
                value = self._as_float_or_none(row.get("frame_index"))
                if value is not None:
                    frame_indices.append(int(value))
                for item in row.get("frame_indices", []) if isinstance(row.get("frame_indices"), list) else []:
                    item_value = self._as_float_or_none(item)
                    if item_value is not None:
                        frame_indices.append(int(item_value))
        return max(frame_indices + [0]) + 1

    def route2_write_lidar_summary(self, output_dir: Path, *, running: bool) -> None:
        rows = self.read_jsonl_artifact(output_dir / "lidar_capture_log.jsonl")
        summary = {
            "capture_kind": "facade_v2_lidar",
            "task_title": self.llm_task_text_var.get().strip() or "facade_v2_search",
            "stream_dir": str(output_dir),
            "frames_dir": str(output_dir / "frames"),
            "reconstruction_dir": str(output_dir / "reconstruction"),
            "running": bool(running),
            "frame_count": len(rows),
            "lidar_depth_min_cm": float(getattr(self.args, "lidar_depth_min_cm", flight.DEFAULT_LIDAR_DEPTH_MIN_CM)),
            "lidar_depth_max_cm": float(getattr(self.args, "lidar_depth_max_cm", flight.DEFAULT_LIDAR_DEPTH_MAX_CM)),
            "lidar_depth_projection": str(getattr(self.args, "lidar_depth_projection", flight.DEFAULT_LIDAR_DEPTH_PROJECTION)),
            "lidar_capture_processing": self.lidar_capture_processing_mode(),
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        (output_dir / "stream_capture_lidar.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        trajectory_payload = dict(summary)
        trajectory_payload["trajectory"] = rows
        (output_dir / "trajectory.json").write_text(json.dumps(trajectory_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def route2_capture_facade_scan_worker(self, session: flight.DroneFlightSession) -> None:
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: missing facade output directory."))
            return
        state = self.route2_selected_state()
        points = [dict(point) for point in state.get("facade_scan_points", []) if isinstance(point, dict)]
        total = len(points)
        interval_s = self.route_capture_interval_s()
        trajectory = state.get("facade_capture_rows", []) if isinstance(state.get("facade_capture_rows"), list) else []
        self.root.after(0, lambda: self.llm_route2_status_var.set(f"LLM Route V2: capture facade {facade} points={total}"))
        for idx, point in enumerate(points, start=1):
            if self.route_stop_event.is_set():
                self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: scan capture stopped."))
                return
            scan_id = str(point.get("scan_id", "") or f"{house_id}_{facade}_{idx:03d}")
            planned_pose = {
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "z": float(point.get("z", 600.0)),
                "yaw": float(point.get("yaw_deg", point.get("yaw", 0.0))),
            }
            self.root.after(0, lambda i=idx, t=total, sid=scan_id: self.llm_route2_status_var.set(f"LLM Route V2: move/capture {i}/{t} {sid}"))
            set_result = self.safe("Route V2 scan set pose", lambda p=planned_pose: session.set_pose(p))
            if isinstance(set_result, dict):
                self.root.after(0, lambda r=set_result: self.apply_state(r))
            if self.route_stop_event.wait(0.3):
                return
            capture_results: List[Dict[str, Any]] = []
            for capture_index in range(self.route_capture_count()):
                if self.route_stop_event.is_set():
                    break
                frame_index = self.route2_next_frame_index(output_dir)
                action_detail = dict(self.build_stream_action_detail())
                action_detail.update(
                    {
                        "source": "llm_route_v2_facade_scan",
                        "scan_id": scan_id,
                        "global_scan_order": point.get("global_scan_order"),
                        "house_id": house_id,
                        "facade": facade,
                        "facade_id": self.route2_facade_id(house_id, facade),
                        "height_band": str(point.get("height_band", "") or ""),
                        "floor_index": point.get("floor_index"),
                        "semantic_region": str(point.get("semantic_region", "") or ""),
                        "capture_index": capture_index + 1,
                        "capture_count": self.route_capture_count(),
                        "planned_pose": planned_pose,
                    }
                )
                result = self.safe(
                    "Route V2 scan capture",
                    lambda frame=frame_index, action=action_detail: session.capture_lidar_stream_frame(
                        output_dir,
                        frame,
                        action_detail=action,
                    ),
                )
                if not isinstance(result, dict):
                    continue
                capture_results.append(result)
                row = {
                    "frame_index": int(result.get("frame_index", frame_index)),
                    "capture_time": result.get("capture_time", ""),
                    "scan_id": scan_id,
                    "global_scan_order": point.get("global_scan_order"),
                    "house_id": house_id,
                    "facade": facade,
                    "facade_id": self.route2_facade_id(house_id, facade),
                    "height_band": str(point.get("height_band", "") or ""),
                    "floor_index": point.get("floor_index"),
                    "semantic_region": str(point.get("semantic_region", "") or ""),
                    "planned_pose": planned_pose,
                    "pose": result.get("pose", {}),
                    "commanded_pose": result.get("commanded_pose", {}),
                    "actual_pose": result.get("actual_pose", {}),
                    "pose_error": result.get("pose_error", {}),
                    "capture_dir": result.get("capture_dir", ""),
                    "rgb_path": result.get("rgb_path", ""),
                    "point_cloud_world_standard_m_npy_path": result.get("point_cloud_world_standard_m_npy_path", ""),
                    "point_cloud_world_standard_m_ply_path": result.get("point_cloud_world_standard_m_ply_path", ""),
                    "point_cloud_preview_path": result.get("point_cloud_preview_path", ""),
                    "point_count": int(result.get("point_count", 0) or 0),
                    "action_detail": action_detail,
                }
                trajectory.append(row)
                self.append_jsonl(output_dir / "lidar_capture_log.jsonl", row)
                self.append_jsonl(facade_dir / "lidar_capture_log.jsonl", row)
                self.route2_write_lidar_summary(output_dir, running=True)
                if capture_index + 1 < self.route_capture_count() and self.route_stop_event.wait(interval_s):
                    break
            capture_status = "ok" if capture_results else "failed"
            execution_entry = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "house_id": house_id,
                "facade": facade,
                "scan_id": scan_id,
                "global_scan_order": point.get("global_scan_order"),
                "height_band": str(point.get("height_band", "") or ""),
                "floor_index": point.get("floor_index"),
                "planned_pose": planned_pose,
                "capture_status": capture_status,
                "capture_count": len(capture_results),
                "frame_indices": [int(item.get("frame_index", 0) or 0) for item in capture_results],
                "capture_dirs": [str(item.get("capture_dir", "") or "") for item in capture_results],
                "point_count": sum(int(item.get("point_count", 0) or 0) for item in capture_results),
            }
            self.append_jsonl(output_dir / "scan_execution_log.jsonl", execution_entry)
            self.append_jsonl(facade_dir / "scan_execution_log.jsonl", execution_entry)
            for state_point in self.llm_route2_state.get("facade_scan_points", []) if isinstance(self.llm_route2_state.get("facade_scan_points"), list) else []:
                if isinstance(state_point, dict) and str(state_point.get("scan_id", "") or "") == scan_id:
                    state_point["status"] = "captured" if capture_status == "ok" else "capture_failed"
            self.route2_update_state(facade_capture_rows=trajectory)
            self.route2_write_state_artifact()
            self.root.after(0, self.refresh_route2_preview)
            self.root.after(0, self.refresh_llm_route2_map)
            if idx < total and self.route_stop_event.wait(interval_s):
                self.root.after(0, lambda: self.llm_route2_status_var.set("LLM Route V2: scan capture stopped."))
                return
        self.route2_write_lidar_summary(output_dir, running=False)
        self.root.after(0, lambda: self.llm_route2_status_var.set(f"LLM Route V2: facade scan capture complete -> {output_dir}"))

    def route2_validate_facade(self) -> Dict[str, Any]:
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            raise RuntimeError("missing facade output directory")
        state = self.route2_selected_state()
        points = [point for point in state.get("facade_scan_points", []) if isinstance(point, dict)]
        geometry = self.scan_point_validation_report(house_id, points)
        execution_rows = self.read_jsonl_artifact(facade_dir / "scan_execution_log.jsonl")
        capture_rows = self.read_jsonl_artifact(facade_dir / "lidar_capture_log.jsonl")
        successful_scan_ids = {str(row.get("scan_id", "") or "") for row in execution_rows if str(row.get("capture_status", "") or "") == "ok"}
        planned_bands = {str(point.get("height_band", "") or "") for point in points if str(point.get("height_band", "") or "")}
        captured_bands = {
            str(row.get("height_band", "") or "")
            for row in execution_rows
            if str(row.get("capture_status", "") or "") == "ok" and str(row.get("height_band", "") or "")
        }
        merged_point_count = sum(int(row.get("point_count", 0) or 0) for row in capture_rows)
        coverage_report = {
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "planned_scan_count": len(points),
            "captured_scan_count": len(successful_scan_ids),
            "capture_success_rate": round(float(len(successful_scan_ids)) / float(max(1, len(points))), 4),
            "planned_height_bands": sorted(planned_bands),
            "captured_height_bands": sorted(captured_bands),
            "missing_height_bands": sorted(planned_bands - captured_bands),
            "merged_point_count": int(merged_point_count),
            "geometry_valid": bool(geometry.get("valid", False)),
            "coverage_mode": "facade_v2_count_and_height_band",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        rescan_points: List[Dict[str, Any]] = []
        missing_bands = planned_bands - captured_bands
        if not bool(geometry.get("valid", False)) or missing_bands or not successful_scan_ids:
            for band in sorted(missing_bands) or sorted(planned_bands):
                candidates = [point for point in points if str(point.get("height_band", "") or "") == band]
                if candidates:
                    item = dict(candidates[len(candidates) // 2])
                    item["scan_id"] = f"{house_id}_{facade}_{band}_rescan_000"
                    item["status"] = "planned"
                    item["view_type"] = "facade_v2_rescan"
                    item["rescan_reason"] = "missing height band capture or geometry check failed"
                    rescan_points.append(item)
        rescan_plan = {
            "house_id": house_id,
            "facade": facade,
            "rescan_points": rescan_points,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        checks = [
            {"name": "scan_points_exist", "passed": len(points) > 0, "detail": f"{len(points)} points"},
            {"name": "geometry_valid", "passed": bool(geometry.get("valid", False)), "detail": geometry},
            {"name": "all_height_bands_captured", "passed": not (planned_bands - captured_bands), "detail": coverage_report["missing_height_bands"]},
            {"name": "lidar_rows_exist", "passed": len(capture_rows) > 0, "detail": f"{len(capture_rows)} rows"},
            {"name": "point_count_positive", "passed": merged_point_count > 0, "detail": f"point_count={merged_point_count}"},
        ]
        validation = {
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "overall_passed": all(bool(check.get("passed", False)) for check in checks),
            "checks": checks,
            "coverage_report": coverage_report,
            "rescan_plan": rescan_plan,
            "validation_report_path": str(facade_dir / "facade_validation_report.json"),
            "validated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(facade_dir / "facade_coverage_report.json", coverage_report)
        self.write_json_artifact(facade_dir / "facade_rescan_plan.json", rescan_plan)
        self.write_json_artifact(facade_dir / "facade_validation_report.json", validation)
        self.route2_update_state(validation_report=validation, facade_coverage_report=coverage_report, facade_rescan_plan=rescan_plan)
        self.route2_write_state_artifact()
        return validation

    def on_route2_validate_facade(self) -> None:
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route2_status_var.set("LLM Route V2: wait for current worker.")
            return
        try:
            validation = self.route2_validate_facade()
            self.refresh_route2_preview()
            self.refresh_llm_route2_map()
            self.llm_route2_status_var.set(
                f"LLM Route V2: validation {'PASS' if validation.get('overall_passed') else 'CHECK'} -> {validation.get('validation_report_path', '')}"
            )
        except Exception as exc:
            LOGGER.warning("Route V2 facade validation failed: %s", exc)
            self.llm_route2_status_var.set(f"LLM Route V2: validation failed: {exc}")

    def make_route3_autosearch_output_dir(self, target_house_id: str) -> Path:
        root = self.resolve_project_path("route_capture_lidar")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_house = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_house_id or "unknown")).strip("_") or "unknown"
        base_name = f"house_{safe_house}_autosearch_v3_{timestamp}"
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

    def route3_float_param(self, var: tk.StringVar, default: float, *, min_value: float, max_value: float) -> float:
        try:
            value = float(var.get().strip())
        except Exception:
            value = float(default)
        return max(float(min_value), min(float(max_value), float(value)))

    def route3_nav_config(self) -> Dict[str, float]:
        return {
            "move_tick_ms": self.route3_float_param(self.llm_route3_move_tick_ms_var, 150.0, min_value=50.0, max_value=2000.0),
            "nav_step_cm": self.route3_float_param(self.llm_route3_nav_step_cm_var, 20.0, min_value=5.0, max_value=200.0),
            "reach_tol_cm": self.route3_float_param(self.llm_route3_reach_tol_cm_var, 60.0, min_value=5.0, max_value=500.0),
            "z_tol_cm": self.route3_float_param(self.llm_route3_z_tol_cm_var, 40.0, min_value=5.0, max_value=500.0),
            "yaw_tol_deg": self.route3_float_param(self.llm_route3_yaw_tol_deg_var, 10.0, min_value=1.0, max_value=90.0),
            "max_stage_s": self.route3_float_param(self.llm_route3_max_stage_s_var, 90.0, min_value=5.0, max_value=900.0),
        }

    def route3_state_output_dir(self) -> Optional[Path]:
        state = self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {}
        raw = str(state.get("output_dir", "") or "")
        if not raw:
            return None
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        (path / "reconstruction").mkdir(parents=True, exist_ok=True)
        (path / "facade_observations").mkdir(parents=True, exist_ok=True)
        return path

    def route3_update_state(self, **updates: Any) -> Dict[str, Any]:
        state = dict(self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {})
        state.update(updates)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_route3_state = state
        return state

    def route3_write_state_artifact(self) -> None:
        output_dir = self.route3_state_output_dir()
        if output_dir is None:
            return
        self.write_json_artifact(output_dir / "autonomy_state.json", self.llm_route3_state)

    def route3_default_facade_priority(self, task_text: str = "") -> List[str]:
        text = str(task_text or "").lower()
        aliases = {
            "south": ("south", "\u5357"),
            "east": ("east", "\u4e1c"),
            "north": ("north", "\u5317"),
            "west": ("west", "\u897f", "front", "\u6b63\u9762", "\u5165\u53e3\u9762"),
        }
        priority: List[str] = []
        for facade, tokens in aliases.items():
            if any(token in text for token in tokens) and facade not in priority:
                priority.append(facade)
        for facade in ("west", "south", "east", "north"):
            if facade not in priority:
                priority.append(facade)
        return priority

    def route3_house_registry_for_task_plan(self) -> Dict[str, Any]:
        provider = getattr(self, "house_registry_for_llm_plan", None)
        if callable(provider):
            return provider()
        records_provider = getattr(self, "house_records_for_route_planning", None)
        records = records_provider() if callable(records_provider) else []
        available: List[Dict[str, Any]] = []
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            house_id = str(record.get("id", record.get("house_id", "")) or "").strip()
            if not house_id:
                continue
            numeric = str(int(house_id)) if house_id.isdigit() else house_id
            name = str(record.get("name", f"House_{numeric}") or f"House_{numeric}")
            available.append(
                {
                    "house_id": house_id,
                    "house_name": name,
                    "aliases": [house_id, numeric, name, name.lower(), f"house {numeric}", f"house_{numeric}", f"House_{numeric}"],
                    "status": str(record.get("status", "") or ""),
                    "center_x": record.get("x", record.get("center_x")),
                    "center_y": record.get("y", record.get("center_y")),
                    "radius_cm": record.get("radius_cm"),
                }
            )
        selected = self.selected_route_target_house_id() if hasattr(self, "selected_route_target_house_id") else ""
        return {"current_target_id": selected, "selected_target_id": selected, "available_houses": available}

    def route3_explicit_house_sequence_from_text(self, task_text: str, registry: Dict[str, Any]) -> List[str]:
        alias_map = self.build_task_alias_map(registry)
        ordered: List[str] = []
        text = str(task_text or "")
        for match in re.finditer(r"(?:house|House|house_|\u623f\u5c4b|\u623f\u5b50)\s*0*(\d{1,3})", text):
            house_id = self.resolve_plan_house_id(match.group(1), alias_map)
            if house_id and house_id not in ordered:
                ordered.append(house_id)
        return ordered

    def route3_task_target_entries_from_ids(self, house_ids: List[str], *, source: str = "route3_task") -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for house_id in house_ids:
            raw_house_id = str(house_id or "").strip()
            if not raw_house_id:
                continue
            hid = raw_house_id.zfill(3) if raw_house_id.isdigit() else raw_house_id
            if any(str(item.get("house_id", "")) == hid for item in entries):
                continue
            entries.append(
                {
                    "order": len(entries) + 1,
                    "house_id": hid,
                    "house_alias": self.house_display_by_id.get(hid, hid) if hasattr(self, "house_display_by_id") else hid,
                    "goal": "search_entry",
                    "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
                    "status": "pending",
                    "source": source,
                }
            )
        return entries

    def route3_task_plan_status_text(self, plan: Dict[str, Any]) -> str:
        target_sequence = plan.get("target_sequence", []) if isinstance(plan.get("target_sequence"), list) else []
        sequence_text = "->".join(str(item) for item in target_sequence[:6]) if target_sequence else str(plan.get("target_house_id", "-") or "-")
        if len(target_sequence) > 6:
            sequence_text += "..."
        return (
            f"Task Plan: targets={sequence_text} current={plan.get('target_house_id', '-')} "
            f"start={plan.get('preferred_start_facade', '-')} order={','.join(plan.get('facade_priority', [])[:4])}"
        )

    def route3_target_sequence_status_text(self, plan: Dict[str, Any]) -> str:
        targets = plan.get("ordered_targets", []) if isinstance(plan.get("ordered_targets"), list) else []
        if not targets:
            sequence = plan.get("target_sequence", []) if isinstance(plan.get("target_sequence"), list) else []
            targets = self.route3_task_target_entries_from_ids([str(item) for item in sequence], source="target_sequence")
        if not targets:
            return "Targets: n/a"
        chunks: List[str] = []
        active = str(plan.get("target_house_id", "") or "")
        for item in targets:
            if not isinstance(item, dict):
                continue
            order = int(item.get("order", len(chunks) + 1) or (len(chunks) + 1))
            house_id = str(item.get("house_id", "") or "").strip()
            status = str(item.get("status", "pending") or "pending")
            marker = "*" if house_id == active else ""
            chunks.append(f"{order}.{house_id}{marker}:{status}")
        return "Targets: " + "  ".join(chunks)

    def route3_apply_task_plan_state(self, plan: Dict[str, Any]) -> None:
        self.route3_update_state(
            task_plan=plan,
            target_house_id=str(plan.get("target_house_id", "") or ""),
            task_subtasks=plan.get("subtasks", []),
            preferred_start_facade=plan.get("preferred_start_facade", ""),
            facade_priority=plan.get("facade_priority", []),
            target_sequence=plan.get("target_sequence", []),
            ordered_targets=plan.get("ordered_targets", []),
            active_target_index=int(plan.get("active_target_index", 0) or 0),
        )

    def route3_sync_selected_target_from_plan(self, plan: Dict[str, Any]) -> str:
        target_house_id = str(plan.get("target_house_id", "") or "").strip()
        if not target_house_id:
            return ""
        try:
            self.root.after(0, lambda hid=target_house_id: self.set_selected_route_target_house(hid))
        except Exception:
            try:
                self.set_selected_route_target_house(target_house_id)
            except Exception:
                pass
        return target_house_id

    def route3_refresh_task_plan_labels(self, plan: Dict[str, Any]) -> None:
        try:
            self.llm_route3_task_status_var.set(self.route3_task_plan_status_text(plan))
            self.llm_route3_target_sequence_var.set(self.route3_target_sequence_status_text(plan))
        except Exception:
            pass

    def route3_normalize_task_plan(self, parsed: Dict[str, Any], target_house_id: str, task_text: str) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}
        registry = self.route3_house_registry_for_task_plan()
        alias_map = self.build_task_alias_map(registry)
        explicit_sequence = self.route3_explicit_house_sequence_from_text(task_text, registry)
        raw_selected_target = str(target_house_id or "").strip()
        selected_target = self.resolve_plan_house_id(raw_selected_target, alias_map)
        if not selected_target and raw_selected_target:
            selected_target = raw_selected_target.zfill(3) if raw_selected_target.isdigit() else raw_selected_target
        parsed_target = self.resolve_plan_house_id(parsed.get("target_house_id", ""), alias_map)
        parsed_ordered_targets: List[Dict[str, Any]] = []
        if isinstance(parsed.get("ordered_targets"), list):
            parsed_multi = self.normalize_llm_task_plan(parsed, task_text, registry)
            parsed_ordered_targets = [
                dict(item)
                for item in (parsed_multi.get("ordered_targets", []) if isinstance(parsed_multi.get("ordered_targets"), list) else [])
                if isinstance(item, dict)
            ]
        if explicit_sequence:
            ordered_targets = self.route3_task_target_entries_from_ids(explicit_sequence, source="explicit_task_text")
            planner_target_source = "explicit_task_text"
        elif parsed_ordered_targets:
            ordered_targets = parsed_ordered_targets
            planner_target_source = "llm_ordered_targets"
        elif parsed_target:
            ordered_targets = self.route3_task_target_entries_from_ids([parsed_target], source="llm_target_house_id")
            planner_target_source = "llm_target_house_id"
        elif selected_target:
            ordered_targets = self.route3_task_target_entries_from_ids([selected_target], source="selected_target_house")
            planner_target_source = "selected_target_house"
        else:
            local_plan = self.local_task_plan_from_text(task_text, registry)
            ordered_targets = [
                dict(item)
                for item in (local_plan.get("ordered_targets", []) if isinstance(local_plan.get("ordered_targets"), list) else [])
                if isinstance(item, dict)
            ]
            planner_target_source = "local_fallback"
        target_sequence = [
            str(item.get("house_id", "") or "").strip()
            for item in ordered_targets
            if isinstance(item, dict) and str(item.get("house_id", "") or "").strip()
        ]
        active_target_id = target_sequence[0] if target_sequence else (parsed_target or selected_target)
        normalized_ordered_targets: List[Dict[str, Any]] = []
        for index, item in enumerate(ordered_targets, start=1):
            if not isinstance(item, dict):
                continue
            house_id = str(item.get("house_id", "") or "").strip()
            if not house_id:
                continue
            normalized = dict(item)
            normalized["order"] = index
            if house_id == active_target_id and str(normalized.get("status", "pending") or "pending") == "pending":
                normalized["status"] = "in_progress"
            normalized_ordered_targets.append(normalized)
        ordered_targets = normalized_ordered_targets
        facade_priority: List[str] = []
        raw_priority = parsed.get("facade_priority", parsed.get("preferred_facade_order", []))
        if isinstance(raw_priority, list):
            for item in raw_priority:
                facade = str(item or "").strip().lower()
                if facade in {"south", "east", "north", "west"} and facade not in facade_priority:
                    facade_priority.append(facade)
        preferred = str(parsed.get("preferred_start_facade", "") or "").strip().lower()
        if preferred in {"south", "east", "north", "west"} and preferred not in facade_priority:
            facade_priority.insert(0, preferred)
        for facade in self.route3_default_facade_priority(task_text):
            if facade not in facade_priority:
                facade_priority.append(facade)
        if preferred not in {"south", "east", "north", "west"}:
            preferred = facade_priority[0] if facade_priority else "west"
        raw_subtasks = parsed.get("subtasks", []) if isinstance(parsed.get("subtasks"), list) else []
        subtasks: List[Dict[str, Any]] = []
        for item in raw_subtasks:
            if not isinstance(item, dict):
                continue
            facade = str(item.get("facade", "") or "").strip().lower()
            if facade not in {"south", "east", "north", "west"}:
                continue
            subtasks.append(
                {
                    "order": len(subtasks) + 1,
                    "facade": facade,
                    "goal": str(item.get("goal", "observe_analyze_scan_validate") or "observe_analyze_scan_validate"),
                    "status": str(item.get("status", "pending") or "pending"),
                }
            )
        if not subtasks:
            subtasks = [
                {"order": index, "facade": facade, "goal": "observe_analyze_scan_validate", "status": "pending"}
                for index, facade in enumerate(facade_priority, start=1)
            ]
        return {
            "schema": "route3_task_plan_v1",
            "target_house_id": str(active_target_id or ""),
            "active_target_index": 0,
            "target_sequence": target_sequence,
            "ordered_targets": ordered_targets,
            "target_source": planner_target_source,
            "selected_target_house_id_before_analysis": str(target_house_id or ""),
            "major_task": str(parsed.get("major_task", task_text) or task_text or "Search selected house entrance."),
            "task_text": str(task_text or ""),
            "subtasks": subtasks,
            "preferred_start_facade": preferred,
            "facade_priority": facade_priority,
            "completion_criteria": str(
                parsed.get("completion_criteria", "")
                or "All reachable facades have RGB/VLM analysis, scan captures, and validation."
            ),
            "reason": str(parsed.get("reason", "") or "Normalized route3 task plan."),
            "planner_source": str(parsed.get("planner_source", "llm_route3_task_analysis") or "llm_route3_task_analysis"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def route3_analyze_task_plan(self, target_house_id: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
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
                }
                response_payload = self.call_configured_llm_text(
                    system_prompt=(
                        "You are the high-level task planner for an autonomous UAV house search. "
                        "Split the user's task into ordered target houses, facade subtasks, a preferred start facade, "
                        "and a facade priority order. The task text has priority when it explicitly names houses. "
                        "Return strict compact JSON only."
                    ),
                    user_prompt=(
                        "Plan the house search task. If the task text explicitly names one or more houses, use that "
                        "as the target sequence and only use the selected target house when the text is ambiguous. "
                        "Do not output low-level movement commands.\n"
                        f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}\n"
                        f"Expected JSON:\n{json.dumps(LLM_ROUTE3_TASK_PLAN_SCHEMA, indent=2, ensure_ascii=False)}"
                    ),
                    max_output_tokens=700,
                    json_schema=LLM_ROUTE3_TASK_PLAN_SCHEMA,
                )
                parsed = extract_json_object(str(response_payload.get("raw_text", "") or ""))
            except Exception as exc:
                LOGGER.warning("Route V3 task analysis fallback: %s", exc)
                response_payload = {"error": str(exc), "fallback_used": True}
        if not parsed:
            explicit_targets = self.route3_explicit_house_sequence_from_text(task_text, registry)
            fallback_target = explicit_targets[0] if explicit_targets else str(target_house_id or "")
            local_targets = explicit_targets or ([fallback_target] if fallback_target else [])
            local_ordered_targets = self.route3_task_target_entries_from_ids(local_targets, source="local_fallback_task_text" if explicit_targets else "selected_target_house")
            parsed = {
                "target_house_id": str(fallback_target or ""),
                "ordered_targets": local_ordered_targets,
                "major_task": task_text or "Search selected house entrance.",
                "facade_priority": self.route3_default_facade_priority(task_text),
                "preferred_start_facade": "",
                "completion_criteria": "All reachable facades have RGB/VLM analysis, scan captures, and validation.",
                "reason": "Local fallback task analysis.",
                "planner_source": "local_fallback_no_or_failed_api",
            }
        plan = self.route3_normalize_task_plan(parsed, target_house_id, task_text)
        self.route3_apply_task_plan_state(plan)
        self.route3_sync_selected_target_from_plan(plan)
        if output_dir is not None:
            self.write_json_artifact(output_dir / "route3_task_plan.json", {"plan": plan, "llm_response": response_payload})
            self.route3_log_event(output_dir, "task_plan_analysis", {"task_plan": plan, "llm_response": response_payload})
            self.route3_write_state_artifact()
        try:
            self.root.after(0, lambda p=plan: self.route3_refresh_task_plan_labels(p))
        except Exception:
            pass
        return plan

    def route3_task_plan_valid(self, target_house_id: str) -> bool:
        state = self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {}
        plan = state.get("task_plan", {}) if isinstance(state.get("task_plan"), dict) else {}
        return bool(plan and str(plan.get("target_house_id", "") or "") == str(target_house_id or ""))

    def route3_ensure_task_plan(self, target_house_id: str, output_dir: Path) -> Dict[str, Any]:
        if self.route3_task_plan_valid(target_house_id):
            plan = self.llm_route3_state.get("task_plan", {})
            return plan if isinstance(plan, dict) else {}
        return self.route3_analyze_task_plan(target_house_id, output_dir=output_dir)

    def route3_log_event(self, output_dir: Optional[Path], event_type: str, payload: Dict[str, Any]) -> None:
        if output_dir is None:
            return
        row = {
            "event_type": str(event_type),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            **(payload if isinstance(payload, dict) else {}),
        }
        self.append_jsonl(output_dir / "autonomy_events.jsonl", row)

    def route3_set_stage(
        self,
        stage: str,
        *,
        output_dir: Optional[Path] = None,
        facade: str = "",
        message: str = "",
        target: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        update: Dict[str, Any] = {"stage": stage}
        if facade:
            update["current_facade"] = facade
        if target is not None:
            update["current_target_pose"] = target
        if error is not None:
            update["last_error"] = error
        self.route3_update_state(**update)
        self.route3_write_state_artifact()
        self.route3_log_event(output_dir or self.route3_state_output_dir(), "stage", {"stage": stage, "facade": facade, "message": message})
        self.root.after(0, lambda s=stage: self.llm_route3_stage_var.set(f"Stage: {s}"))
        if facade:
            self.root.after(0, lambda f=facade: self.llm_route3_active_var.set(f"Active: facade={f}"))
        if target:
            self.root.after(
                0,
                lambda t=target: self.llm_route3_target_var.set(
                    f"Target: ({float(t.get('x', 0.0)):.1f},{float(t.get('y', 0.0)):.1f},{float(t.get('z', 0.0)):.1f}) yaw={float(t.get('yaw', t.get('yaw_deg', 0.0))):.1f}"
                ),
            )
        if error:
            self.root.after(0, lambda e=error: self.llm_route3_error_var.set(f"Error: {e.get('reason', e.get('status', 'CHECK'))}"))
        if message:
            self.root.after(0, lambda m=message: self.llm_route3_status_var.set(f"LLM Route V3: {m}"))
        self.root.after(0, self.refresh_route3_support_views)

    def route3_pose_from_payload(self, payload: Dict[str, Any]) -> Dict[str, float]:
        pose = payload.get("pose", payload) if isinstance(payload, dict) else {}
        if not isinstance(pose, dict):
            return {}
        x = self._as_float_or_none(pose.get("x"))
        y = self._as_float_or_none(pose.get("y"))
        z = self._as_float_or_none(pose.get("z"))
        yaw = self._as_float_or_none(pose.get("task_yaw", pose.get("yaw")))
        if x is None or y is None:
            return {}
        return {
            "x": float(x),
            "y": float(y),
            "z": float(z if z is not None else 0.0),
            "yaw": float(yaw if yaw is not None else 0.0),
        }

    def route3_current_pose(self, session: Optional[flight.DroneFlightSession] = None) -> Dict[str, float]:
        pose = self.route3_pose_from_payload(self.latest_state)
        if pose:
            return pose
        if session is not None:
            state = self.safe("Route V3 get state", session.get_state)
            if isinstance(state, dict):
                self.root.after(0, lambda r=state: self.apply_state(r))
                return self.route3_pose_from_payload(state)
        return {}

    def route3_target_pose_from_point(self, point: Dict[str, Any]) -> Dict[str, float]:
        return {
            "x": float(point.get("x", point.get("world_x", 0.0)) or 0.0),
            "y": float(point.get("y", point.get("world_y", 0.0)) or 0.0),
            "z": float(point.get("z", 300.0) or 300.0),
            "yaw": float(point.get("yaw_deg", point.get("yaw", 0.0)) or 0.0),
        }

    def route3_pose_error(self, current: Dict[str, float], target: Dict[str, float], config: Dict[str, float]) -> Dict[str, Any]:
        dx = float(target["x"]) - float(current["x"])
        dy = float(target["y"]) - float(current["y"])
        dz = float(target["z"]) - float(current.get("z", 0.0))
        yaw_error = self._normalize_angle_deg(float(target.get("yaw", target.get("yaw_deg", 0.0))) - float(current.get("yaw", 0.0)))
        dist_xy = float(math.hypot(dx, dy))
        return {
            "dx": round(dx, 3),
            "dy": round(dy, 3),
            "dz": round(dz, 3),
            "dist_xy_cm": round(dist_xy, 3),
            "yaw_error_deg": round(float(yaw_error), 3),
            "reached": bool(
                dist_xy <= float(config["reach_tol_cm"])
                and abs(dz) <= float(config["z_tol_cm"])
                and abs(yaw_error) <= float(config["yaw_tol_deg"])
            ),
        }

    def route3_movement_payload_for_target(self, current: Dict[str, float], target: Dict[str, float], config: Dict[str, float]) -> Dict[str, Any]:
        error = self.route3_pose_error(current, target, config)
        yaw_rad = math.radians(float(current.get("yaw", 0.0)))
        dx = float(error["dx"])
        dy = float(error["dy"])
        forward = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
        right = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        horizontal_len = math.hypot(forward, right)
        step = float(config["nav_step_cm"])
        if horizontal_len <= float(config["reach_tol_cm"]):
            forward = 0.0
            right = 0.0
        elif horizontal_len > step and horizontal_len > 1e-6:
            scale = step / horizontal_len
            forward *= scale
            right *= scale
        up = float(error["dz"])
        if abs(up) <= float(config["z_tol_cm"]):
            up = 0.0
        else:
            up = max(-step, min(step, up))
        yaw_delta = float(error["yaw_error_deg"])
        if abs(yaw_delta) <= float(config["yaw_tol_deg"]):
            yaw_delta = 0.0
        else:
            yaw_delta = max(-30.0, min(30.0, yaw_delta))
        if abs(forward) <= 1e-6 and abs(right) <= 1e-6 and abs(up) <= 1e-6 and abs(yaw_delta) <= 1e-6:
            action = "hold"
        else:
            action = "route3_nav"
        return {
            "forward_cm": round(float(forward), 3),
            "right_cm": round(float(right), 3),
            "up_cm": round(float(up), 3),
            "yaw_delta_deg": round(float(yaw_delta), 3),
            "action_name": action,
        }

    def route3_predict_next_pose(self, current: Dict[str, float], payload: Dict[str, Any]) -> Dict[str, float]:
        yaw_rad = math.radians(float(current.get("yaw", 0.0)))
        forward = float(payload.get("forward_cm", 0.0) or 0.0)
        right = float(payload.get("right_cm", 0.0) or 0.0)
        return {
            "x": float(current["x"]) + forward * math.cos(yaw_rad) - right * math.sin(yaw_rad),
            "y": float(current["y"]) + forward * math.sin(yaw_rad) + right * math.cos(yaw_rad),
            "z": float(current.get("z", 0.0)) + float(payload.get("up_cm", 0.0) or 0.0),
            "yaw": self._normalize_angle_deg(float(current.get("yaw", 0.0)) + float(payload.get("yaw_delta_deg", 0.0) or 0.0)),
        }

    def route3_safety_report_for_pose(self, target_house_id: str, pose: Dict[str, float]) -> Dict[str, Any]:
        bounds = self.route2_observation_map_bounds_report(float(pose["x"]), float(pose["y"]))
        if not bool(bounds.get("in_bounds", True)):
            return {"safe": False, "reason": "map_boundary", "map_bounds": bounds}
        target_bbox = self.house_world_bbox_for_id(str(target_house_id or "").strip())
        if target_bbox and self.point_inside_open_bbox(float(pose["x"]), float(pose["y"]), target_bbox):
            return {
                "safe": False,
                "reason": "target_house_bbox",
                "blocking_house_id": str(target_house_id or ""),
                "obstacle": target_bbox,
                "map_bounds": bounds,
            }
        obstacle = self.route2_observation_blocking_house(
            target_house_id,
            float(pose["x"]),
            float(pose["y"]),
            clearance_cm=float(LLM_ROUTE_HOUSE_CLEARANCE_CM),
        )
        if obstacle:
            return {
                "safe": False,
                "reason": "non_target_house_clearance",
                "blocking_house_id": str(obstacle.get("house_id", "") or ""),
                "obstacle": obstacle,
                "map_bounds": bounds,
            }
        return {"safe": True, "reason": "", "map_bounds": bounds}

    def route3_navigation_world_bounds(
        self,
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        target_house_id: str,
    ) -> Tuple[float, float, float, float]:
        xs = [float(start_pose["x"]), float(target_pose["x"])]
        ys = [float(start_pose["y"]), float(target_pose["y"])]
        house_ids = {str(target_house_id or "").strip()}
        for item in self.house_records_for_route_planning():
            if not isinstance(item, dict):
                continue
            house_id = str(item.get("id", item.get("house_id", "")) or "")
            if house_id:
                house_ids.add(house_id)
        for house_id in house_ids:
            bbox = self.house_world_bbox_for_id(house_id)
            if not bbox:
                continue
            xs.extend([float(bbox["min_x"]), float(bbox["max_x"])])
            ys.extend([float(bbox["min_y"]), float(bbox["max_y"])])
        bounds = getattr(self, "map_world_bounds", None)
        if isinstance(bounds, tuple) and len(bounds) == 4:
            try:
                min_x, min_y, max_x, max_y = [float(value) for value in bounds]
                xs.extend([min_x, max_x])
                ys.extend([min_y, max_y])
            except Exception:
                pass
        margin = max(1200.0, 3.0 * float(LLM_ROUTE3_ASTAR_GRID_CM))
        return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin

    def route3_navigation_obstacles(self, target_house_id: str) -> List[Dict[str, Any]]:
        planner_buffer = 2.0
        raw_obstacles = [
            dict(item)
            for item in self.route_forbidden_house_bboxes(
                target_house_id=str(target_house_id or "").strip(),
                clearance_cm=float(LLM_ROUTE_HOUSE_CLEARANCE_CM),
            )
            if isinstance(item, dict)
        ]
        target_bbox = self.house_world_bbox_for_id(str(target_house_id or "").strip())
        if target_bbox:
            target_raw = dict(target_bbox)
            target_raw["house_id"] = str(target_house_id or "").strip()
            target_raw["clearance_cm"] = 0.0
            target_raw["obstacle_role"] = "target_house_raw_bbox_no_fly"
            raw_obstacles.append(target_raw)
        obstacles: List[Dict[str, Any]] = []
        for obstacle in raw_obstacles:
            item = dict(obstacle)
            try:
                item["min_x"] = float(item["min_x"]) - planner_buffer
                item["max_x"] = float(item["max_x"]) + planner_buffer
                item["min_y"] = float(item["min_y"]) - planner_buffer
                item["max_y"] = float(item["max_y"]) + planner_buffer
                item["planner_buffer_cm"] = planner_buffer
            except Exception:
                pass
            obstacles.append(item)
        return obstacles

    def route3_point_blocked_by_obstacles(self, x: float, y: float, obstacles: List[Dict[str, Any]]) -> Dict[str, Any]:
        for obstacle in obstacles:
            if self.point_inside_open_bbox(float(x), float(y), obstacle):
                return dict(obstacle)
        return {}

    def route3_segment_blocked_by_obstacles(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any],
        obstacles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ax = float(start["x"])
        ay = float(start["y"])
        bx = float(end["x"])
        by = float(end["y"])
        for obstacle in obstacles:
            if self.segment_intersects_open_bbox(ax, ay, bx, by, obstacle):
                item = dict(obstacle)
                item["segment_block_source"] = "analytic_bbox_intersection"
                return item
        distance = math.hypot(bx - ax, by - ay)
        steps = max(2, int(math.ceil(distance / max(1.0, float(LLM_ROUTE3_SEGMENT_SAFETY_SAMPLE_CM)))))
        for index in range(1, steps):
            t = float(index) / float(steps)
            x = ax + (bx - ax) * t
            y = ay + (by - ay) * t
            blocker = self.route3_point_blocked_by_obstacles(x, y, obstacles)
            if blocker:
                blocker["segment_block_source"] = "sampled_bbox_intersection"
                blocker["segment_sample_t"] = round(float(t), 4)
                blocker["segment_sample_x"] = round(float(x), 3)
                blocker["segment_sample_y"] = round(float(y), 3)
                return blocker
        return {}

    def route3_obstacle_identity(self, obstacle: Dict[str, Any]) -> str:
        return str(obstacle.get("house_id", obstacle.get("id", "")) or "")

    def route3_escape_waypoint_from_obstacle(
        self,
        start_pose: Dict[str, float],
        obstacle: Dict[str, Any],
        target_pose: Dict[str, float],
        obstacles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        margin = max(
            float(LLM_ROUTE3_ESCAPE_MARGIN_CM),
            float(LLM_ROUTE3_NAV_SEGMENT_REACH_TOL_CM) + 0.5 * float(LLM_ROUTE3_ASTAR_GRID_CM),
        )
        x = float(start_pose["x"])
        y = float(start_pose["y"])
        min_x = float(obstacle["min_x"])
        max_x = float(obstacle["max_x"])
        min_y = float(obstacle["min_y"])
        max_y = float(obstacle["max_y"])
        target_x = float(target_pose["x"])
        target_y = float(target_pose["y"])

        def clamp(value: float, low: float, high: float) -> float:
            return max(float(low), min(float(high), float(value)))

        candidates: List[Dict[str, Any]] = []
        seen: set[Tuple[int, int, str]] = set()

        def add_candidate(cx: float, cy: float, side: str) -> None:
            key = (int(round(float(cx) * 10.0)), int(round(float(cy) * 10.0)), side)
            if key in seen:
                return
            seen.add(key)
            candidates.append({"x": float(cx), "y": float(cy), "exit_side": side})

        side_y_values = [
            y,
            clamp(target_y, min_y - margin, max_y + margin),
            min_y - margin,
            max_y + margin,
        ]
        for side_y in side_y_values:
            add_candidate(min_x - margin, side_y, "west")
            add_candidate(max_x + margin, side_y, "east")
        side_x_values = [
            x,
            clamp(target_x, min_x - margin, max_x + margin),
            min_x - margin,
            max_x + margin,
        ]
        for side_x in side_x_values:
            add_candidate(side_x, min_y - margin, "south")
            add_candidate(side_x, max_y + margin, "north")
        valid: List[Dict[str, Any]] = []
        for candidate in candidates:
            bounds_report = self.route2_observation_map_bounds_report(float(candidate["x"]), float(candidate["y"]))
            if not bool(bounds_report.get("in_bounds", True)):
                continue
            blocker = self.route3_point_blocked_by_obstacles(float(candidate["x"]), float(candidate["y"]), obstacles)
            if blocker:
                continue
            travel = math.hypot(float(candidate["x"]) - x, float(candidate["y"]) - y)
            to_target = math.hypot(float(target_pose["x"]) - float(candidate["x"]), float(target_pose["y"]) - float(candidate["y"]))
            item = {
                "x": round(float(candidate["x"]), 3),
                "y": round(float(candidate["y"]), 3),
                "z": round(float(start_pose.get("z", target_pose.get("z", 0.0))), 3),
                "yaw": round(float(start_pose.get("yaw", target_pose.get("yaw", 0.0))), 3),
                "waypoint_index": 0,
                "waypoint_final": False,
                "waypoint_role": "escape",
                "escape_from_obstacle_house_id": self.route3_obstacle_identity(obstacle),
                "escape_from_obstacle": obstacle,
                "escape_side": candidate["exit_side"],
                "escape_margin_cm": round(float(margin), 2),
                "strict_reach_tol_cm": float(LLM_ROUTE3_ESCAPE_REACH_TOL_CM),
                "strict_yaw_tol_deg": 180.0,
                "escape_travel_cm": round(float(travel), 2),
                "escape_score": round(float(travel + 0.25 * to_target), 2),
                "map_bounds": bounds_report,
            }
            valid.append(item)
        if not valid:
            return {}
        valid.sort(key=lambda item: (float(item["escape_score"]), float(item["escape_travel_cm"])))
        return valid[0]

    def route3_escape_safety_allowed(
        self,
        current: Dict[str, float],
        predicted: Dict[str, float],
        target_pose: Dict[str, float],
        safety: Dict[str, Any],
        escape_obstacle: Optional[Dict[str, Any]],
    ) -> bool:
        if not escape_obstacle:
            return False
        if str(safety.get("reason", "") or "") not in {"non_target_house_clearance", "target_house_bbox"}:
            return False
        safety_obstacle = safety.get("obstacle", {}) if isinstance(safety.get("obstacle"), dict) else {}
        safety_id = self.route3_obstacle_identity(safety_obstacle)
        escape_id = self.route3_obstacle_identity(escape_obstacle)
        if safety_id and escape_id and safety_id != escape_id:
            return False
        if self.point_inside_open_bbox(float(target_pose["x"]), float(target_pose["y"]), escape_obstacle):
            return False
        current_inside = self.point_inside_open_bbox(float(current["x"]), float(current["y"]), escape_obstacle)
        predicted_inside = self.point_inside_open_bbox(float(predicted["x"]), float(predicted["y"]), escape_obstacle)
        if not current_inside and not predicted_inside:
            return False
        current_dist = math.hypot(float(target_pose["x"]) - float(current["x"]), float(target_pose["y"]) - float(current["y"]))
        predicted_dist = math.hypot(float(target_pose["x"]) - float(predicted["x"]), float(target_pose["y"]) - float(predicted["y"]))
        return bool(predicted_dist <= current_dist + 5.0)

    def route3_astar_index_for_point(
        self,
        x: float,
        y: float,
        *,
        min_x: float,
        min_y: float,
        grid_cm: float,
    ) -> Tuple[int, int]:
        return (
            int(round((float(x) - float(min_x)) / float(grid_cm))),
            int(round((float(y) - float(min_y)) / float(grid_cm))),
        )

    def route3_point_for_astar_index(
        self,
        node: Tuple[int, int],
        *,
        min_x: float,
        min_y: float,
        grid_cm: float,
    ) -> Dict[str, float]:
        return {
            "x": float(min_x) + float(node[0]) * float(grid_cm),
            "y": float(min_y) + float(node[1]) * float(grid_cm),
        }

    def route3_smooth_waypoints(
        self,
        waypoints: List[Dict[str, float]],
        obstacles: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        if len(waypoints) <= 2:
            return waypoints
        smoothed: List[Dict[str, float]] = [waypoints[0]]
        anchor_index = 0
        while anchor_index < len(waypoints) - 1:
            next_index = len(waypoints) - 1
            while next_index > anchor_index + 1:
                if not self.route3_segment_blocked_by_obstacles(waypoints[anchor_index], waypoints[next_index], obstacles):
                    break
                next_index -= 1
            smoothed.append(waypoints[next_index])
            anchor_index = next_index
        return smoothed

    def route3_plan_navigation_waypoints(
        self,
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        target_house_id: str,
        *,
        grid_cm: float = LLM_ROUTE3_ASTAR_GRID_CM,
    ) -> Dict[str, Any]:
        target_safety = self.route3_safety_report_for_pose(target_house_id, target_pose)
        if not bool(target_safety.get("safe", False)):
            return {"status": "blocked", "reason": "unsafe_target", "target_safety": target_safety, "waypoints": []}
        obstacles = self.route3_navigation_obstacles(target_house_id)
        start_obstacle = self.route3_point_blocked_by_obstacles(float(start_pose["x"]), float(start_pose["y"]), obstacles)
        target_obstacle = self.route3_point_blocked_by_obstacles(float(target_pose["x"]), float(target_pose["y"]), obstacles)
        if target_obstacle:
            return {"status": "blocked", "reason": "target_inside_obstacle", "obstacle": target_obstacle, "waypoints": []}
        planning_start = dict(start_pose)
        escape_waypoint: Dict[str, Any] = {}
        if start_obstacle:
            escape_waypoint = self.route3_escape_waypoint_from_obstacle(start_pose, start_obstacle, target_pose, obstacles)
            if not escape_waypoint:
                return {"status": "blocked", "reason": "start_inside_obstacle", "obstacle": start_obstacle, "waypoints": []}
            planning_start = {
                "x": float(escape_waypoint["x"]),
                "y": float(escape_waypoint["y"]),
                "z": float(escape_waypoint.get("z", start_pose.get("z", target_pose.get("z", 0.0)))),
                "yaw": float(escape_waypoint.get("yaw", start_pose.get("yaw", target_pose.get("yaw", 0.0)))),
            }
        direct_blocker = self.route3_segment_blocked_by_obstacles(planning_start, target_pose, obstacles)
        direct_bounds = self.route2_observation_map_bounds_report(float(target_pose["x"]), float(target_pose["y"]))
        if not direct_blocker and bool(direct_bounds.get("in_bounds", True)):
            waypoints = [dict(target_pose)]
            raw_waypoints = [dict(planning_start), dict(target_pose)]
            reason = "direct_path_clear"
            if escape_waypoint:
                waypoints = [dict(escape_waypoint), dict(target_pose)]
                raw_waypoints = [dict(start_pose), dict(escape_waypoint), dict(target_pose)]
                reason = "start_escape_then_direct_path"
            return {
                "status": "ok",
                "reason": reason,
                "grid_cm": float(grid_cm),
                "waypoints": waypoints,
                "raw_waypoints": raw_waypoints,
                "start_obstacle": start_obstacle,
                "escape_waypoint": escape_waypoint,
                "obstacle_count": len(obstacles),
            }

        min_x, min_y, max_x, max_y = self.route3_navigation_world_bounds(planning_start, target_pose, target_house_id)
        grid = max(40.0, float(grid_cm))
        start_node = self.route3_astar_index_for_point(float(planning_start["x"]), float(planning_start["y"]), min_x=min_x, min_y=min_y, grid_cm=grid)
        target_node = self.route3_astar_index_for_point(float(target_pose["x"]), float(target_pose["y"]), min_x=min_x, min_y=min_y, grid_cm=grid)
        max_ix = int(math.ceil((max_x - min_x) / grid))
        max_iy = int(math.ceil((max_y - min_y) / grid))

        def node_valid(node: Tuple[int, int]) -> bool:
            if node == start_node or node == target_node:
                return True
            ix, iy = node
            if ix < 0 or iy < 0 or ix > max_ix or iy > max_iy:
                return False
            point = self.route3_point_for_astar_index(node, min_x=min_x, min_y=min_y, grid_cm=grid)
            bounds_report = self.route2_observation_map_bounds_report(point["x"], point["y"])
            if not bool(bounds_report.get("in_bounds", True)):
                return False
            return not bool(self.route3_point_blocked_by_obstacles(point["x"], point["y"], obstacles))

        neighbors = [
            (-1, -1, math.sqrt(2.0)),
            (-1, 0, 1.0),
            (-1, 1, math.sqrt(2.0)),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, -1, math.sqrt(2.0)),
            (1, 0, 1.0),
            (1, 1, math.sqrt(2.0)),
        ]
        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start_node))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        best_cost: Dict[Tuple[int, int], float] = {start_node: 0.0}
        visited = 0
        max_visits = max(1000, (max_ix + 1) * (max_iy + 1))
        found = False
        while open_heap and visited <= max_visits:
            _priority, cost, node = heapq.heappop(open_heap)
            if cost > best_cost.get(node, float("inf")) + 1e-6:
                continue
            visited += 1
            if node == target_node:
                found = True
                break
            for dx, dy, step_cost in neighbors:
                nxt = (node[0] + dx, node[1] + dy)
                if not node_valid(nxt):
                    continue
                if node == start_node:
                    current_point = {"x": float(planning_start["x"]), "y": float(planning_start["y"])}
                else:
                    current_point = self.route3_point_for_astar_index(node, min_x=min_x, min_y=min_y, grid_cm=grid)
                if nxt == target_node:
                    next_point = {"x": float(target_pose["x"]), "y": float(target_pose["y"])}
                else:
                    next_point = self.route3_point_for_astar_index(nxt, min_x=min_x, min_y=min_y, grid_cm=grid)
                if self.route3_segment_blocked_by_obstacles(current_point, next_point, obstacles):
                    continue
                new_cost = cost + step_cost
                if new_cost + 1e-6 < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = new_cost
                    came_from[nxt] = node
                    heuristic = math.hypot(float(target_node[0] - nxt[0]), float(target_node[1] - nxt[1]))
                    heapq.heappush(open_heap, (new_cost + heuristic, new_cost, nxt))
        if not found:
            return {
                "status": "blocked",
                "reason": "astar_no_path",
                "grid_cm": grid,
                "bounds": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
                "visited": visited,
                "obstacle_count": len(obstacles),
                "direct_blocker": direct_blocker,
                "waypoints": [],
            }
        nodes = [target_node]
        while nodes[-1] != start_node:
            nodes.append(came_from[nodes[-1]])
        nodes.reverse()
        raw_xy = [self.route3_point_for_astar_index(node, min_x=min_x, min_y=min_y, grid_cm=grid) for node in nodes]
        raw_xy[0] = {"x": float(planning_start["x"]), "y": float(planning_start["y"])}
        raw_xy[-1] = {"x": float(target_pose["x"]), "y": float(target_pose["y"])}
        smooth_xy = self.route3_smooth_waypoints(raw_xy, obstacles)
        waypoints: List[Dict[str, float]] = []
        for idx, point in enumerate(smooth_xy[1:], start=1):
            is_last = idx == len(smooth_xy) - 1
            waypoint = {
                "x": round(float(point["x"]), 3),
                "y": round(float(point["y"]), 3),
                "z": round(float(target_pose["z"] if is_last else start_pose.get("z", target_pose["z"])), 3),
                "yaw": round(float(target_pose["yaw"] if is_last else start_pose.get("yaw", target_pose["yaw"])), 3),
                "waypoint_index": idx,
                "waypoint_final": bool(is_last),
            }
            waypoints.append(waypoint)
        raw_waypoints_out: List[Dict[str, Any]] = raw_xy
        if escape_waypoint:
            waypoints = [dict(escape_waypoint)] + waypoints
            raw_waypoints_out = [dict(start_pose), dict(escape_waypoint)] + raw_xy[1:]
        return {
            "status": "ok",
            "reason": "start_escape_then_astar_path" if escape_waypoint else "astar_path",
            "grid_cm": grid,
            "bounds": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
            "visited": visited,
            "raw_waypoints": raw_waypoints_out,
            "waypoints": waypoints,
            "start_obstacle": start_obstacle,
            "escape_waypoint": escape_waypoint,
            "obstacle_count": len(obstacles),
            "direct_blocker": direct_blocker,
        }

    def route3_navigation_plan_cost_cm(self, start_pose: Dict[str, float], plan: Dict[str, Any]) -> float:
        if not isinstance(plan, dict) or plan.get("status") != "ok":
            return float("inf")
        waypoints = [item for item in plan.get("waypoints", []) if isinstance(item, dict)]
        if not waypoints:
            return 0.0
        total = 0.0
        previous = dict(start_pose)
        for waypoint in waypoints:
            total += math.hypot(float(waypoint.get("x", 0.0)) - float(previous.get("x", 0.0)), float(waypoint.get("y", 0.0)) - float(previous.get("y", 0.0)))
            previous = waypoint
        return float(total)

    def route3_observation_attempts_for_facade(
        self,
        target_house_id: str,
        facade: str,
        base_candidate: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        hid = str(target_house_id or "").strip()
        facade = str(facade or "").strip().lower()
        bbox = self.house_world_bbox_for_id(hid)
        if not hid or facade not in {"south", "east", "north", "west"} or not bbox:
            return []
        base = dict(base_candidate or {})
        corridor_info = self.scan_corridor_standoff_by_facade(hid, bbox)
        facade_info = corridor_info.get(facade, {}) if isinstance(corridor_info.get(facade), dict) else {}
        axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
        axis_center = self.route2_facade_center_axis(bbox, facade)
        facade_length = abs(float(axis_max) - float(axis_min))
        facade_depth = abs(float(bbox["max_y"]) - float(bbox["min_y"])) if facade in {"south", "north"} else abs(float(bbox["max_x"]) - float(bbox["min_x"]))
        desired_standoff, observation_meta = self.route2_observation_standoff_cm(
            facade_length_cm=facade_length,
            facade_depth_cm=facade_depth,
            facade_info=facade_info,
        )
        base_standoff = self._as_float_or_none(base.get("standoff_cm"))
        base_standoff = float(base_standoff if base_standoff is not None else desired_standoff)
        base_z = self._as_float_or_none(base.get("z"))
        if base_z is None:
            pose = self.current_route_pose()
            pose_z = self._as_float_or_none(pose.get("z")) if pose else None
            base_z = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), float(pose_z if pose_z is not None else LLM_ROUTE2_OBSERVATION_Z_CM))
        base_z = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), min(float(LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM), float(base_z)))
        attempts: List[Dict[str, Any]] = []
        seen: set[Tuple[int, int, int, int]] = set()

        def add_attempt(axis_value: float, standoff: float, z_cm: float, source: str, *, seed: Optional[Dict[str, Any]] = None) -> None:
            if len(attempts) >= int(LLM_ROUTE3_OBSERVATION_ATTEMPT_MAX):
                return
            seed = dict(seed or {})
            if seed and self._as_float_or_none(seed.get("x")) is not None and self._as_float_or_none(seed.get("y")) is not None:
                pose_payload = {
                    "x": round(float(seed.get("x", 0.0)), 2),
                    "y": round(float(seed.get("y", 0.0)), 2),
                    "z": round(float(seed.get("z", z_cm) or z_cm), 2),
                    "yaw_deg": round(float(seed.get("yaw_deg", seed.get("yaw", 0.0)) or 0.0), 2),
                    "target_x": seed.get("target_x"),
                    "target_y": seed.get("target_y"),
                }
                axis = self._as_float_or_none(seed.get("axis_value"))
                if axis is None:
                    axis = axis_value
                selected_standoff = self._as_float_or_none(seed.get("standoff_cm"))
                selected_standoff = float(selected_standoff if selected_standoff is not None else standoff)
                meta = {key: value for key, value in seed.items() if str(key).startswith("observation_")}
                boundary_adjustment = seed.get("observation_boundary_adjustment", {}) if isinstance(seed.get("observation_boundary_adjustment"), dict) else {}
            else:
                axis = float(axis_value)
                selected_standoff = float(standoff)
                meta = self.route2_observation_meta_for_standoff(observation_meta, facade_info, selected_standoff, source)
                pose_payload = self.route2_facade_pose_from_axis(bbox, facade, axis, selected_standoff, z_cm)
                boundary_adjustment = {}
                blocking_obstacle = self.route2_observation_blocking_house(hid, float(pose_payload["x"]), float(pose_payload["y"]))
                if blocking_obstacle:
                    adjusted = self.route2_adjust_observation_to_blocking_boundary(
                        bbox,
                        facade,
                        axis,
                        z_cm,
                        selected_standoff,
                        blocking_obstacle,
                    )
                    if adjusted:
                        pose_payload = adjusted.get("pose", pose_payload) if isinstance(adjusted.get("pose"), dict) else pose_payload
                        selected_standoff = float(adjusted.get("standoff_cm", selected_standoff))
                        meta = self.route2_observation_meta_for_standoff(
                            observation_meta,
                            facade_info,
                            selected_standoff,
                            f"{source}_boundary_adjusted",
                        )
                        boundary_adjustment = adjusted.get("adjustment", {}) if isinstance(adjusted.get("adjustment"), dict) else {}
            key = (
                int(round(float(pose_payload.get("x", 0.0)) * 10.0)),
                int(round(float(pose_payload.get("y", 0.0)) * 10.0)),
                int(round(float(pose_payload.get("z", z_cm)) * 10.0)),
                int(round(float(pose_payload.get("yaw_deg", 0.0)) * 10.0)),
            )
            if key in seen:
                return
            seen.add(key)
            bounds_report = self.route2_observation_map_bounds_report(float(pose_payload["x"]), float(pose_payload["y"]))
            blocking = self.route2_observation_blocking_house(hid, float(pose_payload["x"]), float(pose_payload["y"]))
            status = "planned" if not blocking and bool(bounds_report.get("in_bounds", True)) else "blocked"
            attempt = {
                **pose_payload,
                "label": f"{hid}_{facade}_obs_{len(attempts) + 1}",
                "route_point_type": "observation_point",
                "house_id": hid,
                "facade": facade,
                "facade_id": self.route2_facade_id(hid, facade),
                "axis_value": round(float(axis), 2),
                "axis_center_cm": round(float(axis_center), 2),
                "axis_center_error_cm": round(abs(float(axis) - float(axis_center)), 2),
                "standoff_cm": round(float(selected_standoff), 2),
                **meta,
                "observation_attempt_index": len(attempts),
                "observation_attempt_source": source,
                "observation_map_bounds": bounds_report,
                "observation_boundary_adjustment": boundary_adjustment,
                "observation_blocking_house_id": str(blocking.get("house_id", "") or ""),
                "observation_block_reason": "non_target_house" if blocking else ("" if bounds_report.get("in_bounds", True) else "map_boundary"),
                "status": status,
            }
            attempts.append(attempt)

        add_attempt(float(base.get("axis_value", axis_center) or axis_center), base_standoff, base_z, "base_candidate", seed=base if base else None)
        add_attempt(axis_center, base_standoff, base_z, "center_projection")
        add_attempt(float(axis_min) + (float(axis_max) - float(axis_min)) / 3.0, base_standoff, base_z, "left_third_projection")
        add_attempt(float(axis_min) + 2.0 * (float(axis_max) - float(axis_min)) / 3.0, base_standoff, base_z, "right_third_projection")
        far_standoff = min(float(LLM_ROUTE2_OBSERVATION_MAX_STANDOFF_CM), max(base_standoff + 350.0, base_standoff * 1.25))
        add_attempt(axis_center, far_standoff, base_z, "far_center_projection")
        elevated_z = min(float(LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM), base_z + float(LLM_ROUTE3_OBSTACLE_RAISE_STEP_CM))
        add_attempt(axis_center, base_standoff, elevated_z, "elevated_center_projection")
        return attempts

    def route3_task_facade_priority(self) -> List[str]:
        state = self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {}
        plan = state.get("task_plan", {}) if isinstance(state.get("task_plan"), dict) else {}
        raw = plan.get("facade_priority", state.get("facade_priority", [])) if isinstance(plan, dict) else []
        priority: List[str] = []
        if isinstance(raw, list):
            for item in raw:
                facade = str(item or "").strip().lower()
                if facade in {"south", "east", "north", "west"} and facade not in priority:
                    priority.append(facade)
        for facade in ("west", "south", "east", "north"):
            if facade not in priority:
                priority.append(facade)
        return priority

    def route3_rank_observation_candidates(
        self,
        target_house_id: str,
        candidates: List[Dict[str, Any]],
        completed: set[str],
        blocked: set[str],
        *,
        start_pose: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        current = dict(start_pose or self.route3_current_pose() or self.current_route_pose() or {})
        priority = self.route3_task_facade_priority()
        priority_index = {facade: idx for idx, facade in enumerate(priority)}
        base_by_facade: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
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
                    item["route3_navigation_status"] = "unknown_no_current_pose"
                    item["route3_navigation_cost_cm"] = float(item.get("observation_selection_score", item.get("distance_to_uav_cm", 0.0)) or 0.0)
                    ranked_attempts.append(item)
                    continue
                plan = self.route3_plan_navigation_waypoints(current, pose, target_house_id, grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM))
                item["route3_navigation_plan"] = plan
                item["route3_navigation_status"] = str(plan.get("status", "blocked") or "blocked")
                item["route3_navigation_reason"] = str(plan.get("reason", "") or "")
                if plan.get("status") == "ok":
                    item["route3_navigation_cost_cm"] = round(self.route3_navigation_plan_cost_cm(current, plan), 2)
                    item["status"] = "planned"
                else:
                    item["route3_navigation_cost_cm"] = float("inf")
                    item["status"] = "blocked"
                    item["observation_block_reason"] = str(plan.get("reason", item.get("observation_block_reason", "navigation_blocked")) or "navigation_blocked")
                ranked_attempts.append(item)
            ranked_attempts.sort(
                key=lambda item: (
                    0 if item.get("route3_navigation_status") == "ok" else 1,
                    float(item.get("route3_navigation_cost_cm", float("inf"))),
                    int(item.get("observation_attempt_index", 999) or 999),
                )
            )
            feasible = next((item for item in ranked_attempts if item.get("route3_navigation_status") == "ok"), None)
            selected = dict(feasible or (ranked_attempts[0] if ranked_attempts else base))
            rank_penalty = 25.0 * float(priority_index.get(facade, len(priority)))
            nav_cost = float(selected.get("route3_navigation_cost_cm", selected.get("distance_to_uav_cm", 0.0)) or 0.0)
            if not math.isfinite(nav_cost):
                nav_cost = 1_000_000.0
            selected["route3_observation_rank_score"] = round(float(nav_cost + rank_penalty), 2)
            selected["route3_facade_priority_index"] = priority_index.get(facade, len(priority))
            selected["selected_observation_attempt"] = dict(feasible or selected)
            selected["observation_attempts"] = ranked_attempts
            selected["observation_attempt_count"] = len(ranked_attempts)
            if feasible is None:
                selected["status"] = "blocked"
                selected["route3_navigation_status"] = str(selected.get("route3_navigation_status", "blocked") or "blocked")
            ranked.append(selected)
        ranked.sort(
            key=lambda item: (
                0 if item.get("route3_navigation_status") == "ok" and item.get("status") != "blocked" else 1,
                float(item.get("route3_observation_rank_score", 1_000_000.0)),
                int(item.get("route3_facade_priority_index", 99) or 99),
            )
        )
        return ranked

    def route3_ordered_observation_attempts(self, selected: Dict[str, Any]) -> List[Dict[str, Any]]:
        attempts = [dict(item) for item in selected.get("observation_attempts", []) if isinstance(item, dict)]
        primary = selected.get("selected_observation_attempt", {}) if isinstance(selected.get("selected_observation_attempt"), dict) else {}
        ordered: List[Dict[str, Any]] = []
        seen: set[Tuple[int, int, int]] = set()

        def add(item: Dict[str, Any]) -> None:
            if not item:
                return
            key = (
                int(round(float(item.get("x", 0.0)) * 10.0)),
                int(round(float(item.get("y", 0.0)) * 10.0)),
                int(round(float(item.get("z", 0.0)) * 10.0)),
            )
            if key in seen:
                return
            seen.add(key)
            ordered.append(dict(item))

        add(primary)
        for item in attempts:
            if item.get("route3_navigation_status") == "ok" or item.get("status") != "blocked":
                add(item)
        for item in attempts:
            add(item)
        return ordered

    def route3_prepare_next_facade_hint(
        self,
        target_house_id: str,
        completed: set[str],
        blocked: set[str],
        *,
        exclude_facade: str = "",
        start_pose: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        completed_for_hint = set(completed)
        if exclude_facade:
            completed_for_hint.add(str(exclude_facade).strip().lower())
        raw_candidates = self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False)
        ranked = self.route3_rank_observation_candidates(
            target_house_id,
            raw_candidates,
            completed_for_hint,
            blocked,
            start_pose=start_pose,
        )
        feasible = next((item for item in ranked if item.get("route3_navigation_status") == "ok" and item.get("status") != "blocked"), None)
        if not feasible:
            return {"target_facade": "", "reason": "no_feasible_next_facade", "ranked_facade_candidates": ranked}
        observation = feasible.get("selected_observation_attempt", feasible)
        return {
            "target_facade": str(feasible.get("facade", "") or ""),
            "observation_point": observation,
            "navigation_cost_cm": feasible.get("route3_navigation_cost_cm", feasible.get("route3_observation_rank_score")),
            "reason": "nearest feasible next facade hint",
            "ranked_facade_candidates": ranked,
        }

    def route3_hold(self, session: flight.DroneFlightSession, *, output_dir: Optional[Path] = None, reason: str = "hold") -> Dict[str, Any]:
        payload = {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "hold"}
        result = self.safe("Route V3 hold", lambda: session.move_relative(payload))
        self.route3_log_event(output_dir, "hold", {"reason": reason, "payload": payload})
        return result if isinstance(result, dict) else {}

    def route3_set_control_lock(self, locked: bool) -> None:
        self.llm_route3_control_locked = bool(locked)
        try:
            if locked:
                self.root.after(0, lambda: self.stop_keyboard_control(send_hold=False))
                self.root.after(0, lambda: self.update_keyboard_status("locked by LLM Route V3"))
            else:
                self.root.after(0, self.update_keyboard_status)
        except Exception:
            pass

    def route3_enable_physics_movement(self, session: flight.DroneFlightSession) -> None:
        self.route3_set_control_lock(True)
        result = self.safe("Route V3 movement mode physics", lambda: session.set_movement_mode("physics"))
        self.safe("Route V3 enable movement", lambda: session.set_movement_enabled(True))
        self.movement_mode_state = "physics"
        try:
            self.root.after(0, lambda: self.movement_mode_var.set("physics"))
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_state(r))
        except Exception:
            pass

    def route3_wait_if_paused(self, session: flight.DroneFlightSession, output_dir: Optional[Path]) -> bool:
        held = False
        while self.llm_route3_pause_event.is_set() and not self.llm_route3_stop_event.is_set():
            if not held:
                self.route3_hold(session, output_dir=output_dir, reason="paused")
                held = True
            self.root.after(0, lambda: self.llm_route3_status_var.set("LLM Route V3: paused."))
            time.sleep(0.2)
        return bool(self.llm_route3_stop_event.is_set())

    def route3_follow_navigation_waypoint_with_movement(
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
        step_index = 0
        last_error: Dict[str, Any] = {}
        target_safety = self.route3_safety_report_for_pose(target_house_id, target_pose)
        if not bool(target_safety.get("safe", False)):
            self.route3_hold(session, output_dir=output_dir, reason="unsafe_waypoint")
            return {
                "status": "blocked",
                "reason": "unsafe_waypoint",
                "stage": stage,
                "facade": facade,
                "target_id": target_id,
                "waypoint_index": waypoint_index,
                "waypoint_count": waypoint_count,
                "safety": target_safety,
            }
        while not self.llm_route3_stop_event.is_set():
            if self.route3_wait_if_paused(session, output_dir):
                break
            error = self.route3_pose_error(current, target_pose, config)
            last_error = error
            self.root.after(
                0,
                lambda e=error: self.llm_route3_error_var.set(
                    f"Error: xy={float(e['dist_xy_cm']):.1f} z={float(e['dz']):.1f} yaw={float(e['yaw_error_deg']):.1f}"
                ),
            )
            if bool(error.get("reached", False)):
                self.route3_hold(session, output_dir=output_dir, reason="target_reached")
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
                self.route3_hold(session, output_dir=output_dir, reason="nav_timeout")
                return {
                    "status": "timeout",
                    "reason": "nav_timeout",
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "waypoint_index": waypoint_index,
                    "waypoint_count": waypoint_count,
                    "pose_error": error,
                    "elapsed_s": round(time.time() - started_at, 3),
                    "current_pose": current,
                }
            payload = self.route3_movement_payload_for_target(current, target_pose, config)
            predicted = self.route3_predict_next_pose(current, payload)
            safety = self.route3_safety_report_for_pose(target_house_id, predicted)
            escape_safety_allowed = self.route3_escape_safety_allowed(
                current,
                predicted,
                target_pose,
                safety,
                escape_obstacle,
            )
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
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            if escape_safety_allowed:
                trace["safety_override"] = {
                    "allowed": True,
                    "reason": "escape_start_inside_obstacle",
                    "escape_from_obstacle_house_id": self.route3_obstacle_identity(escape_obstacle or {}),
                }
            self.append_jsonl(output_dir / "movement_trace.jsonl", trace)
            self.root.after(
                0,
                lambda p=payload: self.llm_route3_payload_var.set(
                    f"Payload: f={p['forward_cm']} r={p['right_cm']} u={p['up_cm']} yaw={p['yaw_delta_deg']}"
                ),
            )
            if not bool(safety.get("safe", False)) and not escape_safety_allowed:
                self.route3_hold(session, output_dir=output_dir, reason=str(safety.get("reason", "unsafe_next_step")))
                self.route3_log_event(output_dir, "navigation_blocked", trace)
                return {
                    "status": "blocked",
                    "reason": str(safety.get("reason", "unsafe_next_step")),
                    "stage": stage,
                    "facade": facade,
                    "target_id": target_id,
                    "waypoint_index": waypoint_index,
                    "waypoint_count": waypoint_count,
                    "pose_error": error,
                    "safety": safety,
                    "current_pose": current,
                }
            if escape_safety_allowed:
                self.route3_log_event(output_dir, "navigation_escape_safety_override", trace)
            result = self.safe("Route V3 movement tick", lambda p=payload: session.move_relative(p))
            if not isinstance(result, dict):
                self.route3_hold(session, output_dir=output_dir, reason="movement_failed")
                return {
                    "status": "failed",
                    "reason": "movement_failed",
                    "pose_error": error,
                    "current_pose": current,
                    "waypoint_index": waypoint_index,
                    "waypoint_count": waypoint_count,
                }
            self.root.after(0, lambda r=result: self.apply_state(r))
            next_pose = self.route3_pose_from_payload(result)
            if next_pose:
                current = next_pose
            else:
                current = predicted
            step_index += 1
            if self.llm_route3_stop_event.wait(max(0.01, tick_s)):
                break
        self.route3_hold(session, output_dir=output_dir, reason="stopped")
        return {
            "status": "stopped",
            "reason": "stopped",
            "pose_error": last_error,
            "current_pose": current,
            "waypoint_index": waypoint_index,
            "waypoint_count": waypoint_count,
        }

    def route3_navigate_to_pose_with_movement(
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
        base_config = self.route3_nav_config()
        self.route3_enable_physics_movement(session)
        current = self.route3_current_pose(session)
        if not current:
            return {"status": "failed", "reason": "missing_current_pose"}
        max_replans = int(LLM_ROUTE3_ASTAR_MAX_REPLANS)
        replan_count = 0
        started_at = time.time()
        last_result: Dict[str, Any] = {}
        while replan_count <= max_replans and not self.llm_route3_stop_event.is_set():
            plan = self.route3_plan_navigation_waypoints(
                current,
                target_pose,
                target_house_id,
                grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM),
            )
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
            self.append_jsonl(output_dir / "navigation_plan.jsonl", plan_log)
            self.route3_log_event(output_dir, "navigation_plan", plan_log)
            if plan.get("status") != "ok":
                self.route3_hold(session, output_dir=output_dir, reason=str(plan.get("reason", "navigation_plan_failed")))
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
            waypoints = [dict(item) for item in plan.get("waypoints", []) if isinstance(item, dict)]
            if not waypoints:
                waypoints = [dict(target_pose)]
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
                    segment_config["reach_tol_cm"] = max(
                        float(segment_config["reach_tol_cm"]),
                        float(LLM_ROUTE3_NAV_SEGMENT_REACH_TOL_CM),
                    )
                    segment_config["yaw_tol_deg"] = 180.0
                result = self.route3_follow_navigation_waypoint_with_movement(
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
        self.route3_hold(session, output_dir=output_dir, reason="replan_exhausted")
        return {
            "status": "blocked",
            "reason": "replan_exhausted",
            "stage": stage,
            "facade": facade,
            "target_id": target_id,
            "last_result": last_result,
            "replan_count": replan_count,
            "elapsed_s": round(time.time() - started_at, 3),
        }

    def route3_observation_needs_panorama(self, observation: Dict[str, Any]) -> bool:
        if not isinstance(observation, dict) or not observation:
            return False
        adjustment = observation.get("observation_boundary_adjustment", {})
        if isinstance(adjustment, dict) and adjustment:
            return True
        coverage = self._as_float_or_none(observation.get("observation_panorama_coverage_ratio"))
        if coverage is not None and float(coverage) < float(LLM_ROUTE3_PANORAMA_COVERAGE_THRESHOLD):
            return True
        return False

    def route3_facade_target_point_from_axis(self, bbox: Dict[str, Any], facade: str, axis_value: float) -> Dict[str, float]:
        facade = str(facade or "").strip().lower()
        axis = float(axis_value)
        if facade == "south":
            return {"x": axis, "y": float(bbox["min_y"])}
        if facade == "north":
            return {"x": axis, "y": float(bbox["max_y"])}
        if facade == "east":
            return {"x": float(bbox["max_x"]), "y": axis}
        return {"x": float(bbox["min_x"]), "y": axis}

    def route3_panorama_observation_poses(
        self,
        house_id: str,
        facade: str,
        planned_pose: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        bbox = self.house_world_bbox_for_id(house_id)
        if not bbox:
            return [{"label": "center", **dict(planned_pose)}]
        axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
        axis_center = self.route2_facade_center_axis(bbox, facade)
        pose_x = float(planned_pose.get("x", 0.0))
        pose_y = float(planned_pose.get("y", 0.0))

        def yaw_to_axis(axis_value: float) -> float:
            target = self.route3_facade_target_point_from_axis(bbox, facade, axis_value)
            return math.degrees(math.atan2(float(target["y"]) - pose_y, float(target["x"]) - pose_x))

        center_yaw = yaw_to_axis(axis_center)

        def clamped_side_yaw(axis_value: float, fallback_sign: float) -> float:
            raw = yaw_to_axis(axis_value)
            delta = self._normalize_angle_deg(raw - center_yaw)
            sign = -1.0 if delta < 0.0 else (1.0 if delta > 0.0 else fallback_sign)
            magnitude = max(
                float(LLM_ROUTE3_PANORAMA_MIN_YAW_DELTA_DEG),
                min(float(LLM_ROUTE3_PANORAMA_MAX_YAW_DELTA_DEG), abs(float(delta))),
            )
            return self._normalize_angle_deg(center_yaw + sign * magnitude)

        return [
            {"label": "left", **dict(planned_pose), "yaw": round(float(clamped_side_yaw(axis_min, -1.0)), 2)},
            {"label": "center", **dict(planned_pose), "yaw": round(float(center_yaw), 2)},
            {"label": "right", **dict(planned_pose), "yaw": round(float(clamped_side_yaw(axis_max, 1.0)), 2)},
        ]

    def route3_copy_capture_rgb(self, capture_result: Dict[str, Any], output_path: Path) -> str:
        rgb_path = self.route2_candidate_path(capture_result.get("rgb_path"))
        if rgb_path is not None and rgb_path.exists() and rgb_path.is_file():
            try:
                if rgb_path.resolve() != output_path.resolve():
                    output_path.write_bytes(rgb_path.read_bytes())
            except Exception:
                output_path.write_bytes(rgb_path.read_bytes())
        return str(output_path if output_path.exists() else (rgb_path or ""))

    def route3_write_panorama_image(self, image_paths: List[Path], output_path: Path) -> bool:
        images: List[np.ndarray] = []
        for path in image_paths:
            try:
                if not path.exists():
                    continue
                data = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is not None and image.size:
                    images.append(image)
            except Exception:
                continue
        if not images:
            return False
        min_height = min(int(image.shape[0]) for image in images)
        resized: List[np.ndarray] = []
        for image in images:
            scale = float(min_height) / max(1.0, float(image.shape[0]))
            width = max(1, int(round(float(image.shape[1]) * scale)))
            resized.append(cv2.resize(image, (width, min_height), interpolation=cv2.INTER_AREA))
        panorama = cv2.hconcat(resized)
        ok, encoded = cv2.imencode(".png", panorama)
        if not ok:
            return False
        output_path.write_bytes(encoded.tobytes())
        return True

    def route3_capture_facade_rgb_panorama_current(
        self,
        session: flight.DroneFlightSession,
        *,
        output_dir: Path,
        facade_dir: Path,
        house_id: str,
        facade: str,
        planned_pose: Dict[str, float],
    ) -> Dict[str, Any]:
        poses = self.route3_panorama_observation_poses(house_id, facade, planned_pose)
        rows: List[Dict[str, Any]] = []
        captures: Dict[str, Any] = {}
        image_paths: Dict[str, str] = {}
        for pose in poses:
            label = str(pose.get("label", "center") or "center")
            yaw_pose = {
                "x": float(pose.get("x", planned_pose["x"])),
                "y": float(pose.get("y", planned_pose["y"])),
                "z": float(pose.get("z", planned_pose["z"])),
                "yaw": float(pose.get("yaw", planned_pose.get("yaw", 0.0))),
            }
            nav = self.route3_navigate_to_pose_with_movement(
                session,
                yaw_pose,
                output_dir=output_dir,
                stage="CAPTURE_RGB_PANORAMA",
                facade=facade,
                target_id=f"{house_id}_{facade}_panorama_{label}",
                target_house_id=house_id,
            )
            if nav.get("status") != "ok":
                return {"status": "failed", "reason": "panorama_yaw_navigation_failed", "label": label, "navigation": nav}
            action_detail = dict(self.build_stream_action_detail())
            action_detail.update(
                {
                    "source": "llm_route_v3_facade_observation_panorama",
                    "house_id": house_id,
                    "facade": facade,
                    "facade_id": self.route2_facade_id(house_id, facade),
                    "planned_pose": yaw_pose,
                    "panorama_label": label,
                    "panorama_pose_count": len(poses),
                }
            )
            frame_index = self.route2_next_frame_index(output_dir)
            capture_result = self.safe(
                f"Route V3 panorama RGB capture {label}",
                lambda idx=frame_index, action=action_detail: session.capture_lidar_stream_frame(output_dir, idx, action_detail=action),
            )
            if not isinstance(capture_result, dict):
                return {"status": "failed", "reason": "panorama_capture_failed", "label": label}
            image_path = facade_dir / f"coarse_rgb_{label}.png"
            image_paths[label] = self.route3_copy_capture_rgb(capture_result, image_path)
            captures[label] = capture_result
            row = {
                "frame_index": int(capture_result.get("frame_index", frame_index)),
                "capture_time": capture_result.get("capture_time", ""),
                "capture_kind": "facade_coarse_observation_v3_panorama",
                "scan_id": f"{house_id}_{facade}_coarse_observation_{label}",
                "house_id": house_id,
                "facade": facade,
                "facade_id": self.route2_facade_id(house_id, facade),
                "planned_pose": yaw_pose,
                "pose": capture_result.get("pose", {}),
                "commanded_pose": capture_result.get("commanded_pose", {}),
                "actual_pose": capture_result.get("actual_pose", {}),
                "pose_error": capture_result.get("pose_error", {}),
                "capture_dir": capture_result.get("capture_dir", ""),
                "rgb_path": capture_result.get("rgb_path", ""),
                "copied_rgb_path": image_paths[label],
                "point_cloud_world_standard_m_npy_path": capture_result.get("point_cloud_world_standard_m_npy_path", ""),
                "point_cloud_world_standard_m_ply_path": capture_result.get("point_cloud_world_standard_m_ply_path", ""),
                "point_cloud_preview_path": capture_result.get("point_cloud_preview_path", ""),
                "point_count": int(capture_result.get("point_count", 0) or 0),
                "action_detail": action_detail,
            }
            rows.append(row)
            self.append_jsonl(output_dir / "lidar_capture_log.jsonl", row)
            self.append_jsonl(facade_dir / "coarse_lidar_capture_log.jsonl", row)
        center_path = self.route2_candidate_path(image_paths.get("center"))
        if center_path is not None and center_path.exists():
            try:
                (facade_dir / "coarse_rgb.png").write_bytes(center_path.read_bytes())
            except Exception:
                pass
        panorama_path = facade_dir / "coarse_rgb_panorama.png"
        ordered_paths = [
            self.route2_candidate_path(image_paths.get("left")),
            self.route2_candidate_path(image_paths.get("center")),
            self.route2_candidate_path(image_paths.get("right")),
        ]
        stitch_ok = self.route3_write_panorama_image([path for path in ordered_paths if path is not None], panorama_path)
        if not stitch_ok and center_path is not None and center_path.exists():
            panorama_path.write_bytes(center_path.read_bytes())
        capture_payload = {
            "capture_kind": "facade_coarse_observation_v3_panorama",
            "house_id": house_id,
            "facade": facade,
            "planned_pose": planned_pose,
            "panorama_poses": poses,
            "capture_results": captures,
            "rows": rows,
            "coarse_rgb_left_path": image_paths.get("left", ""),
            "coarse_rgb_center_path": image_paths.get("center", ""),
            "coarse_rgb_right_path": image_paths.get("right", ""),
            "coarse_rgb_panorama_path": str(panorama_path if panorama_path.exists() else ""),
            "coarse_rgb_path": str(panorama_path if panorama_path.exists() else (center_path or "")),
            "panorama_stitch_ok": bool(stitch_ok),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(facade_dir / "panorama_capture.json", capture_payload)
        self.write_json_artifact(facade_dir / "coarse_capture.json", capture_payload)
        self.route2_write_lidar_summary(output_dir, running=True)
        self.route2_update_state(
            coarse_capture=capture_payload,
            panorama_capture=capture_payload,
            coarse_rgb_path=capture_payload["coarse_rgb_path"],
            coarse_rgb_panorama_path=capture_payload["coarse_rgb_panorama_path"],
        )
        self.route2_write_state_artifact()
        return {"status": "ok", "capture": capture_payload, "rows": rows}

    def route3_capture_facade_rgb_current(
        self,
        session: flight.DroneFlightSession,
        *,
        output_dir: Path,
        facade_dir: Path,
        house_id: str,
        facade: str,
        planned_pose: Dict[str, float],
    ) -> Dict[str, Any]:
        state = self.route2_selected_state()
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        if self.route3_observation_needs_panorama(observation):
            return self.route3_capture_facade_rgb_panorama_current(
                session,
                output_dir=output_dir,
                facade_dir=facade_dir,
                house_id=house_id,
                facade=facade,
                planned_pose=planned_pose,
            )
        action_detail = dict(self.build_stream_action_detail())
        action_detail.update(
            {
                "source": "llm_route_v3_facade_observation",
                "house_id": house_id,
                "facade": facade,
                "facade_id": self.route2_facade_id(house_id, facade),
                "planned_pose": planned_pose,
            }
        )
        frame_index = self.route2_next_frame_index(output_dir)
        capture_result = self.safe(
            "Route V3 coarse RGB/LiDAR capture",
            lambda: session.capture_lidar_stream_frame(output_dir, frame_index, action_detail=action_detail),
        )
        if not isinstance(capture_result, dict):
            return {"status": "failed", "reason": "capture_failed"}
        coarse_rgb_path = facade_dir / "coarse_rgb.png"
        rgb_path = self.route2_candidate_path(capture_result.get("rgb_path"))
        if rgb_path is not None and rgb_path.exists() and rgb_path.is_file():
            try:
                if rgb_path.resolve() != coarse_rgb_path.resolve():
                    coarse_rgb_path.write_bytes(rgb_path.read_bytes())
            except Exception:
                coarse_rgb_path.write_bytes(rgb_path.read_bytes())
        capture_payload = {
            "capture_kind": "facade_coarse_observation_v3",
            "house_id": house_id,
            "facade": facade,
            "planned_pose": planned_pose,
            "capture_result": capture_result,
            "coarse_rgb_path": str(coarse_rgb_path if coarse_rgb_path.exists() else (rgb_path or "")),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.write_json_artifact(facade_dir / "coarse_capture.json", capture_payload)
        row = {
            "frame_index": int(capture_result.get("frame_index", frame_index)),
            "capture_time": capture_result.get("capture_time", ""),
            "capture_kind": "facade_coarse_observation_v3",
            "scan_id": f"{house_id}_{facade}_coarse_observation",
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "planned_pose": planned_pose,
            "pose": capture_result.get("pose", {}),
            "commanded_pose": capture_result.get("commanded_pose", {}),
            "actual_pose": capture_result.get("actual_pose", {}),
            "pose_error": capture_result.get("pose_error", {}),
            "capture_dir": capture_result.get("capture_dir", ""),
            "rgb_path": capture_result.get("rgb_path", ""),
            "point_cloud_world_standard_m_npy_path": capture_result.get("point_cloud_world_standard_m_npy_path", ""),
            "point_cloud_world_standard_m_ply_path": capture_result.get("point_cloud_world_standard_m_ply_path", ""),
            "point_cloud_preview_path": capture_result.get("point_cloud_preview_path", ""),
            "point_count": int(capture_result.get("point_count", 0) or 0),
            "action_detail": action_detail,
        }
        self.append_jsonl(output_dir / "lidar_capture_log.jsonl", row)
        self.append_jsonl(facade_dir / "coarse_lidar_capture_log.jsonl", row)
        self.route2_write_lidar_summary(output_dir, running=True)
        self.route2_update_state(coarse_capture=capture_payload, coarse_rgb_path=capture_payload["coarse_rgb_path"])
        self.route2_write_state_artifact()
        return {"status": "ok", "capture": capture_payload, "row": row}

    def route3_obstacle_fallback_analysis(
        self,
        image_path: Optional[Path],
        planned_pose: Dict[str, float],
        reason: str = "",
    ) -> Dict[str, Any]:
        current_z = float(planned_pose.get("z", LLM_ROUTE2_OBSERVATION_Z_CM) or LLM_ROUTE2_OBSERVATION_Z_CM)
        green_ratio = 0.0
        center_green_ratio = 0.0
        try:
            if image_path is not None and image_path.exists():
                data = np.fromfile(str(image_path), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is not None and image.size:
                    h, w = image.shape[:2]
                    roi = image[int(h * 0.40): h, :]
                    center = roi[:, int(w * 0.30): int(w * 0.70)]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    hsv_center = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
                    green = (
                        (hsv[:, :, 0] >= 32)
                        & (hsv[:, :, 0] <= 92)
                        & (hsv[:, :, 1] >= 35)
                        & (hsv[:, :, 2] >= 35)
                    )
                    green_center = (
                        (hsv_center[:, :, 0] >= 32)
                        & (hsv_center[:, :, 0] <= 92)
                        & (hsv_center[:, :, 1] >= 35)
                        & (hsv_center[:, :, 2] >= 35)
                    )
                    green_ratio = float(np.mean(green)) if green.size else 0.0
                    center_green_ratio = float(np.mean(green_center)) if green_center.size else 0.0
        except Exception as exc:
            reason = reason or str(exc)
        obstacle = bool(green_ratio >= 0.38 or center_green_ratio >= 0.32)
        severity = "high" if center_green_ratio >= 0.48 or green_ratio >= 0.55 else ("medium" if obstacle else "none")
        recommended_z = current_z
        if obstacle:
            recommended_z = min(
                float(LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM),
                max(float(LLM_ROUTE2_OBSERVATION_Z_CM), current_z + float(LLM_ROUTE3_OBSTACLE_RAISE_STEP_CM)),
            )
        return {
            "foreground_obstacle_present": obstacle,
            "obstacle_type": "vegetation_or_foreground_occluder" if obstacle else "none",
            "severity": severity,
            "facade_visibility": "blocked" if severity == "high" else ("partially_blocked" if obstacle else "clear"),
            "recommend_raise": obstacle,
            "recommended_observation_z_cm": round(float(recommended_z), 2),
            "green_ratio_lower_view": round(float(green_ratio), 4),
            "green_ratio_lower_center": round(float(center_green_ratio), 4),
            "planner_source": "vision_heuristic_fallback",
            "reason": reason or ("foreground vegetation likely blocks the facade" if obstacle else "no strong foreground obstacle detected"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def route3_normalize_obstacle_analysis(
        self,
        parsed: Dict[str, Any],
        planned_pose: Dict[str, float],
        fallback_reason: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}
        fallback = self.route3_obstacle_fallback_analysis(None, planned_pose, fallback_reason)
        present_raw = parsed.get("foreground_obstacle_present", parsed.get("obstacle_visible", parsed.get("blocked")))
        if isinstance(present_raw, str):
            present = present_raw.strip().lower() in {"true", "yes", "y", "1", "blocked", "present"}
        elif present_raw is None:
            present = bool(fallback["foreground_obstacle_present"])
        else:
            present = bool(present_raw)
        severity = str(parsed.get("severity", fallback["severity"]) or "").strip().lower()
        if severity not in {"none", "low", "medium", "high"}:
            severity = "medium" if present else "none"
        visibility = str(parsed.get("facade_visibility", fallback["facade_visibility"]) or "").strip().lower()
        if visibility not in {"clear", "partially_blocked", "blocked", "unknown"}:
            visibility = "blocked" if severity == "high" else ("partially_blocked" if present else "clear")
        recommend_raise_raw = parsed.get("recommend_raise", parsed.get("raise_altitude", parsed.get("should_climb")))
        if isinstance(recommend_raise_raw, str):
            recommend_raise = recommend_raise_raw.strip().lower() in {"true", "yes", "y", "1", "raise", "climb"}
        elif recommend_raise_raw is None:
            recommend_raise = bool(present and severity in {"medium", "high"})
        else:
            recommend_raise = bool(recommend_raise_raw)
        current_z = float(planned_pose.get("z", LLM_ROUTE2_OBSERVATION_Z_CM) or LLM_ROUTE2_OBSERVATION_Z_CM)
        recommended_raw = self._as_float_or_none(parsed.get("recommended_observation_z_cm", parsed.get("recommended_z_cm")))
        if recommended_raw is None:
            recommended_z = current_z + float(LLM_ROUTE3_OBSTACLE_RAISE_STEP_CM) if recommend_raise else current_z
        else:
            recommended_z = float(recommended_raw)
        recommended_z = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), min(float(LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM), recommended_z))
        if recommend_raise and recommended_z <= current_z + 5.0:
            recommended_z = min(float(LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM), current_z + float(LLM_ROUTE3_OBSTACLE_RAISE_STEP_CM))
        return {
            "foreground_obstacle_present": present,
            "obstacle_type": str(parsed.get("obstacle_type", fallback["obstacle_type"]) or "unknown"),
            "severity": severity,
            "facade_visibility": visibility,
            "recommend_raise": bool(recommend_raise),
            "recommended_observation_z_cm": round(float(recommended_z), 2),
            "reason": str(parsed.get("reason", fallback["reason"]) or fallback["reason"]),
            "planner_source": "vlm_obstacle_check" if parsed else fallback["planner_source"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def route3_analyze_observation_obstacle(
        self,
        facade_dir: Path,
        *,
        house_id: str,
        facade: str,
        planned_pose: Dict[str, float],
    ) -> Dict[str, Any]:
        image_path = self.route2_current_rgb_path()
        response_payload: Dict[str, Any] = {}
        analysis: Dict[str, Any]
        try:
            if image_path is None:
                raise RuntimeError("missing coarse RGB image")
            image_b64 = self.route2_read_image_b64(image_path)
            if not image_b64:
                raise RuntimeError("missing coarse RGB image")
            if not self.effective_llm_api_key():
                raise RuntimeError("missing API key")
            context = {
                "house_id": house_id,
                "facade": facade,
                "planned_pose": planned_pose,
                "preferred_min_observation_z_cm": LLM_ROUTE2_OBSERVATION_Z_CM,
                "max_observation_z_cm": LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM,
            }
            response_payload = self.call_configured_llm_text(
                system_prompt=(
                    "You inspect a UAV forward RGB image before facade search. "
                    "Detect foreground occluders such as hedges, bushes, fences, walls, trees, cars, or awnings "
                    "that block the view of the house facade. Return compact JSON only."
                ),
                user_prompt=(
                    "Decide whether the UAV should climb before facade analysis. "
                    "Use recommend_raise=true when the lower or center facade is blocked by foreground objects. "
                    "For normal observation prefer at least 280cm height; if blocked, recommend a higher z, usually 400-550cm.\n"
                    "Return JSON keys: foreground_obstacle_present, obstacle_type, severity(none|low|medium|high), "
                    "facade_visibility(clear|partially_blocked|blocked), recommend_raise, recommended_observation_z_cm, reason.\n"
                    f"Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
                ),
                max_output_tokens=500,
                json_schema=LLM_ROUTE3_OBSERVATION_OBSTACLE_SCHEMA,
                image_b64=image_b64,
            )
            parsed = extract_json_object(str(response_payload.get("raw_text", "") or ""))
            analysis = self.route3_normalize_obstacle_analysis(parsed, planned_pose, "")
            response_payload["parsed"] = parsed
        except Exception as exc:
            LOGGER.warning("Route V3 observation obstacle fallback: %s", exc)
            analysis = self.route3_obstacle_fallback_analysis(image_path, planned_pose, str(exc))
            response_payload = {"error": str(exc), "fallback_used": True}
        self.write_json_artifact(facade_dir / "observation_obstacle_response.json", response_payload)
        self.write_json_artifact(facade_dir / "observation_obstacle_analysis.json", analysis)
        state = self.route2_selected_state()
        checks = list(state.get("observation_obstacle_checks", [])) if isinstance(state.get("observation_obstacle_checks"), list) else []
        checks.append(analysis)
        self.route2_update_state(observation_obstacle_analysis=analysis, observation_obstacle_checks=checks)
        self.route2_write_state_artifact()
        return analysis

    def route3_raise_observation_if_obstructed(
        self,
        session: flight.DroneFlightSession,
        *,
        output_dir: Path,
        facade_dir: Path,
        house_id: str,
        facade: str,
        obs_pose: Dict[str, float],
        obstacle_analysis: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not bool(obstacle_analysis.get("recommend_raise", False)):
            return obs_pose, {"status": "not_needed", "obstacle_analysis": obstacle_analysis}
        target_z = self._as_float_or_none(obstacle_analysis.get("recommended_observation_z_cm"))
        if target_z is None:
            return obs_pose, {"status": "not_needed", "reason": "missing_recommended_z", "obstacle_analysis": obstacle_analysis}
        target_z = max(float(LLM_ROUTE2_OBSERVATION_Z_CM), min(float(LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM), float(target_z)))
        if target_z <= float(obs_pose.get("z", 0.0)) + 5.0:
            return obs_pose, {"status": "not_needed", "reason": "target_z_not_higher", "obstacle_analysis": obstacle_analysis}
        raised_pose = dict(obs_pose)
        raised_pose["z"] = round(float(target_z), 2)
        self.route3_set_stage(
            "CHECK_OBS_OCCLUSION",
            output_dir=output_dir,
            facade=facade,
            target=raised_pose,
            message=f"{facade} foreground obstacle detected; climbing to {raised_pose['z']:.0f}cm",
        )
        nav_result = self.route3_navigate_to_pose_with_movement(
            session,
            raised_pose,
            output_dir=output_dir,
            stage="CHECK_OBS_OCCLUSION",
            facade=facade,
            target_id=f"{house_id}_{facade}_obs_raise",
            target_house_id=house_id,
        )
        self.route3_log_event(output_dir, "observation_raise_navigation", nav_result)
        if nav_result.get("status") != "ok":
            return obs_pose, {"status": "raise_failed", "navigation": nav_result, "obstacle_analysis": obstacle_analysis}
        state = self.route2_selected_state()
        observation = dict(state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {})
        observation.update({"z": raised_pose["z"], "observation_raise_reason": obstacle_analysis})
        self.route2_update_state(observation_point=observation)
        self.write_json_artifact(facade_dir / "facade_observation_point.json", observation)
        self.route2_write_state_artifact()
        return raised_pose, {"status": "raised", "navigation": nav_result, "obstacle_analysis": obstacle_analysis}

    def route3_plan_facade_scan_current(self) -> Dict[str, Any]:
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        if not analysis:
            analysis = self.route2_fallback_facade_analysis("Route V3 used fallback before VLM analysis.")
        points = self.route2_generate_facade_scan_points(analysis)
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        if output_dir is None or facade_dir is None:
            raise RuntimeError("missing facade output directory")
        next_hint = state.get("next_facade_hint", {}) if isinstance(state.get("next_facade_hint"), dict) else {}
        next_observation = next_hint.get("observation_point", {}) if isinstance(next_hint.get("observation_point"), dict) else {}
        points = self.route2_order_scan_points_continuously(points, next_observation_pose=next_observation)
        points = self.route2_assign_global_scan_ids(output_dir, house_id, facade, points)
        validation = self.scan_point_validation_report(house_id, points)
        search_plan = {
            "schema": "facade_v3_scan_plan",
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "observation_point": state.get("observation_point", {}),
            "next_facade_hint": next_hint,
            "facade_analysis": analysis,
            "scan_points": points,
            "scan_point_validation_report": validation,
            "route_blocked_by_safety": not bool(validation.get("valid", False)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(facade_dir / "facade_search_plan.json", search_plan)
        merged_points = self.route2_write_merged_scan_points(output_dir, house_id)
        self.route2_update_state(facade_analysis=analysis, facade_search_plan=search_plan, facade_scan_points=points, validation_report=validation)
        self.route2_write_state_artifact()
        return {"search_plan": search_plan, "points": points, "validation": validation, "merged_points": merged_points}

    def route3_capture_scan_point_current(
        self,
        session: flight.DroneFlightSession,
        *,
        output_dir: Path,
        facade_dir: Path,
        point: Dict[str, Any],
        planned_pose: Dict[str, float],
    ) -> Dict[str, Any]:
        capture_count = max(1, self.route_capture_count())
        capture_results: List[Dict[str, Any]] = []
        scan_id = str(point.get("scan_id", "") or "")
        for capture_idx in range(capture_count):
            action_detail = dict(self.build_stream_action_detail())
            action_detail.update(
                {
                    "source": "llm_route_v3_facade_scan",
                    "scan_id": scan_id,
                    "house_id": point.get("house_id"),
                    "facade": point.get("facade"),
                    "facade_id": point.get("facade_id"),
                    "global_scan_order": point.get("global_scan_order"),
                    "height_band": point.get("height_band"),
                    "floor_index": point.get("floor_index"),
                    "planned_pose": planned_pose,
                    "capture_index": capture_idx,
                }
            )
            frame_index = self.route2_next_frame_index(output_dir)
            capture_result = self.safe(
                "Route V3 scan capture",
                lambda idx=frame_index, action=action_detail: session.capture_lidar_stream_frame(
                    output_dir,
                    idx,
                    action_detail=action,
                ),
            )
            if isinstance(capture_result, dict):
                capture_results.append(capture_result)
        capture_status = "ok" if capture_results else "failed"
        rows: List[Dict[str, Any]] = []
        for result in capture_results:
            row = {
                "frame_index": int(result.get("frame_index", 0) or 0),
                "capture_time": result.get("capture_time", ""),
                "capture_kind": "facade_scan_v3",
                "scan_id": scan_id,
                "global_scan_order": point.get("global_scan_order"),
                "house_id": point.get("house_id"),
                "facade": point.get("facade"),
                "facade_id": point.get("facade_id"),
                "height_band": point.get("height_band"),
                "floor_index": point.get("floor_index"),
                "planned_pose": planned_pose,
                "pose": result.get("pose", {}),
                "commanded_pose": result.get("commanded_pose", {}),
                "actual_pose": result.get("actual_pose", {}),
                "pose_error": result.get("pose_error", {}),
                "capture_dir": result.get("capture_dir", ""),
                "rgb_path": result.get("rgb_path", ""),
                "point_cloud_world_standard_m_npy_path": result.get("point_cloud_world_standard_m_npy_path", ""),
                "point_cloud_world_standard_m_ply_path": result.get("point_cloud_world_standard_m_ply_path", ""),
                "point_cloud_preview_path": result.get("point_cloud_preview_path", ""),
                "point_count": int(result.get("point_count", 0) or 0),
                "action_detail": result.get("action_detail", {}),
            }
            rows.append(row)
            self.append_jsonl(output_dir / "lidar_capture_log.jsonl", row)
            self.append_jsonl(facade_dir / "lidar_capture_log.jsonl", row)
        execution_entry = {
            "scan_id": scan_id,
            "global_scan_order": point.get("global_scan_order"),
            "house_id": point.get("house_id"),
            "facade": point.get("facade"),
            "height_band": point.get("height_band"),
            "floor_index": point.get("floor_index"),
            "planned_pose": planned_pose,
            "capture_status": capture_status,
            "capture_count": len(capture_results),
            "frame_indices": [int(item.get("frame_index", 0) or 0) for item in capture_results],
            "capture_dirs": [str(item.get("capture_dir", "") or "") for item in capture_results],
            "point_count": sum(int(item.get("point_count", 0) or 0) for item in capture_results),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
        }
        self.append_jsonl(output_dir / "scan_execution_log.jsonl", execution_entry)
        self.append_jsonl(facade_dir / "scan_execution_log.jsonl", execution_entry)
        trajectory = list(self.route2_selected_state().get("facade_capture_rows", [])) if isinstance(self.route2_selected_state().get("facade_capture_rows"), list) else []
        trajectory.extend(rows)
        for state_point in self.llm_route2_state.get("facade_scan_points", []) if isinstance(self.llm_route2_state.get("facade_scan_points"), list) else []:
            if isinstance(state_point, dict) and str(state_point.get("scan_id", "") or "") == scan_id:
                state_point["status"] = "captured" if capture_status == "ok" else "capture_failed"
        self.route2_update_state(facade_capture_rows=trajectory)
        self.route2_write_state_artifact()
        self.route2_write_lidar_summary(output_dir, running=True)
        return {"status": capture_status, "rows": rows, "execution": execution_entry}

    def route3_decide_next_facade(
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
            and str(candidate.get("route3_navigation_status", "ok") or "ok") == "ok"
        ]
        fallback = {
            "next_action": "select_facade" if available else "done",
            "target_facade": str(available[0].get("facade", "") or "") if available else "",
            "reason": "fallback nearest feasible uncompleted facade",
            "rescan_required": False,
            "stop_condition_met": not bool(available),
            "planner_source": "rule_fallback",
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
            }
            response = self.call_configured_llm_text(
                system_prompt=(
                    "You are a high-level UAV house facade search supervisor. "
                    "Choose the next facade only; do not output low-level movement commands. "
                    "Return compact JSON."
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
                    parsed["planner_source"] = "llm_high_level_rank_corrected"
                    return parsed
                parsed["planner_source"] = "llm_high_level"
                return parsed
        except Exception as exc:
            LOGGER.warning("Route V3 high-level LLM decision fallback: %s", exc)
        return fallback

    def route3_initialize_run(self, target_house_id: str, *, force_new: bool = False) -> Path:
        current = self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {}
        current_target = str(current.get("target_house_id", "") or "")
        output_dir = self.route3_state_output_dir()
        if force_new or output_dir is None or current_target != str(target_house_id):
            output_dir = self.make_route3_autosearch_output_dir(target_house_id)
            self.llm_route3_completed_facades = set()
            self.llm_route3_blocked_facades = set()
            self.llm_route3_state = {
                "mode": "facade_autosearch_v3",
                "target_house_id": target_house_id,
                "output_dir": str(output_dir),
                "stage": "INIT_RUN",
                "completed_facades": [],
                "blocked_facades": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "nav_config": self.route3_nav_config(),
            }
        self.llm_route2_state = {
            "mode": "facade_by_facade_vlm_v2",
            "target_house_id": target_house_id,
            "output_dir": str(output_dir),
            "facade": "",
            "facade_id": "",
            "completed_facades": sorted(self.llm_route3_completed_facades),
        }
        self.llm_route2_completed_facades = set(self.llm_route3_completed_facades)
        self.route3_write_state_artifact()
        return output_dir

    def route3_run_summary(self, output_dir: Path, *, status: str) -> Dict[str, Any]:
        capture_rows = self.read_jsonl_artifact(output_dir / "lidar_capture_log.jsonl")
        execution_rows = self.read_jsonl_artifact(output_dir / "scan_execution_log.jsonl")
        attempted_facades = set(self.llm_route3_completed_facades) | set(self.llm_route3_blocked_facades)
        blocked_reasons = self.llm_route3_state.get("blocked_facade_reasons", {}) if isinstance(self.llm_route3_state.get("blocked_facade_reasons"), dict) else {}
        missing_facade_rgb: List[str] = []
        for facade in ("south", "east", "north", "west"):
            facade_dir = output_dir / "facade_observations" / self.route2_facade_id(str(self.llm_route3_state.get("target_house_id", "") or ""), facade)
            if facade in attempted_facades and not ((facade_dir / "coarse_rgb.png").exists() or (facade_dir / "coarse_rgb_panorama.png").exists()):
                missing_facade_rgb.append(facade)
        summary = {
            "schema": "facade_autosearch_v3_summary",
            "status": status,
            "target_house_id": str(self.llm_route3_state.get("target_house_id", "") or ""),
            "completed_facades": sorted(self.llm_route3_completed_facades),
            "blocked_facades": sorted(self.llm_route3_blocked_facades),
            "attempted_facades": sorted(attempted_facades),
            "blocked_facade_reasons": blocked_reasons,
            "missing_facade_rgb": missing_facade_rgb,
            "capture_count": len(capture_rows),
            "scan_execution_count": len(execution_rows),
            "output_dir": str(output_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(output_dir / "autosearch_summary.json", summary)
        return summary

    def route3_full_search_worker(self, session: flight.DroneFlightSession, *, single_facade: bool = False, force_new: bool = False) -> None:
        selected_target_house_id = self.selected_route_target_house_id()
        if not selected_target_house_id:
            self.root.after(0, lambda: self.llm_route3_status_var.set("LLM Route V3: select a target house first."))
            return
        self.route3_set_control_lock(True)
        target_house_id = selected_target_house_id
        output_dir: Path
        try:
            self.route3_set_stage("TASK_ANALYSIS", output_dir=None, message=f"analyzing task for selected house={selected_target_house_id}")
            task_plan = self.route3_analyze_task_plan(selected_target_house_id, output_dir=None)
            resolved_target = str(task_plan.get("target_house_id", "") or selected_target_house_id).strip()
            if resolved_target and resolved_target != target_house_id:
                target_house_id = resolved_target
            output_dir = self.route3_initialize_run(target_house_id, force_new=force_new)
            self.route3_apply_task_plan_state(task_plan)
            self.route3_update_state(output_dir=str(output_dir), target_house_id=target_house_id)
            self.write_json_artifact(output_dir / "route3_task_plan.json", {"plan": task_plan, "llm_response": {}})
            self.route3_log_event(output_dir, "task_plan_analysis", {"task_plan": task_plan, "selected_target_house_id": selected_target_house_id})
            self.route3_write_state_artifact()
        except Exception as exc:
            LOGGER.warning("Route V3 task analysis failed during start: %s", exc)
            output_dir = self.route3_initialize_run(target_house_id, force_new=force_new)
            self.route3_set_stage("TASK_ANALYSIS", output_dir=output_dir, message=f"task analysis fallback for house={target_house_id}")
            self.route3_ensure_task_plan(target_house_id, output_dir)
        self.route3_set_stage("PLAN_4_FACADES", output_dir=output_dir, message=f"planning facade candidates for house={target_house_id}")
        self.sync_capture_options_to_session(session)
        self.route3_enable_physics_movement(session)
        status = "running"
        try:
            while not self.llm_route3_stop_event.is_set():
                completed = set(self.llm_route3_completed_facades)
                blocked = set(self.llm_route3_blocked_facades)
                if len(completed | blocked) >= 4:
                    status = "done"
                    break
                raw_candidates = self.route2_all_facade_observation_candidates(target_house_id, skip_completed=False)
                ranked_candidates = self.route3_rank_observation_candidates(
                    target_house_id,
                    raw_candidates,
                    completed,
                    blocked,
                    start_pose=self.route3_current_pose(session),
                )
                feasible_candidates = [
                    item for item in ranked_candidates
                    if isinstance(item, dict)
                    and str(item.get("route3_navigation_status", "") or "") == "ok"
                    and str(item.get("status", "") or "") != "blocked"
                ]
                current_status = {
                    "stage": str(self.llm_route3_state.get("stage", "") or ""),
                    "completed_facades": sorted(completed),
                    "blocked_facades": sorted(blocked),
                    "feasible_facades": [str(item.get("facade", "") or "") for item in feasible_candidates],
                }
                next_status = {
                    "facade": str(feasible_candidates[0].get("facade", "") or "") if feasible_candidates else "",
                    "observation": feasible_candidates[0].get("selected_observation_attempt", {}) if feasible_candidates else {},
                    "navigation_cost_cm": feasible_candidates[0].get("route3_navigation_cost_cm") if feasible_candidates else None,
                }
                self.route3_update_state(
                    candidate_observation_points=raw_candidates,
                    ranked_facade_candidates=ranked_candidates,
                    current_exploration_status=current_status,
                    next_exploration_status=next_status,
                    completed_facades=sorted(completed),
                    blocked_facades=sorted(blocked),
                )
                self.route3_write_state_artifact()
                if not feasible_candidates:
                    newly_blocked = [
                        str(item.get("facade", "") or "")
                        for item in ranked_candidates
                        if str(item.get("facade", "") or "") not in completed and str(item.get("facade", "") or "") not in blocked
                    ]
                    blocked_reasons = dict(self.llm_route3_state.get("blocked_facade_reasons", {}) if isinstance(self.llm_route3_state.get("blocked_facade_reasons"), dict) else {})
                    for item in ranked_candidates:
                        facade_name = str(item.get("facade", "") or "")
                        if facade_name in newly_blocked:
                            self.llm_route3_blocked_facades.add(facade_name)
                            blocked_reasons[facade_name] = {
                                "reason": str(item.get("observation_block_reason", item.get("route3_navigation_reason", "no_feasible_observation")) or "no_feasible_observation"),
                                "observation_attempt_count": int(item.get("observation_attempt_count", 0) or 0),
                                "observation_attempts": item.get("observation_attempts", []),
                            }
                    self.route3_update_state(blocked_facades=sorted(self.llm_route3_blocked_facades), blocked_facade_reasons=blocked_reasons)
                    self.route3_write_state_artifact()
                    status = "done"
                    break
                decision = self.route3_decide_next_facade(target_house_id, ranked_candidates, completed, blocked)
                self.route3_log_event(output_dir, "high_level_decision", decision)
                facade = str(decision.get("target_facade", "") or "").strip().lower()
                if not facade or bool(decision.get("stop_condition_met", False)):
                    status = "done"
                    break
                selected = next((dict(item) for item in ranked_candidates if str(item.get("facade", "") or "") == facade), {})
                if not selected:
                    self.llm_route3_blocked_facades.add(facade)
                    continue
                self.route3_set_stage("SELECT_NEXT_FACADE", output_dir=output_dir, facade=facade, message=f"selected facade={facade}")
                self.llm_route2_state = {"target_house_id": target_house_id, "output_dir": str(output_dir)}
                facade_dir = self.route2_facade_dir(output_dir, target_house_id, facade)
                observation_attempts = self.route3_ordered_observation_attempts(selected)
                self.route3_update_state(
                    observation_attempts={**(self.llm_route3_state.get("observation_attempts", {}) if isinstance(self.llm_route3_state.get("observation_attempts"), dict) else {}), facade: observation_attempts},
                    selected_observation_attempt={},
                )
                observation: Dict[str, Any] = {}
                obs_pose: Dict[str, float] = {}
                nav_result: Dict[str, Any] = {}
                for attempt_index, attempt in enumerate(observation_attempts, start=1):
                    if self.llm_route3_stop_event.is_set():
                        break
                    attempt_status = str(attempt.get("status", "") or "")
                    if attempt_status == "blocked" and str(attempt.get("route3_navigation_status", "") or "") != "ok":
                        self.route3_log_event(
                            output_dir,
                            "observation_attempt_skipped",
                            {"facade": facade, "attempt_index": attempt_index, "attempt": attempt, "reason": attempt.get("observation_block_reason", "blocked_attempt")},
                        )
                        continue
                    self.apply_route2_observation_plan(target_house_id, attempt, ranked_candidates, status_label=f"v3 selected attempt {attempt_index}/{len(observation_attempts)}")
                    observation = self.route2_selected_state().get("observation_point", attempt)
                    obs_pose = self.route3_target_pose_from_point(observation)
                    self.route3_update_state(
                        selected_observation_attempt=observation,
                        current_exploration_status={
                            "stage": "NAV_TO_OBS",
                            "facade": facade,
                            "observation_attempt_index": attempt_index,
                            "observation_attempt_count": len(observation_attempts),
                            "target_pose": obs_pose,
                        },
                    )
                    self.route3_set_stage(
                        "NAV_TO_OBS",
                        output_dir=output_dir,
                        facade=facade,
                        target=obs_pose,
                        message=f"navigating to {facade} observation attempt {attempt_index}/{len(observation_attempts)}",
                    )
                    nav_result = self.route3_navigate_to_pose_with_movement(
                        session,
                        obs_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_OBS",
                        facade=facade,
                        target_id=f"{target_house_id}_{facade}_obs_attempt_{attempt_index}",
                        target_house_id=target_house_id,
                    )
                    self.route3_log_event(
                        output_dir,
                        "observation_attempt_navigation_result",
                        {"facade": facade, "attempt_index": attempt_index, "attempt": attempt, "navigation": nav_result},
                    )
                    if nav_result.get("status") == "ok":
                        break
                if nav_result.get("status") != "ok":
                    self.llm_route3_blocked_facades.add(facade)
                    blocked_reasons = dict(self.llm_route3_state.get("blocked_facade_reasons", {}) if isinstance(self.llm_route3_state.get("blocked_facade_reasons"), dict) else {})
                    blocked_reasons[facade] = {
                        "reason": str(nav_result.get("reason", "observation_navigation_failed") or "observation_navigation_failed"),
                        "observation_attempt_count": len(observation_attempts),
                        "observation_attempts": observation_attempts,
                        "last_navigation_result": nav_result,
                    }
                    self.route3_update_state(blocked_facades=sorted(self.llm_route3_blocked_facades), blocked_facade_reasons=blocked_reasons)
                    self.route3_write_state_artifact()
                    self.route3_set_stage("DECIDE_NEXT", output_dir=output_dir, facade=facade, error=nav_result, message=f"{facade} observation navigation blocked")
                    if single_facade:
                        break
                    continue
                self.route3_log_event(output_dir, "navigation_result", nav_result)
                if self.llm_route3_stop_event.is_set():
                    break
                self.route3_set_stage("CAPTURE_RGB", output_dir=output_dir, facade=facade, target=obs_pose, message=f"capturing {facade} RGB")
                rgb_result = self.route3_capture_facade_rgb_current(
                    session,
                    output_dir=output_dir,
                    facade_dir=facade_dir,
                    house_id=target_house_id,
                    facade=facade,
                    planned_pose=obs_pose,
                )
                self.route3_log_event(output_dir, "facade_rgb_capture", rgb_result)
                self.root.after(0, self.refresh_route3_support_views)
                if rgb_result.get("status") == "ok":
                    self.route3_set_stage("CHECK_OBS_OCCLUSION", output_dir=output_dir, facade=facade, target=obs_pose, message=f"checking {facade} foreground obstacle")
                    obstacle_analysis = self.route3_analyze_observation_obstacle(
                        facade_dir,
                        house_id=target_house_id,
                        facade=facade,
                        planned_pose=obs_pose,
                    )
                    self.route3_log_event(output_dir, "observation_obstacle_analysis", obstacle_analysis)
                    raised_pose, raise_result = self.route3_raise_observation_if_obstructed(
                        session,
                        output_dir=output_dir,
                        facade_dir=facade_dir,
                        house_id=target_house_id,
                        facade=facade,
                        obs_pose=obs_pose,
                        obstacle_analysis=obstacle_analysis,
                    )
                    self.route3_log_event(output_dir, "observation_raise_result", raise_result)
                    if raise_result.get("status") == "raised":
                        obs_pose = raised_pose
                        self.route3_set_stage("CAPTURE_RGB", output_dir=output_dir, facade=facade, target=obs_pose, message=f"recapturing {facade} RGB after climb")
                        rgb_result = self.route3_capture_facade_rgb_current(
                            session,
                            output_dir=output_dir,
                            facade_dir=facade_dir,
                            house_id=target_house_id,
                            facade=facade,
                            planned_pose=obs_pose,
                        )
                        self.route3_log_event(output_dir, "facade_rgb_recapture_after_raise", rgb_result)
                        if rgb_result.get("status") == "ok":
                            post_obstacle_analysis = self.route3_analyze_observation_obstacle(
                                facade_dir,
                                house_id=target_house_id,
                                facade=facade,
                                planned_pose=obs_pose,
                            )
                            self.route3_log_event(output_dir, "observation_obstacle_analysis_after_raise", post_obstacle_analysis)
                    self.root.after(0, self.refresh_route3_support_views)

                self.route3_set_stage("ANALYZE_VLM", output_dir=output_dir, facade=facade, message=f"analyzing {facade} facade")
                self.route2_analyze_facade_vlm_worker()
                self.root.after(0, self.refresh_route3_support_views)

                next_hint = self.route3_prepare_next_facade_hint(
                    target_house_id,
                    completed,
                    blocked,
                    exclude_facade=facade,
                    start_pose=self.route3_current_pose(session) or obs_pose,
                )
                self.route2_update_state(next_facade_hint=next_hint)
                self.route3_update_state(
                    next_facade_hint=next_hint,
                    next_exploration_status={
                        "facade": str(next_hint.get("target_facade", "") or ""),
                        "observation": next_hint.get("observation_point", {}),
                        "navigation_cost_cm": next_hint.get("navigation_cost_cm"),
                        "reason": next_hint.get("reason", ""),
                    },
                )
                self.route3_write_state_artifact()
                self.route3_log_event(output_dir, "next_facade_hint", next_hint)

                self.route3_set_stage("PLAN_SCAN", output_dir=output_dir, facade=facade, message=f"planning {facade} scan")
                plan_result = self.route3_plan_facade_scan_current()
                points = [point for point in plan_result.get("points", []) if isinstance(point, dict)]
                self.route3_log_event(output_dir, "facade_scan_plan", {"facade": facade, "point_count": len(points), "validation": plan_result.get("validation", {})})
                self.root.after(0, self.refresh_route3_support_views)
                if not points:
                    self.llm_route3_blocked_facades.add(facade)
                    self.route3_update_state(blocked_facades=sorted(self.llm_route3_blocked_facades))
                    self.route3_write_state_artifact()
                    self.route3_set_stage("DECIDE_NEXT", output_dir=output_dir, facade=facade, message=f"{facade} scan has no valid points")
                    if single_facade:
                        break
                    continue

                total = len(points)
                for idx, point in enumerate(points, start=1):
                    if self.llm_route3_stop_event.is_set():
                        break
                    scan_id = str(point.get("scan_id", "") or f"{facade}_{idx}")
                    target_pose = self.route3_target_pose_from_point(point)
                    self.route3_update_state(
                        current_exploration_status={
                            "stage": "NAV_TO_SCAN_POINT",
                            "facade": facade,
                            "point_index": idx,
                            "point_total": total,
                            "scan_id": scan_id,
                            "target_pose": target_pose,
                        }
                    )
                    self.route3_set_stage(
                        "NAV_TO_SCAN_POINT",
                        output_dir=output_dir,
                        facade=facade,
                        target=target_pose,
                        message=f"scan {idx}/{total} {scan_id}",
                    )
                    nav_scan = self.route3_navigate_to_pose_with_movement(
                        session,
                        target_pose,
                        output_dir=output_dir,
                        stage="NAV_TO_SCAN_POINT",
                        facade=facade,
                        target_id=scan_id,
                        target_house_id=target_house_id,
                    )
                    self.route3_log_event(output_dir, "scan_navigation_result", nav_scan)
                    if nav_scan.get("status") != "ok":
                        point["status"] = "blocked"
                        point["block_reason"] = nav_scan.get("reason", "navigation_failed")
                        continue
                    self.route3_set_stage("CAPTURE_SCAN", output_dir=output_dir, facade=facade, target=target_pose, message=f"capturing {scan_id}")
                    capture = self.route3_capture_scan_point_current(
                        session,
                        output_dir=output_dir,
                        facade_dir=facade_dir,
                        point=point,
                        planned_pose=target_pose,
                    )
                    self.route3_log_event(output_dir, "scan_capture", {"scan_id": scan_id, **capture})
                    progress = 100.0 * (len(completed) + (idx / max(1, total))) / 4.0
                    self.root.after(0, lambda v=progress: self.llm_route3_progress_var.set(max(0.0, min(100.0, v))))
                    self.root.after(0, lambda i=idx, t=total, f=facade: self.llm_route3_progress_text_var.set(f"Autonomy: {f} {i}/{t}"))
                    self.root.after(0, self.refresh_route3_support_views)

                self.route2_write_lidar_summary(output_dir, running=False)
                self.route3_set_stage("VALIDATE_FACADE", output_dir=output_dir, facade=facade, message=f"validating {facade}")
                validation = self.route2_validate_facade()
                self.route3_log_event(output_dir, "facade_validation", validation)
                self.llm_route3_completed_facades.add(facade)
                self.route3_update_state(completed_facades=sorted(self.llm_route3_completed_facades), blocked_facades=sorted(self.llm_route3_blocked_facades))
                self.route3_write_state_artifact()
                self.root.after(0, self.refresh_route3_support_views)
                if single_facade:
                    status = "single_facade_complete"
                    break
                self.route3_set_stage("DECIDE_NEXT", output_dir=output_dir, facade=facade, message=f"{facade} complete; deciding next facade")
            if self.llm_route3_stop_event.is_set():
                status = "stopped"
            if status == "done" and self.llm_route3_blocked_facades:
                status = "done_with_blocked"
            final_stage = "DONE" if status == "done" else ("DONE_WITH_BLOCKED" if status == "done_with_blocked" else "DECIDE_NEXT")
            self.route3_set_stage(final_stage, output_dir=output_dir, message=f"autosearch {status}")
            summary = self.route3_run_summary(output_dir, status=status)
            self.route3_log_event(output_dir, "summary", summary)
            self.root.after(0, lambda s=status, d=output_dir: self.llm_route3_status_var.set(f"LLM Route V3: {s} -> {d}"))
            self.root.after(0, self.refresh_route3_support_views)
        except Exception as exc:
            LOGGER.warning("Route V3 autosearch failed: %s", exc)
            self.route3_log_event(output_dir, "error", {"reason": str(exc)})
            self.route3_run_summary(output_dir, status="failed")
            self.root.after(0, lambda e=exc: self.llm_route3_status_var.set(f"LLM Route V3: failed: {e}"))
        finally:
            self.route3_set_control_lock(False)

    def refresh_route3_preview(self) -> None:
        preview_text = getattr(self, "llm_route3_preview_text", None)
        if preview_text is None:
            return
        payload = {
            "autonomy_state": self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {},
            "active_facade_state": self.llm_route2_state if isinstance(getattr(self, "llm_route2_state", None), dict) else {},
        }
        try:
            if not preview_text.winfo_exists():
                self.llm_route3_preview_text = None
                return
            preview_text.configure(state="normal")
            preview_text.delete("1.0", "end")
            preview_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            preview_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route3_preview_text = None

    def refresh_route3_analysis_view(self) -> None:
        analysis_text = getattr(self, "llm_route3_analysis_text", None)
        if analysis_text is None:
            return
        state = self.route2_selected_state()
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        payload = analysis if analysis else {"facade": state.get("facade", ""), "status": "No facade analysis yet."}
        try:
            if not analysis_text.winfo_exists():
                self.llm_route3_analysis_text = None
                return
            analysis_text.configure(state="normal")
            analysis_text.delete("1.0", "end")
            analysis_text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
            analysis_text.configure(state="disabled")
        except tk.TclError:
            self.llm_route3_analysis_text = None

    def refresh_route3_rgb_display(self) -> None:
        widget = getattr(self, "llm_route3_rgb_label", None)
        if widget is None:
            return
        image_path = self.route2_current_rgb_path()
        if image_path is None:
            try:
                self.route2_draw_rgb_preview_message(widget, "No facade RGB")
                self.llm_route3_rgb_photo = None
                self.llm_route2_rgb_status_var.set("Facade RGB: none")
            except tk.TclError:
                self.llm_route3_rgb_label = None
            return
        try:
            image = Image.open(image_path).convert("RGB")
            photo = ImageTk.PhotoImage(self.route2_rgb_preview_image(image, widget))
            self.route2_draw_rgb_preview_photo(widget, photo)
            self.llm_route3_rgb_photo = photo
            suffix = " (panorama used)" if image_path.name == "coarse_rgb_panorama.png" else ""
            self.llm_route2_rgb_status_var.set(f"Facade RGB: {image_path.name}{suffix}")
        except Exception as exc:
            LOGGER.warning("Refresh Route6_entrance_search v3 facade RGB failed: %s", exc)
            try:
                self.route2_draw_rgb_preview_message(widget, f"RGB load failed:\n{exc}")
                self.llm_route3_rgb_photo = None
                self.llm_route2_rgb_status_var.set("Facade RGB: load failed")
            except tk.TclError:
                self.llm_route3_rgb_label = None

    def refresh_llm_route3_map(self) -> None:
        widget = getattr(self, "llm_route3_map_widget", None)
        if widget is None:
            return
        try:
            if not self.load_map_resources(force=not bool(self.map_config)):
                self.llm_route3_map_status_var.set("Route V3 Map: map unavailable")
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
            route3_state = self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {}
            candidates_for_map = route3_state.get("ranked_facade_candidates", [])
            if not isinstance(candidates_for_map, list) or not candidates_for_map:
                candidates_for_map = state.get("candidate_observation_points", []) if isinstance(state.get("candidate_observation_points"), list) else []
            for candidate in candidates_for_map:
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
            output_dir = self.route3_state_output_dir()
            if output_dir is not None:
                trace_rows = self.read_jsonl_artifact(output_dir / "movement_trace.jsonl")
                trajectory = [row.get("current_pose", {}) for row in trace_rows[-400:] if isinstance(row.get("current_pose"), dict)]
                widget.set_trajectory(trajectory)
            self.llm_route3_map_status_var.set(
                f"Route V3 Map: houses={len(houses)} route_points={len(route_points)} completed={len(self.llm_route3_completed_facades)}/4"
            )
        except tk.TclError:
            pass
        except Exception as exc:
            LOGGER.warning("Refresh LLM Route6_entrance_search v3 map failed: %s", exc)
            self.llm_route3_map_status_var.set(f"Route V3 Map: failed: {exc}")

    def refresh_route3_support_views(self) -> None:
        completed = len(getattr(self, "llm_route3_completed_facades", set()) or set())
        blocked = len(getattr(self, "llm_route3_blocked_facades", set()) or set())
        progress = 100.0 * float(min(4, completed + blocked)) / 4.0
        state = self.llm_route3_state if isinstance(getattr(self, "llm_route3_state", None), dict) else {}
        current_status = state.get("current_exploration_status", {}) if isinstance(state.get("current_exploration_status"), dict) else {}
        next_status = state.get("next_exploration_status", {}) if isinstance(state.get("next_exploration_status"), dict) else {}
        try:
            self.llm_route3_progress_var.set(max(0.0, min(100.0, progress)))
            self.llm_route3_progress_text_var.set(f"Autonomy: completed={completed} blocked={blocked}")
            if current_status:
                facade = str(current_status.get("facade", state.get("current_facade", "")) or state.get("current_facade", "") or "-")
                stage = str(current_status.get("stage", state.get("stage", "")) or state.get("stage", "") or "-")
                point_index = current_status.get("point_index")
                point_total = current_status.get("point_total")
                suffix = f" {point_index}/{point_total}" if point_index is not None and point_total is not None else ""
                self.llm_route3_current_status_var.set(f"Current: {stage} {facade}{suffix}")
            else:
                self.llm_route3_current_status_var.set(f"Current: {state.get('stage', 'idle')} {state.get('current_facade', '')}")
            next_facade = str(next_status.get("facade", next_status.get("target_facade", "")) or "")
            obs = next_status.get("observation", next_status.get("observation_point", {})) if isinstance(next_status, dict) else {}
            if next_facade and isinstance(obs, dict):
                self.llm_route3_next_status_var.set(
                    f"Next: {next_facade} obs=({float(obs.get('x', 0.0)):.0f},{float(obs.get('y', 0.0)):.0f})"
                )
            else:
                self.llm_route3_next_status_var.set("Next: n/a")
            plan = state.get("task_plan", {}) if isinstance(state.get("task_plan"), dict) else {}
            if plan:
                self.route3_refresh_task_plan_labels(plan)
        except tk.TclError:
            pass
        self.refresh_route3_preview()
        self.refresh_route3_analysis_view()
        self.refresh_route3_rgb_display()
        self.refresh_llm_route3_map()

    def schedule_route3_auto_refresh(self) -> None:
        if not bool(self.llm_route3_auto_refresh_var.get()):
            self.cancel_route3_auto_refresh()
            return
        self.cancel_route3_auto_refresh()

        def tick() -> None:
            self.llm_route3_auto_refresh_job = None
            if not bool(self.llm_route3_auto_refresh_var.get()):
                return
            self.refresh_route3_support_views()
            try:
                self.llm_route3_auto_refresh_job = self.root.after(int(LLM_ROUTE3_AUTO_REFRESH_MS), tick)
            except tk.TclError:
                self.llm_route3_auto_refresh_job = None

        try:
            self.llm_route3_auto_refresh_job = self.root.after(int(LLM_ROUTE3_AUTO_REFRESH_MS), tick)
        except tk.TclError:
            self.llm_route3_auto_refresh_job = None

    def cancel_route3_auto_refresh(self) -> None:
        job = getattr(self, "llm_route3_auto_refresh_job", None)
        if job:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        self.llm_route3_auto_refresh_job = None

    def on_route3_auto_refresh_toggle(self) -> None:
        if bool(self.llm_route3_auto_refresh_var.get()):
            self.refresh_route3_support_views()
            self.schedule_route3_auto_refresh()
        else:
            self.cancel_route3_auto_refresh()

    def on_route3_analyze_task_plan(self) -> None:
        target_house_id = self.selected_route_target_house_id()
        if not target_house_id:
            self.llm_route3_status_var.set("LLM Route V3: select a target house first.")
            return
        self.llm_route3_status_var.set("LLM Route V3: analyzing task plan.")
        try:
            plan = self.route3_analyze_task_plan(target_house_id, output_dir=None)
            resolved_target = str(plan.get("target_house_id", "") or target_house_id).strip()
            output_dir = self.route3_state_output_dir()
            expected_prefix = f"house_{resolved_target or target_house_id}_"
            if output_dir is None or not output_dir.name.startswith(expected_prefix):
                output_dir = self.route3_initialize_run(resolved_target or target_house_id, force_new=False)
            self.route3_apply_task_plan_state(plan)
            self.route3_update_state(output_dir=str(output_dir), target_house_id=resolved_target or target_house_id)
            self.write_json_artifact(output_dir / "route3_task_plan.json", {"plan": plan, "llm_response": {}})
            self.route3_log_event(output_dir, "task_plan_analysis", {"task_plan": plan, "selected_target_house_id": target_house_id})
            self.route3_write_state_artifact()
            self.llm_route3_status_var.set(
                f"LLM Route V3: task plan targets={'->'.join(plan.get('target_sequence', []) or [plan.get('target_house_id', target_house_id)])}"
            )
            self.refresh_route3_support_views()
        except Exception as exc:
            LOGGER.warning("Route V3 task plan analyze failed: %s", exc)
            self.llm_route3_status_var.set(f"LLM Route V3: task plan failed: {exc}")

    def on_route3_start_full_search(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.llm_route3_thread is not None and self.llm_route3_thread.is_alive():
            self.llm_route3_status_var.set("LLM Route V3: already running.")
            return
        self.llm_route3_stop_event.clear()
        self.llm_route3_pause_event.clear()
        self.llm_route3_paused_var.set(False)
        self.llm_route3_thread = threading.Thread(
            target=lambda: self.route3_full_search_worker(session, single_facade=False, force_new=True),
            daemon=True,
        )
        self.llm_route3_thread.start()

    def on_route3_step_stage(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.llm_route3_thread is not None and self.llm_route3_thread.is_alive():
            self.llm_route3_status_var.set("LLM Route V3: wait for current worker.")
            return
        self.llm_route3_stop_event.clear()
        self.llm_route3_pause_event.clear()
        self.llm_route3_thread = threading.Thread(
            target=lambda: self.route3_full_search_worker(session, single_facade=True, force_new=False),
            daemon=True,
        )
        self.llm_route3_thread.start()

    def on_route3_toggle_pause(self) -> None:
        if self.llm_route3_pause_event.is_set():
            self.llm_route3_pause_event.clear()
            self.llm_route3_paused_var.set(False)
            self.llm_route3_status_var.set("LLM Route V3: resumed.")
        else:
            self.llm_route3_pause_event.set()
            self.llm_route3_paused_var.set(True)
            session = self.session
            if session is not None and session.started:
                self.route3_hold(session, output_dir=self.route3_state_output_dir(), reason="pause_button")
            self.llm_route3_status_var.set("LLM Route V3: paused.")

    def on_route3_stop(self) -> None:
        self.llm_route3_stop_event.set()
        self.route_stop_event.set()
        self.llm_route3_pause_event.clear()
        session = self.session
        if session is not None and session.started:
            self.route3_hold(session, output_dir=self.route3_state_output_dir(), reason="stop_button")
        self.llm_route3_status_var.set("LLM Route V3: stop requested.")

    def on_route3_clear(self) -> None:
        if self.llm_route3_thread is not None and self.llm_route3_thread.is_alive():
            self.llm_route3_status_var.set("LLM Route V3: stop before clearing.")
            return
        self.llm_route3_state = {}
        self.llm_route3_completed_facades = set()
        self.llm_route3_blocked_facades = set()
        self.llm_route3_stage_var.set("Stage: idle")
        self.llm_route3_active_var.set("Active: n/a")
        self.llm_route3_target_var.set("Target: n/a")
        self.llm_route3_error_var.set("Error: n/a")
        self.llm_route3_payload_var.set("Payload: hold")
        self.llm_route3_task_status_var.set("Task Plan: n/a")
        self.llm_route3_target_sequence_var.set("Targets: n/a")
        self.llm_route3_current_status_var.set("Current: idle")
        self.llm_route3_next_status_var.set("Next: n/a")
        self.llm_route3_status_var.set("LLM Route V3: cleared.")
        self.refresh_route3_support_views()

    def on_route3_validate_run(self) -> None:
        output_dir = self.route3_state_output_dir()
        if output_dir is None:
            self.llm_route3_status_var.set("LLM Route V3: no run to validate.")
            return
        summary = self.route3_run_summary(output_dir, status=str(self.llm_route3_state.get("stage", "manual_validate") or "manual_validate"))
        self.llm_route3_status_var.set(f"LLM Route V3: run summary -> {output_dir / 'autosearch_summary.json'}")
        self.route3_log_event(output_dir, "manual_validate_run", summary)
        self.refresh_route3_support_views()

    def on_llm_task_analyze(self) -> None:
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route_status_var.set("LLM Route: wait for current Route6_entrance_search worker.")
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
                LOGGER.warning("LLM Route6_entrance_search plan failed: %s", exc)
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

    def on_route_lidar_capture(self) -> None:
        session = self.active_session()
        if session is None:
            return
        if self.route_thread is not None and self.route_thread.is_alive():
            self.llm_route_status_var.set("Route Capture: worker already running.")
            return
        target_house_id = self.selected_route_target_house_id() or "001"
        if not target_house_id:
            self.llm_route_status_var.set("Route Capture: no target house selected.")
            return
        if str(self.llm_route_plan.get("target_house_id", "") or "") != target_house_id or not self.llm_route_scan_points:
            try:
                plan = self.fallback_route_plan(target_house_id)
                plan["planner_source"] = "route_capture_lidar_fallback"
                self.apply_route_plan(plan, status_prefix="Route Capture Setup")
            except Exception as exc:
                self.llm_route_status_var.set(f"Route Capture setup failed: {exc}")
                return
        scan_validation = self.scan_point_validation_report(target_house_id, self.llm_route_scan_points)
        if not bool(scan_validation.get("valid", False)):
            self.llm_route_status_var.set("Route Capture: scan point validation failed.")
            self.llm_route_validation_report = {"scan_point_validation_report": scan_validation, "overall_passed": False}
            self.refresh_route_preview()
            return
        output_dir = self.make_route_capture_output_dir(target_house_id)
        self.prepare_active_plan_for_route_capture(target_house_id, output_dir)
        self.sync_capture_options_to_session(session)
        movement_state = self.safe("Route capture enable movement", lambda: session.set_movement_enabled(True))
        if isinstance(movement_state, dict):
            self.root.after(0, lambda r=movement_state: self.apply_state(r))
        self.route_stop_event.clear()
        self.route_thread = threading.Thread(
            target=lambda: self.route_lidar_capture_worker(session, target_house_id, output_dir),
            daemon=True,
        )
        self.route_thread.start()

    def route_capture_target_point(self, target_house_id: str, x: float, y: float) -> Tuple[float, float]:
        bbox = self.house_world_bbox_for_id(target_house_id)
        if not bbox:
            return float(x), float(y)
        try:
            target_x = min(max(float(x), float(bbox["min_x"])), float(bbox["max_x"]))
            target_y = min(max(float(y), float(bbox["min_y"])), float(bbox["max_y"]))
            if math.hypot(target_x - float(x), target_y - float(y)) <= 1e-6:
                target_x = float(bbox.get("center_x", target_x))
                target_y = float(bbox.get("center_y", target_y))
            return target_x, target_y
        except Exception:
            return float(x), float(y)

    def route_capture_desired_yaw_deg(self, target_house_id: str, x: float, y: float) -> float:
        target_x, target_y = self.route_capture_target_point(target_house_id, x, y)
        return math.degrees(math.atan2(float(target_y) - float(y), float(target_x) - float(x)))

    def route_capture_align_yaw_with_qe(
        self,
        session: flight.DroneFlightSession,
        target_house_id: str,
        point: Dict[str, Any],
    ) -> Dict[str, Any]:
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        desired_yaw = self.route_capture_desired_yaw_deg(target_house_id, x, y)
        yaw_steps: List[Dict[str, Any]] = []
        tolerance = float(LLM_ROUTE_PATH_CAPTURE_YAW_TOLERANCE_DEG)
        max_steps = 12
        final_yaw: Optional[float] = None
        for _step in range(max_steps):
            state = self.safe("Route capture yaw state", session.get_state)
            pose = state.get("pose", {}) if isinstance(state, dict) and isinstance(state.get("pose"), dict) else {}
            yaw = self._as_float_or_none(pose.get("task_yaw", pose.get("yaw")))
            if yaw is None:
                break
            final_yaw = float(yaw)
            delta = self._normalize_angle_deg(desired_yaw - float(yaw))
            if abs(delta) <= tolerance:
                break
            key = "e" if delta > 0.0 else "q"
            yaw_step = min(float(LLM_ROUTE_PATH_CAPTURE_YAW_STEP_DEG), abs(float(delta)))
            payload = dict(MOVE_COMMANDS[key])
            payload["yaw_delta_deg"] = yaw_step if key == "e" else -yaw_step
            payload["action_name"] = f"route_capture_{key}"
            response = self.safe("Route capture q/e yaw", lambda p=payload: session.move_relative(p))
            if isinstance(response, dict):
                response_pose = response.get("pose", {}) if isinstance(response.get("pose"), dict) else {}
                response_yaw = self._as_float_or_none(response_pose.get("task_yaw", response_pose.get("yaw")))
                if response_yaw is not None:
                    final_yaw = float(response_yaw)
                self.root.after(0, lambda r=response: self.apply_state(r))
            yaw_steps.append(
                {
                    "key": key,
                    "yaw_delta_deg": round(float(payload["yaw_delta_deg"]), 3),
                    "remaining_error_before_deg": round(float(delta), 3),
                }
            )
            if not isinstance(response, dict) or self.route_stop_event.is_set():
                break
        final_error = self._normalize_angle_deg(desired_yaw - final_yaw) if final_yaw is not None else None
        return {
            "desired_yaw_deg": round(float(desired_yaw), 3),
            "final_yaw_deg": round(float(final_yaw), 3) if final_yaw is not None else None,
            "final_error_deg": round(float(final_error), 3) if final_error is not None else None,
            "yaw_key_sequence": [step["key"] for step in yaw_steps],
            "yaw_steps": yaw_steps,
            "tolerance_deg": tolerance,
        }

    def capture_route_lidar_at_point(
        self,
        session: flight.DroneFlightSession,
        output_dir: Path,
        target_house_id: str,
        point: Dict[str, Any],
        *,
        point_index: int,
        total: int,
    ) -> Dict[str, Any]:
        scan_id = str(point.get("scan_id", "") or point.get("label", f"route_capture_{point_index:03d}"))
        capture_count = self.route_capture_count()
        interval_s = self.route_capture_interval_s()
        capture_results: List[Dict[str, Any]] = []
        yaw_result: Dict[str, Any] = {}
        for capture_index in range(capture_count):
            if self.route_stop_event.is_set():
                break
            yaw_result = self.route_capture_align_yaw_with_qe(session, target_house_id, point)
            frame_index = len(self.llm_route_lidar_trajectory) + 1
            action_detail = dict(self.build_stream_action_detail())
            action_detail.update(
                {
                    "source": "route_capture_lidar",
                    "route_capture_mode": "scan_point_path_lidar_qe_yaw",
                    "scan_id": scan_id,
                    "house_id": target_house_id,
                    "facade": str(point.get("facade", "") or ""),
                    "capture_index": capture_index + 1,
                    "capture_count": capture_count,
                    "capture_interval_s": interval_s,
                    "planned_pose": {
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                        "z": float(point.get("z", self.target_house_capture_altitude_cm(target_house_id))),
                    },
                    "yaw_alignment": yaw_result,
                }
            )
            result = self.safe(
                "Route capture lidar frame",
                lambda idx=frame_index, action=action_detail: session.capture_lidar_stream_frame(
                    output_dir,
                    idx,
                    action_detail=action,
                ),
            )
            if isinstance(result, dict):
                capture_results.append(result)
                trajectory_entry = {
                    "frame_index": int(result.get("frame_index", frame_index)),
                    "capture_time": result.get("capture_time", ""),
                    "scan_id": scan_id,
                    "house_id": target_house_id,
                    "facade": str(point.get("facade", "") or ""),
                    "route_capture_mode": "scan_point_path_lidar_qe_yaw",
                    "capture_interval_s": interval_s,
                    "point_index": point_index,
                    "point_total": total,
                    "planned_pose": action_detail["planned_pose"],
                    "yaw_alignment": yaw_result,
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
                self.append_jsonl(output_dir / "lidar_capture_log.jsonl", trajectory_entry)
                self.append_jsonl(output_dir / "route_capture_lidar_log.jsonl", trajectory_entry)
                self.write_house_search_lidar_summary(output_dir, running=True)
                self.root.after(
                    0,
                    lambda sid=scan_id, c=len(self.llm_route_lidar_trajectory): self.llm_route_status_var.set(
                        f"Route Capture: captured {sid} frames={c}"
                    ),
                )
            if capture_index + 1 < capture_count and self.route_stop_event.wait(interval_s):
                break
        state = self.safe("Route capture final state", session.get_state)
        actual_pose = state.get("pose", {}) if isinstance(state, dict) and isinstance(state.get("pose"), dict) else {}
        capture_status = "ok" if capture_results else "failed"
        execution_entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "house_id": target_house_id,
            "scan_id": scan_id,
            "facade": str(point.get("facade", "") or ""),
            "route_capture_mode": "scan_point_path_lidar_qe_yaw",
            "capture_interval_s": interval_s,
            "planned_pose": {
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "z": float(point.get("z", self.target_house_capture_altitude_cm(target_house_id))),
            },
            "actual_pose": actual_pose,
            "yaw_alignment": yaw_result,
            "safety_state": "SAFE",
            "capture_status": capture_status,
            "capture_count": len(capture_results),
            "capture_dirs": [str(item.get("capture_dir", "") or "") for item in capture_results],
            "point_cloud_paths": [
                str(item.get("point_cloud_world_standard_m_ply_path", "") or item.get("point_cloud_world_ply_path", "") or "")
                for item in capture_results
            ],
        }
        self.append_jsonl(output_dir / "scan_execution_log.jsonl", execution_entry)
        self.update_route_point_runtime_status(point, "captured" if capture_status == "ok" else "capture_failed")
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
        self.write_json_artifact(output_dir / "execution_summary.json", self.llm_route_execution_summary)
        self.root.after(0, self.refresh_route_preview)
        return execution_entry

    def route_lidar_capture_worker(
        self,
        session: flight.DroneFlightSession,
        target_house_id: str,
        output_dir: Path,
    ) -> None:
        scan_points = self.scan_points_as_route_points(self.llm_route_scan_points)
        route_points = self.llm_route_plan.get("route_points", []) if isinstance(self.llm_route_plan.get("route_points"), list) else []
        index_by_scan_id = {
            str(point.get("scan_id", "") or ""): idx
            for idx, point in enumerate(route_points)
            if isinstance(point, dict) and str(point.get("scan_id", "") or "")
        }
        total = len(scan_points)
        interval_s = self.route_capture_interval_s()
        captured_any = False
        self.root.after(
            0,
            lambda: self.llm_route_status_var.set(
                f"Route Capture: starting house={target_house_id} points={total} interval={interval_s:.2f}s"
            ),
        )
        for point_index, point in enumerate(scan_points, start=1):
            if self.route_stop_event.is_set():
                self.root.after(0, lambda: self.llm_route_status_var.set("Route Capture: stopped."))
                return
            scan_id = str(point.get("scan_id", "") or "")
            if scan_id in index_by_scan_id:
                point["_route_point_index"] = index_by_scan_id[scan_id]
            target_pose = {
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "z": float(point.get("z", self.target_house_capture_altitude_cm(target_house_id))),
            }
            self.root.after(
                0,
                lambda i=point_index, t=total, sid=scan_id: self.llm_route_status_var.set(
                    f"Route Capture: move/capture {i}/{t} {sid}"
                ),
            )
            response = self.safe("Route capture set point", lambda p=target_pose: session.set_pose(p))
            if isinstance(response, dict):
                self.root.after(0, lambda r=response: self.apply_state(r))
            else:
                self.update_route_point_runtime_status(point, "capture_failed")
                continue
            capture_entry = self.capture_route_lidar_at_point(
                session,
                output_dir,
                target_house_id,
                point,
                point_index=point_index,
                total=total,
            )
            captured_any = captured_any or capture_entry.get("capture_status") == "ok"
            self.root.after(0, self.refresh_llm_route_map)
            if point_index < total and self.route_stop_event.wait(interval_s):
                self.root.after(0, lambda: self.llm_route_status_var.set("Route Capture: stopped."))
                return
        if captured_any:
            try:
                finalize_result = self.finalize_house_search_outputs()
                validation = self.validate_house_search_data(finalize_result=finalize_result)
                self.root.after(
                    0,
                    lambda v=validation: self.llm_route_status_var.set(
                        f"Route Capture: complete, validation={'PASS' if v.get('overall_passed') else 'CHECK'} -> {output_dir}"
                    ),
                )
            except Exception as exc:
                LOGGER.warning("Route capture finalize/validate failed: %s", exc)
                self.root.after(0, lambda e=exc: self.llm_route_status_var.set(f"Route Capture: finalize failed: {e}"))
        else:
            self.root.after(0, lambda: self.llm_route_status_var.set("Route Capture: no captures completed."))

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
            self.llm_route_status_var.set("LLM Route: blocked by Route6_entrance_search/scan safety validation.")
            self.refresh_route_preview()
            return
        self.sync_capture_options_to_session(session)
        points = self.route_points_to_follow(auto=auto)
        if not points:
            self.llm_route_status_var.set("LLM Route: no Route6_entrance_search point to follow.")
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
