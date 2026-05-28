from __future__ import annotations

from PIL import ImageDraw

from .common import *
from obstacle_avoidance.collect_route_episodes import distance_3d_cm


class Route7PlanningMixin:
    def ensure_route7_state(self) -> None:
        self.ensure_route5_state()
        try:
            if not str(self.llm_route7_map_layer_var.get() or "").strip():
                self.llm_route7_map_layer_var.set("z_300")
        except Exception:
            pass

    def route7_default_exploration_z_cm(self) -> float:
        return 300.0

    def route7_default_layer_key(self) -> str:
        return f"z_{int(round(self.route7_default_exploration_z_cm())):03d}"

    def route7_normalize_house_bbox(self, bbox: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(bbox, dict) or not bbox:
            return {}
        try:
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            points = bbox.get("points", []) if isinstance(bbox.get("points", []), list) else []
            xs: List[float] = []
            ys: List[float] = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    xs.append(float(point.get("x", 0.0) or 0.0))
                    ys.append(float(point.get("y", 0.0) or 0.0))
                except Exception:
                    continue
            if not xs or not ys:
                return {}
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        low_x, high_x = sorted((float(min_x), float(max_x)))
        low_y, high_y = sorted((float(min_y), float(max_y)))
        normalized = dict(bbox)
        normalized.update(
            {
                "min_x": low_x,
                "max_x": high_x,
                "min_y": low_y,
                "max_y": high_y,
                "center_x": float(bbox.get("center_x", 0.5 * (low_x + high_x)) or 0.5 * (low_x + high_x)),
                "center_y": float(bbox.get("center_y", 0.5 * (low_y + high_y)) or 0.5 * (low_y + high_y)),
            }
        )
        return normalized

    def route7_static_house_bbox_for_id(self, house_id: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        hid = str(house_id or "").strip()
        if not hid:
            return {}
        bases: List[Dict[str, Any]] = []
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        state_base = state.get("route7_static_house_base", {}) if isinstance(state.get("route7_static_house_base", {}), dict) else {}
        if state_base:
            bases.append(state_base)
        if output_dir is not None:
            try:
                path = Path(output_dir) / "map" / "route7_static_house_base.json"
                payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                if isinstance(payload, dict) and payload:
                    bases.append(payload)
            except Exception:
                pass
        static_base_fn = getattr(self, "route7_static_house_base", None)
        if callable(static_base_fn):
            try:
                payload = static_base_fn()
                if isinstance(payload, dict) and payload:
                    bases.append(payload)
            except Exception:
                pass
        for base in bases:
            houses = base.get("houses", []) if isinstance(base.get("houses", []), list) else []
            for house in houses:
                if not isinstance(house, dict):
                    continue
                record_id = str(house.get("house_id", house.get("id", "")) or "").strip()
                if record_id != hid:
                    continue
                raw_bbox = house.get("bbox_world", {}) if isinstance(house.get("bbox_world", {}), dict) else {}
                if not raw_bbox:
                    raw_bbox = house.get("bbox", {}) if isinstance(house.get("bbox", {}), dict) else {}
                bbox = self.route7_normalize_house_bbox(raw_bbox)
                if not bbox:
                    continue
                bbox["route7_house_bbox_source"] = "route7_static_house_base"
                bbox["route7_static_house_base_source"] = str(base.get("source", "") or "")
                bbox["route7_static_house_label"] = str(house.get("label", house.get("name", "")) or "")
                return bbox
        return {}

    def route7_house_bbox_for_id(self, house_id: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        bbox = self.route7_static_house_bbox_for_id(house_id, output_dir=output_dir)
        if bbox:
            return bbox
        try:
            fallback = self.house_world_bbox_for_id(str(house_id or "")) if callable(getattr(self, "house_world_bbox_for_id", None)) else {}
        except Exception:
            fallback = {}
        bbox = self.route7_normalize_house_bbox(fallback)
        if bbox:
            bbox["route7_house_bbox_source"] = str(bbox.get("source", "") or "house_world_bbox_for_id")
        return bbox

    def route7_observation_facade_corridor_report(
        self,
        pose: Dict[str, Any],
        bbox: Dict[str, Any],
        facade: str,
        *,
        axis_margin_cm: float = 250.0,
        side_margin_cm: float = 150.0,
        expected_standoff_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        facade_name = str(facade or "").strip().lower()
        pose = pose if isinstance(pose, dict) else {}
        bbox = self.route7_normalize_house_bbox(bbox)
        if facade_name not in {"south", "east", "north", "west"} or not bbox:
            return {
                "same_facade_corridor": True,
                "enforced": False,
                "reason": "facade_bbox_unavailable",
                "facade": facade_name,
                "pose": self.route5_json_safe(pose),
            }
        try:
            x = float(pose.get("x", 0.0) or 0.0)
            y = float(pose.get("y", 0.0) or 0.0)
            min_x = float(bbox["min_x"])
            max_x = float(bbox["max_x"])
            min_y = float(bbox["min_y"])
            max_y = float(bbox["max_y"])
        except Exception:
            return {
                "same_facade_corridor": True,
                "enforced": False,
                "reason": "pose_or_bbox_invalid",
                "facade": facade_name,
                "pose": self.route5_json_safe(pose),
                "bbox": self.route5_json_safe(bbox),
            }
        axis_margin = max(0.0, float(axis_margin_cm))
        side_margin = max(0.0, float(side_margin_cm))
        if facade_name == "east":
            side_ok = x >= max_x - side_margin
            axis_ok = min_y - axis_margin <= y <= max_y + axis_margin
            side_distance_cm = x - max_x
            axis_value = y
            axis_min, axis_max = min_y, max_y
        elif facade_name == "west":
            side_ok = x <= min_x + side_margin
            axis_ok = min_y - axis_margin <= y <= max_y + axis_margin
            side_distance_cm = min_x - x
            axis_value = y
            axis_min, axis_max = min_y, max_y
        elif facade_name == "north":
            side_ok = y >= max_y - side_margin
            axis_ok = min_x - axis_margin <= x <= max_x + axis_margin
            side_distance_cm = y - max_y
            axis_value = x
            axis_min, axis_max = min_x, max_x
        else:
            side_ok = y <= min_y + side_margin
            axis_ok = min_x - axis_margin <= x <= max_x + axis_margin
            side_distance_cm = min_y - y
            axis_value = x
            axis_min, axis_max = min_x, max_x
        standoff_ok = True
        max_side_distance_cm: Optional[float] = None
        if expected_standoff_cm is not None:
            expected = max(0.0, float(expected_standoff_cm))
            max_side_distance_cm = expected + max(300.0, expected * 0.5)
            standoff_ok = side_distance_cm <= max_side_distance_cm
        same = bool(side_ok and axis_ok and standoff_ok)
        if not axis_ok:
            reason = "axis_outside_target_house_bounds"
        elif not side_ok:
            reason = "wrong_target_facade_side"
        elif not standoff_ok:
            reason = "outside_expected_standoff_band"
        else:
            reason = "ok"
        return {
            "same_facade_corridor": same,
            "enforced": True,
            "reason": reason,
            "facade": facade_name,
            "pose": self.route5_json_safe(pose),
            "bbox": self.route5_json_safe(bbox),
            "side_ok": bool(side_ok),
            "axis_ok": bool(axis_ok),
            "standoff_ok": bool(standoff_ok),
            "side_distance_cm": round(float(side_distance_cm), 3),
            "axis_value_cm": round(float(axis_value), 3),
            "axis_min_cm": round(float(axis_min), 3),
            "axis_max_cm": round(float(axis_max), 3),
            "axis_margin_cm": float(axis_margin),
            "side_margin_cm": float(side_margin),
            "expected_standoff_cm": None if expected_standoff_cm is None else round(float(expected_standoff_cm), 3),
            "max_side_distance_cm": None if max_side_distance_cm is None else round(float(max_side_distance_cm), 3),
        }

    def route7_static_house_base_snapshot(self, target_house_id: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        target_id = str(target_house_id or "").strip()
        try:
            if callable(getattr(self, "load_map_resources", None)):
                self.load_map_resources(force=not bool(getattr(self, "map_config", {})))
        except Exception:
            pass
        houses_raw = []
        known_records_fn = getattr(self, "route6_known_house_polygon_records", None)
        known_records = known_records_fn() if callable(known_records_fn) else []
        if isinstance(known_records, list) and known_records:
            for record in known_records[:5]:
                if not isinstance(record, dict):
                    continue
                house_id = str(record.get("house_id", record.get("id", "")) or "").strip()
                bbox = record.get("bbox", {}) if isinstance(record.get("bbox", {}), dict) else {}
                try:
                    min_x = float(bbox["min_x"])
                    max_x = float(bbox["max_x"])
                    min_y = float(bbox["min_y"])
                    max_y = float(bbox["max_y"])
                except Exception:
                    points = record.get("points", []) if isinstance(record.get("points", []), list) else []
                    xs: List[float] = []
                    ys: List[float] = []
                    for point in points:
                        if not isinstance(point, dict):
                            continue
                        try:
                            xs.append(float(point.get("x", 0.0) or 0.0))
                            ys.append(float(point.get("y", 0.0) or 0.0))
                        except Exception:
                            continue
                    if not xs or not ys:
                        continue
                    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
                bbox_world = {
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "center_x": 0.5 * (min_x + max_x),
                    "center_y": 0.5 * (min_y + max_y),
                    "source": "route6_operator_known_coordinates",
                }
                houses_raw.append(
                    {
                        "house_id": house_id,
                        "label": str(record.get("name", f"H{house_id}") or f"H{house_id}"),
                        "bbox_world": self.route5_json_safe(bbox_world),
                        "is_target_house": bool(house_id == target_id),
                    }
                )
        if houses_raw:
            snapshot = {
                "schema": "route7_static_house_base_v1",
                "target_house_id": target_id,
                "default_layer_key": self.route7_default_layer_key(),
                "default_z_cm": self.route7_default_exploration_z_cm(),
                "houses": self.route5_json_safe(houses_raw),
                "house_count": len(houses_raw),
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                "source": "route6_operator_known_house_polygons_frozen_at_route7_start",
            }
            if output_dir is not None:
                try:
                    self.write_json_artifact(Path(output_dir) / "map" / "route7_static_house_base.json", snapshot)
                except Exception:
                    pass
            return snapshot
        map_config = getattr(self, "map_config", {}) if isinstance(getattr(self, "map_config", {}), dict) else {}
        for house in (map_config.get("houses", []) if isinstance(map_config.get("houses", []), list) else [])[:5]:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("id", "") or "").strip()
            if not house_id:
                continue
            try:
                bbox = self.house_world_bbox_for_id(house_id) if callable(getattr(self, "house_world_bbox_for_id", None)) else {}
            except Exception:
                bbox = {}
            if not bbox:
                continue
            houses_raw.append(
                {
                    "house_id": house_id,
                    "label": str(house.get("label", house.get("name", f"H{house_id}")) or f"H{house_id}"),
                    "bbox_world": self.route5_json_safe(bbox),
                    "is_target_house": bool(house_id == target_id),
                }
            )
        if target_id and not any(str(item.get("house_id", "")) == target_id for item in houses_raw):
            try:
                bbox = self.house_world_bbox_for_id(target_id) if callable(getattr(self, "house_world_bbox_for_id", None)) else {}
            except Exception:
                bbox = {}
            if bbox:
                houses_raw.append(
                    {
                        "house_id": target_id,
                        "label": f"H{target_id}",
                        "bbox_world": self.route5_json_safe(bbox),
                        "is_target_house": True,
                    }
                )
        snapshot = {
            "schema": "route7_static_house_base_v1",
            "target_house_id": target_id,
            "default_layer_key": self.route7_default_layer_key(),
            "default_z_cm": self.route7_default_exploration_z_cm(),
            "houses": self.route5_json_safe(houses_raw),
            "house_count": len(houses_raw),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "source": "map_config_house_world_bbox_frozen_at_route7_start",
        }
        if output_dir is not None:
            try:
                self.write_json_artifact(Path(output_dir) / "map" / "route7_static_house_base.json", snapshot)
            except Exception:
                pass
        return snapshot

    def route7_prepare_new_map_output_dir(self, target_house_id: str) -> Path:
        self.ensure_route7_state()
        output_dir = self.make_route7_fused_output_dir(target_house_id)
        route7_static_house_base = self.route7_static_house_base_snapshot(target_house_id, output_dir=output_dir)
        self.route5_update_state(
            mode="route7_llm_route_oa3_or3_1_fusion",
            route_window_label="V7",
            target_house_id=str(target_house_id or ""),
            output_dir=str(output_dir),
            route7_map_output_dir=str(output_dir),
            route7_primary_representation="or3_1",
            or3_1_model_path=str(self.default_route7_or31_model_path()),
            route7_static_house_base=route7_static_house_base,
            observation_z_override_cm=self.route7_default_exploration_z_cm(),
            stage="INIT_RUN",
        )
        if callable(getattr(self, "ensure_route6_state", None)):
            self.ensure_route6_state()
        if callable(getattr(self, "route6_record_update_map_output_dir", None)):
            self.route6_record_update_map_output_dir(output_dir, source="route7_fresh_start")
        return output_dir

    def route7_current_map_output_dir(self) -> Optional[Path]:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        for key in ("route7_map_output_dir", "output_dir"):
            raw = str(state.get(key, "") or "")
            if raw:
                path = Path(raw)
                if path.exists():
                    return path
        route6_state = getattr(self, "llm_route6_state", {}) if isinstance(getattr(self, "llm_route6_state", {}), dict) else {}
        raw = str(route6_state.get("route6_update_map_output_dir", "") or "")
        if raw:
            path = Path(raw)
            if path.exists():
                return path
        return None

    def route7_start_update_map_realtime(self, session: Any, output_dir: Path) -> bool:
        self.ensure_route7_state()
        if callable(getattr(self, "ensure_route6_state", None)):
            self.ensure_route6_state()
        realtime_thread = getattr(self, "route6_update_map_realtime_thread", None)
        if realtime_thread is not None and realtime_thread.is_alive():
            self.route6_update_map_realtime_stop_event.set()
            try:
                realtime_thread.join(timeout=1.0)
            except RuntimeError:
                pass
        capture_thread = getattr(self, "route6_update_map_capture_thread", None)
        if capture_thread is not None and capture_thread.is_alive():
            self.route6_update_map_capture_stop_event.set()
            try:
                capture_thread.join(timeout=1.0)
            except RuntimeError:
                pass
        realtime_thread = getattr(self, "route6_update_map_realtime_thread", None)
        if realtime_thread is not None and realtime_thread.is_alive():
            self.route6_update_map_status_var.set("Route 6 Update Map: waiting for previous realtime update to stop.")
            return False
        capture_thread = getattr(self, "route6_update_map_capture_thread", None)
        if capture_thread is not None and capture_thread.is_alive():
            self.route6_update_map_status_var.set("Route 6 Update Map: waiting for previous capture to stop.")
            return False
        self.route6_record_update_map_output_dir(Path(output_dir), source="route7_realtime")
        self.llm_route6_state["route6_update_map_capture_frame_index"] = 0
        self.route6_update_map_capture_stop_event.clear()
        self.route6_update_map_realtime_stop_event.clear()
        self.route6_update_map_realtime_thread = threading.Thread(
            target=lambda: self.route6_update_map_realtime_worker(session, Path(output_dir)),
            daemon=True,
        )
        self.route6_update_map_realtime_thread.start()
        self.route6_update_map_status_var.set(f"Route 6 Update Map: V7 realtime update started -> {output_dir}")
        return True

    def route7_map_layer_z_cm_from_key(self, key: str) -> float:
        text = str(key or "").strip().lower()
        if text.startswith("z_"):
            text = text[2:]
        try:
            return float(text)
        except Exception:
            return self.route7_default_exploration_z_cm()

    def route7_selected_map_layer_record(self, output_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], str, float, Dict[str, Any]]:
        manifest: Dict[str, Any] = {}
        load_manifest = getattr(self, "route6_update_map_load_manifest", None)
        if callable(load_manifest):
            manifest = load_manifest(build_if_missing=True, output_dir=output_dir) or {}
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        selected_key = self.route7_select_update_map_layer_key(layers) if layers else self.route7_default_layer_key()
        key_fn = getattr(self, "_route6_update_map_layer_key", None)
        selected_layer = next(
            (
                layer
                for layer in layers
                if isinstance(layer, dict)
                and ((str(key_fn(layer)) if callable(key_fn) else f"z_{int(float(layer.get('z_cm', 0) or 0)):03d}") == selected_key)
            ),
            {},
        )
        layer_z = self.route7_map_layer_z_cm_from_key(selected_key)
        if selected_layer:
            try:
                layer_z = float(selected_layer.get("z_cm", layer_z) or layer_z)
            except Exception:
                pass
        return dict(selected_layer), selected_key, float(layer_z), manifest

    def route7_map_layer_key_for_record(self, layer_record: Dict[str, Any]) -> str:
        key_fn = getattr(self, "_route6_update_map_layer_key", None)
        if callable(key_fn):
            try:
                return str(key_fn(layer_record))
            except Exception:
                pass
        try:
            return f"z_{int(round(float((layer_record if isinstance(layer_record, dict) else {}).get('z_cm', 0.0) or 0.0))):03d}"
        except Exception:
            return self.route7_default_layer_key()

    def route7_available_map_layer_records(self, output_dir: Optional[Path] = None) -> Tuple[List[Tuple[Dict[str, Any], str, float]], Dict[str, Any]]:
        manifest: Dict[str, Any] = {}
        load_manifest = getattr(self, "route6_update_map_load_manifest", None)
        if callable(load_manifest):
            manifest = load_manifest(build_if_missing=True, output_dir=output_dir) or {}
        layers = manifest.get("layers", []) if isinstance(manifest.get("layers", []), list) else []
        records: List[Tuple[Dict[str, Any], str, float]] = []
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            layer_key = self.route7_map_layer_key_for_record(layer)
            try:
                layer_z = float(layer.get("z_cm", self.route7_map_layer_z_cm_from_key(layer_key)) or self.route7_map_layer_z_cm_from_key(layer_key))
            except Exception:
                layer_z = self.route7_map_layer_z_cm_from_key(layer_key)
            records.append((dict(layer), layer_key, float(layer_z)))
        return records, manifest

    def route7_order_map_layers_for_route(
        self,
        layers: List[Tuple[Dict[str, Any], str, float]],
        *,
        preferred_layer_key: str = "",
        target_z_cm: float = 300.0,
    ) -> List[Tuple[Dict[str, Any], str, float]]:
        preferred = str(preferred_layer_key or self.route7_default_layer_key()).strip() or self.route7_default_layer_key()
        default_key = self.route7_default_layer_key()

        def score(item: Tuple[Dict[str, Any], str, float]) -> Tuple[int, float, float, str]:
            _record, key, z_cm = item
            if key == preferred:
                rank = 0
            elif key == default_key:
                rank = 1
            else:
                rank = 2
            return rank, abs(float(z_cm) - float(target_z_cm)), abs(float(z_cm) - self.route7_default_exploration_z_cm()), str(key)

        return sorted(list(layers), key=score)

    def route7_layer_pose_block_report(
        self,
        layer_record: Dict[str, Any],
        pose: Dict[str, Any],
        *,
        layer_key: str,
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        loaded = self.route7_load_layer_occupancy_grid(layer_record)
        if loaded.get("status") != "ok":
            return {"status": "unavailable", "reason": str(loaded.get("reason", loaded.get("status", "missing_route7_layer_grid"))), "layer_key": layer_key}
        metadata = loaded["metadata"]
        grid = loaded["grid"]
        raw_cell = self.route7_pose_to_layer_cell(metadata, pose)
        if raw_cell is None:
            return {"status": "outside_layer_grid", "reason": "pose_outside_route7_layer_grid", "layer_key": layer_key, "cell": []}
        mask = self.route7_inflated_occupied_mask(grid, inflation_cells=inflation_cells)
        blocked = self.route7_layer_cell_blocked(mask, raw_cell)
        nearest = self.route7_nearest_free_layer_cell(mask, raw_cell)
        return {
            "status": "ok",
            "layer_key": layer_key,
            "cell": list(raw_cell),
            "blocked": bool(blocked),
            "nearest_free_cell": list(nearest) if nearest else [],
            "inflation_cells": int(inflation_cells),
        }

    def route7_axis_values_near_house_edge(self, bbox: Dict[str, Any], facade: str, safe_intervals: List[Dict[str, Any]]) -> List[float]:
        axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
        low, high = sorted((float(axis_min), float(axis_max)))
        span = max(0.0, high - low)
        if span <= 0.0:
            return [low, high]
        inset = min(max(120.0, span * 0.12), span * 0.25)
        raw_values = [low + inset, high - inset] if span > inset * 2.0 else [low + span * 0.25, low + span * 0.75]
        intervals = [
            (float(item.get("min")), float(item.get("max")))
            for item in safe_intervals
            if isinstance(item, dict) and self._as_float_or_none(item.get("min")) is not None and self._as_float_or_none(item.get("max")) is not None
        ]
        if not intervals:
            return raw_values
        adjusted: List[float] = []
        for value in raw_values:
            containing = [(min(lo, hi), max(lo, hi)) for lo, hi in intervals if min(lo, hi) <= value <= max(lo, hi)]
            if containing:
                adjusted.append(float(value))
                continue
            nearest = min(
                ((min(lo, hi), max(lo, hi)) for lo, hi in intervals),
                key=lambda pair: min(abs(value - pair[0]), abs(value - pair[1])),
            )
            adjusted.append(min(max(float(value), nearest[0]), nearest[1]))
        if len(adjusted) >= 2 and abs(adjusted[0] - adjusted[1]) < 1.0:
            lo, hi = min(intervals, key=lambda pair: abs((pair[1] - pair[0]) - span))
            adjusted = [lo, hi] if hi > lo else raw_values
        return adjusted[:2]

    def route7_layer_pose_occupancy_score(self, layer_record: Dict[str, Any], pose: Dict[str, Any], *, radius_cells: int = 2) -> Dict[str, Any]:
        metadata = self.route6_update_map_load_layer_metadata(layer_record) if callable(getattr(self, "route6_update_map_load_layer_metadata", None)) else {}
        grid_path = Path(str((layer_record if isinstance(layer_record, dict) else {}).get("occupancy_grid_path", "") or ""))
        if not metadata or not grid_path.is_file():
            return {"status": "no_layer_grid", "occupied_cell_count": None, "free_cell_count": None}
        try:
            width = int(metadata.get("width", 0) or 0)
            height = int(metadata.get("height", 0) or 0)
            resolution = float(metadata.get("resolution_m", 0.25) or 0.25)
            origin_x, origin_y = [float(value) for value in metadata.get("origin_standard_m", [0.0, 0.0])]
            standard_x_m = float(pose.get("x", 0.0) or 0.0) / 100.0
            standard_y_m = -float(pose.get("y", 0.0) or 0.0) / 100.0
            col = int(math.floor((standard_x_m - origin_x) / resolution))
            row = int(math.floor((standard_y_m - origin_y) / resolution))
            if col < 0 or col >= width or row < 0 or row >= height:
                return {"status": "outside_layer_grid", "occupied_cell_count": None, "free_cell_count": None, "grid_col": col, "grid_row": row}
            grid = np.asarray(np.load(grid_path), dtype=np.int16)
            r = max(1, int(radius_cells))
            patch = grid[max(0, row - r): min(grid.shape[0], row + r + 1), max(0, col - r): min(grid.shape[1], col + r + 1)]
            occupied = int(np.sum(patch >= 100))
            free = int(np.sum(patch == 0))
            return {"status": "ok", "occupied_cell_count": occupied, "free_cell_count": free, "grid_col": col, "grid_row": row}
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "occupied_cell_count": None, "free_cell_count": None}

    def route7_lidar_radius_cm(self) -> float:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        sensing = state.get("sensing_config", {}) if isinstance(state.get("sensing_config", {}), dict) else {}
        candidates = [
            getattr(getattr(self, "args", None), "lidar_depth_max_cm", None),
            sensing.get("lidar_depth_max_cm"),
            state.get("lidar_depth_max_cm"),
            state.get("route7_lidar_radius_cm"),
        ]
        for value in candidates:
            try:
                radius = float(value)
            except Exception:
                continue
            if math.isfinite(radius) and radius > 0.0:
                return max(100.0, radius)
        return 1200.0

    def route7_lidar_edge_observation_formula(self, bbox: Dict[str, Any], facade: str) -> Dict[str, Any]:
        axis_min, axis_max = self.route2_facade_axis_range(bbox, facade)
        low, high = sorted((float(axis_min), float(axis_max)))
        edge_length = max(0.0, high - low)
        lidar_radius = self.route7_lidar_radius_cm()
        effective_radius = max(1.0, 0.8 * float(lidar_radius))
        point_count = max(1, int(math.ceil(edge_length / effective_radius))) if edge_length > 0.0 else 1
        spacing = edge_length / float(point_count) if point_count > 0 else edge_length
        upper_standoff = max(50.0, 0.95 * effective_radius)
        lower_standoff = min(250.0, upper_standoff)
        standoff = min(max(0.65 * effective_radius, lower_standoff), upper_standoff)
        axis_values = [low + (float(index) + 0.5) * spacing for index in range(point_count)]
        return {
            "schema": "route7_lidar_edge_observation_formula_v1",
            "facade": str(facade or "").strip().lower(),
            "axis_min": round(low, 3),
            "axis_max": round(high, 3),
            "edge_length_cm": round(edge_length, 3),
            "lidar_radius_cm": round(float(lidar_radius), 3),
            "effective_lidar_radius_cm": round(float(effective_radius), 3),
            "effective_lidar_multiplier": 0.8,
            "point_count": int(point_count),
            "spacing_cm": round(float(spacing), 3),
            "observation_standoff_cm": round(float(standoff), 3),
            "axis_values": [round(float(value), 3) for value in axis_values],
            "z_cm": round(float(self.route7_default_exploration_z_cm()), 3),
            "formula": "N=max(1,ceil(L_edge/(0.8*R_lidar))); axis_i=axis_min+(i+0.5)*(L_edge/N); D_obs=clamp(0.65*(0.8*R_lidar),250,0.95*(0.8*R_lidar))",
        }

    def route7_adjust_observation_point_to_free_cell(
        self,
        point: Dict[str, Any],
        *,
        output_dir: Optional[Path] = None,
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        adjusted = dict(point)
        layer_record, layer_key, layer_z, _manifest = self.route7_selected_map_layer_record(output_dir)
        adjusted["route7_map_layer_key"] = layer_key
        adjusted["route7_map_layer_z_cm"] = round(float(layer_z), 3)
        loaded = self.route7_load_layer_occupancy_grid(layer_record)
        if loaded.get("status") != "ok":
            adjusted["route7_observation_map_status"] = str(loaded.get("status", "missing_layer_grid") or "missing_layer_grid")
            adjusted["route7_observation_replan_reason"] = str(loaded.get("reason", adjusted["route7_observation_map_status"]) or "")
            return adjusted
        metadata = loaded["metadata"]
        grid = loaded["grid"]
        raw_cell = self.route7_pose_to_layer_cell(metadata, adjusted)
        if raw_cell is None:
            adjusted["route7_observation_map_status"] = "outside_layer_grid"
            adjusted["route7_map_cell"] = []
            return adjusted
        mask = self.route7_inflated_occupied_mask(grid, inflation_cells=inflation_cells)
        blocked = self.route7_layer_cell_blocked(mask, raw_cell)
        adjusted["route7_observation_map_status"] = "blocked" if blocked else "free"
        adjusted["route7_map_cell"] = [int(raw_cell[0]), int(raw_cell[1])]
        adjusted["route7_observation_inflation_cells"] = int(inflation_cells)
        if not blocked:
            return adjusted
        nearest = self.route7_nearest_free_layer_cell(mask, raw_cell)
        adjusted["route7_original_pose"] = self.route5_json_safe(point)
        adjusted["route7_original_map_cell"] = [int(raw_cell[0]), int(raw_cell[1])]
        adjusted["route7_observation_replanned_from_blocked"] = True
        adjusted["route7_observation_replan_reason"] = "route7_observation_point_inside_occupied_cell"
        if nearest is None:
            adjusted["status"] = "blocked"
            adjusted["route7_observation_map_status"] = "blocked_no_nearest_free_cell"
            return adjusted
        free_pose = self.route7_layer_cell_to_pose(
            metadata,
            nearest,
            z_cm=float(adjusted.get("z", layer_z) or layer_z),
            yaw_deg=float(adjusted.get("yaw_deg", adjusted.get("yaw", 0.0)) or 0.0),
        )
        adjusted.update(free_pose)
        adjusted["yaw_deg"] = round(float(free_pose.get("yaw", adjusted.get("yaw_deg", 0.0)) or 0.0), 3)
        adjusted["route7_map_cell"] = [int(nearest[0]), int(nearest[1])]
        adjusted["route7_observation_map_status"] = "replanned_to_nearest_free_cell"
        adjusted["status"] = "planned"
        return adjusted

    def route7_edge_observation_attempts_for_facade(
        self,
        target_house_id: str,
        facade: str,
        *,
        output_dir: Optional[Path] = None,
        inflation_cells: int = 1,
    ) -> List[Dict[str, Any]]:
        hid = str(target_house_id or "").strip()
        facade = str(facade or "").strip().lower()
        if not hid or facade not in {"south", "east", "north", "west"}:
            return []
        bbox = self.route7_house_bbox_for_id(hid, output_dir=output_dir)
        if not bbox:
            return []
        bbox_source = str(bbox.get("route7_house_bbox_source", bbox.get("source", "house_world_bbox_for_id")) or "house_world_bbox_for_id")
        formula = self.route7_lidar_edge_observation_formula(bbox, facade)
        formula["bbox_source"] = bbox_source
        formula["bbox_world"] = self.route5_json_safe(bbox)
        z_cm = float(formula.get("z_cm", self.route7_default_exploration_z_cm()) or self.route7_default_exploration_z_cm())
        standoff = float(formula.get("observation_standoff_cm", 300.0) or 300.0)
        attempts: List[Dict[str, Any]] = []
        for index, axis in enumerate(formula.get("axis_values", []) if isinstance(formula.get("axis_values", []), list) else [], start=1):
            pose = self.route2_facade_pose_from_axis(bbox, facade, float(axis), standoff, z_cm)
            yaw = float(pose.get("yaw_deg", pose.get("yaw", 0.0)) or 0.0)
            attempt: Dict[str, Any] = {
                **pose,
                "yaw": round(yaw, 3),
                "yaw_deg": round(yaw, 3),
                "target_house_id": hid,
                "facade": facade,
                "target_id": f"{hid}_{facade}_edge_obs_{index:03d}",
                "label": f"{hid}_{facade}_edge_obs_{index:03d}",
                "route_point_type": "route7_edge_observation_point",
                "view_type": "route7_lidar_edge_observation",
                "observation_attempt_index": index,
                "observation_attempt_source": "route7_lidar_edge_minimal",
                "observation_selection_score": round(float(index), 3),
                "axis_value": round(float(axis), 3),
                "standoff_cm": round(float(standoff), 3),
                "route7_observation_formula": self.route5_json_safe(formula),
                "route7_lidar_radius_cm": formula.get("lidar_radius_cm"),
                "route7_effective_lidar_radius_cm": formula.get("effective_lidar_radius_cm"),
                "route7_edge_length_cm": formula.get("edge_length_cm"),
                "route7_edge_point_count": formula.get("point_count"),
                "route7_house_bbox_source": bbox_source,
                "route7_house_bbox": self.route5_json_safe(bbox),
                "status": "planned",
            }
            adjusted = self.route7_adjust_observation_point_to_free_cell(
                attempt,
                output_dir=output_dir,
                inflation_cells=inflation_cells,
            )
            corridor = self.route7_observation_facade_corridor_report(
                adjusted,
                bbox,
                facade,
                expected_standoff_cm=standoff,
            )
            adjusted["route7_observation_facade_corridor"] = self.route5_json_safe(corridor)
            if not bool(corridor.get("same_facade_corridor", True)):
                adjusted["status"] = "blocked"
                adjusted["route7_observation_map_status"] = "outside_target_facade_corridor"
                adjusted["route7_observation_replan_reason"] = "route7_observation_point_outside_target_facade_corridor"
            attempts.append(adjusted)
        return attempts

    def route7_edge_observation_candidates_for_house(
        self,
        target_house_id: str,
        *,
        output_dir: Optional[Path] = None,
        facades: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        facade_order = [str(item).strip().lower() for item in (facades or self.route5_task_facade_priority()) if str(item).strip().lower()]
        if not facade_order:
            facade_order = ["south", "east", "north", "west"]
        candidates: List[Dict[str, Any]] = []
        for facade in facade_order:
            if facade not in {"south", "east", "north", "west"}:
                continue
            attempts = self.route7_edge_observation_attempts_for_facade(target_house_id, facade, output_dir=output_dir)
            if not attempts:
                candidates.append(
                    {
                        "target_house_id": str(target_house_id or ""),
                        "facade": facade,
                        "status": "blocked",
                        "reason": "route7_no_lidar_edge_observation_attempts",
                        "route7_edge_observation_attempts": [],
                        "observation_attempts": [],
                        "observation_attempt_count": 0,
                    }
                )
                continue
            valid_attempts = [
                dict(item)
                for item in attempts
                if isinstance(item, dict)
                and str(item.get("status", "") or "").strip().lower() != "blocked"
                and bool(
                    (
                        item.get("route7_observation_facade_corridor", {})
                        if isinstance(item.get("route7_observation_facade_corridor", {}), dict)
                        else {}
                    ).get("same_facade_corridor", True)
                )
            ]
            if not valid_attempts:
                candidates.append(
                    {
                        "target_house_id": str(target_house_id or ""),
                        "facade": facade,
                        "status": "blocked",
                        "reason": "route7_no_valid_lidar_edge_observation_attempts",
                        "route7_edge_observation_attempts": self.route5_json_safe(attempts),
                        "observation_attempts": self.route5_json_safe(attempts),
                        "observation_attempt_count": len(attempts),
                    }
                )
                continue
            selected = dict(valid_attempts[0])
            candidate = {
                **selected,
                "target_house_id": str(target_house_id or ""),
                "facade": facade,
                "status": "planned",
                "route7_edge_observation_candidate": True,
                "route7_edge_observation_attempts": self.route5_json_safe(attempts),
                "selected_observation_attempt": selected,
                "observation_attempts": self.route5_json_safe(attempts),
                "observation_attempt_count": len(attempts),
            }
            candidates.append(candidate)
        return candidates

    def route7_load_layer_occupancy_grid(self, layer_record: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self.route6_update_map_load_layer_metadata(layer_record) if callable(getattr(self, "route6_update_map_load_layer_metadata", None)) else {}
        grid_path = Path(str((layer_record if isinstance(layer_record, dict) else {}).get("occupancy_grid_path", "") or ""))
        if not metadata or not grid_path.is_file():
            return {"status": "missing_layer_grid", "reason": "missing_route7_layer_grid", "metadata": metadata, "grid": None}
        try:
            grid = np.asarray(np.load(grid_path), dtype=np.int16)
        except Exception as exc:
            return {"status": "error", "reason": f"load_route7_layer_grid_failed:{exc}", "metadata": metadata, "grid": None}
        if grid.ndim != 2 or grid.size <= 0:
            return {"status": "error", "reason": "invalid_route7_layer_grid_shape", "metadata": metadata, "grid": None}
        metadata = dict(metadata)
        metadata["height"] = int(metadata.get("height", grid.shape[0]) or grid.shape[0])
        metadata["width"] = int(metadata.get("width", grid.shape[1]) or grid.shape[1])
        metadata["resolution_m"] = float(metadata.get("resolution_m", 0.25) or 0.25)
        metadata.setdefault("origin_standard_m", [0.0, 0.0])
        return {"status": "ok", "metadata": metadata, "grid": grid, "grid_path": str(grid_path)}

    def route7_pose_to_layer_cell(self, metadata: Dict[str, Any], pose: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        try:
            width = int(metadata.get("width", 0) or 0)
            height = int(metadata.get("height", 0) or 0)
            resolution = float(metadata.get("resolution_m", 0.25) or 0.25)
            origin_x, origin_y = [float(value) for value in metadata.get("origin_standard_m", [0.0, 0.0])]
            standard_x_m = float(pose.get("x", 0.0) or 0.0) / 100.0
            standard_y_m = -float(pose.get("y", 0.0) or 0.0) / 100.0
        except Exception:
            return None
        if width <= 0 or height <= 0 or resolution <= 0:
            return None
        col = int(math.floor((standard_x_m - origin_x) / resolution))
        row = int(math.floor((standard_y_m - origin_y) / resolution))
        if col < 0 or col >= width or row < 0 or row >= height:
            return None
        return row, col

    def route7_layer_cell_to_pose(self, metadata: Dict[str, Any], cell: Tuple[int, int], *, z_cm: float, yaw_deg: float) -> Dict[str, float]:
        resolution = float(metadata.get("resolution_m", 0.25) or 0.25)
        origin_x, origin_y = [float(value) for value in metadata.get("origin_standard_m", [0.0, 0.0])]
        row, col = int(cell[0]), int(cell[1])
        x_cm = (origin_x + (float(col) + 0.5) * resolution) * 100.0
        y_cm = -(origin_y + (float(row) + 0.5) * resolution) * 100.0
        return {"x": round(x_cm, 3), "y": round(y_cm, 3), "z": round(float(z_cm), 3), "yaw": round(float(yaw_deg), 3)}

    def route7_inflated_occupied_mask(self, grid: np.ndarray, *, inflation_cells: int = 1) -> np.ndarray:
        occupied = np.asarray(grid, dtype=np.int16) >= 100
        radius = max(0, int(inflation_cells))
        if radius <= 0:
            return occupied
        padded = np.pad(occupied, radius, mode="constant", constant_values=False)
        inflated = np.zeros_like(occupied, dtype=bool)
        height, width = occupied.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                inflated |= padded[radius + dr: radius + dr + height, radius + dc: radius + dc + width]
        return inflated

    def route7_layer_segment_cells(self, start_cell: Tuple[int, int], target_cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r0, c0 = int(start_cell[0]), int(start_cell[1])
        r1, c1 = int(target_cell[0]), int(target_cell[1])
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        cells: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for idx in range(steps + 1):
            t = float(idx) / float(steps)
            cell = (int(round(r0 + (r1 - r0) * t)), int(round(c0 + (c1 - c0) * t)))
            if cell in seen:
                continue
            seen.add(cell)
            cells.append(cell)
        return cells

    def route7_layer_cell_blocked(self, mask: np.ndarray, cell: Tuple[int, int]) -> bool:
        row, col = int(cell[0]), int(cell[1])
        return row < 0 or col < 0 or row >= int(mask.shape[0]) or col >= int(mask.shape[1]) or bool(mask[row, col])

    def route7_layer_segment_blocked(self, mask: np.ndarray, start_cell: Tuple[int, int], target_cell: Tuple[int, int]) -> bool:
        return any(self.route7_layer_cell_blocked(mask, cell) for cell in self.route7_layer_segment_cells(start_cell, target_cell))

    def route7_nearest_free_layer_cell(self, mask: np.ndarray, cell: Tuple[int, int], *, max_radius: int = 24) -> Optional[Tuple[int, int]]:
        row, col = int(cell[0]), int(cell[1])
        height, width = int(mask.shape[0]), int(mask.shape[1])
        if 0 <= row < height and 0 <= col < width and not bool(mask[row, col]):
            return row, col
        for radius in range(1, max(1, int(max_radius)) + 1):
            candidates: List[Tuple[float, Tuple[int, int]]] = []
            for rr in range(row - radius, row + radius + 1):
                for cc in (col - radius, col + radius):
                    if 0 <= rr < height and 0 <= cc < width and not bool(mask[rr, cc]):
                        candidates.append((math.hypot(rr - row, cc - col), (rr, cc)))
            for cc in range(col - radius + 1, col + radius):
                for rr in (row - radius, row + radius):
                    if 0 <= rr < height and 0 <= cc < width and not bool(mask[rr, cc]):
                        candidates.append((math.hypot(rr - row, cc - col), (rr, cc)))
            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
        return None

    def route7_smooth_layer_path(self, mask: np.ndarray, cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        smoothed: List[Tuple[int, int]] = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            next_index = len(cells) - 1
            while next_index > anchor + 1:
                if not self.route7_layer_segment_blocked(mask, cells[anchor], cells[next_index]):
                    break
                next_index -= 1
            smoothed.append(cells[next_index])
            anchor = next_index
        return smoothed

    def route7_layer_nearest_occupied_distance_cm(self, grid: np.ndarray, cell: Tuple[int, int], metadata: Dict[str, Any]) -> Optional[float]:
        occupied = np.argwhere(np.asarray(grid, dtype=np.int16) >= 100)
        if occupied.size <= 0:
            return None
        row, col = int(cell[0]), int(cell[1])
        deltas = occupied.astype(np.float64) - np.asarray([[float(row), float(col)]], dtype=np.float64)
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        resolution_cm = float(metadata.get("resolution_m", 0.25) or 0.25) * 100.0
        return round(float(np.min(distances)) * resolution_cm, 3)

    def route7_map_route_clearance_report(
        self,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        *,
        output_dir: Optional[Path] = None,
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        layer_record, layer_key, layer_z_cm, _manifest = self.route7_selected_map_layer_record(output_dir)
        loaded = self.route7_load_layer_occupancy_grid(layer_record)
        if loaded.get("status") != "ok":
            return {"status": "unavailable", "reason": str(loaded.get("reason", loaded.get("status", "missing_route7_layer_grid"))), "layer_key": layer_key}
        metadata = loaded["metadata"]
        grid = loaded["grid"]
        current_cell = self.route7_pose_to_layer_cell(metadata, current_pose)
        target_cell = self.route7_pose_to_layer_cell(metadata, target_pose)
        if current_cell is None or target_cell is None:
            return {
                "status": "outside_layer_grid",
                "reason": "current_or_target_outside_route7_layer_grid",
                "layer_key": layer_key,
                "current_cell": current_cell,
                "target_cell": target_cell,
            }
        mask = self.route7_inflated_occupied_mask(grid, inflation_cells=inflation_cells)
        segment_cells = self.route7_layer_segment_cells(current_cell, target_cell)
        blocked_cells = [cell for cell in segment_cells if self.route7_layer_cell_blocked(mask, cell)]
        nearest_distance = self.route7_layer_nearest_occupied_distance_cm(grid, current_cell, metadata)
        first_blocked = blocked_cells[0] if blocked_cells else None
        return {
            "status": "ok",
            "layer_key": layer_key,
            "layer_z_cm": round(float(layer_z_cm), 3),
            "route_blocked": bool(blocked_cells),
            "blocked_cell_count": len(blocked_cells),
            "first_blocked_cell": list(first_blocked) if first_blocked else [],
            "current_cell": list(current_cell),
            "target_cell": list(target_cell),
            "path_sample_count": len(segment_cells),
            "nearest_occupied_distance_cm": nearest_distance,
            "inflation_cells": int(inflation_cells),
        }

    def route7_plan_navigation_waypoints_from_map(
        self,
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        *,
        output_dir: Optional[Path] = None,
        stage: str = "",
        target_id: str = "",
        target_house_id: str = "",
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        layer_record, layer_key, layer_z_cm, _manifest = self.route7_selected_map_layer_record(output_dir)
        loaded = self.route7_load_layer_occupancy_grid(layer_record)
        if loaded.get("status") != "ok":
            return {
                "status": "fallback",
                "reason": str(loaded.get("reason", loaded.get("status", "missing_route7_layer_grid"))),
                "planner_source": "route7_layered_occupancy_astar",
                "route7_map_route_replan_policy": "map_first_runtime_revalidate",
                "layer_key": layer_key,
                "waypoints": [],
            }
        metadata = loaded["metadata"]
        grid = loaded["grid"]
        mask = self.route7_inflated_occupied_mask(grid, inflation_cells=inflation_cells)
        start_cell_raw = self.route7_pose_to_layer_cell(metadata, start_pose)
        target_cell_raw = self.route7_pose_to_layer_cell(metadata, target_pose)
        if start_cell_raw is None or target_cell_raw is None:
            return {
                "status": "fallback",
                "reason": "start_or_target_outside_route7_layer_grid",
                "planner_source": "route7_layered_occupancy_astar",
                "route7_map_route_replan_policy": "map_first_runtime_revalidate",
                "layer_key": layer_key,
                "start_cell": list(start_cell_raw) if start_cell_raw else [],
                "target_cell": list(target_cell_raw) if target_cell_raw else [],
                "waypoints": [],
            }
        start_cell = self.route7_nearest_free_layer_cell(mask, start_cell_raw)
        target_cell = self.route7_nearest_free_layer_cell(mask, target_cell_raw)
        if start_cell is None or target_cell is None:
            return {
                "status": "blocked",
                "reason": "route7_map_start_or_target_enclosed",
                "planner_source": "route7_layered_occupancy_astar",
                "route7_map_route_replan_policy": "map_first_runtime_revalidate",
                "layer_key": layer_key,
                "start_cell": list(start_cell_raw),
                "target_cell": list(target_cell_raw),
                "waypoints": [],
            }

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

        def node_valid(node: Tuple[int, int]) -> bool:
            return not self.route7_layer_cell_blocked(mask, node)

        def diagonal_clear(node: Tuple[int, int], dr: int, dc: int) -> bool:
            if abs(dr) != 1 or abs(dc) != 1:
                return True
            return node_valid((node[0] + dr, node[1])) and node_valid((node[0], node[1] + dc))

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start_cell))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        best_cost: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        visited = 0
        max_visits = min(100000, max(1000, int(mask.size) * 2))
        found = start_cell == target_cell
        while open_heap and not found and visited <= max_visits:
            _priority, cost, node = heapq.heappop(open_heap)
            if cost > best_cost.get(node, float("inf")) + 1e-6:
                continue
            visited += 1
            for dr, dc, step_cost in neighbors:
                nxt = (node[0] + dr, node[1] + dc)
                if not node_valid(nxt) or not diagonal_clear(node, dr, dc):
                    continue
                new_cost = cost + step_cost
                if new_cost + 1e-6 >= best_cost.get(nxt, float("inf")):
                    continue
                best_cost[nxt] = new_cost
                came_from[nxt] = node
                heuristic = math.hypot(float(target_cell[0] - nxt[0]), float(target_cell[1] - nxt[1]))
                heapq.heappush(open_heap, (new_cost + heuristic, new_cost, nxt))
                if nxt == target_cell:
                    found = True
                    break
        if not found:
            return {
                "status": "blocked",
                "reason": "route7_layered_occupancy_astar_no_path",
                "planner_source": "route7_layered_occupancy_astar",
                "route7_map_route_replan_policy": "map_first_runtime_revalidate",
                "layer_key": layer_key,
                "layer_z_cm": round(float(layer_z_cm), 3),
                "start_cell": list(start_cell),
                "target_cell": list(target_cell),
                "visited": visited,
                "inflation_cells": int(inflation_cells),
                "occupied_cell_count": int(np.sum(np.asarray(grid) >= 100)),
                "waypoints": [],
            }
        cells = [target_cell]
        while cells[-1] != start_cell:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        smooth_cells = self.route7_smooth_layer_path(mask, cells)
        waypoints: List[Dict[str, Any]] = []
        previous_pose = dict(start_pose)
        for idx, cell in enumerate(smooth_cells[1:], start=1):
            is_last = idx == len(smooth_cells) - 1
            waypoint = dict(target_pose) if is_last else self.route7_layer_cell_to_pose(
                metadata,
                cell,
                z_cm=float(target_pose.get("z", layer_z_cm) or layer_z_cm),
                yaw_deg=float(target_pose.get("yaw", start_pose.get("yaw", 0.0)) or 0.0),
            )
            if not is_last:
                dx = float(waypoint.get("x", 0.0)) - float(previous_pose.get("x", 0.0))
                dy = float(waypoint.get("y", 0.0)) - float(previous_pose.get("y", 0.0))
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    waypoint["yaw"] = round(math.degrees(math.atan2(-dy, dx)), 3)
            waypoint.update(
                {
                    "waypoint_index": idx,
                    "waypoint_final": bool(is_last),
                    "route_point_type": "navigation_waypoint",
                    "route7_map_layer_key": layer_key,
                    "route7_map_cell": list(cell),
                    "route7_map_route_replan_policy": "map_first_runtime_revalidate",
                }
            )
            waypoints.append(waypoint)
            previous_pose = waypoint
        return {
            "status": "ok",
            "reason": "route7_layered_occupancy_astar_path",
            "planner_source": "route7_layered_occupancy_astar",
            "route7_map_route_replan_policy": "map_first_runtime_revalidate",
            "layer_key": layer_key,
            "layer_z_cm": round(float(layer_z_cm), 3),
            "stage": str(stage or ""),
            "target_id": str(target_id or ""),
            "target_house_id": str(target_house_id or ""),
            "grid_resolution_m": round(float(metadata.get("resolution_m", 0.25) or 0.25), 4),
            "start_cell": list(start_cell),
            "target_cell": list(target_cell),
            "raw_cells": [list(cell) for cell in cells],
            "smooth_cells": [list(cell) for cell in smooth_cells],
            "visited": visited,
            "inflation_cells": int(inflation_cells),
            "occupied_cell_count": int(np.sum(np.asarray(grid) >= 100)),
            "inflated_occupied_cell_count": int(np.sum(mask)),
            "waypoints": waypoints or [dict(target_pose)],
        }

    def route7_plan_navigation_waypoints_on_layer(
        self,
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        layer_record: Dict[str, Any],
        layer_key: str,
        layer_z_cm: float,
        *,
        output_dir: Optional[Path] = None,
        stage: str = "",
        target_id: str = "",
        target_house_id: str = "",
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        var = getattr(self, "llm_route7_map_layer_var", None)
        previous_key = ""
        try:
            previous_key = str(var.get() or "") if var is not None and hasattr(var, "get") else ""
        except Exception:
            previous_key = ""
        try:
            if var is not None and hasattr(var, "set"):
                var.set(str(layer_key))
            layer_start = dict(start_pose)
            layer_target = dict(target_pose)
            layer_start["z"] = round(float(layer_z_cm), 3)
            layer_target["z"] = round(float(layer_z_cm), 3)
            plan = self.route7_plan_navigation_waypoints_from_map(
                layer_start,
                layer_target,
                output_dir=output_dir,
                stage=stage,
                target_id=target_id,
                target_house_id=target_house_id,
                inflation_cells=inflation_cells,
            )
        finally:
            try:
                if var is not None and hasattr(var, "set") and previous_key:
                    var.set(previous_key)
            except Exception:
                pass
        start_block = self.route7_layer_pose_block_report(layer_record, start_pose, layer_key=layer_key, inflation_cells=inflation_cells)
        target_block = self.route7_layer_pose_block_report(layer_record, target_pose, layer_key=layer_key, inflation_cells=inflation_cells)
        plan = dict(plan if isinstance(plan, dict) else {})
        plan.update(
            {
                "layer_key": str(layer_key),
                "layer_z_cm": round(float(layer_z_cm), 3),
                "route7_layer_start_block_report": self.route5_json_safe(start_block),
                "route7_layer_target_block_report": self.route5_json_safe(target_block),
                "route7_layer_start_blocked": bool(start_block.get("blocked", False)),
                "route7_layer_target_blocked": bool(target_block.get("blocked", False)),
            }
        )
        if bool(start_block.get("blocked", False)) or bool(target_block.get("blocked", False)):
            plan["route7_layer_candidate_blocked"] = True
            plan["route7_layer_candidate_blocked_reason"] = (
                "route7_layer_start_blocked" if bool(start_block.get("blocked", False)) else "route7_layer_target_blocked"
            )
        return plan

    def route7_make_route_segments(
        self,
        start_pose: Dict[str, Any],
        waypoints: List[Dict[str, Any]],
        *,
        default_layer_key: str,
    ) -> List[Dict[str, Any]]:
        points = [dict(start_pose if isinstance(start_pose, dict) else {})]
        points[0].setdefault("route7_map_layer_key", default_layer_key)
        points.extend(dict(item) for item in waypoints if isinstance(item, dict))
        segments: List[Dict[str, Any]] = []
        for idx, (start, end) in enumerate(zip(points, points[1:]), start=1):
            try:
                dx = float(end.get("x", 0.0) or 0.0) - float(start.get("x", 0.0) or 0.0)
                dy = float(end.get("y", 0.0) or 0.0) - float(start.get("y", 0.0) or 0.0)
                dz = float(end.get("z", 0.0) or 0.0) - float(start.get("z", 0.0) or 0.0)
            except Exception:
                dx, dy, dz = 0.0, 0.0, 0.0
            from_layer = str(start.get("route7_map_layer_key", default_layer_key) or default_layer_key)
            to_layer = str(end.get("route7_map_layer_key", default_layer_key) or default_layer_key)
            vertical = abs(dx) <= 1.0 and abs(dy) <= 1.0 and abs(dz) > 5.0
            layer_key = to_layer if vertical else (to_layer or from_layer or default_layer_key)
            segments.append(
                {
                    "segment_index": idx,
                    "kind": "vertical_transition" if vertical else "horizontal_route",
                    "from_layer_key": from_layer,
                    "to_layer_key": to_layer,
                    "layer_key": layer_key,
                    "from_pose": self.route5_json_safe(start),
                    "to_pose": self.route5_json_safe(end),
                    "distance_cm": round(math.sqrt(dx * dx + dy * dy + dz * dz), 3),
                    "horizontal_distance_cm": round(math.hypot(dx, dy), 3),
                    "vertical_delta_cm": round(dz, 3),
                }
            )
        return segments

    def route7_reindex_route_waypoints(self, waypoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = [dict(item) for item in waypoints if isinstance(item, dict)]
        total = len(items)
        for idx, item in enumerate(items, start=1):
            item["waypoint_index"] = idx
            item["waypoint_final"] = bool(idx == total)
        return items

    def route7_compose_realtime_multilayer_plan(
        self,
        selected_plan: Dict[str, Any],
        *,
        start_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        selected_layer_key: str,
        selected_layer_z_cm: float,
        preferred_layer_key: str,
        preferred_layer_z_cm: float,
        preferred_attempt: Dict[str, Any],
        layer_attempts: List[Dict[str, Any]],
        stage: str,
        target_id: str,
        target_house_id: str,
    ) -> Dict[str, Any]:
        selected_plan = dict(selected_plan if isinstance(selected_plan, dict) else {})
        target_z = float(target_pose.get("z", selected_layer_z_cm) or selected_layer_z_cm)
        start_z = float(start_pose.get("z", selected_layer_z_cm) or selected_layer_z_cm)
        waypoints: List[Dict[str, Any]] = []
        transition_count = 0
        multilayer = str(selected_layer_key) != str(preferred_layer_key) or abs(float(selected_layer_z_cm) - target_z) > 25.0 or abs(float(selected_layer_z_cm) - start_z) > 25.0
        selected_start = dict(start_pose)
        selected_start["z"] = round(float(selected_layer_z_cm), 3)
        selected_start["route7_map_layer_key"] = str(selected_layer_key)
        if abs(start_z - float(selected_layer_z_cm)) > 5.0:
            transition_count += 1
            waypoints.append(
                {
                    "x": round(float(start_pose.get("x", 0.0) or 0.0), 3),
                    "y": round(float(start_pose.get("y", 0.0) or 0.0), 3),
                    "z": round(float(selected_layer_z_cm), 3),
                    "yaw": round(float(start_pose.get("yaw", target_pose.get("yaw", 0.0)) or 0.0), 3),
                    "route_point_type": "route7_layer_transition",
                    "route7_map_layer_key": str(selected_layer_key),
                    "route7_transition_from_layer_key": str(preferred_layer_key),
                    "route7_transition_to_layer_key": str(selected_layer_key),
                    "route7_transition_direction": "ascend" if float(selected_layer_z_cm) > start_z else "descend",
                    "label": f"to {selected_layer_key}",
                }
            )
        horizontal_waypoints = [dict(item) for item in selected_plan.get("waypoints", []) if isinstance(item, dict)]
        for item in horizontal_waypoints:
            item.setdefault("route_point_type", "navigation_waypoint")
            item["route7_map_layer_key"] = str(selected_layer_key)
            item["z"] = round(float(selected_layer_z_cm), 3)
            waypoints.append(item)
        preferred_target_blocked = bool((preferred_attempt.get("route7_layer_target_block_report", {}) if isinstance(preferred_attempt.get("route7_layer_target_block_report", {}), dict) else {}).get("blocked", False))
        can_return_to_target_layer = not preferred_target_blocked
        if str(selected_layer_key) != str(preferred_layer_key) and abs(target_z - float(selected_layer_z_cm)) > 5.0 and can_return_to_target_layer:
            transition_count += 1
            waypoints.append(
                {
                    "x": round(float(target_pose.get("x", 0.0) or 0.0), 3),
                    "y": round(float(target_pose.get("y", 0.0) or 0.0), 3),
                    "z": round(target_z, 3),
                    "yaw": round(float(target_pose.get("yaw", 0.0) or 0.0), 3),
                    "route_point_type": "route7_layer_transition",
                    "route7_map_layer_key": str(preferred_layer_key),
                    "route7_transition_from_layer_key": str(selected_layer_key),
                    "route7_transition_to_layer_key": str(preferred_layer_key),
                    "route7_transition_direction": "descend" if target_z < float(selected_layer_z_cm) else "ascend",
                    "label": f"to {preferred_layer_key}",
                }
            )
        elif str(selected_layer_key) != str(preferred_layer_key) and horizontal_waypoints:
            waypoints[-1]["route7_target_z_adjusted_by_map"] = True
            waypoints[-1]["route7_original_target_z_cm"] = round(target_z, 3)
        waypoints = self.route7_reindex_route_waypoints(waypoints or [dict(target_pose)])
        segments = self.route7_make_route_segments(start_pose, waypoints, default_layer_key=preferred_layer_key)
        plan = {
            **selected_plan,
            "status": "ok",
            "reason": "route7_realtime_multilayer_occupancy_path" if multilayer else "route7_realtime_selected_layer_occupancy_path",
            "planner_source": "route7_realtime_multilayer_occupancy_astar",
            "route7_map_route_replan_policy": "map_first_realtime_multilayer_revalidate",
            "route7_realtime_route_plan": True,
            "route7_multilayer_route": bool(multilayer),
            "route7_layer_transition_count": int(transition_count),
            "route7_preferred_layer_key": str(preferred_layer_key),
            "route7_preferred_layer_z_cm": round(float(preferred_layer_z_cm), 3),
            "route7_selected_layer_key": str(selected_layer_key),
            "layer_key": str(selected_layer_key),
            "layer_z_cm": round(float(selected_layer_z_cm), 3),
            "stage": str(stage or ""),
            "target_id": str(target_id or ""),
            "target_house_id": str(target_house_id or ""),
            "start_pose": self.route5_json_safe(start_pose),
            "target_pose": self.route5_json_safe(target_pose),
            "route7_layer_attempts": self.route5_json_safe(layer_attempts),
            "route7_route_segments": self.route5_json_safe(segments),
            "waypoints": waypoints,
        }
        plan["route7_route_signature"] = "|".join(
            f"{item.get('route7_map_layer_key', '')}:{float(item.get('x', 0.0) or 0.0):.1f},{float(item.get('y', 0.0) or 0.0):.1f},{float(item.get('z', 0.0) or 0.0):.1f}"
            for item in waypoints
            if isinstance(item, dict)
        )
        return plan

    def route7_plan_realtime_navigation_route_from_map(
        self,
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        *,
        output_dir: Optional[Path] = None,
        stage: str = "",
        target_id: str = "",
        target_house_id: str = "",
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        layers, manifest = self.route7_available_map_layer_records(output_dir)
        if not layers:
            fallback = self.route7_plan_navigation_waypoints_from_map(
                start_pose,
                target_pose,
                output_dir=output_dir,
                stage=stage,
                target_id=target_id,
                target_house_id=target_house_id,
                inflation_cells=inflation_cells,
            )
            fallback["route7_realtime_route_plan"] = True
            fallback["route7_multilayer_route"] = False
            return fallback
        values = [key for _record, key, _z in layers]
        preferred_key = self.route7_select_update_map_layer_key([record for record, _key, _z in layers]) if values else self.route7_default_layer_key()
        target_z = float(target_pose.get("z", self.route7_default_exploration_z_cm()) or self.route7_default_exploration_z_cm())
        ordered = self.route7_order_map_layers_for_route(layers, preferred_layer_key=preferred_key, target_z_cm=target_z)
        preferred_z = next((z for _record, key, z in layers if key == preferred_key), self.route7_map_layer_z_cm_from_key(preferred_key))
        attempts: List[Dict[str, Any]] = []
        preferred_attempt: Dict[str, Any] = {}
        selected: Optional[Tuple[Dict[str, Any], str, float, Dict[str, Any]]] = None
        for layer_record, layer_key, layer_z in ordered:
            attempt_plan = self.route7_plan_navigation_waypoints_on_layer(
                start_pose,
                target_pose,
                layer_record,
                layer_key,
                layer_z,
                output_dir=output_dir,
                stage=stage,
                target_id=target_id,
                target_house_id=target_house_id,
                inflation_cells=inflation_cells,
            )
            attempt_summary = {
                "layer_key": layer_key,
                "layer_z_cm": round(float(layer_z), 3),
                "status": str(attempt_plan.get("status", "") or ""),
                "reason": str(attempt_plan.get("reason", "") or ""),
                "start_blocked": bool(attempt_plan.get("route7_layer_start_blocked", False)),
                "target_blocked": bool(attempt_plan.get("route7_layer_target_blocked", False)),
                "waypoint_count": len([item for item in attempt_plan.get("waypoints", []) if isinstance(item, dict)]) if isinstance(attempt_plan.get("waypoints", []), list) else 0,
            }
            attempts.append(attempt_summary)
            if layer_key == preferred_key:
                preferred_attempt = attempt_plan
            layer_ok = (
                str(attempt_plan.get("status", "") or "") == "ok"
                and not bool(attempt_plan.get("route7_layer_start_blocked", False))
                and not bool(attempt_plan.get("route7_layer_target_blocked", False))
            )
            if layer_ok:
                selected = (layer_record, layer_key, layer_z, attempt_plan)
                break
        if selected is None:
            last = attempts[-1] if attempts else {}
            return {
                "status": "blocked",
                "reason": "route7_realtime_multilayer_no_free_path",
                "planner_source": "route7_realtime_multilayer_occupancy_astar",
                "route7_map_route_replan_policy": "map_first_realtime_multilayer_revalidate",
                "route7_realtime_route_plan": True,
                "route7_multilayer_route": False,
                "route7_preferred_layer_key": preferred_key,
                "route7_layer_attempts": self.route5_json_safe(attempts),
                "last_attempt": self.route5_json_safe(last),
                "manifest_layer_count": len(layers),
                "manifest": self.route5_json_safe({"schema": manifest.get("schema", ""), "layer_count": len(layers)}),
                "waypoints": [],
                "route7_route_segments": [],
            }
        _record, selected_key, selected_z, selected_plan = selected
        return self.route7_compose_realtime_multilayer_plan(
            selected_plan,
            start_pose=start_pose,
            target_pose=target_pose,
            selected_layer_key=selected_key,
            selected_layer_z_cm=selected_z,
            preferred_layer_key=preferred_key,
            preferred_layer_z_cm=preferred_z,
            preferred_attempt=preferred_attempt or selected_plan,
            layer_attempts=attempts,
            stage=stage,
            target_id=target_id,
            target_house_id=target_house_id,
        )

    def route7_update_realtime_navigation_route(
        self,
        current_pose: Dict[str, float],
        target_pose: Dict[str, float],
        *,
        output_dir: Path,
        stage: str = "",
        target_id: str = "",
        target_house_id: str = "",
        update_reason: str = "runtime_tick",
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        plan = self.route7_plan_realtime_navigation_route_from_map(
            current_pose,
            target_pose,
            output_dir=output_dir,
            stage=stage,
            target_id=target_id,
            target_house_id=target_house_id,
            inflation_cells=inflation_cells,
        )
        route_state = {
            "schema": "route7_realtime_route_plan_v1",
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
            "update_reason": str(update_reason or "runtime_tick"),
            "stage": str(stage or ""),
            "target_id": str(target_id or ""),
            "target_house_id": str(target_house_id or ""),
            "current_pose": self.route5_json_safe(current_pose),
            "target_pose": self.route5_json_safe(target_pose),
            "status": str(plan.get("status", "") or ""),
            "reason": str(plan.get("reason", "") or ""),
            "planner_source": str(plan.get("planner_source", "") or ""),
            "layer_key": str(plan.get("layer_key", plan.get("route7_selected_layer_key", "")) or ""),
            "route7_preferred_layer_key": str(plan.get("route7_preferred_layer_key", "") or ""),
            "route7_multilayer_route": bool(plan.get("route7_multilayer_route", False)),
            "route7_layer_transition_count": int(plan.get("route7_layer_transition_count", 0) or 0),
            "route7_route_signature": str(plan.get("route7_route_signature", "") or ""),
            "waypoints": self.route5_json_safe(plan.get("waypoints", [])),
            "route7_route_segments": self.route5_json_safe(plan.get("route7_route_segments", [])),
            "route7_layer_attempts": self.route5_json_safe(plan.get("route7_layer_attempts", [])),
        }
        plan["route7_realtime_route_state"] = route_state
        try:
            if route_state.get("route7_route_segments"):
                self.route5_update_state(route7_realtime_route_plan=route_state, route7_last_drawable_route_plan=route_state)
            else:
                self.route5_update_state(route7_realtime_route_plan=route_state)
        except Exception:
            pass
        try:
            map_dir = Path(output_dir) / "map"
            self.write_json_artifact(map_dir / "route7_realtime_route_plan.json", route_state)
            self.append_jsonl(map_dir / "route7_realtime_route_plan.jsonl", self.route5_json_safe(route_state))
        except Exception as exc:
            plan["route7_realtime_route_write_error"] = f"{type(exc).__name__}: {exc}"
        return plan

    def route7_plan_spatial_navigation_path(
        self,
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        *,
        output_dir: Optional[Path] = None,
        stage: str = "",
        target_id: str = "",
        target_house_id: str = "",
        update_reason: str = "route7_spatial_astar",
        inflation_cells: int = 1,
    ) -> Dict[str, Any]:
        resolved_output: Optional[Path] = Path(output_dir) if output_dir is not None else self.route7_current_map_output_dir()
        if resolved_output is None:
            fallback = self.route3_plan_navigation_waypoints(
                start_pose,
                target_pose,
                target_house_id,
                grid_cm=float(LLM_ROUTE3_ASTAR_GRID_CM),
            )
            fallback = dict(fallback if isinstance(fallback, dict) else {})
            fallback["planner_source"] = "route7_spatial_occupancy_astar_unavailable_map_fallback"
            fallback["route7_space_astar"] = False
            fallback["route7_astar_dimensions"] = ["x_cell", "y_cell"]
            return fallback
        plan = self.route7_update_realtime_navigation_route(
            start_pose,
            target_pose,
            output_dir=resolved_output,
            stage=stage,
            target_id=target_id,
            target_house_id=target_house_id,
            update_reason=update_reason,
            inflation_cells=inflation_cells,
        )
        plan = dict(plan if isinstance(plan, dict) else {})
        plan["planner_source_original"] = str(plan.get("planner_source", "") or "")
        plan["planner_source"] = "route7_spatial_occupancy_astar"
        plan["route7_space_astar"] = True
        plan["route7_astar_dimensions"] = ["x_cell", "y_cell", "z_layer"]
        plan["route7_map_route_replan_policy"] = str(plan.get("route7_map_route_replan_policy", "") or "map_first_realtime_multilayer_revalidate")
        waypoints = [dict(item) for item in plan.get("waypoints", []) if isinstance(item, dict)]
        for waypoint in waypoints:
            waypoint.setdefault("route_point_type", "navigation_waypoint")
            waypoint["route7_space_astar"] = True
            waypoint.setdefault("route7_map_route_replan_policy", plan["route7_map_route_replan_policy"])
        plan["waypoints"] = waypoints
        return plan

    def route7_realtime_plan_needs_execution_replan(
        self,
        plan: Dict[str, Any],
        current_waypoint: Dict[str, Any],
        *,
        tolerance_cm: float = 90.0,
    ) -> bool:
        if str((plan if isinstance(plan, dict) else {}).get("status", "") or "") != "ok":
            return False
        waypoints = plan.get("waypoints", []) if isinstance(plan.get("waypoints", []), list) else []
        first = next((item for item in waypoints if isinstance(item, dict)), None)
        if not isinstance(first, dict) or not isinstance(current_waypoint, dict):
            return False
        try:
            distance = distance_3d_cm(first, current_waypoint)
        except Exception:
            distance = 0.0
        return bool(distance > float(tolerance_cm))

    def route7_write_navigation_plan_visualization(
        self,
        output_dir: Path,
        plan: Dict[str, Any],
        *,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        target_id: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(plan, dict) or str(plan.get("planner_source", "") or "") not in {"route7_layered_occupancy_astar", "route7_realtime_multilayer_occupancy_astar", "route7_spatial_occupancy_astar"}:
            return {"status": "skipped", "reason": "not_route7_layered_occupancy_plan"}
        out_path = Path(output_dir)
        layer_record, layer_key, _layer_z_cm, _manifest = self.route7_selected_map_layer_record(out_path)
        plan_layer_key = str(plan.get("layer_key", plan.get("route7_selected_layer_key", "")) or "")
        if plan_layer_key:
            layers, _manifest2 = self.route7_available_map_layer_records(out_path)
            matched = next(((record, key, z_cm) for record, key, z_cm in layers if key == plan_layer_key), None)
            if matched is not None:
                layer_record, layer_key, _layer_z_cm = matched
        loaded = self.route7_load_layer_occupancy_grid(layer_record)
        if loaded.get("status") != "ok":
            return {"status": "skipped", "reason": str(loaded.get("reason", loaded.get("status", "missing_layer_grid")))}
        metadata = loaded["metadata"]
        grid = np.asarray(loaded["grid"], dtype=np.int16)
        height, width = grid.shape
        scale = max(2, min(8, int(900 / max(1, max(width, height)))))
        preview_path = Path(str(layer_record.get("occupancy_preview_path", "") or ""))
        if preview_path.is_file():
            try:
                image = Image.open(preview_path).convert("RGB")
                if image.width != width * scale or image.height != height * scale:
                    image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
            except Exception:
                image = Image.new("RGB", (width * scale, height * scale), "white")
        else:
            image = Image.new("RGB", (width * scale, height * scale), "white")
            draw_grid = ImageDraw.Draw(image)
            occupied = np.argwhere(grid >= 100)
            for row, col in occupied:
                x0 = int(col) * scale
                y0 = (height - 1 - int(row)) * scale
                draw_grid.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=(20, 20, 20))
        draw = ImageDraw.Draw(image)

        def cell_pixel(cell: Any) -> Optional[Tuple[int, int]]:
            if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                return None
            try:
                row = int(cell[0])
                col = int(cell[1])
            except Exception:
                return None
            if row < 0 or row >= height or col < 0 or col >= width:
                return None
            return int(col * scale + scale / 2), int((height - 1 - row) * scale + scale / 2)

        def pose_pixel(pose: Dict[str, Any]) -> Optional[Tuple[int, int]]:
            cell = self.route7_pose_to_layer_cell(metadata, pose if isinstance(pose, dict) else {})
            return cell_pixel(cell) if cell is not None else None

        raw_pixels = [pixel for pixel in (cell_pixel(cell) for cell in plan.get("raw_cells", []) if isinstance(plan.get("raw_cells", []), list)) if pixel is not None]
        smooth_pixels = [pixel for pixel in (cell_pixel(cell) for cell in plan.get("smooth_cells", []) if isinstance(plan.get("smooth_cells", []), list)) if pixel is not None]
        if len(raw_pixels) >= 2:
            draw.line(raw_pixels, fill=(126, 211, 153), width=max(1, scale))
        if len(smooth_pixels) >= 2:
            draw.line(smooth_pixels, fill=(0, 128, 64), width=max(2, scale + 1))
        current_pixel = pose_pixel(current_pose)
        target_pixel = pose_pixel(target_pose)
        for pixel, color, label in (
            (current_pixel, (21, 101, 192), "start"),
            (target_pixel, (194, 24, 91), "target"),
        ):
            if pixel is None:
                continue
            radius = max(4, scale * 2)
            draw.ellipse((pixel[0] - radius, pixel[1] - radius, pixel[0] + radius, pixel[1] + radius), fill=color, outline=(0, 0, 0), width=1)
            draw.text((pixel[0] + radius + 2, pixel[1] - radius), label, fill=color)
        for idx, waypoint in enumerate(plan.get("waypoints", []) if isinstance(plan.get("waypoints", []), list) else [], start=1):
            if not isinstance(waypoint, dict):
                continue
            pixel = pose_pixel(waypoint)
            if pixel is None:
                continue
            radius = max(3, scale)
            draw.rectangle((pixel[0] - radius, pixel[1] - radius, pixel[0] + radius, pixel[1] + radius), outline=(255, 152, 0), width=max(1, scale // 2))
            draw.text((pixel[0] + radius + 2, pixel[1] + radius + 2), f"wp{idx}", fill=(255, 111, 0))
        title = f"Route7 planned trajectory {layer_key} target={target_id or plan.get('target_id', '')}"
        draw.rectangle((0, 0, min(image.width - 1, 560), 22), fill=(255, 255, 255))
        draw.text((6, 4), title[:92], fill=(0, 0, 0))
        safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_id or plan.get("target_id", "route7_plan") or "route7_plan")).strip("_") or "route7_plan"
        vis_dir = out_path / "map"
        vis_dir.mkdir(parents=True, exist_ok=True)
        latest_path = vis_dir / "route7_planned_trajectory_latest.png"
        target_path = vis_dir / f"route7_planned_trajectory_{safe_target}.png"
        image.save(latest_path)
        if target_path != latest_path:
            image.save(target_path)
        return {
            "status": "ok",
            "schema": "route7_planned_trajectory_visualization_v1",
            "visualization_path": str(latest_path),
            "target_visualization_path": str(target_path),
            "layer_key": layer_key,
            "raw_cell_count": len(raw_pixels),
            "smooth_cell_count": len(smooth_pixels),
            "waypoint_count": len([item for item in plan.get("waypoints", []) if isinstance(item, dict)]) if isinstance(plan.get("waypoints", []), list) else 0,
        }

    def route7_should_use_map_route_planner(self, stage: str, target_id: str = "", *, output_dir: Optional[Path] = None) -> bool:
        state = self.llm_route5_state if isinstance(getattr(self, "llm_route5_state", None), dict) else {}
        route_label = str(state.get("route_window_label", "") or "").strip().upper()
        output_text = str(output_dir or state.get("route7_map_output_dir", state.get("output_dir", "")) or "")
        is_route7 = route_label == "V7" or "llm_route_7_fusion_runs" in output_text.replace("/", "\\")
        if not is_route7:
            return False
        stage_text = str(stage or "").upper()
        target_text = str(target_id or "").lower()
        return stage_text in {"NAV_TO_OBS", "NAV_TO_SCAN_POINT", "ACTIVE_NBV_NAV_TO_SCAN_POINT"} or "_z_" in target_text or "map_layer" in target_text

    def route7_navigation_safety_report(
        self,
        target_house_id: str,
        pose: Dict[str, Any],
        *,
        stage: str = "",
        facade: str = "",
        target_id: str = "",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        safety = self.route3_safety_report_for_pose(target_house_id, pose)
        if bool(safety.get("safe", False)):
            return safety
        if not self.route7_should_use_map_route_planner(stage, target_id, output_dir=output_dir):
            return safety
        if str(safety.get("reason", "") or "") != "target_house_bbox":
            return safety
        corridor = self.route5_pose_facade_corridor_check(
            target_house_id,
            facade,
            pose,
            axis_margin_cm=250.0,
            side_margin_cm=180.0,
        )
        if not bool(corridor.get("same_facade_corridor", False)):
            checked = dict(safety)
            checked["route7_target_house_bbox_allowed"] = False
            checked["route7_target_house_bbox_recheck"] = {
                "reason": "not_on_requested_facade_boundary",
                "facade_corridor": self.route5_json_safe(corridor),
            }
            return checked
        clearance = self.route7_map_route_clearance_report(pose, pose, output_dir=output_dir)
        if clearance.get("status") == "ok" and not bool(clearance.get("route_blocked", False)):
            return {
                "safe": True,
                "reason": "route7_target_house_boundary_waypoint_allowed",
                "route7_target_house_bbox_allowed": True,
                "original_safety": self.route5_json_safe(safety),
                "facade_corridor": self.route5_json_safe(corridor),
                "map_route_clearance": self.route5_json_safe(clearance),
            }
        checked = dict(safety)
        checked["route7_target_house_bbox_allowed"] = False
        checked["route7_target_house_bbox_recheck"] = {
            "reason": "map_layer_cell_blocked_or_unavailable",
            "facade_corridor": self.route5_json_safe(corridor),
            "map_route_clearance": self.route5_json_safe(clearance),
        }
        return checked

    def route7_local_3d_replan_decision(
        self,
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        payload: Dict[str, Any],
        local_3d_safety: Dict[str, Any],
        gate: Dict[str, Any],
        *,
        output_dir: Optional[Path] = None,
        stage: str = "",
        target_id: str = "",
    ) -> Dict[str, Any]:
        front_min = self._as_float_or_none((gate if isinstance(gate, dict) else {}).get("front_min_depth_cm"))
        front_min_cm = float(front_min) if front_min is not None else 0.0
        must_stop = bool((gate if isinstance(gate, dict) else {}).get("must_stop", False))
        can_forward = bool((gate if isinstance(gate, dict) else {}).get("can_forward", False))
        risk_state = str((gate if isinstance(gate, dict) else {}).get("front_risk_state", "") or "")
        if must_stop or risk_state == "must_stop":
            return {
                "action": "stop",
                "reason": "route7_hard_obstacle_stop",
                "front_min_depth_cm": round(front_min_cm, 3),
                "risk_state": risk_state,
                "route7_map_route_replan_policy": "map_first_runtime_revalidate",
            }
        clearance = self.route7_map_route_clearance_report(current_pose, target_pose, output_dir=output_dir)
        action = "replan"
        reason = "route7_map_route_replan_required"
        if clearance.get("status") == "ok" and not bool(clearance.get("route_blocked", False)) and can_forward:
            action = "continue_cautious"
            reason = "route7_map_route_clear_local_3d_soft_block"
        elif clearance.get("status") == "ok" and bool(clearance.get("route_blocked", False)):
            action = "replan"
            reason = "route7_map_route_obstacle_replan"
        return {
            "action": action,
            "reason": reason,
            "front_min_depth_cm": round(front_min_cm, 3),
            "risk_state": risk_state,
            "can_forward": can_forward,
            "must_stop": must_stop,
            "local_3d_safety": self.route5_json_safe(local_3d_safety),
            "payload": self.route5_json_safe(payload),
            "map_route_clearance": self.route5_json_safe(clearance),
            "route7_map_route_replan_policy": "map_first_runtime_revalidate",
            "stage": str(stage or ""),
            "target_id": str(target_id or ""),
        }

    def route7_or2_soft_obstacle_policy(
        self,
        event: Dict[str, Any],
        gate: Dict[str, Any],
        current_pose: Dict[str, Any],
        target_pose: Dict[str, Any],
        *,
        output_dir: Optional[Path] = None,
        rule: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_dict = event if isinstance(event, dict) else {}
        stage = str(event_dict.get("route5_stage", event_dict.get("stage", "")) or "")
        target_id = str(event_dict.get("target_id", "") or "")
        if output_dir is None:
            raw_output_dir = str(event_dict.get("route5_output_dir", "") or "")
            output_dir = Path(raw_output_dir) if raw_output_dir else None
        if not self.route7_should_use_map_route_planner(stage, target_id, output_dir=output_dir):
            return {"mode": "not_route7_map_route"}
        risk_state = str((gate if isinstance(gate, dict) else {}).get("front_risk_state", "") or "").strip().lower()
        front_min = self._as_float_or_none((gate if isinstance(gate, dict) else {}).get("front_min_depth_cm"))
        front_min_cm = float(front_min) if front_min is not None else 0.0
        must_stop = bool((gate if isinstance(gate, dict) else {}).get("must_stop", False))
        deep_red = self.route7_front_square_deep_red_takeover(event_dict, gate if isinstance(gate, dict) else {}, rule=rule)
        base = {
            "schema": "route7_soft_obstacle_policy_v1",
            "route7_deep_red_only_avoidance": True,
            "front_min_depth_cm": round(front_min_cm, 3),
            "front_risk_state": risk_state,
            "must_stop": must_stop,
            "route7_front_square_deep_red_takeover": self.route5_json_safe(deep_red),
            "stage": stage,
            "target_id": target_id,
        }
        if bool(deep_red.get("active", False)):
            return {**base, "mode": "hard_avoidance", "reason": "route7_front_square_deep_red_takeover"}
        route_state = event_dict.get("route7_realtime_route_plan", {}) if isinstance(event_dict.get("route7_realtime_route_plan", {}), dict) else {}
        if route_state:
            base["route7_realtime_route_plan"] = self.route5_json_safe(route_state)
            if str(route_state.get("status", "") or "") == "blocked":
                return {**base, "mode": "replan", "reason": "route7_realtime_route_blocked_replan"}
            if str(route_state.get("status", "") or "") == "ok":
                return {**base, "mode": "continue_planned_route", "reason": "route7_realtime_route_clear_continue_planned_route"}
        clearance = self.route7_map_route_clearance_report(current_pose, target_pose, output_dir=output_dir)
        base["map_route_clearance"] = self.route5_json_safe(clearance)
        if clearance.get("status") == "ok" and bool(clearance.get("route_blocked", False)):
            return {**base, "mode": "replan", "reason": "route7_map_route_obstacle_replan"}
        return {**base, "mode": "continue_planned_route", "reason": "route7_soft_obstacle_not_deep_red_continue_planned_route"}

    def route7_front_square_deep_red_takeover(
        self,
        event: Dict[str, Any],
        gate: Dict[str, Any],
        *,
        rule: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_dict = event if isinstance(event, dict) else {}
        gate_dict = gate if isinstance(gate, dict) else {}
        rule_dict = rule if isinstance(rule, dict) else {}
        corridor_risks = rule_dict.get("corridor_risks") if isinstance(rule_dict.get("corridor_risks"), dict) else {}
        if not corridor_risks:
            corridor_risks = event_dict.get("or2_corridor_risks") if isinstance(event_dict.get("or2_corridor_risks"), dict) else {}
        if not corridor_risks:
            corridor_risks = gate_dict.get("or2_corridor_risks") if isinstance(gate_dict.get("or2_corridor_risks"), dict) else {}
        front = corridor_risks.get("front_center", {}) if isinstance(corridor_risks.get("front_center", {}), dict) else {}
        front_stop = self.route5_event_float(front.get("stop_fraction"), default=0.0)
        front_warning = self.route5_event_float(front.get("warning_fraction"), default=0.0)
        risk_state = str(gate_dict.get("front_risk_state", event_dict.get("or2_front_risk_state", "")) or "").strip().lower()
        must_stop = bool(gate_dict.get("must_stop", event_dict.get("or2_must_stop", False)))
        has_front_square_stats = bool(front)
        or3_source = gate_dict
        if "front_box_stop_fraction" not in or3_source:
            event_or3 = event_dict.get("or3_prediction", {}) if isinstance(event_dict.get("or3_prediction"), dict) else {}
            if "front_box_stop_fraction" in event_or3:
                or3_source = event_or3
        has_or3_stats = "front_box_stop_fraction" in (or3_source if isinstance(or3_source, dict) else {})
        if has_or3_stats:
            front_stop = self.route5_event_float(or3_source.get("front_box_stop_fraction"), default=0.0)
            front_warning = self.route5_event_float(or3_source.get("front_box_warning_fraction"), default=0.0)
            front_clearance = self.route5_event_float(or3_source.get("front_box_clearance_fraction"), default=0.0)
            threshold = self.route5_event_float(or3_source.get("projection_box_stop_threshold"), default=0.01)
            threshold = threshold if threshold > 0.0 else 0.01
            projection_box = or3_source.get("projection_box", {}) if isinstance(or3_source.get("projection_box", {}), dict) else {}
            active = bool(front_stop > threshold)
            reason = "or3_projection_box_stop_fraction" if active else "or3_projection_box_not_deep_red"
            return {
                "schema": "route7_front_square_deep_red_takeover_v2",
                "active": bool(active),
                "reason": reason,
                "front_stop_fraction": round(float(front_stop), 6),
                "front_warning_fraction": round(float(front_warning), 6),
                "front_clearance_fraction": round(float(front_clearance), 6),
                "front_stop_threshold": threshold,
                "threshold": threshold,
                "has_front_square_stats": True,
                "or3_projection_box_gate": True,
                "projection_box": self.route5_json_safe(projection_box),
                "projection_box_stop_threshold": threshold,
                "front_box_stop_fraction": round(float(front_stop), 6),
                "front_box_warning_fraction": round(float(front_warning), 6),
                "front_box_clearance_fraction": round(float(front_clearance), 6),
                "front_risk_state": risk_state,
                "must_stop": bool(must_stop),
            }
        threshold = 0.01
        active = bool(front_stop > threshold)
        reason = "front_square_stop_fraction" if active else "front_square_not_deep_red"
        return {
            "schema": "route7_front_square_deep_red_takeover_v1",
            "active": bool(active),
            "reason": reason,
            "front_stop_fraction": round(float(front_stop), 6),
            "front_warning_fraction": round(float(front_warning), 6),
            "front_stop_threshold": threshold,
            "has_front_square_stats": bool(has_front_square_stats),
            "front_risk_state": risk_state,
            "must_stop": bool(must_stop),
        }

    def route7_select_standoff_for_layer_edge(
        self,
        bbox: Dict[str, Any],
        facade: str,
        axis_value: float,
        layer_z_cm: float,
        preferred_standoff_cm: float,
        layer_record: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        preferred = max(80.0, min(1500.0, float(preferred_standoff_cm)))
        candidates: List[float] = []
        for raw in (preferred, preferred + 120.0, preferred + 240.0, preferred - 100.0):
            value = max(80.0, min(1500.0, float(raw)))
            if all(abs(value - existing) > 1.0 for existing in candidates):
                candidates.append(value)
        scored: List[Tuple[int, float, float, Dict[str, Any]]] = []
        for standoff in candidates:
            pose = self.route2_facade_pose_from_axis(bbox, facade, float(axis_value), float(standoff), float(layer_z_cm))
            score = self.route7_layer_pose_occupancy_score(layer_record, pose)
            occupied = score.get("occupied_cell_count")
            occupied_count = int(occupied) if occupied is not None else 0
            scored.append((occupied_count, abs(float(standoff) - preferred), float(standoff), {**score, "candidate_pose": pose}))
        if not scored:
            return preferred, {"status": "no_standoff_candidates"}
        occupied_count, _delta, selected, score = min(scored, key=lambda item: (item[0], item[1], item[2]))
        score["selected_occupied_cell_count"] = occupied_count
        score["preferred_standoff_cm"] = round(float(preferred), 2)
        score["selected_standoff_cm"] = round(float(selected), 2)
        return float(selected), score

    def route7_build_update_map_after_observation(
        self,
        output_dir: Optional[Path],
        *,
        facade: str = "",
        observation: Optional[Dict[str, Any]] = None,
        rgb_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out_path = Path(output_dir) if output_dir is not None else self.route7_current_map_output_dir()
        if out_path is None:
            return {"status": "skipped", "reason": "missing_route7_output_dir"}
        result: Dict[str, Any] = {"status": "skipped", "reason": "route6_update_map_build_unavailable", "output_dir": str(out_path)}
        build_map = getattr(self, "route6_update_map_build_from_pointcloud", None)
        if callable(build_map):
            result = build_map(out_path) or {}
            if not result:
                result = {"status": "empty", "reason": "no_pointcloud_for_observation_map", "output_dir": str(out_path)}
        result["schema"] = "route7_observation_map_build_v1"
        result["facade"] = str(facade or "")
        result["observation"] = self.route5_json_safe(observation or {})
        result["rgb_status"] = str((rgb_result if isinstance(rgb_result, dict) else {}).get("status", "") or "")
        result["purpose"] = "observe_house_surroundings_for_same_layer_obstacle_map"
        result["preferred_layer_key"] = self.route7_default_layer_key()
        self.route5_update_state(route7_last_observation_map_build=self.route5_json_safe(result))
        try:
            self.route6_update_map_request_window_refresh()
        except Exception:
            pass
        try:
            self.root.after(0, lambda: self.refresh_llm_route7_update_map(build_if_missing=False))
        except Exception:
            pass
        return result

    def route7_plan_facade_map_layer_scan_current(self) -> Dict[str, Any]:
        state = self.route2_selected_state()
        output_dir, facade_dir, house_id, facade = self.route2_facade_paths()
        observation = state.get("observation_point", {}) if isinstance(state.get("observation_point"), dict) else {}
        bbox = self.house_world_bbox_for_id(house_id)
        if output_dir is None or facade_dir is None:
            raise RuntimeError("missing facade output directory")
        if not house_id or not facade or not observation or not bbox:
            return {"points": [], "scan_counts": {}, "validation": {"valid": False, "reason": "missing_v7_scan_context"}}
        analysis = state.get("facade_analysis", {}) if isinstance(state.get("facade_analysis"), dict) else {}
        layer_record, layer_key, layer_z_cm, manifest = self.route7_selected_map_layer_record(output_dir)
        preferred_standoff = self._as_float_or_none(observation.get("standoff_cm"))
        scan_geometry: Dict[str, Any] = {}
        try:
            scan_geometry = self.route2_select_scan_geometry(house_id, facade, bbox, analysis or {})
            preferred_standoff = self._as_float_or_none(scan_geometry.get("standoff_cm")) or preferred_standoff
        except Exception:
            scan_geometry = {}
        if preferred_standoff is None:
            try:
                preferred_standoff = float(self.route_scan_standoff_cm())
            except Exception:
                preferred_standoff = 650.0
        safe_intervals = [
            item for item in scan_geometry.get("safe_intervals", [])
            if isinstance(item, dict)
        ] if isinstance(scan_geometry.get("safe_intervals", []), list) else []
        axis_values = self.route7_axis_values_near_house_edge(bbox, facade, safe_intervals)
        points: List[Dict[str, Any]] = []
        for idx, axis_value in enumerate(axis_values, start=1):
            standoff, occupancy_score = self.route7_select_standoff_for_layer_edge(
                bbox,
                facade,
                float(axis_value),
                float(layer_z_cm),
                float(preferred_standoff),
                layer_record,
            )
            pose = self.route2_facade_pose_from_axis(bbox, facade, float(axis_value), float(standoff), float(layer_z_cm))
            side_label = "near_min_edge" if idx == 1 else "near_max_edge"
            points.append(
                {
                    "scan_id": f"{house_id}_{facade}_{layer_key}_edge_{idx:03d}",
                    "local_scan_index": idx - 1,
                    "house_id": house_id,
                    "facade": facade,
                    "facade_id": self.route2_facade_id(house_id, facade),
                    "route_point_type": "map_layer_edge_capture",
                    "height_band": layer_key,
                    "floor_index": 1,
                    "semantic_region": side_label,
                    "target_score": str(analysis.get("target_score", "medium") or "medium"),
                    "translation_span": "map_layer_two_edge_points",
                    "rule_density": "route7_map_layer_two_points",
                    "planned_facade_sample_count": 2,
                    "axis_value": round(float(axis_value), 2),
                    "x": pose["x"],
                    "y": pose["y"],
                    "z": pose["z"],
                    "yaw_deg": pose["yaw_deg"],
                    "yaw_source": "route7_face_house_edge",
                    "standoff_cm": round(float(standoff), 2),
                    "preferred_standoff_cm": round(float(preferred_standoff), 2),
                    "scan_standoff_mode": "route7_same_layer_obstacle_aware",
                    "view_type": "route7_map_layer_edge_capture",
                    "capture_trigger": "arrive_align_hover_capture",
                    "safe_interval_index": -1,
                    "safe_interval_count": len(safe_intervals),
                    "safe_axis_min": round(min(float(value) for value in axis_values), 2),
                    "safe_axis_max": round(max(float(value) for value in axis_values), 2),
                    "route7_map_layer_key": layer_key,
                    "route7_map_layer_z_cm": round(float(layer_z_cm), 2),
                    "route7_map_guided_capture": True,
                    "route7_map_layer_point_count": int(layer_record.get("point_count", 0) or 0) if layer_record else 0,
                    "route7_map_layer_occupied_cell_count": int(layer_record.get("occupied_cell_count", 0) or 0) if layer_record else 0,
                    "route7_layer_occupancy_score": self.route5_json_safe(occupancy_score),
                    "route7_observation_purpose": "observation_points_build_obstacle_map_before_edge_capture",
                    "status": "planned",
                }
            )
        points = self.route2_order_scan_points_continuously(points, start_pose=observation)
        validation = self.scan_point_validation_report(house_id, points)
        search_plan = {
            "schema": "facade_v7_map_layer_edge_scan_plan",
            "house_id": house_id,
            "facade": facade,
            "facade_id": self.route2_facade_id(house_id, facade),
            "observation_point": observation,
            "facade_analysis": analysis,
            "scan_points": points,
            "scan_point_validation_report": validation,
            "route7_map_layer_key": layer_key,
            "route7_map_layer_z_cm": round(float(layer_z_cm), 2),
            "route7_map_manifest_path": str(manifest.get("manifest_path", "") or ""),
            "route7_map_layer_source": "selected_layer" if layer_record else "default_300cm_no_manifest",
            "route7_capture_policy": "two_points_near_house_edge_on_selected_layer",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_json_artifact(facade_dir / "facade_search_plan.json", search_plan)
        merged_points = self.route5_write_merged_scan_points(
            output_dir,
            house_id,
            policy={"source": "route7_map_layer_edge_scan", "layer_key": layer_key, "point_count": len(points)},
        )
        self.route2_update_state(facade_analysis=analysis, facade_search_plan=search_plan, facade_scan_points=points, validation_report=validation)
        self.route5_update_state(
            current_facade_scan_points=points,
            route7_last_map_layer_scan_plan={
                "facade": facade,
                "layer_key": layer_key,
                "layer_z_cm": round(float(layer_z_cm), 2),
                "point_count": len(points),
                "merged_scan_count": len(merged_points),
            },
        )
        self.route2_write_state_artifact()
        return {
            "points": points,
            "validation": validation,
            "scan_counts": {"physical_axis_sample_count": len(points), "total_capture_record_count": len(points), "yaw_supplement_record_count": 0},
            "boundary_policy": {"source": "route7_map_layer_edge_scan", "layer_key": layer_key},
        }

