from __future__ import annotations

from .common import *


class Route5HouseMemoryMixin:
    def route5_empty_house_memory(self, target_house_id: str, *, history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "schema": "route5_house_exploration_memory_v1",
            "target_house_id": str(target_house_id or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "history": history if isinstance(history, dict) else {},
            "facades": {
                facade: {
                    "facade": facade,
                    "status": "pending",
                    "attempt_count": 0,
                    "observation_attempts": [],
                    "failure_reasons": [],
                    "degraded_reasons": [],
                    "obstacle_label_conflicts": [],
                    "success_frames": [],
                    "scan_coverage": {},
                    "entrance_candidates": [],
                    "llm_decision_reasons": [],
                    "last_safe_observation_pose": {},
                    "last_updated_at": "",
                }
                for facade in ("west", "south", "east", "north")
            },
        }

    def route5_house_memory_path(self, output_dir: Path) -> Path:
        return output_dir / "house_exploration_memory.json"

    def route5_house_memory_events_path(self, output_dir: Path) -> Path:
        return output_dir / "house_exploration_memory_events.jsonl"

    def route5_cross_run_house_memory_path(self, output_dir: Path, target_house_id: str) -> Path:
        return output_dir.parent / "house_memory" / f"house_{str(target_house_id or '').strip()}.json"

    def route5_initialize_house_memory(self, output_dir: Path, target_house_id: str) -> Dict[str, Any]:
        history_path = self.route5_cross_run_house_memory_path(output_dir, target_house_id)
        history = flight.read_json_object(history_path) if history_path.is_file() else {}
        memory = self.route5_empty_house_memory(target_house_id, history=history)
        self.write_json_artifact(self.route5_house_memory_path(output_dir), memory)
        agenda = {facade: item.get("status", "pending") for facade, item in memory["facades"].items()}
        self.route5_update_state(house_memory=self.route5_house_memory_summary(memory), mandatory_facade_agenda=agenda)
        self.route5_write_state_artifact()
        self.append_jsonl(
            self.route5_house_memory_events_path(output_dir),
            {
                "event_type": "memory_initialized",
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                "target_house_id": str(target_house_id or ""),
                "history_loaded": bool(history),
            },
        )
        return memory

    def route5_read_house_memory(self, output_dir: Path, target_house_id: str) -> Dict[str, Any]:
        path = self.route5_house_memory_path(output_dir)
        if path.is_file():
            payload = flight.read_json_object(path)
            if isinstance(payload, dict) and isinstance(payload.get("facades"), dict):
                return payload
        return self.route5_initialize_house_memory(output_dir, target_house_id)

    def route5_house_memory_summary(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        facades = memory.get("facades", {}) if isinstance(memory, dict) and isinstance(memory.get("facades"), dict) else {}
        return {
            "target_house_id": str(memory.get("target_house_id", "") or "") if isinstance(memory, dict) else "",
            "facade_status": {str(facade): str(item.get("status", "") or "") for facade, item in facades.items() if isinstance(item, dict)},
            "updated_at": str(memory.get("updated_at", "") or "") if isinstance(memory, dict) else "",
        }

    def route5_append_unique(self, values: List[Any], item: Any) -> List[Any]:
        safe_item = self.route5_json_safe(item)
        encoded = json.dumps(safe_item, sort_keys=True, ensure_ascii=False)
        seen = {json.dumps(self.route5_json_safe(value), sort_keys=True, ensure_ascii=False) for value in values}
        if encoded not in seen:
            values.append(safe_item)
        return values

    def route5_update_house_memory(
        self,
        output_dir: Path,
        target_house_id: str,
        facade: str,
        *,
        status: str,
        reason: str = "",
        observation_attempt: Optional[Dict[str, Any]] = None,
        nav_result: Optional[Dict[str, Any]] = None,
        frame_id: Optional[int] = None,
        scan_coverage: Optional[Dict[str, Any]] = None,
        entrance_candidates: Optional[List[Any]] = None,
        llm_decision_reason: str = "",
        obstacle_label_conflict: Optional[Dict[str, Any]] = None,
        safe_observation_pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        facade = str(facade or "").strip().lower()
        memory = self.route5_read_house_memory(output_dir, target_house_id)
        facades = memory.setdefault("facades", {})
        item = facades.setdefault(facade, self.route5_empty_house_memory(target_house_id)["facades"].get(facade, {"facade": facade}))
        previous_status = str(item.get("status", "pending") or "pending")
        item["status"] = str(status or previous_status)
        item["attempt_count"] = int(item.get("attempt_count", 0) or 0) + 1
        item["last_reason"] = str(reason or item.get("last_reason", "") or "")
        item["last_updated_at"] = datetime.now().isoformat(timespec="milliseconds")
        if observation_attempt:
            self.route5_append_unique(item.setdefault("observation_attempts", []), observation_attempt)
        if reason and str(status).lower() in {"soft_blocked", "failed_blocked"}:
            self.route5_append_unique(item.setdefault("failure_reasons", []), reason)
        if reason and str(status).lower() == "degraded_completed":
            self.route5_append_unique(item.setdefault("degraded_reasons", []), reason)
        if nav_result:
            item["last_navigation_result"] = self.route5_json_safe(nav_result)
        if frame_id is not None:
            self.route5_append_unique(item.setdefault("success_frames", []), int(frame_id))
        if scan_coverage:
            item["scan_coverage"] = self.route5_json_safe(scan_coverage)
        if entrance_candidates:
            item["entrance_candidates"] = self.route5_json_safe(entrance_candidates)
        if llm_decision_reason:
            self.route5_append_unique(item.setdefault("llm_decision_reasons", []), llm_decision_reason)
        if obstacle_label_conflict:
            self.route5_append_unique(item.setdefault("obstacle_label_conflicts", []), obstacle_label_conflict)
        if safe_observation_pose:
            item["last_safe_observation_pose"] = self.route5_json_safe(safe_observation_pose)
        memory["updated_at"] = datetime.now().isoformat(timespec="seconds")
        facades[facade] = item
        self.write_json_artifact(self.route5_house_memory_path(output_dir), memory)
        cross_path = self.route5_cross_run_house_memory_path(output_dir, target_house_id)
        cross_payload = dict(memory)
        cross_payload["history"] = {}
        self.write_json_artifact(cross_path, cross_payload)
        event = {
            "event_type": "facade_memory_update",
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "target_house_id": str(target_house_id or ""),
            "facade": facade,
            "previous_status": previous_status,
            "status": item["status"],
            "reason": reason,
        }
        self.append_jsonl(self.route5_house_memory_events_path(output_dir), event)
        agenda = {name: str(data.get("status", "pending") or "pending") for name, data in facades.items() if isinstance(data, dict)}
        self.route5_update_state(house_memory=self.route5_house_memory_summary(memory), mandatory_facade_agenda=agenda)
        self.route5_write_state_artifact()
        return memory

