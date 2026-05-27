from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


PLAN_SCHEMA_VERSION = 1
DEFAULT_PLAN_FILENAME = "obstacle_avoidance_3_plans.json"
DEFAULT_PROJECT_ID = "route_obstacle_or2_collection_v1"
DEFAULT_ENVIRONMENT_ID = "default_unreal_scene"
DEFAULT_METHOD_ID = "obstacle_representation_direction_rule_v1"

DEFAULT_ROUTE_EPISODES: List[Dict[str, Any]] = [
    {"episode_id": "E01", "start_pose": [325.5, 526.0, 211.2, 0.2], "goal_pose": [734.3, 552.0, 211.2, 0.2]},
    {"episode_id": "E02", "start_pose": [1298.8, 513.8, 258.2, 0.3], "goal_pose": [2048.9, 443.8, 259.7, 0.3]},
    {"episode_id": "E03", "start_pose": [2475.1, 1129.3, 175.3, 74.2], "goal_pose": [2883.4, 1852.3, 162.0, 65.4]},
    {"episode_id": "E04", "start_pose": [3546.2, 2361.3, 162.0, -55.8], "goal_pose": [4425.2, 1100.3, 200.0, -55.6]},
    {"episode_id": "E05", "start_pose": [4870.3, -655.2, 257.9, -105.3], "goal_pose": [3820.9, -2436.0, 267.9, 123.3]},
    {"episode_id": "E06", "start_pose": [1875.2, -1390.4, 223.5, -89.8], "goal_pose": [1877.1, 1911.1, 223.5, -89.8]},
    {"episode_id": "E07", "start_pose": [1916.6, -2605.6, 223.5, 178.0], "goal_pose": [1429.5, -2646.4, 223.5, 178.0]},
    {"episode_id": "E08", "start_pose": [-549.9, -2703.1, 262.5, 93.0], "goal_pose": [-589.8, -1948.0, 262.4, 93.0]},
    {"episode_id": "E09", "start_pose": [-497.1, 1640.1, 195.2, 85.6], "goal_pose": [-462.9, 2084.6, 201.2, 85.6]},
    {"episode_id": "E10", "start_pose": [1960.5, 1975.9, 153.1, -87.4], "goal_pose": [1983.1, 1474.2, 153.1, -87.4]},
]

DEFAULT_ENVIRONMENTS: List[Dict[str, Any]] = [
    {"environment_id": DEFAULT_ENVIRONMENT_ID, "name": "Default Unreal scene"},
    {"environment_id": "tree_trunk_or_pole", "name": "Tree trunk / pole"},
    {"environment_id": "fence_or_rail", "name": "Fence / rail"},
    {"environment_id": "building_or_roof", "name": "Building / roof"},
    {"environment_id": "mixed_obstacles", "name": "Mixed obstacles"},
]

DEFAULT_METHODS: List[Dict[str, Any]] = [
    {
        "method_id": DEFAULT_METHOD_ID,
        "name": "OR2 risk-region direction rule v1",
        "runnable": True,
        "description": "Uses OR2 RGB-D risk masks. Deep red blocks only that corridor; choose clear left/right/up before backoff.",
    },
]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def sanitize_id(value: Any, default: str, *, max_len: int = 80) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    text = text or default
    if max_len > 0 and len(text) > max_len:
        text = text[:max_len].rstrip("._-") or default
    return text


def coerce_pose(raw: Any) -> List[float]:
    if isinstance(raw, str):
        parts = [part for part in re.split(r"[\s,;]+", raw.replace("\uff0c", ",")) if part]
    elif isinstance(raw, Iterable):
        parts = list(raw)
    else:
        raise ValueError("pose must be a list or comma-separated string")
    if len(parts) != 4:
        raise ValueError(f"pose must contain 4 values [x, y, z, yaw_deg], got {len(parts)}")
    values: List[float] = []
    for part in parts:
        value = float(part)
        if not math.isfinite(value):
            raise ValueError("pose values must be finite")
        values.append(value)
    return values


def make_default_episodes() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in DEFAULT_ROUTE_EPISODES:
        episode_id = str(item["episode_id"])
        rows.append(
            {
                "episode_id": episode_id,
                "enabled": True,
                "start_pose": list(item["start_pose"]),
                "goal_pose": list(item["goal_pose"]),
                "scenario_id": f"oa3_route_{episode_id}",
                "environment_id": DEFAULT_ENVIRONMENT_ID,
                "method": DEFAULT_METHOD_ID,
                "obstacle_hint": "unknown",
                "operator_note": "",
            }
        )
    return rows


def _active_project(data: Dict[str, Any]) -> Dict[str, Any]:
    projects = data.get("projects") if isinstance(data, dict) else []
    if not isinstance(projects, list):
        return {}
    active_id = str(data.get("active_project_id", "") or "")
    for project in projects:
        if isinstance(project, dict) and str(project.get("project_id", "")) == active_id:
            return project
    for project in projects:
        if isinstance(project, dict):
            return project
    return {}


def oa2_episodes_for_oa3(oa2_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    project = _active_project(oa2_data)
    raw_episodes = project.get("episodes") if isinstance(project, dict) else []
    if not isinstance(raw_episodes, list):
        return []
    rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_episodes, start=1):
        if not isinstance(raw, dict):
            continue
        start = raw.get("start_pose") or raw.get("start")
        goal = raw.get("goal_pose") or raw.get("goal")
        try:
            start_pose = coerce_pose(start)
            goal_pose = coerce_pose(goal)
        except Exception:
            continue
        episode_id = str(raw.get("episode_id") or f"E{index:02d}")
        rows.append(
            {
                "episode_id": episode_id,
                "enabled": bool(raw.get("enabled", True)),
                "start_pose": start_pose,
                "goal_pose": goal_pose,
                "scenario_id": str(raw.get("scenario_id") or f"oa3_route_{episode_id}"),
                "environment_id": str(raw.get("environment_id") or project.get("environment_id", DEFAULT_ENVIRONMENT_ID)),
                "method": DEFAULT_METHOD_ID,
                "obstacle_hint": str(raw.get("obstacle_hint") or "unknown"),
                "operator_note": str(raw.get("operator_note") or ""),
            }
        )
    return rows


def sync_oa2_plan_into_oa3(oa3_data: Dict[str, Any], oa2_data: Dict[str, Any]) -> Dict[str, Any]:
    episodes = oa2_episodes_for_oa3(oa2_data)
    if not episodes:
        return normalize_plans(oa3_data)
    synced = normalize_plans(oa3_data)
    oa2_project = _active_project(oa2_data)
    projects = synced.setdefault("projects", [])
    if not projects:
        projects.append(make_default_plans()["projects"][0])
    project = projects[0]
    project["project_id"] = DEFAULT_PROJECT_ID
    project["name"] = f"OA3 synced from {oa2_project.get('project_id', 'OA2')}"
    project["environment_id"] = str(oa2_project.get("environment_id") or DEFAULT_ENVIRONMENT_ID)
    project["default_method"] = DEFAULT_METHOD_ID
    project["episodes"] = episodes
    project.setdefault("experiment_defaults", make_default_plans()["projects"][0]["experiment_defaults"])
    project["experiment_defaults"]["method"] = DEFAULT_METHOD_ID
    synced["active_project_id"] = DEFAULT_PROJECT_ID
    synced["updated_at"] = timestamp()
    return normalize_plans(synced)


def make_default_plans() -> Dict[str, Any]:
    return {
        "version": PLAN_SCHEMA_VERSION,
        "updated_at": timestamp(),
        "active_project_id": DEFAULT_PROJECT_ID,
        "environments": deepcopy(DEFAULT_ENVIRONMENTS),
        "methods": deepcopy(DEFAULT_METHODS),
        "projects": [
            {
                "project_id": DEFAULT_PROJECT_ID,
                "name": "OA3 default OR2 risk-region collection",
                "environment_id": DEFAULT_ENVIRONMENT_ID,
                "default_method": DEFAULT_METHOD_ID,
                "episodes": make_default_episodes(),
                "experiment_defaults": {
                    "stage": "route_episode_3",
                    "method": DEFAULT_METHOD_ID,
                    "reach_tol_cm": 180.0,
                    "interval_s": 0.5,
                    "max_ticks_per_episode": 220,
                },
            }
        ],
    }


def normalize_plans(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    default = make_default_plans()
    normalized = deepcopy(data)
    normalized.setdefault("version", PLAN_SCHEMA_VERSION)
    normalized.setdefault("active_project_id", DEFAULT_PROJECT_ID)
    if not isinstance(normalized.get("environments"), list) or not normalized.get("environments"):
        normalized["environments"] = deepcopy(DEFAULT_ENVIRONMENTS)
    if not isinstance(normalized.get("methods"), list) or not normalized.get("methods"):
        normalized["methods"] = deepcopy(DEFAULT_METHODS)
    if not isinstance(normalized.get("projects"), list) or not normalized["projects"]:
        normalized["projects"] = deepcopy(default["projects"])
    for project in normalized.get("projects", []):
        if not isinstance(project, dict):
            continue
        project.setdefault("project_id", DEFAULT_PROJECT_ID)
        project.setdefault("name", "OA3 Route6_entrance_search obstacle collection")
        project.setdefault("environment_id", DEFAULT_ENVIRONMENT_ID)
        project.setdefault("default_method", DEFAULT_METHOD_ID)
        if not isinstance(project.get("episodes"), list) or not project.get("episodes"):
            project["episodes"] = make_default_episodes()
        for episode in project["episodes"]:
            if not isinstance(episode, dict):
                continue
            episode.setdefault("enabled", True)
            episode.setdefault("scenario_id", f"oa3_route_{episode.get('episode_id', 'episode')}")
            episode.setdefault("environment_id", project.get("environment_id", DEFAULT_ENVIRONMENT_ID))
            episode.setdefault("method", project.get("default_method", DEFAULT_METHOD_ID))
            episode.setdefault("obstacle_hint", "unknown")
            episode.setdefault("operator_note", "")
            episode["start_pose"] = coerce_pose(episode.get("start_pose", [0, 0, 100, 0]))
            episode["goal_pose"] = coerce_pose(episode.get("goal_pose", [0, 0, 100, 0]))
        project.setdefault("experiment_defaults", deepcopy(default["projects"][0]["experiment_defaults"]))
    normalized["updated_at"] = str(normalized.get("updated_at") or timestamp())
    return normalized


def load_plans(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        data = make_default_plans()
        save_plans(path, data)
        return data
    return normalize_plans(json.loads(path.read_text(encoding="utf-8")))


def save_plans(path: Path, data: Dict[str, Any]) -> None:
    payload = normalize_plans(data)
    payload["updated_at"] = timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
