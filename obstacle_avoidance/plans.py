from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PLAN_SCHEMA_VERSION = 1
DEFAULT_PLAN_FILENAME = "obstacle_avoidance_plans.json"
DEFAULT_PROJECT_ID = "default_route_episodes"
DEFAULT_ENVIRONMENT_ID = "default_unreal_scene"
DEFAULT_METHOD_ID = "geometry_rule_v0"


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
    {
        "environment_id": DEFAULT_ENVIRONMENT_ID,
        "name": "Default Unreal scene",
        "description": "Metadata label for the current UE obstacle-avoidance scene.",
    },
    {
        "environment_id": "tree_trunk_or_pole",
        "name": "Tree trunk / pole obstacle",
        "description": "Semantic plan label for narrow vertical objects that should prefer lateral offsets.",
    },
    {
        "environment_id": "tree_canopy_or_cluster",
        "name": "Tree canopy / cluster obstacle",
        "description": "Semantic plan label for wider tree obstacles that may need lateral offsets or climb only when side space is poor.",
    },
    {
        "environment_id": "tree_or_pole",
        "name": "Tree / pole obstacle (legacy)",
        "description": "Legacy semantic label; the runner now splits it by observed width.",
    },
    {
        "environment_id": "fence_or_rail",
        "name": "Fence / rail obstacle",
        "description": "Semantic plan label for obstacles that may require climb-over behavior.",
    },
    {
        "environment_id": "building_or_roof",
        "name": "Building / roof obstacle",
        "description": "Semantic plan label for large structures that require Route6_entrance_search changes or overflight.",
    },
    {
        "environment_id": "mixed_obstacles",
        "name": "Mixed obstacles",
        "description": "Metadata label for mixed hard cases.",
    },
]


DEFAULT_METHODS: List[Dict[str, Any]] = [
    {
        "method_id": "geometry_rule_v0",
        "name": "Point-cloud geometry rule v0",
        "runnable": True,
        "description": "Implemented Route6_entrance_search episode baseline using point-cloud geometry and safety scoring.",
    },
    {
        "method_id": "distance_rule",
        "name": "Distance threshold rule",
        "runnable": False,
        "description": "Experiment label only until a Route6_entrance_search-episode runner branch is implemented.",
    },
    {
        "method_id": "no_avoidance",
        "name": "No avoidance",
        "runnable": False,
        "description": "Experiment label only; the current Route6_entrance_search runner does not expose this behavior switch.",
    },
    {
        "method_id": "route_follow",
        "name": "Route follow only",
        "runnable": False,
        "description": "Experiment label only; the current Route6_entrance_search runner still applies the v0 safety shield.",
    },
    {
        "method_id": "vlm_pointcloud_semantic_v1",
        "name": "VLM + point-cloud semantic v1",
        "runnable": False,
        "description": "Planned semantic method; not executable until VLM inference and policy logic are added.",
    },
]


RUNNABLE_METHOD_IDS = {
    str(item.get("method_id"))
    for item in DEFAULT_METHODS
    if bool(item.get("runnable"))
}


def utc_timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def sanitize_id(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or default


def coerce_pose(raw: Any) -> List[float]:
    if isinstance(raw, str):
        parts = [part for part in re.split(r"[\s,;]+", raw.replace("，", ",")) if part]
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
    for row in DEFAULT_ROUTE_EPISODES:
        episode_id = str(row["episode_id"])
        rows.append(
            {
                "episode_id": episode_id,
                "enabled": True,
                "start_pose": list(row["start_pose"]),
                "goal_pose": list(row["goal_pose"]),
                "scenario_id": f"route_episode_{episode_id}",
                "environment_id": DEFAULT_ENVIRONMENT_ID,
                "method": DEFAULT_METHOD_ID,
                "obstacle_hint": "unknown",
                "operator_note": "",
            }
        )
    return rows


def make_default_plans() -> Dict[str, Any]:
    return {
        "version": PLAN_SCHEMA_VERSION,
        "updated_at": utc_timestamp(),
        "active_project_id": DEFAULT_PROJECT_ID,
        "methods": deepcopy(DEFAULT_METHODS),
        "environments": deepcopy(DEFAULT_ENVIRONMENTS),
        "projects": [
            {
                "project_id": DEFAULT_PROJECT_ID,
                "name": "Default 10 Route6_entrance_search episodes",
                "environment_id": DEFAULT_ENVIRONMENT_ID,
                "default_method": DEFAULT_METHOD_ID,
                "episodes": make_default_episodes(),
                "experiment_defaults": {
                    "stage": "route_episode",
                    "launch_sleep": 5,
                    "interval_s": 5.0,
                    "width": 320,
                    "height": 240,
                    "reach_tol_cm": 180,
                    "max_ticks_per_episode": 220,
                    "continue_on_failure": True,
                },
            }
        ],
    }


def normalize_plan_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("plans JSON must be an object")
    normalized = deepcopy(data)
    normalized.setdefault("version", PLAN_SCHEMA_VERSION)
    normalized.setdefault("updated_at", utc_timestamp())
    normalized.setdefault("methods", deepcopy(DEFAULT_METHODS))
    normalized.setdefault("environments", deepcopy(DEFAULT_ENVIRONMENTS))
    projects = normalized.get("projects")
    if not isinstance(projects, list) or not projects:
        normalized["projects"] = make_default_plans()["projects"]
    for index, project in enumerate(normalized["projects"], start=1):
        if not isinstance(project, dict):
            raise ValueError(f"project #{index} must be an object")
        project.setdefault("project_id", f"project_{index:02d}")
        project["project_id"] = sanitize_id(project.get("project_id"), f"project_{index:02d}")
        project.setdefault("name", project["project_id"])
        project.setdefault("environment_id", DEFAULT_ENVIRONMENT_ID)
        project.setdefault("default_method", DEFAULT_METHOD_ID)
        project.setdefault("experiment_defaults", {})
        project.setdefault("episodes", [])
        if not isinstance(project["episodes"], list):
            raise ValueError(f"project {project['project_id']} episodes must be a list")
        for episode_index, episode in enumerate(project["episodes"], start=1):
            if not isinstance(episode, dict):
                raise ValueError(f"project {project['project_id']} episode #{episode_index} must be an object")
            episode.setdefault("episode_id", f"E{episode_index:02d}")
            episode.setdefault("enabled", True)
            episode.setdefault("scenario_id", f"route_episode_{episode['episode_id']}")
            episode.setdefault("environment_id", project.get("environment_id", DEFAULT_ENVIRONMENT_ID))
            episode.setdefault("method", project.get("default_method", DEFAULT_METHOD_ID))
            episode.setdefault("obstacle_hint", "unknown")
            episode.setdefault("operator_note", "")
    active_project_id = normalized.get("active_project_id")
    project_ids = {str(project.get("project_id")) for project in normalized["projects"]}
    if str(active_project_id) not in project_ids:
        normalized["active_project_id"] = str(normalized["projects"][0]["project_id"])
    return normalized


def load_plans(path: Path | str) -> Dict[str, Any]:
    plan_path = Path(path)
    if not plan_path.exists():
        return make_default_plans()
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return normalize_plan_data(data)


def save_plans(path: Path | str, data: Dict[str, Any]) -> None:
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_plan_data(data)
    payload["updated_at"] = utc_timestamp()
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_plan_episode(episode: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(episode, dict):
        return ["episode must be an object"]
    if not str(episode.get("episode_id", "")).strip():
        errors.append("episode_id is required")
    for key in ("start_pose", "goal_pose"):
        try:
            coerce_pose(episode.get(key))
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    return errors


def project_by_id(data: Dict[str, Any], project_id: str) -> Dict[str, Any]:
    for project in data.get("projects", []):
        if str(project.get("project_id")) == str(project_id):
            return project
    raise KeyError(f"project not found: {project_id}")


def method_is_runnable(data: Dict[str, Any], method_id: str) -> bool:
    for method in data.get("methods", []):
        if str(method.get("method_id")) == str(method_id):
            return bool(method.get("runnable"))
    return method_id in RUNNABLE_METHOD_IDS


def selected_episode_rows(project: Dict[str, Any], episode_ids: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    wanted = {str(value) for value in (episode_ids or []) if str(value).strip()}
    rows: List[Dict[str, Any]] = []
    for episode in project.get("episodes", []):
        episode_id = str(episode.get("episode_id", ""))
        if wanted and episode_id not in wanted:
            continue
        if not wanted and not bool(episode.get("enabled", True)):
            continue
        errors = validate_plan_episode(episode)
        if errors:
            raise ValueError(f"{episode_id or '<unnamed>'}: {'; '.join(errors)}")
        rows.append(deepcopy(episode))
    return rows


def export_selected_episodes(project: Dict[str, Any], episode_ids: Sequence[str] | None, path: Path | str) -> Path:
    rows = selected_episode_rows(project, episode_ids)
    if not rows:
        raise ValueError("no enabled or selected episodes to export")
    export_rows = []
    for row in rows:
        export_rows.append(
            {
                "episode_id": str(row.get("episode_id")),
                "start_pose": coerce_pose(row.get("start_pose")),
                "goal_pose": coerce_pose(row.get("goal_pose")),
                "scenario_id": row.get("scenario_id", ""),
                "environment_id": row.get("environment_id", project.get("environment_id", DEFAULT_ENVIRONMENT_ID)),
                "method": row.get("method", project.get("default_method", DEFAULT_METHOD_ID)),
                "obstacle_hint": row.get("obstacle_hint", "unknown"),
                "operator_note": row.get("operator_note", ""),
            }
        )
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps(
            {
                "project_id": project.get("project_id", ""),
                "project_name": project.get("name", ""),
                "environment_id": project.get("environment_id", DEFAULT_ENVIRONMENT_ID),
                "default_method": project.get("default_method", DEFAULT_METHOD_ID),
                "episodes": export_rows,
                "exported_at": utc_timestamp(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return export_path
