from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from obstacle_avoidance_2.plans import (
    DEFAULT_ENVIRONMENTS as OA2_DEFAULT_ENVIRONMENTS,
    DEFAULT_ROUTE_EPISODES,
    coerce_pose,
    sanitize_id,
    validate_plan_episode,
)


PLAN_SCHEMA_VERSION = 1
DEFAULT_PLAN_FILENAME = "obstacle_avoidance_llm_plans.json"
DEFAULT_PROJECT_ID = "route_obstacle_llm_collection_v1"
DEFAULT_ENVIRONMENT_ID = "default_unreal_scene"
LLM_DIRECT_METHOD_ID = "llm_direct_control_v1"
LLM_STRATEGY_METHOD_ID = "llm_strategy_pointcloud_rule_v1"
DEFAULT_METHOD_ID = LLM_STRATEGY_METHOD_ID

DEFAULT_METHODS: List[Dict[str, Any]] = [
    {
        "method_id": LLM_DIRECT_METHOD_ID,
        "name": "LLM direct control v1",
        "runnable": True,
        "description": "Per-tick LLM action policy using RGB, point-cloud summary, and relative target. Safety shield is always applied.",
    },
    {
        "method_id": LLM_STRATEGY_METHOD_ID,
        "name": "LLM strategy + point-cloud rule v1",
        "runnable": True,
        "description": "LLM classifies environment and strategy once at start, then existing pointcloud_direction_rule executes the route.",
    },
    {
        "method_id": "pointcloud_direction_rule",
        "name": "Point-cloud direction rule reference",
        "runnable": False,
        "description": "Reference OA2 method label; use OA2 window to run the pure rule baseline.",
    },
    {
        "method_id": "geometry_rule_v0",
        "name": "Geometry rule v0 reference",
        "runnable": False,
        "description": "Reference OA2 method label.",
    },
    {
        "method_id": "distance_rule",
        "name": "Distance rule reference",
        "runnable": False,
        "description": "Reference OA2 method label.",
    },
    {
        "method_id": "no_avoidance",
        "name": "No avoidance reference",
        "runnable": False,
        "description": "Reference OA2 method label.",
    },
    {
        "method_id": "route_follow",
        "name": "Route follow reference",
        "runnable": False,
        "description": "Reference OA2 method label.",
    },
]

EXECUTABLE_METHOD_IDS = {LLM_DIRECT_METHOD_ID, LLM_STRATEGY_METHOD_ID}


def timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def make_default_episodes() -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for row in DEFAULT_ROUTE_EPISODES:
        episode_id = str(row["episode_id"])
        episodes.append(
            {
                "episode_id": episode_id,
                "enabled": True,
                "start_pose": list(row["start_pose"]),
                "goal_pose": list(row["goal_pose"]),
                "scenario_id": f"oa_llm_route_{episode_id}",
                "environment_id": DEFAULT_ENVIRONMENT_ID,
                "method": DEFAULT_METHOD_ID,
                "obstacle_hint": "unknown",
                "operator_note": "",
            }
        )
    return episodes


def make_default_plans() -> Dict[str, Any]:
    return {
        "version": PLAN_SCHEMA_VERSION,
        "updated_at": timestamp(),
        "active_project_id": DEFAULT_PROJECT_ID,
        "environments": deepcopy(OA2_DEFAULT_ENVIRONMENTS),
        "methods": deepcopy(DEFAULT_METHODS),
        "projects": [
            {
                "project_id": DEFAULT_PROJECT_ID,
                "name": "OA-LLM default route obstacle collection",
                "environment_id": DEFAULT_ENVIRONMENT_ID,
                "default_method": DEFAULT_METHOD_ID,
                "episodes": make_default_episodes(),
                "experiment_defaults": {
                    "stage": "route_episode_llm",
                    "interval_s": 5.0,
                    "reach_tol_cm": 180,
                    "max_ticks_per_episode": 220,
                    "continue_on_failure": True,
                },
            }
        ],
    }


def normalize_plan_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("plan JSON must be an object")
    normalized = deepcopy(data)
    normalized.setdefault("version", PLAN_SCHEMA_VERSION)
    normalized.setdefault("updated_at", timestamp())
    normalized.setdefault("environments", deepcopy(OA2_DEFAULT_ENVIRONMENTS))
    normalized.setdefault("methods", [])
    existing_methods = {str(item.get("method_id", "")) for item in normalized.get("methods", []) if isinstance(item, dict)}
    for method in DEFAULT_METHODS:
        method_id = str(method.get("method_id", ""))
        if method_id not in existing_methods:
            normalized["methods"].append(deepcopy(method))
            existing_methods.add(method_id)
    for method in normalized.get("methods", []):
        if isinstance(method, dict) and str(method.get("method_id", "")) in EXECUTABLE_METHOD_IDS:
            method["runnable"] = True
    if not isinstance(normalized.get("projects"), list) or not normalized["projects"]:
        normalized["projects"] = make_default_plans()["projects"]
    for project_index, project in enumerate(normalized["projects"], start=1):
        if not isinstance(project, dict):
            raise ValueError(f"project #{project_index} must be an object")
        project["project_id"] = sanitize_id(project.get("project_id"), f"llm_project_{project_index:02d}")
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
            episode.setdefault("scenario_id", f"oa_llm_route_{episode['episode_id']}")
            episode.setdefault("environment_id", project.get("environment_id", DEFAULT_ENVIRONMENT_ID))
            episode.setdefault("method", project.get("default_method", DEFAULT_METHOD_ID))
            episode.setdefault("obstacle_hint", "unknown")
            episode.setdefault("operator_note", "")
    project_ids = {str(project.get("project_id")) for project in normalized["projects"]}
    if str(normalized.get("active_project_id")) not in project_ids:
        normalized["active_project_id"] = str(normalized["projects"][0]["project_id"])
    return normalized


def load_plans(path: Path | str) -> Dict[str, Any]:
    plan_path = Path(path)
    if not plan_path.exists():
        return make_default_plans()
    return normalize_plan_data(json.loads(plan_path.read_text(encoding="utf-8")))


def save_plans(path: Path | str, data: Dict[str, Any]) -> None:
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_plan_data(data)
    payload["updated_at"] = timestamp()
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def project_by_id(data: Dict[str, Any], project_id: str) -> Dict[str, Any]:
    for project in data.get("projects", []):
        if str(project.get("project_id")) == str(project_id):
            return project
    raise KeyError(f"project not found: {project_id}")


def method_is_runnable(data: Dict[str, Any], method_id: str) -> bool:
    if str(method_id) in EXECUTABLE_METHOD_IDS:
        return True
    for method in data.get("methods", []):
        if str(method.get("method_id")) == str(method_id):
            return bool(method.get("runnable"))
    return False


def selected_episode_rows(project: Dict[str, Any], episode_ids: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    wanted = {str(item) for item in (episode_ids or []) if str(item).strip()}
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
        raise ValueError("no selected or enabled episodes to export")
    episodes: List[Dict[str, Any]] = []
    for row in rows:
        episodes.append(
            {
                "episode_id": str(row.get("episode_id", "")),
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
                "source": "obstacle_avoidance_llm_plan_export",
                "project_id": project.get("project_id", ""),
                "project_name": project.get("name", ""),
                "environment_id": project.get("environment_id", DEFAULT_ENVIRONMENT_ID),
                "default_method": project.get("default_method", DEFAULT_METHOD_ID),
                "episodes": episodes,
                "exported_at": timestamp(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return export_path
