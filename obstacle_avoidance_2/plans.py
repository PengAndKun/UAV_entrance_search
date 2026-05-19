from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PLAN_SCHEMA_VERSION = 2
DEFAULT_PLAN_FILENAME = "obstacle_avoidance_2_plans.json"
DEFAULT_PROJECT_ID = "route_obstacle_collection_v2"
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
    {"environment_id": DEFAULT_ENVIRONMENT_ID, "name": "Default Unreal scene", "description": "Current UE scene metadata label."},
    {"environment_id": "tree_trunk_or_pole", "name": "Tree trunk / pole", "description": "Narrow vertical obstacle; prefer lateral offset."},
    {
        "environment_id": "tree_canopy_or_cluster",
        "name": "Tree canopy / cluster",
        "description": "Wider tree obstacle; prefer lateral offset, climb only when side space is poor.",
    },
    {"environment_id": "tree_or_pole", "name": "Tree / pole (legacy)", "description": "Legacy label; runner splits it by observed width."},
    {"environment_id": "fence_or_rail", "name": "Fence / rail", "description": "Crossing obstacle; likely climb-over behavior."},
    {"environment_id": "building_or_roof", "name": "Building / roof", "description": "Large obstacle; likely overflight or route change."},
    {"environment_id": "mixed_obstacles", "name": "Mixed obstacles", "description": "Mixed hard cases."},
]


DEFAULT_METHODS: List[Dict[str, Any]] = [
    {
        "method_id": "geometry_rule_v0",
        "name": "Point-cloud geometry rule v0",
        "runnable": True,
        "description": "Implemented collector: straight route following plus v0 point-cloud safety shield.",
    },
    {
        "method_id": "pointcloud_direction_rule",
        "name": "Point-cloud direction rule",
        "runnable": True,
        "description": "Executable point-cloud rule: vertical obstacles choose left/right clearance; low or overhead obstacles choose vertical/lateral clearance.",
    },
    {
        "method_id": "distance_rule",
        "name": "Distance threshold rule",
        "runnable": True,
        "description": "Executable baseline: front distance threshold plus left/right/up clearance choice.",
    },
    {
        "method_id": "no_avoidance",
        "name": "No avoidance",
        "runnable": True,
        "description": "Executable ablation: straight start-goal route following without obstacle recovery.",
    },
    {
        "method_id": "route_follow",
        "name": "Route follow only",
        "runnable": True,
        "description": "Executable baseline: straight route following with a hard safety hold when front is blocked.",
    },
    {
        "method_id": "vlm_pointcloud_semantic_v1",
        "name": "VLM + point-cloud semantic v1",
        "runnable": False,
        "description": "Planned semantic policy; registered for dataset planning but not executable yet.",
    },
]

EXECUTABLE_METHOD_IDS = {"geometry_rule_v0", "pointcloud_direction_rule", "distance_rule", "no_avoidance", "route_follow"}


def timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def sanitize_id(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or default


def coerce_pose(raw: Any) -> List[float]:
    if isinstance(raw, str):
        normalized = raw.replace("\uff0c", ",")
        parts = [part for part in re.split(r"[\s,;]+", normalized) if part]
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
    episodes: List[Dict[str, Any]] = []
    for row in DEFAULT_ROUTE_EPISODES:
        episode_id = str(row["episode_id"])
        episodes.append(
            {
                "episode_id": episode_id,
                "enabled": True,
                "start_pose": list(row["start_pose"]),
                "goal_pose": list(row["goal_pose"]),
                "scenario_id": f"oa2_route_{episode_id}",
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
        "environments": deepcopy(DEFAULT_ENVIRONMENTS),
        "methods": deepcopy(DEFAULT_METHODS),
        "projects": [
            {
                "project_id": DEFAULT_PROJECT_ID,
                "name": "OA2 default route obstacle collection",
                "environment_id": DEFAULT_ENVIRONMENT_ID,
                "default_method": DEFAULT_METHOD_ID,
                "episodes": make_default_episodes(),
                "experiment_defaults": {
                    "stage": "route_episode_2",
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
        raise ValueError("plan JSON must be an object")
    normalized = deepcopy(data)
    normalized.setdefault("version", PLAN_SCHEMA_VERSION)
    normalized.setdefault("updated_at", timestamp())
    normalized.setdefault("environments", deepcopy(DEFAULT_ENVIRONMENTS))
    normalized.setdefault("methods", deepcopy(DEFAULT_METHODS))
    if not isinstance(normalized.get("methods"), list):
        normalized["methods"] = []
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
        project["project_id"] = sanitize_id(project.get("project_id"), f"project_{project_index:02d}")
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
            episode.setdefault("scenario_id", f"oa2_route_{episode['episode_id']}")
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


def validate_plan_episode(episode: Dict[str, Any]) -> List[str]:
    if not isinstance(episode, dict):
        return ["episode must be an object"]
    errors: List[str] = []
    if not str(episode.get("episode_id", "")).strip():
        errors.append("episode_id is required")
    for key in ("start_pose", "goal_pose"):
        try:
            coerce_pose(episode.get(key))
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    return errors


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
                "source": "obstacle_avoidance_2_plan_export",
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
