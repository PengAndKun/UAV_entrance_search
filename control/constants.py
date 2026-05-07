from __future__ import annotations

from typing import Any, Dict

DEFAULT_MAP_CONFIG_PATH = "assets/overhead_map/houses_config.json"
DEFAULT_MAP_BOUNDS = (1000.0, -500.0, 5000.0, 3000.0)
DEFAULT_CORRECTED_MAP_CONFIG_NAME = "corrected_houses_config.json"
LLM_API_STYLE_OPTIONS = ("openai_chat", "openai_responses", "anthropic_sdk")
LLM_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com"
LLM_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
LLM_ROUTE_WAYPOINT_REACHED_CM = 180.0
LLM_ROUTE_STANDOFF_CM = 850.0
LLM_ROUTE_HOUSE_CLEARANCE_CM = 180.0
LLM_ROUTE_MIN_PERIMETER_STANDOFF_CM = 120.0
LLM_ROUTE_FACE_STANDOFF_BUFFER_CM = 10.0
LLM_ROUTE_ALIGN_TOLERANCE_DEG = 12.0
LLM_ROUTE_DEFAULT_REPEAT_CAP = 6
LLM_ROUTE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "target_house_id": "002",
    "route_name": "safe_route_to_house_002",
    "route_points": [
        {"label": "start", "x": 1650.0, "y": 626.2, "status": "visited"},
        {"label": "via_open_space", "x": 1650.0, "y": 300.0, "status": "planned"},
        {"label": "target_standoff", "x": 900.0, "y": 300.0, "status": "planned"},
    ],
    "perimeter_search_order": ["front", "side", "back", "other_side"],
    "avoid_house_ids": ["001", "003"],
    "replan_triggers": ["depth_blocked", "off_route", "target_switch"],
    "reason": "Short map-level route planning rationale.",
}
LLM_TASK_PLAN_OUTPUT_SCHEMA: Dict[str, Any] = {
    "plan_id": "llm_task_plan_20260507_123000",
    "task_text": "Search house 1 entrance, then house 3 entrance.",
    "ordered_targets": [
        {
            "order": 1,
            "house_id": "001",
            "house_alias": "house 1",
            "goal": "search_entry",
            "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
            "status": "pending",
        }
    ],
    "reason": "Short task planning rationale.",
}


MOVE_COMMANDS: Dict[str, Dict[str, Any]] = {
    "w": {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "forward"},
    "s": {"forward_cm": -20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "backward"},
    "a": {"forward_cm": 0.0, "right_cm": -20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "left"},
    "d": {"forward_cm": 0.0, "right_cm": 20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "right"},
    "r": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 20.0, "yaw_delta_deg": 0.0, "action_name": "up"},
    "f": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": -20.0, "yaw_delta_deg": 0.0, "action_name": "down"},
    "q": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": -30.0, "action_name": "yaw_left"},
    "e": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 30.0, "action_name": "yaw_right"},
    "x": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "hold"},
}
YAW_GRID_STEP_DEG = 30.0
YAW_SNAP_SYMBOLS = {"q", "e"}

DEFAULT_KEYBOARD_INTERVAL_MS = 90
KEYBOARD_SYMBOL_ALIASES = {
    "space": "r",
    "control_l": "f",
    "control_r": "f",
    "ctrl_l": "f",
    "ctrl_r": "f",
    "h": "x",
}
KEYBOARD_SYMBOL_ORDER = ("w", "s", "a", "d", "r", "f", "q", "e")
