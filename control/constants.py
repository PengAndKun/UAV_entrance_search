from __future__ import annotations

from typing import Any, Dict

DEFAULT_BASE_MAP_CONFIG_PATH = "assets/overhead_map/houses_config.json"
DEFAULT_MANUAL_SHIFT_MAP_CONFIG_NAME = "manual_shift_houses_config.json"
DEFAULT_MAP_CONFIG_PATH = f"assets/overhead_map/{DEFAULT_MANUAL_SHIFT_MAP_CONFIG_NAME}"
DEFAULT_MAP_BOUNDS = (1000.0, -500.0, 5000.0, 3000.0)
DEFAULT_CORRECTED_MAP_CONFIG_NAME = "corrected_houses_config.json"
DEFAULT_SETTING_MAP_CONFIG_NAME = "setting_map_houses_config.json"
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
LLM_ROUTE_SCAN_STANDOFF_CM = 850.0
LLM_ROUTE_SCAN_SPACING_CM = 150.0
LLM_ROUTE_CAPTURE_COUNT = 1
LLM_ROUTE_PATH_CAPTURE_INTERVAL_S = 0.5
LLM_ROUTE_PATH_CAPTURE_YAW_STEP_DEG = 30.0
LLM_ROUTE_PATH_CAPTURE_YAW_TOLERANCE_DEG = 8.0
LLM_ROUTE2_FACADE_OPTIONS = ("auto", "south", "east", "north", "west")
LLM_ROUTE2_DEFAULT_FLOOR_HEIGHT_M = 3.0
LLM_ROUTE2_DEFAULT_FLOORS = 2
LLM_ROUTE2_LOW_Z_CM = 250.0
LLM_ROUTE2_Z_STEP_CM = 350.0
LLM_ROUTE2_MAX_FLOORS = 4
LLM_ROUTE2_OBSERVATION_Z_CM = 280.0
LLM_ROUTE2_OBSERVATION_MIN_STANDOFF_CM = 850.0
LLM_ROUTE2_OBSERVATION_MAX_STANDOFF_CM = 1800.0
LLM_ROUTE2_OBSERVATION_DEPTH_FACTOR = 0.515
LLM_ROUTE2_OBSERVATION_LENGTH_FACTOR = 0.95
LLM_ROUTE2_SINGLE_BAND_Z_OFFSET_CM = 50.0
LLM_ROUTE3_OBSTACLE_RAISE_STEP_CM = 120.0
LLM_ROUTE3_OBSTACLE_MAX_OBSERVATION_Z_CM = 650.0
LLM_ROUTE3_ASTAR_GRID_CM = 100.0
LLM_ROUTE3_ASTAR_MAX_REPLANS = 3
LLM_ROUTE3_SEGMENT_SAFETY_SAMPLE_CM = 20.0
LLM_ROUTE3_NAV_SEGMENT_REACH_TOL_CM = 80.0
LLM_ROUTE3_OBSERVATION_BOUNDARY_GAP_CM = 100.0
LLM_ROUTE3_ESCAPE_MARGIN_CM = 140.0
LLM_ROUTE3_ESCAPE_REACH_TOL_CM = 15.0
LLM_ROUTE3_PANORAMA_COVERAGE_THRESHOLD = 0.85
LLM_ROUTE3_PANORAMA_MIN_YAW_DELTA_DEG = 15.0
LLM_ROUTE3_PANORAMA_MAX_YAW_DELTA_DEG = 45.0
LLM_ROUTE3_OBSERVATION_ATTEMPT_MAX = 6
LLM_ROUTE3_AUTO_REFRESH_MS = 1000
LLM_ROUTE2_DENSITY_SPACING_CM: Dict[str, float] = {
    "high": 80.0,
    "medium": 120.0,
    "low": 200.0,
}
LLM_ROUTE2_TRANSLATION_SPACING_CM: Dict[str, float] = {
    "small": 80.0,
    "medium": 140.0,
    "large": 220.0,
}
LLM_ROUTE2_RULE_DENSITY_SPACING_CM: Dict[str, float] = {
    "dense": 120.0,
    "medium": 220.0,
    "sparse": 600.0,
}
LLM_ROUTE2_RULE_DENSITY_LIMITS: Dict[str, Dict[str, int]] = {
    "dense": {"min": 8, "max": 28},
    "medium": {"min": 4, "max": 12},
    "sparse": {"min": 2, "max": 7},
}
LLM_ROUTE2_HOUSE_FACADE_FALLBACK_DENSITY: Dict[str, Dict[str, str]] = {
    "001": {
        "west": "dense",
        "south": "medium",
        "north": "sparse",
        "east": "sparse",
    }
}
LLM_ROUTE2_SCAN_STANDOFF_RATIO: Dict[str, float] = {
    "small": 0.333,
    "medium": 0.333,
    "large": 0.42,
}
LLM_ROUTE2_SCAN_STANDOFF_MIN_CM = 300.0
LLM_ROUTE2_SCAN_STANDOFF_MAX_CM = 500.0
LLM_ROUTE2_FACADE_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "facade_id": "001_east",
    "floor_count_estimate": 2,
    "semantic_complexity": "medium",
    "terrace_or_awning_risk": "unknown",
    "target_score": "medium",
    "detected_cues": [
        {"type": "door_candidate", "region": "low-center", "confidence": 0.5}
    ],
    "recommended_translation_span": "medium",
    "recommended_height_bands": ["single_300cm"],
    "reason": "Short facade analysis rationale.",
}
LLM_ROUTE3_OBSERVATION_OBSTACLE_SCHEMA: Dict[str, Any] = {
    "foreground_obstacle_present": False,
    "obstacle_type": "none",
    "severity": "none",
    "facade_visibility": "clear",
    "recommend_raise": False,
    "recommended_observation_z_cm": LLM_ROUTE2_OBSERVATION_Z_CM,
    "reason": "Short explanation.",
}
LLM_ROUTE3_TASK_PLAN_SCHEMA: Dict[str, Any] = {
    "target_house_id": "001",
    "target_sequence": ["001"],
    "ordered_targets": [
        {"order": 1, "house_id": "001", "goal": "search_entry", "status": "pending"}
    ],
    "major_task": "Search the selected house entrance.",
    "subtasks": [
        {"order": 1, "facade": "west", "goal": "observe_analyze_scan_validate", "status": "pending"}
    ],
    "preferred_start_facade": "west",
    "facade_priority": ["west", "south", "east", "north"],
    "completion_criteria": "All reachable facades have RGB/VLM analysis, scan captures, and validation.",
    "reason": "Short high-level task decomposition rationale.",
}
LLM_ROUTE_SCAN_POINTS_SCHEMA: Dict[str, Any] = {
    "scan_points": [
        {
            "scan_id": "001_south_000",
            "house_id": "001",
            "facade": "south",
            "face_role": "front",
            "x": 0.0,
            "y": 0.0,
            "z": 600.0,
            "yaw_deg": 90.0,
            "standoff_cm": LLM_ROUTE_SCAN_STANDOFF_CM,
            "scan_spacing_cm": LLM_ROUTE_SCAN_SPACING_CM,
            "expected_lidar_range_cm": [20.0, 1200.0],
            "capture_trigger": "arrive_align_hover_capture",
            "view_type": "face_view",
            "status": "planned",
            "safe_interval_index": 0,
            "safe_interval_count": 1,
            "safe_axis_min": 0.0,
            "safe_axis_max": 0.0,
            "safe_interval_source": "bbox_clearance_clipped",
            "corridor_mode": "open_default",
            "corridor_gap_cm": None,
            "corridor_side_margin_cm": None,
            "corridor_blocking_house_id": "",
            "corridor_clearance_cm": LLM_ROUTE_HOUSE_CLEARANCE_CM,
            "corridor_safe": True,
        }
    ]
}
LLM_ROUTE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "target_house_id": "002",
    "route_name": "safe_route_to_house_002",
    "route_points": [
        {"label": "start", "x": 1650.0, "y": 626.2, "status": "visited"},
        {"label": "via_open_space", "x": 1650.0, "y": 300.0, "status": "planned"},
        {"label": "target_standoff", "x": 900.0, "y": 300.0, "status": "planned"},
    ],
    "perimeter_search_order": ["front", "side", "back", "other_side"],
    "preferred_facade_order": ["south", "east", "north", "west"],
    "avoid_house_ids": ["001", "003"],
    "replan_triggers": ["depth_blocked", "off_route", "target_switch"],
    "reason": "Short map-level Route6_entrance_search planning rationale.",
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
