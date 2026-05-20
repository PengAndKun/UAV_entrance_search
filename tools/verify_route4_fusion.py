from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from control.route4_fusion_control import Route4FusionControlMixin
from verify_route3_autonomy import _Route3Harness


class _Route4Harness(Route4FusionControlMixin, _Route3Harness):
    def __init__(self) -> None:
        super().__init__()
        self.llm_route4_status_var = _Var("LLM Route V4: idle")
        self.llm_route4_map_status_var = _Var("Route V4 Map: idle")
        self.llm_route4_stage_var = _Var("Stage: idle")
        self.llm_route4_active_var = _Var("Active: n/a")
        self.llm_route4_target_var = _Var("Target: n/a")
        self.llm_route4_error_var = _Var("Error: n/a")
        self.llm_route4_payload_var = _Var("Payload: hold")
        self.llm_route4_progress_text_var = _Var("Fusion: 0%")
        self.llm_route4_progress_var = _Var(0.0)
        self.llm_route4_current_status_var = _Var("Current: idle")
        self.llm_route4_next_status_var = _Var("Next: n/a")
        self.llm_route4_avoidance_status_var = _Var("Avoidance: idle")
        self.llm_route4_representation_status_var = _Var("Representation: idle")
        self.llm_route4_paused_var = _Var(False)
        self.llm_route4_auto_refresh_var = _Var(False)
        self.llm_route4_move_tick_ms_var = _Var(175)
        self.llm_route4_nav_step_cm_var = _Var(25)
        self.llm_route4_reach_tol_cm_var = _Var(70)
        self.llm_route4_z_tol_cm_var = _Var(45)
        self.llm_route4_yaw_tol_deg_var = _Var(12)
        self.llm_route4_max_stage_s_var = _Var(95)
        self.llm_route4_sensing_interval_s_var = _Var(1.25)
        self.llm_route4_representation_model_var = _Var("missing-model.pt")
        self.llm_route4_window = None
        self.llm_route4_map_widget = None
        self.llm_route4_preview_text = None
        self.llm_route4_analysis_text = None
        self.llm_route4_rgb_label = None
        self.llm_route4_rgb_photo = None
        self.llm_route4_state = {}
        self.llm_route4_completed_facades = set()
        self.llm_route4_blocked_facades = set()
        self.llm_route4_thread = None
        self.llm_route4_stop_event = threading.Event()
        self.llm_route4_pause_event = threading.Event()
        self.llm_route4_control_locked = False
        self.llm_route4_auto_refresh_job = None
        self.llm_route3_state = {"mode": "keep_v3_state", "output_dir": "do_not_touch"}
        self.llm_route3_completed_facades = {"west"}
        self.llm_route3_blocked_facades = {"north"}
        self._temp_root = Path(tempfile.mkdtemp())
        self._llm_call_count = 0

    def resolve_project_path(self, path: str | Path) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            return raw
        return self._temp_root / raw

    def effective_llm_api_key(self) -> str:
        return ""

    def effective_llm_model(self) -> str:
        return ""

    def call_configured_llm_text(self, **_kwargs):
        self._llm_call_count += 1
        return {"raw_text": "{}"}

    def active_session(self):
        return None


class _Var:
    def __init__(self, value: object) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value

    def set(self, value: object) -> None:
        self.value = str(value)


def main() -> None:
    harness = _Route4Harness()

    nav = harness.route4_nav_config()
    assert nav["nav_step_cm"] == 25.0, nav
    assert nav["move_tick_ms"] == 175.0, nav
    assert harness.route3_nav_config()["nav_step_cm"] == 20.0

    output_dir = harness.route4_initialize_run("007", force_new=True)
    assert "autosearch_v4_fused" in output_dir.name, output_dir
    assert harness.llm_route4_state["mode"] == "route4_llm_route_avoidance_fusion"
    assert harness.llm_route4_state["target_house_id"] == "007"
    assert harness.llm_route4_state["sensing_config"]["sensing_interval_s"] == 1.25
    assert harness.llm_route3_state == {"mode": "keep_v3_state", "output_dir": "do_not_touch"}
    assert harness.llm_route3_completed_facades == {"west"}
    assert harness.llm_route3_blocked_facades == {"north"}
    for artifact_name in (
        "route4_fusion_state.json",
        "route4_fusion_events.jsonl",
        "route4_navigation_plan.jsonl",
        "route4_movement_trace.jsonl",
        "avoidance_events.jsonl",
        "avoidance_session_summary.json",
    ):
        assert (output_dir / artifact_name).exists(), artifact_name

    harness.route4_update_state(stage="TEST_STAGE")
    harness.route4_write_state_artifact()
    state_path = output_dir / "route4_fusion_state.json"
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["stage"] == "TEST_STAGE", saved

    assert harness.route4_semantic_hint_from_prediction({"predicted_label": "fence_or_rail"}) == "fence_or_rail"
    assert harness.route4_semantic_hint_from_prediction({"predicted_label": "open_path"}) == "unknown"
    assert harness.route4_semantic_hint_from_prediction({"predicted_label": "tree_trunk_or_pole"}) == "tree_trunk_or_pole"

    event = {
        "episode_id": "fusion_probe",
        "frame_id": 1,
        "rgb_path": "",
        "pointcloud_summary": {
            "front_min_depth_cm": 180.0,
            "forward_swept_clear": False,
            "obstacle_geometry": "low_obstacle",
            "obstacle_width_cm": 360.0,
            "up_swept_clear": True,
        },
        "current_pose": {"x": 0.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
        "goal_pose": {"x": 500.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
        "representation_prediction": {"predicted_label": "fence_or_rail", "confidence": 0.83},
    }
    strategy = harness.route4_strategy_for_event(event)
    assert strategy["obstacle_hint"] == "fence_or_rail", strategy
    assert strategy["strategy_source"] in {"representation_prediction_no_api", "representation_prediction_cached"}, strategy
    assert harness._llm_call_count == 0
    cached = harness.route4_strategy_for_event(dict(event))
    assert cached["strategy_source"] == "representation_prediction_cached", cached
    assert harness._llm_call_count == 0

    normalized = harness.route4_normalize_avoidance_event(
        {
            "collision_state": True,
            "selected_action_payload": {"action_name": "backoff", "forward_cm": -20.0},
        }
    )
    assert normalized["avoidance_failed"] is True
    normalized = harness.route4_normalize_avoidance_event(
        {
            "collision_state": False,
            "selected_action_payload": {"action_name": "forward", "forward_cm": 10.0},
        }
    )
    assert normalized["avoidance_failed"] is False

    print("OK route4 fusion verification: state isolation, strategy cache, semantic hints, and collision contract passed")


if __name__ == "__main__":
    main()
