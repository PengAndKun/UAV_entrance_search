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
        self.llm_route4_thinking_status_var = _Var("Thinking: idle")
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


class _FakeRoot:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []
        self.cancelled: list[object] = []

    def after(self, delay_ms: int, callback):
        token = f"after-{len(self.after_calls) + 1}"
        self.after_calls.append((int(delay_ms), callback))
        return token

    def after_cancel(self, job) -> None:
        self.cancelled.append(job)


def _dense_west_facade_scan_fixture(axis_min: float = -1604.49, axis_max: float = 1655.84) -> tuple[list[dict], float, float]:
    points: list[dict] = []
    local_index = 0
    for band, floor_index, z_cm in (
        ("low_ground_250cm", 1, 250.0),
        ("upper_floor_600cm", 2, 600.0),
    ):
        for sample_idx in range(28):
            ratio = float(sample_idx) / 27.0
            axis_value = axis_min + (axis_max - axis_min) * ratio
            points.append(
                {
                    "scan_id": f"001_west_{band}_{sample_idx:03d}",
                    "local_scan_index": local_index,
                    "house_id": "001",
                    "facade": "west",
                    "facade_id": "001_west",
                    "height_band": band,
                    "floor_index": floor_index,
                    "semantic_region": "full_facade",
                    "planned_facade_sample_count": 28,
                    "x": 2103.87,
                    "y": round(axis_value, 2),
                    "z": z_cm,
                    "yaw_deg": 0.0,
                    "target_x": 2503.47,
                    "target_y": round(axis_value, 2),
                    "standoff_cm": 399.6,
                    "scan_spacing_cm": 120.0,
                    "view_type": "facade_floor_band_scan",
                    "capture_trigger": "arrive_align_hover_capture",
                    "safe_interval_index": 0,
                    "safe_interval_count": 1,
                    "safe_axis_min": axis_min,
                    "safe_axis_max": axis_max,
                    "status": "planned",
                }
            )
            local_index += 1
    for cue_idx in range(5):
        axis_value = axis_min + (axis_max - axis_min) * (cue_idx + 1.0) / 6.0
        points.append(
            {
                "scan_id": f"001_west_low_ground_250cm_confirm_{cue_idx:03d}",
                "local_scan_index": local_index,
                "house_id": "001",
                "facade": "west",
                "facade_id": "001_west",
                "height_band": "low_ground_250cm",
                "floor_index": 1,
                "semantic_region": "low-center",
                "cue_type": "door_candidate",
                "x": 2103.87,
                "y": round(axis_value, 2),
                "z": 250.0,
                "yaw_deg": 0.0,
                "target_x": 2503.47,
                "target_y": round(axis_value, 2),
                "standoff_cm": 399.6,
                "scan_spacing_cm": 120.0,
                "view_type": "semantic_candidate_confirm_scan",
                "capture_trigger": "arrive_align_hover_capture",
                "safe_interval_index": 0,
                "safe_interval_count": 1,
                "safe_axis_min": axis_min,
                "safe_axis_max": axis_max,
                "status": "planned",
            }
        )
        local_index += 1
    return points, axis_min, axis_max


def main() -> None:
    harness = _Route4Harness()

    nav = harness.route4_nav_config()
    assert nav["nav_step_cm"] == 25.0, nav
    assert nav["move_tick_ms"] == 175.0, nav
    assert harness.route3_nav_config()["nav_step_cm"] == 20.0

    output_dir = harness.route4_initialize_run("007", force_new=True)
    assert "autosearch_v4_fused" in output_dir.name, output_dir
    assert output_dir.parent.name == "llm_route4_fusion_runs", output_dir
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

    memory = harness.route4_initialize_house_memory(output_dir, "007")
    assert memory["target_house_id"] == "007", memory
    assert set(memory["facades"].keys()) == {"west", "south", "east", "north"}, memory
    assert all(item["status"] == "pending" for item in memory["facades"].values()), memory
    cross_memory_path = output_dir.parent / "house_memory" / "house_007.json"
    harness.route4_update_house_memory(
        output_dir,
        "007",
        "north",
        status="soft_blocked",
        reason="nav_timeout",
        observation_attempt={"label": "007_north_obs_1"},
    )
    assert (output_dir / "house_exploration_memory.json").is_file()
    assert cross_memory_path.is_file()
    saved_memory = json.loads((output_dir / "house_exploration_memory.json").read_text(encoding="utf-8"))
    assert saved_memory["facades"]["north"]["status"] == "soft_blocked", saved_memory
    assert saved_memory["facades"]["north"]["attempt_count"] == 1, saved_memory
    assert harness.llm_route4_state["mandatory_facade_agenda"]["north"] == "soft_blocked"
    second_run = harness.route4_initialize_run("007", force_new=True)
    second_memory = harness.route4_initialize_house_memory(second_run, "007")
    assert second_memory["history"]["facades"]["north"]["status"] == "soft_blocked", second_memory
    assert second_memory["facades"]["north"]["status"] == "pending", second_memory

    llm_ref = harness.route4_log_llm_call(
        output_dir,
        "oa_strategy",
        {"frame_id": 42, "facade": "north", "target_id": "007_north_obs"},
        {"raw_text": "{\"recommended_method\":\"pointcloud_direction_rule\",\"reason\":\"test\"}", "api_key": "SECRET"},
        frame_id=42,
        facade="north",
        target_id="007_north_obs",
        decision={"reason": "test"},
    )
    assert llm_ref["call_id"], llm_ref
    llm_lines = (output_dir / "route4_llm_calls.jsonl").read_text(encoding="utf-8")
    assert "SECRET" not in llm_lines, llm_lines
    event = {
        "frame_id": 42,
        "route4_stage": "NAV_TO_OBS",
        "facade": "north",
        "target_id": "007_north_obs",
        "current_pose": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 45.0},
        "target_waypoint": {"x": 10.0, "y": 20.0, "z": 30.0, "yaw": -90.0},
        "capture_dir": str(output_dir / "frames" / "frame_000042"),
        "depth_lookahead_plan": {"decision_state": "SENSE_LOOK_YAW", "thinking_status": "Thinking: test"},
        "pointcloud_summary": {"front_min_depth_cm": 250.0, "forward_swept_clear": True},
        "representation_prediction": {"predicted_label": "fence_or_rail", "confidence": 0.9},
        "avoidance_gate": {"avoidance_active": False, "reason": "front_clear_depth_250cm"},
        "llm_strategy": {"strategy_source": "representation_prediction_cached", "strategy_cache_key": "north|fence|wall", "strategy_cache_hit": True},
        "selected_action": "route3_nav",
        "selected_action_reason": "clear Route6_entrance_search",
        "risk_state": "CLEAR",
        "nominal_action": {"forward_cm": 20.0},
        "selected_action_payload": {"forward_cm": 20.0},
        "collision_state": False,
        "avoidance_failed": False,
        "llm_call_refs": [llm_ref],
        "house_memory_updates": [{"facade": "north", "status": "in_progress"}],
    }
    decision_doc = harness.route4_write_frame_decision(output_dir, event, final_payload={"forward_cm": 20.0})
    decision_path = output_dir / "frames" / "frame_000042" / "decision.json"
    assert decision_path.is_file(), decision_doc
    saved_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert saved_decision["decision_reason"] == "clear Route6_entrance_search", saved_decision
    assert saved_decision["llm_calls"][0]["call_id"] == llm_ref["call_id"], saved_decision
    assert saved_decision["strategy_cache"]["hit"] is True, saved_decision
    assert (output_dir / "route4_frame_decisions.jsonl").read_text(encoding="utf-8").strip(), "missing frame decision jsonl"

    harness.custom_bboxes = {
        "001": {
            "min_x": 2503.47,
            "max_x": 4290.0,
            "min_y": -1604.49,
            "max_y": 1655.84,
            "center_x": 3396.735,
            "center_y": 25.675,
        }
    }
    dense_points, axis_min, axis_max = _dense_west_facade_scan_fixture()
    assert harness.route2_interval_sample_counts(
        [{"min": axis_min, "max": axis_max}],
        spacing=120.0,
        density="dense",
    ) == [28]
    compacted = harness.route4_compact_boundary_scan_points(dense_points, house_id="001", facade="west")
    compact_points = compacted["points"]
    policy = compacted["policy"]
    assert policy["source"] == "route4_scan_boundary_compaction_v1", policy
    assert policy["max_physical_axis_samples_per_band"] == 7, policy
    assert compacted["raw_point_count"] == 61, compacted
    assert compacted["physical_axis_sample_count"] == 14, compacted
    assert compacted["total_capture_record_count"] == len(compact_points), compacted

    for band in ("low_ground_250cm", "upper_floor_600cm"):
        band_full = [
            point for point in compact_points
            if point.get("height_band") == band and point.get("view_type") == "facade_floor_band_scan"
        ]
        band_axes = sorted({round(float(point["y"]), 2) for point in band_full})
        assert len(band_axes) <= 7, (band, band_axes)
        assert band_axes[0] == round(axis_min, 2), (band, band_axes)
        assert band_axes[-1] == round(axis_max, 2), (band, band_axes)
        assert all(axis_min <= float(point["y"]) <= axis_max for point in band_full), band_full
        boundary_points = [point for point in band_full if point.get("boundary_role") in {"left_boundary", "right_boundary"}]
        assert {point["boundary_role"] for point in boundary_points} == {"left_boundary", "right_boundary"}, boundary_points
        for boundary in boundary_points:
            assert boundary.get("axis_clamped") is True, boundary
            assert boundary.get("physical_axis_sample_count") == 7, boundary
            supplement = boundary.get("yaw_supplement", {})
            assert supplement.get("enabled") is True, supplement
            assert abs(float(supplement.get("offset_deg", 999.0))) <= 45.0, supplement
        band_supplements = [
            point for point in compact_points
            if point.get("height_band") == band and point.get("view_type") == "boundary_yaw_supplement_scan"
        ]
        assert {point.get("boundary_role") for point in band_supplements} == {"left_boundary", "right_boundary"}, band_supplements
        assert all(axis_min <= float(point["y"]) <= axis_max for point in band_supplements), band_supplements

    confirms = [point for point in compact_points if point.get("view_type") == "semantic_candidate_confirm_scan"]
    assert len(confirms) == 5, confirms
    assert all(axis_min <= float(point["y"]) <= axis_max for point in confirms), confirms
    assert harness.route2_interval_sample_counts(
        [{"min": axis_min, "max": axis_max}],
        spacing=120.0,
        density="dense",
    ) == [28]

    wide_points, wide_axis_min, wide_axis_max = _dense_west_facade_scan_fixture(-1900.0, 1900.0)
    wide_compacted = harness.route4_compact_boundary_scan_points(wide_points, house_id="001", facade="west")
    house_axis_min = -1604.49
    house_axis_max = 1655.84
    wide_policy = wide_compacted["policy"]
    assert wide_policy["clamp_axis_to_house_facade_bbox"] is True, wide_policy
    assert wide_policy["axis_clamp_source"] == "safe_interval_intersect_house_facade_bbox", wide_policy
    wide_scan_points = wide_compacted["points"]
    wide_axis_points = [
        point for point in wide_scan_points
        if point.get("view_type") in {
            "facade_floor_band_scan",
            "boundary_yaw_supplement_scan",
            "semantic_candidate_confirm_scan",
        }
    ]
    assert all(house_axis_min <= float(point["y"]) <= house_axis_max for point in wide_axis_points), wide_axis_points
    wide_left = [
        point for point in wide_scan_points
        if point.get("view_type") == "facade_floor_band_scan" and point.get("boundary_role") == "left_boundary"
    ]
    wide_right = [
        point for point in wide_scan_points
        if point.get("view_type") == "facade_floor_band_scan" and point.get("boundary_role") == "right_boundary"
    ]
    assert {round(float(point["y"]), 2) for point in wide_left} == {round(house_axis_min, 2)}, wide_left
    assert {round(float(point["y"]), 2) for point in wide_right} == {round(house_axis_max, 2)}, wide_right
    assert all(point.get("axis_clamp_source") == "safe_interval_intersect_house_facade_bbox" for point in wide_axis_points), wide_axis_points
    assert harness.route2_interval_sample_counts(
        [{"min": wide_axis_min, "max": wide_axis_max}],
        spacing=120.0,
        density="dense",
    ) == [28]

    with tempfile.TemporaryDirectory() as temp_dir:
        harness.llm_route2_state = {
            "mode": "facade_by_facade_vlm_v4_fused",
            "target_house_id": "001",
            "output_dir": temp_dir,
            "facade": "west",
            "facade_id": harness.route2_facade_id("001", "west"),
            "observation_point": {
                "house_id": "001",
                "facade": "west",
                "x": 2103.87,
                "y": 25.67,
                "z": 340.0,
                "yaw_deg": 0.0,
                "standoff_cm": 399.6,
                "status": "planned",
            },
            "facade_analysis": {
                "planner_source": "verify_route4_boundary_fixture",
                "floor_count_estimate": 3,
                "semantic_complexity": "high",
                "recommended_scan_density": "high",
                "target_score": "high",
                "detected_cues": [{"type": "door_candidate", "region": "low-center", "confidence": 0.8}],
                "recommended_translation_span": "large",
            },
        }
        artifact_plan = harness.route4_plan_facade_scan_current()
        artifact_policy = artifact_plan["search_plan"]["route4_scan_boundary_policy"]
        assert artifact_policy["source"] == "route4_scan_boundary_compaction_v1", artifact_policy
        assert artifact_plan["scan_counts"]["physical_axis_sample_count"] <= 14, artifact_plan["scan_counts"]
        facade_plan_path = Path(temp_dir) / "facade_observations" / "001_west" / "facade_search_plan.json"
        saved_facade_plan = json.loads(facade_plan_path.read_text(encoding="utf-8"))
        assert saved_facade_plan["schema"] == "facade_v4_fused_scan_plan", saved_facade_plan
        assert saved_facade_plan["route4_scan_boundary_policy"]["max_physical_axis_samples_per_band"] == 7
        merged_payload = json.loads((Path(temp_dir) / "scan_points.json").read_text(encoding="utf-8"))
        assert merged_payload["schema"] == "facade_v4_fused_global_scan_points", merged_payload
        assert merged_payload["route4_scan_boundary_policy"]["source"] == "route4_scan_boundary_compaction_v1"
        assert any(
            point.get("view_type") == "boundary_yaw_supplement_scan"
            for point in merged_payload["scan_points"]
        ), merged_payload

    assert harness.route4_facade_transition_rank("east", {"west", "north"}, last_completed_facade="north") < harness.route4_facade_transition_rank(
        "south",
        {"west", "north"},
        last_completed_facade="north",
    )

    north_rescue = harness.route4_observation_rescue_candidate(
        target_house_id="001",
        facade="north",
        original_observation={
            "house_id": "001",
            "facade": "north",
            "x": 3396.7,
            "y": 2471.17,
            "z": 280.0,
            "yaw_deg": -90.0,
            "standoff_cm": 815.33,
            "target_x": 3396.7,
            "target_y": 1655.84,
        },
        nav_result={
            "status": "blocked",
            "reason": "non_target_house_clearance",
            "target_id": "001_north_obs_attempt_1",
            "current_pose": {"x": 2638.165, "y": 2544.165, "z": 241.574, "yaw": -49.15},
            "navigation_plan": {
                "escape_waypoint": {
                    "x": 2638.165,
                    "y": 2338.17,
                    "z": 241.574,
                    "yaw": -49.15,
                    "escape_from_obstacle_house_id": "house_12",
                }
            },
        },
    )
    assert north_rescue["status"] == "planned", north_rescue
    assert north_rescue["route4_observation_rescue"] is True, north_rescue
    assert north_rescue["route4_rescue_capture_without_additional_navigation"] is True, north_rescue
    assert north_rescue["observation_attempt_source"] == "route4_rescue_current_pose", north_rescue
    assert abs(float(north_rescue["x"]) - 2638.165) <= 0.02, north_rescue
    assert abs(float(north_rescue["y"]) - 2544.165) <= 0.02, north_rescue
    assert round(float(north_rescue["target_y"]), 2) == 1655.84, north_rescue
    assert 850.0 <= float(north_rescue["standoff_cm"]) <= 900.0, north_rescue
    assert abs(float(north_rescue["yaw_deg"]) + 90.0) <= 1.0, north_rescue
    assert north_rescue["route4_rescue_original_observation"]["standoff_cm"] == 815.33, north_rescue

    north_degraded = harness.route4_degraded_observation_candidate(
        target_house_id="001",
        facade="north",
        original_observation=north_rescue["route4_rescue_original_observation"],
        nav_result={
            "status": "timeout",
            "reason": "nav_timeout",
            "target_id": "001_north_obs_attempt_1",
            "current_pose": {"x": 2396.319, "y": 1749.856, "z": 453.241, "yaw": 51.078},
        },
        fallback_pose={"x": 2396.319, "y": 1749.856, "z": 453.241, "yaw": 51.078},
    )
    assert north_degraded["status"] == "planned", north_degraded
    assert north_degraded["route4_degraded_observation"] is True, north_degraded
    assert north_degraded["route4_completion_status"] == "degraded_completed", north_degraded
    assert north_degraded["observation_attempt_source"] == "route4_degraded_current_pose", north_degraded
    with tempfile.TemporaryDirectory() as temp_dir:
        degraded_dir = Path(temp_dir)
        harness.llm_route4_completed_facades = {"west"}
        harness.llm_route4_blocked_facades = set()
        result = harness.route4_mark_facade_degraded_completed(
            degraded_dir,
            "001",
            "north",
            observation=north_degraded,
            reason="nav_timeout",
            nav_result={"status": "timeout", "reason": "nav_timeout"},
        )
        assert result["status"] == "degraded_completed", result
        assert "north" in harness.llm_route4_completed_facades, result
        assert "north" not in harness.llm_route4_blocked_facades, result
        state = harness.llm_route4_state
        assert state["facade_completion_status"]["north"] == "degraded_completed", state
        memory_payload = json.loads((degraded_dir / "house_exploration_memory.json").read_text(encoding="utf-8"))
        assert memory_payload["facades"]["north"]["status"] == "degraded_completed", memory_payload
        summary = harness.route4_run_summary(degraded_dir, status="done")
        assert summary["degraded_completed_facades"] == ["north"], summary
        assert summary["house_memory"]["facade_status"]["north"] == "degraded_completed", summary

    east_rescue = harness.route4_observation_rescue_candidate(
        target_house_id="001",
        facade="east",
        original_observation={
            "house_id": "001",
            "facade": "east",
            "x": 5209.95,
            "y": 569.07,
            "z": 280.0,
            "yaw_deg": 180.0,
            "target_x": 4289.93,
            "target_y": 569.07,
        },
        nav_result={
            "status": "blocked",
            "reason": "non_target_house_clearance",
            "current_pose": {"x": 2638.165, "y": 2544.165, "z": 241.574, "yaw": -49.15},
        },
    )
    assert east_rescue == {}, east_rescue

    with tempfile.TemporaryDirectory() as temp_dir:
        harness.root = _FakeRoot()
        harness.llm_route2_state = {"target_house_id": "001", "output_dir": temp_dir}
        rescue_result = harness.route4_try_observation_rescue(
            output_dir=Path(temp_dir),
            target_house_id="001",
            facade="north",
            original_observation=north_rescue["route4_rescue_original_observation"],
            ranked_candidates=[north_rescue],
            nav_result={
                "status": "blocked",
                "reason": "non_target_house_clearance",
                "target_id": "001_north_obs_attempt_1",
                "current_pose": {"x": 2638.165, "y": 2544.165, "z": 241.574, "yaw": -49.15},
            },
            attempt_index=1,
        )
        assert rescue_result["status"] == "ok", rescue_result
        assert rescue_result["observation"]["route4_observation_rescue"] is True, rescue_result
        saved_rescue = json.loads((Path(temp_dir) / "facade_observations" / "001_north" / "facade_observation_point.json").read_text(encoding="utf-8"))
        assert saved_rescue["observation_attempt_source"] == "route4_rescue_current_pose", saved_rescue
        rescue_events = (Path(temp_dir) / "route4_fusion_events.jsonl").read_text(encoding="utf-8")
        assert "observation_rescue_applied" in rescue_events, rescue_events

    fake_root = _FakeRoot()
    harness.root = fake_root
    harness.llm_route4_auto_refresh_var.set(True)
    harness.on_route4_auto_refresh_toggle()
    assert fake_root.after_calls, "route4 auto refresh should schedule a tick"
    assert harness.llm_route4_auto_refresh_job is not None
    harness.llm_route4_auto_refresh_var.set(False)
    harness.on_route4_auto_refresh_toggle()
    assert fake_root.cancelled, "route4 auto refresh should cancel pending tick"
    harness.custom_bboxes = {}

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

    clear_tree_event = {
        "target_id": "scan_clear_tree",
        "representation_prediction": {"predicted_label": "tree_canopy_or_cluster", "confidence": 0.52},
        "pointcloud_summary": {
            "available": True,
            "front_min_depth_cm": 566.5,
            "forward_swept_clear": True,
            "obstacle_geometry": "overhang_beam",
        },
    }
    gate = harness.route4_should_apply_avoidance(clear_tree_event, {"obstacle_hint": "building"})
    assert gate["avoidance_active"] is False, gate
    assert gate["semantic_hint"] == "tree_canopy_or_cluster", gate
    assert gate["trigger_distance_cm"] == 120.0, gate

    blocked_lookahead_event = {
        "target_id": "001_scan_0014_west_single_300cm",
        "representation_prediction": {"predicted_label": "tree_canopy_or_cluster", "confidence": 0.62},
        "pointcloud_summary": {
            "available": True,
            "front_min_depth_cm": 181.125,
            "forward_swept_clear": False,
            "right_swept_clear": False,
            "up_swept_clear": False,
            "left_swept_clear": True,
            "down_swept_clear": True,
            "obstacle_geometry": "vertical_wall",
        },
    }
    nominal_payload = {
        "forward_cm": 12.056,
        "right_cm": 15.958,
        "up_cm": 20.0,
        "yaw_delta_deg": -17.728,
        "action_name": "route3_nav",
    }
    travel_yaw = harness.route4_navigation_travel_yaw_deg(
        {"x": 2100.0, "y": 1062.0, "z": 250.0, "yaw": 0.0},
        {"x": 2103.0, "y": 1180.0, "z": 300.0, "yaw": 0.0},
    )
    assert 88.0 <= travel_yaw <= 89.5, travel_yaw
    lookahead_plan = harness.route4_depth_lookahead_plan(
        {"x": 2100.0, "y": 1062.0, "z": 250.0, "yaw": 0.0},
        {"x": 2103.0, "y": 1180.0, "z": 300.0, "yaw": 0.0},
        stage="NAV_TO_SCAN_POINT",
        facade="west",
        target_id="001_scan_0014_west_single_300cm",
    )
    assert lookahead_plan["enabled"] is True, lookahead_plan
    assert lookahead_plan["look_direction"] == "waypoint_bearing", lookahead_plan
    assert lookahead_plan["capture_yaw_deg"] == 0.0, lookahead_plan
    assert 88.0 <= float(lookahead_plan["look_yaw_deg"]) <= 89.5, lookahead_plan
    assert "look waypoint" in lookahead_plan["thinking_status"], lookahead_plan
    route4_payload = harness.route4_movement_payload_for_target_with_lookahead(
        {"x": 2100.0, "y": 1062.0, "z": 250.0, "yaw": 86.0},
        {"x": 2103.0, "y": 1180.0, "z": 300.0, "yaw": 0.0},
        harness.route4_nav_config(),
        stage="NAV_TO_SCAN_POINT",
    )
    assert route4_payload["forward_cm"] > 0.0, route4_payload
    assert abs(float(route4_payload["right_cm"])) <= 2.0, route4_payload
    assert abs(float(route4_payload["yaw_delta_deg"])) <= 10.0, route4_payload
    final_yaw_payload = harness.route4_movement_payload_for_target_with_lookahead(
        {"x": 2103.0, "y": 1112.0, "z": 300.0, "yaw": 86.0},
        {"x": 2103.0, "y": 1112.0, "z": 300.0, "yaw": 0.0},
        harness.route4_nav_config(),
        stage="NAV_TO_SCAN_POINT",
    )
    assert final_yaw_payload["forward_cm"] == 0.0 and final_yaw_payload["right_cm"] == 0.0, final_yaw_payload
    assert float(final_yaw_payload["yaw_delta_deg"]) < 0.0, final_yaw_payload
    assert harness.route4_should_precheck_depth_before_payload("NAV_TO_SCAN_POINT", nominal_payload) is True
    assert harness.route4_should_precheck_depth_before_payload("NAV_TO_OBS", nominal_payload) is True
    assert harness.route4_should_precheck_depth_before_payload(
        "NAV_TO_SCAN_POINT",
        {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 20.0, "action_name": "route3_yaw"},
    ) is False
    assert harness.route4_should_precheck_depth_before_payload("CAPTURE_SCAN", nominal_payload) is False
    gate = harness.route4_should_apply_avoidance(blocked_lookahead_event, {})
    assert gate["avoidance_active"] is False, gate
    lookahead_gate = harness.route4_depth_lookahead_gate(
        blocked_lookahead_event,
        gate,
        nominal_payload,
        stage="NAV_TO_SCAN_POINT",
    )
    assert lookahead_gate["avoidance_active"] is True, lookahead_gate
    assert lookahead_gate["semantic_hint"] == "building", lookahead_gate
    assert lookahead_gate["semantic_source"] == "depth_lookahead_pointcloud", lookahead_gate
    assert lookahead_gate["reason"] == "depth_lookahead_blocked_before_collection", lookahead_gate
    assert set(lookahead_gate["lookahead_blocked_directions"]) == {"forward", "right", "up"}, lookahead_gate

    clear_lookahead_gate = harness.route4_depth_lookahead_gate(
        {
            "pointcloud_summary": {
                "available": True,
                "front_min_depth_cm": 620.0,
                "forward_swept_clear": True,
                "right_swept_clear": True,
                "up_swept_clear": True,
                "obstacle_geometry": "open_path",
            }
        },
        {"avoidance_active": False, "reason": "front_clear_depth_620cm"},
        nominal_payload,
        stage="NAV_TO_SCAN_POINT",
    )
    assert clear_lookahead_gate["avoidance_active"] is False, clear_lookahead_gate

    near_tree_event = {
        "representation_prediction": {"predicted_label": "fence_or_rail", "confidence": 0.84},
        "pointcloud_summary": {
            "available": True,
            "front_min_depth_cm": 98.0,
            "forward_swept_clear": False,
            "obstacle_geometry": "low_obstacle",
        },
    }
    gate = harness.route4_should_apply_avoidance(near_tree_event, {"obstacle_hint": "fence_or_rail"})
    assert gate["avoidance_active"] is True, gate
    assert gate["trigger_distance_cm"] == 120.0, gate

    far_building_event = {
        "representation_prediction": {"predicted_label": "building", "confidence": 0.91},
        "pointcloud_summary": {
            "available": True,
            "front_min_depth_cm": 540.0,
            "forward_swept_clear": True,
            "obstacle_geometry": "vertical_wall",
        },
    }
    gate = harness.route4_should_apply_avoidance(far_building_event, {"obstacle_hint": "building"})
    assert gate["avoidance_active"] is False, gate
    assert gate["trigger_distance_cm"] == 500.0, gate

    near_building_event = dict(far_building_event)
    near_building_event["pointcloud_summary"] = dict(far_building_event["pointcloud_summary"], front_min_depth_cm=480.0)
    gate = harness.route4_should_apply_avoidance(near_building_event, {"obstacle_hint": "building"})
    assert gate["avoidance_active"] is True, gate

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

    print(
        "OK route4 fusion verification: state isolation, scan boundary compaction, "
        "house-bbox clamp, observation rescue, auto refresh, transition rank, "
        "depth lookahead, frame decisions, house memory, degraded completion, "
        "strategy cache, semantic hints, and collision contract passed"
    )


if __name__ == "__main__":
    main()
