from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def panel_source() -> str:
    return (PROJECT_ROOT / "control" / "panel.py").read_text(encoding="utf-8")


def test_panel_route6_button_wiring() -> None:
    source = panel_source()
    assert_true(
        "Route6ExploreControlMixin" in source,
        "RunDroneFlightPanel must import and inherit Route6ExploreControlMixin",
    )
    assert_true(
        'text="Open LLM Route Window 6"' in source or "text='Open LLM Route Window 6'" in source,
        "main Route6_entrance_search scan row must expose the Open LLM Route Window 6 button",
    )
    assert_true(
        "command=self.open_llm_route_window6" in source,
        "Open LLM Route Window 6 button must call open_llm_route_window6",
    )


def test_route6_mixin_contract() -> None:
    module = importlib.import_module("control.route6_explore_control")
    mixin = getattr(module, "Route6ExploreControlMixin", None)
    assert_true(mixin is not None, "control.route6_explore_control must define Route6ExploreControlMixin")
    for method_name in (
        "ensure_route6_state",
        "open_llm_route_window6",
        "close_llm_route_window6",
        "on_route6_start_nearest_map_search",
        "on_route6_pause",
        "on_route6_resume",
        "on_route6_stop",
        "on_route6_force_next_house",
        "on_route6_clear",
        "on_route6_save_corrected_map_config",
        "on_route6_open_latest_output",
        "refresh_llm_route6_map",
        "refresh_llm_route6_realtime_map",
        "route6_build_realtime_map_planning_context",
        "route6_select_llm_target_from_context",
        "route6_apply_selected_target_to_scan_plan",
        "route6_start_realtime_map_for_target_search",
        "on_route6_full_stop_uav",
        "route6_apply_full_stop",
        "route6_build_semantic_map_context",
        "route6_extract_layered_object_candidates",
        "route6_select_llm_semantic_target",
        "route6_plan_llm_navigation_target",
        "refresh_llm_route6_map_analysis_panel",
        "refresh_llm_route6_or_avoidance_display",
        "route6_or_avoidance_display_payload",
        "_on_llm_route6_mousewheel",
        "route6_map_overlay_points",
        "route6_update_runtime_metrics",
    ):
        assert_true(
            callable(getattr(mixin, method_name, None)),
            f"Route6ExploreControlMixin must provide {method_name}()",
        )


def test_control_package_exports_route6_mixin() -> None:
    source = (PROJECT_ROOT / "control" / "__init__.py").read_text(encoding="utf-8")
    assert_true(
        "Route6ExploreControlMixin" in source,
        "control package should export Route6ExploreControlMixin for controller composition and tests",
    )


def test_route6_window_declares_design_doc_and_core_controls() -> None:
    source = (PROJECT_ROOT / "control" / "route6_explore_control.py").read_text(encoding="utf-8")
    assert_true(
        "route6_window6_realtime_map_llm_targeting.md" in source,
        "Route 6 window should link the v002 realtime-map LLM targeting document",
    )
    for label in (
        "Start LLM Target Map Search",
        "Pause",
        "Resume",
        "Stop",
        "Full Stop UAV",
        "Force Next House",
        "Clear",
        "Save Corrected Map Config",
        "Open Latest Route 6 Output",
        "Select LLM Target",
        "Realtime Layered Map",
        "Load Latest Map",
        "Start Realtime Update",
        "Stop Realtime Update",
        "OR Avoidance",
        "Refresh OR",
        "LLM Map Analysis",
    ):
        assert_true(label in source, f"Route 6 window should expose control: {label}")
    start_handler_source = source[source.find("def on_route6_start_nearest_map_search"):source.find("def on_route6_pause")]
    assert_true(
        "route6_start_realtime_map_for_target_search" in start_handler_source,
        "Start LLM Target Map Search should also start Route 6 realtime map updates",
    )
    stop_handler_source = source[source.find("def on_route6_stop"):source.find("def on_route6_force_next_house")]
    assert_true(
        "route6_update_map_realtime_stop_event" in stop_handler_source,
        "Route 6 Stop should request realtime map update shutdown too",
    )
    for text in (
        "llm_route6_scroll_canvas",
        "llm_route6_content_frame",
        "llm_route6_or_status_var",
        "llm_route6_or_detail_var",
        "<MouseWheel>",
        "<Button-4>",
        "<Button-5>",
        "_bind_llm_route6_mousewheel_tree",
    ):
        assert_true(text in source, f"Route 6 window should declare scroll/OR display support: {text}")
    assert_true(
        "bind_all" not in source[source.find("def open_llm_route_window6"):source.find("def close_llm_route_window6")],
        "Window 6 wheel handling should stay local and must not bind globally to the main window",
    )
    assert_true(
        "Route 6 Map Overlay" not in source[source.find("def open_llm_route_window6"):source.find("def close_llm_route_window6")],
        "Window 6 should no longer use the old overhead map overlay as its primary map",
    )
    assert_true(
        "llm_route6_realtime_map_preview_label" in source and "refresh_llm_route6_realtime_map" in source,
        "Route 6 window should consume the realtime layered occupancy preview",
    )
    assert_true(
        "llm_route6_metrics_var" in source and "Metrics:" in source,
        "Route 6 window should expose mapped/searched/blocked counts, map confidence, and latest corrected config path",
    )
    doc_path = PROJECT_ROOT / "overleaf" / "Route6_entrance_search" / "v002" / "route6_window6_realtime_map_llm_targeting.md"
    assert_true(doc_path.is_file(), f"Route 6 v002 document should exist: {doc_path}")


def main() -> None:
    tests = [
        test_panel_route6_button_wiring,
        test_route6_mixin_contract,
        test_control_package_exports_route6_mixin,
        test_route6_window_declares_design_doc_and_core_controls,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PASS route6 button verifier")


if __name__ == "__main__":
    main()
