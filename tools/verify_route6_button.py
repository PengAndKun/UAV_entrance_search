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
        "route6_nearest_house_pointcloud_map_design.md" in source,
        "Route 6 window should link the design document that drives implementation",
    )
    for label in (
        "Start Nearest House Map Search",
        "Pause",
        "Resume",
        "Stop",
        "Force Next House",
        "Clear",
        "Save Corrected Map Config",
        "Open Latest Route 6 Output",
        "Refresh Map",
    ):
        assert_true(label in source, f"Route 6 window should expose control: {label}")
    assert_true(
        "llm_route6_map_widget" in source and "OverheadMapWidget" in source,
        "Route 6 window should embed an overhead map widget for map overlay",
    )
    assert_true(
        "llm_route6_metrics_var" in source and "Metrics:" in source,
        "Route 6 window should expose mapped/searched/blocked counts, map confidence, and latest corrected config path",
    )


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
