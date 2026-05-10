from __future__ import annotations

import importlib.util
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTE_SCAN_PATH = PROJECT_ROOT / "control" / "route_scan.py"
SPEC = importlib.util.spec_from_file_location("route_scan_under_test", ROUTE_SCAN_PATH)
assert SPEC is not None and SPEC.loader is not None
route_scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(route_scan)


def main() -> None:
    bbox = {
        "min_x": 0.0,
        "max_x": 300.0,
        "min_y": 100.0,
        "max_y": 400.0,
        "center_x": 150.0,
        "center_y": 250.0,
    }
    first = route_scan.generate_rule_scan_points(
        house_id="001",
        bbox_world=bbox,
        standoff_cm=850.0,
        scan_spacing_cm=150.0,
        altitude_cm=600.0,
    )
    second = route_scan.generate_rule_scan_points(
        house_id="001",
        bbox_world=bbox,
        standoff_cm=850.0,
        scan_spacing_cm=150.0,
        altitude_cm=600.0,
    )
    assert first == second, "scan points should be stable for identical inputs"
    assert {"south", "east", "north", "west"} == {point["facade"] for point in first}
    assert all(point["z"] == 600.0 for point in first)

    ordered = route_scan.generate_rule_scan_points(
        house_id="001",
        bbox_world=bbox,
        facade_order=["east", "south"],
        standoff_cm=850.0,
        scan_spacing_cm=150.0,
        altitude_cm=600.0,
    )
    east_first = ordered[0]
    expected_yaw = math.degrees(math.atan2(250.0 - east_first["y"], 300.0 - east_first["x"]))
    assert east_first["scan_id"].startswith("001_east_")
    assert east_first["x"] == 1150.0
    assert round(expected_yaw, 2) == east_first["yaw_deg"]
    print(f"OK route scan verification: {len(first)} default points, {len(ordered)} ordered points")


if __name__ == "__main__":
    main()
