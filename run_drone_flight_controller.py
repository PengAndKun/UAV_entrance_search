from __future__ import annotations

from control.cli import main, parse_args
from control.constants import *
from control.map_utils import (
    affine_rmse_px,
    build_corrected_map_config,
    corrected_anchors_from_touch_state,
    image_to_world_with_affine,
    rebuild_houses_for_corrected_affine,
    solve_affine_from_anchor_points,
    world_to_image_with_affine,
)
from control.panel import RunDroneFlightPanel
from control.utils import default_llm_api_style, extract_json_object, normalize_llm_api_style


if __name__ == "__main__":
    main()
