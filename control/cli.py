from __future__ import annotations

import argparse
import logging

import run_drone_flight as flight

from .constants import (
    DEFAULT_KEYBOARD_INTERVAL_MS,
    DEFAULT_MAP_CONFIG_PATH,
    LLM_API_STYLE_OPTIONS,
)
from .utils import default_llm_api_style
from .panel import RunDroneFlightPanel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tk controller for run_drone_flight.py")
    parser.add_argument("--env_platform", "--platform", choices=["auto", "win", "mac", "linux"], default="auto")
    parser.add_argument("--env_root", default=None)
    parser.add_argument("--env_bin", default=None)
    parser.add_argument("--output_dir", default="results/drone_flight_controller")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--launch_sleep", type=int, default=15)
    parser.add_argument("--time_dilation", type=int, default=0)
    parser.add_argument("--step_delay", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--enhance_rgb", dest="enhance_rgb", action="store_true", default=flight.DEFAULT_RGB_ENHANCE_ENABLED)
    parser.add_argument("--no_enhance_rgb", dest="enhance_rgb", action="store_false")
    parser.add_argument("--rgb_enhance_gamma", type=float, default=flight.DEFAULT_RGB_ENHANCE_GAMMA)
    parser.add_argument("--rgb_enhance_gain", type=float, default=flight.DEFAULT_RGB_ENHANCE_GAIN)
    parser.add_argument("--rgb_source_order", choices=["rgb", "bgr"], default=flight.DEFAULT_RGB_SOURCE_ORDER)
    parser.add_argument("--first_person_camera_config", default=flight.DEFAULT_FIRST_PERSON_CAMERA_CONFIG)
    parser.add_argument("--native_viewport_camera_config", default=flight.DEFAULT_NATIVE_VIEWPORT_CAMERA_CONFIG)
    parser.add_argument("--temp_capture_dir", default=flight.DEFAULT_TEMP_CAPTURE_DIR)
    parser.add_argument("--temp_capture_lidar_dir", default=flight.DEFAULT_TEMP_CAPTURE_LIDAR_DIR)
    parser.add_argument("--stream_capture_dir", default=flight.DEFAULT_STREAM_CAPTURE_DIR)
    parser.add_argument("--stream_capture_lidar_dir", default=flight.DEFAULT_STREAM_CAPTURE_LIDAR_DIR)
    parser.add_argument("--stream_interval_s", type=float, default=flight.DEFAULT_STREAM_CAPTURE_INTERVAL_S)
    parser.add_argument("--depth_min_cm", type=float, default=flight.DEFAULT_DEPTH_MIN_CM)
    parser.add_argument("--depth_max_cm", type=float, default=flight.DEFAULT_DEPTH_MAX_CM)
    parser.add_argument("--lidar_depth_min_cm", type=float, default=flight.DEFAULT_LIDAR_DEPTH_MIN_CM)
    parser.add_argument("--lidar_depth_max_cm", type=float, default=flight.DEFAULT_LIDAR_DEPTH_MAX_CM)
    parser.add_argument("--lidar_depth_projection", choices=["auto", "plane", "ray"], default=flight.DEFAULT_LIDAR_DEPTH_PROJECTION)
    parser.add_argument("--lidar_capture_processing", choices=sorted(flight.LIDAR_CAPTURE_PROCESSING_MODES), default=flight.DEFAULT_LIDAR_CAPTURE_PROCESSING)
    parser.add_argument("--force_kill_unreal_on_stop", dest="force_kill_unreal_on_stop", action="store_true",
                        default=flight.DEFAULT_FORCE_KILL_UNREAL_ON_STOP)
    parser.add_argument("--no_force_kill_unreal_on_stop", dest="force_kill_unreal_on_stop", action="store_false")
    parser.add_argument("--movement_mode", choices=["pose_lock", "physics"], default=flight.DEFAULT_MOVEMENT_MODE)
    parser.add_argument("--keyboard_interval_ms", type=int, default=DEFAULT_KEYBOARD_INTERVAL_MS)
    parser.add_argument("--initial_pos", nargs="+", type=float, default=flight.DEFAULT_INITIAL_POS)
    parser.add_argument("--orbit_center", nargs=2, type=float, default=flight.DEFAULT_ORBIT_CENTER)
    parser.add_argument("--orbit_radius", type=float, default=flight.DEFAULT_ORBIT_RADIUS)
    parser.add_argument("--orbit_altitude", type=float, default=flight.DEFAULT_ORBIT_ALTITUDE)
    parser.add_argument("--orbit_steps", type=int, default=flight.DEFAULT_ORBIT_STEPS)
    parser.add_argument("--orbit_start_angle", type=float, default=flight.DEFAULT_ORBIT_START_ANGLE)
    parser.add_argument("--orbit_clockwise", action="store_true")
    parser.add_argument("--state_interval_ms", type=int, default=1500)
    parser.add_argument("--preview_interval_ms", type=int, default=1500)
    parser.add_argument("--map_config", default=DEFAULT_MAP_CONFIG_PATH)
    parser.add_argument("--map_image", default="")
    parser.add_argument("--map_interval_ms", type=int, default=1000)
    parser.add_argument("--map_trajectory_limit", type=int, default=500)
    parser.add_argument("--llm_api_style", default=default_llm_api_style(), choices=LLM_API_STYLE_OPTIONS)
    parser.add_argument("--llm_base_url", default="")
    parser.add_argument("--llm_api_key", default="")
    parser.add_argument("--llm_model", default="")
    parser.add_argument("--llm_route_timeout_s", type=float, default=60.0)
    parser.add_argument("--route_step_cm", type=float, default=120.0)
    parser.add_argument("--route_delay_ms", type=float, default=100.0)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    RunDroneFlightPanel(args).run()


if __name__ == "__main__":
    main()
