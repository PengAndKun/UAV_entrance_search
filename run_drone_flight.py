import argparse
import json
import logging
import math
import os
import plistlib
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

import gym
import gym_unrealcv
import cv2
import numpy as np
from PIL import Image

if sys.platform == "darwin":
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/uav-flow-mpl")
else:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TEMP", ".")) / "uav-flow-mpl"))

from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation


LOGGER = logging.getLogger(__name__)
_OPEN3D_MODULE: Any = None
_OPEN3D_IMPORT_ERROR: Optional[str] = None


DEFAULT_ENV_ID = "UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0"
DEFAULT_OUTPUT_DIR = "results/drone_flight_test"
DEFAULT_ENV_ROOT = "UnrealEnv"
DEFAULT_WIN_ENV_BIN = "UE4_ExampleScene_Win/UE4_ExampleScene/Binaries/Win64/UE4_ExampleScene.exe"
DEFAULT_MAC_ENV_BIN = "UE4_ExampleScene_Mac/UE4_ExampleScene.app"
DEFAULT_INITIAL_POS = [0.0, 0.0, 100.0, 0.0]
DEFAULT_ORBIT_CENTER = [995.3, -203.73]
DEFAULT_ORBIT_RADIUS = 550.0
DEFAULT_ORBIT_ALTITUDE = 180.0
DEFAULT_ORBIT_STEPS = 96
DEFAULT_ORBIT_START_ANGLE = 180.0
DEFAULT_MOVEMENT_MODE = "physics"
DEFAULT_POSE_SETTLE_TIMEOUT = 0.8
DEFAULT_POSE_SETTLE_POS_TOLERANCE = 5.0
DEFAULT_POSE_SETTLE_YAW_TOLERANCE = 2.0
DEFAULT_CALIBRATION_MARKER_CLASS = "BP_drone01_C"
DEFAULT_CALIBRATION_MARKER_CLASSES = ("BP_drone01_C", "target_C", "bp_character_C", "Cube_C")
DEFAULT_CALIBRATION_MARKER_SCALE = [0.1, 0.1, 0.1]
DEFAULT_RGB_ENHANCE_ENABLED = True
DEFAULT_RGB_ENHANCE_GAMMA = 0.72
DEFAULT_RGB_ENHANCE_GAIN = 1.12
DEFAULT_RGB_SOURCE_ORDER = "bgr"
DEFAULT_TEMP_CAPTURE_DIR = "temp_capture"
DEFAULT_TEMP_CAPTURE_LIDAR_DIR = "temp_capture_lidar"
DEFAULT_STREAM_CAPTURE_DIR = "stream_capture"
DEFAULT_STREAM_CAPTURE_LIDAR_DIR = "stream_capture_lidar"
DEFAULT_STREAM_CAPTURE_INTERVAL_S = 0.5
DEFAULT_DEPTH_MIN_CM = 20.0
DEFAULT_DEPTH_MAX_CM = 1200.0
DEFAULT_LIDAR_DEPTH_MIN_CM = DEFAULT_DEPTH_MIN_CM
DEFAULT_LIDAR_DEPTH_MAX_CM = DEFAULT_DEPTH_MAX_CM
DEFAULT_LIDAR_DEPTH_PROJECTION = "auto"
DEFAULT_LIDAR_CAPTURE_PROCESSING = "smooth"
DEFAULT_LIDAR_RECON_VOXEL_CM = 8.0
DEFAULT_LIDAR_RECON_MAX_POINTS = 500000
DEFAULT_LIDAR_RECON_WRITE_EVERY = 5
DEFAULT_OPEN3D_VOXEL_CM = DEFAULT_LIDAR_RECON_VOXEL_CM
DEFAULT_OPEN3D_NORMAL_RADIUS_CM = 30.0
DEFAULT_OPEN3D_NORMAL_MAX_NN = 30
DEFAULT_FORCE_KILL_UNREAL_ON_STOP = True
DEFAULT_ACTION_PLAN = [
    ("hover", [0.0, 0.0, 0.0, 0.0]),
    ("up", [0.0, 0.0, 0.5, 0.0]),
    ("forward", [0.5, 0.0, 0.0, 0.0]),
    ("right", [0.0, 0.5, 0.0, 0.0]),
    ("yaw_left", [0.0, 0.0, 0.0, 0.8]),
    ("down", [0.0, 0.0, -0.4, 0.0]),
    ("stop", [0.0, 0.0, 0.0, 0.0]),
]
KEYBOARD_HELP = """
Keyboard control is active. Keep this process running, then click the Unreal window if you want to watch the drone view.

  W/S       forward / backward
  A/D       left / right
  Space/R   up
  F/Ctrl    down
  Q/E       yaw left / yaw right
  X/H       hover / stop
  Esc       quit
"""


def normalize_calibration_marker_scale(value: Any = None) -> List[float]:
    if value is None:
        return list(DEFAULT_CALIBRATION_MARKER_SCALE)
    try:
        if isinstance(value, str):
            parts = [part for part in value.replace(",", " ").split() if part]
            if len(parts) >= 3:
                return [max(0.05, float(parts[0])), max(0.05, float(parts[1])), max(0.05, float(parts[2]))]
            if len(parts) == 1:
                scalar = float(parts[0])
                return [max(0.05, scalar), max(0.05, scalar), max(0.05, scalar)]
        if isinstance(value, (int, float)):
            scalar = float(value)
            return [max(0.05, scalar), max(0.05, scalar), max(0.05, scalar)]
        values = list(value)
        if len(values) == 1:
            scalar = float(values[0])
            return [max(0.05, scalar), max(0.05, scalar), max(0.05, scalar)]
        if len(values) >= 3:
            return [max(0.05, float(values[0])), max(0.05, float(values[1])), max(0.05, float(values[2]))]
    except Exception:
        pass
    return list(DEFAULT_CALIBRATION_MARKER_SCALE)


def host_platform() -> str:
    if sys.platform.startswith("win"):
        return "win"
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def resolve_path(path_value: Optional[str], default_path: Path, repo_root: Path) -> Path:
    if path_value:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()
    return default_path.resolve()


def path_for_unrealcv_setting(path: Path, env_root: Path) -> str:
    try:
        return path.resolve().relative_to(env_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def path_from_unrealcv_setting(path_value: str, env_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = env_root / path
    return path.resolve()


def find_macos_app_executable(app_bundle: Path) -> Path:
    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    info_plist = contents_dir / "Info.plist"
    executable_name = None
    if info_plist.exists():
        try:
            with info_plist.open("rb") as f:
                executable_name = plistlib.load(f).get("CFBundleExecutable")
        except (OSError, plistlib.InvalidFileException, ValueError):
            executable_name = None
    if executable_name:
        executable = macos_dir / executable_name
        if executable.exists():
            return executable
    executables = sorted(
        path for path in macos_dir.iterdir()
        if path.is_file() and os.access(path, os.X_OK)
    ) if macos_dir.exists() else []
    return executables[0] if executables else macos_dir / app_bundle.stem


def find_macos_unrealcv_ini(app_bundle: Path) -> Path:
    contents_dir = app_bundle / "Contents"
    for engine_dir_name in ("UE4", "UE"):
        matches = sorted((contents_dir / engine_dir_name).glob("*/Binaries/Mac/unrealcv.ini"))
        if matches:
            return matches[0]
    return contents_dir / "UE4" / app_bundle.stem / "Binaries" / "Mac" / "unrealcv.ini"


def find_windows_env_binary(env_root: Path) -> Optional[Path]:
    if not env_root.exists():
        return None
    candidates = []
    for candidate in env_root.rglob("*.exe"):
        name = candidate.name.lower()
        if "prereq" in name or "setup" in name:
            continue
        parts = {part.lower() for part in candidate.parts}
        score = 0
        if (candidate.parent / "unrealcv.ini").exists():
            score += 10
        if "binaries" in parts and "win64" in parts:
            score += 5
        if candidate.stem.lower() in {"ue4_examplescene", "collection"}:
            score += 2
        candidates.append((score, len(candidate.parts), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], str(item[2]).lower()))
    return candidates[0][2]


def find_macos_env_bundle(env_root: Path) -> Optional[Path]:
    if not env_root.exists():
        return None
    bundles = sorted(env_root.rglob("*.app"), key=lambda path: (len(path.parts), str(path).lower()))
    return bundles[0] if bundles else None


def resolve_env_binary(args: argparse.Namespace, platform_name: str, env_root: Path) -> Optional[str]:
    if args.env_bin:
        env_bin_path = Path(args.env_bin).expanduser()
        if not env_bin_path.is_absolute():
            env_bin_path = env_root / env_bin_path
        return path_for_unrealcv_setting(env_bin_path, env_root)

    if platform_name == "win":
        default_path = env_root / DEFAULT_WIN_ENV_BIN
        env_bin_path = default_path if default_path.exists() else find_windows_env_binary(env_root)
        return path_for_unrealcv_setting(env_bin_path, env_root) if env_bin_path else DEFAULT_WIN_ENV_BIN

    if platform_name == "mac":
        default_path = env_root / DEFAULT_MAC_ENV_BIN
        env_bin_path = default_path if default_path.exists() else find_macos_env_bundle(env_root)
        return path_for_unrealcv_setting(env_bin_path, env_root) if env_bin_path else DEFAULT_MAC_ENV_BIN

    return None


def configure_local_unreal_env(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent
    local_env_root = repo_root / DEFAULT_ENV_ROOT
    unreal_env_var = os.getenv("UnrealEnv")
    env_root_default = Path(unreal_env_var) if unreal_env_var else local_env_root
    env_root = resolve_path(args.env_root, env_root_default, repo_root)
    platform_name = host_platform() if args.env_platform == "auto" else args.env_platform
    if platform_name != host_platform() and not args.dry_run:
        raise ValueError(
            f"Cannot launch a '{platform_name}' Unreal package on host platform '{host_platform()}'. "
            "Use --dry_run only for cross-platform config checks."
        )
    env_bin = resolve_env_binary(args, platform_name, env_root)

    os.environ["UnrealEnv"] = str(env_root)
    args.resolved_env_root = env_root
    args.resolved_env_platform = platform_name
    args.resolved_env_bin = env_bin


def validate_unreal_env_config(args: argparse.Namespace) -> None:
    env_bin = getattr(args, "resolved_env_bin", None)
    if not env_bin:
        return
    env_bin_path = path_from_unrealcv_setting(env_bin, Path(args.resolved_env_root))
    if not env_bin_path.exists():
        raise FileNotFoundError(
            f"Unreal binary for platform '{args.resolved_env_platform}' was not found: {env_bin_path}. "
            "Use --env_root or --env_bin to point at the packaged Unreal environment."
        )
    if args.resolved_env_platform == "mac" and env_bin_path.suffix == ".app":
        executable = find_macos_app_executable(env_bin_path)
        unrealcv_ini = find_macos_unrealcv_ini(env_bin_path)
        if not executable.exists():
            raise FileNotFoundError(f"macOS Unreal .app is missing its executable: {executable}")
        if not unrealcv_ini.exists():
            raise FileNotFoundError(f"macOS Unreal .app is missing unrealcv.ini: {unrealcv_ini}")
    if args.resolved_env_platform == "win":
        unrealcv_ini = env_bin_path.parent / "unrealcv.ini"
        if not unrealcv_ini.exists():
            raise FileNotFoundError(
                f"Windows Unreal binary was found, but unrealcv.ini is missing next to it: {unrealcv_ini}. "
                "Point --env_bin at the executable under Binaries/Win64, not the top-level launcher."
            )


def unreal_process_names_for_args(args: argparse.Namespace) -> List[str]:
    names: Set[str] = {"UE4_ExampleScene", "UE4_ExampleScene.exe"}
    env_root_value = getattr(args, "resolved_env_root", None) or args.env_root or DEFAULT_ENV_ROOT
    env_bin_value = getattr(args, "resolved_env_bin", None) or args.env_bin or DEFAULT_WIN_ENV_BIN
    try:
        env_bin_path = path_from_unrealcv_setting(str(env_bin_value), Path(env_root_value))
        if env_bin_path.suffix == ".app":
            names.add(find_macos_app_executable(env_bin_path).name)
        elif env_bin_path.name:
            names.add(env_bin_path.name)
            names.add(env_bin_path.stem)
    except Exception:
        raw_name = Path(str(env_bin_value)).name
        if raw_name:
            names.add(raw_name)
            names.add(Path(raw_name).stem)
    return sorted(name for name in names if name)


def force_kill_unreal_processes(args: argparse.Namespace) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    names = unreal_process_names_for_args(args)
    if host_platform() == "win":
        expanded: Set[str] = set()
        for name in names:
            expanded.add(name if name.lower().endswith(".exe") else f"{name}.exe")
        for name in sorted(expanded):
            try:
                completed = subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", name],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                results.append({
                    "name": name,
                    "returncode": completed.returncode,
                    "stdout": (completed.stdout or "").strip(),
                    "stderr": (completed.stderr or "").strip(),
                })
            except Exception as exc:
                results.append({"name": name, "error": str(exc)})
        return results

    signal_names = [name[:-4] if name.lower().endswith(".exe") else name for name in names]
    for name in sorted(set(signal_names)):
        for signal_name in ("-TERM", "-KILL"):
            try:
                completed = subprocess.run(
                    ["pkill", signal_name, "-x", name],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                results.append({
                    "name": name,
                    "signal": signal_name,
                    "returncode": completed.returncode,
                    "stdout": (completed.stdout or "").strip(),
                    "stderr": (completed.stderr or "").strip(),
                })
            except Exception as exc:
                results.append({"name": name, "signal": signal_name, "error": str(exc)})
    return results


def patch_env_setting_loader(args: argparse.Namespace) -> None:
    env_bin = getattr(args, "resolved_env_bin", None)
    if not env_bin:
        return

    from gym_unrealcv.envs.utils import misc

    original_loader = getattr(misc.load_env_setting, "_original_loader", misc.load_env_setting)
    platform_key = {
        "linux": "env_bin",
        "mac": "env_bin_mac",
        "win": "env_bin_win",
    }.get(args.resolved_env_platform)

    def load_env_setting_with_launch_config(filename: str) -> Dict[str, Any]:
        setting = original_loader(filename)
        if platform_key:
            setting[platform_key] = env_bin
        return setting

    load_env_setting_with_launch_config._original_loader = original_loader  # type: ignore[attr-defined]
    misc.load_env_setting = load_env_setting_with_launch_config


def patch_unrealcv_client_thread_lifecycle() -> None:
    import unrealcv

    Client = unrealcv.Client
    if getattr(Client, "_uav_thread_lifecycle_patch", False):
        return

    def connect_with_daemon_receive_thread(self: Any, timeout: float = 1) -> bool:
        if self.isconnected():
            return True
        try:
            if self.type == "unix":
                print("=>Info: using uds socket")
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            elif self.type == "inet":
                print("=>Info: using ip-port socket")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            else:
                raise NotImplementedError
            sock.connect(self.endpoint)
            self.sock = sock
            message = unrealcv.SocketMessage.ReceivePayload(self.sock)
            if message is not None and message.startswith(b"connected"):
                self.t = threading.Thread(target=self.receive_loop_queue, daemon=True)
                self.t.start()
                return True
            self.disconnect()
            return False
        except Exception as exc:
            LOGGER.debug("UnrealCV connect failed: %s", exc)
            self.disconnect()
            return False

    def disconnect_with_timeout(self: Any) -> None:
        try:
            if self.isconnected():
                try:
                    self.sock.shutdown(socket.SHUT_RD)
                except Exception:
                    pass
                try:
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass
                self.sock = None
                time.sleep(0.1)
        finally:
            thread = getattr(self, "t", None)
            if thread is not None and thread.is_alive():
                try:
                    self.recv_num_q.put(None)
                except Exception:
                    pass
                thread.join(timeout=0.5)

    Client.connect = connect_with_daemon_receive_thread
    Client.disconnect = disconnect_with_timeout
    Client._uav_thread_lifecycle_patch = True


def print_launch_config(args: argparse.Namespace) -> None:
    print("Unreal launch config:")
    print(f"  platform: {args.resolved_env_platform}")
    print(f"  env_root: {args.resolved_env_root}")
    print(f"  env_bin:  {args.resolved_env_bin or '(from setting file)'}")


def ensure_legacy_gym_entrypoint_loader() -> None:
    import importlib
    from gym.envs import registration

    def load(entry_point: str):
        module_name, attr_name = entry_point.split(":", 1)
        obj = importlib.import_module(module_name)
        for part in attr_name.split("."):
            obj = getattr(obj, part)
        return obj

    registration.load = load


def patch_macos_launchservices_launcher() -> None:
    if "darwin" not in sys.platform:
        return

    import atexit
    from unrealcv.launcher import RunUnreal

    if getattr(RunUnreal, "_launchservices_patch", False):
        return

    original_init = RunUnreal.__init__
    original_start = RunUnreal.start
    original_close = RunUnreal.close

    def init_with_app_support(self, ENV_BIN, ENV_MAP=None):
        env_bin_text = str(ENV_BIN)
        env_root = Path(os.getenv("UnrealEnv", ".")).expanduser().resolve()
        app_bundle = path_from_unrealcv_setting(env_bin_text, env_root)
        if app_bundle.suffix == ".app" and app_bundle.exists():
            self.path2env = str(env_root)
            self.env_bin = path_for_unrealcv_setting(app_bundle, env_root)
            self.env_map = ENV_MAP
            self.path2unrealcv = str(find_macos_unrealcv_ini(app_bundle))
            self.path2binary = str(find_macos_app_executable(app_bundle))
            assert os.path.exists(self.path2binary), \
                "Please load env binary in UnrealEnv and Check the env_bin in setting file!"
            self.ue_pid = None
            return
        original_init(self, ENV_BIN, ENV_MAP)

    def find_unreal_pids(binary_path: str) -> List[int]:
        process_name = Path(binary_path).name
        try:
            output = subprocess.check_output(["pgrep", "-x", process_name], text=True)
        except subprocess.CalledProcessError:
            return []
        pids = []
        for value in output.split():
            try:
                pid = int(value)
            except ValueError:
                continue
            if pid != os.getpid():
                pids.append(pid)
        return pids

    def start_with_launchservices(self, docker=False, resolution=(640, 480), display=None, opengl=False,
                                  offscreen=False, nullrhi=False, gpu_id=None, local_host=True,
                                  sleep_time=8, log_file_path=None):
        app_bundle = next((path for path in Path(self.path2binary).parents if path.suffix == ".app"), None)
        if docker or app_bundle is None:
            return original_start(self, docker=docker, resolution=resolution, display=display, opengl=opengl,
                                  offscreen=offscreen, nullrhi=nullrhi, gpu_id=gpu_id, local_host=local_host,
                                  sleep_time=sleep_time, log_file_path=log_file_path)

        port = self.read_port()
        self.write_resolution(resolution)
        self.use_docker = False
        env_ip = "127.0.0.1"
        if local_host:
            while not self.isPortFree(env_ip, port):
                port += 1
                self.write_port(port)

        cmd_exe = [os.path.abspath(self.path2binary)]
        self.set_ue_options(cmd_exe, opengl, offscreen, nullrhi, gpu_id)
        subprocess.run(["open", "-n", str(app_bundle), "--args", *cmd_exe[1:]], check=True)
        time.sleep(sleep_time)

        pids = find_unreal_pids(self.path2binary)
        self.ue_pid = pids[-1] if pids else None
        self.env = None
        atexit.register(self.close)
        print("Running macOS LaunchServices env, pid:{}".format(self.ue_pid or "unknown"))
        print("Please wait for a while to launch env......")
        return env_ip, port

    def close_with_launchservices(self):
        if getattr(self, "use_docker", False):
            return original_close(self)
        ue_pid = getattr(self, "ue_pid", None)
        if ue_pid is None:
            pids = find_unreal_pids(getattr(self, "path2binary", ""))
            ue_pid = pids[-1] if pids else None
        if ue_pid is None:
            return
        try:
            os.kill(ue_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    RunUnreal.__init__ = init_with_app_support
    RunUnreal.start = start_with_launchservices
    RunUnreal.close = close_with_launchservices
    RunUnreal._launchservices_patch = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch UnrealCV and run a drone flight controller or scripted flight.",
        epilog=(
            "Examples:\n"
            "  Windows: python run_drone_flight.py --env_platform win --mode keyboard\n"
            "  macOS:   python run_drone_flight.py --env_platform mac --env_root /path/to/UnrealEnv --mode keyboard"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_id", default=DEFAULT_ENV_ID, help="Gym UnrealCV environment id")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for images and trajectory logs")
    parser.add_argument("--env_platform", "--platform", choices=["auto", "win", "mac", "linux"], default="auto",
                        help="Packaged Unreal environment platform to launch; auto uses the current OS")
    parser.add_argument("--env_root", default=None,
                        help=f"Directory containing packaged Unreal environments; default is ./{DEFAULT_ENV_ROOT}")
    parser.add_argument("--env_bin", default=None,
                        help="Packaged Unreal binary or .app bundle, relative to --env_root or absolute")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the resolved Unreal launch config and exit without starting Unreal")
    parser.add_argument("--width", type=int, default=256, help="Observation width")
    parser.add_argument("--height", type=int, default=256, help="Observation height")
    parser.add_argument("--launch_sleep", type=int, default=int(os.getenv("UNREALCV_LAUNCH_SLEEP", "5")),
                        help="Seconds to wait for the UnrealCV server to start")
    parser.add_argument("--time_dilation", type=int, default=10, help="Time dilation reference fps; use 0 to disable")
    parser.add_argument("--mode", choices=["keyboard", "scripted", "orbit"], default="keyboard",
                        help="keyboard keeps the drone view alive for manual control; scripted runs the smoke test plan; "
                             "orbit flies one deterministic loop around a house center")
    parser.add_argument("--keyboard_backend", choices=["global", "terminal"], default="global",
                        help="global captures keys while the Unreal window is focused; terminal reads keys from stdin")
    parser.add_argument("--control_dt", type=float, default=0.1, help="Keyboard control step interval, in seconds")
    parser.add_argument("--linear_speed", type=float, default=0.5, help="Keyboard X/Y movement speed")
    parser.add_argument("--vertical_speed", type=float, default=0.5, help="Keyboard upward movement speed")
    parser.add_argument("--down_speed", type=float, default=0.4, help="Keyboard downward movement speed")
    parser.add_argument("--yaw_speed", type=float, default=0.8, help="Keyboard yaw speed")
    parser.add_argument("--max_steps", type=int, default=0, help="Maximum keyboard steps; 0 means unlimited")
    parser.add_argument("--auto_action",
                        choices=["none", "hover", "up", "down", "forward", "backward", "left", "right",
                                 "yaw_left", "yaw_right"],
                        default="none",
                        help="Run one action automatically for testing instead of reading the keyboard")
    parser.add_argument("--drone_index", type=int, default=None,
                        help="Index in env.unwrapped.player_list; default is the first/only drone")
    parser.add_argument("--keep_reset_pose", action="store_true",
                        help="Keep selected drone at its reset-time pose instead of moving it to --initial_pos")
    parser.add_argument("--steps_per_action", type=int, default=20, help="Repeated env steps for each test action")
    parser.add_argument("--step_delay", type=float, default=0.1, help="Delay after each step, in seconds")
    parser.add_argument("--save_every", type=int, default=0, help="Save one image every N steps; use 0 to disable")
    parser.add_argument("--enhance_rgb", dest="enhance_rgb", action="store_true", default=DEFAULT_RGB_ENHANCE_ENABLED,
                        help="Brighten saved RGB frames and controller previews")
    parser.add_argument("--no_enhance_rgb", dest="enhance_rgb", action="store_false",
                        help="Save/show raw RGB frames without brightness enhancement")
    parser.add_argument("--rgb_enhance_gamma", type=float, default=DEFAULT_RGB_ENHANCE_GAMMA,
                        help="Gamma used for RGB output enhancement; lower values brighten shadows")
    parser.add_argument("--rgb_enhance_gain", type=float, default=DEFAULT_RGB_ENHANCE_GAIN,
                        help="Gain used for RGB output enhancement")
    parser.add_argument("--rgb_source_order", choices=["rgb", "bgr"], default=DEFAULT_RGB_SOURCE_ORDER,
                        help="Color channel order returned by UnrealCV before display/save")
    parser.add_argument("--temp_capture_dir", default=DEFAULT_TEMP_CAPTURE_DIR,
                        help="Directory for controller Temp Capture bundles")
    parser.add_argument("--temp_capture_lidar_dir", default=DEFAULT_TEMP_CAPTURE_LIDAR_DIR,
                        help="Directory for controller Temp Capture Lidar point cloud bundles")
    parser.add_argument("--stream_capture_dir", default=DEFAULT_STREAM_CAPTURE_DIR,
                        help="Directory for timed stream capture task folders")
    parser.add_argument("--stream_capture_lidar_dir", default=DEFAULT_STREAM_CAPTURE_LIDAR_DIR,
                        help="Directory for timed stream capture lidar point cloud task folders")
    parser.add_argument("--stream_interval_s", type=float, default=DEFAULT_STREAM_CAPTURE_INTERVAL_S,
                        help="Default interval in seconds for controller stream capture")
    parser.add_argument("--depth_min_cm", type=float, default=DEFAULT_DEPTH_MIN_CM,
                        help="Minimum depth in cm for temp depth preview scaling")
    parser.add_argument("--depth_max_cm", type=float, default=DEFAULT_DEPTH_MAX_CM,
                        help="Maximum depth in cm for temp depth preview scaling")
    parser.add_argument("--lidar_depth_min_cm", type=float, default=DEFAULT_LIDAR_DEPTH_MIN_CM,
                        help="Minimum depth in cm retained in Temp Capture Lidar point clouds")
    parser.add_argument("--lidar_depth_max_cm", type=float, default=DEFAULT_LIDAR_DEPTH_MAX_CM,
                        help="Maximum depth in cm retained in Temp Capture Lidar point clouds")
    parser.add_argument("--lidar_depth_projection", choices=["auto", "plane", "ray"], default=DEFAULT_LIDAR_DEPTH_PROJECTION,
                        help="Depth backprojection for standard Lidar clouds; auto uses plane depth for UnrealCV depth npy")
    parser.add_argument("--lidar_capture_processing", choices=["smooth", "full"], default=DEFAULT_LIDAR_CAPTURE_PROCESSING,
                        help="Lidar stream processing mode: smooth stores raw frames first; full writes point clouds every frame")
    parser.add_argument("--force_kill_unreal_on_stop", dest="force_kill_unreal_on_stop", action="store_true",
                        default=DEFAULT_FORCE_KILL_UNREAL_ON_STOP,
                        help="Force-kill packaged Unreal processes when stopping a session")
    parser.add_argument("--no_force_kill_unreal_on_stop", dest="force_kill_unreal_on_stop", action="store_false",
                        help="Use only UnrealCV/gym close without taskkill/pkill on stop")
    parser.add_argument("--movement_mode", choices=["pose_lock", "physics"], default=DEFAULT_MOVEMENT_MODE,
                        help="Controller movement mode: pose_lock directly sets target poses; physics uses drone velocity")
    parser.add_argument("--pose_settle_timeout", type=float, default=DEFAULT_POSE_SETTLE_TIMEOUT,
                        help="Seconds to wait for pose_lock pose convergence")
    parser.add_argument("--pose_settle_pos_tolerance", type=float, default=DEFAULT_POSE_SETTLE_POS_TOLERANCE,
                        help="Position tolerance in cm for pose_lock convergence")
    parser.add_argument("--pose_settle_yaw_tolerance", type=float, default=DEFAULT_POSE_SETTLE_YAW_TOLERANCE,
                        help="Yaw tolerance in degrees for pose_lock convergence")
    parser.add_argument("--initial_pos", nargs="+", type=float, default=DEFAULT_INITIAL_POS,
                        metavar="POSE",
                        help="Initial drone pose after reset: X Y Z YAW or X Y Z ROLL YAW PITCH")
    parser.add_argument("--orbit_center", nargs=2, type=float, default=DEFAULT_ORBIT_CENTER,
                        metavar=("X", "Y"),
                        help="House center for --mode orbit, in Unreal world X/Y coordinates")
    parser.add_argument("--orbit_radius", type=float, default=DEFAULT_ORBIT_RADIUS,
                        help="Radius for --mode orbit")
    parser.add_argument("--orbit_altitude", type=float, default=DEFAULT_ORBIT_ALTITUDE,
                        help="Drone altitude for --mode orbit")
    parser.add_argument("--orbit_steps", type=int, default=DEFAULT_ORBIT_STEPS,
                        help="Number of segments in the full 360-degree orbit")
    parser.add_argument("--orbit_start_angle", type=float, default=DEFAULT_ORBIT_START_ANGLE,
                        help="Starting angle around --orbit_center, in degrees")
    parser.add_argument("--orbit_clockwise", action="store_true",
                        help="Fly the orbit clockwise instead of counter-clockwise")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def default_session_args(**overrides: Any) -> argparse.Namespace:
    args = build_arg_parser().parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(f"Unknown run_drone_flight argument: {key}")
        setattr(args, key, value)
    return args


def make_output_dir(base_dir: str) -> Path:
    run_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_project_output_path(value: Any, default_value: str) -> Path:
    raw = str(value or default_value).strip() or default_value
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def sanitize_capture_task_title(title: Any, default_value: str = "stream_task") -> str:
    text = str(title or "").strip()
    invalid_chars = set('<>:"/\\|?*')
    pieces: List[str] = []
    last_separator = False
    for char in text:
        if char in invalid_chars or ord(char) < 32 or char.isspace():
            if not last_separator:
                pieces.append("_")
                last_separator = True
            continue
        pieces.append(char)
        last_separator = False
    cleaned = "".join(pieces).strip(" ._")
    return (cleaned[:80] or default_value)


def make_unique_child_dir(root_path: Path, dirname: str) -> Path:
    root_path.mkdir(parents=True, exist_ok=True)
    base = root_path / dirname
    for index in range(1000):
        candidate = base if index == 0 else root_path / f"{dirname}-{index:02d}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique directory under {root_path}")


def prepare_observation_rgb(
    observation: Any,
    *,
    enhance: bool = DEFAULT_RGB_ENHANCE_ENABLED,
    gamma: float = DEFAULT_RGB_ENHANCE_GAMMA,
    gain: float = DEFAULT_RGB_ENHANCE_GAIN,
    source_order: str = DEFAULT_RGB_SOURCE_ORDER,
) -> Optional[np.ndarray]:
    image = np.asarray(observation)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        return None
    image = image[:, :, :3]
    image = np.clip(image, 0, 255).astype(np.uint8)
    if str(source_order or DEFAULT_RGB_SOURCE_ORDER).strip().lower() == "bgr":
        image = image[:, :, ::-1]
    if not enhance:
        return image
    gamma_value = max(0.2, min(2.5, float(gamma)))
    gain_value = max(0.1, min(4.0, float(gain)))
    linear = image.astype(np.float32) / 255.0
    linear = np.power(np.clip(linear * gain_value, 0.0, 1.0), gamma_value)
    return np.clip(linear * 255.0, 0, 255).astype(np.uint8)


def rgb_enhance_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "enhance": bool(getattr(args, "enhance_rgb", DEFAULT_RGB_ENHANCE_ENABLED)),
        "gamma": float(getattr(args, "rgb_enhance_gamma", DEFAULT_RGB_ENHANCE_GAMMA)),
        "gain": float(getattr(args, "rgb_enhance_gain", DEFAULT_RGB_ENHANCE_GAIN)),
        "source_order": str(getattr(args, "rgb_source_order", DEFAULT_RGB_SOURCE_ORDER) or DEFAULT_RGB_SOURCE_ORDER),
    }


def save_color_observation(
    observation: Any,
    path: Path,
    *,
    enhance: bool = DEFAULT_RGB_ENHANCE_ENABLED,
    gamma: float = DEFAULT_RGB_ENHANCE_GAMMA,
    gain: float = DEFAULT_RGB_ENHANCE_GAIN,
    source_order: str = DEFAULT_RGB_SOURCE_ORDER,
) -> None:
    image = prepare_observation_rgb(observation, enhance=enhance, gamma=gamma, gain=gain, source_order=source_order)
    if image is None:
        return
    Image.fromarray(image).save(path)


def save_color_observation_for_args(args: argparse.Namespace, observation: Any, path: Path) -> None:
    save_color_observation(observation, path, **rgb_enhance_options(args))


def coerce_depth_planar_image(depth_image: Any) -> np.ndarray:
    depth = np.asarray(depth_image)
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            depth = depth[:, :, 0]
    if depth.ndim != 2:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be 2D after coercion, got shape {depth.shape}")
    return depth.astype(np.float32, copy=False)


def summarize_depth_image(
    depth_image: Any,
    *,
    min_depth_cm: float = DEFAULT_DEPTH_MIN_CM,
    max_depth_cm: float = DEFAULT_DEPTH_MAX_CM,
) -> Dict[str, Any]:
    depth = coerce_depth_planar_image(depth_image)
    finite_depth = depth[np.isfinite(depth)]
    valid_depth = finite_depth[
        (finite_depth >= float(min_depth_cm))
        & (finite_depth <= float(max_depth_cm))
    ]
    h, w = depth.shape[:2]
    patch = depth[int(h * 0.55):int(h * 0.9), int(w * 0.4):int(w * 0.6)]
    patch_valid = patch[np.isfinite(patch)]
    return {
        "available": bool(finite_depth.size),
        "min_depth": float(np.min(valid_depth)) if valid_depth.size else float(min_depth_cm),
        "max_depth": float(np.max(valid_depth)) if valid_depth.size else float(max_depth_cm),
        "front_min_depth": float(np.min(patch_valid)) if patch_valid.size else 0.0,
        "front_mean_depth": float(np.mean(patch_valid)) if patch_valid.size else 0.0,
        "finite_count": int(finite_depth.size),
        "valid_count": int(valid_depth.size),
        "image_width": int(w),
        "image_height": int(h),
        "source_mode": "unrealcv_depth_npy",
    }


def render_depth_preview(
    depth_image: Any,
    *,
    min_depth_cm: float = DEFAULT_DEPTH_MIN_CM,
    max_depth_cm: float = DEFAULT_DEPTH_MAX_CM,
    source_mode: str = "unrealcv_depth_npy",
) -> np.ndarray:
    depth = coerce_depth_planar_image(depth_image)
    canvas = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    canvas[:] = (12, 12, 18)
    finite = depth[np.isfinite(depth)]
    if finite.size:
        preview_min = float(np.min(finite)) if min_depth_cm <= 0 else float(min_depth_cm)
        preview_max = float(np.max(finite)) if max_depth_cm <= preview_min else float(max_depth_cm)
        if preview_max <= preview_min:
            preview_max = preview_min + 1.0
        valid_mask = np.isfinite(depth) & (depth >= preview_min) & (depth <= preview_max)
        clipped = np.clip(depth, preview_min, preview_max)
        normalized = 1.0 - ((clipped - preview_min) / (preview_max - preview_min))
        preview_u8 = np.clip(np.nan_to_num(normalized) * 255.0, 0.0, 255.0).astype(np.uint8)
        canvas = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
        canvas[~valid_mask] = (16, 16, 20)
    cv2.putText(
        canvas,
        f"Depth mode: {source_mode}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_depth_capture_outputs(
    depth_image: Any,
    *,
    depth_cm_path: Path,
    depth_preview_path: Path,
    depth_npy_path: Path,
    min_depth_cm: float = DEFAULT_DEPTH_MIN_CM,
    max_depth_cm: float = DEFAULT_DEPTH_MAX_CM,
) -> Dict[str, Any]:
    depth = coerce_depth_planar_image(depth_image)
    depth_u16 = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(depth_u16).save(depth_cm_path)
    Image.fromarray(
        render_depth_preview(depth, min_depth_cm=min_depth_cm, max_depth_cm=max_depth_cm)
    ).save(depth_preview_path)
    np.save(depth_npy_path, depth.astype(np.float32, copy=False))
    return summarize_depth_image(depth, min_depth_cm=min_depth_cm, max_depth_cm=max_depth_cm)


def parse_camera_fov_degrees(value: Any, default_value: float = 90.0) -> Tuple[float, str]:
    try:
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        fov = float(str(value).strip())
        if math.isfinite(fov) and 1.0 <= fov <= 179.0:
            return fov, "unrealcv"
    except Exception:
        pass
    return float(default_value), "fallback"


def camera_intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> Dict[str, float]:
    w = max(1, int(width))
    h = max(1, int(height))
    fov_rad = math.radians(max(1.0, min(179.0, float(horizontal_fov_deg))))
    fx = float(w) / (2.0 * math.tan(fov_rad / 2.0))
    return {
        "fx": fx,
        "fy": fx,
        "cx": (float(w) - 1.0) / 2.0,
        "cy": (float(h) - 1.0) / 2.0,
        "horizontal_fov_deg": float(horizontal_fov_deg),
    }


def unreal_rotation_matrix_from_camera(rotation: Dict[str, Any]) -> np.ndarray:
    pitch = math.radians(float(rotation.get("pitch", 0.0) or 0.0))
    yaw = math.radians(float(rotation.get("yaw", 0.0) or 0.0))
    roll = math.radians(float(rotation.get("roll", 0.0) or 0.0))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp], dtype=np.float32)
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp], dtype=np.float32)
    return np.stack([forward, right, up], axis=1)


def normalize_lidar_depth_projection(value: Any) -> str:
    projection = str(value or DEFAULT_LIDAR_DEPTH_PROJECTION).strip().lower()
    if projection not in {"auto", "plane", "ray"}:
        return DEFAULT_LIDAR_DEPTH_PROJECTION
    return projection


def normalize_lidar_capture_processing(value: Any) -> str:
    mode = str(value or DEFAULT_LIDAR_CAPTURE_PROCESSING).strip().lower()
    if mode not in {"smooth", "full"}:
        return DEFAULT_LIDAR_CAPTURE_PROCESSING
    return mode


def select_lidar_depth_projection(value: Any, *, depth_input_mode: str = "unrealcv_depth_npy") -> str:
    projection = normalize_lidar_depth_projection(value)
    if projection == "ray":
        return "ray_depth"
    # UnrealCV get_depth returns a depth image that behaves as planar camera Z for
    # this visualization path; keep ray-depth as diagnostics instead of default.
    return "plane_depth"


def standard_voxel_size_m(voxel_cm: float = DEFAULT_LIDAR_RECON_VOXEL_CM) -> float:
    voxel = float(voxel_cm)
    if not math.isfinite(voxel) or voxel <= 0.0:
        return 0.0
    return voxel / 100.0


def point_cloud_xyz_stats(point_cloud: np.ndarray) -> Dict[str, Any]:
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        return {"point_count": 0}
    xyz = points[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    if not np.all(finite):
        xyz = xyz[finite]
    if xyz.shape[0] == 0:
        return {"point_count": 0}
    return {
        "point_count": int(xyz.shape[0]),
        "min_xyz": [float(value) for value in np.nanmin(xyz, axis=0)],
        "max_xyz": [float(value) for value in np.nanmax(xyz, axis=0)],
        "mean_xyz": [float(value) for value in np.nanmean(xyz, axis=0)],
    }


def unreal_world_cloud_to_standard_zup_m(point_cloud: np.ndarray) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        return np.zeros((0, 6), dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyz = points[:, :3].astype(np.float32, copy=False)
    standard_xyz = np.column_stack((xyz[:, 0], -xyz[:, 1], xyz[:, 2])).astype(np.float32, copy=False) / 100.0
    return np.column_stack((standard_xyz, points[:, 3:6])).astype(np.float32, copy=False)


def standard_world_m_to_unreal_world_cm(point_cloud: np.ndarray) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        return np.zeros((0, 6), dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyz = points[:, :3].astype(np.float32, copy=False) * 100.0
    unreal_xyz = np.column_stack((xyz[:, 0], -xyz[:, 1], xyz[:, 2])).astype(np.float32, copy=False)
    return np.column_stack((unreal_xyz, points[:, 3:6])).astype(np.float32, copy=False)


def build_standard_point_clouds_from_depth(
    rgb_image: np.ndarray,
    depth_image: Any,
    camera_info: Dict[str, Any],
    *,
    min_depth_cm: float = DEFAULT_DEPTH_MIN_CM,
    max_depth_cm: float = DEFAULT_DEPTH_MAX_CM,
    lidar_depth_projection: str = DEFAULT_LIDAR_DEPTH_PROJECTION,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    depth_cm = coerce_depth_planar_image(depth_image)
    h, w = depth_cm.shape[:2]
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB image must be HxWx3, got shape {rgb.shape}")
    rgb_resized = False
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb[:, :, :3], (w, h), interpolation=cv2.INTER_LINEAR)
        rgb_resized = True
    else:
        rgb = rgb[:, :, :3]

    fov, fov_source = parse_camera_fov_degrees(camera_info.get("horizontal_fov_deg"))
    intrinsics = camera_intrinsics_from_fov(w, h, fov)
    ys, xs = np.indices((h, w), dtype=np.float32)
    depth_cm = depth_cm.astype(np.float32, copy=False)
    min_depth = max(0.0, float(min_depth_cm))
    max_depth = float(max_depth_cm)
    if not math.isfinite(max_depth) or max_depth <= min_depth:
        max_depth = float(np.nanmax(depth_cm)) if np.isfinite(depth_cm).any() else min_depth + 1.0
    valid = np.isfinite(depth_cm) & (depth_cm >= min_depth) & (depth_cm <= max_depth)
    invalid_depth_count = int(depth_cm.size - np.count_nonzero(valid))
    requested_projection = normalize_lidar_depth_projection(lidar_depth_projection)
    selected_projection = select_lidar_depth_projection(requested_projection)

    base_summary: Dict[str, Any] = {
        "point_count": 0,
        "invalid_depth_count": invalid_depth_count,
        "image_width": int(w),
        "image_height": int(h),
        "intrinsics": intrinsics,
        "fov_source": fov_source,
        "rgb_resized_for_depth": bool(rgb_resized),
        "min_depth_cm": min_depth,
        "max_depth_cm": max_depth,
        "depth_input_mode": "unrealcv_depth_npy",
        "depth_projection_requested": requested_projection,
        "depth_projection_selected": selected_projection,
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
        "projection_corrected": True,
        "coordinate_conventions": {
            "camera_standard_m": "right-handed camera: x right, y up, z forward; units m",
            "world_standard_m": "right-handed world: x Unreal X, y -Unreal Y, z Unreal Z; units m",
            "legacy_unreal_cm": "Unreal world coordinates; units cm",
        },
    }
    if np.count_nonzero(valid) <= 0:
        empty = np.zeros((0, 6), dtype=np.float32)
        base_summary["projection_comparison"] = {
            "plane_depth": {"point_count": 0},
            "ray_depth": {"point_count": 0},
        }
        return empty, empty, base_summary

    x_norm = (xs - float(intrinsics["cx"])) / float(intrinsics["fx"])
    y_norm = (ys - float(intrinsics["cy"])) / float(intrinsics["fy"])
    ray_norm = np.sqrt(1.0 + np.square(x_norm) + np.square(y_norm)).astype(np.float32, copy=False)
    z_plane = depth_cm
    z_ray = depth_cm / ray_norm
    selected_z = z_ray if selected_projection == "ray_depth" else z_plane
    x_right = x_norm * selected_z
    y_down = y_norm * selected_z
    colors = np.clip(rgb.astype(np.float32), 0.0, 255.0)
    color_values = colors[valid].astype(np.float32, copy=False)

    camera_xyz_m = np.column_stack(
        (
            x_right[valid],
            -y_down[valid],
            selected_z[valid],
        )
    ).astype(np.float32, copy=False) / 100.0
    camera_cloud_standard_m = np.column_stack((camera_xyz_m, color_values)).astype(np.float32, copy=False)

    local_unreal_cm = np.column_stack(
        (
            selected_z[valid],
            x_right[valid],
            -y_down[valid],
        )
    ).astype(np.float32, copy=False)
    location = camera_info.get("location", {}) if isinstance(camera_info.get("location"), dict) else {}
    camera_location = np.array(
        [
            float(location.get("x", 0.0) or 0.0),
            float(location.get("y", 0.0) or 0.0),
            float(location.get("z", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )
    rotation = camera_info.get("rotation", {}) if isinstance(camera_info.get("rotation"), dict) else {}
    world_unreal_cm = local_unreal_cm @ unreal_rotation_matrix_from_camera(rotation).T + camera_location
    world_xyz_m = np.column_stack(
        (
            world_unreal_cm[:, 0],
            -world_unreal_cm[:, 1],
            world_unreal_cm[:, 2],
        )
    ).astype(np.float32, copy=False) / 100.0
    world_cloud_standard_m = np.column_stack((world_xyz_m, color_values)).astype(np.float32, copy=False)

    plane_camera_xyz_m = np.column_stack(
        (
            (x_norm * z_plane)[valid],
            -(y_norm * z_plane)[valid],
            z_plane[valid],
        )
    ).astype(np.float32, copy=False) / 100.0
    ray_camera_xyz_m = np.column_stack(
        (
            (x_norm * z_ray)[valid],
            -(y_norm * z_ray)[valid],
            z_ray[valid],
        )
    ).astype(np.float32, copy=False) / 100.0
    z_delta_cm = z_plane[valid] - z_ray[valid]
    base_summary.update(
        {
            "point_count": int(camera_cloud_standard_m.shape[0]),
            "projection_comparison": {
                "plane_depth": point_cloud_xyz_stats(np.column_stack((plane_camera_xyz_m, color_values))),
                "ray_depth": point_cloud_xyz_stats(np.column_stack((ray_camera_xyz_m, color_values))),
                "mean_plane_minus_ray_z_cm": float(np.nanmean(z_delta_cm)) if z_delta_cm.size else 0.0,
                "max_abs_plane_minus_ray_z_cm": float(np.nanmax(np.abs(z_delta_cm))) if z_delta_cm.size else 0.0,
            },
            "camera_standard_m_stats": point_cloud_xyz_stats(camera_cloud_standard_m),
            "world_standard_m_stats": point_cloud_xyz_stats(world_cloud_standard_m),
        }
    )
    return camera_cloud_standard_m, world_cloud_standard_m, base_summary


def build_colored_point_clouds_from_depth(
    rgb_image: np.ndarray,
    depth_image: Any,
    camera_info: Dict[str, Any],
    *,
    min_depth_cm: float = DEFAULT_DEPTH_MIN_CM,
    max_depth_cm: float = DEFAULT_DEPTH_MAX_CM,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    point_depth = coerce_depth_planar_image(depth_image)
    h, w = point_depth.shape[:2]
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB image must be HxWx3, got shape {rgb.shape}")
    rgb_resized = False
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb[:, :, :3], (w, h), interpolation=cv2.INTER_LINEAR)
        rgb_resized = True
    else:
        rgb = rgb[:, :, :3]

    fov, fov_source = parse_camera_fov_degrees(camera_info.get("horizontal_fov_deg"))
    intrinsics = camera_intrinsics_from_fov(w, h, fov)
    ys, xs = np.indices((h, w), dtype=np.float32)
    point_depth = point_depth.astype(np.float32, copy=False)
    min_depth = max(0.0, float(min_depth_cm))
    max_depth = float(max_depth_cm)
    if not math.isfinite(max_depth) or max_depth <= min_depth:
        max_depth = float(np.nanmax(point_depth)) if np.isfinite(point_depth).any() else min_depth + 1.0
    valid = np.isfinite(point_depth) & (point_depth >= min_depth) & (point_depth <= max_depth)
    invalid_depth_count = int(point_depth.size - np.count_nonzero(valid))

    if np.count_nonzero(valid) <= 0:
        empty = np.zeros((0, 6), dtype=np.float32)
        return empty, empty, {
            "point_count": 0,
            "invalid_depth_count": invalid_depth_count,
            "image_width": int(w),
            "image_height": int(h),
            "intrinsics": intrinsics,
            "fov_source": fov_source,
            "rgb_resized_for_depth": bool(rgb_resized),
            "min_depth_cm": min_depth,
            "max_depth_cm": max_depth,
            "depth_input_mode": "unrealcv_point_depth_cm",
            "depth_projection": "ray_depth_legacy",
            "coordinate_frame": "legacy_unreal",
            "coordinate_units": "cm",
        }

    x_norm = (xs - float(intrinsics["cx"])) / float(intrinsics["fx"])
    y_norm = (ys - float(intrinsics["cy"])) / float(intrinsics["fy"])
    ray_norm = np.sqrt(1.0 + np.square(x_norm) + np.square(y_norm)).astype(np.float32, copy=False)
    z = point_depth / ray_norm
    x = x_norm * z
    y = y_norm * z
    colors = np.clip(rgb.astype(np.float32), 0.0, 255.0)
    camera_xyz = np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32, copy=False)
    color_values = colors[valid].astype(np.float32, copy=False)
    camera_cloud = np.column_stack((camera_xyz, color_values)).astype(np.float32, copy=False)

    local_unreal = np.column_stack((camera_xyz[:, 2], camera_xyz[:, 0], -camera_xyz[:, 1])).astype(np.float32, copy=False)
    location = camera_info.get("location", {}) if isinstance(camera_info.get("location"), dict) else {}
    camera_location = np.array(
        [
            float(location.get("x", 0.0) or 0.0),
            float(location.get("y", 0.0) or 0.0),
            float(location.get("z", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )
    rotation = camera_info.get("rotation", {}) if isinstance(camera_info.get("rotation"), dict) else {}
    world_xyz = local_unreal @ unreal_rotation_matrix_from_camera(rotation).T + camera_location
    world_cloud = np.column_stack((world_xyz.astype(np.float32, copy=False), color_values)).astype(np.float32, copy=False)

    return camera_cloud, world_cloud, {
        "point_count": int(camera_cloud.shape[0]),
        "invalid_depth_count": invalid_depth_count,
        "image_width": int(w),
        "image_height": int(h),
        "intrinsics": intrinsics,
        "fov_source": fov_source,
        "rgb_resized_for_depth": bool(rgb_resized),
        "min_depth_cm": min_depth,
        "max_depth_cm": max_depth,
        "depth_input_mode": "unrealcv_point_depth_cm",
        "depth_projection": "ray_depth_legacy",
        "coordinate_frame": "legacy_unreal",
        "coordinate_units": "cm",
        "legacy_world_cm_stats": point_cloud_xyz_stats(world_cloud),
    }


def write_colored_point_cloud_ply(path: Path, point_cloud: np.ndarray) -> None:
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError(f"Point cloud must be Nx6, got shape {points.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    colors = np.clip(np.rint(points[:, 3:6]), 0, 255).astype(np.float32, copy=False)
    payload = np.column_stack((points[:, :3], colors))
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(header)
        if payload.size:
            np.savetxt(handle, payload, fmt=("%.4f", "%.4f", "%.4f", "%d", "%d", "%d"))


def render_point_cloud_preview(
    point_cloud: np.ndarray,
    *,
    width: int = 900,
    height: int = 700,
    point_radius: int = 1,
    title: str = "Point cloud preview: camera X/Y, depth color",
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (12, 14, 18)
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6 or points.shape[0] == 0:
        cv2.putText(canvas, "Point cloud preview: no valid points", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
        return canvas

    xyz = points[:, :3]
    max_preview_points = 120000
    if xyz.shape[0] > max_preview_points:
        step = int(math.ceil(xyz.shape[0] / max_preview_points))
        xyz = xyz[::step]

    x_values = xyz[:, 0]
    y_values = xyz[:, 1]
    z_values = xyz[:, 2]
    x_min, x_max = np.percentile(x_values, [1.0, 99.0])
    y_min, y_max = np.percentile(y_values, [1.0, 99.0])
    z_min, z_max = np.percentile(z_values, [2.0, 98.0])
    if not math.isfinite(float(x_min)) or not math.isfinite(float(x_max)) or x_max <= x_min:
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values) + 1.0)
    if not math.isfinite(float(y_min)) or not math.isfinite(float(y_max)) or y_max <= y_min:
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values) + 1.0)
    if not math.isfinite(float(z_min)) or not math.isfinite(float(z_max)) or z_max <= z_min:
        z_min, z_max = float(np.min(z_values)), float(np.max(z_values) + 1.0)

    margin = 48
    plot_w = max(1, width - margin * 2)
    plot_h = max(1, height - margin * 2)
    px = np.clip(((x_values - x_min) / max(1e-6, x_max - x_min) * plot_w + margin).astype(np.int32), margin, width - margin - 1)
    py = np.clip((height - margin - ((y_values - y_min) / max(1e-6, y_max - y_min) * plot_h)).astype(np.int32), margin, height - margin - 1)
    depth_norm = 1.0 - ((np.clip(z_values, z_min, z_max) - z_min) / max(1e-6, z_max - z_min))
    depth_u8 = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)
    colors = cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)[:, 0, :]
    colors = np.maximum(colors, 42).astype(np.uint8, copy=False)
    radius = max(0, int(point_radius))
    if radius <= 0:
        canvas[py, px] = colors
    else:
        for dy in range(-radius, radius + 1):
            yy = np.clip(py + dy, margin, height - margin - 1)
            for dx in range(-radius, radius + 1):
                xx = np.clip(px + dx, margin, width - margin - 1)
                canvas[yy, xx] = colors
    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (120, 120, 120), 1)
    cv2.putText(canvas, str(title), (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"points={points.shape[0]}", (24, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def downsample_colored_point_cloud_voxel(
    point_cloud: np.ndarray,
    *,
    voxel_cm: float = DEFAULT_LIDAR_RECON_VOXEL_CM,
    max_points: int = DEFAULT_LIDAR_RECON_MAX_POINTS,
) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6 or points.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    finite = np.isfinite(points[:, :3]).all(axis=1)
    if not np.all(finite):
        points = points[finite]
    if points.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    voxel = float(voxel_cm)
    if math.isfinite(voxel) and voxel > 0.0:
        keys = np.floor(points[:, :3] / voxel).astype(np.int64, copy=False)
        _unique_keys, unique_indices = np.unique(keys, axis=0, return_index=True)
        unique_indices.sort()
        points = points[unique_indices]

    limit = int(max_points)
    if limit > 0 and points.shape[0] > limit:
        step = int(math.ceil(points.shape[0] / limit))
        points = points[::step][:limit]
    return points.astype(np.float32, copy=False)


def get_open3d_module() -> Any:
    global _OPEN3D_MODULE, _OPEN3D_IMPORT_ERROR
    if _OPEN3D_MODULE is not None:
        return _OPEN3D_MODULE
    if _OPEN3D_IMPORT_ERROR is not None:
        return None
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        _OPEN3D_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _OPEN3D_MODULE = o3d
    _OPEN3D_IMPORT_ERROR = None
    return _OPEN3D_MODULE


def open3d_status() -> Dict[str, Any]:
    o3d = get_open3d_module()
    if o3d is None:
        return {
            "available": False,
            "version": "",
            "error": _OPEN3D_IMPORT_ERROR or "open3d is not installed",
        }
    return {
        "available": True,
        "version": str(getattr(o3d, "__version__", "")),
        "error": "",
    }


def colored_point_cloud_to_open3d(point_cloud: np.ndarray) -> Tuple[Any, int]:
    o3d = get_open3d_module()
    if o3d is None:
        raise RuntimeError(open3d_status()["error"])
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError(f"Point cloud must be Nx6, got shape {points.shape}")
    finite = np.isfinite(points[:, :3]).all(axis=1)
    if not np.all(finite):
        points = points[finite]

    pcd = o3d.geometry.PointCloud()
    if points.shape[0] == 0:
        return pcd, 0

    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(
        np.clip(points[:, 3:6] / 255.0, 0.0, 1.0).astype(np.float64, copy=False)
    )
    return pcd, int(points.shape[0])


def open3d_point_cloud_to_colored_numpy(pcd: Any) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape != points.shape:
        colors = np.ones_like(points, dtype=np.float32)
    colors = np.clip(colors * 255.0, 0.0, 255.0)
    return np.column_stack((points, colors)).astype(np.float32, copy=False)


def limit_open3d_point_cloud(pcd: Any, max_points: int) -> Any:
    point_count = int(np.asarray(pcd.points).shape[0])
    limit = int(max_points)
    if limit <= 0 or point_count <= limit:
        return pcd
    step = int(math.ceil(point_count / limit))
    indices = list(range(0, point_count, step))[:limit]
    return pcd.select_by_index(indices)


def save_open3d_point_cloud_outputs(
    point_cloud: np.ndarray,
    output_dir: Path,
    *,
    basename: str,
    voxel_cm: float = DEFAULT_OPEN3D_VOXEL_CM,
    voxel_size: Optional[float] = None,
    max_points: int = DEFAULT_LIDAR_RECON_MAX_POINTS,
    estimate_normals: bool = True,
    normal_radius_cm: float = DEFAULT_OPEN3D_NORMAL_RADIUS_CM,
    normal_radius: Optional[float] = None,
    normal_max_nn: int = DEFAULT_OPEN3D_NORMAL_MAX_NN,
    coordinate_units: str = "Unreal world centimeters",
) -> Dict[str, Any]:
    status = open3d_status()
    effective_voxel = float(voxel_cm) if voxel_size is None else float(voxel_size)
    effective_normal_radius = float(normal_radius_cm) if normal_radius is None else float(normal_radius)
    summary: Dict[str, Any] = {
        "backend": "open3d",
        "available": bool(status["available"]),
        "version": status.get("version", ""),
        "error": status.get("error", ""),
        "basename": str(basename),
        "voxel_cm": float(voxel_cm),
        "voxel_size": float(effective_voxel),
        "max_points": int(max_points),
        "normal_radius_cm": float(normal_radius_cm),
        "normal_radius": float(effective_normal_radius),
        "normal_max_nn": int(normal_max_nn),
        "coordinate_units": str(coordinate_units),
    }
    if not status["available"]:
        return summary

    o3d = get_open3d_module()
    output_dir.mkdir(parents=True, exist_ok=True)
    pcd, source_count = colored_point_cloud_to_open3d(point_cloud)
    summary["source_point_count"] = int(source_count)
    ply_path = output_dir / f"{basename}.ply"
    pcd_path = output_dir / f"{basename}.pcd"
    npy_path = output_dir / f"{basename}.npy"

    if source_count <= 0:
        np.save(npy_path, np.zeros((0, 6), dtype=np.float32))
        summary.update(
            {
                "processed_point_count": 0,
                "normals_estimated": False,
                "skipped_empty": True,
                "ply_path": "",
                "pcd_path": "",
                "npy_path": str(npy_path),
            }
        )
        return summary

    if source_count > 0 and math.isfinite(effective_voxel) and effective_voxel > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(effective_voxel))
    pcd = limit_open3d_point_cloud(pcd, int(max_points))

    processed_count = int(np.asarray(pcd.points).shape[0])
    normals_estimated = False
    if estimate_normals and processed_count > 0:
        radius = max(1e-6, float(effective_normal_radius))
        max_nn = max(1, int(normal_max_nn))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        normals_estimated = True

    if processed_count <= 0:
        np.save(npy_path, np.zeros((0, 6), dtype=np.float32))
        summary.update(
            {
                "processed_point_count": 0,
                "normals_estimated": False,
                "skipped_empty": True,
                "ply_path": "",
                "pcd_path": "",
                "npy_path": str(npy_path),
            }
        )
        return summary

    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False, print_progress=False)
    o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False, print_progress=False)
    np.save(npy_path, open3d_point_cloud_to_colored_numpy(pcd))

    summary.update(
        {
            "processed_point_count": processed_count,
            "normals_estimated": normals_estimated,
            "skipped_empty": False,
            "ply_path": str(ply_path),
            "pcd_path": str(pcd_path),
            "npy_path": str(npy_path),
        }
    )
    return summary


def save_open3d_point_cloud_from_npy_task(task: Dict[str, Any]) -> Dict[str, Any]:
    npy_path = Path(str(task.get("npy_path", "")))
    output_dir = Path(str(task.get("output_dir", "")))
    basename = str(task.get("basename", npy_path.stem))
    frame_index = int(task.get("frame_index", 0) or 0)
    reuse_existing = bool(task.get("reuse_existing", False))
    ply_path = output_dir / f"{basename}.ply"
    pcd_path = output_dir / f"{basename}.pcd"
    out_npy_path = output_dir / f"{basename}.npy"
    if reuse_existing and ply_path.exists() and pcd_path.exists() and out_npy_path.exists():
        try:
            point_count = int(np.load(out_npy_path, mmap_mode="r").shape[0])
        except Exception:
            point_count = 0
        return {
            "frame_index": frame_index,
            "backend": "open3d",
            "available": True,
            "basename": basename,
            "source_path": str(npy_path),
            "source_point_count": point_count,
            "processed_point_count": point_count,
            "skipped_empty": point_count <= 0,
            "skipped_existing": True,
            "ply_path": str(ply_path) if point_count > 0 else "",
            "pcd_path": str(pcd_path) if point_count > 0 else "",
            "npy_path": str(out_npy_path),
            "coordinate_units": str(task.get("coordinate_units", "m")),
        }

    cloud = np.load(npy_path).astype(np.float32, copy=False)
    result = save_open3d_point_cloud_outputs(
        cloud,
        output_dir,
        basename=basename,
        voxel_cm=float(task.get("voxel_cm", DEFAULT_OPEN3D_VOXEL_CM)),
        voxel_size=task.get("voxel_size"),
        max_points=int(task.get("max_points", DEFAULT_LIDAR_RECON_MAX_POINTS)),
        estimate_normals=bool(task.get("estimate_normals", False)),
        normal_radius_cm=float(task.get("normal_radius_cm", DEFAULT_OPEN3D_NORMAL_RADIUS_CM)),
        normal_radius=task.get("normal_radius"),
        normal_max_nn=int(task.get("normal_max_nn", DEFAULT_OPEN3D_NORMAL_MAX_NN)),
        coordinate_units=str(task.get("coordinate_units", "m")),
    )
    result.update(
        {
            "frame_index": frame_index,
            "source_path": str(npy_path),
            "skipped_existing": False,
        }
    )
    return result


def save_lidar_reconstruction_outputs(
    point_cloud_world: np.ndarray,
    reconstruction_dir: Path,
    *,
    source_frame_count: int = 0,
    source_point_count: int = 0,
    voxel_cm: float = DEFAULT_LIDAR_RECON_VOXEL_CM,
    max_points: int = DEFAULT_LIDAR_RECON_MAX_POINTS,
    coordinate_frame: str = "legacy_unreal",
    coordinate_units: str = "cm",
) -> Dict[str, Any]:
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    units = str(coordinate_units or "cm").strip().lower()
    frame = str(coordinate_frame or ("standard_zup" if units == "m" else "legacy_unreal")).strip()
    is_standard = units == "m" or frame == "standard_zup"
    voxel_size = standard_voxel_size_m(voxel_cm) if is_standard else float(voxel_cm)
    merged_cloud = downsample_colored_point_cloud_voxel(
        point_cloud_world,
        voxel_cm=voxel_size,
        max_points=max_points,
    )
    if is_standard:
        merged_npy_path = reconstruction_dir / "merged_point_cloud_world_standard_m.npy"
        merged_ply_path = reconstruction_dir / "merged_point_cloud_world_standard_m.ply"
        reconstruction_npy_path = reconstruction_dir / "reconstruction_world_standard_m.npy"
        reconstruction_ply_path = reconstruction_dir / "reconstruction_world_standard_m.ply"
        open3d_basename = "reconstruction_world_standard_m"
        open3d_coordinate_units = "m"
        normal_radius = DEFAULT_OPEN3D_NORMAL_RADIUS_CM / 100.0
    else:
        merged_npy_path = reconstruction_dir / "merged_point_cloud_world.npy"
        merged_ply_path = reconstruction_dir / "merged_point_cloud_world.ply"
        reconstruction_npy_path = merged_npy_path
        reconstruction_ply_path = merged_ply_path
        open3d_basename = "merged_point_cloud_world_open3d"
        open3d_coordinate_units = "Unreal world centimeters"
        normal_radius = DEFAULT_OPEN3D_NORMAL_RADIUS_CM
    preview_path = reconstruction_dir / "merged_point_cloud_preview.png"
    summary_path = reconstruction_dir / "reconstruction_summary.json"

    np.save(merged_npy_path, merged_cloud)
    write_colored_point_cloud_ply(merged_ply_path, merged_cloud)
    if reconstruction_npy_path != merged_npy_path:
        np.save(reconstruction_npy_path, merged_cloud)
    if reconstruction_ply_path != merged_ply_path:
        write_colored_point_cloud_ply(reconstruction_ply_path, merged_cloud)
    Image.fromarray(
        render_point_cloud_preview(
            merged_cloud,
            title="Point cloud reconstruction: standard Z-up meters" if is_standard else "Point cloud reconstruction: world X/Y, height color",
        )
    ).save(preview_path)
    try:
        open3d_outputs = save_open3d_point_cloud_outputs(
            merged_cloud,
            reconstruction_dir / "open3d",
            basename=open3d_basename,
            voxel_cm=voxel_cm,
            voxel_size=voxel_size,
            max_points=max_points,
            estimate_normals=True,
            normal_radius_cm=DEFAULT_OPEN3D_NORMAL_RADIUS_CM,
            normal_radius=normal_radius,
            coordinate_units=open3d_coordinate_units,
        )
    except Exception as exc:
        open3d_outputs = {
            "backend": "open3d",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    summary = {
        "source_mode": "depth_backprojected_lidar",
        "reconstruction_mode": "world_pose_accumulated_voxel_downsample",
        "source_frame_count": int(source_frame_count),
        "source_point_count": int(source_point_count),
        "merged_point_count": int(merged_cloud.shape[0]),
        "voxel_cm": float(voxel_cm),
        "voxel_size": float(voxel_size),
        "max_points": int(max_points),
        "coordinate_frame": "standard_zup" if is_standard else frame,
        "coordinate_units": "m" if is_standard else units,
        "merged_point_cloud_world_npy_path": str(merged_npy_path),
        "merged_point_cloud_world_ply_path": str(merged_ply_path),
        "merged_point_cloud_preview_path": str(preview_path),
        "reconstruction_summary_path": str(summary_path),
        "open3d": open3d_outputs,
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    if is_standard:
        summary.update(
            {
                "merged_point_cloud_world_standard_m_npy_path": str(merged_npy_path),
                "merged_point_cloud_world_standard_m_ply_path": str(merged_ply_path),
                "reconstruction_world_standard_m_npy_path": str(reconstruction_npy_path),
                "reconstruction_world_standard_m_ply_path": str(reconstruction_ply_path),
                "depth_projection_selected": "standard_zup_merge",
                "projection_corrected": True,
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def save_lidar_capture_outputs(
    *,
    rgb_image: np.ndarray,
    depth_image: Any,
    camera_info: Dict[str, Any],
    output_dir: Path,
    min_depth_cm: float = DEFAULT_DEPTH_MIN_CM,
    max_depth_cm: float = DEFAULT_DEPTH_MAX_CM,
    lidar_depth_projection: str = DEFAULT_LIDAR_DEPTH_PROJECTION,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_cloud, world_cloud, summary = build_colored_point_clouds_from_depth(
        rgb_image,
        depth_image,
        camera_info,
        min_depth_cm=min_depth_cm,
        max_depth_cm=max_depth_cm,
    )
    camera_standard_m, world_standard_m, standard_summary = build_standard_point_clouds_from_depth(
        rgb_image,
        depth_image,
        camera_info,
        min_depth_cm=min_depth_cm,
        max_depth_cm=max_depth_cm,
        lidar_depth_projection=lidar_depth_projection,
    )
    camera_npy_path = output_dir / "point_cloud_camera.npy"
    world_npy_path = output_dir / "point_cloud_world.npy"
    camera_ply_path = output_dir / "point_cloud_camera.ply"
    world_ply_path = output_dir / "point_cloud_world.ply"
    camera_standard_m_npy_path = output_dir / "point_cloud_camera_standard_m.npy"
    world_standard_m_npy_path = output_dir / "point_cloud_world_standard_m.npy"
    camera_standard_m_ply_path = output_dir / "point_cloud_camera_standard_m.ply"
    world_standard_m_ply_path = output_dir / "point_cloud_world_standard_m.ply"
    preview_path = output_dir / "point_cloud_preview.png"
    camera_info_path = output_dir / "camera_info.json"
    diagnostics_path = output_dir / "projection_diagnostics.json"

    np.save(camera_npy_path, camera_cloud)
    np.save(world_npy_path, world_cloud)
    np.save(camera_standard_m_npy_path, camera_standard_m)
    np.save(world_standard_m_npy_path, world_standard_m)
    write_colored_point_cloud_ply(camera_ply_path, camera_cloud)
    write_colored_point_cloud_ply(world_ply_path, world_cloud)
    write_colored_point_cloud_ply(camera_standard_m_ply_path, camera_standard_m)
    write_colored_point_cloud_ply(world_standard_m_ply_path, world_standard_m)
    Image.fromarray(
        render_point_cloud_preview(
            camera_standard_m,
            title="Point cloud preview: standard camera meters",
        )
    ).save(preview_path)
    try:
        open3d_dir = output_dir / "open3d"
        camera_open3d_outputs = save_open3d_point_cloud_outputs(
            camera_cloud,
            open3d_dir,
            basename="point_cloud_camera_open3d",
            voxel_cm=0.0,
            max_points=0,
            estimate_normals=False,
            coordinate_units="Camera coordinates centimeters",
        )
        world_open3d_outputs = save_open3d_point_cloud_outputs(
            world_cloud,
            open3d_dir,
            basename="point_cloud_world_open3d",
            voxel_cm=0.0,
            max_points=0,
            estimate_normals=False,
            coordinate_units="Unreal world coordinates centimeters",
        )
        camera_standard_open3d_outputs = save_open3d_point_cloud_outputs(
            camera_standard_m,
            open3d_dir,
            basename="point_cloud_camera_standard_m",
            voxel_cm=0.0,
            voxel_size=0.0,
            max_points=0,
            estimate_normals=False,
            coordinate_units="m",
        )
        world_standard_open3d_outputs = save_open3d_point_cloud_outputs(
            world_standard_m,
            open3d_dir,
            basename="point_cloud_world_standard_m",
            voxel_cm=0.0,
            voxel_size=0.0,
            max_points=0,
            estimate_normals=False,
            coordinate_units="m",
        )
    except Exception as exc:
        error_payload = {
            "backend": "open3d",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        camera_open3d_outputs = dict(error_payload)
        world_open3d_outputs = dict(error_payload)
        camera_standard_open3d_outputs = dict(error_payload)
        world_standard_open3d_outputs = dict(error_payload)

    diagnostics_payload = {
        "projection": standard_summary["depth_projection_selected"],
        "depth_projection_requested": standard_summary["depth_projection_requested"],
        "depth_projection_selected": standard_summary["depth_projection_selected"],
        "projection_corrected": bool(standard_summary["projection_corrected"]),
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
        "intrinsics": standard_summary["intrinsics"],
        "fov_source": standard_summary["fov_source"],
        "image_width": int(standard_summary["image_width"]),
        "image_height": int(standard_summary["image_height"]),
        "min_depth_cm": float(standard_summary["min_depth_cm"]),
        "max_depth_cm": float(standard_summary["max_depth_cm"]),
        "invalid_depth_count": int(standard_summary["invalid_depth_count"]),
        "legacy_unreal_cm_stats": point_cloud_xyz_stats(world_cloud),
        "standard_camera_m_stats": point_cloud_xyz_stats(camera_standard_m),
        "standard_world_m_stats": point_cloud_xyz_stats(world_standard_m),
        "projection_comparison": standard_summary.get("projection_comparison", {}),
        "coordinate_conventions": standard_summary.get("coordinate_conventions", {}),
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    camera_info_payload = dict(camera_info)
    camera_info_payload.update(
        {
            "image_width": int(standard_summary["image_width"]),
            "image_height": int(standard_summary["image_height"]),
            "intrinsics": standard_summary["intrinsics"],
            "fov_source": standard_summary["fov_source"],
            "rgb_resized_for_depth": bool(standard_summary["rgb_resized_for_depth"]),
            "min_depth_cm": float(standard_summary["min_depth_cm"]),
            "max_depth_cm": float(standard_summary["max_depth_cm"]),
            "depth_input_mode": standard_summary["depth_input_mode"],
            "depth_projection": standard_summary["depth_projection_selected"],
            "depth_projection_requested": standard_summary["depth_projection_requested"],
            "depth_projection_selected": standard_summary["depth_projection_selected"],
            "projection_corrected": True,
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "coordinate_conventions": {
                "camera_legacy": "CV: x right, y down, z forward; units cm",
                "world_legacy": "Unreal world coordinates; units cm",
                "camera_standard_m": "right-handed camera: x right, y up, z forward; units m",
                "world_standard_m": "right-handed world: x Unreal X, y -Unreal Y, z Unreal Z; units m",
            },
        }
    )
    camera_info_path.write_text(json.dumps(camera_info_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "point_cloud_camera_npy_path": str(camera_npy_path),
        "point_cloud_world_npy_path": str(world_npy_path),
        "point_cloud_camera_ply_path": str(camera_ply_path),
        "point_cloud_world_ply_path": str(world_ply_path),
        "point_cloud_camera_standard_m_npy_path": str(camera_standard_m_npy_path),
        "point_cloud_world_standard_m_npy_path": str(world_standard_m_npy_path),
        "point_cloud_camera_standard_m_ply_path": str(camera_standard_m_ply_path),
        "point_cloud_world_standard_m_ply_path": str(world_standard_m_ply_path),
        "point_cloud_preview_path": str(preview_path),
        "camera_info_path": str(camera_info_path),
        "projection_diagnostics_path": str(diagnostics_path),
        "open3d_camera": camera_open3d_outputs,
        "open3d_world": world_open3d_outputs,
        "open3d_camera_standard_m": camera_standard_open3d_outputs,
        "open3d_world_standard_m": world_standard_open3d_outputs,
        "point_count": int(standard_summary["point_count"]),
        "legacy_point_count": int(summary["point_count"]),
        "invalid_depth_count": int(standard_summary["invalid_depth_count"]),
        "source_mode": "depth_backprojected_lidar",
        "point_cloud_shape": list(camera_standard_m.shape),
        "legacy_point_cloud_shape": list(camera_cloud.shape),
        "min_depth_cm": float(standard_summary["min_depth_cm"]),
        "max_depth_cm": float(standard_summary["max_depth_cm"]),
        "depth_input_mode": standard_summary["depth_input_mode"],
        "depth_projection": standard_summary["depth_projection_selected"],
        "depth_projection_requested": standard_summary["depth_projection_requested"],
        "depth_projection_selected": standard_summary["depth_projection_selected"],
        "projection_corrected": True,
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
    }


def read_json_object(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def resolve_capture_output_path(capture_dir: Path, payload: Dict[str, Any], key: str, default_name: str) -> Path:
    raw = str(payload.get(key, "") or "").strip()
    path = Path(raw) if raw else capture_dir / default_name
    if not path.is_absolute():
        path = (capture_dir / path).resolve()
    return path


def write_legacy_converted_standard_lidar_outputs(
    legacy_world_path: Path,
    output_dir: Path,
    *,
    capture_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_cloud = np.load(legacy_world_path).astype(np.float32, copy=False)
    standard_cloud = unreal_world_cloud_to_standard_zup_m(legacy_cloud)
    standard_npy_path = output_dir / "point_cloud_world_standard_m.npy"
    standard_ply_path = output_dir / "point_cloud_world_standard_m.ply"
    diagnostics_path = output_dir / "projection_diagnostics.json"
    np.save(standard_npy_path, standard_cloud)
    write_colored_point_cloud_ply(standard_ply_path, standard_cloud)
    diagnostics_payload = {
        "projection": "legacy_converted",
        "depth_projection_selected": "legacy_converted",
        "depth_projection_requested": "legacy_converted",
        "projection_corrected": False,
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
        "legacy_source_path": str(legacy_world_path),
        "legacy_unreal_cm_stats": point_cloud_xyz_stats(legacy_cloud),
        "standard_world_m_stats": point_cloud_xyz_stats(standard_cloud),
        "note": "Legacy point_cloud_world.npy did not include raw depth; converted Unreal [x,y,z] cm to standard [x,-y,z] m only.",
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "point_cloud_world_standard_m_npy_path": str(standard_npy_path),
        "point_cloud_world_standard_m_ply_path": str(standard_ply_path),
        "projection_diagnostics_path": str(diagnostics_path),
        "point_count": int(standard_cloud.shape[0]),
        "depth_projection": "legacy_converted",
        "depth_projection_selected": "legacy_converted",
        "projection_corrected": False,
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
        "legacy_source_path": str(legacy_world_path),
    }
    capture_json_path = output_dir / "capture.json"
    payload = dict(capture_payload or read_json_object(capture_json_path))
    if payload:
        payload.update(result)
        capture_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def ensure_standard_world_cloud_for_capture(
    capture_dir: Path,
    *,
    capture_payload: Optional[Dict[str, Any]] = None,
    lidar_depth_projection: str = DEFAULT_LIDAR_DEPTH_PROJECTION,
    min_depth_cm: Optional[float] = None,
    max_depth_cm: Optional[float] = None,
) -> Dict[str, Any]:
    capture_dir = Path(capture_dir)
    payload = dict(capture_payload or read_json_object(capture_dir / "capture.json"))
    diagnostics_path = capture_dir / "projection_diagnostics.json"
    standard_path = resolve_capture_output_path(
        capture_dir,
        payload,
        "point_cloud_world_standard_m_npy_path",
        "point_cloud_world_standard_m.npy",
    )
    if standard_path.exists():
        diagnostics = read_json_object(diagnostics_path)
        return {
            "point_cloud_world_standard_m_npy_path": str(standard_path),
            "point_cloud_world_standard_m_ply_path": str(
                resolve_capture_output_path(
                    capture_dir,
                    payload,
                    "point_cloud_world_standard_m_ply_path",
                    "point_cloud_world_standard_m.ply",
                )
            ),
            "projection_diagnostics_path": str(diagnostics_path),
            "point_count": int(payload.get("point_count", 0) or diagnostics.get("standard_world_m_stats", {}).get("point_count", 0) or 0),
            "depth_projection": diagnostics.get("depth_projection_selected", payload.get("depth_projection_selected", "plane_depth")),
            "depth_projection_selected": diagnostics.get("depth_projection_selected", payload.get("depth_projection_selected", "plane_depth")),
            "projection_corrected": bool(diagnostics.get("projection_corrected", payload.get("projection_corrected", True))),
            "coordinate_frame": "standard_zup",
            "coordinate_units": "m",
            "legacy_source_path": diagnostics.get("legacy_source_path", payload.get("legacy_source_path", "")),
        }

    depth_path = resolve_capture_output_path(capture_dir, payload, "depth_npy_path", "depth.npy")
    rgb_path = resolve_capture_output_path(capture_dir, payload, "rgb_path", "rgb.png")
    camera_info_path = resolve_capture_output_path(capture_dir, payload, "camera_info_path", "camera_info.json")
    if depth_path.exists() and rgb_path.exists() and camera_info_path.exists():
        rgb_image = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth_image = np.load(depth_path).astype(np.float32, copy=False)
        camera_info = read_json_object(camera_info_path)
        min_depth = float(min_depth_cm if min_depth_cm is not None else payload.get("lidar_depth_min_cm", payload.get("min_depth_cm", DEFAULT_LIDAR_DEPTH_MIN_CM)))
        max_depth = float(max_depth_cm if max_depth_cm is not None else payload.get("lidar_depth_max_cm", payload.get("max_depth_cm", DEFAULT_LIDAR_DEPTH_MAX_CM)))
        result = save_lidar_capture_outputs(
            rgb_image=rgb_image,
            depth_image=depth_image,
            camera_info=camera_info,
            output_dir=capture_dir,
            min_depth_cm=min_depth,
            max_depth_cm=max_depth,
            lidar_depth_projection=lidar_depth_projection,
        )
        capture_json_path = capture_dir / "capture.json"
        merged_payload = dict(payload or read_json_object(capture_json_path))
        if merged_payload:
            merged_payload.update(result)
            capture_json_path.write_text(json.dumps(merged_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    legacy_world_path = resolve_capture_output_path(capture_dir, payload, "point_cloud_world_npy_path", "point_cloud_world.npy")
    if legacy_world_path.exists():
        return write_legacy_converted_standard_lidar_outputs(
            legacy_world_path,
            capture_dir,
            capture_payload=payload,
        )
    return {}


def update_capture_json(capture_dir: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    capture_path = Path(capture_dir) / "capture.json"
    payload = read_json_object(capture_path)
    payload.update(updates)
    capture_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def postprocess_lidar_stream_capture(
    stream_dir: Path,
    *,
    lidar_depth_projection: str = DEFAULT_LIDAR_DEPTH_PROJECTION,
    min_depth_cm: float = DEFAULT_LIDAR_DEPTH_MIN_CM,
    max_depth_cm: float = DEFAULT_LIDAR_DEPTH_MAX_CM,
    voxel_cm: float = DEFAULT_LIDAR_RECON_VOXEL_CM,
    max_points: int = DEFAULT_LIDAR_RECON_MAX_POINTS,
) -> Dict[str, Any]:
    stream_path = Path(stream_dir).resolve()
    frames_root = stream_path / "frames"
    started_at = datetime.now().isoformat(timespec="milliseconds")
    frame_dirs = sorted(path for path in frames_root.glob("frame_*") if path.is_dir()) if frames_root.exists() else []
    frame_results: List[Dict[str, Any]] = []
    cloud_paths: List[Path] = []
    source_point_count = 0
    failed_count = 0

    summary_path = stream_path / "stream_capture_lidar.json"
    trajectory_path = stream_path / "trajectory.json"
    summary = read_json_object(summary_path)
    trajectory_payload = read_json_object(trajectory_path)
    trajectory = trajectory_payload.get("trajectory", []) if isinstance(trajectory_payload.get("trajectory"), list) else []

    summary.update(
        {
            "postprocess_status": "running",
            "postprocess_started_at": started_at,
            "postprocess_finished_at": "",
            "postprocess_error": "",
            "updated_at": started_at,
        }
    )
    if summary:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for capture_dir in frame_dirs:
        capture_payload = read_json_object(capture_dir / "capture.json")
        frame_index = int(capture_payload.get("frame_index", 0) or 0)
        frame_started_at = datetime.now().isoformat(timespec="milliseconds")
        try:
            capture_payload = update_capture_json(
                capture_dir,
                {
                    "postprocess_status": "running",
                    "postprocess_started_at": frame_started_at,
                    "postprocess_finished_at": "",
                    "postprocess_error": "",
                },
            )
            ensured = ensure_standard_world_cloud_for_capture(
                capture_dir,
                capture_payload=capture_payload,
                lidar_depth_projection=lidar_depth_projection,
                min_depth_cm=float(min_depth_cm),
                max_depth_cm=float(max_depth_cm),
            )
            raw_path = str(ensured.get("point_cloud_world_standard_m_npy_path", "") or "")
            point_count = int(ensured.get("point_count", 0) or 0)
            if raw_path:
                cloud_path = Path(raw_path)
                if cloud_path.exists():
                    cloud_paths.append(cloud_path)
                    if point_count <= 0:
                        try:
                            point_count = int(np.load(cloud_path, mmap_mode="r").shape[0])
                        except Exception:
                            point_count = 0
                    source_point_count += max(0, point_count)
            frame_finished_at = datetime.now().isoformat(timespec="milliseconds")
            frame_update = {
                **ensured,
                "raw_capture_only": False,
                "postprocess_status": "done",
                "postprocess_finished_at": frame_finished_at,
                "postprocess_error": "",
            }
            update_capture_json(capture_dir, frame_update)
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "capture_dir": str(capture_dir),
                    "point_count": point_count,
                    "postprocess_status": "done",
                    **ensured,
                }
            )
        except Exception as exc:
            failed_count += 1
            error_text = f"{type(exc).__name__}: {exc}"
            update_capture_json(
                capture_dir,
                {
                    "postprocess_status": "failed",
                    "postprocess_finished_at": datetime.now().isoformat(timespec="milliseconds"),
                    "postprocess_error": error_text,
                },
            )
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "capture_dir": str(capture_dir),
                    "postprocess_status": "failed",
                    "postprocess_error": error_text,
                }
            )

    merged = np.zeros((0, 6), dtype=np.float32)
    for cloud_path in cloud_paths:
        cloud = np.load(cloud_path).astype(np.float32, copy=False)
        if cloud.ndim != 2 or cloud.shape[1] != 6 or cloud.shape[0] == 0:
            continue
        merged = cloud if merged.shape[0] == 0 else np.vstack((merged, cloud))
        merged = downsample_colored_point_cloud_voxel(
            merged,
            voxel_cm=standard_voxel_size_m(voxel_cm),
            max_points=max_points,
        )

    reconstruction = save_lidar_reconstruction_outputs(
        merged,
        stream_path / "reconstruction",
        source_frame_count=len(cloud_paths),
        source_point_count=source_point_count,
        voxel_cm=voxel_cm,
        max_points=max_points,
        coordinate_frame="standard_zup",
        coordinate_units="m",
    )
    finished_at = datetime.now().isoformat(timespec="milliseconds")
    status = "failed" if failed_count else "done"

    frame_result_by_dir = {str(Path(item.get("capture_dir", "")).resolve()): item for item in frame_results}
    if isinstance(trajectory, list):
        for entry in trajectory:
            if not isinstance(entry, dict):
                continue
            raw_capture_dir = str(entry.get("capture_dir", "") or "")
            if not raw_capture_dir:
                continue
            capture_dir = Path(raw_capture_dir)
            if not capture_dir.is_absolute():
                capture_dir = (stream_path / capture_dir).resolve()
            frame_result = frame_result_by_dir.get(str(capture_dir.resolve()))
            if frame_result:
                entry.update(frame_result)
                entry["raw_capture_only"] = False if frame_result.get("postprocess_status") == "done" else bool(entry.get("raw_capture_only", True))

    final_summary = {
        **summary,
        "capture_kind": "stream_capture_lidar",
        "stream_dir": str(stream_path),
        "frames_dir": str(frames_root),
        "reconstruction_dir": str(stream_path / "reconstruction"),
        "frame_count": len(frame_dirs),
        "source_point_count": int(source_point_count),
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
        "depth_projection_selected": select_lidar_depth_projection(lidar_depth_projection),
        "postprocess_status": status,
        "postprocess_started_at": started_at,
        "postprocess_finished_at": finished_at,
        "postprocess_error": "" if status != "failed" else f"{failed_count} frame(s) failed during postprocess",
        "failed_frame_count": int(failed_count),
        "reconstruction": reconstruction,
        "updated_at": finished_at,
    }
    summary_path.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    trajectory_payload.update(final_summary)
    trajectory_payload["trajectory"] = trajectory
    trajectory_path.write_text(json.dumps(trajectory_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        **final_summary,
        "frame_results": frame_results,
        "source_frame_count": len(cloud_paths),
        "input_cloud_file_count": len(cloud_paths),
        "merged_point_count": int(reconstruction.get("merged_point_count", 0) or 0),
    }


def make_env(args: argparse.Namespace):
    patch_unrealcv_client_thread_lifecycle()
    env = gym.make(args.env_id)
    if args.time_dilation > 0:
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)

    # Match batch_run_act_all.py: reset removes non-drone roles, then RandomPopulationWrapper
    # adds the requested drone population back into the scene.
    env.agents_category = ["drone"]
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(args.width, args.height), sleep_time=args.launch_sleep)
    env = augmentation.RandomPopulationWrapper(env, 1, 1, random_target=False)
    env.seed(int(args.seed))
    return env


def mark_episode_started(env: Any) -> None:
    current = env
    while current is not None:
        if hasattr(current, "_episode_started_at") and hasattr(current, "_elapsed_steps"):
            current._episode_started_at = time.time()
            current._elapsed_steps = 0
            return
        current = getattr(current, "env", None)


def step_drone_flight_env(env: Any, actions: List[Any]) -> Tuple[Any, Any, bool, Dict[str, Any]]:
    base_env = env.unwrapped
    if len(base_env.player_list) == 1:
        from gym_unrealcv.envs.base_env import UnrealCv_base

        return UnrealCv_base.step(base_env, actions)
    return env.step(actions)


def set_drone_camera(env: Any, drone_name: str) -> None:
    agent_cfg = env.unwrapped.agents.get(drone_name, {})
    rel_loc = agent_cfg.get("relative_location", [0, 0, 0])
    rel_rot = agent_cfg.get("relative_rotation", [0, 0, 0])
    env.unwrapped.unrealcv.set_cam(drone_name, rel_loc, rel_rot)


def bind_drone_view(env: Any, drone_name: str) -> None:
    set_drone_camera(env, drone_name)
    env.unwrapped.unrealcv.set_viewport(drone_name)


def set_drone_world_pose(env: Any, drone_name: str, pose: List[float], *,
                         update_camera: bool = False, update_viewport: bool = False,
                         disable_physics: bool = True, stop_motion: bool = True) -> None:
    env.unwrapped.unrealcv.set_obj_location(drone_name, pose[:3])
    env.unwrapped.unrealcv.set_rotation(drone_name, pose[4] - 180)
    if disable_physics:
        env.unwrapped.unrealcv.set_phy(drone_name, 0)
    if stop_motion:
        env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
    if update_camera:
        set_drone_camera(env, drone_name)
    if update_viewport:
        env.unwrapped.unrealcv.set_viewport(drone_name)


def reset_drone_pose(env: Any, drone_name: str, pose: List[float]) -> None:
    set_drone_world_pose(env, drone_name, pose, update_camera=True, update_viewport=True)
    time.sleep(1.0)


def normalize_initial_pose(pose: List[float]) -> List[float]:
    values = list(pose)
    if len(values) == 4:
        x, y, z, yaw = values
        return [x, y, z, 0.0, yaw, 0.0]
    if len(values) == 6:
        return values
    raise ValueError("--initial_pos must have 4 values (X Y Z YAW) or 6 values (X Y Z ROLL YAW PITCH)")


def pose_from_info(info: Dict[str, Any]) -> List[float]:
    poses = info.get("Pose", [])
    if len(poses) == 0:
        return []
    pose = poses[0]
    if hasattr(pose, "tolist"):
        return pose.tolist()
    return list(pose)


def read_drone_pose(env: Any, drone_name: str) -> List[float]:
    loc = env.unwrapped.unrealcv.get_obj_location(drone_name)
    rot = env.unwrapped.unrealcv.get_obj_rotation(drone_name)
    return list(loc) + list(rot)


def pose_values_to_dict(pose_values: List[float]) -> Dict[str, Any]:
    if len(pose_values) < 6:
        return {"raw": list(pose_values)}
    return {
        "x": float(pose_values[0]),
        "y": float(pose_values[1]),
        "z": float(pose_values[2]),
        "roll": float(pose_values[3]),
        "yaw": float(pose_values[4]),
        "task_yaw": float(pose_values[4]),
        "pitch": float(pose_values[5]),
        "raw": [float(value) for value in pose_values[:6]],
    }


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return str(value)


def read_drone_observation(env: Any, drone_name: str) -> Optional[np.ndarray]:
    cam_id = env.unwrapped.agents.get(drone_name, {}).get("cam_id")
    if cam_id is None or cam_id < 0:
        return None
    return env.unwrapped.unrealcv.read_image(cam_id, "lit", "direct")


def read_drone_depth(env: Any, drone_name: str) -> Optional[np.ndarray]:
    cam_id = env.unwrapped.agents.get(drone_name, {}).get("cam_id")
    if cam_id is None or cam_id < 0:
        return None
    return env.unwrapped.unrealcv.get_depth(cam_id, show=False)


def read_drone_camera_info(env: Any, drone_name: str, *, rgb: Any = None, depth: Any = None) -> Dict[str, Any]:
    cam_id = env.unwrapped.agents.get(drone_name, {}).get("cam_id")
    if cam_id is None or cam_id < 0:
        raise RuntimeError("Camera id is unavailable for current drone")
    unrealcv = env.unwrapped.unrealcv
    location_values = [float(value) for value in unrealcv.get_cam_location(cam_id)]
    rotation_values = [float(value) for value in unrealcv.get_cam_rotation(cam_id)]
    fov, fov_source = parse_camera_fov_degrees(unrealcv.get_cam_fov(cam_id))
    depth_shape = list(np.asarray(depth).shape) if depth is not None else []
    rgb_shape = list(np.asarray(rgb).shape) if rgb is not None else []
    if depth is not None:
        image_height, image_width = coerce_depth_planar_image(depth).shape[:2]
    elif rgb is not None and np.asarray(rgb).ndim >= 2:
        image_height, image_width = np.asarray(rgb).shape[:2]
    else:
        image_width = int(getattr(env.unwrapped.unrealcv, "resolution", [0, 0])[0] or 0)
        image_height = int(getattr(env.unwrapped.unrealcv, "resolution", [0, 0])[1] or 0)
    return {
        "camera_id": int(cam_id),
        "location": {
            "x": location_values[0],
            "y": location_values[1],
            "z": location_values[2],
            "raw": location_values[:3],
        },
        "rotation": {
            "pitch": rotation_values[0],
            "yaw": rotation_values[1],
            "roll": rotation_values[2],
            "raw": rotation_values[:3],
        },
        "horizontal_fov_deg": float(fov),
        "fov_source": fov_source,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "rgb_shape": rgb_shape,
        "depth_shape": depth_shape,
        "coordinate_frame": "standard_zup",
        "coordinate_units": "m",
        "coordinate_conventions": {
            "world_standard_m": "right-handed world: x Unreal X, y -Unreal Y, z Unreal Z; units m",
            "legacy_unreal_cm": "Unreal world coordinates; units cm",
        },
    }


def stabilize_drone_at_current_pose(env: Any, drone_name: str) -> None:
    env.unwrapped.unrealcv.set_phy(drone_name, 0)
    env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
    bind_drone_view(env, drone_name)
    time.sleep(0.5)


def log_drone_inventory(env: Any, label: str) -> None:
    LOGGER.info("%s player_list: %s", label, env.unwrapped.player_list)
    for index, name in enumerate(env.unwrapped.player_list):
        cam_id = env.unwrapped.agents.get(name, {}).get("cam_id")
        LOGGER.info(
            "%s drone[%s] name=%s cam_id=%s pose=%s",
            label,
            index,
            name,
            cam_id,
            read_drone_pose(env, name),
        )


def select_drone_name(env: Any, args: argparse.Namespace) -> str:
    player_count = len(env.unwrapped.player_list)
    if player_count == 0:
        raise RuntimeError("No drone player found after reset")
    requested_index = 0 if args.drone_index is None else args.drone_index
    if requested_index < 0 or requested_index >= player_count:
        LOGGER.warning("drone_index=%s is out of range; using 0", requested_index)
        requested_index = 0
    return env.unwrapped.player_list[requested_index]


class GlobalKeyboardReader:
    def __init__(self) -> None:
        self._pressed: Set[str] = set()
        self._exit_requested = False
        self._listener = None
        self._keyboard = None

    def __enter__(self) -> "GlobalKeyboardReader":
        from pynput import keyboard

        self._keyboard = keyboard

        def normalize_key(key: Any) -> Optional[str]:
            try:
                return key.char.lower()
            except AttributeError:
                if key == keyboard.Key.space:
                    return "space"
                if key == keyboard.Key.esc:
                    return "esc"
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    return "ctrl"
                return None

        def on_press(key: Any) -> None:
            key_name = normalize_key(key)
            if key_name is None:
                return
            if key_name == "esc":
                self._exit_requested = True
            self._pressed.add(key_name)

        def on_release(key: Any) -> None:
            key_name = normalize_key(key)
            if key_name is not None:
                self._pressed.discard(key_name)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._listener is not None:
            self._listener.stop()

    def snapshot(self) -> Tuple[Set[str], bool]:
        return set(self._pressed), self._exit_requested


class TerminalKeyboardReader:
    def __init__(self) -> None:
        self._old_settings = None
        self._exit_requested = False

    def __enter__(self) -> "TerminalKeyboardReader":
        if termios is None or tty is None:
            raise RuntimeError("terminal keyboard backend is only available on Unix-like terminals")
        if not sys.stdin.isatty():
            raise RuntimeError("terminal keyboard backend needs a TTY")
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def snapshot(self) -> Tuple[Set[str], bool]:
        pressed: Set[str] = set()
        while select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if char == "\x1b":
                self._exit_requested = True
            elif char == "\x03":
                raise KeyboardInterrupt
            elif char == " ":
                pressed.add("space")
            elif char:
                pressed.add(char.lower())
        return pressed, self._exit_requested


def keyboard_reader(args: argparse.Namespace):
    if args.keyboard_backend == "terminal":
        return TerminalKeyboardReader()
    return GlobalKeyboardReader()


def action_from_keys(keys: Set[str], args: argparse.Namespace) -> Tuple[str, List[float]]:
    if "x" in keys or "h" in keys:
        return "hover", [0.0, 0.0, 0.0, 0.0]

    x_axis = float("w" in keys) - float("s" in keys)
    y_axis = float("d" in keys) - float("a" in keys)
    z_axis = float("space" in keys or "r" in keys) - float("f" in keys or "ctrl" in keys)
    yaw_axis = float("e" in keys) - float("q" in keys)

    action = [
        args.linear_speed * x_axis,
        args.linear_speed * y_axis,
        args.vertical_speed * z_axis if z_axis > 0 else args.down_speed * z_axis,
        args.yaw_speed * yaw_axis,
    ]

    labels = []
    if x_axis > 0:
        labels.append("forward")
    elif x_axis < 0:
        labels.append("backward")
    if y_axis > 0:
        labels.append("right")
    elif y_axis < 0:
        labels.append("left")
    if z_axis > 0:
        labels.append("up")
    elif z_axis < 0:
        labels.append("down")
    if yaw_axis > 0:
        labels.append("yaw_left")
    elif yaw_axis < 0:
        labels.append("yaw_right")

    return "+".join(labels) if labels else "hover", action


def action_from_name(name: str, args: argparse.Namespace) -> Tuple[str, List[float]]:
    actions = {
        "hover": [0.0, 0.0, 0.0, 0.0],
        "up": [0.0, 0.0, args.vertical_speed, 0.0],
        "down": [0.0, 0.0, -args.down_speed, 0.0],
        "forward": [args.linear_speed, 0.0, 0.0, 0.0],
        "backward": [-args.linear_speed, 0.0, 0.0, 0.0],
        "left": [0.0, -args.linear_speed, 0.0, 0.0],
        "right": [0.0, args.linear_speed, 0.0, 0.0],
        "yaw_left": [0.0, 0.0, 0.0, -args.yaw_speed],
        "yaw_right": [0.0, 0.0, 0.0, args.yaw_speed],
    }
    return name, actions[name]


def prepare_drone_env(args: argparse.Namespace) -> Tuple[Any, str, Any]:
    env = make_env(args)
    LOGGER.info("Starting one-drone UnrealCV environment %s", args.env_id)
    base_env = env.unwrapped
    if not base_env.launched:
        base_env.launched = base_env.launch_ue_env()
        base_env.init_agents()
        base_env.init_objects()
    base_env.num_agents = 1
    base_env.set_population(1)
    base_env.tracker_id = 0
    base_env.target_id = 0
    log_drone_inventory(env, "After reset")
    drone_name = select_drone_name(env, args)
    if args.keep_reset_pose:
        stabilize_drone_at_current_pose(env, drone_name)
    else:
        reset_drone_pose(env, drone_name, normalize_initial_pose(args.initial_pos))
    LOGGER.info("Using drone: %s", drone_name)
    LOGGER.info("Selected drone pose: %s", read_drone_pose(env, drone_name))
    LOGGER.info("Initial camera config: %s", env.unwrapped.unrealcv.get_camera_config())
    observation = read_drone_observation(env, drone_name)
    mark_episode_started(env)
    return env, drone_name, observation


def append_step_log(log: List[Dict[str, Any]], step: int, phase: str, action: List[float],
                    reward: Any, done: bool, info: Dict[str, Any]) -> None:
    log.append({
        "step": step,
        "phase": phase,
        "action": action,
        "pose": pose_from_info(info),
        "reward": np.asarray(reward).tolist(),
        "done": bool(done),
    })


def yaw_toward_point(x: float, y: float, target_x: float, target_y: float) -> float:
    return math.degrees(math.atan2(target_y - y, target_x - x))


def run_orbit_house_plan(args: argparse.Namespace, env: Any, drone_name: str, observation: Any,
                         run_dir: Path, log: List[Dict[str, Any]]) -> int:
    if args.orbit_radius <= 0:
        raise ValueError("--orbit_radius must be greater than 0")
    if args.orbit_steps <= 0:
        raise ValueError("--orbit_steps must be greater than 0")

    total_step = 0
    center_x, center_y = args.orbit_center
    direction = -1.0 if args.orbit_clockwise else 1.0
    phase = "orbit_house"
    save_color_observation_for_args(args, observation, run_dir / "step_0000_reset.png")
    LOGGER.info(
        "Orbit center=(%.3f, %.3f) radius=%.3f altitude=%.3f steps=%s clockwise=%s",
        center_x,
        center_y,
        args.orbit_radius,
        args.orbit_altitude,
        args.orbit_steps,
        args.orbit_clockwise,
    )

    for index in range(args.orbit_steps + 1):
        angle_deg = args.orbit_start_angle + direction * 360.0 * index / args.orbit_steps
        angle_rad = math.radians(angle_deg)
        x = center_x + args.orbit_radius * math.cos(angle_rad)
        y = center_y + args.orbit_radius * math.sin(angle_rad)
        yaw = yaw_toward_point(x, y, center_x, center_y)
        pose = [x, y, args.orbit_altitude, 0.0, yaw, 0.0]

        set_drone_world_pose(env, drone_name, pose)
        actual_pose = wait_for_drone_pose(
            env,
            drone_name,
            pose,
            timeout_s=float(getattr(args, "pose_settle_timeout", DEFAULT_POSE_SETTLE_TIMEOUT)),
            position_tolerance_cm=float(getattr(args, "pose_settle_pos_tolerance", DEFAULT_POSE_SETTLE_POS_TOLERANCE)),
            yaw_tolerance_deg=float(getattr(args, "pose_settle_yaw_tolerance", DEFAULT_POSE_SETTLE_YAW_TOLERANCE)),
        )
        time.sleep(args.step_delay)
        total_step += 1
        log.append({
            "step": total_step,
            "phase": phase,
            "action": [x, y, args.orbit_altitude, yaw],
            "commanded_pose": list(pose),
            "actual_pose": actual_pose,
            "pose": actual_pose,
            "pose_error": pose_error_summary(actual_pose, pose),
            "reward": None,
            "done": False,
        })

        should_save = (
            index == 0
            or index == args.orbit_steps
            or (args.save_every > 0 and total_step % args.save_every == 0)
        )
        if should_save:
            image = read_drone_observation(env, drone_name)
            if image is not None:
                save_color_observation_for_args(args, image, run_dir / f"step_{total_step:04d}_{phase}.png")

    return total_step


def run_scripted_plan(args: argparse.Namespace, env: Any, drone_name: str, observation: Any,
                      run_dir: Path, log: List[Dict[str, Any]]) -> int:
    total_step = 0
    save_color_observation_for_args(args, observation, run_dir / "step_0000_reset.png")

    for phase, action in DEFAULT_ACTION_PLAN:
        LOGGER.info("Phase %-8s action=%s", phase, action)
        for _ in range(args.steps_per_action):
            env_actions = [None] * len(env.unwrapped.player_list)
            env_actions[0] = action
            observation, reward, done, info = step_drone_flight_env(env, env_actions)
            total_step += 1
            append_step_log(log, total_step, phase, action, reward, done, info)
            if args.save_every > 0 and total_step % args.save_every == 0:
                save_color_observation_for_args(args, observation, run_dir / f"step_{total_step:04d}_{phase}.png")
            if done:
                LOGGER.warning("Environment returned done=True at step %s", total_step)
                break
            time.sleep(args.step_delay)

    return total_step


def run_keyboard_control(args: argparse.Namespace, env: Any, drone_name: str, observation: Any,
                         run_dir: Path, log: List[Dict[str, Any]]) -> int:
    total_step = 0
    last_phase = None
    physics_enabled = False
    save_color_observation_for_args(args, observation, run_dir / "step_0000_reset.png")
    print(KEYBOARD_HELP)
    LOGGER.info("Keyboard control started. Output directory: %s", run_dir)

    def apply_action(phase: str, action: List[float]) -> bool:
        nonlocal total_step, last_phase, physics_enabled
        if phase != last_phase:
            LOGGER.info("Keyboard action %-20s action=%s", phase, action)
            last_phase = phase
        if not physics_enabled and any(abs(value) > 1e-9 for value in action):
            env.unwrapped.unrealcv.set_phy(drone_name, 1)
            physics_enabled = True
            LOGGER.info("Enabled drone physics for flight")
        env.unwrapped.unrealcv.set_move_bp(drone_name, action)
        total_step += 1
        log.append({
            "step": total_step,
            "phase": phase,
            "action": action,
            "pose": read_drone_pose(env, drone_name),
            "reward": None,
            "done": False,
        })
        if args.save_every > 0 and total_step % args.save_every == 0:
            image = env.unwrapped.unrealcv.read_image(env.unwrapped.agents[drone_name]["cam_id"], "lit", "direct")
            save_color_observation_for_args(args, image, run_dir / f"step_{total_step:04d}_{phase}.png")
        if args.max_steps > 0 and total_step >= args.max_steps:
            LOGGER.info("Reached max keyboard steps: %s", args.max_steps)
            return True
        time.sleep(args.control_dt)
        return False

    if args.auto_action != "none":
        LOGGER.info("Auto action test mode: %s", args.auto_action)
        while True:
            phase, action = action_from_name(args.auto_action, args)
            if apply_action(phase, action):
                break
        return total_step

    with keyboard_reader(args) as reader:
        while True:
            keys, should_exit = reader.snapshot()
            if should_exit:
                LOGGER.info("Exit requested from keyboard")
                break

            phase, action = action_from_keys(keys, args)
            if apply_action(phase, action):
                break

    return total_step


def normalize_angle_deg(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized


def angle_error_deg(actual: float, target: float) -> float:
    return normalize_angle_deg(float(actual) - float(target))


def pose_error_summary(actual_pose: Optional[List[float]], commanded_pose: Optional[List[float]]) -> Dict[str, float]:
    if not actual_pose or not commanded_pose or len(actual_pose) < 6 or len(commanded_pose) < 6:
        return {"position_cm": 0.0, "yaw_deg": 0.0}
    dx = float(actual_pose[0]) - float(commanded_pose[0])
    dy = float(actual_pose[1]) - float(commanded_pose[1])
    dz = float(actual_pose[2]) - float(commanded_pose[2])
    return {
        "position_cm": float(math.sqrt(dx * dx + dy * dy + dz * dz)),
        "yaw_deg": float(angle_error_deg(float(actual_pose[4]), float(commanded_pose[4]))),
    }


def wait_for_drone_pose(env: Any, drone_name: str, commanded_pose: List[float], *,
                        timeout_s: float = DEFAULT_POSE_SETTLE_TIMEOUT,
                        position_tolerance_cm: float = DEFAULT_POSE_SETTLE_POS_TOLERANCE,
                        yaw_tolerance_deg: float = DEFAULT_POSE_SETTLE_YAW_TOLERANCE,
                        poll_s: float = 0.05) -> List[float]:
    deadline = time.time() + max(0.0, float(timeout_s))
    actual_pose = read_drone_pose(env, drone_name)
    while time.time() < deadline:
        error = pose_error_summary(actual_pose, commanded_pose)
        if (
            abs(error["yaw_deg"]) <= max(0.0, float(yaw_tolerance_deg))
            and error["position_cm"] <= max(0.0, float(position_tolerance_cm))
        ):
            return actual_pose
        time.sleep(max(0.01, float(poll_s)))
        actual_pose = read_drone_pose(env, drone_name)
    return actual_pose


def serialized_unrealcv_method(method: Any) -> Any:
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock = getattr(self, "api_lock", None)
        if lock is None:
            return method(self, *args, **kwargs)
        with lock:
            return method(self, *args, **kwargs)
    return wrapped


class DroneFlightSession:
    """Reusable in-process controller for GUI or service wrappers."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.env = None
        self.drone_name: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.log: List[Dict[str, Any]] = []
        self.total_step = 0
        self.movement_enabled = False
        self.movement_mode = str(getattr(args, "movement_mode", DEFAULT_MOVEMENT_MODE) or DEFAULT_MOVEMENT_MODE)
        if self.movement_mode not in {"pose_lock", "physics"}:
            self.movement_mode = DEFAULT_MOVEMENT_MODE
        self.commanded_pose: Optional[List[float]] = None
        self.last_actual_pose: Optional[List[float]] = None
        self.map_touch_calibration: Dict[str, Any] = {}
        self.api_lock = threading.RLock()
        self.last_action = "idle"
        self.last_move_command: Dict[str, Any] = {}
        self.last_command_detail: Dict[str, Any] = {}
        self.last_observation: Any = None

    @property
    def started(self) -> bool:
        return self.env is not None and self.drone_name is not None

    def start(self) -> Dict[str, Any]:
        if self.started:
            return self.get_state(message="Session already started")

        configure_local_unreal_env(self.args)
        validate_unreal_env_config(self.args)
        patch_env_setting_loader(self.args)
        ensure_legacy_gym_entrypoint_loader()
        patch_macos_launchservices_launcher()
        np.random.seed(self.args.seed)
        self.run_dir = make_output_dir(self.args.output_dir)
        self.env, self.drone_name, self.last_observation = prepare_drone_env(self.args)
        self.last_actual_pose = self.get_pose_list()
        if getattr(self.args, "keep_reset_pose", False):
            self.commanded_pose = list(self.last_actual_pose)
        else:
            self.commanded_pose = normalize_initial_pose(self.args.initial_pos)
        if self.last_observation is not None:
            save_color_observation_for_args(self.args, self.last_observation, self.run_dir / "step_0000_reset.png")
        self.last_action = "started"
        return self.get_state(message="Session started")

    def require_started(self) -> Tuple[Any, str]:
        if not self.started:
            raise RuntimeError("Drone flight session is not started")
        return self.env, str(self.drone_name)

    def force_kill_unreal_processes(self) -> List[Dict[str, Any]]:
        return force_kill_unreal_processes(self.args)

    def close(self, *, force_kill_unreal: Optional[bool] = None) -> None:
        force_kill = bool(
            getattr(self.args, "force_kill_unreal_on_stop", DEFAULT_FORCE_KILL_UNREAL_ON_STOP)
            if force_kill_unreal is None
            else force_kill_unreal
        )
        if force_kill:
            self.force_kill_unreal_processes()

        lock_acquired = False
        if force_kill:
            lock_acquired = self.api_lock.acquire(timeout=2.0)
        else:
            self.api_lock.acquire()
            lock_acquired = True
        try:
            env = self.env
            drone_name = self.drone_name
            try:
                if lock_acquired and env is not None and drone_name is not None and not force_kill:
                    self.stop_map_touch_calibration(cleanup=True)
                    env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
                    time.sleep(0.2)
            finally:
                if self.run_dir is not None:
                    with open(self.run_dir / "trajectory.json", "w") as f:
                        json.dump(self.log, f, indent=2)
                if lock_acquired and env is not None and not force_kill:
                    env.close()
                self.env = None
                self.drone_name = None
                self.movement_enabled = False
                self.commanded_pose = None
                self.last_actual_pose = None
                self.map_touch_calibration = {}
                self.last_action = "force_stopped" if force_kill else "closed"
        finally:
            if lock_acquired:
                self.api_lock.release()
        if force_kill:
            self.force_kill_unreal_processes()

    def set_movement_enabled(self, enabled: bool) -> Dict[str, Any]:
        self.movement_enabled = bool(enabled)
        self.last_action = "movement_enabled" if self.movement_enabled else "movement_disabled"
        return self.get_state(message=f"Movement enabled={int(self.movement_enabled)}")

    @serialized_unrealcv_method
    def set_movement_mode(self, movement_mode: str) -> Dict[str, Any]:
        if movement_mode not in {"pose_lock", "physics"}:
            raise ValueError("movement_mode must be 'pose_lock' or 'physics'")
        self.movement_mode = movement_mode
        self.args.movement_mode = movement_mode
        if movement_mode == "pose_lock" and self.started:
            self.commanded_pose = self.get_pose_list()
        self.last_action = f"movement_mode_{movement_mode}"
        return self.get_state(message=f"Movement mode={movement_mode}")

    def get_pose_list(self) -> List[float]:
        with self.api_lock:
            env, drone_name = self.require_started()
            self.last_actual_pose = [float(value) for value in read_drone_pose(env, drone_name)]
            return list(self.last_actual_pose)

    def get_state(self, *, status: str = "ok", message: str = "") -> Dict[str, Any]:
        with self.api_lock:
            pose_values: List[float] = []
            if self.started:
                try:
                    pose_values = self.get_pose_list()
                except Exception as exc:
                    return {"status": "error", "message": str(exc), "started": self.started}

            pose = {}
            if len(pose_values) >= 6:
                pose = {
                    "x": pose_values[0],
                    "y": pose_values[1],
                    "z": pose_values[2],
                    "roll": pose_values[3],
                    "yaw": pose_values[4],
                    "task_yaw": pose_values[4],
                    "pitch": pose_values[5],
                    "raw": pose_values,
                }

            commanded = {}
            if self.commanded_pose and len(self.commanded_pose) >= 6:
                commanded = {
                    "x": float(self.commanded_pose[0]),
                    "y": float(self.commanded_pose[1]),
                    "z": float(self.commanded_pose[2]),
                    "roll": float(self.commanded_pose[3]),
                    "yaw": float(self.commanded_pose[4]),
                    "task_yaw": float(self.commanded_pose[4]),
                    "pitch": float(self.commanded_pose[5]),
                    "raw": list(self.commanded_pose),
                }

            error = pose_error_summary(pose_values, self.commanded_pose)
            return {
                "status": status,
                "message": message,
                "started": self.started,
                "drone_name": self.drone_name or "",
                "pose": pose,
                "commanded_pose": commanded,
                "pose_error": error,
                "movement_enabled": self.movement_enabled,
                "movement_mode": self.movement_mode,
                "last_action": self.last_action,
                "step_count": self.total_step,
                "run_dir": str(self.run_dir) if self.run_dir is not None else "",
                "env_platform": getattr(self.args, "resolved_env_platform", self.args.env_platform),
                "env_root": str(getattr(self.args, "resolved_env_root", self.args.env_root or "")),
                "env_bin": str(getattr(self.args, "resolved_env_bin", self.args.env_bin or "")),
                "map_touch_calibration": self.get_map_touch_calibration_state(),
            }

    def record_step(self, phase: str, action: Dict[str, Any],
                    actual_pose: Optional[List[float]] = None,
                    commanded_pose: Optional[List[float]] = None) -> None:
        pose = list(actual_pose) if actual_pose is not None else self.get_pose_list()
        commanded = list(commanded_pose) if commanded_pose is not None else (
            list(self.commanded_pose) if self.commanded_pose is not None else []
        )
        self.total_step += 1
        safe_action = make_json_safe(action)
        error = pose_error_summary(pose, commanded)
        entry = {
            "step": self.total_step,
            "phase": phase,
            "action": safe_action,
            "commanded_pose": commanded,
            "actual_pose": pose,
            "pose": pose,
            "pose_error": error,
            "reward": None,
            "done": False,
        }
        self.last_move_command = safe_action if isinstance(safe_action, dict) else {"value": safe_action}
        self.last_command_detail = make_json_safe({
            "step": self.total_step,
            "phase": phase,
            "action": safe_action,
            "commanded_pose": commanded,
            "actual_pose": pose,
            "pose_error": error,
            "recorded_at": datetime.now().isoformat(timespec="milliseconds"),
        })
        self.log.append(entry)

    def build_action_detail_snapshot(
        self,
        actual_pose_values: Optional[List[float]] = None,
        *,
        controller_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        actual_pose = list(actual_pose_values) if actual_pose_values is not None else (
            list(self.last_actual_pose) if self.last_actual_pose is not None else []
        )
        commanded_pose = list(self.commanded_pose) if self.commanded_pose is not None else []
        detail: Dict[str, Any] = {
            "schema_version": 1,
            "source": "controller" if controller_action else "session",
            "last_action": self.last_action,
            "movement_mode": self.movement_mode,
            "movement_enabled": bool(self.movement_enabled),
            "step_count": int(self.total_step),
            "last_move_command": self.last_move_command,
            "last_recorded_command": self.last_command_detail,
            "commanded_pose": pose_values_to_dict(commanded_pose) if commanded_pose else {},
            "actual_pose": pose_values_to_dict(actual_pose) if actual_pose else {},
            "pose_error": pose_error_summary(actual_pose, commanded_pose),
        }
        if controller_action:
            detail["controller"] = make_json_safe(controller_action)
        return make_json_safe(detail)

    def get_trajectory_points(self, limit: int = 500) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        for entry in list(self.log):
            pose = entry.get("actual_pose", entry.get("pose", []))
            if not isinstance(pose, list) or len(pose) < 2:
                continue
            try:
                point = {
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "z": float(pose[2]) if len(pose) > 2 else 0.0,
                    "yaw": float(pose[4]) if len(pose) > 4 else 0.0,
                    "step": float(entry.get("step", len(points) + 1)),
                }
            except Exception:
                continue
            points.append(point)
        if self.last_actual_pose and len(self.last_actual_pose) >= 2:
            try:
                current = {
                    "x": float(self.last_actual_pose[0]),
                    "y": float(self.last_actual_pose[1]),
                    "z": float(self.last_actual_pose[2]) if len(self.last_actual_pose) > 2 else 0.0,
                    "yaw": float(self.last_actual_pose[4]) if len(self.last_actual_pose) > 4 else 0.0,
                    "step": float(self.total_step),
                }
                if not points or abs(points[-1]["x"] - current["x"]) > 1e-6 or abs(points[-1]["y"] - current["y"]) > 1e-6:
                    points.append(current)
            except Exception:
                pass
        if limit > 0:
            points = points[-int(limit):]
        return points

    def _map_touch_marker_name(self, label: str, marker_set_id: str = "") -> str:
        safe_label = "".join(ch for ch in str(label or "") if ch.isalnum() or ch == "_")
        safe_set_id = "".join(ch for ch in str(marker_set_id or "") if ch.isalnum() or ch == "_")
        if safe_set_id:
            return f"UAVCalib_{safe_set_id}_{safe_label or 'P'}"
        return f"UAVCalib_{safe_label or 'P'}"

    def _legacy_map_touch_marker_names(self) -> List[str]:
        return [self._map_touch_marker_name(f"P{index}") for index in range(1, 6)]

    def _map_touch_api(self) -> Any:
        if self.env is None:
            raise RuntimeError("Drone flight session is not started")
        return self.env.unwrapped.unrealcv

    def _safe_map_touch_call(self, method_name: str, *args: Any) -> bool:
        try:
            api = self._map_touch_api()
            method = getattr(api, method_name, None)
            if method is None:
                return False
            method(*args)
            return True
        except Exception as exc:
            LOGGER.debug("Map touch %s%r failed: %s", method_name, args, exc)
            return False

    def _destroy_map_touch_marker(self, marker_name: str) -> None:
        if not marker_name or self.env is None:
            return
        try:
            self._map_touch_api().destroy_obj(marker_name)
        except Exception as exc:
            LOGGER.debug("Destroy calibration marker %s failed: %s", marker_name, exc)

    def _spawn_map_touch_marker(self, point: Dict[str, Any], marker_classes: List[str]) -> Dict[str, Any]:
        api = self._map_touch_api()
        marker_name = str(point["marker_name"])
        loc = [
            float(point["target_world_x"]),
            float(point["target_world_y"]),
            float(point["target_world_z"]),
        ]
        errors: List[str] = []
        for class_name in marker_classes:
            try:
                if hasattr(api, "new_obj"):
                    api.new_obj(class_name, marker_name, loc, [0.0, 0.0, 0.0])
                elif hasattr(api, "set_new_obj"):
                    result = api.set_new_obj(class_name, marker_name)
                    if result is None:
                        raise RuntimeError("spawn returned no object name")
                else:
                    raise AttributeError("UnrealCV API has no object spawn method")
                self._safe_map_touch_call("set_obj_location", marker_name, loc)
                marker_scale = normalize_calibration_marker_scale(point.get("marker_scale"))
                self._safe_map_touch_call("set_obj_scale", marker_name, marker_scale)
                self._safe_map_touch_call("set_obj_color", marker_name, [255, 235, 40])
                self._safe_map_touch_call("set_phy", marker_name, 0)
                point["marker_class"] = class_name
                point["marker_spawned"] = True
                return point
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
                LOGGER.debug("Spawn calibration marker %s as %s failed: %s", marker_name, class_name, exc)
        raise RuntimeError(f"Failed to spawn {marker_name}; tried {', '.join(errors)}")

    def _set_map_touch_marker_visibility(self, active_index: int) -> None:
        state = self.map_touch_calibration if isinstance(self.map_touch_calibration, dict) else {}
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        for index, point in enumerate(points):
            if not isinstance(point, dict):
                continue
            marker_name = str(point.get("marker_name", "") or "")
            if not marker_name:
                continue
            visible = index == int(active_index) and bool(state.get("running", False))
            if visible:
                self._safe_map_touch_call("set_show_obj", marker_name)
                self._safe_map_touch_call("set_obj_color", marker_name, [255, 235, 40])
            else:
                self._safe_map_touch_call("set_hide_obj", marker_name)

    def _public_map_touch_state(self) -> Dict[str, Any]:
        if not isinstance(self.map_touch_calibration, dict) or not self.map_touch_calibration:
            return {
                "status": "idle",
                "running": False,
                "active_index": -1,
                "active_point": "",
                "completed_count": 0,
                "points": [],
                "message": "Map touch calibration is idle",
            }
        try:
            return json.loads(json.dumps(self.map_touch_calibration))
        except Exception:
            return dict(self.map_touch_calibration)

    @serialized_unrealcv_method
    def start_map_touch_calibration(
        self,
        anchors: List[Dict[str, Any]],
        marker_z: Optional[float] = None,
        marker_class: str = DEFAULT_CALIBRATION_MARKER_CLASS,
        marker_scale: Optional[Any] = None,
    ) -> Dict[str, Any]:
        self.require_started()
        self.stop_map_touch_calibration(cleanup=True)
        source_anchors = sorted(
            [anchor for anchor in anchors if isinstance(anchor, dict)],
            key=lambda item: float(item.get("index", 9999)),
        )[:5]
        if len(source_anchors) < 5:
            raise ValueError("Map touch calibration requires P1-P5 anchors")

        current_pose = self.get_pose_list()
        z_value = float(marker_z) if marker_z is not None else float(current_pose[2])
        marker_classes = [str(marker_class or DEFAULT_CALIBRATION_MARKER_CLASS)]
        for fallback in DEFAULT_CALIBRATION_MARKER_CLASSES:
            if fallback not in marker_classes:
                marker_classes.append(fallback)
        scale_value = normalize_calibration_marker_scale(marker_scale)
        marker_set_id = f"{int(time.time() * 1000) % 100000000:08d}"

        points: List[Dict[str, Any]] = []
        for order, anchor in enumerate(source_anchors, start=1):
            label = str(anchor.get("label", f"P{order}") or f"P{order}")
            point = {
                "index": int(float(anchor.get("index", order))),
                "order": order,
                "label": label,
                "image_x": float(anchor["image_x"]),
                "image_y": float(anchor["image_y"]),
                "target_world_x": float(anchor["world_x"]),
                "target_world_y": float(anchor["world_y"]),
                "target_world_z": z_value,
                "marker_name": self._map_touch_marker_name(label, marker_set_id),
                "legacy_marker_name": self._map_touch_marker_name(label),
                "marker_class": "",
                "marker_scale": list(scale_value),
                "marker_spawned": False,
                "status": "pending",
                "distance_xy_cm": None,
                "distance_z_cm": None,
                "contact_pose": None,
                "contact_time": None,
                "contact_world_x": None,
                "contact_world_y": None,
                "contact_world_z": None,
            }
            points.append(point)

        self.map_touch_calibration = {
            "status": "starting",
            "running": True,
            "active_index": 0,
            "active_point": points[0]["label"],
            "marker_set_id": marker_set_id,
            "marker_z": z_value,
            "marker_scale": list(scale_value),
            "marker_class_preference": str(marker_class or DEFAULT_CALIBRATION_MARKER_CLASS),
            "started_at": time.time(),
            "completed_count": 0,
            "distance_xy_cm": None,
            "distance_z_cm": None,
            "points": points,
            "message": "Map touch calibration starting",
        }
        try:
            for point in points:
                self._spawn_map_touch_marker(point, marker_classes)
            points[0]["status"] = "active"
            self.map_touch_calibration.update({
                "status": "running",
                "message": "Map touch calibration started",
            })
            self._set_map_touch_marker_visibility(0)
            return self._public_map_touch_state()
        except Exception:
            self.stop_map_touch_calibration(cleanup=True)
            raise

    @serialized_unrealcv_method
    def poll_map_touch_calibration(
        self,
        xy_tolerance_cm: float = 60.0,
        z_tolerance_cm: float = 80.0,
    ) -> Dict[str, Any]:
        state = self.map_touch_calibration if isinstance(self.map_touch_calibration, dict) else {}
        if not state or not bool(state.get("running", False)):
            return self._public_map_touch_state()
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        active_index = int(state.get("active_index", 0) or 0)
        if active_index < 0 or active_index >= len(points):
            state.update({
                "status": "complete",
                "running": False,
                "active_index": -1,
                "active_point": "",
                "completed_count": len([p for p in points if isinstance(p, dict) and p.get("status") == "done"]),
                "completed_at": time.time(),
                "message": "Map touch calibration complete",
            })
            self._set_map_touch_marker_visibility(-1)
            return self._public_map_touch_state()

        pose = self.get_pose_list()
        active = points[active_index]
        target_x = float(active.get("target_world_x", 0.0))
        target_y = float(active.get("target_world_y", 0.0))
        target_z = float(active.get("target_world_z", state.get("marker_z", 0.0)))
        dx = float(pose[0]) - target_x
        dy = float(pose[1]) - target_y
        dz = float(pose[2]) - target_z
        distance_xy = float(math.sqrt(dx * dx + dy * dy))
        distance_z = float(abs(dz))
        active["distance_xy_cm"] = distance_xy
        active["distance_z_cm"] = distance_z
        state["distance_xy_cm"] = distance_xy
        state["distance_z_cm"] = distance_z
        state["active_point"] = str(active.get("label", ""))
        state["message"] = f"Waiting for {state['active_point']} contact"

        if distance_xy <= max(0.0, float(xy_tolerance_cm)) and distance_z <= max(0.0, float(z_tolerance_cm)):
            contact_pose = [float(value) for value in pose[:6]]
            active.update({
                "status": "done",
                "contact_pose": contact_pose,
                "contact_time": time.time(),
                "contact_world_x": contact_pose[0],
                "contact_world_y": contact_pose[1],
                "contact_world_z": contact_pose[2],
                "xy_tolerance_cm": float(xy_tolerance_cm),
                "z_tolerance_cm": float(z_tolerance_cm),
            })
            completed_count = len([p for p in points if isinstance(p, dict) and p.get("status") == "done"])
            next_index = active_index + 1
            if next_index < len(points):
                points[next_index]["status"] = "active"
                state.update({
                    "status": "running",
                    "running": True,
                    "active_index": next_index,
                    "active_point": str(points[next_index].get("label", "")),
                    "completed_count": completed_count,
                    "message": f"{active.get('label')} touched; advance to {points[next_index].get('label')}",
                })
                self._set_map_touch_marker_visibility(next_index)
            else:
                state.update({
                    "status": "complete",
                    "running": False,
                    "active_index": -1,
                    "active_point": "",
                    "completed_count": completed_count,
                    "completed_at": time.time(),
                    "message": "Map touch calibration complete",
                })
                self._set_map_touch_marker_visibility(-1)
        return self._public_map_touch_state()

    @serialized_unrealcv_method
    def stop_map_touch_calibration(self, cleanup: bool = True) -> Dict[str, Any]:
        state = self.map_touch_calibration if isinstance(self.map_touch_calibration, dict) else {}
        points = state.get("points", []) if isinstance(state.get("points"), list) else []
        marker_names = {
            str(point.get("marker_name", "") or "")
            for point in points
            if isinstance(point, dict)
        }
        marker_names.update(self._legacy_map_touch_marker_names())
        if cleanup:
            for marker_name in sorted(name for name in marker_names if name):
                self._destroy_map_touch_marker(marker_name)
        if not state:
            self.map_touch_calibration = {
                "status": "stopped",
                "running": False,
                "active_index": -1,
                "active_point": "",
                "completed_count": 0,
                "points": [],
                "message": "Map touch calibration stopped",
            }
        else:
            state.update({
                "status": "stopped",
                "running": False,
                "active_index": -1,
                "active_point": "",
                "stopped_at": time.time(),
                "message": "Map touch calibration stopped",
            })
            self.map_touch_calibration = state
        return self._public_map_touch_state()

    def get_map_touch_calibration_state(self) -> Dict[str, Any]:
        return self._public_map_touch_state()

    @serialized_unrealcv_method
    def reset_map_touch_calibration_markers(self) -> Dict[str, Any]:
        state = self.stop_map_touch_calibration(cleanup=True)
        self.map_touch_calibration = {
            "status": "idle",
            "running": False,
            "active_index": -1,
            "active_point": "",
            "completed_count": 0,
            "points": [],
            "message": "Calibration markers reset",
            "previous_state": state,
        }
        return self._public_map_touch_state()

    @serialized_unrealcv_method
    def capture_observation(self, *, save: bool = False, label: str = "capture") -> Optional[np.ndarray]:
        env, drone_name = self.require_started()
        image = read_drone_observation(env, drone_name)
        self.last_observation = image
        if save and image is not None and self.run_dir is not None:
            save_color_observation_for_args(self.args, image, self.run_dir / f"step_{self.total_step:04d}_{label}.png")
        return image

    def _capture_bundle_to_dir(
        self,
        env: Any,
        drone_name: str,
        capture_dir: Path,
        *,
        extra_result: Optional[Dict[str, Any]] = None,
        extra_pose: Optional[Dict[str, Any]] = None,
        controller_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        capture_dir.mkdir(parents=True, exist_ok=True)
        rgb = read_drone_observation(env, drone_name)
        depth = read_drone_depth(env, drone_name)
        pose_values = [float(value) for value in read_drone_pose(env, drone_name)]
        pose = pose_values_to_dict(pose_values)
        action_detail = self.build_action_detail_snapshot(pose_values, controller_action=controller_action)
        if rgb is None:
            raise RuntimeError("RGB observation is unavailable for current drone camera")
        if depth is None:
            raise RuntimeError("Depth observation is unavailable for current drone camera")

        rgb_path = capture_dir / "rgb.png"
        depth_cm_path = capture_dir / "depth_cm.png"
        depth_preview_path = capture_dir / "depth_preview.png"
        depth_npy_path = capture_dir / "depth.npy"
        pose_json_path = capture_dir / "pose.json"
        action_json_path = capture_dir / "action.json"
        capture_json_path = capture_dir / "capture.json"

        save_color_observation_for_args(self.args, rgb, rgb_path)
        depth_summary = save_depth_capture_outputs(
            depth,
            depth_cm_path=depth_cm_path,
            depth_preview_path=depth_preview_path,
            depth_npy_path=depth_npy_path,
            min_depth_cm=float(getattr(self.args, "depth_min_cm", DEFAULT_DEPTH_MIN_CM)),
            max_depth_cm=float(getattr(self.args, "depth_max_cm", DEFAULT_DEPTH_MAX_CM)),
        )
        self.last_observation = rgb
        self.last_actual_pose = pose_values

        capture_time = datetime.now().isoformat(timespec="milliseconds")
        pose_payload = {
            "capture_time": capture_time,
            "drone_name": drone_name,
            "pose": pose,
            "movement_mode": self.movement_mode,
            "movement_enabled": bool(self.movement_enabled),
            "last_action": self.last_action,
            "step_count": int(self.total_step),
            "commanded_pose": action_detail.get("commanded_pose", {}),
            "actual_pose": action_detail.get("actual_pose", {}),
            "pose_error": action_detail.get("pose_error", {}),
            "action_detail": action_detail,
        }
        if extra_pose:
            pose_payload.update(extra_pose)
        pose_json_path.write_text(json.dumps(pose_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        action_json_path.write_text(json.dumps(action_detail, indent=2, ensure_ascii=False), encoding="utf-8")

        result = {
            "status": "ok",
            "message": f"Capture saved: {capture_dir}",
            "capture_time": capture_time,
            "drone_name": drone_name,
            "capture_dir": str(capture_dir),
            "rgb_path": str(rgb_path),
            "depth_cm_path": str(depth_cm_path),
            "depth_preview_path": str(depth_preview_path),
            "depth_npy_path": str(depth_npy_path),
            "pose_json_path": str(pose_json_path),
            "action_json_path": str(action_json_path),
            "capture_json_path": str(capture_json_path),
            "rgb_source_order": str(getattr(self.args, "rgb_source_order", DEFAULT_RGB_SOURCE_ORDER) or DEFAULT_RGB_SOURCE_ORDER),
            "enhance_rgb": bool(getattr(self.args, "enhance_rgb", DEFAULT_RGB_ENHANCE_ENABLED)),
            "depth_min_cm": float(getattr(self.args, "depth_min_cm", DEFAULT_DEPTH_MIN_CM)),
            "depth_max_cm": float(getattr(self.args, "depth_max_cm", DEFAULT_DEPTH_MAX_CM)),
            "pose": pose,
            "commanded_pose": action_detail.get("commanded_pose", {}),
            "actual_pose": action_detail.get("actual_pose", {}),
            "pose_error": action_detail.get("pose_error", {}),
            "action_detail": action_detail,
            "depth_summary": depth_summary,
        }
        if extra_result:
            result.update(extra_result)
        capture_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def _capture_lidar_bundle_to_dir(
        self,
        env: Any,
        drone_name: str,
        capture_dir: Path,
        *,
        extra_result: Optional[Dict[str, Any]] = None,
        extra_pose: Optional[Dict[str, Any]] = None,
        controller_action: Optional[Dict[str, Any]] = None,
        raw_capture_only: bool = False,
    ) -> Dict[str, Any]:
        capture_dir.mkdir(parents=True, exist_ok=True)
        rgb = read_drone_observation(env, drone_name)
        depth = read_drone_depth(env, drone_name)
        pose_values = [float(value) for value in read_drone_pose(env, drone_name)]
        pose = pose_values_to_dict(pose_values)
        action_detail = self.build_action_detail_snapshot(pose_values, controller_action=controller_action)
        if rgb is None:
            raise RuntimeError("RGB observation is unavailable for current drone camera")
        if depth is None:
            raise RuntimeError("Depth observation is unavailable for current drone camera")

        rgb_image = prepare_observation_rgb(rgb, **rgb_enhance_options(self.args))
        if rgb_image is None:
            raise RuntimeError("RGB observation could not be converted to an image")

        rgb_path = capture_dir / "rgb.png"
        depth_cm_path = capture_dir / "depth_cm.png"
        depth_preview_path = capture_dir / "depth_preview.png"
        depth_npy_path = capture_dir / "depth.npy"
        pose_json_path = capture_dir / "pose.json"
        action_json_path = capture_dir / "action.json"
        capture_json_path = capture_dir / "capture.json"
        camera_info_path = capture_dir / "camera_info.json"

        Image.fromarray(rgb_image).save(rgb_path)
        depth_summary = save_depth_capture_outputs(
            depth,
            depth_cm_path=depth_cm_path,
            depth_preview_path=depth_preview_path,
            depth_npy_path=depth_npy_path,
            min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", DEFAULT_LIDAR_DEPTH_MIN_CM)),
            max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", DEFAULT_LIDAR_DEPTH_MAX_CM)),
        )
        camera_info = read_drone_camera_info(env, drone_name, rgb=rgb_image, depth=depth)
        if raw_capture_only:
            camera_info_payload = dict(camera_info)
            camera_info_payload.update(
                {
                    "min_depth_cm": float(getattr(self.args, "lidar_depth_min_cm", DEFAULT_LIDAR_DEPTH_MIN_CM)),
                    "max_depth_cm": float(getattr(self.args, "lidar_depth_max_cm", DEFAULT_LIDAR_DEPTH_MAX_CM)),
                    "depth_input_mode": "unrealcv_depth_npy",
                    "depth_projection_requested": str(getattr(self.args, "lidar_depth_projection", DEFAULT_LIDAR_DEPTH_PROJECTION)),
                    "depth_projection_selected": select_lidar_depth_projection(
                        getattr(self.args, "lidar_depth_projection", DEFAULT_LIDAR_DEPTH_PROJECTION)
                    ),
                    "coordinate_frame": "standard_zup",
                    "coordinate_units": "m",
                }
            )
            camera_info_path.write_text(json.dumps(camera_info_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            lidar_outputs = {
                "point_count": 0,
                "invalid_depth_count": 0,
                "source_mode": "raw_depth_lidar",
                "point_cloud_shape": [0, 6],
                "min_depth_cm": float(getattr(self.args, "lidar_depth_min_cm", DEFAULT_LIDAR_DEPTH_MIN_CM)),
                "max_depth_cm": float(getattr(self.args, "lidar_depth_max_cm", DEFAULT_LIDAR_DEPTH_MAX_CM)),
                "depth_input_mode": "unrealcv_depth_npy",
                "depth_projection": select_lidar_depth_projection(
                    getattr(self.args, "lidar_depth_projection", DEFAULT_LIDAR_DEPTH_PROJECTION)
                ),
                "depth_projection_selected": select_lidar_depth_projection(
                    getattr(self.args, "lidar_depth_projection", DEFAULT_LIDAR_DEPTH_PROJECTION)
                ),
                "projection_corrected": False,
                "coordinate_frame": "standard_zup",
                "coordinate_units": "m",
                "camera_info_path": str(camera_info_path),
                "raw_capture_only": True,
                "postprocess_status": "pending",
                "postprocess_started_at": "",
                "postprocess_finished_at": "",
            }
        else:
            lidar_outputs = save_lidar_capture_outputs(
                rgb_image=rgb_image,
                depth_image=depth,
                camera_info=camera_info,
                output_dir=capture_dir,
                min_depth_cm=float(getattr(self.args, "lidar_depth_min_cm", DEFAULT_LIDAR_DEPTH_MIN_CM)),
                max_depth_cm=float(getattr(self.args, "lidar_depth_max_cm", DEFAULT_LIDAR_DEPTH_MAX_CM)),
                lidar_depth_projection=str(getattr(self.args, "lidar_depth_projection", DEFAULT_LIDAR_DEPTH_PROJECTION)),
            )
            lidar_outputs.update(
                {
                    "raw_capture_only": False,
                    "postprocess_status": "done",
                    "postprocess_started_at": "",
                    "postprocess_finished_at": datetime.now().isoformat(timespec="milliseconds"),
                }
            )
        self.last_observation = rgb
        self.last_actual_pose = pose_values

        capture_time = datetime.now().isoformat(timespec="milliseconds")
        pose_payload = {
            "capture_time": capture_time,
            "drone_name": drone_name,
            "pose": pose,
            "movement_mode": self.movement_mode,
            "movement_enabled": bool(self.movement_enabled),
            "last_action": self.last_action,
            "step_count": int(self.total_step),
            "commanded_pose": action_detail.get("commanded_pose", {}),
            "actual_pose": action_detail.get("actual_pose", {}),
            "pose_error": action_detail.get("pose_error", {}),
            "action_detail": action_detail,
        }
        if extra_pose:
            pose_payload.update(extra_pose)
        pose_json_path.write_text(json.dumps(pose_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        action_json_path.write_text(json.dumps(action_detail, indent=2, ensure_ascii=False), encoding="utf-8")

        result = {
            "status": "ok",
            "message": f"Capture saved: {capture_dir}",
            "capture_time": capture_time,
            "drone_name": drone_name,
            "capture_dir": str(capture_dir),
            "rgb_path": str(rgb_path),
            "pose_json_path": str(pose_json_path),
            "action_json_path": str(action_json_path),
            "capture_json_path": str(capture_json_path),
            "depth_cm_path": str(depth_cm_path),
            "depth_preview_path": str(depth_preview_path),
            "depth_npy_path": str(depth_npy_path),
            "rgb_source_order": str(getattr(self.args, "rgb_source_order", DEFAULT_RGB_SOURCE_ORDER) or DEFAULT_RGB_SOURCE_ORDER),
            "enhance_rgb": bool(getattr(self.args, "enhance_rgb", DEFAULT_RGB_ENHANCE_ENABLED)),
            "lidar_depth_min_cm": float(getattr(self.args, "lidar_depth_min_cm", DEFAULT_LIDAR_DEPTH_MIN_CM)),
            "lidar_depth_max_cm": float(getattr(self.args, "lidar_depth_max_cm", DEFAULT_LIDAR_DEPTH_MAX_CM)),
            "lidar_depth_projection": str(getattr(self.args, "lidar_depth_projection", DEFAULT_LIDAR_DEPTH_PROJECTION)),
            "lidar_capture_processing": normalize_lidar_capture_processing(
                getattr(self.args, "lidar_capture_processing", DEFAULT_LIDAR_CAPTURE_PROCESSING)
            ),
            "pose": pose,
            "commanded_pose": action_detail.get("commanded_pose", {}),
            "actual_pose": action_detail.get("actual_pose", {}),
            "pose_error": action_detail.get("pose_error", {}),
            "action_detail": action_detail,
            "depth_summary": depth_summary,
            **lidar_outputs,
        }
        if extra_result:
            result.update(extra_result)
        capture_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    @serialized_unrealcv_method
    def capture_temp_bundle(self, output_root: Optional[Any] = None) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        root_value = output_root or getattr(self.args, "temp_capture_dir", DEFAULT_TEMP_CAPTURE_DIR)
        root_path = resolve_project_output_path(root_value, DEFAULT_TEMP_CAPTURE_DIR)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        capture_dir = make_unique_child_dir(root_path, timestamp)
        return self._capture_bundle_to_dir(
            env,
            drone_name,
            capture_dir,
            extra_result={
                "capture_kind": "temp_capture",
                "message": f"Temp capture saved: {capture_dir}",
            },
        )

    @serialized_unrealcv_method
    def capture_temp_lidar_bundle(self, output_root: Optional[Any] = None) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        root_value = output_root or getattr(self.args, "temp_capture_lidar_dir", DEFAULT_TEMP_CAPTURE_LIDAR_DIR)
        root_path = resolve_project_output_path(root_value, DEFAULT_TEMP_CAPTURE_LIDAR_DIR)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        capture_dir = make_unique_child_dir(root_path, timestamp)
        return self._capture_lidar_bundle_to_dir(
            env,
            drone_name,
            capture_dir,
            extra_result={
                "capture_kind": "temp_capture_lidar",
                "message": f"Temp capture lidar saved: {capture_dir}",
            },
        )

    @serialized_unrealcv_method
    def capture_stream_frame(
        self,
        stream_dir: Any,
        frame_index: int,
        action_detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        stream_path = resolve_project_output_path(stream_dir, DEFAULT_STREAM_CAPTURE_DIR)
        frames_dir = stream_path / "frames"
        frame_number = max(0, int(frame_index))
        frame_name = f"frame_{frame_number:06d}"
        capture_dir = make_unique_child_dir(frames_dir, frame_name)
        return self._capture_bundle_to_dir(
            env,
            drone_name,
            capture_dir,
            extra_pose={"frame_index": frame_number, "stream_dir": str(stream_path)},
            extra_result={
                "capture_kind": "stream_capture",
                "frame_index": frame_number,
                "stream_dir": str(stream_path),
                "frames_dir": str(frames_dir),
                "message": f"Stream frame {frame_number} saved: {capture_dir}",
            },
            controller_action=action_detail,
        )

    @serialized_unrealcv_method
    def capture_lidar_stream_frame(
        self,
        stream_dir: Any,
        frame_index: int,
        action_detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        stream_path = resolve_project_output_path(stream_dir, DEFAULT_STREAM_CAPTURE_LIDAR_DIR)
        frames_dir = stream_path / "frames"
        frame_number = max(0, int(frame_index))
        frame_name = f"frame_{frame_number:06d}"
        capture_dir = make_unique_child_dir(frames_dir, frame_name)
        processing_mode = normalize_lidar_capture_processing(
            getattr(self.args, "lidar_capture_processing", DEFAULT_LIDAR_CAPTURE_PROCESSING)
        )
        return self._capture_lidar_bundle_to_dir(
            env,
            drone_name,
            capture_dir,
            extra_pose={"frame_index": frame_number, "stream_dir": str(stream_path)},
            extra_result={
                "capture_kind": "stream_capture_lidar",
                "frame_index": frame_number,
                "stream_dir": str(stream_path),
                "frames_dir": str(frames_dir),
                "lidar_capture_processing": processing_mode,
                "message": f"Stream lidar frame {frame_number} saved: {capture_dir}",
            },
            controller_action=action_detail,
            raw_capture_only=(processing_mode == "smooth"),
        )

    @serialized_unrealcv_method
    def set_pose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        if "pose" in payload and isinstance(payload["pose"], list):
            pose = normalize_initial_pose([float(value) for value in payload["pose"]])
        else:
            current = self.get_pose_list()
            pose = [
                float(payload.get("x", current[0])),
                float(payload.get("y", current[1])),
                float(payload.get("z", current[2])),
                float(payload.get("roll", current[3] if len(current) > 3 else 0.0)),
                float(payload.get("yaw", payload.get("task_yaw", current[4] if len(current) > 4 else 0.0))),
                float(payload.get("pitch", current[5] if len(current) > 5 else 0.0)),
            ]
        self.commanded_pose = list(pose)
        set_drone_world_pose(env, drone_name, pose)
        actual_pose = wait_for_drone_pose(
            env,
            drone_name,
            pose,
            timeout_s=float(getattr(self.args, "pose_settle_timeout", DEFAULT_POSE_SETTLE_TIMEOUT)),
            position_tolerance_cm=float(getattr(self.args, "pose_settle_pos_tolerance", DEFAULT_POSE_SETTLE_POS_TOLERANCE)),
            yaw_tolerance_deg=float(getattr(self.args, "pose_settle_yaw_tolerance", DEFAULT_POSE_SETTLE_YAW_TOLERANCE)),
        )
        time.sleep(max(0.0, float(getattr(self.args, "step_delay", 0.1))))
        self.last_action = "set_pose"
        self.record_step("set_pose", {"pose": pose}, actual_pose=actual_pose, commanded_pose=pose)
        return self.get_state(message="Pose set")

    @serialized_unrealcv_method
    def move_relative(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        if not self.movement_enabled:
            return self.get_state(status="disabled", message="Basic movement is disabled")

        if self.movement_mode == "physics":
            return self.move_relative_physics(payload)

        pose = list(self.commanded_pose) if self.commanded_pose is not None else self.get_pose_list()
        yaw = float(pose[4]) if len(pose) > 4 else 0.0
        forward_cm = float(payload.get("forward_cm", 0.0) or 0.0)
        right_cm = float(payload.get("right_cm", 0.0) or 0.0)
        up_cm = float(payload.get("up_cm", 0.0) or 0.0)
        yaw_delta_deg = float(payload.get("yaw_delta_deg", 0.0) or 0.0)
        if payload.get("yaw_target_deg") is not None:
            target_yaw = normalize_angle_deg(float(payload["yaw_target_deg"]))
        else:
            target_yaw = normalize_angle_deg(yaw + yaw_delta_deg)

        yaw_rad = math.radians(yaw)
        dx = forward_cm * math.cos(yaw_rad) - right_cm * math.sin(yaw_rad)
        dy = forward_cm * math.sin(yaw_rad) + right_cm * math.cos(yaw_rad)
        next_pose = [
            float(pose[0]) + dx,
            float(pose[1]) + dy,
            float(pose[2]) + up_cm,
            float(pose[3]) if len(pose) > 3 else 0.0,
            target_yaw,
            float(pose[5]) if len(pose) > 5 else 0.0,
        ]
        self.commanded_pose = list(next_pose)
        set_drone_world_pose(env, drone_name, next_pose)
        actual_pose = wait_for_drone_pose(
            env,
            drone_name,
            next_pose,
            timeout_s=float(getattr(self.args, "pose_settle_timeout", DEFAULT_POSE_SETTLE_TIMEOUT)),
            position_tolerance_cm=float(getattr(self.args, "pose_settle_pos_tolerance", DEFAULT_POSE_SETTLE_POS_TOLERANCE)),
            yaw_tolerance_deg=float(getattr(self.args, "pose_settle_yaw_tolerance", DEFAULT_POSE_SETTLE_YAW_TOLERANCE)),
        )
        time.sleep(max(0.0, float(getattr(self.args, "step_delay", 0.1))))
        action_name = str(payload.get("action_name", "move_relative") or "move_relative")
        self.last_action = action_name
        self.record_step("move_relative", {
            "action_name": action_name,
            "forward_cm": forward_cm,
            "right_cm": right_cm,
            "up_cm": up_cm,
            "yaw_delta_deg": yaw_delta_deg,
            "yaw_target_deg": target_yaw,
        }, actual_pose=actual_pose, commanded_pose=next_pose)
        if self.args.save_every > 0 and self.total_step % self.args.save_every == 0:
            self.capture_observation(save=True, label=action_name)
        return self.get_state(message=f"Moved {action_name}")

    @serialized_unrealcv_method
    def move_relative_physics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        forward_cm = float(payload.get("forward_cm", 0.0) or 0.0)
        right_cm = float(payload.get("right_cm", 0.0) or 0.0)
        up_cm = float(payload.get("up_cm", 0.0) or 0.0)
        yaw_delta_deg = float(payload.get("yaw_delta_deg", 0.0) or 0.0)
        action_name = str(payload.get("action_name", "move_relative") or "move_relative")

        linear_step = 20.0
        yaw_step = 30.0
        action = [
            float(np.clip(forward_cm / linear_step * self.args.linear_speed, -1.0, 1.0)),
            float(np.clip(right_cm / linear_step * self.args.linear_speed, -1.0, 1.0)),
            float(np.clip(up_cm / linear_step * (self.args.vertical_speed if up_cm >= 0 else self.args.down_speed), -1.0, 1.0)),
            float(np.clip(yaw_delta_deg / yaw_step * self.args.yaw_speed, -1.0, 1.0)),
        ]

        if any(abs(value) > 1e-9 for value in action):
            env.unwrapped.unrealcv.set_phy(drone_name, 1)
            env.unwrapped.unrealcv.set_move_bp(drone_name, action)
            time.sleep(max(0.0, float(getattr(self.args, "control_dt", 0.1))))
        env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])

        actual_pose = self.get_pose_list()
        self.commanded_pose = list(actual_pose)
        self.last_action = action_name
        self.record_step("move_physics", {
            "action_name": action_name,
            "forward_cm": forward_cm,
            "right_cm": right_cm,
            "up_cm": up_cm,
            "yaw_delta_deg": yaw_delta_deg,
            "velocity_action": action,
        }, actual_pose=actual_pose, commanded_pose=self.commanded_pose)
        if self.args.save_every > 0 and self.total_step % self.args.save_every == 0:
            self.capture_observation(save=True, label=action_name)
        return self.get_state(message=f"Moved {action_name} ({self.movement_mode})")

    @serialized_unrealcv_method
    def run_scripted(self) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        observation = self.capture_observation()
        if observation is None:
            observation = self.last_observation
        if observation is None:
            observation = read_drone_observation(env, drone_name)
        offset = self.total_step
        start_index = len(self.log)
        steps = run_scripted_plan(self.args, env, drone_name, observation, self.run_dir or Path("."), self.log)
        for entry in self.log[start_index:]:
            entry["step"] = int(entry.get("step", 0) or 0) + offset
        self.total_step += int(steps)
        self.commanded_pose = self.get_pose_list()
        self.last_action = "scripted"
        return self.get_state(message=f"Scripted plan completed: {steps} steps")

    @serialized_unrealcv_method
    def run_orbit(self, *, center: List[float], radius: float, altitude: float, steps: int,
                  start_angle: float, clockwise: bool) -> Dict[str, Any]:
        env, drone_name = self.require_started()
        self.args.orbit_center = [float(center[0]), float(center[1])]
        self.args.orbit_radius = float(radius)
        self.args.orbit_altitude = float(altitude)
        self.args.orbit_steps = int(steps)
        self.args.orbit_start_angle = float(start_angle)
        self.args.orbit_clockwise = bool(clockwise)
        observation = self.capture_observation()
        if observation is None:
            observation = self.last_observation
        if observation is None:
            observation = read_drone_observation(env, drone_name)
        offset = self.total_step
        start_index = len(self.log)
        orbit_steps = run_orbit_house_plan(self.args, env, drone_name, observation, self.run_dir or Path("."), self.log)
        for entry in self.log[start_index:]:
            entry["step"] = int(entry.get("step", 0) or 0) + offset
        if len(self.log) > start_index:
            last_commanded = self.log[-1].get("commanded_pose")
            if isinstance(last_commanded, list) and len(last_commanded) >= 6:
                self.commanded_pose = [float(value) for value in last_commanded[:6]]
        self.total_step += int(orbit_steps)
        self.last_action = "orbit"
        return self.get_state(message=f"Orbit completed: {orbit_steps} steps")


def run_flight_test(args: argparse.Namespace) -> Path:
    configure_local_unreal_env(args)
    validate_unreal_env_config(args)
    patch_env_setting_loader(args)
    LOGGER.info("UnrealEnv root: %s", args.resolved_env_root)
    LOGGER.info("Unreal platform: %s", args.resolved_env_platform)
    LOGGER.info("Unreal binary: %s", args.resolved_env_bin or "from setting file")
    ensure_legacy_gym_entrypoint_loader()
    patch_macos_launchservices_launcher()
    np.random.seed(args.seed)
    run_dir = make_output_dir(args.output_dir)
    env = None
    log: List[Dict[str, Any]] = []
    total_step = 0

    try:
        env, drone_name, observation = prepare_drone_env(args)
        if args.mode == "scripted":
            total_step = run_scripted_plan(args, env, drone_name, observation, run_dir, log)
        elif args.mode == "orbit":
            total_step = run_orbit_house_plan(args, env, drone_name, observation, run_dir, log)
        else:
            total_step = run_keyboard_control(args, env, drone_name, observation, run_dir, log)

        env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
        time.sleep(0.5)
    finally:
        with open(run_dir / "trajectory.json", "w") as f:
            json.dump(log, f, indent=2)
        if env is not None:
            env.close()

    LOGGER.info("Saved %s steps to %s", total_step, run_dir)
    return run_dir


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(asctime)s - %(message)s",
    )
    if args.dry_run:
        configure_local_unreal_env(args)
        validate_unreal_env_config(args)
        print_launch_config(args)
        sys.exit(0)
    output = run_flight_test(args)
    print(f"Drone flight test output: {output}")
