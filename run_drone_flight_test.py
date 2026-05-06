import argparse
import json
import logging
import os
import plistlib
import select
import signal
import subprocess
import sys
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
import numpy as np
from PIL import Image

if sys.platform == "darwin":
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/uav-flow-mpl")
else:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TEMP", ".")) / "uav-flow-mpl"))

from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation


LOGGER = logging.getLogger(__name__)


DEFAULT_ENV_ID = "UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0"
DEFAULT_OUTPUT_DIR = "results/drone_flight_test"
DEFAULT_ENV_ROOT = "UnrealEnv"
DEFAULT_WIN_ENV_BIN = "UE4_ExampleScene_Win/UE4_ExampleScene/Binaries/Win64/UE4_ExampleScene.exe"
DEFAULT_MAC_ENV_BIN = "UE4_ExampleScene_Mac/UE4_ExampleScene.app"
DEFAULT_INITIAL_POS = [0.0, 0.0, 100.0, 0.0]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch UnrealCV and run a simple drone flight smoke test.",
        epilog=(
            "Examples:\n"
            "  Windows: python run_drone_flight_test.py --env_platform win --mode keyboard\n"
            "  macOS:   python run_drone_flight_test.py --env_platform mac --env_root /path/to/UnrealEnv --mode keyboard"
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
    parser.add_argument("--mode", choices=["keyboard", "scripted"], default="keyboard",
                        help="keyboard keeps the drone view alive for manual control; scripted runs the smoke test plan")
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
    parser.add_argument("--initial_pos", nargs="+", type=float, default=DEFAULT_INITIAL_POS,
                        metavar="POSE",
                        help="Initial drone pose after reset: X Y Z YAW or X Y Z ROLL YAW PITCH")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def make_output_dir(base_dir: str) -> Path:
    run_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_color_observation(observation: Any, path: Path) -> None:
    image = np.asarray(observation)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        return
    image = image[:, :, :3]
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def make_env(args: argparse.Namespace):
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


def reset_drone_pose(env: Any, drone_name: str, pose: List[float]) -> None:
    env.unwrapped.unrealcv.set_obj_location(drone_name, pose[:3])
    env.unwrapped.unrealcv.set_rotation(drone_name, pose[4] - 180)
    env.unwrapped.unrealcv.set_phy(drone_name, 0)
    env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
    set_drone_camera(env, drone_name)
    env.unwrapped.unrealcv.set_viewport(drone_name)
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


def read_drone_observation(env: Any, drone_name: str) -> Optional[np.ndarray]:
    cam_id = env.unwrapped.agents.get(drone_name, {}).get("cam_id")
    if cam_id is None or cam_id < 0:
        return None
    return env.unwrapped.unrealcv.read_image(cam_id, "lit", "direct")


def stabilize_drone_at_current_pose(env: Any, drone_name: str) -> None:
    env.unwrapped.unrealcv.set_phy(drone_name, 0)
    env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
    set_drone_camera(env, drone_name)
    env.unwrapped.unrealcv.set_viewport(drone_name)
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


def run_scripted_plan(args: argparse.Namespace, env: Any, drone_name: str, observation: Any,
                      run_dir: Path, log: List[Dict[str, Any]]) -> int:
    total_step = 0
    save_color_observation(observation, run_dir / "step_0000_reset.png")

    for phase, action in DEFAULT_ACTION_PLAN:
        LOGGER.info("Phase %-8s action=%s", phase, action)
        for _ in range(args.steps_per_action):
            env_actions = [None] * len(env.unwrapped.player_list)
            env_actions[0] = action
            observation, reward, done, info = step_drone_flight_env(env, env_actions)
            env.unwrapped.unrealcv.set_viewport(drone_name)
            total_step += 1
            append_step_log(log, total_step, phase, action, reward, done, info)
            if args.save_every > 0 and total_step % args.save_every == 0:
                save_color_observation(observation, run_dir / f"step_{total_step:04d}_{phase}.png")
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
    save_color_observation(observation, run_dir / "step_0000_reset.png")
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
            save_color_observation(image, run_dir / f"step_{total_step:04d}_{phase}.png")
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
