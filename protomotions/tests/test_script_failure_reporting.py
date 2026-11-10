# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from protomotions.utils import simulator_imports

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUTORIAL_NAMES = (
    "0_create_simulator",
    "1_add_terrain",
    "2_load_robot",
    "3_scene_creation",
    "4_basic_environment",
    "5_motion_manager",
    "6_mimic_environment",
    "7_deepmimic",
)
EXAMPLE_DEFINITIONS = {
    "env_kinematic_playback": ("create_parser", "main"),
    "motion_libs_visualizer": (
        "RobotSpec",
        "_diff",
        "normalized_jerk_from_vel",
        "oscillation_index_from_vel",
        "purposeful_jerk_from_vel",
        "create_checkerboard_ground",
        "_longest_clip_length",
        "_exact_frame_state",
        "MotionVisualizerSmoothness",
        "main",
    ),
    "random_pose_visualizer": ("RobotSpec", "RandomPoseVisualizer", "main"),
    "visualize_soma_scenes": (
        "make_capsule_mesh",
        "_rotation_between",
        "make_box_mesh",
        "make_sphere_mesh",
        "build_body_local_meshes",
        "quat_xyzw_to_wxyz",
        "load_motion_data",
        "load_scene_data",
        "get_motion_frames",
        "load_scene_meshes",
        "main",
    ),
}
EXAMPLE_ARGUMENTS = {
    "env_kinematic_playback": [
        "--robot-name",
        "g1",
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        "--motion-file",
        "unused.pt",
        "--experiment-path",
        "unused.py",
    ],
    "motion_libs_visualizer": ["--motion_files", "unused.pt"],
}
ENTRYPOINTS = tuple(f"examples/tutorial/{name}.py" for name in TUTORIAL_NAMES) + tuple(
    f"examples/{name}.py" for name in EXAMPLE_DEFINITIONS
)
CLEANUP_ENTRYPOINTS = (
    *(f"examples/tutorial/{name}.py" for name in TUTORIAL_NAMES),
    "examples/env_kinematic_playback.py",
    "examples/motion_libs_visualizer.py",
    "examples/random_pose_visualizer.py",
)
CLEANUP_HARNESS = """
import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from protomotions.launch.startup import run_reporting_failures


def close():
    print("explicit close ran", flush=True)
    os._exit(0)


path = Path(sys.argv[1])
mode = sys.argv[2]
tree = ast.parse(path.read_text(), filename=str(path))
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
loop_index = max(index for index, node in enumerate(main.body) if isinstance(node, ast.Try))
# Keep the script's actual exception handlers and cleanup placement; omit asset construction and loop work.
main.body = main.body[loop_index:]
main.body[0].body = ast.parse({
    "failure": 'raise RuntimeError("injected loop failure")',
    "success": 'print("loop finished")',
    "interrupt": 'raise KeyboardInterrupt',
    "exit_none": 'raise SystemExit(None)',
    "exit_zero": 'raise SystemExit(0)',
    "exit_failure": 'raise SystemExit(7)',
    "exit_message": 'raise SystemExit("requested failure")',
    "exit_float_zero": 'raise SystemExit(0.0)',
}[mode]).body
simulator = SimpleNamespace(close=close)
env = simulator
visualizer = SimpleNamespace(simulator=simulator)
exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(path), "exec"))
run_reporting_failures(main)
"""

VIEWER_CLOSE_HARNESS = """
import ast
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from protomotions.launch.startup import run_reporting_failures

root = Path(sys.argv[1])
render_path = root / "protomotions/simulator/isaacgym/simulator.py"
render_class = next(node for node in ast.parse(render_path.read_text()).body
                    if isinstance(node, ast.ClassDef) and node.name == "IsaacGymSimulator")
render = next(node for node in render_class.body
              if isinstance(node, ast.FunctionDef) and node.name == "render")
exec(compile(ast.Module(body=[render], type_ignores=[]), str(render_path), "exec"))

class Simulator:
    render = render
    headless = False
    _viewer = object()
    _gym = SimpleNamespace(query_viewer_has_closed=lambda viewer: True)

    def __init__(self):
        self._temp_dir = tempfile.TemporaryDirectory(dir=sys.argv[2], prefix="viewer-assets-")
        print("temporary assets created", flush=True)

    def _update_camera(self):
        pass

    def close(self):
        self._temp_dir.cleanup()
        print("simulator close removed assets", flush=True)

class RandomPoseVisualizer:
    def __init__(self, **kwargs):
        self.simulator = Simulator()

    def run(self):
        self.simulator.render()

torch = SimpleNamespace(device=lambda value: value)
args = SimpleNamespace(cpu_only=True, simulator="isaacgym", robot="g1", num_envs=1, headless=False)
path = root / "examples/random_pose_visualizer.py"
main = next(node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "main")
exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"))
run_reporting_failures(main)
"""

FAILURE_HARNESS = """
import atexit
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from protomotions.launch import startup
from protomotions.utils import simulator_imports


def shutdown():
    print("atexit ran", flush=True)
    os._exit(0)


def fail_startup(simulator):
    raise RuntimeError("injected tutorial startup failure")


def fail_main(frame, event, arg):
    if event == "call" and frame.f_code is startup.run_reporting_failures.__code__:
        print("shared reporter entered", flush=True)
    if event == "call" and frame.f_code.co_name == "main" and frame.f_code.co_filename == str(path):
        raise RuntimeError("injected main failure")
    return None


atexit.register(shutdown)
path = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
sys.argv = [str(path), *sys.argv[3:]]
# Only the backend pre-import and optional browser viewer need fakes; script imports and definitions stay real.
simulator_imports.import_simulator_before_torch = lambda simulator: None
sys.modules["viser"] = ModuleType("viser")
if mode == "startup":
    simulator_imports.import_simulator_before_torch = fail_startup
else:
    sys.settrace(fail_main)
runpy.run_path(str(path), run_name="__main__")
"""


def _script_arguments(path: Path) -> list[str]:
    if path.parent.name == "examples":
        return EXAMPLE_ARGUMENTS.get(path.stem, [])
    arguments = ["--simulator", "mujoco", "--cpu-only", "--headless"]
    if path.stem in {"2_load_robot", "3_scene_creation", "4_basic_environment"}:
        arguments.extend(["--robot", "g1"])
    return arguments


def _run_child(code: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *arguments],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


@pytest.mark.parametrize("relative_path", ENTRYPOINTS)
def test_script_main_failure_survives_shutdown_hook(relative_path: str) -> None:
    """Every script reports its main failure before a shutdown hook can replace the exit status."""
    path = PROJECT_ROOT / relative_path
    result = _run_child(FAILURE_HARNESS, str(path), "main", *_script_arguments(path))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback (most recent call last)" in result.stderr
    assert "RuntimeError: injected main failure" in result.stderr
    assert "shared reporter entered" in result.stdout
    assert path.name in result.stderr
    assert "atexit ran" not in result.stdout


@pytest.mark.parametrize("name", TUTORIAL_NAMES)
def test_tutorial_startup_failure_survives_shutdown_hook(name: str) -> None:
    """Tutorial setup errors retain their traceback and nonzero status without loading a simulator."""
    path = PROJECT_ROOT / "examples" / "tutorial" / f"{name}.py"
    result = _run_child(FAILURE_HARNESS, str(path), "startup", *_script_arguments(path))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "RuntimeError: injected tutorial startup failure" in result.stderr
    assert "Traceback (most recent call last)" in result.stderr
    assert path.name in result.stderr
    assert "atexit ran" not in result.stdout


@pytest.mark.parametrize("relative_path", CLEANUP_ENTRYPOINTS)
@pytest.mark.parametrize(
    "outcome",
    [
        "failure",
        "success",
        "interrupt",
        "exit_none",
        "exit_zero",
        "exit_failure",
        "exit_message",
        "exit_float_zero",
    ],
)
def test_loop_cleanup_preserves_failure_status(
    relative_path: str, outcome: str
) -> None:
    """Failures bypass terminating cleanup while normal returns, interrupts, and successful exits close."""
    result = _run_child(CLEANUP_HARNESS, str(PROJECT_ROOT / relative_path), outcome)

    if outcome == "failure":
        assert result.returncode == 1, result.stdout + result.stderr
        assert "RuntimeError: injected loop failure" in result.stderr
        assert "explicit close ran" not in result.stdout
    elif outcome in {"exit_failure", "exit_message", "exit_float_zero"}:
        assert result.returncode == (7 if outcome == "exit_failure" else 1), (
            result.stdout + result.stderr
        )
        assert "explicit close ran" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "explicit close ran" in result.stdout
        assert result.stderr == ""


def test_isaacgym_viewer_close_removes_temporary_assets(tmp_path: Path) -> None:
    """The actual viewer-close render path reaches main's cleanup before a successful process exit."""
    result = _run_child(VIEWER_CLOSE_HARNESS, str(PROJECT_ROOT), str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "temporary assets created" in result.stdout
    assert "simulator close removed assets" in result.stdout
    assert not list(tmp_path.iterdir())


def test_successful_main_retains_normal_shutdown() -> None:
    """A successful main returns to its caller and still runs interpreter shutdown hooks."""
    result = _run_child("""
import atexit
from protomotions.launch.startup import run_reporting_failures

atexit.register(print, "atexit ran")
run_reporting_failures(lambda: print("main succeeded"))
print("caller resumed")
""")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "main succeeded",
        "caller resumed",
        "atexit ran",
    ]
    assert result.stderr == ""


@pytest.mark.parametrize("name", EXAMPLE_DEFINITIONS)
def test_example_definitions_remain_importable_and_picklable(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Example functions and classes keep stable module identities for imports and pickle references."""
    path = PROJECT_ROOT / "examples" / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(path), *_script_arguments(path)])
    monkeypatch.setattr(
        simulator_imports, "import_simulator_before_torch", lambda simulator: None
    )
    monkeypatch.setitem(sys.modules, "viser", ModuleType("viser"))
    module_name = f"examples.{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    for definition_name in EXAMPLE_DEFINITIONS[name]:
        definition = getattr(module, definition_name)
        assert definition.__module__ == module_name
        assert definition.__qualname__ == definition_name
        assert (
            getattr(importlib.import_module(module_name), definition_name) is definition
        )
        assert pickle.loads(pickle.dumps(definition)) is definition
