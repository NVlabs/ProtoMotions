# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from protomotions.simulator.base_simulator.user_interface import UserInterface
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.mujoco.keyboard import MujocoViewerKeyboard
from protomotions.simulator.mujoco.simulator import MujocoSimulator


class FakeViewer:
    """Model MuJoCo's change-detected, bidirectional render-flag sync."""

    def __init__(self):
        self.opt = SimpleNamespace(flags=np.zeros(4), geomgroup=np.ones(6))
        self.user_scn = SimpleNamespace(flags=np.zeros(3))
        self.native_flags = self.user_scn.flags.copy()
        self.previous_flags = self.user_scn.flags.copy()
        self.cam = SimpleNamespace()
        self.sync_count = 0

    def lock(self):
        return nullcontext()

    def sync(self):
        changed = self.user_scn.flags != self.previous_flags
        self.native_flags[changed] = self.user_scn.flags[changed]
        self.user_scn.flags[:] = self.native_flags
        self.previous_flags[:] = self.native_flags
        self.sync_count += 1


def simulator_shell():
    simulator = object.__new__(MujocoSimulator)
    simulator.user_interface = UserInterface()
    simulator._viewer_keyboard = MujocoViewerKeyboard()
    return simulator


def test_mujoco_rejects_body_mass_randomization_before_initialization():
    config = SimpleNamespace(domain_randomization=SimpleNamespace(body_mass=object()))
    with pytest.raises(NotImplementedError, match="body-mass domain randomization"):
        MujocoSimulator(config, None, None, torch.device("cpu"), None)


def test_viewer_launch_wires_native_callback_without_second_window(monkeypatch):
    import mujoco.viewer

    simulator = simulator_shell()
    simulator.model, simulator.data = object(), object()
    viewer = FakeViewer()
    launch = Mock(return_value=viewer)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr("atexit.register", lambda callback: None)
    simulator._init_viewer()
    assert launch.call_args.kwargs["key_callback"] == simulator._viewer_keyboard.on_key
    assert not hasattr(simulator, "_keyboard_window")
    calls = []
    simulator.user_interface.register_key(
        "R", owner="test", description="Reset", on_press=lambda: calls.append("reset")
    )
    launch.call_args.kwargs["key_callback"](ord("R"))
    assert calls == []  # Viewer thread never runs task callbacks.
    simulator._poll_keyboard_events()
    assert calls == ["reset"]


def test_repeated_presses_use_existing_ui_queue():
    simulator = simulator_shell()
    key = simulator.user_interface.register_key(
        "W", owner="test", description="Forward"
    )
    for _ in range(3):
        simulator._viewer_keyboard.on_key(ord("W"))
    simulator.user_interface.begin_step()
    simulator._poll_keyboard_events()
    for _ in range(3):
        assert key.consume()
        assert not key.consume()
        simulator.user_interface.begin_step()
    assert not key.consume()
    assert not key.down()  # Passive viewer does not report releases / held state.


def test_physics_step_dispatches_keys_before_control_and_integration(monkeypatch):
    import mujoco
    from protomotions.robot_configs.base import ControlType

    simulator = simulator_shell()
    simulator.config = SimpleNamespace(use_implicit_pd=True)
    simulator.control_type = ControlType.TORQUE
    simulator._max_action_latency_ms = 0.0
    simulator._step_count = 0
    simulator.decimation = 1
    simulator.model, simulator.data = object(), object()
    order = []
    simulator._apply_control = lambda: order.append("control")
    simulator._print_state_debug = lambda: None
    monkeypatch.setattr(mujoco, "mj_step", lambda model, data: order.append("physics"))
    simulator.user_interface.register_key(
        "J",
        owner="test",
        description="Projectile",
        on_press=lambda: order.append("key"),
    )
    simulator._viewer_keyboard.on_key(ord("J"))
    simulator._physics_step()
    assert order == ["key", "control", "physics"]


@pytest.mark.parametrize("keycode", [-1, 0, 32, 127, 256, 1000])
def test_non_task_keys_are_left_to_native_viewer(keycode):
    keyboard = MujocoViewerKeyboard()
    keyboard.on_key(keycode)
    assert list(keyboard.drain_presses()) == []


def test_native_mode_suppresses_tasks_and_period_always_switches_back():
    simulator = simulator_shell()
    keyboard = simulator._viewer_keyboard
    simulator.user_interface.register_key(
        ".", owner="mujoco", description="Native mode", on_press=keyboard.toggle_native
    )
    calls = []
    simulator.user_interface.register_key(
        "R", owner="test", description="Reset", on_press=lambda: calls.append("reset")
    )
    for char in "R.R.R":
        keyboard.on_key(ord(char))
    simulator._poll_keyboard_events()
    assert calls == ["reset", "reset"]
    assert not keyboard.native


def test_task_mode_restores_options_and_native_render_flags():
    viewer = FakeViewer()
    keyboard = MujocoViewerKeyboard()
    keyboard.capture(viewer)
    viewer.opt.flags[0] = 1  # N: island colouring
    viewer.opt.geomgroup[0] = 0  # digits: hide geometries
    viewer.native_flags[0] = 1  # W/R/L: private render scene changed
    keyboard.sync(viewer)
    np.testing.assert_array_equal(viewer.opt.flags, 0)
    np.testing.assert_array_equal(viewer.opt.geomgroup, 1)
    np.testing.assert_array_equal(viewer.native_flags, 0)
    assert viewer.sync_count == 2
    keyboard.sync(viewer)
    assert viewer.sync_count == 3  # No extra sync in the steady state.


def test_native_mode_leaves_visualization_changes_alone():
    viewer = FakeViewer()
    keyboard = MujocoViewerKeyboard()
    keyboard.capture(viewer)
    keyboard.toggle_native()
    viewer.opt.flags[0] = 1
    viewer.native_flags[0] = 1
    keyboard.sync(viewer)
    assert viewer.opt.flags[0] == viewer.native_flags[0] == 1
    keyboard.toggle_native()
    keyboard.sync(viewer)
    assert viewer.opt.flags[0] == viewer.native_flags[0] == 0


def test_render_uses_keyboard_sync_without_changing_mouse_orbit(monkeypatch):
    simulator = simulator_shell()
    simulator.headless = False
    simulator._viewer_initialized = True
    simulator.viewer = FakeViewer()
    simulator.viewer.is_running = lambda: True
    simulator.viewer.cam.distance = 4.0
    simulator.viewer.cam.azimuth = 75.0
    simulator.viewer.cam.elevation = -25.0
    monkeypatch.setattr(Simulator, "render", lambda self: None)
    simulator._viewer_keyboard.sync = Mock()
    simulator.render()
    simulator._viewer_keyboard.sync.assert_called_once_with(simulator.viewer)
    assert simulator.viewer.cam.distance == 4.0
    assert simulator.viewer.cam.azimuth == 75.0
    assert simulator.viewer.cam.elevation == -25.0


def test_headless_render_and_shutdown_do_not_need_a_viewer(monkeypatch):
    simulator = simulator_shell()
    simulator.headless = True
    simulator._viewer_initialized = False
    simulator.viewer = None
    monkeypatch.setattr(Simulator, "render", lambda self: None)
    simulator._viewer_keyboard.sync = Mock()
    simulator.render()
    simulator.shutdown()
    simulator.shutdown()
    simulator._viewer_keyboard.sync.assert_not_called()
