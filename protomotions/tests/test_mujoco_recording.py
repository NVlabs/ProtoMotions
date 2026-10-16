# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MuJoCo viewer-camera video capture."""

from contextlib import nullcontext
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pytest

from protomotions.simulator.mujoco.simulator import MujocoSimulator


@pytest.mark.parametrize("viewport_size", [None, (0, 0)])
def test_viewport_capture_uses_live_viewer_camera(monkeypatch, tmp_path, viewport_size):
    calls = []

    class FakeRenderer:
        def __init__(self, model, *, height, width, max_geom):
            assert (height, width) == (480, 640)
            self.scene = mujoco.MjvScene(model, maxgeom=max_geom)

        def update_scene(self, data, *, camera=-1, scene_option=None):
            calls.append((data, camera, scene_option))
            # A debug overlay must append to, not replace, the model scene.
            self.scene.geoms[0].pos[:] = [5, 6, 7]
            self.scene.ngeom = 1

        def render(self):
            assert self.scene.ngeom == 2
            np.testing.assert_array_equal(self.scene.geoms[0].pos, [5, 6, 7])
            np.testing.assert_array_equal(self.scene.geoms[1].pos, [1, 2, 3])
            assert self.scene.geoms[1].label == "goal"
            np.testing.assert_array_equal(self.scene.flags, debug_scene.flags)
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(plt, "imsave", lambda path, pixels: None)

    camera = object()
    option = object()
    data = object()
    simulator = object.__new__(MujocoSimulator)
    simulator.model = mujoco.MjModel.from_xml_string("<mujoco/>")
    simulator.data = data
    simulator._user_recording_frame = 0
    debug_scene = mujoco.MjvScene(simulator.model, maxgeom=4)
    mujoco.mjv_initGeom(
        debug_scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.ones(3),
        np.array([1.0, 2.0, 3.0]),
        np.eye(3).ravel(),
        np.ones(4),
    )
    debug_scene.geoms[0].label = "goal"
    debug_scene.ngeom = 1
    debug_scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    simulator.viewer = SimpleNamespace(
        cam=camera, opt=option, user_scn=debug_scene, lock=lambda: nullcontext()
    )
    if viewport_size is not None:
        simulator.viewer.viewport = mujoco.MjrRect(0, 0, *viewport_size)
    simulator._viewer_initialized = True

    simulator._write_viewport_to_file(str(tmp_path / "frame.png"))

    assert calls == [(data, camera, option)]


def test_viewport_capture_falls_back_without_viewer(monkeypatch, tmp_path):
    calls = []

    class FakeRenderer:
        def __init__(self, model, *, height, width, max_geom):
            pass

        def update_scene(self, data, *, camera=-1):
            calls.append((data, camera))

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(plt, "imsave", lambda path, pixels: None)

    data = object()
    simulator = object.__new__(MujocoSimulator)
    simulator.model = mujoco.MjModel.from_xml_string("<mujoco/>")
    simulator.data = data
    simulator._user_recording_frame = 0
    simulator.viewer = None
    simulator._viewer_initialized = False

    simulator._write_viewport_to_file(str(tmp_path / "frame.png"))

    assert calls == [(data, -1)]


def test_viewport_capture_renders_markers_and_arrows(tmp_path):
    """Exercise the real offscreen renderer, including hiding debug geometry."""
    simulator = object.__new__(MujocoSimulator)
    simulator.model = mujoco.MjModel.from_xml_string("<mujoco/>")
    simulator.data = mujoco.MjData(simulator.model)
    mujoco.mj_forward(simulator.model, simulator.data)
    scene = mujoco.MjvScene(simulator.model, maxgeom=4)
    simulator._marker_overflow_warned = False
    simulator._add_mujoco_marker_geom(
        scene,
        "sphere",
        "regular",
        np.array([0.0, -0.2, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 1.0]),
    )
    simulator._add_mujoco_marker_geom(
        scene,
        "arrow",
        "small",
        np.array([0.0, 0.2, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 1.0]),
    )
    camera = mujoco.MjvCamera()
    camera.distance = 2
    camera.azimuth = 90
    camera.elevation = -30
    camera.lookat[:] = [0.3, 0, 0]
    simulator.viewer = SimpleNamespace(
        cam=camera, opt=mujoco.MjvOption(), user_scn=scene, lock=lambda: nullcontext()
    )
    simulator._viewer_initialized = True
    simulator._user_recording_frame = 0
    simulator.viewer.viewport = mujoco.MjrRect(0, 0, 1281, 721)
    path = tmp_path / "visible.png"
    simulator._write_viewport_to_file(str(path))
    pixels = plt.imread(path)
    assert pixels.shape[:2] == (720, 1280)
    assert np.count_nonzero(pixels[..., 0] > pixels[..., 1] + 0.1) > 20
    assert np.count_nonzero(pixels[..., 1] > pixels[..., 0] + 0.1) > 20
    scene.ngeom = 0
    simulator._user_recording_frame = 1
    simulator.viewer.viewport = mujoco.MjrRect(0, 0, 640, 480)
    simulator._write_viewport_to_file(str(tmp_path / "hidden.png"))
    hidden = plt.imread(tmp_path / "hidden.png")
    assert hidden.shape == pixels.shape
    assert np.count_nonzero(hidden[..., :3]) == 0
    simulator._user_recording_frame = 0
    simulator._write_viewport_to_file(str(tmp_path / "next_clip.png"))
    assert plt.imread(tmp_path / "next_clip.png").shape[:2] == (480, 640)


def test_viewport_capture_closes_renderer_on_error(monkeypatch, tmp_path):
    closed = []

    class FakeRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def update_scene(self, data):
            raise RuntimeError("render failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    simulator = object.__new__(MujocoSimulator)
    simulator.model = mujoco.MjModel.from_xml_string("<mujoco/>")
    simulator.data = object()
    simulator.viewer = None
    simulator._viewer_initialized = False
    simulator._user_recording_frame = 0
    with pytest.raises(RuntimeError, match="render failure"):
        simulator._write_viewport_to_file(str(tmp_path / "frame.png"))
    assert closed == [True]
