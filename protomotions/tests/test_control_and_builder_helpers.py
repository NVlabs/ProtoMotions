# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for control lifecycle and component builder helpers."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.manager import ControlManager
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
import protomotions.envs.control.manager as control_manager_module
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils import component_builder
from protomotions.utils.hydra_replacement import get_class, instantiate


class _RecordingControl(ControlComponent):
    instances = []

    def __init__(self, config, env):
        super().__init__(config, env)
        self.events = []
        _RecordingControl.instances.append(self)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.events.append(("reset", env_ids.clone()))

    def step(self):
        self.events.append(("step", None))

    def check_resets_and_terminations(self):
        return self.config.reset_mask.clone(), self.config.terminate_mask.clone()

    def populate_context(self, ctx):
        ctx.populated.append(self.config.name)

    def create_visualization_markers(self, headless: bool):
        return {self.config.name: f"marker-{headless}"}

    def get_markers_state(self):
        return {self.config.name: f"state-{self.config.name}"}


def _identity_quat(*shape):
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def test_control_component_defaults_return_empty_buffers_and_markers():
    env = SimpleNamespace(device=torch.device("cpu"), num_envs=2)
    component = _RecordingControl(
        SimpleNamespace(
            name="default",
            reset_mask=torch.tensor([False, False]),
            terminate_mask=torch.tensor([False, False]),
        ),
        env,
    )

    reset_buf, terminate_buf = ControlComponent.check_resets_and_terminations(component)

    assert isinstance(ControlComponentConfig(), ControlComponentConfig)
    assert torch.equal(reset_buf, torch.tensor([False, False]))
    assert torch.equal(terminate_buf, torch.tensor([False, False]))
    assert ControlComponent.step(component) is None
    assert ControlComponent.populate_context(component, SimpleNamespace()) is None
    assert ControlComponent.create_visualization_markers(component, headless=True) == {}
    assert ControlComponent.get_markers_state(component) == {}


def test_control_manager_orchestrates_component_lifecycle(monkeypatch):
    _RecordingControl.instances = []
    monkeypatch.setattr(
        control_manager_module,
        "get_class",
        lambda target: _RecordingControl,
    )
    env = SimpleNamespace(device=torch.device("cpu"), num_envs=3)
    manager = ControlManager(
        {
            "first": SimpleNamespace(
                _target_="unused.First",
                name="first",
                reset_mask=torch.tensor([True, False, False]),
                terminate_mask=torch.tensor([False, True, False]),
            ),
            "second": SimpleNamespace(
                _target_="unused.Second",
                name="second",
                reset_mask=torch.tensor([False, False, True]),
                terminate_mask=torch.tensor([False, False, False]),
            ),
        },
        env,
    )

    manager.step()
    manager.reset(torch.tensor([0, 2]))
    reset_buf, terminate_buf = manager.check_resets_and_terminations()
    ctx = SimpleNamespace(populated=[])
    manager.populate_context(ctx)

    assert set(manager.components) == {"first", "second"}
    assert torch.equal(reset_buf, torch.tensor([True, False, True]))
    assert torch.equal(terminate_buf, torch.tensor([False, True, False]))
    assert ctx.populated == ["first", "second"]
    assert manager.create_visualization_markers(headless=True) == {}
    assert manager.create_visualization_markers(headless=False) == {
        "first": "marker-False",
        "second": "marker-False",
    }
    assert manager.get_markers_state() == {
        "first": "state-first",
        "second": "state-second",
    }
    assert _RecordingControl.instances[0].events[0] == ("step", None)
    assert torch.equal(_RecordingControl.instances[1].events[1][1], torch.tensor([0, 2]))


def test_mimic_control_populates_context_with_clamped_future_reference_data():
    class _MotionLib:
        def __init__(self):
            self.state_queries = []
            self.length_queries = []

        def get_motion_length(self, motion_ids):
            self.length_queries.append(motion_ids.clone())
            lengths = {3: 1.0, 5: 0.4}
            return torch.tensor([lengths[int(motion_id)] for motion_id in motion_ids])

        def get_motion_state(self, motion_ids, motion_times):
            self.state_queries.append((motion_ids.clone(), motion_times.clone()))
            n = len(motion_ids)
            pos = torch.zeros(n, 2, 3)
            pos[:, 0, 0] = motion_ids.float()
            pos[:, 0, 1] = motion_times
            pos[:, 1, 0] = motion_ids.float() + 100.0
            pos[:, 1, 1] = motion_times + 100.0
            return RobotState(
                state_conversion=None,
                rigid_body_pos=pos,
                rigid_body_rot=_identity_quat(n, 2),
                rigid_body_vel=pos + 1000.0,
                rigid_body_ang_vel=pos + 2000.0,
                dof_pos=motion_times.unsqueeze(-1),
                dof_vel=motion_ids.float().unsqueeze(-1),
            )

    offsets = []

    def terrain_offset(ref_pos):
        offsets.append(ref_pos.clone())
        return torch.tensor(
            [[[10.0, 0.0, 1.0]], [[20.0, 0.0, 2.0]]],
            dtype=ref_pos.dtype,
        )

    motion_lib = _MotionLib()
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        motion_manager=SimpleNamespace(
            motion_ids=torch.tensor([3, 5]),
            motion_times=torch.tensor([0.8, 0.25]),
        ),
        motion_lib=motion_lib,
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=terrain_offset,
        robot_config=SimpleNamespace(
            anchor_body_index=1,
            kinematic_info=SimpleNamespace(
                hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])}
            ),
        ),
    )
    control = MimicControl(MimicControlConfig(future_steps=[1, 3]), env)
    ctx = SimpleNamespace()

    control.populate_context(ctx)

    current_ids, current_times = motion_lib.state_queries[0]
    future_ids, future_times = motion_lib.state_queries[1]
    assert torch.equal(current_ids, torch.tensor([3, 5]))
    assert torch.allclose(current_times, torch.tensor([0.8, 0.25]))
    assert torch.equal(future_ids, torch.tensor([3, 3, 5, 5]))
    assert torch.allclose(future_times, torch.tensor([0.9, 1.0, 0.35, 0.4]))
    assert torch.equal(motion_lib.length_queries[0], torch.tensor([3, 5]))

    assert len(offsets) == 2
    assert offsets[0].shape == (2, 2, 3)
    assert offsets[1].shape == (2, 2, 3)
    assert ctx.mimic.future_pos.shape == (2, 2, 2, 3)
    assert ctx.mimic.future_rot.shape == (2, 2, 2, 4)
    assert ctx.mimic.future_dof_pos.shape == (2, 2, 1)
    assert ctx.mimic.ref_lr.shape == (2, 1, 4)
    assert ctx.mimic.anchor_idx == 1
    assert torch.allclose(
        ctx.mimic.ref_state.rigid_body_pos[:, 0],
        torch.tensor([[13.0, 0.8, 1.0], [25.0, 0.25, 2.0]]),
    )
    assert torch.allclose(
        ctx.mimic.future_root_pos[:, :, :2],
        torch.tensor([[[13.0, 0.9], [13.0, 1.0]], [[25.0, 0.35], [25.0, 0.4]]]),
    )


def test_mimic_control_int_future_steps_and_reset_bootstrap_modes():
    class _MotionLib:
        def __init__(self):
            self.state_queries = []

        def get_motion_length(self, motion_ids):
            return torch.ones_like(motion_ids, dtype=torch.float32)

        def get_motion_state(self, motion_ids, motion_times):
            self.state_queries.append((motion_ids.clone(), motion_times.clone()))
            n = len(motion_ids)
            return RobotState(
                state_conversion=None,
                rigid_body_pos=torch.zeros(n, 1, 3),
                rigid_body_rot=_identity_quat(n, 1),
                rigid_body_vel=torch.zeros(n, 1, 3),
                rigid_body_ang_vel=torch.zeros(n, 1, 3),
                dof_pos=motion_times.unsqueeze(-1),
                dof_vel=torch.zeros(n, 1),
            )

    done_tracks = torch.tensor([True, False])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.2,
        motion_manager=SimpleNamespace(
            motion_ids=torch.tensor([0, 1]),
            motion_times=torch.tensor([0.1, 0.4]),
            get_done_tracks=lambda: done_tracks,
        ),
        motion_lib=_MotionLib(),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda ref: torch.zeros(
            ref.shape[0], 1, 3
        ),
        robot_config=SimpleNamespace(
            anchor_body_index=0,
            kinematic_info=SimpleNamespace(
                hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])}
            ),
        ),
    )
    control = MimicControl(MimicControlConfig(future_steps=2), env)
    ctx = SimpleNamespace()

    assert control.step() is None
    control.populate_context(ctx)
    reset_buf, terminate_buf = control.check_resets_and_terminations()

    assert torch.equal(reset_buf, done_tracks)
    assert torch.equal(terminate_buf, torch.tensor([False, False]))
    assert torch.allclose(
        env.motion_lib.state_queries[1][1],
        torch.tensor([0.3, 0.5, 0.6, 0.8]),
    )
    assert ctx.mimic.future_dof_pos.shape == (2, 2, 1)

    no_bootstrap = MimicControl(
        MimicControlConfig(future_steps=1, bootstrap_on_episode_end=False), env
    )
    _, terminate_buf = no_bootstrap.check_resets_and_terminations()
    assert torch.equal(terminate_buf, done_tracks)


def test_mimic_control_visualization_marker_config_and_state():
    class _MotionLib:
        def __init__(self):
            self.queries = []

        def get_motion_state(self, motion_ids, motion_times):
            self.queries.append((motion_ids.clone(), motion_times.clone()))
            return RobotState(
                state_conversion=None,
                rigid_body_pos=torch.tensor(
                    [
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    ]
                ),
            )

    motion_lib = _MotionLib()
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        simulator=SimpleNamespace(headless=False),
        motion_manager=SimpleNamespace(
            motion_ids=torch.tensor([2, 3]),
            motion_times=torch.tensor([0.25, 0.5]),
        ),
        motion_lib=motion_lib,
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda ref: torch.tensor(
            [[[0.0, 0.0, 1.0]], [[10.0, 0.0, 2.0]]]
        ),
        robot_config=SimpleNamespace(
            mimic_small_marker_bodies={"hand"},
            kinematic_info=SimpleNamespace(body_names=["pelvis", "hand"]),
        ),
    )
    control = MimicControl(MimicControlConfig(), env)

    assert control.create_visualization_markers(headless=True) == {}
    marker_cfg = control.create_visualization_markers(headless=False)[
        "body_markers_red"
    ]
    assert marker_cfg.type == "sphere"
    assert marker_cfg.color == (1.0, 0.0, 0.0)
    assert [marker.size for marker in marker_cfg.markers] == ["regular", "small"]

    markers = control.get_markers_state()

    assert torch.equal(motion_lib.queries[0][0], torch.tensor([2, 3]))
    assert torch.allclose(
        markers["body_markers_red"].translation,
        torch.tensor(
            [
                [[1.0, 2.0, 4.0], [4.0, 5.0, 7.0]],
                [[17.0, 8.0, 11.0], [20.0, 11.0, 14.0]],
            ]
        ),
    )
    assert torch.equal(
        markers["body_markers_red"].orientation,
        torch.zeros(2, 2, 4),
    )

    env.simulator.headless = True
    assert control.get_markers_state() == {}


def test_hydra_replacement_get_class_and_instantiate_variants():
    assert get_class("types.SimpleNamespace") is SimpleNamespace

    from_dict = instantiate(
        {"_target_": "types.SimpleNamespace", "a": 1, "b": 2},
        b=3,
    )
    from_object = instantiate(
        SimpleNamespace(_target_="types.SimpleNamespace", a=4, _private=99)
    )

    assert from_dict.a == 1
    assert from_dict.b == 3
    assert from_object.a == 4
    assert not hasattr(from_object, "_private")
    with pytest.raises(ValueError, match="Config must have"):
        instantiate(SimpleNamespace(a=1))


def test_component_builder_builds_simulator_and_orchestrates_all(monkeypatch):
    class _Simulator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(component_builder, "get_class", lambda target: _Simulator)
    simulator = component_builder.build_simulator_from_config(
        simulator_config=SimpleNamespace(_target_="unused.Simulator"),
        robot_config="robot",
        terrain="terrain",
        scene_lib="scene",
        device=torch.device("cpu"),
        extra_flag=True,
    )
    assert simulator.kwargs == {
        "config": simulator.kwargs["config"],
        "robot_config": "robot",
        "terrain": "terrain",
        "scene_lib": "scene",
        "device": torch.device("cpu"),
        "extra_flag": True,
    }
    assert component_builder.build_terrain_from_config(None, 4, torch.device("cpu")) is None

    calls = []
    sim_cfg = SimpleNamespace(num_envs=4)
    motion_cfg = SimpleNamespace(motion_file=None)
    monkeypatch.setattr(
        component_builder,
        "build_terrain_from_config",
        lambda cfg, num_envs, device: calls.append(("terrain", cfg, num_envs, device))
        or "terrain",
    )
    monkeypatch.setattr(
        component_builder,
        "build_scene_lib_from_config",
        lambda cfg, num_envs, device, terrain, scene_weights=None: calls.append(
            ("scene", cfg, num_envs, device, terrain, scene_weights)
        )
        or "scene",
    )
    monkeypatch.setattr(
        component_builder,
        "build_motion_lib_from_config",
        lambda cfg, device: calls.append(("motion", cfg, device)) or "motion",
    )
    monkeypatch.setattr(
        component_builder,
        "build_simulator_from_config",
        lambda sim_cfg, robot_cfg, terrain, scene_lib, device, **extra: calls.append(
            ("simulator", sim_cfg, robot_cfg, terrain, scene_lib, device, extra)
        )
        or "simulator",
    )

    result = component_builder.build_all_components(
        terrain_config="terrain_cfg",
        scene_lib_config="scene_cfg",
        motion_lib_config=motion_cfg,
        simulator_config=sim_cfg,
        robot_config="robot_cfg",
        device=torch.device("cpu"),
        custom="value",
    )

    assert result == {
        "terrain": "terrain",
        "scene_lib": "scene",
        "motion_lib": "motion",
        "simulator": "simulator",
    }
    assert calls == [
        ("terrain", "terrain_cfg", 4, torch.device("cpu")),
        ("scene", "scene_cfg", 4, torch.device("cpu"), "terrain", None),
        ("motion", motion_cfg, torch.device("cpu")),
        (
            "simulator",
            sim_cfg,
            "robot_cfg",
            "terrain",
            "scene",
            torch.device("cpu"),
            {"custom": "value"},
        ),
    ]


def test_component_builder_constructs_dependency_classes_and_scene_weights(monkeypatch):
    built = []

    class _Terrain:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            built.append(("terrain", kwargs))

    class _SceneLib:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            built.append(("scene", kwargs))

    class _MotionLib:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            built.append(("motion", kwargs))

    class _BaseEnv:
        @staticmethod
        def apply_motion_weights_to_scene_weights(save_dir, motion_file, device):
            built.append(("weights", save_dir, motion_file, device))
            return [0.1, 0.9]

    terrain_module = types.ModuleType("protomotions.components.terrains.terrain")
    terrain_module.Terrain = _Terrain
    scene_module = types.ModuleType("protomotions.components.scene_lib")
    scene_module.SceneLib = _SceneLib
    motion_module = types.ModuleType("protomotions.components.motion_lib")
    motion_module.MotionLib = _MotionLib
    env_module = types.ModuleType("protomotions.envs.base_env.env")
    env_module.BaseEnv = _BaseEnv
    monkeypatch.setitem(
        sys.modules, "protomotions.components.terrains.terrain", terrain_module
    )
    monkeypatch.setitem(sys.modules, "protomotions.components.scene_lib", scene_module)
    monkeypatch.setitem(sys.modules, "protomotions.components.motion_lib", motion_module)
    monkeypatch.setitem(sys.modules, "protomotions.envs.base_env.env", env_module)

    class _Simulator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(component_builder, "get_class", lambda target: _Simulator)
    device = torch.device("cpu")
    terrain_cfg = SimpleNamespace(name="terrain")
    scene_cfg = SimpleNamespace(scene_file=None, inline_scenes=None)
    motion_cfg = SimpleNamespace(motion_file="motions.yaml")
    simulator_cfg = SimpleNamespace(num_envs=2, _target_="unused.Simulator")

    result = component_builder.build_all_components(
        terrain_config=terrain_cfg,
        scene_lib_config=scene_cfg,
        motion_lib_config=motion_cfg,
        simulator_config=simulator_cfg,
        robot_config="robot",
        device=device,
        save_dir="/tmp/checkpoint",
        custom="value",
    )

    assert isinstance(result["terrain"], _Terrain)
    assert isinstance(result["scene_lib"], _SceneLib)
    assert isinstance(result["motion_lib"], _MotionLib)
    assert isinstance(result["simulator"], _Simulator)
    assert built[0] == ("terrain", {"config": terrain_cfg, "num_envs": 2, "device": device})
    assert built[1] == ("weights", "/tmp/checkpoint", "motions.yaml", device)
    assert built[2][0] == "scene"
    assert built[2][1]["scene_weights"] == [0.1, 0.9]
    assert built[2][1]["scenes"] is None
    assert built[3] == ("motion", {"config": motion_cfg, "device": device})


def test_motion_manager_ignores_exclusion_file_that_disappears_after_probe(
    tmp_path, monkeypatch, capsys
):
    import pathlib

    exclusion_file = tmp_path / "exclude.txt"
    exclusion_file.write_text("1\n")
    real_exists = pathlib.Path.exists

    def exists_after_probe(path):
        if path == exclusion_file:
            return False
        return real_exists(path)

    class _MotionLib:
        motion_weights = torch.ones(3)
        motion_lengths = torch.ones(3)
        motion_file = "motions.yaml"

        def num_motions(self):
            return 3

    monkeypatch.setattr(pathlib.Path, "exists", exists_after_probe)

    manager = MotionManager(
        MotionManagerConfig(
            init_start_prob=0.0,
            exclude_motions_file=str(exclusion_file),
        ),
        num_envs=2,
        env_dt=0.1,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
    )

    assert manager.excluded_motion_ids is None
    assert "exclude_motions_file not found" in capsys.readouterr().out
