# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for control lifecycle and component builder helpers."""

import pickle
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.context import (
    control_producer,
    merge_with_producers,
    published_vocabulary_errors,
)
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.manager import ControlManager, TeleportQueue
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
import protomotions.envs.control.manager as control_manager_module
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils import component_builder
from protomotions.utils.hydra_replacement import get_class, instantiate
from protomotions.components.motion_lib import MotionFileSwitchMode
from protomotions.tests.support.fake_reference import fake_reference
from protomotions.tests.support.task_inputs import (
    make_robot,
    make_robot_state,
    make_task_inputs,
)


class _RecordingControl(ControlComponent):
    instances = []

    def __init__(self, config, inputs):
        super().__init__(config, inputs)
        self.events = []
        self.command = self.state.add("command", (), torch.float, 0.0)
        _RecordingControl.instances.append(self)

    def reset_rows(self, env_ids, state):
        super().reset_rows(env_ids, state)
        self.events.append(("reset_rows", env_ids.clone()))

    def advance(self, state):
        self.events.append(("advance", state))
        return dict(self.config.logging)

    def publish(self, env_ids):
        return {f"{self.config.name}.seen": env_ids}

    def create_visualization_markers(self, headless: bool):
        return {self.config.name: f"marker-{headless}"}

    def get_markers_state(self, state):
        return {self.config.name: f"state-{self.config.name}-{state}"}


def _identity_quat(*shape):
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _recording_config(name, **logging):
    return SimpleNamespace(_target_=f"unused.{name}", name=name, logging=logging)


def test_control_component_defaults_return_empty_checks_and_markers():
    inputs = make_task_inputs(2)
    component = _RecordingControl(_recording_config("default"), inputs)

    assert isinstance(ControlComponentConfig(), ControlComponentConfig)
    assert ControlComponent.terminations(component.config, inputs.robot) == {}
    assert ControlComponent.resample(component, torch.arange(2), None) is None
    assert ControlComponent.create_visualization_markers(component, headless=True) == {}
    assert ControlComponent.get_markers_state(component, None) == {}
    assert ControlComponent.provides(component.config, "name")  # a truthy field
    assert not ControlComponent.provides(component.config, "missing")

    component.command[:] = torch.tensor([1.0, 2.0])
    snapshot = component.save_state()
    component.command.zero_()
    component.load_state(snapshot)
    assert torch.equal(component.command, torch.tensor([1.0, 2.0]))
    # The default row reset puts the registry's rows back to their reset value.
    ControlComponent.reset_rows(component, torch.tensor([1]), None)
    assert component.command.tolist() == [1.0, 0.0]
    # A tensor whose shape changed is an error, not a silent rebind that would
    # break anything else holding the old tensor.
    with pytest.raises(ValueError, match="shape changed"):
        component.load_state({"command": torch.zeros(1)})


def test_control_manager_orchestrates_component_lifecycle(monkeypatch):
    _RecordingControl.instances = []
    monkeypatch.setattr(
        control_manager_module,
        "get_class",
        lambda target: _RecordingControl,
    )
    inputs = make_task_inputs(3)
    teleports = TeleportQueue()
    manager = ControlManager(
        {
            "first": _recording_config("first", a=torch.ones(3)),
            "second": _recording_config("second", b=torch.zeros(3)),
        },
        inputs,
        teleports,
    )
    manager.check_holds_no_handles()

    state = make_robot_state(torch.zeros(3, 3))
    logging = manager.advance(state)
    manager.reset_rows(torch.tensor([0, 2]), state)
    env_ids = torch.tensor([1])
    # The manager has no merged publish of its own: ``BaseEnv._publish`` walks
    # ``components`` and merges each dict under the component's producer name.
    assert not hasattr(manager, "publish")

    def publish_parts(rows):
        return {
            control_producer(name): component.publish(rows)
            for name, component in manager.components.items()
        }

    assert set(manager.components) == {"first", "second"}
    # The components' logging dicts are merged; a key logged twice is refused.
    assert set(logging) == {"a", "b"}
    manager.components["second"].config.logging = {"a": torch.zeros(3)}
    with pytest.raises(ValueError, match="'first' and 'second' both log 'a'"):
        manager.advance(state)
    manager.components["second"].config.logging = {}
    # No component owns a check here.
    assert manager.terminations() == {}
    # Every component's names, in component order, with the rows passed
    # through, each recorded under its own producer.
    merged, producers = merge_with_producers(publish_parts(env_ids))
    assert list(merged) == ["first.seen", "second.seen"]
    assert merged["first.seen"] is env_ids
    assert producers == {"first.seen": "control/first", "second.seen": "control/second"}
    # Two components publishing one name is an error naming both producers.
    manager.components["second"].config.name = "first"
    with pytest.raises(ValueError, match="'first.seen'.*control/first.*control/second"):
        merge_with_producers(publish_parts(None))
    manager.components["second"].config.name = "second"
    assert manager.create_visualization_markers(headless=True) == {}
    assert manager.create_visualization_markers(headless=False) == {
        "first": "marker-False",
        "second": "marker-False",
    }
    assert manager.get_markers_state("frame") == {
        "first": "state-first-frame",
        "second": "state-second-frame",
    }
    assert _RecordingControl.instances[0].events[0] == ("advance", state)
    assert torch.equal(
        _RecordingControl.instances[1].events[1][1], torch.tensor([0, 2])
    )

    manager.components["first"].command[:] = 3.0
    manager.components["second"].command[:] = 4.0
    snapshot = manager.save_state()
    manager.components["first"].command.zero_()
    manager.components["second"].command.zero_()
    manager.load_state(snapshot)
    assert manager.components["first"].command[0].item() == 3.0
    assert manager.components["second"].command[0].item() == 4.0
    # Every configured component must be restored; an incomplete snapshot is a bug.
    with pytest.raises(KeyError, match="second"):
        manager.load_state({"first": snapshot["first"]})


def test_control_manager_keys_every_check_by_control_and_refuses_handles(monkeypatch):
    class _Checked(_RecordingControl):
        @classmethod
        def terminations(cls, config, robot_config):
            return {"done": f"rule-{config.name}-{robot_config.anchor_body_index}"}

    class _Holder(_RecordingControl):
        def __init__(self, config, inputs):
            super().__init__(config, inputs)
            self.manager = MotionManager.__new__(MotionManager)

    monkeypatch.setattr(control_manager_module, "get_class", lambda target: _Checked)
    inputs = make_task_inputs(2)
    manager = ControlManager(
        {"mimic": _recording_config("mimic")}, inputs, TeleportQueue()
    )
    assert manager.terminations() == {"mimic/done": "rule-mimic-0"}

    monkeypatch.setattr(control_manager_module, "get_class", lambda target: _Holder)
    manager = ControlManager({"bad": _recording_config("bad")}, inputs, TeleportQueue())
    with pytest.raises(TypeError, match="_Holder.manager holds a MotionManager"):
        manager.check_holds_no_handles()


def test_teleport_queue_applies_requests_in_order_then_clears():
    queue = TeleportQueue()
    simulator = SimpleNamespace(calls=[])
    simulator.reset_envs = lambda robot, objects, env_ids: simulator.calls.append(
        (robot, objects, env_ids)
    )
    queue.request(torch.tensor([0]), "pose-a", None)
    queue.request(torch.tensor([1]), "pose-b", "objects-b")
    queue.apply(simulator)
    assert simulator.calls == [
        ("pose-a", None, torch.tensor([0])),
        ("pose-b", "objects-b", torch.tensor([1])),
    ] or [c[:2] for c in simulator.calls] == [("pose-a", None), ("pose-b", "objects-b")]
    queue.apply(simulator)
    assert len(simulator.calls) == 2


def _mimic_inputs(
    motion_manager, motion_lib, *, dt, robot=None, placement_offset=None, headless=True
):
    motion = SimpleNamespace(
        num_envs=len(motion_manager.motion_ids),
        device=torch.device("cpu"),
        motion_manager=motion_manager,
        motion_lib=motion_lib,
    )
    return make_task_inputs(
        motion.num_envs,
        robot=robot,
        reference=fake_reference(motion, placement_offset=placement_offset),
        dt=dt,
        headless=headless,
    )


def test_mimic_control_normalises_the_int_a_cli_override_leaves_behind():
    """The override funnel writes after ``__post_init__``, so a saved config may
    hold ``future_steps=3``; the control's time offsets still come from the
    normalised tuple."""
    config = MimicControlConfig()
    config.future_steps = 3
    inputs = _mimic_inputs(
        SimpleNamespace(
            motion_ids=torch.tensor([0, 0]),
            motion_times=torch.zeros(2),
            get_done_tracks=lambda env_ids=None: torch.tensor([False, False]),
        ),
        SimpleNamespace(),
        dt=0.1,
    )
    control = MimicControl(config, inputs)
    assert torch.allclose(control.future_time_offsets, torch.tensor([0.1, 0.2, 0.3]))


@pytest.mark.parametrize("smoothing_seconds", [0.0, 0.2])
def test_mimic_control_populates_context_with_clamped_future_reference_data(
    smoothing_seconds,
):
    class _MotionLib:
        def __init__(self):
            self.state_queries = []
            self.length_queries = []
            self.position_queries = []

        def get_motion_positions(self, motion_ids, motion_times):
            self.position_queries.append((motion_ids.clone(), motion_times.clone()))
            pos = torch.zeros(len(motion_ids), 2, 3)
            pos[:, 1, 2] = motion_ids.float()
            return pos

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

    def terrain_offset(ref_pos, env_ids=None):
        offsets.append(ref_pos.clone())
        values = torch.tensor(
            [[[10.0, 0.0, 1.0]], [[20.0, 0.0, 2.0]]],
            dtype=ref_pos.dtype,
        )
        return values if env_ids is None else values[env_ids]

    motion_lib = _MotionLib()
    inputs = _mimic_inputs(
        SimpleNamespace(
            motion_ids=torch.tensor([3, 5]),
            motion_times=torch.tensor([0.8, 0.25]),
            get_done_tracks=lambda env_ids=None: torch.tensor([False, False]),
        ),
        motion_lib,
        dt=0.1,
        robot=make_robot(
            ("root", "anchor"),
            anchor_body_index=1,
            hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
        ),
        placement_offset=terrain_offset,
    )
    control = MimicControl(
        MimicControlConfig(
            future_steps=[1, 3], reference_height_smoothing_seconds=smoothing_seconds
        ),
        inputs,
    )
    published = control.publish(None)

    assert len(motion_lib.state_queries) == 1
    combined_ids, combined_times = motion_lib.state_queries[0]
    assert torch.equal(combined_ids, torch.tensor([3, 5, 3, 3, 5, 5]))
    assert torch.allclose(
        combined_times, torch.tensor([0.8, 0.25, 0.9, 1.0, 0.35, 0.4])
    )
    assert torch.equal(motion_lib.length_queries[0], torch.tensor([3, 5]))

    assert len(offsets) == (3 if smoothing_seconds else 2)
    if smoothing_seconds:
        assert len(motion_lib.position_queries) == 1
        torch.testing.assert_close(
            published["mimic.smoothed_ref_anchor_height"], torch.tensor([4.0, 7.0])
        )
    else:
        assert not motion_lib.position_queries
        # Only computed with smoothing: absent, not None.
        assert "mimic.smoothed_ref_anchor_height" not in published
    assert offsets[0].shape == (2, 2, 3)
    assert offsets[1].shape == (2, 2, 3)
    assert published["mimic.future_pos"].shape == (2, 2, 2, 3)
    assert published["mimic.future_rot"].shape == (2, 2, 2, 4)
    assert published["mimic.future_dof_pos"].shape == (2, 2, 1)
    assert published["mimic.ref_lr"].shape == (2, 1, 4)
    assert published["mimic.anchor_idx"] == 1
    assert torch.allclose(
        published["mimic.ref_state.rigid_body_pos"][:, 0],
        torch.tensor([[13.0, 0.8, 1.0], [25.0, 0.25, 2.0]]),
    )
    assert torch.allclose(
        published["mimic.future_root_pos"][:, :, :2],
        torch.tensor([[[13.0, 0.9], [13.0, 1.0]], [[25.0, 0.35], [25.0, 0.4]]]),
    )
    # The derived anchor names are slices of the same tensors.
    assert torch.equal(
        published["mimic.ref_anchor_pos"],
        published["mimic.ref_state.rigid_body_pos"][:, 1],
    )
    assert torch.equal(
        published["mimic.future_anchor_rot"], published["mimic.future_rot"][:, :, 1]
    )
    assert torch.equal(published["mimic.clip_done"], torch.tensor([False, False]))
    # Everything published is declared under mimic.*, nothing required is missing.
    assert published_vocabulary_errors(published) == []


def test_mimic_control_future_time_offsets_are_a_registered_constant():
    inputs = make_task_inputs(2, dt=0.1)
    control = MimicControl(MimicControlConfig(future_steps=[1, 3]), inputs)

    assert torch.allclose(control.future_time_offsets, torch.tensor([0.1, 0.3]))
    # A constant: neither per-env state nor part of the snapshot.
    assert control.state.owns(control.future_time_offsets)
    assert control.save_state() == {}
    assert torch.allclose(
        MimicControl(MimicControlConfig(future_steps=2), inputs).future_time_offsets,
        torch.tensor([0.1, 0.2]),
    )


def test_mimic_control_int_future_steps_and_clip_done_modes():
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
    motion_lib = _MotionLib()
    inputs = _mimic_inputs(
        SimpleNamespace(
            motion_ids=torch.tensor([0, 1]),
            motion_times=torch.tensor([0.1, 0.4]),
            get_done_tracks=lambda env_ids=None: done_tracks,
        ),
        motion_lib,
        dt=0.2,
        robot=make_robot(
            ("root",), hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])}
        ),
    )
    config = MimicControlConfig(future_steps=2)
    control = MimicControl(config, inputs)

    assert control.advance(None) == {}
    published = control.publish(None)

    assert torch.equal(published["mimic.clip_done"], done_tracks)
    assert torch.allclose(
        motion_lib.state_queries[0][1],
        torch.tensor([0.1, 0.4, 0.3, 0.5, 0.6, 0.8]),
    )
    assert published["mimic.future_dof_pos"].shape == (2, 2, 1)

    # The one check: a truncation under bootstrapping, a failure without it.
    bootstrap = MimicControl.terminations(config, inputs.robot)["clip_done"]
    assert bootstrap.truncation is True and bootstrap.threshold is None
    assert bootstrap.component.dynamic_vars["clip_done"].path == "mimic.clip_done"
    no_bootstrap = MimicControl.terminations(
        MimicControlConfig(future_steps=1, bootstrap_on_episode_end=False), inputs.robot
    )["clip_done"]
    assert no_bootstrap.truncation is False


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
    inputs = _mimic_inputs(
        SimpleNamespace(
            motion_ids=torch.tensor([2, 3]),
            motion_times=torch.tensor([0.25, 0.5]),
        ),
        motion_lib,
        dt=0.1,
        robot=make_robot(("pelvis", "hand"), mimic_small_marker_bodies={"hand"}),
        placement_offset=lambda ref, env_ids: torch.tensor(
            [[[0.0, 0.0, 1.0]], [[10.0, 0.0, 2.0]]]
        ),
        headless=False,
    )
    control = MimicControl(MimicControlConfig(), inputs)

    assert control.create_visualization_markers(headless=True) == {}
    marker_cfg = control.create_visualization_markers(headless=False)[
        "body_markers_red"
    ]
    assert marker_cfg.type == "sphere"
    assert marker_cfg.color == (1.0, 0.0, 0.0)
    assert [marker.size for marker in marker_cfg.markers] == ["regular", "small"]

    markers = control.get_markers_state(None)

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

    headless = MimicControl(
        MimicControlConfig(),
        _mimic_inputs(
            SimpleNamespace(
                motion_ids=torch.tensor([2, 3]), motion_times=torch.tensor([0.25, 0.5])
            ),
            motion_lib,
            dt=0.1,
            robot=make_robot(("pelvis", "hand")),
        ),
    )
    assert headless.get_markers_state(None) == {}


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
    assert (
        component_builder.build_terrain_from_config(None, 4, torch.device("cpu"))
        is None
    )

    calls = []
    sim_cfg = SimpleNamespace(num_envs=4)
    motion_cfg = SimpleNamespace(
        motion_file=None,
        motion_file_switch_mode=MotionFileSwitchMode.FIXED,
    )
    scene_cfg = SimpleNamespace(scene_file=None)
    monkeypatch.setattr(
        component_builder,
        "build_terrain_from_config",
        lambda cfg, num_envs, device: calls.append(("terrain", cfg, num_envs, device))
        or "terrain",
    )
    scene_lib_result = SimpleNamespace(pairing=SimpleNamespace(is_paired=False))

    def _build_scene(
        cfg, num_envs, device, terrain, scene_weights=None, motion_weights=None
    ):
        calls.append(("scene", cfg, num_envs, device, terrain, scene_weights))
        return scene_lib_result

    monkeypatch.setattr(component_builder, "build_scene_lib_from_config", _build_scene)
    monkeypatch.setattr(
        component_builder,
        "build_motion_lib_from_config",
        lambda cfg, device, shard_cycle=0: calls.append(
            ("motion", cfg, device, shard_cycle)
        )
        or "motion",
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
        scene_lib_config=scene_cfg,
        motion_lib_config=motion_cfg,
        simulator_config=sim_cfg,
        robot_config="robot_cfg",
        device=torch.device("cpu"),
        custom="value",
        motion_shard_cycle=5,
    )

    assert result == {
        "terrain": "terrain",
        "scene_lib": scene_lib_result,
        "motion_lib": "motion",
        "simulator": "simulator",
    }
    assert calls == [
        ("terrain", "terrain_cfg", 4, torch.device("cpu")),
        ("motion", motion_cfg, torch.device("cpu"), 5),
        ("scene", scene_cfg, 4, torch.device("cpu"), "terrain", None),
        (
            "simulator",
            sim_cfg,
            "robot_cfg",
            "terrain",
            scene_lib_result,
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
            self.pairing = SimpleNamespace(is_paired=False)
            built.append(("scene", kwargs))

    class _MotionLib:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.motion_file = kwargs["config"].motion_file
            built.append(("motion", kwargs))

    def apply_motion_weights_to_scene_weights(save_dir, motion_file, device):
        built.append(("weights", save_dir, motion_file, device))
        return [0.1, 0.9]

    terrain_module = types.ModuleType("protomotions.components.terrains.terrain")
    terrain_module.Terrain = _Terrain
    scene_module = types.ModuleType("protomotions.components.scene_lib")
    scene_module.SceneLib = _SceneLib
    motion_module = types.ModuleType("protomotions.components.motion_lib")
    motion_module.MotionLib = _MotionLib
    monkeypatch.setitem(
        sys.modules, "protomotions.components.terrains.terrain", terrain_module
    )
    monkeypatch.setitem(sys.modules, "protomotions.components.scene_lib", scene_module)
    monkeypatch.setitem(
        sys.modules, "protomotions.components.motion_lib", motion_module
    )
    monkeypatch.setattr(
        component_builder,
        "apply_motion_weights_to_scene_weights",
        apply_motion_weights_to_scene_weights,
    )

    class _Simulator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(component_builder, "get_class", lambda target: _Simulator)
    device = torch.device("cpu")
    terrain_cfg = SimpleNamespace(name="terrain")
    scene_cfg = SimpleNamespace(scene_file=None, inline_scenes=None)
    motion_cfg = SimpleNamespace(
        motion_file="motions.yaml",
        motion_file_switch_mode=MotionFileSwitchMode.FIXED,
    )
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
    assert built[0] == (
        "terrain",
        {"config": terrain_cfg, "num_envs": 2, "device": device},
    )
    assert built[1] == (
        "motion",
        {"config": motion_cfg, "device": device, "shard_cycle": 0},
    )
    assert built[2] == ("weights", "/tmp/checkpoint", "motions.yaml", device)
    assert built[3][0] == "scene"
    assert built[3][1]["motion_weights"] == [0.1, 0.9]
    assert built[3][1]["scenes"] is None


def test_build_env_from_configs_writes_the_friction_conversion_back(monkeypatch):
    """The one path from a bundle to an env converts the terrain friction to
    the simulator's combine mode and writes the converted configs back into
    the bundle, so a bundle saved afterwards describes what the env was built
    from; the components and the env take the converted configs. (The launch
    tests fake the conversion away, so only this test sees a dropped write-back.)
    """
    from protomotions.components.terrains.config import (
        CombineMode,
        TerrainConfig,
        TerrainSimConfig,
    )
    from protomotions.simulator.base_simulator.config import (
        DomainRandomizationConfig,
        FrictionDomainRandomizationConfig,
        SimParams,
        SimulatorConfig,
    )
    from protomotions.utils.experiment_configs import ExperimentConfigs

    built = {"targets": []}

    def build_all_components(**kwargs):
        built["components"] = kwargs
        return {
            "terrain": "terrain",
            "scene_lib": "scene_lib",
            "motion_lib": "motion_lib",
            "simulator": "simulator",
        }

    class _Env:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def get_class(target):
        built["targets"].append(target)
        return _Env

    monkeypatch.setattr(component_builder, "build_all_components", build_all_components)
    monkeypatch.setattr(component_builder, "get_class", get_class)

    terrain = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1.0, combine_mode=CombineMode.AVERAGE
        )
    )
    simulator = SimulatorConfig(
        _target_="protomotions.simulator.newton.simulator.NewtonSimulator",
        w_last=True,
        headless=True,
        num_envs=2,
        sim=SimParams(),
        experiment_name="unit",
        domain_randomization=DomainRandomizationConfig(
            friction=FrictionDomainRandomizationConfig(body_indices=[0])
        ),
    )
    env_config = SimpleNamespace(_target_="env.Target", save_dir="")
    configs = ExperimentConfigs(
        robot="robot",
        simulator=simulator,
        terrain=terrain,
        scene_lib="scenes",
        motion_lib="motions",
        env=env_config,
        agent=None,
    )
    device = torch.device("cpu")

    env = component_builder.build_env_from_configs(
        configs, device, motion_shard_cycle=3, simulation_app="app"
    )

    # Written back: the bundle records Newton's MAX mode and the converted
    # friction range, not the AVERAGE form the recipe produced.
    assert configs.terrain is not terrain
    assert configs.terrain.sim_config.combine_mode is CombineMode.MAX
    assert configs.simulator.domain_randomization.friction.static_friction_range[
        0
    ] == pytest.approx(0.75)
    # The components and the env were built from the converted configs.
    assert built["components"] == {
        "terrain_config": configs.terrain,
        "scene_lib_config": "scenes",
        "motion_lib_config": "motions",
        "simulator_config": configs.simulator,
        "robot_config": "robot",
        "device": device,
        "save_dir": "",
        "motion_shard_cycle": 3,
        "simulation_app": "app",
    }
    assert built["targets"] == ["env.Target"]
    assert env.kwargs == {
        "config": env_config,
        "robot_config": "robot",
        "device": device,
        "terrain": "terrain",
        "scene_lib": "scene_lib",
        "motion_lib": "motion_lib",
        "simulator": "simulator",
    }
    # A bundle saved after the build (resume, inference) is already in the
    # simulator's mode, so building it again converts nothing more.
    converted = (configs.terrain, configs.simulator)
    component_builder.build_env_from_configs(configs, device)
    assert configs.terrain is converted[0]
    assert configs.simulator is converted[1]


class _ShardBuilderScene:
    def __init__(self, humanoid_motion_ids=None):
        self.humanoid_motion_ids = humanoid_motion_ids

    @property
    def pairing(self):
        ids = self.humanoid_motion_ids or []
        return SimpleNamespace(is_paired=any(i >= 0 for i in ids))


def _patch_shard_builder(monkeypatch, motion_lib, scene_lib, captured):
    monkeypatch.setattr(
        component_builder,
        "build_terrain_from_config",
        lambda config, num_envs, device: "terrain",
    )
    monkeypatch.setattr(
        component_builder,
        "build_motion_lib_from_config",
        lambda config, device, shard_cycle=0: captured.setdefault(
            "motion_cycles", []
        ).append(shard_cycle)
        or motion_lib,
    )

    def build_scene(config, *args, **kwargs):
        captured["scene_config"] = config
        return scene_lib

    monkeypatch.setattr(component_builder, "build_scene_lib_from_config", build_scene)
    monkeypatch.setattr(
        component_builder,
        "build_simulator_from_config",
        lambda *args, **kwargs: "simulator",
    )


def test_component_builder_uses_selected_shard_for_paired_scene_and_weights(
    tmp_path, monkeypatch
):
    scene_pattern = tmp_path / "scenes_slurmrank.pt"
    selected_scene = tmp_path / "scenes_003.pt"
    selected_scene.touch()
    motion_lib = SimpleNamespace(
        motion_file=str(tmp_path / "motions_003.pt"),
        motion_file_shard_index=3,
    )
    scene_lib = _ShardBuilderScene()
    captured = {}
    _patch_shard_builder(monkeypatch, motion_lib, scene_lib, captured)

    def apply_motion_weights_to_scene_weights(save_dir, motion_file, device):
        captured["weight_motion_file"] = motion_file
        return [1.0]

    monkeypatch.setattr(
        component_builder,
        "apply_motion_weights_to_scene_weights",
        apply_motion_weights_to_scene_weights,
    )
    scene_config = SimpleNamespace(scene_file=str(scene_pattern), inline_scenes=None)
    motion_config = SimpleNamespace(
        motion_file="motions_slurmrank.pt",
        motion_file_switch_mode=MotionFileSwitchMode.RESTART,
    )

    component_builder.build_all_components(
        terrain_config=None,
        scene_lib_config=scene_config,
        motion_lib_config=motion_config,
        simulator_config=SimpleNamespace(num_envs=1),
        robot_config=SimpleNamespace(),
        device=torch.device("cpu"),
        save_dir="checkpoint",
        motion_shard_cycle=7,
    )

    assert captured["motion_cycles"] == [7]
    assert captured["scene_config"] is not scene_config
    assert captured["scene_config"].scene_file == str(selected_scene)
    assert scene_config.scene_file == str(scene_pattern)
    assert captured["weight_motion_file"] == str(tmp_path / "motions_003.pt")


def test_component_builder_rejects_incompatible_paired_scene_switching(monkeypatch):
    motion_lib = SimpleNamespace(motion_file="motions_0.pt", motion_file_shard_index=0)
    captured = {}
    _patch_shard_builder(monkeypatch, motion_lib, _ShardBuilderScene([0]), captured)

    with pytest.raises(ValueError, match="matching sharded scene file"):
        component_builder.build_all_components(
            terrain_config=None,
            scene_lib_config=SimpleNamespace(
                scene_file="scenes.pt", inline_scenes=None
            ),
            motion_lib_config=SimpleNamespace(
                motion_file_switch_mode=MotionFileSwitchMode.RESTART
            ),
            simulator_config=SimpleNamespace(num_envs=1),
            robot_config=SimpleNamespace(),
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="motion-paired scenes"):
        component_builder.build_all_components(
            terrain_config=None,
            scene_lib_config=SimpleNamespace(scene_file=None, inline_scenes=None),
            motion_lib_config=SimpleNamespace(
                motion_file_switch_mode=MotionFileSwitchMode.LIVE
            ),
            simulator_config=SimpleNamespace(num_envs=1),
            robot_config=SimpleNamespace(),
            device=torch.device("cpu"),
        )


def test_component_builder_rejects_paired_sharded_motion_without_scene_shard_in_fixed(
    monkeypatch,
):
    """FIXED too: a sharded motion file with paired scenes needs a sharded scene file."""
    motion_lib = SimpleNamespace(motion_file="motions_0.pt", motion_file_shard_index=0)
    captured = {}
    _patch_shard_builder(monkeypatch, motion_lib, _ShardBuilderScene([0]), captured)

    with pytest.raises(ValueError, match="matching sharded scene file"):
        component_builder.build_all_components(
            terrain_config=None,
            scene_lib_config=SimpleNamespace(
                scene_file="scenes.pt", inline_scenes=None
            ),
            motion_lib_config=SimpleNamespace(
                motion_file_switch_mode=MotionFileSwitchMode.FIXED
            ),
            simulator_config=SimpleNamespace(num_envs=1),
            robot_config=SimpleNamespace(),
            device=torch.device("cpu"),
        )


def test_component_builder_builds_paired_scene_with_unsharded_motion_in_fixed(
    monkeypatch,
):
    """A paired scene library on a single unsharded motion file builds (SAMP/Metatron)."""
    motion_lib = SimpleNamespace(motion_file="motions.pt", motion_file_shard_index=None)
    captured = {}
    _patch_shard_builder(monkeypatch, motion_lib, _ShardBuilderScene([0, 1]), captured)

    # No refusal: the sharding rule is a no-op when the motion file is unsharded.
    component_builder.build_all_components(
        terrain_config=None,
        scene_lib_config=SimpleNamespace(scene_file="scenes.pt", inline_scenes=None),
        motion_lib_config=SimpleNamespace(
            motion_file_switch_mode=MotionFileSwitchMode.FIXED
        ),
        simulator_config=SimpleNamespace(num_envs=1),
        robot_config=SimpleNamespace(),
        device=torch.device("cpu"),
    )
    assert captured["scene_config"].scene_file == "scenes.pt"


def test_component_builder_rejects_live_sharded_scene_geometry(tmp_path, monkeypatch):
    captured = {}
    _patch_shard_builder(
        monkeypatch,
        SimpleNamespace(motion_file="motions_0.pt", motion_file_shard_index=0),
        _ShardBuilderScene(),
        captured,
    )

    with pytest.raises(ValueError, match="cannot replace simulator scene geometry"):
        component_builder.build_all_components(
            terrain_config=None,
            scene_lib_config=SimpleNamespace(
                scene_file=str(tmp_path / "scenes_slurmrank.pt"), inline_scenes=None
            ),
            motion_lib_config=SimpleNamespace(
                motion_file_switch_mode=MotionFileSwitchMode.LIVE
            ),
            simulator_config=SimpleNamespace(num_envs=1),
            robot_config=SimpleNamespace(),
            device=torch.device("cpu"),
        )


def test_component_builder_rejects_sharded_scene_without_selected_motion(
    tmp_path, monkeypatch
):
    captured = {}
    _patch_shard_builder(
        monkeypatch,
        SimpleNamespace(motion_file="motions.pt", motion_file_shard_index=None),
        _ShardBuilderScene(),
        captured,
    )

    with pytest.raises(ValueError, match="requires a sharded motion file"):
        component_builder.build_all_components(
            terrain_config=None,
            scene_lib_config=SimpleNamespace(
                scene_file=str(tmp_path / "scenes_slurmrank.pt"), inline_scenes=None
            ),
            motion_lib_config=SimpleNamespace(
                motion_file_switch_mode=MotionFileSwitchMode.FIXED
            ),
            simulator_config=SimpleNamespace(num_envs=1),
            robot_config=SimpleNamespace(),
            device=torch.device("cpu"),
        )


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
        generator=None,
    )

    assert manager.excluded_motion_ids is None
    assert "exclude_motions_file not found" in capsys.readouterr().out


def test_apply_motion_weights_to_scene_weights_loads_expected_checkpoint(tmp_path):
    weights_of = component_builder.apply_motion_weights_to_scene_weights
    device = torch.device("cpu")
    assert weights_of(None, "motion.npy", device) is None
    assert weights_of(str(tmp_path), None, device) is None
    assert weights_of(str(tmp_path), "/tmp/missing.npy", device) is None

    checkpoint = tmp_path / "env_clip.npy.ckpt"
    torch.save(
        {"motion_manager": {"motion_weights": torch.tensor([0.25, 0.75])}}, checkpoint
    )
    assert weights_of(str(tmp_path), "/tmp/clip.npy", device) == [0.25, 0.75]

    no_manager = tmp_path / "env_no_manager.npy.ckpt"
    torch.save({"other": True}, no_manager)
    assert weights_of(str(tmp_path), "/tmp/no_manager.npy", device) is None

    bad_checkpoint = tmp_path / "env_bad.npy.ckpt"
    bad_checkpoint.write_text("not a torch checkpoint")
    with pytest.raises(pickle.UnpicklingError):
        weights_of(str(tmp_path), "/tmp/bad.npy", device)


def test_scene_curriculum_does_not_hide_malformed_motion_manager(tmp_path):
    torch.save({"motion_manager": {}}, tmp_path / "env_clip.npy.ckpt")
    with pytest.raises(KeyError, match="motion_weights"):
        component_builder.apply_motion_weights_to_scene_weights(
            str(tmp_path), "/tmp/clip.npy", torch.device("cpu")
        )
