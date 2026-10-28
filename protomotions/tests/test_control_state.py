# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation snapshots restore every control's registry in place; the
registry refuses bare per-env tensors, a changed shape and a changed key set."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.base_env.episode_tensors import EpisodeTensors
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.external_kinematic_control import (
    ExternalKinematicControl,
    ExternalKinematicControlConfig,
)
from protomotions.envs.control.gpc_sft_reference_target_control import (
    GPCSFTReferenceTargetControl,
    GPCSFTReferenceTargetControlConfig,
)
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControlConfig,
    MaskedMimicStyleConfig,
    MaskedMimicVelocityCommandConfig,
)
from protomotions.envs.control.path_follower_control import (
    PathFollowerControl,
    PathFollowerControlConfig,
)
from protomotions.envs.control.steering_control import (
    SteeringControl,
    SteeringControlConfig,
)
from protomotions.envs.control.target_control import TargetControl, TargetControlConfig
from protomotions.tests.support.task_inputs import (
    make_reset_state,
    make_robot,
    make_robot_state,
    make_task_inputs,
)
from protomotions.tests.test_structured_target_sampling import make_control


@pytest.mark.parametrize(
    "control_type, config_type",
    [
        (SteeringControl, SteeringControlConfig),
        (GPCSFTReferenceTargetControl, GPCSFTReferenceTargetControlConfig),
        (TargetControl, TargetControlConfig),
    ],
)
def test_registry_tensors_restore_in_place(control_type, config_type):
    config = config_type()
    if control_type is TargetControl:
        config.enable_fall_termination = False
    control = control_type(config, make_task_inputs(2))
    tensors = dict(control.state.tensors())
    assert tensors, "a control with no per-env state"
    for index, tensor in enumerate(tensors.values(), start=1):
        tensor.fill_(index)
    snapshot = control.save_state()
    for tensor in tensors.values():
        tensor.zero_()
    control.load_state(snapshot)
    for name, tensor in tensors.items():
        assert dict(control.state.tensors())[name] is tensor
        torch.testing.assert_close(tensor, snapshot[name])
        tensor.zero_()
        # The snapshot owns its clones: zeroing the live tensor leaves it.
        assert snapshot[name].count_nonzero() == snapshot[name].numel()


def test_target_history_cursor_is_restored_with_history():
    control = TargetControl(
        TargetControlConfig(enable_fall_termination=False), make_task_inputs(2)
    )
    control.history_index.fill_(3)
    control.root_pos_history.fill_(2)
    snapshot = control.save_state()
    control.history_index.fill_(5)
    control.root_pos_history.zero_()
    control.load_state(snapshot)
    assert int(control.history_index) == 3
    torch.testing.assert_close(control.root_pos_history, snapshot["root_pos_history"])


@pytest.mark.parametrize("all_channels", [False, True])
def test_masked_mimic_restores_masks_times_and_plans(all_channels):
    config = MaskedMimicControlConfig(
        local_target_condition_probability=1.0 if all_channels else 0.0,
        anchor_pose_submasking=all_channels,
        style=MaskedMimicStyleConfig() if all_channels else None,
        velocity_command=MaskedMimicVelocityCommandConfig() if all_channels else None,
    )
    control = make_control(config)
    num_envs = control.inputs.num_envs
    state = make_robot_state(torch.zeros(num_envs, 3, 3))
    control.reset_rows(torch.arange(num_envs), state)
    snapshot = control.save_state()
    # Constants are not snapshotted; the streams are nested child registries.
    assert "conditionable_body_ids" not in snapshot
    assert set(snapshot["global_stream"]) == {
        "times",
        "offsets",
        "frame_masks",
        "body_masks",
    }
    assert ("local_stream" in snapshot) is all_channels
    tensors = dict(control.state.tensors())
    assert ("local_stream.times" in tensors) is all_channels
    assert ("style_frame_masks" in tensors) is all_channels

    control.reset_rows(torch.arange(num_envs), state)
    for tensor in tensors.values():
        tensor.zero_()

    control.load_state(snapshot)

    assert control.planned.all()
    for name, tensor in tensors.items():
        assert dict(control.state.tensors())[name] is tensor
        saved = snapshot
        for part in name.split("."):
            saved = saved[part]
        torch.testing.assert_close(tensor, saved)
        assert tensor.data_ptr() != saved.data_ptr()
    # Restoration can be repeated: neither reset nor restore consumes the snapshot.
    control.global_stream.times.zero_()
    control.load_state(snapshot)
    torch.testing.assert_close(
        control.global_stream.times, snapshot["global_stream"]["times"]
    )


def test_path_snapshot_preserves_flat_vertex_alias():
    inputs = make_task_inputs(
        2, robot=make_robot(("head",), head_body_name="head"), max_episode_length=100
    )
    control = PathFollowerControl(PathFollowerControlConfig(), inputs)
    generator = control.path_generator
    generator.verts.fill_(3)
    snapshot = control.save_state()
    flat = generator.verts_flat
    generator.verts.zero_()
    control.load_state(snapshot)
    assert generator.verts_flat is flat
    torch.testing.assert_close(flat, snapshot["path"]["verts"].flatten(0, 1))
    generator.verts.zero_()
    verts = snapshot["path"]["verts"]
    assert verts.count_nonzero() == verts.numel()


@pytest.mark.parametrize("pending", [False, True])
def test_external_pose_and_initialization_are_owned_snapshots(pending):
    inputs = make_task_inputs(2)
    control = ExternalKinematicControl(ExternalKinematicControlConfig(), inputs)
    if pending:
        control.reset_rows(torch.arange(2), make_robot_state(torch.zeros(2, 3)))
        control.set_next_pose(make_reset_state())
    snapshot = control.save_state()
    if pending:
        control._next_pose.root_pos.fill_(9)
    control._next_pose = None
    control.initialized.fill_(not pending)
    control.load_state(snapshot)
    assert bool(control.initialized.all()) is pending
    if pending:
        assert control._next_pose is not snapshot["next_pose"]
        torch.testing.assert_close(control._next_pose.root_pos, torch.zeros(2, 3))
    else:
        assert control._next_pose is None


# ---------------------------------------------------------------------------
# The registry and the attribute guard
# ---------------------------------------------------------------------------


def test_registry_add_shared_constant_child_and_row_reset():
    state = EpisodeTensors(3, torch.device("cpu"))
    counter = state.add("counter", (), torch.long, 7)
    direction = state.add("direction", (2,), torch.float, torch.tensor([1.0, 0.0]))
    cursor = state.shared("cursor", torch.zeros((), dtype=torch.long))
    table = state.constant("table", torch.arange(3))
    child = state.child("sub")
    inner = child.add("inner", (), torch.bool, True)

    assert torch.equal(counter, torch.full((3,), 7))
    assert torch.equal(direction, torch.tensor([[1.0, 0.0]] * 3))
    counter[:] = 0
    direction[:] = 5.0
    inner[:] = False
    state.reset_rows(torch.tensor([1]))
    assert counter.tolist() == [0, 7, 0]
    assert direction[1].tolist() == [1.0, 0.0] and direction[0].tolist() == [5.0, 5.0]
    assert inner.tolist() == [False, True, False]

    assert state.owns(counter) and state.owns(cursor)
    assert state.owns(table) and state.owns(inner)
    assert not state.owns(torch.zeros(3))
    assert dict(state.tensors()).keys() == {
        "counter",
        "direction",
        "cursor",
        "sub.inner",
    }
    snapshot = state.save_state()
    assert set(snapshot) == {"counter", "direction", "cursor", "sub"}
    with pytest.raises(ValueError, match="already exists"):
        state.add("counter", (), torch.long, 0)
    with pytest.raises(ValueError, match="already exists"):
        state.shared("table", torch.zeros(1))
    # A constant may be re-registered under its name (a cache keyed on config).
    state.constant("table", torch.arange(4))
    with pytest.raises(ValueError, match="does not match the slot shape"):
        state.add("bad", (3,), torch.float, torch.zeros(2))
    with pytest.raises(ValueError, match="shape changed"):
        state.load_state({**snapshot, "counter": torch.zeros(1, dtype=torch.long)})
    with pytest.raises(ValueError, match="do not match the registry"):
        state.load_state({k: v for k, v in snapshot.items() if k != "cursor"})


class _Config(ControlComponentConfig):
    pass


class _Bare(ControlComponent):
    def __init__(self, config, inputs):
        super().__init__(config, inputs)
        self.command = torch.zeros(inputs.num_envs)

    def advance(self, state):
        return {}

    def publish(self, env_ids):
        return {}


def test_a_bare_per_env_tensor_attribute_is_refused():
    with pytest.raises(TypeError, match="bare tensor with a num_envs first dimension"):
        _Bare(_Config(), make_task_inputs(4))


class _Registered(ControlComponent):
    def __init__(self, config, inputs):
        super().__init__(config, inputs)
        self.command = self.state.add("command", (), torch.float, 0.0)
        self.table = self.state.constant("table", torch.arange(inputs.num_envs))
        self.scalar = torch.tensor(3.0)  # no num_envs dimension: not per-env state
        self.names = SimpleNamespace(count=inputs.num_envs)

    def advance(self, state):
        return {}

    def publish(self, env_ids):
        return {}


def test_registered_tensors_and_scalars_pass_the_guard():
    control = _Registered(_Config(), make_task_inputs(4))
    control.command[:] = 2.0
    control.reset_rows(torch.tensor([0, 2]), None)
    assert control.command.tolist() == [0.0, 2.0, 0.0, 2.0]
    snapshot = control.save_state()
    assert set(snapshot) == {"command"}
    assert torch.equal(snapshot["command"], control.command)
    assert control.resample(torch.arange(4), None) is None
