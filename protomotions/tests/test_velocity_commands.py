# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent velocity command geometry and reset/context contracts."""

import math
import pickle
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.context import published_vocabulary_errors
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
    MaskedMimicStyleConfig,
    MaskedMimicVelocityCommandConfig,
)
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.cast import to_float
from protomotions.envs.obs.masked_mimic_velocity_command import (
    compute_masked_mimic_velocity_command_obs,
)
from protomotions.tests.support.task_inputs import motion_fakes
from protomotions.tests.test_structured_target_sampling import make_control, make_inputs


def command_control(probability=1.0):
    inputs = make_inputs(
        placement_offset=lambda pos, env_ids: torch.tensor(
            [100.0, 200.0, 3.0]
        ).expand_as(pos)
    )
    motion = motion_fakes(inputs)
    motion.motion_lib.queries = []

    def state(ids, times):
        motion.motion_lib.queries.append(times.clone())
        pos = torch.zeros(len(ids), 3, 3)
        pos[:, 0, :2] = times[:, None] * torch.tensor([2.0, -1.0])
        rot = torch.zeros(len(ids), 3, 4)
        rot[..., 3] = 1
        return SimpleNamespace(
            rigid_body_pos=pos,
            rigid_body_rot=rot,
            rigid_body_vel=torch.zeros_like(pos),
            rigid_body_ang_vel=torch.zeros_like(pos),
            dof_pos=torch.zeros(len(ids), 2),
            dof_vel=torch.zeros(len(ids), 2),
        )

    motion.motion_lib.get_motion_state = state
    return MaskedMimicControl(
        MaskedMimicControlConfig(
            velocity_command=MaskedMimicVelocityCommandConfig(probability=probability),
            local_target_condition_probability=1,
            style=MaskedMimicStyleConfig(),
        ),
        inputs,
    )


@pytest.mark.parametrize("horizon", [0, -1, float("nan"), float("inf")])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        MaskedMimicVelocityCommandConfig(horizon_seconds=horizon)


@pytest.mark.parametrize("probability", [-1, 2, float("nan"), float("inf")])
def test_invalid_probability(probability):
    with pytest.raises(ValueError):
        MaskedMimicVelocityCommandConfig(probability=probability)


def test_config_pickle_and_live_validation():
    cfg = MaskedMimicControlConfig(velocity_command=MaskedMimicVelocityCommandConfig())
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    cfg.velocity_command.horizon_seconds = float("nan")
    with pytest.raises(ValueError):
        cfg.validate()


def test_heading_frame_and_hidden_zero():
    rotation = torch.tensor([[0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]]).repeat(2, 1)
    velocity = torch.tensor([[2.0, 0], [float("nan"), float("inf")]])
    mask = torch.tensor([[True], [False]])
    obs = compute_masked_mimic_velocity_command_obs(rotation, velocity, mask)
    torch.testing.assert_close(obs[0], torch.tensor([0.0, -2.0]), atol=1e-6, rtol=0)
    assert torch.equal(obs[1], torch.zeros(2))


def test_rolling_horizon_subset_and_clip_end():
    control = command_control()
    motion = motion_fakes(control.inputs)
    motion.motion_manager.motion_times[:] = torch.tensor([9.75, 20, 31])
    # The nonzero spawn/terrain stub must not leak into the XY velocity.
    velocity = control._compute_average_velocity()
    torch.testing.assert_close(velocity, torch.tensor([[2.0, -1], [0, 0], [0, 0]]))
    torch.testing.assert_close(
        motion.motion_lib.queries[-1][3:], torch.tensor([10.0, 20, 30])
    )
    ids = torch.tensor([2, 0])
    torch.testing.assert_close(control._compute_average_velocity(ids), velocity[ids])
    motion.motion_manager.motion_times[0] = 1
    control.config.velocity_command.horizon_seconds = 2
    torch.testing.assert_close(
        control._compute_average_velocity()[0], torch.tensor([2.0, -1])
    )
    assert motion.motion_lib.queries[-1][3] == 3


def test_short_remaining_duration_uses_simulation_step_floor():
    control = command_control()
    motion_fakes(control.inputs).motion_manager.motion_times[0] = 9.99
    velocity = control._compute_average_velocity()
    remaining = 10 - motion_fakes(control.inputs).motion_manager.motion_times[0]
    torch.testing.assert_close(
        velocity[0], torch.tensor([2.0, -1]) * remaining / control.inputs.dt
    )


def test_visibility_reset_and_stream_independence():
    control = command_control(0.5)
    torch.manual_seed(14)
    control.reset_rows(torch.arange(3), None)
    masks = control.velocity_command_mask.clone()
    other_streams = [
        control.global_stream.body_masks, control.local_stream.body_masks,
        control.style_condition_masks, control.style_frame_masks,
    ]
    before = [x.clone() for x in other_streams]
    control.config.velocity_command.horizon_seconds = 1.5
    control.velocity_command_mask[:] = ~masks
    control._compute_average_velocity()
    for original, value in zip(before, other_streams):
        assert torch.equal(original, value)
    control.config.velocity_command.probability = 0
    control.reset_rows(torch.tensor([1]), None)
    assert not control.velocity_command_mask[1].any()
    assert torch.equal(control.velocity_command_mask[[0, 2]], ~masks[[0, 2]])
    control.reset_rows(torch.tensor([], dtype=torch.long), None)
    control.advance(None)
    assert not control.velocity_command_mask[1].any()


def test_context_and_observation_subset(monkeypatch):
    control = command_control()
    control.reset_rows(torch.arange(3), None)
    monkeypatch.setattr(control, "_query_style_samples", lambda ids: {})
    monkeypatch.setattr(MimicControl, "publish", lambda self, env_ids: {})
    ids = torch.tensor([2, 0])
    ctx = control.publish(ids)
    # Every published masked_mimic name is declared, and every required one is here.
    assert published_vocabulary_errors(ctx) == []
    torch.testing.assert_close(
        ctx["masked_mimic.average_velocity"], torch.tensor([[2.0, -1]]).repeat(2, 1)
    )
    assert ctx["masked_mimic.velocity_command_mask"].shape == (2, 1)
    # The recipe's two lines: the command kernel on its own defaults (it reads
    # the anchor, not the root) and a float cast of the published mask.
    command = MdpComponent.from_kernel(compute_masked_mimic_velocity_command_obs)
    assert command.dynamic_vars["anchor_rot"].path == "current.anchor_rot"
    assert command.dynamic_vars["average_velocity"].path == (
        "masked_mimic.average_velocity"
    )
    mask = MdpComponent.from_kernel(
        to_float, x=EnvContext.masked_mimic.velocity_command_mask
    )
    assert mask.compute(ctx).dtype == torch.float32
    assert mask.compute(ctx).shape == (2, 1)


def test_disabled_command_preserves_global_only_default():
    control = make_control()
    assert control.velocity_command_mask is None
