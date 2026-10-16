# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The noisy CurrentStateView must carry the dedicated root/anchor noise.

apply_observation_noise noises anchor_rot, root_local_ang_vel and
anchor_local_ang_vel on their own (anchor_rot_noise, root_ang_vel_noise,
anchor_ang_vel_noise). CurrentStateView used to re-derive those three from the
whole-body tensors, so actor observations such as projected gravity and the
pelvis gyro saw clean values even with that noise configured.
"""

import pytest
import torch

from protomotions.envs.context_views import CurrentStateView
from protomotions.envs.obs import compute_humanoid_reduced_coords_observations
from protomotions.envs.obs.observation_noise import apply_observation_noise
from protomotions.simulator.base_simulator.config import RobotNoiseConfig
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)

NUM_ENVS, NUM_BODIES, NUM_DOFS = 64, 3, 4


def _robot_state() -> RobotState:
    rot = torch.nn.functional.normalize(torch.randn(NUM_ENVS, NUM_BODIES, 4), dim=-1)
    return RobotState(
        rigid_body_pos=torch.randn(NUM_ENVS, NUM_BODIES, 3),
        rigid_body_rot=rot,
        rigid_body_vel=torch.randn(NUM_ENVS, NUM_BODIES, 3),
        rigid_body_ang_vel=torch.randn(NUM_ENVS, NUM_BODIES, 3),
        dof_pos=torch.randn(NUM_ENVS, NUM_DOFS),
        dof_vel=torch.randn(NUM_ENVS, NUM_DOFS),
        state_conversion=StateConversion.COMMON,
    )


def _reduced_coords_obs(view: CurrentStateView) -> torch.Tensor:
    return compute_humanoid_reduced_coords_observations(
        dof_pos=view.dof_pos,
        dof_vel=view.dof_vel,
        anchor_rot=view.anchor_rot,
        root_local_ang_vel=view.root_local_ang_vel,
        w_last=True,
    )


@pytest.mark.parametrize("anchor_idx", [0, 2])
def test_noisy_view_uses_dedicated_root_and_anchor_noise(anchor_idx):
    torch.manual_seed(0)
    state = _robot_state()
    # Only the dedicated channels are noised, so anything re-derived from the
    # whole-body tensors would come out exactly clean.
    noise_cfg = RobotNoiseConfig(
        anchor_rot_noise=0.05, root_ang_vel_noise=0.2, anchor_ang_vel_noise=0.2
    )
    noisy = apply_observation_noise(noise_cfg, state, anchor_idx, torch.zeros(NUM_ENVS))

    clean_view = CurrentStateView(state, anchor_idx)
    noisy_view = CurrentStateView(noisy, anchor_idx)

    for name in ("anchor_rot", "root_local_ang_vel", "anchor_local_ang_vel"):
        assert getattr(noisy_view, name) is getattr(noisy, name), name
        assert not torch.allclose(
            getattr(noisy_view, name), getattr(clean_view, name)
        ), name

    # Projected gravity and the pelvis gyro are what the actor actually reads.
    assert not torch.allclose(
        _reduced_coords_obs(noisy_view), _reduced_coords_obs(clean_view)
    )


def test_anchor_only_noise_leaves_pelvis_gyro_bit_identical():
    """The G1 BeyondMimic-style configs noise only dofs and the anchor.

    For them the fix must change anchor_rot and nothing else the actor reads:
    root_ang_vel_noise is 0 there, so the pelvis gyro must stay exactly clean.
    """
    torch.manual_seed(0)
    state = _robot_state()
    noise_cfg = RobotNoiseConfig(
        dof_pos_noise=0.01,
        dof_vel_noise=0.5,
        anchor_ang_vel_noise=0.2,
        anchor_rot_noise=0.05,
    )
    noisy = apply_observation_noise(noise_cfg, state, 2, torch.zeros(NUM_ENVS))

    clean_view = CurrentStateView(state, anchor_idx=2)
    noisy_view = CurrentStateView(noisy, anchor_idx=2)

    assert torch.equal(noisy_view.root_local_ang_vel, clean_view.root_local_ang_vel)
    assert torch.equal(noisy_view.root_rot, clean_view.root_rot)
    assert not torch.allclose(noisy_view.anchor_rot, clean_view.anchor_rot)


def test_noisy_view_follows_env_subset_selection():
    torch.manual_seed(0)
    state = _robot_state()
    noise_cfg = RobotNoiseConfig(anchor_rot_noise=0.05, root_ang_vel_noise=0.2)
    noisy = apply_observation_noise(noise_cfg, state, 1, torch.zeros(NUM_ENVS))
    env_ids = torch.tensor([5, 2, 40])

    subset_view = CurrentStateView(noisy.select(env_ids), anchor_idx=1)

    assert torch.equal(subset_view.anchor_rot, noisy.anchor_rot[env_ids])
    assert torch.equal(
        subset_view.root_local_ang_vel, noisy.root_local_ang_vel[env_ids]
    )


def test_clean_view_still_derives_root_and_anchor_from_state():
    torch.manual_seed(0)
    state = _robot_state()

    view = CurrentStateView(state, anchor_idx=2)

    assert torch.equal(view.anchor_rot, state.rigid_body_rot[:, 2, :])
    # With no noise configured the NoisyObservations fields are the clean
    # values, so the noisy view must match the clean one exactly.
    no_noise = apply_observation_noise(None, state, 2, torch.zeros(NUM_ENVS))
    noisy_view = CurrentStateView(no_noise, anchor_idx=2)
    for name in ("anchor_rot", "root_local_ang_vel", "anchor_local_ang_vel"):
        assert torch.equal(getattr(noisy_view, name), getattr(view, name)), name
