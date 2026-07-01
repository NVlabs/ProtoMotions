# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for observation noise containers and history buffers."""

from types import SimpleNamespace

import torch

from protomotions.envs.obs import observation_noise
from protomotions.envs.obs.observation_noise import (
    NoisyObservations,
    _add_euler_rotation_noise,
    _add_noise,
    _add_quaternion_noise,
    apply_observation_noise,
    apply_reset_noise,
)
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _make_noisy_observations(offset: float = 0.0) -> NoisyObservations:
    return NoisyObservations(
        rigid_body_pos=torch.ones(2, 2, 3) + offset,
        rigid_body_rot=_identity_quat(2, 2) + offset,
        rigid_body_vel=torch.ones(2, 2, 3) * 2 + offset,
        rigid_body_ang_vel=torch.ones(2, 2, 3) * 3 + offset,
        dof_pos=torch.ones(2, 3) * 4 + offset,
        dof_vel=torch.ones(2, 3) * 5 + offset,
        root_rot=_identity_quat(2) + offset,
        root_local_ang_vel=torch.ones(2, 3) * 6 + offset,
        anchor_rot=_identity_quat(2) + offset,
        anchor_local_ang_vel=torch.ones(2, 3) * 7 + offset,
        ground_heights=torch.ones(2) * 8 + offset,
    )


def test_noisy_observations_properties_clone_and_subset_update():
    obs = _make_noisy_observations()
    clone = obs.clone()
    clone.rigid_body_pos[0, 0, 0] = -99.0

    assert torch.equal(obs.root_pos, obs.rigid_body_pos[:, 0, :])
    assert torch.equal(obs.root_vel, obs.rigid_body_vel[:, 0, :])
    assert torch.equal(obs.root_ang_vel, obs.rigid_body_ang_vel[:, 0, :])
    assert obs.rigid_body_pos[0, 0, 0] != clone.rigid_body_pos[0, 0, 0]

    replacement = _make_noisy_observations(offset=10.0)
    single_replacement = NoisyObservations(
        **{field: getattr(replacement, field)[:1] for field in replacement.__dataclass_fields__}
    )
    obs.update_subset(torch.tensor([1]), single_replacement)

    assert torch.equal(obs.rigid_body_pos[1], replacement.rigid_body_pos[0])
    assert torch.equal(obs.ground_heights[1:2], replacement.ground_heights[:1])


def test_noise_helpers_return_identity_for_zero_scales_and_normalize_quats():
    tensor = torch.ones(2, 3)
    quat = _identity_quat(2)

    assert _add_noise(tensor, 0.0) is tensor
    assert _add_noise(tensor, [0.0, 0.0, 0.0]) is tensor
    assert _add_quaternion_noise(quat, 0.0) is quat
    assert _add_quaternion_noise(quat, [0.0, 0.0, 0.0, 0.0]) is quat
    assert _add_euler_rotation_noise(quat, 0.0) is quat
    assert _add_euler_rotation_noise(quat, [0.0, 0.0, 0.0]) is quat

    torch.manual_seed(0)
    noisy = _add_noise(tensor, 0.25)
    assert noisy.shape == tensor.shape
    assert not torch.equal(noisy, tensor)

    torch.manual_seed(0)
    noisy_axis = _add_noise(tensor, [0.25, 0.0, 0.5])
    assert torch.equal(noisy_axis[:, 1], tensor[:, 1])

    torch.manual_seed(0)
    noisy_quat = _add_quaternion_noise(quat, 0.05)
    assert torch.allclose(torch.norm(noisy_quat, dim=-1), torch.ones(2))

    torch.manual_seed(0)
    noisy_quat_axis = _add_quaternion_noise(quat, [0.05, 0.0, 0.0, 0.0])
    assert not torch.equal(noisy_quat_axis, quat)
    assert torch.allclose(torch.norm(noisy_quat_axis, dim=-1), torch.ones(2))

    torch.manual_seed(0)
    noisy_euler = _add_euler_rotation_noise(quat, [0.05, 0.0, 0.0])
    assert noisy_euler.shape == quat.shape
    assert torch.allclose(torch.norm(noisy_euler, dim=-1), torch.ones(2), atol=1e-5)

    torch.manual_seed(0)
    noisy_euler_scalar = _add_euler_rotation_noise(quat, 0.05)
    assert noisy_euler_scalar.shape == quat.shape
    assert torch.allclose(
        torch.norm(noisy_euler_scalar, dim=-1),
        torch.ones(2),
        atol=1e-5,
    )


def test_apply_observation_noise_clean_and_zero_noise_paths(monkeypatch):
    robot_state = SimpleNamespace(
        rigid_body_pos=torch.zeros(2, 2, 3),
        rigid_body_rot=_identity_quat(2, 2),
        rigid_body_vel=torch.ones(2, 2, 3),
        rigid_body_ang_vel=torch.ones(2, 2, 3) * 2,
        dof_pos=torch.ones(2, 3),
        dof_vel=torch.ones(2, 3) * 3,
        root_rot=_identity_quat(2),
    )
    ground_heights = torch.tensor([0.1, 0.2])
    monkeypatch.setattr(
        observation_noise,
        "compute_local_ang_vel",
        lambda rot, ang_vel: ang_vel + 1.0,
    )

    clean = apply_observation_noise(None, robot_state, anchor_idx=1, ground_heights=ground_heights)
    zero_noise = apply_observation_noise(
        SimpleNamespace(
            body_pos_noise=0.0,
            body_rot_noise=0.0,
            body_vel_noise=0.0,
            body_ang_vel_noise=0.0,
            dof_pos_noise=0.0,
            dof_vel_noise=0.0,
            root_rot_noise=0.0,
            root_ang_vel_noise=0.0,
            anchor_rot_noise=0.0,
            anchor_ang_vel_noise=0.0,
            ground_height_noise=0.0,
        ),
        robot_state,
        anchor_idx=1,
        ground_heights=ground_heights,
    )

    assert clean.rigid_body_pos is robot_state.rigid_body_pos
    assert torch.equal(clean.anchor_rot, robot_state.rigid_body_rot[:, 1, :])
    assert torch.equal(clean.root_local_ang_vel, robot_state.rigid_body_ang_vel[:, 0, :] + 1.0)
    assert torch.equal(zero_noise.rigid_body_pos, robot_state.rigid_body_pos)
    assert torch.equal(zero_noise.anchor_local_ang_vel, robot_state.rigid_body_ang_vel[:, 1, :] + 1.0)


def test_apply_observation_noise_slices_robot_state_by_env_ids(monkeypatch):
    robot_state = RobotState(
        state_conversion=StateConversion.COMMON,
        rigid_body_pos=torch.arange(4 * 2 * 3, dtype=torch.float).view(4, 2, 3),
        rigid_body_rot=_identity_quat(4, 2),
        rigid_body_vel=torch.arange(100, 100 + 4 * 2 * 3, dtype=torch.float).view(4, 2, 3),
        rigid_body_ang_vel=torch.arange(200, 200 + 4 * 2 * 3, dtype=torch.float).view(4, 2, 3),
        dof_pos=torch.arange(300, 300 + 4 * 3, dtype=torch.float).view(4, 3),
        dof_vel=torch.arange(400, 400 + 4 * 3, dtype=torch.float).view(4, 3),
    )
    env_ids = torch.tensor([2, 0])
    ground_heights = torch.tensor([0.2, 0.4])
    monkeypatch.setattr(
        observation_noise,
        "compute_local_ang_vel",
        lambda rot, ang_vel: ang_vel + 10.0,
    )

    obs = apply_observation_noise(
        None,
        robot_state,
        anchor_idx=1,
        ground_heights=ground_heights,
        env_ids=env_ids,
    )

    assert torch.equal(obs.rigid_body_pos, robot_state.rigid_body_pos[env_ids])
    assert torch.equal(obs.dof_pos, robot_state.dof_pos[env_ids])
    assert torch.equal(obs.root_rot, robot_state.rigid_body_rot[env_ids, 0])
    assert torch.equal(obs.anchor_rot, robot_state.rigid_body_rot[env_ids, 1])
    assert torch.equal(
        obs.root_local_ang_vel,
        robot_state.rigid_body_ang_vel[env_ids, 0] + 10.0,
    )
    assert torch.equal(obs.ground_heights, ground_heights)


def test_apply_reset_noise_clamps_dofs_and_updates_fields():
    reset_state = SimpleNamespace(
        dof_pos=torch.tensor([[-2.0, 0.5, 3.0]]),
        dof_vel=torch.ones(1, 3),
        root_pos=torch.zeros(1, 3),
        root_rot=_identity_quat(1),
        root_vel=torch.ones(1, 3),
        root_ang_vel=torch.ones(1, 3) * 2,
    )
    config = SimpleNamespace(
        dof_pos_noise=0.0,
        dof_vel_noise=0.0,
        root_pos_noise=0.0,
        root_rot_noise=0.0,
        root_vel_noise=0.0,
        root_ang_vel_noise=0.0,
    )

    apply_reset_noise(
        reset_state,
        config,
        dof_limits_lower=torch.tensor([-1.0, -1.0, -1.0]),
        dof_limits_upper=torch.tensor([1.0, 1.0, 1.0]),
    )

    assert torch.equal(reset_state.dof_pos, torch.tensor([[-1.0, 0.5, 1.0]]))
    assert torch.equal(reset_state.root_rot, _identity_quat(1))


def _make_history_buffer(store_noisy: bool = True) -> StateHistoryBuffer:
    return StateHistoryBuffer(
        num_envs=2,
        num_history_steps=2,
        num_bodies=2,
        num_dofs=3,
        action_dim=2,
        num_contact_bodies=2,
        anchor_body_index=1,
        device=torch.device("cpu"),
        store_noisy=store_noisy,
    )


def _history_inputs(base: float = 1.0):
    return {
        "rigid_body_pos": torch.ones(2, 2, 3) * base,
        "rigid_body_rot": _identity_quat(2, 2),
        "rigid_body_vel": torch.ones(2, 2, 3) * (base + 1),
        "rigid_body_ang_vel": torch.ones(2, 2, 3) * (base + 2),
        "dof_pos": torch.ones(2, 3) * (base + 3),
        "dof_vel": torch.ones(2, 3) * (base + 4),
        "actions": torch.ones(2, 2) * (base + 5),
        "ground_heights": torch.ones(2) * (base + 6),
        "body_contacts": torch.tensor([[True, False], [False, True]]),
    }


def _history_sequence(start: int, *shape: int) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    return torch.arange(start, start + count, dtype=torch.float).view(*shape)


def test_state_history_buffer_rotates_updates_properties_and_noisy_values():
    buffer = _make_history_buffer(store_noisy=True)
    first = _history_inputs(base=1.0)
    second = _history_inputs(base=10.0)

    buffer.rotate_and_update(**first)
    buffer.rotate_and_update(
        **second,
        processed_actions=torch.ones(2, 2) * 99.0,
        noisy_rigid_body_pos=torch.ones(2, 2, 3) * -1.0,
        noisy_rigid_body_rot=_identity_quat(2, 2),
        noisy_rigid_body_vel=torch.ones(2, 2, 3) * -2.0,
        noisy_rigid_body_ang_vel=torch.ones(2, 2, 3) * -3.0,
        noisy_dof_pos=torch.ones(2, 3) * -4.0,
        noisy_dof_vel=torch.ones(2, 3) * -5.0,
        noisy_ground_heights=torch.ones(2) * -6.0,
    )

    assert torch.equal(buffer.rigid_body_pos[:, 0], second["rigid_body_pos"])
    assert torch.equal(buffer.rigid_body_pos[:, 1], first["rigid_body_pos"])
    assert torch.equal(buffer.processed_actions[:, 0], torch.ones(2, 2) * 99.0)
    assert torch.equal(buffer.historical_rigid_body_pos, buffer.rigid_body_pos[:, 1:])
    assert torch.equal(buffer.historical_root_pos, buffer.rigid_body_pos[:, 1:, 0, :])
    assert torch.equal(buffer.historical_anchor_pos, buffer.rigid_body_pos[:, 1:, 1, :])
    assert torch.equal(buffer.historical_anchor_vel, buffer.rigid_body_vel[:, 1:, 1, :])
    assert torch.equal(buffer.historical_anchor_ang_vel, buffer.rigid_body_ang_vel[:, 1:, 1, :])
    assert torch.equal(buffer.noisy_rigid_body_pos[:, 0], torch.ones(2, 2, 3) * -1.0)
    assert torch.equal(buffer.noisy_historical_root_pos, buffer.noisy_rigid_body_pos[:, 1:, 0, :])
    assert torch.equal(buffer.noisy_historical_anchor_rot, buffer.noisy_rigid_body_rot[:, 1:, 1, :])
    assert torch.equal(buffer.noisy_historical_ground_heights, buffer.noisy_ground_heights[:, 1:])


def test_state_history_buffer_reset_save_and_load_paths():
    buffer = _make_history_buffer(store_noisy=True)
    env_ids = torch.tensor([0, 1])
    single = _history_inputs(base=2.0)

    buffer.reset_from_single_state(
        env_ids=env_ids,
        rigid_body_pos=single["rigid_body_pos"],
        rigid_body_rot=single["rigid_body_rot"],
        rigid_body_vel=single["rigid_body_vel"],
        rigid_body_ang_vel=single["rigid_body_ang_vel"],
        dof_pos=single["dof_pos"],
        dof_vel=single["dof_vel"],
        ground_heights=single["ground_heights"],
        body_contacts=single["body_contacts"],
    )
    assert torch.equal(buffer.rigid_body_pos[:, 0], single["rigid_body_pos"])
    assert torch.equal(buffer.rigid_body_pos[:, 2], single["rigid_body_pos"])
    assert torch.equal(buffer.actions, torch.zeros_like(buffer.actions))

    historical = {
        "rigid_body_pos": torch.ones(2, 3, 2, 3) * 4.0,
        "rigid_body_rot": _identity_quat(2, 3, 2),
        "rigid_body_vel": torch.ones(2, 3, 2, 3) * 5.0,
        "rigid_body_ang_vel": torch.ones(2, 3, 2, 3) * 6.0,
        "dof_pos": torch.ones(2, 3, 3) * 7.0,
        "dof_vel": torch.ones(2, 3, 3) * 8.0,
        "ground_heights": torch.ones(2, 3) * 9.0,
        "body_contacts": torch.ones(2, 3, 2, dtype=torch.bool),
        "actions": torch.ones(2, 3, 2) * 10.0,
    }
    buffer.reset_from_states(env_ids=env_ids, **historical)
    assert torch.equal(buffer.rigid_body_pos, historical["rigid_body_pos"])
    assert torch.equal(buffer.actions, historical["actions"])
    assert torch.equal(buffer.noisy_dof_vel, historical["dof_vel"])

    state = buffer.save_state()
    buffer.rigid_body_pos.zero_()
    buffer.processed_actions.zero_()
    buffer.load_state(state)
    assert torch.equal(buffer.rigid_body_pos, historical["rigid_body_pos"])
    assert torch.equal(buffer.processed_actions, historical["actions"])

    historical_without_actions = {
        name: value for name, value in historical.items() if name != "actions"
    }
    buffer.actions.fill_(123.0)
    buffer.processed_actions.fill_(456.0)
    buffer.reset_from_states(env_ids=env_ids, **historical_without_actions)
    assert torch.equal(buffer.actions, torch.zeros_like(buffer.actions))
    assert torch.equal(buffer.processed_actions, torch.zeros_like(buffer.processed_actions))


def test_state_history_buffer_clean_noisy_buffers_alias_before_updates():
    buffer = _make_history_buffer(store_noisy=False)

    assert buffer.noisy_rigid_body_pos is buffer.rigid_body_pos
    assert buffer.noisy_dof_vel is buffer.dof_vel
    assert buffer.noisy_historical_dof_pos.shape == (2, 2, 3)


def test_state_history_buffer_historical_views_cover_clean_and_noisy_sources():
    buffer = _make_history_buffer(store_noisy=True)

    buffer.rigid_body_pos.copy_(_history_sequence(0, 2, 3, 2, 3))
    buffer.rigid_body_rot.copy_(_history_sequence(100, 2, 3, 2, 4))
    buffer.rigid_body_vel.copy_(_history_sequence(200, 2, 3, 2, 3))
    buffer.rigid_body_ang_vel.copy_(_history_sequence(300, 2, 3, 2, 3))
    buffer.dof_pos.copy_(_history_sequence(400, 2, 3, 3))
    buffer.dof_vel.copy_(_history_sequence(500, 2, 3, 3))
    buffer.actions.copy_(_history_sequence(600, 2, 3, 2))
    buffer.processed_actions.copy_(_history_sequence(700, 2, 3, 2))
    buffer.ground_heights.copy_(_history_sequence(800, 2, 3))
    buffer.body_contacts.copy_(torch.tensor([[[True, False], [False, True], [True, True]]] * 2))

    buffer.noisy_rigid_body_pos.copy_(_history_sequence(900, 2, 3, 2, 3))
    buffer.noisy_rigid_body_rot.copy_(_history_sequence(1000, 2, 3, 2, 4))
    buffer.noisy_rigid_body_vel.copy_(_history_sequence(1100, 2, 3, 2, 3))
    buffer.noisy_rigid_body_ang_vel.copy_(_history_sequence(1200, 2, 3, 2, 3))
    buffer.noisy_dof_pos.copy_(_history_sequence(1300, 2, 3, 3))
    buffer.noisy_dof_vel.copy_(_history_sequence(1400, 2, 3, 3))

    assert torch.equal(buffer.historical_rigid_body_rot, buffer.rigid_body_rot[:, 1:])
    assert torch.equal(buffer.historical_rigid_body_vel, buffer.rigid_body_vel[:, 1:])
    assert torch.equal(buffer.historical_rigid_body_ang_vel, buffer.rigid_body_ang_vel[:, 1:])
    assert torch.equal(buffer.historical_dof_pos, buffer.dof_pos[:, 1:])
    assert torch.equal(buffer.historical_dof_vel, buffer.dof_vel[:, 1:])
    assert torch.equal(buffer.historical_actions, buffer.actions[:, 1:])
    assert torch.equal(buffer.historical_processed_actions, buffer.processed_actions[:, 1:])
    assert torch.equal(buffer.historical_ground_heights, buffer.ground_heights[:, 1:])
    assert torch.equal(buffer.historical_body_contacts, buffer.body_contacts[:, 1:])
    assert torch.equal(buffer.historical_root_rot, buffer.rigid_body_rot[:, 1:, 0, :])
    assert torch.equal(buffer.historical_root_ang_vel, buffer.rigid_body_ang_vel[:, 1:, 0, :])
    assert torch.equal(buffer.historical_anchor_rot, buffer.rigid_body_rot[:, 1:, 1, :])

    assert torch.equal(buffer.noisy_historical_rigid_body_pos, buffer.noisy_rigid_body_pos[:, 1:])
    assert torch.equal(buffer.noisy_historical_rigid_body_rot, buffer.noisy_rigid_body_rot[:, 1:])
    assert torch.equal(buffer.noisy_historical_rigid_body_vel, buffer.noisy_rigid_body_vel[:, 1:])
    assert torch.equal(
        buffer.noisy_historical_rigid_body_ang_vel,
        buffer.noisy_rigid_body_ang_vel[:, 1:],
    )
    assert torch.equal(buffer.noisy_historical_dof_pos, buffer.noisy_dof_pos[:, 1:])
    assert torch.equal(buffer.noisy_historical_dof_vel, buffer.noisy_dof_vel[:, 1:])
    assert torch.equal(buffer.noisy_historical_root_rot, buffer.noisy_rigid_body_rot[:, 1:, 0, :])
    assert torch.equal(
        buffer.noisy_historical_root_ang_vel,
        buffer.noisy_rigid_body_ang_vel[:, 1:, 0, :],
    )
    assert torch.equal(buffer.noisy_historical_anchor_pos, buffer.noisy_rigid_body_pos[:, 1:, 1, :])
