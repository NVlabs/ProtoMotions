# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for historical humanoid observation builders."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.obs.humanoid_historical import (
    compute_historical_actions_from_state,
    compute_historical_max_coords_from_motion_lib,
    compute_historical_max_coords_from_state,
    compute_historical_poses_with_time,
    compute_historical_poses_with_time_reduced_coords,
    compute_historical_reduced_coords_from_state,
    compute_historical_time_offsets,
)


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _history(num_envs=1, history_steps=3, num_bodies=2):
    body_pos = torch.arange(
        num_envs * history_steps * num_bodies * 3, dtype=torch.float
    ).reshape(num_envs, history_steps, num_bodies, 3)
    return {
        "body_pos": body_pos,
        "body_rot": _identity_quat(num_envs, history_steps, num_bodies),
        "body_vel": torch.ones(num_envs, history_steps, num_bodies, 3),
        "body_ang_vel": torch.ones(num_envs, history_steps, num_bodies, 3) * 2.0,
        "ground": torch.zeros(num_envs, history_steps),
        "contacts": torch.tensor([[[True, False], [False, True], [True, True]]]),
        "dof_pos": torch.arange(num_envs * history_steps * 2, dtype=torch.float).reshape(
            num_envs, history_steps, 2
        ),
        "dof_vel": torch.ones(num_envs, history_steps, 2) * 3.0,
        "root_rot": _identity_quat(num_envs, history_steps),
        "root_ang": torch.ones(num_envs, history_steps, 3) * 4.0,
        "anchor_rot": _identity_quat(num_envs, history_steps),
    }


def _make_motion_state(motion_ids, motion_times, *, num_bodies=3, with_contacts=True):
    batch = motion_ids.shape[0]
    ids = motion_ids.float().view(batch, 1, 1)
    times = motion_times.view(batch, 1, 1)
    body_ids = torch.arange(num_bodies, dtype=torch.float).view(1, num_bodies, 1)

    rigid_body_pos = torch.cat(
        [
            ids + body_ids * 0.1 + times,
            ids * -0.5 + body_ids * 0.2 + times * 0.3,
            0.25 + body_ids * 0.4 + times * 0.5,
        ],
        dim=-1,
    )
    rigid_body_rot = _identity_quat(batch, num_bodies)
    rigid_body_vel = torch.cat(
        [
            times.expand(batch, num_bodies, 1),
            (ids + 1.0).expand(batch, num_bodies, 1),
            (body_ids + 2.0).expand(batch, num_bodies, 1),
        ],
        dim=-1,
    )
    rigid_body_ang_vel = torch.cat(
        [
            (body_ids + 3.0).expand(batch, num_bodies, 1),
            (times + 4.0).expand(batch, num_bodies, 1),
            (ids + 5.0).expand(batch, num_bodies, 1),
        ],
        dim=-1,
    )

    state = {
        "rigid_body_pos": rigid_body_pos,
        "rigid_body_rot": rigid_body_rot,
        "rigid_body_vel": rigid_body_vel,
        "rigid_body_ang_vel": rigid_body_ang_vel,
    }
    if with_contacts:
        body_indices = torch.arange(num_bodies).view(1, num_bodies)
        state["rigid_body_contacts"] = (
            (
                motion_ids.view(batch, 1)
                + body_indices
                + (motion_times * 10).long().view(batch, 1)
            )
            % 2
            == 0
        )
    return SimpleNamespace(**state)


class _RecordingMotionLib:
    def __init__(self, *, motion_lengths=None, with_contacts=True):
        if motion_lengths is not None:
            self.motion_lengths = motion_lengths
        self.with_contacts = with_contacts
        self.calls = []

    def get_motion_state(self, motion_ids, motion_times):
        self.calls.append((motion_ids.clone(), motion_times.clone()))
        return _make_motion_state(
            motion_ids,
            motion_times,
            with_contacts=self.with_contacts,
        )


def test_historical_time_offsets_support_all_selection_forms():
    assert torch.equal(
        compute_historical_time_offsets(3, dt=0.1, num_envs=2, device="cpu"),
        torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]),
    )
    assert torch.equal(
        compute_historical_time_offsets(5, dt=0.2, num_envs=1, device="cpu", history_steps=2),
        torch.tensor([[0.0, 0.2]]),
    )
    assert torch.equal(
        compute_historical_time_offsets(5, dt=0.2, num_envs=1, device="cpu", history_steps=[1, 4]),
        torch.tensor([[0.0, 0.6]]),
    )


def test_historical_reduced_and_max_coords_select_steps_and_validate_enabled_history():
    hist = _history()

    reduced_all = compute_historical_reduced_coords_from_state(
        hist["dof_pos"],
        hist["dof_vel"],
        hist["root_rot"],
        hist["root_ang"],
        hist["anchor_rot"],
        history_steps=None,
        w_last=True,
    )
    reduced_subset = compute_historical_reduced_coords_from_state(
        hist["dof_pos"],
        hist["dof_vel"],
        hist["root_rot"],
        hist["root_ang"],
        hist["anchor_rot"],
        history_steps=[1, 3],
        w_last=True,
    )
    max_subset = compute_historical_max_coords_from_state(
        hist["body_pos"],
        hist["body_rot"],
        hist["body_vel"],
        hist["body_ang_vel"],
        hist["ground"],
        hist["contacts"],
        local_obs=False,
        root_height_obs=True,
        observe_contacts=True,
        history_steps=[1, 3],
        w_last=True,
    )

    assert reduced_all.shape == (1, 30)
    assert reduced_subset.shape == (1, 20)
    assert max_subset.shape == (1, 60)
    max_all = compute_historical_max_coords_from_state(
        hist["body_pos"],
        hist["body_rot"],
        hist["body_vel"],
        hist["body_ang_vel"],
        hist["ground"],
        hist["contacts"],
        local_obs=True,
        root_height_obs=True,
        observe_contacts=False,
        history_steps=None,
        w_last=True,
    )
    assert max_all.shape == (1, 84)

    with pytest.raises(ValueError, match="Historical state tensors are None"):
        compute_historical_reduced_coords_from_state(None, None, None, None, None)
    with pytest.raises(ValueError, match="Historical state tensors are None"):
        compute_historical_max_coords_from_state(None, None, None, None, None, None)


def test_historical_actions_and_time_augmented_pose_builders():
    hist = _history()
    actions = torch.arange(1 * 3 * 2, dtype=torch.float).reshape(1, 3, 2)

    assert torch.equal(
        compute_historical_actions_from_state(actions, history_steps=[1, 3]),
        torch.tensor([[0.0, 1.0, 4.0, 5.0]]),
    )
    assert torch.equal(
        compute_historical_actions_from_state(actions, history_steps=None),
        actions.reshape(1, -1),
    )
    with pytest.raises(ValueError, match="Historical actions tensor is None"):
        compute_historical_actions_from_state(None)

    max_with_time = compute_historical_poses_with_time(
        hist["body_pos"],
        hist["body_rot"],
        hist["body_vel"],
        hist["body_ang_vel"],
        hist["ground"],
        hist["contacts"],
        history_steps=[1, 3],
        local_obs=False,
        root_height_obs=True,
        w_last=True,
        dt=0.1,
    )
    reduced_with_time = compute_historical_poses_with_time_reduced_coords(
        hist["dof_pos"],
        hist["dof_vel"],
        hist["root_ang"],
        hist["root_rot"],
        hist["anchor_rot"],
        history_steps=2,
        w_last=True,
        dt=0.1,
    )

    assert max_with_time.shape == (1, 58)
    assert torch.equal(max_with_time[:, -1:], torch.tensor([[0.2]]))
    assert reduced_with_time.shape == (1, 22)
    assert torch.equal(reduced_with_time[:, -1:], torch.tensor([[0.1]]))


def test_historical_max_coords_from_motion_lib_samples_clamped_past_times():
    class _MotionLib:
        def get_motion_state(self, motion_ids, sample_times):
            num_samples = motion_ids.shape[0]
            body_pos = torch.zeros(num_samples, 2, 3)
            body_pos[:, 0, 2] = sample_times
            return SimpleNamespace(
                rigid_body_pos=body_pos,
                rigid_body_rot=_identity_quat(num_samples, 2),
                rigid_body_vel=torch.ones(num_samples, 2, 3),
                rigid_body_ang_vel=torch.ones(num_samples, 2, 3) * 2.0,
            )

    obs = compute_historical_max_coords_from_motion_lib(
        _MotionLib(),
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.tensor([0.05, 0.4]),
        num_state_history_steps=3,
        dt=0.1,
        local_obs=False,
        root_height_obs=True,
        history_steps=[1, 3],
    )

    assert obs.shape == (2, 56)
    assert obs[0, 0].item() == pytest.approx(0.0)
    assert obs[1, 28].item() == pytest.approx(0.1)

    all_steps = compute_historical_max_coords_from_motion_lib(
        _MotionLib(),
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.4]),
        num_state_history_steps=2,
        dt=0.1,
        local_obs=False,
        root_height_obs=True,
        history_steps=None,
    )
    first_two = compute_historical_max_coords_from_motion_lib(
        _MotionLib(),
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.4]),
        num_state_history_steps=5,
        dt=0.1,
        local_obs=False,
        root_height_obs=True,
        history_steps=2,
    )
    assert all_steps.shape == (1, 56)
    assert torch.equal(all_steps, first_two)


def test_motion_lib_reference_history_supports_missing_motion_lengths_and_step_selection():
    motion_lib = _RecordingMotionLib()
    motion_ids = torch.tensor([2, 4])
    motion_times = torch.tensor([0.5, 0.04])

    obs = compute_historical_max_coords_from_motion_lib(
        motion_lib,
        motion_ids=motion_ids,
        motion_times=motion_times,
        num_state_history_steps=6,
        dt=0.1,
        local_obs=False,
        root_height_obs=True,
        observe_contacts=False,
        history_steps=[1, 3, 6],
    )

    assert [call_times for _, call_times in motion_lib.calls] == [
        pytest.approx(torch.tensor([0.4, 0.0])),
        pytest.approx(torch.tensor([0.2, 0.0])),
        pytest.approx(torch.tensor([0.0, 0.0])),
    ]
    assert obs.shape == (2, 3 * 43)


def test_motion_lib_reference_history_clamps_to_motion_lengths_and_slices_contacts():
    motion_lib = _RecordingMotionLib(motion_lengths=torch.tensor([0.25, 0.32]))
    motion_ids = torch.tensor([0, 1])
    motion_times = torch.tensor([0.8, 0.04])
    contact_body_ids = torch.tensor([2, 0])

    obs = compute_historical_max_coords_from_motion_lib(
        motion_lib,
        motion_ids=motion_ids,
        motion_times=motion_times,
        num_state_history_steps=4,
        dt=0.1,
        local_obs=False,
        root_height_obs=True,
        observe_contacts=True,
        contact_body_ids=contact_body_ids,
        history_steps=[1, 4],
    )

    assert [call_times for _, call_times in motion_lib.calls] == [
        pytest.approx(torch.tensor([0.25, 0.0])),
        pytest.approx(torch.tensor([0.25, 0.0])),
    ]

    step1_state = _make_motion_state(
        motion_ids,
        torch.tensor([0.25, 0.0]),
        with_contacts=True,
    )
    step4_state = _make_motion_state(
        motion_ids,
        torch.tensor([0.25, 0.0]),
        with_contacts=True,
    )
    expected = compute_historical_max_coords_from_state(
        historical_rigid_body_pos=torch.stack(
            [step1_state.rigid_body_pos, step4_state.rigid_body_pos],
            dim=1,
        ),
        historical_rigid_body_rot=torch.stack(
            [step1_state.rigid_body_rot, step4_state.rigid_body_rot],
            dim=1,
        ),
        historical_rigid_body_vel=torch.stack(
            [step1_state.rigid_body_vel, step4_state.rigid_body_vel],
            dim=1,
        ),
        historical_rigid_body_ang_vel=torch.stack(
            [step1_state.rigid_body_ang_vel, step4_state.rigid_body_ang_vel],
            dim=1,
        ),
        historical_ground_heights=torch.zeros(2, 2),
        historical_body_contacts=torch.stack(
            [
                step1_state.rigid_body_contacts[:, contact_body_ids],
                step4_state.rigid_body_contacts[:, contact_body_ids],
            ],
            dim=1,
        ),
        local_obs=False,
        root_height_obs=True,
        observe_contacts=True,
        history_steps=None,
        w_last=True,
    )

    assert torch.allclose(obs, expected, atol=1e-6)


def test_motion_lib_reference_history_zero_fills_missing_contacts_when_requested():
    motion_lib = _RecordingMotionLib(with_contacts=False)

    obs = compute_historical_max_coords_from_motion_lib(
        motion_lib,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.2]),
        num_state_history_steps=1,
        dt=0.1,
        local_obs=False,
        root_height_obs=True,
        observe_contacts=True,
        contact_body_ids=torch.tensor([0, 2]),
        history_steps=1,
    )

    assert torch.equal(obs[:, -2:], torch.zeros(1, 2))
