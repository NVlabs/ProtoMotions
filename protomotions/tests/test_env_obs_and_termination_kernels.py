# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure observation and termination kernels."""

import torch

from protomotions.envs.obs.path import compute_path_obs
from protomotions.envs.obs.steering import compute_steering_obs
from protomotions.envs.obs.target import compute_target_obs
from protomotions.envs.steps import select_steps
from protomotions.envs.terminations import base as base_terms
from protomotions.envs.terminations import task as task_terms
from protomotions.envs.terminations import tracking


def _identity_quat(num_envs: int, *extra_dims: int) -> torch.Tensor:
    quat = torch.zeros(num_envs, *extra_dims, 4)
    quat[..., 3] = 1.0
    return quat


def test_select_steps_narrows_a_leading_run_and_gathers_one_indexed_steps():
    tensor = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    assert torch.equal(select_steps(tensor, (1, 2)), tensor[:, :2])
    assert torch.equal(select_steps(tensor, (1, 3), dim=1), tensor[:, [0, 2]])


def test_path_steering_and_target_observations_use_heading_local_frame():
    root_rot = _identity_quat(1)
    head_pos = torch.tensor([[1.0, 2.0, 0.5]])
    traj_samples = torch.tensor([[[2.0, 4.0, 1.0], [0.0, 1.0, -1.0]]])

    assert torch.allclose(
        compute_path_obs(root_rot, head_pos, traj_samples, height_conditioned=True),
        torch.tensor([[1.0, 2.0, 0.5, -1.0, -1.0, -1.5]]),
    )
    assert torch.allclose(
        compute_path_obs(root_rot, head_pos, traj_samples, height_conditioned=False),
        torch.tensor([[1.0, 2.0, -1.0, -1.0]]),
    )

    steering_obs = compute_steering_obs(
        root_rot=_identity_quat(2),
        tar_dir=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        tar_speed=torch.tensor([2.0, 3.0]),
        tar_face_dir=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
    )
    assert torch.allclose(
        steering_obs,
        torch.tensor([[1.0, 0.0, 2.0, 0.0, 1.0], [0.0, 1.0, 3.0, -1.0, 0.0]]),
    )

    target_obs = compute_target_obs(
        root_pos=torch.tensor([[1.0, 2.0, 3.0]]),
        root_rot=root_rot,
        tar_pos=torch.tensor([[3.0, 1.0, 5.0]]),
    )
    assert torch.allclose(target_obs, torch.tensor([[2.0, -1.0]]))


def test_base_termination_kernels_cover_contact_height_and_threshold_paths():
    contacts = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]
    )
    progress = torch.tensor([0, 2, 3])
    # Matrix rows carry every body here; body "b" (index 1) is allowed support.
    forces = torch.zeros(3, 3, 4, 3)
    forces[:, :, 0, 2] = contacts.float()
    body_index = {"a": 0, "b": 1, "c": 2}
    partner_index = {"terrain": 0, "robot/a": 1, "robot/b": 2, "robot/c": 3}

    rigid_body_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.4], [0.0, 0.0, 0.1], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.8], [0.0, 0.0, 0.1], [0.0, 0.0, 0.9]],
            [[0.0, 0.0, 0.4], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
        ]
    )
    # Named fall: forbidden bodies a and c against terrain, contact AND low
    # height, after the two-step reset grace. Env 0 touches at progress 0
    # (grace), env 1 only touches with the allowed body, env 2 falls.
    assert torch.equal(
        base_terms.named_contact_fall_termination(
            rigid_body_pos=rigid_body_pos,
            contact_forces=forces,
            contact_body_index=body_index,
            contact_partner_index=partner_index,
            ground_heights=torch.zeros(3),
            progress_buf=progress,
            body_ids=torch.tensor([0, 2]),
            body_names=["a", "c"],
            partner_names=["terrain"],
            termination_height=0.5,
        ),
        torch.tensor([False, False, True]),
    )
    assert torch.equal(
        base_terms.check_max_length_term(torch.tensor([8, 9, 10]), 10.0),
        torch.tensor([False, True, True]),
    )


def test_task_termination_kernels_measure_path_and_steering_errors():
    progress = torch.tensor([5, 20, 20])
    head_pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    target_pos = torch.zeros(3, 3)

    # Zero before min_progress; the 3D distance and the height error after it.
    assert torch.equal(
        task_terms.path_distance(head_pos, target_pos, progress, min_progress=10),
        torch.tensor([0.0, 2.0, 3.0]),
    )
    assert torch.equal(
        task_terms.path_height_error(head_pos, target_pos, progress, min_progress=10),
        torch.tensor([0.0, 0.0, 3.0]),
    )

    root_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 0.0]])
    prev_root_pos = torch.zeros(3, 3)
    tar_dir = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert torch.allclose(
        task_terms.steering_speed_error(
            root_pos, prev_root_pos, torch.tensor([1.0, 1.0, 3.0]), dt=1.0
        ),
        torch.tensor([0.0, 0.0, 2.9]),
    )
    # Alignment reads 1.0 (a pass) for a near-stationary robot: |v| = 0.1 is
    # not > 0.1, exactly where the old boolean check skipped the direction test.
    assert torch.allclose(
        task_terms.steering_direction_alignment(
            root_pos, prev_root_pos, tar_dir, dt=1.0
        ),
        torch.tensor([1.0, 0.0, 1.0]),
        atol=1e-5,
    )


def test_tracking_values_and_terminations_cover_global_and_relative_errors():
    current_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    ref_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.0]],
            [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
        ]
    )
    current_rot = _identity_quat(2, 2)
    ref_rot = _identity_quat(2, 2)
    ref_rot[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    current_anchor_pos = current_pos[:, 0, :]
    current_anchor_rot = current_rot[:, 0, :]
    ref_anchor_pos = ref_pos[:, 0, :]
    ref_anchor_rot = ref_rot[:, 0, :]

    assert torch.allclose(
        tracking.mean_body_pos_error(current_pos, ref_pos),
        torch.tensor([0.15, 1.0]),
    )
    assert torch.allclose(
        tracking.max_body_pos_error(current_pos, ref_pos),
        torch.tensor([0.3, 1.0]),
    )
    assert torch.allclose(
        tracking.mean_body_rot_error(current_rot, current_rot),
        torch.zeros(2),
    )
    assert torch.allclose(
        tracking.anchor_pos_error_value(current_anchor_pos, ref_anchor_pos),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(
        tracking.anchor_height_error_value(current_anchor_pos, ref_anchor_pos),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(
        tracking.anchor_ori_error_value(current_anchor_rot, ref_anchor_rot),
        torch.tensor([0.0, 2.0]),
        atol=1e-5,
    )
    assert torch.allclose(
        tracking.relative_body_pos_max_error(
            current_pos,
            ref_pos,
            current_anchor_pos,
            current_anchor_rot,
            ref_anchor_pos,
            ref_anchor_rot,
        ),
        torch.tensor([0.3, 2.0**0.5]),
    )
