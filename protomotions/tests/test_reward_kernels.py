# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure reward kernels."""

import math

import torch

from protomotions.envs.rewards import base, regularization, task, tracking


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def test_base_reward_primitives_handle_shapes_indices_and_exp_modes():
    x = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2.0, 2.0], [4.0, 4.0]],
        ]
    )
    ref = torch.zeros_like(x)
    indices = torch.tensor([1])

    assert torch.allclose(
        base.mean_squared_error(x, ref),
        torch.tensor([0.5, 10.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(x, ref, indices=indices),
        torch.tensor([1.0, 16.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(torch.tensor([[1.0, 3.0]]), torch.zeros(1, 2)),
        torch.tensor([5.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(torch.tensor([2.0]), torch.zeros(1)),
        torch.tensor([4.0]),
    )

    assert torch.allclose(
        base.mean_squared_error_exp(x, ref, coefficient=-1.0),
        torch.exp(torch.tensor([-0.5, -10.0])),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            x,
            ref,
            coefficient=-1.0,
            indices=indices,
            mean_before_exp=False,
        ),
        torch.exp(torch.tensor([[-1.0], [-16.0]])).mean(dim=-1),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            torch.tensor([[1.0, 3.0]]),
            torch.zeros(1, 2),
            coefficient=-2.0,
        ),
        torch.exp(torch.tensor([-10.0])),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            torch.tensor([2.0]),
            torch.zeros(1),
            coefficient=-0.5,
        ),
        torch.exp(torch.tensor([-2.0])),
    )

    assert torch.allclose(
        base.norm(torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])),
        torch.tensor([[5.0, 13.0]]),
    )
    assert torch.allclose(
        base.norm(torch.tensor([[[3.0, 4.0], [5.0, 12.0]]]), indices=indices),
        torch.tensor([[13.0]]),
    )
    assert torch.allclose(
        base.delta_norm(torch.tensor([[3.0, 4.0]]), torch.zeros(1, 2)),
        torch.tensor([5.0]),
    )
    assert torch.allclose(
        base.delta_norm(x, ref, indices=indices),
        torch.tensor([[2.0**0.5], [32.0**0.5]]),
    )
    assert torch.allclose(
        base.delta_logmeanexp(
            torch.tensor([[1.0, 3.0]]),
            torch.zeros(1, 2),
            beta=2.0,
        ),
        (torch.logsumexp(torch.tensor([[2.0, 6.0]]), dim=-1) - math.log(2)) / 2.0,
    )
    assert torch.allclose(
        base.delta_logmeanexp(x, ref, indices=indices, beta=2.0),
        torch.tensor([[1.0], [4.0]]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(x, ref),
        torch.tensor([2.0, 12.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(x, ref, indices=indices),
        torch.tensor([2.0, 8.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(torch.tensor([[1.0, -2.0]]), torch.zeros(1, 2)),
        torch.tensor([3.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(torch.tensor([-3.0]), torch.zeros(1)),
        torch.tensor([3.0]),
    )


def test_base_rotation_and_power_primitives():
    quat = _identity_quat(2, 2)
    ref_quat = quat.clone()
    ref_quat[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert torch.allclose(
        base.rotation_error_exp(quat, quat, coefficient=-1.0),
        torch.ones(2),
    )
    assert torch.allclose(
        base.rotation_error_exp(
            quat,
            quat,
            coefficient=-1.0,
            indices=torch.tensor([1]),
            mean_before_exp=False,
        ),
        torch.ones(2),
    )
    assert base.rotation_error_exp(quat, ref_quat, coefficient=-1.0)[1] < 1.0
    assert torch.allclose(base.rotation_error(quat, quat), torch.zeros(2))
    assert torch.allclose(
        base.rotation_error(quat, quat, indices=torch.tensor([1])),
        torch.zeros(2),
    )

    dof_forces = torch.tensor([[2.0, -3.0], [4.0, 5.0]])
    dof_vel = torch.tensor([[10.0, -2.0], [0.5, -1.0]])
    assert torch.allclose(
        base.power_consumption_sum(dof_forces, dof_vel),
        torch.tensor([26.0, 7.0]),
    )
    assert torch.allclose(
        base.power_consumption_sum(
            dof_forces,
            dof_vel,
            indices=torch.tensor([1]),
        ),
        torch.tensor([6.0, 5.0]),
    )
    assert torch.allclose(
        base.power_consumption_sum(dof_forces, dof_vel, use_torque_squared=True),
        torch.tensor([13.0, 41.0]),
    )
    assert torch.allclose(
        base.power_consumption_exp(dof_forces, dof_vel, coefficient=-0.1),
        torch.exp(torch.tensor([-2.6, -0.7])),
    )
    assert torch.allclose(
        base.power_consumption_exp(
            dof_forces,
            dof_vel,
            coefficient=-0.1,
            use_torque_squared=True,
            indices=torch.tensor([1]),
        ),
        torch.exp(torch.tensor([-0.9, -2.5])),
    )
    assert torch.allclose(
        base.velocity_squared_sum(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])),
        torch.tensor([30.0]),
    )
    assert torch.allclose(
        base.velocity_squared_sum(torch.tensor([[1.0, 2.0, 3.0]])),
        torch.tensor([14.0]),
    )
    assert torch.allclose(
        base.velocity_squared_sum(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            indices=torch.tensor([1]),
        ),
        torch.tensor([25.0]),
    )


def test_regularization_rewards_and_helpers():
    current_action = torch.tensor([[1.0, 3.0], [4.0, 4.0]])
    previous_action = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    dof_pos = torch.tensor([[-2.0, 0.0, 2.0], [0.0, 2.0, 4.0]])
    lower = torch.tensor([-1.0, -1.0, -1.0])
    upper = torch.tensor([1.0, 1.0, 3.0])

    assert torch.allclose(
        regularization.compute_action_smoothness(current_action, previous_action),
        torch.tensor([2.0, 5.0]),
    )
    assert torch.allclose(
        regularization.compute_action_smoothness_logmeanexp(
            current_action,
            previous_action,
            beta=2.0,
        ),
        base.delta_logmeanexp(current_action, previous_action, beta=2.0),
    )
    assert torch.allclose(
        regularization.compute_pow_rew(
            torch.tensor([[2.0, -3.0]]),
            torch.tensor([[10.0, -2.0]]),
        ),
        torch.tensor([26.0]),
    )
    assert torch.allclose(
        regularization.compute_pow_rew(
            torch.tensor([[2.0, -3.0]]),
            torch.tensor([[10.0, -2.0]]),
            use_torque_squared=True,
        ),
        torch.tensor([13.0]),
    )
    assert torch.allclose(
        regularization.compute_soft_pos_limit_rew(dof_pos, lower, upper),
        torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(
        regularization.joint_limit_violation(
            dof_pos,
            lower,
            upper,
            indices=torch.tensor([0, 2]),
        ),
        torch.tensor([1.0, 1.0]),
    )

    sim_contacts = torch.tensor([[True, False, True], [False, True, False]])
    ref_contacts = torch.tensor([[True, True, False], [True, True, False]])
    assert torch.allclose(
        regularization.compute_contact_match_rew(
            sim_contacts,
            ref_contacts,
            contact_body_ids=torch.tensor([1, 2]),
        ),
        torch.tensor([2.0, 0.0]),
    )
    assert torch.allclose(
        regularization.contact_mismatch_sum(
            sim_contacts,
            ref_contacts,
            indices=torch.tensor([0, 1]),
        ),
        torch.tensor([1.0, 1.0]),
    )
    assert torch.allclose(
        regularization.compute_contact_force_change_rew(
            torch.tensor([[10.0, 50.0], [100.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [20.0, 40.0]]),
            threshold=30.0,
        ),
        torch.tensor([20.0, 60.0]),
    )
    assert torch.allclose(
        regularization.impact_force_penalty(
            torch.tensor([[10.0, 50.0], [100.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [20.0, 40.0]]),
            indices=torch.tensor([0]),
            threshold=30.0,
        ),
        torch.tensor([0.0, 50.0]),
    )


def test_task_rewards_cover_direction_path_target_and_object_terms():
    root_rot = _identity_quat(3)
    heading_reward = task.compute_heading_velocity_rew(
        root_pos=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        prev_root_pos=torch.zeros(3, 3),
        root_rot=root_rot,
        tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        tar_speed=torch.tensor([1.0, 1.0, 1.0]),
        tar_face_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        dt=1.0,
    )
    assert torch.allclose(heading_reward[0], torch.tensor(1.0))
    assert torch.allclose(heading_reward[1], torch.tensor(0.3))
    assert heading_reward[2] < 0.7

    assert torch.allclose(
        task.compute_path_following_rew(
            head_pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
            tar_pos=torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
            height_conditioned=True,
            pos_err_scale=2.0,
            height_err_scale=1.0,
        ),
        torch.tensor([math.exp(-1.0), math.exp(-2.0)]),
    )
    assert torch.allclose(
        task.compute_path_following_rew(
            head_pos=torch.tensor([[0.0, 0.0, 0.0]]),
            tar_pos=torch.tensor([[0.0, 0.0, 10.0]]),
            height_conditioned=False,
        ),
        torch.ones(1),
    )

    target_reward = task.compute_target_rew(
        root_pos=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        tar_pos=torch.tensor([[0.1, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        tar_proximity_threshold=0.5,
        pos_err_scale=0.5,
    )
    assert torch.allclose(target_reward, torch.tensor([1.0, math.exp(-1.0)]))


def test_tracking_rewards_cover_standard_and_beyond_mimic_variants():
    current_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    ref_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
        ]
    )
    current_rot = _identity_quat(2, 2)
    ref_rot = _identity_quat(2, 2)
    ref_rot[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    current_vel = torch.ones(2, 2, 3)
    ref_vel = torch.zeros(2, 2, 3)
    current_anchor_pos = current_pos[:, 0, :]
    current_anchor_rot = current_rot[:, 0, :]

    assert torch.allclose(tracking.compute_gt_rew(current_pos, current_pos), torch.ones(2))
    assert torch.allclose(tracking.compute_gr_rew(current_rot, current_rot), torch.ones(2))
    assert torch.allclose(tracking.compute_gv_rew(current_vel, current_vel), torch.ones(2))
    assert torch.allclose(tracking.compute_gav_rew(current_vel, current_vel), torch.ones(2))
    assert torch.allclose(
        tracking.compute_rh_rew(current_pos[:, 0, 2], current_pos),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_global_position_error_exp(current_pos, current_pos, sigma=0.5),
        torch.ones(2),
    )
    assert tracking.compute_global_position_error_exp(
        current_pos,
        ref_pos,
        sigma=1.0,
        indices=torch.tensor([1]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_global_anchor_pos_rew(
            current_anchor_pos,
            ref_pos,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.tensor([1.0, math.exp(-1.0)]),
    )
    assert torch.allclose(
        tracking.compute_global_orientation_error_exp(current_rot, current_rot, sigma=1.0),
        torch.ones(2),
    )
    assert tracking.compute_global_anchor_ori_rew(
        current_anchor_rot,
        ref_rot,
        anchor_idx=0,
        sigma=1.0,
    )[1] < 1.0

    assert torch.allclose(
        tracking.compute_relative_body_pos_rew(
            current_pos,
            current_pos,
            current_anchor_rot,
            current_rot,
            current_anchor_pos,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.ones(2),
    )
    assert tracking.compute_relative_body_pos_rew(
        current_pos,
        ref_pos,
        current_anchor_rot,
        ref_rot,
        current_anchor_pos,
        anchor_idx=0,
        sigma=1.0,
        body_indices=torch.tensor([1]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_relative_body_ori_rew(
            current_rot,
            current_rot,
            current_anchor_rot,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.ones(2),
    )
    assert tracking.compute_relative_body_ori_rew(
        current_rot,
        ref_rot,
        current_anchor_rot,
        anchor_idx=0,
        sigma=1.0,
        body_indices=torch.tensor([0]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_global_body_lin_vel_rew(current_vel, current_vel),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_global_body_ang_vel_rew(current_vel, current_vel),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_gt_rel_rew(
            current_pos,
            current_pos,
            current_anchor_rot,
            current_rot,
            anchor_idx=0,
            body_indices=[0, 1],
        ),
        torch.ones(2),
    )
    assert tracking.compute_gt_rel_rew(
        current_pos,
        ref_pos,
        current_anchor_rot,
        ref_rot,
        anchor_idx=0,
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_anchor_xy_rew(
            current_anchor_pos,
            current_pos,
            anchor_idx=0,
        ),
        torch.ones(2),
    )
