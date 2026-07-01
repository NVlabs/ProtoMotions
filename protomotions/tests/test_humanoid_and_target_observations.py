# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for humanoid and mimic target observation builders."""

import pytest
import torch

from protomotions.envs.obs import humanoid
from protomotions.envs.obs.humanoid import (
    compute_humanoid_max_coords_observations,
    compute_humanoid_reduced_coords_observations,
    compute_local_ang_vel,
    dof_to_local,
    dof_to_obs,
    obs_to_dof,
    root_projected_gravity,
)
from protomotions.envs.obs.masked_mimic import (
    compute_target_masks_only,
    compute_target_poses_only,
    compute_target_time_offsets,
)
from protomotions.envs.obs.target_poses import (
    build_corrupted_xy_offset,
    build_deploy_target_poses,
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
    build_reduced_coords_target_poses,
    build_sparse_target_poses,
    build_target_height,
    build_target_root_ang_vel,
    build_target_root_rot,
    build_target_root_vel,
    build_target_xy_offset,
)


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _body_state(num_envs=1, num_bodies=2):
    body_pos = torch.arange(num_envs * num_bodies * 3, dtype=torch.float).reshape(
        num_envs, num_bodies, 3
    )
    return {
        "pos": body_pos,
        "rot": _identity_quat(num_envs, num_bodies),
        "vel": torch.ones(num_envs, num_bodies, 3),
        "ang_vel": torch.ones(num_envs, num_bodies, 3) * 2.0,
    }


def _future_body_state(num_envs=1, future_steps=3, num_bodies=2):
    base = torch.arange(
        num_envs * future_steps * num_bodies * 3, dtype=torch.float
    ).reshape(num_envs, future_steps, num_bodies, 3)
    return {
        "pos": base + 1.0,
        "rot": _identity_quat(num_envs, future_steps, num_bodies),
        "vel": torch.ones(num_envs, future_steps, num_bodies, 3) * 3.0,
        "ang_vel": torch.ones(num_envs, future_steps, num_bodies, 3) * 4.0,
    }


def test_dof_conversion_helpers_cover_shape_validation_and_round_trips(monkeypatch):
    hinge_axes_map = {
        1: torch.tensor([[1.0, 0.0, 0.0]]),
        2: torch.eye(3),
    }

    def fake_extract_transforms(axes_map, pose, qpos_is_exp_map_on_3dof_joints):
        return torch.eye(3).expand(pose.shape[0], len(axes_map), 3, 3).clone()

    monkeypatch.setattr(
        humanoid,
        "extract_transforms_from_qpos_non_root_ignore_fixed_helper",
        fake_extract_transforms,
    )

    pose = torch.zeros(2, 4)
    local = dof_to_local(pose, hinge_axes_map, w_last=True)
    obs = dof_to_obs(pose, hinge_axes_map, w_last=True)
    recovered = obs_to_dof(obs, hinge_axes_map, w_last=True)

    assert local.shape == (2, 2, 4)
    assert obs.shape == (2, 12)
    assert torch.allclose(recovered, torch.zeros(2, 4), atol=1e-5)

    with pytest.raises(ValueError, match="Input pose must be 2D"):
        dof_to_local(torch.zeros(2, 2, 2), hinge_axes_map, w_last=True)

    with pytest.raises(ValueError, match="Unsupported number of DOFs"):
        obs_to_dof(torch.zeros(2, 6), {0: torch.ones(2, 3)}, w_last=True)


def test_projected_gravity_and_local_ang_vel_handle_identity_and_batched_history():
    root_rot = _identity_quat(2)
    ang_vel = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    history_rot = _identity_quat(2, 3)
    history_ang_vel = torch.arange(18, dtype=torch.float).reshape(2, 3, 3)

    assert root_projected_gravity(None) is None
    assert torch.equal(
        root_projected_gravity(root_rot),
        torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
    )
    assert torch.equal(compute_local_ang_vel(root_rot, ang_vel), ang_vel)
    assert torch.equal(
        compute_local_ang_vel(history_rot, history_ang_vel),
        history_ang_vel,
    )


def test_reduced_coords_observations_include_optional_velocity_and_height():
    obs = compute_humanoid_reduced_coords_observations(
        dof_pos=torch.tensor([[0.1, 0.2]]),
        dof_vel=torch.tensor([[1.0, 2.0]]),
        anchor_rot=_identity_quat(1),
        root_local_ang_vel=torch.tensor([[3.0, 4.0, 5.0]]),
        root_rot=_identity_quat(1),
        root_pos=torch.tensor([[0.0, 0.0, 1.5]]),
        root_vel=torch.tensor([[6.0, 7.0, 8.0]]),
        ground_height=torch.tensor([0.5]),
        root_height_obs=True,
        root_vel_obs=True,
        w_last=True,
    )

    assert torch.allclose(
        obs,
        torch.tensor([[0.1, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, -1.0, 6.0, 7.0, 8.0, 1.0]]),
    )

    with pytest.raises(ValueError, match="root_vel is required"):
        compute_humanoid_reduced_coords_observations(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            _identity_quat(1),
            torch.zeros(1, 3),
            root_vel_obs=True,
            w_last=True,
        )
    with pytest.raises(ValueError, match="root_rot is required"):
        compute_humanoid_reduced_coords_observations(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            _identity_quat(1),
            torch.zeros(1, 3),
            root_vel=torch.zeros(1, 3),
            root_vel_obs=True,
            w_last=True,
        )
    with pytest.raises(ValueError, match="root_pos is required"):
        compute_humanoid_reduced_coords_observations(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            _identity_quat(1),
            torch.zeros(1, 3),
            root_height_obs=True,
            w_last=True,
        )
    height_without_ground = compute_humanoid_reduced_coords_observations(
        dof_pos=torch.zeros(1, 1),
        dof_vel=torch.zeros(1, 1),
        anchor_rot=_identity_quat(1),
        root_local_ang_vel=torch.zeros(1, 3),
        root_pos=torch.tensor([[0.0, 0.0, 1.25]]),
        root_height_obs=True,
        w_last=True,
    )
    assert height_without_ground[0, -1].item() == pytest.approx(1.25)


def test_max_coords_observations_cover_local_global_height_and_contact_options():
    state = _body_state(num_envs=1, num_bodies=2)

    global_obs = compute_humanoid_max_coords_observations(
        body_pos=state["pos"],
        body_rot=state["rot"],
        body_vel=state["vel"],
        body_ang_vel=state["ang_vel"],
        ground_height=torch.tensor([0.5]),
        body_contacts=torch.tensor([[True, False]]),
        local_obs=False,
        root_height_obs=False,
        observe_contacts=False,
        w_last=True,
    )
    local_obs = compute_humanoid_max_coords_observations(
        body_pos=state["pos"],
        body_rot=state["rot"],
        body_vel=state["vel"],
        body_ang_vel=state["ang_vel"],
        ground_height=torch.tensor([[0.5]]),
        body_contacts=torch.tensor([[True, False]]),
        local_obs=True,
        root_height_obs=True,
        observe_contacts=True,
        w_last=True,
    )

    assert global_obs.shape == (1, 28)
    assert global_obs[0, 0].item() == 0.0
    assert local_obs.shape == (1, 30)
    assert local_obs[0, 0].item() == pytest.approx(1.5)
    assert torch.equal(local_obs[0, -2:], torch.tensor([1.0, 0.0]))


def test_target_pose_scalar_builders_select_steps_and_transform_identity_frames():
    current_pos = torch.tensor([[1.0, 2.0, 3.0]])
    current_rot = _identity_quat(1)
    ref_pos = torch.tensor([[[4.0, 5.0, 1.0], [6.0, 7.0, 2.0], [8.0, 9.0, 3.0]]])
    ref_rot = _identity_quat(1, 3, 1)
    ref_vel = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    ref_ang_vel = ref_vel + 10.0

    assert build_target_root_rot(current_rot, ref_rot, future_steps=[1, 3]).shape == (1, 12)
    assert torch.equal(
        build_target_xy_offset(
            current_pos,
            current_rot,
            ref_pos,
            future_steps=2,
            current_ref_anchor_pos=torch.tensor([[0.5, 1.5, 3.0]]),
        ),
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
    )
    assert torch.equal(
        build_target_xy_offset(current_pos, current_rot, ref_pos, future_steps=2, zero_xy_offset=True),
        torch.zeros(1, 4),
    )
    assert torch.equal(build_target_height(ref_pos, future_steps=[1, 3]), torch.tensor([[1.0, 3.0]]))
    assert torch.equal(build_target_root_vel(current_rot, ref_vel, future_steps=2), ref_vel[:, :2].reshape(1, -1))
    assert torch.equal(
        build_target_root_ang_vel(current_rot, ref_ang_vel, future_steps=[2]),
        ref_ang_vel[:, 1:2].reshape(1, -1),
    )


def test_target_pose_composite_builders_cover_optional_components_and_errors():
    current = _body_state(num_envs=1, num_bodies=2)
    future = _future_body_state(num_envs=1, future_steps=3, num_bodies=2)
    dof_pos = torch.arange(1 * 3 * 2, dtype=torch.float).reshape(1, 3, 2)
    dof_vel = dof_pos + 10.0

    assert build_max_coords_target_poses_future_rel(
        current["pos"],
        current["rot"],
        future["pos"],
        future["rot"],
        future_steps=[1, 3],
        w_last=True,
    ).shape == (1, 72)
    assert build_max_coords_target_poses(
        current["pos"],
        current["rot"],
        current["vel"],
        current["ang_vel"],
        future["pos"],
        future["rot"],
        future["vel"],
        future["ang_vel"],
        with_velocities=False,
        with_relative=False,
        future_steps=2,
        w_last=True,
    ).shape == (1, 36)
    assert build_max_coords_target_poses(
        current["pos"],
        current["rot"],
        current["vel"],
        current["ang_vel"],
        future["pos"],
        future["rot"],
        future["vel"],
        future["ang_vel"],
        with_velocities=True,
        with_relative=True,
        future_steps=2,
        w_last=True,
    ).shape == (1, 96)

    reduced = build_reduced_coords_target_poses(
        current_state_anchor_rot=current["rot"][:, 0],
        current_state_anchor_pos=current["pos"][:, 0],
        mimic_ref_anchor_rot=future["rot"][:, :, 0],
        mimic_ref_anchor_pos=future["pos"][:, :, 0],
        mimic_ref_anchor_vel=future["vel"][:, :, 0],
        mimic_ref_anchor_ang_vel=future["ang_vel"][:, :, 0],
        mimic_ref_dof_vel=dof_vel,
        mimic_ref_dof_pos=dof_pos,
        include_xy_offset=True,
        include_height=True,
        include_dof_vel=True,
        include_anchor_vel=True,
        include_anchor_ang_vel=True,
        future_steps=[1, 3],
        current_ref_anchor_pos=torch.zeros(1, 3),
        w_last=True,
    )
    assert reduced.shape == (1, 38)

    zeroed_offset = build_reduced_coords_target_poses(
        current_state_anchor_rot=current["rot"][:, 0],
        current_state_anchor_pos=current["pos"][:, 0],
        mimic_ref_anchor_rot=future["rot"][:, :, 0],
        mimic_ref_anchor_pos=future["pos"][:, :, 0],
        mimic_ref_dof_vel=dof_vel,
        mimic_ref_dof_pos=dof_pos,
        include_xy_offset=True,
        include_dof_vel=False,
        zero_xy_offset=True,
        future_steps=1,
        w_last=True,
    )
    assert zeroed_offset.shape == (1, 10)
    assert torch.equal(zeroed_offset[:, 8:10], torch.zeros(1, 2))

    with pytest.raises(ValueError, match="mimic_ref_anchor_vel is required"):
        build_reduced_coords_target_poses(
            current["rot"][:, 0],
            future["rot"][:, :, 0],
            dof_vel,
            dof_pos,
            include_anchor_vel=True,
        )
    with pytest.raises(ValueError, match="mimic_ref_anchor_ang_vel is required"):
        build_reduced_coords_target_poses(
            current["rot"][:, 0],
            future["rot"][:, :, 0],
            dof_vel,
            dof_pos,
            include_anchor_ang_vel=True,
        )


def test_sparse_masked_and_deploy_target_pose_builders_cover_masks_and_step_selection():
    current = _body_state(num_envs=1, num_bodies=2)
    future = _future_body_state(num_envs=1, future_steps=3, num_bodies=2)
    conditionable_body_ids = torch.tensor([1])
    masks = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    time_offsets = torch.tensor([[0.1, 0.2, 0.3]])
    dof_pos = torch.arange(1 * 3 * 2, dtype=torch.float).reshape(1, 3, 2)
    dof_vel = dof_pos + 10.0

    sparse_full = build_sparse_target_poses(
        current["pos"],
        current["rot"],
        future["pos"],
        future["rot"],
        conditionable_body_ids=conditionable_body_ids,
        include_root_relative=True,
        future_steps=[1, 3],
        w_last=True,
    )
    sparse_reduced = build_sparse_target_poses(
        current["pos"],
        current["rot"],
        future["pos"],
        future["rot"],
        conditionable_body_ids=conditionable_body_ids,
        include_root_relative=False,
        future_steps=[1, 3],
        w_last=True,
    )
    masked = compute_target_poses_only(
        current["pos"],
        current["rot"],
        future["pos"],
        future["rot"],
        masks,
        conditionable_body_ids,
        future_steps=[1, 3],
        include_root_relative=False,
    )

    assert sparse_full.shape == (1, 48)
    assert sparse_reduced.shape == (1, 24)
    assert masked.shape == (1, 24)
    assert torch.equal(masked[:, 6:12], torch.zeros(1, 6))
    assert torch.equal(compute_target_masks_only(masks, conditionable_body_ids, None), masks)
    assert torch.equal(
        compute_target_masks_only(masks, conditionable_body_ids, future_steps=[1, 3]),
        torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
    )
    assert torch.equal(compute_target_time_offsets(time_offsets, None), time_offsets)
    assert torch.equal(
        compute_target_time_offsets(time_offsets, future_steps=[1, 3]),
        torch.tensor([[0.1, 0.3]]),
    )

    deploy = build_deploy_target_poses(
        current_anchor_rot=current["rot"][:, 0],
        mimic_ref_rot=future["rot"],
        mimic_ref_dof_pos=dof_pos,
        mimic_ref_dof_vel=dof_vel,
        include_dof_vel=True,
        future_steps=[1, 3],
        w_last=True,
    )
    deploy_without_vel = build_deploy_target_poses(
        current_anchor_rot=current["rot"][:, 0],
        mimic_ref_rot=future["rot"],
        mimic_ref_dof_pos=dof_pos,
        mimic_ref_dof_vel=dof_vel,
        include_dof_vel=False,
        future_steps=1,
        w_last=True,
    )
    assert deploy.shape == (1, 32)
    assert deploy_without_vel.shape == (1, 14)


def test_corrupted_xy_offset_uses_shared_corruption_with_identity_noise():
    offset = build_corrupted_xy_offset(
        current_state_anchor_pos=torch.tensor([[1.0, 2.0, 0.0]]),
        current_state_anchor_rot=_identity_quat(1),
        ref_rigid_body_pos=torch.tensor([[[0.0, 0.0, 0.0], [4.0, 6.0, 0.0]]]),
        anchor_idx=1,
        odom_scale=torch.ones(1),
        odom_yaw_cos_sin=torch.tensor([[1.0, 0.0]]),
        log_noise_std=0.0,
        soft_threshold=0.15,
        w_last=True,
    )

    assert torch.allclose(offset, torch.tensor([[3.0, 4.0]]), atol=1e-6)
