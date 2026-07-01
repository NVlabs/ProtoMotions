# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure observation and termination kernels."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.obs.path import compute_path_obs
from protomotions.envs.obs.steering import compute_steering_obs
from protomotions.envs.obs.target import compute_target_obs
from protomotions.envs.obs.terrain_obs import TerrainObs
from protomotions.envs.obs.utils import select_step_indices
from protomotions.envs.terminations import base as base_terms
from protomotions.envs.terminations import task as task_terms
from protomotions.envs.terminations import tracking


def _identity_quat(num_envs: int, *extra_dims: int) -> torch.Tensor:
    quat = torch.zeros(num_envs, *extra_dims, 4)
    quat[..., 3] = 1.0
    return quat


def test_select_step_indices_supports_counts_and_one_indexed_lists():
    tensor = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    assert torch.equal(select_step_indices(tensor, 2), tensor[:, :2])
    assert torch.equal(
        select_step_indices(tensor, [1, 3], dim=1),
        tensor[:, [0, 2]],
    )


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


def test_terrain_obs_updates_selected_envs_and_returns_clones():
    class _Terrain:
        num_height_points = 3

        def get_height_maps(self, root_states, env_ids):
            return root_states + env_ids[:, None].float()

    class _Simulator:
        def get_root_state(self, env_ids):
            return torch.stack(
                [env_ids.float(), env_ids.float() + 1.0, env_ids.float() + 2.0],
                dim=-1,
            )

    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        terrain=_Terrain(),
        simulator=_Simulator(),
    )
    obs = TerrainObs(config=SimpleNamespace(), env=env)

    obs.compute_observations(torch.tensor([0, 2]))
    result = obs.get_obs()
    result["terrain"][0, 0] = -99.0

    assert torch.equal(obs.terrain_obs[0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(obs.terrain_obs[1], torch.zeros(3))
    assert torch.equal(obs.terrain_obs[2], torch.tensor([4.0, 5.0, 6.0]))


def test_base_termination_kernels_cover_contact_height_and_threshold_paths():
    contacts = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]
    )
    progress = torch.tensor([0, 2, 3])
    allowed = torch.tensor([1])
    assert torch.equal(
        base_terms.check_fall_contact_term(contacts, allowed, progress),
        torch.tensor([False, False, True]),
    )
    assert torch.equal(
        base_terms.contact_termination(contacts, allowed, progress),
        torch.tensor([False, False, True]),
    )

    rigid_body_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.4], [0.0, 0.0, 0.1], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.8], [0.0, 0.0, 0.1], [0.0, 0.0, 0.9]],
            [[0.0, 0.0, 0.4], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
        ]
    )
    termination_heights = torch.full((3,), 0.5)
    assert torch.equal(
        base_terms.check_height_term(rigid_body_pos, termination_heights, allowed),
        torch.tensor([True, False, True]),
    )
    assert torch.equal(
        base_terms.combine_fall_termination(
            contacts,
            rigid_body_pos,
            termination_heights,
            allowed,
            progress,
        ),
        torch.tensor([False, False, True]),
    )
    assert torch.equal(
        base_terms.height_termination(
            rigid_body_pos,
            ground_heights=torch.zeros(3),
            termination_height=0.5,
            non_termination_body_ids=allowed,
        ),
        torch.tensor([True, False, True]),
    )
    assert torch.equal(
        base_terms.fall_termination(
            rigid_body_pos,
            contacts,
            ground_heights=torch.zeros(3),
            termination_height=0.5,
            non_termination_contact_body_ids=allowed,
            progress_buf=progress,
        ),
        torch.tensor([False, False, True]),
    )
    assert torch.equal(
        base_terms.check_max_length_term(torch.tensor([8, 9, 10]), 10.0),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(
        base_terms.threshold_termination(
            torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
            threshold=4.0,
            greater_than=True,
        ),
        torch.tensor([False, True]),
    )
    assert torch.equal(
        base_terms.threshold_termination(
            torch.tensor([1.0, 5.0]),
            threshold=4.0,
            greater_than=False,
        ),
        torch.tensor([True, False]),
    )


def test_task_termination_kernels_cover_path_and_steering_errors():
    progress = torch.tensor([5, 20, 20])
    head_pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    target_pos = torch.zeros(3, 3)

    assert torch.equal(
        task_terms.check_path_distance_term(
            head_pos,
            target_pos,
            fail_dist=1.0,
            progress_buf=progress,
            min_progress=10,
        ),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(
        task_terms.check_path_height_term(
            head_pos,
            target_pos,
            fail_height_dist=1.0,
            progress_buf=progress,
            min_progress=10,
        ),
        torch.tensor([False, False, True]),
    )

    assert torch.equal(
        task_terms.check_steering_velocity_error(
            root_pos=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 0.0]]
            ),
            prev_root_pos=torch.zeros(3, 3),
            tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            tar_speed=torch.tensor([1.0, 1.0, 3.0]),
            dt=1.0,
            speed_tolerance=0.25,
            direction_tolerance=0.5,
        ),
        torch.tensor([False, True, True]),
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
        tracking.anchor_pos_error_value(current_anchor_pos, ref_pos, anchor_idx=0),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(
        tracking.anchor_height_error_value(current_anchor_pos, ref_pos, anchor_idx=0),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(
        tracking.anchor_ori_error_value(current_anchor_rot, ref_rot, anchor_idx=0),
        torch.tensor([0.0, 2.0]),
        atol=1e-5,
    )
    assert torch.allclose(
        tracking.relative_body_pos_max_error(
            current_pos,
            ref_pos,
            current_anchor_pos,
            current_anchor_rot,
            ref_rot,
            anchor_idx=0,
        ),
        torch.tensor([0.3, 2.0**0.5]),
    )

    assert torch.equal(
        tracking.compute_tracking_error(current_pos, ref_pos, threshold=0.5),
        torch.tensor([False, True]),
    )
    assert torch.equal(
        tracking.compute_anchor_pos_error_term(
            current_anchor_pos, ref_pos, anchor_idx=0, threshold=0.5
        ),
        torch.tensor([False, True]),
    )
    assert torch.equal(
        tracking.compute_anchor_height_error_term(
            current_anchor_pos, ref_pos, anchor_idx=0, threshold=0.5
        ),
        torch.tensor([False, True]),
    )
    assert torch.equal(
        tracking.compute_anchor_ori_error_term(
            current_anchor_rot, ref_rot, anchor_idx=0, threshold=0.5
        ),
        torch.tensor([False, True]),
    )
    assert torch.equal(
        tracking.compute_relative_body_pos_error_term(
            current_pos,
            ref_pos,
            current_anchor_pos,
            current_anchor_rot,
            ref_rot,
            anchor_idx=0,
            threshold=0.5,
        ),
        torch.tensor([False, True]),
    )


def test_motion_clip_done_uses_motion_lib_lengths():
    class _MotionLib:
        def get_motion_length(self, motion_ids):
            return torch.tensor([1.0, 2.0, 3.0])[motion_ids]

    assert torch.equal(
        tracking.motion_clip_done(
            motion_times=torch.tensor([1.0, 1.5, 4.0]),
            motion_ids=torch.tensor([0, 1, 2]),
            motion_lib=_MotionLib(),
        ),
        torch.tensor([True, False, True]),
    )
