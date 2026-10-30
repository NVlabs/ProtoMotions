# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BakedAnchorFK.body_kinematics: root-frame poses and joint-induced velocities."""

import numpy as np
import pytest
import torch

from protomotions.tests.support.sensor_chain import (
    chain_state,
    load_chain,
    random_chain_batch,
)
from protomotions.utils import rotations as rot
from protomotions.utils.baked_fk import BakedAnchorFK


@pytest.fixture
def chain(tmp_path):
    model, info = load_chain(tmp_path)
    qpos, qvel = random_chain_batch(np.random.default_rng(5), n=4)
    # Root fixed at the origin with identity orientation and no root motion:
    # every velocity MuJoCo reports is then induced by the joints alone.
    qpos[:, :3] = 0.0
    qpos[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    qvel[:, :6] = 0.0
    return info, chain_state(model, qpos, qvel)


def test_body_kinematics_matches_mujoco_for_a_fixed_root(chain):
    info, truth = chain
    fk = BakedAnchorFK(
        info, anchor_idx=0, trackable_body_indices=torch.arange(info.num_bodies)
    )
    pos, rot_mat, omega, vel = fk.body_kinematics(truth.dof_pos, truth.dof_vel)
    torch.testing.assert_close(pos, truth.rigid_body_pos, atol=1e-5, rtol=1e-5)
    expected_rot = rot.quaternion_to_matrix(
        truth.rigid_body_rot.reshape(-1, 4), w_last=True
    ).reshape(4, -1, 3, 3)
    torch.testing.assert_close(rot_mat, expected_rot, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(omega, truth.rigid_body_ang_vel, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(vel, truth.rigid_body_vel, atol=1e-5, rtol=1e-5)


def test_body_kinematics_rejects_mismatched_joint_arrays(chain):
    info, truth = chain
    fk = BakedAnchorFK(
        info, anchor_idx=0, trackable_body_indices=torch.arange(info.num_bodies)
    )
    with pytest.raises(ValueError):
        fk.body_kinematics(truth.dof_pos, truth.dof_vel[:, :-1])


@pytest.fixture
def posed_chain(tmp_path):
    # Random root pose too: world_body_poses must place the chain from the
    # anchor's world pose alone, not assume an identity root.
    model, info = load_chain(tmp_path)
    qpos, qvel = random_chain_batch(np.random.default_rng(7), n=5)
    return info, chain_state(model, qpos, qvel)


@pytest.mark.parametrize("anchor", [0, 1])
def test_world_body_poses_reproduce_mujoco_from_the_anchor_pose(posed_chain, anchor):
    info, truth = posed_chain
    fk = BakedAnchorFK(
        info, anchor_idx=anchor, trackable_body_indices=torch.arange(info.num_bodies)
    )
    anchor_pos = truth.rigid_body_pos[:, anchor]
    anchor_rot = truth.rigid_body_rot[:, anchor]
    pos, quat = fk.world_body_poses(truth.dof_pos, anchor_pos, anchor_rot)
    assert pos.shape == truth.rigid_body_pos.shape
    assert quat.shape == truth.rigid_body_rot.shape
    torch.testing.assert_close(pos, truth.rigid_body_pos, atol=1e-5, rtol=1e-5)
    # Unit quaternions equal to MuJoCo's up to the sign.
    norms = quat.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=0)
    dots = (quat * truth.rigid_body_rot).sum(-1).abs()
    torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-5, rtol=0)
    # The anchor row is the given pose itself (no FK round trip through it).
    torch.testing.assert_close(pos[:, anchor], anchor_pos, atol=1e-6, rtol=0)


def test_world_body_poses_reject_a_wrong_joint_count(posed_chain):
    info, truth = posed_chain
    fk = BakedAnchorFK(
        info, anchor_idx=0, trackable_body_indices=torch.arange(info.num_bodies)
    )
    with pytest.raises(ValueError, match=r"needs \[B, 3\] joint angles"):
        fk.world_body_poses(
            truth.dof_pos[:, :-1], truth.rigid_body_pos[:, 0], truth.rigid_body_rot[:, 0]
        )
