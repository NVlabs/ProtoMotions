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
