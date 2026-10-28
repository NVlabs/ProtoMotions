# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacGym reset FK, including Gym's coupled three-DOF joint convention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_transforms_from_qpos_non_root,
    matrix_to_quaternion,
)
from protomotions.utils import rotations

if TYPE_CHECKING:
    from protomotions.components.pose_lib import KinematicInfo
    from protomotions.simulator.base_simulator.simulator_state import ResetState


def reset_body_state(
    kinematic_info: KinematicInfo, state: ResetState
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return common-order link poses and origin velocities without advancing time.

    Gym represents a three-axis joint by an exponential-map position and a
    child-body-frame angular velocity, not three sequential hinge coordinates.
    All Gym-specific interpretation stays here, outside sensor/observation code.
    """
    ki = kinematic_info
    local = extract_transforms_from_qpos_non_root(
        ki, state.dof_pos, qpos_is_exp_map_on_3dof_joints=True
    )
    local[:, 0] = rotations.quaternion_to_matrix(state.root_rot, w_last=True)
    pos, mats = compute_forward_kinematics_from_transforms(ki, state.root_pos, local)
    ref = ki.local_rot_ref_mat.to(state.dof_pos)
    local_omega = {}
    start = 0
    for body, axes in ki.hinge_axes_map.items():
        count = len(axes)
        if count == 3:
            local_omega[body] = state.dof_vel[:, start : start + 3]
        elif count == 1:
            local_omega[body] = state.dof_vel[:, start : start + 1] * axes[0].to(
                state.dof_pos
            )
        else:
            raise ValueError("IsaacGym reset FK supports one- or three-DOF joints")
        start += count
    angular, velocity = [state.root_ang_vel], [state.root_vel]
    for body in range(1, ki.num_bodies):
        parent = ki.parent_indices[body]
        relative = local_omega.get(body)
        omega = angular[parent]
        if relative is not None:
            frame = (
                mats[:, body]
                if len(ki.hinge_axes_map[body]) == 3
                else mats[:, parent] @ ref[body]
            )
            omega = omega + (frame @ relative[..., None]).squeeze(-1)
        angular.append(omega)
        velocity.append(
            velocity[parent]
            + torch.cross(angular[parent], pos[:, body] - pos[:, parent], dim=-1)
        )
    return (
        pos,
        matrix_to_quaternion(mats, w_last=True),
        torch.stack(velocity, 1),
        torch.stack(angular, 1),
    )
