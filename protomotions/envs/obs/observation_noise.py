# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Observation noise utilities for domain randomization.

Provides functions to apply configurable noise to robot state observations
for sim-to-real transfer training.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.config import RobotNoiseConfig
from protomotions.simulator.base_simulator.simulator_state import ResetState, RobotState
from protomotions.utils import rotations
from protomotions.envs.obs.humanoid import compute_local_ang_vel


@dataclass
class NoisyObservations:
    """Container for noisy observation tensors.

    When noise is not configured, these point to the same tensors as clean observations.
    """

    # Whole-body
    rigid_body_pos: Tensor
    rigid_body_rot: Tensor
    rigid_body_vel: Tensor
    rigid_body_ang_vel: Tensor

    # DOF
    dof_pos: Tensor
    dof_vel: Tensor

    # Root
    root_rot: Tensor
    root_local_ang_vel: Tensor

    # Anchor
    anchor_rot: Tensor
    anchor_local_ang_vel: Tensor

    # Ground height
    ground_heights: Tensor

    # Computed root properties (matching RobotState interface)
    @property
    def root_pos(self) -> Tensor:
        """Root body position [num_envs, 3]."""
        return self.rigid_body_pos[:, 0, :]

    @property
    def root_vel(self) -> Tensor:
        """Root body linear velocity [num_envs, 3]."""
        return self.rigid_body_vel[:, 0, :]

    @property
    def root_ang_vel(self) -> Tensor:
        """Root body angular velocity in world frame [num_envs, 3]."""
        return self.rigid_body_ang_vel[:, 0, :]

    def clone(self) -> "NoisyObservations":
        """Return a deep copy with all tensors cloned."""
        return NoisyObservations(
            rigid_body_pos=self.rigid_body_pos.clone(),
            rigid_body_rot=self.rigid_body_rot.clone(),
            rigid_body_vel=self.rigid_body_vel.clone(),
            rigid_body_ang_vel=self.rigid_body_ang_vel.clone(),
            dof_pos=self.dof_pos.clone(),
            dof_vel=self.dof_vel.clone(),
            root_rot=self.root_rot.clone(),
            root_local_ang_vel=self.root_local_ang_vel.clone(),
            anchor_rot=self.anchor_rot.clone(),
            anchor_local_ang_vel=self.anchor_local_ang_vel.clone(),
            ground_heights=self.ground_heights.clone(),
        )

    def update_subset(self, env_ids: Tensor, other: "NoisyObservations") -> None:
        """Update specific environment rows in-place from another NoisyObservations.

        Args:
            env_ids: Environment indices to update.
            other: NoisyObservations with tensors shaped [len(env_ids), ...].
        """
        self.rigid_body_pos[env_ids] = other.rigid_body_pos
        self.rigid_body_rot[env_ids] = other.rigid_body_rot
        self.rigid_body_vel[env_ids] = other.rigid_body_vel
        self.rigid_body_ang_vel[env_ids] = other.rigid_body_ang_vel
        self.dof_pos[env_ids] = other.dof_pos
        self.dof_vel[env_ids] = other.dof_vel
        self.root_rot[env_ids] = other.root_rot
        self.root_local_ang_vel[env_ids] = other.root_local_ang_vel
        self.anchor_rot[env_ids] = other.anchor_rot
        self.anchor_local_ang_vel[env_ids] = other.anchor_local_ang_vel
        self.ground_heights[env_ids] = other.ground_heights


def apply_observation_noise(
    obs_noise_cfg: Optional[RobotNoiseConfig],
    robot_state: RobotState,
    anchor_idx: int,
    ground_heights: Tensor,
    env_ids: Optional[Tensor] = None,
) -> NoisyObservations:
    """Apply observation noise to robot state.

    Derives root/anchor local angular velocities internally from robot_state
    and anchor_idx, ensuring a single source of truth for these computations.

    Noise is applied independently to each component from clean values,
    not cascaded (e.g., root noise is applied to clean root, not noisy whole-body).

    Args:
        obs_noise_cfg: Observation noise configuration. If None, returns clean tensors.
        robot_state: Clean robot state from simulator [num_envs, ...].
        anchor_idx: Index of the anchor body.
        ground_heights: Clean ground heights [N, 1].
        env_ids: Optional environment indices. When provided, robot_state is
            sliced by env_ids internally and ground_heights must already be
            shaped [len(env_ids), ...]. Returns subset-sized NoisyObservations.

    Returns:
        NoisyObservations containing potentially noisy tensors.
        When obs_noise_cfg is None, returns references to the clean tensors.
    """
    # Slice robot_state if env_ids provided; ground_heights is pre-sliced by caller
    if env_ids is not None:
        robot_state = robot_state[env_ids]

    # Derive root/anchor local angular velocities from robot_state
    root_local_ang_vel = compute_local_ang_vel(
        robot_state.root_rot,
        robot_state.rigid_body_ang_vel[:, 0, :],
    )
    anchor_rot = robot_state.rigid_body_rot[:, anchor_idx, :]
    anchor_local_ang_vel = compute_local_ang_vel(
        anchor_rot,
        robot_state.rigid_body_ang_vel[:, anchor_idx, :],
    )

    if obs_noise_cfg is None:
        # No noise - return clean tensors
        return NoisyObservations(
            rigid_body_pos=robot_state.rigid_body_pos,
            rigid_body_rot=robot_state.rigid_body_rot,
            rigid_body_vel=robot_state.rigid_body_vel,
            rigid_body_ang_vel=robot_state.rigid_body_ang_vel,
            dof_pos=robot_state.dof_pos,
            dof_vel=robot_state.dof_vel,
            root_rot=robot_state.root_rot,
            root_local_ang_vel=root_local_ang_vel,
            anchor_rot=anchor_rot,
            anchor_local_ang_vel=anchor_local_ang_vel,
            ground_heights=ground_heights,
        )

    noisy_rigid_body_pos = _add_noise(
        robot_state.rigid_body_pos, obs_noise_cfg.body_pos_noise
    )
    noisy_rigid_body_rot = _add_quaternion_noise(
        robot_state.rigid_body_rot, obs_noise_cfg.body_rot_noise
    )
    noisy_rigid_body_vel = _add_noise(
        robot_state.rigid_body_vel, obs_noise_cfg.body_vel_noise
    )
    noisy_rigid_body_ang_vel = _add_noise(
        robot_state.rigid_body_ang_vel, obs_noise_cfg.body_ang_vel_noise
    )
    noisy_dof_pos = _add_noise(robot_state.dof_pos, obs_noise_cfg.dof_pos_noise)
    noisy_dof_vel = _add_noise(robot_state.dof_vel, obs_noise_cfg.dof_vel_noise)
    noisy_root_rot = _add_quaternion_noise(
        robot_state.root_rot, obs_noise_cfg.root_rot_noise
    )
    noisy_root_local_ang_vel = _add_noise(
        root_local_ang_vel, obs_noise_cfg.root_ang_vel_noise
    )
    noisy_anchor_rot = _add_quaternion_noise(anchor_rot, obs_noise_cfg.anchor_rot_noise)
    noisy_anchor_local_ang_vel = _add_noise(
        anchor_local_ang_vel, obs_noise_cfg.anchor_ang_vel_noise
    )
    noisy_ground_heights = _add_noise(ground_heights, obs_noise_cfg.ground_height_noise)

    return NoisyObservations(
        rigid_body_pos=noisy_rigid_body_pos,
        rigid_body_rot=noisy_rigid_body_rot,
        rigid_body_vel=noisy_rigid_body_vel,
        rigid_body_ang_vel=noisy_rigid_body_ang_vel,
        dof_pos=noisy_dof_pos,
        dof_vel=noisy_dof_vel,
        root_rot=noisy_root_rot,
        root_local_ang_vel=noisy_root_local_ang_vel,
        anchor_rot=noisy_anchor_rot,
        anchor_local_ang_vel=noisy_anchor_local_ang_vel,
        ground_heights=noisy_ground_heights,
    )


def _add_noise(tensor: Tensor, noise_scale: Union[float, List[float]]) -> Tensor:
    """Add uniform noise to a tensor.

    Args:
        tensor: Input tensor of any shape.
        noise_scale: Scalar or per-axis list. When a list, it is broadcast
            against the last dimension of *tensor*.
    """
    if isinstance(noise_scale, list):
        scale_t = torch.tensor(noise_scale, device=tensor.device, dtype=tensor.dtype)
        if scale_t.any():
            return tensor + (torch.rand_like(tensor) * 2.0 - 1.0) * scale_t
        return tensor
    if noise_scale > 0.0:
        return tensor + (torch.rand_like(tensor) * 2.0 - 1.0) * noise_scale
    return tensor


def _add_quaternion_noise(
    quat: Tensor, noise_scale: Union[float, List[float]]
) -> Tensor:
    """Add uniform noise to quaternions and re-normalize."""
    if isinstance(noise_scale, list):
        scale_t = torch.tensor(noise_scale, device=quat.device, dtype=quat.dtype)
        if scale_t.any():
            noisy_quat = quat + (torch.rand_like(quat) * 2.0 - 1.0) * scale_t
            return noisy_quat / torch.norm(noisy_quat, dim=-1, keepdim=True)
        return quat
    if noise_scale > 0.0:
        sampled_noise = (torch.rand_like(quat) * 2.0 - 1.0) * noise_scale
        noisy_quat = quat + sampled_noise
        return noisy_quat / torch.norm(noisy_quat, dim=-1, keepdim=True)
    return quat


def _add_euler_rotation_noise(
    quat: Tensor, noise_scale: Union[float, List[float]]
) -> Tensor:
    """Perturb quaternions by sampling uniform euler angle noise and composing.

    Args:
        quat: Quaternions [N, 4] in xyzw convention (w_last=True).
        noise_scale: Scalar or [roll, pitch, yaw] noise scales in radians.
    """
    if isinstance(noise_scale, list):
        scale_t = torch.tensor(noise_scale, device=quat.device, dtype=quat.dtype)
        if not scale_t.any():
            return quat
    else:
        if noise_scale == 0.0:
            return quat
        scale_t = noise_scale

    n = quat.shape[0]
    rpy_noise = (
        torch.rand(n, 3, device=quat.device, dtype=quat.dtype) * 2.0 - 1.0
    ) * scale_t
    noise_quat = rotations.quat_from_euler_xyz(
        rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2], w_last=True
    )
    return rotations.quat_mul(noise_quat, quat, w_last=True)


def apply_reset_noise(
    reset_state: ResetState,
    config: RobotNoiseConfig,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
) -> None:
    """Apply Reference State Initialization (RSI) noise to a reset state in-place.

    Adds uniform noise to root position, rotation, velocities, and DOF
    states to help the policy learn recovery from imperfect initial conditions.

    Args:
        reset_state: The reset state to perturb (modified in-place).
        config: Noise configuration with per-field scales.
        dof_limits_lower: Lower DOF limits [num_dof] for clamping.
        dof_limits_upper: Upper DOF limits [num_dof] for clamping.
    """
    # DOF noise
    reset_state.dof_pos = _add_noise(reset_state.dof_pos, config.dof_pos_noise)
    reset_state.dof_pos = torch.clamp(
        reset_state.dof_pos, min=dof_limits_lower, max=dof_limits_upper
    )
    reset_state.dof_vel = _add_noise(reset_state.dof_vel, config.dof_vel_noise)

    # Root position noise
    reset_state.root_pos = _add_noise(reset_state.root_pos, config.root_pos_noise)

    # Root rotation noise (euler perturbation composed with original quat)
    reset_state.root_rot = _add_euler_rotation_noise(
        reset_state.root_rot, config.root_rot_noise
    )

    # Root velocity noise
    reset_state.root_vel = _add_noise(reset_state.root_vel, config.root_vel_noise)
    reset_state.root_ang_vel = _add_noise(
        reset_state.root_ang_vel, config.root_ang_vel_noise
    )
