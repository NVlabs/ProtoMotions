# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical reset perturbations, not measurement noise.

These draws change the initial simulator state before sensors are read.
Observation noise belongs to the sensor measurement model.
Keep the streams and responsibilities separate: reset perturbations use the
global Torch RNG, while sensor noise uses its isolated generator.
"""

from typing import List, Optional, Union

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.config import RobotNoiseConfig
from protomotions.simulator.base_simulator.simulator_state import ResetState
from protomotions.utils import rotations


def _uniform_like(tensor: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    """``torch.rand`` of ``tensor``'s shape, device and dtype, on ``generator``
    (the draws ``rand_like`` made, with the stream named)."""
    return torch.rand(
        tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator
    )


def _add_noise(
    tensor: Tensor,
    noise_scale: Union[float, List[float]],
    generator: Optional[torch.Generator],
) -> Tensor:
    """Add uniform noise to a tensor.

    Args:
        tensor: Input tensor of any shape.
        noise_scale: Scalar or per-axis list. When a list, it is broadcast
            against the last dimension of *tensor*.
        generator: The reset-noise stage's generator (None: the global stream).
    """
    if isinstance(noise_scale, list):
        scale_t = torch.tensor(noise_scale, device=tensor.device, dtype=tensor.dtype)
        if scale_t.any():
            return tensor + (_uniform_like(tensor, generator) * 2.0 - 1.0) * scale_t
        return tensor
    if noise_scale > 0.0:
        return tensor + (_uniform_like(tensor, generator) * 2.0 - 1.0) * noise_scale
    return tensor


def _add_euler_rotation_noise(
    quat: Tensor,
    noise_scale: Union[float, List[float]],
    generator: Optional[torch.Generator],
) -> Tensor:
    """Perturb quaternions by sampling uniform euler angle noise and composing.

    Args:
        quat: Quaternions [N, 4] in xyzw convention (w_last=True).
        noise_scale: Scalar or [roll, pitch, yaw] noise scales in radians.
        generator: The reset-noise stage's generator (None: the global stream).
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
        torch.rand(n, 3, device=quat.device, dtype=quat.dtype, generator=generator)
        * 2.0
        - 1.0
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
    generator: Optional[torch.Generator],
) -> None:
    """Apply Reference State Initialization (RSI) noise to a reset state in-place.

    Adds uniform noise to root position, rotation, velocities, and DOF
    states to help the policy learn recovery from imperfect initial conditions.

    Args:
        reset_state: The reset state to perturb (modified in-place).
        config: Noise configuration with per-field scales.
        dof_limits_lower: Lower DOF limits [num_dof] for clamping.
        dof_limits_upper: Upper DOF limits [num_dof] for clamping.
        generator: The reset-noise stage's generator (None: the global stream).
    """
    # DOF noise
    reset_state.dof_pos = _add_noise(
        reset_state.dof_pos, config.dof_pos_noise, generator
    )
    reset_state.dof_pos = torch.clamp(
        reset_state.dof_pos, min=dof_limits_lower, max=dof_limits_upper
    )
    reset_state.dof_vel = _add_noise(
        reset_state.dof_vel, config.dof_vel_noise, generator
    )

    # Root position noise
    reset_state.root_pos = _add_noise(
        reset_state.root_pos, config.root_pos_noise, generator
    )

    # Root rotation noise (euler perturbation composed with original quat)
    reset_state.root_rot = _add_euler_rotation_noise(
        reset_state.root_rot, config.root_rot_noise, generator
    )

    # Root velocity noise
    reset_state.root_vel = _add_noise(
        reset_state.root_vel, config.root_vel_noise, generator
    )
    reset_state.root_ang_vel = _add_noise(
        reset_state.root_ang_vel, config.root_ang_vel_noise, generator
    )
