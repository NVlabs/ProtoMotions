# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immediate XY velocity commands, separate from timed pose targets.

A recipe binds the command and its mask as two components::

    "masked_mimic_velocity_command": MdpComponent.from_kernel(
        compute_masked_mimic_velocity_command_obs
    ),
    "masked_mimic_velocity_command_mask": MdpComponent.from_kernel(
        to_float, x=EnvContext.masked_mimic.velocity_command_mask
    ),

The anchor rotation is the privileged one by default; ``sensed(...)`` moves it
to a sensor source.
"""

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.utils import rotations


def compute_masked_mimic_velocity_command_obs(
    anchor_rot: Tensor = EnvContext.current.anchor_rot,
    average_velocity: Tensor = EnvContext.masked_mimic.average_velocity,
    velocity_command_mask: Tensor = EnvContext.masked_mimic.velocity_command_mask,
) -> Tensor:
    """Return the heading-relative XY velocity, zeroed when hidden, shape [envs, 2]."""
    heading_inv = rotations.calc_heading_quat_inv(anchor_rot, w_last=True)
    velocity = torch.cat(
        (average_velocity, torch.zeros_like(average_velocity[:, :1])), -1
    )
    local_velocity = rotations.quat_rotate(heading_inv, velocity, w_last=True)[:, :2]
    return torch.where(
        velocity_command_mask.bool(), local_velocity, torch.zeros_like(local_velocity)
    )
