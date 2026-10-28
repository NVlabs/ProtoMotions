# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation compute kernel for path following tasks.

A pure tensor function whose parameter defaults are its context bindings: the
heading comes from the anchor body, the waypoints from the path control's
namespace. A recipe binds it in one line::

    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.path import compute_path_obs

    observation_components = {
        "path": MdpComponent.from_kernel(compute_path_obs),
    }
"""

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.utils import rotations


def compute_path_obs(
    root_rot: Tensor = EnvContext.current.anchor_rot,
    head_pos: Tensor = EnvContext.path.head_pos,
    traj_samples: Tensor = EnvContext.path.traj_samples,
    height_conditioned: bool = EnvContext.path.height_conditioned,
) -> Tensor:
    """Compute path observations in the robot's local frame.

    Transforms the future waypoints from world frame to the robot's local frame.

    Args:
        root_rot: Heading-giving orientation quaternions [num_envs, 4] (w-last
            format); the anchor body's rotation by default.
        head_pos: Head positions [num_envs, 3] in ground-relative frame.
        traj_samples: Future waypoint positions [num_envs, num_samples, 3].
        height_conditioned: Whether to include height in observations.

    Returns:
        Path observations [num_envs, num_samples * (2 or 3)].
    """
    heading_rot = rotations.calc_heading_quat_inv(root_rot, True)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )

    traj_samples_delta = traj_samples - head_pos.unsqueeze(-2)

    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )

    local_traj_pos = rotations.quat_rotate(
        heading_rot_exp, traj_samples_delta_flat, True
    )
    if not height_conditioned:
        local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs


__all__ = ["compute_path_obs"]
