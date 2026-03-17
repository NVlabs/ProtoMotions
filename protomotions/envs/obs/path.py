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
"""Observation compute kernel for path following tasks.

Pure tensor function (kernel) for computing path following observations.
Use MdpComponent in experiment configs to bind kernel to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.path import compute_path_obs
    
    observation_components = {
        "path": MdpComponent(
            compute_func=compute_path_obs,
            dynamic_vars={
                "root_rot": EnvContext.current.root_rot,
                "head_pos": EnvContext.path.head_pos,
                "traj_samples": EnvContext.path.traj_samples,
                "height_conditioned": EnvContext.path.height_conditioned,
            },
        ),
    }
"""

import torch
from torch import Tensor

from protomotions.utils import rotations


def compute_path_obs(
    root_rot: Tensor,
    head_pos: Tensor,
    traj_samples: Tensor,
    height_conditioned: bool,
) -> Tensor:
    """Compute path observations in the robot's local frame.

    Transforms the future waypoints from world frame to the robot's local frame.

    Args:
        root_rot: Root orientation quaternions [num_envs, 4] (w-last format).
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
