# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
"""Stateless observation functions for path following tasks.

These functions compute observations from path control context and can be used
with the ObservationComponentConfig system.
"""

import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.envs.obs.observation_component import ObservationComponentConfig


def path_obs_factory() -> ObservationComponentConfig:
    """Factory for path following observation component.

    Returns an ObservationComponentConfig that computes path observations
    (future waypoints in local frame).

    Returns:
        ObservationComponentConfig for path observations.
    """
    return ObservationComponentConfig(
        function=compute_path_obs,
        variables={
            "root_rot": "current_state_root_rot",
            "head_pos": "head_pos",  # From PathFollowerControl.get_context()
            "traj_samples": "traj_samples",  # From PathFollowerControl.get_context()
            "height_conditioned": "height_conditioned",  # From PathFollowerControl.get_context()
        },
    )


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

