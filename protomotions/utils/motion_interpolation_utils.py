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
"""Motion interpolation utilities.

Provides functions for smoothly interpolating between motion frames,
including linear position interpolation and spherical quaternion interpolation (SLERP).
"""

import torch
from protomotions.utils import rotations


def interpolate_pos(pos0, pos1, blend):
    """Linear interpolation between two position tensors.

    Args:
        pos0: Starting positions [batch, ...] or [batch, bodies, 3]
        pos1: Ending positions [batch, ...] or [batch, bodies, 3]
        blend: Blend factor [batch] where 0=pos0, 1=pos1

    Returns:
        Interpolated positions with same shape as pos0/pos1
    """

    if pos1.dim() == 2:
        blend = blend.unsqueeze(-1)
    elif pos1.dim() == 3:
        blend = blend.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"pos1 has {pos1.dim()} dimensions, expected 2 or 3")

    return (1.0 - blend) * pos0 + blend * pos1


def interpolate_quat(rot0, rot1, blend):
    """Spherical linear interpolation (SLERP) between quaternions.

    Args:
        rot0: Starting quaternions [batch, 4] or [batch, bodies, 4]
        rot1: Ending quaternions [batch, 4] or [batch, bodies, 4]
        blend: Blend factor [batch] where 0=rot0, 1=rot1

    Returns:
        Interpolated quaternions with same shape as rot0/rot1
    """
    if rot1.dim() == 2:
        blend = blend.unsqueeze(-1)
    elif rot1.dim() == 3:
        blend = blend.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"rot1 has {rot1.dim()} dimensions, expected 2 or 3")

    return rotations.slerp(rot0, rot1, blend)


def calc_frame_blend(time, length, num_frames, dt):
    """
    Calculate frame indices and blend factor for interpolation.

    Args:
        time (torch.Tensor): Current time.
        length (torch.Tensor): Length of the motion sequence in seconds.
        num_frames (torch.Tensor): Number of frames in the motion sequence.
        dt (torch.Tensor): Time step between frames.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Frame index 0, frame index 1, and blend factor.
    """
    phase = time / length
    phase = torch.clip(phase, 0.0, 1.0)

    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0 * dt) / dt

    return frame_idx0, frame_idx1, blend
