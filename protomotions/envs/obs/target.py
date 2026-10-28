# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation compute kernel for target-reaching tasks.

Bound in one line, ``MdpComponent.from_kernel(compute_target_obs)``: the
anchor body gives the position and heading, the target control the goal.
"""

from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.obs.utils import heading_local_xy_delta


def compute_target_obs(
    root_pos: Tensor = EnvContext.current.anchor_pos,
    root_rot: Tensor = EnvContext.current.anchor_rot,
    tar_pos: Tensor = EnvContext.target.tar_pos,
) -> Tensor:
    """Return target XY offset in the heading frame of the anchor body."""
    return heading_local_xy_delta(root_pos, root_rot, tar_pos, w_last=True)


__all__ = ["compute_target_obs"]
