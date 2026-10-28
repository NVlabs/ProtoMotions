# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for observation processing."""

from torch import Tensor

from protomotions.utils import rotations


def heading_local_xy_delta(
    origin_pos: Tensor,
    origin_rot: Tensor,
    target_pos: Tensor,
    w_last: bool = True,
) -> Tensor:
    """Return target XY displacement in the origin heading frame.

    The vertical component is intentionally discarded before rotation so this
    helper can be shared by target-reaching observations and odometer-style
    target-pose offsets.
    """
    rel_pos = target_pos - origin_pos
    rel_pos = rel_pos.clone()
    rel_pos[..., 2] = 0.0
    heading_inv = rotations.calc_heading_quat_inv(origin_rot, w_last)
    return rotations.quat_rotate(heading_inv, rel_pos, w_last)[..., :2]
