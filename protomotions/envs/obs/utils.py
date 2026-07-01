# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for observation processing."""

from typing import Union, List

import torch
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


def select_step_indices(
    tensor: Tensor,
    steps: Union[int, List[int]],
    dim: int = 1
) -> Tensor:
    """Select steps from tensor by index.

    Supports both consecutive steps (int) and arbitrary step indices (list).
    Uses 1-indexed step numbers that are converted to 0-indexed tensor positions.

    Args:
        tensor: Input tensor with steps along dim.
        steps: If int N, selects first N steps (like arange(1, N+1)).
               If list, selects specific 1-indexed steps (e.g., [1, 3, 5] -> indices [0, 2, 4]).
        dim: Dimension containing steps.

    Returns:
        Tensor with selected steps.
    """
    if isinstance(steps, int):
        return tensor.narrow(dim, 0, steps)
    else:
        indices = torch.tensor([s - 1 for s in steps], device=tensor.device, dtype=torch.long)
        return tensor.index_select(dim, indices)
