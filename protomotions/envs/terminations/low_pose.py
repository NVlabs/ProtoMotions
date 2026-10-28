# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-height-gated tracking limits; low poses ignore foot XY only."""

from typing import List, Optional

import torch
from torch import Tensor

from protomotions.envs.terminations.foot_tracking import (
    aligned_foot_position_termination,
)


def tracking_failure_ratio(
    current_position: Tensor,
    reference_position: Tensor,
    smoothed_ref_anchor_height: Tensor,
    upper_body_indices: List[int],
    foot_indices: List[int],
    upright_limit: float,
    upright_body_indices: Optional[List[int]] = None,
    upright_mean: bool = False,
    low_height: float = 0.5,
    low_upper_limit: float = 0.5,
    low_foot_z_limit: float = 0.3,
) -> Tensor:
    """Return [N] error/limit ratios; values strictly greater than one fail.

    Positions are [N, B, 3] in the same world frame, heights are [N]. Limits
    are positive distances in meters; upper-body and foot selections are
    nonempty. Height comes from the reference, never the falling robot.
    Upright checks retain the selected bodies and mean/max reduction. In low
    poses, check maximum upper-body XYZ and foot Z errors independently.
    """
    delta = reference_position - current_position
    selected = delta if upright_body_indices is None else delta[:, upright_body_indices]
    distances = torch.linalg.vector_norm(selected, dim=-1)
    upright = (
        distances.mean(-1) if upright_mean else distances.amax(-1)
    ) / upright_limit
    upper = (
        torch.linalg.vector_norm(delta[:, upper_body_indices], dim=-1).amax(-1)
        / low_upper_limit
    )
    feet = delta[:, foot_indices, 2].abs().amax(-1) / low_foot_z_limit
    return torch.where(
        smoothed_ref_anchor_height < low_height, torch.maximum(upper, feet), upright
    )


def upright_foot_position_failed(
    current_position: Tensor,
    reference_position: Tensor,
    current_rotation: Tensor,
    reference_rotation: Tensor,
    smoothed_ref_anchor_height: Tensor,
    body_indices: List[int],
    anchor_index: int,
    distance_limit: float = 0.2,
    low_height: float = 0.5,
) -> Tensor:
    """Apply aligned foot XYZ limits only to upright reference poses."""
    failed = aligned_foot_position_termination(
        current_position,
        reference_position,
        current_rotation,
        reference_rotation,
        body_indices,
        anchor_index,
        distance_limit,
    )
    return failed & (smoothed_ref_anchor_height >= low_height)


def upright_foot_height_failed(
    current_position: Tensor,
    reference_position: Tensor,
    smoothed_ref_anchor_height: Tensor,
    body_indices: List[int],
    height_limit: float = 0.15,
    low_height: float = 0.5,
) -> Tensor:
    """Apply world-Z foot limits only to upright reference poses."""
    error = (
        current_position[:, body_indices, 2] - reference_position[:, body_indices, 2]
    ).abs()
    return (error > height_limit).any(-1) & (smoothed_ref_anchor_height >= low_height)
