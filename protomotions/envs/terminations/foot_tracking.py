# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foot tracking relative to the robot's anchor XY and reference anchor height."""

from typing import List

import torch
from torch import Tensor

from protomotions.utils.rotations import (
    calc_heading_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)


def aligned_foot_position_termination(
    current_position: Tensor,
    reference_position: Tensor,
    current_rotation: Tensor,
    reference_rotation: Tensor,
    body_indices: List[int],
    anchor_index: int,
    distance_limit: float = 0.2,
) -> Tensor:
    """Return [N] failures for any selected foot exceeding the XYZ limit.

    Positions are [N, B, 3], rotations are [N, B, 4] XYZW quaternions.
    Match the reference tracker: take yaw from the relative anchor rotation,
    align anchor XY, and retain reference world Z. This does not measure
    absolute world-XY tracking; the global body check supplies that constraint.
    """
    anchor_pos = current_position[:, anchor_index].clone()
    anchor_pos[:, 2] = reference_position[:, anchor_index, 2]
    delta_heading = calc_heading_quat(
        quat_mul(
            current_rotation[:, anchor_index],
            quat_conjugate(reference_rotation[:, anchor_index], w_last=True),
            w_last=True,
        ),
        w_last=True,
    )
    offsets = (
        reference_position[:, body_indices] - reference_position[:, anchor_index, None]
    )
    aligned_reference = anchor_pos[:, None] + quat_rotate(
        delta_heading[:, None].expand(-1, len(body_indices), -1).reshape(-1, 4),
        offsets.reshape(-1, 3),
        w_last=True,
    ).reshape_as(offsets)
    errors = torch.linalg.vector_norm(
        aligned_reference - current_position[:, body_indices], dim=-1
    )
    return (errors > distance_limit).any(dim=-1)
