# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def compute_contact_labels_from_pos_and_vel(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    vel_thres: float = 0.1,
    height_thresh: float = 0.08,
) -> torch.Tensor:
    """Compute contact labels for all bodies using heuristics combining body height and velocities.

    Args:
        positions (torch.Tensor): [T, N_bodies, 3] global body positions
        velocity (torch.Tensor): [T, N_bodies, 3] velocities, already multiplied by 1 / dt
        vel_thres (float): threshold for body velocity (default: 0.15 m/s)
        height_thresh (float): threshold for body height (default: 0.1 m)

    Returns:
        torch.Tensor: [T, N_bodies] contact labels, 1 for body contact with ground
    """
    # Compute velocity magnitude for each body
    body_vel_magnitude = torch.linalg.norm(velocity, dim=-1)  # [T, N_bodies]

    # Get height (z-coordinate) for each body
    body_heights = positions[:, :, 2]  # [T, N_bodies] - assuming z is up

    # Contact occurs when both velocity is low AND height is low
    contacts = torch.logical_and(
        body_vel_magnitude < vel_thres,
        body_heights < height_thresh,
    ).to(positions.dtype)

    return contacts
