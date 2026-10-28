# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific scalar termination kernels (path following, steering).

Each kernel is a scalar measurement; a recipe cuts it with a
``TerminationRule(kernel, threshold, fail_above=...)`` and may place the same
rule in the evaluator's failure components. There is no boolean check helper:
a termination is one rule over a scalar kernel, shared with evaluation.
"""

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext


def path_distance(
    head_pos: Tensor = EnvContext.path.head_pos,
    target_pos: Tensor = EnvContext.path.tar_pos,
    progress_buf: Tensor = EnvContext.progress_buf,
    min_progress: int = 10,
) -> Tensor:
    """3D distance from the head to the target path position, in metres.

    Zero while ``progress_buf <= min_progress`` (the check is off early, as the
    old ``progress_buf > min_progress`` gate was). ``TerminationRule(path_distance,
    fail_dist)`` equals the old ``d**2 > fail_dist**2`` for nonnegative ``d``.
    """
    distance = (target_pos - head_pos).pow(2).sum(dim=-1).sqrt()
    return torch.where(
        progress_buf > min_progress, distance, torch.zeros_like(distance)
    )


def path_height_error(
    head_pos: Tensor = EnvContext.path.head_pos,
    target_pos: Tensor = EnvContext.path.tar_pos,
    progress_buf: Tensor = EnvContext.progress_buf,
    min_progress: int = 10,
) -> Tensor:
    """Absolute height (Z) deviation from the target path; zero early."""
    height_error = (target_pos[..., 2] - head_pos[..., 2]).abs()
    return torch.where(
        progress_buf > min_progress, height_error, torch.zeros_like(height_error)
    )


def steering_speed_error(
    root_pos: Tensor = EnvContext.current.anchor_pos,
    prev_root_pos: Tensor = EnvContext.steering.prev_root_pos,
    tar_speed: Tensor = EnvContext.steering.tar_speed,
    dt: float = EnvContext.dt,
) -> Tensor:
    """``|planar speed - target speed|``: the speed magnitude today's check compares.

    The planar speed is ``|(root_pos - prev_root_pos)[:, :2] / dt|``; the velocity
    projected onto ``tar_dir`` belongs to the steering reward, not this check.
    """
    vel_xy = (root_pos - prev_root_pos)[:, :2] / dt
    speed = torch.norm(vel_xy, dim=-1)
    return torch.abs(speed - tar_speed)


def steering_direction_alignment(
    root_pos: Tensor = EnvContext.current.anchor_pos,
    prev_root_pos: Tensor = EnvContext.steering.prev_root_pos,
    tar_dir: Tensor = EnvContext.steering.tar_dir,
    dt: float = EnvContext.dt,
) -> Tensor:
    """Dot of the planar heading with the target direction.

    ``1.0`` (a passing value for a ``fail_above=False`` rule) where the robot is
    nearly stationary (``|v_xy| <= 0.1``), since today's guard only fails a moving
    robot. ``TerminationRule(steering_speed_error, speed_tol)`` OR
    ``TerminationRule(steering_direction_alignment, dir_tol, fail_above=False)``
    equals the old steering boolean on every input.
    """
    vel_xy = (root_pos - prev_root_pos)[:, :2] / dt
    speed = torch.norm(vel_xy, dim=-1)
    vel_dir = vel_xy / (speed.unsqueeze(-1) + 1e-6)
    alignment = (vel_dir * tar_dir).sum(dim=-1)
    return torch.where(speed > 0.1, alignment, torch.ones_like(alignment))
