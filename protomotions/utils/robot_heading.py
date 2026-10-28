# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot anatomical heading helpers.

The generic quaternion heading utilities in :mod:`protomotions.utils.rotations`
define yaw by rotating local ``+X``. That convention is pretrained-policy
critical for observations and relative tracking terms. These helpers answer a
separate question: which local robot axis is anatomical forward?

The semantic forward axis is xy-only by design. ProtoMotions assumes Z-up
robots, so heading is a horizontal direction; upright standing orientation is a
separate asset/configuration responsibility. Keeping the axis 2D avoids needing
to re-orient each robot into a standing pose before heading utilities can be
used.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from protomotions.utils.rotations import calc_heading_quat, heading_to_quat, quat_rotate


def semantic_forward_axis_tensor(
    semantic_forward_axis_xy: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return a normalized robot-local anatomical forward axis.

    The axis is intentionally 2D in robot-local xy coordinates. ProtoMotions
    assumes Z-up humanoids, so this value controls horizontal heading only and
    does not encode pitch/roll or how to rotate an asset into an upright
    standing pose.

    Every robot config declares the axis (``RobotConfig.__post_init__`` refuses
    ``None``, and a saved robot subtree without the field fails to load), so
    ``None`` here is a caller's mistake and is refused with the same sentence.
    """
    if semantic_forward_axis_xy is None:
        raise ValueError(
            "RobotConfig.semantic_forward_axis_xy must be declared explicitly "
            "as the robot-local xy axis that points in the anatomical "
            "forward direction, e.g. (1.0, 0.0) or (0.0, -1.0). Run `python "
            "scripts/identify_robot_facing_axis.py --robots <robot-name> "
            "--view` to identify and visualize it."
        )

    axis = torch.as_tensor(semantic_forward_axis_xy, device=device, dtype=dtype)
    if axis.shape[-1] != 2:
        raise ValueError(
            "semantic_forward_axis_xy must have shape (..., 2). The axis is "
            "xy-only because ProtoMotions assumes Z-up robots and uses it for "
            "horizontal heading, not upright pose re-orientation."
        )
    norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return axis / norm


def semantic_forward_xy(
    quat: Tensor,
    semantic_forward_axis_xy: Any,
    *,
    w_last: bool,
) -> Tensor:
    """Project a robot's anatomical forward axis into world xy.

    Only the yaw component of ``quat`` is used, matching steering/path tasks
    that compare heading without penalizing pitch or roll.
    """
    axis = semantic_forward_axis_tensor(
        semantic_forward_axis_xy,
        device=quat.device,
        dtype=quat.dtype,
    )
    local_forward = torch.zeros(*quat.shape[:-1], 3, device=quat.device, dtype=quat.dtype)
    local_forward[..., :2] = axis.expand(*quat.shape[:-1], 2)
    heading_rot = calc_heading_quat(quat, w_last=w_last)
    return quat_rotate(heading_rot, local_forward, w_last=w_last)[..., :2]


def heading_to_quat_for_forward_axis(
    target_yaw: Tensor,
    semantic_forward_axis_xy: Any,
    *,
    w_last: bool,
) -> Tensor:
    """Build a yaw quaternion that aims the semantic forward axis at ``target_yaw``."""
    axis = semantic_forward_axis_tensor(
        semantic_forward_axis_xy,
        device=target_yaw.device,
        dtype=target_yaw.dtype,
    )
    axis_yaw = torch.atan2(axis[..., 1], axis[..., 0])
    return heading_to_quat(target_yaw - axis_yaw, w_last=w_last)
