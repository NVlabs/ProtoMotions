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

import logging
from typing import Any

import torch
from torch import Tensor

from protomotions.utils.rotations import calc_heading_quat, heading_to_quat, quat_rotate

_LOG = logging.getLogger(__name__)
_LOGGED_LEGACY_FORWARD_AXIS_FALLBACK = False


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

    ``None`` is accepted only for legacy resolved configs pickled before the
    field existed. It falls back to local ``+X``, preserving old behavior while
    fresh robot configs declare the axis explicitly.
    """
    global _LOGGED_LEGACY_FORWARD_AXIS_FALLBACK
    if semantic_forward_axis_xy is None:
        if not _LOGGED_LEGACY_FORWARD_AXIS_FALLBACK:
            _LOG.warning(
                "Robot config has no semantic_forward_axis_xy; falling back to "
                "legacy local +X facing semantics. Regenerate the config to "
                "declare this field explicitly. Run `python "
                "scripts/identify_robot_facing_axis.py --robots <robot-name> "
                "--view` to identify and visualize it."
            )
            _LOGGED_LEGACY_FORWARD_AXIS_FALLBACK = True
        semantic_forward_axis_xy = (1.0, 0.0)

    axis = torch.as_tensor(semantic_forward_axis_xy, device=device, dtype=dtype)
    if axis.shape[-1] != 2:
        raise ValueError(
            "semantic_forward_axis_xy must have shape (..., 2). The axis is "
            "xy-only because ProtoMotions assumes Z-up robots and uses it for "
            "horizontal heading, not upright pose re-orientation."
        )
    norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return axis / norm


def robot_semantic_forward_axis_tensor(
    robot_config: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Read a robot config's semantic forward axis with legacy fallback."""
    return semantic_forward_axis_tensor(
        getattr(robot_config, "semantic_forward_axis_xy", None),
        device=device,
        dtype=dtype,
    )


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
