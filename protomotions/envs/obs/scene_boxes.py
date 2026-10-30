# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Object-box observations for scene conditioning: corners in the robot's heading frame.

A scene-conditioned policy sees a fixed-capacity set of boxes, ``objects`` times
``boxes per object``: each box is a centre and full extents in its object's
frame (asset scale included), each object has a world pose, each box a validity
flag (padding boxes are invalid) and each object a visibility flag. The names
live in the ``scene_boxes`` control namespace (``context_views.py``); the
``SceneBoxesControl`` placeholder publishes them with no boxes, and a host
supplies real values at deployment.

The observation is the eight corners of every box in the heading frame of the
robot's root body, zero for an invalid box, plus a per-box mask that is visible
AND valid. A per-episode bias of the observed object pose (a position offset and
a yaw offset) is applied before the projection, so a training recipe can
perturb what the policy sees; a robot host feeds zeros.

The corners kernel computes the root pose from the joint angles and the anchor
pose with the recipe's ``BakedAnchorFK``, the way ``compute_fk_target_poses_only``
does, so an exported graph takes raw readings and the eight scene inputs and no
body pose. The projection itself is a plain function of the root pose
(:func:`scene_box_corners_in_heading_frame`), so a privileged binding is one
line away if a training recipe ever wants one; none is kept in the tree.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.utils import rotations
from protomotions.utils.baked_fk import BakedAnchorFK

# The kinematic root is body 0 in every robot's body order (pose_lib's FK
# holds it at the origin); its pose is what ``current.root_pos`` / ``root_rot``
# carry.
ROOT_BODY = 0


def observed_object_pose(
    object_pos: Tensor, object_rot: Tensor, position_offset: Tensor, yaw_offset: Tensor
) -> Tuple[Tensor, Tensor]:
    """The object pose the policy observes: the true pose plus a per-env bias.

    The yaw turns the object about the world vertical after its own rotation
    (``yaw * object_rot``), the way a biased pose estimate would.

    Args:
        object_pos: ``[E, O, 3]`` world positions.
        object_rot: ``[E, O, 4]`` world rotations, xyzw.
        position_offset: ``[E, 3]`` position bias, added to every object.
        yaw_offset: ``[E]`` yaw bias in radians, composed onto every object.
    """
    half = 0.5 * yaw_offset
    zero = torch.zeros_like(half)
    yaw = torch.stack([zero, zero, torch.sin(half), torch.cos(half)], dim=-1)
    rot = rotations.quat_mul(yaw[:, None].expand_as(object_rot), object_rot, True)
    return object_pos + position_offset[:, None], rot


def scene_box_corners_in_heading_frame(
    object_pos: Tensor,
    object_rot: Tensor,
    centers: Tensor,
    extents: Tensor,
    valid: Tensor,
    position_offset: Tensor,
    yaw_offset: Tensor,
    root_pos: Tensor,
    root_rot: Tensor,
) -> Tensor:
    """Every box's eight corners in the root heading frame, ``[E, O * B, 24]``.

    Corner order is lexicographic in the sign of (x, y, z): (-,-,-), (-,-,+),
    (-,+,-), ... Corners keep the object's orientation, which bare extents
    would lose. An invalid box is zero, whatever its unused pose holds.

    Args:
        object_pos, object_rot: object world poses, ``[E, O, 3]`` / ``[E, O, 4]`` xyzw.
        centers, extents: box centres and full side lengths in the object
            frame, ``[E, O, B, 3]``, asset scale included.
        valid: ``[E, O, B]`` per-box presence (padding is false).
        position_offset, yaw_offset: the observed-pose bias (``[E, 3]``, ``[E]``).
        root_pos, root_rot: the robot root's world pose, ``[E, 3]`` / ``[E, 4]`` xyzw.
    """
    object_pos, object_rot = observed_object_pose(
        object_pos, object_rot, position_offset, yaw_offset
    )
    signs = extents.new_tensor(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    local = centers.unsqueeze(-2) + 0.5 * extents.unsqueeze(-2) * signs
    shape = local.shape  # [E, O, B, 8, 3]
    obj_rot = object_rot[:, :, None, None, :].expand(*shape[:-1], 4)
    world = rotations.quat_rotate(obj_rot.reshape(-1, 4), local.reshape(-1, 3), True)
    world = world.reshape(shape) + object_pos[:, :, None, None, :]
    heading = rotations.calc_heading_quat_inv(root_rot, True)
    heading = heading[:, None, None, None, :].expand(*shape[:-1], 4)
    relative = world - root_pos[:, None, None, None, :]
    corners = rotations.quat_rotate(
        heading.reshape(-1, 4), relative.reshape(-1, 3), True
    ).reshape(shape)
    corners = torch.where(
        valid[..., None, None].bool(), corners, torch.zeros_like(corners)
    )
    return corners.flatten(-2).flatten(1, 2)


def compute_fk_scene_box_corners(
    object_pos: Tensor = EnvContext.scene_boxes.object_pos,
    object_rot: Tensor = EnvContext.scene_boxes.object_rot,
    centers: Tensor = EnvContext.scene_boxes.centers,
    extents: Tensor = EnvContext.scene_boxes.extents,
    valid: Tensor = EnvContext.scene_boxes.valid,
    position_offset: Tensor = EnvContext.scene_boxes.position_offset,
    yaw_offset: Tensor = EnvContext.scene_boxes.yaw_offset,
    dof_pos: Tensor = EnvContext.current.dof_pos,
    anchor_pos: Tensor = EnvContext.current.anchor_pos,
    anchor_rot: Tensor = EnvContext.current.anchor_rot,
    anchor_body_index: int = EnvContext.current.anchor_idx,
    *,
    fk: BakedAnchorFK,
) -> Tensor:
    """Box corners in the root heading frame, the root pose computed by ``fk``.

    ``[E, O * B, 24]``, see :func:`scene_box_corners_in_heading_frame`. The
    root's world pose is body ``ROOT_BODY`` of ``fk.world_body_poses`` on the
    joint angles and the anchor pose, the same kinematics the sensor runtime
    uses, so an exported graph takes the raw readings it already has and no
    body pose. ``sensed(...)`` moves the four state bindings to the sensor
    source. ``anchor_body_index`` is the body whose pose the state publishes
    as the anchor; a kernel anchored on another body would place every box
    wrong by the offset between the two, so it is refused, as the goal kernel
    refuses it.
    """
    if fk.anchor_idx != anchor_body_index:
        raise ValueError(
            f"the FK kernel is anchored on body {fk.anchor_idx} but the state "
            f"publishes the pose of anchor body {anchor_body_index}; build the "
            "BakedAnchorFK with the robot's anchor"
        )
    body_pos, body_rot = fk.world_body_poses(dof_pos, anchor_pos, anchor_rot)
    return scene_box_corners_in_heading_frame(
        object_pos,
        object_rot,
        centers,
        extents,
        valid,
        position_offset,
        yaw_offset,
        body_pos[:, ROOT_BODY],
        body_rot[:, ROOT_BODY],
    )


def compute_scene_box_mask(
    visible: Tensor = EnvContext.scene_boxes.visible,
    valid: Tensor = EnvContext.scene_boxes.valid,
) -> Tensor:
    """Per-box token mask, ``[E, O * B]`` bool: object visible AND box valid."""
    return (visible.bool() & valid.bool()).flatten(1)


__all__ = [
    "ROOT_BODY",
    "observed_object_pose",
    "scene_box_corners_in_heading_frame",
    "compute_fk_scene_box_corners",
    "compute_scene_box_mask",
]
