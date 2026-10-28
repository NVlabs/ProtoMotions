# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""World-space body goals and packing into MaskedMimic observation inputs.

Packing translates goals into the training observation format; it does not
synthesize actions or modify simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict

import torch
from torch import Tensor


@dataclass
class PackedGoals:
    """Goals packed into MaskedMimic's conditioning canvas.

    One field per ``masked_mimic.*`` name of the context vocabulary
    (``context_views.MaskedMimicContext``); every modality has a tensor here,
    empty masks standing for the unused ones. ``names()`` is what the task
    controller publishes.
    """

    ref_pos: Tensor
    ref_rot: Tensor
    target_times: Tensor
    time_offsets: Tensor
    target_poses_masks: Tensor
    target_bodies_masks: Tensor
    anchor_translation_mask: Tensor
    anchor_rotation_mask: Tensor
    local_ref_pos: Tensor
    local_ref_rot: Tensor
    local_target_times: Tensor
    local_time_offsets: Tensor
    local_target_poses_masks: Tensor
    local_target_bodies_masks: Tensor
    style_pose_samples: Tensor
    style_velocity_samples: Tensor
    style_condition_masks: Tensor
    style_frame_masks: Tensor
    average_velocity: Tensor
    velocity_command_mask: Tensor

    def names(self) -> Dict[str, Tensor]:
        """The packed goals as ``masked_mimic.*`` context names."""
        return {
            f"masked_mimic.{field.name}": getattr(self, field.name)
            for field in fields(self)
        }


@dataclass(frozen=True)
class BodyGoal:
    """One body's world target at one slot, with a per-environment deadline.

    Quaternions are xyzw. Visible has shape [E]; deadline has shape [E].
    Position/rotation, when present, have shapes [E,3] / [E,4]. Anchor
    submodes are (XY, height) and (full rotation, heading only).
    """

    body_id: int
    slot: int
    deadline: Tensor
    visible: Tensor
    position: Tensor | None = None
    rotation: Tensor | None = None
    translation: tuple[bool, bool] = (True, True)
    orientation: tuple[bool, bool] = (True, False)


def pack_goals(
    goals,
    *,
    clock,
    body_count,
    conditionable_ids,
    anchor_id,
    frames,
    style_frames,
    ground_height=None,
    velocity=None,
    velocity_mask=None,
):
    """Pack body IDs in canvas order; unused modalities have empty masks.

    Sensor-only target features are measured from the target pose's lowest
    body, the only height reference available without terrain sensing. A sparse
    goal has no support foot, so hidden bodies are placed on the terrain under
    the anchor goal to stand in for it. They are never visible goals.
    """
    count = clock.shape[0]
    ids = list(conditionable_ids)
    if len(set(ids)) != len(ids) or any(i < 0 or i >= body_count for i in ids):
        raise ValueError("Conditionable body IDs must be unique and in range")
    if anchor_id not in ids or frames < 1 or style_frames < 0:
        raise ValueError(
            "Native goals require a conditionable anchor and valid frame counts"
        )
    pos = clock.new_zeros(count, frames, body_count, 3)
    if ground_height is not None:
        if ground_height.shape != (count,) or not torch.isfinite(ground_height).all():
            raise ValueError("Ground height must be a finite [E] tensor")
        pos[..., 2] = ground_height[:, None, None]
    rot = clock.new_zeros(count, frames, body_count, 4)
    rot[..., 3] = 1
    masks = torch.zeros(
        count, frames, len(ids), 2, device=clock.device, dtype=torch.bool
    )
    times = clock[:, None].expand(-1, frames).clone()
    used_frames = torch.zeros(count, frames, device=clock.device, dtype=torch.bool)
    translation = torch.zeros(count, frames, 2, device=clock.device, dtype=torch.bool)
    orientation = torch.zeros_like(translation)
    seen = set()
    for goal in goals:
        if goal.body_id not in ids or not 0 <= goal.slot < frames:
            raise ValueError("Goal body/slot is outside the conditioning canvas")
        pair = (goal.body_id, goal.slot)
        if pair in seen:
            raise ValueError("A body/slot may appear only once in a target spec")
        seen.add(pair)
        if (
            goal.visible.shape != (count,)
            or goal.visible.dtype != torch.bool
            or goal.deadline.shape != (count,)
        ):
            raise ValueError("Goal visibility/deadlines must have shape [environments]")
        active = goal.visible
        if not torch.isfinite(goal.deadline[active]).all():
            raise ValueError("Visible goal deadlines must be finite")
        conflict = (
            active & used_frames[:, goal.slot] & (times[:, goal.slot] != goal.deadline)
        )
        if conflict.any():
            raise ValueError("Bodies in the same frame must share a deadline")
        if goal.position is None and goal.rotation is None:
            raise ValueError("A goal must specify a position or rotation")
        index = ids.index(goal.body_id)
        for property_index, (value, output, width) in enumerate(
            ((goal.position, pos, 3), (goal.rotation, rot, 4))
        ):
            if value is None:
                continue
            if value.shape != (count, width) or not torch.isfinite(value[active]).all():
                raise ValueError(
                    "Visible goal pose has an invalid shape or nonfinite value"
                )
            if property_index == 1 and not torch.allclose(
                value[active].norm(dim=-1), torch.ones_like(clock[active]), atol=1e-4
            ):
                raise ValueError("Visible rotations must be unit xyzw quaternions")
            output[active, goal.slot, goal.body_id] = value[active]
            masks[:, goal.slot, index, property_index] = active
        times[active, goal.slot] = goal.deadline[active]
        used_frames[:, goal.slot] |= active
        if goal.body_id == anchor_id:
            if goal.position is not None:
                translation[:, goal.slot] = active[:, None] & torch.tensor(
                    goal.translation, device=clock.device
                )
            if goal.rotation is not None:
                if sum(goal.orientation) != 1:
                    raise ValueError("Choose exactly one anchor rotation mode")
                orientation[:, goal.slot] = active[:, None] & torch.tensor(
                    goal.orientation, device=clock.device
                )
    world_velocity = clock.new_zeros(count, 2)
    command_mask = torch.zeros(count, 1, device=clock.device, dtype=torch.bool)
    if velocity is not None:
        if (
            velocity.shape != (count, 2)
            or velocity_mask is None
            or velocity_mask.shape != (count, 1)
        ):
            raise ValueError(
                "Velocity requires [E,2] world XY values and an [E,1] mask"
            )
        command_mask = velocity_mask.bool()
        world_velocity = velocity.masked_fill(~command_mask, 0)
        if not torch.isfinite(world_velocity).all():
            raise ValueError("Visible velocity must be finite")
    return PackedGoals(
        ref_pos=pos,
        ref_rot=rot,
        target_times=times,
        time_offsets=(times - clock[:, None]).clamp_min(0),
        target_poses_masks=masks.flatten(2).any(-1),
        target_bodies_masks=masks.flatten(1),
        anchor_translation_mask=translation,
        anchor_rotation_mask=orientation,
        local_ref_pos=pos,
        local_ref_rot=rot,
        local_target_times=clock[:, None].expand(-1, frames),
        local_time_offsets=torch.zeros_like(times),
        local_target_poses_masks=torch.zeros_like(used_frames),
        local_target_bodies_masks=torch.zeros_like(masks.flatten(1)),
        style_pose_samples=clock.new_zeros(count, style_frames * len(ids) * 9),
        style_velocity_samples=clock.new_zeros(count, style_frames * len(ids) * 6),
        style_condition_masks=torch.zeros(
            count, style_frames, 2 * len(ids) + 1, device=clock.device, dtype=torch.bool
        ),
        style_frame_masks=torch.zeros(
            count, style_frames, device=clock.device, dtype=torch.bool
        ),
        average_velocity=world_velocity,
        velocity_command_mask=command_mask,
    )
