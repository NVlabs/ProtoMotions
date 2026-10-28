# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation compute kernels for masked mimic tasks.

Pure tensor functions over ``EnvContext.masked_mimic``, whose parameter
defaults are their context bindings. The five timed-target observations of the
goal prompt are composed once by :func:`masked_mimic_goal_observations` from
the control's ``GoalCanvas``; the local-target, style and velocity-command
observations are single ``MdpComponent.from_kernel`` lines in a recipe::

    "masked_mimic_local_target_poses": MdpComponent.from_kernel(
        compute_local_target_poses,
        conditionable_body_ids=BodySelection.named(*canvas.conditionable_bodies),
    ),
    "masked_mimic_style_poses": MdpComponent.from_kernel(
        to_float, x=EnvContext.masked_mimic.style_pose_samples
    ),
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.body_selection import BodySelection
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.cast import to_float
from protomotions.envs.obs.target_poses import build_sparse_target_poses
from protomotions.envs.steps import StepSelection, select_steps
from protomotions.utils import rotations

if TYPE_CHECKING:
    from protomotions.envs.control.masked_mimic_prompt import GoalCanvas


def _grounded_body_positions(body_pos: Tensor) -> Tensor:
    """Remove the pose's own minimum body-origin height without changing XY."""
    offset = torch.zeros_like(body_pos[..., :1, :])
    offset[..., 0, 2] = body_pos[..., 2].amin(dim=-1)
    return body_pos - offset


def _target_body_masks(
    masks: Tensor,
    num_bodies: int,
    future_steps: Optional[StepSelection] = None,
) -> Tensor:
    """Unpack body/property masks and select the same 1-based steps as poses."""
    num_steps = masks.shape[1] // (num_bodies * 2)
    masks = masks.view(masks.shape[0], num_steps, num_bodies, 2)
    if future_steps is not None:
        masks = select_steps(masks, future_steps)
    return masks


def compute_target_poses_only(
    conditionable_body_ids: Tensor,
    current_state_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    current_state_body_rot: Tensor = EnvContext.current.rigid_body_rot,
    masked_mimic_ref_pos: Tensor = EnvContext.masked_mimic.ref_pos,
    masked_mimic_ref_rot: Tensor = EnvContext.masked_mimic.ref_rot,
    masked_mimic_target_bodies_masks: Tensor = (
        EnvContext.masked_mimic.target_bodies_masks
    ),
    anchor_body_index: int = EnvContext.mimic.anchor_idx,
    future_steps: Optional[StepSelection] = None,
    include_root_relative: bool = True,
    anchor_translation_mask: Optional[Tensor] = None,
    anchor_rotation_mask: Optional[Tensor] = None,
    ground_independently: bool = False,
) -> Tensor:
    """Compute masked target poses (hidden bodies zeroed out).

    The body selection is the one required static; the current pose and the
    masked-mimic stream are the kernel's default bindings (``sensed(...)``
    moves the current pose to a sensor source).

    Args:
        conditionable_body_ids: The bodies that can be conditioned on, as a
            ``BodySelection`` in a recipe (the build resolves it to indices).
        current_state_body_pos: Current body positions [num_envs, num_bodies, 3].
        current_state_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        masked_mimic_ref_pos: Reference body positions [num_envs, future_steps, num_bodies, 3].
        masked_mimic_ref_rot: Reference body rotations [num_envs, future_steps, num_bodies, 4].
        masked_mimic_target_bodies_masks: Binary visibility masks [num_envs, future_steps * num_bodies * 2].
        anchor_body_index: Robot anchor body; the anchor-relative features and
            anchor submasks apply to this body. Bound as a component it reads
            the control's published anchor unless the recipe rebinds it.
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.
        include_root_relative: If True, output 24 features per body (body-relative + anchor-relative).
            If False, output 12 features per body (body-relative only: pos delta + rot delta).
        anchor_translation_mask: Optional anchor submode mask [envs, steps, 2] for
            [XY, Z]. Values are relative to the current anchor.
        anchor_rotation_mask: Optional anchor submode mask [envs, steps, 2] for
            [full rotation, heading]. Values are relative to the current anchor.

    Returns:
        Masked target pose observations [num_envs, features].
    """
    # Height-free goals compare each pose above its own lowest body origin.
    # This is not terrain-relative height; preserve XY localization error.
    if ground_independently:
        current_state_body_pos = _grounded_body_positions(current_state_body_pos)
        masked_mimic_ref_pos = _grounded_body_positions(masked_mimic_ref_pos)
    num_envs = current_state_body_pos.shape[0]
    num_conditionable_bodies = len(conditionable_body_ids)
    conditionable_body_ids = torch.as_tensor(
        conditionable_body_ids, device=current_state_body_pos.device, dtype=torch.long
    )
    if (anchor_translation_mask is None) != (anchor_rotation_mask is None):
        raise ValueError(
            "Anchor translation and rotation masks must be provided together"
        )

    if future_steps is not None:
        masked_mimic_ref_pos = select_steps(masked_mimic_ref_pos, future_steps)
        masked_mimic_ref_rot = select_steps(masked_mimic_ref_rot, future_steps)
        if anchor_translation_mask is not None:
            anchor_translation_mask = select_steps(
                anchor_translation_mask, future_steps
            )
        if anchor_rotation_mask is not None:
            anchor_rotation_mask = select_steps(anchor_rotation_mask, future_steps)

    num_future_steps_actual = masked_mimic_ref_pos.shape[1]

    obs = build_sparse_target_poses(
        current_state_body_pos=current_state_body_pos,
        current_state_body_rot=current_state_body_rot,
        masked_mimic_ref_pos=masked_mimic_ref_pos,
        masked_mimic_ref_rot=masked_mimic_ref_rot,
        conditionable_body_ids=conditionable_body_ids,
        w_last=True,
        include_root_relative=include_root_relative,
        anchor_body_index=anchor_body_index,
    )

    # Reshape for masking: [envs, steps, bodies, types, features]
    # With include_root_relative=True: [envs, steps, bodies, 2, 12] -> 24 per body
    # With include_root_relative=False: [envs, steps, bodies, 2, 6] -> 12 per body
    features_per_body = 24 if include_root_relative else 12
    features_per_type = features_per_body // 2
    obs = obs.view(
        num_envs,
        num_future_steps_actual,
        num_conditionable_bodies,
        2,
        features_per_type,
    )

    anchor_body_mask = (conditionable_body_ids == anchor_body_index).view(
        1, 1, num_conditionable_bodies, 1
    )
    if anchor_translation_mask is not None:
        anchor_translation_mask = anchor_translation_mask.bool()
        xy_active = anchor_translation_mask[..., 0:1]
        z_active = anchor_translation_mask[..., 1:2]
        # Construct the selector directly. Slice assignment broadcasts a
        # batch-one shape into the ONNX graph instead of keeping batch dynamic.
        position_axes = torch.cat((xy_active, xy_active, z_active), dim=-1)
        anchor_position_selector = (
            torch.cat((position_axes, torch.zeros_like(position_axes)), dim=-1)
            .unsqueeze(2)
            .to(obs.dtype)
        )
        if include_root_relative:
            anchor_position_selector = torch.cat(
                (anchor_position_selector, anchor_position_selector), dim=-1
            )
        position_selector = torch.where(
            anchor_body_mask,
            anchor_position_selector,
            torch.ones_like(anchor_position_selector),
        )
        obs[..., 0, :] *= position_selector

    if anchor_rotation_mask is not None:
        anchor_rotation_mask = anchor_rotation_mask.bool()
        current_anchor_rot = (
            current_state_body_rot[:, anchor_body_index]
            .unsqueeze(1)
            .expand(-1, num_future_steps_actual, -1)
        )
        target_anchor_rot = masked_mimic_ref_rot[:, :, anchor_body_index]

        relative_full = rotations.quat_mul(
            rotations.quat_conjugate(current_anchor_rot, True),
            target_anchor_rot,
            True,
        )
        current_heading = rotations.calc_heading_quat(current_anchor_rot, True)
        target_heading = rotations.calc_heading_quat(target_anchor_rot, True)
        relative_heading = rotations.quat_mul(
            rotations.quat_conjugate(current_heading, True),
            target_heading,
            True,
        )
        full_features = rotations.quat_to_tan_norm(relative_full, True)
        heading_features = rotations.quat_to_tan_norm(relative_heading, True)
        full_active = anchor_rotation_mask[..., 0:1]
        heading_active = anchor_rotation_mask[..., 1:2]
        # Full rotation includes heading, so if a caller accidentally enables
        # both modes, full rotation takes precedence instead of adding two
        # rotation encodings together.
        rotation_features = torch.where(
            full_active,
            full_features,
            heading_features * heading_active,
        )
        if include_root_relative:
            rotation_features = torch.cat(
                (rotation_features, rotation_features), dim=-1
            )
        obs[..., 1, :] = torch.where(
            anchor_body_mask,
            rotation_features.unsqueeze(2),
            obs[..., 1, :],
        )

    mask = _target_body_masks(
        masked_mimic_target_bodies_masks, num_conditionable_bodies, future_steps
    ).reshape(num_envs, num_future_steps_actual, num_conditionable_bodies, 2, 1)
    masked_obs = obs * mask

    return masked_obs.flatten(1)


def compute_target_masks_only(
    conditionable_body_ids: Tensor,
    masked_mimic_target_bodies_masks: Tensor = (
        EnvContext.masked_mimic.target_bodies_masks
    ),
    future_steps: Optional[StepSelection] = None,
    anchor_translation_mask: Optional[Tensor] = None,
    anchor_rotation_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return binary target masks.

    Args:
        conditionable_body_ids: The bodies that can be conditioned on (a
            ``BodySelection`` in a recipe; the build resolves it).
        masked_mimic_target_bodies_masks: Binary visibility masks [num_envs, future_steps * num_bodies * 2].
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.
        anchor_translation_mask: Optional anchor [XY, Z] submode mask.
        anchor_rotation_mask: Optional anchor [full, heading] submode mask.

    Returns:
        Binary body masks followed by anchor submode masks for each selected step.
    """
    masks_reshaped = _target_body_masks(
        masked_mimic_target_bodies_masks, len(conditionable_body_ids)
    ).flatten(2)
    if (anchor_translation_mask is None) != (anchor_rotation_mask is None):
        raise ValueError(
            "anchor_translation_mask and anchor_rotation_mask must be provided together."
        )
    if anchor_translation_mask is not None:
        masks_reshaped = torch.cat(
            (
                masks_reshaped,
                anchor_translation_mask,
                anchor_rotation_mask,
            ),
            dim=-1,
        )

    if future_steps is not None:
        masks_reshaped = select_steps(masks_reshaped, future_steps)
    return masks_reshaped.flatten(1)


def compute_local_pose_samples(
    target_body_pos: Tensor,
    target_body_rot: Tensor,
    conditionable_body_ids: Tensor,
    anchor_body_index: int,
) -> Tensor:
    """Express target poses in their own anchor-heading and ground frame.

    XY is relative to the target anchor. Z is relative to the lowest body in
    that target pose. Rotations have the target anchor heading removed. The
    anchor therefore contributes only height for translation and roll/pitch
    for rotation; it never carries world XY or heading into the local stream.
    """
    if target_body_pos.ndim != 4 or target_body_rot.ndim != 4:
        raise ValueError("target poses must have shape [envs, steps, bodies, features]")
    if target_body_pos.shape[:3] != target_body_rot.shape[:3]:
        raise ValueError("target position and rotation shapes must agree")
    if target_body_pos.shape[-1] != 3 or target_body_rot.shape[-1] != 4:
        raise ValueError("target positions/rotations must have features 3/4")

    body_ids = conditionable_body_ids.to(device=target_body_pos.device).reshape(-1)
    anchor_slots = torch.nonzero(
        body_ids == anchor_body_index, as_tuple=False
    ).flatten()
    if anchor_slots.numel() != 1:
        raise ValueError("conditionable bodies must contain the anchor exactly once")

    selected_pos = target_body_pos.index_select(2, body_ids)
    selected_rot = target_body_rot.index_select(2, body_ids)
    anchor_pos = target_body_pos[..., anchor_body_index, :]
    anchor_rot = target_body_rot[..., anchor_body_index, :]
    heading_inv = rotations.calc_heading_quat_inv(anchor_rot, w_last=True)

    relative_xy = selected_pos[..., :2] - anchor_pos.unsqueeze(2)[..., :2]
    relative_xy_3d = torch.cat(
        (relative_xy, torch.zeros_like(relative_xy[..., :1])), dim=-1
    )
    heading_for_bodies = heading_inv.unsqueeze(2).expand_as(selected_rot)
    local_xy = rotations.quat_rotate(
        heading_for_bodies.reshape(-1, 4),
        relative_xy_3d.reshape(-1, 3),
        w_last=True,
    ).reshape_as(relative_xy_3d)[..., :2]

    lowest_height = target_body_pos[..., 2].amin(dim=-1, keepdim=True)
    local_z = selected_pos[..., 2:3] - lowest_height.unsqueeze(2)
    local_pos = torch.cat((local_xy, local_z), dim=-1)

    local_rot = rotations.quat_mul(heading_for_bodies, selected_rot, w_last=True)
    local_rot_features = rotations.quat_to_tan_norm(local_rot, w_last=True)
    return torch.cat((local_pos, local_rot_features), dim=-1)


def compute_local_target_poses(
    conditionable_body_ids: Tensor,
    masked_mimic_ref_pos: Tensor = EnvContext.masked_mimic.local_ref_pos,
    masked_mimic_ref_rot: Tensor = EnvContext.masked_mimic.local_ref_rot,
    masked_mimic_target_bodies_masks: Tensor = (
        EnvContext.masked_mimic.local_target_bodies_masks
    ),
    masked_mimic_target_poses_masks: Tensor = (
        EnvContext.masked_mimic.local_target_poses_masks
    ),
    anchor_body_index: int = EnvContext.mimic.anchor_idx,
    future_steps: Optional[StepSelection] = None,
) -> Tensor:
    """Build a masked, timed local target stream with nine features per body.

    Reads the control's local (anchor-relative) stream by default; the anchor
    is the control's published ``mimic.anchor_idx``. ``future_steps`` are the
    1-based steps of the stream to read (None = every published step).
    """
    num_bodies = conditionable_body_ids.numel()
    if future_steps is not None:
        masked_mimic_ref_pos = select_steps(masked_mimic_ref_pos, future_steps)
        masked_mimic_ref_rot = select_steps(masked_mimic_ref_rot, future_steps)
        masked_mimic_target_poses_masks = select_steps(
            masked_mimic_target_poses_masks, future_steps
        )
    body_masks = _target_body_masks(
        masked_mimic_target_bodies_masks, num_bodies, future_steps
    ).reshape(*masked_mimic_ref_pos.shape[:2], num_bodies, 2)

    samples = compute_local_pose_samples(
        masked_mimic_ref_pos,
        masked_mimic_ref_rot,
        conditionable_body_ids,
        anchor_body_index,
    )
    visible = body_masks.bool() & masked_mimic_target_poses_masks.bool().unsqueeze(
        -1
    ).unsqueeze(-1)
    samples[..., :3] *= visible[..., 0:1]
    samples[..., 3:] *= visible[..., 1:2]
    return samples.flatten(1)


def compute_target_time_offsets(
    masked_mimic_time_offsets: Tensor = EnvContext.masked_mimic.time_offsets,
    future_steps: Optional[StepSelection] = None,
) -> Tensor:
    """Return target time offsets.

    Args:
        masked_mimic_time_offsets: Time offsets for each future step [num_envs, future_steps].
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.

    Returns:
        Time offsets to each target frame [num_envs, selected_steps].
    """
    if future_steps is None:
        return masked_mimic_time_offsets
    return select_steps(masked_mimic_time_offsets, future_steps)


def masked_mimic_goal_observations(
    canvas: "GoalCanvas",
    *,
    ground_independently: bool = False,
    include_root_relative: bool = True,
) -> Dict[str, MdpComponent]:
    """The five timed-target observations of the goal prompt, bound for ``canvas``.

    ``canvas`` is the control's ``GoalCanvas`` (``GoalCanvas.derive(control_config,
    robot_config)``): the bodies are the canvas's conditionable bodies as a
    ``BodySelection.named(...)`` (the build resolves them, the saved
    configuration keeps the names) and, with
    ``canvas.anchor_pose_submasking``, the anchor sub-mask pair is bound to both
    the pose and the mask kernel, so the policy sees which anchor components the
    zeroed features stand for. All five outputs cover every control frame so the
    transformer's token and mask inputs stay aligned; ``sensed(...)`` over the
    result moves the current pose to a sensor source.
    """
    bodies = BodySelection.named(*canvas.conditionable_bodies)
    submasks: Dict[str, object] = {}
    if canvas.anchor_pose_submasking:
        submasks = {
            "anchor_translation_mask": EnvContext.masked_mimic.anchor_translation_mask,
            "anchor_rotation_mask": EnvContext.masked_mimic.anchor_rotation_mask,
        }
    return {
        "masked_mimic_target_poses": MdpComponent.from_kernel(
            compute_target_poses_only,
            conditionable_body_ids=bodies,
            include_root_relative=include_root_relative,
            ground_independently=ground_independently,
            **submasks,
        ),
        "masked_mimic_target_masks": MdpComponent.from_kernel(
            compute_target_masks_only, conditionable_body_ids=bodies, **submasks
        ),
        "masked_mimic_target_times": MdpComponent.from_kernel(
            compute_target_time_offsets
        ),
        "masked_mimic_target_poses_masks": MdpComponent.from_kernel(
            to_float, x=EnvContext.masked_mimic.target_poses_masks
        ),
        "masked_mimic_target_bodies_masks": MdpComponent.from_kernel(
            to_float, x=EnvContext.masked_mimic.target_bodies_masks
        ),
    }


STYLE_POSE_FEATURES_PER_BODY = 9
STYLE_VELOCITY_FEATURES_PER_BODY = 6


def compute_style_pose_samples(
    style_ref_pos: Tensor,
    style_ref_rot: Tensor,
    anchor_body_index: int = EnvContext.mimic.anchor_idx,
    conditionable_body_ids: Optional[Tensor] = None,
) -> Tensor:
    """Build untimed style poses with the same geometry as local targets.

    Inputs are [envs, frames, bodies, 3/4] positions/XYZW rotations. Output is
    [envs, frames * selected_bodies * 9], with translation (3) followed by
    tangent-normal rotation (6) for each selected body.
    """
    if conditionable_body_ids is None:
        conditionable_body_ids = torch.arange(
            style_ref_pos.shape[2], device=style_ref_pos.device
        )
    pose = compute_local_pose_samples(
        style_ref_pos,
        style_ref_rot,
        conditionable_body_ids,
        anchor_body_index,
    )
    return pose.flatten(1)


def compute_style_velocity_samples(
    style_ref_vel: Tensor,
    style_ref_ang_vel: Tensor,
    style_ref_rot: Tensor,
    w_last: bool = True,
    anchor_body_index: int = EnvContext.mimic.anchor_idx,
    conditionable_body_ids: Optional[Tensor] = None,
) -> Tensor:
    """Build heading-aligned style velocity samples.

    Linear and angular reference velocities are rotated into each frame's
    anchor-heading basis.  The anchor velocity itself is retained, so style
    can communicate locomotion dynamics (including moving versus standing)
    without exposing world yaw or translation. Each body contributes the
    ``STYLE_VELOCITY_FEATURES_PER_BODY`` values: linear velocity (3) followed
    by angular velocity (3).

    Inputs are [envs, frames, bodies, 3] velocities and matching rotations
    with four quaternion components. Output is flattened to
    [envs, frames * selected_bodies * 6].
    """
    if style_ref_vel.ndim != 4 or style_ref_ang_vel.ndim != 4:
        raise ValueError(
            "style velocities must have shape [envs, steps, bodies, features]"
        )
    if style_ref_vel.shape[-1] != 3 or style_ref_ang_vel.shape[-1] != 3:
        raise ValueError("style velocities must have last dimension 3")
    if style_ref_vel.shape != style_ref_ang_vel.shape:
        raise ValueError("linear and angular style velocity shapes must agree")
    if style_ref_rot.ndim != 4 or style_ref_rot.shape[:3] != style_ref_vel.shape[:3]:
        raise ValueError(
            "style rotations and velocities must have matching leading shapes"
        )
    if style_ref_rot.shape[-1] != 4:
        raise ValueError("style_ref_rot must have last dimension 4")
    num_bodies = style_ref_vel.shape[2]
    if type(anchor_body_index) is not int or not 0 <= anchor_body_index < num_bodies:
        raise ValueError("anchor_body_index is out of range")

    if conditionable_body_ids is None:
        conditionable_body_ids = torch.arange(num_bodies, device=style_ref_vel.device)
    else:
        conditionable_body_ids = conditionable_body_ids.to(device=style_ref_vel.device)

    anchor_rot = style_ref_rot[..., anchor_body_index, :]
    heading_inv = rotations.calc_heading_quat_inv(anchor_rot, w_last)
    selected_vel = style_ref_vel.index_select(2, conditionable_body_ids)
    selected_ang_vel = style_ref_ang_vel.index_select(2, conditionable_body_ids)
    heading_inv = heading_inv.unsqueeze(-2).expand(*selected_vel.shape[:-1], 4)
    local_vel = rotations.quat_rotate(heading_inv, selected_vel, w_last)
    local_ang_vel = rotations.quat_rotate(heading_inv, selected_ang_vel, w_last)
    velocity = torch.cat((local_vel, local_ang_vel), dim=-1)
    return velocity.flatten(1)


def apply_style_condition_masks(
    pose_samples: Tensor,
    velocity_samples: Tensor,
    condition_masks: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Apply the per-frame ``bodies x 2 + velocity`` style mask contract.

    Translation and rotation bits mask their corresponding pose features. The
    final bit enables velocities for every body with either pose property
    visible in that frame.
    """
    if condition_masks.ndim != 3 or condition_masks.shape[-1] < 3:
        raise ValueError("condition_masks must have shape [envs, frames, bodies*2+1]")
    num_envs, num_frames, mask_width = condition_masks.shape
    if (mask_width - 1) % 2 != 0:
        raise ValueError("style mask width must be bodies*2+1")
    num_bodies = (mask_width - 1) // 2
    expected_pose_width = num_frames * num_bodies * STYLE_POSE_FEATURES_PER_BODY
    expected_velocity_width = num_frames * num_bodies * STYLE_VELOCITY_FEATURES_PER_BODY
    if pose_samples.shape != (num_envs, expected_pose_width):
        raise ValueError("pose_samples width does not match the style mask")
    if velocity_samples.shape != (num_envs, expected_velocity_width):
        raise ValueError("velocity_samples width does not match the style mask")

    body_masks = condition_masks[..., :-1].view(num_envs, num_frames, num_bodies, 2)
    pose = pose_samples.view(
        num_envs, num_frames, num_bodies, STYLE_POSE_FEATURES_PER_BODY
    ).clone()
    pose[..., :3] *= body_masks[..., 0:1]
    pose[..., 3:] *= body_masks[..., 1:2]

    visible_bodies = body_masks.any(dim=-1, keepdim=True)
    velocity_active = condition_masks[..., -1:].unsqueeze(2)
    velocity = velocity_samples.view(
        num_envs, num_frames, num_bodies, STYLE_VELOCITY_FEATURES_PER_BODY
    ).clone()
    velocity *= visible_bodies & velocity_active
    return pose.flatten(1), velocity.flatten(1)


__all__ = [
    "compute_style_pose_samples",
    "compute_style_velocity_samples",
    "apply_style_condition_masks",
    "compute_local_pose_samples",
    "compute_local_target_poses",
    "compute_target_poses_only",
    "compute_target_masks_only",
    "compute_target_time_offsets",
    "masked_mimic_goal_observations",
]
