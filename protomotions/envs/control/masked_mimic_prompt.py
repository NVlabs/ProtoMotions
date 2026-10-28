# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MaskedMimic goal prompt's layout, derived once and checked at build.

A masked-mimic policy is trained against a prompt of a fixed shape: how many
future slots the full mimic stream and the masked target stream carry, which
bodies may be conditioned and in which order, which body is the anchor,
whether the anchor's translation and rotation can be masked separately,
whether a second, anchor-relative (local) target stream exists, how many
style frames there are and whether a velocity command is part of it. The
training sampler (:class:`~.masked_mimic_control.MaskedMimicControl`) and
every other prompt producer (the interactive task controller, a scripted
task) must agree on that layout, and the observations bound on the actor
must match it.

:class:`GoalCanvas` is that layout as one frozen value, derived from the
trained :class:`~.masked_mimic_control.MaskedMimicControlConfig` and the
robot (:meth:`GoalCanvas.derive`) and validated against the bound context
paths (:meth:`GoalCanvas.validate`), so the three copies of these fields
that used to live in the training config, the task config and the recipe
cannot drift apart silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from protomotions.envs.steps import normalize_steps

ANCHOR_IN_TRACKABLE = (
    "Anchor/local/style goals require the anchor in trackable_bodies_subset"
)
GETUP_NEEDS_LOCAL_STREAM = (
    "GetupRecoveryConfig needs a checkpoint with a local target stream "
    "(local_target_condition_probability > 0)"
)
SUBMASKING_UNBOUND = (
    "anchor_pose_submasking is on but no observation binds "
    "masked_mimic.anchor_translation_mask / anchor_rotation_mask"
)
ANCHOR_MASK_PATHS = frozenset(
    {"masked_mimic.anchor_translation_mask", "masked_mimic.anchor_rotation_mask"}
)
# The ``masked_mimic.*`` names of each optional modality, by the prefix or the
# exact name a declared path has (see ``MaskedMimicContext``).
LOCAL_STREAM_PREFIX = "masked_mimic.local_"
STYLE_STREAM_PREFIX = "masked_mimic.style_"
VELOCITY_PATHS = frozenset(
    {"masked_mimic.average_velocity", "masked_mimic.velocity_command_mask"}
)


@dataclass(frozen=True)
class GoalCanvas:
    """The goal prompt's layout: what the trained policy expects, once."""

    future_steps: Tuple[int, ...]  # the full mimic stream's step indices
    num_masked_future_steps: int  # slots of the masked target stream(s)
    conditionable_bodies: Tuple[str, ...]  # body names, in prompt order
    anchor_body: str
    anchor_pose_submasking: bool
    has_local_stream: bool
    style_frames: Optional[int]  # None without the style stream
    has_velocity_command: bool

    @classmethod
    def derive(cls, control_config, robot_config) -> "GoalCanvas":
        """The layout a ``MaskedMimicControlConfig`` trains against on ``robot_config``.

        Refuses a configuration whose anchor, local or style goals need the
        anchor body but whose ``trackable_bodies_subset`` does not carry it.
        """
        body_names = robot_config.kinematic_info.body_names
        anchor_body = body_names[robot_config.anchor_body_index]
        conditionable = tuple(robot_config.trackable_bodies_subset)
        style = control_config.style
        has_local_stream = control_config.local_stream
        needs_anchor = (
            control_config.anchor_pose_submasking
            or has_local_stream
            or style is not None
        )
        if needs_anchor and anchor_body not in conditionable:
            raise ValueError(ANCHOR_IN_TRACKABLE)
        return cls(
            # An override written after __post_init__ may have left the int.
            future_steps=normalize_steps(control_config.future_steps),
            num_masked_future_steps=control_config.num_masked_future_steps,
            conditionable_bodies=conditionable,
            anchor_body=anchor_body,
            anchor_pose_submasking=control_config.anchor_pose_submasking,
            has_local_stream=has_local_stream,
            style_frames=None if style is None else style.num_frames,
            has_velocity_command=control_config.velocity_command is not None,
        )

    def _missing_modality(self, path: str) -> Optional[str]:
        """What ``path`` reads that this canvas does not carry (a stream, or the
        anchor sub-masks), as the words of the refusal; None when it is served."""
        if path.startswith(LOCAL_STREAM_PREFIX) and not self.has_local_stream:
            return "local stream"
        if path.startswith(STYLE_STREAM_PREFIX) and self.style_frames is None:
            return "style stream"
        if path in VELOCITY_PATHS and not self.has_velocity_command:
            return "velocity command"
        if path in ANCHOR_MASK_PATHS and not self.anchor_pose_submasking:
            return "anchor sub-masks (anchor_pose_submasking is off)"
        return None

    def validate(self, bound_paths: FrozenSet[str], getup: bool) -> None:
        """Refuse what this layout cannot serve: a get-up preset without a local
        stream, a binding to a stream or the anchor sub-masks the canvas lacks,
        and anchor sub-masking that no observation reads."""
        if getup and not self.has_local_stream:
            raise ValueError(GETUP_NEEDS_LOCAL_STREAM)
        for path in sorted(bound_paths):
            missing = self._missing_modality(path)
            if missing is not None:
                raise ValueError(
                    f"observation binds '{path}' but the goal canvas has no {missing}"
                )
        if self.anchor_pose_submasking and not (bound_paths & ANCHOR_MASK_PATHS):
            raise ValueError(SUBMASKING_UNBOUND)


__all__ = ["GoalCanvas"]
