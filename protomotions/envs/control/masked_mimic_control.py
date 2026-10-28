# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MaskedMimic global targets with coherent episode-level conditioning plans.

The sampler of the training prompt: per episode, a target-stream plan (slot
offsets, frame visibility, body/property masks) for the global stream and,
when configured, the local stream, the anchor sub-masks, the style plan and
the velocity-command visibility, all per-env state in the control's registry
(the streams in child registries) and all drawn on the control stage's
generator. Publishes the parent's ``mimic.*`` names and the
``masked_mimic.*`` names of the enabled modalities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from protomotions.envs.base_env.episode_tensors import EpisodeTensors
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.envs.obs.masked_mimic import (
    STYLE_POSE_FEATURES_PER_BODY,
    STYLE_VELOCITY_FEATURES_PER_BODY,
    apply_style_condition_masks,
    compute_style_pose_samples,
    compute_style_velocity_samples,
)
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator_state import RobotState


@dataclass
class FixedBodyCondition:
    """Fixed conditioning for a specific body."""

    body_name: str
    constraint_state: int  # 0=translation only, 1=both, 2=rotation only


@dataclass
class MaskedMimicVelocityCommandConfig:
    """Episode visibility and rolling reference lookahead for an XY command."""

    horizon_seconds: float = 0.5
    probability: float = 0.5

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not math.isfinite(self.horizon_seconds) or self.horizon_seconds <= 0:
            raise ValueError("velocity horizon_seconds must be finite and positive")
        if not 0 <= self.probability <= 1:
            raise ValueError("velocity probability must be in [0, 1]")


@dataclass
class MaskedMimicStyleConfig:
    """Untimed reference samples: fixed episode offsets, sliding future window.

    Each visible frame selects body translation/rotation and optionally dynamics.
    The default distribution hides style in half the episodes and spreads the
    remaining probability evenly across one through five visible frames.
    """

    window_seconds: float = 2.0
    num_frames: int = 5
    near_time_bias_power: float = 2.0
    velocity_probability: float = 0.5
    visible_frame_count_probabilities: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        if (
            self.visible_frame_count_probabilities is None
            and isinstance(self.num_frames, int)
            and self.num_frames > 0
        ):
            self.visible_frame_count_probabilities = (0.5,) + (
                0.5 / self.num_frames,
            ) * self.num_frames
        self.validate()

    def validate(self):
        if type(self.num_frames) is not int or not 1 <= self.num_frames <= 5:
            raise ValueError("style num_frames must be an integer in [1, 5]")
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("style window_seconds must be finite and positive")
        if (
            not math.isfinite(self.near_time_bias_power)
            or self.near_time_bias_power < 1
        ):
            raise ValueError("style near_time_bias_power must be finite and at least 1")
        if not 0 <= self.velocity_probability <= 1:
            raise ValueError("style velocity_probability must be in [0, 1]")
        weights = self.visible_frame_count_probabilities
        if (
            len(weights) != self.num_frames + 1
            or any(not math.isfinite(p) or p < 0 for p in weights)
            or not math.isfinite(sum(weights))
            or sum(weights) <= 0
        ):
            raise ValueError(
                "style count weights require finite nonnegative mass for counts 0 through num_frames"
            )


@dataclass
class MaskedMimicControlConfig(MimicControlConfig):
    """Sample one dense/intermediate/sparse global prompt per episode.

    Body/property masks and frame visibility remain fixed until reset. Time
    remaining counts down until the first target time is reached, even if that
    target is hidden. All target times then move forward using the same offsets
    from the current motion time.
    Inherited future_steps controls the separate, full mimic reference stream.
    """

    _target_: str = "protomotions.envs.control.masked_mimic_control.MaskedMimicControl"

    num_masked_future_steps: int = 5
    # Relative weights for dense, intermediate and sparse episodes.
    target_profile_probabilities: Tuple[float, float, float] = (0.2, 0.6, 0.2)
    target_gap_min_seconds: float = 0.1
    target_gap_max_seconds: float = 2.0
    # Frame visibility for intermediate episodes; all-hidden prompts are valid.
    visible_target_pose_prob: float = 0.8
    # Overrides body/property selection, not profile spacing or frame visibility.
    fixed_conditioning: Optional[List[FixedBodyCondition]] = None
    # Independent episode-level enables; defaults preserve global-only prompts.
    sparse_target_condition_probability: float = 1.0
    local_target_condition_probability: float = 0.0
    anchor_pose_submasking: bool = False
    style: Optional[MaskedMimicStyleConfig] = None
    velocity_command: Optional[MaskedMimicVelocityCommandConfig] = None

    @property
    def local_stream(self) -> bool:
        """Whether episodes may carry local (anchor-relative) targets: the one
        read of ``local_target_condition_probability > 0``, the switch behind
        the ``requires="local_stream"`` names and the goal canvas."""
        return self.local_target_condition_probability > 0

    def __post_init__(self):
        super().__post_init__()
        self.validate()

    def validate(self) -> None:
        if self.velocity_command is not None:
            self.velocity_command.validate()
        if self.style is not None:
            self.style.validate()
        probabilities = self.target_profile_probabilities
        if (
            len(probabilities) != 3
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isfinite(sum(probabilities))
            or sum(probabilities) <= 0
        ):
            raise ValueError(
                "target_profile_probabilities requires three finite nonnegative "
                "weights with a finite positive sum"
            )
        if not (
            math.isfinite(self.target_gap_min_seconds)
            and math.isfinite(self.target_gap_max_seconds)
            and 0 < self.target_gap_min_seconds <= self.target_gap_max_seconds
        ):
            raise ValueError(
                "Require finite 0 < target_gap_min_seconds <= target_gap_max_seconds"
            )
        if self.num_masked_future_steps < 1:
            raise ValueError("num_masked_future_steps must be positive")
        if not 0 <= self.visible_target_pose_prob <= 1:
            raise ValueError("visible_target_pose_prob must be in [0, 1]")
        if not 0 <= self.sparse_target_condition_probability <= 1:
            raise ValueError("sparse_target_condition_probability must be in [0, 1]")
        if not 0 <= self.local_target_condition_probability <= 1:
            raise ValueError("local_target_condition_probability must be in [0, 1]")


class TargetStream:
    """Episode-fixed lookahead plan for one timed target stream.

    Offsets, frame visibility and body/property masks stay at their slot
    indices until reset. Times are absolute clip timestamps rebased from the
    current motion time whenever the first slot is reached, so a later visible
    goal advances before its own deadline. Slots do not shift like a rolling
    queue: the episode's sampled offsets and masks are reused at each rebase.
    """

    def __init__(
        self, state: EpisodeTensors, num_steps: int, num_bodies: int, probability: float
    ):
        self.probability = probability
        # The plan's four per-env tensors, allocated through the control's
        # child registry (row reset, snapshot and restore reach them).
        self.times = state.add("times", (num_steps,), torch.float, 0.0)
        self.offsets = state.add("offsets", (num_steps,), torch.float, 0.0)
        self.frame_masks = state.add("frame_masks", (num_steps,), torch.bool, False)
        self.body_masks = state.add(
            "body_masks", (num_steps * num_bodies * 2,), torch.bool, False
        )

    def write_plan(self, env_ids: Tensor, offsets: Tensor, body_masks: Tensor) -> None:
        """Store offsets [E, S] and visibility [E, S, B, 2] for the reset envs."""
        self.offsets[env_ids] = offsets
        self.body_masks[env_ids] = body_masks.flatten(1)
        self.frame_masks[env_ids] = body_masks.flatten(2).any(-1)

    def rebase_times(
        self, env_ids: Tensor, motion_times: Tensor, motion_lengths: Tensor
    ) -> None:
        self.times[env_ids] = torch.minimum(
            motion_times[:, None] + self.offsets[env_ids], motion_lengths[:, None]
        )

    def due(self, motion_times: Tensor) -> Tensor:
        """The envs whose first slot's time is reached."""
        return torch.nonzero(motion_times >= self.times[:, 0]).squeeze(-1)


class MaskedMimicControl(MimicControl):
    """Own a stable body/property mask and reference lookahead plan per episode."""

    config: MaskedMimicControlConfig
    # The parent's ``mimic.*`` names plus ``masked_mimic.*``. Each modality
    # beyond the global targets is a switch of the config (the ``requires=``
    # tags in ``context_views.MaskedMimicContext`` name the switches), and a
    # disabled modality is not published.
    namespaces = ("mimic", "masked_mimic")

    @classmethod
    def check_bindings(cls, config, robot_config, bound_paths):
        """Refuse a binding the trained goal layout cannot serve (``GoalCanvas``)."""
        GoalCanvas.derive(config, robot_config).validate(bound_paths, getup=False)

    def __init__(self, config: MaskedMimicControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        if not math.isfinite(inputs.dt) or inputs.dt <= 0:
            raise ValueError("Control dt must be finite and positive")
        robot, device = inputs.robot, inputs.device
        constant = self.state.constant

        # The layout this sampler produces, derived once; it refuses an anchor
        # that anchor, local or style goals need but cannot condition.
        canvas = GoalCanvas.derive(config, robot)
        body_names = robot.kinematic_info.body_names
        trackable_names = robot.trackable_bodies_subset
        self.conditionable_body_ids = constant(
            "conditionable_body_ids",
            torch.tensor(
                [body_names.index(name) for name in trackable_names],
                device=device,
                dtype=torch.long,
            ),
        )
        self.num_conditionable_bodies = len(trackable_names)
        if not self.num_conditionable_bodies:
            raise ValueError("MaskedMimic requires at least one trackable body")

        # Device constants reused across resets (registered so the attribute
        # guard knows they are deliberate).
        self._profile_gap_lower = constant(
            "profile_gap_lower", torch.tensor([0.0, 0.25, 0.65], device=device)
        )
        self._profile_gap_upper = constant(
            "profile_gap_upper", torch.tensor([0.25, 0.65, 1.0], device=device)
        )
        self._future_step_indices = constant(
            "future_step_indices",
            torch.arange(1, config.num_masked_future_steps + 1, device=device),
        )
        self._body_ranks = constant(
            "body_ranks", torch.arange(self.num_conditionable_bodies, device=device)
        )
        self._profile_weights_key = None
        self._profile_weights = None

        # False until a row's first reset (the registry's row reset writes
        # True); the streams rebase once any row has a plan.
        self.planned = self.state.add("planned", (), torch.bool, True)
        self.planned.zero_()

        num_steps = config.num_masked_future_steps
        self.global_stream = TargetStream(
            self.state.child("global_stream"),
            num_steps,
            self.num_conditionable_bodies,
            config.sparse_target_condition_probability,
        )
        self.local_stream = None
        if canvas.has_local_stream:
            self.local_stream = TargetStream(
                self.state.child("local_stream"),
                num_steps,
                self.num_conditionable_bodies,
                config.local_target_condition_probability,
            )

        self._conditionable_anchor_index = None
        if (
            canvas.anchor_pose_submasking
            or canvas.has_local_stream
            or canvas.style_frames is not None
        ):
            self._conditionable_anchor_index = trackable_names.index(canvas.anchor_body)

        add = self.state.add
        self.anchor_translation_mask = None
        self.anchor_rotation_mask = None
        if config.anchor_pose_submasking:
            self.anchor_translation_mask = add(
                "anchor_translation_mask", (num_steps, 2), torch.bool, False
            )
            self.anchor_rotation_mask = add(
                "anchor_rotation_mask", (num_steps, 2), torch.bool, False
            )

        self.style_time_offsets = None
        self.style_frame_masks = None
        self.style_condition_masks = None
        if config.style is not None:
            frames = config.style.num_frames
            self.style_time_offsets = add(
                "style_time_offsets", (frames,), torch.float, 0.0
            )
            self.style_frame_masks = add(
                "style_frame_masks", (frames,), torch.bool, False
            )
            self.style_condition_masks = add(
                "style_condition_masks",
                (frames, self.num_conditionable_bodies * 2 + 1),
                torch.bool,
                False,
            )
            total = sum(config.style.visible_frame_count_probabilities)
            self._style_count_weights = constant(
                "style_count_weights",
                torch.tensor(
                    [p / total for p in config.style.visible_frame_count_probabilities],
                    device=device,
                ),
            )

        self.velocity_command_mask = None
        if config.velocity_command is not None:
            self.velocity_command_mask = add(
                "velocity_command_mask", (1,), torch.bool, False
            )

        self._fixed_body_mask = None
        if config.fixed_conditioning is not None:
            self._fixed_body_mask = constant(
                "fixed_body_mask",
                torch.zeros(
                    self.num_conditionable_bodies, 2, dtype=torch.bool, device=device
                ),
            )
            seen_bodies = set()
            for condition in config.fixed_conditioning:
                if (
                    condition.body_name in seen_bodies
                    or condition.constraint_state not in (0, 1, 2)
                ):
                    raise ValueError(
                        "Fixed conditions require unique bodies and states 0, 1, or 2"
                    )
                seen_bodies.add(condition.body_name)
                body_index = trackable_names.index(condition.body_name)
                self._fixed_body_mask[body_index, 0] = condition.constraint_state <= 1
                self._fixed_body_mask[body_index, 1] = condition.constraint_state >= 1

    def _streams(self) -> List[TargetStream]:
        streams = [self.global_stream]
        if self.local_stream is not None:
            streams.append(self.local_stream)
        return streams

    def reset_rows(self, env_ids: Tensor, state: RobotState) -> None:
        """Resample only the reset environments' lookaheads and visibility
        (the registry's clear first: it also marks the rows planned)."""
        super().reset_rows(env_ids, state)
        if len(env_ids) == 0:
            return
        body_masks, profiles = self._reset_stream(self.global_stream, env_ids)
        if self.config.anchor_pose_submasking:
            translation, rotation = self._sample_anchor_masks(body_masks, profiles)
            self.anchor_translation_mask[env_ids] = translation
            self.anchor_rotation_mask[env_ids] = rotation
        if self.local_stream is not None:
            self._reset_stream(self.local_stream, env_ids)
        if self.config.style is not None:
            self._sample_style_plan(env_ids)
        if self.config.velocity_command is not None:
            self.velocity_command_mask[env_ids] = (
                torch.rand(
                    len(env_ids),
                    1,
                    device=self.inputs.device,
                    generator=self.inputs.generator,
                )
                < self.config.velocity_command.probability
            )

    def _reset_stream(
        self, stream: TargetStream, env_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        offsets, body_masks, profiles = self._sample_target_plan(len(env_ids))
        self._mask_disabled_stream(body_masks, stream.probability)
        stream.write_plan(env_ids, offsets, body_masks)
        self._rebase_stream_times(stream, env_ids)
        return body_masks, profiles

    def _rebase_stream_times(self, stream: TargetStream, env_ids: Tensor) -> None:
        reference = self.inputs.reference
        stream.rebase_times(
            env_ids,
            reference.times[env_ids],
            reference.lengths(reference.ids[env_ids]),
        )

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        """Keep target times fixed until the first is reached, then move them forward."""
        logging = super().advance(state)
        if not bool(self.planned.any()):
            return logging
        # Visibility masks observations, not the schedule clock: a hidden first
        # slot still rebases every slot of its stream.
        for stream in self._streams():
            env_ids = stream.due(self.inputs.reference.times)
            if len(env_ids) > 0:
                self._rebase_stream_times(stream, env_ids)
        return logging

    def _compute_average_velocity(self, env_ids: Optional[Tensor] = None) -> Tensor:
        """Reference anchor XY displacement over the available lookahead.

        Spawn and terrain offsets are identical for both frames of one
        environment, so raw reference positions give the same XY velocity.
        """
        selected = slice(None) if env_ids is None else env_ids
        reference = self.inputs.reference
        motion_ids = reference.ids[selected]
        motion_times = reference.times[selected]
        future_times = torch.minimum(
            motion_times + self.config.velocity_command.horizon_seconds,
            reference.lengths(motion_ids),
        )
        anchor_pos = reference.raw(
            torch.cat((motion_ids, motion_ids)), torch.cat((motion_times, future_times))
        ).rigid_body_pos[:, self.inputs.robot.anchor_body_index, :2]
        current_pos, future_pos = anchor_pos.chunk(2)
        duration = (future_times - motion_times).unsqueeze(-1)
        velocity = (future_pos - current_pos) / duration.clamp_min(self.inputs.dt)
        return torch.where(duration > 0, velocity, torch.zeros_like(velocity))

    def _sample_style_plan(self, env_ids: Tensor) -> None:
        """Keep offsets, visible frames and body/property choices until reset."""
        style = self.config.style
        num_envs, device = len(env_ids), self.inputs.device
        generator = self.inputs.generator
        self.style_time_offsets[env_ids] = (
            torch.rand(
                num_envs, style.num_frames, device=device, generator=generator
            ).pow(style.near_time_bias_power)
            * style.window_seconds
        )
        counts = torch.multinomial(
            self._style_count_weights, num_envs, replacement=True, generator=generator
        )
        # The second argsort inverts the permutation into each frame's rank.
        ranks = (
            torch.rand(num_envs, style.num_frames, device=device, generator=generator)
            .argsort(-1)
            .argsort(-1)
        )
        frame_masks = ranks < counts[:, None]
        profiles = self._sample_profiles(num_envs)
        body_masks = self._sample_body_masks(profiles)
        velocity_mask = (
            torch.rand(num_envs, 1, device=device, generator=generator)
            < style.velocity_probability
        )
        condition_masks = torch.cat((body_masks.flatten(1), velocity_mask), -1)
        self.style_frame_masks[env_ids] = frame_masks
        self.style_condition_masks[env_ids] = (
            condition_masks[:, None] & frame_masks[..., None]
        )

    def _query_style_samples(self, env_ids: Optional[Tensor]) -> Dict[str, Tensor]:
        """Query the moving reference window without exposing its times.

        For E environments, F frames and B conditionable bodies, return poses
        [E, F*B*9], velocities [E, F*B*6], condition masks [E, F, 2*B+1], and
        frame masks [E, F].
        """
        selected = slice(None) if env_ids is None else env_ids
        reference = self.inputs.reference
        motion_ids = reference.ids[selected]
        motion_times = reference.times[selected]
        lengths = reference.lengths(motion_ids)
        frame_masks = (
            self.style_frame_masks[selected] & (motion_times <= lengths)[:, None]
        )
        condition_masks = self.style_condition_masks[selected] & frame_masks[..., None]
        num_envs, num_frames = frame_masks.shape
        num_bodies = self.num_conditionable_bodies
        pose = torch.zeros(
            num_envs,
            num_frames * num_bodies * STYLE_POSE_FEATURES_PER_BODY,
            device=self.inputs.device,
        )
        velocity = torch.zeros(
            num_envs,
            num_frames * num_bodies * STYLE_VELOCITY_FEATURES_PER_BODY,
            device=self.inputs.device,
        )
        active_env_ids = frame_masks.any(-1).nonzero().squeeze(-1)
        if len(active_env_ids):
            query_times = torch.minimum(
                motion_times[active_env_ids, None]
                + self.style_time_offsets[selected][active_env_ids],
                lengths[active_env_ids, None],
            )
            # Raw clip poses: the style samples are expressed relative to each
            # frame's own anchor, so the env's placement cancels.
            state = reference.raw(
                motion_ids[active_env_ids, None].expand(-1, num_frames).reshape(-1),
                query_times.reshape(-1),
            )
            pos = state.rigid_body_pos.reshape(len(active_env_ids), num_frames, -1, 3)
            rot = state.rigid_body_rot.reshape(len(active_env_ids), num_frames, -1, 4)
            pose[active_env_ids] = compute_style_pose_samples(
                pos,
                rot,
                anchor_body_index=self.inputs.robot.anchor_body_index,
                conditionable_body_ids=self.conditionable_body_ids,
            )
            velocity[active_env_ids] = compute_style_velocity_samples(
                state.rigid_body_vel.reshape_as(pos),
                state.rigid_body_ang_vel.reshape_as(pos),
                rot,
                anchor_body_index=self.inputs.robot.anchor_body_index,
                conditionable_body_ids=self.conditionable_body_ids,
            )
        pose, velocity = apply_style_condition_masks(pose, velocity, condition_masks)
        return dict(
            style_pose_samples=pose,
            style_velocity_samples=velocity,
            style_condition_masks=condition_masks,
            style_frame_masks=frame_masks,
        )

    def _mask_disabled_stream(self, masks: Tensor, probability: float) -> None:
        # No random draws for the default global-only path or deterministic gates.
        if probability == 0:
            masks.zero_()
        elif probability < 1:
            enabled = (
                torch.rand(
                    masks.shape[0], device=masks.device, generator=self.inputs.generator
                )
                < probability
            )
            masks &= enabled[:, None, None, None]

    def _sample_anchor_masks(
        self, body_masks: Tensor, profiles: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One submode per episode, intersected with each frame's visibility."""
        n, device, generator = (
            body_masks.shape[0],
            body_masks.device,
            self.inputs.generator,
        )
        translation_mode = torch.randint(3, (n,), device=device, generator=generator)
        rotation_mode = torch.randint(2, (n,), device=device, generator=generator)
        translation = torch.stack((translation_mode != 1, translation_mode != 0), -1)
        rotation = torch.stack((rotation_mode == 0, rotation_mode == 1), -1)
        translation[profiles == 0] = True
        rotation[profiles == 0, 0] = True
        rotation[profiles == 0, 1] = False
        active = body_masks[:, :, self._conditionable_anchor_index]
        return (
            translation[:, None] & active[..., 0:1],
            rotation[:, None] & active[..., 1:2],
        )

    def _sample_target_plan(self, num_envs: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return offsets, per-frame body/property masks, and density profiles."""
        config = self.config
        device, generator = self.inputs.device, self.inputs.generator
        num_steps = config.num_masked_future_steps
        if num_envs == 0:
            return (
                torch.empty(0, num_steps, device=device),
                torch.empty(
                    0,
                    num_steps,
                    self.num_conditionable_bodies,
                    2,
                    dtype=torch.bool,
                    device=device,
                ),
                torch.empty(0, dtype=torch.long, device=device),
            )
        profile_ids = self._sample_profiles(num_envs)
        is_dense, is_sparse = profile_ids == 0, profile_ids == 2

        # Dense uses the first 25% of the gap range, intermediate the middle
        # 40%, and sparse the final 35%. Each episode has one fixed spacing.
        gap_span = config.target_gap_max_seconds - config.target_gap_min_seconds
        gap_min = (
            config.target_gap_min_seconds
            + gap_span * self._profile_gap_lower[profile_ids]
        )
        gap_max = (
            config.target_gap_min_seconds
            + gap_span * self._profile_gap_upper[profile_ids]
        )
        gap = gap_min + torch.rand(num_envs, device=device, generator=generator) * (
            gap_max - gap_min
        ).clamp_min(0.0)
        schedule_offsets = (
            gap.clamp_min(self.inputs.dt).unsqueeze(-1) * self._future_step_indices
        )
        body_mask = self._sample_body_masks(profile_ids)

        frame_visible = (
            torch.rand(num_envs, num_steps, device=device, generator=generator)
            < config.visible_target_pose_prob
        )
        frame_visible[is_dense] = True
        sparse_env_ids = torch.nonzero(is_sparse).squeeze(-1)
        if len(sparse_env_ids) > 0:
            frame_visible[sparse_env_ids] = False
            sparse_frame_ids = torch.randint(
                num_steps, (len(sparse_env_ids),), device=device, generator=generator
            )
            frame_visible[sparse_env_ids, sparse_frame_ids] = True
        return (
            schedule_offsets,
            body_mask.unsqueeze(1) & frame_visible[:, :, None, None],
            profile_ids,
        )

    def _sample_profiles(self, num_envs: int) -> Tensor:
        config, device = self.config, self.inputs.device
        # Keep runtime probability edits working without repeated host-to-device copies.
        weights_key = tuple(config.target_profile_probabilities)
        if weights_key != self._profile_weights_key:
            # Normalize in Python first so large finite weights do not overflow float32.
            total = sum(weights_key)
            self._profile_weights = self.state.constant(
                "profile_weights",
                torch.tensor([p / total for p in weights_key], device=device),
            )
            self._profile_weights_key = weights_key
        return torch.multinomial(
            self._profile_weights,
            num_envs,
            replacement=True,
            generator=self.inputs.generator,
        )

    def _sample_body_masks(self, profile_ids: Tensor) -> Tensor:
        num_envs, num_bodies = len(profile_ids), self.num_conditionable_bodies
        device, generator = self.inputs.device, self.inputs.generator
        is_dense, is_sparse = profile_ids == 0, profile_ids == 2
        # Preserve the training sampler's draw order, including fixed-body plans.
        body_count = (
            (
                torch.rand(num_envs, device=device, generator=generator)
                * (num_bodies - 1)
                + 1
            )
            .round()
            .long()
        )
        body_count[is_dense] = num_bodies
        body_count[is_sparse] = 1
        constraint_states = torch.randint(
            3, (num_envs, num_bodies), device=device, generator=generator
        )
        if self._fixed_body_mask is None:
            body_order = torch.rand(
                num_envs, num_bodies, device=device, generator=generator
            ).argsort(1)
            # Invert the permutation in linear work instead of sorting it again.
            body_ranks = torch.empty_like(body_order).scatter_(
                1, body_order, self._body_ranks.expand_as(body_order)
            )
            selected_bodies = body_ranks < body_count.unsqueeze(-1)
            body_mask = torch.stack(
                (constraint_states <= 1, constraint_states >= 1), dim=-1
            )
            body_mask &= selected_bodies.unsqueeze(-1)
            body_mask[is_dense] = True
        else:
            body_mask = self._fixed_body_mask.unsqueeze(0).expand(num_envs, -1, -1)

        return body_mask

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Add the ``masked_mimic.*`` names to the parent's ``mimic.*`` names.

        Only the enabled modalities are published: a binding to a disabled
        one is refused by the config-only check and by the build check, which
        name the switch (``requires=`` in ``context_views.py``).
        """
        published = super().publish(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        motion_times = self.inputs.reference.times[selected]

        fields: Dict[str, Tensor] = self._stream_context(
            "", self.global_stream, motion_times, env_ids
        )
        if self.anchor_translation_mask is not None:
            fields.update(
                anchor_translation_mask=self.anchor_translation_mask[selected],
                anchor_rotation_mask=self.anchor_rotation_mask[selected],
            )
        if self.local_stream is not None:
            fields.update(
                self._stream_context("local_", self.local_stream, motion_times, env_ids)
            )
        if self.config.style is not None:
            fields.update(self._query_style_samples(env_ids))
        if self.config.velocity_command is not None:
            fields.update(
                average_velocity=self._compute_average_velocity(env_ids),
                velocity_command_mask=self.velocity_command_mask[selected],
            )
        published.update(
            {f"masked_mimic.{name}": value for name, value in fields.items()}
        )
        return published

    def _stream_context(
        self, prefix: str, stream: TargetStream, motion_times, env_ids
    ) -> Dict[str, Tensor]:
        selected = slice(None) if env_ids is None else env_ids
        times = stream.times[selected]
        # Either stream in the same placed world frame: every slot shares the
        # placement of the first (target times are already clipped to the
        # motion length when written).
        window = self.inputs.reference.window(env_ids, times)
        return {
            f"{prefix}ref_pos": window.pos,
            f"{prefix}ref_rot": window.rot,
            f"{prefix}target_times": times,
            f"{prefix}time_offsets": times - motion_times.unsqueeze(-1),
            f"{prefix}target_poses_masks": stream.frame_masks[selected],
            f"{prefix}target_bodies_masks": stream.body_masks[selected],
        }

    _MARKER_COLORS = {
        "body_markers_blue": (0.0, 0.0, 1.0),
        "body_markers_yellow": (1.0, 1.0, 0.0),
        "body_markers_red": (1.0, 0.0, 0.0),
    }

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Conditionable-body spheres colored by time to the first visible target.

        Blue: more than 1s away. Yellow: between 0.1s and 1s. Red: within 0.1s.
        """
        if headless:
            return {}
        small_bodies = self.inputs.robot.mimic_small_marker_bodies or ()
        body_markers = [
            MarkerConfig(size="small" if body_name in small_bodies else "regular")
            for body_name in self.inputs.robot.trackable_bodies_subset
        ]
        return {
            name: VisualizationMarkerConfig(
                type="sphere", color=color, markers=body_markers
            )
            for name, color in self._MARKER_COLORS.items()
        }

    def get_markers_state(self, state: RobotState) -> Dict[str, MarkerState]:
        """Show the first visible target's translation goals, colored by time left."""
        if self.inputs.headless or not bool(self.planned.any()):
            return {}

        num_envs, device = self.inputs.num_envs, self.inputs.device
        stream = self.global_stream
        env_ids = torch.arange(num_envs, device=device)
        first_visible_frame = torch.argmax(stream.frame_masks.float(), dim=1)
        target_motion_times = stream.times[env_ids, first_visible_frame]
        reference = self.inputs.reference
        time_remaining = target_motion_times - reference.times

        target_pos = reference.poses_at(None, target_motion_times).rigid_body_pos
        target_pos = target_pos[:, self.conditionable_body_ids, :]

        body_masks = stream.body_masks.view(
            num_envs,
            self.config.num_masked_future_steps,
            self.num_conditionable_bodies,
            2,
        )
        visible_positions = body_masks[env_ids, first_visible_frame, :, 0].clone()
        if self.anchor_translation_mask is not None:
            # A point marker implies a full XYZ goal, not an XY plane or height.
            visible_positions[:, self._conditionable_anchor_index] &= (
                self.anchor_translation_mask[env_ids, first_visible_frame].all(-1)
            )

        time_masks = (
            time_remaining > 1.0,
            (time_remaining > 0.1) & (time_remaining <= 1.0),
            time_remaining <= 0.1,
        )
        markers_state = {}
        for name, time_mask in zip(self._MARKER_COLORS, time_masks):
            positions = target_pos.clone()
            # Move inactive markers off screen.
            positions[~(visible_positions & time_mask[:, None])] += 100
            markers_state[name] = MarkerState(
                translation=positions.view(num_envs, -1, 3),
                orientation=torch.zeros(
                    num_envs, self.num_conditionable_bodies, 4, device=device
                ),
            )
        return markers_state
