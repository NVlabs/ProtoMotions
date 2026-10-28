# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic control component for motion tracking tasks.

Publishes the reference motion a tracking task follows (``mimic.*``: the
current reference pose, the future window, the anchor slices, the clip-done
flag), read through the placed reference query; owns one episode-ending
check, ``clip_done`` (the clip runs out within the next step), a truncation
under ``bootstrap_on_episode_end`` and a failure otherwise; draws the target
pose markers. The env moves the reference clock, so ``advance`` has nothing
to do.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.humanoid import dof_to_local
from protomotions.envs.steps import normalize_steps
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator_state import RobotState


@dataclass
class MimicControlConfig(ControlComponentConfig):
    """Configuration for mimic control component.

    Attributes:
        bootstrap_on_episode_end: If True, don't terminate when motion clip ends.
        future_steps: The 1-based future reference steps to publish, in order.
            A recipe may write an int N for the consecutive steps ``1..N`` or
            the specific steps (``[1, 3, 5, 9, 15]`` for non-uniform sampling);
            ``__post_init__`` normalises either to a tuple of ints. A CLI
            override is written after ``__post_init__`` and may leave the int,
            so every reader goes through :func:`normalize_steps` as well.
        reference_height_smoothing_seconds: EMA time constant for reference
            anchor height. Zero disables sampling. Uses 51 past-to-current
            samples over five time constants, independently of episode history.
        low_pose_grace_seconds: Trailing minimum window for smoothed reference
            height. Keeps low-pose termination limits briefly after standing up.
            Rounded up to the height sampler's tau/10 interval; zero disables it.
    """

    _target_: str = "protomotions.envs.control.mimic_control.MimicControl"

    bootstrap_on_episode_end: bool = True
    future_steps: Union[int, Tuple[int, ...]] = 1
    reference_height_smoothing_seconds: float = 0.0
    low_pose_grace_seconds: float = 0.0

    def __post_init__(self):
        # The one normalisation of the published horizon: every reader
        # (the control, the recipes, the exporter) sees a tuple of 1-based steps.
        self.future_steps = normalize_steps(self.future_steps)


def clip_done_flag(clip_done: Tensor = EnvContext.mimic.clip_done) -> Tensor:
    """The published clip-done flag is the check: a boolean kernel."""
    return clip_done


class MimicControl(ControlComponent):
    """The reference motion of a tracking task, published; the clip-done check."""

    config: MimicControlConfig
    namespaces = ("mimic",)

    @classmethod
    def terminations(
        cls, config: MimicControlConfig, robot_config
    ) -> Dict[str, TerminationRule]:
        """``clip_done``: the episode ends when the reference clip runs out
        within the next step; under ``bootstrap_on_episode_end`` it is a
        truncation (the env resets, the agent bootstraps), otherwise a
        failure (reset and terminated)."""
        return {
            "clip_done": TerminationRule(
                MdpComponent.from_kernel(clip_done_flag),
                truncation=config.bootstrap_on_episode_end,
            )
        }

    def __init__(self, config: MimicControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        if (
            not math.isfinite(config.reference_height_smoothing_seconds)
            or config.reference_height_smoothing_seconds < 0
        ):
            raise ValueError(
                "reference_height_smoothing_seconds must be finite and nonnegative"
            )
        if (
            not math.isfinite(config.low_pose_grace_seconds)
            or config.low_pose_grace_seconds < 0
        ):
            raise ValueError("low_pose_grace_seconds must be finite and nonnegative")
        if (
            config.low_pose_grace_seconds > 0
            and config.reference_height_smoothing_seconds == 0
        ):
            raise ValueError("low_pose_grace_seconds requires reference height smoothing")
        # The future window's time offsets, in seconds: a constant of the config.
        # normalize_steps again: an override written after __post_init__ may
        # have left the int form.
        self.future_time_offsets = self.state.constant(
            "future_time_offsets",
            inputs.dt
            * torch.as_tensor(
                normalize_steps(config.future_steps),
                device=inputs.device,
                dtype=torch.float32,
            ),
        )

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        """Nothing to advance: the env moves the reference clock before any
        control component advances, so the reference time is already this
        step's here."""
        return {}

    def _smoothed_reference_height(self, env_ids: Optional[Tensor]) -> Tensor:
        """Finite causal EMA, seeded at the oldest sample, with no rollout state.

        Sample spacing is tau/10, not the simulation timestep. Clipping queries
        to the clip boundaries extends endpoint poses rather than wrapping into
        another motion. Terrain correction matches ordinary reference poses.
        """
        reference = self.inputs.reference
        motion_ids = reference.ids if env_ids is None else reference.ids[env_ids]
        motion_times = reference.times if env_ids is None else reference.times[env_ids]
        tau = self.config.reference_height_smoothing_seconds
        grace_steps = math.ceil(self.config.low_pose_grace_seconds / (tau / 10))
        sample_count = 51 + grace_steps
        ages = torch.arange(
            sample_count - 1, -1, -1, device=motion_times.device, dtype=motion_times.dtype
        )
        times = (motion_times[:, None] - ages * (tau / 10)).clamp_min(0)
        times = torch.minimum(times, reference.lengths(motion_ids)[:, None])
        ids = motion_ids[:, None].expand_as(times).reshape(-1)
        positions = reference.raw_positions(ids, times.reshape(-1))
        if env_ids is None:
            env_ids = torch.arange(self.inputs.num_envs, device=motion_ids.device)
        offset = reference.placement_offset(
            positions, env_ids.repeat_interleave(sample_count)
        )
        anchor = self.inputs.robot.anchor_body_index
        # Terrain correction applies one shared vertical translation to each pose.
        heights = (positions[:, anchor, 2] + offset[:, 0, 2]).reshape(-1, sample_count)
        weights = torch.exp(-ages[-51:] / 10) * (-math.expm1(-0.1))
        # The oldest sample seeds the recurrence and absorbs the omitted tail.
        weights[0] = math.exp(-5)
        # Share one motion query across overlapping EMA windows. Reference-only
        # grace cannot be extended by the robot falling, repeated context reads,
        # or resets; snapshot restore therefore needs no extra rollout state.
        smoothed = (heights.unfold(-1, 51, 1) * weights).sum(-1)
        return smoothed.amin(-1)

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Publish the ``mimic.*`` names (see ``MimicContext`` in context_views).

        - ``mimic.ref_state.<field>``: every tensor the single-step reference
          state at the current time carries (for rewards)
        - ``mimic.future_*``: multi-step future reference poses
          [envs, future_steps, ...] and their root / anchor slices
        - ``mimic.ref_anchor_*``: the current-frame reference anchor's
          position, rotation, velocity and angular velocity
        - ``mimic.clip_done``: whether the clip ends within the next step
        - ``mimic.smoothed_ref_anchor_height`` with reference height smoothing
        """
        reference = self.inputs.reference
        # The current frame (placed per pose) and the future window (placed
        # from its first slot), from one library query.
        ref_state, future = reference.current_and_window(
            env_ids, self.future_time_offsets
        )
        future_pos, future_rot = future.pos, future.rot
        future_vel, future_ang_vel = future.vel, future.ang_vel
        future_dof_pos, future_dof_vel = future.dof_pos, future.dof_vel

        hinge_axes_map = self.inputs.robot.kinematic_info.hinge_axes_map
        ref_lr = dof_to_local(ref_state.dof_pos, hinge_axes_map, True)
        anchor_idx = self.inputs.robot.anchor_body_index

        # Every tensor the reference state carries (contacts and local
        # rotations only when the motion file has them).
        published: Dict[str, Any] = {}
        for field in fields(ref_state):
            value = getattr(ref_state, field.name)
            if torch.is_tensor(value):
                published[f"mimic.ref_state.{field.name}"] = value
        published.update(
            {
                "mimic.future_pos": future_pos,
                "mimic.future_rot": future_rot,
                "mimic.future_vel": future_vel,
                "mimic.future_ang_vel": future_ang_vel,
                "mimic.future_dof_pos": future_dof_pos,
                "mimic.future_dof_vel": future_dof_vel,
                "mimic.anchor_idx": anchor_idx,
                "mimic.ref_lr": ref_lr,
                # Root (body 0) and anchor slices of the targets, and the
                # current-frame reference anchor.
                "mimic.future_root_pos": future_pos[:, :, 0, :],
                "mimic.future_root_rot": future_rot[:, :, 0, :],
                "mimic.future_root_vel": future_vel[:, :, 0, :],
                "mimic.future_root_ang_vel": future_ang_vel[:, :, 0, :],
                "mimic.ref_anchor_pos": ref_state.rigid_body_pos[:, anchor_idx, :],
                "mimic.ref_anchor_rot": ref_state.rigid_body_rot[:, anchor_idx, :],
                "mimic.ref_anchor_vel": ref_state.rigid_body_vel[:, anchor_idx, :],
                "mimic.ref_anchor_ang_vel": ref_state.rigid_body_ang_vel[
                    :, anchor_idx, :
                ],
                "mimic.future_anchor_pos": future_pos[:, :, anchor_idx, :],
                "mimic.future_anchor_rot": future_rot[:, :, anchor_idx, :],
                "mimic.future_anchor_vel": future_vel[:, :, anchor_idx, :],
                "mimic.future_anchor_ang_vel": future_ang_vel[:, :, anchor_idx, :],
                "mimic.clip_done": reference.done(env_ids),
            }
        )
        if self.config.reference_height_smoothing_seconds > 0:
            published["mimic.smoothed_ref_anchor_height"] = (
                self._smoothed_reference_height(env_ids)
            )
        return published

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Red spheres on every body of the reference pose."""
        if headless:
            return {}
        robot = self.inputs.robot
        small = robot.mimic_small_marker_bodies
        body_markers = [
            MarkerConfig(
                size="small" if small is not None and name in small else "regular"
            )
            for name in robot.kinematic_info.body_names
        ]
        return {
            "body_markers_red": VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 0.0), markers=body_markers
            )
        }

    def get_markers_state(self, state: RobotState) -> Dict[str, MarkerState]:
        """The reference pose at the current time, placed where each env spawned."""
        if self.inputs.headless:
            return {}
        reference = self.inputs.reference
        num_envs, device = self.inputs.num_envs, self.inputs.device
        target_pos = reference.poses_at(None, reference.times).rigid_body_pos
        target_pos = target_pos.view(num_envs, -1, 3)
        return {
            "body_markers_red": MarkerState(
                translation=target_pos,
                orientation=torch.zeros(
                    num_envs, target_pos.shape[1], 4, device=device
                ),
            )
        }
