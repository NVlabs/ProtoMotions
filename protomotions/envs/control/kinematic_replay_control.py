# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kinematic replay control: plays the reference motions without physics.

An animation player: every step it advances the clip time, resamples the
clips that ended, and asks the env to write the reference pose (at rest) and
the scene objects where each env spawned (``TaskInputs.teleport``, applied
once every control advanced) and to start the episodes over
(``TaskInputs.restart_episode``), so the env's own done checks never fire on
a replay. The viewer's reset key (``TaskInputs.viewer_reset``) resamples
every clip from time zero. Publishes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.simulator.base_simulator.simulator_state import (
    ResetState,
    RobotState,
)


@dataclass
class KinematicReplayControlConfig(ControlComponentConfig):
    _target_: str = (
        "protomotions.envs.control.kinematic_replay_control.KinematicReplayControl"
    )


class KinematicReplayControl(ControlComponent):
    """Plays reference motions kinematically (bypasses physics)."""

    config: KinematicReplayControlConfig

    def __init__(self, config: KinematicReplayControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        inputs = self.inputs
        reference = inputs.reference
        all_env_ids = torch.arange(
            inputs.num_envs, dtype=torch.long, device=inputs.device
        )

        # The viewer's reset key: resample every clip and restart it from t=0.
        if inputs.viewer_reset.consume():
            reference.resample(all_env_ids)
            reference.assign(all_env_ids, times=torch.zeros_like(reference.times))

        # Advance the clip time by one env step.
        reference.advance_time(inputs.dt)

        # Clips that ended get a new clip.
        done_clip = reference.done()
        if bool(done_clip.any()):
            reference.resample(torch.where(done_clip)[0])

        # The reference pose in the clip's frame, at rest (kinematic replay).
        ref_state = reference.raw(reference.ids, reference.times)
        ref_state.dof_vel *= 0
        ref_state.rigid_body_vel *= 0
        ref_state.rigid_body_ang_vel *= 0
        ref_reset_state = ResetState.from_robot_state(ref_state)

        # The scene objects at rest (None without scenes), and the root-only
        # placement that puts both where the env spawned.
        ref_object_state = reference.scene_poses_at(all_env_ids, reference.times)
        offset = reference.placement_offset(
            ref_reset_state.root_pos[:, None, :], all_env_ids
        ).squeeze(1)
        ref_reset_state.root_pos += offset
        if ref_object_state is not None:
            ref_object_state.root_pos += offset.unsqueeze(1)

        # The two declared writes: the pose, and the episode clock back to zero.
        inputs.teleport(all_env_ids, ref_reset_state, ref_object_state)
        inputs.restart_episode(all_env_ids)
        return {}

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Kinematic replay publishes no context names."""
        return {}
