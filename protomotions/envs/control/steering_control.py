# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steering control component for locomotion tasks.

Owns the heading schedule of a steering task: a target direction, a target
facing direction and a target speed per env, resampled when the env's
``heading_change_steps`` is reached (a random walk from the previous heading
and speed, or a fresh draw), and the double buffer of the anchor position
the velocity reward reads. Publishes the ``steering.*`` names
(``SteeringContext``) and owns no episode-ending check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils import rotations


@dataclass
class SteeringControlConfig(ControlComponentConfig):
    """Configuration for steering control component.

    Attributes:
        tar_speed_min: Minimum target speed.
        tar_speed_max: Maximum target speed.
        heading_change_steps_min: Minimum steps between heading changes.
        heading_change_steps_max: Maximum steps between heading changes.
        random_heading_probability: Probability of fully random heading vs incremental change.
        standard_heading_change: Maximum incremental heading change (radians).
        standard_speed_change: Maximum incremental speed change.
        stop_probability: Probability of setting speed to zero.
        enable_rand_facing: Enable independent random facing direction (for strafing, etc).
    """

    _target_: str = "protomotions.envs.control.steering_control.SteeringControl"

    tar_speed_min: float = 0.0
    tar_speed_max: float = 2.0
    heading_change_steps_min: int = 50
    heading_change_steps_max: int = 150
    random_heading_probability: float = 0.1
    standard_heading_change: float = 0.5  # radians
    standard_speed_change: float = 0.5
    stop_probability: float = 0.1
    enable_rand_facing: bool = True


class SteeringControl(ControlComponent):
    """The heading schedule and the anchor double buffer of a steering task."""

    config: SteeringControlConfig
    namespaces = ("steering",)

    def __init__(self, config: SteeringControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        add = self.state.add
        self.heading_change_steps = add("heading_change_steps", (), torch.int64, 0)
        self.tar_dir_theta = add("tar_dir_theta", (), torch.float, 0.0)
        # Default: forward.
        self.tar_dir = add("tar_dir", (2,), torch.float, torch.tensor([1.0, 0.0]))
        # The facing direction can differ from the movement direction (strafing).
        self.tar_face_dir = add(
            "tar_face_dir", (2,), torch.float, torch.tensor([1.0, 0.0])
        )
        self.tar_speed = add("tar_speed", (), torch.float, 1.0)
        # Double buffer of the anchor position (prev = t-1, curr = t), so the
        # velocity reward can difference two consecutive positions.
        self.prev_root_pos = add("prev_root_pos", (3,), torch.float, 0.0)
        self.curr_root_pos = add("curr_root_pos", (3,), torch.float, 0.0)

    def reset_rows(self, env_ids: Tensor, state: RobotState) -> None:
        """Seed the anchor double buffer from the post-reset pose and sample a
        new episode's first commands.

        Deliberately NOT the registry's clear: the schedule is a random walk,
        and ``tar_dir_theta`` and ``tar_speed`` carry the previous episode's
        values into the incremental branch of ``_sample_commands`` (the
        reference behaviour); every other slot is rewritten here.
        """
        if len(env_ids) == 0:
            return
        anchor_pos = state.rigid_body_pos[:, self.inputs.robot.anchor_body_index]
        self.prev_root_pos[env_ids] = anchor_pos
        self.curr_root_pos[env_ids] = anchor_pos
        self._sample_commands(env_ids)

    def resample(self, env_ids: Tensor, state: RobotState) -> None:
        """New commands inside the episode (``advance`` found the change due).
        The anchor double buffer is episode-start state and is not touched, so
        the velocity reward sees a true displacement on the change step."""
        self._sample_commands(env_ids)

    def _sample_commands(self, env_ids: Tensor) -> None:
        """The command draw for ``env_ids``: a per-env choice between a fresh
        heading and speed and an incremental change from the current ones, the
        next change step, a stop and the facing direction. Every draw is on the
        control stage's generator."""
        config = self.config
        device, generator = self.inputs.device, self.inputs.generator
        n = len(env_ids)

        # Per-environment sampling: which envs get a random heading vs incremental
        rand_probs = torch.ones(n, device=device) * config.random_heading_probability
        use_random = torch.bernoulli(rand_probs, generator=generator).bool()

        # Fully random heading and speed (for envs with use_random=True)
        rand_dir_theta = (
            2 * np.pi * torch.rand(n, device=device, generator=generator) - np.pi
        )
        rand_tar_speed = (config.tar_speed_max - config.tar_speed_min) * torch.rand(
            n, device=device, generator=generator
        ) + config.tar_speed_min

        # Incremental change from the current heading/speed (use_random=False)
        dir_delta_theta = (
            2
            * config.standard_heading_change
            * torch.rand(n, device=device, generator=generator)
            - config.standard_heading_change
        )
        inc_dir_theta = (dir_delta_theta + self.tar_dir_theta[env_ids] + np.pi) % (
            2 * np.pi
        ) - np.pi

        speed_delta = (
            2
            * config.standard_speed_change
            * torch.rand(n, device=device, generator=generator)
            - config.standard_speed_change
        )
        inc_tar_speed = torch.clamp(
            speed_delta + self.tar_speed[env_ids],
            min=config.tar_speed_min,
            max=config.tar_speed_max,
        )

        # Select per-environment based on the use_random mask
        dir_theta = torch.where(use_random, rand_dir_theta, inc_dir_theta)
        tar_speed = torch.where(use_random, rand_tar_speed, inc_tar_speed)

        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)

        # Sample when to change heading next
        change_steps = torch.randint(
            low=config.heading_change_steps_min,
            high=config.heading_change_steps_max,
            size=(n,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )

        # Randomly set some targets to stop (speed=0)
        stop_probs = torch.ones(n, device=device) * config.stop_probability
        should_stop = torch.bernoulli(stop_probs, generator=generator)

        # Facing direction: independent from the movement direction if enabled
        if config.enable_rand_facing:
            face_theta = (
                2 * np.pi * torch.rand(n, device=device, generator=generator) - np.pi
            )
        else:
            face_theta = dir_theta  # face the movement direction
        tar_face_dir = torch.stack(
            [torch.cos(face_theta), torch.sin(face_theta)], dim=-1
        )

        self.tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self.tar_dir_theta[env_ids] = dir_theta
        self.tar_dir[env_ids] = tar_dir
        self.tar_face_dir[env_ids] = tar_face_dir
        progress = self.inputs.clock.progress_buf[env_ids]
        self.heading_change_steps[env_ids] = progress + change_steps

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        """Rotate the anchor double buffer, then resample the envs whose
        heading change is due."""
        anchor = self.inputs.robot.anchor_body_index
        self.prev_root_pos[:] = self.curr_root_pos
        self.curr_root_pos[:] = state.rigid_body_pos[:, anchor]

        due = self.inputs.clock.progress_buf >= self.heading_change_steps
        env_ids = due.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.resample(env_ids, state)
        return {}

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Publish the ``steering.*`` names (see ``SteeringContext``)."""
        rows = slice(None) if env_ids is None else env_ids
        return {
            "steering.tar_dir": self.tar_dir[rows],
            "steering.tar_dir_theta": self.tar_dir_theta[rows],
            "steering.tar_speed": self.tar_speed[rows],
            "steering.tar_face_dir": self.tar_face_dir[rows],
            "steering.prev_root_pos": self.prev_root_pos[rows],
        }

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Two arrows: red for the movement direction, blue for the facing."""
        if headless:
            return {}
        return {
            "movement_markers": VisualizationMarkerConfig(
                type="arrow",
                color=(0.8, 0.0, 0.0),
                markers=[MarkerConfig(size="regular")],
            ),
            "facing_markers": VisualizationMarkerConfig(
                type="arrow",
                color=(0.0, 0.0, 0.8),
                markers=[MarkerConfig(size="regular")],
            ),
        }

    def get_markers_state(self, state: RobotState) -> Dict[str, MarkerState]:
        """The two arrows at the anchor, pointing along the two directions."""
        if self.inputs.headless:
            return {}
        num_envs = self.inputs.num_envs
        root_pos = state.rigid_body_pos[:, self.inputs.robot.anchor_body_index]
        heading_axis = torch.zeros_like(root_pos)
        heading_axis[..., -1] = 1.0

        movement_marker_pos = root_pos.clone()
        movement_marker_pos[..., 0:2] += self.tar_dir
        movement_theta = torch.atan2(self.tar_dir[..., 1], self.tar_dir[..., 0])
        movement_rot = rotations.quat_from_angle_axis(
            movement_theta, heading_axis, True
        )

        facing_marker_pos = root_pos.clone()
        facing_marker_pos[..., 0:2] += self.tar_face_dir
        facing_theta = torch.atan2(self.tar_face_dir[..., 1], self.tar_face_dir[..., 0])
        facing_rot = rotations.quat_from_angle_axis(facing_theta, heading_axis, True)

        return {
            "movement_markers": MarkerState(
                translation=movement_marker_pos.view(num_envs, -1, 3),
                orientation=movement_rot.view(num_envs, -1, 4),
            ),
            "facing_markers": MarkerState(
                translation=facing_marker_pos.view(num_envs, -1, 3),
                orientation=facing_rot.view(num_envs, -1, 4),
            ),
        }
