# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path follower control component for navigation tasks.

Owns the path of a path-following task (a :class:`PathGenerator` polyline per
env, sampled at reset from the head's ground-relative position) and latches
the ground-relative head position every step; publishes the ``path.*`` names
(the target on the path now, the head position, the future waypoint samples)
and owns the path-deviation checks (``terminations``): ``path_distance``
(the head too far from the path's target) and, when height conditioned,
``path_height`` (the head too far above or below it), both only with
``enable_path_termination``. A failed check resets the env as every other
rule does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.terminations.task import path_distance, path_height_error
from protomotions.envs.utils.path_generator import PathGenerator, PathGeneratorConfig
from protomotions.components.pose_lib import build_body_ids_tensor
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator_state import RobotState


@dataclass
class PathFollowerControlConfig(ControlComponentConfig):
    """Configuration for path follower control component.

    Attributes:
        num_traj_samples: Number of future waypoint samples to provide.
        traj_sample_timestep: Time interval between waypoint samples.
        path_generator: Path generation parameters.
        enable_path_termination: Whether to terminate on path deviation.
        fail_dist: Maximum horizontal distance from path before termination.
        fail_height_dist: Maximum height deviation from path before termination.
    """

    _target_: str = (
        "protomotions.envs.control.path_follower_control.PathFollowerControl"
    )

    num_traj_samples: int = 10
    traj_sample_timestep: float = 0.3
    path_generator: PathGeneratorConfig = field(default_factory=PathGeneratorConfig)
    enable_path_termination: bool = True
    fail_dist: float = 1.0
    fail_height_dist: float = 0.5


# The path checks ignore the first steps of an episode.
MIN_PROGRESS = 10


class PathFollowerControl(ControlComponent):
    """The path and the head position of a path-following task."""

    config: PathFollowerControlConfig
    namespaces = ("path",)

    @classmethod
    def terminations(
        cls, config: PathFollowerControlConfig, robot_config
    ) -> Dict[str, TerminationRule]:
        """``path_distance`` (and ``path_height`` when height conditioned),
        cuts over the scalar path kernels, only with ``enable_path_termination``."""
        if not config.enable_path_termination:
            return {}
        entries = {
            "path_distance": TerminationRule(
                MdpComponent.from_kernel(path_distance, min_progress=MIN_PROGRESS),
                threshold=config.fail_dist,
            )
        }
        if config.path_generator.height_conditioned:
            entries["path_height"] = TerminationRule(
                MdpComponent.from_kernel(path_height_error, min_progress=MIN_PROGRESS),
                threshold=config.fail_height_dist,
            )
        return entries

    def __init__(self, config: PathFollowerControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        robot = inputs.robot
        # The head body: the path is followed by the head.
        self.head_body_id = build_body_ids_tensor(
            robot.kinematic_info.body_names,
            robot.common_naming_to_robot_body_names["head_body_name"],
            inputs.device,
        ).item()

        # The path, in a child registry (its vertices are per-env state).
        episode_dur = inputs.clock.max_episode_length * inputs.dt
        self.path_generator = PathGenerator(
            config.path_generator,
            self.state.child("path"),
            inputs.generator,
            inputs.num_envs,
            episode_dur,
            config.path_generator.height_conditioned,
        )
        # The head's ground-relative position, latched from the state the env
        # read (the lowest body is the ground estimate).
        self.head_pos = self.state.add("head_pos", (3,), torch.float, 0.0)

    @property
    def height_conditioned(self) -> bool:
        """Whether paths include height conditioning."""
        return self.config.path_generator.height_conditioned

    @staticmethod
    def _ground_relative_head(rigid_body_pos: Tensor, head_body_id: int) -> Tensor:
        ground_below = torch.min(rigid_body_pos, dim=1).values[..., 2]
        head_position = rigid_body_pos[:, head_body_id, :].clone()
        head_position[..., 2] -= ground_below
        return head_position

    def reset_rows(self, env_ids: Tensor, state: RobotState) -> None:
        """A new path from the anchor's ground-relative position; the head latched."""
        if len(env_ids) == 0:
            return
        super().reset_rows(env_ids, state)
        anchor_pos = state.rigid_body_pos[:, self.inputs.robot.anchor_body_index]

        # Build the path's start from the root x,y and an approximate head height
        head_position = anchor_pos.clone()
        head_position[..., 2] -= self.inputs.ground_height(head_position)
        self.path_generator.reset(env_ids, head_position)

        self.head_pos[env_ids] = self._ground_relative_head(
            state.rigid_body_pos, self.head_body_id
        )

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        """Latch the head position; the path itself is fixed for the episode."""
        self.head_pos[:] = self._ground_relative_head(
            state.rigid_body_pos, self.head_body_id
        )
        return {}

    def _all_env_ids(self) -> Tensor:
        return torch.arange(
            self.inputs.num_envs, device=self.inputs.device, dtype=torch.long
        )

    def calc_target_pos(self, env_ids: Optional[Tensor] = None) -> Tensor:
        """The target position on the path at the current time, [len(env_ids), 3]."""
        if env_ids is None:
            env_ids = self._all_env_ids()
        time = self.inputs.clock.progress_buf[env_ids] * self.inputs.dt
        return self.path_generator.calc_pos(env_ids, time)

    def fetch_path_samples(self, env_ids: Optional[Tensor] = None) -> Tensor:
        """Future waypoint samples along the path,
        [len(env_ids), num_traj_samples, 3]."""
        if env_ids is None:
            env_ids = self._all_env_ids()

        timestep_beg = self.inputs.clock.progress_buf[env_ids] * self.inputs.dt
        timesteps = torch.arange(
            self.config.num_traj_samples, device=self.inputs.device, dtype=torch.float
        )
        timesteps = timesteps * self.config.traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        return torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self.config.num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Publish the ``path.*`` names (see ``PathContext``)."""
        rows = slice(None) if env_ids is None else env_ids
        if env_ids is None:
            env_ids = self._all_env_ids()
        return {
            "path.tar_pos": self.calc_target_pos(env_ids),
            "path.head_pos": self.head_pos[rows],
            "path.traj_samples": self.fetch_path_samples(env_ids),
            "path.height_conditioned": self.height_conditioned,
            "path.head_body_id": self.head_body_id,
        }

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Red spheres on the future waypoint samples."""
        if headless:
            return {}
        path_markers = [
            MarkerConfig(size="regular") for _ in range(self.config.num_traj_samples)
        ]
        return {
            "path_markers": VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 0.0), markers=path_markers
            )
        }

    def get_markers_state(self, state: RobotState) -> Dict[str, MarkerState]:
        """The waypoint samples, lifted onto the terrain."""
        if self.inputs.headless:
            return {}
        num_envs, device = self.inputs.num_envs, self.inputs.device
        traj_samples = self.fetch_path_samples().clone()
        if not self.height_conditioned:
            traj_samples[..., 2] = 0.8  # 0.8 m when not height conditioned

        ground_below_marker = self.inputs.ground_height(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])
        traj_samples[..., 2] += ground_below_marker

        traj_samples = traj_samples.view(num_envs, -1, 3)
        return {
            "path_markers": MarkerState(
                translation=traj_samples,
                orientation=torch.zeros(
                    num_envs, traj_samples.shape[1], 4, device=device
                ),
            )
        }
