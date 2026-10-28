# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor lifecycle: one measurement, one noise draw, one reconstruction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Callable, Optional

import torch
from torch import Tensor

from protomotions.envs.sensors.config import NoNoise, SensorConfig
from protomotions.envs.sensors.estimation import (
    estimate_body_state,
    with_episode_odometry,
)
from protomotions.envs.sensors.measurement import SensorReadings, measure
from protomotions.envs.sensors.noise import DriftState, NoiseDraws, add_sensor_noise
from protomotions.envs.state import BodyState
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils.baked_fk import BakedAnchorFK
from protomotions.utils.rotations import calc_heading_quat_inv


@dataclass
class SensorEstimates:
    """Two estimates from the same physical sample, in world SI units."""

    clean: BodyState
    noisy: BodyState


@dataclass
class EpisodeStart:
    """Localization drift origin and each estimate's own odometry reference."""

    localization_origin: Tensor
    drift_scale: Tensor
    drift_yaw: Tensor
    clean_anchor_xy: Tensor
    clean_heading_inv: Tensor
    noisy_anchor_xy: Tensor
    noisy_heading_inv: Tensor

    def map(self, transform: Callable[[Tensor], Tensor]) -> EpisodeStart:
        """Select or clone all episode tensors together."""
        return EpisodeStart(
            **{
                field.name: transform(getattr(self, field.name))
                for field in fields(self)
            }
        )


@dataclass
class SensorSnapshot:
    """Runtime state only; reconstructed estimates live in state history."""

    rng: Tensor
    episode: Optional[EpisodeStart]


class SensorRuntime:
    """Own sensor randomness and episode references, never state history.

    ``reset`` receives the post-reset simulator state for the supplied rows.
    ``step`` receives all environments after physics. Both return estimates;
    callers store those estimates alongside privileged state exactly once.
    """

    def __init__(
        self,
        config: SensorConfig,
        robot: RobotConfig,
        num_envs: int,
        device: torch.device,
        *,
        ground_query: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self.mounts = config.mounts(robot)
        if config.height is not None and ground_query is None:
            raise ValueError("The height sensor requires a ground-height query")
        self.config = config
        self.anchor_idx = robot.anchor_body_index
        self.ground_query = ground_query
        self.generator = torch.Generator(device=device)
        seed = config.seed if config.seed is not None else torch.initial_seed() + 17389
        self.generator.manual_seed(seed % (2**63))
        # FK constants belong on the runtime device from construction. Keep the
        # saved robot config unchanged and avoid host-to-device copies per step.
        self.fk = BakedAnchorFK(
            deepcopy(robot.kinematic_info).to(device),
            self.anchor_idx,
            torch.arange(robot.kinematic_info.num_bodies, device=device),
        )
        self.env_ids = torch.arange(num_envs, device=device)
        self.episode = None
        if config.localization is not None:
            identity = torch.zeros(num_envs, 4, device=device)
            identity[:, 3] = 1
            self.episode = EpisodeStart(
                localization_origin=torch.zeros(num_envs, 3, device=device),
                drift_scale=torch.ones(num_envs, device=device),
                drift_yaw=torch.zeros(num_envs, device=device),
                clean_anchor_xy=torch.zeros(num_envs, 2, device=device),
                clean_heading_inv=identity.clone(),
                noisy_anchor_xy=torch.zeros(num_envs, 2, device=device),
                noisy_heading_inv=identity.clone(),
            )

    @torch.no_grad()
    def step(self, state: RobotState) -> SensorEstimates:
        """Measure all environments and return clean/noisy world estimates."""
        estimates = self._estimate(self._measure(state), self.env_ids)
        return self._add_odometry(estimates, self.env_ids)

    @torch.no_grad()
    def reset(self, env_ids: Tensor, state: RobotState) -> SensorEstimates:
        """Measure reset rows; start drift before noise and odometry after FK."""
        readings = self._measure(state)
        if self.episode is not None:
            self.episode.localization_origin[env_ids] = readings.localization
            self._sample_drift(env_ids)
        estimates = self._estimate(readings, env_ids)
        if self.episode is not None:
            self._record_start(
                env_ids,
                estimates.clean,
                self.episode.clean_anchor_xy,
                self.episode.clean_heading_inv,
            )
            self._record_start(
                env_ids,
                estimates.noisy,
                self.episode.noisy_anchor_xy,
                self.episode.noisy_heading_inv,
            )
        return self._add_odometry(estimates, env_ids)

    def save_state(self) -> SensorSnapshot:
        """Save RNG and episode values without sampling or running FK."""
        episode = None if self.episode is None else self.episode.map(torch.clone)
        return SensorSnapshot(self.generator.get_state().clone(), episode)

    def load_state(self, snapshot: SensorSnapshot) -> None:
        """Restore exactly; the environment restores history separately."""
        self.generator.set_state(snapshot.rng)
        self.episode = (
            None if snapshot.episode is None else snapshot.episode.map(torch.clone)
        )

    def _measure(self, state: RobotState) -> SensorReadings:
        ground = None
        if self.config.height is not None:
            ground = self.ground_query(state.rigid_body_pos[:, 0]).squeeze(-1)
        return measure(state, self.config, self.mounts, ground_heights=ground)

    def _estimate(self, readings: SensorReadings, env_ids: Tensor) -> SensorEstimates:
        draws = NoiseDraws(
            measurement=readings.map(
                lambda value: torch.rand(
                    value.shape,
                    device=value.device,
                    dtype=value.dtype,
                    generator=self.generator,
                )
                * 2
                - 1
            ),
            localization_log=torch.randn(
                len(env_ids), 1, device=env_ids.device, generator=self.generator
            ),
        )
        drift = None
        if self._drift_config() is not None:
            drift = DriftState(
                self.episode.localization_origin[env_ids],
                self.episode.drift_scale[env_ids],
                self.episode.drift_yaw[env_ids],
            )
        noisy = add_sensor_noise(readings, self.config, draws, drift)
        # Both sources follow identical reconstruction math in one 2*N FK call.
        estimate = estimate_body_state(
            SensorReadings.cat([readings, noisy]), self.fk, self.mounts
        )
        count = len(env_ids)
        return SensorEstimates(
            estimate.map(lambda value: value[:count]),
            estimate.map(lambda value: value[count:]),
        )

    def _drift_config(self):
        """The localization drift law, or None when there is no localization channel."""
        localization = self.config.localization
        return None if localization is None else localization.drift

    def _sample_drift(self, env_ids: Tensor) -> None:
        config = self._drift_config()
        if config is None:
            return
        draws = (
            torch.rand(len(env_ids), 2, device=env_ids.device, generator=self.generator)
            * 2
            - 1
        )
        scale = 0.0 if isinstance(config.scale, NoNoise) else config.scale
        yaw = 0.0 if isinstance(config.yaw, NoNoise) else config.yaw
        self.episode.drift_scale[env_ids] = 1 + draws[:, 0] * scale
        self.episode.drift_yaw[env_ids] = draws[:, 1] * yaw

    def _record_start(
        self, env_ids: Tensor, state: BodyState, anchor_xy: Tensor, heading_inv: Tensor
    ) -> None:
        anchor_xy[env_ids] = state.rigid_body_pos[:, self.anchor_idx, :2]
        heading_inv[env_ids] = calc_heading_quat_inv(
            state.rigid_body_rot[:, self.anchor_idx], w_last=True
        )

    def _add_odometry(
        self, estimates: SensorEstimates, env_ids: Tensor
    ) -> SensorEstimates:
        if self.episode is None:
            return estimates
        episode = self.episode
        return SensorEstimates(
            with_episode_odometry(
                estimates.clean,
                self.anchor_idx,
                episode.clean_anchor_xy[env_ids],
                episode.clean_heading_inv[env_ids],
            ),
            with_episode_odometry(
                estimates.noisy,
                self.anchor_idx,
                episode.noisy_anchor_xy[env_ids],
                episode.noisy_heading_inv[env_ids],
            ),
        )
