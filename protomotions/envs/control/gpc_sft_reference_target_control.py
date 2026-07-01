# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-clip target source for GPC supervised fine-tuning.

This control does not expose full-body mimic tracking targets. It populates the
same ``TargetContext`` used by ``target_control.py`` with an XY steering target
whose offset comes from a future root position in the active mimic reference
clip. GPC SFT uses this to bootstrap a target-reaching policy from motion clips:
the policy sees a normal target-reaching task, while the target source is derived
from reference motion instead of random or keyboard commands.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext, TargetContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.utils.rotations import (
    calc_heading_quat,
    calc_heading_quat_inv,
    quat_rotate,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class GPCSFTReferenceTargetControlConfig(ControlComponentConfig):
    """GPC SFT target source derived from future root XY in a mimic clip."""

    _target_: str = (
        "protomotions.envs.control.gpc_sft_reference_target_control.GPCSFTReferenceTargetControl"
    )
    lookahead_seconds_min: float = 0.5
    lookahead_seconds_max: float = 2.0
    target_jitter_radius: float = 0.1
    tar_proximity_threshold: float = 0.3
    random_target_fraction: float = 0.0
    random_target_xy_radius: float = 6.0


class GPCSFTReferenceTargetControl(ControlComponent):
    """Populate ``TargetContext`` from future root XY in the current reference clip."""

    config: GPCSFTReferenceTargetControlConfig

    def __init__(self, config: GPCSFTReferenceTargetControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._tar_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._target_motion_time = torch.zeros(env.num_envs, device=env.device)
        self._sample_motion_time = torch.zeros(env.num_envs, device=env.device)
        self._xy_jitter = torch.zeros(env.num_envs, 2, device=env.device)
        self._target_mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Tensor):
        if len(env_ids) == 0:
            return
        if self.config.random_target_fraction <= 0:
            self._target_mode[env_ids] = 0
            self._sample_mimic_target(env_ids)
            return

        is_random = (
            torch.rand(len(env_ids), device=self.env.device)
            < self.config.random_target_fraction
        )
        mimic_ids = env_ids[~is_random]
        random_ids = env_ids[is_random]
        self._target_mode[mimic_ids] = 0
        self._target_mode[random_ids] = 1
        self._sample_mimic_target(mimic_ids)
        self._sample_random_target(random_ids)

    def step(self):
        motion_times = self.env.motion_manager.motion_times
        should_resample = (
            motion_times >= self._target_motion_time
        ) | (motion_times < self._sample_motion_time)
        env_ids = should_resample.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._sample_mimic_target(env_ids)

        if self.config.random_target_fraction > 0:
            self.env.extras["task/gpc_sft_reference_target_random_fraction"] = (
                self._target_mode == 1
            ).float().mean().detach()

    def _sample_mimic_target(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return

        count = len(env_ids)
        device = self.env.device
        cfg = self.config
        motion_lib = self.env.motion_lib
        motion_manager = self.env.motion_manager
        motion_ids = motion_manager.motion_ids[env_ids]
        motion_times = motion_manager.motion_times[env_ids]
        motion_lengths = motion_lib.get_motion_length(motion_ids)

        lookahead = (
            torch.rand(count, device=device)
            * (cfg.lookahead_seconds_max - cfg.lookahead_seconds_min)
            + cfg.lookahead_seconds_min
        )
        target_times = torch.minimum(motion_times + lookahead, motion_lengths)
        self._target_motion_time[env_ids] = target_times
        self._sample_motion_time[env_ids] = motion_times
        self._sample_xy_jitter(env_ids)

        state_now = motion_lib.get_motion_state(motion_ids, motion_times)
        state_target = motion_lib.get_motion_state(motion_ids, target_times)
        char_state = self.env.simulator.get_root_state(env_ids)

        delta = torch.zeros(count, 3, device=device)
        delta[:, :2] = (
            state_target.rigid_body_pos[:, 0, :2]
            - state_now.rigid_body_pos[:, 0, :2]
        )
        clip_heading_inv = calc_heading_quat_inv(
            state_now.rigid_body_rot[:, 0, :],
            w_last=True,
        )
        char_heading = calc_heading_quat(char_state.root_rot, w_last=True)
        delta_local = quat_rotate(clip_heading_inv, delta, w_last=True)
        delta_world = quat_rotate(char_heading, delta_local, w_last=True)

        self._tar_pos[env_ids, :2] = (
            char_state.root_pos[:, :2]
            + delta_world[:, :2]
            + self._xy_jitter[env_ids]
        )
        self._tar_pos[env_ids, 2] = self._ground_heights(self._tar_pos[env_ids])

    def _sample_xy_jitter(self, env_ids: Tensor) -> None:
        radius = self.config.target_jitter_radius
        if len(env_ids) == 0 or radius <= 0:
            self._xy_jitter[env_ids] = 0.0
            return
        device = self.env.device
        r = radius * torch.sqrt(torch.rand(len(env_ids), device=device))
        theta = torch.rand(len(env_ids), device=device) * (2.0 * torch.pi)
        self._xy_jitter[env_ids, 0] = r * torch.cos(theta)
        self._xy_jitter[env_ids, 1] = r * torch.sin(theta)

    def _sample_random_target(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return
        device = self.env.device
        radius = self.config.random_target_xy_radius
        r = radius * torch.sqrt(torch.rand(len(env_ids), device=device))
        theta = torch.rand(len(env_ids), device=device) * (2.0 * torch.pi)
        root_pos = self.env.simulator.get_root_state(env_ids).root_pos
        self._tar_pos[env_ids, 0] = root_pos[:, 0] + r * torch.cos(theta)
        self._tar_pos[env_ids, 1] = root_pos[:, 1] + r * torch.sin(theta)
        self._tar_pos[env_ids, 2] = self._ground_heights(self._tar_pos[env_ids])
        self._target_motion_time[env_ids] = float("inf")
        self._sample_motion_time[env_ids] = float("-inf")
        self._xy_jitter[env_ids] = 0.0

    def _ground_heights(self, positions: Tensor) -> Tensor:
        if self.env.terrain is None:
            return torch.zeros(positions.shape[0], device=self.env.device)
        return self.env.terrain.get_ground_heights(positions).squeeze(-1)

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        zeros = torch.zeros(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.env.device,
        )
        return zeros, zeros

    def populate_context(self, ctx: EnvContext) -> None:
        ctx.target = TargetContext(
            tar_pos=self._tar_pos,
            tar_proximity_threshold=self.config.tar_proximity_threshold,
        )

    def create_visualization_markers(
        self,
        headless: bool,
    ) -> Dict[str, VisualizationMarkerConfig]:
        if headless:
            return {}
        return {
            "target_markers": VisualizationMarkerConfig(
                type="sphere",
                color=(0.0, 0.0, 1.0),
                markers=[MarkerConfig(size="huge")],
            )
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        if self.env.simulator.headless:
            return {}
        tar_pos = self._tar_pos.view(self.env.num_envs, 1, 3).clone()
        tar_pos[..., 2] += 0.1
        tar_rot = torch.zeros(self.env.num_envs, 1, 4, device=self.env.device)
        tar_rot[..., -1] = 1.0
        return {"target_markers": MarkerState(translation=tar_pos, orientation=tar_rot)}
