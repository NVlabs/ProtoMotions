# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-clip target source for GPC supervised fine-tuning.

This control does not expose full-body mimic tracking targets. It populates the
same ``TargetContext`` used by ``target_control.py`` with an XY steering target
whose offset comes from a future root position in the active mimic reference
clip. GPC SFT uses this to bootstrap a target-reaching policy from motion clips:
the policy sees a normal target-reaching task, while the target source is derived
from reference motion instead of random or keyboard commands. It owns no
episode-ending check and none of the failure-check names of ``TargetContext``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
from protomotions.utils.rotations import (
    calc_heading_quat,
    calc_heading_quat_inv,
    quat_rotate,
)


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
    namespaces = ("target",)

    def __init__(self, config: GPCSFTReferenceTargetControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        add = self.state.add
        self.tar_pos = add("tar_pos", (3,), torch.float, 0.0)
        self.target_motion_time = add("target_motion_time", (), torch.float, 0.0)
        self.sample_motion_time = add("sample_motion_time", (), torch.float, 0.0)
        self.xy_jitter = add("xy_jitter", (2,), torch.float, 0.0)
        # 0: the target follows the clip; 1: a random target for the episode.
        self.target_mode = add("target_mode", (), torch.long, 0)

    def reset_rows(self, env_ids: Tensor, state: RobotState) -> None:
        super().reset_rows(env_ids, state)
        if len(env_ids) == 0:
            return
        root_pos, root_rot = state.rigid_body_pos[:, 0], state.rigid_body_rot[:, 0]
        if self.config.random_target_fraction <= 0:
            self.target_mode[env_ids] = 0
            self._sample_mimic_target(env_ids, root_pos, root_rot)
            return

        is_random = (
            torch.rand(
                len(env_ids), device=self.inputs.device, generator=self.inputs.generator
            )
            < self.config.random_target_fraction
        )
        mimic_ids = env_ids[~is_random]
        random_ids = env_ids[is_random]
        self.target_mode[mimic_ids] = 0
        self.target_mode[random_ids] = 1
        self._sample_mimic_target(mimic_ids, root_pos[~is_random], root_rot[~is_random])
        self._sample_random_target(random_ids, root_pos[is_random])

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        """Resample the envs whose target time is reached (or whose clip time
        moved back); log the random-target fraction when it is in play."""
        motion_times = self.inputs.reference.times
        should_resample = (motion_times >= self.target_motion_time) | (
            motion_times < self.sample_motion_time
        )
        env_ids = should_resample.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._sample_mimic_target(
                env_ids,
                state.rigid_body_pos[env_ids, 0],
                state.rigid_body_rot[env_ids, 0],
            )

        if self.config.random_target_fraction > 0:
            return {
                "task/gpc_sft_reference_target_random_fraction": (self.target_mode == 1)
                .float()
                .mean()
                .detach()
            }
        return {}

    def _sample_mimic_target(
        self, env_ids: Tensor, root_pos: Tensor, root_rot: Tensor
    ) -> None:
        """``root_pos`` / ``root_rot`` are the rows of ``env_ids``."""
        if len(env_ids) == 0:
            return

        count = len(env_ids)
        device, generator = self.inputs.device, self.inputs.generator
        cfg = self.config
        reference = self.inputs.reference
        motion_ids = reference.ids[env_ids]
        motion_times = reference.times[env_ids]
        motion_lengths = reference.lengths(motion_ids)

        lookahead = (
            torch.rand(count, device=device, generator=generator)
            * (cfg.lookahead_seconds_max - cfg.lookahead_seconds_min)
            + cfg.lookahead_seconds_min
        )
        target_times = torch.minimum(motion_times + lookahead, motion_lengths)
        self.target_motion_time[env_ids] = target_times
        self.sample_motion_time[env_ids] = motion_times
        self._sample_xy_jitter(env_ids)

        # Raw clip poses: the target is the displacement between two poses of
        # one env, in which the env's placement cancels.
        state_now = reference.raw(motion_ids, motion_times)
        state_target = reference.raw(motion_ids, target_times)

        delta = torch.zeros(count, 3, device=device)
        delta[:, :2] = (
            state_target.rigid_body_pos[:, 0, :2] - state_now.rigid_body_pos[:, 0, :2]
        )
        clip_heading_inv = calc_heading_quat_inv(
            state_now.rigid_body_rot[:, 0, :],
            w_last=True,
        )
        char_heading = calc_heading_quat(root_rot, w_last=True)
        delta_local = quat_rotate(clip_heading_inv, delta, w_last=True)
        delta_world = quat_rotate(char_heading, delta_local, w_last=True)

        self.tar_pos[env_ids, :2] = (
            root_pos[:, :2] + delta_world[:, :2] + self.xy_jitter[env_ids]
        )
        self.tar_pos[env_ids, 2] = self._ground_heights(self.tar_pos[env_ids])

    def _sample_xy_jitter(self, env_ids: Tensor) -> None:
        radius = self.config.target_jitter_radius
        if len(env_ids) == 0 or radius <= 0:
            self.xy_jitter[env_ids] = 0.0
            return
        device, generator = self.inputs.device, self.inputs.generator
        r = radius * torch.sqrt(
            torch.rand(len(env_ids), device=device, generator=generator)
        )
        theta = torch.rand(len(env_ids), device=device, generator=generator) * (
            2.0 * torch.pi
        )
        self.xy_jitter[env_ids, 0] = r * torch.cos(theta)
        self.xy_jitter[env_ids, 1] = r * torch.sin(theta)

    def _sample_random_target(self, env_ids: Tensor, root_pos: Tensor) -> None:
        """``root_pos`` is the rows of ``env_ids``."""
        if len(env_ids) == 0:
            return
        device, generator = self.inputs.device, self.inputs.generator
        radius = self.config.random_target_xy_radius
        r = radius * torch.sqrt(
            torch.rand(len(env_ids), device=device, generator=generator)
        )
        theta = torch.rand(len(env_ids), device=device, generator=generator) * (
            2.0 * torch.pi
        )
        self.tar_pos[env_ids, 0] = root_pos[:, 0] + r * torch.cos(theta)
        self.tar_pos[env_ids, 1] = root_pos[:, 1] + r * torch.sin(theta)
        self.tar_pos[env_ids, 2] = self._ground_heights(self.tar_pos[env_ids])
        self.target_motion_time[env_ids] = float("inf")
        self.sample_motion_time[env_ids] = float("-inf")
        self.xy_jitter[env_ids] = 0.0

    def _ground_heights(self, positions: Tensor) -> Tensor:
        return self.inputs.ground_height(positions)

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Publish the same ``target.*`` goal names as ``TargetControl``."""
        tar_pos = self.tar_pos if env_ids is None else self.tar_pos[env_ids]
        return {
            "target.tar_pos": tar_pos,
            "target.tar_proximity_threshold": self.config.tar_proximity_threshold,
        }

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

    def get_markers_state(self, state: RobotState) -> Dict[str, MarkerState]:
        if self.inputs.headless:
            return {}
        num_envs, device = self.inputs.num_envs, self.inputs.device
        tar_pos = self.tar_pos.view(num_envs, 1, 3).clone()
        tar_pos[..., 2] += 0.1
        tar_rot = torch.zeros(num_envs, 1, 4, device=device)
        tar_rot[..., -1] = 1.0
        return {"target_markers": MarkerState(translation=tar_pos, orientation=tar_rot)}
