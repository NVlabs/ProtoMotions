# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External kinematic control component.

An animation player fed from outside the env: pose computation happens
elsewhere (a prior model's inference, for example), the caller hands the
next pose to ``set_next_pose`` between steps, and the control asks the env to
write it (``TaskInputs.teleport``) and to start the episodes over
(``TaskInputs.restart_episode``) during the next ``advance``. Publishes
nothing.
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
class ExternalKinematicControlConfig(ControlComponentConfig):
    """Configuration for external kinematic control component."""

    _target_: str = (
        "protomotions.envs.control.external_kinematic_control.ExternalKinematicControl"
    )


class ExternalKinematicControl(ControlComponent):
    """Control component that applies externally-provided poses.

    Usage:
        hook = env.control_manager.components["external_kinematic"]
        hook.set_next_pose(reset_state)
        env.step(actions)  # the pose is written during the step's advance
    """

    config: ExternalKinematicControlConfig

    def __init__(self, config: ExternalKinematicControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        # False until a row's first reset; the registry's row reset writes True.
        self.initialized = self.state.add("initialized", (), torch.bool, True)
        self.initialized.zero_()
        # The pending pose: an input queue of one, not per-env state (the
        # caller hands in a whole batch), so the snapshot carries it beside
        # the registry; see save_state.
        self._next_pose: Optional[ResetState] = None

    def reset_rows(self, env_ids: Tensor, state: RobotState) -> None:
        super().reset_rows(env_ids, state)
        self._next_pose = None

    def save_state(self) -> Dict[str, Any]:
        """The registry's snapshot plus the pending pose, the one thing this
        control holds that is not a per-env slot (an input queued from
        outside, consumed by the next advance)."""
        snapshot = super().save_state()
        snapshot["next_pose"] = (
            None if self._next_pose is None else self._next_pose.clone()
        )
        return snapshot

    def load_state(self, snapshot: Dict[str, Any]) -> None:
        snapshot = dict(snapshot)
        pose = snapshot.pop("next_pose")
        super().load_state(snapshot)
        self._next_pose = None if pose is None else pose.clone()

    def set_next_pose(self, reset_state: ResetState) -> None:
        """Set the pose to apply on the next step."""
        self._next_pose = reset_state

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        """Queue the stored pose for every env (no object support) and restart
        the episodes; nothing before the first reset or without a pose."""
        if self._next_pose is None or not bool(self.initialized.any()):
            return {}
        env_ids = torch.arange(
            self.inputs.num_envs, dtype=torch.long, device=self.inputs.device
        )
        self.inputs.teleport(env_ids, self._next_pose, None)
        self.inputs.restart_episode(env_ids)
        self._next_pose = None
        return {}

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """External kinematic control publishes no context names."""
        return {}
