# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The target control's command sources: what moves the target and when.

A command source owns the policy for where the next target goes and when it
changes; the control (``target_control.py``) owns the target's per-env state
and publishes it. ``RandomTargetCommandSource`` samples heading-relative
targets on a random schedule for training; ``KeyboardTargetCommandSource``
moves one env's target from registered viewer keys. A source writes the
control's registered tensors through the control it is handed and keeps no
per-env tensor of its own; every random draw passes
``generator=control.inputs.generator``. The configs carry each source's
dotted path in ``_target_`` and are what a recipe puts in
``TargetControlConfig.command_source``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils.rotations import calc_heading_quat, quat_rotate

if TYPE_CHECKING:
    from protomotions.envs.control.target_control import TargetControl


@dataclass
class TargetKeyBindingConfig:
    """Keyboard binding for interactive target updates.

    ``action`` becomes the attribute name on the scope returned by
    :meth:`UserInterface.scope`. Must be a valid non-private Python
    identifier (e.g. ``move_forward``); validation happens at registration.
    """

    key: str
    action: str
    delta_xy: Tuple[float, float]
    description: str


@dataclass
class TargetCommandSourceConfig:
    """Base config for target command sources."""

    _target_: str = "protomotions.envs.control.command_source.TargetCommandSource"


@dataclass
class RandomTargetCommandSourceConfig(TargetCommandSourceConfig):
    """Training-time target sampler."""

    _target_: str = "protomotions.envs.control.command_source.RandomTargetCommandSource"

    tar_change_time_min: float = 6.0
    tar_change_time_max: float = 8.0
    tar_dist_max: float = 6.0

    target_bounds: Optional[Tuple[float, float, float, float]] = None
    fixed_target_position: Optional[Tuple[float, float]] = None
    target_positions: Optional[Tuple[Tuple[float, float], ...]] = None


@dataclass
class KeyboardTargetCommandSourceConfig(TargetCommandSourceConfig):
    """Interactive target source driven by registered simulator UI keys."""

    _target_: str = (
        "protomotions.envs.control.command_source.KeyboardTargetCommandSource"
    )

    key_bindings: Tuple[TargetKeyBindingConfig, ...] = (
        TargetKeyBindingConfig(
            key="W",
            action="move_forward",
            delta_xy=(0.0, 1.0),
            description="Move target forward",
        ),
        TargetKeyBindingConfig(
            key="S",
            action="move_backward",
            delta_xy=(0.0, -1.0),
            description="Move target backward",
        ),
        TargetKeyBindingConfig(
            key="A",
            action="move_left",
            delta_xy=(-1.0, 0.0),
            description="Move target left",
        ),
        TargetKeyBindingConfig(
            key="D",
            action="move_right",
            delta_xy=(1.0, 0.0),
            description="Move target right",
        ),
    )
    fail_if_headless: bool = True


class TargetCommandSource:
    """Source of target updates for :class:`TargetControl`."""

    def __init__(self, config: TargetCommandSourceConfig, control: "TargetControl"):
        self.config = config
        self.control = control

    def reset(self, env_ids: Tensor, anchor_pos: Tensor, anchor_rot: Tensor) -> None:
        """A new episode's target for ``env_ids``, from their anchor pose."""
        raise NotImplementedError

    def advance(self, state: RobotState) -> None:
        """Once per step, on the robot state the env read."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class RandomTargetCommandSource(TargetCommandSource):
    """Sample target positions from the training configuration."""

    config: RandomTargetCommandSourceConfig

    def __init__(
        self, config: RandomTargetCommandSourceConfig, control: "TargetControl"
    ):
        super().__init__(config, control)
        self.config = config
        self._target_bounds = config.target_bounds

    def reset(self, env_ids: Tensor, anchor_pos: Tensor, anchor_rot: Tensor) -> None:
        self._set_random_target(env_ids, anchor_pos, anchor_rot)

    def advance(self, state: RobotState) -> None:
        control = self.control
        progress_time = control.inputs.clock.elapsed_seconds
        reset_ids = (
            (progress_time >= control.tar_change_time).nonzero(as_tuple=False).flatten()
        )
        if len(reset_ids) > 0:
            anchor = control.inputs.robot.anchor_body_index
            self._set_random_target(
                reset_ids,
                state.rigid_body_pos[reset_ids, anchor],
                state.rigid_body_rot[reset_ids, anchor],
            )

    def _set_random_target(
        self, env_ids: Tensor, root_pos: Tensor, root_rot: Tensor
    ) -> None:
        control = self.control
        inputs = control.inputs

        if self.config.target_positions is not None:
            self._sample_from_target_positions(env_ids)
            return

        if self.config.fixed_target_position is not None:
            fixed_x, fixed_y = self.config.fixed_target_position
            control.tar_pos[env_ids, 0] = fixed_x
            control.tar_pos[env_ids, 1] = fixed_y
            control._update_target_heights(env_ids)
            control.tar_change_time[env_ids] = float("inf")
            return

        self._sample_heading_relative_target(env_ids, root_pos, root_rot)

        tar_change = (
            torch.rand(len(env_ids), device=inputs.device, generator=inputs.generator)
            * (self.config.tar_change_time_max - self.config.tar_change_time_min)
            + self.config.tar_change_time_min
        )
        current_time = inputs.clock.progress_buf[env_ids] * inputs.dt
        control.tar_change_time[env_ids] = current_time + tar_change

    def _sample_from_target_positions(self, env_ids: Tensor) -> None:
        control = self.control
        positions = self.config.target_positions
        indices = torch.randint(
            0,
            len(positions),
            (len(env_ids),),
            device=control.inputs.device,
            generator=control.inputs.generator,
        )
        for row, env_id in enumerate(env_ids):
            control.tar_pos[env_id, 0] = positions[indices[row].item()][0]
            control.tar_pos[env_id, 1] = positions[indices[row].item()][1]
        control._update_target_heights(env_ids)
        control.tar_change_time[env_ids] = float("inf")

    def _sample_heading_relative_target(
        self, env_ids: Tensor, root_pos: Tensor, root_rot: Tensor
    ) -> None:
        control = self.control
        device, generator = control.inputs.device, control.inputs.generator
        num_envs = len(env_ids)
        rand_dist = (
            torch.rand(num_envs, device=device, generator=generator)
            * self.config.tar_dist_max
        )
        rand_angle = (
            torch.rand(num_envs, device=device, generator=generator) * 2 * torch.pi
        )

        local_offset = torch.zeros(num_envs, 3, device=device)
        local_offset[:, 0] = rand_dist * torch.cos(rand_angle)
        local_offset[:, 1] = rand_dist * torch.sin(rand_angle)
        heading_rot = calc_heading_quat(root_rot, w_last=True)
        world_offset = quat_rotate(heading_rot, local_offset, w_last=True)
        control.tar_pos[env_ids, :2] = root_pos[:, :2] + world_offset[:, :2]

        if self._target_bounds is not None:
            x_min, x_max, y_min, y_max = self._target_bounds
            control.tar_pos[env_ids, 0] = control.tar_pos[env_ids, 0].clamp(
                x_min, x_max
            )
            control.tar_pos[env_ids, 1] = control.tar_pos[env_ids, 1].clamp(
                y_min, y_max
            )
        control._update_target_heights(env_ids)


class KeyboardTargetCommandSource(TargetCommandSource):
    """Move one active environment target from registered keyboard bindings."""

    config: KeyboardTargetCommandSourceConfig

    def __init__(
        self, config: KeyboardTargetCommandSourceConfig, control: "TargetControl"
    ):
        super().__init__(config, control)
        self.config = config
        if config.fail_if_headless and control.inputs.headless:
            raise RuntimeError(
                "Keyboard target command source requires a non-headless simulator"
            )
        self._key_bindings = control.inputs.ui.scope("target_control")
        self._action_bindings = []
        for binding in config.key_bindings:
            handle = self._key_bindings.register(
                binding.key,
                binding.action,
                binding.description,
            )
            self._action_bindings.append((handle, binding.delta_xy))

    def reset(self, env_ids: Tensor, anchor_pos: Tensor, anchor_rot: Tensor) -> None:
        control = self.control
        control.tar_pos[env_ids] = anchor_pos
        control._update_target_heights(env_ids)
        control.tar_change_time[env_ids] = float("inf")

    def advance(self, state: RobotState) -> None:
        control = self.control
        ui = control.inputs.ui
        for handle, (dx, dy) in self._action_bindings:
            if handle.consume():
                env_id = int(ui.active_env_id)
                if env_id < 0 or env_id >= control.inputs.num_envs:
                    raise IndexError(
                        f"Active env id {env_id} is outside "
                        f"[0, {control.inputs.num_envs})"
                    )
                control._move_targets(
                    torch.tensor([env_id], device=control.inputs.device), dx=dx, dy=dy
                )

    def close(self) -> None:
        self._key_bindings.unregister_all()


__all__ = [
    "KeyboardTargetCommandSource",
    "KeyboardTargetCommandSourceConfig",
    "RandomTargetCommandSource",
    "RandomTargetCommandSourceConfig",
    "TargetCommandSource",
    "TargetCommandSourceConfig",
    "TargetKeyBindingConfig",
]
