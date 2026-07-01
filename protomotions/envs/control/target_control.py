# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-reaching control component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext, TargetContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.utils.hydra_replacement import get_class
from protomotions.utils.rotations import calc_heading_quat, quat_rotate

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


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

    _target_: str = "protomotions.envs.control.target_control.TargetCommandSource"


@dataclass
class RandomTargetCommandSourceConfig(TargetCommandSourceConfig):
    """Training-time target sampler."""

    _target_: str = (
        "protomotions.envs.control.target_control.RandomTargetCommandSource"
    )

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
        "protomotions.envs.control.target_control.KeyboardTargetCommandSource"
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


@dataclass
class TargetControlConfig(ControlComponentConfig):
    """Configuration for target-reaching tasks."""

    _target_: str = "protomotions.envs.control.target_control.TargetControl"

    tar_proximity_threshold: float = 0.25
    command_source: TargetCommandSourceConfig = field(
        default_factory=RandomTargetCommandSourceConfig
    )

    enable_fall_termination: bool = True
    fall_termination_height: float = 0.2
    enable_gap_termination: bool = False
    gap_termination_threshold: float = 0.5
    enable_stuck_termination: bool = False
    stuck_window_frames: int = 15
    stuck_movement_threshold: float = 0.1
    stuck_height_threshold: float = 0.4
    reset_grace_period: int = 5


class TargetCommandSource:
    """Source of target updates for :class:`TargetControl`."""

    def __init__(self, config: TargetCommandSourceConfig, control: "TargetControl"):
        self.config = config
        self.control = control

    def reset(self, env_ids: Tensor) -> None:
        raise NotImplementedError

    def step(self) -> None:
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

    def reset(self, env_ids: Tensor) -> None:
        self._set_random_target(env_ids)

    def step(self) -> None:
        control = self.control
        progress_time = control.env.progress_buf * control.env.dt
        reset_ids = (progress_time >= control._tar_change_time).nonzero(
            as_tuple=False
        ).flatten()
        if len(reset_ids) > 0:
            self._set_random_target(reset_ids)

    def _set_random_target(self, env_ids: Tensor) -> None:
        control = self.control
        root_state = control.env.simulator.get_root_state(env_ids)
        root_pos = root_state.root_pos
        root_rot = root_state.root_rot

        if self.config.target_positions is not None:
            self._sample_from_target_positions(env_ids)
            return

        if self.config.fixed_target_position is not None:
            fixed_x, fixed_y = self.config.fixed_target_position
            control._tar_pos[env_ids, 0] = fixed_x
            control._tar_pos[env_ids, 1] = fixed_y
            control._update_target_heights(env_ids)
            control._tar_change_time[env_ids] = float("inf")
            return

        self._sample_heading_relative_target(env_ids, root_pos, root_rot)

        tar_change = torch.rand(len(env_ids), device=control.env.device) * (
            self.config.tar_change_time_max - self.config.tar_change_time_min
        ) + self.config.tar_change_time_min
        current_time = control.env.progress_buf[env_ids] * control.env.dt
        is_env_reset = control.env.reset_buf[env_ids] | control.env.terminate_buf[
            env_ids
        ]
        current_time = torch.where(
            is_env_reset, torch.zeros_like(current_time), current_time
        )
        control._tar_change_time[env_ids] = current_time + tar_change

    def _sample_from_target_positions(self, env_ids: Tensor) -> None:
        control = self.control
        positions = self.config.target_positions
        indices = torch.randint(
            0, len(positions), (len(env_ids),), device=control.env.device
        )
        for row, env_id in enumerate(env_ids):
            control._tar_pos[env_id, 0] = positions[indices[row].item()][0]
            control._tar_pos[env_id, 1] = positions[indices[row].item()][1]
        control._update_target_heights(env_ids)
        control._tar_change_time[env_ids] = float("inf")

    def _sample_heading_relative_target(
        self, env_ids: Tensor, root_pos: Tensor, root_rot: Tensor
    ) -> None:
        control = self.control
        num_envs = len(env_ids)
        rand_dist = (
            torch.rand(num_envs, device=control.env.device) * self.config.tar_dist_max
        )
        rand_angle = torch.rand(num_envs, device=control.env.device) * 2 * torch.pi

        local_offset = torch.zeros(num_envs, 3, device=control.env.device)
        local_offset[:, 0] = rand_dist * torch.cos(rand_angle)
        local_offset[:, 1] = rand_dist * torch.sin(rand_angle)
        heading_rot = calc_heading_quat(root_rot, w_last=True)
        world_offset = quat_rotate(heading_rot, local_offset, w_last=True)
        control._tar_pos[env_ids, :2] = root_pos[:, :2] + world_offset[:, :2]

        if self._target_bounds is not None:
            x_min, x_max, y_min, y_max = self._target_bounds
            control._tar_pos[env_ids, 0] = control._tar_pos[env_ids, 0].clamp(
                x_min, x_max
            )
            control._tar_pos[env_ids, 1] = control._tar_pos[env_ids, 1].clamp(
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
        if config.fail_if_headless and control.env.simulator.headless:
            raise RuntimeError(
                "Keyboard target command source requires a non-headless simulator"
            )
        ui = control.env.simulator.user_interface
        self._key_bindings = ui.scope("target_control")
        self._action_bindings = []
        for binding in config.key_bindings:
            handle = self._key_bindings.register(
                binding.key,
                binding.action,
                binding.description,
            )
            self._action_bindings.append((handle, binding.delta_xy))

    def reset(self, env_ids: Tensor) -> None:
        self.control._tar_pos[env_ids] = self.control.env.simulator.get_root_state(
            env_ids
        ).root_pos
        self.control._update_target_heights(env_ids)
        self.control._tar_change_time[env_ids] = float("inf")

    def step(self) -> None:
        ui = self.control.env.simulator.user_interface
        for handle, (dx, dy) in self._action_bindings:
            if handle.consume():
                env_id = int(ui.active_env_id)
                if env_id < 0 or env_id >= self.control.env.num_envs:
                    raise IndexError(
                        f"Active env id {env_id} is outside [0, {self.control.env.num_envs})"
                    )
                self.control._move_targets(
                    torch.tensor([env_id], device=self.control.env.device),
                    dx=dx,
                    dy=dy,
                )

    def close(self) -> None:
        self._key_bindings.unregister_all()


class TargetControl(ControlComponent):
    """Stateful target sampler for target-reaching tasks."""

    config: TargetControlConfig

    def __init__(self, config: TargetControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config = config
        self._tar_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._tar_change_time = torch.zeros(env.num_envs, device=env.device)
        self._last_support_root_height = torch.zeros(env.num_envs, device=env.device)
        self._last_support_ground_height = torch.zeros(env.num_envs, device=env.device)
        self._root_pos_history = torch.zeros(
            env.num_envs, config.stuck_window_frames, 3, device=env.device
        )
        self._root_pos_history_idx = 0
        command_source_cls = get_class(config.command_source._target_)
        self.command_source = command_source_cls(config.command_source, self)

    def reset(self, env_ids: Tensor):
        if len(env_ids) == 0:
            return
        self.command_source.reset(env_ids)

        root_pos = self.env.simulator.get_root_state(env_ids).root_pos
        self._last_support_root_height[env_ids] = root_pos[:, 2]
        self._last_support_ground_height[env_ids] = self._ground_heights(root_pos)
        self._root_pos_history[env_ids] = root_pos.unsqueeze(1).expand(
            -1, self.config.stuck_window_frames, -1
        )

    def step(self):
        self.command_source.step()

        if self.config.enable_stuck_termination:
            root_pos = self.env.simulator.get_root_state().root_pos
            self._root_pos_history[:, self._root_pos_history_idx] = root_pos
            self._root_pos_history_idx = (
                self._root_pos_history_idx + 1
            ) % self.config.stuck_window_frames

    def close(self) -> None:
        """Release UI handles held by the command source."""
        self.command_source.close()

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        terminated = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        if not (
            self.config.enable_fall_termination
            or self.config.enable_gap_termination
            or self.config.enable_stuck_termination
        ):
            return terminated.clone(), terminated

        robot_state = self.env.simulator.get_robot_state()
        root_pos = robot_state.rigid_body_pos[:, 0]
        root_z = root_pos[:, 2]
        ground_z = self._ground_heights(root_pos)
        past_grace = self.env.progress_buf > self.config.reset_grace_period

        on_elevated_ground = ground_z > 0.0
        self._last_support_root_height[on_elevated_ground] = root_z[on_elevated_ground]
        self._last_support_ground_height[on_elevated_ground] = ground_z[
            on_elevated_ground
        ]

        if self.config.enable_fall_termination:
            from protomotions.envs.terminations import fall_termination

            fall_ref_height = torch.maximum(ground_z, self._last_support_ground_height)
            has_fallen = fall_termination(
                rigid_body_pos=robot_state.rigid_body_pos,
                rigid_body_contacts=robot_state.rigid_body_contacts,
                ground_heights=fall_ref_height,
                termination_height=self.config.fall_termination_height,
                non_termination_contact_body_ids=(
                    self.env.non_termination_contact_body_ids
                ),
                progress_buf=self.env.progress_buf,
            )
            terminated = terminated | (has_fallen & past_grace)

        if self.config.enable_gap_termination:
            fell_in_gap = root_z < (
                self._last_support_root_height - self.config.gap_termination_threshold
            )
            terminated = terminated | (fell_in_gap & past_grace)

        if self.config.enable_stuck_termination:
            oldest_pos = self._root_pos_history[:, self._root_pos_history_idx]
            displacement = torch.linalg.norm(root_pos - oldest_pos, dim=-1)
            is_stuck = displacement < self.config.stuck_movement_threshold
            is_below_support = root_z < (
                self._last_support_root_height - self.config.stuck_height_threshold
            )
            enough_history = self.env.progress_buf >= self.config.stuck_window_frames
            terminated = terminated | (
                is_stuck & is_below_support & enough_history & past_grace
            )

        # Fall, gap, and stuck failures are terminal resets: value bootstrap should
        # be zeroed for the same envs that reset. Return a separate reset tensor so
        # callers can safely mutate reset_buf without aliasing terminate_buf.
        return terminated.clone(), terminated

    def populate_context(self, ctx: EnvContext) -> None:
        ctx.target = TargetContext(
            tar_pos=self._tar_pos,
            tar_proximity_threshold=self.config.tar_proximity_threshold,
        )

    def _ground_heights(self, positions: Tensor) -> Tensor:
        if self.env.terrain is None:
            return torch.zeros(positions.shape[0], device=self.env.device)
        return self.env.terrain.get_ground_heights(positions).squeeze(-1)

    def _update_target_heights(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._tar_pos[env_ids, 2] = self._ground_heights(self._tar_pos[env_ids])

    def _move_targets(self, env_ids: Tensor, dx: float, dy: float) -> None:
        self._tar_pos[env_ids, 0] += dx
        self._tar_pos[env_ids, 1] += dy
        self._update_target_heights(env_ids)

    def create_visualization_markers(
        self, headless: bool
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
