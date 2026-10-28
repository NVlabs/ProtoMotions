# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-reaching control component.

Owns the target position of a target-reaching task; the source that moves it
(random draws for training, keyboard keys for a viewer) is a
``command_source.TargetCommandSource`` the control constructs from its
config's ``_target_``. Publishes the ``target.*`` names, and owns the task's
three episode-ending checks as bound
kernels (``terminations``): ``fall`` (a forbidden body touching the terrain
while low over the fall reference height), ``gap`` (the anchor dropped below
the last elevated support) and ``stuck`` (little movement over a window while
below the last support). The checks read what ``advance`` latches and
``publish`` hands out under ``failure_checks``: the fall reference height
(the ground under the anchor or the last elevated support, whichever is
higher), the last-support root height and the oldest anchor position of the
stuck window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from protomotions.components.pose_lib import build_body_ids_tensor
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig

# The module, not its names: a name imported here would be an attribute of
# this module too, and a bundle written before the sources moved (naming
# ``...target_control.RandomTargetCommandSourceConfig``) would then load
# instead of being refused with the regenerate message.
from protomotions.envs.control import command_source as command_sources
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.base import named_contact_fall_termination
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils.hydra_replacement import get_class


@dataclass
class TargetControlConfig(ControlComponentConfig):
    """Configuration for target-reaching tasks."""

    _target_: str = "protomotions.envs.control.target_control.TargetControl"

    tar_proximity_threshold: float = 0.25
    command_source: command_sources.TargetCommandSourceConfig = field(
        default_factory=command_sources.RandomTargetCommandSourceConfig
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


def _checks_enabled(config: TargetControlConfig) -> bool:
    return bool(
        config.enable_fall_termination
        or config.enable_gap_termination
        or config.enable_stuck_termination
    )


# ---------------------------------------------------------------------------
# The episode-ending checks, as kernels over the context
# ---------------------------------------------------------------------------


def fall_bodies(robot_config) -> Tuple[List[str], Tensor]:
    """The bodies whose terrain contact is a fall: every body not in
    ``robot.non_termination_contact_bodies`` (``"all"`` allows every body).
    Returns their names (the contact matrix's rows) and ids (the kinematics')."""
    names = list(robot_config.kinematic_info.body_names)
    allowed = robot_config.non_termination_contact_bodies
    allowed = set(names) if allowed == "all" else set(allowed)
    forbidden = [name for name in names if name not in allowed]
    return forbidden, build_body_ids_tensor(names, forbidden, torch.device("cpu"))


def target_fall(
    rigid_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    contact_forces: Tensor = EnvContext.contact_forces,
    contact_body_index: Dict[str, int] = EnvContext.contact_body_index,
    contact_partner_index: Dict[str, int] = EnvContext.contact_partner_index,
    fall_ref_height: Tensor = EnvContext.target.fall_ref_height,
    progress_buf: Tensor = EnvContext.progress_buf,
    *,
    body_ids: Tensor,
    body_names: List[str],
    partner_names: List[str],
    termination_height: float,
    reset_grace_period: int,
) -> Tensor:
    """Boolean: a forbidden body touches the terrain while low over the fall
    reference height, past the reset grace period. Stays a boolean kernel: a
    contact conjunction under a grace mask has no one scalar to cut."""
    fallen = named_contact_fall_termination(
        rigid_body_pos=rigid_body_pos,
        contact_forces=contact_forces,
        contact_body_index=contact_body_index,
        contact_partner_index=contact_partner_index,
        ground_heights=fall_ref_height,
        progress_buf=progress_buf,
        body_ids=body_ids,
        body_names=body_names,
        partner_names=partner_names,
        termination_height=termination_height,
    )
    return fallen & (progress_buf > reset_grace_period)


def target_gap(
    anchor_pos: Tensor = EnvContext.current.anchor_pos,
    last_support_root_height: Tensor = EnvContext.target.last_support_root_height,
    progress_buf: Tensor = EnvContext.progress_buf,
    *,
    gap_threshold: float,
    reset_grace_period: int,
) -> Tensor:
    """Boolean: the anchor is more than ``gap_threshold`` below the root height at
    the last elevated support, outside the reset grace period. The comparison is
    the old ``z < last_support - threshold`` as written, so the two agree at the
    one float boundary where ``last_support - z > threshold`` would not."""
    below = anchor_pos[..., 2] < last_support_root_height - gap_threshold
    return below & (progress_buf > reset_grace_period)


def target_stuck(
    anchor_pos: Tensor = EnvContext.current.anchor_pos,
    oldest_root_pos: Tensor = EnvContext.target.oldest_root_pos,
    last_support_root_height: Tensor = EnvContext.target.last_support_root_height,
    progress_buf: Tensor = EnvContext.progress_buf,
    *,
    movement_threshold: float,
    height_threshold: float,
    window_frames: int,
    reset_grace_period: int,
) -> Tensor:
    """Boolean: the anchor moved less than ``movement_threshold`` over the
    stuck window while more than ``height_threshold`` below the last support,
    with a full window of history and past the grace period. Stays a boolean
    kernel: two cuts in one check."""
    displacement = torch.linalg.norm(anchor_pos - oldest_root_pos, dim=-1)
    is_stuck = displacement < movement_threshold
    is_below_support = anchor_pos[..., 2] < (
        last_support_root_height - height_threshold
    )
    enough_history = progress_buf >= window_frames
    return (
        is_stuck
        & is_below_support
        & enough_history
        & (progress_buf > reset_grace_period)
    )


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------


class TargetControl(ControlComponent):
    """Stateful target sampler for target-reaching tasks."""

    config: TargetControlConfig
    namespaces = ("target",)

    @classmethod
    def provides(cls, config: TargetControlConfig, tag: str) -> bool:
        """``failure_checks`` (the three names the checks read) is on when
        any of the three checks is; every other tag reads the config field."""
        if tag == "failure_checks":
            return _checks_enabled(config)
        return super().provides(config, tag)

    @classmethod
    def terminations(
        cls, config: TargetControlConfig, robot_config
    ) -> Dict[str, TerminationRule]:
        """``fall``, ``gap`` and ``stuck``, each only when its switch is on.
        A robot without contact bodies is refused on the ``contact_forces``
        binding by the standard absence sentence; a forbidden body missing
        from the contact matrix by ``required_keys``."""
        entries: Dict[str, TerminationRule] = {}
        grace = config.reset_grace_period
        if config.enable_fall_termination:
            body_names, body_ids = fall_bodies(robot_config)
            entries["fall"] = TerminationRule(
                MdpComponent.from_kernel(
                    target_fall,
                    body_ids=body_ids,
                    body_names=body_names,
                    partner_names=["terrain"],
                    termination_height=config.fall_termination_height,
                    reset_grace_period=grace,
                    required_keys={
                        "contact_body_index": body_names,
                        "contact_partner_index": ["terrain"],
                    },
                )
            )
        if config.enable_gap_termination:
            entries["gap"] = TerminationRule(
                MdpComponent.from_kernel(
                    target_gap,
                    gap_threshold=config.gap_termination_threshold,
                    reset_grace_period=grace,
                )
            )
        if config.enable_stuck_termination:
            entries["stuck"] = TerminationRule(
                MdpComponent.from_kernel(
                    target_stuck,
                    movement_threshold=config.stuck_movement_threshold,
                    height_threshold=config.stuck_height_threshold,
                    window_frames=config.stuck_window_frames,
                    reset_grace_period=grace,
                )
            )
        return entries

    def __init__(self, config: TargetControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        add = self.state.add
        self.tar_pos = add("tar_pos", (3,), torch.float, 0.0)
        self.tar_change_time = add("tar_change_time", (), torch.float, 0.0)
        # What the checks read: the root and ground heights at the last
        # elevated support, the fall reference height, the stuck window.
        self.last_support_root_height = add(
            "last_support_root_height", (), torch.float, 0.0
        )
        self.last_support_ground_height = add(
            "last_support_ground_height", (), torch.float, 0.0
        )
        self.fall_ref_height = add("fall_ref_height", (), torch.float, 0.0)
        self.root_pos_history = add(
            "root_pos_history", (config.stuck_window_frames, 3), torch.float, 0.0
        )
        # The ring's write cursor is one for every env: a shared tensor.
        self.history_index = self.state.shared(
            "history_index", torch.zeros((), dtype=torch.long, device=inputs.device)
        )
        self.checks_enabled = _checks_enabled(config)
        command_source_cls = get_class(config.command_source._target_)
        self.command_source = command_source_cls(config.command_source, self)

    def reset_rows(self, env_ids: Tensor, state: RobotState) -> None:
        if len(env_ids) == 0:
            return
        super().reset_rows(env_ids, state)
        anchor = self.inputs.robot.anchor_body_index
        root_pos = state.rigid_body_pos[:, anchor]
        self.command_source.reset(env_ids, root_pos, state.rigid_body_rot[:, anchor])

        self.last_support_root_height[env_ids] = root_pos[:, 2]
        ground = self._ground_heights(root_pos)
        self.last_support_ground_height[env_ids] = ground
        self.fall_ref_height[env_ids] = ground
        self.root_pos_history[env_ids] = root_pos.unsqueeze(1).expand(
            -1, self.config.stuck_window_frames, -1
        )

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        """Move the target when due, then latch what the checks read: the
        stuck window's newest anchor position, the last elevated support and
        the fall reference height."""
        self.command_source.advance(state)
        if self.checks_enabled:
            root_pos = state.rigid_body_pos[:, self.inputs.robot.anchor_body_index]
            index = int(self.history_index)
            self.root_pos_history[:, index] = root_pos
            self.history_index.fill_((index + 1) % self.config.stuck_window_frames)

            root_z = root_pos[:, 2]
            ground_z = self._ground_heights(root_pos)
            on_elevated_ground = ground_z > 0.0
            self.last_support_root_height[on_elevated_ground] = root_z[
                on_elevated_ground
            ]
            self.last_support_ground_height[on_elevated_ground] = ground_z[
                on_elevated_ground
            ]
            self.fall_ref_height[:] = torch.maximum(
                ground_z, self.last_support_ground_height
            )
        return {}

    def close(self) -> None:
        """Release UI handles held by the command source."""
        self.command_source.close()

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Publish the ``target.*`` names (see ``TargetContext``); the three
        failure-check names only when a check is enabled."""
        rows = slice(None) if env_ids is None else env_ids
        published: Dict[str, Any] = {
            "target.tar_pos": self.tar_pos[rows],
            "target.tar_proximity_threshold": self.config.tar_proximity_threshold,
        }
        if self.checks_enabled:
            published.update(
                {
                    "target.fall_ref_height": self.fall_ref_height[rows],
                    "target.last_support_root_height": self.last_support_root_height[
                        rows
                    ],
                    "target.oldest_root_pos": self.root_pos_history[
                        rows, int(self.history_index)
                    ],
                }
            )
        return published

    def _ground_heights(self, positions: Tensor) -> Tensor:
        return self.inputs.ground_height(positions)

    def _update_target_heights(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.tar_pos[env_ids, 2] = self._ground_heights(self.tar_pos[env_ids])

    def _move_targets(self, env_ids: Tensor, dx: float, dy: float) -> None:
        self.tar_pos[env_ids, 0] += dx
        self.tar_pos[env_ids, 1] += dy
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

    def get_markers_state(self, state: RobotState) -> Dict[str, MarkerState]:
        if self.inputs.headless:
            return {}
        num_envs, device = self.inputs.num_envs, self.inputs.device
        tar_pos = self.tar_pos.view(num_envs, 1, 3).clone()
        tar_pos[..., 2] += 0.1
        tar_rot = torch.zeros(num_envs, 1, 4, device=device)
        tar_rot[..., -1] = 1.0
        return {"target_markers": MarkerState(translation=tar_pos, orientation=tar_rot)}
