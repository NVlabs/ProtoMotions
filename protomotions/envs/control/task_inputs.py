# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a control component may read: one frozen value, built by the env.

A control is constructed from its config and a :class:`TaskInputs` and reads
``self.inputs`` and nothing else: no env, no simulator, no motion library.
The value carries exactly what the in-tree controls read today: the batch
(``num_envs``, ``device``, ``dt``), the robot config, the read-only episode
clock, the placed reference query, the terrain's ground-height query, the
control stage's random generator, and the declared UI exceptions
(``headless``, ``ui``, ``viewer_reset``). The robot state is never fetched:
``reset_rows``, ``advance`` and ``get_markers_state`` receive the state the
env already read. Two declared writes exist for the animation players only:
``restart_episode(env_ids)`` zeroes the episode clock of those rows and
``teleport(env_ids, robot_state, object_state)`` queues a pose write the env
applies after the controls advanced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    # Annotations only: the env package imports the control base, which
    # imports this module, so the env-side types cannot be imported here.
    from protomotions.envs.base_env.owners import EpisodeClock
    from protomotions.envs.base_env.reset.placed_reference import PlacedReference
    from protomotions.simulator.base_simulator.simulator_state import (
        ObjectState,
        ResetState,
    )
    from protomotions.simulator.base_simulator.user_interface import (
        KeyBinding,
        UserInterface,
    )


@dataclass(frozen=True)
class TaskInputs:
    """Everything a control component reads; see the module docstring."""

    num_envs: int
    device: torch.device
    dt: float
    robot: object  # the RobotConfig
    clock: "EpisodeClock"
    reference: "PlacedReference"
    ground_height: Callable[[Tensor], Tensor]  # points [n, 2 or 3] -> heights [n]
    generator: Optional[torch.Generator]  # the control stage; None: the global stream
    headless: bool
    ui: "UserInterface"
    viewer_reset: "KeyBinding"  # the env's R key; consume() is the edge
    restart_episode: Callable[[Tensor], None]
    teleport: Callable[[Tensor, "ResetState", Optional["ObjectState"]], None]


__all__ = ["TaskInputs"]
