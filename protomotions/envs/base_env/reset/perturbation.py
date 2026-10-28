# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset-time perturbation of the pose about to be written.

:class:`ResetPerturbation` applies the robot config's ``reset_noise`` (the
Reference State Initialization noise of ``protomotions/envs/reset_noise.py``)
to the reset state, in place, within the robot's joint limits. It holds the
robot config and reads ``reset_noise`` off it on every ``apply``, as the
inline reset did, so a noise config changed on a built env is honoured. The
env keeps the switch (``BaseEnv.perturbations_enabled``, which the evaluator
turns off so success metrics measure tracking, not robustness) and passes it
in: a disabled or unconfigured perturbation draws nothing from the global
torch generator. ``apply_reset_noise`` is looked up in this module at call
time, which is where the parity harness spies it.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.reset_noise import apply_reset_noise
from protomotions.simulator.base_simulator.simulator_state import ResetState


class ResetPerturbation:
    """The robot config's reset noise, applied under the env's switch.

    ``robot_config.reset_noise`` is read on every ``apply`` (not captured at
    build); the joint limits are the kinematic tensors the build moved to
    the device.
    """

    def __init__(
        self,
        robot_config,
        dof_limits_lower: Tensor,
        dof_limits_upper: Tensor,
        generator: Optional[torch.Generator],
    ):
        self.robot_config = robot_config
        self.dof_limits_lower = dof_limits_lower
        self.dof_limits_upper = dof_limits_upper
        # The reset-noise stage's generator (None: the global stream).
        self.generator = generator

    def apply(self, reset_state: ResetState, enabled: bool) -> None:
        """Perturb ``reset_state`` in place when ``enabled`` and the robot
        config has a ``reset_noise``."""
        config = self.robot_config.reset_noise
        if not enabled or config is None:
            return
        apply_reset_noise(
            reset_state=reset_state,
            config=config,
            dof_limits_lower=self.dof_limits_lower,
            dof_limits_upper=self.dof_limits_upper,
            generator=self.generator,
        )


__all__ = ["ResetPerturbation"]
