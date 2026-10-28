# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery resets: a cache of physically settled fall poses.

:class:`RecoveryBank` holds the cache (``fall_states``) and does the two
things ``recovery_reset`` asks for. ``generate`` fills it once, at startup:
every env is dropped at a random orientation onto flat ground, driven with
random joint targets for ``fall_sim_steps`` physics steps, and the settled
poses are kept with their velocities zeroed. It runs inside a save/restore
of the whole env (the callables it is handed are ``BaseEnv.save_state`` /
``restore_state``, the owner walk), so the simulator, the history and the
caches are exactly as before afterwards. ``swap_in`` is the reset chain's
recovery phase: a Bernoulli draw picks the subset of the envs being reset
that start from a cached fall pose instead of the sampled one.

Both draw from the global torch generator (``randn``, ``rand``, ``randint``)
and ``generate`` also draws the flat spawn locations from the terrain's
private generator; the parity harness records every generator state, so the
calls and their order are kept exactly as the inline versions made them.
The cache is generated once and only ever read, so the bank is on no
snapshot walk (the completeness test allowlists it as a constant). The bank
captures its ``recovery_reset`` config at build, because the cache is
generated from it then: changing ``config.recovery_reset`` on a built env
has no effect.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor

from protomotions.envs.base_env.config import RecoveryResetConfig
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
)


class RecoveryBank:
    """The fall-pose cache and the reset-time swap into it.

    Holds the ``RecoveryResetConfig`` it was built with (None for a config
    that predates the field); the cache is generated from it at build, so
    a ``config.recovery_reset`` changed afterwards has no effect.
    """

    def __init__(
        self,
        config: Optional[RecoveryResetConfig],
        simulator,
        terrain,
        dof_limits_lower: Tensor,
        dof_limits_upper: Tensor,
        num_actions: int,
        num_envs: int,
        device: torch.device,
        generator: Optional[torch.Generator],
    ):
        self.config = config
        self.simulator = simulator
        self.terrain = terrain
        self.dof_limits_lower = dof_limits_lower
        self.dof_limits_upper = dof_limits_upper
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.device = device
        # The recovery stage's generator (None: the global stream).
        self.generator = generator
        self.fall_states: Optional[ResetState] = None

    @property
    def enabled(self) -> bool:
        """Whether resets may use a fall pose (a positive ``recovery_prob``)."""
        return self.config is not None and self.config.recovery_prob > 0

    def generate(
        self,
        default_state: ResetState,
        default_object_state: ObjectState,
        *,
        save_state: Callable[[], Any],
        restore_state: Callable[[Any], None],
    ) -> None:
        """Fill the cache by letting every env fall from a random orientation.

        ``default_state`` is the simulator's default reset pose for every
        env; with ``fall_sim_steps <= 0`` the cache is that pose itself.
        ``save_state`` / ``restore_state`` bracket the physics so the env is
        unchanged afterwards (the harness's ``fall_cache_leaks_into_training``
        mutation is this method without the restore).
        """
        config = self.config
        if config.fall_sim_steps <= 0:
            self.fall_states = default_state.clone()
            return

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        snapshot = save_state()
        try:
            fall_states = default_state.clone()
            flat_xy = self.terrain.sample_valid_locations(
                num_envs=self.num_envs, sample_flat=True
            )
            ground_heights = self.terrain.get_ground_heights(
                torch.cat(
                    [
                        flat_xy,
                        torch.zeros(self.num_envs, 1, device=self.device),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            fall_states.root_pos[:, :2] = flat_xy
            fall_states.root_pos[:, 2] += ground_heights

            random_quat = torch.randn(
                self.num_envs, 4, device=self.device, generator=self.generator
            )
            random_quat = random_quat / random_quat.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            fall_states.root_rot = random_quat
            fall_states.root_vel.zero_()
            fall_states.root_ang_vel.zero_()
            if fall_states.dof_vel is not None:
                fall_states.dof_vel.zero_()

            self.simulator.reset_envs(fall_states, default_object_state, env_ids)

            dof_lower = self.dof_limits_lower
            dof_upper = self.dof_limits_upper
            random_targets = (
                torch.rand(
                    self.num_envs,
                    self.num_actions,
                    dtype=torch.float,
                    device=self.device,
                    generator=self.generator,
                )
                * (dof_upper - dof_lower)
                + dof_lower
            )
            for _ in range(config.fall_sim_steps):
                self.simulator.step(random_targets, markers_callback=None)

            self.fall_states = ResetState.from_robot_state(
                self.simulator.get_robot_state()
            ).clone()
            self.fall_states.root_vel.zero_()
            self.fall_states.root_ang_vel.zero_()
            if self.fall_states.dof_vel is not None:
                self.fall_states.dof_vel.zero_()
        finally:
            restore_state(snapshot)

    def swap_in(
        self, env_ids: Tensor, reset_states: ResetState, respawn_root_offset: Tensor
    ) -> Tuple[ResetState, Tensor]:
        """Replace a Bernoulli subset of ``reset_states`` with cached fall poses.

        The swapped rows take a random cached pose's orientation and joint
        positions with zero velocities, keep their sampled XY, and sit at the
        cached root height above their spawn's terrain height
        (``respawn_root_offset[:, 2]`` of the envs being reset). Returns the
        states and the mask of the swapped rows.
        """
        recovery_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if not self.enabled:
            return reset_states, recovery_mask
        if self.fall_states is None:
            raise RuntimeError(
                "recovery_prob is positive but fall reset states were not generated"
            )

        recovery_mask = (
            torch.rand(len(env_ids), device=self.device, generator=self.generator)
            < self.config.recovery_prob
        )
        if not recovery_mask.any():
            return reset_states, recovery_mask

        local_ids = torch.nonzero(recovery_mask, as_tuple=False).flatten()
        random_ids = torch.randint(
            0,
            self.fall_states.root_pos.shape[0],
            (len(local_ids),),
            device=self.device,
            generator=self.generator,
        )
        fall_states = self.fall_states[random_ids]

        reset_states.root_rot[local_ids] = fall_states.root_rot
        reset_states.root_vel[local_ids] = 0.0
        reset_states.root_ang_vel[local_ids] = 0.0
        reset_states.dof_pos[local_ids] = fall_states.dof_pos
        if reset_states.dof_vel is not None and fall_states.dof_vel is not None:
            reset_states.dof_vel[local_ids] = 0.0

        env_subset = env_ids[local_ids]
        reset_states.root_pos[local_ids, 2] = (
            respawn_root_offset[env_subset, 2] + fall_states.root_pos[:, 2]
        )
        return reset_states, recovery_mask


__all__ = ["RecoveryBank"]
