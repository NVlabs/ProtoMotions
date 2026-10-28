# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pose the reset writes: default, reference, or a mix of the two.

:class:`ResetPose` builds the ``ResetState`` (robot) and ``ObjectState``
(scene objects) for the envs being reset. ``default`` starts from the
simulator's default pose and the scene library's default objects;
``reference`` from the clip the reference motion sampled and the scene
pose at that clip time. Both let the spawn placer sample where the env
lands (``sample_spawn``) and translate the pose there (``move_to_spawn``),
in that order. ``compute`` is the branch ``BaseEnv.reset`` takes: every env
default, every env reference, or (with ``force_default_mask``) the default
rows first and then the reference rows written into one combined state.
The mixed branch has no production caller today; it stays because the
``force_default_mask`` keyword is part of ``reset``'s surface.

The two default states are cached on first use, as the env's cached
properties were: the simulator's default pose is read once, after the
simulator is initialised.
"""

from __future__ import annotations

from functools import cached_property
from typing import Optional, Tuple

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
)


class ResetPose:
    """The default/reference/mixed pose of the envs being reset, at their spawn."""

    def __init__(
        self,
        simulator,
        scene_lib,
        motion,
        spawn,
        ref_object_respawn_offset: float,
        device: torch.device,
    ):
        self.simulator = simulator
        self.scene_lib = scene_lib
        self._motion = motion
        self._spawn = spawn
        self.ref_object_respawn_offset = ref_object_respawn_offset
        self.device = device

    @cached_property
    def default_reset_state(self) -> ResetState:
        """Default robot reset state from simulator."""
        return self.simulator.get_default_robot_reset_state()

    @cached_property
    def default_object_state(self) -> ObjectState:
        """Default object state (empty if no scenes)."""
        return self.scene_lib.get_default_object_state(self.device)

    def default(
        self, env_ids: Tensor, sample_flat: bool = False
    ) -> Tuple[ResetState, ObjectState]:
        """The default pose of ``env_ids``, at a freshly sampled spawn."""
        new_states = self.default_reset_state[env_ids].clone()
        new_object_states = self.default_object_state[env_ids].clone()

        self._spawn.sample_spawn(env_ids, ref_state=None, sample_flat=sample_flat)

        return self._spawn.move_to_spawn(env_ids, new_states, new_object_states)

    def reference(
        self,
        env_ids: Tensor,
        motion_ids: Tensor,
        motion_times: Tensor,
        sample_flat: bool = False,
    ) -> Tuple[ResetState, ObjectState]:
        """The pose of ``env_ids`` at ``motion_times`` of ``motion_ids``, with
        the scene objects at that time (velocities zeroed), at a fresh spawn."""
        ref_state = self._motion.motion_lib.get_motion_state(motion_ids, motion_times)
        new_states = ResetState.from_robot_state(ref_state).clone()

        new_object_states = self.scene_lib.get_scene_pose(
            env_ids, motion_times, respawn_offset=self.ref_object_respawn_offset
        )
        new_object_states.root_vel = torch.zeros_like(new_object_states.root_pos)
        new_object_states.root_ang_vel = torch.zeros_like(new_object_states.root_pos)

        self._spawn.sample_spawn(env_ids, ref_state=ref_state, sample_flat=sample_flat)

        return self._spawn.move_to_spawn(env_ids, new_states, new_object_states)

    def compute(
        self,
        env_ids: Tensor,
        ref_env_ids: Tensor,
        motion_ids: Optional[Tensor],
        motion_times: Optional[Tensor],
        sample_flat: bool = False,
    ) -> Tuple[ResetState, ObjectState]:
        """The pose of ``env_ids`` given which of them (``ref_env_ids``) reset
        from a clip: all default, all reference, or the mix (default rows
        computed first, then the reference rows)."""
        default_mask = ~torch.isin(env_ids, ref_env_ids)
        default_indices = default_mask.nonzero(as_tuple=True)[0]

        if ref_env_ids.numel() == 0:
            return self.default(env_ids, sample_flat)
        if default_indices.numel() == 0:
            return self.reference(ref_env_ids, motion_ids, motion_times, sample_flat)

        new_states = self.default_reset_state[env_ids].clone()
        new_object_states = self.default_object_state[env_ids].clone()

        default_env_ids = env_ids[default_indices]
        default_states, default_object_states = self.default(
            default_env_ids, sample_flat
        )
        new_states[default_indices] = default_states
        new_object_states[default_indices] = default_object_states

        ref_states, ref_object_states = self.reference(
            ref_env_ids, motion_ids, motion_times, sample_flat
        )
        ref_indices = torch.isin(env_ids, ref_env_ids).nonzero(as_tuple=True)[0]
        new_states[ref_indices] = ref_states
        new_object_states[ref_indices] = ref_object_states
        return new_states, new_object_states


__all__ = ["ResetPose"]
