# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where each env's robot is placed when it resets.

:class:`SpawnPlacer` owns ``respawn_root_offset``: the root translation from
the frame the reset pose was expressed in (the reference clip's, or the
simulator's default pose) to the place on the terrain the env spawned at.
It samples that place (``sample_spawn``: a scene env spawns at its scene's
position, every other env at a random valid terrain location, both lifted
onto the terrain so no body starts below ground), applies the offset to the
pose and the scene objects about to be written (``move_to_spawn``), and
hands the tracking tasks the per-body offset that puts a reference pose into
the same frame (``ref_pose_offset``, read through
``PlacedReference.placement_offset`` by the controls, the mimic evaluator and
ADD; nothing outside the reset chain reads the raw offset).

The placer is also on the step walk (an ``Advancer``): with the motion
manager's ``realign_motion_with_humanoid_on_each_step`` the XY of every
env's offset is re-anchored on where its robot is now after every step
(``align_motion_with_humanoid``), so tracking compares local pose and
global orientation, never global XY. It reads the current manager through
the :class:`~.motion.ReferenceMotion` (the live shard switch replaces it)
and the robot's root from the simulator it holds (``get_root_state``), as
the recovery bank and the reset pose do.

On the snapshot walk the offset is saved and restored (the evaluator's
write-back). No ``reset_rows``: the reset chain writes it through
``sample_spawn`` in its pose phase.

Terrain sampling draws from the terrain's private generator; the parity
harness logs ``terrain.sample_valid_locations``, so the one call per reset
and its arguments are kept exactly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
)


class SpawnPlacer:
    """The respawn offsets and everything that samples, applies or realigns them."""

    name = "spawn"

    def __init__(
        self,
        terrain,
        scene_lib,
        motion,
        simulator,
        ref_respawn_offset: float,
        skip_correct_terrain_height_on_flat: bool,
        num_envs: int,
        device: torch.device,
    ):
        self.terrain = terrain
        self.scene_lib = scene_lib
        self._motion = motion
        self.simulator = simulator
        self.ref_respawn_offset = ref_respawn_offset
        # Skipping the height query is only safe on flat terrain, where the
        # correction is identically zero.
        self.skip_height_correction = (
            skip_correct_terrain_height_on_flat and terrain.is_flat()
        )
        self.num_envs = num_envs
        self.device = device
        self.respawn_root_offset = torch.zeros(
            num_envs, 3, dtype=torch.float, device=device
        )

    def scene_non_scene_mask(self, env_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """``(scene_mask, non_scene_mask)`` over ``env_ids``: which envs spawn
        at a scene. For now either every env requires a scene or none does."""
        num_envs = len(env_ids)
        if self.scene_lib.num_scenes() > 0:
            scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        else:
            scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        return scene_mask, non_scene_mask

    def sample_spawn(
        self,
        env_ids: Tensor,
        ref_state: Optional[RobotState] = None,
        sample_flat: bool = False,
    ) -> None:
        """Sample a spawn location for ``env_ids`` and write their offsets.

        A scene env spawns at its scene's XY; every other env at a random
        valid terrain location (``sample_flat`` restricts it to the flat
        region). ``ref_state`` is the pose about to be spawned (None for the
        default pose at the origin): the XY offset is relative to its root,
        and the height correction lifts the whole body set so the body
        furthest below the terrain lands on it. During co-training scene
        groups use flat terrain, but at inference the resolved terrain may
        be complex (with negative heights that get normalised), so the
        correction applies to scene and non-scene envs alike unless the
        terrain is entirely flat. ``ref_respawn_offset`` is added on top.
        """
        respawn_offset = torch.zeros((len(env_ids), 3), device=self.device)

        scene_mask, non_scene_mask = self.scene_non_scene_mask(env_ids)

        if scene_mask.any():
            scene_pos = self.scene_lib.get_scene_positions(self.terrain, self.device)
            respawn_offset[scene_mask, :2] = scene_pos[env_ids[scene_mask], :2]

            # Scene envs also need terrain height correction: the object
            # playground is flat at height-field 0, but terrain normalisation
            # (shifting min height to z=0) can raise the playground above
            # world z=0. Without correction the agent spawns underground.
            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[scene_mask, 2] = terrain_heights

        if non_scene_mask.any():
            num_non_scene = non_scene_mask.sum().item()
            respawn_position_xy = self.terrain.sample_valid_locations(
                num_envs=num_non_scene, sample_flat=sample_flat
            )

            if ref_state is None:
                ref_root = torch.zeros((num_non_scene, 2), device=self.device)
            else:
                ref_root = ref_state.root_pos[non_scene_mask, :2]
            respawn_offset[non_scene_mask, :2] = respawn_position_xy - ref_root

            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[non_scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        non_scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[non_scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[non_scene_mask, 2] = terrain_heights

        respawn_offset[:, 2] += self.ref_respawn_offset

        self.respawn_root_offset[env_ids] = respawn_offset

    def move_to_spawn(
        self,
        env_ids: Tensor,
        new_states: ResetState,
        new_object_states: ObjectState,
    ) -> Tuple[ResetState, ObjectState]:
        """Translate the pose (and the scene objects) by the sampled offsets."""
        new_states.root_pos += self.respawn_root_offset[env_ids]
        if self.scene_lib.num_scenes() > 0:
            new_object_states.root_pos += self.respawn_root_offset[env_ids].unsqueeze(1)

        return new_states, new_object_states

    def ref_pose_offset(
        self, target_pos: Tensor, env_ids: Optional[Tensor] = None
    ) -> Tensor:
        """The per-body offset that puts a reference pose where the env spawned.

        Args:
            target_pos: Reference body positions [num_envs, num_bodies, 3]
                without the spawn offset applied.
            env_ids: Environment indices [num_envs]. If None, every env.

        Returns:
            Offset to add to ``target_pos`` [num_envs, num_bodies, 3]: every
            body shares the env's XY offset, and the Z offset that lifts the
            body furthest below the terrain onto it, so the rigid structure
            of the pose is preserved.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        new_offset = torch.zeros_like(target_pos)
        new_offset[:, :, :2] = self.respawn_root_offset[env_ids, :2][:, None, :]

        if not self.skip_height_correction:
            target_pos_spawned = target_pos.clone() + new_offset
            z_offset = self.terrain.find_terrain_height_for_max_below_body(
                target_pos_spawned
            )
            new_offset[:, :, 2] = z_offset.unsqueeze(1)

        return new_offset

    def align_motion_with_humanoid(self, env_ids: Tensor, root_pos: Tensor) -> None:
        """Re-anchor the XY offset of ``env_ids`` so the reference clip's
        current root lands on ``root_pos`` [len(env_ids), 3]."""
        ref_state = self._motion.motion_lib.get_motion_state(
            self._motion.motion_manager.motion_ids[env_ids],
            self._motion.motion_manager.motion_times[env_ids],
        )

        self.respawn_root_offset[env_ids, :2] = (
            root_pos[:, :2] - ref_state.rigid_body_pos[:, 0, :2]
        )

    def advance(self, state) -> None:
        """Re-anchor every env's clip on its robot after the step, when the
        current motion manager's config asks for it; otherwise nothing. Reads
        the simulator's root itself (the realignment's own read), not the
        state argument."""
        manager = self._motion.motion_manager
        if (
            manager is None
            or not manager.config.realign_motion_with_humanoid_on_each_step
        ):
            return
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.align_motion_with_humanoid(
            env_ids, self.simulator.get_root_state().root_pos
        )

    def save_state(self) -> dict:
        return {"respawn_root_offset": self.respawn_root_offset.clone()}

    def load_state(self, snapshot: dict) -> None:
        self.respawn_root_offset.copy_(snapshot["respawn_root_offset"])


__all__ = ["SpawnPlacer"]
