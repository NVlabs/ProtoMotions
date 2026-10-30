# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the reference clip's pose is on this terrain, at which time, for which rows.

:class:`PlacedReference` is the one object a task reads the reference motion
through. It composes the installed library and manager
(:class:`~.motion.ReferenceMotion`, read live on every call because the shard
switcher replaces both) with the spawn placer's per-env placement
(:meth:`~.spawn.SpawnPlacer.ref_pose_offset`), so no task composes a spawn
offset by hand.

Reads: ``has_clips``; ``ids`` and ``times`` (the manager's own tensors, read
only by contract); ``lengths``; ``done``; ``raw`` and ``raw_positions`` (the
library's poses in the clip's own frame, for differences of two poses of one
env); ``placement_offset``; ``poses_at`` (one placement per pose); ``window``
and ``current_and_window`` (a window of future times placed from its first
slot, the mimic convention); ``scene_poses_at``; ``spawn_translation`` (the rows'
spawn offset, with the spawn-only height bump separable).

Writers, all the manager's clip-assignment API: ``set_assignment``,
``assign``, ``resample`` and ``advance_time`` (the one clock tick the
kinematic players make themselves; the env's step clock is
``ReferenceMotion.advance``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    RobotState,
)

NO_CLIPS = "no reference motion is installed: the motion library has no clips"
NO_DONE_TRACKS = (
    "the reference motion has no done tracks: configure MimicMotionManagerConfig"
)


@dataclass(frozen=True)
class ReferenceWindow:
    """Reference poses at ``times[n, k]``: ``k`` future slots for ``n`` rows,
    placed where each row spawned (one placement per row, from slot 0)."""

    times: Tensor  # [n, k]
    pos: Tensor  # [n, k, bodies, 3]
    rot: Tensor  # [n, k, bodies, 4]
    vel: Tensor  # [n, k, bodies, 3]
    ang_vel: Tensor  # [n, k, bodies, 3]
    dof_pos: Tensor  # [n, k, dofs]
    dof_vel: Tensor  # [n, k, dofs]


class PlacedReference:
    """Every reference-motion question a task asks, answered in the env's frame."""

    def __init__(
        self,
        motion,
        spawn,
        scene_lib,
        ref_object_respawn_offset: float,
        num_envs: int,
        device: torch.device,
    ):
        self._motion = motion
        self._spawn = spawn
        self._scene_lib = scene_lib
        self._ref_object_respawn_offset = ref_object_respawn_offset
        self.num_envs = num_envs
        self.device = device

    # --- what is installed ---------------------------------------------------
    @property
    def has_clips(self) -> bool:
        """Whether a manager exists (the installed library has clips)."""
        return self._motion.motion_manager is not None

    @property
    def has_scene_objects(self) -> bool:
        """Whether the scene library places objects (it has scenes)."""
        return self._scene_lib.num_scenes() > 0

    def _manager(self):
        manager = self._motion.motion_manager
        if manager is None:
            raise RuntimeError(NO_CLIPS)
        return manager

    @property
    def ids(self) -> Tensor:
        """Every env's clip id (the manager's tensor; read only by contract)."""
        return self._manager().motion_ids

    @property
    def times(self) -> Tensor:
        """Every env's clip time in seconds (the manager's tensor; read only)."""
        return self._manager().motion_times

    def _rows(self, tensor: Tensor, env_ids: Optional[Tensor]) -> Tensor:
        return tensor if env_ids is None else tensor[env_ids]

    def lengths(self, ids: Tensor) -> Tensor:
        """Clip lengths in seconds."""
        return self._motion.motion_lib.get_motion_length(ids)

    def done(self, env_ids: Optional[Tensor] = None) -> Tensor:
        """Which rows' clips end within the next step (bool)."""
        tick = getattr(self._manager(), "get_done_tracks", None)
        if tick is None:
            raise RuntimeError(NO_DONE_TRACKS)
        return tick(env_ids)

    # --- poses in the clip's own frame ----------------------------------------
    def raw(self, ids: Tensor, times: Tensor) -> RobotState:
        """The library's poses at ``(ids, times)``, unplaced: for differences of
        two poses of one env, where the placement cancels."""
        return self._motion.motion_lib.get_motion_state(ids, times)

    def raw_positions(self, ids: Tensor, times: Tensor) -> Tensor:
        """Body positions [n, bodies, 3] at ``(ids, times)``, unplaced."""
        return self._motion.motion_lib.get_motion_positions(ids, times)

    # --- placement -----------------------------------------------------------
    def placement_offset(self, positions: Tensor, env_ids: Optional[Tensor]) -> Tensor:
        """The per-body offset [n, bodies, 3] that puts ``positions`` (one pose
        per row of ``env_ids``) where those envs spawned: the env's XY offset
        and one terrain height per pose."""
        return self._spawn.ref_pose_offset(positions, env_ids)

    def poses_at(self, env_ids: Optional[Tensor], times: Tensor) -> RobotState:
        """The rows' clips at ``times`` [n], each pose placed on its own."""
        state = self.raw(self._rows(self.ids, env_ids), times)
        positions = state.rigid_body_pos.clone()
        positions += self.placement_offset(positions, env_ids)
        state.rigid_body_pos = positions
        return state

    def window(self, env_ids: Optional[Tensor], times: Tensor) -> ReferenceWindow:
        """The rows' clips at ``times[n, k]``, every slot placed from slot 0."""
        ids = self._rows(self.ids, env_ids)
        num_rows, num_slots = times.shape
        flat_ids = ids.unsqueeze(-1).expand(num_rows, num_slots).reshape(-1)
        return self._place_window(self.raw(flat_ids, times.reshape(-1)), times, env_ids)

    def current_and_window(
        self, env_ids: Optional[Tensor], time_offsets: Tensor
    ) -> Tuple[RobotState, ReferenceWindow]:
        """The current pose and the window ``time_offsets[k]`` seconds ahead of it
        (clamped to the clip's end), from ONE library query: the current frame
        placed per pose, the window placed from its first slot."""
        ids = self._rows(self.ids, env_ids)
        times = self._rows(self.times, env_ids)
        num_rows, num_slots = ids.shape[0], time_offsets.shape[0]
        future_times = times.unsqueeze(-1) + time_offsets
        future_times = torch.minimum(future_times, self.lengths(ids).unsqueeze(-1))
        flat_ids = ids.unsqueeze(-1).expand(num_rows, num_slots).reshape(-1)
        combined = self.raw(
            torch.cat((ids, flat_ids), dim=0),
            torch.cat((times, future_times.reshape(-1)), dim=0),
        )
        current, future = combined[:num_rows], combined[num_rows:]
        positions = current.rigid_body_pos.clone()
        positions += self.placement_offset(positions, env_ids)
        current.rigid_body_pos = positions
        return current, self._place_window(future, future_times, env_ids)

    def _place_window(
        self, state: RobotState, times: Tensor, env_ids: Optional[Tensor]
    ) -> ReferenceWindow:
        num_rows, num_slots = times.shape
        num_bodies = state.rigid_body_pos.shape[1]
        num_dofs = state.dof_pos.shape[1]
        pos = state.rigid_body_pos.view(num_rows, num_slots, num_bodies, 3).clone()
        # All slots share one placement, taken from the first step.
        pos += self.placement_offset(pos[:, 0], env_ids).unsqueeze(1)
        return ReferenceWindow(
            times=times,
            pos=pos,
            rot=state.rigid_body_rot.view(num_rows, num_slots, num_bodies, 4),
            vel=state.rigid_body_vel.view(num_rows, num_slots, num_bodies, 3),
            ang_vel=state.rigid_body_ang_vel.view(num_rows, num_slots, num_bodies, 3),
            dof_pos=state.dof_pos.view(num_rows, num_slots, num_dofs),
            dof_vel=state.dof_vel.view(num_rows, num_slots, num_dofs),
        )

    def scene_poses_at(self, env_ids: Tensor, times: Tensor) -> Optional[ObjectState]:
        """The rows' scene objects at ``times``, at rest (velocities zeroed);
        None when the scene library has no scenes."""
        if self._scene_lib.num_scenes() == 0:
            return None
        objects = self._scene_lib.get_scene_pose(
            env_ids, times, self._ref_object_respawn_offset
        )
        objects.root_vel = torch.zeros_like(objects.root_pos)
        objects.root_ang_vel = torch.zeros_like(objects.root_pos)
        return objects

    def spawn_translation(self, env_ids: Optional[Tensor]) -> Tensor:
        """The rows' spawn offset ``[n, 3]`` with the spawn-only height bump
        (``EnvConfig.ref_respawn_offset``, added to z at every spawn) taken
        out: scene XY and terrain height only, which is what a replay adds
        back (the mimic evaluator's inverse of the placement). An owned copy."""
        rows = (
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            if env_ids is None
            else env_ids
        )
        offset = self._spawn.respawn_root_offset[rows].clone()
        offset[:, 2] -= self._spawn.ref_respawn_offset
        return offset

    # --- the clip assignment's writers ---------------------------------------
    def set_assignment(self, ids: Tensor, times: Tensor) -> None:
        """Overwrite every env's clip and time in place."""
        self._manager().set_assignment(ids, times)

    def assign(
        self,
        env_ids: Tensor,
        *,
        ids: Optional[Tensor] = None,
        times: Optional[Tensor] = None,
    ) -> None:
        """Write the clip and/or time of ``env_ids`` only, in place."""
        self._manager().assign(env_ids, motion_ids=ids, motion_times=times)

    def resample(self, env_ids: Tensor) -> None:
        """Sample new clips (and start times) for ``env_ids``."""
        self._manager().sample_motions(env_ids)

    def advance_time(self, dt: float) -> None:
        """Tick every env's clip time by ``dt`` seconds."""
        self._manager().motion_times += dt


__all__: Tuple[Any, ...] = ("PlacedReference", "ReferenceWindow")
