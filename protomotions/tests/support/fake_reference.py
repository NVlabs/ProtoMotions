# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real :class:`PlacedReference` over a hand-built fake env.

The control, agent and evaluator tests build their env as a ``SimpleNamespace``
(or a small class) with ``motion_lib`` and ``motion_manager`` fakes and read
the reference motion through ``env.reference``. :func:`fake_reference` gives
them the production query over those fakes: the clip assignment and the poses
come from ``env.motion_manager`` / ``env.motion_lib`` (read live, so a test may
still swap a fake's method after construction), the placement from
``placement_offset`` (zeros unless given), the scene objects from
``scene_lib`` (none unless given) and ``spawn_translation()`` from
``respawn_root_offset`` (zeros unless given).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from protomotions.envs.base_env.reset.placed_reference import PlacedReference


class _NoScenes:
    has_pointclouds = False  # an object-less library computes none

    def num_scenes(self) -> int:
        return 0


class _FakeSpawn:
    """The two things the query reads off the spawn placer."""

    def __init__(
        self,
        placement_offset: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]],
        respawn_root_offset: Tensor,
        ref_respawn_offset: float,
    ):
        self._placement_offset = placement_offset
        self.respawn_root_offset = respawn_root_offset
        self.ref_respawn_offset = ref_respawn_offset

    def ref_pose_offset(self, target_pos: Tensor, env_ids: Optional[Tensor] = None):
        if self._placement_offset is None:
            return torch.zeros_like(target_pos)
        return self._placement_offset(target_pos, env_ids)


def fake_reference(
    env,
    *,
    placement_offset: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    scene_lib=None,
    ref_object_respawn_offset: float = 0.0,
    respawn_root_offset: Optional[Tensor] = None,
    ref_respawn_offset: float = 0.0,
) -> PlacedReference:
    """The production reference query over ``env``'s motion fakes.

    ``env`` needs ``num_envs``, ``device``, ``motion_lib`` and
    ``motion_manager`` (the last may be None: then only ``has_clips`` answers).
    ``placement_offset(positions, env_ids)`` stands in for the spawn placer's
    per-body offset; ``scene_lib`` for the scene library (``num_scenes`` and
    ``get_scene_pose``).
    """
    if respawn_root_offset is None:
        respawn_root_offset = torch.zeros(env.num_envs, 3, device=env.device)
    spawn = _FakeSpawn(placement_offset, respawn_root_offset, ref_respawn_offset)
    return PlacedReference(
        env,
        spawn,
        _NoScenes() if scene_lib is None else scene_lib,
        ref_object_respawn_offset,
        env.num_envs,
        env.device,
    )


__all__ = ["fake_reference"]
