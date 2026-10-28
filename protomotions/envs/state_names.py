# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-body names derived from one robot state, for the context and for export.

A ``BodyState`` holds the raw quantities of one state source (body positions,
rotations, velocities, joint state ...). Observation kernels also read the
root and anchor slices of those quantities, the root height and the local
angular velocities, so :func:`body_state_names` derives them once. The
environment's state publisher calls it for every state source it publishes
(``current``, ``noisy``, ``historical`` ...), and the ONNX exporters call it
to build the sample a traced observation kernel reads, so the two can never
drift apart on how a derived name is computed.
"""

from __future__ import annotations

from typing import Any, Dict

from protomotions.envs.obs.humanoid import compute_local_ang_vel
from protomotions.envs.state import BodyState


def body_state_names(state: BodyState, anchor_idx: int) -> Dict[str, Any]:
    """The names one state source publishes, derived from a ``BodyState``.

    Every available tensor under its own name, ``anchor_idx``, the root
    (body 0) and anchor slices of each body quantity the state has,
    ``root_height``, and the root and anchor angular velocities in their own
    frames. Leading dimensions are preserved, so the same derivation serves
    the current step (``[env, ...]``) and the history (``[env, slot, ...]``).
    Quantities the source did not measure stay absent, never None.
    """
    names: Dict[str, Any] = dict(state.tensors())
    names["anchor_idx"] = anchor_idx
    for suffix in ("pos", "rot", "vel", "ang_vel"):
        value = getattr(state, f"rigid_body_{suffix}")
        if value is not None:
            names[f"root_{suffix}"] = value[..., 0, :]
            names[f"anchor_{suffix}"] = value[..., anchor_idx, :]
    if state.rigid_body_pos is not None:
        names["root_height"] = names["root_pos"][..., 2]
    names["root_local_ang_vel"] = compute_local_ang_vel(
        names["root_rot"], names["root_ang_vel"]
    )
    names["anchor_local_ang_vel"] = compute_local_ang_vel(
        names["anchor_rot"], names["anchor_ang_vel"]
    )
    return names


# The raw ``BodyState`` field each derived name reads: the root and anchor
# slices come from the ``rigid_body_*`` field they slice, ``root_height`` from
# the positions, and the two local angular velocities from ``rigid_body_ang_vel``
# (they are the angular velocity re-expressed, not a new measurement). Every
# other name is its own field (``dof_pos``, ``rigid_body_rot``, ...), so a
# pass-through maps to itself.
_DERIVED_FROM = {
    "root_pos": "rigid_body_pos",
    "anchor_pos": "rigid_body_pos",
    "root_height": "rigid_body_pos",
    "root_rot": "rigid_body_rot",
    "anchor_rot": "rigid_body_rot",
    "root_vel": "rigid_body_vel",
    "anchor_vel": "rigid_body_vel",
    "root_ang_vel": "rigid_body_ang_vel",
    "anchor_ang_vel": "rigid_body_ang_vel",
    "root_local_ang_vel": "rigid_body_ang_vel",
    "anchor_local_ang_vel": "rigid_body_ang_vel",
}


def raw_field_of(leaf: str) -> str:
    """The raw ``BodyState`` field a ``body_state_names`` leaf derives from.

    A slice or a frame-change (``anchor_rot``, ``root_local_ang_vel``, ...)
    reports the field it was computed from; every other leaf is its own field.
    The deployment export reads this to find which sensor channel feeds a name.
    """
    return _DERIVED_FROM.get(leaf, leaf)


__all__ = ["body_state_names", "raw_field_of"]
