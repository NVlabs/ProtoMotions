# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton bookkeeping for the partner-resolved contact force matrix.

The matrix is reduced directly from Newton's per-contact records
(``Contacts.rigid_contact_shape0/1``, ``rigid_contact_normal`` and the
requested ``force`` attribute, newtons in world axes): per-shape maps say
which flat row and which partner column a shape belongs to, and the reduction
projects each contact force onto its unit normal before summing. Plain
Python/Torch over model lists, so everything is unit-testable without a GPU.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def resolve_world_body_indices(
    body_labels: Sequence[str],
    body_world: Sequence[int],
    num_worlds: int,
    body_names: Sequence[str],
) -> List[List[int]]:
    """Newton body index of every common robot body, per world: ``[world][body]``.

    ``replicate`` keeps the same label in every world and distinguishes copies
    by ``body_world``, so a body is identified by (world, short label). The short
    label is the last path segment of ``body_label``. Each (world, name) must
    resolve exactly once; anything else is a model/asset mismatch.
    """
    found: Dict[Tuple[int, str], int] = {}
    for index, (label, world) in enumerate(zip(body_labels, body_world)):
        key = (int(world), label.rsplit("/", 1)[-1])
        if key[1] in body_names and key[0] >= 0:
            if key in found:
                raise ValueError(
                    f"Newton body '{key[1]}' appears twice in world {key[0]}"
                )
            found[key] = index
    missing = [
        (world, name)
        for world in range(num_worlds)
        for name in body_names
        if (world, name) not in found
    ]
    if missing:
        raise ValueError(f"Newton model lacks robot bodies {missing[:5]}")
    return [
        [found[(world, name)] for name in body_names] for world in range(num_worlds)
    ]


def resolve_world_projectile_indices(
    body_world: Sequence[int],
    num_worlds: int,
    robot_indices: List[List[int]],
    num_projectiles: int,
) -> List[List[int]]:
    """Projectile bodies per world, in creation order: ``[world][projectile]``.

    Projectiles are the only non-robot bodies the backend adds to a world, in
    ``ProjectileConfig`` order, so they are the remaining bodies of that world.
    """
    robot = {index for world in robot_indices for index in world}
    result = []
    for world in range(num_worlds):
        others = [
            index
            for index, owner in enumerate(body_world)
            if int(owner) == world and index not in robot
        ]
        if len(others) != num_projectiles:
            raise ValueError(
                f"World {world} has {len(others)} non-robot bodies, expected "
                f"{num_projectiles} projectiles"
            )
        result.append(others)
    return result


def shape_axes(
    shape_body: Sequence[int],
    shape_world: Sequence[int],
    shape_flags: Sequence[int],
    collide_flag: int,
    row_of_body: Dict[int, int],
    column_of_body: Dict[int, int],
    terrain_column: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-shape flat row and partner column, -1 where a shape takes no part.

    ``row_of_body`` maps a Newton body index to its flat ``env * rows + row``
    position for selected sensing bodies; ``column_of_body`` maps partner
    bodies to layout columns. Global static shapes (world -1, no body) are the
    terrain on this backend: the simulator adds exactly one ground plane, which
    gets the terrain column only when that partner is selected. Shapes without
    collision never report contacts and are left out on both axes.
    """
    rows, columns = [], []
    for body, world, flags in zip(shape_body, shape_world, shape_flags):
        body, world = int(body), int(world)
        collides = bool(int(flags) & collide_flag)
        ground = body < 0 and world < 0
        rows.append(row_of_body.get(body, -1) if collides else -1)
        if not collides:
            columns.append(-1)
        elif ground:
            columns.append(-1 if terrain_column is None else terrain_column)
        else:
            columns.append(column_of_body.get(body, -1))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(columns, dtype=torch.long)


def contact_force_matrix(
    count: torch.Tensor,
    shape0: torch.Tensor,
    shape1: torch.Tensor,
    force: torch.Tensor,
    normal: torch.Tensor,
    shape_row: torch.Tensor,
    shape_column: torch.Tensor,
    num_envs: int,
    num_rows: int,
    num_partners: int,
) -> torch.Tensor:
    """Sum per-contact normal forces into ``[envs, rows, partners, xyz]``.

    ``force[k]`` is the contact force on ``shape0[k]`` (its linear part; the
    opposite acts on ``shape1[k]``). Each contact contributes
    ``(f_k . n_k) n_k`` with a unit ``n_k``, so a cell is the vector sum of
    normal forces on the row body from that partner; friction is dropped and
    the sign of the stored normal does not matter. Slots beyond ``count`` and
    pairs where either side is not a selected row/partner are ignored. Both
    orderings are handled, so a selected body is a source whether Newton
    stored it as shape0 or shape1.
    """
    slots = torch.arange(shape0.shape[0], device=shape0.device)
    live = slots < count
    unit = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normal_force = (force[:, :3] * unit).sum(dim=-1, keepdim=True) * unit
    out = normal_force.new_zeros(num_envs * num_rows * num_partners, 3)
    for source, partner, sign in ((shape0, shape1, 1.0), (shape1, shape0, -1.0)):
        rows = shape_row[source.clamp_min(0)]
        columns = shape_column[partner.clamp_min(0)]
        valid = live & (source >= 0) & (partner >= 0) & (rows >= 0) & (columns >= 0)
        flat = rows[valid] * num_partners + columns[valid]
        out.index_add_(0, flat, sign * normal_force[valid])
    return out.view(num_envs, num_rows, num_partners, 3)
