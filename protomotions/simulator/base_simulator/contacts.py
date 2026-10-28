# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static axes for robot-body contact forces, independent of simulator ordering."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Dict, Sequence, Union

CONTACT_FORCE_THRESHOLD = 0.01  # N; common default for binary contact features.


@dataclass(frozen=True)
class ContactLayout:
    """Indices into privileged ``forces[env, body, partner, xyz]``.

    Rows select common robot body names. Columns select terrain, robot links,
    scene-object slots, and projectiles. Object slots follow SceneLib order;
    ``object/0`` can represent different assets in different environments.

    Each cell is sum(normal_force_magnitude * world_normal_on_row_body) over
    contact points for that pair, in N. This is a resultant force, not a unit
    contact normal. Normals within a pair can cancel. No friction or torque.
    """

    body_index: Dict[str, int]
    partner_index: Dict[str, int]
    common_body_ids: tuple[int, ...]
    is_complete: bool

    @classmethod
    def build(
        cls,
        body_names: Sequence[str],
        num_objects: int,
        num_projectiles: int,
        *,
        contact_bodies: Union[str, Sequence[str]] = "all",
        contact_partners: Union[str, Sequence[str]] = "all",
    ) -> "ContactLayout":
        """Resolve selections once, preserving canonical order on both axes.

        Selectors are exact names or glob patterns, e.g. ``robot/*`` and
        ``object/7``. Empty/unknown selectors fail instead of silently giving
        zeros for an unobserved contact. ``all`` selects the whole axis.
        """
        partners = (
            ["terrain"]
            + [f"robot/{name}" for name in body_names]
            + [f"object/{i}" for i in range(num_objects)]
            + [f"projectile/{i}" for i in range(num_projectiles)]
        )
        bodies = _select_names(body_names, contact_bodies, "body")
        selected_partners = _select_names(partners, contact_partners, "partner")
        return cls(
            body_index={name: i for i, name in enumerate(bodies)},
            partner_index={name: i for i, name in enumerate(selected_partners)},
            common_body_ids=tuple(body_names.index(name) for name in bodies),
            is_complete=len(bodies) == len(body_names)
            and len(selected_partners) == len(partners),
        )


def _select_names(
    available: Sequence[str], selectors: Union[str, Sequence[str]], axis: str
) -> list[str]:
    if selectors == "all":
        return list(available)
    patterns = [selectors] if isinstance(selectors, str) else selectors
    if not patterns:
        raise ValueError(f"Contact {axis} selection cannot be empty")
    selected = set()
    for pattern in patterns:
        matches = {name for name in available if fnmatchcase(name, pattern)}
        if not matches:
            raise ValueError(f"Contact {axis} selector {pattern!r} matches nothing")
        selected.update(matches)
    return [name for name in available if name in selected]
