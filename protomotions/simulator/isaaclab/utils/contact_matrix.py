# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit PhysX contact paths, in environment/body/partner order."""

from typing import Dict, List, Sequence, Tuple

from protomotions.simulator.base_simulator.contacts import ContactLayout


def contact_view_patterns(
    layout: ContactLayout,
    body_prim_paths: Dict[str, str],
    env_prim_paths: Sequence[str],
) -> Tuple[List[str], List[List[str]]]:
    """One exact source and partner list per robot link, with no wildcard matches.

    PhysX's many-to-many API accepts one filter list per source path. Exact
    paths avoid matching both a nested rigid link and its same-named visual
    geometry. Each source is paired only with partners in its own environment.
    """
    sources, filters = [], []
    for env_path in env_prim_paths:
        bodies = {
            name: f"{env_path}/Robot/{body_prim_paths[name]}"
            for name in layout.body_index
        }
        partners = {"terrain": "/World/ground/terrain/mesh"}
        for name in layout.partner_index:
            category, _, slot = name.partition("/")
            if category == "robot":
                partners[name] = f"{env_path}/Robot/{body_prim_paths[slot]}"
            elif category == "object":
                partners[name] = f"{env_path}/Object_{slot}"
            elif category == "projectile":
                partners[name] = f"{env_path}/Projectile_{slot}"
        ordered_partners = [partners[name] for name in layout.partner_index]
        sources.extend(bodies.values())
        filters.extend([ordered_partners] * len(bodies))
    return sources, filters
