# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure reductions of current, world-frame normal contact forces in newtons,
and the two checks a kernel makes on the names and the threshold it was bound
with."""

import math
from typing import List, Mapping, Optional, Sequence

from torch import Tensor

from protomotions.simulator.base_simulator.contacts import CONTACT_FORCE_THRESHOLD


def contact_rows(
    index: Mapping[str, int], names: Sequence[str], argument: str
) -> List[int]:
    """The rows (or columns) of the compact contact matrix ``names`` address.

    A kernel addresses the matrix by exact registered name through the
    published ``contact_body_index`` / ``contact_partner_index``. The build's
    required-keys check accepts a glob pattern, so a pattern written where the
    kernel reads exact names would pass that check and fail as a ``KeyError``
    at the first compute; it is refused here with ``argument`` named, as is a
    name the layout does not measure, an empty selection (a kernel over no rows
    would silently never fire or count zero) and a repeated name.
    """
    if not names:
        raise ValueError(f"{argument} must name at least one contact")
    repeated = sorted({name for name in names if list(names).count(name) > 1})
    if repeated:
        raise ValueError(f"{argument} must not repeat a name: {repeated}")
    rows = []
    for name in names:
        if "*" in name or "?" in name or "[" in name:
            raise ValueError(
                f"{argument} must be exact contact names, not patterns: {name!r}"
            )
        if name not in index:
            raise ValueError(
                f"{argument}: {name!r} is not a measured contact name "
                f"(measured: {sorted(index)})"
            )
        rows.append(index[name])
    return rows


def check_force_threshold(force_threshold: float) -> None:
    """Refuse a threshold that cannot be compared with a force: NaN, infinite
    or negative (every magnitude exceeds a negative one)."""
    if not math.isfinite(force_threshold) or force_threshold < 0:
        raise ValueError(
            f"force_threshold must be finite and nonnegative, got {force_threshold!r}"
        )


def contact_magnitudes(
    contact_forces: Tensor,
    body_ids: Optional[Tensor] = None,
    partner_ids: Optional[Tensor] = None,
) -> Tensor:
    """Largest pair magnitude per selected body, without inter-partner cancellation.

    Input is [env, body, partner, xyz]; output is [env, selected body], in N.
    None selects the whole corresponding axis. Collision points within each
    pair have already been summed by the simulator; friction is not included.
    """
    forces = contact_forces
    if body_ids is not None:
        forces = forces.index_select(1, body_ids)
    if partner_ids is not None:
        forces = forces.index_select(2, partner_ids)
    return forces.norm(dim=-1).amax(dim=-1)


def contact_flags(
    contact_forces: Tensor,
    body_ids: Optional[Tensor] = None,
    partner_ids: Optional[Tensor] = None,
    force_threshold: float = CONTACT_FORCE_THRESHOLD,
) -> Tensor:
    """Whether any selected partner exceeds the force threshold (N), per body."""
    return contact_magnitudes(contact_forces, body_ids, partner_ids) > force_threshold
