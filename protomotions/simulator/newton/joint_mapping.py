# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name-based builder DOF addressing, before ArticulationView exists."""

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    import newton


def builder_joint_mapping(
    builder: "newton.ModelBuilder", dof_names: "Sequence[str]"
) -> Tuple[Dict[str, Union[str, List[str]]], Dict[str, int]]:
    """Resolve scalar/compound MJCF joint labels without assuming common order.

    Newton combines consecutive scalar joints into a D6 joint whose label joins
    their names with underscores. Match complete names, not arbitrary substrings.
    Missing, duplicated or ambiguous names fail before physics construction.
    """
    remaining = set(dof_names)
    if len(remaining) != len(dof_names):
        raise ValueError("Duplicate common DOF names")
    groups, indices = {}, {}
    for joint, label in enumerate(builder.joint_label):
        short = label.rsplit("/", 1)[-1]
        count = sum(builder.joint_dof_dim[joint])
        if short in groups:
            raise ValueError(f"Duplicate Newton joint label {short!r}")
        tail, names = short, []
        while tail:
            # A scalar joint named hip_x is not ambiguous with another joint
            # named hip. Compound labels still require unambiguous boundaries.
            candidates = (
                [tail]
                if count == 1 and tail in remaining
                else [n for n in remaining if tail == n or tail.startswith(n + "_")]
            )
            if not candidates:
                break
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous Newton joint {label!r}: {candidates}")
            name = candidates[0]
            names.append(name)
            # Matching guarantees either the whole tail or a following "_".
            # Slicing also keeps this config-time helper usable on Python 3.8.
            tail = tail[len(name) + 1 :]
            remaining.remove(name)
        if not names:
            continue  # Free/fixed root and non-robot joints have no named action DOFs.
        if tail or len(names) != count:
            raise ValueError(
                f"Newton joint {label!r} does not match {count} named DOFs"
            )
        groups[short] = names if count > 1 else names[0]
        for axis, name in enumerate(names):
            indices[name] = builder.joint_qd_start[joint] + axis
    if remaining:
        raise ValueError(f"Missing Newton DOFs: {sorted(remaining)}")
    return groups, indices
