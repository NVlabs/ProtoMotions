# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bodies a kernel reads, named by the recipe and resolved to indices once.

A recipe spells the bodies a kernel takes as a :class:`BodySelection`: by name
(``BodySelection.named("left_hand", "right_hand")``), by semantic group
(``BodySelection.group("hands", "feet")``, through the robot's
``common_naming_to_robot_body_names``) or by a regular expression over the body
names (``BodySelection.matching(r".*_ankle_.*")``). The selection is a static
kernel parameter and stays names in the saved configuration; ``build_env``
calls :func:`resolve_body_selections` on each component table once, which
copies every component holding a selection with the ``LongTensor`` of indices
in its place. The configs are never mutated, and ``ComponentManager`` refuses
a selection that reached it unresolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from protomotions.envs.mdp_component import MdpComponent


@dataclass(frozen=True)
class BodySelection:
    """Bodies by name, by semantic group or by pattern; exactly one kind.

    Attributes:
        names: Body names, kept in the order written.
        groups: Semantic groups (``RobotConfig.body_indices`` resolves them,
            ``'root'`` and ``'all'`` included), in the order written.
        pattern: A regular expression every selected body name fully matches,
            in body order.
    """

    names: Tuple[str, ...] = ()
    groups: Tuple[str, ...] = ()
    pattern: Optional[str] = None

    def __post_init__(self):
        kinds = sum((bool(self.names), bool(self.groups), self.pattern is not None))
        if kinds != 1:
            raise ValueError(
                "BodySelection takes exactly one of names, groups or pattern"
            )
        if not all(isinstance(name, str) for name in (*self.names, *self.groups)):
            raise TypeError("BodySelection names and groups are strings")
        if self.pattern is not None and not isinstance(self.pattern, str):
            raise TypeError("BodySelection.pattern is a regular expression string")

    @classmethod
    def named(cls, *names: str) -> "BodySelection":
        """The bodies with these names, in this order."""
        return cls(names=tuple(names))

    @classmethod
    def group(cls, *semantic_groups: str) -> "BodySelection":
        """The bodies of these semantic groups, in this order."""
        return cls(groups=tuple(semantic_groups))

    @classmethod
    def matching(cls, regex: str) -> "BodySelection":
        """The bodies whose name fully matches ``regex``, in body order."""
        return cls(pattern=regex)

    def resolve(self, robot_config) -> List[int]:
        """The body indices this selection names on ``robot_config``."""
        body_names = list(robot_config.kinematic_info.body_names)
        robot = type(robot_config).__name__
        if self.names:
            indices = []
            for name in self.names:
                if name not in body_names:
                    raise ValueError(
                        f"BodySelection: '{name}' is not a body of {robot} "
                        f"(bodies: {body_names})"
                    )
                indices.append(body_names.index(name))
            return indices
        if self.groups:
            return list(robot_config.body_indices(*self.groups))
        indices = [
            index
            for index, name in enumerate(body_names)
            if re.fullmatch(self.pattern, name)
        ]
        if not indices:
            raise ValueError(
                f"BodySelection: pattern '{self.pattern}' matches no body of {robot}"
            )
        return indices


def unresolved_selections(component: MdpComponent) -> Dict[str, BodySelection]:
    """The static parameters of ``component`` that are still a BodySelection."""
    return {
        name: value
        for name, value in component.static_params.items()
        if isinstance(value, BodySelection)
    }


def resolve_body_selections(
    components: Mapping[str, MdpComponent], robot_config
) -> Mapping[str, MdpComponent]:
    """``components`` with every BodySelection replaced by its index tensor.

    A component holding a selection is copied with ``torch.tensor(indices,
    dtype=torch.long)`` in the selection's place; one holding none is the same
    object, and a table in which none holds one is the same mapping. The input
    mapping and its components are not mutated, so a saved configuration keeps
    the names.
    """
    resolved: Dict[str, MdpComponent] = {}
    changed = False
    for name, component in components.items():
        selections = unresolved_selections(component)
        if not selections:
            resolved[name] = component
            continue
        changed = True
        static_params = dict(component.static_params)
        for parameter, selection in selections.items():
            static_params[parameter] = torch.tensor(
                selection.resolve(robot_config), dtype=torch.long
            )
        resolved[name] = MdpComponent(
            component.compute_func,
            dict(component.dynamic_vars),
            static_params,
            meta=component.meta,
            required_keys=component.required_keys,
        )
    return resolved if changed else components


def resolve_rule_selections(entries: Mapping[str, object], robot_config) -> Mapping:
    """:func:`resolve_body_selections` for a table of ``TerminationRule`` entries.

    Each rule is rebuilt around its resolved component; a rule whose component
    holds no selection is the same object, and a table with nothing to resolve
    is the same mapping.
    """
    components = {name: entry.component for name, entry in entries.items()}
    resolved = resolve_body_selections(components, robot_config)
    if resolved is components:
        return entries
    return {
        name: (
            entry
            if resolved[name] is entry.component
            else replace(entry, component=resolved[name])
        )
        for name, entry in entries.items()
    }


__all__ = [
    "BodySelection",
    "resolve_body_selections",
    "resolve_rule_selections",
    "unresolved_selections",
]
