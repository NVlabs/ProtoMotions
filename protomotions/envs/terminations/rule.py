# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one termination entry type and the rules that combine the group.

Every termination is a :class:`TerminationRule` over a bound kernel:

* a boolean kernel (clip done, contact fall, joint limits) takes no threshold —
  its ``torch.bool`` output is the flag;
* a scalar kernel takes a float ``threshold`` and ``fail_above`` (the cut);
* ``truncation=True`` ends the episode without terminating (the agent still
  bootstraps through the boundary), for a check like the mimic clip end.

There is no boolean twin, no wrapper type and no second entry type. The
evaluator's failure is the same rule, bare or wrapped in a ``ThresholdFailure``
(``protomotions.agents.evaluators.components``) for a per-trial reduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from protomotions.envs.mdp_component import MdpComponent


@dataclass(frozen=True)
class TerminationRule:
    """A termination entry: a bound kernel, an optional cut, a truncation bit.

    Attributes:
        component: The bound kernel (``MdpComponent``). A boolean kernel for a
            thresholdless rule, a scalar kernel when ``threshold`` is given.
        threshold: The float cut for a scalar kernel; ``None`` for a boolean one.
        fail_above: With a threshold, fail when ``value > threshold`` (else
            ``value < threshold``).
        truncation: End the episode without marking it terminated (the agent
            bootstraps through the boundary).
    """

    component: MdpComponent
    threshold: Optional[float] = None
    fail_above: bool = True
    truncation: bool = False

    def __post_init__(self):
        if not isinstance(self.component, MdpComponent):
            raise TypeError(
                "TerminationRule.component must be an MdpComponent (bind the "
                f"kernel with from_kernel first), got {type(self.component).__name__}"
            )
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("TerminationRule.threshold must be finite or None")
        if not isinstance(self.fail_above, bool) or not isinstance(
            self.truncation, bool
        ):
            raise TypeError("TerminationRule.fail_above and truncation must be bool")

    def compare(self, value: Tensor, name: str = "") -> Tensor:
        """The boolean flag for this rule's kernel output.

        A thresholdless rule returns the boolean kernel's value (which must be
        ``torch.bool``); a thresholded rule cuts the scalar value. ``name`` is
        the entry name for the refusal's wording.
        """
        if self.threshold is None:
            if value.dtype != torch.bool:
                who = (
                    f"termination component {name!r}"
                    if name
                    else ("termination component")
                )
                raise ValueError(
                    f"{who} returned {value.dtype}; a boolean kernel returns "
                    "torch.bool, a scalar one needs threshold="
                )
            return value
        return value > self.threshold if self.fail_above else value < self.threshold


def components_of(entries: Mapping[str, TerminationRule]) -> Dict[str, MdpComponent]:
    """The bound kernels of a termination table, by name.

    Refuses a bare ``MdpComponent`` (not wrapped in a rule) and a
    ``ThresholdFailure`` (an evaluator entry) with a sentence pointing at the
    right shape.
    """
    out: Dict[str, MdpComponent] = {}
    for name, entry in entries.items():
        if isinstance(entry, TerminationRule):
            out[name] = entry.component
        elif isinstance(entry, MdpComponent):
            raise TypeError(
                f"termination '{name}' is a bare MdpComponent; wrap it: "
                "TerminationRule(component) for a boolean kernel, "
                "TerminationRule(component, threshold) for a scalar one"
            )
        elif type(entry).__name__ == "ThresholdFailure":
            raise TypeError(
                f"termination '{name}' is a ThresholdFailure; it carries a "
                "per-trial reduction and belongs in the evaluator's "
                "failure_components: pass its .rule here"
            )
        else:
            raise TypeError(
                f"termination '{name}' must be a TerminationRule, got "
                f"{type(entry).__name__}"
            )
    return out


def evaluate_terminations(
    manager, entries: Mapping[str, TerminationRule], ctx
) -> Dict[str, Tensor]:
    """Run each rule's kernel over ``ctx`` and reduce to its boolean flag.

    The kernels run under their entry names, so the component manager's compile
    cache is unchanged from running them as bare components.
    """
    values = manager.execute_all(components_of(entries), ctx)
    return {name: entries[name].compare(values[name], name) for name in entries}


def or_terminations(
    flags: Mapping[str, Tensor],
    entries: Mapping[str, TerminationRule],
    num_envs: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """OR the per-rule flags into the reset and terminated masks, plus logging.

    Every flag resets the env; a flag whose rule is ``truncation=True`` resets
    without marking the env terminated (the agent bootstraps through it). One
    ``termination/<name>`` float flag is logged per entry.
    """
    reset = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    logging: Dict[str, Tensor] = {}
    for name, flag in flags.items():
        reset = reset | flag
        if not entries[name].truncation:
            terminated = terminated | flag
        logging[f"termination/{name}"] = flag.float().clone()
    return reset, terminated, logging


__all__ = [
    "TerminationRule",
    "components_of",
    "evaluate_terminations",
    "or_terminations",
]
