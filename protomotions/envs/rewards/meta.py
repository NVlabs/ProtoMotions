# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward metadata: how a reward term enters the combined reward.

A reward ``MdpComponent`` carries one ``RewardMeta`` (passed as ``meta=`` to
``MdpComponent.from_kernel``). The kernel computes a raw per-env value; the
metadata says how that value is weighted, clamped and summed by
``protomotions.envs.rewards.combine.combine_rewards``. Only reward components
carry a ``RewardMeta``; an observation, termination or evaluation component
never does.

Two mutually exclusive shapes:

* **additive** (the default): the raw value is scaled by ``weight`` (required,
  finite and nonzero), optionally clamped to ``[min_value, max_value]``, and
  added to the reward sum.
* **multiplicative**: the raw value multiplies into a product (an alive bonus,
  for example) that is added to the additive sum. A multiplicative term takes
  no ``weight`` and no clamp.

``zero_during_grace_period`` zeroes the term on the rows within the reset grace
period, for rewards that are unreliable right after a reset (power, contacts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RewardMeta:
    """How one reward term is weighted, clamped and combined.

    Attributes:
        weight: Scale applied to the raw additive term. Required, finite and
            nonzero for an additive term; must be ``None`` for a multiplicative
            one.
        multiplicative: If True, the term multiplies into the reward product
            instead of being weighted and added.
        min_value: Lower clamp on the scaled additive term (after weighting).
        max_value: Upper clamp on the scaled additive term (after weighting).
        zero_during_grace_period: If True, the term is zeroed on rows within the
            reset grace period.
    """

    weight: Optional[float] = None
    multiplicative: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    zero_during_grace_period: bool = False

    def __post_init__(self) -> None:
        if self.multiplicative:
            if (
                self.weight is not None
                or self.min_value is not None
                or self.max_value is not None
            ):
                raise ValueError("a multiplicative RewardMeta takes no weight or clamp")
            return
        if self.weight is None or not math.isfinite(self.weight) or self.weight == 0:
            raise ValueError(
                "RewardMeta.weight must be a finite nonzero float; delete the "
                "component instead of weighting it 0"
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and not self.min_value < self.max_value
        ):
            raise ValueError("RewardMeta min_value must be less than max_value")


__all__ = ["RewardMeta"]
