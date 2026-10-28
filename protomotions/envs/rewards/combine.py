# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine raw reward terms into the single reward the agent receives.

This is the one place the reward-combining rule lives: weighting, the
multiplicative product, clamping and the reset grace period. Each term's
``RewardMeta`` (``protomotions.envs.rewards.meta``) decides how it enters.
``base_env/mdp_managers.py``'s ``RewardManager`` calls this; look here when a
reward does not behave as its metadata says.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.meta import RewardMeta


def combine_rewards(
    raw_rewards: Dict[str, Tensor],
    components: Mapping[str, MdpComponent],
    grace_mask: Optional[Tensor] = None,
    num_envs: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Combine the raw reward terms into the one reward the agent receives.

    Each term's :class:`~protomotions.envs.rewards.meta.RewardMeta` decides how
    it enters: a ``multiplicative`` term multiplies into a product that is added
    to the weighted sum of the others (an alive bonus, for example); otherwise
    the term is scaled by ``weight``, clamped to ``min_value`` / ``max_value``
    when given, and summed. A term with ``zero_during_grace_period`` is zeroed
    where ``grace_mask`` is True (the rows just reset). Returns the combined
    reward and a logging dict with the raw and scaled values of every term.

    ``components`` maps each name to its ``MdpComponent``; every reward
    component carries a ``RewardMeta`` on ``.meta``.
    """
    mult_reward = torch.ones(num_envs, device=device, dtype=torch.float)
    add_reward = torch.zeros(num_envs, device=device, dtype=torch.float)
    any_mult = False
    logging_dict: Dict[str, Tensor] = {}

    for name, reward in raw_rewards.items():
        meta = components[name].meta
        if not isinstance(meta, RewardMeta):
            raise TypeError(
                f"reward component '{name}' has no RewardMeta: pass "
                "meta=RewardMeta(weight=...) to from_kernel"
            )

        if meta.zero_during_grace_period and grace_mask is not None:
            reward = reward.clone()
            reward[grace_mask] = 0.0

        assert torch.all(torch.isfinite(reward)), f"Reward '{name}' not finite"
        logging_dict[f"raw_r/{name}"] = reward.clone()

        if meta.multiplicative:
            mult_reward *= reward
            any_mult = True
        else:
            scaled = reward * meta.weight
            if meta.min_value is not None:
                scaled = torch.clamp(scaled, min=meta.min_value)
            if meta.max_value is not None:
                scaled = torch.clamp(scaled, max=meta.max_value)
            logging_dict[f"scaled_r/{name}"] = scaled.clone()
            add_reward += scaled

    if any_mult:
        logging_dict["multiplicative_reward"] = mult_reward.clone()
        logging_dict["additive_reward"] = add_reward.clone()
        return add_reward + mult_reward, logging_dict

    return add_reward, logging_dict


__all__ = ["combine_rewards"]
