# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State-dict filtering for PEFT adapter checkpoints.

High-level actor and agent code should talk in terms of "save adapter" and
"load adapter". This module owns the low-level key rules that decide which
entries are adapter/task state and which entries belong to the frozen prior,
critic, or training-only scaffolding.
"""

from __future__ import annotations

import torch
from torch import nn

from protomotions.agents.utils.normalization import (
    materialize_lazy_running_stats_from_state_dict,
)


def strip_actor_prefix(key: str) -> str:
    """Accept both model-level keys and actor-local keys."""
    return key[len("_actor.") :] if key.startswith("_actor.") else key


def is_adapter_state_key(key: str) -> bool:
    """Return True for keys that belong in a slim PEFT checkpoint."""
    key = strip_actor_prefix(key)
    if key.startswith("prior_with_peft._anchor_transformer."):
        return False
    if key.startswith("prior_with_peft.reference_prior."):
        return False
    if key.startswith("prior_with_peft.reference_film_input_norm."):
        return False
    if key.startswith("actor_peft_model."):
        return True
    if key.startswith("prior_with_peft.film_input_norm."):
        return True
    return key.startswith("prior_with_peft.") and (
        ".lora." in key
        or ".gamma." in key
        or ".beta." in key
        or key.endswith(".m")
    )


def contains_adapter_state(state_dict: dict) -> bool:
    """Return True when a checkpoint contains at least one adapter key."""
    return any(is_adapter_state_key(key) for key in state_dict)


def build_adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Build an adapter-only state dict from a live actor module."""
    return {
        key: value
        for key, value in module.state_dict().items()
        if is_adapter_state_key(key)
    }


def load_adapter_state_dict(
    module: nn.Module,
    state_dict: dict,
    *,
    strict: bool = True,
):
    """Load adapter/task keys into ``module`` without touching frozen prior state."""
    actor_state = {strip_actor_prefix(key): value for key, value in state_dict.items()}
    materialize_lazy_running_stats_from_state_dict(module, actor_state)
    expected_keys = set(build_adapter_state_dict(module).keys())
    adapter_state = {}
    unexpected_keys = []

    for raw_key, value in state_dict.items():
        key = strip_actor_prefix(raw_key)
        if key in expected_keys:
            adapter_state[key] = value
        elif is_adapter_state_key(key):
            unexpected_keys.append(key)

    missing_keys = sorted(expected_keys - set(adapter_state.keys()))
    unexpected_keys = sorted(unexpected_keys)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Adapter state dict mismatch: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )

    merged_state = module.state_dict()
    merged_state.update(adapter_state)
    module.load_state_dict(merged_state, strict=True)
    return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
