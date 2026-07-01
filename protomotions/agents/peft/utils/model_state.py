# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-state loading rules for discrete-prior PEFT checkpoints.

The agent should not need to know why a missing key is acceptable. This module
owns those low-level checkpoint rules and exposes one readable entry point:
load a PEFT model state if it is adapter-only or a compatible full training
checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from protomotions.agents.peft.utils.adapter_state import is_adapter_state_key
from protomotions.agents.utils.normalization import (
    materialize_lazy_running_stats_from_state_dict,
)


REFERENCE_FULL_CHECKPOINT_PREFIXES = (
    "_actor.prior_with_peft.reference_prior.",
    "_actor.prior_with_peft.reference_film_input_norm.",
)


def has_reference_state(model_state: dict) -> bool:
    """Return True when a full PEFT checkpoint carries reference-policy state."""
    return any(
        key.startswith("_actor.prior_with_peft.reference_prior.")
        or key.startswith("prior_with_peft.reference_prior.")
        or key.startswith("_actor.prior_with_peft.reference_film_input_norm.")
        or key.startswith("prior_with_peft.reference_film_input_norm.")
        for key in model_state
    )


def load_compatible_peft_model_state(
    module: nn.Module,
    model_state: dict,
) -> None:
    """Load adapter-only state or a compatible full PEFT training checkpoint."""
    if is_adapter_only_model_state(model_state):
        module._actor.load_adapter_state_dict(model_state, strict=True)
        return

    reference_state_present = has_reference_state(model_state)
    if reference_state_present:
        module._actor.prior_with_peft.ensure_reference_modules()
    materialize_lazy_running_stats_from_state_dict(module, model_state)
    missing, unexpected = module.load_state_dict(model_state, strict=False)
    optional_prefixes = optional_full_checkpoint_state_prefixes(module)
    if reference_state_present:
        optional_prefixes = tuple(
            prefix
            for prefix in optional_prefixes
            if prefix not in REFERENCE_FULL_CHECKPOINT_PREFIXES
        )
    bad_missing = [key for key in missing if not key.startswith(optional_prefixes)]
    bad_unexpected = [
        key for key in unexpected if not key.startswith(optional_prefixes)
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Unexpected PEFT model state_dict mismatch: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
    if reference_state_present:
        module._actor.prior_with_peft.mark_reference_loaded()


def optional_full_checkpoint_state_prefixes(module: nn.Module) -> tuple[str, ...]:
    """Return state_dict prefixes that are optional for full PEFT checkpoints.

    Each module owns the optional state it introduces by defining
    ``optional_full_checkpoint_state_prefixes()`` with local state_dict
    prefixes. This helper qualifies those local prefixes by walking
    ``named_modules()``, keeping checkpoint compatibility rules close to the
    modules that create the state instead of centralizing private path strings.
    """
    prefixes: list[str] = []
    for module_name, submodule in module.named_modules():
        hook = getattr(submodule, "optional_full_checkpoint_state_prefixes", None)
        if not callable(hook):
            continue
        base = f"{module_name}." if module_name else ""
        for prefix in _as_prefix_tuple(hook()):
            prefixes.append(f"{base}{prefix}")
    return tuple(dict.fromkeys(prefixes))


def _as_prefix_tuple(prefixes: str | Iterable[str]) -> tuple[str, ...]:
    result = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    if not all(isinstance(prefix, str) for prefix in result):
        raise TypeError("optional checkpoint state prefixes must be strings.")
    return result


def is_adapter_only_model_state(model_state: dict) -> bool:
    """Return True when every checkpoint entry is adapter/task state."""
    return bool(model_state) and all(is_adapter_state_key(key) for key in model_state)
