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
from dataclasses import dataclass

from torch import nn

from protomotions.agents.peft.utils.adapter_state import (
    REFERENCE_STATE_PREFIXES,
    STUDENT_TO_REFERENCE_PREFIX,
    is_adapter_checkpoint_key,
    refuse_reference_without_conditioning,
    strip_actor_prefix,
)
from protomotions.agents.utils.checkpoint import (
    prepare_model_state_dict, expected_checkpoint_state, validate_checkpoint_state,
)
from protomotions.agents.utils.normalization import (
    materialize_lazy_running_stats_from_state_dict,
)


REFERENCE_FULL_CHECKPOINT_PREFIXES = tuple(
    "_actor." + prefix for prefix in REFERENCE_STATE_PREFIXES
)


def has_reference_state(model_state: dict) -> bool:
    """Return True when a PEFT checkpoint carries any reference-policy state."""
    return any(
        strip_actor_prefix(key).startswith(REFERENCE_STATE_PREFIXES)
        for key in model_state
    )


@dataclass(frozen=True)
class PEFTModelStateLoad:
    """What ``load_compatible_peft_model_state`` found in a checkpoint.

    Attributes:
        adapter_only: True for an adapter checkpoint without frozen prior weights.
        reference_loaded: True when the checkpoint restored the reference policy.
    """

    adapter_only: bool
    reference_loaded: bool


def load_compatible_peft_model_state(
    module: nn.Module,
    model_state: dict,
) -> PEFTModelStateLoad:
    """Load adapter-only state or a compatible full PEFT training checkpoint.

    Either kind restores the reference policy when the checkpoint stores one.
    A reference saved before it kept its own conditioning network is refused
    (``scripts/migrate_peft_reference_conditioning.py`` writes a migrated
    file). Resume and inference use the restored reference; a new RLFT run
    replaces it at training start.
    """
    if is_adapter_only_model_state(model_state):
        result = module._actor.load_adapter_state_dict(model_state, strict=True)
        return PEFTModelStateLoad(
            adapter_only=True,
            reference_loaded=result["reference_loaded"],
        )

    model_state = prepare_model_state_dict(module, model_state)
    refuse_reference_without_conditioning(model_state)
    reference_state_present = has_reference_state(model_state)
    expected = expected_checkpoint_state(module, model_state)
    optional_prefixes = optional_full_checkpoint_state_prefixes(module)
    if reference_state_present:
        # The reference has the student's architecture. Derive its expected
        # keys/shapes without allocating or changing a live reference module.
        for student_prefix, reference_prefix in STUDENT_TO_REFERENCE_PREFIX.items():
            source, target = "_actor." + student_prefix, "_actor." + reference_prefix
            expected.update({target + key[len(source):]: value
                             for key, value in list(expected.items())
                             if key.startswith(source)})
        optional_prefixes = tuple(
            prefix
            for prefix in optional_prefixes
            if prefix not in REFERENCE_FULL_CHECKPOINT_PREFIXES
        )
    validate_checkpoint_state(expected, model_state, optional_prefixes=optional_prefixes)
    if reference_state_present:
        module._actor.ensure_reference_modules()
    materialize_lazy_running_stats_from_state_dict(module, model_state)
    module.load_state_dict(model_state, strict=False)
    if reference_state_present:
        module._actor.mark_reference_loaded()
    return PEFTModelStateLoad(
        adapter_only=False,
        reference_loaded=reference_state_present,
    )


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
    """Return True for an adapter checkpoint: adapter/task state and reference state only."""
    return bool(model_state) and all(is_adapter_checkpoint_key(key) for key in model_state)
