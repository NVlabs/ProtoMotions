# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State-dict filtering for PEFT adapter checkpoints.

High-level actor and agent code should talk in terms of "save adapter" and
"load adapter". This module owns the low-level key rules that decide which
entries are adapter/task state and which entries belong to the frozen prior,
critic, or training-only scaffolding.

An adapter checkpoint holds everything that cannot be rebuilt from the frozen prior
checkpoint: the active (student) adapter and task state, plus the state of the
RLFT reference policy. The reference policy is a frozen copy of the student
networks: the adapter-wrapped prior, the FiLM input normalizer and the
conditioning network. Its frozen transformer weights are rebuilt from the
configured prior instead of stored.
"""

from __future__ import annotations

import torch
from torch import nn
from protomotions.agents.utils.checkpoint import (
    expected_checkpoint_state, validate_checkpoint_state,
)

from protomotions.agents.utils.normalization import (
    materialize_lazy_running_stats_from_state_dict,
)

# Writes a migrated copy of a file saved before the reference kept its own
# conditioning network; the loaders refuse such a file and name it.
MIGRATION_SCRIPT = "scripts/migrate_peft_reference_conditioning.py"

REFERENCE_PRIOR_PREFIX = "prior_with_peft.reference_prior."
REFERENCE_FILM_INPUT_NORM_PREFIX = "prior_with_peft.reference_film_input_norm."
REFERENCE_PEFT_MODEL_PREFIX = "reference_peft_model."
STUDENT_PRIOR_PREFIX = "prior_with_peft.base_prior."
STUDENT_FILM_INPUT_NORM_PREFIX = "prior_with_peft.film_input_norm."
STUDENT_PEFT_MODEL_PREFIX = "actor_peft_model."
# Each reference network is a frozen copy of a student network, stored under
# the student's keys with this prefix swapped in.
STUDENT_TO_REFERENCE_PREFIX = {
    STUDENT_PRIOR_PREFIX: REFERENCE_PRIOR_PREFIX,
    STUDENT_FILM_INPUT_NORM_PREFIX: REFERENCE_FILM_INPUT_NORM_PREFIX,
    STUDENT_PEFT_MODEL_PREFIX: REFERENCE_PEFT_MODEL_PREFIX,
}
REFERENCE_STATE_PREFIXES = tuple(STUDENT_TO_REFERENCE_PREFIX.values())


def strip_actor_prefix(key: str) -> str:
    """Accept both model-level keys and actor-local keys."""
    return key[len("_actor.") :] if key.startswith("_actor.") else key


def _is_adapter_parameter(key: str) -> bool:
    return ".lora." in key or ".gamma." in key or ".beta." in key or key.endswith(".m")


def is_adapter_state_key(key: str) -> bool:
    """Return True for the active adapter and task keys of an adapter PEFT checkpoint."""
    key = strip_actor_prefix(key)
    if key.startswith("prior_with_peft._anchor_transformer."):
        return False
    if key.startswith(REFERENCE_STATE_PREFIXES):
        return False
    if key.startswith(STUDENT_PEFT_MODEL_PREFIX):
        return True
    if key.startswith(STUDENT_FILM_INPUT_NORM_PREFIX):
        return True
    return key.startswith("prior_with_peft.") and _is_adapter_parameter(key)


def is_reference_state_key(key: str) -> bool:
    """Return True for the reference-policy keys an adapter PEFT checkpoint stores.

    These are the reference prior's adapter tensors, the reference's FiLM input
    normalizer and its conditioning network. The reference prior's frozen
    weights are not included because they equal the configured frozen prior.
    """
    key = strip_actor_prefix(key)
    if key.startswith((REFERENCE_FILM_INPUT_NORM_PREFIX, REFERENCE_PEFT_MODEL_PREFIX)):
        return True
    return key.startswith(REFERENCE_PRIOR_PREFIX) and _is_adapter_parameter(key)


def _student_key_to_reference_key(key: str) -> str:
    for student_prefix, reference_prefix in STUDENT_TO_REFERENCE_PREFIX.items():
        if key.startswith(student_prefix):
            return reference_prefix + key[len(student_prefix) :]
    raise KeyError(f"The reference policy has no copy of {key!r}.")


def reference_state_from_policy_state(state_dict: dict) -> dict:
    """Rename the trained (student) policy saved in a checkpoint to reference keys.

    Accepts full and adapter checkpoints, with or without the ``_actor.``
    prefix. The checkpoint's own reference, its frozen prior weights and any
    critic or discriminator state are left out.
    """
    student_prefixes = tuple(STUDENT_TO_REFERENCE_PREFIX)
    reference_state = {}
    for key, value in state_dict.items():
        key = strip_actor_prefix(key)
        if is_adapter_state_key(key) and key.startswith(student_prefixes):
            reference_state[_student_key_to_reference_key(key)] = value
    return reference_state


def reference_lacks_conditioning_copy(state_dict: dict) -> bool:
    """True for a file saved before the reference kept its own conditioning
    network: it stores reference state and a student conditioning network, but
    no ``reference_peft_model.*`` keys. A partial copy is a key mismatch that
    validation reports, not such a file; a conditioning network that saves no
    state leaves nothing to copy, so a file without student network keys is not
    one either.
    """
    local_keys = [strip_actor_prefix(key) for key in state_dict]
    has_reference = any(key.startswith(REFERENCE_STATE_PREFIXES) for key in local_keys)
    has_copy = any(key.startswith(REFERENCE_PEFT_MODEL_PREFIX) for key in local_keys)
    has_student_network = any(
        key.startswith(STUDENT_PEFT_MODEL_PREFIX) for key in local_keys
    )
    return has_reference and has_student_network and not has_copy


class ReferenceWithoutConditioningError(RuntimeError):
    """The refusal of a file saved before the reference kept its own conditioning
    network; the load entry points that know the file's path re-raise it with
    the path in front."""


def refuse_reference_without_conditioning(state_dict: dict) -> None:
    """Refuse a file saved before the reference kept its own conditioning network.

    Such a reference read the student's live conditioning network, so the
    student network saved in the same file is the one it read at save time;
    :data:`MIGRATION_SCRIPT` copies it under the reference prefix into a new
    file. Nothing fills the copy in on load.
    """
    if reference_lacks_conditioning_copy(state_dict):
        raise ReferenceWithoutConditioningError(
            "This checkpoint's reference policy has no conditioning network of "
            f"its own (no '{REFERENCE_PEFT_MODEL_PREFIX}*' keys beside its other "
            "reference state): it was saved before references kept one, when the "
            "reference read the student's live conditioning network. Write a "
            f"migrated file with `python {MIGRATION_SCRIPT} <checkpoint> <out>`, "
            "which copies the student network saved in this file under the "
            "reference prefix, and load that file."
        )


def is_adapter_checkpoint_key(key: str) -> bool:
    """Return True for any key that belongs in an adapter PEFT checkpoint."""
    return is_adapter_state_key(key) or is_reference_state_key(key)


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


def build_reference_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Build the adapter reference-policy state from a live actor module.

    The result is empty when no reference policy has been built yet.
    """
    return {
        key: value
        for key, value in module.state_dict().items()
        if is_reference_state_key(key)
    }


def validate_adapter_checkpoint(module: nn.Module, state_dict: dict, *, strict=True):
    """Validate student and reference together, before either network is changed."""
    state = {strip_actor_prefix(key): value for key, value in state_dict.items()}
    expected = expected_checkpoint_state(module, state)
    student = {key: value for key, value in state.items() if is_adapter_state_key(key)}
    expected_student = {
        key: value for key, value in expected.items() if is_adapter_state_key(key)
    }
    validate_checkpoint_state(expected_student, student, strict=strict)
    reference = {key: value for key, value in state.items() if is_reference_state_key(key)}
    if reference:
        expected_reference = {
            _student_key_to_reference_key(key): value
            for key, value in expected_student.items()
        }
        validate_checkpoint_state(expected_reference, reference)


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


def load_reference_state_dict(module: nn.Module, state_dict: dict) -> bool:
    """Restore the reference policy stored in an adapter checkpoint.

    Returns False, and changes nothing, when ``state_dict`` stores no
    reference. Otherwise the reference prior is rebuilt from the module's
    current frozen prior, then its adapter tensors, FiLM input normalizer and
    conditioning network are overwritten with the stored ones, so it matches
    the saved reference exactly.
    """
    reference_state = {
        strip_actor_prefix(key): value
        for key, value in state_dict.items()
        if is_reference_state_key(key)
    }
    if not reference_state:
        return False

    # The reference mirrors the student key for key. Check names and shapes
    # before touching the module, so a rejected file leaves it unchanged.
    expected = {
        _student_key_to_reference_key(key): value
        for key, value in build_adapter_state_dict(module).items()
    }
    validate_checkpoint_state(expected, reference_state)

    module.ensure_reference_modules()
    # Start from the student's state so the reference's frozen weights equal
    # the configured prior, whatever an earlier reference held.
    peft = module.prior_with_peft
    peft.reference_prior.load_state_dict(peft.base_prior.state_dict())
    materialize_lazy_running_stats_from_state_dict(module, reference_state)

    merged_state = module.state_dict()
    merged_state.update(reference_state)
    module.load_state_dict(merged_state, strict=True)
    module.mark_reference_loaded()
    return True
