# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for the frozen prior consumed by discrete-prior PEFT."""

from __future__ import annotations

from torch import nn

from protomotions.agents.common.autoregressive import (
    DiscreteAutoregressiveTransformer,
)


def require_frozen_prior_attr(
    pretrained_prior_model: nn.Module,
    attr: str,
    expected_type: type,
):
    """Return a required attribute from the whole loaded prior model."""
    if not hasattr(pretrained_prior_model, attr):
        raise AttributeError(
            f"DiscretePriorPEFTActor expected the loaded prior model to expose '{attr}'. "
            f"Got {type(pretrained_prior_model).__name__}. Make sure "
            "'pretrained_modules[\"prior\"]' points at the whole "
            "DiscreteAutoregressiveLatentPriorModel, not an old actor.mu "
            "submodule."
        )
    value = getattr(pretrained_prior_model, attr)
    if not isinstance(value, expected_type):
        raise TypeError(
            f"DiscretePriorPEFTActor expected '{attr}' to be a "
            f"{expected_type.__name__}, got {type(value).__name__}."
        )
    return value


def resolve_frozen_prior_input_keys(pretrained_prior_model: nn.Module) -> list[str]:
    """Read context input keys from the loaded frozen prior transformer."""
    prior_transformer = require_frozen_prior_attr(
        pretrained_prior_model,
        "prior",
        DiscreteAutoregressiveTransformer,
    )
    input_keys = list(prior_transformer.context_in_keys)
    if not input_keys:
        raise ValueError(
            "DiscretePriorPEFTActor requires the loaded prior transformer to expose "
            "at least one context input key."
        )
    return input_keys
