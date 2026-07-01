# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEFT-side loading/inspection of the frozen base-prior checkpoint.

These helpers intentionally reach into the discrete latent prior checkpoint
layout so the PEFT agent can keep its control flow readable. If more agents
need this behavior, move the layout-specific pieces onto the prior model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def reload_frozen_base_prior_from_checkpoint(peft, checkpoint_path: str | Path, device):
    """Copy configured prior weights into an already-built PEFT wrapper."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Prior checkpoint not found: {checkpoint_path}")

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    prior_state = (
        state["model"] if isinstance(state, dict) and "model" in state else state
    )
    base_prior = peft.base_prior

    _reload_projection_modules(base_prior, prior_state)
    transformer_state = _extract_transformer_state(prior_state)
    _reload_positional_embedding(peft, prior_state, device)
    # PEFT wraps each frozen transformer layer, so checkpoint weights are loaded
    # into the wrapped inner layer instead of replacing the adapter wrapper.
    _reload_wrapped_transformer_layers(base_prior, transformer_state)

    log.info("Reloaded frozen base prior from %s", checkpoint_path)


def _reload_projection_modules(base_prior, prior_state: dict) -> None:
    for attr in ("_context_encoder", "_token_encoder", "_output_head"):
        module = getattr(base_prior, attr, None)
        if module is None:
            continue
        prefix = f"prior.{attr}."
        module_state = _strip_prefixed_state(prior_state, prefix)
        if module_state:
            module.load_state_dict(module_state, strict=False)


def _extract_transformer_state(prior_state: dict) -> dict:
    return _strip_prefixed_state(prior_state, "prior._transformer.")


def _reload_positional_embedding(peft, prior_state: dict, device) -> None:
    pos_emb = prior_state.get("prior._pos_emb")
    if pos_emb is None:
        return
    pos_emb = pos_emb.to(device)
    peft.base_prior._pos_emb.data.copy_(pos_emb)


def _reload_wrapped_transformer_layers(base_prior, transformer_state: dict) -> None:
    if not transformer_state:
        return

    patched_layers = 0
    for index, layer in enumerate(base_prior._transformer.layers):
        wrapped_layer = getattr(layer, "transformer_layer", None)
        if wrapped_layer is None:
            continue
        layer_state = _strip_prefixed_state(transformer_state, f"layers.{index}.")
        if layer_state:
            wrapped_layer.load_state_dict(layer_state, strict=True)
            patched_layers += 1

    if patched_layers == 0:
        raise RuntimeError(
            "No PEFT transformer layers exposed 'transformer_layer'; "
            "could not reload frozen base prior transformer weights."
        )


def _strip_prefixed_state(state_dict: dict, prefix: str) -> dict:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
