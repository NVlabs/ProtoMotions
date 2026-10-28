# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint rules shared by policy loaders."""

from copy import copy

from torch import nn
import torch

from protomotions.agents.utils.normalization import RunningMeanStd


IGNORE_IF_TARGET_ABSENT = ("_critic", "_discriminator", "_disc_critic")


def prepare_model_state_dict(model: nn.Module, state_dict: dict) -> dict:
    """Discard unused networks; retain only explicitly permitted new networks.

    The target model owns permission to initialize a wholly absent network.
    Partial saved networks are never filled in: normal validation must reject
    them. Actor weights are never optional. The input mapping is unchanged.
    """
    state = copy(state_dict)
    for name in IGNORE_IF_TARGET_ABSENT:
        prefix = name + "."
        if getattr(model, name, None) is None:
            for key in list(state):
                if key.startswith(prefix):
                    del state[key]

    optional_modules = getattr(model, "optional_checkpoint_modules", lambda: ())()
    for name in optional_modules:
        if name not in IGNORE_IF_TARGET_ABSENT:
            raise ValueError(f"Network {name!r} cannot be optional on checkpoint load")
        prefix = name + "."
        network = getattr(model, name, None)
        if network is not None and not any(key.startswith(prefix) for key in state):
            state.update(
                (prefix + key, value) for key, value in network.state_dict().items()
            )
    return state


def expected_checkpoint_state(model: nn.Module, state: dict) -> dict:
    """Describe expected tensors without creating live lazy normalization buffers."""
    expected = dict(model.state_dict())
    for name, module in model.named_modules():
        if isinstance(module, RunningMeanStd) and not module._initialized:
            prefix = name + "." if name else ""
            mean = state.get(prefix + "mean")
            if isinstance(mean, torch.Tensor):
                # Lazy statistics have no inferred observation width yet, but
                # mean/variance must agree and the count must remain scalar.
                shape = module.shape if module.shape is not None else mean.shape
                expected[prefix + "mean"] = torch.empty(shape, device="meta")
                expected[prefix + "var"] = torch.empty(shape, device="meta")
                expected[prefix + "count"] = torch.empty((), device="meta")
    return expected


def validate_checkpoint_state(expected, state, *, optional_prefixes=(), strict=True):
    """Preflight keys and shapes; do not copy any checkpoint tensors."""
    missing = sorted(
        key for key in expected
        if key not in state and not key.startswith(optional_prefixes)
    )
    unexpected = sorted(
        key for key in state
        if key not in expected and not key.startswith(optional_prefixes)
    )
    bad_shapes = []
    for key in expected.keys() & state.keys():
        value, target = state[key], expected[key]
        if not isinstance(value, torch.Tensor):
            bad_shapes.append(f"{key}: expected a tensor")
        elif not nn.parameter.is_lazy(target) and value.shape != target.shape:
            bad_shapes.append(f"{key}: {tuple(value.shape)} != {tuple(target.shape)}")
    if bad_shapes or (strict and (missing or unexpected)):
        raise RuntimeError(
            f"Checkpoint mismatch: missing_keys={missing}, unexpected_keys={unexpected}, "
            f"shape_mismatches={bad_shapes}"
        )
    return missing, unexpected
