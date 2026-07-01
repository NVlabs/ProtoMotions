# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading frozen pretrained modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from protomotions.agents.utils.normalization import (
    materialize_lazy_running_stats_from_state_dict,
)
from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint
from protomotions.utils.hydra_replacement import get_class


def freeze_module(module: nn.Module) -> None:
    """Freeze a module in eval mode."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_path(root: Any, path: str) -> Any:
    """Resolve a dotted attribute/key path.

    Module paths may use public names such as ``actor`` even when the model
    stores the module as ``_actor``. An empty path returns ``root`` unchanged,
    which lets callers request the whole loaded model.
    """
    if not path:
        return root

    current = root
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple, nn.ModuleList, nn.Sequential)) and part.isdigit():
            current = current[int(part)]
        elif hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, f"_{part}"):
            current = getattr(current, f"_{part}")
        else:
            raise AttributeError(f"Could not resolve '{path}' at '{part}'")
    return current


def _set_dotted_attr(root: Any, path: str, value: Any) -> None:
    """Assign ``value`` at a dotted config path below ``root``.

    This is used for checkpoint-path overrides on loaded resolved configs.
    Intermediate path segments can walk through dictionaries or object
    attributes; the final segment is overwritten in-place. Empty paths are
    rejected so callers cannot accidentally replace the whole config object.
    """
    if not path:
        raise ValueError("checkpoint_path override cannot use an empty path")

    parts = path.split(".")

    current = root
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)

    if isinstance(current, dict):
        current[parts[-1]] = value
    else:
        setattr(current, parts[-1], value)


def load_pretrained_model_module(
    config,
    device: torch.device,
    checkpoint_path_overrides: Mapping[str, str] | None = None,
    prefer_inference_config: bool = False,
) -> nn.Module:
    """Instantiate a checkpoint-backed or embedded frozen module."""
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
    if checkpoint_path is None or not checkpoint_path.exists():
        module_config = getattr(config, "module_config", None)
        if module_config is not None:
            module_cls = get_class(module_config._target_)
            module = module_cls(config=module_config).to(device)
            module.eval()
            if config.freeze:
                freeze_module(module)
            return module

        raise FileNotFoundError(
            f"Pretrained model checkpoint not found: {checkpoint_path}"
        )

    resolved_configs = load_resolved_configs_from_checkpoint(
        checkpoint_path,
        prefer_inference=prefer_inference_config,
    )
    model_config = get_path(resolved_configs, config.config_path)
    for path, value in (checkpoint_path_overrides or {}).items():
        _set_dotted_attr(model_config, path, value)

    model_cls = get_class(model_config._target_)
    model = model_cls(config=model_config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint[config.state_dict_key]
    model.materialize_from_state_dict(state_dict)
    materialize_lazy_running_stats_from_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=config.strict)

    model.eval()

    module = get_path(model, config.module_path)
    if config.freeze:
        freeze_module(module)
    return module
