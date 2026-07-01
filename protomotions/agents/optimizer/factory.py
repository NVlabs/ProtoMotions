# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer construction helpers."""

from torch import nn

from protomotions.utils.hydra_replacement import get_class, instantiate


def _optimizer_target(optimizer_config) -> str:
    if isinstance(optimizer_config, dict):
        return optimizer_config.get("_target_")
    return getattr(optimizer_config, "_target_", None)


def instantiate_optimizer(
    optimizer_config,
    module: nn.Module,
    params=None,
):
    """Instantiate an optimizer from config.

    Optimizers that declare ``accepts_module_params`` receive the owning module,
    so they can split trainable parameters using module and parameter names.
    Standard optimizers receive an explicit parameter iterable.
    """
    target = _optimizer_target(optimizer_config)
    optimizer_cls = get_class(target)
    if getattr(optimizer_cls, "accepts_module_params", False):
        optimizer_params = module
        optimizer_kwargs = {"params": optimizer_params}
        if params is not None:
            optimizer_kwargs["param_subset"] = list(params)
        return instantiate(optimizer_config, **optimizer_kwargs)
    elif params is not None:
        optimizer_params = list(params)
    else:
        optimizer_params = list(module.parameters())
    return instantiate(optimizer_config, params=optimizer_params)


def optimizer_learning_rate(optimizer_config, optimizer) -> float:
    """Return the learning rate PPO should track for adaptive scheduling."""
    if isinstance(optimizer_config, dict):
        if "lr" in optimizer_config:
            return optimizer_config["lr"]
    if hasattr(optimizer_config, "lr"):
        return optimizer_config.lr
    return optimizer.param_groups[0]["lr"]


def scale_optimizer_learning_rates(optimizer, old_lr: float, new_lr: float) -> None:
    """Scale optimizer param-group learning rates while preserving group ratios."""
    if old_lr == 0:
        raise ValueError("Cannot scale optimizer learning rates from old_lr=0.")
    if old_lr == new_lr:
        return

    lr_scale = new_lr / old_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] *= lr_scale
