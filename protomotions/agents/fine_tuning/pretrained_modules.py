# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixin for agents that load frozen modules before model construction."""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch import nn

from protomotions.agents.common.pretrained import load_pretrained_model_module

log = logging.getLogger(__name__)


class PretrainedModulesMixin:
    """Load configured frozen modules through the BaseAgent setup lifecycle.

    The mixin is intentionally independent of the training algorithm. PPO-style
    fine-tuning and supervised fine-tuning can both populate ``self.pretrained``
    before ``create_model()`` and then drop that temporary handle after the model
    captures the modules it needs.
    """

    pretrained: Dict[str, nn.Module]

    def _before_create_model(self) -> None:
        super()._before_create_model()
        self.pretrained = self._load_pretrained_modules()

    def _after_model_reset(self) -> None:
        self._post_create_model_hook()
        self.pretrained = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super()._after_model_reset()

    def _after_create_optimizers(self) -> None:
        self._print_param_info()
        super()._after_create_optimizers()

    def _pretrained_module_configs(self):
        """Return the config mapping consumed by ``_load_pretrained_modules``."""
        return self.config.pretrained_modules

    def _pretrained_module_load_kwargs(self, _name, _pretrained_config) -> dict:
        """Return per-module keyword arguments for ``load_pretrained_model_module``."""
        return {}

    def _load_pretrained_modules(self) -> Dict[str, nn.Module]:
        configs = self._pretrained_module_configs()
        modules: Dict[str, nn.Module] = {}
        for name, pretrained_config in configs.items():
            log.info(
                "Loading pretrained module '%s' from %s (module_path=%s)",
                name,
                pretrained_config.checkpoint_path or "<embedded module_config>",
                pretrained_config.module_path,
            )
            modules[name] = load_pretrained_model_module(
                pretrained_config,
                device=self.fabric.device,
                **self._pretrained_module_load_kwargs(name, pretrained_config),
            )
        return modules

    def _post_create_model_hook(self) -> None:
        """Hook called after create_model(), device transfer, and model reset."""

    def _print_param_info(self) -> None:
        """Optional hook for trainable/frozen parameter diagnostics."""
