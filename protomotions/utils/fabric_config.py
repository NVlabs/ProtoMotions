# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for Lightning Fabric distributed training."""

import os
from datetime import timedelta
from typing import Dict, Any, Union, Optional, List
from omegaconf import DictConfig
from dataclasses import dataclass, field, fields
from lightning import fabric

from protomotions.utils.hydra_replacement import instantiate


def _default_ddp_strategy() -> fabric.strategies.DDPStrategy:
    """Build the default DDPStrategy with a configurable process-group timeout.

    2026-07-04 crash-rootcause fix: PyTorch's default 30-min (1800s) collective
    timeout was firing on BaseAgent.__init__'s one-time world-size all_gather
    whenever a rank's Isaac-env/motion-lib construction (documented to
    legitimately take 25+ min under NFS contention) pushed past 30 min,
    aborting the entire 8-rank job even though nothing was actually hung.
    PG_TIMEOUT_SEC (env, default 3600s = 1h) raises that ceiling above the
    known JIT/NFS-load variance without weakening real-hang detection (a
    genuine deadlock still eventually aborts, just later).
    """
    timeout_sec = int(os.environ.get("PG_TIMEOUT_SEC", "3600"))
    return fabric.strategies.DDPStrategy(timeout=timedelta(seconds=timeout_sec))


@dataclass
class FabricConfig:
    """Configuration for Lightning Fabric distributed training."""

    accelerator: str = field(
        default="gpu",
        metadata={"help": "Hardware accelerator: 'gpu', 'cpu', 'tpu', 'auto'."}
    )
    devices: Union[int, str] = field(
        default=1,
        metadata={"help": "Number of devices or 'auto' for all available."}
    )
    num_nodes: Union[int, str] = field(
        default=1,
        metadata={"help": "Number of nodes for distributed training.", "min": 1}
    )
    strategy: Union[Dict, fabric.strategies.Strategy] = field(
        default_factory=_default_ddp_strategy,
        metadata={"help": "Distributed training strategy (DDP, FSDP, etc)."}
    )
    precision: Union[str, int] = field(
        default="32-true",
        metadata={"help": "Training precision: '32-true', '16-mixed', 'bf16-mixed'."}
    )
    loggers: Optional[List[Union[Dict, fabric.loggers.Logger]]] = field(
        default=None,
        metadata={"help": "List of logging backends (WandB, TensorBoard, etc)."}
    )
    callbacks: Optional[List[Union[Dict, Any]]] = field(
        default=None,
        metadata={"help": "List of training callbacks."}
    )

    def __post_init__(self):
        if self.strategy is not None and (
            isinstance(self.strategy, dict) or isinstance(self.strategy, DictConfig)
        ):
            self.strategy = instantiate(self.strategy)
        if self.loggers is not None:
            loggers = []
            for logger in self.loggers:
                if isinstance(logger, dict) or isinstance(logger, DictConfig):
                    loggers.append(instantiate(logger))
                else:
                    loggers.append(logger)
            self.loggers = loggers
        if self.callbacks is not None:
            callbacks = []
            for callback in self.callbacks:
                if isinstance(callback, dict) or isinstance(callback, DictConfig):
                    callbacks.append(instantiate(callback))
                else:
                    callbacks.append(callback)
            self.callbacks = callbacks

    def as_kwargs(self) -> Dict[str, Any]:
        """Return Fabric constructor kwargs without deep-copying live objects."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def as_loggable_dict(self) -> Dict[str, Any]:
        """Return a safe summary for logs without touching logger internals."""

        def summarize(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [summarize(item) for item in value]
            return value.__class__.__name__

        return {field.name: summarize(getattr(self, field.name)) for field in fields(self)}
