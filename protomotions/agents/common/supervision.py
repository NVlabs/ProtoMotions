# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared supervision losses for imitation and distillation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn.functional as F


class SupervisionLossType(str, Enum):
    """Supported supervised losses over configured prediction and target keys."""

    MSE = "mse"
    DISCRETE_CROSS_ENTROPY = "discrete_cross_entropy"
    DISCRETE_KL = "discrete_kl"
    CONTINUOUS_GAUSSIAN_KL = "continuous_gaussian_kl"


@dataclass
class SupervisionLossConfig:
    """Key-based supervised loss over model outputs and labels.

    Distillation agents use ``prediction_key`` and ``target_key`` to select
    tensors from a TensorDict batch, so the same loss config can supervise
    actions, discrete latent tokens, or distribution parameters.
    """

    loss_type: SupervisionLossType = SupervisionLossType.MSE
    prediction_key: str = "privileged_action"
    target_key: str = "expert_actions"
    prediction_logvar_key: Optional[str] = None
    target_logvar_key: Optional[str] = None
    label_smoothing: float = 0.0
    weight: float = 1.0
    log_prefix: str = "supervision"
    enabled: bool = True
    extra: Dict[str, str] = field(default_factory=dict)


def _get(batch, key: str) -> torch.Tensor:
    if key in batch:
        return batch[key]
    raise KeyError(f"Missing tensor '{key}' for supervision loss")


def _discrete_kl(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(target_logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def _gaussian_kl(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target_mean: torch.Tensor,
    target_logvar: torch.Tensor,
) -> torch.Tensor:
    var = logvar.exp()
    target_var = target_logvar.exp()
    kl = 0.5 * (
        target_logvar
        - logvar
        + (var + (mean - target_mean) ** 2) / target_var
        - 1
    )
    return kl.sum(dim=-1).mean()


def compute_supervision_loss(batch, config: SupervisionLossConfig):
    """Compute a configured supervised loss."""
    if not config.enabled:
        prediction = _get(batch, config.prediction_key)
        zero = prediction.sum() * 0.0
        return zero, {f"{config.log_prefix}/loss": zero.detach()}

    loss_type = SupervisionLossType(config.loss_type)
    prefix = config.log_prefix

    if loss_type == SupervisionLossType.MSE:
        raw_loss = F.mse_loss(
            _get(batch, config.prediction_key),
            _get(batch, config.target_key),
        )
        metrics = {f"{prefix}/mse": raw_loss.detach()}
    elif loss_type == SupervisionLossType.DISCRETE_CROSS_ENTROPY:
        logits = _get(batch, config.prediction_key)
        target = _get(batch, config.target_key)
        raw_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            label_smoothing=config.label_smoothing,
        )
        with torch.no_grad():
            accuracy = (
                logits.argmax(dim=-1).reshape(-1) == target.reshape(-1)
            ).float().mean()
        metrics = {
            f"{prefix}/cross_entropy": raw_loss.detach(),
            f"{prefix}/accuracy": accuracy,
            f"{prefix}/perplexity": torch.exp(raw_loss.detach()),
        }
    elif loss_type == SupervisionLossType.DISCRETE_KL:
        raw_loss = _discrete_kl(
            _get(batch, config.prediction_key),
            _get(batch, config.target_key),
        )
        metrics = {f"{prefix}/discrete_kl": raw_loss.detach()}
    elif loss_type == SupervisionLossType.CONTINUOUS_GAUSSIAN_KL:
        if config.prediction_logvar_key is None or config.target_logvar_key is None:
            raise ValueError(
                "Continuous Gaussian KL requires prediction_logvar_key and "
                "target_logvar_key."
            )
        raw_loss = _gaussian_kl(
            _get(batch, config.prediction_key),
            _get(batch, config.prediction_logvar_key),
            _get(batch, config.target_key),
            _get(batch, config.target_logvar_key),
        )
        metrics = {f"{prefix}/gaussian_kl": raw_loss.detach()}
    else:
        raise NotImplementedError(f"Unsupported supervision loss type: {loss_type}")

    weighted_loss = raw_loss * config.weight
    metrics[f"{prefix}/loss"] = weighted_loss.detach()
    return weighted_loss, metrics
