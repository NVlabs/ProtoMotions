# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared supervision losses."""

import torch
import torch.nn.functional as F
import pytest

from protomotions.agents.common.latent import LATENT_LOGITS_KEY
from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
    compute_supervision_loss,
)


def test_discrete_supervision_loss_uses_cross_entropy_and_reports_accuracy():
    logits = torch.tensor(
        [
            [[4.0, 0.0, -1.0], [0.0, 5.0, -2.0]],
            [[-1.0, 0.0, 6.0], [3.0, 0.0, -1.0]],
        ]
    )
    target = torch.tensor([[0, 1], [2, 0]])
    batch = {
        LATENT_LOGITS_KEY: logits,
        "target_latent": target,
    }
    config = SupervisionLossConfig(
        loss_type=SupervisionLossType.DISCRETE_CROSS_ENTROPY,
        prediction_key=LATENT_LOGITS_KEY,
        target_key="target_latent",
        label_smoothing=0.0,
    )

    loss, metrics = compute_supervision_loss(batch, config)
    expected = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["supervision/accuracy"], torch.tensor(1.0))


def test_continuous_distribution_supervision_uses_gaussian_kl():
    batch = {
        "latent_mu": torch.tensor([[0.0, 1.0], [2.0, -1.0]]),
        "latent_logvar": torch.tensor([[0.0, 0.2], [-0.4, 0.0]]),
        "target_latent_mu": torch.tensor([[0.5, 0.5], [1.5, -1.0]]),
        "target_latent_logvar": torch.tensor([[0.1, -0.1], [0.0, 0.3]]),
    }
    config = SupervisionLossConfig(
        loss_type=SupervisionLossType.CONTINUOUS_GAUSSIAN_KL,
        prediction_key="latent_mu",
        target_key="target_latent_mu",
        prediction_logvar_key="latent_logvar",
        target_logvar_key="target_latent_logvar",
    )

    loss, metrics = compute_supervision_loss(batch, config)

    pred_var = batch["latent_logvar"].exp()
    target_var = batch["target_latent_logvar"].exp()
    expected = 0.5 * (
        batch["target_latent_logvar"]
        - batch["latent_logvar"]
        + (pred_var + (batch["latent_mu"] - batch["target_latent_mu"]) ** 2)
        / target_var
        - 1
    )
    expected = expected.sum(dim=-1).mean()

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["supervision/gaussian_kl"], expected.detach())


def test_action_loss_is_just_a_configured_prediction_target_mse():
    batch = {
        "privileged_action": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "expert_actions": torch.tensor([[2.0, 2.0], [1.0, 5.0]]),
    }
    config = SupervisionLossConfig(
        loss_type=SupervisionLossType.MSE,
        prediction_key="privileged_action",
        target_key="expert_actions",
        log_prefix="masked_mimic",
    )

    loss, metrics = compute_supervision_loss(batch, config)
    expected = F.mse_loss(batch["privileged_action"], batch["expert_actions"])

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["masked_mimic/mse"], expected.detach())


def test_disabled_supervision_loss_returns_zero_without_target_key():
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    config = SupervisionLossConfig(
        prediction_key="prediction",
        target_key="missing_target",
        enabled=False,
    )

    loss, metrics = compute_supervision_loss({"prediction": prediction}, config)

    assert torch.equal(loss, torch.zeros(()))
    assert torch.equal(metrics["supervision/loss"], torch.zeros(()))

    loss.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_discrete_kl_supervision_matches_manual_distribution_kl():
    logits = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
    target_logits = torch.tensor([[0.5, 0.0], [1.0, -0.5]])
    config = SupervisionLossConfig(
        loss_type=SupervisionLossType.DISCRETE_KL,
        prediction_key="logits",
        target_key="target_logits",
        log_prefix="prior",
        weight=0.25,
    )

    loss, metrics = compute_supervision_loss(
        {"logits": logits, "target_logits": target_logits},
        config,
    )

    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(target_logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    raw = (p * (log_p - log_q)).sum(dim=-1).mean()

    assert torch.allclose(loss, raw * 0.25)
    assert torch.allclose(metrics["prior/discrete_kl"], raw.detach())
    assert torch.allclose(metrics["prior/loss"], (raw * 0.25).detach())


def test_gaussian_kl_supervision_requires_logvar_keys():
    config = SupervisionLossConfig(
        loss_type=SupervisionLossType.CONTINUOUS_GAUSSIAN_KL,
        prediction_key="latent_mu",
        target_key="target_latent_mu",
    )

    with pytest.raises(ValueError, match="requires prediction_logvar_key"):
        compute_supervision_loss(
            {
                "latent_mu": torch.zeros(1, 2),
                "target_latent_mu": torch.zeros(1, 2),
            },
            config,
        )


def test_supervision_loss_reports_missing_tensors_by_configured_key():
    config = SupervisionLossConfig(
        prediction_key="prediction",
        target_key="target",
    )

    with pytest.raises(KeyError, match="target"):
        compute_supervision_loss({"prediction": torch.zeros(1, 2)}, config)


def test_supervision_loss_rejects_unknown_loss_type_values():
    config = SupervisionLossConfig(
        loss_type="unsupported",
        prediction_key="prediction",
        target_key="target",
    )

    with pytest.raises(ValueError, match="not a valid SupervisionLossType"):
        compute_supervision_loss(
            {
                "prediction": torch.zeros(1, 2),
                "target": torch.zeros(1, 2),
            },
            config,
        )
