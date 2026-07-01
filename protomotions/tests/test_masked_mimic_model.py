# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MaskedMimic model inference helpers."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel


class _InferencePrior(ProtoMotionsTensorDictModule):
    in_keys = ["obs"]
    out_keys = ["prior_mu", "prior_logvar"]

    def __init__(self, config):
        super().__init__()

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        batch = tensordict.batch_size[0]
        tensordict["prior_mu"] = torch.zeros(batch, 2)
        tensordict["prior_logvar"] = torch.zeros(batch, 2)
        return tensordict


class _InferenceEncoder(ProtoMotionsTensorDictModule):
    in_keys = ["expert_obs"]
    out_keys = ["encoder_mu", "encoder_logvar"]

    def __init__(self, config):
        super().__init__()

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        batch = tensordict.batch_size[0]
        tensordict["encoder_mu"] = torch.zeros(batch, 2)
        tensordict["encoder_logvar"] = torch.zeros(batch, 2)
        return tensordict


class _InferenceTrunk(ProtoMotionsTensorDictModule):
    in_keys = ["vae_latent"]
    out_keys = ["trunk_action"]

    def __init__(self, config):
        super().__init__()

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        tensordict["trunk_action"] = tensordict["vae_latent"] + 1.0
        return tensordict


def test_masked_mimic_forward_inference_uses_prior_and_trunk_only():
    model = MaskedMimicModel(
        SimpleNamespace(
            prior=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferencePrior"
            ),
            encoder=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferenceEncoder"
            ),
            trunk=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferenceTrunk"
            ),
            vae=SimpleNamespace(vae_latent_dim=2, vae_noise_type="zeros"),
        )
    )
    tensordict = TensorDict(
        {
            "obs": torch.ones(2, 3),
            "vae_noise": torch.zeros(2, 2),
        },
        batch_size=2,
    )

    out = model.forward_inference(tensordict)

    assert torch.equal(out["action"], torch.ones(2, 2))
    assert "privileged_action" not in out.keys()


def _masked_mimic_model_with_kld_schedule(schedule):
    return MaskedMimicModel(
        SimpleNamespace(
            prior=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferencePrior"
            ),
            encoder=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferenceEncoder"
            ),
            trunk=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferenceTrunk"
            ),
            vae=SimpleNamespace(
                vae_latent_dim=2,
                vae_noise_type="zeros",
                kld_schedule=schedule,
            ),
        )
    )


def test_masked_mimic_kld_schedule_handles_zero_length_transition():
    schedule = SimpleNamespace(
        start_epoch=5,
        end_epoch=5,
        init_kld_coeff=0.25,
        end_kld_coeff=0.75,
    )
    model = _masked_mimic_model_with_kld_schedule(schedule)

    assert model._kld_coefficient(current_epoch=4) == 0.25
    assert model._kld_coefficient(current_epoch=5) == 0.75
    assert model._kld_coefficient(current_epoch=9) == 0.75


def test_masked_mimic_model_loss_without_tensordict_returns_grad_connected_zero():
    schedule = SimpleNamespace(
        start_epoch=0,
        end_epoch=10,
        init_kld_coeff=0.5,
        end_kld_coeff=0.5,
    )
    model = _masked_mimic_model_with_kld_schedule(schedule)
    zero_loss = torch.tensor(2.0, requires_grad=True)

    loss, logs = model.compute_model_loss(
        tensordict=None,
        current_epoch=0,
        zero_loss=zero_loss,
    )

    assert torch.equal(loss, zero_loss * 0.0)
    assert torch.equal(logs["model/kld_loss"], torch.zeros(()))
    loss.backward()
    assert torch.equal(zero_loss.grad, torch.zeros_like(zero_loss))
