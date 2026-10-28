# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MaskedMimic model inference helpers."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.evaluators.config import MaskedMimicActionSource
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel
from protomotions.agents.supervised.masked_mimic_config import (
    KLDScheduleConfig,
    MaskedMimicVAEConfig,
    MaskedMimicModelConfig,
)


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


def test_masked_mimic_forward_deployable_uses_prior_and_trunk_only():
    model = _masked_mimic_model_with_kld_schedule(None)
    tensordict = TensorDict(
        {
            "obs": torch.ones(2, 3),
            "vae_noise": torch.zeros(2, 2),
        },
        batch_size=2,
    )

    out = model.forward_deployable(tensordict)

    assert torch.equal(out["action"], torch.ones(2, 2))
    assert "privileged_action" not in out.keys()


class _ConditionalPrior(_InferencePrior):
    out_keys = ["prior_mu", "prior_logvar", "decoder_condition"]

    def forward(self, tensordict, log_internals=False):
        tensordict = super().forward(tensordict, log_internals)
        tensordict["decoder_condition"] = tensordict["obs"][:, :2] * 2
        return tensordict


class _ConditionalTrunk(_InferenceTrunk):
    in_keys = ["vae_latent", "decoder_condition", "decoder_obs"]

    def forward(self, tensordict, log_internals=False):
        tensordict["trunk_action"] = (
            tensordict["vae_latent"]
            + tensordict["decoder_condition"]
            + tensordict["decoder_obs"]
        )
        return tensordict


class _ConditionalEncoder(_InferenceEncoder):
    out_keys = ["encoder_mu", "encoder_logvar", "decoder_condition"]

    def forward(self, tensordict, log_internals=False):
        tensordict = super().forward(tensordict, log_internals)
        tensordict["decoder_condition"] = tensordict["expert_obs"][:, :2]
        return tensordict


def test_prior_condition_is_internal_for_training_and_deployment():
    config = _masked_mimic_model_with_kld_schedule(None).config
    config.prior._target_ = __name__ + "._ConditionalPrior"
    config.trunk._target_ = __name__ + "._ConditionalTrunk"
    model = MaskedMimicModel(config)
    assert model.in_keys == ["obs", "expert_obs", "decoder_obs"]
    assert model.deployable_in_keys == ["obs", "decoder_obs"]
    inputs = TensorDict(
        {
            "obs": torch.ones(2, 3),
            "decoder_obs": torch.ones(2, 2),
            "vae_noise": torch.zeros(2, 2),
        },
        batch_size=2,
    )
    output = model.materialize(inputs.clone())
    assert torch.equal(output["action"], torch.full((2, 2), 3.0))
    assert "privileged_action" not in output
    inputs["expert_obs"] = torch.ones(2, 3)
    output = model(inputs)
    assert torch.equal(output["privileged_action"], output["action"])
    assert torch.equal(model.kl_loss(output), torch.zeros(2, 2))


def test_encoder_output_does_not_satisfy_deployable_decoder_input():
    config = _masked_mimic_model_with_kld_schedule(None).config
    config.encoder._target_ = __name__ + "._ConditionalEncoder"
    config.trunk._target_ = __name__ + "._ConditionalTrunk"
    model = MaskedMimicModel(config)
    # The condition is not produced by this prior: keep the missing input
    # visible instead of borrowing privileged data to make deployment pass.
    assert "decoder_condition" in model.deployable_in_keys
    with pytest.raises(KeyError):
        model.forward_deployable(
            TensorDict(
                {
                    "obs": torch.ones(1, 3),
                    "decoder_obs": torch.ones(1, 2),
                    "vae_noise": torch.zeros(1, 2),
                },
                batch_size=1,
            )
        )


def test_masked_mimic_evaluator_selects_real_model_posterior_or_prior_only():
    from protomotions.agents.evaluators.masked_mimic_evaluator import MaskedMimicEvaluator

    model = _masked_mimic_model_with_kld_schedule(None).eval()
    evaluator = MaskedMimicEvaluator.__new__(MaskedMimicEvaluator)
    evaluator.agent = SimpleNamespace(model=model)

    def distinct_posterior(module, inputs, output):
        output["encoder_mu"] = output["encoder_mu"] + 2.
        return output

    hook = model._encoder.register_forward_hook(distinct_posterior)
    evaluator.config = SimpleNamespace(action_source=MaskedMimicActionSource.POSTERIOR)
    obs = TensorDict({"obs": torch.ones(2, 3), "vae_noise": torch.zeros(2, 2)}, batch_size=2)
    assert torch.equal(evaluator._policy_action(obs), torch.full((2, 2), 3.))
    hook.remove()

    def forbid_encoder(*args):
        raise AssertionError("Inference must not execute the privileged encoder")

    model._encoder.forward = forbid_encoder
    evaluator.config.action_source = MaskedMimicActionSource.PRIOR
    assert torch.equal(evaluator._policy_action(obs), torch.ones(2, 2))


def _masked_mimic_model_with_kld_schedule(schedule):
    return MaskedMimicModel(
        MaskedMimicModelConfig(
            prior=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferencePrior"
            ),
            encoder=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferenceEncoder"
            ),
            trunk=SimpleNamespace(
                _target_="protomotions.tests.test_masked_mimic_model._InferenceTrunk"
            ),
            vae=MaskedMimicVAEConfig(
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
        cyclical=False,
    )
    model = _masked_mimic_model_with_kld_schedule(schedule)

    assert model._kld_coefficient(current_epoch=4) == 0.25
    assert model._kld_coefficient(current_epoch=5) == 0.75
    assert model._kld_coefficient(current_epoch=9) == 0.75


def _cyclical_schedule(cyclical=True):
    return KLDScheduleConfig(
        init_kld_coeff=0.0,
        end_kld_coeff=1.0,
        start_epoch=10,
        end_epoch=14,
        cyclical=cyclical,
    )


def test_masked_mimic_kld_schedule_noncyclical_stays_at_end_value():
    model = _masked_mimic_model_with_kld_schedule(_cyclical_schedule(False))

    assert model._kld_coefficient(current_epoch=14) == 1.0
    assert model._kld_coefficient(current_epoch=15) == 1.0
    assert model._kld_coefficient(current_epoch=100) == 1.0


def test_masked_mimic_kld_schedule_cyclical_wraps_to_start():
    model = _masked_mimic_model_with_kld_schedule(_cyclical_schedule())

    assert model._kld_coefficient(current_epoch=5) == 0.0
    assert model._kld_coefficient(current_epoch=10) == 0.0
    assert model._kld_coefficient(current_epoch=12) == 0.5
    assert model._kld_coefficient(current_epoch=14) == 1.0
    # Period is 5 epochs: 15 restarts the ramp, 19 is its end, 20 restarts again.
    assert model._kld_coefficient(current_epoch=15) == 0.0
    assert model._kld_coefficient(current_epoch=17) == 0.5
    assert model._kld_coefficient(current_epoch=19) == 1.0
    assert model._kld_coefficient(current_epoch=20) == 0.0


def test_masked_mimic_kld_schedule_cyclical_requires_nonempty_ramp():
    with pytest.raises(ValueError, match="end_epoch > start_epoch"):
        KLDScheduleConfig(start_epoch=5, end_epoch=5, cyclical=True)
