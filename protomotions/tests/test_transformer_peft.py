# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reusable transformer PEFT helpers."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDict

from protomotions.agents.common.autoregressive import DiscreteAutoregressiveTransformer
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    MLPWithConcatConfig,
    ModuleContainerConfig,
    NormObsBaseConfig,
)
from protomotions.agents.common.common import NormObsBase
from protomotions.agents.peft.adapters import (
    LoRALayer,
    TransformerEncoderWithConditioning,
    TransformerLayerWithDoRA,
    TransformerLayerWithLoRA,
    inject_transformer_peft,
)
from protomotions.agents.peft.prior_with_peft import DiscretePriorWithPEFT


def _prior():
    return DiscreteAutoregressiveTransformer(
        DiscreteAutoregressiveTransformerConfig(
            in_keys=["prior_context", "prior_tokens"],
            out_keys=["latent_logits"],
            context_key="prior_context",
            token_key="prior_tokens",
            logits_key="latent_logits",
            generated_tokens_key="latent",
            logprob_key="latent_logprob",
            d_model=8,
            num_heads=2,
            num_layers=1,
            ff_size=16,
            dropout=0.0,
            num_tokens=3,
            vocab_size=5,
        )
    )


def _prior_with_distinct_context_in_key():
    return DiscreteAutoregressiveTransformer(
        DiscreteAutoregressiveTransformerConfig(
            in_keys=["max_coords_obs", "prior_tokens"],
            out_keys=["latent_logits"],
            context_key="prior_context",
            token_key="prior_tokens",
            logits_key="latent_logits",
            generated_tokens_key="latent",
            logprob_key="latent_logprob",
            context_encoder=ModuleContainerConfig(
                in_keys=["max_coords_obs"],
                out_keys=["context_embedding"],
                models=[
                    MLPWithConcatConfig(
                        in_keys=["max_coords_obs"],
                        out_keys=["context_embedding"],
                        num_out=8,
                        layers=[],
                    )
                ],
            ),
            d_model=8,
            num_heads=2,
            num_layers=1,
            ff_size=16,
            dropout=0.0,
            num_tokens=3,
            vocab_size=5,
        )
    )


def _materialize_prior(prior):
    tokens = torch.randint(0, 5, (2, 3))
    context_key = prior.context_in_keys[0]
    prior(
        TensorDict(
            {
                context_key: torch.randn(2, 4),
                "prior_tokens": F.one_hot(tokens, 5).float(),
            },
            batch_size=2,
        )
    )
    return tokens


def test_inject_transformer_peft_freezes_base_and_enables_adapters():
    layer = nn.TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(layer, num_layers=2)

    wrapped = inject_transformer_peft(
        transformer=transformer,
        conditioning_dim=5,
        rank=2,
        alpha=1.0,
        peft_type="dora",
    )

    trainable = {
        name
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    }

    assert isinstance(wrapped, TransformerEncoderWithConditioning)
    assert trainable
    assert all(
        any(part in name for part in ("lora", "gamma", "beta", ".m"))
        for name in trainable
    )

    x = torch.randn(3, 4, 8)
    task_c = torch.randn(3, 5)
    out = wrapped(x, task_c=task_c)
    assert out.shape == x.shape


def test_lora_layer_and_mixed_conditioning_encoder_cover_plain_layers():
    lora = LoRALayer(
        c_dim=3,
        in_dim=4,
        out_dim=5,
        rank=2,
        alpha=0.5,
    )
    x = torch.randn(2, 3, 4)
    c_mul = torch.ones(2, 2)
    c_add = torch.zeros(2, 2)

    assert lora(x, c_mul, c_add).shape == (2, 3, 5)

    conditioned = TransformerLayerWithLoRA(
        nn.TransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            batch_first=True,
            dropout=0.0,
        ),
        c_dim=3,
        rank=2,
        alpha=1.0,
    )
    plain = nn.TransformerEncoderLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        batch_first=True,
        dropout=0.0,
    )
    encoder = TransformerEncoderWithConditioning(nn.ModuleList([conditioned, plain]))

    out = encoder(torch.randn(2, 3, 4), task_c=torch.randn(2, 3))

    assert out.shape == (2, 3, 4)


def test_dora_adapter_starts_near_frozen_prior_scale():
    layer = TransformerLayerWithDoRA(
        nn.TransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            batch_first=True,
            dropout=0.0,
        ),
        c_dim=3,
        rank=2,
        alpha=1.0,
    )

    assert layer.m.abs().max() <= 1e-5
    assert layer.lora.B.abs().max() < 1e-3


def test_prior_with_peft_wraps_common_autoregressive_transformer():
    prior = _prior()
    tokens = _materialize_prior(prior)

    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="dora",
        top_p=1.0,
    )
    peft.init_peft()
    logits = peft(
        {
            "prior_context": torch.randn(2, 4),
            "tokens": F.one_hot(tokens, 5).float(),
            "task_cond": torch.randn(2, 2),
        }
    )
    generated, gen_logits, logprob = peft.generate(
        {
            "prior_context": torch.randn(2, 4),
            "task_cond": torch.randn(2, 2),
        },
        return_logits=True,
        return_logprob=True,
    )

    assert logits.shape == (2, 3, 5)
    assert generated.shape == (2, 3)
    assert gen_logits.shape == (2, 3, 5)
    assert logprob.shape == (2, 3)


def test_prior_with_peft_persists_reference_position_embedding_after_capture():
    prior = _prior()
    _materialize_prior(prior)

    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="dora",
        top_p=1.0,
    )
    peft.init_peft()

    assert not any(key.startswith("_anchor_") for key in peft.state_dict())
    assert peft.capture_reference() is True
    state = peft.state_dict()

    assert "reference_prior._pos_emb" in state
    assert torch.equal(state["reference_prior._pos_emb"], peft.base_prior._pos_emb)


def test_prior_with_peft_routes_raw_context_to_frozen_prior_context_encoder():
    prior = _prior_with_distinct_context_in_key()
    tokens = _materialize_prior(prior)
    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="dora",
        top_p=1.0,
    )
    peft.init_peft()

    logits = peft(
        {
            "max_coords_obs": torch.randn(2, 4),
            "tokens": F.one_hot(tokens, 5).float(),
            "task_cond": torch.randn(2, 2),
        }
    )

    assert logits.shape == (2, 3, 5)


def test_prior_with_peft_train_keeps_frozen_prior_normalizers_eval():
    prior = _prior()
    _materialize_prior(prior)
    prior.input_normalizer = nn.BatchNorm1d(4)
    prior.extra_obs_norm = NormObsBase(NormObsBaseConfig(normalize_obs=True))

    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="dora",
        top_p=1.0,
    )
    peft.init_peft()

    trainable = {
        name
        for name, parameter in peft.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert any("lora" in name or "gamma" in name or "beta" in name for name in trainable)
    assert prior.input_normalizer.training is False
    assert prior.extra_obs_norm._freeze_running is True

    prior.input_normalizer.train()
    prior.extra_obs_norm.train()
    peft.train(True)

    assert peft.training is True
    assert prior.training is False
    assert prior.input_normalizer.training is False
    assert prior.extra_obs_norm.training is False
    assert prior.extra_obs_norm._freeze_running is True
    assert all(
        not layer.transformer_layer.training
        for layer in peft.base_prior._transformer.layers
        if hasattr(layer, "transformer_layer")
    )
    assert all(
        layer.training
        for layer in peft.base_prior._transformer.layers
        if hasattr(layer, "lora")
    )
    assert all(
        parameter.requires_grad
        for name, parameter in peft.named_parameters()
        if name in trainable
    )


def test_prior_with_peft_requires_known_conditioning_dim():
    peft = DiscretePriorWithPEFT(
        prior=_prior(),
        rank=2,
        alpha=1.0,
    )

    with pytest.raises(RuntimeError, match="conditioning_dim"):
        _ = peft.task_c_dim


def test_prior_with_peft_init_can_prime_context_dim_from_warmup_obs():
    prior = _prior()
    tokens = torch.randint(0, 5, (2, 3))
    peft = DiscretePriorWithPEFT(
        prior=prior,
        rank=2,
        alpha=1.0,
        top_p=1.0,
    )

    peft.init_peft(
        warmup_obs={
            "prior_context": torch.randn(2, 4),
            "prior_tokens": F.one_hot(tokens, 5).float(),
            "task_cond": torch.randn(2, 2),
        }
    )

    assert peft.base_prior.context_dim is not None
    assert peft.task_c_dim == 6


def test_prior_with_peft_direct_conditioning_and_reference_prior_paths():
    prior = _prior()
    tokens = _materialize_prior(prior)
    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="lora",
        top_p=1.0,
    )
    peft.init_peft()
    peft.train(True)

    input_dict = {
        "prior_context": torch.randn(2, 4),
        "prior_tokens": F.one_hot(tokens, 5).float(),
        "task_cond": torch.randn(2, 2),
    }

    logits = peft(input_dict)
    peft.capture_reference()
    prior_logits = peft.forward_prior(input_dict)
    generated_only = peft.generate(
        {
            "prior_context": torch.randn(2, 4),
            "task_cond": torch.randn(2, 2),
        },
        return_logits=False,
        return_logprob=False,
    )

    assert logits.shape == (2, 3, 5)
    assert prior_logits.shape == (2, 3, 5)
    assert generated_only.shape == (2, 3)
    assert all(
        not layer.transformer_layer.training
        for layer in peft.base_prior._transformer.layers
        if hasattr(layer, "transformer_layer")
    )


def test_prior_with_peft_can_sample_under_frozen_prior_constraint():
    prior = _prior()
    _materialize_prior(prior)
    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        top_p=1.0,
        prior_top_p=1.0,
        sampling_mode="prior_constraint",
    )
    peft.init_peft()

    generated, logits = peft.generate(
        {
            "prior_context": torch.randn(2, 4),
            "task_cond": torch.randn(2, 2),
        },
        return_logits=True,
        return_logprob=False,
    )

    assert generated.shape == (2, 3)
    assert logits.shape == (2, 3, 5)


def test_prior_with_peft_reference_can_be_pinned_and_clear_keeps_reference():
    prior = _prior()
    _materialize_prior(prior)
    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="dora",
        top_p=1.0,
    )
    peft.init_peft()
    layer = peft.base_prior._transformer.layers[0]
    layer.m.data.fill_(0.25)

    assert peft.capture_reference() is True
    assert peft.reference_ready is True
    reference_layer = peft.reference_prior._transformer.layers[0]
    assert torch.equal(reference_layer.m, layer.m)

    layer.m.data.fill_(0.5)
    assert peft.capture_reference() is False
    assert torch.all(reference_layer.m == 0.25)

    peft.clear_peft()
    assert torch.all(layer.m == 0)
    assert torch.all(reference_layer.m == 0.25)


def test_prior_with_peft_film_input_norm_learns_until_reference_capture():
    prior = _prior()
    _materialize_prior(prior)
    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="dora",
        top_p=1.0,
        film_input_norm=True,
        film_input_norm_clamp=3.0,
    )
    peft.init_peft()
    peft.train(True)

    input_dict = {
        "prior_context": torch.full((4, 4), 2.0),
        "task_cond": torch.full((4, 2), 3.0),
    }
    assert peft.film_input_norm is not None
    assert peft.film_input_norm.running_obs_norm.clamp_value == 3.0

    assert peft.film_input_norm._freeze_running is False
    peft._build_task_c(input_dict, peft.film_input_norm)
    mean_after_train_forward = peft.film_input_norm.running_obs_norm.mean.clone()
    assert not torch.allclose(
        mean_after_train_forward,
        torch.zeros_like(mean_after_train_forward),
    )

    assert peft.capture_reference() is True
    assert peft.film_input_norm._freeze_running is True
    assert peft.reference_film_input_norm is not None
    peft._build_task_c(
        {
            "prior_context": torch.full((4, 4), 10.0),
            "task_cond": torch.full((4, 2), 10.0),
        },
        peft.film_input_norm,
    )
    assert torch.equal(peft.film_input_norm.running_obs_norm.mean, mean_after_train_forward)


def test_prior_with_peft_generate_restores_training_mode():
    prior = _prior()
    _materialize_prior(prior)
    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        peft_type="dora",
        top_p=1.0,
    )
    peft.init_peft()
    peft.train(True)
    saw_training_modes = []

    def fake_context_embedding(input_dict, prior=None):
        return torch.zeros(2, peft.base_prior.d_model)

    def fake_generate_from_context(context, num_tokens, **kwargs):
        saw_training_modes.append(peft.training)
        return (
            torch.zeros(2, num_tokens, dtype=torch.long),
            torch.zeros(2, num_tokens, peft.base_prior.vocab_size),
            torch.zeros(2, num_tokens),
        )

    peft._context_embedding = fake_context_embedding
    peft.base_prior.generate_from_context = fake_generate_from_context

    peft.generate(
        {
            "prior_context": torch.randn(2, 4),
            "task_cond": torch.randn(2, 2),
        },
        return_logits=True,
        return_logprob=True,
    )

    assert saw_training_modes == [False]
    assert peft.training is True


def test_prior_constraint_sampling_uses_pinned_reference_prior_logits(monkeypatch):
    prior = _prior()
    _materialize_prior(prior)
    peft = DiscretePriorWithPEFT(
        prior=prior,
        conditioning_dim=6,
        rank=2,
        alpha=1.0,
        top_p=1.0,
        prior_top_p=0.5,
        sampling_mode="prior_constraint",
    )
    peft.init_peft()
    calls = []

    def fake_prior_constraint_log_probs(logits, prior_logits, p, temperature):
        calls.append((logits, prior_logits, p, temperature))
        log_probs = torch.full_like(logits, -torch.inf)
        log_probs[:, 0] = 0.0
        return log_probs

    monkeypatch.setattr(
        "protomotions.agents.common.autoregressive.prior_constrained_sampling_log_probs",
        fake_prior_constraint_log_probs,
    )

    generated = peft.generate(
        {
            "prior_context": torch.randn(2, 4),
            "task_cond": torch.randn(2, 2),
        },
        return_logits=False,
        return_logprob=False,
    )

    assert generated.shape == (2, 3)
    assert len(calls) == prior.num_tokens
    assert all(prior_logits.shape == (2, 5) for _, prior_logits, _, _ in calls)
    assert all(p == 0.5 for _, _, p, _ in calls)
