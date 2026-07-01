# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for common autoregressive categorical modules."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.autoregressive import (
    DiscreteAutoregressiveTransformer,
    generate_causal_mask,
    kl_divergence_categorical,
    kl_divergence_sampling_distribution,
    nucleus_sampling,
    nucleus_sampling_prior_constraint,
    prior_constrained_sampling_log_probs,
    resolve_discrete_autoregressive_config,
    sampling_log_probs,
)
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    ModuleContainerConfig,
)


def _config():
    return DiscreteAutoregressiveTransformerConfig(
        in_keys=["context_obs", "teacher_tokens"],
        out_keys=["token_logits"],
        context_key="context_obs",
        token_key="teacher_tokens",
        logits_key="token_logits",
        num_tokens=3,
        vocab_size=5,
        d_model=8,
        num_heads=2,
        num_layers=1,
        ff_size=16,
        dropout=0.0,
    )


def test_autoregressive_transformer_uses_configured_tensordict_keys():
    model = DiscreteAutoregressiveTransformer(_config())
    tokens = torch.randint(0, 5, (2, 3))
    td = TensorDict(
        {
            "context_obs": torch.randn(2, 4),
            "teacher_tokens": F.one_hot(tokens, 5).float(),
        },
        batch_size=2,
    )

    out = model(td)

    assert out["token_logits"].shape == (2, 3, 5)
    assert model.context_dim == 4
    assert model.in_keys == ["context_obs", "teacher_tokens"]
    assert model.out_keys == ["token_logits"]


def test_autoregressive_transformer_accepts_integer_teacher_tokens():
    model = DiscreteAutoregressiveTransformer(_config())
    tokens = torch.tensor([[0, 1, 2], [3, 4, 0]])
    td = TensorDict(
        {
            "context_obs": torch.randn(2, 4),
            "teacher_tokens": tokens,
        },
        batch_size=2,
    )

    out = model(td)

    assert out["token_logits"].shape == (2, 3, 5)


def test_autoregressive_transformer_generation_uses_configured_context_key():
    model = DiscreteAutoregressiveTransformer(_config())
    td = TensorDict({"context_obs": torch.randn(2, 4)}, batch_size=2)

    td = model(td)

    indices = td["token_logits_tokens"]
    logits = td["token_logits"]
    assert indices.shape == (2, 3)
    assert logits.shape == (2, 3, 5)
    assert torch.all(indices >= 0)
    assert torch.all(indices < 5)


def test_autoregressive_generation_can_use_prior_constraint_each_step():
    model = DiscreteAutoregressiveTransformer(_config())
    calls = []

    def prior_constraint(token_indices, step):
        calls.append((None if token_indices is None else token_indices.clone(), step))
        logits = torch.full((2, model.vocab_size), -20.0)
        logits[:, 2] = 20.0
        return logits

    generated, logits, logps = model.generate_from_context(
        torch.zeros(2, model.d_model),
        num_tokens=2,
        top_p=0.5,
        prior_constraint=prior_constraint,
    )

    assert torch.equal(generated, torch.full((2, 2), 2))
    assert logits.shape == (2, 2, model.vocab_size)
    assert logps.shape == (2, 2)
    assert calls[0] == (None, 0)
    assert calls[1][1] == 1
    assert torch.equal(calls[1][0], torch.full((2, 1), 2))


def test_autoregressive_transformer_can_return_generated_token_logprobs():
    config = _config()
    config.logprob_key = "token_logp"
    model = DiscreteAutoregressiveTransformer(config)
    td = TensorDict({"context_obs": torch.randn(2, 4)}, batch_size=2)

    td = model(td)

    assert td["token_logits_tokens"].shape == (2, 3)
    assert td["token_logp"].shape == (2, 3)
    assert torch.all(torch.isfinite(td["token_logp"]))


def test_autoregressive_transformer_rejects_bad_teacher_token_shape():
    model = DiscreteAutoregressiveTransformer(_config())
    td = TensorDict(
        {
            "context_obs": torch.randn(2, 4),
            "teacher_tokens": torch.zeros(2, 3, 4),
        },
        batch_size=2,
    )

    try:
        model(td)
    except ValueError as error:
        assert "Expected token indices" in str(error)
    else:
        raise AssertionError("Expected invalid teacher token shape to fail")


def test_autoregressive_transformer_rejects_sequence_past_position_capacity():
    model = DiscreteAutoregressiveTransformer(_config())
    sequence = torch.zeros(2, 2, model.d_model)
    pos_emb = torch.zeros(1, 1, model.d_model)

    try:
        model.add_positions(sequence, pos_emb=pos_emb)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("Expected overlong sequence to fail")


def test_autoregressive_transformer_rejects_unresolved_shape():
    config = _config()
    config.num_tokens = 0
    config.vocab_size = 0

    try:
        DiscreteAutoregressiveTransformer(config)
    except ValueError as exc:
        assert "resolved positive num_tokens" in str(exc)
    else:
        raise AssertionError("Expected unresolved autoregressive shape to fail")


def test_generate_causal_mask_blocks_context_to_targets_and_future_tokens():
    mask = generate_causal_mask(num_target=3, num_context=2)

    assert mask.shape == (5, 5)
    assert torch.all(mask[:2, :2] == 0)
    assert torch.isneginf(mask[:2, 2:]).all()
    assert torch.all(mask[2:, :2] == 0)
    assert torch.all(mask[2:, 2:].tril() == 0)
    assert torch.isneginf(mask[2, 3:]).all()
    assert torch.isneginf(mask[3, 4:]).all()


def test_nucleus_sampling_collapses_to_only_kept_token():
    logits = torch.tensor([[10.0, 0.0, -10.0], [-5.0, 8.0, 0.0]])

    sampled = nucleus_sampling(logits, p=0.5)

    assert torch.equal(sampled, torch.tensor([0, 1]))


def test_sampling_log_probs_match_top_p_sampling_distribution():
    logits = torch.tensor([[4.0, 3.0, 0.0]])

    log_probs = sampling_log_probs(logits, p=0.5)

    assert torch.allclose(log_probs.exp(), torch.tensor([[1.0, 0.0, 0.0]]))


def test_prior_constrained_sampling_log_probs_masks_without_nan_actor_gradients():
    logits = torch.tensor([[4.0, 3.0, 0.0]], requires_grad=True)
    prior_logits = torch.tensor([[8.0, 1.0, 0.0]])

    log_probs = prior_constrained_sampling_log_probs(
        logits,
        prior_logits,
        p=0.5,
    )
    loss = log_probs[:, 0].sum()
    loss.backward()

    assert torch.isfinite(logits.grad).all()


def test_sampling_kl_handles_mismatched_nucleus_support_without_nan_gradients():
    logits = torch.tensor([[8.0, 1.0, 0.0]], requires_grad=True)
    prior_logits = torch.tensor([[1.0, 8.0, 0.0]])

    kl = kl_divergence_sampling_distribution(
        logits,
        prior_logits,
        p=0.8,
        reduction="none",
    )
    kl.sum().backward()

    assert torch.isfinite(kl).all()
    assert torch.all(kl > 0.0)
    assert torch.isfinite(logits.grad).all()


def test_nucleus_sampling_rejects_zero_temperature():
    with pytest.raises(ValueError, match="temperature"):
        nucleus_sampling(torch.zeros(1, 3), temperature=0.0)


def test_generate_from_context_records_filtered_token_logprobs():
    model = DiscreteAutoregressiveTransformer(_config())
    logits = torch.tensor([[4.0, 3.0, 0.0]])

    def fixed_logits(context, token_indices=None, **kwargs):
        return logits.expand(context.shape[0], -1)

    model.next_logits_from_context = fixed_logits

    generated, _, logps = model.generate_from_context(
        torch.zeros(2, model.d_model),
        num_tokens=1,
        top_p=0.5,
    )

    assert torch.equal(generated, torch.zeros(2, 1, dtype=torch.long))
    assert torch.allclose(logps, torch.zeros(2, 1))


def test_generate_temporarily_uses_eval_mode_and_restores_training():
    model = DiscreteAutoregressiveTransformer(_config())
    model.train(True)
    saw_training_modes = []

    def fake_encode_context(tensordict):
        saw_training_modes.append(model.training)
        return torch.zeros(tensordict.batch_size[0], model.d_model)

    def fake_generate_from_context(context, num_tokens, **kwargs):
        batch_size = context.shape[0]
        return (
            torch.zeros(batch_size, num_tokens, dtype=torch.long),
            torch.zeros(batch_size, num_tokens, model.vocab_size),
            torch.zeros(batch_size, num_tokens),
        )

    model.encode_context = fake_encode_context
    model.generate_from_context = fake_generate_from_context

    model.generate(TensorDict({"context_obs": torch.zeros(2, 4)}, batch_size=2))

    assert saw_training_modes == [False]
    assert model.training


def test_autoregressive_transformer_uses_base_rollout_context_lifecycle():
    assert (
        DiscreteAutoregressiveTransformer.reset_rollout_context
        is ProtoMotionsTensorDictModule.reset_rollout_context
    )
    assert (
        DiscreteAutoregressiveTransformer.rollout_context_keys
        is ProtoMotionsTensorDictModule.rollout_context_keys
    )


def test_prior_constraint_samples_only_from_prior_nucleus():
    model_logits = torch.tensor([[10.0, 0.0, -10.0]])
    prior_logits = torch.tensor([[-10.0, 0.0, 10.0]])

    sampled = nucleus_sampling_prior_constraint(
        model_logits,
        prior_logits,
        p=0.5,
    )

    assert torch.equal(sampled, torch.tensor([2]))


def test_prior_constraint_falls_back_to_prior_nucleus_when_model_mass_is_tiny():
    model_logits = torch.tensor([[-10000.0, 0.0, -10000.0]])
    prior_logits = torch.tensor([[10.0, 0.0, -10.0]])

    sampled = nucleus_sampling_prior_constraint(
        model_logits,
        prior_logits,
        p=0.1,
    )

    assert torch.equal(sampled, torch.tensor([0]))


def test_prior_constraint_treats_near_zero_overlap_as_no_overlap():
    model_logits = torch.tensor([[0.0, 8.0, 8.0]])
    prior_logits = torch.tensor([[10.0, 0.0, -10.0]])

    sampled = nucleus_sampling_prior_constraint(
        model_logits,
        prior_logits,
        p=0.1,
    )

    assert torch.equal(sampled, torch.tensor([0]))


def test_kl_divergence_categorical_matches_manual_reductions():
    logits = torch.tensor([[2.0, 0.0], [0.5, -0.5]])
    prior_logits = torch.tensor([[0.0, 1.0], [0.25, 0.0]])
    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(prior_logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    per_row = (p * (log_p - log_q)).sum(dim=-1)

    assert torch.allclose(
        kl_divergence_categorical(logits, prior_logits, reduction="none"),
        per_row,
    )
    assert torch.allclose(
        kl_divergence_categorical(logits, prior_logits, reduction="sum"),
        per_row.sum(),
    )
    assert torch.allclose(
        kl_divergence_categorical(logits, prior_logits),
        per_row.mean(),
    )


def test_resolve_discrete_autoregressive_config_copies_and_updates_logits_head():
    config = _config()

    resolved = resolve_discrete_autoregressive_config(
        config,
        num_tokens=4,
        vocab_size=7,
    )

    assert resolved is not config
    assert resolved.num_tokens == 4
    assert resolved.vocab_size == 7
    assert resolved.output_head.models[0].num_out == 7
    assert config.num_tokens == 3
    assert config.vocab_size == 5
    assert config.output_head.models[0].num_out == 5


def test_autoregressive_config_derives_keys_from_context_encoder_container():
    config = DiscreteAutoregressiveTransformerConfig(
        context_encoder=ModuleContainerConfig(
            in_keys=["state", "phase"],
            out_keys=["context_z"],
            models=[SimpleNamespace(in_keys=["unused"], out_keys=["unused"])],
        ),
        token_key="tokens",
        logits_key="logits",
        out_keys=["custom_logits"],
        num_tokens=2,
        vocab_size=3,
    )

    assert config.context_embedding_key == "context_z"
    assert config.context_encoder.in_keys == ["state", "phase"]
    assert config.context_encoder.out_keys == ["context_z"]
    assert config.in_keys == ["state", "phase", "tokens"]
    assert config.out_keys == ["custom_logits"]
    assert config.logits_key == "custom_logits"


def test_autoregressive_config_requires_context_encoder_to_declare_output_key():
    with pytest.raises(AssertionError, match="must declare"):
        DiscreteAutoregressiveTransformerConfig(
            context_encoder=ModuleContainerConfig(
                models=[SimpleNamespace(in_keys=["state"], out_keys=[])],
            ),
            token_key="tokens",
            num_tokens=2,
            vocab_size=3,
        )
