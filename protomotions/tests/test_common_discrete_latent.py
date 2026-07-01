# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for common discrete latent helpers."""

from types import SimpleNamespace

import torch
import pytest
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import nn

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common import discrete_latent as discrete_latent_module
from protomotions.agents.common.discrete_latent import (
    FSQTokenization,
    load_pretrained_discrete_latent_decoder,
    load_pretrained_discrete_latent_target_encoder,
    make_discrete_latent_decoder,
    make_discrete_latent_target_encoder,
)


class DummyEncoder(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ["obs"]
        self.out_keys = ["latent"]
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tensordict):
        tensordict["latent"] = tensordict["obs"] * self.weight
        return tensordict


class DummyDecoder(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ["state", "latent"]
        self.out_keys = ["mu"]
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tensordict):
        tensordict["mu"] = tensordict["state"] + tensordict["latent"] * self.weight
        return tensordict


class DummyQuantizer(nn.Module):
    num_fsq_levels = 5
    num_fsq_scalars = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def quantize(self, latent):
        return latent * self.scale

    def codes_to_indices(self, codes):
        return codes.long()

    def indices_to_codes(self, indices):
        return indices.float()


def test_fsq_tokenization_round_trip():
    tokenization = FSQTokenization(
        num_fsq_levels=5,
        num_fsq_scalars=4,
        fsq_scalars_per_prior_token=2,
    )
    fsq_indices = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])

    prior_tokens = tokenization.fsq_indices_to_prior_tokens(fsq_indices)
    round_trip = tokenization.prior_tokens_to_fsq_indices(prior_tokens)

    assert torch.equal(round_trip, fsq_indices)
    assert prior_tokens.shape == (2, 2)
    assert tokenization.one_hot_prior_tokens(prior_tokens).shape == (2, 2, 25)


def test_fsq_tokenization_validates_shape():
    with pytest.raises(ValueError, match="fsq_scalars_per_prior_token must be positive"):
        FSQTokenization(
            num_fsq_levels=5,
            num_fsq_scalars=4,
            fsq_scalars_per_prior_token=0,
        )

    with pytest.raises(ValueError, match="must evenly divide"):
        FSQTokenization(
            num_fsq_levels=5,
            num_fsq_scalars=5,
            fsq_scalars_per_prior_token=2,
        )


def test_fsq_tokenization_one_hot_uses_prior_token_vocab_size():
    tokenization = FSQTokenization(
        num_fsq_levels=3,
        num_fsq_scalars=4,
        fsq_scalars_per_prior_token=2,
    )
    prior_tokens = torch.tensor([[0, 8]])

    one_hot = tokenization.one_hot_prior_tokens(prior_tokens)

    assert one_hot.shape == (1, 2, 9)
    assert torch.equal(
        one_hot[0, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert torch.equal(
        one_hot[0, 1],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_discrete_latent_decoder_uses_only_quantizer_and_decoder():
    module = nn.Module()
    module.decoder = DummyDecoder()
    module.quantizer = DummyQuantizer()
    module.latent_key = "latent"
    decoder = make_discrete_latent_decoder(module)
    td = TensorDict({"state": torch.ones(1, 4)}, batch_size=1)

    decoded = decoder.decode(td, decoder.indices_to_codes(torch.tensor([[0, 1, 2, 3]])))

    assert torch.equal(decoded, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert all(not parameter.requires_grad for parameter in decoder.parameters())


def test_discrete_latent_decoder_keeps_frozen_parts_in_eval_mode():
    module = nn.Module()
    module.decoder = DummyDecoder()
    module.quantizer = DummyQuantizer()

    decoder = make_discrete_latent_decoder(module)
    decoder.train(True)

    assert decoder.training
    assert not decoder.decoder.training
    assert not decoder.quantizer.training


def test_discrete_latent_target_encoder_writes_prior_token_indices():
    module = nn.Module()
    module.encoder = DummyEncoder()
    module.quantizer = DummyQuantizer()
    tokenization = FSQTokenization(
        num_fsq_levels=5,
        num_fsq_scalars=4,
        fsq_scalars_per_prior_token=2,
    )
    target_encoder = make_discrete_latent_target_encoder(
        module,
        tokenization=tokenization,
        target_key="target_latent",
    )
    td = TensorDict({"obs": torch.tensor([[0, 1, 2, 3]])}, batch_size=1)

    out = target_encoder(td)

    assert torch.equal(out["target_latent"], torch.tensor([[5, 17]]))
    assert all(not parameter.requires_grad for parameter in target_encoder.parameters())


def test_discrete_latent_target_encoder_keeps_frozen_parts_in_eval_mode():
    module = nn.Module()
    module.encoder = DummyEncoder()
    module.quantizer = DummyQuantizer()
    tokenization = FSQTokenization(
        num_fsq_levels=5,
        num_fsq_scalars=4,
        fsq_scalars_per_prior_token=2,
    )

    target_encoder = make_discrete_latent_target_encoder(
        module,
        tokenization=tokenization,
        target_key="target_latent",
    )
    target_encoder.train(True)

    assert target_encoder.training
    assert not target_encoder.encoder.training
    assert not target_encoder.quantizer.training


def test_discrete_latent_target_encoder_uses_base_rollout_context_lifecycle():
    target_encoder_cls = discrete_latent_module.DiscreteLatentTargetEncoder

    assert (
        target_encoder_cls.reset_rollout_context
        is ProtoMotionsTensorDictModule.reset_rollout_context
    )
    assert (
        target_encoder_cls.rollout_context_keys
        is ProtoMotionsTensorDictModule.rollout_context_keys
    )


def test_pretrained_discrete_latent_adapters_delegate_to_loader(monkeypatch):
    module = nn.Module()
    module.decoder = DummyDecoder()
    module.encoder = DummyEncoder()
    module.quantizer = DummyQuantizer()
    config = SimpleNamespace(freeze=False)
    device = torch.device("cpu")
    calls = []

    def fake_load_pretrained_model_module(load_config, device):
        calls.append((load_config, device))
        return module

    monkeypatch.setattr(
        discrete_latent_module,
        "load_pretrained_model_module",
        fake_load_pretrained_model_module,
    )

    decoder = load_pretrained_discrete_latent_decoder(config, device=device)
    target_encoder = load_pretrained_discrete_latent_target_encoder(
        config,
        tokenization=FSQTokenization(
            num_fsq_levels=5,
            num_fsq_scalars=4,
            fsq_scalars_per_prior_token=2,
        ),
        target_key="target_latent",
        device=device,
    )

    assert decoder.decoder is module.decoder
    assert target_encoder.encoder is module.encoder
    assert all(call == (config, device) for call in calls)
    assert len(calls) == 2
    assert any(parameter.requires_grad for parameter in decoder.parameters())
    assert any(parameter.requires_grad for parameter in target_encoder.parameters())


def test_discrete_latent_adapters_fail_loudly_on_missing_parts():
    with pytest.raises(TypeError, match="decoder"):
        make_discrete_latent_decoder(nn.Module())

    with pytest.raises(TypeError, match="encoder"):
        make_discrete_latent_target_encoder(
            nn.Module(),
            tokenization=FSQTokenization(
                num_fsq_levels=5,
                num_fsq_scalars=4,
                fsq_scalars_per_prior_token=2,
            ),
            target_key="target_latent",
        )
