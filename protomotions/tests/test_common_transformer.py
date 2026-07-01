# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic Transformer TensorDict module."""

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.config import TransformerConfig
from protomotions.agents.common.transformer import Transformer


class _CaptureTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_tokens = None
        self.last_mask = None

    def forward(self, tokens, src_key_padding_mask=None):
        self.last_tokens = tokens.detach().clone()
        self.last_mask = src_key_padding_mask.detach().clone()
        return tokens + 1.0


def test_transformer_concatenates_tokens_and_builds_padding_mask():
    config = TransformerConfig(
        in_keys=["single", "sequence", "object", "single_mask", "object_mask"],
        out_keys=["transformer_out"],
        input_and_mask_mapping={
            "single": "single_mask",
            "object": "object_mask",
        },
        latent_dim=4,
        transformer_token_size=4,
        num_heads=2,
        num_layers=1,
        ff_size=8,
        dropout=0.0,
    )
    model = Transformer(config)
    capture = _CaptureTransformerEncoder()
    model.seqTransEncoder = capture
    td = TensorDict(
        {
            "single": torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ),
            "sequence": torch.ones(2, 2, 4) * 10.0,
            "object": torch.ones(2, 1, 4) * 20.0,
            "single_mask": torch.tensor([True, False]),
            "object_mask": torch.tensor(
                [
                    [True, False, False],
                    [False, False, False],
                ]
            ),
        },
        batch_size=2,
    )

    out = model(td)

    assert capture.last_tokens.shape == (2, 4, 4)
    assert torch.equal(capture.last_tokens[:, 0], td["single"])
    assert torch.equal(capture.last_tokens[:, 1:3], td["sequence"])
    assert torch.equal(capture.last_tokens[:, 3:], td["object"])
    assert torch.equal(
        capture.last_mask,
        torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, True],
            ]
        ),
    )
    assert torch.equal(out["transformer_out"], td["single"] + 1.0)


def test_transformer_expands_short_masks_to_sequence_length():
    config = TransformerConfig(
        in_keys=["sequence", "sequence_mask"],
        out_keys=["transformer_out"],
        input_and_mask_mapping={"sequence": "sequence_mask"},
        latent_dim=4,
        transformer_token_size=4,
        num_heads=2,
        num_layers=1,
        ff_size=8,
        dropout=0.0,
    )
    model = Transformer(config)
    capture = _CaptureTransformerEncoder()
    model.seqTransEncoder = capture
    td = TensorDict(
        {
            "sequence": torch.ones(2, 3, 4),
            "sequence_mask": torch.tensor([[True], [False]]),
        },
        batch_size=2,
    )

    model(td)

    assert torch.equal(
        capture.last_mask,
        torch.tensor(
            [
                [False, False, False],
                [True, True, True],
            ]
        ),
    )


def test_transformer_output_activation_is_applied_to_first_token():
    config = TransformerConfig(
        in_keys=["single"],
        out_keys=["transformer_out"],
        latent_dim=4,
        transformer_token_size=4,
        num_heads=2,
        num_layers=1,
        ff_size=8,
        dropout=0.0,
        output_activation="tanh",
    )
    model = Transformer(config)
    model.seqTransEncoder = _CaptureTransformerEncoder()
    td = TensorDict({"single": torch.zeros(2, 4)}, batch_size=2)

    out = model(td)

    assert torch.allclose(out["transformer_out"], torch.tanh(torch.ones(2, 4)))


def test_transformer_real_encoder_uses_context_and_respects_padding_mask():
    torch.manual_seed(7)
    config = TransformerConfig(
        in_keys=["query", "context", "context_mask"],
        out_keys=["transformer_out"],
        input_and_mask_mapping={"context": "context_mask"},
        latent_dim=4,
        transformer_token_size=4,
        num_heads=2,
        num_layers=1,
        ff_size=16,
        dropout=0.0,
    )
    model = Transformer(config)
    model.eval()

    query = torch.tensor([[0.2, -0.1, 0.4, -0.3]])
    context_a = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    context_b = torch.tensor([[[3.0, -2.0, 1.5, 0.5]]])

    def run(context: torch.Tensor, context_valid: bool) -> torch.Tensor:
        td = TensorDict(
            {
                "query": query.clone(),
                "context": context.clone(),
                "context_mask": torch.tensor([[context_valid]]),
            },
            batch_size=1,
        )
        with torch.no_grad():
            return model(td)["transformer_out"].clone()

    valid_a = run(context_a, context_valid=True)
    valid_b = run(context_b, context_valid=True)
    masked_a = run(context_a, context_valid=False)
    masked_b = run(context_b, context_valid=False)

    assert not torch.allclose(valid_a, valid_b)
    torch.testing.assert_close(masked_a, masked_b, atol=1e-6, rtol=1e-6)


def test_transformer_config_validates_mask_mapping_keys():
    with pytest.raises(AssertionError, match="mask key"):
        TransformerConfig(
            in_keys=["tokens"],
            out_keys=["out"],
            input_and_mask_mapping={"tokens": "missing_mask"},
        )

    with pytest.raises(AssertionError, match="input key"):
        TransformerConfig(
            in_keys=["tokens", "mask"],
            out_keys=["out"],
            input_and_mask_mapping={"missing_tokens": "mask"},
        )
