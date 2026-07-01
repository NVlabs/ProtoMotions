# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared MLP modules."""

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.config import (
    MLPLayerConfig,
    MLPWithConcatConfig,
)
from protomotions.agents.common.mlp import (
    MLPWithConcat,
    build_mlp,
)


def test_build_mlp_uses_lazy_layers_and_first_layer_norm_only():
    config = MLPWithConcatConfig(
        in_keys=["obs"],
        out_keys=["out"],
        num_out=2,
        layers=[
            MLPLayerConfig(units=5, activation="relu", use_layer_norm=True),
            MLPLayerConfig(units=3, activation="tanh", use_layer_norm=True),
        ],
    )

    mlp = build_mlp(config)

    assert isinstance(mlp[0], nn.LazyLinear)
    assert isinstance(mlp[-1], nn.LazyLinear)
    out = mlp(torch.ones(4, 6))

    assert out.shape == (4, 2)
    assert sum(isinstance(module, nn.LayerNorm) for module in mlp) == 1
    assert isinstance(mlp[0], nn.Linear)
    assert isinstance(mlp[-1], nn.Linear)


def test_mlp_with_concat_normalizes_concatenated_inputs_and_applies_activation():
    config = MLPWithConcatConfig(
        in_keys=["obs_a", "obs_b"],
        out_keys=["out"],
        normalize_obs=True,
        norm_clamp_value=100.0,
        output_activation="tanh",
        num_out=4,
        layers=[],
    )
    model = MLPWithConcat(config)
    model.mlp = nn.Identity()
    model.eval()
    td = TensorDict(
        {
            "obs_a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "obs_b": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        },
        batch_size=2,
    )

    out = model(td)

    expected_norm = torch.cat([td["obs_a"], td["obs_b"]], dim=-1) / torch.sqrt(
        torch.tensor(1.0 + 1e-5)
    )
    assert torch.allclose(out["norm_obs_a"], expected_norm)
    assert torch.allclose(out["out"], torch.tanh(expected_norm))


def test_mlp_with_concat_requires_explicit_single_output_key():
    with pytest.raises(AssertionError, match="obs_key"):
        MLPWithConcat(MLPWithConcatConfig(out_keys=["out"], num_out=1))

    with pytest.raises(AssertionError, match="exactly one output key"):
        MLPWithConcat(
            MLPWithConcatConfig(
                in_keys=["obs"],
                out_keys=["out_a", "out_b"],
                num_out=1,
            )
        )


def test_mlp_with_concat_exposes_common_runtime_hooks():
    model = MLPWithConcat(
        MLPWithConcatConfig(in_keys=["obs"], out_keys=["out"], num_out=1)
    )

    model.reset_rollout_context(env_ids=torch.tensor([0]), num_envs=1, device="cpu")

    assert model.rollout_context_keys() == []
    assert model.experience_buffer_keys() == ["out"]
