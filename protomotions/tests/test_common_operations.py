# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared module operations and initialization helpers."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.common import (
    NormObsBase,
    ObsProcessor,
    apply_module_operations,
    get_params,
    weight_init,
    weight_init_trainable,
)
from protomotions.agents.common.config import (
    ModuleOperationExpandConfig,
    ModuleOperationForwardConfig,
    ModuleOperationPermuteConfig,
    ModuleOperationReshapeConfig,
    ModuleOperationSphereProjectionConfig,
    ModuleOperationSqueezeConfig,
    ModuleOperationUnsqueezeConfig,
    NormObsBaseConfig,
    ObsProcessorConfig,
)


def _fabric():
    return SimpleNamespace(world_size=1, global_rank=0)


def test_get_params_accepts_flat_parameters_and_optimizer_param_groups():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    flat_params = [model[0].weight, model[1].weight]
    grouped_params = [
        {"params": [model[0].weight]},
        {"params": [model[1].weight, model[1].bias]},
    ]

    assert get_params(flat_params) == flat_params
    assert get_params(grouped_params) == [
        model[0].weight,
        model[1].weight,
        model[1].bias,
    ]


def test_weight_init_trainable_skips_frozen_leaf_modules():
    frozen = nn.Linear(2, 2)
    trainable = nn.Linear(2, 2)
    original_frozen_weight = frozen.weight.detach().clone()
    original_trainable_weight = trainable.weight.detach().clone()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)

    frozen.apply(lambda module: weight_init_trainable(module, orthogonal=True))
    trainable.apply(lambda module: weight_init_trainable(module, orthogonal=True))

    assert torch.equal(frozen.weight, original_frozen_weight)
    assert not torch.equal(trainable.weight, original_trainable_weight)
    assert torch.equal(trainable.bias, torch.zeros_like(trainable.bias))


def test_weight_init_resets_modules_with_reset_parameters():
    norm = nn.LayerNorm(2)
    norm.weight.data.fill_(3.0)
    norm.bias.data.fill_(4.0)

    weight_init(norm)

    assert torch.equal(norm.weight, torch.ones_like(norm.weight))
    assert torch.equal(norm.bias, torch.zeros_like(norm.bias))


def test_norm_obs_base_records_moments_while_training():
    normalizer = NormObsBase(
        NormObsBaseConfig(
            normalize_obs=True,
            norm_clamp_value=100.0,
            norm_ema_decay=None,
        )
    )
    normalizer.running_obs_norm.fabric = _fabric()
    normalizer.train()
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    assert not hasattr(normalizer.running_obs_norm, "count")

    output = normalizer(values)

    assert output.shape == values.shape
    assert normalizer.running_obs_norm.count == 3
    assert torch.allclose(
        normalizer.running_obs_norm.mean.float(),
        values.mean(dim=0) * (2.0 / 3.0),
    )


def test_apply_module_operations_handles_common_shape_operations():
    obs = torch.arange(2 * 3 * 4, dtype=torch.float).reshape(2, 3, 4)

    result = apply_module_operations(
        obs,
        [
            ModuleOperationPermuteConfig(new_order=[0, 2, 1]),
            ModuleOperationReshapeConfig(new_shape=["batch_size * 4", 3]),
            ModuleOperationUnsqueezeConfig(unsqueeze_dim=1),
            ModuleOperationExpandConfig(expand_shape=[-1, 2, -1]),
            ModuleOperationReshapeConfig(new_shape=["batch_size", 2, 4, 3]),
            ModuleOperationSphereProjectionConfig(),
        ],
        normalizer=None,
    )

    output = result["output"]
    assert output.shape == (2, 2, 4, 3)
    assert torch.allclose(output.norm(dim=-1), torch.ones(2, 2, 4))


def test_apply_module_operations_normalizes_before_forward_model():
    normalizer = NormObsBase(
        NormObsBaseConfig(normalize_obs=True, norm_clamp_value=100.0)
    )
    normalizer.running_obs_norm.fabric = _fabric()
    normalizer.eval()
    forward_model = nn.Linear(2, 1, bias=False)
    forward_model.weight.data.fill_(2.0)
    obs = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    result = apply_module_operations(
        obs,
        [ModuleOperationForwardConfig()],
        normalizer=normalizer,
        forward_model=forward_model,
    )

    assert result["output"].shape == (1, 2, 1)
    assert result["norm_obs"].shape == (1, 2, 2)
    assert torch.allclose(
        result["output"].squeeze(-1),
        torch.tensor([[8.0, 24.0]]),
        atol=1e-5,
    )


def test_apply_module_operations_rejects_unknown_operation():
    try:
        apply_module_operations(
            torch.ones(1, 2),
            [object()],
            normalizer=None,
        )
    except NotImplementedError as error:
        assert "not implemented" in str(error)
    else:
        raise AssertionError("Expected unknown module operation to fail")


def test_obs_processor_concatenates_inputs_and_writes_configured_output_key():
    processor = ObsProcessor(
        ObsProcessorConfig(
            in_keys=["a", "b"],
            out_keys=["processed"],
            module_operations=[
                ModuleOperationReshapeConfig(new_shape=["batch_size", 1, 4]),
                ModuleOperationSqueezeConfig(squeeze_dim=1),
            ],
        )
    )
    td = TensorDict(
        {
            "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "b": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        },
        batch_size=2,
    )

    out = processor(td)

    assert processor.in_keys == ["a", "b"]
    assert processor.out_keys == ["processed"]
    assert torch.equal(
        out["processed"],
        torch.tensor([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]),
    )
