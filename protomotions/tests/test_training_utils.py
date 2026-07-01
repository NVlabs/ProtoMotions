# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for training utilities that can run without distributed launch."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.utils.training import (
    aggregate_scalar_metrics,
    bounds_loss,
    get_activation_func,
    handle_model_grad_clipping,
)


class _GatherFabric:
    world_size = 2
    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0

    def all_gather(self, tensor):
        self.calls += 1
        if self.calls == 1:
            return torch.tensor([2.0, 6.0])
        if torch.allclose(tensor.float(), torch.tensor(4.0)):
            return torch.tensor([4.0, 8.0])
        return torch.tensor([1.0, 3.0])


class _SingleFabric:
    world_size = 1
    device = torch.device("cpu")


class _ClipFabric:
    def __init__(self):
        self.calls = []

    def clip_gradients(self, model, optimizer, max_norm, error_if_nonfinite):
        self.calls.append((model, optimizer, max_norm, error_if_nonfinite))
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            error_if_nonfinite=error_if_nonfinite,
        )


def _grad_config(**kwargs):
    defaults = {
        "check_grad_mag": True,
        "fail_on_bad_grads": False,
        "gradient_clip_val": 0.1,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_bounds_loss_penalizes_only_values_outside_soft_bounds():
    mu = torch.tensor([[-2.0, -0.5, 0.0, 1.5], [0.25, 0.5, 1.0, -1.0]])

    loss = bounds_loss(mu)

    assert torch.allclose(loss, torch.tensor([1.25, 0.0]))


def test_handle_model_grad_clipping_reports_and_clips_norms_without_fabric():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1.0)
    output = model(torch.ones(4, 2)).sum()
    output.backward()

    metrics = handle_model_grad_clipping(
        config=_grad_config(gradient_clip_val=0.5),
        fabric=None,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        model_name="actor",
    )

    assert metrics["actor/grad_norm_before_clip"] > 0.5
    assert metrics["actor/grad_norm_after_clip"] <= 0.5001
    assert metrics["actor/bad_grads_count"] == 0


def test_handle_model_grad_clipping_zeros_bad_grads_when_configured_to_continue():
    model = nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))

    metrics = handle_model_grad_clipping(
        config=_grad_config(gradient_clip_val=0.0, fail_on_bad_grads=False),
        fabric=None,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        model_name="critic",
    )

    assert metrics["critic/bad_grads_count"] == 1
    assert all(
        torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        for parameter in model.parameters()
    )


def test_handle_model_grad_clipping_can_fail_or_ignore_large_finite_grads():
    nan_model = nn.Linear(2, 1)
    for parameter in nan_model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))

    with pytest.raises(ValueError, match="NaN gradient in actor"):
        handle_model_grad_clipping(
            config=_grad_config(gradient_clip_val=0.0, fail_on_bad_grads=True),
            fabric=None,
            model=nan_model,
            optimizer=torch.optim.SGD(nan_model.parameters(), lr=0.1),
            model_name="actor",
        )

    large_grad_model = nn.Linear(1, 1, bias=False)
    large_grad_model.weight.grad = torch.full_like(large_grad_model.weight, 2_000_000.0)
    metrics = handle_model_grad_clipping(
        config=_grad_config(
            check_grad_mag=False,
            gradient_clip_val=0.0,
        ),
        fabric=None,
        model=large_grad_model,
        optimizer=torch.optim.SGD(large_grad_model.parameters(), lr=0.1),
        model_name="large",
    )

    assert metrics["large/bad_grads_count"] == 0
    assert large_grad_model.weight.grad.item() == 2_000_000.0


def test_handle_model_grad_clipping_uses_fabric_when_available():
    model = nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fabric = _ClipFabric()

    metrics = handle_model_grad_clipping(
        config=_grad_config(gradient_clip_val=0.25),
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        model_name="fabric",
    )

    assert fabric.calls == [(model, optimizer, 0.25, True)]
    assert metrics["fabric/grad_norm_after_clip"] <= 0.2501


def test_aggregate_scalar_metrics_keeps_strings_and_uses_weighted_average():
    metrics = aggregate_scalar_metrics(
        {
            "loss": torch.tensor([0.0, 2.0]),
            "count": 4,
            "phase": "train",
        },
        fabric=_GatherFabric(),
        weight=2,
    )

    assert metrics["loss"] == 2.5
    assert metrics["count"] == 7.0
    assert metrics["phase"] == "train"


def test_aggregate_scalar_metrics_single_rank_uses_local_values():
    metrics = aggregate_scalar_metrics(
        {
            "loss": torch.tensor([1.0, 3.0]),
            "scalar": torch.tensor(7.0),
            "count": 4,
            "phase": "eval",
        },
        fabric=_SingleFabric(),
        weight=8,
    )

    assert metrics == {"loss": 2.0, "scalar": 7.0, "count": 4.0, "phase": "eval"}


def test_get_activation_func_returns_modules_and_functionals_and_rejects_unknowns():
    activation = get_activation_func("silu", return_type="nn")
    functional = get_activation_func("silu", return_type="functional")
    values = torch.tensor([-1.0, 0.0, 1.0])

    assert isinstance(activation, nn.SiLU)
    assert torch.allclose(activation(values), functional(values))
    assert isinstance(get_activation_func("tanh", return_type="nn"), nn.Tanh)
    assert isinstance(get_activation_func("relu", return_type="nn"), nn.ReLU)
    assert isinstance(get_activation_func("elu", return_type="nn"), nn.ELU)
    assert isinstance(get_activation_func("gelu", return_type="nn"), nn.GELU)
    assert isinstance(get_activation_func("identity", return_type="nn"), nn.Identity)
    assert isinstance(get_activation_func("mish", return_type="nn"), nn.Mish)
    assert torch.equal(
        get_activation_func("identity", return_type="functional")(values),
        values,
    )

    try:
        get_activation_func("not_an_activation")
    except NotImplementedError as error:
        assert "not_an_activation" in str(error)
    else:
        raise AssertionError("Expected unknown activation name to fail")

    with pytest.raises(NotImplementedError, match="Return type bad"):
        get_activation_func("relu", return_type="bad")
