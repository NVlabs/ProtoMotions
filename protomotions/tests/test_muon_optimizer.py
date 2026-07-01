# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generalized Muon optimizer parameter splitting."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.base_agent.config import MuonWithAuxAdamConfig, OptimizerConfig
from protomotions.agents.optimizer.factory import (
    instantiate_optimizer,
    optimizer_learning_rate,
    scale_optimizer_learning_rates,
)
from protomotions.agents.optimizer import muon as muon_module
from protomotions.agents.optimizer.muon import (
    Muon,
    MuonWithAuxAdam,
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
    _distributed_rank_world,
    _sync_updated_params,
    adam_update,
    make_muon_param_groups,
    muon_update,
    split_muon_parameters,
    zeropower_via_newtonschulz5,
)
from protomotions.agents.ppo.agent import PPO


class _FallbackModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.adapter = nn.Linear(4, 4)
        self.embed = nn.Embedding(3, 4)
        self.head = nn.Linear(4, 2)
        self.scalar = nn.Parameter(torch.ones(()))
        self.frozen = nn.Parameter(torch.ones(2, 2), requires_grad=False)

    def muon_adam_fallback_modules(self):
        return [self.adapter]


def test_muon_with_aux_adam_uses_generic_fallback_rules():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Linear(4, 4),
        nn.LayerNorm(4),
        nn.Linear(4, 2),
    )

    optimizer = MuonWithAuxAdam(
        params=model,
        lr=0.012,
        weight_decay=0.34,
        momentum=0.91,
        adam_lr=0.056,
        adam_betas=(0.8, 0.88),
        adam_eps=1e-6,
        adam_weight_decay=0.78,
    )

    hidden_group = next(group for group in optimizer.param_groups if group["use_muon"])
    adam_group = next(group for group in optimizer.param_groups if not group["use_muon"])

    assert hidden_group["lr"] == 0.012
    assert hidden_group["weight_decay"] == 0.34
    assert hidden_group["momentum"] == 0.91
    assert adam_group["lr"] == 0.056
    assert adam_group["betas"] == (0.8, 0.88)
    assert adam_group["eps"] == 1e-6
    assert adam_group["weight_decay"] == 0.78
    assert set(hidden_group["params"]) == {model[1].weight}
    assert set(adam_group["params"]) == {
        model[0].weight,
        model[0].bias,
        model[1].bias,
        model[2].weight,
        model[2].bias,
        model[3].weight,
        model[3].bias,
    }


def test_split_muon_parameters_handles_declared_and_named_fallbacks():
    model = _FallbackModule()

    hidden_params, adam_params, hidden_names, adam_names = split_muon_parameters(
        model,
        adam_fallback_module_patterns=["head"],
        adam_fallback_parameter_patterns=["*.bias"],
        use_adam_for_sequential_projections=False,
    )

    assert hidden_params == [model.hidden.weight]
    assert hidden_names == ["hidden.weight"]
    assert set(adam_params) == {
        model.hidden.bias,
        model.adapter.weight,
        model.adapter.bias,
        model.embed.weight,
        model.head.weight,
        model.head.bias,
        model.scalar,
    }
    assert "frozen" not in hidden_names
    assert "frozen" not in adam_names


def test_split_muon_parameters_supports_plain_iterables_and_subset_filtering():
    matrix = nn.Parameter(torch.ones(2, 2))
    vector = nn.Parameter(torch.ones(2))
    skipped = nn.Parameter(torch.ones(2, 2))

    hidden_params, adam_params, hidden_names, adam_names = split_muon_parameters(
        [matrix, vector, skipped],
        param_subset=[matrix, vector],
    )

    assert hidden_params == [matrix]
    assert adam_params == [vector]
    assert hidden_names == ["0"]
    assert adam_names == ["1"]


def test_make_muon_param_groups_raises_for_empty_splits():
    with pytest.raises(ValueError, match="custom empty"):
        make_muon_param_groups([], [], empty_error="custom empty")


def test_optimizer_factory_passes_module_to_muon_splitter():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    optimizer_config = MuonWithAuxAdamConfig(
        lr=0.012,
        weight_decay=0.34,
    )

    optimizer = instantiate_optimizer(optimizer_config, model)
    hidden_group = next(group for group in optimizer.param_groups if group["use_muon"])
    adam_group = next(group for group in optimizer.param_groups if not group["use_muon"])

    assert len(hidden_group["params"]) == 1
    assert hidden_group["params"][0] is model[1].weight
    assert set(adam_group["params"]) == {
        model[0].weight,
        model[0].bias,
        model[1].bias,
        model[2].weight,
        model[2].bias,
    }
    assert optimizer_learning_rate(optimizer_config, optimizer) == 0.012


def test_optimizer_factory_honors_explicit_param_subset_for_muon():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    optimizer_config = MuonWithAuxAdamConfig(
        lr=0.012,
    )
    params = [model[1].weight, model[1].bias]

    optimizer = instantiate_optimizer(optimizer_config, model, params=params)
    hidden_group = next(group for group in optimizer.param_groups if group["use_muon"])
    adam_group = next(group for group in optimizer.param_groups if not group["use_muon"])

    assert len(hidden_group["params"]) == 1
    assert hidden_group["params"][0] is model[1].weight
    assert len(adam_group["params"]) == 1
    assert adam_group["params"][0] is model[1].bias


def test_optimizer_factory_uses_explicit_params_for_standard_optimizers():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    optimizer_config = OptimizerConfig(_target_="torch.optim.AdamW", lr=0.004)
    params = [model[1].weight]

    optimizer = instantiate_optimizer(optimizer_config, model, params=params)

    assert len(optimizer.param_groups[0]["params"]) == 1
    assert optimizer.param_groups[0]["params"][0] is model[1].weight
    assert optimizer_learning_rate(optimizer_config, optimizer) == 0.004


def test_optimizer_factory_supports_dict_configs_and_default_module_params():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    optimizer_config = {"_target_": "torch.optim.SGD", "lr": 0.02}

    optimizer = instantiate_optimizer(optimizer_config, model)

    assert len(optimizer.param_groups[0]["params"]) == 4
    assert optimizer_learning_rate(optimizer_config, optimizer) == 0.02
    assert optimizer_learning_rate({}, optimizer) == 0.02


def test_optimizer_factory_uses_combined_muon_config_directly():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    optimizer_config = MuonWithAuxAdamConfig(
        lr=0.012,
        weight_decay=0.34,
    )

    optimizer = instantiate_optimizer(optimizer_config, model)
    hidden_group = next(group for group in optimizer.param_groups if group["use_muon"])

    assert isinstance(optimizer, MuonWithAuxAdam)
    assert hidden_group["lr"] == 0.012
    assert hidden_group["weight_decay"] == 0.34


def test_muon_orders_hidden_params_by_element_count():
    shape_tuple_larger = nn.Parameter(torch.ones(1000, 1))
    numel_larger = nn.Parameter(torch.ones(2, 1000))

    muon_optimizer = Muon([shape_tuple_larger, numel_larger])
    muon_params = muon_optimizer.param_groups[0]["params"]

    assert muon_params[0] is numel_larger
    assert muon_params[1] is shape_tuple_larger

    aux_optimizer = MuonWithAuxAdam(
        [
            {"params": [shape_tuple_larger, numel_larger], "use_muon": True},
            {"params": [nn.Parameter(torch.ones(3))], "use_muon": False},
        ]
    )
    aux_muon_params = next(
        group["params"] for group in aux_optimizer.param_groups if group["use_muon"]
    )

    assert aux_muon_params[0] is numel_larger
    assert aux_muon_params[1] is shape_tuple_larger


def test_scale_optimizer_learning_rates_preserves_group_ratios():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    optimizer = MuonWithAuxAdam(
        params=model,
        lr=0.012,
        adam_lr=0.003,
    )

    scale_optimizer_learning_rates(optimizer, old_lr=0.012, new_lr=0.006)

    hidden_group = next(group for group in optimizer.param_groups if group["use_muon"])
    adam_group = next(group for group in optimizer.param_groups if not group["use_muon"])
    assert hidden_group["lr"] == 0.006
    assert adam_group["lr"] == 0.0015


def test_scale_optimizer_learning_rates_handles_noop_and_rejects_zero_old_lr():
    model = nn.Linear(2, 1)
    optimizer = instantiate_optimizer(
        {"_target_": "torch.optim.SGD", "lr": 0.1},
        model,
    )

    scale_optimizer_learning_rates(optimizer, old_lr=0.1, new_lr=0.1)
    assert optimizer.param_groups[0]["lr"] == 0.1

    with pytest.raises(ValueError, match="old_lr=0"):
        scale_optimizer_learning_rates(optimizer, old_lr=0.0, new_lr=0.03)


def test_ppo_adaptive_lr_update_preserves_optimizer_group_ratios():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    actor_optimizer = MuonWithAuxAdam(params=model, lr=0.012, adam_lr=0.003)
    critic_optimizer = MuonWithAuxAdam(params=model, lr=0.02, adam_lr=0.005)
    agent = SimpleNamespace(
        actor_lr=0.012,
        critic_lr=0.02,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        config=SimpleNamespace(
            adaptive_lr=SimpleNamespace(
                desired_kl=0.01,
                min_lr=1e-6,
                max_lr=1.0,
            )
        ),
    )

    PPO._update_learning_rate(agent, kl_mean=0.03)

    actor_hidden = next(
        group for group in actor_optimizer.param_groups if group["use_muon"]
    )
    actor_adam = next(
        group for group in actor_optimizer.param_groups if not group["use_muon"]
    )
    critic_hidden = next(
        group for group in critic_optimizer.param_groups if group["use_muon"]
    )
    critic_adam = next(
        group for group in critic_optimizer.param_groups if not group["use_muon"]
    )
    assert actor_hidden["lr"] == pytest.approx(0.008)
    assert actor_adam["lr"] == pytest.approx(0.002)
    assert critic_hidden["lr"] == pytest.approx(0.02 / 1.5)
    assert critic_adam["lr"] == pytest.approx(0.005 / 1.5)


def test_distributed_helpers_use_rank_world_and_sync_owned_param_buckets(monkeypatch):
    broadcasts = []

    monkeypatch.setattr(muon_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(muon_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(muon_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(muon_module.dist, "get_world_size", lambda: 3)
    def fake_broadcast(tensor, src):
        broadcasts.append((tensor.clone(), src))
        tensor.copy_(
            torch.arange(
                tensor.numel(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            + (src + 1) * 100
        )

    monkeypatch.setattr(muon_module.dist, "broadcast", fake_broadcast)

    assert _distributed_rank_world() == (1, 3)
    params = [
        torch.tensor([0.0, 1.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0, 4.0, 5.0]),
        torch.tensor([6.0]),
    ]

    _sync_updated_params(params, world_size=1)
    assert broadcasts == []

    _sync_updated_params(params, world_size=2)

    assert [(src, tensor.tolist()) for tensor, src in broadcasts] == [
        (0, [0.0, 1.0, 3.0, 4.0, 5.0]),
        (1, [2.0, 6.0]),
    ]
    assert params[0].tolist() == [100.0, 101.0]
    assert params[1].tolist() == [200.0]
    assert params[2].tolist() == [102.0, 103.0, 104.0]
    assert params[3].tolist() == [201.0]


def test_sync_updated_params_uses_stable_mixed_dtype_bucket_order(monkeypatch):
    broadcasts = []

    def fake_broadcast(tensor, src):
        broadcasts.append((src, tensor.dtype, tensor.numel()))

    monkeypatch.setattr(muon_module.dist, "broadcast", fake_broadcast)

    params = [
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([2.0], dtype=torch.bfloat16),
        torch.tensor([3.0], dtype=torch.bfloat16),
    ]

    _sync_updated_params(params, world_size=2)

    assert broadcasts == [
        (0, torch.bfloat16, 1),
        (0, torch.float32, 1),
        (1, torch.bfloat16, 1),
        (1, torch.float32, 1),
    ]


def test_sync_updated_params_handles_uneven_and_empty_owner_buckets(monkeypatch):
    broadcasts = []

    def fake_broadcast(tensor, src):
        broadcasts.append((src, tensor.numel()))
        tensor.copy_(torch.full_like(tensor, src + 10))

    monkeypatch.setattr(muon_module.dist, "broadcast", fake_broadcast)

    uneven_params = [
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
        torch.tensor([4.0]),
    ]

    _sync_updated_params(uneven_params, world_size=2)

    assert broadcasts == [(0, 3), (1, 2)]
    assert [param.item() for param in uneven_params] == [10.0, 11.0, 10.0, 11.0, 10.0]

    broadcasts.clear()
    single_param = [torch.tensor([5.0])]

    _sync_updated_params(single_param, world_size=3)

    assert broadcasts == [(0, 1)]
    assert single_param[0].item() == 10.0


def test_muon_math_helpers_cover_shapes_and_bias_correction():
    tall = torch.arange(6, dtype=torch.float32).view(3, 2)
    assert zeropower_via_newtonschulz5(tall, steps=1).shape == tall.shape

    with pytest.raises(AssertionError):
        zeropower_via_newtonschulz5(torch.ones(3), steps=1)

    grad = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 4)
    momentum = torch.zeros_like(grad)
    update = muon_update(grad.clone(), momentum, beta=0.5, ns_steps=1, nesterov=False)
    assert update.shape == (1, 8)
    assert torch.count_nonzero(momentum) > 0

    exp_avg = torch.zeros(2)
    exp_avg_sq = torch.zeros(2)
    adam_step = adam_update(
        torch.tensor([1.0, 2.0]),
        exp_avg,
        exp_avg_sq,
        step=1,
        betas=(0.5, 0.5),
        eps=1e-8,
    )
    assert torch.allclose(adam_step, torch.ones(2), atol=1e-5)


def test_muon_and_single_device_muon_step_initialize_state_and_return_closure_loss():
    param = nn.Parameter(torch.eye(2))
    optimizer = Muon([param], lr=0.01, weight_decay=0.1, momentum=0.5)
    before = param.detach().clone()

    loss = optimizer.step(lambda: torch.tensor(3.0))

    assert loss.item() == 3.0
    assert torch.allclose(param, before * 0.999)
    assert torch.equal(param.grad, torch.zeros_like(param))
    assert "momentum_buffer" in optimizer.state[param]

    single_param = nn.Parameter(torch.eye(2))
    single_param.grad = torch.ones_like(single_param)
    single_optimizer = SingleDeviceMuon(
        [single_param],
        lr=0.01,
        weight_decay=0.1,
        momentum=0.5,
    )
    single_before = single_param.detach().clone()

    single_loss = single_optimizer.step(lambda: torch.tensor(4.0))

    assert single_loss.item() == 4.0
    assert "momentum_buffer" in single_optimizer.state[single_param]
    assert not torch.allclose(single_param, single_before)

    missing_grad_param = nn.Parameter(torch.eye(2))
    missing_grad_optimizer = SingleDeviceMuon([missing_grad_param])
    missing_grad_optimizer.step()
    assert torch.equal(missing_grad_param.grad, torch.zeros_like(missing_grad_param))


def test_muon_with_aux_adam_direct_groups_step_both_update_paths():
    muon_param = nn.Parameter(torch.eye(2))
    adam_param = nn.Parameter(torch.ones(2))

    optimizer = MuonWithAuxAdam(
        [
            {"params": [muon_param], "use_muon": True},
            {
                "params": [adam_param],
                "use_muon": False,
                "lr": 0.01,
                "weight_decay": 0.1,
            },
        ]
    )

    loss = optimizer.step(lambda: torch.tensor(5.0))

    assert loss.item() == 5.0
    assert torch.equal(muon_param.grad, torch.zeros_like(muon_param))
    assert "momentum_buffer" in optimizer.state[muon_param]
    assert torch.equal(adam_param.grad, torch.zeros_like(adam_param))
    assert optimizer.state[adam_param]["step"] == 1
    assert torch.allclose(adam_param, torch.full((2,), 0.999))


def test_muon_with_aux_adam_requires_params_or_groups():
    with pytest.raises(ValueError, match="requires params or param_groups"):
        MuonWithAuxAdam()


def test_single_device_muon_with_aux_adam_defaults_and_steps_all_groups():
    muon_param = nn.Parameter(torch.eye(2))
    adam_param = nn.Parameter(torch.ones(2))
    optimizer = SingleDeviceMuonWithAuxAdam(
        [
            {"params": [muon_param], "use_muon": True},
            {"params": [adam_param], "use_muon": False},
        ]
    )

    loss = optimizer.step(lambda: torch.tensor(6.0))

    assert loss.item() == 6.0
    assert torch.equal(muon_param.grad, torch.zeros_like(muon_param))
    assert torch.equal(adam_param.grad, torch.zeros_like(adam_param))
    assert "momentum_buffer" in optimizer.state[muon_param]
    assert optimizer.state[adam_param]["step"] == 1
