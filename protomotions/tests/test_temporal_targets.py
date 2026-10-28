# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal targets must follow time, not shuffled or env-major adjacency."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.common.temporal import adjacent_targets, masked_mse
from protomotions.agents.common.supervision import SupervisionLossConfig
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.supervised.masked_mimic_config import TemporalLossConfig
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel
from protomotions.agents.utils.data import swap_and_flatten01
from protomotions.tests.test_conditional_vae import _model, _batch, _mlp


@pytest.mark.parametrize("forward", [False, True])
def test_targets_preserve_env_identity_and_boundaries(forward):
    values = torch.tensor(
        [[[1.0], [100.0]], [[2.0], [200.0]], [[3.0], [300.0]]], requires_grad=True
    )
    dones = torch.tensor([[False, True], [True, False], [False, False]])
    target, valid = adjacent_targets(values, dones, forward=forward)
    expected = (
        [[[2.0], [0.0]], [[0.0], [300.0]], [[0.0], [0.0]]]
        if forward
        else [[[0.0], [0.0]], [[1.0], [0.0]], [[0.0], [200.0]]]
    )
    torch.testing.assert_close(target, torch.tensor(expected))
    assert not target.requires_grad
    assert valid.sum() == 2
    order = torch.tensor([4, 0, 3, 2, 1, 5])
    flat, mask = swap_and_flatten01(target)[order], swap_and_flatten01(valid)[order]
    current = swap_and_flatten01(values.detach())[order]
    expected_delta = torch.where(current < 100, 1.0, 100.0)
    torch.testing.assert_close(
        (flat - current)[mask], (expected_delta if forward else -expected_delta)[mask]
    )
    torch.testing.assert_close(
        flat[mask].sort(dim=0).values,
        torch.tensor([[2.0], [300.0]]) if forward else torch.tensor([[1.0], [200.0]]),
    )


def test_masked_loss_averages_valid_rows_with_zero_gradient_elsewhere():
    prediction = torch.tensor([[1.0, 3.0], [5.0, 5.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    loss = masked_mse(prediction, target, torch.tensor([True, False]))
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))


def test_masked_loss_propagates_nonfinite_prediction_on_invalid_row():
    prediction = torch.tensor([[1.0, 3.0], [float("nan"), 0.0]])
    target = torch.zeros(2, 2)
    loss = masked_mse(prediction, target, torch.tensor([True, False]))
    assert torch.isnan(loss)


@pytest.mark.parametrize("forward", [False, True])
def test_no_valid_transition_is_finite_grad_connected_zero(forward):
    prediction = torch.randn(3, 2, requires_grad=True)
    loss = masked_mse(prediction, torch.randn(3, 2), torch.zeros(3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert prediction.grad.count_nonzero() == 0
    _, valid = adjacent_targets(torch.ones(1, 2, 3), torch.zeros(1, 2), forward=forward)
    assert not valid.any()


def test_model_targets_use_posterior_not_executed_actions_and_replay_noise():
    config = _model().config
    config.vae.kld_schedule = None
    config.temporal_losses = [
        TemporalLossConfig("privileged_action", 0.05),
        TemporalLossConfig("solution_features", 0.001),
        TemporalLossConfig(
            "decoded_action",
            0.001,
            prediction_key="predicted_next_action",
            forward=True,
        ),
    ]
    config.trunk.models.extend(
        [
            _mlp(["decoded_action", "vae_latent"], "solution_features"),
            _mlp(["solution_features"], "predicted_next_action"),
        ]
    )
    config.trunk.out_keys.extend(
        ["solution_features", "predicted_next_action"]
    )
    model = MaskedMimicModel(config).eval()
    model.reset_rollout_context(num_envs=5, device="cpu")
    original_noise = model.vae_noise.clone()
    outputs = [model(_batch()) for _ in range(3)]
    rollout = {
        key: torch.stack([out[key].detach() for out in outputs])
        for key in model.experience_buffer_keys()
    }
    rollout["action"] = torch.full_like(rollout["privileged_action"], 999.0)
    rollout["dones"] = torch.zeros(3, 5, dtype=torch.bool)
    rollout["dones"][0, 1] = True
    targets = model.build_training_targets(rollout)
    torch.testing.assert_close(
        targets["previous_target_privileged_action"][1, 0],
        rollout["privileged_action"][0, 0],
    )
    assert not targets["previous_target_valid"][1, 1]
    assert not targets["next_target_valid"][-1].any()
    torch.testing.assert_close(
        targets["next_target_decoded_action"][0, 0],
        rollout["decoded_action"][1, 0],
    )
    batch = _batch()
    noise = batch["vae_noise"].clone()
    out = model(batch)
    for key, value in targets.items():
        out[key] = value[1]
    loss, logs = model.compute_model_loss(out, 0, out["action"].sum() * 0)
    assert torch.isfinite(loss) and len(logs) == 3
    loss.backward()
    for key in (
        "model/temporal_previous_privileged_action",
        "model/temporal_previous_solution_features",
        "model/temporal_next_predicted_next_action",
    ):
        assert logs[key] >= 0
    torch.testing.assert_close(model.vae_noise, original_noise)
    torch.testing.assert_close(out["vae_noise"], noise)


@pytest.mark.parametrize("coefficient", [-1.0, float("inf"), float("nan")])
def test_loss_coefficients_validated(coefficient):
    with pytest.raises(ValueError):
        TemporalLossConfig("key", coefficient)


def test_loss_sources_must_be_declared():
    config = _model().config
    config.temporal_losses = [TemporalLossConfig("hidden", 1)]
    with pytest.raises(ValueError, match="declared model output"):
        MaskedMimicModel(config)
    config.temporal_losses = [
        TemporalLossConfig("decoded_action", 1, prediction_key="hidden")
    ]
    with pytest.raises(ValueError, match="declared model output"):
        MaskedMimicModel(config)


def test_duplicate_key_direction_pairs_rejected():
    config = _model().config
    config.temporal_losses = [
        TemporalLossConfig("decoded_action", 1),
        TemporalLossConfig("decoded_action", 2),
    ]
    with pytest.raises(ValueError, match="unique"):
        MaskedMimicModel(config)
    config.temporal_losses = [
        TemporalLossConfig("decoded_action", 1),
        TemporalLossConfig("decoded_action", 1, forward=True),
    ]
    MaskedMimicModel(config)


def _temporal_model():
    config = _model().config
    config.vae.kld_schedule = None
    config.temporal_losses = [
        TemporalLossConfig("privileged_action", 0.05),
        TemporalLossConfig(
            "decoded_action",
            0.001,
            prediction_key="predicted_next_action",
            forward=True,
        ),
    ]
    config.trunk.models.append(_mlp(["decoded_action"], "predicted_next_action"))
    config.trunk.out_keys.append("predicted_next_action")
    model = MaskedMimicModel(config)
    model.reset_rollout_context(num_envs=5, device="cpu")
    return model


class _RecordingBuffer:
    def __init__(self):
        self.data = {}

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()


class _GoalExpert:
    in_keys = ["goals"]
    deployable_action_key = "mean_action"

    def forward_deployable(self, tensordict):
        tensordict["mean_action"] = torch.full((tensordict.batch_size[0], 3), 7.0)
        return tensordict


def _supervised_agent(model, rollout_actor):
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        rollout_actor=rollout_actor,
        loss=SupervisionLossConfig(target_key="expert_actions"),
    )
    agent.model = model
    agent.training_model = model
    agent.model_output_keys = model.experience_buffer_keys()
    agent.expert_model = None
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.experience_buffer = _RecordingBuffer()
    agent.calculate_extra_loss = lambda batch, actions: (torch.zeros(()), {})
    return agent


@pytest.mark.parametrize("expert_drives", [False, True])
def test_collect_rollout_stores_student_temporal_sources(expert_drives):
    agent = _supervised_agent(_temporal_model(), RolloutActor.STUDENT)
    obs_td = _batch().exclude("vae_noise")
    if expert_drives:
        agent.expert_model = agent.expert_actor = _GoalExpert()
        agent.expert_actor_in_keys = ["goals"]
        agent._expert_episode_mask = torch.ones(5, dtype=torch.bool)
        obs_td["expert_goals"] = obs_td["goals"]
    else:
        obs_td["expert_actions"] = torch.zeros(5, 3)

    out = SupervisedAgent.collect_rollout_step(agent, obs_td, step=2)

    stored = agent.experience_buffer.data
    for key in ("privileged_action", "decoded_action", "predicted_next_action"):
        assert (key, 2) in stored
    if expert_drives:
        # Execute teacher actions but retain student predictions for temporal losses.
        torch.testing.assert_close(out["action"], torch.full((5, 3), 7.0))
        assert not torch.allclose(stored[("privileged_action", 2)], out["action"])
    else:
        torch.testing.assert_close(out["action"], stored[("privileged_action", 2)])


def test_supervised_step_applies_temporal_losses_from_replayed_batch():
    model = _temporal_model()
    agent = _supervised_agent(model, RolloutActor.STUDENT)
    rollout = {
        key: torch.stack([model(_batch())[key].detach() for _ in range(3)])
        for key in model.experience_buffer_keys()
    }
    rollout["dones"] = torch.zeros(3, 5, dtype=torch.bool)
    targets = model.build_training_targets(rollout)
    batch = {key: value[1] for key, value in {**rollout, **targets}.items()}
    batch.update(_batch().to_dict())
    batch["expert_actions"] = torch.randn(5, 3)

    loss, log_dict = agent.supervised_step(batch)

    temporal_keys = [
        "model/temporal_previous_privileged_action",
        "model/temporal_next_predicted_next_action",
    ]
    assert all(log_dict[key] > 0 for key in temporal_keys)
    torch.testing.assert_close(
        log_dict["supervised/model_loss"], sum(log_dict[key] for key in temporal_keys)
    )
    torch.testing.assert_close(
        loss.detach(), log_dict["supervised/loss"] + log_dict["supervised/model_loss"]
    )
