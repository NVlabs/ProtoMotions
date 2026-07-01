# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MaskedMimic supervised losses."""

from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch import nn
from tensordict import TensorDict

from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel
from protomotions.agents.common.config import MLPWithConcatConfig, ModuleContainerConfig
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    PRIVILEGED_LATENT_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
)
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.supervised.config import SupervisedAgentConfig, RolloutActor
from protomotions.agents.supervised.masked_mimic_config import (
    MaskedMimicSupervisedAgentConfig,
    VAENoiseType,
)
from protomotions.utils.config_utils import clean_dict_for_storage
from protomotions.utils.hydra_replacement import get_class


class _DummyLatentPolicy(nn.Module):
    out_keys = ["action", "privileged_action", "latent"]

    def forward(self, tensordict):
        tensordict["privileged_action"] = tensordict["action"] + 10.0
        tensordict["latent"] = tensordict["model_latent"]
        return tensordict

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        return zero_loss * 0.0, {}


class _DummyModelLossPolicy(_DummyLatentPolicy):
    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        loss = tensordict["model_loss_source"].mean() + current_epoch
        return loss, {f"{log_prefix}/dummy_loss": loss.detach()}


class _DummyDiscretePriorPolicy(nn.Module):
    out_keys = ["action", LATENT_LOGITS_KEY]

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones(()))

    def forward(self, tensordict):
        batch = tensordict["action"].shape[0]
        tensordict[LATENT_LOGITS_KEY] = self.logit_scale * torch.ones(
            batch,
            2,
            3,
        )
        return tensordict

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        return zero_loss * 0.0, {}


class _DummyRolloutModel(nn.Module):
    out_keys = ["action", "mean_action"]

    def forward(self, tensordict):
        tensordict["action"] = torch.full_like(tensordict["obs"].float(), 1.0)
        tensordict["mean_action"] = tensordict["action"]
        return tensordict

    def collect_expert_rollout(self, tensordict):
        tensordict[TARGET_LATENT_KEY] = tensordict["obs"].long()
        tensordict["action"] = tensordict[TARGET_LATENT_KEY].float() + 10.0
        tensordict["mean_action"] = tensordict["action"]
        return tensordict


class _DummyNoModelExpertRollout(nn.Module):
    out_keys = ["action", "mean_action"]

    def forward(self, tensordict):
        tensordict["action"] = torch.full_like(tensordict["obs"].float(), 1.0)
        tensordict["mean_action"] = tensordict["action"]
        return tensordict


class _DummyPrivilegedStudentModel(nn.Module):
    out_keys = ["action", "mean_action", "privileged_action"]

    def forward(self, tensordict):
        tensordict["action"] = torch.full_like(tensordict["obs"].float(), 1.0)
        tensordict["mean_action"] = tensordict["action"]
        tensordict["privileged_action"] = tensordict["action"] + 100.0
        return tensordict


class _DummyExperienceBuffer:
    def __init__(self):
        self.data = {}
        self.counts = {}

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()
        self.counts[(key, step)] = self.counts.get((key, step), 0) + 1


class _DummyExternalExpert:
    in_keys = ["obs"]

    def __call__(self, tensordict):
        tensordict["mean_action"] = tensordict["obs"].float() + 20.0
        return tensordict


class _RawDummyExternalExpert:
    in_keys = ["obs"]

    def __call__(self, tensordict):
        tensordict["mean_action"] = tensordict["obs"].float() + 30.0
        return tensordict


class _CheckpointLinear(nn.Linear):
    def materialize_from_state_dict(self, state_dict):
        pass


def _mlp(in_keys, out_key, num_out):
    return MLPWithConcatConfig(in_keys=in_keys, out_keys=[out_key], num_out=num_out)


def test_masked_mimic_config_uses_generic_supervised_agent():
    assert get_class(MaskedMimicSupervisedAgentConfig._target_) is SupervisedAgent


def test_masked_mimic_forward_uses_training_contract():
    latent_dim = 2
    config = SimpleNamespace(
        encoder=ModuleContainerConfig(
            in_keys=["privileged_obs"],
            out_keys=["encoder_mu", "encoder_logvar"],
            models=[
                _mlp(["privileged_obs"], "encoder_mu", latent_dim),
                _mlp(["privileged_obs"], "encoder_logvar", latent_dim),
            ],
        ),
        prior=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["prior_mu", "prior_logvar"],
            models=[
                _mlp(["obs"], "prior_mu", latent_dim),
                _mlp(["obs"], "prior_logvar", latent_dim),
            ],
        ),
        trunk=ModuleContainerConfig(
            in_keys=["obs", "vae_latent"],
            out_keys=["decoded_action"],
            models=[_mlp(["obs", "vae_latent"], "decoded_action", 1)],
        ),
        vae=SimpleNamespace(vae_latent_dim=latent_dim, vae_noise_type="zeros"),
    )
    model = MaskedMimicModel(config)
    model.reset_rollout_context(num_envs=3, device="cpu")

    out = model(
        TensorDict(
            {
                "obs": torch.randn(3, 3),
                "privileged_obs": torch.randn(3, 2),
            },
            batch_size=3,
        )
    )

    assert out["action"].shape == (3, 1)
    assert out["privileged_action"].shape == (3, 1)
    assert out[LATENT_KEY].shape == (3, latent_dim)
    assert out[PRIVILEGED_LATENT_KEY].shape == (3, latent_dim)


def test_supervised_config_uses_loss_target_instead_of_rollout_labeler():
    assert "rollout_labeler" not in SupervisedAgentConfig.__dataclass_fields__
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)

    assert config.rollout_actor is RolloutActor.STUDENT


def test_rollout_actor_serializes_as_readable_value():
    config = SupervisedAgentConfig(
        batch_size=1,
        training_max_steps=1,
        rollout_actor=RolloutActor.EXPERT,
    )

    stored = clean_dict_for_storage(asdict(config))

    assert stored["rollout_actor"] == "expert"


def test_rollout_actor_and_vae_noise_type_parse_case_insensitive_values():
    assert RolloutActor.from_str("STUDENT") is RolloutActor.STUDENT
    assert RolloutActor.from_str("expert") is RolloutActor.EXPERT
    assert VAENoiseType.from_str("NORMAL") is VAENoiseType.NORMAL
    assert VAENoiseType.from_str("uniform") is VAENoiseType.UNIFORM

    try:
        RolloutActor.from_str("teacher")
    except ValueError as error:
        assert "not a valid RolloutActor" in str(error)
    else:
        raise AssertionError("Expected invalid rollout actor to fail")

    try:
        VAENoiseType.from_str("gaussian")
    except ValueError as error:
        assert "not a valid VAENoiseType" in str(error)
    else:
        raise AssertionError("Expected invalid VAE noise type to fail")


def test_supervised_expert_rollout_requires_external_or_model_owned_expert():
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        rollout_actor=RolloutActor.EXPERT,
        loss=SupervisionLossConfig(target_key=TARGET_LATENT_KEY),
    )
    agent.model = _DummyNoModelExpertRollout()
    agent.expert_model = None

    try:
        SupervisedAgent._collect_rollout_output(
            agent,
            TensorDict({"obs": torch.tensor([[1]])}, batch_size=1),
        )
    except ValueError as error:
        assert "rollout_actor=EXPERT needs an expert source" in str(error)
    else:
        raise AssertionError("Expected expert rollout without an expert source to fail")


def test_masked_mimic_does_not_own_expert_rollout_actions():
    assert not hasattr(MaskedMimicModel, "collect_expert_rollout")


def test_supervised_collect_rollout_stores_external_expert_actions_once():
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        rollout_actor=RolloutActor.STUDENT,
        loss=SupervisionLossConfig(target_key="expert_actions"),
    )
    agent.model = _DummyRolloutModel()
    agent.model_output_keys = agent.model.out_keys
    agent.expert_model = _DummyExternalExpert()
    agent.device = torch.device("cpu")
    agent.experience_buffer = _DummyExperienceBuffer()
    obs_td = TensorDict(
        {"obs": torch.tensor([[1], [2]]), "expert_obs": torch.tensor([[1], [2]])},
        batch_size=2,
    )

    out = SupervisedAgent.collect_rollout_step(agent, obs_td, step=4)

    assert torch.equal(out["action"], torch.tensor([[1.0], [1.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("expert_actions", 4)],
        torch.tensor([[21.0], [22.0]]),
    )
    assert agent.experience_buffer.counts[("expert_actions", 4)] == 1


def test_supervised_external_expert_can_be_raw_frozen_module():
    agent = object.__new__(SupervisedAgent)
    agent.expert_model = _RawDummyExternalExpert()
    agent.device = torch.device("cpu")
    obs_td = TensorDict(
        {"obs": torch.tensor([[1], [2]]), "expert_obs": torch.tensor([[1], [2]])},
        batch_size=2,
    )

    action = SupervisedAgent._collect_external_expert_action(agent, obs_td)

    assert torch.equal(action, torch.tensor([[31.0], [32.0]]))


def test_supervised_expert_rollout_ignores_student_privileged_action():
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        rollout_actor=RolloutActor.EXPERT,
        loss=SupervisionLossConfig(target_key="expert_actions"),
    )
    agent.model = _DummyPrivilegedStudentModel()
    agent.model_output_keys = agent.model.out_keys
    agent.expert_model = _DummyExternalExpert()
    agent.device = torch.device("cpu")
    agent.experience_buffer = _DummyExperienceBuffer()
    obs_td = TensorDict(
        {"obs": torch.tensor([[1], [2]]), "expert_obs": torch.tensor([[1], [2]])},
        batch_size=2,
    )

    out = SupervisedAgent.collect_rollout_step(agent, obs_td, step=5)

    assert torch.equal(out["action"], torch.tensor([[21.0], [22.0]]))


def test_supervised_collect_rollout_rejects_string_rollout_actor():
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        rollout_actor="expert",
        loss=SupervisionLossConfig(target_key=TARGET_LATENT_KEY),
    )
    agent.model = _DummyRolloutModel()
    agent.expert_model = None

    try:
        SupervisedAgent._collect_rollout_output(
            agent,
            TensorDict({"obs": torch.tensor([[1]])}, batch_size=1),
        )
    except ValueError as error:
        assert "Unsupported supervised rollout_actor" in str(error)
    else:
        raise AssertionError("Expected a ValueError for string rollout_actor")


def test_supervised_step_uses_configured_supervision_loss():
    agent = object.__new__(SupervisedAgent)
    policy = _DummyLatentPolicy()
    agent.config = SimpleNamespace(
        model=SimpleNamespace(vae=None),
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.MSE,
            prediction_key="latent",
            target_key="target_latent",
            log_prefix="latent_bc",
        ),
    )
    agent.training_model = policy
    agent.model = policy
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.calculate_extra_loss = lambda batch, actions: (
        torch.zeros((), device=agent.device),
        {},
    )

    batch = {
        "action": torch.zeros(2, 1),
        "model_latent": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "target_latent": torch.tensor([[1.0, 0.0], [5.0, 4.0]]),
    }

    loss, log_dict = agent.supervised_step(batch)

    expected = torch.nn.functional.mse_loss(batch["model_latent"], batch["target_latent"])
    assert torch.allclose(loss, expected)
    assert torch.allclose(log_dict["latent_bc/mse"], expected.detach())
    assert torch.allclose(log_dict["supervised/loss"], expected.detach())


def test_supervised_step_adds_model_exposed_loss_without_model_specific_config():
    agent = object.__new__(SupervisedAgent)
    policy = _DummyModelLossPolicy()
    agent.config = SimpleNamespace(
        model=SimpleNamespace(),
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.MSE,
            prediction_key="latent",
            target_key="target_latent",
            log_prefix="supervision",
        ),
    )
    agent.training_model = policy
    agent.model = policy
    agent.device = torch.device("cpu")
    agent.current_epoch = 3
    agent.calculate_extra_loss = lambda batch, actions: (
        torch.zeros((), device=agent.device),
        {},
    )

    batch = {
        "action": torch.zeros(2, 1),
        "model_latent": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "target_latent": torch.tensor([[1.0, 0.0], [5.0, 4.0]]),
        "model_loss_source": torch.tensor([2.0, 4.0]),
    }

    loss, log_dict = agent.supervised_step(batch)

    expected_supervision = torch.nn.functional.mse_loss(
        batch["model_latent"],
        batch["target_latent"],
    )
    expected_model_loss = batch["model_loss_source"].mean() + agent.current_epoch
    assert torch.allclose(loss, expected_supervision + expected_model_loss)
    assert torch.allclose(
        log_dict["supervised/model_loss"],
        expected_model_loss.detach(),
    )
    assert torch.allclose(
        log_dict["model/dummy_loss"],
        expected_model_loss.detach(),
    )


def _make_checkpoint_agent():
    agent = object.__new__(SupervisedAgent)
    model = _CheckpointLinear(1, 1, bias=False)
    model.weight.data.fill_(1.0)

    agent.model = model
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.current_epoch = 7
    agent.step_count = 11
    agent.fit_start_time = 13.0
    agent.best_evaluated_score = 17.0
    agent.evaluator = None
    agent.supervised_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return agent


def test_supervised_training_checkpoint_keeps_resume_state_separate_from_inference():
    agent = _make_checkpoint_agent()

    training_state = SupervisedAgent.get_state_dict(agent, {})
    inference_state = SupervisedAgent.get_inference_state_dict(agent, {})

    assert torch.allclose(training_state["model"]["weight"], torch.tensor([[1.0]]))
    assert "supervised_optimizer" in training_state

    assert torch.allclose(inference_state["model"]["weight"], torch.tensor([[1.0]]))
    assert "supervised_optimizer" not in inference_state


def test_supervised_inference_load_does_not_require_optimizer_state():
    agent = object.__new__(SupervisedAgent)
    model = _CheckpointLinear(1, 1, bias=False)
    model.weight.data.zero_()

    agent.model = model
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent.supervised_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    SupervisedAgent.load_parameters(
        agent,
        {
            "model": {"weight": torch.tensor([[5.0]])},
            "epoch": 0,
        },
        load_training_state=False,
    )

    assert torch.allclose(agent.model.weight, torch.tensor([[5.0]]))
