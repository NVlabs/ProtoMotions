# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPC/FSQ prior latent supervision."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    LATENT_LOGPROB_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
)
from protomotions.agents.common.supervision import compute_supervision_loss
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.peft import prior_setup as prior_setup_module
from protomotions.agents.peft.prior_config import DiscretePriorPEFTSFTAgentConfig
from protomotions.agents.peft.sft_agent import DiscretePriorPEFTSFTAgent
from protomotions.agents.peft.sft_model import DiscretePriorPEFTSFTModel
from protomotions.agents.peft.actor import DiscretePriorPEFTActor


class _DummyDiscretePriorPEFTActor:
    encoder = object()
    kl_coeff = 0.0

    def __init__(self, logits=None):
        self.logits = logits

    def __call__(self, prior_dict):
        return self.logits

    def predict_target_prior_tokens(self, tensordict: TensorDict):
        return tensordict[TARGET_LATENT_KEY]

    def normalize_obs(self, obs):
        return obs

    def build_task_conditioning(self, batch):
        return batch["task_obs"]

    def terrain_observation(self, batch):
        return None

    def build_prior_input(self, batch, tokens=None):
        prior_dict = {
            "max_coords_obs": batch["max_coords_obs"],
            "task_cond": self.build_task_conditioning(batch),
        }
        if tokens is not None:
            prior_dict["tokens"] = self.one_hot_prior_tokens(tokens)
        return prior_dict

    def one_hot_prior_tokens(self, indices):
        return torch.nn.functional.one_hot(indices, num_classes=3).float()

    def perturb_tokens(self, tokens, *, rate, mode):
        return tokens

    def batch_size_from_input(self, batch):
        return batch["max_coords_obs"].shape[0]


class _CtorOnlyDiscretePriorPEFTActor(torch.nn.Module):
    def __init__(self, config, pretrained_prior_model, mimic_target_poses_dim=0):
        super().__init__()
        self.config = config
        self.pretrained_prior_model = pretrained_prior_model
        self.mimic_target_poses_dim = mimic_target_poses_dim
        self.in_keys = ["max_coords_obs"]
        self.out_keys = ["action", "mean_action", "neglogp", "prior_tokens"]


def test_prior_peft_sft_agent_config_uses_discrete_token_loss():
    config = DiscretePriorPEFTSFTAgentConfig(batch_size=2, training_max_steps=4)

    assert config.loss.loss_type == SupervisionLossType.DISCRETE_CROSS_ENTROPY
    assert config.loss.prediction_key == LATENT_LOGITS_KEY
    assert config.loss.target_key == TARGET_LATENT_KEY


def test_prior_peft_sft_agent_create_model_initializes_no_external_expert(
    monkeypatch,
):
    monkeypatch.setattr(
        prior_setup_module.DiscretePriorPEFTSetupMixin,
        "create_model",
        lambda self: object(),
    )
    agent = object.__new__(DiscretePriorPEFTSFTAgent)

    DiscretePriorPEFTSFTAgent.create_model(agent)
    SupervisedAgent.register_algorithm_experience_buffer_keys(agent)

    assert agent.expert_model is None


def test_prior_peft_sft_model_config_does_not_require_critic_field():
    config = SimpleNamespace(
        actor=SimpleNamespace(
            _target_=(
                "protomotions.tests.test_fsq_prior_latent_loss."
                "_CtorOnlyDiscretePriorPEFTActor"
            )
        ),
        token_perturb_rate=0.0,
        token_perturb_mode="replace",
    )

    model = DiscretePriorPEFTSFTModel(
        config=config,
        pretrained_prior_model=object(),
    )

    assert model._critic is None
    assert model.in_keys == ["max_coords_obs"]
    assert model.out_keys == ["action", "mean_action", "neglogp", "prior_tokens"]


def test_prior_peft_sft_model_routes_through_supervision_loss_config():
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 4.0, -2.0]],
            [[-1.0, 0.0, 5.0], [2.0, 0.0, -1.0]],
        ]
    )
    target = torch.tensor([[0, 1], [2, 0]])
    model = object.__new__(DiscretePriorPEFTSFTModel)
    torch.nn.Module.__init__(model)
    model._actor = _DummyDiscretePriorPEFTActor(logits=logits)
    model.config = SimpleNamespace(token_perturb_rate=0.0, token_perturb_mode="replace")
    loss_cfg = SupervisionLossConfig(
        loss_type=SupervisionLossType.DISCRETE_CROSS_ENTROPY,
        prediction_key=LATENT_LOGITS_KEY,
        target_key=TARGET_LATENT_KEY,
        log_prefix="gpc_latent",
    )

    td = TensorDict(
        {
            "max_coords_obs": torch.randn(2, 4),
            "task_obs": torch.randn(2, 1),
            TARGET_LATENT_KEY: target,
        },
        batch_size=2,
    )

    td = DiscretePriorPEFTSFTModel.forward(model, td)
    loss, log_dict = compute_supervision_loss(td, loss_cfg)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))

    assert torch.allclose(loss, expected)
    assert torch.allclose(log_dict["gpc_latent/cross_entropy"], expected.detach())
    assert torch.allclose(log_dict["gpc_latent/accuracy"], torch.tensor(1.0))


class _DummyPEFT:
    def __init__(self, indices, logits):
        self.indices = indices
        self.logits = logits

    def generate(self, prior_dict, return_logits=True, return_logprob=False):
        outputs = [self.indices]
        if return_logits:
            outputs.append(self.logits)
        if return_logprob:
            outputs.append(
                torch.distributions.Categorical(logits=self.logits).log_prob(
                    self.indices
                )
            )
        if len(outputs) > 1:
            return tuple(outputs)
        return self.indices


def test_prior_peft_actor_returns_latent_selection_distribution_and_logprob():
    indices = torch.tensor([[0, 1], [2, 0]])
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 4.0, -2.0]],
            [[-1.0, 0.0, 5.0], [2.0, 0.0, -1.0]],
        ]
    )
    actor = object.__new__(DiscretePriorPEFTActor)
    actor.prior_with_peft = _DummyPEFT(indices, logits)
    actor.normalize_obs = lambda obs: obs
    actor.build_task_conditioning = lambda td: td["task_obs"]
    actor.terrain_observation = lambda td: None
    actor.build_prior_input = lambda td: {
        "max_coords_obs": td["max_coords_obs"],
        "task_cond": td["task_obs"],
    }
    actor.prior_tokens_to_fsq_indices = lambda prior_tokens: prior_tokens
    actor.fsq_indices_to_codes = lambda fsq_indices: fsq_indices.float()
    actor._decode = lambda td, fsq_codes: fsq_codes.sum(dim=-1, keepdim=True)

    td = TensorDict(
        {
            "max_coords_obs": torch.randn(2, 4),
            "task_obs": torch.randn(2, 1),
        },
        batch_size=2,
    )

    out = DiscretePriorPEFTActor.get_action_and_logp(actor, td)
    expected_logprob = torch.distributions.Categorical(logits=logits).log_prob(indices)

    assert torch.equal(out[LATENT_KEY], indices)
    assert torch.equal(out["prior_tokens"], indices)
    assert LATENT_LOGITS_KEY not in out.keys()
    assert torch.allclose(out[LATENT_LOGPROB_KEY], expected_logprob)
    assert torch.allclose(out["neglogp"], -expected_logprob)
