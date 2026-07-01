# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-owned losses in PPO optimization."""

from types import SimpleNamespace

import torch
from torch import nn
from tensordict import TensorDict

from protomotions.agents.base_agent.model import BaseModel, ProtoMotionsTensorDictModule
from protomotions.agents.common.config import MLPWithConcatConfig
from protomotions.agents.common.common import MODULE_INTERNALS_KEY
from protomotions.agents.common.mlp import MLPWithConcat
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.model import PPOActor, PPOModel


class _ActorWithModelLoss(nn.Module):
    in_keys = ["model_loss_source"]

    def __init__(self):
        super().__init__()
        self.logstd = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, tensordict, log_internals=False):
        tensordict["mean_action"] = torch.zeros_like(tensordict["action"])
        if log_internals:
            tensordict[MODULE_INTERNALS_KEY] = TensorDict(
                {
                    "perplexity": torch.tensor([2, 4]),
                    "ppo_loss": torch.tensor([99, 101]),
                },
                batch_size=tensordict.batch_size,
            )
        return tensordict

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        loss = tensordict["model_loss_source"].mean() + current_epoch
        return loss, {f"{log_prefix}/dummy_loss": loss.detach()}


class _CriticWithModelLoss(nn.Module):
    def forward(self, tensordict):
        tensordict["value"] = torch.zeros_like(tensordict["returns"]).unsqueeze(-1)
        return tensordict

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        loss = tensordict["model_loss_source"].mean() + current_epoch
        return loss, {f"{log_prefix}/dummy_loss": loss.detach()}


class _PlainCritic(ProtoMotionsTensorDictModule):
    def forward(self, tensordict):
        tensordict["value"] = torch.full_like(
            tensordict["returns"].unsqueeze(-1), 1.0
        )
        return tensordict


class _ResettableCritic(ProtoMotionsTensorDictModule):
    in_keys = ["obs"]
    out_keys = ["value"]

    def __init__(self, config):
        super().__init__()
        self.reset_args = None

    def reset_rollout_context(self, env_ids=None, num_envs: int = None, device=None):
        self.reset_args = (env_ids, num_envs, device)

    def forward(self, tensordict):
        tensordict["value"] = torch.zeros(tensordict.batch_size[0], 1)
        return tensordict


class _StatefulMuModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.in_keys = ["obs"]
        self.out_keys = ["mu"]
        self.reset_args = None

    def rollout_context_keys(self) -> list:
        return ["vae_noise"]

    def reset_rollout_context(self, env_ids=None, num_envs: int = None, device=None):
        self.reset_args = (env_ids, num_envs, device)

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["mu"] = torch.zeros(tensordict.batch_size[0], 1)
        tensordict["vae_noise"] = torch.ones(tensordict.batch_size[0], 2)
        return tensordict


class _MuWithInternalsAndLoss(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.in_keys = ["obs"]
        self.out_keys = ["mu"]

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["mu"] = tensordict["obs"][:, :1] + 0.25
        if log_internals:
            tensordict[MODULE_INTERNALS_KEY] = TensorDict(
                {"usage": torch.tensor([1.0, 3.0])},
                batch_size=tensordict.batch_size,
            )
        return tensordict

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        loss = tensordict["loss_source"].mean() + current_epoch
        return loss, {f"{log_prefix}/mu_loss": loss.detach()}


class _ValueCritic(nn.Module):
    in_keys = ["obs"]
    out_keys = ["value"]

    def __init__(self, config):
        super().__init__()

    def forward(self, tensordict):
        tensordict["value"] = torch.zeros(tensordict.batch_size[0], 1)
        return tensordict

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        return zero_loss * 0.0, {}


class _ChildWithHooks(ProtoMotionsTensorDictModule):
    in_keys = []
    out_keys = ["child_out"]

    def __init__(self):
        super().__init__()
        self.reset_args = None
        self.reset_count = 0

    def rollout_context_keys(self) -> list:
        return ["child_state"]

    def reset_rollout_context(self, env_ids=None, num_envs: int = None, device=None):
        self.reset_args = (env_ids, num_envs, device)
        self.reset_count += 1

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        loss = tensordict["loss_source"].mean() + current_epoch
        return loss, {f"{log_prefix}/child_loss": loss.detach()}

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["child_out"] = tensordict["loss_source"].unsqueeze(-1)
        return tensordict


class _ParentWithoutManualDelegation(ProtoMotionsTensorDictModule):
    in_keys = ["loss_source"]
    out_keys = ["parent_out"]

    def __init__(self):
        super().__init__()
        self.child = _ChildWithHooks()

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["parent_out"] = tensordict["loss_source"].unsqueeze(-1)
        return self.child(tensordict, log_internals=log_internals)


class _PlainBranch(nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child


class _ParentWithSharedChildThroughTwoBranches(ProtoMotionsTensorDictModule):
    in_keys = ["loss_source"]
    out_keys = ["parent_out"]

    def __init__(self):
        super().__init__()
        child = _ChildWithHooks()
        self.left = _PlainBranch(child)
        self.right = _PlainBranch(child)
        self.child = child

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["parent_out"] = tensordict["loss_source"].unsqueeze(-1)
        return self.child(tensordict, log_internals=log_internals)


class _PlainMuModule(nn.Module):
    in_keys = ["obs"]
    out_keys = ["mu"]

    def __init__(self, config):
        super().__init__()

    def forward(self, tensordict):
        tensordict["mu"] = torch.zeros(tensordict.batch_size[0], 1)
        return tensordict


def _make_ppo_agent():
    agent = object.__new__(PPO)
    agent.device = torch.device("cpu")
    agent.current_epoch = 2
    agent.e_clip = 0.2
    agent.num_envs = 2
    agent.config = SimpleNamespace(
        bounds_loss_coef=0.0,
        entropy_coef=0.0,
        clip_critic_loss=False,
        model=SimpleNamespace(
            actor=SimpleNamespace(learnable_std=False),
        ),
        adaptive_lr=SimpleNamespace(enabled=False),
    )
    agent.calculate_extra_actor_loss = lambda batch_td: (
        torch.zeros((), device=agent.device),
        {},
    )
    return agent


def _actor_config(mu_target, learnable_std=False):
    return SimpleNamespace(
        mu_model=SimpleNamespace(_target_=mu_target),
        in_keys=["obs"],
        out_keys=["action", "mean_action", "neglogp"],
        mu_key="mu",
        num_out=1,
        actor_logstd=-1.0,
        learnable_std=learnable_std,
    )


def test_ppo_actor_samples_actions_and_recomputes_neglogp_from_configured_mu_key():
    torch.manual_seed(0)
    actor = PPOActor(
        _actor_config(
            "protomotions.tests.test_ppo_model_loss._StatefulMuModel",
            learnable_std=False,
        )
    )
    td = TensorDict({"obs": torch.ones(3, 2)}, batch_size=3)

    out = actor(td)

    expected_std = torch.exp(torch.tensor(-1.0))
    dist = torch.distributions.Normal(out["mean_action"], expected_std)
    assert out["mean_action"].shape == (3, 1)
    assert out["action"].shape == (3, 1)
    assert out["neglogp"].shape == (3,)
    assert torch.allclose(
        out["neglogp"],
        -dist.log_prob(out["action"]).sum(dim=-1),
    )
    assert not actor.logstd.requires_grad


def test_ppo_actor_forwards_mu_internals_and_model_owned_loss():
    actor = PPOActor(
        _actor_config(
            "protomotions.tests.test_ppo_model_loss._MuWithInternalsAndLoss",
            learnable_std=True,
        )
    )
    td = TensorDict(
        {
            "obs": torch.ones(2, 2),
            "loss_source": torch.tensor([2.0, 4.0]),
        },
        batch_size=2,
    )

    out = actor(td, log_internals=True)
    loss, logs = actor.compute_model_loss(
        out,
        current_epoch=5,
        zero_loss=torch.zeros(()),
        log_prefix="actor_model",
    )

    assert actor.logstd.requires_grad
    assert torch.equal(out[MODULE_INTERNALS_KEY]["usage"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(loss, torch.tensor(8.0))
    assert torch.allclose(logs["actor_model/mu_loss"], torch.tensor(8.0))


def test_proto_module_default_hooks_recurse_into_proto_children():
    module = _ParentWithoutManualDelegation()
    env_ids = torch.tensor([0, 2])
    zero_loss = torch.zeros(())
    td = TensorDict({"loss_source": torch.tensor([2.0, 4.0])}, batch_size=2)

    module.reset_rollout_context(env_ids=env_ids, num_envs=3, device="cpu")
    loss, logs = module.compute_model_loss(
        td,
        current_epoch=5,
        zero_loss=zero_loss,
        log_prefix="parent",
    )

    assert module.rollout_context_keys() == ["child_state"]
    assert module.experience_buffer_keys() == ["parent_out", "child_state"]
    assert module.child.reset_args == (env_ids, 3, "cpu")
    assert torch.allclose(loss, torch.tensor(8.0))
    assert torch.allclose(logs["parent/child_loss"], torch.tensor(8.0))


def test_proto_module_default_hooks_deduplicate_shared_children_by_identity():
    module = _ParentWithSharedChildThroughTwoBranches()
    env_ids = torch.tensor([0, 2])
    zero_loss = torch.zeros(())
    td = TensorDict({"loss_source": torch.tensor([2.0, 4.0])}, batch_size=2)

    module.reset_rollout_context(env_ids=env_ids, num_envs=3, device="cpu")
    loss, logs = module.compute_model_loss(
        td,
        current_epoch=5,
        zero_loss=zero_loss,
        log_prefix="parent",
    )

    assert module.rollout_context_keys() == ["child_state"]
    assert module.child.reset_count == 1
    assert torch.allclose(loss, torch.tensor(8.0))
    assert torch.allclose(logs["parent/child_loss"], torch.tensor(8.0))


def test_ppo_actor_non_base_mu_has_no_stateful_keys_or_model_loss():
    actor = PPOActor(
        _actor_config(
            "protomotions.tests.test_ppo_model_loss._PlainMuModule",
            learnable_std=False,
        )
    )

    loss, logs = actor.compute_model_loss(
        TensorDict({}, batch_size=[]),
        current_epoch=1,
        zero_loss=torch.ones(()),
    )

    assert actor.rollout_context_keys() == []
    assert actor.experience_buffer_keys() == ["action", "mean_action", "neglogp"]
    assert torch.equal(loss, torch.zeros(()))
    assert logs == {}


def test_ppo_model_exposes_nested_actor_rollout_context_keys_for_buffer_storage():
    config = SimpleNamespace(
        actor=SimpleNamespace(
            _target_="protomotions.agents.ppo.model.PPOActor",
            mu_model=SimpleNamespace(
                _target_="protomotions.tests.test_ppo_model_loss._StatefulMuModel",
            ),
            in_keys=["obs"],
            out_keys=["action", "mean_action", "neglogp"],
            mu_key="mu",
            num_out=1,
            actor_logstd=0.0,
            learnable_std=False,
        ),
        critic=SimpleNamespace(
            _target_="protomotions.tests.test_ppo_model_loss._ValueCritic",
        ),
        in_keys=["obs"],
        out_keys=["action", "mean_action", "neglogp", "value"],
    )

    model = PPOModel(config)

    assert "vae_noise" in model.experience_buffer_keys()
    assert "mu" not in model.experience_buffer_keys()


def test_ppo_model_resets_nested_rollout_context():
    config = SimpleNamespace(
        actor=SimpleNamespace(
            _target_="protomotions.agents.ppo.model.PPOActor",
            mu_model=SimpleNamespace(
                _target_="protomotions.tests.test_ppo_model_loss._StatefulMuModel",
            ),
            in_keys=["obs"],
            out_keys=["action", "mean_action", "neglogp"],
            mu_key="mu",
            num_out=1,
            actor_logstd=0.0,
            learnable_std=False,
        ),
        critic=SimpleNamespace(
            _target_="protomotions.tests.test_ppo_model_loss._ResettableCritic",
        ),
        in_keys=["obs"],
        out_keys=["action", "mean_action", "neglogp", "value"],
    )
    env_ids = torch.tensor([1, 3])
    device = torch.device("cpu")

    model = PPOModel(config)

    model.reset_rollout_context(env_ids=env_ids, num_envs=4, device=device)

    assert model._actor.mu.reset_args == (env_ids, 4, device)
    assert model._critic.reset_args == (env_ids, 4, device)


def test_ppo_actor_loss_includes_model_owned_loss():
    agent = _make_ppo_agent()
    agent.actor = _ActorWithModelLoss()
    batch = {
        "action": torch.zeros(2, 1),
        "neglogp": torch.zeros(2),
        "advantages": torch.zeros(2),
        "model_loss_source": torch.tensor([3.0, 5.0]),
    }

    loss, log_dict = agent.actor_step(batch)

    expected_model_loss = batch["model_loss_source"].mean() + agent.current_epoch
    assert torch.allclose(loss, expected_model_loss)
    assert torch.allclose(log_dict["actor/model_loss"], expected_model_loss.detach())
    assert torch.allclose(
        log_dict["actor_model/dummy_loss"],
        expected_model_loss.detach(),
    )
    assert torch.allclose(log_dict["actor/internals/perplexity"], torch.tensor(3.0))
    assert torch.allclose(log_dict["actor/internals/ppo_loss"], torch.tensor(100.0))
    assert not torch.allclose(log_dict["actor/ppo_loss"], torch.tensor(100.0))


def test_ppo_critic_loss_includes_model_owned_loss():
    agent = _make_ppo_agent()
    agent.critic = _CriticWithModelLoss()
    batch = {
        "action": torch.zeros(2, 1),
        "returns": torch.zeros(2),
        "value": torch.zeros(2, 1),
        "model_loss_source": torch.tensor([2.0, 4.0]),
    }

    loss, log_dict = agent.critic_step(batch)

    expected_model_loss = batch["model_loss_source"].mean() + agent.current_epoch
    assert torch.allclose(loss, expected_model_loss)
    assert torch.allclose(log_dict["critic/model_loss"], expected_model_loss.detach())
    assert torch.allclose(
        log_dict["critic_model/dummy_loss"],
        expected_model_loss.detach(),
    )


def test_ppo_critic_loss_allows_plain_critic_without_model_owned_loss():
    agent = _make_ppo_agent()
    agent.critic = _PlainCritic()
    batch = {
        "action": torch.zeros(3, 1),
        "returns": torch.tensor([0.0, 1.0, 2.0]),
        "value": torch.zeros(3, 1),
    }

    loss, log_dict = agent.critic_step(batch)

    assert torch.allclose(loss, torch.tensor(2.0 / 3.0))
    assert torch.equal(log_dict["critic/model_loss"], torch.zeros(()))


def test_ppo_critic_loss_allows_mlp_with_concat_critic_without_model_owned_loss():
    agent = _make_ppo_agent()
    agent.critic = MLPWithConcat(
        MLPWithConcatConfig(in_keys=["obs"], out_keys=["value"], num_out=1)
    )
    batch = {
        "obs": torch.ones(3, 2),
        "action": torch.zeros(3, 1),
        "returns": torch.tensor([0.0, 1.0, 2.0]),
        "value": torch.zeros(3, 1),
    }

    loss, log_dict = agent.critic_step(batch)

    assert torch.isfinite(loss)
    assert torch.equal(log_dict["critic/model_loss"], torch.zeros(()))
    assert not any(key.startswith("critic_model/") for key in log_dict)
