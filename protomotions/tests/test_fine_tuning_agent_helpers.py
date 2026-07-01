# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reusable fine-tuning pretrained-module agent base."""

import torch
from torch import nn

from protomotions.agents.common.config import PretrainedModelConfig
from protomotions.agents.fine_tuning.config import FineTuningAgentConfig
from protomotions.agents.fine_tuning.agent import FineTuningAgent
from protomotions.agents.ppo.config import PPOAgentConfig


class _DummyFabric:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def call(self, hook_name):
        self.calls.append(hook_name)


class _DummyEnv:
    def get_obs(self):
        return {"obs": torch.zeros(2, 3)}


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(1)
        self.reset_args = None
        self.materialized_with = None

    def reset_rollout_context(self, num_envs, device):
        self.reset_args = (num_envs, device)

    def materialize(self, obs):
        self.materialized_with = obs
        return self.linear(obs["obs"])


class _DummyFineTuningAgent(FineTuningAgent):
    def __init__(self, config):
        self.fabric = _DummyFabric()
        self.device = self.fabric.device
        self.env = _DummyEnv()
        self.config = config
        self.num_envs = 2
        self.events = []
        self.optimizer_created_for = None

    def add_agent_info_to_obs(self, obs):
        self.events.append("add_agent_info_to_obs")
        return obs

    def obs_dict_to_tensordict(self, obs):
        self.events.append("obs_dict_to_tensordict")
        return obs

    def create_model(self):
        self.events.append(("create_model", tuple(self.pretrained)))
        self.created_pretrained = dict(self.pretrained)
        return _DummyModel()

    def create_optimizers(self, model):
        self.events.append("create_optimizers")
        self.optimizer_created_for = model

    def _post_create_model_hook(self):
        self.events.append("_post_create_model_hook")

    def _materialize_lazy_modules(self, dummy_obs_td):
        self.events.append("_materialize_lazy_modules")
        self.model.materialize(dummy_obs_td)

    def _print_param_info(self):
        self.events.append("_print_param_info")


def test_fine_tuning_agent_config_defaults_to_empty_pretrained_modules():
    config = FineTuningAgentConfig(batch_size=4, training_max_steps=64)

    assert config.pretrained_modules == {}
    assert config._target_ == "protomotions.agents.fine_tuning.agent.FineTuningAgent"
    assert isinstance(config, PPOAgentConfig)


def test_fine_tuning_setup_loads_pretrained_modules_before_model_creation(monkeypatch):
    loaded_module = nn.Linear(3, 1)
    pretrained_config = PretrainedModelConfig(
        checkpoint_path="stage.ckpt",
        module_path="actor",
    )
    config = FineTuningAgentConfig(
        batch_size=4,
        training_max_steps=64,
        pretrained_modules={"stage": pretrained_config},
    )
    load_calls = []

    def fake_load_pretrained_model_module(load_config, device):
        load_calls.append((load_config, device))
        return loaded_module

    monkeypatch.setattr(
        "protomotions.agents.fine_tuning.pretrained_modules.load_pretrained_model_module",
        fake_load_pretrained_model_module,
    )
    agent = _DummyFineTuningAgent(config)

    agent.setup()

    assert load_calls == [(pretrained_config, torch.device("cpu"))]
    assert agent.created_pretrained == {"stage": loaded_module}
    assert agent.model.reset_args == (2, torch.device("cpu"))
    assert agent.optimizer_created_for is agent.model
    assert agent.pretrained == {}
    assert agent.fabric.calls == [
        "on_model_init_start",
        "on_model_init_end",
        "on_optimizer_init_start",
        "on_optimizer_init_end",
    ]
    assert agent.events == [
        ("create_model", ("stage",)),
        "_post_create_model_hook",
        "add_agent_info_to_obs",
        "obs_dict_to_tensordict",
        "_materialize_lazy_modules",
        "create_optimizers",
        "_print_param_info",
    ]
