# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DiscretePriorPEFTRLFTAgent helper paths that do not need checkpoints."""

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.autoregressive import (
    prior_constrained_sampling_log_probs,
)
from protomotions.agents.common.latent import LATENT_LOGITS_KEY, TARGET_LATENT_KEY
from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
)
from protomotions.agents.base_agent import agent as base_agent_module
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    ModuleContainerConfig,
    PretrainedModelConfig,
)
from protomotions.agents.fine_tuning import agent as fine_tuning_agent_module
from protomotions.agents.fine_tuning import pretrained_modules as pretrained_modules_module
from protomotions.agents.peft import prior_agent as prior_agent_module
from protomotions.agents.peft import prior_setup as prior_setup_module
from protomotions.agents.peft.utils import model_state as model_state_module
from protomotions.agents.peft.utils.adapter_state import is_adapter_state_key
from protomotions.agents.peft.prior_amp_agent import DiscretePriorPEFTRLFTAMPAgent
from protomotions.agents.peft.prior_agent import DiscretePriorPEFTRLFTAgent
from protomotions.agents.peft.sft_agent import DiscretePriorPEFTSFTAgent
from protomotions.agents.peft.sft_model import DiscretePriorPEFTSFTModel
from protomotions.agents.peft.prior_config import (
    DiscretePriorPEFTConfig,
    DiscretePriorPEFTActorConfig,
    DiscretePriorPEFTRLFTAgentConfig,
    DiscretePriorPEFTSFTAgentConfig,
)
from protomotions.agents.ppo import agent as ppo_agent_module
from protomotions.agents.ppo.utils import discount_values
from protomotions.agents.utils.metering import TensorAverageMeterDict


class _ExperienceBufferRecorder:
    def __init__(self):
        self.registered = []
        self.data = {}
        self.make_dict_calls = 0

    def register_key(self, key, shape=(), dtype=None):
        self.registered.append((key, shape, dtype))

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()

    def batch_update_data(self, key, value):
        setattr(self, key, value.clone())
        self.data[(key, "batch")] = value.clone()

    def make_dict(self):
        self.make_dict_calls += 1
        return {"made": True}


class _Actor(nn.Module):
    num_prior_tokens = 3
    prior_token_vocab_size = 5
    fsq_scalars_per_prior_token = 1
    num_fsq_levels = 5

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def get_action_and_logp(self, obs_td):
        batch = obs_td.batch_size[0]
        action = obs_td["max_coords_obs"][:, :2] * self.weight
        return {
            "action": action,
            "mean_action": action + 0.5,
            "neglogp": torch.arange(batch, dtype=torch.float),
            "prior_tokens": torch.arange(batch * self.num_prior_tokens).view(
                batch,
                self.num_prior_tokens,
            )
            % self.prior_token_vocab_size,
        }

    def prior_tokens_to_fsq_indices(self, tokens):
        return tokens

    def fsq_indices_to_prior_tokens(self, flat):
        return flat + 100


class _Critic(nn.Module):
    in_keys = ["max_coords_obs", "task_obs"]
    out_keys = ["value"]

    def forward(self, tensordict):
        tensordict["value"] = (
            tensordict["max_coords_obs"][:, :1] + tensordict["task_obs"][:, :1]
        )
        return tensordict


class _ParamCritic(_Critic):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(()))

    def forward(self, tensordict):
        tensordict["value"] = (
            tensordict["max_coords_obs"][:, :1]
            + tensordict["task_obs"][:, :1]
            + self.bias
        )
        return tensordict


class _Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, value):
        if isinstance(value, TensorDict):
            return self.module(value)
        context = value.get("max_coords_obs", value.get("self_obs"))
        batch = context.shape[0]
        return torch.zeros(batch, 3, 5, requires_grad=True)

    def train(self, mode=True):
        self.training = mode
        return self


class _OptimizerRecorder:
    def __init__(self, state=None):
        self.zero_grad_calls = []
        self.steps = 0
        self.state = state or {"state": "initial"}
        self.loaded_state = None

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls.append(set_to_none)

    def step(self):
        self.steps += 1

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.loaded_state = state


class _SingleRankFabric:
    world_size = 1
    global_rank = 0


class _Env:
    def __init__(self, obs):
        self.obs = obs

    def get_obs(self):
        return {key: value.clone() for key, value in self.obs.items()}


def _obs(batch=2):
    return {
        "max_coords_obs": torch.arange(batch * 4, dtype=torch.float).view(batch, 4),
        "task_obs": torch.ones(batch, 3),
        "mimic_target_poses": torch.ones(batch, 5),
        "masked_mimic_target_poses": torch.ones(batch, 7),
        "masked_mimic_target_masks": torch.ones(batch, 2),
        "masked_mimic_target_times": torch.ones(batch, 1),
        "historical_pose_obs": torch.ones(batch, 4),
    }


_DEFAULT_CONDITIONING_KEYS = (
    "task_obs",
    "masked_mimic_target_poses",
    "masked_mimic_target_masks",
    "masked_mimic_target_times",
    "historical_pose_obs",
)


def _config(token_perturb_rate=0.0, token_perturb_mode="replace"):
    return SimpleNamespace(
        normalize_rewards=False,
        task_reward_w=1.0,
        entropy_coef=0.01,
        pretrained_modules={},
        advantage_normalization=SimpleNamespace(enabled=False),
        adaptive_lr=SimpleNamespace(enabled=False),
        model=SimpleNamespace(
            actor=SimpleNamespace(
                in_keys=[*_DEFAULT_CONDITIONING_KEYS],
                learnable_std=False,
                peft=SimpleNamespace(
                    model=SimpleNamespace(
                        in_keys=[*_DEFAULT_CONDITIONING_KEYS],
                        out_keys=["task_cond"],
                    ),
                    clear_peft=False,
                ),
            ),
            token_perturb_rate=token_perturb_rate,
            token_perturb_mode=token_perturb_mode,
            critic=SimpleNamespace(),
        ),
    )


def _agent(has_critic=True, token_perturb_rate=0.0, token_perturb_mode="replace"):
    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    actor = _Actor()
    critic = _Critic() if has_critic else None

    class _AgentModel:
        _actor = actor
        _critic = critic

        def __call__(self, obs_td):
            output_td = self._actor.get_action_and_logp(obs_td)
            if self._critic is not None:
                for key, value in output_td.items():
                    obs_td[key] = value
                output_td = self._critic(obs_td)
            return output_td

        def experience_buffer_keys(self):
            keys = ["action", "mean_action", "neglogp", "prior_tokens"]
            if self._critic is not None:
                keys.append("value")
            return keys

    agent.model = _AgentModel()
    agent.actor = _Wrapper(actor)
    agent.critic = _Wrapper(critic) if critic is not None else None
    agent.config = _config(token_perturb_rate, token_perturb_mode)
    agent.config.model.critic = SimpleNamespace() if has_critic else None
    agent.device = torch.device("cpu")
    agent.gamma = 0.9
    agent.tau = 0.8
    agent.e_clip = 0.2
    return agent


class _DiscretePriorWithPEFT:
    def __init__(self, prior_logits):
        self.prior_logits = prior_logits
        self.temperature = 0.75
        self.top_p = 0.5
        self.prior_top_p = 0.7
        self.sampling_mode = "prior_constraint"

    def forward_prior(self, prior_dict):
        self.last_prior_dict = prior_dict
        return self.prior_logits


class _LatentActor(_Actor):
    def __init__(
        self,
        *,
        encoder,
        kl_coeff=0.0,
        extra_context=None,
        prior_logits=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.kl_coeff = kl_coeff
        self.extra_context = extra_context
        self.prior_with_peft = (
            _DiscretePriorWithPEFT(prior_logits)
            if prior_logits is not None
            else None
        )

    def predict_target_prior_tokens(self, tensordict):
        return tensordict[TARGET_LATENT_KEY]

    def normalize_obs(self, obs):
        self.normalized_obs = obs
        return obs + 1.0

    def build_task_conditioning(self, batch):
        return batch["task_obs"] * 2.0

    def extra_prior_context(self, batch):
        return self.extra_context

    def build_prior_input(self, batch, tokens=None):
        prior_dict = {
            "max_coords_obs": batch["max_coords_obs"],
            "task_cond": self.build_task_conditioning(batch),
        }
        if tokens is not None:
            prior_dict["tokens"] = self.one_hot_prior_tokens(tokens)
        extra_context = self.extra_prior_context(batch)
        if extra_context is not None:
            prior_dict["extra_context"] = extra_context
        return prior_dict

    def one_hot_prior_tokens(self, indices):
        return torch.nn.functional.one_hot(indices, num_classes=self.prior_token_vocab_size).float()

    def perturb_tokens(self, tokens, *, rate, mode):
        return tokens

    def batch_size_from_input(self, batch):
        return batch["max_coords_obs"].shape[0]


def test_prior_peft_properties_resolve_wrapped_actor():
    agent = _agent(has_critic=True)

    assert agent.has_critic is True
    assert agent.actor_module is agent.model._actor


def test_prior_peft_captures_rl_reference_once_and_optionally_clears_student():
    class _PEFT:
        def __init__(self):
            self.reference_ready = False
            self.captured = 0
            self.cleared = 0

        def capture_reference(self):
            if self.reference_ready:
                return False
            self.captured += 1
            self.reference_ready = True
            return True

        def clear_peft(self):
            self.cleared += 1

    agent = _agent(has_critic=True)
    peft = _PEFT()
    agent.model._actor.prior_with_peft = peft
    agent.config.model.actor.peft.clear_peft = True

    DiscretePriorPEFTRLFTAgent._prepare_rlft_prior_reference(agent)
    DiscretePriorPEFTRLFTAgent._prepare_rlft_prior_reference(agent)

    assert peft.captured == 1
    assert peft.cleared == 1
    assert not hasattr(DiscretePriorPEFTSFTAgent, "_prepare_rlft_prior_reference")


def test_prior_peft_requires_reference_when_loading_training_state():
    class _PEFT:
        def __init__(self, ready):
            self.reference_ready = ready
            self.required = 0
            self.captured = 0

        def require_reference(self):
            self.required += 1
            if not self.reference_ready:
                raise RuntimeError("missing reference")

        def capture_reference(self):
            self.captured += 1
            self.reference_ready = True
            return True

    agent = _agent(has_critic=True)
    peft = _PEFT(ready=True)
    agent.model._actor.prior_with_peft = peft

    DiscretePriorPEFTRLFTAgent._prepare_rlft_prior_reference(
        agent,
        require_existing=True,
    )

    assert peft.required == 1
    assert peft.captured == 0

    peft = _PEFT(ready=False)
    agent.model._actor.prior_with_peft = peft
    with pytest.raises(RuntimeError, match="missing reference"):
        DiscretePriorPEFTRLFTAgent._prepare_rlft_prior_reference(
            agent,
            require_existing=True,
        )

def test_prior_peft_clamps_dora_m_from_config():
    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = nn.Parameter(torch.tensor([[-2.0, 0.25, 2.0]]))

    layer = _Layer()
    transformer = nn.Module()
    transformer.layers = nn.ModuleList([layer])
    agent = _agent(has_critic=True)
    agent.config.model.actor.peft.m_clamp = 0.5
    agent.model._actor.prior_with_peft = SimpleNamespace(
        base_prior=SimpleNamespace(
            _transformer=transformer,
        )
    )

    DiscretePriorPEFTRLFTAgent._clamp_peft_m(agent)

    assert torch.equal(layer.m, torch.tensor([[-0.5, 0.25, 0.5]]))


def test_prior_peft_allows_missing_film_input_norm_checkpoint_state():
    class _PriorWithPEFT(nn.Module):
        def optional_full_checkpoint_state_prefixes(self):
            return ("film_input_norm.",)

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.prior_with_peft = _PriorWithPEFT()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = _Actor()

    key = "_actor.prior_with_peft.film_input_norm.running_obs_norm.mean"
    assert key.startswith(
        model_state_module.optional_full_checkpoint_state_prefixes(_Model())
    )


def test_prior_peft_allows_missing_reference_checkpoint_state_for_warm_start():
    class _PriorWithPEFT(nn.Module):
        def optional_full_checkpoint_state_prefixes(self):
            return ("reference_prior.", "reference_film_input_norm.")

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.prior_with_peft = _PriorWithPEFT()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = _Actor()

    key = "_actor.prior_with_peft.reference_prior._pos_emb"
    assert key.startswith(
        model_state_module.optional_full_checkpoint_state_prefixes(_Model())
    )


def test_prior_peft_treats_film_input_norm_as_adapter_state():
    assert is_adapter_state_key(
        "prior_with_peft.film_input_norm.running_obs_norm.mean"
    )
    assert is_adapter_state_key(
        "_actor.prior_with_peft.film_input_norm.running_obs_norm.var"
    )


def test_prior_peft_target_kl_skips_actor_update_but_keeps_critic_update():
    agent = _agent(has_critic=True)
    agent.config.target_kl = 0.1
    agent.config.actor_clip_frac_threshold = None
    agent.actor_optimizer = _OptimizerRecorder()
    agent.critic_optimizer = _OptimizerRecorder()
    steps = []

    def actor_step(batch):
        return torch.tensor(1.0, requires_grad=True), {
            "actor/kl": torch.tensor(0.2),
        }

    def critic_step(batch):
        return torch.tensor(1.0, requires_grad=True), {
            "losses/critic_loss": torch.tensor(1.0),
        }

    def step_optimizer(loss, model, optimizer, model_name):
        steps.append(model_name)
        optimizer.step()
        return {f"{model_name}/grad_norm_before_clip": torch.tensor(0.0)}

    agent.actor_step = actor_step
    agent.critic_step = critic_step
    agent._step_optimizer = step_optimizer

    log_dict = DiscretePriorPEFTRLFTAgent.perform_optimization_step(agent, {}, batch_idx=0)

    assert steps == ["critic"]
    assert agent.actor_optimizer.steps == 0
    assert agent.critic_optimizer.steps == 1
    assert log_dict["ppo/kl_early_stopped"].item() == 1.0
    assert log_dict["actor/update_skipped"].item() == 1.0


def test_prior_peft_config_requires_explicit_pretrained_modules():
    field_names = {field.name for field in fields(DiscretePriorPEFTRLFTAgentConfig)}
    assert "prior_checkpoint" not in field_names
    assert "tracker_checkpoint" not in field_names
    assert "training_mode" not in field_names

    prior = PretrainedModelConfig(
        checkpoint_path="prior.ckpt",
        module_path="",
    )
    config = DiscretePriorPEFTRLFTAgentConfig(
        batch_size=8,
        training_max_steps=16,
        pretrained_modules={"prior": prior},
    )
    sft_config = DiscretePriorPEFTSFTAgentConfig(
        batch_size=8,
        training_max_steps=16,
        pretrained_modules={"prior": prior},
    )

    assert config.pretrained_modules == {"prior": prior}
    assert config.entropy_coef == 0.0
    assert config.model.actor.peft.prior_top_p == 0.99
    assert config.model.actor.peft.clear_peft is False
    assert config.save_inference_checkpoint is True
    assert not hasattr(sft_config.model, "critic")
    assert sft_config.loss.target_key == TARGET_LATENT_KEY


def test_prior_peft_load_pretrained_modules_loads_prior_only(monkeypatch):
    calls = []

    def fake_load(
        config,
        device,
        checkpoint_path_overrides=None,
        prefer_inference_config=False,
    ):
        calls.append(
            (
                config.checkpoint_path,
                device,
                checkpoint_path_overrides,
                prefer_inference_config,
            )
        )
        return f"loaded:{config.checkpoint_path}"

    monkeypatch.setattr(pretrained_modules_module, "load_pretrained_model_module", fake_load)

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.fabric = SimpleNamespace(device=torch.device("cpu"))
    agent.config = SimpleNamespace(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path="prior.ckpt",
                module_path="",
            ),
        }
    )

    modules = DiscretePriorPEFTRLFTAgent._load_pretrained_modules(agent)

    assert modules == {"prior": "loaded:prior.ckpt"}
    assert calls == [(
        "prior.ckpt",
        torch.device("cpu"),
        None,
        True,
    )]


def test_prior_peft_load_pretrained_modules_rejects_extra_modules():
    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.fabric = SimpleNamespace(device=torch.device("cpu"))
    agent.config = SimpleNamespace(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path="prior.ckpt",
                module_path="",
            ),
            "extra": PretrainedModelConfig(
                checkpoint_path="extra.ckpt",
                module_path="",
            ),
        }
    )

    with pytest.raises(ValueError, match="Unexpected modules: \\['extra'\\]"):
        DiscretePriorPEFTRLFTAgent._load_pretrained_modules(agent)


def test_prior_peft_load_pretrained_modules_requires_prior():
    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.fabric = SimpleNamespace(device=torch.device("cpu"))
    agent.config = SimpleNamespace(pretrained_modules={})

    with pytest.raises(RuntimeError, match="pretrained_modules\\['prior'\\]"):
        DiscretePriorPEFTRLFTAgent._load_pretrained_modules(agent)


def test_prior_peft_init_delegates_base_init_for_rlft(monkeypatch):
    captured = {}

    def fake_fine_tuning_init(self, fabric, env, config, root_dir=None):
        captured["args"] = (self, fabric, env, config, root_dir)
        self.e_clip = config.e_clip
        self.tau = config.tau

    monkeypatch.setattr(fine_tuning_agent_module.FineTuningAgent, "__init__", fake_fine_tuning_init)

    fabric = object()
    env = object()
    config = SimpleNamespace(
        pretrained_modules={},
        e_clip=0.3,
        tau=0.7,
        model=SimpleNamespace(critic=SimpleNamespace()),
    )
    root_dir = Path("/tmp/protomotions-test")

    agent = DiscretePriorPEFTRLFTAgent(fabric, env, config, root_dir=root_dir)

    assert not hasattr(agent, "training_mode")
    assert agent.e_clip == 0.3
    assert agent.tau == 0.7
    assert captured["args"] == (agent, fabric, env, config, root_dir)

def test_prior_peft_init_rejects_missing_rlft_critic_or_sft_critic(monkeypatch):
    monkeypatch.setattr(
        fine_tuning_agent_module.FineTuningAgent,
        "__init__",
        lambda self, fabric, env, config, root_dir=None: None,
    )
    monkeypatch.setattr(
        "protomotions.agents.supervised.agent.SupervisedAgent.__init__",
        lambda self, fabric, env, config, root_dir=None: None,
    )

    with pytest.raises(ValueError, match="requires config.model.critic"):
        DiscretePriorPEFTRLFTAgent(
            object(),
            object(),
            SimpleNamespace(model=SimpleNamespace(critic=None)),
        )

    with pytest.raises(ValueError, match="does not use a critic"):
        DiscretePriorPEFTSFTAgent(
            object(),
            object(),
            SimpleNamespace(model=SimpleNamespace(critic=SimpleNamespace())),
        )


def test_prior_peft_amp_agent_requires_critic_and_installs_amp_component(monkeypatch):
    def fake_prior_init(self, fabric, env, config, root_dir=None):
        self.fabric = fabric
        self.env = env
        self.config = config
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.num_steps = 4
        self.num_mini_epochs = 1
        self.gamma = 0.95

    monkeypatch.setattr(DiscretePriorPEFTRLFTAgent, "__init__", fake_prior_init)

    config = _config()
    config.batch_size = 4
    config.normalize_rewards = False
    config.normalized_reward_clamp_value = 5.0
    config.amp_parameters = SimpleNamespace(
        use_disc_critic=True,
        discriminator_replay_size=8,
        discriminator_batch_size=2,
    )

    agent = DiscretePriorPEFTRLFTAMPAgent(object(), object(), config)

    assert not hasattr(agent, "training_mode")
    assert agent.amp_component.agent is agent
    assert agent.amp_component.replay_buffer is not None
    assert agent.amp_component.use_disc_critic is True

    bad_config = _config()
    bad_config.model.critic = None
    with pytest.raises(ValueError, match="requires config.model.critic"):
        DiscretePriorPEFTRLFTAMPAgent(object(), object(), bad_config)


def test_prior_peft_amp_warm_starts_from_peft_checkpoint_without_amp_state(
    monkeypatch,
):
    class _LoadablePEFTAMPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = nn.Linear(1, 1, bias=False)
            self._critic = nn.Linear(1, 1, bias=False)
            self._discriminator = nn.Linear(1, 1, bias=False)
            self._disc_critic = nn.Linear(1, 1, bias=False)

    model = _LoadablePEFTAMPModel()
    model._actor.weight.data.fill_(1.0)
    model._critic.weight.data.fill_(2.0)
    model._discriminator.weight.data.fill_(3.0)
    model._disc_critic.weight.data.fill_(4.0)

    checkpoint_model = {
        "_actor.weight": torch.tensor([[10.0]]),
        "_critic.weight": torch.tensor([[20.0]]),
    }
    state = {
        "model": checkpoint_model,
        "epoch": 5,
        "actor_optimizer": {"actor": 1},
        "critic_optimizer": {"critic": 2},
    }
    agent = object.__new__(DiscretePriorPEFTRLFTAMPAgent)
    agent.model = model
    agent.config = _config()
    agent.actor_optimizer = _OptimizerRecorder()
    agent.critic_optimizer = _OptimizerRecorder()
    agent.evaluator = None
    agent.amp_component = SimpleNamespace(has_model_state=lambda state_dict: False)

    monkeypatch.setattr(
        model_state_module,
        "materialize_lazy_running_stats_from_state_dict",
        lambda module, model_state: None,
    )

    DiscretePriorPEFTRLFTAMPAgent.load_parameters(agent, state, load_training_state=True)

    assert torch.equal(model._actor.weight, torch.tensor([[10.0]]))
    assert torch.equal(model._critic.weight, torch.tensor([[20.0]]))
    assert torch.equal(model._discriminator.weight, torch.tensor([[3.0]]))
    assert torch.equal(model._disc_critic.weight, torch.tensor([[4.0]]))
    assert agent.current_epoch == 5
    assert agent.actor_optimizer.loaded_state == {"actor": 1}
    assert agent.critic_optimizer.loaded_state == {"critic": 2}


def test_prior_peft_amp_optimization_runs_peft_then_amp_tail(monkeypatch):
    calls = []

    def fake_peft_step(self, batch_dict, batch_idx=0):
        calls.append(("peft", batch_idx))
        return {"peft/loss": torch.tensor(1.0)}

    monkeypatch.setattr(
        DiscretePriorPEFTRLFTAgent,
        "perform_optimization_step",
        fake_peft_step,
    )

    agent = object.__new__(DiscretePriorPEFTRLFTAMPAgent)
    agent.config = _config()
    agent.config.amp_parameters = SimpleNamespace(
        discriminator_optimization_ratio=1,
    )
    agent._skip_actor_for_epoch = False
    agent.use_disc_critic = True
    agent.disc_critic = nn.Linear(1, 1)
    agent.discriminator = nn.Linear(1, 1)
    agent.disc_critic_optimizer = _OptimizerRecorder()
    agent.discriminator_optimizer = _OptimizerRecorder()

    def fake_amp_tail(batch_dict, batch_idx, iter_log_dict):
        calls.append(("amp_tail", batch_idx, batch_dict["action"].shape[0]))
        return {**iter_log_dict, "amp/tail": torch.tensor(1.0)}

    agent.amp_component = SimpleNamespace(optimize_batch_tail=fake_amp_tail)

    batch = {"action": torch.zeros(2, 1)}

    log_dict = DiscretePriorPEFTRLFTAMPAgent.perform_optimization_step(
        agent,
        batch,
        batch_idx=4,
    )

    assert calls == [("peft", 4), ("amp_tail", 4, 2)]
    assert set(log_dict) == {"peft/loss", "amp/tail"}
    assert torch.equal(log_dict["peft/loss"], torch.tensor(1.0))
    assert torch.equal(log_dict["amp/tail"], torch.tensor(1.0))


def test_prior_peft_config_default_keeps_task_reward_advantages_enabled():
    config = DiscretePriorPEFTRLFTAgentConfig(batch_size=2, training_max_steps=4)

    assert config.task_reward_w == 1.0


def test_prior_peft_actor_config_validates_input_role_mapping():
    with pytest.raises(AssertionError, match="in_keys"):
        DiscretePriorPEFTActorConfig(
            in_keys=["task_obs"],
            peft=DiscretePriorPEFTConfig(
                model=ModuleContainerConfig(
                    in_keys=["max_coords_obs", "task_obs"],
                    out_keys=["task_cond"],
                    models=[
                        MLPWithConcatConfig(
                            in_keys=["max_coords_obs", "task_obs"],
                            out_keys=["task_cond"],
                            num_out=16,
                        ),
                    ],
                ),
            ),
        )

    with pytest.raises(AssertionError, match="task_cond"):
        DiscretePriorPEFTActorConfig(
            in_keys=["max_coords_obs"],
            peft=DiscretePriorPEFTConfig(
                model=ModuleContainerConfig(
                    in_keys=["max_coords_obs"],
                    out_keys=["max_coords_obs"],
                ),
            ),
        )


def test_prior_peft_validates_peft_inputs_and_resolves_mimic_dimension(monkeypatch):
    monkeypatch.setattr(
        prior_setup_module,
        "resolve_frozen_prior_input_keys",
        lambda prior_model: ["max_coords_obs"],
    )
    agent = _agent()
    agent.pretrained = {"prior": object()}
    agent.env = _Env(_obs())

    DiscretePriorPEFTRLFTAgent._validate_peft_inputs(agent)
    mimic_dim = DiscretePriorPEFTRLFTAgent._resolve_mimic_target_dim(agent)

    assert mimic_dim == 5

    agent.env = _Env({"max_coords_obs": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="actor inputs"):
        DiscretePriorPEFTRLFTAgent._validate_peft_inputs(agent)
    assert DiscretePriorPEFTRLFTAgent._resolve_mimic_target_dim(agent) == 0

    agent.config.model.actor.peft.model.in_keys = [
        "historical_pose_obs",
    ]
    agent.env = _Env(_obs())
    DiscretePriorPEFTRLFTAgent._validate_peft_inputs(agent)
    assert DiscretePriorPEFTRLFTAgent._resolve_mimic_target_dim(agent) == 5


def test_prior_peft_create_model_passes_encoder_dims_for_sft_with_mimic_targets(
    monkeypatch,
):
    constructed = []

    class _RecordingModel:
        def __init__(
            self, config, pretrained_prior_model, mimic_target_poses_dim
        ):
            constructed.append(
                {
                    "config": config,
                    "pretrained_prior_model": pretrained_prior_model,
                    "mimic_target_poses_dim": mimic_target_poses_dim,
                }
            )

    monkeypatch.setattr(prior_setup_module, "get_class", lambda target: _RecordingModel)
    monkeypatch.setattr(
        prior_setup_module,
        "resolve_frozen_prior_input_keys",
        lambda prior_model: ["max_coords_obs"],
    )

    agent = _agent()
    agent.env = _Env(_obs())
    prior_model = object()
    agent.pretrained = {"prior": prior_model}
    agent.config.model._target_ = "unused.model"

    model = DiscretePriorPEFTRLFTAgent.create_model(agent)

    assert isinstance(model, _RecordingModel)
    assert constructed[-1]["config"] is agent.config.model
    assert constructed[-1]["pretrained_prior_model"] is prior_model
    assert constructed[-1]["mimic_target_poses_dim"] == 0

    no_encoder_obs = {
        "max_coords_obs": torch.zeros(2, 4),
        "mimic_target_poses": torch.ones(2, 5),
    }
    agent.config.model.actor.in_keys = []
    agent.config.model.actor.peft.model.in_keys = []
    agent.env = _Env(no_encoder_obs)

    DiscretePriorPEFTRLFTAgent.create_model(agent)

    assert constructed[-1]["mimic_target_poses_dim"] == 0

    sft_agent = object.__new__(DiscretePriorPEFTSFTAgent)
    sft_agent.env = agent.env
    sft_agent.pretrained = agent.pretrained
    sft_agent.config = agent.config

    DiscretePriorPEFTSFTAgent.create_model(sft_agent)

    assert constructed[-1]["mimic_target_poses_dim"] == 5


def test_prior_peft_setup_loads_models_materializes_and_initializes_optimizers(
    monkeypatch,
):
    calls = []

    class _FakeRunningMeanStd(nn.Module):
        pass

    monkeypatch.setattr(base_agent_module, "RunningMeanStd", _FakeRunningMeanStd)

    class _Fabric:
        device = torch.device("cpu")

        def call(self, name, *args):
            calls.append(name)

    class _SetupActor(_Actor):
        def __init__(self):
            super().__init__()
            self.prior = SimpleNamespace(context_in_keys=[])
            self.init_warmup_obs = None

        def init_peft(self, warmup_obs=None):
            self.init_warmup_obs = warmup_obs
            calls.append("init_peft")

    class _SetupModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = _SetupActor()
            self._critic = _Critic()
            self.rms = _FakeRunningMeanStd()
            self.reset_args = None

        def reset_rollout_context(self, num_envs, device):
            self.reset_args = (num_envs, device)

        def forward(self, tensordict):
            return self._actor.get_action_and_logp(tensordict)

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.fabric = _Fabric()
    agent.device = agent.fabric.device
    agent.env = _Env(_obs())
    agent.num_envs = 2
    agent.add_agent_info_to_obs = lambda obs: {
        **obs,
        "agent_info": torch.ones(obs["max_coords_obs"].shape[0], 1),
    }
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(
        obs,
        batch_size=obs["max_coords_obs"].shape[0],
    )
    model = _SetupModel()
    agent._load_pretrained_modules = lambda: {"prior": object()}
    agent.create_model = lambda: model
    agent.create_optimizers = lambda created_model: calls.append(
        ("create_optimizers", created_model)
    )
    agent._print_param_info = lambda: calls.append("print_param_info")

    DiscretePriorPEFTRLFTAgent.setup(agent)

    assert agent.model is model
    assert model.reset_args == (2, torch.device("cpu"))
    assert torch.equal(
        model._actor.init_warmup_obs["max_coords_obs"],
        _obs()["max_coords_obs"],
    )
    assert torch.equal(
        model._actor.init_warmup_obs["agent_info"],
        torch.ones(2, 1),
    )
    assert model.rms.fabric is agent.fabric
    assert agent.pretrained == {}
    assert calls[:2] == ["on_model_init_start", "init_peft"]
    assert "on_model_init_end" in calls
    assert "on_optimizer_init_start" in calls
    assert "on_optimizer_init_end" in calls
    assert ("create_optimizers", model) in calls
    assert "print_param_info" in calls


def test_prior_peft_create_optimizers_wraps_actor_and_critic(monkeypatch):
    agent = _agent(has_critic=True)
    model = SimpleNamespace(_actor=_Actor(), _critic=_ParamCritic())
    agent.model = model
    agent.config.model.actor_optimizer = SimpleNamespace(name="actor")
    agent.config.model.critic_optimizer = SimpleNamespace(name="critic")
    created = []

    def instantiate_optimizer(config, module, params=None):
        params = None if params is None else list(params)
        created.append((config.name, module, params))
        return _OptimizerRecorder(state={"optimizer": config.name})

    def setup_model_optimizer(module, optimizer):
        return _Wrapper(module), optimizer

    monkeypatch.setattr(ppo_agent_module, "instantiate_optimizer", instantiate_optimizer)
    agent._setup_model_optimizer = setup_model_optimizer

    DiscretePriorPEFTRLFTAgent.create_optimizers(agent, model)

    assert created[0] == ("actor", model._actor, None)
    assert created[1][0] == "critic"
    assert created[1][1] is model._critic
    assert created[1][2] is None
    assert agent.actor.module is model._actor
    assert agent.critic.module is model._critic


def test_prior_peft_print_param_info_reports_trainable_and_critic_counts(caplog):
    caplog.set_level("INFO", logger=prior_setup_module.log.name)
    agent = _agent(has_critic=True)
    agent.model._actor.frozen = nn.Parameter(torch.ones(2), requires_grad=False)

    DiscretePriorPEFTRLFTAgent._print_param_info(agent)

    out = caplog.text
    assert "PEFT parameters: actor trainable 1 / 3" in out
    assert "critic 0 trainable" in out
    assert "[TRAIN]" not in out


def test_prior_peft_registers_model_outputs_and_algorithm_keys():
    agent = _agent(has_critic=True)
    agent.experience_buffer = _ExperienceBufferRecorder()

    assert agent.model.experience_buffer_keys() == [
        "action",
        "mean_action",
        "neglogp",
        "prior_tokens",
        "value",
    ]
    agent.experience_buffer.value = torch.zeros(2, 4, 1)
    DiscretePriorPEFTRLFTAgent.register_algorithm_experience_buffer_keys(agent)

    registered = {key for key, _, _ in agent.experience_buffer.registered}
    assert {
        "next_value",
        "returns",
        "advantages",
    }.issubset(registered)
    assert "values" not in registered
    assert "next_values" not in registered

    assert not hasattr(DiscretePriorPEFTRLFTAgent, "_collect_rollout_step_sft")


def test_prior_peft_collect_rollout_records_policy_outputs_and_values():
    agent = _agent(has_critic=True)
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent.model_output_keys = agent.model.experience_buffer_keys()
    obs_td = TensorDict(_obs(), batch_size=2)

    output_td = DiscretePriorPEFTRLFTAgent.collect_rollout_step(agent, obs_td, step=4)

    assert torch.equal(output_td["action"], torch.tensor([[0.0, 1.0], [4.0, 5.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("prior_tokens", 4)],
        torch.tensor([[0, 1, 2], [3, 4, 0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("value", 4)],
        torch.tensor([[1.0], [5.0]]),
    )
    assert ("neglogp", 4) in agent.experience_buffer.data
    assert ("values", 4) not in agent.experience_buffer.data


def test_prior_peft_record_rollout_step_records_next_value_after_base_metrics():
    agent = _agent(has_critic=True)
    agent.current_rewards = torch.zeros(2)
    agent.current_lengths = torch.zeros(2, dtype=torch.long)
    agent.episode_reward_meter = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.episode_length_meter = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.episode_env_tensors = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.experience_buffer = _ExperienceBufferRecorder()

    DiscretePriorPEFTRLFTAgent.record_rollout_step(
        agent,
        next_obs_td=TensorDict(_obs(), batch_size=2),
        actions=torch.zeros(2, 2),
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([False, True]),
        terminated=torch.tensor([False, True]),
        done_indices=torch.tensor([1]),
        extras={},
        step=1,
    )

    assert torch.equal(
        agent.experience_buffer.data[("rewards", 1)],
        torch.tensor([1.0, 2.0]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("next_value", 1)],
        torch.tensor([[1.0], [0.0]]),
    )
    assert ("next_values", 1) not in agent.experience_buffer.data


def test_prior_peft_fit_runs_one_epoch_rollout_eval_and_checkpoint_branches(
    monkeypatch,
):
    calls = []
    buffer = _ExperienceBufferRecorder()
    monkeypatch.setattr(
        base_agent_module,
        "ExperienceBuffer",
        lambda *args, **kwargs: buffer,
    )
    monkeypatch.setattr(
        base_agent_module,
        "track",
        lambda iterable, description=None: iterable,
    )
    monkeypatch.setattr(
        base_agent_module,
        "aggregate_scalar_metrics",
        lambda metrics, fabric, weight=1: dict(metrics),
    )

    class _Fabric:
        def call(self, name, *args):
            calls.append(("fabric", name, len(args)))

        def broadcast(self, value):
            return value

    class _FitEnv:
        def __init__(self):
            self.reset_args = []
            self.step_actions = []
            self.epoch_ends = []
            self.max_episode_length = 3

        def reset(self, done_indices=None):
            self.reset_args.append(None if done_indices is None else done_indices.clone())
            return _obs(), {}

        def step(self, action):
            self.step_actions.append(action.clone())
            return (
                _obs(),
                torch.tensor([1.0, 2.0]),
                torch.tensor([False, True]),
                torch.tensor([False, True]),
                {"raw": torch.ones(2)},
            )

        def on_epoch_end(self, epoch):
            self.epoch_ends.append(epoch)

    class _Evaluator:
        config = SimpleNamespace(eval_metrics_every=1)

        def evaluate(self):
            calls.append(("evaluate",))
            return {"eval/success": 2.0}, 3.0, 4

    class _LengthManager:
        def current_max_episode_length(self, epoch):
            calls.append(("length_manager", epoch))
            return 10 + epoch

    class _FitModel:
        training = True

        def modules(self):
            yield self

        def __call__(self, obs_td):
            calls.append(("register_outputs",))
            return TensorDict(
                {"action": torch.zeros(2, 2)},
                batch_size=2,
            )

        def experience_buffer_keys(self):
            return ["action"]

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.num_envs = 2
    agent.num_steps = 2
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.max_epochs = 1
    agent.fit_start_time = None
    agent.step_count = 0
    agent._skip_next_policy_update = False
    agent.just_loaded_checkpoint_should_evaluate = False
    agent.best_evaluated_score = None
    agent._should_stop = False
    agent.fabric = _Fabric()
    agent.env = _FitEnv()
    agent.evaluator = _Evaluator()
    agent.model = _FitModel()
    agent.config = SimpleNamespace(
        normalize_rewards=True,
        save_epoch_checkpoint_every=1,
        save_last_checkpoint_every=1,
        max_episode_length_manager=_LengthManager(),
    )
    agent.time_report = SimpleNamespace(report=lambda: calls.append(("time_report",)))
    agent.add_agent_info_to_obs = lambda obs: obs
    agent.add_agent_info_to_next_obs = lambda obs: obs
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(
        obs,
        batch_size=obs["max_coords_obs"].shape[0],
    )
    agent.register_algorithm_experience_buffer_keys = lambda: calls.append(
        ("register_algo",)
    )
    agent.eval = lambda: calls.append(("eval_mode",))
    agent.pre_collect_step = lambda step: calls.append(("pre_collect", step))
    agent.collect_rollout_step = lambda obs_td, step: TensorDict(
        {"action": torch.ones(2, 2) * (step + 1)},
        batch_size=2,
    )
    agent.check_for_nans = lambda *tds: calls.append(("check_nans", tds[0].batch_size))
    agent.post_env_step_modifications = lambda dones, terminated, extras: (
        dones,
        terminated,
        extras,
    )
    agent.record_rollout_step = lambda *args, **kwargs: calls.append(
        ("record_rollout", kwargs["step"] if "step" in kwargs else args[-1])
    )
    agent.get_step_count_increment = lambda: 5
    agent.normalize_rewards_in_buffer = lambda: calls.append(("normalize_rewards",))
    agent.optimize_model = lambda: {"train/loss": 1.0}
    agent.save = lambda checkpoint_name, new_high_score=False: calls.append(
        ("save", checkpoint_name, new_high_score)
    )
    agent.post_epoch_logging = lambda log_dict: calls.append(("post_log", dict(log_dict)))
    agent._prepare_rlft_prior_reference = lambda: calls.append(("prepare_reference",))

    DiscretePriorPEFTRLFTAgent.fit(agent)

    assert buffer.registered
    assert ("rewards", (), None) in buffer.registered
    assert ("unnormalized_rewards", (), None) in buffer.registered
    assert ("dones", (), torch.long) in buffer.registered
    assert ("register_outputs",) in calls
    assert ("register_algo",) in calls
    assert [call for call in calls if call[0] == "pre_collect"] == [
        ("pre_collect", 0),
        ("pre_collect", 1),
    ]
    assert agent.step_count == 10
    assert agent.current_epoch == 1
    assert agent.best_evaluated_score == 3.0
    assert agent._skip_next_policy_update is True
    assert agent.env.max_episode_length == 11
    assert ("save", "epoch_1.ckpt", False) in calls
    assert ("save", "last.ckpt", True) in calls
    assert calls[-1] == ("fabric", "on_fit_end", 1)
    assert ("time_report",) in calls


def test_prior_peft_fit_skip_update_branch_can_stop_training(monkeypatch):
    calls = []
    buffer = _ExperienceBufferRecorder()
    monkeypatch.setattr(
        base_agent_module,
        "ExperienceBuffer",
        lambda *args, **kwargs: buffer,
    )
    monkeypatch.setattr(
        base_agent_module,
        "track",
        lambda iterable, description=None: iterable,
    )

    class _Fabric:
        def call(self, name, *args):
            calls.append(("fabric", name))

        def broadcast(self, value):
            return value

    class _FitEnv:
        max_episode_length = 3

        def reset(self, done_indices=None):
            return _obs(), {}

        def step(self, action):
            return (
                _obs(),
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

        def on_epoch_end(self, epoch):
            calls.append(("epoch_end", epoch))

    class _FitModel:
        training = True

        def modules(self):
            yield self

        def __call__(self, obs_td):
            return TensorDict(
                {"action": torch.zeros(2, 2)},
                batch_size=2,
            )

        def experience_buffer_keys(self):
            return ["action"]

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.num_envs = 2
    agent.num_steps = 1
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.max_epochs = 2
    agent.fit_start_time = 1.0
    agent.step_count = 0
    agent._skip_next_policy_update = True
    agent.just_loaded_checkpoint_should_evaluate = False
    agent.best_evaluated_score = None
    agent._should_stop = True
    agent.fabric = _Fabric()
    agent.env = _FitEnv()
    agent.evaluator = None
    agent.model = _FitModel()
    agent.config = SimpleNamespace(
        normalize_rewards=False,
        save_epoch_checkpoint_every=None,
        save_last_checkpoint_every=10,
        max_episode_length_manager=None,
    )
    agent.time_report = SimpleNamespace(report=lambda: calls.append(("time_report",)))
    agent.add_agent_info_to_obs = lambda obs: obs
    agent.add_agent_info_to_next_obs = lambda obs: obs
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(
        obs,
        batch_size=obs["max_coords_obs"].shape[0],
    )
    agent.register_algorithm_experience_buffer_keys = lambda: None
    agent.eval = lambda: None
    agent.pre_collect_step = lambda step: None
    agent.collect_rollout_step = lambda obs_td, step: TensorDict(
        {"action": torch.zeros(2, 2)},
        batch_size=2,
    )
    agent.check_for_nans = lambda *tds: None
    agent.post_env_step_modifications = lambda dones, terminated, extras: (
        dones,
        terminated,
        extras,
    )
    agent.record_rollout_step = lambda *args, **kwargs: None
    agent.get_step_count_increment = lambda: 1
    agent.normalize_rewards_in_buffer = lambda: None
    agent.pre_process_dataset = lambda: calls.append(("pre_process",))
    agent.optimize_model = lambda: pytest.fail("skip branch should avoid optimize_model")
    agent.save = lambda checkpoint_name, new_high_score=False: calls.append(
        ("save", checkpoint_name, new_high_score)
    )
    agent.post_epoch_logging = lambda log_dict: calls.append(("post_log", dict(log_dict)))
    agent._prepare_rlft_prior_reference = lambda: calls.append(("prepare_reference",))

    DiscretePriorPEFTRLFTAgent.fit(agent)

    assert buffer.make_dict_calls == 1
    assert ("pre_process",) in calls
    assert ("post_log", {"skipped_policy_update": 1.0, "epoch": 0}) in calls
    assert ("fabric", "on_training_stop") in calls
    assert ("save", "last.ckpt", False) in calls
    assert ("time_report",) not in calls


def test_prior_peft_pre_process_dataset_computes_returns_and_normalized_advantages():
    agent = _agent(has_critic=True)
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent.experience_buffer.dones = torch.tensor([[0, 0], [1, 0]], dtype=torch.float)
    agent.experience_buffer.value = torch.tensor([[[0.5], [1.0]], [[1.5], [2.0]]])
    agent.experience_buffer.rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    agent.experience_buffer.next_value = torch.tensor(
        [[[0.25], [0.5]], [[0.75], [1.0]]]
    )

    DiscretePriorPEFTRLFTAgent.pre_process_dataset(agent)

    expected_advantages = discount_values(
        agent.experience_buffer.dones,
        agent.experience_buffer.value.squeeze(-1),
        agent.experience_buffer.rewards,
        agent.experience_buffer.next_value.squeeze(-1),
        agent.gamma,
        agent.tau,
    )
    assert torch.allclose(
        agent.experience_buffer.returns,
        expected_advantages + agent.experience_buffer.value.squeeze(-1),
    )
    assert torch.allclose(
        agent.experience_buffer.advantages,
        expected_advantages,
    )


def test_prior_peft_actor_step_uses_rlft_task_loss(monkeypatch):
    agent = _agent()
    batch = {"max_coords_obs": torch.zeros(1, 4)}
    called = []

    monkeypatch.setattr(
        agent,
        "_actor_step_discrete_ppo",
        lambda received: called.append(received) or (torch.tensor(2.0), {}),
    )

    loss, _ = DiscretePriorPEFTRLFTAgent.actor_step(agent, batch)

    assert torch.equal(loss, torch.tensor(2.0))
    assert called == [batch]
    assert not hasattr(DiscretePriorPEFTRLFTAgent, "_actor_step_sft")


def test_prior_peft_sft_model_teacher_forces_without_model_loss():
    logits = torch.tensor(
        [
            [
                [3.0, 0.0, -1.0, -2.0, -3.0],
                [0.0, 4.0, -2.0, -3.0, -4.0],
                [0.0, 1.0, 5.0, -1.0, -2.0],
            ],
            [
                [-1.0, 0.0, 5.0, -2.0, -3.0],
                [2.0, 0.0, -1.0, -2.0, -3.0],
                [0.0, 1.0, -1.0, 4.0, -2.0],
            ],
        ],
        requires_grad=True,
    )
    prior_logits = torch.zeros_like(logits)
    extra_context = torch.ones(2, 2)
    actor = _LatentActor(
        encoder=object(),
        kl_coeff=0.5,
        extra_context=extra_context,
        prior_logits=prior_logits,
    )
    actor.forward = lambda prior_dict: logits

    model = object.__new__(DiscretePriorPEFTSFTModel)
    nn.Module.__init__(model)
    model._actor = actor
    model.config = SimpleNamespace(token_perturb_rate=0.0, token_perturb_mode="replace")
    target = torch.tensor([[0, 1, 2], [2, 0, 3]])
    td = TensorDict({**_obs(), TARGET_LATENT_KEY: target}, batch_size=2)

    td = DiscretePriorPEFTSFTModel.forward(model, td)
    loss, log_dict = DiscretePriorPEFTSFTModel.compute_model_loss(
        model,
        td,
        current_epoch=0,
        zero_loss=logits.sum() * 0.0,
        log_prefix="sft_model",
    )

    assert torch.equal(loss, logits.sum() * 0.0)
    assert log_dict == {}
    assert not hasattr(actor.prior_with_peft, "last_prior_dict")
    assert torch.equal(td[LATENT_LOGITS_KEY], logits)


def test_prior_peft_task_actor_step_uses_critic_advantages_and_prior_kl():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0, -2.0, -3.0],
                [0.0, 2.0, -1.0, -2.0, -3.0],
                [0.0, -1.0, 2.0, -2.0, -3.0],
            ],
            [
                [-1.0, 0.0, 2.0, -2.0, -3.0],
                [2.0, 0.0, -1.0, -2.0, -3.0],
                [0.0, -1.0, -2.0, 2.0, -3.0],
            ],
        ],
        requires_grad=True,
    )
    prior_logits = torch.zeros_like(logits)
    extra_context = torch.ones(2, 2)
    actor = _LatentActor(
        encoder=None,
        kl_coeff=0.25,
        extra_context=extra_context,
        prior_logits=prior_logits,
    )
    agent = _agent(has_critic=True)
    agent.model._actor = actor
    agent.actor = lambda prior_dict: logits
    advantages = torch.tensor([-0.5, 1.5])
    batch = {
        **_obs(),
        "prior_tokens": torch.tensor([[0, 1, 2], [2, 0, 3]]),
        "neglogp": torch.zeros(2, 3),
        "advantages": advantages,
        "rewards": torch.tensor([1.0, 3.0]),
    }

    loss, log_dict = DiscretePriorPEFTRLFTAgent._actor_step_discrete_ppo(agent, batch)

    assert loss.requires_grad is True
    assert torch.equal(
        actor.prior_with_peft.last_prior_dict["extra_context"],
        extra_context,
    )
    actor_logp = prior_constrained_sampling_log_probs(
        logits,
        prior_logits,
        p=actor.prior_with_peft.prior_top_p,
        temperature=actor.prior_with_peft.temperature,
    )
    reference_logp = prior_constrained_sampling_log_probs(
        prior_logits,
        prior_logits,
        p=actor.prior_with_peft.prior_top_p,
        temperature=actor.prior_with_peft.temperature,
    )
    actor_probs = actor_logp.exp()
    expected_kl = torch.where(
        actor_probs > 0,
        actor_probs * (actor_logp - reference_logp),
        torch.zeros_like(actor_probs),
    ).sum(dim=-1).mean()
    assert torch.allclose(log_dict["actor/kl_prior_loss"], expected_kl)
    assert torch.allclose(log_dict["actor/adv_mean"], torch.tensor(0.0))
    assert torch.allclose(log_dict["actor/adv_std"], torch.tensor(1.0))
    assert torch.allclose(log_dict["stats/reward_mean"], torch.tensor(2.0))


def test_prior_peft_task_actor_step_requires_advantages():
    actor = _LatentActor(encoder=None, kl_coeff=0.0, extra_context=None)
    agent = _agent()
    agent.model._actor = actor
    batch = {
        **_obs(),
        "prior_tokens": torch.tensor([[0, 1, 2], [2, 0, 3]]),
        "neglogp": torch.zeros(2, 3),
        "rewards": torch.tensor([1.0, 3.0]),
    }

    with pytest.raises(KeyError, match="advantages"):
        DiscretePriorPEFTRLFTAgent._actor_step_discrete_ppo(agent, batch)


def test_prior_peft_task_actor_step_uses_critic_advantages_without_prior_kl():
    logits = torch.zeros(2, 3, 5, requires_grad=True)
    actor = _LatentActor(encoder=None, kl_coeff=0.0, extra_context=None)
    agent = _agent(has_critic=True)
    agent.model._actor = actor
    agent.actor = lambda prior_dict: logits
    advantages = torch.tensor([0.5, -0.25])
    batch = {
        **_obs(),
        "prior_tokens": torch.tensor([[0, 1, 2], [2, 0, 3]]),
        "neglogp": torch.zeros(2, 3),
        "advantages": advantages,
        "rewards": torch.tensor([1.0, 3.0]),
    }

    _, log_dict = DiscretePriorPEFTRLFTAgent._actor_step_discrete_ppo(agent, batch)

    assert torch.allclose(log_dict["actor/adv_mean"], torch.tensor(0.0))
    assert torch.allclose(log_dict["actor/adv_std"], torch.tensor(1.0))
    assert torch.equal(log_dict["actor/kl_prior_loss"], torch.tensor(0.0))


def test_prior_peft_critic_step_reports_value_and_return_stats():
    agent = _agent(has_critic=True)

    class _RecordingCritic(_Critic):
        def __init__(self):
            super().__init__()
            self.seen_keys = None

        def forward(self, tensordict):
            self.seen_keys = set(tensordict.keys())
            return super().forward(tensordict)

    critic = _RecordingCritic()
    agent.model._critic = critic
    agent.critic = _Wrapper(critic)
    batch = {
        **_obs(),
        "action": torch.zeros(2, 2),
        "prior_tokens": torch.zeros(2, 3, dtype=torch.long),
        "returns": torch.tensor([2.0, 6.0]),
    }

    loss, log_dict = DiscretePriorPEFTRLFTAgent.critic_step(agent, batch)

    assert torch.allclose(loss, torch.tensor(1.0))
    assert torch.equal(log_dict["stats/value_mean"], torch.tensor(3.0))
    assert torch.equal(log_dict["stats/return_mean"], torch.tensor(4.0))
    assert {"max_coords_obs", "task_obs", "action", "prior_tokens"}.issubset(
        critic.seen_keys
    )


def test_prior_peft_get_state_dict_adds_actor_and_critic_optimizers():
    agent = _agent(has_critic=True)
    agent.model = nn.Module()
    agent.model._actor = _Actor()
    agent.model._critic = _ParamCritic()
    agent.config.normalize_rewards = True
    agent.running_reward_norm = SimpleNamespace(state_dict=lambda: {"reward": 1})
    agent.evaluator = SimpleNamespace(get_state_dict=lambda: {"eval": 2})
    agent.current_epoch = 4
    agent.step_count = 12
    agent.fit_start_time = 34.0
    agent.best_evaluated_score = 56.0
    agent.actor_optimizer = _OptimizerRecorder(state={"actor_opt": 1})
    agent.critic_optimizer = _OptimizerRecorder(state={"critic_opt": 2})

    state = DiscretePriorPEFTRLFTAgent.get_state_dict(agent, {})

    assert "model" in state
    assert state["epoch"] == 4
    assert state["actor_optimizer"] == {"actor_opt": 1}
    assert state["critic_optimizer"] == {"critic_opt": 2}
    assert state["running_reward_norm"] == {"reward": 1}
    assert state["evaluator"] == {"eval": 2}


def test_prior_peft_get_state_dict_saves_advantage_ema_when_enabled():
    agent = _agent(has_critic=True)
    agent.model = nn.Module()
    agent.model._actor = _Actor()
    agent.model._critic = _ParamCritic()
    agent.config.advantage_normalization = SimpleNamespace(
        enabled=True,
        use_ema=True,
    )
    agent.config.normalize_rewards = False
    agent.evaluator = None
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent.actor_optimizer = _OptimizerRecorder(state={"actor_opt": 1})
    agent.critic_optimizer = _OptimizerRecorder(state={"critic_opt": 2})
    agent.adv_mean_ema = torch.tensor([3.0])
    agent.adv_std_ema = torch.tensor([4.0])

    state = DiscretePriorPEFTRLFTAgent.get_state_dict(agent, {})

    assert torch.equal(state["adv_mean_ema"], torch.tensor([3.0]))
    assert torch.equal(state["adv_std_ema"], torch.tensor([4.0]))


def test_prior_peft_load_parameters_restores_training_state_and_optimizers(
    monkeypatch,
):
    loaded = []

    class _LoadableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = _Actor()
            self._critic = _ParamCritic()

        def load_state_dict(self, state_dict, strict=True):
            loaded.append(("model", strict, set(state_dict)))
            return super().load_state_dict(state_dict, strict=strict)

    model = _LoadableModel()
    state = {
        "model": model.state_dict(),
        "epoch": 7,
        "step_count": 99,
        "run_start_time": 123.0,
        "best_evaluated_score": 8.5,
        "running_reward_norm": {"reward_norm": 1},
        "evaluator": {"eval": 2},
        "actor_optimizer": {"actor": 3},
        "critic_optimizer": {"critic": 4},
        "adv_mean_ema": torch.tensor([5.0]),
        "adv_std_ema": torch.tensor([6.0]),
    }
    agent = _agent(has_critic=True)
    agent.model = model
    agent.config.normalize_rewards = True
    agent.config.advantage_normalization = SimpleNamespace(
        enabled=True,
        use_ema=True,
    )
    agent.adv_mean_ema = torch.zeros(1)
    agent.adv_std_ema = torch.ones(1)
    agent.running_reward_norm = SimpleNamespace(
        load_state_dict=lambda state_dict: loaded.append(("reward_norm", state_dict))
    )
    agent.evaluator = SimpleNamespace(
        load_state_dict=lambda state_dict: loaded.append(("evaluator", state_dict))
    )
    agent.actor_optimizer = _OptimizerRecorder()
    agent.critic_optimizer = _OptimizerRecorder()
    monkeypatch.setattr(
        model_state_module,
        "materialize_lazy_running_stats_from_state_dict",
        lambda module, model_state: loaded.append(("materialize", module, set(model_state))),
    )

    DiscretePriorPEFTRLFTAgent.load_parameters(agent, state, load_training_state=True)

    assert agent.current_epoch == 7
    assert agent.step_count == 99
    assert agent.fit_start_time == 123.0
    assert agent.best_evaluated_score == 8.5
    assert ("reward_norm", {"reward_norm": 1}) in loaded
    assert ("evaluator", {"eval": 2}) in loaded
    assert agent.actor_optimizer.loaded_state == {"actor": 3}
    assert agent.critic_optimizer.loaded_state == {"critic": 4}
    assert torch.equal(agent.adv_mean_ema, torch.tensor([5.0]))
    assert torch.equal(agent.adv_std_ema, torch.tensor([6.0]))
    assert any(item[0] == "materialize" for item in loaded)
    assert any(item[:2] == ("model", False) for item in loaded)


def test_prior_peft_load_parameters_warm_starts_rl_from_sft_checkpoint(
    monkeypatch,
):
    loaded = []

    class _LoadableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = _Actor()
            self._critic = _ParamCritic()

        def load_state_dict(self, state_dict, strict=True):
            loaded.append(("model", strict, set(state_dict)))
            return super().load_state_dict(state_dict, strict=strict)

    model = _LoadableModel()
    state = {
        "model": model.state_dict(),
        "epoch": 7,
        "step_count": 99,
        "run_start_time": 123.0,
        "best_evaluated_score": 8.5,
        "actor_optimizer": {"actor": 3},
    }
    agent = _agent(has_critic=True)
    agent.model = model
    agent.config.normalize_rewards = True
    agent.actor_optimizer = _OptimizerRecorder()
    agent.critic_optimizer = _OptimizerRecorder()
    agent.evaluator = None
    monkeypatch.setattr(
        model_state_module,
        "materialize_lazy_running_stats_from_state_dict",
        lambda module, model_state: loaded.append(("materialize", module, set(model_state))),
    )

    DiscretePriorPEFTRLFTAgent.load_parameters(agent, state, load_training_state=True)

    assert agent.current_epoch == 0
    assert agent.step_count == 0
    assert agent.fit_start_time is None
    assert agent.best_evaluated_score is None
    assert agent.actor_optimizer.loaded_state == {"actor": 3}
    assert agent.critic_optimizer.loaded_state is None
    assert any(item[0] == "materialize" for item in loaded)
    assert any(item[:2] == ("model", False) for item in loaded)


def test_prior_peft_load_parameters_reuses_ppo_training_state_restore(monkeypatch):
    calls = []

    def fake_ppo_load(self, state_dict, load_training_state=True):
        calls.append((state_dict, load_training_state))

    monkeypatch.setattr(
        "protomotions.agents.ppo.agent.PPO.load_parameters",
        fake_ppo_load,
    )
    agent = _agent(has_critic=True)
    state = {"model": {}}

    DiscretePriorPEFTRLFTAgent.load_parameters(agent, state, load_training_state=False)

    assert calls == [(state, False)]


def test_prior_peft_restores_configured_frozen_prior_from_checkpoint(tmp_path):
    class _Recorder:
        def __init__(self):
            self.loaded = []

        def load_state_dict(self, state_dict, strict=True):
            self.loaded.append((dict(state_dict), strict))

    context = _Recorder()
    token = _Recorder()
    output = _Recorder()
    layer = _Recorder()
    pos_emb = torch.zeros(1, 2, 3)

    base_prior = SimpleNamespace(
        _context_encoder=context,
        _token_encoder=token,
        _output_head=output,
        _transformer=SimpleNamespace(
            layers=[SimpleNamespace(transformer_layer=layer)]
        ),
        _pos_emb=pos_emb,
    )
    peft = SimpleNamespace(
        base_prior=base_prior,
    )
    checkpoint = tmp_path / "prior.ckpt"
    prior_pos_emb = torch.full_like(pos_emb, 7.0)
    torch.save(
        {
            "model": {
                "prior._context_encoder.weight": torch.tensor([1.0]),
                "prior._token_encoder.weight": torch.tensor([2.0]),
                "prior._output_head.weight": torch.tensor([3.0]),
                "prior._transformer.layers.0.weight": torch.tensor([4.0]),
                "prior._pos_emb": prior_pos_emb,
            }
        },
        checkpoint,
    )

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.device = torch.device("cpu")
    agent.config = SimpleNamespace(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=str(checkpoint),
                module_path="",
            )
        }
    )
    agent.model = SimpleNamespace(
        _actor=SimpleNamespace(prior_with_peft=peft)
    )

    DiscretePriorPEFTRLFTAgent._restore_configured_frozen_prior(agent)

    assert context.loaded[0][1] is False
    assert token.loaded[0][1] is False
    assert output.loaded[0][1] is False
    assert layer.loaded[0][1] is True
    assert torch.equal(context.loaded[0][0]["weight"], torch.tensor([1.0]))
    assert torch.equal(token.loaded[0][0]["weight"], torch.tensor([2.0]))
    assert torch.equal(output.loaded[0][0]["weight"], torch.tensor([3.0]))
    assert torch.equal(layer.loaded[0][0]["weight"], torch.tensor([4.0]))
    assert torch.equal(base_prior._pos_emb, prior_pos_emb)


def test_prior_peft_restore_configured_frozen_prior_requires_patchable_layers(tmp_path):
    checkpoint = tmp_path / "prior.ckpt"
    torch.save(
        {"model": {"prior._transformer.layers.0.weight": torch.tensor([4.0])}},
        checkpoint,
    )
    peft = SimpleNamespace(
        base_prior=SimpleNamespace(
            _transformer=SimpleNamespace(layers=[SimpleNamespace()]),
        ),
    )
    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.device = torch.device("cpu")
    agent.config = SimpleNamespace(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=str(checkpoint),
                module_path="",
            )
        }
    )
    agent.model = SimpleNamespace(
        _actor=SimpleNamespace(prior_with_peft=peft)
    )

    with pytest.raises(RuntimeError, match="transformer_layer"):
        DiscretePriorPEFTRLFTAgent._restore_configured_frozen_prior(agent)


def test_prior_peft_load_adapter_checkpoint_loads_adapter_only_state(tmp_path):
    class _AdapterActor:
        def __init__(self):
            self.loaded = None

        def load_adapter_state_dict(self, state_dict, strict=True):
            adapter_state = {
                key[len("_actor.") :] if key.startswith("_actor.") else key: value
                for key, value in state_dict.items()
                if is_adapter_state_key(key)
            }
            self.loaded = (adapter_state, strict)
            return {"loaded": sorted(adapter_state)}

    actor = _AdapterActor()
    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.model = SimpleNamespace(_actor=actor)
    agent.device = torch.device("cpu")

    checkpoint_path = tmp_path / "adapter.ckpt"
    torch.save(
        {
            "model": {
                (
                    "_actor.prior_with_peft.base_prior._transformer.layers.0."
                    "lora.A"
                ): torch.tensor([1.0]),
                (
                    "_actor.prior_with_peft.base_prior._transformer.layers.0."
                    "transformer_layer.weight"
                ): torch.tensor([2.0]),
                "_critic.weight": torch.tensor([3.0]),
            }
        },
        checkpoint_path,
    )

    result = DiscretePriorPEFTRLFTAgent.load_adapter_checkpoint(agent, checkpoint_path, strict=False)

    assert result == {
        "loaded": [
            "prior_with_peft.base_prior._transformer.layers.0.lora.A",
        ]
    }
    loaded_state, loaded_strict = actor.loaded
    assert loaded_strict is False
    assert torch.equal(
        loaded_state["prior_with_peft.base_prior._transformer.layers.0.lora.A"],
        torch.tensor([1.0]),
    )
    assert len(loaded_state) == 1


def test_prior_peft_load_adapter_checkpoint_rejects_without_adapter_weights(tmp_path):
    class _AdapterActor:
        pass

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.model = SimpleNamespace(_actor=_AdapterActor())
    agent.device = torch.device("cpu")

    checkpoint_path = tmp_path / "no_adapter.ckpt"
    torch.save(
        {
            "model": {
                "_actor.prior_with_peft.base_prior.weight": torch.tensor([1.0]),
                "_critic.weight": torch.tensor([2.0]),
            }
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="No PEFT adapter weights"):
        DiscretePriorPEFTRLFTAgent.load_adapter_checkpoint(agent, checkpoint_path)


def test_prior_peft_load_model_state_dict_accepts_adapter_only_checkpoint():
    class _AdapterActor:
        def __init__(self):
            self.loaded = None

        def load_adapter_state_dict(self, state_dict, strict=True):
            self.loaded = (dict(state_dict), strict)
            return {"missing_keys": [], "unexpected_keys": []}

    class _Model:
        def __init__(self):
            self._actor = _AdapterActor()

        def load_state_dict(self, state_dict, strict=True):
            raise AssertionError("adapter-only checkpoints should not load full model")

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.model = _Model()
    state = {
        "actor_peft_model.0.weight": torch.tensor([1.0]),
        "prior_with_peft.base_prior._transformer.layers.0.lora.A": torch.tensor([2.0]),
        "prior_with_peft.film_input_norm.running_obs_norm.mean": torch.tensor([3.0]),
    }

    DiscretePriorPEFTRLFTAgent._load_model_state_dict(agent, state)

    loaded_state, loaded_strict = agent.model._actor.loaded
    assert loaded_strict is True
    assert loaded_state == state


def test_prior_peft_get_inference_state_dict_emits_adapter_only_model():
    class _InferenceActor:
        def adapter_state_dict(self):
            return {
                "actor_peft_model.0.weight": torch.tensor([1.0, 2.0]),
                "prior_with_peft.base_prior._transformer.layers.0.lora.A": torch.tensor([3.0]),
                "prior_with_peft.base_prior._transformer.layers.0.m": torch.tensor([4.0]),
            }

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.model = SimpleNamespace(_actor=_InferenceActor())
    agent.current_epoch = 17
    agent.step_count = 42_000
    agent.fit_start_time = 1700000000.0
    agent.best_evaluated_score = 0.91

    state = DiscretePriorPEFTRLFTAgent.get_inference_state_dict(
        agent,
        {},
        model_state_dict={
            "_critic.weight": torch.tensor([99.0]),
            "prior_with_peft.base_prior._transformer.layers.0.transformer_layer.weight": torch.tensor([100.0]),
        },
    )

    assert state["epoch"] == 17
    assert state["step_count"] == 42_000
    assert state["run_start_time"] == 1700000000.0
    assert state["best_evaluated_score"] == 0.91
    assert set(state["model"].keys()) == {
        "actor_peft_model.0.weight",
        "prior_with_peft.base_prior._transformer.layers.0.lora.A",
        "prior_with_peft.base_prior._transformer.layers.0.m",
    }
    # No critic, no frozen base prior weights.
    assert not any(key.startswith("_critic") for key in state["model"])
    assert not any(
        key.startswith("prior_with_peft.base_prior._transformer.layers.0.transformer_layer")
        for key in state["model"]
    )
    # Tensors are detached cpu clones so the saved file does not pin GPU memory
    # or share storage with the live model.
    for value in state["model"].values():
        assert value.device.type == "cpu"
        assert value.requires_grad is False
