# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ASE agent helper logic that can run without a simulator."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

import protomotions.agents.ase.agent as ase_agent_module
from protomotions.agents.amp.component import AMPTrainingComponent
from protomotions.agents.amp.agent import AMP
from protomotions.agents.ase.agent import ASE


def _new_ase_agent():
    agent = object.__new__(ASE)
    component = object.__new__(AMPTrainingComponent)
    component.agent = agent
    agent.amp_component = component
    return agent


class _DoneStepTracker:
    def __init__(self, done_ids):
        self.done_ids = done_ids
        self.advance_calls = 0
        self.reset_ids = None

    def advance(self):
        self.advance_calls += 1

    def done_indices(self):
        return self.done_ids

    def reset_steps(self, env_ids):
        self.reset_ids = env_ids.clone()


class _MIEncoderModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(out_keys=["disc_logits", "mi_enc_output"])

    def forward(self, tensordict):
        tensordict["disc_logits"] = tensordict["disc_obs"][:, :1]
        tensordict["mi_enc_output"] = torch.nn.functional.normalize(
            tensordict["disc_obs"],
            dim=-1,
        )
        return tensordict

    def compute_disc_reward(self, disc_logits):
        return disc_logits + 1.0

    def compute_mi_reward(self, tensordict, mi_hypersphere_reward_shift):
        reward = (tensordict["mi_enc_output"] * tensordict["latents"]).sum(
            dim=-1,
            keepdim=True,
        )
        if mi_hypersphere_reward_shift:
            reward = reward + 1.0
        return reward

    def calc_von_mises_fisher_enc_error(self, pred, latents):
        return (pred - latents).pow(2).sum(dim=-1)

    def enc_weights(self):
        return [self.weight]


class _ModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, tensordict):
        return self.module(tensordict)


class _ConstantCritic(nn.Module):
    def __init__(self, out_key, value):
        super().__init__()
        self.config = SimpleNamespace(out_keys=[out_key])
        self.value = value

    def forward(self, tensordict):
        tensordict[self.config.out_keys[0]] = torch.full(
            (tensordict.batch_size[0], 1),
            self.value,
        )
        return tensordict


class _LatentActor(nn.Module):
    def forward(self, tensordict):
        tensordict["mean_action"] = tensordict["latents"][..., :2]
        return tensordict


class _ExperienceBufferRecorder:
    def __init__(self):
        self.registered = []
        self.data = {}
        self.value = torch.zeros(2, 3, 1)

    def register_key(self, key, shape=(), dtype=None):
        self.registered.append((key, shape, dtype))

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()

    def batch_update_data(self, key, value):
        setattr(self, key, value.clone())
        self.data[(key, "batch")] = value.clone()


class _OptimizerRecorder:
    def __init__(self, state=None):
        self.loaded = None
        self.zero_grad_calls = []
        self.step_calls = 0
        self._state = state or {"optimizer": "state"}

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self.loaded = state

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls.append(set_to_none)

    def step(self):
        self.step_calls += 1


def test_ase_init_builds_mi_reward_normalizer_when_reward_norm_enabled(monkeypatch):
    parent_calls = []

    def fake_parent_init(self, fabric, env, config, root_dir=None):
        parent_calls.append((fabric, env, config, root_dir))
        self.fabric = fabric
        self.env = env
        self.config = config
        self.gamma = 0.95
        self.device = torch.device("cpu")

    class _RewardRunningMeanStd:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(AMP, "__init__", fake_parent_init)
    monkeypatch.setattr(
        ase_agent_module,
        "RewardRunningMeanStd",
        _RewardRunningMeanStd,
    )
    config = _ase_config()
    config.normalize_rewards = True

    agent = ASE("fabric", "env", config, root_dir="root")

    assert parent_calls == [("fabric", "env", config, "root")]
    assert agent.running_mi_enc_norm.kwargs == {
        "shape": (1,),
        "fabric": "fabric",
        "gamma": 0.95,
        "device": torch.device("cpu"),
        "clamp_value": config.normalized_reward_clamp_value,
    }


def test_ase_init_skips_mi_reward_normalizer_when_reward_norm_disabled(monkeypatch):
    def fake_parent_init(self, fabric, env, config, root_dir=None):
        self.fabric = fabric
        self.env = env
        self.config = config
        self.gamma = 0.95
        self.device = torch.device("cpu")

    monkeypatch.setattr(AMP, "__init__", fake_parent_init)
    config = _ase_config()
    config.normalize_rewards = False

    agent = ASE("fabric", "env", config)

    assert agent.running_mi_enc_norm is None


class _RewardNorm:
    def __init__(self, offset):
        self.offset = offset
        self.recorded = []
        self.var = torch.tensor([1.0])

    def normalize(self, values, un_norm: bool = False):
        return values - self.offset if un_norm else values + self.offset

    def record_reward(self, rewards, terminated):
        self.recorded.append((rewards.clone(), terminated.clone()))


def _ase_config(**overrides):
    ase_params = SimpleNamespace(
        latent_dim=3,
        latent_steps_min=1,
        latent_steps_max=2,
        mi_hypersphere_reward_shift=True,
        mi_reward_w=0.25,
        uniformity_kernel_scale=2.0,
        diversity_bonus=1.0,
        diversity_tar=0.5,
        mi_enc_grad_penalty=0.0,
        mi_enc_weight_decay=0.0,
        latent_uniformity_weight=0.1,
        conditional_discriminator=False,
    )
    for key, value in overrides.items():
        setattr(ase_params, key, value)
    return SimpleNamespace(
        normalize_rewards=False,
        normalized_reward_clamp_value=10.0,
        clip_critic_loss=True,
        e_clip=0.2,
        ase_parameters=ase_params,
        amp_parameters=SimpleNamespace(discriminator_batch_size=2),
    )


def test_ase_setup_initializes_latent_state_before_parent_setup(monkeypatch):
    setup_calls = []

    class _StepTracker:
        def __init__(self, num_envs, min_steps, max_steps, device):
            setup_calls.append((num_envs, min_steps, max_steps, device))
            self.num_envs = num_envs

    def fake_parent_setup(self):
        setup_calls.append(("parent", self.latents.clone(), self.latent_reset_steps))

    monkeypatch.setattr(ase_agent_module, "StepTracker", _StepTracker)
    monkeypatch.setattr(AMP, "setup", fake_parent_setup)
    agent = _new_ase_agent()
    agent.num_envs = 2
    agent.device = torch.device("cpu")
    agent.config = _ase_config(latent_dim=5, latent_steps_min=3, latent_steps_max=7)

    ASE.setup(agent)

    assert setup_calls[0] == (2, 3, 7, torch.device("cpu"))
    assert setup_calls[1][0] == "parent"
    assert torch.equal(setup_calls[1][1], torch.zeros(2, 5))
    assert isinstance(agent.latent_reset_steps, _StepTracker)


def test_ase_create_optimizers_builds_global_mi_critic_with_fabric(monkeypatch):
    calls = []
    optimizer = _OptimizerRecorder()
    mi_critic = nn.Linear(2, 1)
    model = SimpleNamespace(_mi_critic=mi_critic)

    monkeypatch.setattr(AMP, "create_optimizers", lambda self, model: calls.append("parent"))
    monkeypatch.setattr(
        ase_agent_module,
        "instantiate",
        lambda config, params: calls.append((config, list(params))) or optimizer,
    )
    agent = _new_ase_agent()
    agent.config = SimpleNamespace(
        model=SimpleNamespace(mi_critic_optimizer=SimpleNamespace(name="mi"))
    )

    def fake_setup_model_optimizer(module, opt):
        calls.append((module, opt))
        return "wrapped-mi-critic", "wrapped-mi-optimizer"

    agent._setup_model_optimizer = fake_setup_model_optimizer

    ASE.create_optimizers(agent, model)

    assert calls[0] == "parent"
    assert calls[1][0] is agent.config.model.mi_critic_optimizer
    assert calls[1][1] == list(mi_critic.parameters())
    assert calls[2] == (mi_critic, optimizer)
    assert agent.mi_critic == "wrapped-mi-critic"
    assert agent.mi_critic_optimizer == "wrapped-mi-optimizer"


def test_ase_load_training_state_restores_full_mi_state(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        AMP,
        "_load_training_state",
        lambda self, state_dict: parent_calls.append(state_dict),
    )
    state = {
        "mi_critic_optimizer": {"lr": 1e-4},
        "running_mi_enc_norm": {"mean": torch.tensor([1.0])},
    }
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.normalize_rewards = True
    agent.mi_critic_optimizer = _OptimizerRecorder()
    agent.running_mi_enc_norm = _RewardNorm(offset=0.0)
    agent.running_mi_enc_norm.load_state_dict = lambda norm_state: setattr(
        agent.running_mi_enc_norm, "loaded", norm_state
    )

    ASE._load_training_state(agent, state)

    assert parent_calls == [state]
    assert agent.mi_critic_optimizer.loaded == {"lr": 1e-4}
    assert agent.running_mi_enc_norm.loaded is state["running_mi_enc_norm"]


def test_ase_get_state_dict_adds_mi_training_state(monkeypatch):
    monkeypatch.setattr(
        AMP,
        "get_state_dict",
        lambda self, state_dict: state_dict | {"parent": True},
    )
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.normalize_rewards = True
    agent.mi_critic_optimizer = _OptimizerRecorder({"mi": "optimizer"})
    agent.running_mi_enc_norm = SimpleNamespace(
        state_dict=lambda: {"mi_norm": torch.tensor([2.0])}
    )

    state = ASE.get_state_dict(agent, {"existing": True})

    assert state["existing"] is True
    assert state["parent"] is True
    assert state["mi_critic_optimizer"] == {"mi": "optimizer"}
    assert torch.equal(state["running_mi_enc_norm"]["mi_norm"], torch.tensor([2.0]))


def test_ase_latent_reset_sampling_and_observation_injection_are_stateful():
    torch.manual_seed(0)
    agent = _new_ase_agent()
    agent.num_envs = 3
    agent.device = torch.device("cpu")
    agent.config = _ase_config(latent_dim=4)
    agent.latents = torch.zeros(3, 4)

    ASE.reset_latents(agent)

    assert agent.latents.shape == (3, 4)
    assert torch.allclose(agent.latents.norm(dim=-1), torch.ones(3), atol=1e-6)

    previous = agent.latents.clone()
    done_ids = torch.tensor([0, 2])
    agent.latent_reset_steps = _DoneStepTracker(done_ids)
    ASE.update_latents(agent)
    obs = ASE.add_agent_info_to_obs(agent, {"obs": torch.ones(3, 1)})

    assert agent.latent_reset_steps.advance_calls == 1
    assert torch.equal(agent.latent_reset_steps.reset_ids, done_ids)
    assert not torch.equal(agent.latents[done_ids], previous[done_ids])
    assert torch.equal(obs["latents"], agent.latents)
    assert obs["latents"].data_ptr() != agent.latents.data_ptr()


def test_ase_update_latents_noops_when_tracker_has_no_done_indices():
    agent = _new_ase_agent()
    agent.latent_reset_steps = _DoneStepTracker(torch.empty(0, dtype=torch.long))
    agent.reset_latents = lambda env_ids=None: (_ for _ in ()).throw(
        AssertionError("reset should not run")
    )

    ASE.update_latents(agent)

    assert agent.latent_reset_steps.advance_calls == 1
    assert agent.latent_reset_steps.reset_ids is None


def test_ase_mi_encoder_forward_converts_observation_dict_to_tensordict():
    agent = _new_ase_agent()
    agent.amp_component.discriminator = _ModuleWrapper(_MIEncoderModule())
    obs = {"disc_obs": torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])}

    encoded = ASE.mi_enc_forward(agent, obs)

    assert torch.allclose(
        encoded,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )


def test_ase_registers_mi_experience_keys_with_normalized_variants(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        AMP,
        "register_algorithm_experience_buffer_keys",
        lambda self: parent_calls.append("parent"),
    )
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.normalize_rewards = True
    agent.experience_buffer = _ExperienceBufferRecorder()

    ASE.register_algorithm_experience_buffer_keys(agent)

    registered = {key for key, _, _ in agent.experience_buffer.registered}
    assert parent_calls == ["parent"]
    assert {
        "unnormalized_mi_rewards",
        "mi_rewards",
        "next_mi_value",
        "mi_returns",
        "unnormalized_mi_value",
        "unnormalized_next_mi_value",
    }.issubset(registered)


def test_ase_pre_collect_step_delegates_then_updates_latents(monkeypatch):
    parent_steps = []
    monkeypatch.setattr(
        AMP,
        "pre_collect_step",
        lambda self, step: parent_steps.append(step),
    )
    agent = _new_ase_agent()
    agent.update_latents = lambda: parent_steps.append("latents")

    ASE.pre_collect_step(agent, 4)

    assert parent_steps == [4, "latents"]


def test_ase_record_rollout_step_records_mi_reward_value_and_running_norm(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        AMP,
        "record_rollout_step",
        lambda self, *args, **kwargs: parent_calls.append((args, kwargs)),
    )
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.normalize_rewards = True
    agent.amp_component.discriminator = _ModuleWrapper(_MIEncoderModule())
    agent.mi_critic = _ModuleWrapper(_ConstantCritic("mi_value", value=5.0))
    agent.running_mi_enc_norm = _RewardNorm(offset=10.0)
    agent.experience_buffer = _ExperienceBufferRecorder()
    next_obs_td = TensorDict(
        {
            "disc_obs": torch.eye(3)[:2],
            "latents": torch.eye(3)[:2],
        },
        batch_size=2,
    )
    terminated = torch.tensor([False, True])

    ASE.record_rollout_step(
        agent,
        next_obs_td=next_obs_td,
        actions=torch.zeros(2, 1),
        rewards=torch.zeros(2),
        dones=torch.zeros(2, dtype=torch.bool),
        terminated=terminated,
        done_indices=torch.tensor([1]),
        extras={},
        step=3,
    )

    assert len(parent_calls) == 1
    assert torch.equal(
        agent.experience_buffer.data[("next_mi_value", 3)],
        torch.tensor([[5.0], [0.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("mi_rewards", 3)],
        torch.tensor([2.0, 2.0]),
    )
    assert agent.running_mi_enc_norm.recorded[0][1].equal(terminated)


def test_ase_record_rollout_step_composes_real_amp_and_mi_paths():
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.model = SimpleNamespace(critic=SimpleNamespace())
    agent.model = SimpleNamespace(_critic=_ConstantCritic("value", value=3.0))
    agent.config.amp_parameters.discriminator_reward_threshold = 0.5
    agent.config.amp_parameters.discriminator_max_cumulative_bad_transitions = 2
    agent.current_rewards = torch.zeros(2)
    agent.current_lengths = torch.zeros(2)
    agent.episode_reward_meter = SimpleNamespace(add=lambda data: None)
    agent.episode_length_meter = SimpleNamespace(add=lambda data: None)
    agent.episode_env_tensors = SimpleNamespace(add=lambda data: None)
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent.amp_component.num_cumulative_bad_transitions = torch.zeros(2, dtype=torch.int32)
    agent.amp_component.use_disc_critic = True
    agent.amp_component.discriminator = _ModuleWrapper(_MIEncoderModule())
    agent.amp_component.disc_critic = _ModuleWrapper(_ConstantCritic("disc_value", value=7.0))
    agent.mi_critic = _ModuleWrapper(_ConstantCritic("mi_value", value=5.0))
    next_obs_td = TensorDict(
        {
            "disc_obs": torch.eye(3)[:2],
            "latents": torch.eye(3)[:2],
        },
        batch_size=2,
    )
    dones = torch.tensor([False, True])
    terminated = torch.tensor([False, True])

    ASE.record_rollout_step(
        agent,
        next_obs_td=next_obs_td,
        actions=torch.zeros(2, 1),
        rewards=torch.tensor([0.5, 1.5]),
        dones=dones,
        terminated=terminated,
        done_indices=torch.tensor([1]),
        extras={},
        step=3,
    )

    assert torch.equal(
        agent.experience_buffer.data[("rewards", 3)],
        torch.tensor([0.5, 1.5]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("amp_rewards", 3)],
        torch.tensor([2.0, 1.0]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("next_disc_value", 3)],
        torch.tensor([[7.0], [0.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("mi_rewards", 3)],
        torch.tensor([2.0, 2.0]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("next_mi_value", 3)],
        torch.tensor([[5.0], [0.0]]),
    )


def test_ase_normalize_rewards_returns_when_disabled_and_updates_mi_buffers(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        AMP,
        "normalize_rewards_in_buffer",
        lambda self: parent_calls.append("parent"),
    )
    disabled = _new_ase_agent()
    disabled.config = _ase_config()
    disabled.config.normalize_rewards = False

    ASE.normalize_rewards_in_buffer(disabled)

    assert parent_calls == ["parent"]

    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.normalize_rewards = True
    agent.running_mi_enc_norm = _RewardNorm(offset=2.0)
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent.experience_buffer.mi_rewards = torch.tensor([[1.0, 2.0]])
    agent.experience_buffer.mi_value = torch.tensor([[[3.0], [4.0]]])
    agent.experience_buffer.next_mi_value = torch.tensor([[[5.0], [6.0]]])

    ASE.normalize_rewards_in_buffer(agent)

    assert parent_calls == ["parent", "parent"]
    assert torch.equal(
        agent.experience_buffer.data[("unnormalized_mi_rewards", "batch")],
        torch.tensor([[1.0, 2.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("mi_rewards", "batch")],
        torch.tensor([[3.0, 4.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("unnormalized_mi_value", "batch")],
        torch.tensor([[[1.0], [2.0]]]),
    )


def test_ase_compute_advantages_combines_task_and_mi_advantages(monkeypatch):
    monkeypatch.setattr(
        AMP,
        "compute_advantages",
        lambda self: {
            "returns": torch.tensor([10.0, 20.0]),
            "advantages": torch.tensor([1.0, 2.0]),
        },
    )
    agent = _new_ase_agent()
    agent.config = _ase_config(mi_reward_w=0.5)
    agent.config.normalize_rewards = False
    agent.gamma = 0.0
    agent.tau = 1.0
    agent.experience_buffer = SimpleNamespace(
        dones=torch.zeros(2),
        mi_rewards=torch.tensor([4.0, 6.0]),
        mi_value=torch.tensor([[1.0], [2.0]]),
        next_mi_value=torch.zeros(2, 1),
    )
    updates = {}
    agent.experience_buffer.batch_update_data = (
        lambda key, value: updates.setdefault(key, value.clone())
    )

    advantages = ASE.compute_advantages(agent)

    assert torch.equal(updates["mi_returns"], torch.tensor([4.0, 6.0]))
    assert torch.equal(advantages["returns"], torch.tensor([10.0, 20.0]))
    assert torch.equal(advantages["advantages"], torch.tensor([2.5, 4.0]))


def test_ase_compute_advantages_uses_normalized_mi_inputs_and_returns(monkeypatch):
    monkeypatch.setattr(
        AMP,
        "compute_advantages",
        lambda self: {"returns": torch.tensor([0.0]), "advantages": torch.tensor([1.0])},
    )
    agent = _new_ase_agent()
    agent.config = _ase_config(mi_reward_w=0.5)
    agent.config.normalize_rewards = True
    agent.gamma = 0.0
    agent.tau = 1.0
    agent.running_mi_enc_norm = _RewardNorm(offset=10.0)
    agent.experience_buffer = SimpleNamespace(
        dones=torch.zeros(1),
        unnormalized_mi_rewards=torch.tensor([4.0]),
        unnormalized_mi_value=torch.tensor([[1.0]]),
        unnormalized_next_mi_value=torch.zeros(1, 1),
    )
    updates = {}
    agent.experience_buffer.batch_update_data = (
        lambda key, value: updates.setdefault(key, value.clone())
    )

    advantages = ASE.compute_advantages(agent)

    assert torch.equal(updates["mi_returns"], torch.tensor([14.0]))
    assert torch.equal(advantages["advantages"], torch.tensor([2.5]))


def test_ase_produce_negative_expert_obs_preserves_motion_and_resamples_latents():
    torch.manual_seed(0)
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.model = SimpleNamespace(
        _discriminator=SimpleNamespace(in_keys=["disc_obs", "latents"])
    )
    batch = {
        "expert_disc_obs": torch.arange(6, dtype=torch.float).view(3, 2),
        "agent_latents": torch.ones(3, 3),
    }

    negative = ASE.produce_negative_expert_obs(agent, batch)

    assert torch.equal(negative["disc_obs"], batch["expert_disc_obs"][:2])
    assert negative["latents"].shape == (2, 3)
    assert torch.allclose(negative["latents"].norm(dim=-1), torch.ones(2))


def test_ase_mi_critic_step_uses_clipped_loss_when_enabled():
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.mi_critic = _ModuleWrapper(_ConstantCritic("mi_value", value=1.0))
    batch = {
        "action": torch.zeros(3, 1),
        "mi_returns": torch.tensor([0.0, 1.0, 2.0]),
        "mi_value": torch.zeros(3, 1),
    }

    loss, log_dict = ASE.mi_critic_step(agent, batch)

    assert torch.isfinite(loss)
    assert torch.equal(log_dict["losses/mi_critic_loss"], loss.detach())


def test_ase_mi_critic_step_supports_unclipped_loss():
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.clip_critic_loss = False
    agent.mi_critic = _ModuleWrapper(_ConstantCritic("mi_value", value=1.0))
    batch = {
        "action": torch.zeros(2, 1),
        "mi_returns": torch.ones(2),
        "mi_value": torch.zeros(2, 1),
    }

    loss, log_dict = ASE.mi_critic_step(agent, batch)

    assert loss.item() == 0.0
    assert torch.equal(log_dict["losses/mi_critic_loss"], loss.detach())


def test_ase_perform_optimization_step_skips_or_updates_mi_critic(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        AMP,
        "perform_optimization_step",
        lambda self, batch_dict, batch_idx: parent_calls.append(batch_idx)
        or {"parent": torch.tensor(float(batch_idx))},
    )
    step_calls = []
    agent = _new_ase_agent()
    agent._skip_actor_for_epoch = True
    agent.mi_critic_step = lambda batch_dict: (_ for _ in ()).throw(
        AssertionError("MI critic should be skipped")
    )

    skipped = ASE.perform_optimization_step(agent, {"x": torch.tensor([1.0])}, 3)

    assert skipped["parent"].item() == 3.0
    assert parent_calls == [3]

    optimizer = _OptimizerRecorder()
    agent._skip_actor_for_epoch = False
    agent.config = _ase_config()
    agent.mi_critic = nn.Linear(1, 1)
    agent.mi_critic_optimizer = optimizer
    agent.mi_critic_step = lambda batch_dict: (
        torch.tensor(2.0, requires_grad=True),
        {"losses/mi_critic_loss": torch.tensor(2.0)},
    )
    agent._step_optimizer = (
        lambda loss, model, optimizer, model_name: step_calls.append(
            {
                "loss": loss,
                "model": model,
                "optimizer": optimizer,
                "model_name": model_name,
            }
        )
        or optimizer.zero_grad(set_to_none=True)
        or optimizer.step()
        or {"clip/mi_critic": torch.tensor(0.0)}
    )

    updated = ASE.perform_optimization_step(agent, {"x": torch.tensor([1.0])}, 4)

    assert parent_calls == [3, 4]
    assert updated["losses/mi_critic_loss"].item() == 2.0
    assert updated["clip/mi_critic"].item() == 0.0
    assert optimizer.zero_grad_calls == [True]
    assert optimizer.step_calls == 1
    assert step_calls[0]["model_name"] == "mi_critic"


def test_ase_get_expert_disc_obs_adds_predicted_latents(monkeypatch):
    monkeypatch.setattr(
        AMP,
        "get_expert_disc_obs",
        lambda self, num_samples: {"disc_obs": torch.eye(3)[:num_samples]},
    )
    agent = _new_ase_agent()
    agent.amp_component.discriminator = _ModuleWrapper(_MIEncoderModule())

    expert_obs = ASE.get_expert_disc_obs(agent, num_samples=2)

    assert torch.equal(expert_obs["disc_obs"], torch.eye(3)[:2])
    assert torch.equal(expert_obs["latents"], torch.eye(3)[:2])


def test_ase_discriminator_step_adds_mi_encoder_loss_and_weight_decay(monkeypatch):
    monkeypatch.setattr(
        AMPTrainingComponent,
        "discriminator_step",
        lambda self, batch_dict, *, negative_expert_obs=None: (
            torch.tensor(1.0, requires_grad=True),
            {"base/discriminator": torch.tensor(2.0)},
        ),
    )
    agent = _new_ase_agent()
    agent.device = torch.device("cpu")
    agent.config = _ase_config(
        mi_enc_weight_decay=0.1,
        latent_uniformity_weight=0.1,
    )
    agent.amp_component.discriminator = _ModuleWrapper(_MIEncoderModule())
    agent.model = SimpleNamespace(
        _discriminator=SimpleNamespace(in_keys=["disc_obs", "latents"])
    )
    batch = {
        "agent_disc_obs": torch.eye(3)[:2],
        "expert_disc_obs": torch.flip(torch.eye(3)[:2], dims=[0]),
        "agent_latents": torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ),
    }

    loss, log_dict = ASE.discriminator_step(agent, batch)

    assert torch.isfinite(loss)
    assert "base/discriminator" in log_dict
    assert "encoder/loss" in log_dict
    assert log_dict["encoder/l2_loss"].item() == pytest.approx(0.1)
    assert log_dict["encoder/grad_penalty"].item() == 0.0


def test_ase_discriminator_step_supplies_conditional_negative_expert_obs(monkeypatch):
    captured = {}

    def fake_component_discriminator_step(
        self, batch_dict, *, negative_expert_obs=None
    ):
        captured["negative_expert_obs"] = negative_expert_obs
        return torch.tensor(1.0, requires_grad=True), {}

    monkeypatch.setattr(
        AMPTrainingComponent,
        "discriminator_step",
        fake_component_discriminator_step,
    )
    agent = _new_ase_agent()
    agent.device = torch.device("cpu")
    agent.config = _ase_config(conditional_discriminator=True)
    agent.amp_component.discriminator = _ModuleWrapper(_MIEncoderModule())
    agent.model = SimpleNamespace(
        _discriminator=SimpleNamespace(in_keys=["disc_obs", "latents"])
    )
    batch = {
        "agent_disc_obs": torch.eye(3)[:2],
        "expert_disc_obs": torch.flip(torch.eye(3)[:2], dims=[0]),
        "agent_latents": torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ),
    }

    ASE.discriminator_step(agent, batch)

    negative_expert_obs = captured["negative_expert_obs"]
    assert torch.equal(
        negative_expert_obs["disc_obs"],
        batch["expert_disc_obs"],
    )
    assert (
        negative_expert_obs["latents"].shape
        == batch["agent_latents"].shape
    )


def test_ase_discriminator_step_supports_mi_encoder_gradient_penalty(monkeypatch):
    monkeypatch.setattr(
        AMPTrainingComponent,
        "discriminator_step",
        lambda self, batch_dict, *, negative_expert_obs=None: (
            torch.tensor(0.25, requires_grad=True),
            {},
        ),
    )
    agent = _new_ase_agent()
    agent.device = torch.device("cpu")
    agent.config = _ase_config(mi_enc_grad_penalty=0.5)
    agent.amp_component.discriminator = _ModuleWrapper(_MIEncoderModule())
    agent.model = SimpleNamespace(
        _discriminator=SimpleNamespace(in_keys=["disc_obs", "latents"])
    )
    batch = {
        "agent_disc_obs": torch.eye(3)[:2],
        "expert_disc_obs": torch.flip(torch.eye(3)[:2], dims=[0]),
        "agent_latents": torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ),
    }

    loss, log_dict = ASE.discriminator_step(agent, batch)

    assert torch.isfinite(loss)
    assert log_dict["encoder/grad_penalty"].item() > 0.0


def test_ase_discriminator_step_requires_latents(monkeypatch):
    monkeypatch.setattr(
        AMPTrainingComponent,
        "discriminator_step",
        lambda self, batch_dict, *, negative_expert_obs=None: (
            torch.tensor(0.0),
            {},
        ),
    )
    agent = _new_ase_agent()
    agent.config = _ase_config()

    with (
        torch.no_grad(),
        torch.inference_mode(False),
        pytest.raises(KeyError, match="latents"),
    ):
        ASE.discriminator_step(agent, {"agent_disc_obs": torch.zeros(2, 3)})


def test_ase_uniformity_loss_is_finite_and_more_negative_for_separated_points():
    agent = _new_ase_agent()
    agent.config = _ase_config(uniformity_kernel_scale=2.0)
    repeated = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    separated = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    repeated_loss = ASE.compute_uniformity_loss(agent, repeated)
    separated_loss = ASE.compute_uniformity_loss(agent, separated)

    assert torch.isfinite(repeated_loss)
    assert torch.isfinite(separated_loss)
    assert separated_loss < repeated_loss


def test_ase_diversity_loss_compares_action_changes_against_latent_distance():
    agent = _new_ase_agent()
    agent.config = _ase_config(latent_dim=3, diversity_tar=0.25)
    agent.actor = _LatentActor()
    agent.sample_latents = lambda n: torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]][:n]
    )
    batch = TensorDict(
        {
            "latents": torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            "mean_action": torch.tensor([[0.5, 0.0], [-0.5, 0.0]]),
        },
        batch_size=2,
    )

    loss = ASE.diversity_loss(agent, batch)

    assert torch.isfinite(loss)
    assert torch.equal(
        batch["latents"],
        torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
    )


def test_ase_calculate_extra_actor_loss_respects_diversity_bonus(monkeypatch):
    monkeypatch.setattr(
        AMP,
        "calculate_extra_actor_loss",
        lambda self, batch_td: (torch.tensor(0.5), {"parent": torch.tensor(1.0)}),
    )
    agent = _new_ase_agent()
    agent.config = _ase_config(diversity_bonus=0.0)
    batch = TensorDict(
        {
            "latents": torch.tensor([[1.0, 0.0, 0.0]]),
            "mean_action": torch.tensor([[0.5, 0.0]]),
        },
        batch_size=1,
    )

    loss, log_dict = ASE.calculate_extra_actor_loss(agent, batch)
    assert loss.item() == 0.5
    assert "actor/diversity_loss" not in log_dict

    agent.config = _ase_config(diversity_bonus=2.0, diversity_tar=0.25)
    agent.actor = _LatentActor()
    agent.sample_latents = lambda n: torch.tensor([[0.0, 1.0, 0.0]][:n])
    loss, log_dict = ASE.calculate_extra_actor_loss(agent, batch)

    assert loss.item() > 0.5
    assert "actor/diversity_loss" in log_dict


def test_ase_post_epoch_logging_adds_mi_rewards_and_delegates(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        AMP,
        "post_epoch_logging",
        lambda self, training_log_dict: parent_calls.append(training_log_dict),
    )
    agent = _new_ase_agent()
    agent.config = _ase_config()
    agent.config.normalize_rewards = True
    agent.experience_buffer = SimpleNamespace(
        mi_rewards=torch.tensor([1.0, 2.0, 3.0]),
        unnormalized_mi_rewards=torch.tensor([4.0, 5.0, 6.0]),
    )
    log_dict = {}

    ASE.post_epoch_logging(agent, log_dict)

    assert log_dict["rewards/mi_enc_rewards"].item() == 2.0
    assert torch.equal(
        log_dict["rewards/unnormalized_mi_enc_rewards"],
        torch.tensor([4.0, 5.0, 6.0]),
    )
    assert parent_calls == [log_dict]
