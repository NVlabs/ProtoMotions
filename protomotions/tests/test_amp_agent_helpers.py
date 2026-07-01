# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AMP agent helper logic that does not require a live simulator."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.amp import component as amp_component_module
from protomotions.agents.amp.agent import AMP
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.utils.metering import TensorAverageMeterDict
from protomotions.agents.utils.replay_buffer import ReplayBuffer
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_max_coords_from_motion_lib,
    compute_historical_max_coords_from_state,
)
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer


def _new_amp_agent(agent_cls=AMP):
    agent = object.__new__(agent_cls)
    component = object.__new__(amp_component_module.AMPTrainingComponent)
    component.agent = agent
    agent.amp_component = component
    return agent


class _BufferRecorder:
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


class _RewardNorm:
    def __init__(self, offset):
        self.offset = offset
        self.var = torch.tensor([1.0])
        self.recorded = []

    def normalize(self, values, un_norm: bool = False):
        return values - self.offset if un_norm else values + self.offset

    def record_reward(self, rewards, terminated):
        self.recorded.append((rewards.clone(), terminated.clone()))

    def state_dict(self):
        return {"offset": torch.tensor(self.offset)}

    def load_state_dict(self, state):
        self.loaded_state = state


class _ResetRecorder:
    def __init__(self):
        self.env_ids = None

    def reset_rollout_context(self, env_ids=None, num_envs: int = None, device=None):
        self.env_ids = env_ids.clone()


class _ReplayRecorder:
    def __init__(self, buffer_size, current_len):
        self.buffer_size = buffer_size
        self.current_len = current_len
        self.stored = None

    def get_buffer_size(self):
        return self.buffer_size

    def __len__(self):
        return self.current_len

    def store(self, data_dict):
        self.stored = {key: value.clone() for key, value in data_dict.items()}


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


class _Router:
    def __init__(self, fn, params):
        self.fn = fn
        self.params = params

    def get_compute_func(self):
        return self.fn

    def get_params(self):
        return self.params


class _MotionManager:
    def sample_n_motion_ids(self, num_samples):
        return torch.arange(num_samples)

    def sample_time(self, motion_ids):
        return motion_ids.float() * 0.5


class _FixedMotionManager:
    def __init__(self, motion_ids, motion_times):
        self.motion_ids = motion_ids
        self.motion_times = motion_times

    def sample_n_motion_ids(self, num_samples):
        assert num_samples == self.motion_ids.shape[0]
        return self.motion_ids

    def sample_time(self, motion_ids):
        assert torch.equal(motion_ids, self.motion_ids)
        return self.motion_times


class _DeterministicMotionLib:
    def __init__(self, num_bodies=3, num_dofs=2, motion_lengths=None):
        self.num_bodies = num_bodies
        self.num_dofs = num_dofs
        self.motion_lengths = (
            motion_lengths
            if motion_lengths is not None
            else torch.full((16,), 100.0)
        )

    def get_motion_state(self, motion_ids, motion_times):
        batch = motion_ids.shape[0]
        ids = motion_ids.float().view(batch, 1, 1)
        times = motion_times.view(batch, 1, 1)
        body_ids = torch.arange(self.num_bodies, dtype=torch.float).view(
            1, self.num_bodies, 1
        )

        rigid_body_pos = torch.cat(
            [
                ids + body_ids * 0.25 + times,
                ids * 0.5 - body_ids * 0.1 + times * 0.2,
                0.5 + body_ids * 0.15 + times * 0.3,
            ],
            dim=-1,
        )
        rigid_body_rot = torch.zeros(batch, self.num_bodies, 4)
        rigid_body_rot[..., 3] = 1.0
        rigid_body_vel = torch.cat(
            [
                times.expand(batch, self.num_bodies, 1),
                (ids + 0.1).expand(batch, self.num_bodies, 1),
                (body_ids + 0.2).expand(batch, self.num_bodies, 1),
            ],
            dim=-1,
        )
        rigid_body_ang_vel = torch.cat(
            [
                (body_ids + 0.3).expand(batch, self.num_bodies, 1),
                (times + 0.4).expand(batch, self.num_bodies, 1),
                (ids + 0.5).expand(batch, self.num_bodies, 1),
            ],
            dim=-1,
        )
        dof_base = torch.arange(self.num_dofs, dtype=torch.float).view(1, self.num_dofs)
        dof_pos = motion_ids.float().unsqueeze(-1) + motion_times.unsqueeze(-1) + dof_base
        dof_vel = motion_times.unsqueeze(-1) * 0.5 + dof_base

        return SimpleNamespace(
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_contacts=None,
        )


def _simulator_historical_max_coords_from_reference_reset(
    motion_lib,
    motion_ids,
    motion_times,
    *,
    dt,
    num_state_history_steps,
    history_steps,
    local_obs,
    root_height_obs,
):
    buffer_size = num_state_history_steps + 1
    time_offsets = -dt * torch.arange(buffer_size)
    expanded_motion_ids = motion_ids.unsqueeze(1).expand(-1, buffer_size)
    expanded_motion_times = motion_times.unsqueeze(1) + time_offsets.unsqueeze(0)
    expanded_motion_times = expanded_motion_times.clamp(min=0.0)
    motion_lengths = getattr(motion_lib, "motion_lengths", None)
    if motion_lengths is not None:
        expanded_motion_times = torch.min(
            expanded_motion_times,
            motion_lengths[motion_ids].unsqueeze(1).expand(-1, buffer_size),
        )
    historical_state = motion_lib.get_motion_state(
        expanded_motion_ids.reshape(-1),
        expanded_motion_times.reshape(-1),
    )
    history = StateHistoryBuffer(
        num_envs=motion_ids.shape[0],
        num_history_steps=num_state_history_steps,
        num_bodies=motion_lib.num_bodies,
        num_dofs=motion_lib.num_dofs,
        action_dim=2,
        num_contact_bodies=motion_lib.num_bodies,
        anchor_body_index=0,
        device=torch.device("cpu"),
    )
    history.reset_from_states(
        env_ids=torch.arange(motion_ids.shape[0]),
        rigid_body_pos=historical_state.rigid_body_pos.view(
            motion_ids.shape[0], buffer_size, motion_lib.num_bodies, 3
        ),
        rigid_body_rot=historical_state.rigid_body_rot.view(
            motion_ids.shape[0], buffer_size, motion_lib.num_bodies, 4
        ),
        rigid_body_vel=historical_state.rigid_body_vel.view(
            motion_ids.shape[0], buffer_size, motion_lib.num_bodies, 3
        ),
        rigid_body_ang_vel=historical_state.rigid_body_ang_vel.view(
            motion_ids.shape[0], buffer_size, motion_lib.num_bodies, 3
        ),
        dof_pos=historical_state.dof_pos.view(
            motion_ids.shape[0], buffer_size, motion_lib.num_dofs
        ),
        dof_vel=historical_state.dof_vel.view(
            motion_ids.shape[0], buffer_size, motion_lib.num_dofs
        ),
        ground_heights=torch.zeros(motion_ids.shape[0], buffer_size),
        body_contacts=torch.zeros(
            motion_ids.shape[0],
            buffer_size,
            motion_lib.num_bodies,
            dtype=torch.bool,
        ),
    )
    return compute_historical_max_coords_from_state(
        historical_rigid_body_pos=history.historical_rigid_body_pos,
        historical_rigid_body_rot=history.historical_rigid_body_rot,
        historical_rigid_body_vel=history.historical_rigid_body_vel,
        historical_rigid_body_ang_vel=history.historical_rigid_body_ang_vel,
        historical_ground_heights=history.historical_ground_heights,
        historical_body_contacts=history.historical_body_contacts,
        local_obs=local_obs,
        root_height_obs=root_height_obs,
        observe_contacts=False,
        w_last=True,
        history_steps=history_steps,
    )


class _LinearDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_keys=["disc_obs"], out_keys=["disc_logits"])
        self._grad_penalty_keys = ["disc_obs"]
        self.linear = nn.Linear(2, 1)

    def forward(self, tensordict):
        tensordict["disc_logits"] = self.linear(tensordict["disc_obs"])
        return tensordict

    def compute_disc_reward(self, disc_logits):
        return torch.sigmoid(disc_logits)

    def all_discriminator_weights(self):
        return [self.linear.weight]

    def logit_weights(self):
        return [self.linear.weight]


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
        batch = tensordict.batch_size[0]
        tensordict[self.config.out_keys[0]] = torch.full((batch, 1), self.value)
        return tensordict


def _amp_config(**amp_overrides):
    amp_params = SimpleNamespace(
        use_disc_critic=True,
        discriminator_replay_keep_prob=1.0,
        discriminator_replay_size=8,
        discriminator_batch_size=2,
        discriminator_reward_threshold=0.5,
        discriminator_max_cumulative_bad_transitions=2,
        discriminator_grad_penalty=0.25,
        discriminator_weight_decay=0.1,
        discriminator_logit_weight_decay=0.2,
        discriminator_optimization_ratio=1,
        conditional_discriminator=False,
        discriminator_reward_w=0.5,
    )
    for key, value in amp_overrides.items():
        setattr(amp_params, key, value)
    return SimpleNamespace(
        normalize_rewards=True,
        normalized_reward_clamp_value=5.0,
        amp_parameters=amp_params,
        clip_critic_loss=True,
        e_clip=0.1,
        batch_size=2,
        task_reward_w=1.0,
        model=SimpleNamespace(
            actor=SimpleNamespace(learnable_std=True),
            critic=True,
        ),
        advantage_normalization=SimpleNamespace(enabled=False, use_ema=False),
        adaptive_lr=SimpleNamespace(enabled=False),
        reference_obs_components={},
    )


def test_amp_init_builds_replay_buffer_transition_counters_and_reward_norm(
    monkeypatch,
):
    constructed_norms = []

    class _RewardRunningMeanStd:
        def __init__(self, **kwargs):
            constructed_norms.append(kwargs)

    def fake_ppo_init(self, fabric, env, config, root_dir=None):
        self.fabric = fabric
        self.env = env
        self.config = config
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.num_steps = 3
        self.num_mini_epochs = 1
        self.gamma = 0.95

    monkeypatch.setattr(PPO, "__init__", fake_ppo_init)
    monkeypatch.setattr(
        amp_component_module,
        "RewardRunningMeanStd",
        _RewardRunningMeanStd,
    )

    config = _amp_config(
        use_disc_critic=True,
        discriminator_replay_size=5,
        discriminator_batch_size=2,
    )
    agent = AMP(fabric=object(), env=object(), config=config)

    assert agent.amp_component.use_disc_critic is True
    assert agent.amp_component.replay_buffer.get_buffer_size() == 5
    assert torch.equal(
        agent.amp_component.num_cumulative_bad_transitions,
        torch.zeros(2, dtype=torch.int32),
    )
    assert constructed_norms[0]["gamma"] == 0.95
    assert constructed_norms[0]["clamp_value"] == 5.0

    config = _amp_config(use_disc_critic=False, discriminator_batch_size=99)
    config.normalize_rewards = False
    agent = AMP(fabric=object(), env=object(), config=config)

    assert agent.amp_component.use_disc_critic is False
    assert agent.amp_component.running_reward_norm is None


def test_amp_init_exposes_reusable_training_component(monkeypatch):
    def fake_ppo_init(self, fabric, env, config, root_dir=None):
        self.fabric = fabric
        self.env = env
        self.config = config
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.num_steps = 3
        self.num_mini_epochs = 1
        self.gamma = 0.95

    monkeypatch.setattr(PPO, "__init__", fake_ppo_init)

    agent = AMP(
        fabric=object(),
        env=object(),
        config=_amp_config(discriminator_replay_size=5, discriminator_batch_size=2),
    )

    assert agent.amp_component.agent is agent
    assert agent.amp_component.replay_buffer.get_buffer_size() == 5
    assert agent.amp_component.running_reward_norm is not None
    assert agent.amp_component.use_disc_critic is True
    assert not hasattr(agent, "amp_replay_buffer")
    assert not hasattr(agent, "running_amp_reward_norm")
    assert not hasattr(agent, "use_disc_critic")


def test_amp_create_optimizers_wraps_discriminator_and_optional_disc_critic(
    monkeypatch,
):
    parent_calls = []
    instantiated = []

    def fake_parent_create_optimizers(self, model):
        parent_calls.append(model)

    def fake_instantiate(config, module, params=None):
        params = list(params)
        instantiated.append((config.name, module, params))
        return _OptimizerRecorder(state={"optimizer": config.name})

    monkeypatch.setattr(PPO, "create_optimizers", fake_parent_create_optimizers)
    monkeypatch.setattr(
        amp_component_module,
        "instantiate_optimizer",
        fake_instantiate,
    )

    agent = _new_amp_agent()
    agent.amp_component.use_disc_critic = True
    agent.config = _amp_config()
    agent.config.model.discriminator_optimizer = SimpleNamespace(name="discriminator")
    agent.config.model.disc_critic_optimizer = SimpleNamespace(name="disc_critic")
    model = SimpleNamespace(
        _discriminator=nn.Linear(2, 1),
        _disc_critic=nn.Linear(2, 1),
    )
    agent._setup_model_optimizer = lambda module, optimizer: (
        _ModuleWrapper(module),
        optimizer,
    )

    AMP.create_optimizers(agent, model)

    assert parent_calls == [model]
    assert [name for name, _, _ in instantiated] == ["discriminator", "disc_critic"]
    assert instantiated[0][1] is model._discriminator
    assert instantiated[1][1] is model._disc_critic
    assert agent.amp_component.discriminator.module is model._discriminator
    assert agent.amp_component.disc_critic.module is model._disc_critic

    no_disc_critic = _new_amp_agent()
    no_disc_critic.amp_component.use_disc_critic = False
    no_disc_critic.config = agent.config
    no_disc_critic._setup_model_optimizer = agent._setup_model_optimizer
    instantiated.clear()

    AMP.create_optimizers(no_disc_critic, model)

    assert [name for name, _, _ in instantiated] == ["discriminator"]
    assert instantiated[0][1] is model._discriminator
    assert not hasattr(no_disc_critic.amp_component, "disc_critic")


def test_amp_registers_ppo_and_discriminator_experience_keys():
    agent = _new_amp_agent()
    agent.config = _amp_config()
    agent.amp_component.use_disc_critic = True
    agent.experience_buffer = _BufferRecorder()

    AMP.register_algorithm_experience_buffer_keys(agent)

    registered = {key for key, _, _ in agent.experience_buffer.registered}
    assert {
        "next_value",
        "returns",
        "advantages",
        "unnormalized_value",
        "unnormalized_next_value",
        "amp_rewards",
        "unnormalized_amp_rewards",
        "next_disc_value",
        "disc_returns",
        "unnormalized_disc_value",
        "unnormalized_next_disc_value",
    }.issubset(registered)


def test_amp_disc_critic_buffer_registration_requires_value_buffer():
    agent = _new_amp_agent()
    agent.config = _amp_config()
    agent.amp_component.use_disc_critic = True
    agent.experience_buffer = _BufferRecorder()
    del agent.experience_buffer.value

    with pytest.raises(RuntimeError, match="experience_buffer.value"):
        agent.amp_component.register_experience_buffer_keys()


def test_amp_model_load_delegates_full_amp_checkpoint(monkeypatch):
    calls = []

    monkeypatch.setattr(
        PPO,
        "_load_model_state_dict",
        lambda self, model_state: calls.append(model_state),
    )
    agent = _new_amp_agent()
    model_state = {"_discriminator.weight": torch.tensor([1.0])}

    AMP._load_model_state_dict(agent, model_state)

    assert agent._warm_start_from_non_amp_checkpoint is False
    assert calls == [model_state]


def test_amp_load_parameters_warm_starts_from_non_amp_checkpoint():
    class _WarmStartModel:
        def load_state_dict(self, model_state, strict=True):
            assert strict is False
            agent.actor.logstd.data.fill_(9.0)
            return ["_discriminator.weight"], ["unused"]

    agent = _new_amp_agent()
    agent.model = _WarmStartModel()
    agent.actor = SimpleNamespace(logstd=nn.Parameter(torch.ones(2)))
    agent.actor_optimizer = _OptimizerRecorder()
    agent.critic_optimizer = _OptimizerRecorder()
    agent.running_reward_norm = _RewardNorm(offset=0.0)
    agent.adv_mean_ema = torch.zeros(1)
    agent.adv_std_ema = torch.zeros(1)
    agent.evaluator = None
    agent.config = _amp_config()
    agent.config.model.actor.learnable_std = False
    agent.config.advantage_normalization = SimpleNamespace(
        enabled=True,
        use_ema=True,
    )
    state = {
        "model": {"_actor.weight": torch.tensor([1.0])},
        "epoch": 3,
        "step_count": 12,
        "run_start_time": 45.0,
        "best_evaluated_score": 6.0,
        "running_reward_norm": {"reward": torch.tensor([1.0])},
        "actor_optimizer": {"actor": 2},
        "critic_optimizer": {"critic": 3},
        "adv_mean_ema": torch.tensor([4.0]),
        "adv_std_ema": torch.tensor([5.0]),
    }

    AMP._load_model_state_dict(agent, state["model"])
    AMP._load_training_state(agent, state)

    assert agent.current_epoch == 3
    assert agent.step_count == 12
    assert agent.fit_start_time == 45.0
    assert agent.best_evaluated_score == 6.0
    assert torch.equal(agent.actor.logstd, torch.ones(2))
    assert torch.equal(
        agent.running_reward_norm.loaded_state["reward"],
        torch.tensor([1.0]),
    )
    assert agent.actor_optimizer.loaded_state == {"actor": 2}
    assert agent.critic_optimizer.loaded_state == {"critic": 3}
    assert torch.equal(agent.adv_mean_ema, torch.tensor([4.0]))
    assert torch.equal(agent.adv_std_ema, torch.tensor([5.0]))


def test_amp_restores_full_amp_training_state(monkeypatch):
    calls = []

    def fake_parent_training_state(self, state_dict):
        calls.append(("parent", state_dict))

    monkeypatch.setattr(PPO, "_load_training_state", fake_parent_training_state)

    agent = _new_amp_agent()
    agent.amp_component.use_disc_critic = True
    agent.config = _amp_config()
    agent.amp_component.discriminator_optimizer = _OptimizerRecorder()
    agent.amp_component.disc_critic_optimizer = _OptimizerRecorder()
    agent.amp_component.running_reward_norm = _RewardNorm(offset=0.0)
    state = {
        "discriminator_optimizer": {"disc": 1},
        "disc_critic_optimizer": {"disc_critic": 2},
        "running_amp_reward_norm": {"amp": torch.tensor([3.0])},
    }

    AMP._load_training_state(agent, state)

    assert calls == [("parent", state)]
    assert agent.amp_component.discriminator_optimizer.loaded_state == {"disc": 1}
    assert agent.amp_component.disc_critic_optimizer.loaded_state == {"disc_critic": 2}
    assert torch.equal(
        agent.amp_component.running_reward_norm.loaded_state["amp"],
        torch.tensor([3.0]),
    )


def test_amp_get_state_dict_adds_amp_optimizers_and_reward_norm(monkeypatch):
    monkeypatch.setattr(
        PPO,
        "get_state_dict",
        lambda self, state_dict: state_dict | {"base": True},
    )
    agent = _new_amp_agent()
    agent.amp_component.use_disc_critic = True
    agent.config = _amp_config()
    agent.amp_component.discriminator_optimizer = _OptimizerRecorder(state={"disc": 1})
    agent.amp_component.disc_critic_optimizer = _OptimizerRecorder(state={"disc_critic": 2})
    agent.amp_component.running_reward_norm = _RewardNorm(offset=3.0)

    state = AMP.get_state_dict(agent, {})

    assert state["base"] is True
    assert state["discriminator_optimizer"] == {"disc": 1}
    assert state["disc_critic_optimizer"] == {"disc_critic": 2}
    assert torch.equal(state["running_amp_reward_norm"]["offset"], torch.tensor(3.0))

    agent.amp_component.use_disc_critic = False
    agent.config.normalize_rewards = False
    state = AMP.get_state_dict(agent, {})
    assert "disc_critic_optimizer" not in state
    assert "running_amp_reward_norm" not in state


def test_amp_disc_replay_buffer_filters_overfull_buffer_and_caps_large_batches():
    agent = _new_amp_agent()
    agent.device = torch.device("cpu")
    agent.config = _amp_config(discriminator_replay_keep_prob=0.0)
    agent.amp_component.replay_buffer = _ReplayRecorder(buffer_size=3, current_len=4)

    AMP.update_disc_replay_buffer(
        agent,
        {
            "disc_obs": torch.arange(2, dtype=torch.float).view(2, 1),
            "extra_disc_obs": torch.arange(2, dtype=torch.float).view(2, 1),
        },
    )

    assert agent.amp_component.replay_buffer.stored["disc_obs"].shape == (0, 1)
    assert agent.amp_component.replay_buffer.stored["extra_disc_obs"].shape == (0, 1)

    real_buffer_agent = _new_amp_agent()
    real_buffer_agent.device = torch.device("cpu")
    real_buffer_agent.config = _amp_config()
    real_buffer_agent.amp_component.replay_buffer = ReplayBuffer(3, device=torch.device("cpu"))
    torch.manual_seed(0)
    AMP.update_disc_replay_buffer(
        real_buffer_agent,
        {"disc_obs": torch.arange(10, dtype=torch.float).view(10, 1)},
    )

    assert len(real_buffer_agent.amp_component.replay_buffer) == 3
    assert real_buffer_agent.amp_component.replay_buffer.disc_obs.shape == (3, 1)


def test_amp_disc_replay_keep_prob_applies_once_real_buffer_is_full():
    agent = _new_amp_agent()
    agent.device = torch.device("cpu")
    agent.config = _amp_config(discriminator_replay_keep_prob=0.0)
    agent.amp_component.replay_buffer = ReplayBuffer(3, device=torch.device("cpu"))
    agent.amp_component.replay_buffer.store(
        {"disc_obs": torch.tensor([[10.0], [11.0], [12.0]])}
    )

    AMP.update_disc_replay_buffer(
        agent,
        {"disc_obs": torch.tensor([[0.0], [1.0]])},
    )

    assert len(agent.amp_component.replay_buffer) == 3
    torch.testing.assert_close(
        agent.amp_component.replay_buffer.disc_obs,
        torch.tensor([[10.0], [11.0], [12.0]]),
    )


def test_amp_reference_observation_sampling_chunks_and_filters_router_inputs():
    def ref_obs(motion_ids, motion_times, dt, scale):
        return (motion_ids.float() + motion_times + dt + scale).view(-1, 1)

    agent = _new_amp_agent()
    agent.motion_manager = _MotionManager()
    agent.motion_lib = object()
    agent.num_envs = 2
    agent.env = SimpleNamespace(
        simulator=SimpleNamespace(dt=0.25),
        config=SimpleNamespace(num_state_history_steps=4),
    )
    agent.config = _amp_config()
    agent.config.reference_obs_components = {
        "disc_obs": _Router(ref_obs, {"scale": 2.0, "unused": "ignored"}),
        "ignored_obs": _Router(ref_obs, {"scale": 100.0}),
    }
    agent.amp_component.discriminator = SimpleNamespace(module=SimpleNamespace(in_keys=["disc_obs"]))

    expert_obs = AMP.get_expert_disc_obs(agent, num_samples=5)

    expected_ids = torch.arange(5, dtype=torch.float)
    assert set(expert_obs) == {"disc_obs"}
    assert torch.allclose(
        expert_obs["disc_obs"].squeeze(-1),
        expected_ids + expected_ids * 0.5 + 0.25 + 2.0,
    )


def test_amp_reference_historical_obs_matches_simulator_reset_history_buffer():
    motion_ids = torch.tensor([0, 2, 4])
    motion_times = torch.tensor([1.2, 1.6, 2.0])
    dt = 0.1
    num_state_history_steps = 5
    history_steps = [1, 3, 5]
    motion_lib = _DeterministicMotionLib(num_bodies=3, num_dofs=2)

    agent = _new_amp_agent()
    agent.motion_manager = _FixedMotionManager(motion_ids, motion_times)
    agent.motion_lib = motion_lib
    agent.num_envs = 2
    agent.env = SimpleNamespace(
        simulator=SimpleNamespace(dt=dt),
        config=SimpleNamespace(num_state_history_steps=num_state_history_steps),
    )
    agent.config = _amp_config()
    agent.config.reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},
            static_params={
                "history_steps": history_steps,
                "local_obs": True,
                "root_height_obs": True,
            },
        )
    }
    agent.amp_component.discriminator = SimpleNamespace(
        module=SimpleNamespace(in_keys=["historical_max_coords_obs"])
    )

    expert_obs = AMP.get_expert_disc_obs(agent, num_samples=motion_ids.shape[0])

    simulator_obs = _simulator_historical_max_coords_from_reference_reset(
        motion_lib,
        motion_ids,
        motion_times,
        dt=dt,
        num_state_history_steps=num_state_history_steps,
        history_steps=history_steps,
        local_obs=True,
        root_height_obs=True,
    )

    assert torch.allclose(
        expert_obs["historical_max_coords_obs"],
        simulator_obs,
        atol=1e-6,
    )


def test_amp_reference_historical_obs_clamps_to_motion_lengths_like_simulator_reset():
    motion_ids = torch.tensor([0, 1])
    motion_times = torch.tensor([0.9, 0.75])
    motion_lengths = torch.tensor([0.65, 0.55])
    dt = 0.1
    num_state_history_steps = 3
    history_steps = [1, 2, 3]
    motion_lib = _DeterministicMotionLib(
        num_bodies=3,
        num_dofs=2,
        motion_lengths=motion_lengths,
    )

    agent = _new_amp_agent()
    agent.motion_manager = _FixedMotionManager(motion_ids, motion_times)
    agent.motion_lib = motion_lib
    agent.num_envs = 1
    agent.env = SimpleNamespace(
        simulator=SimpleNamespace(dt=dt),
        config=SimpleNamespace(num_state_history_steps=num_state_history_steps),
    )
    agent.config = _amp_config()
    agent.config.reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},
            static_params={
                "history_steps": history_steps,
                "local_obs": False,
                "root_height_obs": True,
            },
        )
    }
    agent.amp_component.discriminator = SimpleNamespace(
        module=SimpleNamespace(in_keys=["historical_max_coords_obs"])
    )

    expert_obs = AMP.get_expert_disc_obs(agent, num_samples=motion_ids.shape[0])

    simulator_obs = _simulator_historical_max_coords_from_reference_reset(
        motion_lib,
        motion_ids,
        motion_times,
        dt=dt,
        num_state_history_steps=num_state_history_steps,
        history_steps=history_steps,
        local_obs=False,
        root_height_obs=True,
    )

    assert torch.allclose(
        expert_obs["historical_max_coords_obs"],
        simulator_obs,
        atol=1e-6,
    )


def test_amp_reference_observation_call_reports_missing_required_context():
    def needs_unknown(motion_ids, missing_value):
        return motion_ids + missing_value

    with pytest.raises(TypeError, match="missing required arguments"):
        AMP._call_ref_obs_fn(
            needs_unknown,
            runtime_context={"motion_ids": torch.arange(2)},
            static_params={},
        )


def test_amp_reference_observation_call_ignores_variadic_params_and_uses_defaults():
    def accepts_variadic(motion_ids, scale=2.0, *args, **kwargs):
        return motion_ids.float() * scale

    result = AMP._call_ref_obs_fn(
        accepts_variadic,
        runtime_context={"motion_ids": torch.arange(3)},
        static_params={},
    )

    assert torch.equal(result, torch.tensor([0.0, 2.0, 4.0]))


def test_amp_get_expert_disc_obs_requires_reference_components():
    agent = _new_amp_agent()
    agent.motion_manager = _MotionManager()
    agent.num_envs = 2
    agent.config = _amp_config()
    agent.config.reference_obs_components = {}

    with pytest.raises(ValueError, match="reference_obs_components"):
        AMP.get_expert_disc_obs(agent, num_samples=2)


def test_amp_process_dataset_adds_discriminator_training_views_and_stores_replay():
    def ref_obs(motion_ids, motion_times, dt):
        return (motion_ids.float() + motion_times + dt).view(-1, 1)

    agent = _new_amp_agent()
    agent.device = torch.device("cpu")
    agent.num_envs = 2
    agent.num_steps = 2
    agent.num_mini_epochs = 1
    agent.config = _amp_config()
    agent.config.batch_size = 2
    agent.motion_manager = _MotionManager()
    agent.motion_lib = object()
    agent.env = SimpleNamespace(
        simulator=SimpleNamespace(dt=0.25),
        config=SimpleNamespace(num_state_history_steps=1),
    )
    agent.config.reference_obs_components = {
        "disc_obs": _Router(ref_obs, {}),
    }
    agent.model = SimpleNamespace(_discriminator=SimpleNamespace(in_keys=["disc_obs"]))
    agent.amp_component.discriminator = SimpleNamespace(module=SimpleNamespace(in_keys=["disc_obs"]))
    agent.amp_component.replay_buffer = ReplayBuffer(8, device=torch.device("cpu"))
    dataset = {
        "disc_obs": torch.arange(4, dtype=torch.float).view(4, 1),
        "action": torch.zeros(4, 1),
    }

    processed = AMP.process_dataset(agent, dataset)

    assert processed.batch_size == 2
    assert set(processed.tensor_dict).issuperset(
        {
            "agent_disc_obs",
            "replay_disc_obs",
            "expert_disc_obs",
        }
    )
    assert torch.equal(processed.tensor_dict["agent_disc_obs"], dataset["disc_obs"])
    assert torch.equal(processed.tensor_dict["replay_disc_obs"], dataset["disc_obs"])
    assert torch.allclose(
        processed.tensor_dict["expert_disc_obs"].squeeze(-1),
        torch.tensor([0.25, 1.75, 3.25, 4.75]),
    )
    assert len(agent.amp_component.replay_buffer) == 4
    assert torch.equal(agent.amp_component.replay_buffer.disc_obs[:4], dataset["disc_obs"])


def test_amp_process_dataset_samples_existing_replay_when_available():
    def ref_obs(motion_ids, motion_times, dt):
        return (motion_ids.float() + motion_times + dt).view(-1, 1)

    agent = _new_amp_agent()
    agent.device = torch.device("cpu")
    agent.num_envs = 2
    agent.num_steps = 2
    agent.num_mini_epochs = 1
    agent.config = _amp_config()
    agent.config.batch_size = 2
    agent.motion_manager = _MotionManager()
    agent.motion_lib = object()
    agent.env = SimpleNamespace(
        simulator=SimpleNamespace(dt=0.25),
        config=SimpleNamespace(num_state_history_steps=1),
    )
    agent.config.reference_obs_components = {"disc_obs": _Router(ref_obs, {})}
    agent.model = SimpleNamespace(_discriminator=SimpleNamespace(in_keys=["disc_obs"]))
    agent.amp_component.discriminator = SimpleNamespace(module=SimpleNamespace(in_keys=["disc_obs"]))
    agent.amp_component.replay_buffer = ReplayBuffer(8, device=torch.device("cpu"))
    agent.amp_component.replay_buffer.store(
        {"disc_obs": torch.tensor([[10.0], [11.0], [12.0], [13.0]])}
    )
    dataset = {
        "disc_obs": torch.arange(4, dtype=torch.float).view(4, 1),
        "action": torch.zeros(4, 1),
    }

    processed = AMP.process_dataset(agent, dataset)

    assert processed.tensor_dict["replay_disc_obs"].shape == (4, 1)
    assert torch.all(processed.tensor_dict["replay_disc_obs"] >= 10.0)
    assert torch.equal(processed.tensor_dict["agent_disc_obs"], dataset["disc_obs"])


def test_amp_post_env_step_modifications_adds_discriminator_termination_and_resets():
    agent = _new_amp_agent()
    agent.model = _ResetRecorder()
    agent.amp_component.num_cumulative_bad_transitions = torch.tensor([0, 3, 2])
    agent.config = _amp_config(discriminator_max_cumulative_bad_transitions=2)
    dones = torch.tensor([False, False, False])
    terminated = torch.tensor([False, False, False])
    extras = {}

    out_dones, out_terminated, out_extras = AMP.post_env_step_modifications(
        agent,
        dones,
        terminated,
        extras,
    )

    assert torch.equal(out_dones, torch.tensor([False, True, True]))
    assert torch.equal(out_terminated, torch.tensor([False, True, True]))
    assert out_extras is extras
    assert torch.equal(agent.model.env_ids, torch.tensor([1, 2]))
    assert torch.equal(
        extras["amp_discriminator_termination"],
        torch.tensor([False, True, True]),
    )


def test_amp_record_rollout_step_updates_ppo_disc_critic_and_amp_rewards():
    agent = _new_amp_agent()
    agent.current_rewards = torch.zeros(3)
    agent.current_lengths = torch.zeros(3, dtype=torch.long)
    agent.episode_reward_meter = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.episode_length_meter = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.episode_env_tensors = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.experience_buffer = _BufferRecorder()
    agent.config = _amp_config()
    agent.running_reward_norm = _RewardNorm(offset=10.0)
    agent.amp_component.running_reward_norm = _RewardNorm(offset=20.0)
    agent.amp_component.use_disc_critic = True
    agent.model = SimpleNamespace(_critic=_ConstantCritic("value", value=7.0))
    agent.amp_component.discriminator = _ModuleWrapper(_LinearDiscriminator())
    agent.amp_component.disc_critic = _ModuleWrapper(_ConstantCritic("disc_value", value=5.0))
    agent.amp_component.num_cumulative_bad_transitions = torch.zeros(3, dtype=torch.int32)
    next_obs_td = TensorDict(
        {"disc_obs": torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0]])},
        batch_size=3,
    )
    terminated = torch.tensor([False, True, False])

    AMP.record_rollout_step(
        agent,
        next_obs_td=next_obs_td,
        actions=torch.zeros(3, 1),
        rewards=torch.ones(3),
        dones=torch.tensor([False, True, False]),
        terminated=terminated,
        done_indices=torch.tensor([1]),
        extras={},
        step=1,
    )

    assert torch.equal(
        agent.experience_buffer.data[("next_value", 1)],
        torch.tensor([[7.0], [0.0], [7.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("next_disc_value", 1)],
        torch.tensor([[5.0], [0.0], [5.0]]),
    )
    assert agent.experience_buffer.data[("amp_rewards", 1)].shape == (3,)
    assert agent.running_reward_norm.recorded
    assert agent.amp_component.running_reward_norm.recorded
    assert agent.amp_component.num_cumulative_bad_transitions.shape == (3,)


def test_amp_normalizes_task_amp_and_disc_critic_buffers():
    agent = _new_amp_agent()
    agent.config = _amp_config()
    agent.amp_component.use_disc_critic = True
    agent.running_reward_norm = _RewardNorm(offset=1.0)
    agent.amp_component.running_reward_norm = _RewardNorm(offset=2.0)
    agent.experience_buffer = _BufferRecorder()
    agent.experience_buffer.rewards = torch.tensor([[1.0, 2.0]])
    agent.experience_buffer.value = torch.tensor([[[3.0], [4.0]]])
    agent.experience_buffer.next_value = torch.tensor([[[5.0], [6.0]]])
    agent.experience_buffer.amp_rewards = torch.tensor([[7.0, 8.0]])
    agent.experience_buffer.disc_value = torch.tensor([[[9.0], [10.0]]])
    agent.experience_buffer.next_disc_value = torch.tensor([[[11.0], [12.0]]])

    AMP.normalize_rewards_in_buffer(agent)

    assert torch.equal(agent.experience_buffer.rewards, torch.tensor([[2.0, 3.0]]))
    assert torch.equal(
        agent.experience_buffer.amp_rewards,
        torch.tensor([[9.0, 10.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.unnormalized_disc_value,
        torch.tensor([[[7.0], [8.0]]]),
    )
    assert torch.equal(
        agent.experience_buffer.unnormalized_next_disc_value,
        torch.tensor([[[9.0], [10.0]]]),
    )


def test_amp_normalize_rewards_returns_when_disabled_or_without_disc_critic():
    disabled = _new_amp_agent()
    disabled.config = _amp_config()
    disabled.config.normalize_rewards = False
    disabled.experience_buffer = SimpleNamespace(
        batch_update_data=lambda *args, **kwargs: pytest.fail("unexpected update")
    )

    AMP.normalize_rewards_in_buffer(disabled)

    agent = _new_amp_agent()
    agent.config = _amp_config()
    agent.amp_component.use_disc_critic = False
    agent.running_reward_norm = _RewardNorm(offset=1.0)
    agent.amp_component.running_reward_norm = _RewardNorm(offset=2.0)
    agent.experience_buffer = _BufferRecorder()
    agent.experience_buffer.rewards = torch.tensor([[1.0, 2.0]])
    agent.experience_buffer.value = torch.zeros(1, 2, 1)
    agent.experience_buffer.next_value = torch.zeros(1, 2, 1)
    agent.experience_buffer.amp_rewards = torch.tensor([[3.0, 4.0]])

    AMP.normalize_rewards_in_buffer(agent)

    assert torch.equal(
        agent.experience_buffer.data[("unnormalized_amp_rewards", "batch")],
        torch.tensor([[3.0, 4.0]]),
    )
    assert "unnormalized_disc_value" not in {
        key for key, _ in agent.experience_buffer.data.keys()
    }


def test_amp_compute_advantages_combines_task_and_raw_discriminator_advantages():
    agent = _new_amp_agent()
    agent.config = _amp_config(discriminator_reward_w=0.25)
    agent.config.normalize_rewards = False
    agent.config.task_reward_w = 2.0
    agent.gamma = 0.0
    agent.tau = 1.0
    agent.amp_component.use_disc_critic = False
    agent.experience_buffer = SimpleNamespace(
        dones=torch.zeros(2),
        rewards=torch.tensor([1.0, 2.0]),
        value=torch.zeros(2, 1),
        next_value=torch.zeros(2, 1),
        amp_rewards=torch.tensor([4.0, 6.0]),
    )

    advantages = AMP.compute_advantages(agent)

    assert torch.equal(advantages["returns"], torch.tensor([1.0, 2.0]))
    assert torch.equal(advantages["advantages"], torch.tensor([3.0, 5.5]))
    assert torch.equal(agent._diag_task_advantages, torch.tensor([2.0, 4.0]))
    assert torch.equal(agent._diag_disc_advantages, torch.tensor([1.0, 1.5]))


def test_amp_compute_advantages_uses_discriminator_critic_when_enabled():
    agent = _new_amp_agent()
    agent.config = _amp_config(discriminator_reward_w=0.5)
    agent.config.normalize_rewards = False
    agent.config.task_reward_w = 1.0
    agent.gamma = 0.0
    agent.tau = 1.0
    agent.amp_component.use_disc_critic = True
    agent.experience_buffer = SimpleNamespace(
        dones=torch.zeros(2),
        rewards=torch.tensor([1.0, 2.0]),
        value=torch.zeros(2, 1),
        next_value=torch.zeros(2, 1),
        amp_rewards=torch.tensor([4.0, 6.0]),
        disc_value=torch.tensor([[1.0], [2.0]]),
        next_disc_value=torch.zeros(2, 1),
    )
    updates = {}
    agent.experience_buffer.batch_update_data = (
        lambda key, value: updates.setdefault(key, value.clone())
    )

    advantages = AMP.compute_advantages(agent)

    assert torch.equal(updates["disc_returns"], torch.tensor([4.0, 6.0]))
    assert torch.equal(advantages["returns"], torch.tensor([1.0, 2.0]))
    assert torch.equal(advantages["advantages"], torch.tensor([2.5, 4.0]))
    assert torch.equal(agent._diag_disc_advantages, torch.tensor([1.5, 2.0]))


def test_amp_compute_advantages_uses_normalized_disc_critic_inputs_and_returns():
    agent = _new_amp_agent()
    agent.config = _amp_config(discriminator_reward_w=0.5)
    agent.config.normalize_rewards = True
    agent.config.task_reward_w = 1.0
    agent.gamma = 0.0
    agent.tau = 1.0
    agent.amp_component.use_disc_critic = True
    agent.running_reward_norm = _RewardNorm(offset=10.0)
    agent.amp_component.running_reward_norm = _RewardNorm(offset=20.0)
    agent.experience_buffer = SimpleNamespace(
        dones=torch.zeros(1),
        unnormalized_rewards=torch.tensor([1.0]),
        unnormalized_value=torch.zeros(1, 1),
        unnormalized_next_value=torch.zeros(1, 1),
        unnormalized_amp_rewards=torch.tensor([4.0]),
        unnormalized_disc_value=torch.tensor([[1.0]]),
        unnormalized_next_disc_value=torch.zeros(1, 1),
    )
    updates = {}
    agent.experience_buffer.batch_update_data = (
        lambda key, value: updates.setdefault(key, value.clone())
    )

    advantages = AMP.compute_advantages(agent)

    assert torch.equal(updates["disc_returns"], torch.tensor([24.0]))
    assert torch.equal(advantages["returns"], torch.tensor([11.0]))
    assert torch.equal(advantages["advantages"], torch.tensor([2.5]))


def test_amp_compute_advantages_uses_raw_normalized_amp_rewards_without_disc_critic():
    agent = _new_amp_agent()
    agent.config = _amp_config(discriminator_reward_w=0.25)
    agent.config.normalize_rewards = True
    agent.config.task_reward_w = 1.0
    agent.gamma = 0.0
    agent.tau = 1.0
    agent.amp_component.use_disc_critic = False
    agent.running_reward_norm = _RewardNorm(offset=10.0)
    agent.experience_buffer = SimpleNamespace(
        dones=torch.zeros(1),
        unnormalized_rewards=torch.tensor([1.0]),
        unnormalized_value=torch.zeros(1, 1),
        unnormalized_next_value=torch.zeros(1, 1),
        unnormalized_amp_rewards=torch.tensor([4.0]),
    )

    advantages = AMP.compute_advantages(agent)

    assert torch.equal(advantages["advantages"], torch.tensor([2.0]))


def test_amp_perform_optimization_step_skips_disc_when_ppo_skips_actor(monkeypatch):
    monkeypatch.setattr(
        PPO,
        "perform_optimization_step",
        lambda self, batch_dict, batch_idx: {"ppo/loss": torch.tensor(1.0)},
    )
    agent = _new_amp_agent()
    agent._skip_actor_for_epoch = True
    agent.amp_component.use_disc_critic = True
    agent.disc_critic_step = lambda batch_dict: pytest.fail("disc critic should skip")
    agent.discriminator_step = lambda batch_dict: pytest.fail("discriminator should skip")

    log_dict = AMP.perform_optimization_step(agent, {}, batch_idx=0)

    assert set(log_dict) == {"ppo/loss"}
    assert torch.equal(log_dict["ppo/loss"], torch.tensor(1.0))


def test_amp_perform_optimization_step_updates_disc_critic_and_discriminator(
    monkeypatch,
):
    monkeypatch.setattr(
        PPO,
        "perform_optimization_step",
        lambda self, batch_dict, batch_idx: {"ppo/loss": torch.tensor(1.0)},
    )
    optimizer_calls = []

    def fake_step_optimizer(loss, model, optimizer, model_name):
        optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        optimizer_calls.append((model_name, loss.detach(), model, optimizer))
        clipped.append((model_name, model, optimizer))
        return {f"{model_name}/grad_norm": torch.tensor(2.0)}

    clipped = []
    agent = _new_amp_agent()
    agent._skip_actor_for_epoch = False
    agent.amp_component.use_disc_critic = True
    agent.config = _amp_config(discriminator_optimization_ratio=1)
    agent.amp_component.disc_critic = _ModuleWrapper(nn.Linear(1, 1, bias=False))
    agent.amp_component.discriminator = _ModuleWrapper(nn.Linear(1, 1, bias=False))
    agent.amp_component.disc_critic_optimizer = _OptimizerRecorder()
    agent.amp_component.discriminator_optimizer = _OptimizerRecorder()
    agent.disc_critic_step = lambda batch_dict: (
        agent.amp_component.disc_critic.module.weight.sum() * 2.0,
        {"losses/disc_critic_loss": torch.tensor(2.0)},
    )
    agent.discriminator_step = lambda batch_dict: (
        agent.amp_component.discriminator.module.weight.sum() * 3.0,
        {"losses/discriminator_loss": torch.tensor(3.0)},
    )
    agent._step_optimizer = fake_step_optimizer

    log_dict = AMP.perform_optimization_step(agent, {}, batch_idx=0)

    assert [name for name, _, _, _ in optimizer_calls] == [
        "disc_critic",
        "discriminator",
    ]
    assert agent.amp_component.disc_critic_optimizer.zero_grad_calls == [True]
    assert agent.amp_component.disc_critic_optimizer.steps == 1
    assert agent.amp_component.discriminator_optimizer.zero_grad_calls == [True]
    assert agent.amp_component.discriminator_optimizer.steps == 1
    assert [name for name, _, _ in clipped] == ["disc_critic", "discriminator"]
    assert torch.equal(log_dict["disc_critic/grad_norm"], torch.tensor(2.0))
    assert torch.equal(log_dict["discriminator/grad_norm"], torch.tensor(2.0))


def test_amp_perform_optimization_step_respects_discriminator_update_ratio(
    monkeypatch,
):
    monkeypatch.setattr(
        PPO,
        "perform_optimization_step",
        lambda self, batch_dict, batch_idx: {"ppo/loss": torch.tensor(1.0)},
    )
    agent = _new_amp_agent()
    agent._skip_actor_for_epoch = False
    agent.amp_component.use_disc_critic = False
    agent.config = _amp_config(discriminator_optimization_ratio=2)
    agent.discriminator_step = lambda batch_dict: pytest.fail("ratio should skip")

    log_dict = AMP.perform_optimization_step(agent, {}, batch_idx=1)

    assert set(log_dict) == {"ppo/loss"}
    assert torch.equal(log_dict["ppo/loss"], torch.tensor(1.0))


def test_amp_disc_critic_step_uses_clipped_value_loss_and_reports_diagnostics():
    agent = _new_amp_agent()
    agent.config = _amp_config()
    agent.e_clip = agent.config.e_clip
    agent.amp_component.disc_critic = _ModuleWrapper(_ConstantCritic("disc_value", value=1.0))
    batch = {
        "action": torch.zeros(3, 1),
        "disc_obs": torch.zeros(3, 2),
        "disc_value": torch.zeros(3, 1),
        "disc_returns": torch.tensor([0.0, 1.0, 2.0]),
    }

    loss, log_dict = AMP.disc_critic_step(agent, batch)

    assert torch.isfinite(loss)
    assert "losses/disc_critic_loss" in log_dict
    assert "disc_critic/explained_variance" in log_dict
    assert log_dict["disc_critic/value_mean"] == 1.0


def test_amp_disc_critic_step_supports_unclipped_loss_and_zero_return_variance():
    agent = _new_amp_agent()
    agent.config = _amp_config()
    agent.config.clip_critic_loss = False
    agent.amp_component.disc_critic = _ModuleWrapper(_ConstantCritic("disc_value", value=1.0))
    batch = {
        "action": torch.zeros(2, 1),
        "disc_obs": torch.zeros(2, 2),
        "disc_value": torch.zeros(2, 1),
        "disc_returns": torch.ones(2),
    }

    loss, log_dict = AMP.disc_critic_step(agent, batch)

    assert loss.item() == pytest.approx(0.0)
    assert log_dict["disc_critic/explained_variance"].item() == 0.0
    assert log_dict["disc_critic/return_std"].item() == 0.0


def test_amp_post_epoch_logging_adds_amp_reward_stats_and_delegates(monkeypatch):
    def parent_post_epoch_logging(self, training_log_dict):
        training_log_dict["parent_called"] = True

    monkeypatch.setattr(PPO, "post_epoch_logging", parent_post_epoch_logging)
    agent = _new_amp_agent()
    agent.config = _amp_config()
    agent.config.normalize_rewards = True
    agent.experience_buffer = SimpleNamespace(
        amp_rewards=torch.tensor([1.0, 2.0, 3.0]),
        unnormalized_amp_rewards=torch.tensor([4.0, 5.0, 6.0]),
    )
    agent.amp_component.running_reward_norm = SimpleNamespace(var=torch.tensor([7.0]))
    agent._diag_task_advantages = torch.tensor([1.0, 3.0])
    agent._diag_disc_advantages = torch.tensor([2.0, 4.0])
    training_log_dict = {}

    AMP.post_epoch_logging(agent, training_log_dict)

    assert training_log_dict["rewards/amp_rewards"].item() == pytest.approx(2.0)
    assert training_log_dict["rewards/amp_rewards_std"].item() == pytest.approx(1.0)
    assert training_log_dict["rewards/unnormalized_amp_rewards"] == pytest.approx(5.0)
    assert training_log_dict["amp_reward_norm/var"] == pytest.approx(7.0)
    assert training_log_dict["advantages/task_mean"].item() == pytest.approx(2.0)
    assert training_log_dict["advantages/disc_std"].item() == pytest.approx(
        torch.tensor([2.0, 4.0]).std().item()
    )
    assert training_log_dict["parent_called"] is True


def test_amp_discriminator_step_trains_against_expert_agent_and_replay_batches():
    torch.manual_seed(0)
    agent = _new_amp_agent()
    agent.device = torch.device("cpu")
    agent.config = _amp_config()
    agent.amp_component.discriminator = _ModuleWrapper(_LinearDiscriminator())
    batch = {
        "agent_disc_obs": torch.tensor([[0.0, 1.0], [1.0, 0.0], [9.0, 9.0]]),
        "replay_disc_obs": torch.tensor([[0.5, 0.5], [1.0, 1.0], [9.0, 9.0]]),
        "expert_disc_obs": torch.tensor([[2.0, 0.0], [0.0, 2.0], [9.0, 9.0]]),
    }

    loss, log_dict = AMP.discriminator_step(agent, batch)

    assert torch.isfinite(loss)
    assert "losses/discriminator_loss" in log_dict
    assert "discriminator/grad_penalty" in log_dict
    assert "discriminator/l2_loss" in log_dict
    assert "discriminator/negative_logit_mean" in log_dict
    expected_weight_total = agent.amp_component.discriminator.module.linear.weight.pow(2).sum()
    torch.testing.assert_close(log_dict["discriminator/l2_total"], expected_weight_total)
    torch.testing.assert_close(
        log_dict["discriminator/l2_loss"],
        expected_weight_total
        * agent.config.amp_parameters.discriminator_weight_decay,
    )
    torch.testing.assert_close(
        log_dict["discriminator/l2_logit_total"],
        expected_weight_total,
    )
    torch.testing.assert_close(
        log_dict["discriminator/l2_logit_loss"],
        expected_weight_total
        * agent.config.amp_parameters.discriminator_logit_weight_decay,
    )


def test_amp_discriminator_step_handles_conditional_negative_and_integer_expert_obs():
    torch.manual_seed(0)
    agent = _new_amp_agent()
    agent.device = torch.device("cpu")
    agent.config = _amp_config(
        conditional_discriminator=True,
        discriminator_weight_decay=0.0,
        discriminator_logit_weight_decay=0.0,
    )
    agent.amp_component.discriminator = _ModuleWrapper(_LinearDiscriminator())
    agent.produce_negative_expert_obs = lambda batch_dict: {
        "disc_obs": torch.tensor([[3.0, 3.0], [4.0, 4.0]])
    }
    batch = {
        "agent_disc_obs": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "replay_disc_obs": torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
        "expert_disc_obs": torch.tensor([[2, 0], [0, 2]], dtype=torch.long),
    }

    loss, log_dict = AMP.discriminator_step(agent, batch)

    assert torch.isfinite(loss)
    assert log_dict["discriminator/l2_loss"].item() == 0.0
    assert log_dict["discriminator/l2_logit_loss"].item() == 0.0
    assert "discriminator/negative_expert_logit_mean" in log_dict
    assert "discriminator/negative_logit_mean" in log_dict


def test_amp_terminate_early_sets_stop_flag():
    agent = _new_amp_agent()

    AMP.terminate_early(agent)

    assert agent._should_stop is True
