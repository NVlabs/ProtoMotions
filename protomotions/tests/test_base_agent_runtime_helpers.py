# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generic BaseAgent runtime helpers."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent import agent as base_agent_module
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.base_agent.config import (
    BaseModelConfig,
    MaxEpisodeLengthManagerConfig,
)
from protomotions.agents.base_agent.model import BaseModel, RolloutStateSpec
from protomotions.agents.utils.data import ExperienceBuffer
from protomotions.agents.amp.component import AMPTrainingComponent
from protomotions.agents.utils.metering import TensorAverageMeterDict
from protomotions.agents.utils.normalization import RunningMeanStd


def _new_amp_agent(agent_cls=None):
    if agent_cls is None:
        from protomotions.agents.amp.agent import AMP as agent_cls

    agent = object.__new__(agent_cls)
    component = object.__new__(AMPTrainingComponent)
    component.agent = agent
    agent.amp_component = component
    return agent


class _ResetRecorder:
    def __init__(self):
        self.env_ids = None

    def reset_rollout_context(self, env_ids=None, num_envs: int = None, device=None):
        self.env_ids = env_ids


class _RolloutModel:
    def __call__(self, obs_td):
        obs_td["action"] = obs_td["obs"] + 1.0
        obs_td["vae_noise"] = obs_td["obs"] * 0.0 + 2.0
        obs_td["not_registered"] = obs_td["obs"] * 0.0 + 3.0
        return obs_td


class _ExperienceBufferRecorder:
    def __init__(self):
        self.data = {}

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()

    def batch_update_data(self, key, value):
        setattr(self, key, value.clone())
        self.data[(key, "batch")] = value.clone()


class _OptimizerRecorder:
    def __init__(self, lr=0.1):
        self.param_groups = [{"lr": lr}]

    def zero_grad(self, set_to_none=False):
        self.zero_grad_set_to_none = set_to_none

    def step(self):
        self.stepped = True

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        self.loaded_state = state


class _StateDictLoadRecorder:
    def load_state_dict(self, state, strict=True):
        self.loaded_state = state
        self.strict = strict
        return ["missing"], []


class _LazyLoadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.materialized_from_keys = None

    def materialize_from_state_dict(self, state_dict):
        self.materialized_from_keys = sorted(state_dict)
        self.linear = nn.Linear(
            state_dict["linear.weight"].shape[1],
            state_dict["linear.weight"].shape[0],
        )


class _PPOLogstdLoadModel(nn.Module):
    def __init__(self, actor_logstd: float):
        super().__init__()
        self._actor = SimpleNamespace(
            logstd=nn.Parameter(torch.tensor([actor_logstd], dtype=torch.float32))
        )
        self.materialized_from_keys = None

    def materialize_from_state_dict(self, state_dict):
        self.materialized_from_keys = sorted(state_dict)

    def load_state_dict(self, state_dict, strict=True):
        self._actor.logstd.data.copy_(state_dict["_actor.logstd"])


def test_base_agent_has_critic_is_false_for_agents_without_critic_contract():
    agent = object.__new__(BaseAgent)
    assert BaseAgent.has_critic.fget(agent) is False

    agent.critic = object()
    assert BaseAgent.has_critic.fget(agent) is False


class _ConcreteBaseModel(BaseModel):
    def forward(self, tensordict, log_internals: bool = False):
        tensordict["action"] = tensordict["obs"] + 1.0
        return tensordict


class _RolloutStateSpecModel(BaseModel):
    def __init__(self):
        super().__init__(BaseModelConfig(out_keys=["action"]))
        self.out_keys = ["action"]

    def rollout_state_specs(self):
        return {
            "bool_state": RolloutStateSpec(dtype=torch.bool),
            "long_state": RolloutStateSpec(dtype=torch.long),
            "float_state": RolloutStateSpec(shape=(2,), dtype=torch.float32),
        }

    def forward(self, tensordict, log_internals: bool = False):
        batch_size = tensordict.batch_size[0]
        device = tensordict.device
        tensordict["action"] = torch.zeros(batch_size, 1, device=device)
        tensordict["bool_state"] = torch.tensor(
            [True, False], dtype=torch.bool, device=device
        )[:batch_size]
        tensordict["long_state"] = torch.arange(
            batch_size, dtype=torch.long, device=device
        )
        tensordict["float_state"] = torch.ones(
            batch_size, 2, dtype=torch.float32, device=device
        )
        return tensordict


class _RolloutStateMismatchModel(_RolloutStateSpecModel):
    def rollout_state_specs(self):
        return {
            "bool_state": RolloutStateSpec(dtype=torch.bool),
            "float_state": RolloutStateSpec(shape=(3,), dtype=torch.float32),
        }


class _OneBatchDataset:
    do_shuffle = True

    def __init__(self, batch):
        self.batch = batch
        self.shuffle_calls = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.batch

    def shuffle(self):
        self.shuffle_calls += 1


class _BatchExperience:
    def __init__(self, data):
        self.data = data

    def make_dict(self):
        return self.data


class _OffsetNormalizer:
    def __init__(self, offset):
        self.offset = offset

    def normalize(self, values, un_norm: bool = False):
        return values - self.offset if un_norm else values + self.offset

    def record_reward(self, rewards, terminated):
        self.recorded = (rewards.clone(), terminated.clone())


class _FabricRecorder:
    device = torch.device("cpu")
    global_rank = 0
    world_size = 2
    loggers = []

    def __init__(self):
        self.calls = []
        self.logged = []
        self.broadcasts = []

    def all_gather(self, value):
        if torch.is_tensor(value) and value.item() == 2:
            return torch.tensor([[2], [3]], device=value.device)
        return torch.stack([value.clone(), value.clone()])

    def broadcast(self, value):
        self.broadcasts.append(value)
        return value

    def backward(self, loss):
        self.calls.append(("backward", (loss,)))
        loss.backward()

    def call(self, name, *args):
        self.calls.append((name, args))

    def log_dict(self, log_dict, step):
        self.logged.append((log_dict, step))


class _SetupFabricRecorder(_FabricRecorder):
    def setup(self, module, optimizer):
        self.calls.append(("setup", (module, optimizer)))
        return SimpleNamespace(module=module), optimizer


class _SetupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = RunningMeanStd(None, shape=(1,), device=torch.device("cpu"))
        self.reset_args = None
        self.materialized_obs = None

    def reset_rollout_context(self, num_envs: int, device):
        self.reset_args = (num_envs, device)

    def materialize(self, obs_td):
        self.materialized_obs = obs_td.clone()
        return obs_td


def test_base_agent_init_sets_distributed_sizes_reward_norm_and_evaluator(
    tmp_path,
    monkeypatch,
):
    constructed_norms = []

    class _RewardRunningMeanStd:
        def __init__(self, **kwargs):
            constructed_norms.append(kwargs)

    class _Evaluator:
        def __init__(self, agent, fabric, config):
            self.agent = agent
            self.fabric = fabric
            self.config = config

    monkeypatch.setattr(base_agent_module, "RewardRunningMeanStd", _RewardRunningMeanStd)
    monkeypatch.setattr(base_agent_module, "get_class", lambda target: _Evaluator)

    env = SimpleNamespace(
        motion_lib="motion-lib",
        motion_manager="motion-manager",
        num_envs=2,
    )
    config = SimpleNamespace(
        num_steps=3,
        num_mini_epochs=2,
        gamma=0.95,
        training_max_steps=300,
        batch_size=4,
        normalize_rewards=True,
        normalized_reward_clamp_value=7.0,
        reward_norm_ema_decay=0.5,
        evaluator=SimpleNamespace(_target_="fake.Evaluator"),
    )

    agent = BaseAgent(_FabricRecorder(), env, config, root_dir=tmp_path)

    assert agent.motion_lib == "motion-lib"
    assert agent.motion_manager == "motion-manager"
    assert agent._total_envs == 5
    assert agent.max_epochs == 20
    assert agent.root_dir == tmp_path
    assert agent.evaluator.agent is agent
    assert constructed_norms[0]["gamma"] == 0.95
    assert constructed_norms[0]["clamp_value"] == 7.0
    assert constructed_norms[0]["ema_decay"] == 0.5
    assert agent.should_stop is False


def test_base_agent_init_rejects_mismatched_batch_counts(tmp_path):
    class _BadFabric(_FabricRecorder):
        def all_gather(self, value):
            if torch.is_tensor(value) and value.item() == 2:
                return torch.tensor([[2], [2]], device=value.device)
            return torch.tensor([[1], [2]], device=value.device)

    env = SimpleNamespace(motion_lib=None, motion_manager=None, num_envs=2)
    config = SimpleNamespace(
        num_steps=3,
        num_mini_epochs=2,
        gamma=0.99,
        training_max_steps=100,
        batch_size=4,
        normalize_rewards=False,
        evaluator=SimpleNamespace(_target_="unused"),
    )

    with pytest.raises(ValueError, match="max_num_batches differs"):
        BaseAgent(_BadFabric(), env, config, root_dir=tmp_path)


def test_base_agent_setup_materializes_model_and_initializes_optimizers():
    fabric = _FabricRecorder()
    model = _SetupModel()
    optimizer_models = []
    agent = object.__new__(BaseAgent)
    agent.fabric = fabric
    agent.device = torch.device("cpu")
    agent.num_envs = 2
    agent.env = SimpleNamespace(get_obs=lambda: {"obs": torch.ones(2, 1)})
    agent.create_model = lambda: model
    agent._after_model_reset = lambda: setattr(
        model,
        "fabric_during_after_model_reset",
        model.normalizer.fabric,
    )
    agent.create_optimizers = lambda created_model: optimizer_models.append(
        created_model
    )

    BaseAgent.setup(agent)

    assert fabric.calls == [
        ("on_model_init_start", ()),
        ("on_model_init_end", ()),
        ("on_optimizer_init_start", ()),
        ("on_optimizer_init_end", ()),
    ]
    assert agent.model is model
    assert model.reset_args == (2, torch.device("cpu"))
    assert model.fabric_during_after_model_reset is fabric
    assert model.normalizer.fabric is fabric
    assert torch.equal(model.materialized_obs["obs"], torch.ones(2, 1))
    assert optimizer_models == [model]


def test_base_agent_step_optimizer_runs_backward_clipping_and_step(monkeypatch):
    clipped = []

    def fake_clip(config, fabric, model, optimizer, model_name):
        clipped.append((config, fabric, model, optimizer, model_name))
        return {f"{model_name}/grad_norm": torch.tensor(2.0)}

    monkeypatch.setattr(base_agent_module, "handle_model_grad_clipping", fake_clip)
    agent = object.__new__(BaseAgent)
    agent.fabric = _FabricRecorder()
    agent.config = SimpleNamespace()
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 1)).sum()

    log_dict = BaseAgent._step_optimizer(
        agent,
        loss=loss,
        model=model,
        optimizer=optimizer,
        model_name="model",
    )

    assert [item[-1] for item in clipped] == ["model"]
    assert clipped[0][1] is agent.fabric
    assert agent.fabric.calls == [("backward", (loss,))]
    assert log_dict == {"model/grad_norm": torch.tensor(2.0)}


def test_base_agent_setup_model_optimizer_uses_fabric_setup():
    agent = object.__new__(BaseAgent)
    agent.fabric = _SetupFabricRecorder()
    module = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    wrapped_module, wrapped_optimizer = BaseAgent._setup_model_optimizer(
        agent,
        module,
        optimizer,
    )

    assert wrapped_module.module is module
    assert wrapped_optimizer is optimizer
    assert agent.fabric.calls == [("setup", (module, optimizer))]


def test_amp_extra_optimizers_use_fabric_setup(monkeypatch):
    from protomotions.agents.amp import component as amp_component_module
    from protomotions.agents.amp.agent import AMP
    from protomotions.agents.ppo.agent import PPO

    monkeypatch.setattr(PPO, "create_optimizers", lambda self, model: None)
    optimizer_calls = []

    def _instantiate_optimizer(config, module, params=None):
        optimizer_calls.append((config, module, params))
        return _OptimizerRecorder(lr=0.1)

    monkeypatch.setattr(
        amp_component_module,
        "instantiate_optimizer",
        _instantiate_optimizer,
    )

    agent = _new_amp_agent()
    agent.config = SimpleNamespace(
        model=SimpleNamespace(
            discriminator_optimizer=SimpleNamespace(),
            disc_critic_optimizer=SimpleNamespace(),
        )
    )
    agent.amp_component.use_disc_critic = True
    setup_calls = []
    agent._setup_model_optimizer = lambda module, optimizer: (
        setup_calls.append((module, optimizer))
        or (SimpleNamespace(module=module), optimizer)
    )
    model = SimpleNamespace(
        _discriminator=nn.Linear(1, 1),
        _disc_critic=nn.Linear(1, 1),
    )

    AMP.create_optimizers(agent, model)

    assert [call[0] for call in setup_calls] == [
        model._discriminator,
        model._disc_critic,
    ]
    assert [call[1] for call in optimizer_calls] == [
        model._discriminator,
        model._disc_critic,
    ]
    assert agent.amp_component.discriminator.module is model._discriminator
    assert agent.amp_component.disc_critic.module is model._disc_critic


def test_ase_extra_optimizer_uses_fabric_setup(monkeypatch):
    from protomotions.agents.amp.agent import AMP
    from protomotions.agents.ase import agent as ase_agent_module
    from protomotions.agents.ase.agent import ASE

    monkeypatch.setattr(AMP, "create_optimizers", lambda self, model: None)
    monkeypatch.setattr(
        ase_agent_module,
        "instantiate",
        lambda config, params: _OptimizerRecorder(lr=0.1),
    )

    agent = _new_amp_agent(ASE)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(mi_critic_optimizer=SimpleNamespace())
    )
    setup_calls = []
    agent._setup_model_optimizer = lambda module, optimizer: (
        setup_calls.append((module, optimizer))
        or (SimpleNamespace(module=module), optimizer)
    )
    model = SimpleNamespace(_mi_critic=nn.Linear(1, 1))

    ASE.create_optimizers(agent, model)

    assert [call[0] for call in setup_calls] == [model._mi_critic]
    assert agent.mi_critic.module is model._mi_critic


def test_amp_and_ase_extra_steps_use_shared_fabric_optimizer_step(monkeypatch):
    from protomotions.agents.amp.agent import AMP
    from protomotions.agents.ase.agent import ASE

    monkeypatch.setattr(
        AMP,
        "discriminator_step",
        lambda self, batch: (torch.ones((), requires_grad=True), {"disc": 1}),
    )
    monkeypatch.setattr(
        AMP,
        "disc_critic_step",
        lambda self, batch: (torch.ones((), requires_grad=True), {"disc_critic": 1}),
    )
    monkeypatch.setattr(
        ASE,
        "mi_critic_step",
        lambda self, batch: (torch.ones((), requires_grad=True), {"mi_critic": 1}),
    )

    calls = []
    monkeypatch.setattr(
        BaseAgent,
        "_step_optimizer",
        lambda self, loss, model, optimizer, model_name: calls.append(model_name)
        or {f"{model_name}/grad": torch.tensor(1.0)},
    )

    amp = _new_amp_agent()
    amp._skip_actor_for_epoch = False
    amp.amp_component.use_disc_critic = True
    amp.config = SimpleNamespace(
        amp_parameters=SimpleNamespace(discriminator_optimization_ratio=1)
    )
    amp.amp_component.disc_critic = object()
    amp.amp_component.disc_critic_optimizer = object()
    amp.amp_component.discriminator = object()
    amp.amp_component.discriminator_optimizer = object()
    monkeypatch.setattr(
        "protomotions.agents.ppo.agent.PPO.perform_optimization_step",
        lambda self, batch, batch_idx: {"ppo": 1},
    )

    amp_log = AMP.perform_optimization_step(amp, {}, batch_idx=0)

    ase = _new_amp_agent(ASE)
    ase._skip_actor_for_epoch = False
    ase.mi_critic = object()
    ase.mi_critic_optimizer = object()
    monkeypatch.setattr(
        AMP,
        "perform_optimization_step",
        lambda self, batch, batch_idx: {"amp": 1},
    )

    ase_log = ASE.perform_optimization_step(ase, {}, batch_idx=0)

    assert calls == ["disc_critic", "discriminator", "mi_critic"]
    assert "disc_critic/grad" in amp_log
    assert "discriminator/grad" in amp_log
    assert "mi_critic/grad" in ase_log


def test_supervised_agent_uses_fabric_for_setup(monkeypatch):
    from protomotions.agents.supervised import agent as supervised_agent_module
    from protomotions.agents.supervised.agent import SupervisedAgent

    monkeypatch.setattr(
        supervised_agent_module,
        "instantiate_optimizer",
        lambda config, module: _OptimizerRecorder(lr=0.1),
    )

    class _Model(nn.Linear):
        def optimization_module(self):
            return self

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(model=SimpleNamespace(optimizer=SimpleNamespace()))
    agent.fabric = _SetupFabricRecorder()
    model = _Model(1, 1)

    SupervisedAgent.create_optimizers(agent, model)

    assert agent.training_model.module is model
    assert agent.supervised_optimizer.param_groups[0]["lr"] == 0.1
    assert agent.fabric.calls == [("setup", (model, agent.supervised_optimizer))]


def test_base_agent_load_uses_explicit_training_state_and_loads_matching_env_state(
    tmp_path,
):
    checkpoint = tmp_path / "inference.ckpt"
    env_checkpoint = tmp_path / "env_task-a.ckpt"
    torch.save(
        {"model": {"w": torch.tensor([1.0])}},
        checkpoint,
    )
    torch.save({"env": torch.tensor([2.0])}, env_checkpoint)

    fabric = _FabricRecorder()
    loaded = {}
    env = SimpleNamespace(
        get_task_id=lambda: "task-a",
        load_state_dict=lambda state: loaded.setdefault("env", state),
    )
    agent = object.__new__(BaseAgent)
    agent.fabric = fabric
    agent.device = torch.device("cpu")
    agent.env = env
    agent.root_dir = tmp_path
    agent.load_parameters = lambda state, load_training_state: loaded.setdefault(
        "agent",
        (state, load_training_state),
    )

    BaseAgent.load(agent, checkpoint, load_env=True, load_training_state=False)

    assert loaded["agent"][1] is False
    assert torch.equal(loaded["agent"][0]["model"]["w"], torch.tensor([1.0]))
    assert torch.equal(loaded["env"]["env"], torch.tensor([2.0]))
    assert agent.just_loaded_checkpoint_should_evaluate is True
    assert fabric.calls == [
        ("on_load_checkpoint_start", ()),
        ("on_load_checkpoint_end", ()),
    ]


def test_agent_training_state_hooks_do_not_thread_load_flag():
    from protomotions.agents.amp.agent import AMP
    from protomotions.agents.ase.agent import ASE
    from protomotions.agents.ppo.agent import PPO
    from protomotions.agents.supervised.agent import SupervisedAgent

    for agent_cls in [PPO, AMP, ASE, SupervisedAgent]:
        signature = inspect.signature(agent_cls._load_training_state)
        assert list(signature.parameters) == ["self", "state_dict"]


def test_ppo_unwrapped_module_properties_handle_wrapped_and_plain_modules():
    from protomotions.agents.ppo.agent import PPO

    agent = object.__new__(PPO)
    actor = SimpleNamespace(name="actor")
    critic = SimpleNamespace(name="critic")
    agent.actor = SimpleNamespace(module=actor)
    agent.critic = SimpleNamespace(module=critic)

    assert agent.actor_module is actor
    assert agent.critic_module is critic

    agent.actor = actor
    agent.critic = critic

    assert agent.actor_module is actor
    assert agent.critic_module is critic


def test_base_agent_save_writes_rank_and_score_checkpoints_without_real_dist(
    tmp_path,
    monkeypatch,
):
    saved = []
    barriers = []

    def fake_save(state, path):
        saved.append((state, path.name))

    def fake_all_gather_object(target, value):
        target[:] = [value, "other-task"]

    def fake_all_gather(target, value):
        for item in target:
            item.copy_(value)

    monkeypatch.setattr(base_agent_module.torch, "save", fake_save)
    monkeypatch.setattr(
        base_agent_module.dist,
        "all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(base_agent_module.dist, "barrier", lambda: barriers.append(True))
    monkeypatch.setattr(base_agent_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(base_agent_module.dist, "all_gather", fake_all_gather)

    agent = object.__new__(BaseAgent)
    agent.fabric = _FabricRecorder()
    agent.device = torch.device("cpu")
    agent.root_dir = tmp_path
    agent.config = SimpleNamespace(save_inference_checkpoint=True)
    agent.env = SimpleNamespace(
        get_task_id=lambda: "task-a",
        get_state_dict=lambda: {"env": torch.tensor([3.0])},
    )
    agent.best_evaluated_score = 10.0
    model_state = {"w": torch.tensor([1.0])}
    inference_calls = []
    agent.get_state_dict = lambda state: state | {
        "training": torch.tensor([1.0]),
        "model": model_state,
    }

    def fake_get_inference_state_dict(state, model_state_dict=None):
        inference_calls.append(model_state_dict)
        return state | {"inference": torch.tensor([2.0]), "model": model_state_dict}

    agent.get_inference_state_dict = fake_get_inference_state_dict
    agent.inference_checkpoint_name = lambda name: f"custom_{name}"

    BaseAgent.save(agent, checkpoint_name="last.ckpt", new_high_score=True)

    assert barriers == [True]
    assert [name for _, name in saved] == [
        "last.ckpt",
        "custom_last.ckpt",
        "env_task-a.ckpt",
        "score_based.ckpt",
        "custom_score_based.ckpt",
    ]
    assert inference_calls == [model_state]
    assert saved[1][0] is saved[4][0]
    assert agent.fabric.calls == [
        ("on_save_checkpoint_start", (agent,)),
        ("on_save_checkpoint_end", (agent,)),
    ]


def test_post_env_step_modifications_resets_model_done_envs():
    agent = object.__new__(BaseAgent)
    agent.model = _ResetRecorder()
    dones = torch.tensor([False, True, False, True])
    terminated = torch.zeros_like(dones)
    extras = {}

    out_dones, out_terminated, out_extras = BaseAgent.post_env_step_modifications(
        agent,
        dones,
        terminated,
        extras,
    )

    assert out_dones is dones
    assert out_terminated is terminated
    assert out_extras is extras
    assert torch.equal(agent.model.env_ids, torch.tensor([1, 3]))


def test_amp_post_env_step_modifications_resets_model_for_amp_terminations():
    from protomotions.agents.amp.agent import AMP

    agent = _new_amp_agent(AMP)
    agent.model = _ResetRecorder()
    agent.config = SimpleNamespace(
        amp_parameters=SimpleNamespace(
            discriminator_max_cumulative_bad_transitions=2,
        )
    )
    agent.amp_component.num_cumulative_bad_transitions = torch.tensor([0, 2, 3, 1])
    dones = torch.tensor([False, False, True, False])
    terminated = torch.zeros_like(dones)
    extras = {}

    out_dones, out_terminated, out_extras = AMP.post_env_step_modifications(
        agent,
        dones,
        terminated,
        extras,
    )

    assert torch.equal(out_dones, torch.tensor([False, True, True, False]))
    assert torch.equal(out_terminated, torch.tensor([False, True, True, False]))
    assert torch.equal(agent.model.env_ids, torch.tensor([1, 2]))
    assert out_extras is extras
    assert "amp_cumulative_bad_transitions" in extras
    assert "amp_discriminator_termination" in extras


def test_collect_rollout_step_records_registered_model_outputs_only():
    agent = object.__new__(BaseAgent)
    agent.model = _RolloutModel()
    agent.model_output_keys = ["action", "vae_noise"]
    agent.experience_buffer = _ExperienceBufferRecorder()
    obs_td = {"obs": torch.tensor([[1.0], [2.0]])}

    output_td = BaseAgent.collect_rollout_step(agent, obs_td, step=4)

    assert torch.equal(output_td["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("action", 4)],
        torch.tensor([[2.0], [3.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("vae_noise", 4)],
        torch.tensor([[2.0], [2.0]]),
    )
    assert ("not_registered", 4) not in agent.experience_buffer.data


def test_load_parameters_materializes_model_before_strict_state_load():
    agent = object.__new__(BaseAgent)
    agent.model = _LazyLoadModel()
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    state_dict = {
        "epoch": 3,
        "step_count": 7,
        "run_start_time": 11.0,
        "best_evaluated_score": 13.0,
        "model": {
            "linear.weight": torch.tensor([[1.0, 2.0]]),
            "linear.bias": torch.tensor([0.5]),
        },
    }

    BaseAgent.load_parameters(agent, state_dict, load_training_state=False)

    assert agent.current_epoch == 0
    assert agent.step_count == 0
    assert agent.fit_start_time is None
    assert agent.best_evaluated_score is None
    assert agent.model.materialized_from_keys == ["linear.bias", "linear.weight"]
    assert torch.equal(agent.model.linear.weight, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(agent.model.linear.bias, torch.tensor([0.5]))


def test_ppo_load_model_state_preserves_fixed_actor_logstd_override():
    from protomotions.agents.ppo.agent import PPO

    agent = object.__new__(PPO)
    agent.model = _PPOLogstdLoadModel(actor_logstd=-1.25)
    agent.actor = agent.model._actor
    agent.config = SimpleNamespace(
        model=SimpleNamespace(
            actor=SimpleNamespace(learnable_std=False, actor_logstd=-1.25)
        )
    )

    PPO._load_model_state_dict(
        agent,
        {"_actor.logstd": torch.tensor([0.5], dtype=torch.float32)},
    )

    assert agent.model.materialized_from_keys == ["_actor.logstd"]
    assert torch.equal(agent.actor.logstd, torch.tensor([-1.25]))


def test_ppo_load_model_state_restores_learnable_actor_logstd():
    from protomotions.agents.ppo.agent import PPO

    agent = object.__new__(PPO)
    agent.model = _PPOLogstdLoadModel(actor_logstd=-1.25)
    agent.actor = agent.model._actor
    agent.config = SimpleNamespace(
        model=SimpleNamespace(
            actor=SimpleNamespace(learnable_std=True, actor_logstd=-1.25)
        )
    )

    PPO._load_model_state_dict(
        agent,
        {"_actor.logstd": torch.tensor([0.5], dtype=torch.float32)},
    )

    assert agent.model.materialized_from_keys == ["_actor.logstd"]
    assert torch.equal(agent.actor.logstd, torch.tensor([0.5]))


def test_load_parameters_uses_model_state_hook():
    calls = []
    agent = object.__new__(BaseAgent)
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent._load_model_state_dict = lambda model_state: calls.append(
        ("model", model_state)
    )
    agent._after_load_model_state_dict = lambda state: calls.append(("after", state))
    agent._load_training_state = lambda state: calls.append(("training", state))
    state_dict = {"model": {"weight": torch.tensor([1.0])}, "epoch": 3}

    BaseAgent.load_parameters(agent, state_dict, load_training_state=True)

    assert calls == [
        ("model", state_dict["model"]),
        ("after", state_dict),
        ("training", state_dict),
    ]


def test_load_parameters_skips_training_state_hook_for_model_only_load():
    calls = []
    agent = object.__new__(BaseAgent)
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent._load_model_state_dict = lambda model_state: calls.append(
        ("model", model_state)
    )
    agent._after_load_model_state_dict = lambda state: calls.append(("after", state))
    agent._load_training_state = lambda state: calls.append(("training", state))
    state_dict = {"model": {"weight": torch.tensor([1.0])}, "epoch": 3}

    BaseAgent.load_parameters(agent, state_dict, load_training_state=False)

    assert calls == [
        ("model", state_dict["model"]),
        ("after", state_dict),
    ]


def test_base_training_state_load_restores_metadata_norm_and_evaluator():
    loaded = {}
    agent = object.__new__(BaseAgent)
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent.config = SimpleNamespace(normalize_rewards=True)
    agent.running_reward_norm = SimpleNamespace(
        load_state_dict=lambda state: loaded.setdefault("reward_norm", state)
    )
    agent.evaluator = SimpleNamespace(
        load_state_dict=lambda state: loaded.setdefault("evaluator", state)
    )
    state_dict = {
        "epoch": 3,
        "step_count": 7,
        "run_start_time": 11.0,
        "best_evaluated_score": 13.0,
        "running_reward_norm": {"mean": torch.tensor([1.0])},
        "evaluator": {"score": torch.tensor([2.0])},
    }

    BaseAgent._load_training_state(agent, state_dict)

    assert agent.current_epoch == 3
    assert agent.step_count == 7
    assert agent.fit_start_time == 11.0
    assert agent.best_evaluated_score == 13.0
    assert loaded["reward_norm"] == state_dict["running_reward_norm"]
    assert loaded["evaluator"] == state_dict["evaluator"]


def test_amp_model_load_detects_non_amp_warm_start_and_uses_partial_load():
    from protomotions.agents.amp.agent import AMP

    agent = object.__new__(AMP)
    agent.model = _StateDictLoadRecorder()
    agent.actor = SimpleNamespace(logstd=SimpleNamespace(data=torch.tensor([0.0])))
    agent.config = SimpleNamespace(
        model=SimpleNamespace(actor=SimpleNamespace(learnable_std=True))
    )
    state = {"_actor.weight": torch.tensor([1.0])}

    AMP._load_model_state_dict(agent, state)

    assert agent._warm_start_from_non_amp_checkpoint is True
    assert agent.model.loaded_state == state
    assert agent.model.strict is False


def test_amp_non_amp_warm_start_training_state_skips_discriminator_optimizers():
    from protomotions.agents.amp.agent import AMP

    calls = []
    agent = object.__new__(AMP)
    agent._warm_start_from_non_amp_checkpoint = True
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent._load_ppo_training_state = lambda state, require_optimizers: calls.append(
        ("ppo", require_optimizers)
    )
    agent.discriminator_optimizer = SimpleNamespace(
        load_state_dict=lambda state: calls.append(("disc", state))
    )
    state = {"epoch": 5, "discriminator_optimizer": {"disc": 1}}

    AMP._load_training_state(agent, state)

    assert agent.current_epoch == 5
    assert calls == [("ppo", False)]


def test_amp_full_training_state_loads_discriminator_optimizer():
    from protomotions.agents.amp.agent import AMP

    agent = object.__new__(AMP)
    agent._warm_start_from_non_amp_checkpoint = False
    agent.config = SimpleNamespace(
        normalize_rewards=False,
        adaptive_lr=SimpleNamespace(enabled=False),
        advantage_normalization=SimpleNamespace(enabled=False, use_ema=False),
    )
    agent.evaluator = None
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent.actor_optimizer = _OptimizerRecorder()
    agent.critic_optimizer = _OptimizerRecorder()
    component = object.__new__(AMPTrainingComponent)
    component.agent = agent
    component.discriminator_optimizer = _OptimizerRecorder()
    component.use_disc_critic = False
    agent.amp_component = component
    state = {
        "epoch": 5,
        "actor_optimizer": {"actor": 1},
        "critic_optimizer": {"critic": 1},
        "discriminator_optimizer": {"disc": 1},
    }

    AMP._load_training_state(agent, state)

    assert agent.current_epoch == 5
    assert agent.actor_optimizer.loaded_state == {"actor": 1}
    assert agent.critic_optimizer.loaded_state == {"critic": 1}
    assert component.discriminator_optimizer.loaded_state == {"disc": 1}


def test_max_episode_length_manager_interpolates_and_handles_fixed_schedule():
    fixed = MaxEpisodeLengthManagerConfig(
        start_length=7,
        end_length=30,
        transition_epochs=0,
    )
    scheduled = MaxEpisodeLengthManagerConfig(
        start_length=10,
        end_length=30,
        transition_epochs=4,
    )

    assert fixed.current_max_episode_length(current_epoch=999) == 7
    assert scheduled.current_max_episode_length(current_epoch=0) == 10
    assert scheduled.current_max_episode_length(current_epoch=2) == 20
    assert scheduled.current_max_episode_length(current_epoch=8) == 30


def test_base_model_default_contracts_are_safe_for_simple_subclasses():
    model = _ConcreteBaseModel(BaseModelConfig(out_keys=["action"]))
    td = TensorDict({"obs": torch.tensor([[1.0], [2.0]])}, batch_size=2)

    model.reset_rollout_context(num_envs=2, device=torch.device("cpu"))
    model.materialize_from_state_dict({})
    materialized = model.materialize(td.clone())

    assert torch.equal(materialized["action"], torch.tensor([[2.0], [3.0]]))
    assert model.optimization_module() is model
    assert model.rollout_context_keys() == []
    assert model.experience_buffer_keys() == ["action"]
    assert not hasattr(model, "collect_rollout")
    assert not hasattr(model, "collect_expert_rollout")


def test_rollout_state_specs_drive_keys_and_reset_allocation():
    model = _RolloutStateSpecModel()

    assert model.rollout_context_keys() == [
        "bool_state",
        "long_state",
        "float_state",
    ]
    assert model.experience_buffer_keys() == [
        "action",
        "bool_state",
        "long_state",
        "float_state",
    ]

    model.reset_rollout_context(num_envs=2, device=torch.device("cpu"))

    assert model.bool_state.shape == (2,)
    assert model.bool_state.dtype == torch.bool
    assert torch.equal(model.bool_state, torch.zeros(2, dtype=torch.bool))
    assert model.long_state.shape == (2,)
    assert model.long_state.dtype == torch.long
    assert torch.equal(model.long_state, torch.zeros(2, dtype=torch.long))
    assert model.float_state.shape == (2, 2)
    assert model.float_state.dtype == torch.float32
    assert torch.equal(model.float_state, torch.zeros(2, 2))


def test_rollout_state_can_be_read_and_reseeded_without_reallocation():
    model = _RolloutStateSpecModel()
    model.reset_rollout_context(num_envs=3, device=torch.device("cpu"))
    initial_float_state = model.float_state

    model.bool_state[:] = torch.tensor([True, True, False])
    model.long_state[:] = torch.tensor([3, 4, 5])
    model.float_state[:] = torch.tensor(
        [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]],
    )

    td = TensorDict({"obs": torch.ones(2, 1)}, batch_size=2)
    model.read_rollout_state(td)

    assert torch.equal(td["bool_state"], torch.tensor([True, True]))
    assert torch.equal(td["long_state"], torch.tensor([3, 4]))
    assert torch.equal(td["float_state"], torch.tensor([[1.0, 1.5], [2.0, 2.5]]))

    model.reset_rollout_context(env_ids=torch.tensor([1]))

    assert model.float_state is initial_float_state
    assert torch.equal(model.bool_state, torch.tensor([True, False, False]))
    assert torch.equal(model.long_state, torch.tensor([3, 0, 5]))
    assert torch.equal(
        model.float_state,
        torch.tensor([[1.0, 1.5], [0.0, 0.0], [3.0, 3.5]]),
    )


def test_register_model_output_keys_preserves_declared_rollout_state_dtypes():
    agent = object.__new__(BaseAgent)
    agent.model = _RolloutStateSpecModel()
    agent.experience_buffer = ExperienceBuffer(
        num_envs=2,
        num_steps=1,
        device=torch.device("cpu"),
    )
    obs_td = TensorDict(
        {"obs": torch.tensor([[1.0], [2.0]])},
        batch_size=2,
        device=torch.device("cpu"),
    )
    output_td = agent.model(obs_td)

    BaseAgent._register_model_output_keys(agent, output_td)
    for key in agent.model_output_keys:
        agent.experience_buffer.update_data(key, 0, output_td[key].detach())
    data = agent.experience_buffer.make_dict()

    assert data["bool_state"].dtype == torch.bool
    assert torch.equal(data["bool_state"], torch.tensor([True, False]))
    assert data["long_state"].dtype == torch.long
    assert torch.equal(data["long_state"], torch.tensor([0, 1]))
    assert data["float_state"].dtype == torch.float32
    assert data["float_state"].shape == (2, 2)


def test_register_model_output_keys_rejects_rollout_state_spec_mismatch():
    agent = object.__new__(BaseAgent)
    agent.model = _RolloutStateMismatchModel()
    agent.experience_buffer = ExperienceBuffer(
        num_envs=2,
        num_steps=1,
        device=torch.device("cpu"),
    )
    obs_td = TensorDict(
        {"obs": torch.tensor([[1.0], [2.0]])},
        batch_size=2,
        device=torch.device("cpu"),
    )
    output_td = agent.model(obs_td)

    with pytest.raises(ValueError, match="float_state.*expected shape.*observed"):
        BaseAgent._register_model_output_keys(agent, output_td)


def test_base_agent_state_dicts_preserve_training_and_inference_state():
    agent = object.__new__(BaseAgent)
    agent.model = nn.Linear(2, 1)
    agent.current_epoch = 3
    agent.step_count = 11
    agent.fit_start_time = 123.0
    agent.best_evaluated_score = 4.5
    agent.config = SimpleNamespace(normalize_rewards=True)
    agent.running_reward_norm = SimpleNamespace(
        state_dict=lambda: {"mean": torch.tensor([1.0])}
    )
    agent.evaluator = SimpleNamespace(
        get_state_dict=lambda: {"evaluated": torch.tensor(1)}
    )

    state = BaseAgent.get_state_dict(agent, {"existing": True})
    inference = BaseAgent.get_inference_state_dict(agent, {})

    assert state["existing"] is True
    assert state["epoch"] == 3
    assert state["step_count"] == 11
    assert state["running_reward_norm"]["mean"].item() == 1.0
    assert state["evaluator"]["evaluated"].item() == 1
    assert inference["model"]["weight"].device.type == "cpu"
    assert inference["model"]["weight"].data_ptr() != agent.model.weight.data_ptr()
    assert BaseAgent.inference_checkpoint_name("last.ckpt") == "inference_last.ckpt"


def test_base_agent_observation_helpers_and_nan_checks_use_tensordicts():
    agent = object.__new__(BaseAgent)
    agent.device = torch.device("cpu")
    obs = {"obs": torch.ones(2, 3)}

    td = BaseAgent.obs_dict_to_tensordict(agent, obs)

    assert td.batch_size == torch.Size([2])
    assert BaseAgent.add_agent_info_to_obs(agent, obs) is obs
    assert BaseAgent.add_agent_info_to_next_obs(agent, obs) is obs
    BaseAgent.check_for_nans(agent, td)
    with pytest.raises(AssertionError, match="NaN/Inf in obs"):
        BaseAgent.check_for_nans(
            agent,
            TensorDict({"obs": torch.tensor([[float("inf")]])}, batch_size=1),
        )


def test_record_rollout_step_tracks_episode_stats_and_extras():
    agent = object.__new__(BaseAgent)
    agent.current_rewards = torch.tensor([1.0, 2.0, 3.0])
    agent.current_lengths = torch.tensor([2, 3, 4])
    agent.episode_reward_meter = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.episode_length_meter = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.episode_env_tensors = TensorAverageMeterDict(device=torch.device("cpu"))
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent.config = SimpleNamespace(
        normalize_rewards=False,
    )

    BaseAgent.record_rollout_step(
        agent,
        next_obs_td=TensorDict({"obs": torch.zeros(3, 1)}, batch_size=3),
        actions=torch.zeros(3, 1),
        rewards=torch.tensor([1.0, 1.0, 1.0]),
        dones=torch.tensor([False, True, False]),
        terminated=torch.tensor([False, True, False]),
        done_indices=torch.tensor([1]),
        extras={
            "raw/debug": torch.tensor([99.0]),
            "scalar": torch.tensor([2.0]),
            "vector": torch.tensor([1.0, 3.0, 5.0]),
            "text": "ignored",
        },
        step=2,
    )

    assert torch.equal(agent.current_rewards, torch.tensor([2.0, 0.0, 4.0]))
    assert torch.equal(agent.current_lengths, torch.tensor([3, 0, 5]))
    assert torch.equal(
        agent.experience_buffer.data[("dones", 2)],
        torch.tensor([False, True, False]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("rewards", 2)],
        torch.tensor([1.0, 1.0, 1.0]),
    )
    env_metrics = agent.episode_env_tensors.mean_and_clear()
    assert env_metrics["scalar"] == 2.0
    assert env_metrics["vector_mean"] == 3.0


def test_normalize_rewards_in_buffer_keeps_raw_rewards_and_writes_normalized_values():
    agent = object.__new__(BaseAgent)
    agent.config = SimpleNamespace(normalize_rewards=True)
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent.experience_buffer.rewards = torch.tensor([[1.0, 2.0]])
    agent.running_reward_norm = _OffsetNormalizer(offset=10.0)

    BaseAgent.normalize_rewards_in_buffer(agent)

    assert torch.equal(
        agent.experience_buffer.unnormalized_rewards,
        torch.tensor([[1.0, 2.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.rewards,
        torch.tensor([[11.0, 12.0]]),
    )


def test_optimize_model_averages_logs_and_reshuffles_reused_dataset():
    agent = object.__new__(BaseAgent)
    dataset = _OneBatchDataset({"x": torch.tensor([1.0])})
    calls = []
    modes = []
    agent.current_epoch = 0
    agent.experience_buffer = _BatchExperience({"x": torch.tensor([1.0])})
    agent.pre_process_dataset = lambda: calls.append("pre")
    agent.process_dataset = lambda data: dataset
    agent.train = lambda: modes.append("train")
    agent.eval = lambda: modes.append("eval")
    agent.max_num_batches = lambda: 2

    def perform(batch_dict, batch_idx):
        calls.append(("step", batch_idx, batch_dict["x"].clone()))
        return {"loss": torch.tensor(float(batch_idx + 1)), "scalar": 3.0}

    agent.perform_optimization_step = perform

    log_dict = BaseAgent.optimize_model(agent)

    assert calls[0] == "pre"
    assert calls[1][0:2] == ("step", 0)
    assert calls[2][0:2] == ("step", 1)
    assert dataset.shuffle_calls == 1
    assert modes == ["train", "eval"]
    assert torch.equal(log_dict["loss"], torch.tensor(1.5))
    assert log_dict["scalar"] == 3.0


def test_optimize_model_rejects_nan_minibatches_before_training_step():
    agent = object.__new__(BaseAgent)
    agent.current_epoch = 0
    agent.experience_buffer = _BatchExperience({"x": torch.tensor([1.0])})
    agent.pre_process_dataset = lambda: None
    agent.process_dataset = lambda data: _OneBatchDataset(
        {"x": torch.tensor([float("nan")])}
    )
    agent.train = lambda: None
    agent.eval = lambda: None
    agent.max_num_batches = lambda: 1
    agent.perform_optimization_step = lambda batch_dict, batch_idx: {}

    with pytest.raises(ValueError, match="NaN in training"):
        BaseAgent.optimize_model(agent)


def test_post_epoch_logging_aggregates_task_env_and_timing_metrics(monkeypatch):
    logged_aggregate_weights = []

    def fake_aggregate(log_dict, fabric, weight):
        logged_aggregate_weights.append(weight)
        return log_dict

    monkeypatch.setattr(base_agent_module, "aggregate_scalar_metrics", fake_aggregate)
    monkeypatch.setattr(base_agent_module.time, "time", lambda: 110.0)

    agent = object.__new__(BaseAgent)
    agent.fabric = _FabricRecorder()
    agent.num_envs = 2
    agent.num_steps = 3
    agent._total_envs = 4
    agent.step_count = 12
    agent.current_epoch = 5
    agent.epoch_start_time = 100.0
    agent.fit_start_time = 50.0
    agent.last_episode_length = torch.tensor(0.0)
    agent.last_episode_reward = torch.tensor(0.0)
    agent.episode_reward_meter = SimpleNamespace(
        mean_and_clear=lambda: {"episode_reward": torch.tensor(8.0)}
    )
    agent.episode_length_meter = SimpleNamespace(
        mean_and_clear=lambda: {"episode_length": torch.tensor(4.0)}
    )
    agent.episode_env_tensors = SimpleNamespace(
        mean_and_clear=lambda: {"balance": torch.tensor(0.5)}
    )
    agent.experience_buffer = SimpleNamespace(
        rewards=torch.tensor([1.0, 3.0]),
        unnormalized_rewards=torch.tensor([5.0, 7.0]),
    )
    agent.config = SimpleNamespace(normalize_rewards=True)
    agent.running_reward_norm = SimpleNamespace(var=torch.tensor([9.0]))

    BaseAgent.post_epoch_logging(agent, {"loss": torch.tensor(0.25)})

    log_dict, step = agent.fabric.logged[0]
    assert step == 5
    assert logged_aggregate_weights == [2]
    assert log_dict["info/episode_length"] == 4.0
    assert log_dict["times/fps_last_epoch"] == pytest.approx(1.2)
    assert log_dict["rewards/task_rewards"] == 2.0
    assert log_dict["rewards/unnormalized_task_rewards"] == 6.0
    assert log_dict["env/balance"] == 0.5
    assert log_dict["loss"] == torch.tensor(0.25)
    assert torch.equal(agent.last_episode_length, torch.tensor(4.0))
    assert torch.equal(agent.last_episode_reward, torch.tensor(8.0))


def test_post_epoch_logging_reuses_last_episode_stats_when_no_episode_finishes(
    monkeypatch,
):
    monkeypatch.setattr(
        base_agent_module,
        "aggregate_scalar_metrics",
        lambda d, *_args, **_kwargs: d,
    )
    monkeypatch.setattr(base_agent_module.time, "time", lambda: 110.0)

    agent = object.__new__(BaseAgent)
    agent.fabric = _FabricRecorder()
    agent.num_envs = 2
    agent.num_steps = 3
    agent._total_envs = 4
    agent.step_count = 12
    agent.current_epoch = 5
    agent.epoch_start_time = 100.0
    agent.fit_start_time = 50.0
    agent.last_episode_length = torch.tensor(255.0)
    agent.last_episode_reward = torch.tensor(190.0)
    agent.episode_reward_meter = SimpleNamespace(mean_and_clear=lambda: {})
    agent.episode_length_meter = SimpleNamespace(mean_and_clear=lambda: {})
    agent.episode_env_tensors = SimpleNamespace(mean_and_clear=lambda: {})
    agent.experience_buffer = SimpleNamespace(rewards=torch.tensor([1.0, 3.0]))
    agent.config = SimpleNamespace(normalize_rewards=False)

    BaseAgent.post_epoch_logging(agent, {})

    log_dict, _ = agent.fabric.logged[0]
    assert log_dict["info/episode_length"].item() == 255.0
    assert log_dict["info/episode_reward"].item() == 190.0


def test_base_agent_noop_hooks_dataset_helpers_and_mode_switching():
    agent = object.__new__(BaseAgent)
    agent.config = SimpleNamespace(batch_size=2)
    agent._total_envs = 6
    agent.num_envs = 2
    agent.num_steps = 3
    agent.num_mini_epochs = 2
    agent.model = nn.Linear(1, 1)

    BaseAgent.pre_collect_step(agent, step=0)
    BaseAgent.pre_process_dataset(agent)
    BaseAgent.register_algorithm_experience_buffer_keys(agent)
    BaseAgent.register_algorithm_experience_buffer_keys_from_obs(
        agent,
        TensorDict({"obs": torch.ones(1, 1)}, batch_size=1),
    )
    BaseAgent.train(agent)
    assert agent.model.training is True
    BaseAgent.eval(agent)
    assert agent.model.training is False
    assert BaseAgent.get_step_count_increment(agent) == 6
    assert BaseAgent.max_num_batches(agent) == 6

    dataset = BaseAgent.process_dataset(
        agent,
        {"obs": torch.arange(4, dtype=torch.float).view(4, 1)},
    )
    sample = dataset[0]

    assert sample["obs"].shape == (2, 1)
    BaseAgent.terminate_early(agent)
    assert agent._should_stop is True
