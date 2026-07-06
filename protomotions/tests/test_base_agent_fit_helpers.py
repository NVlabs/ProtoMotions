# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight lifecycle tests for BaseAgent.fit without simulator startup."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from protomotions.agents.base_agent import agent as base_agent_module
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.utils.metering import TensorAverageMeterDict


class _FitExperienceBuffer:
    def __init__(self, num_envs: int, num_steps: int, device: torch.device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.registered = []
        self.updated = []
        self.make_dict_calls = 0

    def register_key(self, key: str, shape=(), dtype=torch.float):
        self.registered.append((key, shape, dtype))
        setattr(
            self,
            key,
            torch.zeros(
                (self.num_steps, self.num_envs) + shape,
                dtype=dtype,
                device=self.device,
            ),
        )

    def update_data(self, key: str, step: int, value: torch.Tensor):
        self.updated.append((key, step, value.detach().clone()))
        getattr(self, key)[step] = value

    def make_dict(self):
        self.make_dict_calls += 1
        return {
            key: getattr(self, key)
            .transpose(0, 1)
            .reshape(self.num_envs * self.num_steps, *shape)
            for key, shape, _ in self.registered
        }


class _FitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_calls = []
        self.call_modes = []

    def forward(self, obs_td: TensorDict) -> TensorDict:
        self.call_modes.append(self.training)
        return TensorDict(
            {
                "action": torch.cat([obs_td["obs"], obs_td["obs"] + 10.0], dim=-1),
                "value": obs_td["obs"].squeeze(-1) + 0.5,
            },
            batch_size=obs_td.batch_size,
            device=obs_td.device,
        )

    def experience_buffer_keys(self):
        return ["action", "value"]

    def reset_rollout_context(self, env_ids=None):
        self.reset_calls.append(
            None if env_ids is None else env_ids.detach().clone()
        )


class _MixedModeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(1, 1),
        )


class _FitEnv:
    def __init__(self):
        self.num_envs = 2
        self.reset_args = []
        self.step_actions = []
        self.epoch_ends = []
        self.progress_buf = torch.tensor([4, 5])
        self.max_episode_length = 3

    def reset(self, done_indices=None):
        self.reset_args.append(
            None if done_indices is None else done_indices.detach().clone()
        )
        obs_value = float(len(self.reset_args))
        return {"obs": torch.full((self.num_envs, 1), obs_value)}, {}

    def step(self, action):
        self.step_actions.append(action.detach().clone())
        step_idx = len(self.step_actions)
        dones = torch.tensor([False, step_idx == 1])
        terminated = torch.tensor([False, step_idx == 1])
        return (
            {"obs": torch.full((self.num_envs, 1), 100.0 + step_idx)},
            torch.tensor([1.0, 2.0]),
            dones,
            terminated,
            {"metric": torch.tensor([float(step_idx), float(step_idx + 2)])},
        )

    def on_epoch_end(self, epoch):
        self.epoch_ends.append(epoch)


class _Fabric:
    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, len(args)))

    def broadcast(self, value):
        return value


class _Evaluator:
    config = SimpleNamespace(eval_metrics_every=1)

    def __init__(self, calls):
        self.calls = calls

    def evaluate(self):
        self.calls.append(("evaluate",))
        return {"eval/success": torch.tensor(0.75)}, torch.tensor(3.0), 4


class _MaxEpisodeLengthManager:
    def __init__(self, calls):
        self.calls = calls

    def current_max_episode_length(self, current_epoch: int):
        self.calls.append(("max_episode_length", current_epoch))
        return 20 + current_epoch


def _fit_agent(calls, buffer, *, max_epochs=1):
    agent = object.__new__(BaseAgent)
    agent.num_envs = 2
    agent.num_steps = 2
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.max_epochs = max_epochs
    agent.fit_start_time = None
    agent.step_count = 0
    agent._skip_next_policy_update = False
    agent.just_loaded_checkpoint_should_evaluate = False
    agent._skip_next_eval_after_resume = False
    agent.best_evaluated_score = None
    agent._should_stop = False
    agent.fabric = _Fabric()
    agent.env = _FitEnv()
    agent.model = _FitModel()
    agent.evaluator = _Evaluator(calls)
    agent.config = SimpleNamespace(
        normalize_rewards=False,
        save_epoch_checkpoint_every=1,
        save_last_checkpoint_every=1,
        max_episode_length_manager=_MaxEpisodeLengthManager(calls),
    )
    agent.time_report = SimpleNamespace(
        report=lambda: calls.append(("time_report",))
    )
    agent.episode_reward_meter = TensorAverageMeterDict(device=agent.device)
    agent.episode_length_meter = TensorAverageMeterDict(device=agent.device)
    agent.episode_env_tensors = TensorAverageMeterDict(device=agent.device)
    agent.current_rewards = torch.zeros(agent.num_envs)
    agent.current_lengths = torch.zeros(agent.num_envs, dtype=torch.long)
    agent.experience_buffer = buffer
    agent.optimize_model = lambda: calls.append(("optimize",)) or {
        "train/loss": 1.25
    }
    agent.save = lambda checkpoint_name, new_high_score=False: calls.append(
        ("save", checkpoint_name, new_high_score)
    )
    agent.post_epoch_logging = lambda log_dict: calls.append(
        ("post_log", dict(log_dict))
    )
    agent.get_step_count_increment = lambda: 7
    return agent


def test_base_agent_evaluates_on_default_cadence():
    calls = []
    agent = _fit_agent(
        calls,
        _FitExperienceBuffer(num_envs=2, num_steps=2, device=torch.device("cpu")),
    )
    agent.current_epoch = 10225
    agent.evaluator.config.eval_metrics_every = 25

    assert BaseAgent._should_evaluate_this_epoch(agent) is True
    assert agent.just_loaded_checkpoint_should_evaluate is False
    assert agent._skip_next_eval_after_resume is False


def test_base_agent_skip_resume_eval_defers_next_cadence_eval_once():
    calls = []
    agent = _fit_agent(
        calls,
        _FitExperienceBuffer(num_envs=2, num_steps=2, device=torch.device("cpu")),
    )
    agent.current_epoch = 10225
    agent.evaluator.config.eval_metrics_every = 25
    agent._skip_next_eval_after_resume = True

    assert BaseAgent._should_evaluate_this_epoch(agent) is False
    assert agent.just_loaded_checkpoint_should_evaluate is False
    assert agent._skip_next_eval_after_resume is False

    assert BaseAgent._should_evaluate_this_epoch(agent) is True


def test_eval_model_for_buffer_registration_restores_nested_module_modes():
    agent = object.__new__(BaseAgent)
    agent.model = _MixedModeModel()

    agent.model.train()
    agent.model.encoder.eval()
    agent.model.head[0].train()
    agent.model.head[1].eval()

    before_modes = {
        name: module.training for name, module in agent.model.named_modules()
    }

    with BaseAgent._eval_model_for_buffer_registration(agent):
        assert all(not module.training for module in agent.model.modules())

    after_modes = {
        name: module.training for name, module in agent.model.named_modules()
    }
    assert after_modes == before_modes


def test_base_agent_fit_registers_buffers_collects_rollout_evaluates_and_saves(
    monkeypatch,
):
    calls = []
    buffers = []

    def make_buffer(*args, **kwargs):
        buffer = _FitExperienceBuffer(*args, **kwargs)
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(base_agent_module, "ExperienceBuffer", make_buffer)
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

    agent = _fit_agent(calls, buffer=None)

    BaseAgent.fit(agent)

    buffer = buffers[0]
    assert buffer.registered == [
        ("obs", torch.Size([1]), torch.float32),
        ("action", torch.Size([2]), torch.float32),
        ("value", (), torch.float32),
        ("rewards", (), torch.float),
        ("dones", (), torch.long),
    ]
    assert [key for key, _, _ in buffer.updated] == [
        "obs",
        "action",
        "value",
        "dones",
        "rewards",
        "obs",
        "action",
        "value",
        "dones",
        "rewards",
    ]
    assert agent.env.reset_args[0] is None
    assert torch.equal(agent.env.reset_args[1], torch.tensor([0, 1]))
    assert torch.equal(agent.env.reset_args[2], torch.tensor([1]))
    assert torch.equal(agent.model.reset_calls[0], torch.tensor([1]))
    assert torch.equal(
        agent.model.reset_calls[1],
        torch.tensor([], dtype=torch.long),
    )
    assert agent.step_count == 14
    assert agent.current_epoch == 1
    assert agent.best_evaluated_score == torch.tensor(3.0)
    assert agent._skip_next_policy_update is True
    assert agent.env.max_episode_length == 21
    assert ("optimize",) in calls
    assert ("evaluate",) in calls
    assert (
        "post_log",
        {"train/loss": 1.25, "epoch": 0, "eval/success": torch.tensor(0.75)},
    ) in calls
    assert ("save", "epoch_1.ckpt", False) in calls
    assert ("save", "last.ckpt", False) in calls
    assert ("save", "last.ckpt", True) in calls
    assert calls[-2:] == [("time_report",), ("save", "last.ckpt", False)]
    assert agent.fabric.calls[-1] == ("on_fit_end", 1)
    assert agent.model.call_modes[0] is False


def test_base_agent_fit_skip_update_branch_preprocesses_and_stops_cleanly(
    monkeypatch,
):
    calls = []
    buffers = []

    def make_buffer(*args, **kwargs):
        buffer = _FitExperienceBuffer(*args, **kwargs)
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(base_agent_module, "ExperienceBuffer", make_buffer)
    monkeypatch.setattr(
        base_agent_module,
        "track",
        lambda iterable, description=None: iterable,
    )

    agent = _fit_agent(calls, buffer=None, max_epochs=2)
    agent.evaluator = None
    agent.config.save_epoch_checkpoint_every = None
    agent.config.save_last_checkpoint_every = 10
    agent.config.max_episode_length_manager = None
    agent.fit_start_time = 1.0
    agent._skip_next_policy_update = True
    agent._should_stop = True
    agent.pre_process_dataset = lambda: calls.append(("pre_process",))
    agent.optimize_model = lambda: calls.append(("unexpected_optimize",))

    BaseAgent.fit(agent)

    assert buffers[0].make_dict_calls == 1
    assert ("pre_process",) in calls
    assert ("unexpected_optimize",) not in calls
    assert ("post_log", {"skipped_policy_update": 1.0, "epoch": 0}) in calls
    assert agent.fabric.calls[-1] == ("on_training_stop", 1)
    assert calls[-1] == ("save", "last.ckpt", False)
    assert ("time_report",) not in calls
