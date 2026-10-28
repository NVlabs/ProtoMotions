# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for supervised rollout dispatch without live environments."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent import agent as base_agent_module
from protomotions.agents.supervised import agent as supervised_agent_module
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.supervised.config import SupervisedAgentConfig
from protomotions.agents.utils.normalization import RunningMeanStd
from protomotions.components.motion_lib import MotionFileSwitchMode
from protomotions.envs.base_env.config import EnvConfig


class _StudentModel(nn.Module):
    out_keys = ["action", "target_tokens"]
    deployable_action_key = "action"

    def __init__(self):
        super().__init__()
        self.call_modes = []

    def forward(self, obs_td):
        self.call_modes.append(self.training)
        obs_td["action"] = obs_td["obs"] + 1.0
        obs_td["target_tokens"] = obs_td["obs"].long() + 2
        return obs_td

    def collect_expert_rollout(self, obs_td):
        self.call_modes.append(self.training)
        obs_td["action"] = obs_td["obs"] + 10.0
        obs_td["mean_action"] = obs_td["action"]
        obs_td["target_tokens"] = obs_td["obs"].long() + 20
        return obs_td


class _PrivilegedStudentModel(_StudentModel):
    out_keys = ["action", "privileged_action", "target_tokens"]

    def forward(self, obs_td):
        self.call_modes.append(self.training)
        obs_td["action"] = obs_td["obs"] + 1.0
        obs_td["privileged_action"] = obs_td["obs"] + 50.0
        obs_td["target_tokens"] = obs_td["obs"].long() + 2
        return obs_td


class _ExternalExpert:
    in_keys = ["expert_obs"]
    deployable_action_key = "mean_action"

    def forward_deployable(self, obs_td):
        obs_td["mean_action"] = obs_td["expert_obs"] + 100.0
        return obs_td


class _ActionOnlyExternalActor:
    in_keys = ["actor_obs"]
    deployable_action_key = "action"

    def __init__(self):
        self.calls = []

    def forward_deployable(self, obs_td):
        self.calls.append(list(obs_td.keys()))
        obs_td["action"] = obs_td["actor_obs"] + 200.0
        return obs_td


class _ActorCriticExternalExpert:
    in_keys = ["actor_obs", "critic_obs"]

    def __init__(self):
        self._actor = _ActionOnlyExternalActor()

    def __call__(self, obs_td):
        raise AssertionError("external expert collection should call the actor only")


class _CreatedActorCriticExpertActor(nn.Module):
    in_keys = ["obs"]
    deployable_action_key = "action"

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.forward_training_modes = []

    def forward_deployable(self, tensordict):
        self.forward_training_modes.append(self.training)
        tensordict["action"] = self.linear(tensordict["obs"])
        return tensordict


class _CreatedActorCriticExpertModel(nn.Module):
    in_keys = ["obs", "critic_obs"]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._actor = _CreatedActorCriticExpertActor(config)
        self._critic = nn.Linear(1, 1, bias=False)
        self.obs_norm = RunningMeanStd(
            fabric=None,
            shape=(1,),
            device="cpu",
        )
        self.reset_calls = []

    def reset_rollout_context(self, num_envs, device):
        self.reset_calls.append((num_envs, device))

    def forward(self, tensordict):
        raise AssertionError("external expert materialization should call actor only")


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(2.0)

    def optimization_module(self):
        return self


class _ExperienceBufferRecorder:
    def __init__(self):
        self.data = {}
        self.registered = []

    def update_data(self, key, step, value):
        self.data[(key, step)] = value.clone()

    def register_key(self, key, shape=(), dtype=None):
        self.registered.append((key, shape, dtype))


class _OptimizerRecorder:
    def __init__(self):
        self.zero_grad_calls = []
        self.steps = 0
        self.loaded_state = None

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls.append(set_to_none)

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.loaded_state = state


class _FabricRecorder:
    def __init__(self):
        self.backward_losses = []

    def backward(self, loss):
        self.backward_losses.append(loss.detach().clone())
        loss.backward()


class _CreatedStudentModel(nn.Module):
    skip_default_weight_init = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(1, 1, bias=False)


class _CreatedExpertModel(nn.Module):
    in_keys = ["obs"]
    deployable_action_key = "mean_action"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(1, 1, bias=False)
        self.obs_norm = RunningMeanStd(
            fabric=None,
            shape=(1,),
            device="cpu",
        )
        self.reset_calls = []
        self.forward_training_modes = []

    def reset_rollout_context(self, num_envs, device):
        self.reset_calls.append((num_envs, device))

    def forward_deployable(self, tensordict):
        self.forward_training_modes.append(self.training)
        tensordict["mean_action"] = self.linear(tensordict["obs"])
        return tensordict

    def materialize_from_state_dict(self, state_dict):
        pass


class _CheckpointLinear(nn.Linear):
    def materialize_from_state_dict(self, state_dict):
        pass


def _agent(rollout_actor, expert_model=None, expert_rollout_probability=0.0):
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        rollout_actor=rollout_actor,
        loss=SimpleNamespace(target_key="target_tokens"),
        expert_rollout_probability=expert_rollout_probability,
    )
    agent.device = torch.device("cpu")
    agent.model = _StudentModel()
    agent.expert_model = expert_model
    if expert_model is not None:
        agent.expert_actor = getattr(expert_model, "_actor", expert_model)
        agent.expert_actor_in_keys = list(getattr(agent.expert_actor, "in_keys", []))
        agent._expert_episode_mask = torch.zeros(2, dtype=torch.bool)
    return agent


def _obs_td():
    return TensorDict(
        {
            "obs": torch.tensor([[1.0], [2.0]]),
            "expert_expert_obs": torch.tensor([[5.0], [6.0]]),
        },
        batch_size=2,
    )


@pytest.mark.parametrize("rollout_actor", [RolloutActor.STUDENT, RolloutActor.EXPERT])
def test_posterior_evaluator_ignores_training_episode_actor(rollout_actor):
    from protomotions.agents.evaluators.config import MaskedMimicActionSource
    from protomotions.agents.evaluators.masked_mimic_evaluator import MaskedMimicEvaluator

    agent = _agent(rollout_actor, _ExternalExpert())
    agent.model = _PrivilegedStudentModel()
    obs = _obs_td()
    obs["mean_action"] = torch.full((2, 1), -100.0)
    evaluator = object.__new__(MaskedMimicEvaluator)
    evaluator.agent = agent
    evaluator.config = SimpleNamespace(action_source=MaskedMimicActionSource.POSTERIOR)
    with torch.no_grad():
        action = evaluator._policy_action(obs)
    torch.testing.assert_close(action, torch.tensor([[51.0], [52.0]]))
    assert not action.requires_grad


def test_generic_evaluator_uses_model_action():
    from protomotions.agents.evaluators.base_evaluator import BaseEvaluator

    agent = _agent(RolloutActor.STUDENT)
    agent.model.forward_deployable = agent.model.forward
    evaluator = object.__new__(BaseEvaluator)
    evaluator.agent = agent
    torch.testing.assert_close(evaluator._policy_action(_obs_td()), torch.tensor([[2.0], [3.0]]))


@pytest.mark.parametrize("rollout_actor", [RolloutActor.STUDENT, RolloutActor.EXPERT])
def test_create_model_allows_model_owned_supervision_without_external_expert(monkeypatch, rollout_actor):
    monkeypatch.setattr(
        supervised_agent_module,
        "get_class",
        lambda target: _CreatedStudentModel,
    )

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(_target_="student.Target"),
        expert_model_path=None,
        rollout_actor=rollout_actor,
    )

    SupervisedAgent._before_create_model(agent)
    model = SupervisedAgent.create_model(agent)

    assert isinstance(model, _CreatedStudentModel)
    assert agent.expert_model is None
    assert not hasattr(agent, "_expert_episode_mask")


def test_config_rejects_teacher_probability_without_external_expert():
    with pytest.raises(ValueError, match="requires expert_model_path"):
        SupervisedAgentConfig(
            batch_size=2, training_max_steps=32, expert_rollout_probability=0.5
        )


def test_config_rejects_string_rollout_actor():
    with pytest.raises(TypeError, match="RolloutActor"):
        SupervisedAgentConfig(batch_size=2, training_max_steps=32, rollout_actor="student")


def test_before_create_model_allocates_episode_mask_for_external_expert():
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        expert_model_path="expert.pt", expert_rollout_probability=1.0
    )
    agent.device = torch.device("cpu")
    agent.num_envs = 4

    SupervisedAgent._before_create_model(agent)

    assert agent.expert_model is None
    assert agent._expert_episode_mask.tolist() == [True] * 4


def _external_expert_agent(checkpoint_path, monkeypatch, expert_target, expert_model_config):
    def get_class(target):
        if target == "student.Target":
            return _CreatedStudentModel
        if target == expert_target:
            return {"expert.Target": _CreatedExpertModel,
                    "expert.ActorCritic": _CreatedActorCriticExpertModel}[target]
        raise AssertionError(f"Unexpected target: {target}")

    monkeypatch.setattr(supervised_agent_module, "get_class", get_class)
    monkeypatch.setattr(
        supervised_agent_module,
        "load_experiment_configs_from_checkpoint",
        lambda path: SimpleNamespace(
            robot="expert-robot",
            env=EnvConfig(),
            agent=SimpleNamespace(model=expert_model_config),
        ),
    )
    contract_calls = []
    monkeypatch.setattr(
        supervised_agent_module,
        "validate_expert_action_contract",
        lambda *args: contract_calls.append(args),
    )

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(_target_="student.Target"),
        expert_model_path=str(checkpoint_path),
        rollout_actor=RolloutActor.STUDENT,
    )
    agent.device = torch.device("cpu")
    agent.num_envs = 3
    agent.fabric = object()
    agent.env = SimpleNamespace(
        get_obs=lambda: {"expert_obs": torch.ones(3, 1)},
        robot_config="student-robot",
        config=EnvConfig(),
    )
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(obs, batch_size=3)
    return agent, contract_calls


def test_create_model_loads_external_expert_and_freezes_it(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "expert.pt"
    torch.save(
        {"model": _CreatedExpertModel(SimpleNamespace()).state_dict()},
        checkpoint_path,
    )
    agent, contract_calls = _external_expert_agent(
        checkpoint_path,
        monkeypatch,
        "expert.Target",
        SimpleNamespace(_target_="expert.Target", in_keys=["obs"]),
    )

    model = SupervisedAgent.create_model(agent)

    assert isinstance(model, _CreatedStudentModel)
    assert contract_calls == [
        ("expert-robot", EnvConfig(), "student-robot", EnvConfig())
    ]
    assert isinstance(agent.expert_model, _CreatedExpertModel)
    assert agent.expert_actor is agent.expert_model
    assert agent.expert_actor_in_keys == ["obs"]
    assert agent.expert_model.reset_calls == [(3, torch.device("cpu"))]
    assert agent.expert_model.obs_norm.fabric is agent.fabric
    assert agent.expert_model.forward_training_modes == [False]
    assert agent.expert_model.training is False
    assert all(not param.requires_grad for param in agent.expert_model.parameters())


def test_create_model_materializes_external_expert_actor_only(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "expert_actor_critic.pt"
    expert_state = _CreatedActorCriticExpertModel(SimpleNamespace()).state_dict()
    expert_state["_actor.linear.weight"].fill_(3.0)
    torch.save({"model": expert_state}, checkpoint_path)
    agent, _ = _external_expert_agent(
        checkpoint_path,
        monkeypatch,
        "expert.ActorCritic",
        SimpleNamespace(
            _target_="expert.ActorCritic", actor=SimpleNamespace(in_keys=["obs"])
        ),
    )

    SupervisedAgent.create_model(agent)

    assert agent.expert_actor is agent.expert_model._actor
    assert torch.equal(agent.expert_actor.linear.weight, torch.tensor([[3.0]]))
    assert agent.expert_model.reset_calls == [(3, torch.device("cpu"))]
    assert agent.expert_model._actor.forward_training_modes == [False]
    assert agent.expert_model.training is False
    assert all(not param.requires_grad for param in agent.expert_model.parameters())


def test_build_expert_obs_td_requires_prefixed_expert_observations():
    agent = _agent(RolloutActor.STUDENT)
    obs_td = _obs_td()

    expert_td = SupervisedAgent._build_expert_obs_td(
        agent,
        obs_td,
        expert_in_keys=["expert_obs"],
    )

    assert torch.equal(expert_td["expert_obs"], torch.tensor([[5.0], [6.0]]))

    with pytest.raises(KeyError) as error:
        SupervisedAgent._build_expert_obs_td(agent, obs_td, expert_in_keys=["missing"])

    assert "expert_missing" in str(error.value)


def test_collect_rollout_output_uses_student_or_model_owned_expert_by_enum():
    obs_td = _obs_td()

    student_agent = _agent(RolloutActor.STUDENT)
    student_out = SupervisedAgent._collect_rollout_output(student_agent, obs_td.clone())
    assert torch.equal(student_out["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(student_out["target_tokens"], torch.tensor([[3], [4]]))

    expert_agent = _agent(RolloutActor.EXPERT)
    expert_out = SupervisedAgent._collect_rollout_output(expert_agent, obs_td.clone())
    assert torch.equal(expert_out["action"], torch.tensor([[11.0], [12.0]]))
    assert torch.equal(expert_out["mean_action"], torch.tensor([[11.0], [12.0]]))
    assert torch.equal(expert_out["target_tokens"], torch.tensor([[21], [22]]))


def test_external_expert_labels_every_state_and_drives_selected_episodes():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ExternalExpert())
    agent.model_output_keys = agent.model.out_keys
    agent.experience_buffer = _ExperienceBufferRecorder()

    output = SupervisedAgent._collect_rollout_output(agent, _obs_td())
    assert torch.equal(output["expert_actions"], torch.tensor([[105.0], [106.0]]))
    assert torch.equal(output["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(output["target_tokens"], torch.tensor([[3], [4]]))

    agent._expert_episode_mask[:] = True
    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=0)
    assert torch.equal(output["action"], output["expert_actions"])


def test_external_expert_collection_uses_actor_only_inputs_and_declared_action_key():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ActorCriticExternalExpert())
    obs_td = TensorDict(
        {
            "obs": torch.tensor([[1.0], [2.0]]),
            "expert_actor_obs": torch.tensor([[5.0], [6.0]]),
        },
        batch_size=2,
    )

    action = SupervisedAgent._collect_external_expert_action(agent, obs_td)

    # The collected action is read from the key the actor declares, not a
    # hard-coded "action"/"mean_action" fallback: this actor declares "action".
    assert agent.expert_actor.deployable_action_key == "action"
    assert torch.equal(action, torch.tensor([[205.0], [206.0]]))
    assert agent.expert_model._actor.calls == [["actor_obs"]]


def test_require_deployable_expert_refuses_plain_module():
    from protomotions.agents.supervised.agent import _require_deployable_expert

    class _PlainExpert(nn.Module):
        def forward(self, tensordict):
            return tensordict

    with pytest.raises(TypeError) as error:
        _require_deployable_expert(_PlainExpert())

    message = str(error.value)
    assert "forward_deployable" in message
    assert "deployable_action_key" in message
    assert "adapter" in message


def test_collect_rollout_step_records_configured_supervision_target_key():
    agent = _agent(RolloutActor.STUDENT)
    agent.model_output_keys = ["action"]
    agent.experience_buffer = _ExperienceBufferRecorder()

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=7)

    assert torch.equal(output["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("action", 7)],
        torch.tensor([[2.0], [3.0]]),
    )
    assert torch.equal(
        agent.experience_buffer.data[("target_tokens", 7)],
        torch.tensor([[3], [4]]),
    )


def test_registers_external_expert_actions_and_model_generated_target_key():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ExternalExpert())
    agent.env = SimpleNamespace(
        robot_config=SimpleNamespace(number_of_actions=3),
    )
    agent.experience_buffer = _ExperienceBufferRecorder()

    SupervisedAgent.register_algorithm_experience_buffer_keys(agent)
    SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(agent, _obs_td())

    assert agent.experience_buffer.registered == [
        ("expert_actions", (3,), None),
        ("target_tokens", torch.Size([1]), torch.int64),
    ]
    assert agent.model.call_modes == [False]


def test_register_target_key_short_circuits_existing_buffer_key():
    agent = _agent(RolloutActor.STUDENT)
    agent.experience_buffer = SimpleNamespace(target_tokens=torch.zeros(1))

    SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(agent, _obs_td())


def test_register_target_key_reports_missing_model_output():
    agent = _agent(RolloutActor.STUDENT)
    agent.config.loss.target_key = "missing_target"
    agent.experience_buffer = _ExperienceBufferRecorder()

    with pytest.raises(KeyError, match="missing_target"):
        SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(
            agent, _obs_td()
        )
    assert agent.model.call_modes == [False]


def test_register_target_key_names_the_missing_expert_flag():
    """A recipe whose loss targets the external expert's labels (MaskedMimic),
    run without --expert-model-path, is refused by name before anything is
    collected or registered."""
    agent = _agent(RolloutActor.STUDENT)
    agent.config.loss.target_key = "expert_actions"
    agent.experience_buffer = _ExperienceBufferRecorder()

    with pytest.raises(ValueError, match="--expert-model-path"):
        SupervisedAgent.register_algorithm_experience_buffer_keys_from_obs(
            agent, _obs_td()
        )
    assert agent.model.call_modes == []
    assert agent.experience_buffer.registered == []


def test_collect_rollout_step_prefers_privileged_action_for_student_training():
    agent = _agent(RolloutActor.STUDENT)
    agent.model = _PrivilegedStudentModel()
    agent.model_output_keys = agent.model.out_keys
    agent.experience_buffer = _ExperienceBufferRecorder()

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=9)

    assert torch.equal(output["action"], torch.tensor([[51.0], [52.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("privileged_action", 9)],
        torch.tensor([[51.0], [52.0]]),
    )


def test_mixed_rollouts_preserve_posterior_labels_and_hold_episode_choices():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ExternalExpert())
    agent.model = _PrivilegedStudentModel()
    agent.model_output_keys = agent.model.out_keys
    agent.experience_buffer = _ExperienceBufferRecorder()
    agent._expert_episode_mask = torch.tensor([True, False])
    for step in range(3):
        out = agent.collect_rollout_step(_obs_td(), step)
        torch.testing.assert_close(out["action"], torch.tensor([[105.0], [52.0]]))
        torch.testing.assert_close(
            out["privileged_action"], torch.tensor([[51.0], [52.0]])
        )
        torch.testing.assert_close(
            agent.experience_buffer.data[("action", step)], out["action"]
        )
        torch.testing.assert_close(
            agent.experience_buffer.data[("expert_actions", step)],
            torch.tensor([[105.0], [106.0]]),
        )
    assert agent._expert_episode_mask.tolist() == [True, False]


def test_mixed_reset_resamples_only_done_envs_and_resets_both_models():
    from unittest.mock import Mock

    agent = _agent(
        RolloutActor.STUDENT,
        expert_model=Mock(spec=["reset_rollout_context"]),
        expert_rollout_probability=1.0,
    )
    agent.model = Mock()
    agent._expert_episode_mask = torch.tensor([False, False])
    dones = torch.tensor([False, True])
    returned = agent.post_env_step_modifications(dones, dones.clone(), {})
    torch.testing.assert_close(returned[0], dones)
    assert agent._expert_episode_mask.tolist() == [False, True]
    for model in (agent.model, agent.expert_model):
        assert model.reset_rollout_context.call_args.kwargs["env_ids"].tolist() == [1]
    agent.config.expert_rollout_probability = 0.0
    agent._resample_episode_actor(torch.tensor([1]))
    assert not agent._expert_episode_mask.any()


def test_mixed_sampling_frequency_and_empty_reset():
    agent = _agent(RolloutActor.STUDENT, expert_rollout_probability=0.1)
    agent._expert_episode_mask = torch.zeros(10000, dtype=torch.bool)
    torch.manual_seed(5)
    agent._resample_episode_actor(torch.arange(10000))
    assert 0.08 < agent._expert_episode_mask.float().mean() < 0.12
    before = agent._expert_episode_mask.clone()
    agent._resample_episode_actor(torch.empty(0, dtype=torch.long))
    assert torch.equal(before, agent._expert_episode_mask)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_mixed_probability_is_validated(probability):
    with pytest.raises(ValueError, match="probability"):
        SupervisedAgentConfig(
            batch_size=2, training_max_steps=32, expert_rollout_probability=probability
        )


def test_collect_rollout_step_stores_external_expert_actions_when_not_target_key():
    agent = _agent(RolloutActor.STUDENT, expert_model=_ExternalExpert())
    agent.model_output_keys = ["action", "target_tokens"]
    agent.experience_buffer = _ExperienceBufferRecorder()

    output = SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=4)

    assert torch.equal(output["action"], torch.tensor([[2.0], [3.0]]))
    assert torch.equal(
        agent.experience_buffer.data[("expert_actions", 4)],
        torch.tensor([[105.0], [106.0]]),
    )


def test_collect_rollout_step_reports_missing_required_target_key():
    agent = _agent(RolloutActor.STUDENT)
    agent.config.loss.target_key = "missing_target"
    agent.model_output_keys = ["action"]
    agent.experience_buffer = _ExperienceBufferRecorder()

    with pytest.raises(KeyError, match="missing_target"):
        SupervisedAgent.collect_rollout_step(agent, _obs_td(), step=0)


def test_create_optimizers_prepares_model_with_fabric_setup(monkeypatch):
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(optimizer=SimpleNamespace(name="sgd")),
    )
    optimizer = _OptimizerRecorder()
    monkeypatch.setattr(
        supervised_agent_module,
        "instantiate_optimizer",
        lambda config, module: optimizer,
    )
    setup_calls = []
    agent._setup_model_optimizer = lambda module, optimizer: (
        setup_calls.append((module, optimizer))
        or (SimpleNamespace(module=module), optimizer)
    )
    model = _LinearModel()

    SupervisedAgent.create_optimizers(agent, model)

    assert agent.training_model.module is model
    assert setup_calls == [(model, optimizer)]
    assert agent.supervised_optimizer is optimizer


def test_perform_optimization_step_steps_optimizer_and_clips(monkeypatch):
    clipped = []

    def fake_clip(config, fabric, model, optimizer, model_name):
        clipped.append((model_name, model, optimizer))
        return {"model/grad_norm": torch.tensor(3.0)}

    monkeypatch.setattr(base_agent_module, "handle_model_grad_clipping", fake_clip)

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace()
    agent.fabric = _FabricRecorder()
    agent.training_model = _LinearModel()
    agent.supervised_optimizer = _OptimizerRecorder()
    agent.supervised_step = lambda batch_dict: (
        agent.training_model.linear.weight.sum(),
        {"supervised/loss": torch.tensor(1.0)},
    )

    log_dict = SupervisedAgent.perform_optimization_step(
        agent,
        {"action": torch.zeros(1, 1)},
        batch_idx=0,
    )

    assert agent.supervised_optimizer.zero_grad_calls == [True]
    assert agent.supervised_optimizer.steps == 1
    assert [name for name, _, _ in clipped] == ["model"]
    assert clipped[0][1] is agent.training_model
    assert clipped[0][2] is agent.supervised_optimizer
    assert torch.equal(log_dict["model/grad_norm"], torch.tensor(3.0))


def test_training_load_restores_model_weights_and_optimizer():
    agent = object.__new__(SupervisedAgent)
    model = _CheckpointLinear(1, 1, bias=False)
    model.weight.data.zero_()

    agent.model = model
    agent.config = SimpleNamespace(normalize_rewards=False)
    agent.evaluator = None
    agent.motion_lib = SimpleNamespace(
        config=SimpleNamespace(
            motion_file_switch_mode=MotionFileSwitchMode.FIXED,
        )
    )
    agent.supervised_optimizer = _OptimizerRecorder()
    agent.device = torch.device("cpu")

    SupervisedAgent.load_parameters(
        agent,
        {
            "model": {"weight": torch.tensor([[2.0]])},
            "epoch": 4,
            "supervised_optimizer": {"lr": 0.1},
        },
        load_training_state=True,
    )

    assert agent.current_epoch == 4
    assert torch.equal(agent.model.weight, torch.tensor([[2.0]]))
    assert agent.supervised_optimizer.loaded_state == {"lr": 0.1}
