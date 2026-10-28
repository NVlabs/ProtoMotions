# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseEvaluator helper behavior without a live simulator."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule, RolloutStateSpec

import protomotions.agents.evaluators.base_evaluator as base_evaluator_module
from protomotions.agents.evaluators.aggregate_metrics import (
    ActionSmoothnessAggregateMetric,
)
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.masked_mimic_evaluator import MaskedMimicEvaluator
from protomotions.agents.evaluators.config import EvaluatorConfig, MaskedMimicActionSource
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.manager import EvaluationResult
from protomotions.envs.mdp_component import MdpComponent, eager_kernel


class _Fabric:
    device = torch.device("cpu")
    world_size = 1


@eager_kernel
def _zeros(**params):
    return torch.zeros(**params)


class _EvalComponent(MdpComponent):
    def __init__(self, **params):
        self.threshold = params.pop("threshold", None)
        self.fail_above = params.pop("fail_above", True)
        super().__init__(_zeros, {}, params)


class _Model(torch.nn.Module):
    deployable_action_key = "mean_action"

    def reset_rollout_context(self, env_ids):
        pass

    def forward(self, obs_td):
        return {"mean_action": obs_td["obs"] + 1.0}

    def save_rollout_context(self):
        return {}

    def load_rollout_context(self, state):
        pass


class _Agent:
    def inference_step(self, obs_td):
        return self.model(obs_td)

    def __init__(self, env, root_dir):
        self.env = env
        self.root_dir = Path(root_dir)
        self.num_envs = env.num_envs
        self.model = _Model()
        self.eval_calls = 0
        self.pre_steps = []

    def eval(self):
        self.eval_calls += 1

    def pre_collect_step(self, step):
        self.pre_steps.append(step)

    def add_agent_info_to_obs(self, obs):
        obs["agent_info"] = torch.ones_like(obs["obs"])
        return obs

    def obs_dict_to_tensordict(self, obs):
        return TensorDict(obs, batch_size=obs["obs"].shape[0])


def test_masked_mimic_action_mode_is_configured_independently_of_training_feedback(tmp_path):
    class Model(_Model):
        # Like a real MaskedMimic student: the deployable (prior) path produces
        # "action", so that is the key it declares.
        deployable_action_key = "action"

        def forward(self, obs_td):
            assert not self.training
            return {"action": torch.tensor(1.), "privileged_action": torch.tensor(2.)}

        def forward_deployable(self, obs_td):
            assert not self.training
            return {"action": torch.tensor(3.)}

    class Evaluator(BaseEvaluator):
        _policy_action = MaskedMimicEvaluator._policy_action

        def run_evaluation(self):
            actions.append(self._policy_action({}).item())

        def update_training_state(self, result):
            updates.append(True)

    agent = _Agent(_Env(), tmp_path)
    agent.model = Model()
    agent.eval = agent.model.eval
    actions, updates = [], []
    evaluator = Evaluator(agent, _Fabric(), _config({"value": _EvalComponent()}))
    evaluator.config.action_source = MaskedMimicActionSource.POSTERIOR
    result = evaluator.evaluate()
    assert agent.model.training
    assert updates == []
    evaluator.update_training_state(result)
    evaluator.config.action_source = MaskedMimicActionSource.PRIOR
    evaluator.evaluate()
    assert actions == [2., 3.] and updates == [True]
    assert agent.model.training


class _Env:
    def __init__(self):
        self.num_envs = 2
        self.dt = 0.5
        self.context = {"ctx": torch.ones(2)}
        self.robot_config = SimpleNamespace(
            kinematic_info=SimpleNamespace(num_bodies=1),
        )
        self.simulator = SimpleNamespace()
        self.config = SimpleNamespace()
        self.reset_calls = []
        self.step_actions = []
        # The env-side perturbation switch the evaluator flips around an
        # evaluation. Switch writes and the state write-back share one log, so
        # a test can pin their order: perturbations must come back AFTER the
        # write-back, or the restore's reset_envs would reschedule pushes and
        # move the global random stream.
        self.perturbations_enabled = True
        self.perturbation_events = []

    def set_perturbations_enabled(self, enabled):
        self.perturbations_enabled = bool(enabled)
        self.perturbation_events.append(bool(enabled))

    def save_state(self):
        return {}

    def restore_state(self, state):
        self.perturbation_events.append("restore")

    def reset(self, env_ids=None, **kwargs):
        self.reset_calls.append((None if env_ids is None else env_ids.clone(), kwargs))
        return {"obs": torch.zeros(self.num_envs, 1)}, {}

    def step(self, actions):
        self.step_actions.append(actions.clone())
        return (
            {"obs": actions.clone()},
            torch.ones(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"metric": actions.clone()},
        )


class _HookEvaluator(BaseEvaluator):
    def __init__(self, agent, fabric, config):
        self.starts = []
        self.checks = []
        self.steps = []
        super().__init__(agent, fabric, config)

    def _on_episode_start(self, env_ids):
        self.starts.append(env_ids.clone())

    def _get_reset_kwargs(self):
        return {"deterministic": True}

    def _check_evaluation_failures(self, env_ids, trial_ids):
        self.checks.append((env_ids.clone(), trial_ids.clone()))

    def _on_episode_step(self, env_ids, trial_ids, extras, actions):
        self.steps.append((env_ids.clone(), extras["metric"].clone(), actions.clone()))


class _FakeComponentManager:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def execute_all(self, components, context):
        self.calls.append((components, context))
        result = {}
        for key, component in components.items():
            kind, name = key.split("/", 1)
            value = self.values[name].clone()
            result[key] = value
        return result


class _RobotState:
    def get_shape_mapping(self, flattened=True):
        assert flattened is True
        return {"rigid_body_pos": (6,), "dof_pos": (3,)}


class _SimulatorWithState:
    def get_robot_state(self):
        return _RobotState()


class _Plugin:
    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def compute(self, metrics):
        if self.error is not None:
            raise self.error
        return self.result


def _config(components=None, max_eval_steps=2):
    components = components or {}
    failures = {
        name: TerminationRule(component, component.threshold, component.fail_above)
        for name, component in components.items()
        if component.threshold is not None
    }
    return EvaluatorConfig(metric_components=components, failure_components=failures,
                           score_key="eval/success_rate" if failures else None,
                           max_eval_steps=max_eval_steps)


def _evaluator(tmp_path, components=None):
    env = _Env()
    agent = _Agent(env, tmp_path)
    return BaseEvaluator(agent, _Fabric(), _config(components))


def test_base_evaluator_evaluate_short_circuits_without_components(tmp_path):
    evaluator = _evaluator(tmp_path, components={})

    assert evaluator.evaluate() == EvaluationResult({}, None, 0)
    assert evaluator.agent.eval_calls == 0


@pytest.mark.parametrize("failure_phase", ["initialize", "rollout", "report"])
def test_evaluation_restores_state_and_modes_on_failure(tmp_path, failure_phase):
    evaluator = _evaluator(tmp_path, {"height": _EvalComponent(threshold=0.5)})
    env, model = evaluator.env, evaluator.agent.model
    model.child = torch.nn.Linear(1, 1)
    model.train()
    model.child.eval()
    state = {"physical": 7, "context": 11}
    env.save_state = lambda: state["physical"]

    def restore_state(snapshot):
        state.update(physical=snapshot)
        env.perturbation_events.append("restore")

    env.restore_state = restore_state
    model.save_rollout_context = lambda: state["context"]
    model.load_rollout_context = lambda snapshot: state.update(context=snapshot)
    evaluator.agent.eval = lambda: model.eval()
    def fail():
        state.update(physical=99, context=99)
        raise RuntimeError("injected failure")
    original = evaluator.initialize_eval
    def initialize():
        original()
        if failure_phase == "initialize":
            fail()
        return {}
    evaluator.initialize_eval = initialize
    evaluator.run_evaluation = fail if failure_phase == "rollout" else lambda: None
    evaluator.process_eval_results = fail if failure_phase == "report" else lambda: ({}, None, 0)
    with pytest.raises(RuntimeError, match="injected failure"):
        evaluator.evaluate()
    assert state == {"physical": 7, "context": 11}
    assert model.training and not model.child.training
    # Perturbations were switched off for the evaluation and back on only
    # after the state write-back, on every failure path.
    assert env.perturbation_events == [False, "restore", True]
    assert env.perturbations_enabled is True
    assert evaluator._metrics is None


@pytest.mark.parametrize("failure_phase", [None, "initialize", "rollout", "report", "restore"])
def test_evaluation_restores_real_rollout_buffers_and_mixed_modes(tmp_path, failure_phase):
    class Stateful(ProtoMotionsTensorDictModule):
        def rollout_state_specs(self):
            return {"history": RolloutStateSpec(shape=(2,))}

    evaluator = _evaluator(tmp_path, {"height": _EvalComponent()})
    model = Stateful()
    model.reset_rollout_context(num_envs=2, device="cpu")
    model.history.fill_(7)
    model.frozen_child = torch.nn.Linear(2, 2).eval()
    evaluator.agent.model = model
    evaluator.agent.eval = model.eval
    original_initialize = evaluator.initialize_eval

    def stage(name):
        assert not model.training
        model.history.zero_()
        if failure_phase == name:
            raise RuntimeError(name)

    def initialize():
        result = original_initialize()
        stage("initialize")
        return result

    evaluator.initialize_eval = initialize
    evaluator.run_evaluation = lambda: stage("rollout")
    evaluator.process_eval_results = lambda: (stage("report") or {}, None, 0)

    def restore_state(snapshot):
        evaluator.env.perturbation_events.append("restore")
        stage("restore")

    evaluator.env.restore_state = restore_state
    if failure_phase is None:
        assert evaluator.evaluate().num_items == 0
    else:
        with pytest.raises(RuntimeError, match=failure_phase):
            evaluator.evaluate()
    assert model.training and not model.frozen_child.training
    torch.testing.assert_close(model.history, torch.full((2, 2), 7.0))
    # Even when the write-back itself raises, perturbations come back after it.
    assert evaluator.env.perturbation_events == [False, "restore", True]
    assert evaluator.env.perturbations_enabled is True


def test_offline_evaluation_never_calls_training_consumer(tmp_path):
    evaluator = _evaluator(tmp_path, {"height": _EvalComponent(threshold=0.5)})
    evaluator.run_evaluation = lambda: None
    calls = []
    evaluator.update_training_state = lambda result: calls.append(True)
    result = evaluator.evaluate()
    assert calls == []
    evaluator.update_training_state(result)
    assert calls == [True]


def test_base_evaluator_evaluate_restores_perturbations_when_init_returns_none(
    tmp_path,
):
    class _NoMetricsEvaluator(BaseEvaluator):
        def initialize_eval(self):
            return None

    env = _Env()
    evaluator = _NoMetricsEvaluator(
        _Agent(env, tmp_path),
        _Fabric(),
        _config({"height": _EvalComponent(threshold=1.0)}),
    )

    assert evaluator.evaluate() == EvaluationResult({}, None, 0)
    # Nothing was saved, so nothing is written back; the switch still returns.
    assert env.perturbation_events == [False, True]
    assert env.perturbations_enabled is True


def test_base_evaluator_evaluate_full_cycle_restores_and_logs_component_metrics(
    tmp_path,
):
    class _FullCycleEvaluator(BaseEvaluator):
        def initialize_eval(self):
            self._init_eval_component_buffers(num_eval_ids=1)
            self._evaluation.execute_all = _FakeComponentManager(
                {"height": torch.tensor([0.2, 0.7])}
            ).execute_all
            return {}

        def run_evaluation(self):
            self._check_evaluation_failures(
                active_env_ids=torch.tensor([0]),
                active_motion_ids=torch.tensor([0]),
            )

    env = _Env()
    evaluator = _FullCycleEvaluator(
        _Agent(env, tmp_path),
        _Fabric(),
        _config({"height": _EvalComponent(threshold=0.5, fail_above=True)}),
    )

    result = evaluator.evaluate()
    log_dict, score, num_items = result.logs, result.score, result.num_items

    assert evaluator.agent.eval_calls == 1
    assert evaluator.eval_count == 1
    assert log_dict["eval/success_rate"] == 1.0
    assert log_dict["eval/num_evaluated"] == 1
    assert log_dict["eval/height/failure_rate"] == 0.0
    assert log_dict["eval/height/mean"] == pytest.approx(0.2)
    assert score == 1.0
    assert num_items == 1
    # This evaluator's initialize_eval saves no training state, so there is
    # nothing to write back; the switch still returns.
    assert env.perturbation_events == [False, True]
    assert env.perturbations_enabled is True


def test_base_evaluator_default_initialize_run_properties_and_empty_results(tmp_path):
    evaluator = _evaluator(tmp_path, components={"height": _EvalComponent()})
    calls = []
    evaluator.evaluate_episode = lambda env_ids, max_steps: calls.append(
        (env_ids.clone(), max_steps)
    )

    metrics = evaluator.initialize_eval()
    evaluator.run_evaluation()
    log_dict, score, num_items = evaluator.process_eval_results()

    assert metrics == {}
    assert evaluator.max_eval_steps == 2
    assert torch.equal(calls[0][0], torch.tensor([0, 1]))
    assert calls[0][1] == 2
    assert log_dict == {"eval/num_evaluated": 0}
    assert score is None
    assert num_items == 0
    assert _evaluator(tmp_path).process_eval_results() == ({}, None, 0)


def test_evaluate_episode_runs_hooks_and_steps_policy_actions(tmp_path):
    env = _Env()
    evaluator = _HookEvaluator(
        _Agent(env, tmp_path),
        _Fabric(),
        _config({"height": _EvalComponent(threshold=1.0)}, max_eval_steps=2),
    )
    env_ids = torch.tensor([0, 1])

    evaluator._init_eval_component_buffers(2)
    evaluator.evaluate_episode(env_ids, max_steps=2)

    assert torch.equal(evaluator.starts[0], env_ids)
    assert env.reset_calls[0][1] == {"deterministic": True}
    assert evaluator.agent.pre_steps == [0, 1]
    assert len(evaluator.checks) == 2
    assert len(evaluator.steps) == 2
    assert torch.equal(env.step_actions[0], torch.ones(2, 1))


def test_policy_action_reads_models_declared_action_key(tmp_path):
    env = _Env()
    agent = _Agent(env, tmp_path)
    evaluator = BaseEvaluator(agent, _Fabric(), _config())
    mean_action = torch.ones(2, 1)
    sampled_action = torch.zeros(2, 1)

    # The evaluator selects the key the model declares via deployable_action_key,
    # not a hard-coded mean_action/action fallback in the evaluator.
    class _MeanKeyModel:
        deployable_action_key = "mean_action"

        def __call__(self, obs_td):
            return {"mean_action": mean_action, "action": sampled_action}

    class _ActionKeyModel:
        deployable_action_key = "action"

        def __call__(self, obs_td):
            return {"action": sampled_action}

    agent.model = _MeanKeyModel()
    assert evaluator._policy_action(None) is mean_action

    agent.model = _ActionKeyModel()
    assert evaluator._policy_action(None) is sampled_action


def test_default_episode_hooks_work_with_metrics_only_manager(
    tmp_path,
):
    evaluator = _evaluator(tmp_path, components={"height": _EvalComponent()})
    env_ids = torch.tensor([0, 1])

    evaluator._init_eval_component_buffers(2)
    evaluator._evaluation.execute_all = lambda components, context: {"metric/height": torch.zeros(2)}
    evaluator.evaluate_episode(env_ids, max_steps=1)

    assert evaluator._get_reset_kwargs() == {}
    assert len(evaluator.env.reset_calls) == 1
    assert len(evaluator.env.step_actions) == 1


def test_component_failure_buffers_accumulate_values_and_process_success_rate(tmp_path):
    components = {
        "height": _EvalComponent(threshold=0.5, fail_above=True),
        "speed": _EvalComponent(threshold=0.25, fail_above=False),
    }
    evaluator = _evaluator(tmp_path, components=components)
    evaluator.agent.num_envs = 3
    evaluator._init_eval_component_buffers(num_eval_ids=2)
    evaluator._evaluation.execute_all = _FakeComponentManager(
        {
            "height": torch.tensor([0.2, 0.7, 0.8]),
            "speed": torch.tensor([0.1, 0.4, 0.3]),
        }
    ).execute_all

    evaluator._check_evaluation_failures(
        active_env_ids=torch.tensor([0, 2]),
        active_motion_ids=torch.tensor([1, 0]),
    )
    log_dict, score, num_items = evaluator.process_eval_results()

    assert torch.equal(evaluator._evaluation.evaluated, torch.tensor([True, True]))
    evaluator.process_eval_results()
    assert torch.equal(evaluator._evaluation.failed, torch.tensor([True, True]))
    assert log_dict["eval/success_rate"] == 0.0
    assert score == 0.0
    assert num_items == 2
    assert log_dict["eval/height/failure_rate"] == 0.5
    assert log_dict["eval/speed/failure_rate"] == 0.5
    assert log_dict["eval/height/max"] == pytest.approx(0.8)
    assert log_dict["eval/speed/min"] == pytest.approx(0.1)


def test_process_eval_results_handles_no_evaluated_items(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        components={"height": _EvalComponent(threshold=0.5)},
    )
    evaluator._init_eval_component_buffers(num_eval_ids=2)

    log_dict, score, num_items = evaluator.process_eval_results()

    assert log_dict == {"eval/num_evaluated": 0}
    assert score is None
    assert num_items == 0


def test_evaluator_state_cleanup_and_state_dict_round_trip(tmp_path):
    evaluator = _evaluator(tmp_path, components={"height": _EvalComponent()})
    evaluator.eval_count = 4
    evaluator._metrics = {"x": object()}
    evaluator._init_eval_component_buffers(1)

    state = evaluator.get_state_dict()
    evaluator.load_state_dict({"eval_count": 9})
    evaluator.cleanup_after_evaluation()

    assert state == {"eval_count": 4}
    assert evaluator.eval_count == 9
    assert evaluator._metrics is None
    assert evaluator._evaluation is None


def test_robot_state_metric_addition_preserves_existing_metrics(tmp_path):
    evaluator = _evaluator(tmp_path)
    motion_lens = torch.tensor([2, 3])
    reward = MotionMetrics(2, motion_lens, 3, device=evaluator.device)
    metrics = {"reward": reward}

    evaluator.env.simulator = _SimulatorWithState()
    evaluator._add_robot_state_metrics(metrics, 2, motion_lens, 3)

    assert set(metrics).issuperset({"reward", "rigid_body_pos", "dof_pos"})
    assert metrics["rigid_body_pos"].num_sub_features == 6
    assert metrics["reward"] is reward


def test_robot_state_metric_addition_exposes_invalid_simulator(
    tmp_path, caplog
):
    evaluator = _evaluator(tmp_path)
    motion_lens = torch.tensor([1])
    metrics = {}
    delattr(evaluator.env, "simulator")

    with pytest.raises(AttributeError):
        evaluator._add_robot_state_metrics(metrics, 1, motion_lens, 1)
    assert metrics == {}

    evaluator.env.simulator = SimpleNamespace(get_robot_state=lambda: None)
    with pytest.raises(AttributeError):
        evaluator._add_robot_state_metrics(metrics, 1, motion_lens, 1)

    assert metrics == {}


def test_metric_plugins_do_not_suppress_failures(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.metric_plugins = [
        _Plugin({"eval/custom": 1.5}),
        _Plugin(error=RuntimeError("boom")),
    ]

    with pytest.raises(RuntimeError, match="boom"):
        evaluator._compute_additional_metrics({})


def test_metric_plugin_registration_failures_propagate(
    tmp_path,
    monkeypatch,
    caplog,
):
    class _BadSmoothnessMetric:
        def __init__(self, *args, **kwargs):
            raise ValueError("bad smoothness")

    class _BadActionMetric:
        def __init__(self, *args, **kwargs):
            raise TypeError("bad action")

    evaluator = _evaluator(tmp_path)
    monkeypatch.setattr(
        base_evaluator_module,
        "SmoothnessAggregateMetric",
        _BadSmoothnessMetric,
    )
    monkeypatch.setattr(
        base_evaluator_module,
        "ActionSmoothnessAggregateMetric",
        _BadActionMetric,
    )

    with pytest.raises(ValueError, match="bad smoothness"):
        evaluator._register_smoothness_plugin()
    with pytest.raises(TypeError, match="bad action"):
        evaluator._register_action_smoothness_plugin()


def test_action_smoothness_plugin_and_save_list_to_file(tmp_path):
    evaluator = _evaluator(tmp_path)
    assert evaluator._register_action_smoothness_plugin() is True
    assert isinstance(evaluator.metric_plugins[-1], ActionSmoothnessAggregateMetric)

    actions = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([3]),
        max_motion_len=3,
        num_sub_features=2,
        device=torch.device("cpu"),
    )
    actions.data[0, :3] = torch.tensor([[0.0, 0.0], [1.0, -1.0], [3.0, -3.0]])
    actions.frame_counts[:] = 3

    result = evaluator.metric_plugins[-1].compute({"actions": actions})
    evaluator._save_list_to_file(["a", "b"], "items.txt", subdirectory="nested")
    evaluator._save_list_to_file(["root"], "root_items.txt")

    assert result["eval/action_delta_mean_rad"] == pytest.approx(1.5)
    assert result["eval/action_rate_mean_rad_s"] == pytest.approx(3.0)
    assert (tmp_path / "nested" / "items.txt").read_text().splitlines() == ["a", "b"]
    assert (tmp_path / "root_items.txt").read_text().splitlines() == ["root"]




def test_simple_test_policy_steps_the_policy_until_keyboard_interrupt(
    tmp_path,
    capsys,
):
    class _InterruptingEnv(_Env):
        def reset(self, env_ids=None, **kwargs):
            if self.reset_calls:
                raise KeyboardInterrupt
            return super().reset(env_ids, **kwargs)

        def step(self, actions):
            self.step_actions.append(actions.clone())
            return (
                {"obs": actions.clone()},
                torch.ones(self.num_envs),
                torch.tensor([False, True]),
                torch.tensor([False, True]),
                {},
            )

    env = _InterruptingEnv()
    evaluator = BaseEvaluator(_Agent(env, tmp_path), _Fabric(), _config({}))

    evaluator.simple_test_policy()

    output = capsys.readouterr().out
    assert "Stopped after 1 steps." in output
    assert env.reset_calls[0][0] is None
    assert torch.equal(env.step_actions[0], torch.ones(2, 1))


def test_simple_test_policy_requires_declared_action_key_in_outputs(tmp_path):
    class _MissingActionModel:
        deployable_action_key = "action"

        def __call__(self, obs_td):
            return {}

    env = _Env()
    agent = _Agent(env, tmp_path)
    agent.model = _MissingActionModel()
    evaluator = BaseEvaluator(agent, _Fabric(), _config({}))

    with pytest.raises(KeyError, match="action"):
        evaluator.simple_test_policy()

    assert env.step_actions == []
