# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseEvaluator helper behavior without a live simulator."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import builtins

import pytest
import torch
from tensordict import TensorDict

import protomotions.agents.evaluators.base_evaluator as base_evaluator_module
from protomotions.agents.evaluators.aggregate_metrics import ActionSmoothnessAggregateMetric
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import EvaluatorConfig
from protomotions.agents.evaluators.metrics import MotionMetrics


class _Fabric:
    device = torch.device("cpu")


class _EvalComponent(dict):
    @property
    def static_params(self):
        return self


class _Model:
    def __call__(self, obs_td):
        return {"mean_action": obs_td["obs"] + 1.0}


class _Agent:
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


class _Env:
    def __init__(self):
        self.num_envs = 2
        self.dt = 0.5
        self.context = {"ctx": torch.ones(2)}
        self.robot_config = SimpleNamespace(
            reset_noise="noise",
            kinematic_info=SimpleNamespace(num_bodies=1),
        )
        self.simulator = SimpleNamespace(_push_enabled=True)
        self.config = SimpleNamespace()
        self.reset_calls = []
        self.step_actions = []

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

    def _check_eval_components(self, env_ids, step_idx):
        self.checks.append((env_ids.clone(), step_idx))

    def _on_episode_step(self, env_ids, extras, actions):
        self.steps.append((env_ids.clone(), extras["metric"].clone(), actions.clone()))


class _FakeComponentManager:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def execute_all(self, components, context):
        self.calls.append((components, context))
        return {key: value.clone() for key, value in self.values.items()}


class _RobotState:
    def get_shape_mapping(self, flattened=True):
        assert flattened is True
        return {"rigid_body_pos": (6,), "dof_pos": (3,)}


class _SimulatorWithState:
    _push_enabled = True

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
    return EvaluatorConfig(
        evaluation_components=components or {},
        max_eval_steps=max_eval_steps,
    )


def _evaluator(tmp_path, components=None):
    env = _Env()
    agent = _Agent(env, tmp_path)
    return BaseEvaluator(agent, _Fabric(), _config(components))


def test_base_evaluator_evaluate_short_circuits_without_components(tmp_path):
    evaluator = _evaluator(tmp_path, components={})

    assert evaluator.evaluate() == ({}, None, 0)
    assert evaluator.agent.eval_calls == 0


def test_base_evaluator_evaluate_restores_perturbations_when_init_returns_none(tmp_path):
    class _NoMetricsEvaluator(BaseEvaluator):
        def initialize_eval(self):
            return None

    env = _Env()
    evaluator = _NoMetricsEvaluator(
        _Agent(env, tmp_path),
        _Fabric(),
        _config({"height": _EvalComponent(threshold=1.0)}),
    )

    assert evaluator.evaluate() == ({}, None, 0)
    assert env.robot_config.reset_noise == "noise"
    assert env.simulator._push_enabled is True


def test_base_evaluator_evaluate_full_cycle_restores_and_logs_component_metrics(tmp_path):
    class _FullCycleEvaluator(BaseEvaluator):
        def initialize_eval(self):
            self._init_eval_component_buffers(num_eval_ids=1)
            self._component_manager = _FakeComponentManager(
                {"height": torch.tensor([0.2, 0.7])}
            )
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

    log_dict, score, num_items = evaluator.evaluate()

    assert evaluator.agent.eval_calls == 1
    assert evaluator.eval_count == 1
    assert log_dict["eval/success_rate"] == 1.0
    assert log_dict["eval/num_evaluated"] == 1
    assert log_dict["eval/height/failure_rate"] == 0.0
    assert log_dict["eval/height/mean"] == pytest.approx(0.2)
    assert score == 1.0
    assert num_items == 1
    assert env.robot_config.reset_noise == "noise"
    assert env.simulator._push_enabled is True


def test_base_evaluator_default_initialize_run_properties_and_empty_results(tmp_path):
    evaluator = _evaluator(tmp_path, components={"height": _EvalComponent()})
    calls = []
    evaluator.evaluate_episode = (
        lambda env_ids, max_steps: calls.append((env_ids.clone(), max_steps))
    )

    metrics = evaluator.initialize_eval()
    evaluator.run_evaluation()
    log_dict, score, num_items = evaluator.process_eval_results()

    assert metrics == {}
    assert evaluator.max_eval_steps == 2
    assert torch.equal(calls[0][0], torch.tensor([0, 1]))
    assert calls[0][1] == 2
    assert log_dict == {"eval/success_rate": 1.0, "eval/num_evaluated": 0}
    assert score == 1.0
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

    evaluator.evaluate_episode(env_ids, max_steps=2)

    assert torch.equal(evaluator.starts[0], env_ids)
    assert env.reset_calls[0][1] == {"deterministic": True}
    assert evaluator.agent.pre_steps == [0, 1, 2]
    assert [step for _, step in evaluator.checks] == [0, 1]
    assert len(evaluator.steps) == 2
    assert torch.equal(env.step_actions[0], torch.ones(2, 1))


def test_default_episode_hooks_are_noops_and_check_short_circuits_without_component_manager(tmp_path):
    evaluator = _evaluator(tmp_path, components={"height": _EvalComponent()})
    env_ids = torch.tensor([0, 1])

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
    evaluator._component_manager = _FakeComponentManager(
        {
            "height": torch.tensor([0.2, 0.7, 0.8]),
            "speed": torch.tensor([0.1, 0.4, 0.3]),
        }
    )

    evaluator._check_evaluation_failures(
        active_env_ids=torch.tensor([0, 2]),
        active_motion_ids=torch.tensor([1, 0]),
    )
    log_dict, score, num_items = evaluator.process_eval_results()

    assert torch.equal(evaluator._eval_mask, torch.tensor([True, True]))
    assert torch.equal(evaluator._motion_failed, torch.tensor([True, True]))
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

    assert log_dict["eval/success_rate"] == 1.0
    assert log_dict["eval/height/failure_rate"] == 0.0
    assert score == 1.0
    assert num_items == 0


def test_evaluator_state_cleanup_and_state_dict_round_trip(tmp_path):
    evaluator = _evaluator(tmp_path, components={"height": _EvalComponent()})
    evaluator.eval_count = 4
    evaluator._metrics = {"x": object()}
    evaluator._motion_failed = torch.ones(1, dtype=torch.bool)
    evaluator._eval_mask = torch.ones(1, dtype=torch.bool)
    evaluator._per_component_failures = {"height": torch.ones(1, dtype=torch.bool)}
    evaluator._component_manager = object()

    state = evaluator.get_state_dict()
    evaluator.load_state_dict({"eval_count": 9})
    evaluator.cleanup_after_evaluation()

    assert state == {"eval_count": 4}
    assert evaluator.eval_count == 9
    assert evaluator._metrics is None
    assert evaluator._motion_failed is None
    assert evaluator._per_component_failures == {}
    assert evaluator._component_manager is None


def test_metric_creation_robot_state_metric_addition_and_generation(tmp_path):
    evaluator = _evaluator(tmp_path)
    motion_lens = torch.tensor([2, 3])
    metrics = evaluator._create_base_metrics(
        ["reward"],
        num_motions=2,
        motion_num_frames=motion_lens,
        max_eval_steps=3,
    )
    metrics["reward"].data[0, :2, 0] = torch.tensor([1.0, 3.0])
    metrics["reward"].data[1, :3, 0] = torch.tensor([2.0, 4.0, 6.0])
    metrics["reward"].frame_counts[:] = motion_lens

    evaluator.env.simulator = _SimulatorWithState()
    evaluator._add_robot_state_metrics(metrics, 2, motion_lens, 3)
    logs = evaluator._gen_metrics(metrics, ["reward", "missing"], prefix="eval")

    assert set(metrics).issuperset({"reward", "rigid_body_pos", "dof_pos"})
    assert metrics["rigid_body_pos"].num_sub_features == 6
    assert logs["eval_mean/reward"] == pytest.approx(3.0)
    assert logs["eval_max/reward"] == pytest.approx(4.0)
    assert logs["eval_min/reward"] == pytest.approx(2.0)


def test_robot_state_metric_addition_ignores_missing_or_invalid_simulator(tmp_path, caplog):
    evaluator = _evaluator(tmp_path)
    motion_lens = torch.tensor([1])
    metrics = {}
    delattr(evaluator.env, "simulator")

    evaluator._add_robot_state_metrics(metrics, 1, motion_lens, 1)
    assert metrics == {}

    evaluator.env.simulator = SimpleNamespace(get_robot_state=lambda: None)
    with caplog.at_level("WARNING"):
        evaluator._add_robot_state_metrics(metrics, 1, motion_lens, 1)

    assert metrics == {}
    assert "Could not add robot state metrics" in caplog.text


def test_metric_plugins_collect_results_and_suppress_plugin_failures(tmp_path, caplog):
    evaluator = _evaluator(tmp_path)
    evaluator.metric_plugins = [
        _Plugin({"eval/custom": 1.5}),
        _Plugin(error=RuntimeError("boom")),
    ]

    with caplog.at_level("WARNING"):
        result = evaluator._compute_additional_metrics({})

    assert result == {"eval/custom": 1.5}
    assert "Plugin _Plugin failed" in caplog.text


def test_metric_plugin_registration_failure_paths_are_logged(
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

    with caplog.at_level("WARNING"):
        smoothness_registered = evaluator._register_smoothness_plugin()
        action_registered = evaluator._register_action_smoothness_plugin()

    assert smoothness_registered is False
    assert action_registered is False
    assert "Skipping smoothness plugin" in caplog.text
    assert "Skipping action smoothness plugin" in caplog.text


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


def test_plot_per_frame_metrics_filters_features_handles_empty_frames_and_saves(
    tmp_path,
    monkeypatch,
):
    class _Axis:
        def __init__(self):
            self.transAxes = object()
            self.text_calls = []
            self.plot_calls = []
            self.titles = []
            self.labels = []
            self.grid_calls = []
            self.legend_calls = 0

        def text(self, *args, **kwargs):
            self.text_calls.append((args, kwargs))

        def plot(self, *args, **kwargs):
            self.plot_calls.append((args, kwargs))

        def set_xlabel(self, label):
            self.labels.append(("x", label))

        def set_ylabel(self, label):
            self.labels.append(("y", label))

        def set_title(self, title):
            self.titles.append(title)

        def grid(self, *args, **kwargs):
            self.grid_calls.append((args, kwargs))

        def legend(self):
            self.legend_calls += 1

    axes = [_Axis(), _Axis()]
    saved_paths = []
    closed = []
    fake_fig = object()

    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *args, **kwargs: (fake_fig, axes)
    pyplot.tight_layout = lambda: None
    pyplot.savefig = lambda path, **kwargs: saved_paths.append((path, kwargs))
    pyplot.close = lambda fig: closed.append(fig)
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    evaluator = _evaluator(tmp_path)
    empty = MotionMetrics(1, torch.tensor([0]), 2, device=torch.device("cpu"))
    empty.frame_counts[:] = 0
    reward = MotionMetrics(1, torch.tensor([2]), 2, device=torch.device("cpu"))
    reward.data[0, :2, 0] = torch.tensor([1.0, 3.0])
    reward.frame_counts[:] = 2
    vector = MotionMetrics(
        1,
        torch.tensor([2]),
        2,
        num_sub_features=2,
        device=torch.device("cpu"),
    )
    vector.frame_counts[:] = 2

    evaluator._plot_per_frame_metrics(
        {"empty": empty, "reward": reward, "vector": vector},
        keys_to_plot=["empty", "reward", "vector"],
        custom_colors={"reward": "tab:red"},
        output_filename="plot.png",
    )

    assert axes[0].text_calls[0][0][2] == "No data for empty"
    assert axes[0].titles == ["empty"]
    assert axes[1].plot_calls[0][1]["color"] == "tab:red"
    assert axes[1].plot_calls[0][1]["label"] == "reward"
    assert axes[1].labels == [("x", "Time (s)"), ("y", "reward")]
    assert axes[1].titles == ["reward vs Time"]
    assert saved_paths[0][0] == tmp_path / "plot.png"
    assert saved_paths[0][1]["dpi"] == 150
    assert closed == [fake_fig]


def test_plot_per_frame_metrics_reports_when_no_single_feature_metrics(tmp_path, capsys):
    evaluator = _evaluator(tmp_path)
    vector = MotionMetrics(
        1,
        torch.tensor([1]),
        1,
        num_sub_features=2,
        device=torch.device("cpu"),
    )
    vector.frame_counts[:] = 1

    evaluator._plot_per_frame_metrics({"vector": vector})

    assert "No single-feature metrics found for plotting" in capsys.readouterr().out


def test_plot_per_frame_metrics_handles_missing_matplotlib(tmp_path, monkeypatch, capsys):
    real_import = builtins.__import__

    def raising_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)
    evaluator = _evaluator(tmp_path)

    evaluator._plot_per_frame_metrics({})

    assert "matplotlib not available, skipping plotting" in capsys.readouterr().out


def test_plot_per_frame_metrics_wraps_single_axis_from_matplotlib(
    tmp_path,
    monkeypatch,
):
    class _Axis:
        transAxes = object()

        def __init__(self):
            self.plot_calls = []

        def text(self, *args, **kwargs):
            pass

        def plot(self, *args, **kwargs):
            self.plot_calls.append((args, kwargs))

        def set_xlabel(self, label):
            pass

        def set_ylabel(self, label):
            pass

        def set_title(self, title):
            pass

        def grid(self, *args, **kwargs):
            pass

        def legend(self):
            pass

    axis = _Axis()
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *args, **kwargs: (object(), axis)
    pyplot.tight_layout = lambda: None
    pyplot.savefig = lambda *args, **kwargs: None
    pyplot.close = lambda *args, **kwargs: None
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    metric = MotionMetrics(1, torch.tensor([1]), 1, device=torch.device("cpu"))
    metric.data[0, 0, 0] = 1.0
    metric.frame_counts[:] = 1

    _evaluator(tmp_path)._plot_per_frame_metrics({"reward": metric})

    assert len(axis.plot_calls) == 1


def test_simple_test_policy_collects_running_eval_values_until_keyboard_interrupt(
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
                {"eval_values": {"speed": torch.tensor([2.0, 4.0])}},
            )

    env = _InterruptingEnv()
    evaluator = BaseEvaluator(_Agent(env, tmp_path), _Fabric(), _config({}))

    evaluator.simple_test_policy(collect_metrics=True)

    output = capsys.readouterr().out
    assert "Stopped after 1 steps." in output
    assert "Average metrics:" in output
    assert "speed: 3.0000" in output
    assert env.reset_calls[0][0] is None
    assert torch.equal(env.step_actions[0], torch.ones(2, 1))


def test_simple_test_policy_requires_mean_action_or_action(tmp_path):
    class _MissingActionModel:
        def __call__(self, obs_td):
            return {}

    env = _Env()
    agent = _Agent(env, tmp_path)
    agent.model = _MissingActionModel()
    evaluator = BaseEvaluator(agent, _Fabric(), _config({}))

    with pytest.raises(KeyError, match="action"):
        evaluator.simple_test_policy()

    assert env.step_actions == []
