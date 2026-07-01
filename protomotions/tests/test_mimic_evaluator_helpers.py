# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MimicEvaluator motion-specific helper behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.evaluators.config import (
    MimicEvaluatorConfig,
    MotionWeightsRulesConfig,
)
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.mimic_evaluator import (
    MimicEpisodeContext,
    MimicEvaluator,
)


class _Fabric:
    device = torch.device("cpu")
    global_rank = 0


class _RobotState:
    def get_shape_mapping(self, flattened=True):
        assert flattened is True
        return {
            "dof_pos": (2,),
            "rigid_body_pos": (3,),
            "rigid_body_rot": (4,),
        }


class _Simulator:
    def __init__(self):
        self._push_enabled = True
        self.parked = []

    def get_robot_state(self):
        return _RobotState()

    def park_envs(self, env_ids):
        self.parked.append(env_ids.clone())


class _MotionLib:
    def __init__(self):
        self.lengths = torch.tensor([1.0, 2.5, 1.5])
        self.motion_weights = torch.tensor([1.0, 2.0, 3.0])
        self.motion_files = ("a", "b", "c")

    def num_motions(self):
        return self.lengths.numel()

    def get_motion_length(self, motion_ids):
        if motion_ids is None:
            return self.lengths.clone()
        return self.lengths[motion_ids].clone()


class _MotionManager:
    def __init__(self, num_envs=2):
        self.motion_ids = torch.arange(num_envs)
        self.motion_times = torch.arange(num_envs, dtype=torch.float)
        self.motion_weights = torch.ones(3)
        self.fixed = (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
        self.updated_weights = None

    def get_unique_fixed_motions(self):
        return self.fixed

    def update_sampling_weights(self, weights):
        self.updated_weights = weights.clone()


class _Env:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.dt = 0.5
        self.motion_manager = _MotionManager(num_envs)
        self.simulator = _Simulator()
        self.robot_config = SimpleNamespace(
            reset_noise="noise",
            kinematic_info=SimpleNamespace(num_dofs=2, num_bodies=1),
        )
        self.config = SimpleNamespace(
            ref_respawn_offset=0.05,
        )
        self.context = {}
        self.saved_state = {"state": torch.tensor(1)}
        self.restored_state = None
        self.respawn_root_offset = torch.zeros(num_envs, 3)
        self.reset_calls = []
        self.step_actions = []

    def save_state(self):
        return self.saved_state

    def restore_state(self, state):
        self.restored_state = state

    def reset(self, env_ids=None, **kwargs):
        self.reset_calls.append((None if env_ids is None else env_ids.clone(), kwargs))
        return {"obs": torch.zeros(self.num_envs, 2)}, {}

    def step(self, actions):
        self.step_actions.append(actions.clone())
        extras = {
            "raw/dof_pos": torch.arange(self.num_envs * 2, dtype=torch.float).view(
                self.num_envs, 2
            )
        }
        return (
            {"obs": actions.clone()},
            torch.ones(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            extras,
        )


class _Model:
    def __call__(self, obs_td):
        return {"mean_action": obs_td["obs"] + 1.0}


class _Agent:
    def __init__(self, env, root_dir):
        self.env = env
        self.root_dir = Path(root_dir)
        self.num_envs = env.num_envs
        self.motion_lib = _MotionLib()
        self.current_epoch = 7
        self.model = _Model()
        self.pre_steps = []

    def eval(self):
        pass

    def pre_collect_step(self, step):
        self.pre_steps.append(step)

    def add_agent_info_to_obs(self, obs):
        return obs

    def obs_dict_to_tensordict(self, obs):
        return TensorDict(obs, batch_size=obs["obs"].shape[0])


def _config(**overrides):
    config = MimicEvaluatorConfig(
        max_eval_steps=4,
        eval_metrics_every=2,
        save_predicted_motion_lib_every=None,
        motion_weights_rules=MotionWeightsRulesConfig(
            motion_weights_update_success_discount=0.5,
            motion_weights_update_failure_discount=0.5,
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _evaluator(tmp_path, num_envs=2, **config_overrides):
    env = _Env(num_envs=num_envs)
    agent = _Agent(env, tmp_path)
    evaluator = MimicEvaluator(agent, _Fabric(), _config(**config_overrides))
    return evaluator


def _metric(num_motions=3, max_steps=4, features=2):
    return MotionMetrics(
        num_motions=num_motions,
        motion_lens=torch.full((num_motions,), max_steps),
        max_motion_len=max_steps,
        num_sub_features=features,
        device=torch.device("cpu"),
    )


def _packed_metric(motion_lens, features):
    metric = MotionMetrics(
        num_motions=motion_lens.numel(),
        motion_lens=motion_lens,
        max_motion_len=2,
        num_sub_features=features,
        device=torch.device("cpu"),
    )
    values = torch.arange(
        motion_lens.numel() * 2 * features,
        dtype=torch.float,
    ).view(motion_lens.numel(), 2, features)
    metric.data[:] = values
    metric.frame_counts[:] = motion_lens
    return metric


def test_mimic_initialize_eval_creates_metrics_and_caches_environment_state(tmp_path):
    evaluator = _evaluator(tmp_path)

    metrics = evaluator.initialize_eval()

    assert set(metrics).issuperset({"actions", "dof_pos", "rigid_body_pos"})
    assert metrics["actions"].num_sub_features == 2
    assert torch.equal(metrics["actions"].motion_lens, torch.tensor([2, 4, 3]))
    assert torch.equal(evaluator._cached_motion_ids, torch.tensor([0, 1]))
    assert torch.equal(evaluator._cached_motion_times, torch.tensor([0.0, 1.0]))

    evaluator.cleanup_after_evaluation()

    assert evaluator.env.restored_state == {"state": torch.tensor(1)}
    assert evaluator._metrics is None


def test_mimic_motion_sampling_weights_discount_successes_and_failures(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator._motion_failed = torch.tensor([False, True, True])

    evaluator._update_motion_sampling_weights()

    assert torch.allclose(
        evaluator.env.motion_manager.updated_weights,
        torch.tensor([0.25, 4.0, 4.0]),
    )
    failed_file = (
        tmp_path / "failed_motions" / "failed_motions_epoch_7_rank_0.txt"
    )
    assert failed_file.read_text().splitlines() == ["1", "2"]


def test_mimic_motion_sampling_weights_handles_no_failures_and_zero_failure_discount(
    tmp_path,
):
    evaluator = _evaluator(
        tmp_path,
        motion_weights_rules=MotionWeightsRulesConfig(
            motion_weights_update_success_discount=0.5,
            motion_weights_update_failure_discount=0.0,
        ),
    )

    evaluator._motion_failed = None
    evaluator._update_motion_sampling_weights()
    assert evaluator.env.motion_manager.updated_weights is None

    evaluator._motion_failed = torch.tensor([False, True, False])
    evaluator._update_motion_sampling_weights()

    assert torch.allclose(
        evaluator.env.motion_manager.updated_weights,
        torch.tensor([0.25, 1.0, 0.25]),
    )


def test_mimic_parks_inactive_envs_only_when_batch_is_partial(tmp_path):
    evaluator = _evaluator(tmp_path, num_envs=4)

    evaluator._park_inactive_envs(torch.tensor([1, 3]))
    evaluator._park_inactive_envs(torch.arange(4))
    evaluator._park_inactive_envs(None)

    assert len(evaluator.env.simulator.parked) == 1
    assert torch.equal(evaluator.env.simulator.parked[0], torch.tensor([0, 2]))


def test_mimic_build_eval_batches_prefers_fixed_motions_then_chunks_all_motions(tmp_path):
    evaluator = _evaluator(tmp_path, num_envs=2)
    evaluator.motion_manager.fixed = (
        torch.tensor([2, 0]),
        torch.tensor([1, 0]),
    )

    fixed_batches = evaluator._build_eval_batches()
    assert len(fixed_batches) == 1
    assert torch.equal(fixed_batches[0][0], torch.tensor([1, 0]))
    assert torch.equal(fixed_batches[0][1], torch.tensor([2, 0]))

    evaluator.motion_manager.fixed = (
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    batches = evaluator._build_eval_batches()

    assert len(batches) == 2
    assert torch.equal(batches[0][0], torch.tensor([0, 1]))
    assert torch.equal(batches[0][1], torch.tensor([0, 1]))
    assert torch.equal(batches[1][0], torch.tensor([0]))
    assert torch.equal(batches[1][1], torch.tensor([2]))


def test_mimic_run_evaluation_sets_episode_context_and_uses_motion_lengths(tmp_path):
    evaluator = _evaluator(tmp_path, num_envs=2)
    calls = []

    def record_episode(env_ids, max_steps):
        calls.append(
            (
                env_ids.clone(),
                max_steps,
                evaluator._episode_ctx.motion_ids.clone(),
                evaluator._episode_ctx.frame_limits.clone(),
            )
        )

    evaluator.evaluate_episode = record_episode

    evaluator.run_evaluation()

    assert len(calls) == 2
    assert torch.equal(calls[0][0], torch.tensor([0, 1]))
    assert calls[0][1] == 4
    assert torch.equal(calls[0][2], torch.tensor([0, 1]))
    assert torch.equal(calls[0][3], torch.tensor([2, 4]))
    assert torch.equal(calls[1][0], torch.tensor([0]))
    assert calls[1][1] == 3
    assert torch.equal(calls[1][2], torch.tensor([2]))
    assert torch.equal(calls[1][3], torch.tensor([3]))


def test_mimic_hooks_set_motion_state_filter_active_frames_and_record_metrics(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator._episode_ctx = MimicEpisodeContext(
        motion_ids=torch.tensor([1, 0]),
        frame_limits=torch.tensor([1, 3]),
    )
    evaluator._metrics = {"actions": _metric(), "dof_pos": _metric(features=2)}
    checked = []
    evaluator._check_evaluation_failures = (
        lambda active_env_ids, active_motion_ids: checked.append(
            (active_env_ids.clone(), active_motion_ids.clone())
        )
    )

    evaluator._on_episode_start(torch.tensor([0, 1]))
    assert evaluator._get_reset_kwargs() == {
        "sample_flat": True,
        "disable_motion_resample": True,
    }
    evaluator._check_eval_components(torch.tensor([0, 1]), step_idx=1)
    evaluator._on_episode_step(
        torch.tensor([0, 1]),
        {"raw/dof_pos": torch.tensor([[10.0, 11.0], [20.0, 21.0]])},
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert torch.equal(evaluator.motion_manager.motion_ids, torch.tensor([1, 0]))
    assert torch.equal(evaluator.motion_manager.motion_times, torch.zeros(2))
    assert torch.equal(checked[0][0], torch.tensor([1]))
    assert torch.equal(checked[0][1], torch.tensor([0]))
    assert torch.equal(evaluator._metrics["actions"].data[1, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(evaluator._metrics["dof_pos"].data[0, 0], torch.tensor([20.0, 21.0]))


def test_mimic_check_eval_components_skips_when_no_clip_is_active(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator._episode_ctx = MimicEpisodeContext(
        motion_ids=torch.tensor([1, 0]),
        frame_limits=torch.tensor([1, 1]),
    )
    checked = []
    evaluator._check_evaluation_failures = (
        lambda active_env_ids, active_motion_ids: checked.append(
            (active_env_ids, active_motion_ids)
        )
    )

    evaluator._check_eval_components(torch.tensor([0, 1]), step_idx=1)

    assert checked == []


def test_mimic_evaluate_episode_applies_action_ema_and_records_actions(tmp_path):
    evaluator = _evaluator(tmp_path, eval_action_ema_alpha=0.5)
    evaluator._episode_ctx = MimicEpisodeContext(
        motion_ids=torch.tensor([0, 1]),
        frame_limits=torch.tensor([2, 2]),
    )
    evaluator._metrics = {"actions": _metric(num_motions=2, features=2)}
    evaluator._check_evaluation_failures = lambda active_env_ids, active_motion_ids: None

    evaluator.evaluate_episode(torch.tensor([0, 1]), max_steps=2)

    assert torch.equal(evaluator.env.step_actions[0], torch.ones(2, 2))
    assert torch.equal(evaluator.env.step_actions[1], torch.full((2, 2), 1.5))
    assert torch.equal(evaluator._metrics["actions"].data[0, 1], torch.full((2,), 1.5))


def test_mimic_process_eval_results_updates_weights_and_additional_metrics(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator._motion_failed = torch.tensor([False, True, False])
    evaluator._eval_mask = torch.tensor([True, True, True])
    evaluator._per_component_failures = {}
    evaluator._component_value_sum = {}
    evaluator._component_step_count = {}
    evaluator.metric_plugins = [SimpleNamespace(compute=lambda metrics: {"eval/x": 2.0})]
    evaluator._metrics = {"actions": _metric()}

    log_dict, score, num_items = evaluator.process_eval_results()

    assert log_dict["eval/success_rate"] == pytest.approx(2 / 3)
    assert log_dict["eval/x"] == 2.0
    assert score == pytest.approx(2 / 3)
    assert num_items == 3
    assert evaluator.env.motion_manager.updated_weights is not None


def test_mimic_process_eval_results_saves_predicted_motion_lib_on_interval(tmp_path):
    evaluator = _evaluator(tmp_path, save_predicted_motion_lib_every=1)
    evaluator._motion_failed = torch.tensor([False, False, False])
    evaluator._eval_mask = torch.tensor([True, True, True])
    evaluator._per_component_failures = {}
    evaluator._component_value_sum = {}
    evaluator._component_step_count = {}
    evaluator._metrics = {"actions": _metric()}
    saved = []
    evaluator._save_predicted_motion_lib = (
        lambda metrics, epoch: saved.append((metrics, epoch))
    )

    log_dict, score, num_items = evaluator.process_eval_results()

    assert log_dict["eval/success_rate"] == 1.0
    assert score == 1.0
    assert num_items == 3
    assert saved == [(evaluator._metrics, 7)]


def test_mimic_save_predicted_motion_lib_requires_all_metrics(tmp_path):
    evaluator = _evaluator(tmp_path)

    with pytest.raises(ValueError, match="Missing metric 'dof_pos'"):
        evaluator._save_predicted_motion_lib({}, epoch=1)


def test_mimic_plot_per_frame_metrics_prefers_available_eval_component_keys(
    tmp_path,
    monkeypatch,
):
    calls = []

    def record_plot(self, metrics, **kwargs):
        calls.append((metrics, kwargs))

    monkeypatch.setattr(BaseEvaluator, "_plot_per_frame_metrics", record_plot)
    evaluator = _evaluator(
        tmp_path,
        evaluation_components={
            "height": SimpleNamespace(),
            "speed": SimpleNamespace(),
        },
    )
    metrics = {"height": _metric(num_motions=1), "other": _metric(num_motions=1)}

    evaluator._plot_per_frame_metrics(metrics)

    assert calls[0][0] is metrics
    assert calls[0][1]["keys_to_plot"] == ["height"]
    assert calls[0][1]["custom_colors"] == {}
    assert calls[0][1]["output_filename"] == "metrics_per_frame_plot.png"


def test_mimic_save_predicted_motion_lib_packs_fields_and_removes_replay_offset(
    tmp_path,
):
    evaluator = _evaluator(tmp_path)
    motion_lens = torch.tensor([2, 1, 0])
    metrics = {
        "dof_pos": _packed_metric(motion_lens, features=2),
        "dof_vel": _packed_metric(motion_lens, features=2),
        "rigid_body_pos": _packed_metric(motion_lens, features=3),
        "rigid_body_rot": _packed_metric(motion_lens, features=4),
        "rigid_body_vel": _packed_metric(motion_lens, features=3),
        "rigid_body_ang_vel": _packed_metric(motion_lens, features=3),
        "rigid_body_contacts": _packed_metric(motion_lens, features=1),
    }
    metrics["rigid_body_pos"].data[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    metrics["rigid_body_pos"].data[0, 1] = torch.tensor([4.0, 5.0, 6.0])
    metrics["rigid_body_pos"].data[1, 0] = torch.tensor([11.0, 22.0, 33.0])
    metrics["rigid_body_contacts"].data[0, 0, 0] = 0.0
    metrics["rigid_body_contacts"].data[0, 1, 0] = 1.0
    metrics["rigid_body_contacts"].data[1, 0, 0] = 2.0
    evaluator.motion_manager.fixed = (
        torch.tensor([1]),
        torch.tensor([0]),
    )
    evaluator.env.respawn_root_offset[0] = torch.tensor([10.0, 20.0, 0.55])

    evaluator._save_predicted_motion_lib(metrics, epoch=3)

    saved = torch.load(tmp_path / "results" / "predicted_motion_lib_epoch_3.pt")
    assert torch.equal(saved["motion_num_frames"], motion_lens)
    assert torch.equal(saved["length_starts"], torch.tensor([0, 2, 3]))
    assert saved["gts"].shape == (3, 1, 3)
    assert torch.allclose(
        saved["gts"][:, 0],
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 32.5],
            ]
        ),
    )
    assert saved["contacts"].dtype == torch.bool
    assert torch.equal(
        saved["contacts"].flatten(),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(saved["motion_weights"], evaluator.motion_lib.motion_weights)
    assert saved["motion_files"] == evaluator.motion_lib.motion_files
