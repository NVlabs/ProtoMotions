# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real experiment construction preserves benchmark cuts, not env terminations."""

from argparse import Namespace
import importlib
import pickle
from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import MaskedMimicActionSource, MaskedMimicEvaluatorConfig
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config


def test_masked_mimic_inference_defaults_select_the_prior():
    """The prior action path is a framework inference default, not a recipe hook line."""
    from protomotions.agents.supervised.masked_mimic_config import (
        MaskedMimicSupervisedAgentConfig,
    )

    agent = MaskedMimicSupervisedAgentConfig(batch_size=1, training_max_steps=1)
    agent.evaluator = MaskedMimicEvaluatorConfig()
    env = SimpleNamespace(observation_components={})

    assert agent.evaluator.action_source is MaskedMimicActionSource.POSTERIOR
    agent.apply_inference_defaults(env)
    assert agent.evaluator.action_source is MaskedMimicActionSource.PRIOR

    agent.evaluator.validate()
    agent.evaluator.action_source = "invalid"
    with pytest.raises(ValueError, match="action_source"):
        agent.evaluator.validate()


def test_action_source_cli_override_and_pickle():
    from protomotions.utils.config_utils import apply_overrides
    from protomotions.utils.experiment_configs import ExperimentConfigs

    agent = SimpleNamespace(evaluator=MaskedMimicEvaluatorConfig())
    apply_overrides(
        ExperimentConfigs(None, None, None, None, None, None, agent),
        {"agent.evaluator.action_source": "prior"},
    )
    assert agent.evaluator.action_source is MaskedMimicActionSource.PRIOR
    restored = pickle.loads(pickle.dumps(agent.evaluator))
    assert restored.action_source is MaskedMimicActionSource.PRIOR
    restored.validate()


@pytest.mark.parametrize(
    "name",
    [
        "mimic.mlp",
        "mimic.fsq",
        "mimic.mlp_complex_terrain",
        "mimic.mlp_bm_l2c2",
        "add.mlp",
        "steering.mlp",
        "path_follower.mlp",
    ],
)
def test_public_recipe_scoring_contract(name):
    module = importlib.import_module(f"examples.experiments.{name}")
    args = Namespace(
        robot_name="g1",
        simulator="mujoco",
        motion_file="unused.pt",
        scenes_file=None,
        num_envs=4,
        batch_size=32,
        training_max_steps=1024,
        headless=True,
        experiment_name="eval_contract",
    )
    robot = robot_config("g1")
    simulator = simulator_config("mujoco", robot, True, 4, "eval_contract")
    configure = getattr(module, "configure_robot_and_simulator", None)
    if configure is not None:
        configure(robot, simulator, args)
    env_config = module.env_config(robot, args)
    config = module.agent_config(robot, env_config, args).evaluator
    config.validate()
    # Construct the evaluator without creating a simulator.
    evaluator = BaseEvaluator(
        SimpleNamespace(env=SimpleNamespace(config=env_config, dt=0.02)),
        SimpleNamespace(device=torch.device("cpu"), world_size=1),
        config,
    )
    assert evaluator.get_state_dict() == {"eval_count": 0}
    assert config.score_key == "eval/success_rate"
    assert "evaluation_components" not in vars(config)
    if name == "path_follower.mlp":
        # The benchmark cuts the scalar distance at 0.5 m; the control's own
        # episode-ending check stays at 1.0 m.
        rule = config.failure_components["path_distance"]
        assert rule.component.static_params == {"min_progress": 10}
        assert (rule.threshold, rule.fail_above) == (0.5, True)
        assert env_config.control_components["path"].fail_dist == 1.0
    elif name == "steering.mlp":
        too_fast = config.failure_components["steering_speed_error"]
        off_course = config.failure_components["steering_direction_alignment"]
        assert set(config.failure_components) == set(config.metric_components)
        assert (too_fast.threshold, too_fast.fail_above) == (0.5, True)
        assert (off_course.threshold, off_course.fail_above) == (0.7, False)
        assert too_fast.component.static_params == {}
        assert not env_config.termination_components
    elif name == "mimic.mlp_bm_l2c2":
        assert set(config.failure_components) == {"anchor_height_error"}
        assert all(
            c.threshold == 0.25
            for c in config.failure_components.values()
        )
    else:
        assert set(config.metric_components) == {
            "gt_error",
            "gr_error",
            "max_joint_error",
        }
        assert set(config.failure_components) == {"gt_error"}
        assert (
            config.failure_components["gt_error"].threshold
            == 0.5
        )
        if name == "add.mlp":
            assert not env_config.termination_components
