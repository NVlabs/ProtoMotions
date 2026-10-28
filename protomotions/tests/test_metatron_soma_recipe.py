# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep animation-character physics separate from the G1 expert recipe."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.experiments.metatron import soma_student, soma_teacher
from examples.experiments.mimic import fsq
from protomotions.robot_configs.soma23 import Soma23RobotConfig
from protomotions.simulator.factory import simulator_config
from protomotions.utils.config_schema import to_tree
from protomotions.utils.experiment_configs import (
    ExperimentConfigs,
    load_experiment_configs,
    save_experiment_configs,
)


def test_soma_preserves_character_action_mapping_and_motors(tmp_path):
    args = SimpleNamespace(
        robot_name="soma23",
        simulator="mujoco",
        motion_file="soma_motions.pt",
        scenes_file=None,
        batch_size=4,
        training_max_steps=64,
    )
    robot = Soma23RobotConfig()
    sim = simulator_config("mujoco", robot, True, 1, "test")
    control = deepcopy(robot.control)
    soma_teacher.configure_robot_and_simulator(robot, sim, args)
    assert robot.control == control
    assert robot.asset.self_collisions
    assert robot.reset_noise is None
    assert sim.domain_randomization is None
    env = soma_teacher.env_config(robot, args)
    action = to_tree(env.action_config, where="action")
    assert action == to_tree(fsq.env_config(robot, args).action_config, where="action")
    agent = soma_teacher.agent_config(robot, env, args)
    assert agent.num_mini_epochs == 1
    bundle = ExperimentConfigs(
        robot,
        sim,
        soma_teacher.terrain_config(args),
        soma_teacher.scene_lib_config(args),
        soma_teacher.motion_lib_config(args),
        env,
        agent,
    )
    restored = load_experiment_configs(save_experiment_configs(bundle, tmp_path))
    assert restored.robot.control == robot.control
    assert to_tree(restored.env.action_config, where="action") == action
    assert restored.motion_lib.motion_file == "soma_motions.pt"
    assert restored.simulator.domain_randomization is None


def test_soma_rejects_a_robot_recipe_mixup():
    with pytest.raises(ValueError, match="soma23"):
        soma_teacher.configure_robot_and_simulator(
            None, None, SimpleNamespace(robot_name="g1")
        )


def test_soma_student_inherits_teacher_contract_and_roundtrips(tmp_path, monkeypatch):
    from protomotions.utils import experiment_configs
    from protomotions.agents.supervised.expert_utils import (
        validate_expert_action_contract,
    )
    from protomotions.utils.inference_utils import derive_inference_configs

    args = SimpleNamespace(
        robot_name="soma23",
        simulator="mujoco",
        motion_file="soma_motions.pt",
        scenes_file=None,
        batch_size=4,
        training_max_steps=64,
        expert_model_path="teacher/last.ckpt",
        headless=True,
        full_eval=False,
    )
    robot = Soma23RobotConfig()
    sim = simulator_config("mujoco", robot, True, 1, "test")
    soma_teacher.configure_robot_and_simulator(robot, sim, args)
    teacher_env = soma_teacher.env_config(robot, args)
    teacher = ExperimentConfigs(
        robot,
        sim,
        soma_teacher.terrain_config(args),
        soma_teacher.scene_lib_config(args),
        soma_teacher.motion_lib_config(args),
        teacher_env,
        soma_teacher.agent_config(robot, teacher_env, args),
    )
    monkeypatch.setattr(
        experiment_configs, "load_experiment_configs_from_checkpoint", lambda _: teacher
    )
    student_robot, student_sim = deepcopy(robot), deepcopy(sim)
    soma_student.configure_robot_and_simulator(student_robot, student_sim, args)
    env = soma_student.env_config(student_robot, args)
    agent = soma_student.agent_config(student_robot, env, args)
    validate_expert_action_contract(robot, teacher_env, student_robot, env)
    assert student_sim.sim == sim.sim
    assert student_sim.domain_randomization is None
    assert student_robot.control == robot.control
    assert student_robot.control is not robot.control
    assert agent.expert_model_path == args.expert_model_path
    assert {
        key[len("expert_") :]
        for key in env.observation_components
        if key.startswith("expert_")
    } == set(teacher.agent.model.actor.in_keys)
    configs = ExperimentConfigs(
        student_robot,
        student_sim,
        soma_student.terrain_config(args),
        soma_student.scene_lib_config(args),
        soma_student.motion_lib_config(args),
        env,
        agent,
    )
    rebuilt = load_experiment_configs(save_experiment_configs(configs, tmp_path))
    view = derive_inference_configs(
        rebuilt, args, {}, [soma_student.apply_inference_overrides]
    )
    assert view.agent.expert_model_path is None
    assert not any(key.startswith("expert_") for key in view.env.observation_components)


def test_soma_student_requires_matching_robot_and_teacher():
    with pytest.raises(ValueError, match="soma23"):
        soma_student.configure_robot_and_simulator(
            None, None, SimpleNamespace(robot_name="g1")
        )
    with pytest.raises(ValueError, match="expert-model-path"):
        soma_student.terrain_config(SimpleNamespace(expert_model_path=None))
