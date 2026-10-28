# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic SOMA23 masked-goal distillation from one frozen full-body teacher."""

from copy import deepcopy

from examples.experiments.masked_mimic import transformer
from examples.experiments.masked_mimic.transformer import (
    agent_config as agent_config,
    apply_inference_overrides as apply_inference_overrides,
    motion_lib_config as motion_lib_config,
    scene_lib_config as scene_lib_config,
)
from protomotions.utils import experiment_configs


def additional_experiment_arguments(parser):
    parser.add_argument(
        "--expert-model-path", required=True, help="Frozen SOMA23 teacher checkpoint"
    )


def _teacher(args):
    if not args.expert_model_path:
        raise ValueError("Training requires --expert-model-path")
    return experiment_configs.load_experiment_configs_from_checkpoint(
        args.expert_model_path
    )


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    if args.robot_name != "soma23":
        raise ValueError("The SOMA student requires --robot-name soma23")
    teacher = _teacher(args)
    for field in ("body_names", "dof_names"):
        if getattr(robot_cfg.kinematic_info, field) != getattr(
            teacher.robot.kinematic_info, field
        ):
            raise ValueError(f"Student and teacher {field} must match exactly")
    # Distillation must execute the teacher's actions with the same physics.
    robot_cfg.control = deepcopy(teacher.robot.control)
    robot_cfg.reset_noise = deepcopy(teacher.robot.reset_noise)
    robot_cfg.asset.self_collisions = teacher.robot.asset.self_collisions
    robot_cfg.contact_bodies = deepcopy(teacher.robot.contact_bodies)
    robot_cfg.contact_partners = deepcopy(teacher.robot.contact_partners)
    for field in (
        "domain_randomization",
        "default_robot_friction",
        "pd_target_max_accel",
    ):
        setattr(simulator_cfg, field, deepcopy(getattr(teacher.simulator, field)))
    if type(simulator_cfg) is type(teacher.simulator):
        simulator_cfg.sim = deepcopy(teacher.simulator.sim)


def terrain_config(args):
    return deepcopy(_teacher(args).terrain)


def env_config(robot_cfg, args):
    teacher = _teacher(args)
    env = transformer.env_config(robot_cfg, args)
    env.action_config = deepcopy(teacher.env.action_config)
    return env
