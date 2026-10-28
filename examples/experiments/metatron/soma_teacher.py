# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA23 animation expert: full-body FSQ/PPO, original motors, no randomization.

Reuse the shared character recipe rather than G1's sim-to-real controller.
Actions retain the SOMA pi*tanh PD mapping and asset-defined gains/limits.
Motion and scene files are caller-supplied; training starts from scratch unless
the caller explicitly passes a checkpoint.
"""

from examples.experiments.mimic import fsq
from examples.experiments.mimic.fsq import (
    apply_inference_overrides as apply_inference_overrides,
    env_config as env_config,
    motion_lib_config as motion_lib_config,
    scene_lib_config as scene_lib_config,
    terrain_config as terrain_config,
)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    if args.robot_name != "soma23":
        raise ValueError("The SOMA animation expert requires --robot-name soma23")
    fsq.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    simulator_cfg.domain_randomization = None
    robot_cfg.asset.self_collisions = True
    robot_cfg.reset_noise = None


def agent_config(robot_cfg, env_cfg, args):
    config = fsq.agent_config(robot_cfg, env_cfg, args)
    config.num_mini_epochs = 1
    return config
