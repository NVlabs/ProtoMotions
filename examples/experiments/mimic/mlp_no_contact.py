# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``mimic/mlp`` recipe without foot-contact tracking.

Identical to :mod:`examples.experiments.mimic.mlp` except that the robot declares
no contact bodies and the ``contact_match_rew`` reward is not bound. Use it on
IsaacGym, whose GPU pipeline reports one combined force per body and therefore
refuses any robot with contact bodies, and for motion files that carry no contact
labels. Observations, the remaining rewards, terminations and the agent are the
shared recipe's.
"""

import argparse

from examples.experiments.mimic import mlp as _mlp
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

terrain_config = _mlp.terrain_config
scene_lib_config = _mlp.scene_lib_config
motion_lib_config = _mlp.motion_lib_config
agent_config = _mlp.agent_config
apply_inference_overrides = _mlp.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """The shared environment without the foot-contact reward."""
    config = _mlp.env_config(robot_cfg, args)
    del config.reward_components["contact_match_rew"]
    return config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Declare no contact bodies: nothing in this recipe reads contacts."""
