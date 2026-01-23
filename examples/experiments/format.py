# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse


"""
Config System Overview
======================

TRAINING - Config Building Process:
1. robot_cfg = robot_factory()                                    # Factory creates robot config
2. simulator_cfg = simulator_factory()                            # Factory creates simulator config
3. configure_robot_and_simulator(robot_cfg, simulator_cfg, args)  # Customize for this experiment
4. env_cfg = env_config(robot_cfg, args)                          # Build env config
5. agent_cfg = agent_config(robot_cfg, env_cfg, args)             # Build agent config
6. Apply CLI overrides (--overrides) if provided                  # Optional modifications
7. Save all to resolved_configs.pt                                # Frozen for resume/eval

EVALUATION - Override Process:
1. Load frozen configs from resolved_configs.pt                   # Already has training steps 1-6 baked in
2. apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg, args)  # Experiment-specific eval settings (optional)
3. Apply CLI overrides (--overrides) if provided                  # Highest priority

CLI overrides during training are PERMANENT (saved to resolved_configs.pt)!
"""


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """
    Configure robot and simulator for this experiment.

    Called during training AFTER factory creation, BEFORE env/agent configs are built.
    Results are saved to resolved_configs.pt (NOT called during eval - uses frozen configs).

    This is part of config BUILDING, not overrides. Think of it like env_config() and agent_config(),
    but for robot and simulator (which have factories instead of being built from scratch).

    Args:
        robot_cfg: Robot configuration object (from factory)
        simulator_cfg: Simulator configuration object (from factory)
        args: Command line arguments

    Examples:
        # Robot configuration
        robot_cfg.asset.asset_file_name = "mjcf/g1_bm.xml"
        robot_cfg.asset.self_collisions = False
        robot_cfg.update_fields(contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"])

        # Simulator configuration
        simulator_cfg.domain_randomization = DomainRandomizationConfig(
            center_of_mass=CenterOfMassDomainRandomizationConfig(
                com_range={"x": (-0.025, 0.025)},
                body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],  # Must be a list, not a single string
            ),
        )
    """
    pass


def terrain_config(args: argparse.Namespace):
    """
    Build terrain configuration (optional).

    Returns terrain config or None. If None, train_agent will create a default TerrainConfig.
    """
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """
    Build scene library configuration.

    Returns SceneLibConfig with scene_file set (or None for empty SceneLib).
    Always returns a config - empty SceneLib will be created if scene_file is None.
    """
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """
    Build motion library configuration.

    Returns MotionLibConfig with motion_file set (or None for empty MotionLib).
    Always returns a config - empty MotionLib will be created if motion_file is None.
    """
    from protomotions.components.motion_lib import MotionLibConfig

    motion_file = args.motion_file if hasattr(args, "motion_file") else None
    return MotionLibConfig(motion_file=motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """
    Build environment configuration (training defaults).

    This creates the base config - all eval-specific changes go in apply_inference_overrides().
    """
    env_config = None
    return env_config


def agent_config(
    robot_cfg: RobotConfig, env_cfg: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """
    Build agent configuration (training defaults).

    This creates the base config - all eval-specific changes go in apply_inference_overrides().
    """
    agent_config = None
    return agent_config


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    args: argparse.Namespace,
):
    """
    Apply evaluation-specific overrides to configs.

    Use this when your experiment needs different settings for training vs evaluation.
    This keeps train and eval configurations in the same Python file for clarity.

    Called during evaluation AFTER loading frozen configs from resolved_configs.pt.

    Args:
        robot_cfg: Robot configuration to modify
        simulator_cfg: Simulator configuration to modify
        env_cfg: Environment configuration to modify
        agent_cfg: Agent configuration to modify (can be None)
        args: Command line arguments

    Examples:
        # Disable discriminator reward for AMP/ASE evaluation
        if agent_cfg is not None and hasattr(agent_cfg, 'amp_parameters'):
            agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0

        # Disable expert for masked mimic evaluation (no distillation during eval)
        if agent_cfg is not None and hasattr(agent_cfg, 'expert_config'):
            agent_cfg.expert_config = None
    """
    pass
