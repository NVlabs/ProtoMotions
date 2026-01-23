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
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.amp.config import AMPAgentConfig
import argparse


"""
Steering Environment Configuration with AMP
============================================

Locomotion steering task where the agent walks in a target direction at a target speed.
Uses AMP (Adversarial Motion Priors) to encourage natural motion from reference data.
The target direction and speed change periodically to encourage versatile locomotion.
"""


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults)."""
    from protomotions.envs.obs import max_coords_obs_factory, historical_max_coords_obs_factory, steering_obs_factory
    from protomotions.envs.control.steering_control import SteeringControlConfig
    from protomotions.envs.rewards import heading_velocity_reward_factory

    # Control components - steering manages task state (direction/speed)
    control_components = {
        "steering": SteeringControlConfig(),
    }

    # Observation components configuration
    observation_components = {
        # Humanoid self-observations (current state)
        "max_coords_obs": max_coords_obs_factory(),
        # Historical observations for AMP discriminator (from StateHistoryBuffer)
        "historical_max_coords_obs": historical_max_coords_obs_factory(),
        # Steering observation (from control component context)
        "steering": steering_obs_factory(),
    }

    # Reward configuration using the reward component system
    reward_components = {
        # Primary steering reward - heading and velocity matching
        "heading_rew": heading_velocity_reward_factory(weight=1.0),
    }

    env_cfg = EnvConfig(
        max_episode_length=300,
        num_state_history_steps=8,  # Historical obs for AMP discriminator
        control_components=control_components,
        observation_components=observation_components,
        reward_components=reward_components,
    )

    return env_cfg


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> AMPAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig, ModuleContainerConfig
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.amp.config import (
        AMPModelConfig,
        DiscriminatorConfig,
        AMPParametersConfig,
    )
    from protomotions.envs.obs import historical_max_coords_ref_obs_factory

    # For steering with AMP: actor/critic get steering obs, discriminator uses historical body state
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "steering", "historical_max_coords_obs"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=["max_coords_obs", "steering", "historical_max_coords_obs"],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[
                MLPLayerConfig(units=1024, activation="relu"),
                MLPLayerConfig(units=512, activation="relu"),
            ],
            output_activation="tanh",
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "steering", "historical_max_coords_obs"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ],
    )

    # Discriminator only sees historical body state (not steering obs)
    discriminator_config = DiscriminatorConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
                out_keys=["disc_logits"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            )
        ],
    )

    # Disc critic: value network for discriminator rewards
    disc_critic_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "historical_max_coords_obs"],
                out_keys=["disc_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            )
        ],
    )

    # Reference observation components for discriminator expert data
    reference_obs_components = {
        "historical_max_coords_obs": historical_max_coords_ref_obs_factory(),
    }

    agent_cfg = AMPAgentConfig(
        model=AMPModelConfig(
            in_keys=["max_coords_obs", "steering", "historical_max_coords_obs"],
            out_keys=["action", "mean_action", "neglogp", "value", "disc_logits", "disc_value"],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            disc_critic=disc_critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        reference_obs_components=reference_obs_components,
        batch_size=args.batch_size,
        task_reward_w=0.5,  # Balance between task reward (steering) and style reward (AMP)
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        amp_parameters=AMPParametersConfig(
            discriminator_reward_threshold=0.02,
            discriminator_reward_w=0.5,
        ),
    )
    return agent_cfg


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    # Disable AMP discriminator termination during evaluation
    if agent_cfg is not None and hasattr(agent_cfg, "amp_parameters"):
        agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0

    if env_cfg is not None:
        env_cfg.max_episode_length = 1000000
