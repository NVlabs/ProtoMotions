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
from protomotions.envs.path_follower.config import PathFollowerEnvConfig
from protomotions.agents.amp.config import AMPAgentConfig
import argparse


"""
Path Follower Environment Configuration with AMP
=================================================

Path following task where the agent follows predefined trajectories.
Uses AMP (Adversarial Motion Priors) to encourage natural motion from reference data.
The agent receives observations of future waypoints and is rewarded for staying close to the path.
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


def env_config(
    robot_cfg: RobotConfig, args: argparse.Namespace
) -> PathFollowerEnvConfig:
    """Build environment configuration (training defaults)."""
    from protomotions.envs.obs.config import (
        HumanoidObsConfig,
        MaxCoordsSelfObsConfig,
        PathObsConfig,
        PathGeneratorConfig,
    )
    from protomotions.envs.base_env.config import RewardComponentConfig
    from protomotions.envs.utils.rewards import path_following_reward

    # 2D path following (no height conditioning)
    path_obs = PathObsConfig(
        enabled=True,
        path_generator=PathGeneratorConfig(
            height_conditioned=False,
        ),
    )

    # Reward configuration using the reward component system
    reward_config = {
        # Primary path following reward - distance to target
        "path_rew": RewardComponentConfig(
            function=path_following_reward,
            variables={
                "head_pos": "head_pos",
                "tar_pos": "tar_pos",
                "height_conditioned": "height_conditioned",
            },
            weight=1.0,
        ),
    }

    env_cfg = PathFollowerEnvConfig(
        max_episode_length=300,
        path_obs=path_obs,
        humanoid_obs=HumanoidObsConfig(
            max_coords_obs=MaxCoordsSelfObsConfig(
                enabled=True,
                num_historical_steps=8,  # Historical obs for AMP discriminator
            ),
        ),
        reward_config=reward_config,
    )

    return env_cfg


def agent_config(
    robot_config: RobotConfig,
    env_config: PathFollowerEnvConfig,
    args: argparse.Namespace,
) -> AMPAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.amp.config import (
        AMPModelConfig,
        DiscriminatorConfig,
        AMPParametersConfig,
    )

    # For path following with AMP: actor/critic get path obs, discriminator uses historical body state
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "path", "historical_max_coords_obs"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=["max_coords_obs", "path", "historical_max_coords_obs"],
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
        in_keys=["max_coords_obs", "path", "historical_max_coords_obs"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ],
    )

    # Discriminator only sees historical body state (not path obs)
    discriminator_config = DiscriminatorConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["disc_logits"],
        input_models=[
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

    agent_cfg = AMPAgentConfig(
        model=AMPModelConfig(
            in_keys=["max_coords_obs", "path", "historical_max_coords_obs"],
            out_keys=["action", "mean_action", "neglogp", "value", "disc_logits"],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        batch_size=args.batch_size,
        task_reward_w=0.5,  # Balance between task reward (path following) and style reward (AMP)
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
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    # Disable AMP discriminator termination during evaluation
    if agent_cfg is not None and hasattr(agent_cfg, "amp_parameters"):
        agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0

    if env_cfg is not None:
        env_cfg.max_episode_length = 1000000
        # Disable path termination for longer evaluation runs
        if hasattr(env_cfg, "path_obs"):
            env_cfg.path_obs.enable_path_termination = False
