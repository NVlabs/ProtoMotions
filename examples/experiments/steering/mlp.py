# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import RobotConfig
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

    scene_file = args.scenes_file
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults).
    
    Uses MdpComponent-based component configuration with explicit context bindings.
    """
    from protomotions.envs.control.steering_control import SteeringControlConfig
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_max_coords_from_state,
    )

    # Import compute kernels (steering-specific)
    from protomotions.envs.obs.steering import compute_steering_obs
    from protomotions.envs.rewards.task import compute_heading_velocity_rew
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.action import make_pd_action_config

    # Control components - steering manages task state (direction/speed)
    control_components = {
        "steering": SteeringControlConfig(),
    }

    # Observation components configuration
    observation_components = {
        # Humanoid self-observations (current state)
        "max_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        # Historical observations for AMP discriminator (from StateHistoryBuffer)
        "historical_max_coords_obs": MdpComponent.from_kernel(
            compute_historical_max_coords_from_state,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=8,
        ),
        # Steering observation (from control component context)
        "steering": MdpComponent(
            compute_func=compute_steering_obs,
            dynamic_vars={
                # The root on purpose, not the kernel's default anchor_rot: the
                # steering target is a root heading (on SMPL the anchor is the
                # root anyway, so the recorded traces do not move).
                "root_rot": EnvContext.current.root_rot,
                "tar_dir": EnvContext.steering.tar_dir,
                "tar_speed": EnvContext.steering.tar_speed,
                "tar_face_dir": EnvContext.steering.tar_face_dir,
            },
        ),
    }

    # Reward configuration using the reward component system
    reward_components = {
        # Primary steering reward - heading and velocity matching
        # The kernel reads the robot's anchor body, the one the steering
        # control tracks, through its own defaults.
        "heading_rew": MdpComponent.from_kernel(
            compute_heading_velocity_rew, meta=RewardMeta(weight=1.0)
        ),
    }

    env_cfg = EnvConfig(
        max_episode_length=300,
        control_components=control_components,
        observation_components=observation_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
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
    from protomotions.envs.obs.humanoid_historical import compute_historical_max_coords_from_motion_lib
    from protomotions.agents.evaluators.config import EvaluatorConfig

    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.terminations.task import (
        steering_direction_alignment,
        steering_speed_error,
    )

    # For steering with AMP: actor/critic get steering obs, discriminator uses historical body state
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "steering", "historical_max_coords_obs"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[
                MLPLayerConfig(units=1024, activation="relu"),
                MLPLayerConfig(units=512, activation="relu"),
            ],
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
    # Agent injects motion_lib/motion_ids/motion_times/dt at runtime (not in EnvContext)
    from protomotions.envs.mdp_component import MdpComponent
    
    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},  # All parameters injected by agent
            static_params={"history_steps": 8},  # The ring keeps what this reads.
        ),
    }

    # Two scalar checks, logged as metrics; the failure rules below cut them.
    steering_checks = {
        "steering_speed_error": MdpComponent.from_kernel(steering_speed_error),
        "steering_direction_alignment": MdpComponent.from_kernel(
            steering_direction_alignment
        ),
    }

    agent_cfg = AMPAgentConfig(
        model=AMPModelConfig(
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
        evaluator=EvaluatorConfig(
            metric_components=steering_checks,
            failure_components={
                "steering_speed_error": TerminationRule(
                    steering_checks["steering_speed_error"], 0.5
                ),
                "steering_direction_alignment": TerminationRule(
                    steering_checks["steering_direction_alignment"],
                    0.7,
                    fail_above=False,
                ),
            },
            score_key="eval/success_rate",
            max_eval_steps=300,
        ),
    )
    return agent_cfg
