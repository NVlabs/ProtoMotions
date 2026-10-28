# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from protomotions.components.terrains.config import TerrainConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.amp.config import AMPAgentConfig, AMPParametersConfig
from protomotions.envs.presets import tracking_playback
from protomotions.utils.experiment_configs import ExperimentConfigs
from examples.experiments.amp.mlp import (
    apply_inference_overrides as amp_apply_inference_overrides,
)


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
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

    Binds the standard observation, reward and termination kernels directly
    (``MdpComponent.from_kernel``).
    """
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )
    from protomotions.envs.obs.target_poses import (
        build_reduced_coords_target_poses,
    )
    from protomotions.envs.action import make_pd_action_config

    # Control components - MimicControl provides ref_state for mimic target poses
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    # Observation components configuration
    observation_components = {
        "max_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations
        ),
        "mimic_target_poses": MdpComponent.from_kernel(
            build_reduced_coords_target_poses
        ),
    }

    return EnvConfig(
        max_episode_length=1000,
        control_components=control_components,
        observation_components=observation_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.9,
            resample_on_reset=False,  # We resample on reset to ensure the discriminator and policy observe the same data distribution (random).
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> AMPAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.amp.config import AMPModelConfig, DiscriminatorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig

    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations.tracking import (
        max_body_pos_error,
        mean_body_pos_error,
        mean_body_rot_error,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            out_keys=["actor_trunk_out"],
            normalize_obs=True,
            norm_clamp_value=5,
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    discriminator_config = DiscriminatorConfig(
        in_keys=["mimic_target_poses_diff"],
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=["mimic_target_poses_diff"],
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

    # Separate value baseline for discriminator rewards, not task rewards.
    disc_critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses"],
        out_keys=["disc_value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: AMPAgentConfig = AMPAgentConfig(
        _target_="protomotions.agents.mimic.agent_add.MimicADD",
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
        batch_size=args.batch_size,
        task_reward_w=0.0,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            metric_components={
                "gt_error": MdpComponent.from_kernel(mean_body_pos_error),
                "gr_error": MdpComponent.from_kernel(mean_body_rot_error),
                "max_joint_error": MdpComponent.from_kernel(max_body_pos_error),
            },
            failure_components={
                "gt_error": TerminationRule(
                    MdpComponent.from_kernel(mean_body_pos_error), 0.5
                ),
            },
            score_key="eval/success_rate",
        ),
        amp_parameters=AMPParametersConfig(
            discriminator_reward_threshold=0.03,
        ),
    )
    return agent_config


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Play each clip to its end (mimic) and start it at frame 0 (amp)."""
    tracking_playback(configs.env)
    amp_apply_inference_overrides(configs, args)
