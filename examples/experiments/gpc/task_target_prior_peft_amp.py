# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLFT a PEFT-adapted GPC prior for target reaching with AMP rewards.

This is the target-reaching PEFT setup plus reusable AMP discriminator reward
and discriminator critic modules.
"""

import argparse

from examples.experiments.gpc.prior_context import nearest_surface_obs_params
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


DISC_HISTORY_STEPS = [1, 2, 4, 8, 16]


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--prior-checkpoint", required=True)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        historical_max_coords_obs_factory,
        max_coords_obs_factory,
        nearest_surface_obs_factory,
        target_obs_factory,
        target_reward_factory,
    )
    from protomotions.envs.control.target_control import (
        RandomTargetCommandSourceConfig,
        TargetControlConfig,
    )
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "task_obs": target_obs_factory(),
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            history_steps=DISC_HISTORY_STEPS
        ),
        "nearest_surface": nearest_surface_obs_factory(
            **nearest_surface_obs_params(robot_cfg),
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=256,
        num_state_history_steps=max(DISC_HISTORY_STEPS),
        reset_grace_period=0,
        ref_respawn_offset=0.0,
        control_components={
            "target": TargetControlConfig(
                command_source=RandomTargetCommandSourceConfig(
                    tar_change_time_min=2.0,
                    tar_change_time_max=8.0,
                    tar_dist_max=6.0,
                ),
                tar_proximity_threshold=0.25,
                enable_fall_termination=False,
                enable_gap_termination=False,
                enable_stuck_termination=False,
            ),
        },
        observation_components=observation_components,
        reward_components={
            "target": target_reward_factory(weight=1.0),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
):
    from protomotions.agents.amp.config import AMPParametersConfig, DiscriminatorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        PretrainedModelConfig,
    )
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.peft.prior_amp_config import (
        DiscretePriorPEFTRLFTAMPAgentConfig,
        DiscretePriorPEFTRLFTAMPModelConfig,
    )
    from protomotions.agents.peft.prior_config import (
        DiscretePriorPEFTConfig,
        DiscretePriorPEFTActorConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib

    prior_checkpoint = args.prior_checkpoint
    actor_in_keys = ["task_obs"]

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
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
            )
        ],
    )
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
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
            )
        ],
    )

    num_mini_epochs = 2
    discriminator_batch_size = max(1, args.batch_size // num_mini_epochs)

    return DiscretePriorPEFTRLFTAMPAgentConfig(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=prior_checkpoint,
                module_path="",
            ),
        },
        e_clip=0.2,
        tau=0.95,
        model=DiscretePriorPEFTRLFTAMPModelConfig(
            in_keys=["max_coords_obs", "task_obs", "historical_max_coords_obs"],
            out_keys=[
                "action",
                "mean_action",
                "neglogp",
                "prior_tokens",
                "value",
                "disc_logits",
                "disc_value",
            ],
            actor=DiscretePriorPEFTActorConfig(
                in_keys=actor_in_keys,
                peft=DiscretePriorPEFTConfig(
                    peft_type="dora",
                    rank=32,
                    alpha=64,
                    temperature=1.0,
                    top_p=0.9,
                    sampling_mode="prior_constraint",
                    prior_top_p=0.99,
                    kl_coeff=0.0,
                ),
            ),
            critic=MLPWithConcatConfig(
                in_keys=["max_coords_obs", "task_obs"],
                out_keys=["value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
            ),
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.AdamW",
                lr=1e-4,
                weight_decay=0.01,
            ),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
            discriminator=discriminator_config,
            disc_critic=disc_critic_config,
            discriminator_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
            disc_critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=32,
        num_mini_epochs=num_mini_epochs,
        normalize_rewards=True,
        gradient_clip_val=25.0,
        save_last_checkpoint_every=10,
        evaluator=EvaluatorConfig(eval_metrics_every=100),
        amp_parameters=AMPParametersConfig(
            discriminator_batch_size=discriminator_batch_size,
            discriminator_reward_w=0.1,
            discriminator_reward_threshold=0.02,
        ),
        reference_obs_components={
            "historical_max_coords_obs": MdpComponent(
                compute_func=compute_historical_max_coords_from_motion_lib,
                dynamic_vars={},
                static_params={
                    "num_state_history_steps": max(DISC_HISTORY_STEPS),
                    "history_steps": DISC_HISTORY_STEPS,
                },
            )
        },
    )


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
    agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0
