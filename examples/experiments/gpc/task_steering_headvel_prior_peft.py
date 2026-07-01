# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEFT fine-tuning of a GPC prior for heading and velocity steering."""

import argparse

from examples.experiments.gpc.prior_context import (
    add_peft_sampling_mode_argument,
    nearest_surface_obs_params,
    peft_sampling_mode_kwargs,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig



def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--prior-checkpoint", required=True)
    add_peft_sampling_mode_argument(parser)


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
        max_coords_obs_factory,
        nearest_surface_obs_factory,
        previous_actions_factory,
        steering_obs_factory,
        steering_reward_factory,
    )
    from protomotions.envs.control.steering_control import SteeringControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(),
        "task_obs": steering_obs_factory(),
        "nearest_surface": nearest_surface_obs_factory(
            **nearest_surface_obs_params(robot_cfg),
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        num_state_history_steps=1,
        max_episode_length=256,
        reset_grace_period=5,
        ref_respawn_offset=0.0,
        control_components={
            "steering": SteeringControlConfig(
                tar_speed_min=1.2,
                tar_speed_max=4.0,
                heading_change_steps_min=40,
                heading_change_steps_max=150,
                random_heading_probability=0.2,
                standard_heading_change=1.57,
                standard_speed_change=0.3,
                stop_probability=0.05,
                enable_rand_facing=True,
            ),
        },
        observation_components=observation_components,
        reward_components={
            "steering": steering_reward_factory(weight=1.0),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        PretrainedModelConfig,
    )
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.peft.prior_config import (
        DiscretePriorPEFTConfig,
        DiscretePriorPEFTActorConfig,
        DiscretePriorPEFTRLFTAgentConfig,
        DiscretePriorPEFTRLFTModelConfig,
    )

    prior_checkpoint = args.prior_checkpoint
    actor_in_keys = ["task_obs"]

    return DiscretePriorPEFTRLFTAgentConfig(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=prior_checkpoint,
                module_path="",
            ),
        },
        e_clip=0.2,
        tau=0.95,
        model=DiscretePriorPEFTRLFTModelConfig(
            actor=DiscretePriorPEFTActorConfig(
                in_keys=actor_in_keys,
                peft=DiscretePriorPEFTConfig(
                    peft_type="dora",
                    rank=32,
                    alpha=64,
                    temperature=1.0,
                    **peft_sampling_mode_kwargs(args),
                    film_input_norm=True,
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
            actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=64,
        num_mini_epochs=2,
        normalize_rewards=True,
        gradient_clip_val=25.0,
        save_last_checkpoint_every=10,
        evaluator=EvaluatorConfig(eval_metrics_every=100),
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
    env_cfg.max_episode_length = 100000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
