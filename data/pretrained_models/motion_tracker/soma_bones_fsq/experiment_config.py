# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FSQ tracker over max-coordinates proprioception and target poses."""

import argparse

from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.presets import tracking_playback
from protomotions.utils.experiment_configs import ExperimentConfigs


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )
    from protomotions.envs.obs.target_poses import (
        build_max_coords_target_poses,
    )
    from protomotions.envs.body_selection import BodySelection
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import (
        compute_pow_rew,
        compute_terrain_foot_contact_match_rew,
    )
    from protomotions.envs.rewards.tracking import (
        compute_gav_rew,
        compute_gr_rew,
        compute_gt_rew,
        compute_gv_rew,
        compute_rh_rew,
    )
    from protomotions.envs.terminations.tracking import max_body_pos_error
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    foot_contact_bodies = (
        robot_cfg.common_naming_to_robot_body_names["all_left_foot_bodies"]
        + robot_cfg.common_naming_to_robot_body_names["all_right_foot_bodies"]
    )
    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        reset_grace_period=5,
        ref_respawn_offset=0.0,
        control_components={
            "mimic": MimicControlConfig(
                bootstrap_on_episode_end=True,
                future_steps=[1, 2, 5, 7, 12, 18, 25],
            )
        },
        observation_components={
            "max_coords_obs": MdpComponent.from_kernel(
                compute_humanoid_max_coords_observations, root_height_obs=True
            ),
            "mimic_target_poses": MdpComponent.from_kernel(
                build_max_coords_target_poses,
                with_velocities=True,
            ),
        },
        termination_components={
            "tracking_error": TerminationRule(
                MdpComponent.from_kernel(max_body_pos_error), 0.5
            ),
        },
        reward_components={
            "gt_rew": MdpComponent.from_kernel(
                compute_gt_rew,
                meta=RewardMeta(weight=0.5),
                coefficient=-100.0,
            ),
            "gr_rew": MdpComponent.from_kernel(
                compute_gr_rew,
                meta=RewardMeta(weight=0.3),
                coefficient=-5.0,
            ),
            "gv_rew": MdpComponent.from_kernel(
                compute_gv_rew,
                meta=RewardMeta(weight=0.1),
                coefficient=-0.5,
            ),
            "gav_rew": MdpComponent.from_kernel(
                compute_gav_rew,
                meta=RewardMeta(weight=0.1),
                coefficient=-0.1,
            ),
            "rh_rew": MdpComponent.from_kernel(
                compute_rh_rew,
                meta=RewardMeta(weight=0.2),
                coefficient=-100.0,
            ),
            "pow_rew": MdpComponent.from_kernel(
                compute_pow_rew,
                meta=RewardMeta(weight=-1e-5, min_value=-0.5),
            ),
            "contact_match_rew": MdpComponent.from_kernel(
                compute_terrain_foot_contact_match_rew,
                meta=RewardMeta(weight=-0.1, zero_during_grace_period=True),
                required_keys={
                    "contact_body_index": foot_contact_bodies,
                    "contact_partner_index": ["terrain"],
                },
                contact_body_ids=BodySelection.named(*foot_contact_bodies),
                body_names=foot_contact_bodies,
            ),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> "PPOAgentConfig":
    from protomotions.agents.base_agent.config import (
        MuonWithAuxAdamConfig,
        OptimizerConfig,
    )
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
    )
    from protomotions.agents.common.fsq_config import FSQAutoEncoderConfig

    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        AdvantageNormalizationConfig,
        PPOActorConfig,
        PPOAgentConfig,
        PPOModelConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations.tracking import (
        max_body_pos_error,
        mean_body_pos_error,
        mean_body_rot_error,
    )
    from protomotions.envs.terminations.rule import TerminationRule

    num_fsq_levels = 9
    num_fsq_scalars = 40

    encoder_config = MLPWithConcatConfig(
        in_keys=["mimic_target_poses"],
        out_keys=["latent"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=num_fsq_scalars,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
    )

    decoder_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "latent"],
        out_keys=["mu"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=robot_config.number_of_actions,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
    )

    fsq_config = FSQAutoEncoderConfig(
        num_fsq_levels=num_fsq_levels,
        num_fsq_scalars=num_fsq_scalars,
        encoder_out_keys=["latent"],
        decoder_out_keys=["mu"],
        encoder=encoder_config,
        decoder=decoder_config,
    )
    actor_config = PPOActorConfig(
        mu_key="mu",
        in_keys=["max_coords_obs", "mimic_target_poses"],
        mu_model=fsq_config,
        num_out=robot_config.number_of_actions,
        actor_logstd=-2.9,
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=MuonWithAuxAdamConfig(
                lr=5e-4,
                weight_decay=0.01,
                momentum=0.95,
                adam_lr=1e-4,
                adam_betas=(0.9, 0.95),
                adam_eps=1e-10,
                adam_weight_decay=0.01,
            ),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        save_inference_checkpoint=True,
        evaluator=MimicEvaluatorConfig(
            eval_metrics_every=200,
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
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    # Match IsaacLab's implicit robot material when this policy was trained.
    simulator_cfg.default_robot_friction = 0.5
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"],
        contact_partners=["terrain"],
    )


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Play each clip through to its end for evaluation."""
    tracking_playback(configs.env)
