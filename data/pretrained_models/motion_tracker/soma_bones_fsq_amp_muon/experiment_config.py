# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local FSQ mimic tracker with AMP and light domain randomization.

SOMA23-oriented run target, but robot shapes are read from ``robot_cfg``. The
actor is the local-frame FSQ tracker from ``mimic/fsq.py`` with a Muon actor
optimizer. AMP uses clean historical max-coords. Domain randomization is limited
to reset-pose noise and pushes only: no L2C2, friction, COM, action noise, or
observation noise. The FSQ decoder remains reusable as a GPC prior decoder.
"""

import argparse

from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


PROPRIO_ROOT_HEIGHT_OBS = False
NEAREST_SURFACE_TERRAIN_HORIZONTAL_SCALE = 0.1
POINTCLOUD_SAMPLES_PER_OBJECT = 256
DISC_HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]
FSQ_NUM_LEVEL = 9
FSQ_NUM_TOKEN = 40


def _nearest_surface_body_ids(robot_cfg: RobotConfig) -> list[int]:
    return [
        robot_cfg.kinematic_info.body_names.index(name)
        for name in robot_cfg.trackable_bodies_subset
    ]


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(
        scene_file=scene_file,
        pointcloud_samples_per_object=POINTCLOUD_SAMPLES_PER_OBJECT,
    )


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        historical_max_coords_obs_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        nearest_surface_obs_factory,
        pow_rew_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        reset_grace_period=5,
        ref_respawn_offset=0.0,
        num_state_history_steps=max(DISC_HISTORY_STEPS),
        control_components={
            "mimic": MimicControlConfig(
                bootstrap_on_episode_end=True,
                future_steps=[1, 2, 5, 7, 12, 18, 25],
            )
        },
        observation_components={
            "max_coords_obs": max_coords_obs_factory(
                root_height_obs=PROPRIO_ROOT_HEIGHT_OBS,
            ),
            "nearest_surface": nearest_surface_obs_factory(
                terrain_horizontal_scale=NEAREST_SURFACE_TERRAIN_HORIZONTAL_SCALE,
                body_ids=_nearest_surface_body_ids(robot_cfg),
            ),
            "mimic_target_poses": mimic_target_poses_max_coords_factory(
                with_velocities=True,
            ),
            "historical_max_coords_obs": historical_max_coords_obs_factory(
                use_noisy=False,
                local_obs=True,
                root_height_obs=PROPRIO_ROOT_HEIGHT_OBS,
                observe_contacts=False,
                history_steps=DISC_HISTORY_STEPS,
            ),
        },
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.5),
        },
        reward_components={
            **mimic_tracking_rewards_factory(
                gt_weight=0.5,
                gr_weight=0.3,
                gv_weight=0.1,
                gav_weight=0.1,
                rh_weight=0.2,
                gt_coef=-100.0,
                gr_coef=-5.0,
                gv_coef=-0.5,
                gav_coef=-0.1,
                rh_coef=-100.0,
            ),
            "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> "AMPAgentConfig":
    from protomotions.agents.amp.config import (
        AMPAgentConfig,
        AMPModelConfig,
        AMPParametersConfig,
        DiscriminatorConfig,
    )
    from protomotions.agents.base_agent.config import (
        MuonWithAuxAdamConfig,
        OptimizerConfig,
    )
    from protomotions.agents.common.fsq_config import FSQAutoEncoderConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
    )
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        AdaptiveLRConfig,
        AdvantageNormalizationConfig,
        PPOActorConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib

    encoder_config = MLPWithConcatConfig(
        in_keys=["mimic_target_poses"],
        out_keys=["latent"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=FSQ_NUM_TOKEN,
        layers=[
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=1024, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
    )

    decoder_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "nearest_surface", "latent"],
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
        num_fsq_levels=FSQ_NUM_LEVEL,
        num_fsq_scalars=FSQ_NUM_TOKEN,
        encoder_out_keys=["latent"],
        decoder_out_keys=["mu"],
        encoder=encoder_config,
        decoder=decoder_config,
    )
    actor_config = PPOActorConfig(
        mu_key="mu",
        in_keys=["max_coords_obs", "nearest_surface", "mimic_target_poses"],
        mu_model=fsq_config,
        num_out=robot_config.number_of_actions,
        actor_logstd=-2.9,
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "nearest_surface", "mimic_target_poses"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

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
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            )
        ],
    )

    disc_critic_config = ModuleContainerConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
                out_keys=["disc_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=256, activation="relu"),
                ],
            )
        ],
    )

    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},
            static_params={
                "num_state_history_steps": max(DISC_HISTORY_STEPS),
                "local_obs": True,
                "root_height_obs": PROPRIO_ROOT_HEIGHT_OBS,
                "observe_contacts": False,
                "history_steps": DISC_HISTORY_STEPS,
            },
        ),
    }

    return AMPAgentConfig(
        model=AMPModelConfig(
            in_keys=[
                "max_coords_obs",
                "nearest_surface",
                "mimic_target_poses",
                "historical_max_coords_obs",
            ],
            out_keys=[
                "action",
                "mean_action",
                "neglogp",
                "value",
                "disc_logits",
                "disc_value",
            ],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            disc_critic=disc_critic_config,
            actor_optimizer=MuonWithAuxAdamConfig(
                lr=5e-4,
                weight_decay=0.01,
                momentum=0.95,
                adam_lr=1e-4,
                adam_betas=(0.9, 0.95),
                adam_eps=1e-10,
                adam_weight_decay=0.01,
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4, betas=(0.95, 0.99)
            ),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        reference_obs_components=reference_obs_components,
        normalize_rewards=False,
        task_reward_w=0.5,
        amp_parameters=AMPParametersConfig(
            discriminator_reward_w=2.0,
            discriminator_reward_threshold=0.02,
        ),
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        batch_size=args.batch_size,
        num_mini_epochs=2,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        save_inference_checkpoint=True,
        evaluator=MimicEvaluatorConfig(
            eval_metrics_every=200,
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
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
    from protomotions.simulator.base_simulator.config import (
        DomainRandomizationConfig,
        PushDomainRandomizationConfig,
        RobotNoiseConfig,
    )

    # Match IsaacLab's implicit robot material when this policy was trained.
    simulator_cfg.default_robot_friction = 0.5
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.1,
        root_pos_noise=[0.05, 0.05, 0.01],
        root_rot_noise=[0.1, 0.1, 0.2],
        root_vel_noise=[0.1, 0.1, 0.05],
        root_ang_vel_noise=[0.1, 0.1, 0.1],
    )
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        push=PushDomainRandomizationConfig(
            push_interval_range=(1.0, 3.0),
            max_linear_velocity=(0.5, 0.5, 0.2),
            max_angular_velocity=(0.52, 0.52, 0.78),
        )
    )


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
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
    robot_cfg.reset_noise = None
    simulator_cfg.domain_randomization = None
