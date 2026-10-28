# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.supervised.masked_mimic_config import MaskedMimicSupervisedAgentConfig
from protomotions.envs.presets import tracking_playback
from protomotions.envs.steps import normalize_steps
from protomotions.utils.experiment_configs import ExperimentConfigs
import argparse


# Global configuration for masked mimic
NUM_FUTURE_STEPS = 5
TOTAL_STORED_HISTORICAL_STEPS = 5  # How many historical steps we save
NUM_HISTORICAL_CONDITIONED_STEPS = 5  # From those, how many do we sub-sample


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add MaskedMimic-specific CLI arguments."""
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training"
    )


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
    
    Components bind their kernels with ``MdpComponent.from_kernel``: each kernel
    carries its context bindings as defaults, and overrides name its parameters.
    """
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
    from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
    from protomotions.envs.obs.masked_mimic import masked_mimic_goal_observations
    from protomotions.envs.obs.target_poses import build_max_coords_target_poses
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_max_coords_from_state,
        compute_historical_actions_from_state,
    )
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import compute_action_smoothness
    from protomotions.envs.rewards.tracking import (
        compute_gav_rew,
        compute_gr_rew,
        compute_gt_rew,
        compute_gv_rew,
        compute_rh_rew,
    )
    from protomotions.envs.terminations.tracking import max_body_pos_error
    from protomotions.envs.terminations.rule import TerminationRule
    
    from protomotions.envs.obs.humanoid_historical import compute_historical_poses_with_time
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "masked_mimic": MaskedMimicControlConfig(
            num_masked_future_steps=NUM_FUTURE_STEPS,
            future_steps=1,  # Might be increased if expert requires more
            bootstrap_on_episode_end=True,
            visible_target_pose_prob=0.8
        ),
    }

    # Observation components
    observation_components = {
        "max_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "historical_max_coords_obs": MdpComponent.from_kernel(
            compute_historical_max_coords_from_state,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=TOTAL_STORED_HISTORICAL_STEPS,
        ),
        "previous_actions": MdpComponent.from_kernel(
            compute_historical_actions_from_state, history_steps=1
        ),
        "mimic_target_poses": MdpComponent.from_kernel(
            build_max_coords_target_poses,
            with_velocities=True,
            with_relative=True,
        ),
        # The five timed-target observations of the goal prompt, laid out by
        # the control's canvas (its conditionable bodies, resolved at build).
        **masked_mimic_goal_observations(
            GoalCanvas.derive(control_components["masked_mimic"], robot_cfg)
        ),
        "historical_pose_obs": MdpComponent(
            compute_func=compute_historical_poses_with_time,
            dynamic_vars={
                "historical_rigid_body_pos": EnvContext.historical.rigid_body_pos,
                "historical_rigid_body_rot": EnvContext.historical.rigid_body_rot,
                "historical_rigid_body_vel": EnvContext.historical.rigid_body_vel,
                "historical_rigid_body_ang_vel": EnvContext.historical.rigid_body_ang_vel,
                "historical_ground_heights": EnvContext.historical.ground_heights,
                "dt": EnvContext.dt,
            },
            static_params={
                "history_steps": TOTAL_STORED_HISTORICAL_STEPS,
                "local_obs": True,
                "root_height_obs": True,
                "w_last": True,
            },
        ),
    }

    expert_model_path = args.expert_model_path
    if expert_model_path:
        from protomotions.agents.supervised.expert_utils import (
            get_expert_observation_components,
        )
        from protomotions.utils.experiment_configs import (
            load_experiment_configs_from_checkpoint,
        )

        expert_configs = load_experiment_configs_from_checkpoint(expert_model_path)
        expert_env_config = expert_configs.env
        expert_agent_config = expert_configs.agent
        
        if hasattr(expert_env_config, 'control_components') and expert_env_config.control_components:
            for ctrl_cfg in expert_env_config.control_components.values():
                expert_future = getattr(ctrl_cfg, 'future_steps', None)
                if expert_future is not None:
                    expert_steps = normalize_steps(expert_future)
                    masked_mimic_cfg = control_components["masked_mimic"]
                    if max(masked_mimic_cfg.future_steps) < max(expert_steps):
                        masked_mimic_cfg.future_steps = expert_steps
        
        expert_obs_components = get_expert_observation_components(
            expert_env_config,
            expert_agent_config,
            existing_obs_keys=list(observation_components.keys()),
        )
        observation_components.update(expert_obs_components)

    # Termination components configuration
    termination_components = {
        # Always ran at 0.5 m: the 0.25 m set here before never took effect.
        "tracking_error": TerminationRule(
            MdpComponent.from_kernel(max_body_pos_error), 0.5
        ),
    }

    # Reward components
    reward_components = {
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
        "gv_rew": MdpComponent.from_kernel(compute_gv_rew, meta=RewardMeta(weight=0.1)),
        "gav_rew": MdpComponent.from_kernel(
            compute_gav_rew,
            meta=RewardMeta(weight=0.1),
        ),
        "rh_rew": MdpComponent.from_kernel(compute_rh_rew, meta=RewardMeta(weight=0.2)),
        "action_smoothness": MdpComponent.from_kernel(
            compute_action_smoothness,
            meta=RewardMeta(weight=-0.02),
        ),
    }

    env_config: EnvConfig = EnvConfig(
        max_episode_length=1000,
        # Component-based configuration
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        # Motion manager configuration
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MaskedMimicSupervisedAgentConfig:
    from protomotions.agents.supervised.masked_mimic_config import (
        MaskedMimicModelConfig,
        MaskedMimicVAEConfig,
        VAENoiseType,
        KLDScheduleConfig,
    )
    from protomotions.agents.common.config import (
        ObsProcessorConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        MLPWithConcatConfig,
        TransformerConfig,
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig

    from protomotions.agents.evaluators.config import MaskedMimicEvaluatorConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations.tracking import (
        max_body_pos_error,
        mean_body_pos_error,
        mean_body_rot_error,
    )
    from protomotions.envs.terminations.rule import TerminationRule

    transformer_token_size = 512
    transformer_encoder_widths = 256
    vae_latent_dim = 64

    # Encoder: normalizes inputs → MLP trunk → mu/logvar heads (flat structure)
    encoder_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "masked_mimic_target_poses", "masked_mimic_target_bodies_masks", "masked_mimic_target_times", "masked_mimic_target_poses_masks"],
        out_keys=["encoder_mu", "encoder_logvar"],
        models=[
            # Normalizers (parallel - no dependencies)
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["mimic_target_poses"],
                out_keys=["mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["masked_mimic_target_poses_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["masked_mimic_target_times_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            # Trunk MLP (depends on normalizers)
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_norm", "mimic_target_poses_norm", "masked_mimic_target_poses_norm", "masked_mimic_target_bodies_masks", "masked_mimic_target_times_norm", "masked_mimic_target_poses_masks"],
                out_keys=["encoder_trunk_out"],
                num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(5)],
                output_activation="relu",
            ),
            # Output heads (parallel - both depend on trunk)
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_mu"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_logvar"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
        ],
    )

    # Prior: reshape inputs → encode to tokens → transformer → mu/logvar heads (flat structure)
    prior_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "masked_mimic_target_poses", "masked_mimic_target_masks", "masked_mimic_target_times", "masked_mimic_target_poses_masks", "historical_pose_obs"],
        out_keys=["prior_mu", "prior_logvar"],
        models=[
            # Reshape and normalize inputs (parallel)
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_poses"],
                out_keys=["target_poses_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_masks"],
                out_keys=["target_masks_seq"],
                normalize_obs=False,
                module_operations=[ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1])],
            ),
            ObsProcessorConfig(
                in_keys=["masked_mimic_target_times"],
                out_keys=["target_times_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            ObsProcessorConfig(
                in_keys=["historical_pose_obs"],
                out_keys=["historical_pose_obs_seq"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            # Token encoders (depend on reshaped inputs)
            MLPWithConcatConfig(
                in_keys=["max_coords_obs"],
                out_keys=["current_state_token"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=transformer_token_size,
                layers=[MLPLayerConfig(units=transformer_encoder_widths, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["target_poses_seq", "target_masks_seq", "target_times_seq"],
                out_keys=["masked_mimic_target_poses_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[MLPLayerConfig(units=transformer_encoder_widths, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_FUTURE_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["historical_pose_obs_seq"],
                out_keys=["historical_pose_obs_token"],
                normalize_obs=False,
                num_out=transformer_token_size,
                layers=[MLPLayerConfig(units=transformer_encoder_widths, activation="relu") for _ in range(2)],
                module_operations=[
                    ModuleOperationReshapeConfig(new_shape=["batch_size", NUM_HISTORICAL_CONDITIONED_STEPS, -1]),
                    ModuleOperationForwardConfig(),
                ],
            ),
            # Transformer (depends on tokens)
            TransformerConfig(
                in_keys=["current_state_token", "masked_mimic_target_poses_token", "historical_pose_obs_token", "masked_mimic_target_poses_masks"],
                out_keys=["transformer_out"],
                transformer_token_size=transformer_token_size,
                latent_dim=transformer_token_size,
                input_and_mask_mapping={"masked_mimic_target_poses_token": "masked_mimic_target_poses_masks"},
                output_activation="relu",
            ),
            # Output heads (parallel - both depend on transformer)
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_mu"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
            MLPWithConcatConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_logvar"],
                num_out=vae_latent_dim,
                layers=[MLPLayerConfig(units=256, activation="relu"), MLPLayerConfig(units=128, activation="relu")],
            ),
        ],
    )

    # Trunk: normalize inputs → MLP → actions (flat structure)
    trunk_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            # Normalizers (parallel)
            ObsProcessorConfig(
                in_keys=["max_coords_obs"],
                out_keys=["max_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["previous_actions"],
                out_keys=["previous_actions_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            # Output MLP (depends on normalizers + vae_latent)
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_norm", "previous_actions_norm", "vae_latent"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
            ),
        ],
    )

    # Main model configuration
    model_config = MaskedMimicModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        trunk=trunk_config,
        vae=MaskedMimicVAEConfig(
            vae_latent_dim=vae_latent_dim,
            vae_noise_type=VAENoiseType.NORMAL,
            kld_schedule=KLDScheduleConfig(start_epoch=500, end_epoch=2000),
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = MaskedMimicEvaluatorConfig(
        metric_components={
            "gt_error": MdpComponent.from_kernel(mean_body_pos_error),
            "gr_error": MdpComponent.from_kernel(mean_body_rot_error),
            "max_joint_error": MdpComponent.from_kernel(max_body_pos_error),
        },
        failure_components={
            "gt_error": TerminationRule(
                MdpComponent.from_kernel(mean_body_pos_error), 0.25
            ),
        },
        score_key="eval/success_rate",
    )

    # Get expert model path from args (set via --expert-model-path CLI argument)
    expert_model_path = args.expert_model_path
    
    # Agent configuration
    agent_config = MaskedMimicSupervisedAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=expert_model_path,
    )
    return agent_config


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Play each clip to its end for evaluation.

    The prior action path, dropping the expert and its observations, and the
    long episode are framework inference defaults now (the MaskedMimic
    supervised agent and EnvConfig apply them); only the clip playback stays.
    """
    tracking_playback(configs.env)
