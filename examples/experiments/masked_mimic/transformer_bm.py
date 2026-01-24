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
"""MaskedMimic transformer experiment with BeyondMimic-style rewards.

This experiment combines the MaskedMimic transformer-based architecture
with BeyondMimic reward formulations (Gaussian kernels) and domain randomization.
Designed to work with the mimic mlp_bm pre-trained expert for distillation.

Naming convention:
- transformer: MaskedMimic uses a transformer-based prior network
- bm: BeyondMimic-style rewards and domain randomization
"""

from protomotions.robot_configs.base import RobotConfig, ControlType
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    ObservationNoiseDomainRandomizationConfig,
    PushDomainRandomizationConfig,
    DomainRandomizationConfig,
)
from protomotions.components.terrains.config import (
    TerrainConfig,
    TerrainSimConfig,
    CombineMode,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.masked_mimic.config import MaskedMimicAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


# Global configuration for masked mimic
NUM_FUTURE_STEPS = 5
# Note: TOTAL_STORED_HISTORICAL_STEPS must match expert's num_state_history_steps for observation compatibility
TOTAL_STORED_HISTORICAL_STEPS = 3  # How many historical steps we save (matches mlp_bm expert)
NUM_HISTORICAL_CONDITIONED_STEPS = 3  # From those, how many do we sub-sample


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add MaskedMimic-specific CLI arguments."""
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training (e.g., results/mimic_mlp_bm/last.ckpt)"
    )


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration with low friction settings for BeyondMimic."""
    terrain_cfg = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=0.01,
            dynamic_friction=0.01,
            restitution=0.0,
            combine_mode=CombineMode.AVERAGE,
        )
    )
    return terrain_cfg


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration with BeyondMimic rewards and MaskedMimic control."""
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.obs import (
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
        max_coords_obs_factory,
        historical_max_coords_obs_factory,
        historical_actions_factory,
        historical_poses_with_time_reduced_coords_factory,
        mimic_target_poses_reduced_coords_factory,
        mimic_target_poses_max_coords_factory,
        previous_actions_factory,
    )
    from protomotions.envs.obs.general import passthrough_float_factory
    from protomotions.envs.obs.masked_mimic_obs_functions import (
        masked_mimic_target_poses_factory,
        target_masks_factory,
        target_time_offsets_factory,
    )
    from protomotions.envs.rewards import (
        action_smoothness_physical_factory,
        soft_pos_limit_rew_factory,
        global_anchor_pos_rew_factory,
        global_anchor_ori_rew_factory,
        relative_body_pos_rew_factory,
        relative_body_ori_rew_factory,
        global_body_lin_vel_rew_factory,
        global_body_ang_vel_rew_factory,
    )
    from protomotions.envs.terminations import (
        anchor_pos_error_factory,
        anchor_ori_error_factory,
        relative_body_pos_error_factory,
        tracking_error_factory,
    )

    # Control components configuration - MaskedMimic control
    control_components = {
        "masked_mimic": MaskedMimicControlConfig(
            num_masked_future_steps=NUM_FUTURE_STEPS,
            num_future_steps=1,  # Might be increased if expert requires more
            bootstrap_on_episode_end=True,
            time_alpha=2.0,
            time_beta=5.0,
            repeat_mask_probability=0.8,
            force_max_conditioned_bodies_prob=0.1,
            force_small_num_conditioned_bodies_prob=0.1,
            visible_target_pose_prob=0.8,
        ),
    }

    # Compute conditionable body IDs from robot config (baked into factories)
    conditionable_body_ids = [
        robot_cfg.kinematic_info.body_names.index(name)
        for name in robot_cfg.trackable_bodies_subset
    ]

    # MaskedMimic observation components with noisy reduced coordinates for robustness
    observation_components = {
        # Core MaskedMimic observations (with noise for domain randomization)
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(observation_noise=True),
        "noisy_historical_reduced_coords_obs": historical_reduced_coords_obs_factory(observation_noise=True),
        "noisy_mimic_reduced_coords_target_poses": mimic_target_poses_reduced_coords_factory(num_future_steps=1, observation_noise=True),
        "previous_actions": previous_actions_factory(),
        "historical_previous_actions": historical_actions_factory(),
        # Masked mimic specific observations
        "masked_mimic_target_poses": masked_mimic_target_poses_factory(conditionable_body_ids=conditionable_body_ids),
        "masked_mimic_target_masks": target_masks_factory(conditionable_body_ids=conditionable_body_ids),
        "masked_mimic_target_times": target_time_offsets_factory(),
        "historical_pose_obs": historical_poses_with_time_reduced_coords_factory(
            num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
            total_stored_historical_steps=TOTAL_STORED_HISTORICAL_STEPS,
        ),
        "masked_mimic_target_poses_masks": passthrough_float_factory(variable="masked_mimic_target_poses_masks"),
        "masked_mimic_target_bodies_masks": passthrough_float_factory(variable="masked_mimic_target_bodies_masks"),
        # Expert critic observations (mlp_bm critic uses max_coords)
        "max_coords_obs": max_coords_obs_factory(),
        "historical_max_coords_obs": historical_max_coords_obs_factory(),
        "mimic_max_coords_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
    }

    # Validate expert compatibility if expert model path is provided
    # Note: We use matching observation names (noisy_reduced_coords_obs, etc.) so the expert
    # can directly use our observations without needing separate expert_ prefixed components.
    expert_model_path = getattr(args, 'expert_model_path', None)
    if expert_model_path:
        from protomotions.agents.masked_mimic.utils import load_expert_configs
        
        expert_configs = load_expert_configs(expert_model_path)
        expert_env_config = expert_configs["env"]
        
        # Validate history steps are sufficient
        expert_history_steps = getattr(expert_env_config, 'num_state_history_steps', 0)
        assert TOTAL_STORED_HISTORICAL_STEPS >= expert_history_steps, (
            f"Insufficient history: current={TOTAL_STORED_HISTORICAL_STEPS}, expert requires={expert_history_steps}"
        )
        
        # Adjust num_future_steps if expert requires more
        if hasattr(expert_env_config, 'control_components') and expert_env_config.control_components:
            for ctrl_cfg in expert_env_config.control_components.values():
                expert_num_future = getattr(ctrl_cfg, 'num_future_steps', None)
                if expert_num_future is not None:
                    masked_mimic_cfg = control_components["masked_mimic"]
                    if masked_mimic_cfg.num_future_steps < expert_num_future:
                        masked_mimic_cfg.num_future_steps = expert_num_future

    # BeyondMimic-style termination conditions
    termination_components = {
        "bad_ref_pos": anchor_pos_error_factory(threshold=0.5),
        "bad_ref_ori": anchor_ori_error_factory(threshold=0.8),
        "bad_motion_body_pos": relative_body_pos_error_factory(threshold=0.25),
        "tracking_error": tracking_error_factory(threshold=0.25),
    }

    # BeyondMimic reward configuration with density-based body weights
    reward_components = {
        # Global anchor (root) position and orientation
        "global_anchor_pos": global_anchor_pos_rew_factory(weight=0.5, sigma=0.3),
        "global_anchor_ori": global_anchor_ori_rew_factory(weight=0.5, sigma=0.4),
        # Relative body position and orientation (density-weighted)
        "relative_body_pos": relative_body_pos_rew_factory(
            weight=1.0,
            sigma=0.3,
            use_density_weights=True,
        ),
        "relative_body_ori": relative_body_ori_rew_factory(
            weight=1.0,
            sigma=0.4,
            use_density_weights=True,
        ),
        # Global body velocities (density-weighted)
        "body_lin_vel": global_body_lin_vel_rew_factory(weight=1.0, sigma=1.0, use_density_weights=True),
        "body_ang_vel": global_body_ang_vel_rew_factory(weight=1.0, sigma=3.14, use_density_weights=True),
        # Regularization (physical units: radians/step)
        "action_rate": action_smoothness_physical_factory(weight=-0.1),
        "limits_dof_pos": soft_pos_limit_rew_factory(weight=-100.0),
    }

    return EnvConfig(
        ref_respawn_offset=0.01,
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=TOTAL_STORED_HISTORICAL_STEPS,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MaskedMimicAgentConfig:
    """Build MaskedMimic agent configuration with transformer-based prior."""
    from protomotions.agents.masked_mimic.config import (
        MaskedMimicModelConfig,
        VaeConfig,
        VaeNoiseType,
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
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig, MotionWeightsRulesConfig

    transformer_token_size = 512
    transformer_encoder_widths = 256
    vae_latent_dim = 64

    # Encoder: normalizes inputs → MLP trunk → mu/logvar heads (flat structure)
    encoder_config = ModuleContainerConfig(
        in_keys=["noisy_reduced_coords_obs", "noisy_mimic_reduced_coords_target_poses", "masked_mimic_target_poses", "masked_mimic_target_bodies_masks", "masked_mimic_target_times", "masked_mimic_target_poses_masks"],
        out_keys=["encoder_mu", "encoder_logvar"],
        models=[
            # Normalizers (parallel - no dependencies)
            ObsProcessorConfig(
                in_keys=["noisy_reduced_coords_obs"],
                out_keys=["noisy_reduced_coords_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            ObsProcessorConfig(
                in_keys=["noisy_mimic_reduced_coords_target_poses"],
                out_keys=["noisy_mimic_reduced_coords_target_poses_norm"],
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
                in_keys=["noisy_reduced_coords_obs_norm", "noisy_mimic_reduced_coords_target_poses_norm", "masked_mimic_target_poses_norm", "masked_mimic_target_bodies_masks", "masked_mimic_target_times_norm", "masked_mimic_target_poses_masks"],
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
        in_keys=["noisy_reduced_coords_obs", "masked_mimic_target_poses", "masked_mimic_target_masks", "masked_mimic_target_times", "masked_mimic_target_poses_masks", "historical_pose_obs"],
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
                in_keys=["noisy_reduced_coords_obs"],
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
        in_keys=["noisy_reduced_coords_obs", "previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        models=[
            # Normalizers (parallel)
            ObsProcessorConfig(
                in_keys=["noisy_reduced_coords_obs"],
                out_keys=["noisy_reduced_coords_obs_norm"],
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
                in_keys=["noisy_reduced_coords_obs_norm", "previous_actions_norm", "vae_latent"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(3)],
                output_activation="tanh",
            ),
        ],
    )

    # Main model configuration
    model_config = MaskedMimicModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        trunk=trunk_config,
        vae=VaeConfig(
            vae_latent_dim=vae_latent_dim,
            vae_noise_type=VaeNoiseType.NORMAL,
            kld_schedule=KLDScheduleConfig(start_epoch=500, end_epoch=2000),
        ),
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    evaluator_config = MimicEvaluatorConfig(
        motion_weights_rules=MotionWeightsRulesConfig(
            motion_weights_update_success_discount=0.999,
            motion_weights_update_failure_discount=0,
        ),
        eval_metric_keys=[
            "gt_err",
            "gr_err",
            "gr_err_degrees",
            "max_joint_err",
            "action_rate",
        ],
    )

    # Get expert model path from args (set via --expert-model-path CLI argument)
    expert_model_path = getattr(args, 'expert_model_path', None)
    
    # Agent configuration
    agent_config = MaskedMimicAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=expert_model_path,
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot and simulator with BeyondMimic domain randomization."""
    
    robot_cfg.control.control_type = ControlType.BUILT_IN_PD
    robot_cfg.control.action_scale = 1.0

    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.02, 0.02), dof_names=[".*"], dof_indices=None
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.6, 3.0),
            dynamic_friction_range=(0.6, 3.0),
            restitution_range=(0.0, 1.0),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
        observation_noise=ObservationNoiseDomainRandomizationConfig(
            dof_pos_noise=0.01,
            dof_vel_noise=0.5,
            anchor_ang_vel_noise=0.2,
            anchor_rot_noise=0.05,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(1.0, 3.0),
            max_linear_velocity=(0.5, 0.5, 0.2),
            max_angular_velocity=(0.52, 0.52, 0.78),
        ),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: MaskedMimicAgentConfig,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    from protomotions.envs.obs import (
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
    )
    from protomotions.agents.evaluators.config import EvaluatorConfig

    # Disable all termination components for inference
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}
    
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    # Restore normal friction for inference
    terrain_cfg.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
    simulator_cfg.domain_randomization = None

    # Disable observation noise for inference (use clean observations)
    if env_cfg is not None and hasattr(env_cfg, "observation_components") and env_cfg.observation_components is not None:
        env_cfg.observation_components["noisy_reduced_coords_obs"] = reduced_coords_obs_factory(observation_noise=False)
        env_cfg.observation_components["noisy_historical_reduced_coords_obs"] = historical_reduced_coords_obs_factory(observation_noise=False)
        env_cfg.observation_components["noisy_mimic_reduced_coords_target_poses"] = mimic_target_poses_reduced_coords_factory(num_future_steps=1, observation_noise=False)
        
        # Remove expert-only observations (not needed at inference time)
        expert_obs_keys_to_remove = [
            "historical_previous_actions",
            "max_coords_obs",
            "historical_max_coords_obs",
            "mimic_max_coords_target_poses",
        ]
        for key in expert_obs_keys_to_remove:
            if key in env_cfg.observation_components:
                del env_cfg.observation_components[key]
    
    # Clear expert model path for inference
    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        agent_cfg.expert_model_path = None
        agent_cfg.evaluator = EvaluatorConfig()
