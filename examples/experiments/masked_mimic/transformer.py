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
from protomotions.agents.common.config import SequentialModuleConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.masked_mimic.config import MaskedMimicAgentConfig
import argparse


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


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> MimicEnvConfig:
    """Build environment configuration (training defaults)."""
    from protomotions.envs.mimic.config import (
        MimicEarlyTerminationEntry,
        MimicMotionManagerConfig,
    )
    from protomotions.envs.obs.config import (
        HumanoidObsConfig,
        MaxCoordsSelfObsConfig,
        ActionHistoryConfig,
        MimicObsConfig,
        MimicTargetPoseConfig,
        MaskedMimicObsConfig,
        MaskedMimicTargetPoseConfig,
        MaskedMimicMaskingConfig,
        JointMaskingConfig,
        TimeSamplingConfig,
        MaskedMimicHistoricalObsConfig,
        FuturePoseType,
    )
    from protomotions.envs.base_env.config import RewardComponentConfig
    from protomotions.envs.utils.rewards import (
        mean_squared_error_exp,
        rotation_error_exp,
    )

    # Training defaults (eval overrides applied automatically)
    mimic_early_termination = [
        MimicEarlyTerminationEntry(
            mimic_early_termination_key="max_joint_err",
            mimic_early_termination_thresh=0.5,
            less_than=False,
        )
    ]

    # Unified reward configuration - all components in one dict
    reward_config = {
        # Mimic tracking rewards
        "gt_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos",
                "ref_x": "ref_state.rigid_body_pos",
                "coefficient": "-100.0",
            },
            weight=0.5,
        ),
        "gr_rew": RewardComponentConfig(
            function=rotation_error_exp,
            variables={
                "q": "current_state.rigid_body_rot",
                "ref_q": "ref_state.rigid_body_rot",
                "coefficient": "-5.0",
            },
            weight=0.3,
        ),
    }

    env_config: MimicEnvConfig = MimicEnvConfig(
        humanoid_obs=HumanoidObsConfig(
            max_coords_obs=MaxCoordsSelfObsConfig(
                enabled=True, num_historical_steps=120
            ),
            action_history=ActionHistoryConfig(
                enabled=True,
                num_historical_steps=1,
            ),
        ),
        max_episode_length=1000,  # Training default (eval override applied automatically)
        reward_config=reward_config,
        mimic_early_termination=mimic_early_termination,
        mimic_bootstrap_on_episode_end=True,
        mimic_obs=MimicObsConfig(
            enabled=True,
            mimic_target_pose=MimicTargetPoseConfig(
                enabled=True,
                type=FuturePoseType.MAX_COORDS,
                with_velocities=True,
                future_steps=1,
            ),
        ),
        masked_mimic_obs=MaskedMimicObsConfig(
            enabled=True,
            masked_mimic_target_pose=MaskedMimicTargetPoseConfig(
                num_future_steps=5,
            ),
            masked_mimic_masking=MaskedMimicMaskingConfig(
                joint_masking=JointMaskingConfig(
                    masked_mimic_repeat_mask_probability=0.8,
                    force_max_conditioned_bodies_prob=0.1,
                    force_small_num_conditioned_bodies_prob=0.1,
                    visible_target_pose_prob=0.8,
                ),
                time_sampling=TimeSamplingConfig(alpha=2.0, beta=5.0),
            ),
            historical_obs=MaskedMimicHistoricalObsConfig(
                num_historical_conditioned_steps=15
            ),
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: MimicEnvConfig, args: argparse.Namespace
) -> MaskedMimicAgentConfig:
    from protomotions.agents.masked_mimic.config import (
        MaskedMimicModelConfig,
        VaeConfig,
        VaeNoiseType,
        KLDScheduleConfig,
    )
    from protomotions.agents.common.config import (
        FlattenConfig,
        MLPLayerConfig,
        MultiOutputModuleConfig,
        MLPWithConcatConfig,
        TransformerConfig,
    )
    from protomotions.agents.common.config import (
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    transformer_token_size = 512
    transformer_encoder_widths = 256
    vae_latent_dim = 64

    # Encoder: Uses regular mimic observations and body masks
    encoder_config = SequentialModuleConfig(
        in_keys=["mimic_target_poses", "masked_mimic_target_bodies_masks"],
        out_keys=["encoder_mu", "encoder_logvar"],
        input_models=[
            MultiOutputModuleConfig(
                in_keys=["mimic_target_poses", "masked_mimic_target_bodies_masks"],
                out_keys=[
                    "mimic_target_poses_norm",
                    "masked_mimic_target_bodies_masks_flattened",
                ],
                output_models=[
                    FlattenConfig(
                        in_keys=["mimic_target_poses"],
                        out_keys=["mimic_target_poses_norm"],
                        normalize_obs=True,
                        norm_clamp_value=5,
                    ),
                    FlattenConfig(
                        in_keys=["masked_mimic_target_bodies_masks"],
                        out_keys=["masked_mimic_target_bodies_masks_flattened"],
                    ),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "mimic_target_poses_norm",
                    "masked_mimic_target_bodies_masks_flattened",
                ],
                out_keys=["encoder_trunk_out"],
                num_out=512,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(5)
                ],
                output_activation="relu",
            ),
            MultiOutputModuleConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_mu", "encoder_logvar"],
                output_models=[
                    MLPWithConcatConfig(
                        in_keys=["encoder_trunk_out"],
                        out_keys=["encoder_mu"],
                        num_out=vae_latent_dim,
                        layers=[
                            MLPLayerConfig(units=256, activation="relu"),
                            MLPLayerConfig(units=128, activation="relu"),
                        ],
                    ),
                    MLPWithConcatConfig(
                        in_keys=["encoder_trunk_out"],
                        out_keys=["encoder_logvar"],
                        num_out=vae_latent_dim,
                        layers=[
                            MLPLayerConfig(units=256, activation="relu"),
                            MLPLayerConfig(units=128, activation="relu"),
                        ],
                    ),
                ],
            ),
        ],
    )

    # Prior: Uses transformer with masked mimic observations and historical data
    prior_config = SequentialModuleConfig(
        in_keys=[
            "masked_mimic_target_poses",
            "masked_mimic_target_bodies_masks",
            "historical_pose_obs",
        ],
        out_keys=["prior_mu", "prior_logvar"],
        input_models=[
            MultiOutputModuleConfig(
                in_keys=["masked_mimic_target_poses", "historical_pose_obs"],
                out_keys=[
                    "masked_mimic_target_poses_token",
                    "historical_pose_obs_token",
                ],
                output_models=[
                    MLPWithConcatConfig(
                        in_keys=["masked_mimic_target_poses"],
                        out_keys=["masked_mimic_target_poses_token"],
                        normalize_obs=True,
                        norm_clamp_value=5,
                        num_out=transformer_token_size,
                        layers=[
                            MLPLayerConfig(
                                units=transformer_encoder_widths, activation="relu"
                            )
                            for _ in range(2)
                        ],
                        module_operations=[
                            ModuleOperationReshapeConfig(
                                new_shape=[
                                    "batch_size",
                                    env_config.masked_mimic_obs.masked_mimic_target_pose.num_future_steps,
                                    -1,
                                ]
                            ),
                            ModuleOperationForwardConfig(),
                        ],
                    ),
                    MLPWithConcatConfig(
                        in_keys=["historical_pose_obs"],
                        out_keys=["historical_pose_obs_token"],
                        normalize_obs=True,
                        norm_clamp_value=5,
                        num_out=transformer_token_size,
                        layers=[
                            MLPLayerConfig(
                                units=transformer_encoder_widths, activation="relu"
                            )
                            for _ in range(2)
                        ],
                        module_operations=[
                            ModuleOperationReshapeConfig(
                                new_shape=[
                                    "batch_size",
                                    env_config.masked_mimic_obs.historical_obs.num_historical_conditioned_steps,
                                    -1,
                                ]
                            ),
                            ModuleOperationForwardConfig(),
                        ],
                    ),
                ],
            ),
            TransformerConfig(
                in_keys=[
                    "masked_mimic_target_poses_token",
                    "historical_pose_obs_token",
                    "masked_mimic_target_poses_masks",
                ],
                out_keys=["transformer_out"],
                transformer_token_size=transformer_token_size,
                latent_dim=transformer_token_size,
                input_and_mask_mapping={
                    "masked_mimic_target_poses_token": "masked_mimic_target_poses_masks",
                },
                output_activation="relu",
            ),
            MultiOutputModuleConfig(
                in_keys=["transformer_out"],
                out_keys=["prior_mu", "prior_logvar"],
                output_models=[
                    MLPWithConcatConfig(
                        in_keys=["transformer_out"],
                        out_keys=["prior_mu"],
                        num_out=vae_latent_dim,
                        layers=[
                            MLPLayerConfig(units=256, activation="relu"),
                            MLPLayerConfig(units=128, activation="relu"),
                        ],
                    ),
                    MLPWithConcatConfig(
                        in_keys=["transformer_out"],
                        out_keys=["prior_logvar"],
                        num_out=vae_latent_dim,
                        layers=[
                            MLPLayerConfig(units=256, activation="relu"),
                            MLPLayerConfig(units=128, activation="relu"),
                        ],
                    ),
                ],
            ),
        ],
    )

    # Trunk: Processes VAE latent + self obs to produce actions
    trunk_config = SequentialModuleConfig(
        in_keys=["max_coords_obs", "historical_previous_actions", "vae_latent"],
        out_keys=["actor_trunk_out"],
        input_models=[
            MultiOutputModuleConfig(
                in_keys=["max_coords_obs", "historical_previous_actions", "vae_latent"],
                out_keys=[
                    "max_coords_obs_norm",
                    "historical_previous_actions_norm",
                    "vae_latent_flattened",
                ],
                output_models=[
                    FlattenConfig(
                        in_keys=["max_coords_obs"],
                        out_keys=["max_coords_obs_norm"],
                        normalize_obs=True,
                        norm_clamp_value=5,
                    ),
                    FlattenConfig(
                        in_keys=["historical_previous_actions"],
                        out_keys=["historical_previous_actions_norm"],
                        normalize_obs=True,
                        norm_clamp_value=5,
                    ),
                    FlattenConfig(
                        in_keys=["vae_latent"],
                        out_keys=["vae_latent_flattened"],
                        normalize_obs=False,
                    ),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "max_coords_obs_norm",
                    "historical_previous_actions_norm",
                    "vae_latent_flattened",
                ],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(3)
                ],
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
        eval_metric_keys=["gt_err", "gr_err", "gr_err_degrees", "gt_rew", "gr_rew"],
    )

    # Agent configuration
    agent_config = MaskedMimicAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=evaluator_config,
        expert_model_path=None,
    )
    return agent_config


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: MimicEnvConfig,
    agent_cfg: MaskedMimicAgentConfig,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    # Reuse the mimic apply_inference_overrides function from mimic.mlp
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../mimic/mlp.py"
    )
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, args)

    from protomotions.agents.evaluators.config import EvaluatorConfig

    # For masked mimic: disable expert during evaluation (no distillation)
    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        agent_cfg.expert_model_path = None
        # Use simpler evaluator for eval
        agent_cfg.evaluator = EvaluatorConfig()
