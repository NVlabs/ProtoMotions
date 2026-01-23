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
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.masked_mimic.config import MaskedMimicAgentConfig
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

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults)."""
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.rewards import gt_rew_factory, gr_rew_factory
    from protomotions.envs.terminations import tracking_error_factory
    from protomotions.envs.obs.observation_component import ObservationComponentConfig
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.obs import (
        max_coords_obs_factory,
        historical_max_coords_obs_factory,
        historical_poses_with_time_factory,
        mimic_target_poses_max_coords_factory,
        previous_actions_factory,
    )
    from protomotions.envs.obs.general import passthrough_float_factory
    from protomotions.envs.obs.masked_mimic_obs_functions import (
        masked_mimic_target_poses_factory,
        target_masks_factory,
        target_time_offsets_factory,
    )

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
            visible_target_pose_prob=0.8
        ),
    }

    # Compute conditionable body IDs from robot config (baked into factories)
    conditionable_body_ids = [
        robot_cfg.kinematic_info.body_names.index(name)
        for name in robot_cfg.trackable_bodies_subset
    ]

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "historical_max_coords_obs": historical_max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True, num_future_steps=1),
        "masked_mimic_target_poses": masked_mimic_target_poses_factory(conditionable_body_ids=conditionable_body_ids),
        "masked_mimic_target_masks": target_masks_factory(conditionable_body_ids=conditionable_body_ids),
        "masked_mimic_target_times": target_time_offsets_factory(),
        "historical_pose_obs": historical_poses_with_time_factory(
            num_historical_conditioned_steps=NUM_HISTORICAL_CONDITIONED_STEPS,
            total_stored_historical_steps=TOTAL_STORED_HISTORICAL_STEPS,
        ),
        "masked_mimic_target_poses_masks": passthrough_float_factory(variable="masked_mimic_target_poses_masks"),
        "masked_mimic_target_bodies_masks": passthrough_float_factory(variable="masked_mimic_target_bodies_masks"),
    }

    expert_model_path = getattr(args, 'expert_model_path', None)
    if expert_model_path:
        from protomotions.agents.masked_mimic.utils import (
            load_expert_configs,
            get_expert_observation_components,
        )
        expert_configs = load_expert_configs(expert_model_path)
        expert_env_config = expert_configs["env"]
        expert_agent_config = expert_configs["agent"]
        
        expert_history_steps = getattr(expert_env_config, 'num_state_history_steps', 0)
        assert TOTAL_STORED_HISTORICAL_STEPS >= expert_history_steps, (
            f"Insufficient history: current={TOTAL_STORED_HISTORICAL_STEPS}, expert requires={expert_history_steps}"
        )
        
        if hasattr(expert_env_config, 'control_components') and expert_env_config.control_components:
            for ctrl_cfg in expert_env_config.control_components.values():
                expert_num_future = getattr(ctrl_cfg, 'num_future_steps', None)
                if expert_num_future is not None:
                    masked_mimic_cfg = control_components["masked_mimic"]
                    if masked_mimic_cfg.num_future_steps < expert_num_future:
                        masked_mimic_cfg.num_future_steps = expert_num_future
        
        expert_obs_components = get_expert_observation_components(
            expert_env_config,
            expert_agent_config,
            existing_obs_keys=list(observation_components.keys()),
        )
        observation_components.update(expert_obs_components)

    # Termination components configuration
    termination_components = {
        "tracking_error": tracking_error_factory(threshold=0.25),
    }

    # Reward components
    reward_components = {
        "gt_rew": gt_rew_factory(weight=0.5, coefficient=-100.0),
        "gr_rew": gr_rew_factory(weight=0.3, coefficient=-5.0),
    }

    env_config: EnvConfig = EnvConfig(
        max_episode_length=1000,
        num_state_history_steps=TOTAL_STORED_HISTORICAL_STEPS,  # Historical obs for masked mimic prior
        # Component-based configuration
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        # Motion manager configuration
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MaskedMimicAgentConfig:
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
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

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


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: MaskedMimicAgentConfig,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
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
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args)

    from protomotions.agents.evaluators.config import EvaluatorConfig

    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        expert_model_path = agent_cfg.expert_model_path
        
        # Remove expert observation components
        if expert_model_path is not None and env_cfg is not None:
            if hasattr(env_cfg, "observation_components") and env_cfg.observation_components is not None:
                from protomotions.agents.masked_mimic.utils import (
                    load_expert_configs,
                    get_expert_observation_keys,
                )
                expert_configs = load_expert_configs(expert_model_path)
                expert_obs_keys = get_expert_observation_keys(expert_configs["env"], expert_configs["agent"])
                for key in expert_obs_keys:
                    if key in env_cfg.observation_components:
                        del env_cfg.observation_components[key]
        
        agent_cfg.expert_model_path = None
        agent_cfg.evaluator = EvaluatorConfig()
