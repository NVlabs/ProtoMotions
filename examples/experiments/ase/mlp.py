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
from protomotions.agents.ase.config import ASEAgentConfig
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


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.obs.config import HumanoidObsConfig, MaxCoordsSelfObsConfig
    from protomotions.envs.motion_manager.config import MotionManagerConfig
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.scene_lib import SceneLibConfig

    # Conditionally add scene_lib if scenes_file is provided
    scene_lib_config = None
    if hasattr(args, "scenes_file") and args.scenes_file is not None:
        scene_lib_config = SceneLibConfig(scene_file=args.scenes_file)

    env_config: EnvConfig = EnvConfig(
        max_episode_length=300,  # Training default (eval override applied automatically)
        motion_lib=MotionLibConfig(motion_file=args.motion_file),
        humanoid_obs=HumanoidObsConfig(
            max_coords_obs=MaxCoordsSelfObsConfig(enabled=True, num_historical_steps=8),
        ),
        terrain=TerrainConfig(),
        motion_manager=MotionManagerConfig(
            init_start_prob=0.5  # Bias agent to start at the beginning of the motion to prevent getting stuck in a local-minima (standing still).
        ),
        scene_lib=scene_lib_config,
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> ASEAgentConfig:
    from protomotions.agents.common.config import (
        SequentialModuleConfig,
        MultiInputModuleConfig,
        FlattenConfig,
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleOperationForwardConfig,
        ModuleOperationSphereProjectionConfig,
    )
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.ase.config import (
        ASEParametersConfig,
        ASEDiscriminatorEncoderConfig,
    )
    from protomotions.agents.amp.config import AMPParametersConfig, AMPModelConfig

    conditional_discriminator = False

    ase_parameters = ASEParametersConfig(
        latent_dim=64,
        mi_reward_w=0.5,
        mi_hypersphere_reward_shift=True,
        diversity_bonus=0.01 if not conditional_discriminator else 0.0,
        latent_uniformity_weight=0.0 if not conditional_discriminator else 0.01,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "latents"],
        mu_key="actor_trunk_out",
        mu_model=SequentialModuleConfig(
            in_keys=["max_coords_obs", "latents"],
            out_keys=["actor_trunk_out"],
            input_models=[
                MultiInputModuleConfig(  # We use multi-input + Flatten so we only apply running normalization on the max_coords_obs
                    in_keys=["max_coords_obs", "latents"],
                    out_keys=["max_coords_obs_flattened", "latents_processed"],
                    input_models=[
                        FlattenConfig(
                            in_keys=["max_coords_obs"],
                            out_keys=["max_coords_obs_flattened"],
                            normalize_obs=True,
                            norm_clamp_value=5,
                        ),
                        MLPWithConcatConfig(
                            in_keys=["latents"],
                            out_keys=["latents_processed"],
                            num_out=ase_parameters.latent_dim,
                            output_activation="tanh",
                            layers=[
                                MLPLayerConfig(units=512, activation="relu"),
                                MLPLayerConfig(units=256, activation="relu"),
                            ],
                        ),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=["max_coords_obs_flattened", "latents_processed"],
                    out_keys=["actor_trunk_out"],
                    num_out=robot_config.number_of_actions,
                    layers=[
                        MLPLayerConfig(units=1024, activation="relu"),
                        MLPLayerConfig(units=1024, activation="relu"),
                        MLPLayerConfig(units=512, activation="relu"),
                    ],
                    output_activation="tanh",
                ),
            ],
        ),
    )

    critic_config = SequentialModuleConfig(
        in_keys=["max_coords_obs", "latents"],
        out_keys=["value"],
        input_models=[
            MultiInputModuleConfig(  # We use multi-input + Flatten so we only apply running normalization on the max_coords_obs
                in_keys=["max_coords_obs", "latents"],
                out_keys=["max_coords_obs_flattened", "latents_flattened"],
                input_models=[
                    FlattenConfig(
                        in_keys=["max_coords_obs"],
                        out_keys=["max_coords_obs_flattened"],
                        normalize_obs=True,
                        norm_clamp_value=5,
                    ),
                    FlattenConfig(
                        in_keys=["latents"],
                        out_keys=["latents_flattened"],
                        normalize_obs=False,
                    ),
                ],
            ),
            MLPWithConcatConfig(
                in_keys=["max_coords_obs_flattened", "latents_flattened"],
                out_keys=["value"],
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            ),
        ],
    )

    # Build discriminator keys based on conditional flag
    disc_head_in_keys = ["trunk_features"]
    if conditional_discriminator:
        disc_head_in_keys.append("latents")

    from protomotions.agents.common.config import MultiOutputModuleConfig

    discriminator_encoder_config = (
        ASEDiscriminatorEncoderConfig(  # This is a sequential module config
            encoder_out_size=ase_parameters.latent_dim,
            in_keys=["historical_max_coords_obs"]
            + (["latents"] if not conditional_discriminator else []),
            out_keys=["disc_logits", "mi_enc_output"],
            input_models=[  # Models in the sequential module
                # Trunk: process historical_max_coords_obs to features
                MLPWithConcatConfig(
                    in_keys=["historical_max_coords_obs"],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    out_keys=["trunk_features"],
                    num_out=512,
                    layers=[
                        MLPLayerConfig(units=1024, activation="relu"),
                        MLPLayerConfig(units=1024, activation="relu"),
                    ],
                ),
                # Multi-output: discriminator + MI encoder from trunk features
                MultiOutputModuleConfig(
                    in_keys=disc_head_in_keys,
                    out_keys=["disc_logits", "mi_enc_output"],
                    output_models=[
                        # Discriminator head
                        MLPWithConcatConfig(
                            in_keys=disc_head_in_keys,
                            out_keys=["disc_logits"],
                            num_out=1,
                            layers=[
                                MLPLayerConfig(units=512, activation="relu"),
                                MLPLayerConfig(units=256, activation="relu"),
                            ],
                        ),
                        # MI Encoder head with sphere projection
                        MLPWithConcatConfig(
                            in_keys=["trunk_features"],
                            out_keys=["mi_enc_output"],
                            num_out=ase_parameters.latent_dim,
                            layers=[],  # Single projection layer
                            module_operations=[
                                ModuleOperationForwardConfig(),
                                ModuleOperationSphereProjectionConfig(),
                            ],
                        ),
                    ],
                ),
            ],
        )
    )

    agent_config: ASEAgentConfig = ASEAgentConfig(
        model=AMPModelConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs", "latents"],
            out_keys=[
                "action",
                "mean_action",
                "neglogp",
                "value",
                "disc_logits",
                "mi_enc_output",
            ],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_encoder_config,  # ASEDiscriminatorEncoder (extends Discriminator)
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        task_reward_w=0.0,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        amp_parameters=AMPParametersConfig(
            conditional_discriminator=conditional_discriminator,
            discriminator_reward_w=0.5,
            discriminator_reward_threshold=0.05,  # Training default (eval override in apply_inference_overrides if needed)
        ),
        ase_parameters=ase_parameters,
    )
    return agent_config


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    # Reuse the amp apply_inference_overrides function
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../amp/mlp.py"
    )
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, args)
