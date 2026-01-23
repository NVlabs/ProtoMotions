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
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors."""
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
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
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.obs import (
        max_coords_obs_factory,
        historical_actions_factory,
        mimic_target_poses_max_coords_future_rel_factory,
    )
    from protomotions.envs.rewards import (
        action_smoothness_factory,
        gt_rew_factory,
        gr_rew_factory,
        gv_rew_factory,
        gav_rew_factory,
        rh_rew_factory,
        pow_rew_factory,
        contact_match_rew_factory,
        contact_force_change_rew_factory,
    )
    from protomotions.envs.terminations import tracking_error_factory

    num_future_steps = 10
    num_historical_actions = 2

    # Control components
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            num_future_steps=num_future_steps,
        )
    }

    # Observation components configuration
    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "historical_previous_actions": historical_actions_factory(num_steps=num_historical_actions),
        "mimic_target_poses": mimic_target_poses_max_coords_future_rel_factory(
            num_future_steps=num_future_steps
        ),
    }

    # Termination components
    termination_components = {
        "tracking_error": tracking_error_factory(threshold=0.5),
    }

    # Reward components
    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        "gt_rew": gt_rew_factory(weight=0.5, coefficient=-100.0),
        "gr_rew": gr_rew_factory(weight=0.3, coefficient=-5.0),
        "gv_rew": gv_rew_factory(weight=0.1, coefficient=-0.5),
        "gav_rew": gav_rew_factory(weight=0.1, coefficient=-0.1),
        "rh_rew": rh_rew_factory(weight=0.2, coefficient=-100.0),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(weight=-0.1),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=num_historical_actions,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.common.config import TransformerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    transformer_token_size = 512
    transformer_encoder_widths = 256
    
    # Get parameters from config
    num_future_steps = env_config.control_components["mimic"].num_future_steps
    num_historical_actions = env_config.num_state_history_steps

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "historical_previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=ModuleContainerConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "historical_previous_actions",
            ],
            out_keys=["actor_trunk_out"],
            models=[
                MLPWithConcatConfig(
                    in_keys=["max_coords_obs"],
                    out_keys=["max_coords_obs_token"],
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
                        ModuleOperationForwardConfig(),
                        ModuleOperationReshapeConfig(
                            new_shape=["batch_size", 1, transformer_token_size]
                        ),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=["mimic_target_poses"],
                    out_keys=["mimic_target_poses_token"],
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
                                num_future_steps,
                                -1,
                            ]
                        ),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=["historical_previous_actions"],
                    out_keys=["historical_previous_actions_token"],
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
                                num_historical_actions,
                                -1,
                            ]
                        ),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                TransformerConfig(
                    in_keys=[
                        "max_coords_obs_token",
                        "mimic_target_poses_token",
                        "historical_previous_actions_token",
                    ],
                    out_keys=["transformer_out"],
                    transformer_token_size=transformer_token_size,
                    latent_dim=transformer_token_size,
                    output_activation=None,
                ),
                MLPWithConcatConfig(
                    in_keys=["transformer_out"],
                    out_keys=["actor_trunk_out"],
                    normalize_obs=False,
                    num_out=robot_config.kinematic_info.num_dofs,
                    layers=[
                        MLPLayerConfig(units=256, activation="relu"),
                        MLPLayerConfig(units=128, activation="relu"),
                    ],
                    output_activation="tanh",
                ),
            ],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "historical_previous_actions"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )
    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "historical_previous_actions",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=5e-5),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            eval_metric_keys=[
                "gt_err",
                "gr_err",
                "gr_err_degrees",
                "lr_err_degrees",
                "gt_rew",
                "gr_rew",
                "pow_rew",
                "contact_force_change_rew",
            ],
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
    )
    return agent_config


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
    """Apply evaluation-specific overrides."""
    # For mimic: disable early termination during evaluation
    if env_cfg is not None:
        # Clear termination components for inference
        if hasattr(env_cfg, "termination_components"):
            env_cfg.termination_components = {}
        if hasattr(env_cfg, "max_episode_length"):
            env_cfg.max_episode_length = 1000000
        if hasattr(env_cfg, "motion_manager"):
            if hasattr(env_cfg.motion_manager, "resample_on_reset"):
                env_cfg.motion_manager.resample_on_reset = True
            if hasattr(env_cfg.motion_manager, "init_start_prob"):
                env_cfg.motion_manager.init_start_prob = 1.0
