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
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    from protomotions.components.terrains.config import ComplexTerrainConfig

    return ComplexTerrainConfig()


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
        FuturePoseType,
        MimicTargetPoseConfig,
        MimicObsConfig,
    )
    from protomotions.envs.base_env.config import RewardComponentConfig
    from protomotions.envs.utils.rewards import (
        mean_squared_error_exp,
        rotation_error_exp,
        power_consumption_sum,
        norm,
    )

    mimic_early_termination = [
        MimicEarlyTerminationEntry(
            mimic_early_termination_key="max_joint_err",
            mimic_early_termination_thresh=0.5,
            less_than=False,
        )
    ]

    # Unified reward configuration - all components in one dict
    reward_config = {
        # Base rewards
        "action_smoothness": RewardComponentConfig(
            function=norm,
            variables={
                "x": "current_actions - previous_actions",
            },
            weight=-0.02,
        ),
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
        "gv_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_vel",
                "ref_x": "ref_state.rigid_body_vel",
                "coefficient": "-0.5",
            },
            weight=0.1,
        ),
        "gav_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_ang_vel",
                "ref_x": "ref_state.rigid_body_ang_vel",
                "coefficient": "-0.1",
            },
            weight=0.1,
        ),
        "rh_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos[:, 0, 2]",
                "ref_x": "ref_state.rigid_body_pos[:, 0, 2]",
                "coefficient": "-100.0",
            },
            weight=0.2,
        ),
        "pow_rew": RewardComponentConfig(
            function=power_consumption_sum,
            variables={
                "dof_forces": "current_state.dof_forces",
                "dof_vel": "current_state.dof_vel",
                "use_torque_squared": "False",
            },
            weight=-1e-5,
            min_value=-0.5,
            zero_during_grace_period=True,
        ),
    }

    env_config: MimicEnvConfig = MimicEnvConfig(
        max_episode_length=1000,
        reward_config=reward_config,
        mimic_early_termination=mimic_early_termination,
        mimic_bootstrap_on_episode_end=True,
        mimic_obs=MimicObsConfig(
            enabled=True,
            mimic_target_pose=MimicTargetPoseConfig(
                enabled=True,
                type=FuturePoseType.MAX_COORDS_FUTURE_REL,
                with_time=True,
                future_steps=15,
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
) -> PPOAgentConfig:
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        TransformerConfig,
        SequentialModuleConfig,
        MultiOutputModuleConfig,
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    transformer_token_size = 512
    transformer_encoder_widths = 256
    num_future_steps = env_config.mimic_obs.mimic_target_pose.future_steps

    # Actor uses transformer architecture
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "terrain", "mimic_target_poses"],
        mu_key="actor_trunk_out",
        mu_model=SequentialModuleConfig(
            in_keys=["max_coords_obs", "terrain", "mimic_target_poses"],
            out_keys=["actor_trunk_out"],
            input_models=[
                # Encode inputs to transformer tokens
                MultiOutputModuleConfig(
                    in_keys=["max_coords_obs", "terrain", "mimic_target_poses"],
                    out_keys=[
                        "self_obs_token",
                        "terrain_token",
                        "mimic_target_poses_token",
                    ],
                    output_models=[
                        MLPWithConcatConfig(
                            in_keys=["max_coords_obs"],
                            out_keys=["self_obs_token"],
                            normalize_obs=True,
                            norm_clamp_value=5,
                            num_out=transformer_token_size,
                            layers=[
                                MLPLayerConfig(
                                    units=transformer_encoder_widths, activation="relu"
                                )
                                for _ in range(2)
                            ],
                        ),
                        MLPWithConcatConfig(
                            in_keys=["terrain"],
                            out_keys=["terrain_token"],
                            normalize_obs=True,
                            norm_clamp_value=5,
                            num_out=transformer_token_size,
                            layers=[
                                MLPLayerConfig(
                                    units=transformer_encoder_widths, activation="relu"
                                )
                                for _ in range(2)
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
                                    new_shape=["batch_size", num_future_steps, -1]
                                ),
                                ModuleOperationForwardConfig(),
                            ],
                        ),
                    ],
                ),
                # Transformer
                TransformerConfig(
                    in_keys=[
                        "self_obs_token",
                        "terrain_token",
                        "mimic_target_poses_token",
                    ],
                    out_keys=["transformer_out"],
                    transformer_token_size=transformer_token_size,
                    latent_dim=transformer_token_size,
                    output_activation="relu",
                ),
                # Output MLP
                MLPWithConcatConfig(
                    in_keys=["transformer_out"],
                    out_keys=["actor_trunk_out"],
                    num_out=robot_config.kinematic_info.num_dofs,
                    layers=[
                        MLPLayerConfig(units=1024, activation="relu") for _ in range(3)
                    ],
                    output_activation="tanh",
                ),
            ],
        ),
    )

    # Critic uses simple MLP
    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "terrain", "mimic_target_poses"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=["max_coords_obs", "terrain", "mimic_target_poses"],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
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
    args: argparse.Namespace,
):
    # Reuse the mimic apply_inference_overrides function from transformer.mlp
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides("mlp.py")
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, args)
