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
"""BeyondMimic-style experiment with Gaussian kernel rewards."""

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
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration with low friction settings."""
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
    """Build environment configuration with BeyondMimic rewards."""
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.obs import (
        max_coords_obs_factory,
        historical_max_coords_obs_factory,
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
        historical_actions_factory,
        mimic_target_poses_max_coords_factory,
        mimic_target_poses_reduced_coords_factory,
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
    )

    # Control components configuration
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        # Actor observations (observation_noise=True)
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(observation_noise=True),
        "noisy_historical_reduced_coords_obs": historical_reduced_coords_obs_factory(observation_noise=True),
        "noisy_mimic_reduced_coords_target_poses": mimic_target_poses_reduced_coords_factory(observation_noise=True),
        # Critic observations (default: observation_noise=False)
        "max_coords_obs": max_coords_obs_factory(),
        # Clean historical observations for critic (default: observation_noise=False)
        "historical_max_coords_obs": historical_max_coords_obs_factory(),
        # Target poses for mimic
        "mimic_max_coords_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
        # Common observations
        "historical_previous_actions": historical_actions_factory(),
    }

    # BeyondMimic-style termination conditions
    termination_components = {
        "bad_ref_pos": anchor_pos_error_factory(threshold=0.5),
        "bad_ref_ori": anchor_ori_error_factory(threshold=0.8),
        "bad_motion_body_pos": relative_body_pos_error_factory(threshold=0.25),
    }

    # Reward configuration (BeyondMimic with density-based body weights)
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
        num_state_history_steps=3,
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
) -> PPOAgentConfig:
    """Build agent configuration."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig, MotionWeightsRulesConfig

    # Actor configuration - uses noisy observations
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=[
            "noisy_reduced_coords_obs",
            "noisy_mimic_reduced_coords_target_poses",
            "noisy_historical_reduced_coords_obs",
            "historical_previous_actions",
        ],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "noisy_historical_reduced_coords_obs",
                "historical_previous_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
            output_activation="tanh",
        ),
    )

    # Critic configuration - uses clean observations
    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_max_coords_target_poses", "historical_max_coords_obs", "historical_previous_actions"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                # Noisy observations for actor
                "noisy_reduced_coords_obs",
                "noisy_historical_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                # Clean observations for critic
                "max_coords_obs",
                "historical_max_coords_obs",
                "mimic_max_coords_target_poses",
                # Shared observations
                "historical_previous_actions",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=2e-5, betas=(0.95, 0.99)
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4, betas=(0.95, 0.99)
            ),
        ),
        batch_size=args.batch_size,
        num_mini_epochs=2,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
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
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot and simulator for this experiment."""

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
    env_cfg,
    agent_cfg,
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

    # Disable all termination components for inference
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}
    
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    terrain_cfg.sim_config = TerrainSimConfig(
        static_friction=1.,
        dynamic_friction=1.,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
    simulator_cfg.domain_randomization = None

    # Update observation components to use clean observations
    env_cfg.observation_components["noisy_reduced_coords_obs"] = reduced_coords_obs_factory(observation_noise=False)
    env_cfg.observation_components["noisy_historical_reduced_coords_obs"] = historical_reduced_coords_obs_factory(observation_noise=False)
    env_cfg.observation_components["noisy_mimic_reduced_coords_target_poses"] = mimic_target_poses_reduced_coords_factory(observation_noise=False)
