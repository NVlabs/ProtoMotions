# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BeyondMimic experiment with torso anchor + L2C2 regularization.

BeyondMimic L2C2 experiment with a self-contained configuration:
- Actor observations come from the declared sensor suite (``source="noisy"``):
  joint encoders and the robot's IMU, perturbed once at the sensor level
- Clean counterparts (``source="clean"``) are the same sensors without noise;
  the L2C2 term penalizes MSE(mu_noisy, mu_clean)
- Critic observations remain privileged simulator state
- Anchor uses robot config default (torso_link for G1) instead of pelvis override
- Self-contact cost follows BeyondMimic's non-foot/non-hand body selection

Reference: Kobayashi 2022 — "L2C2: Locally Lipschitz Continuous Constraint"
"""

from typing import List

from protomotions.envs.presets import tracking_playback
from protomotions.utils.experiment_configs import ExperimentConfigs
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    RobotNoiseConfig,
    PushDomainRandomizationConfig,
    DomainRandomizationConfig,
)
from protomotions.components.terrains.config import (
    TerrainConfig,
    TerrainSimConfig,
    CombineMode,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    SensorConfig,
)
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


def _self_contact_bodies(robot_cfg: RobotConfig) -> List[str]:
    """BeyondMimic/Holosoma scored links; hands and feet remain contact partners."""
    allowed = {
        name
        for group in (
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "all_left_hand_bodies",
            "all_right_hand_bodies",
        )
        for name in robot_cfg.common_naming_to_robot_body_names[group]
    }
    # G1's hand markers are distinct from the wrist links carrying hand collisions.
    allowed.update(("left_wrist_yaw_link", "right_wrist_yaw_link"))
    return [name for name in robot_cfg.kinematic_info.body_names if name not in allowed]


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    terrain_cfg = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1,
            dynamic_friction=1,
            restitution=0.0,
            combine_mode=CombineMode.MULTIPLY,
        )
    )
    return terrain_cfg


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration with future target poses and L2C2 clean obs.

    Actor sees reduced coords + multi-horizon target poses (no XY offset).
    Additionally provides clean (noise-free) counterparts for L2C2 regularization.
    Rewards use BeyondMimic-style heading-invariant relative body tracking
    plus global anchor tracking (no motion realignment at training).
    """
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
        compute_humanoid_reduced_coords_observations,
    )
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )
    from protomotions.envs.obs.target_poses import (
        build_max_coords_target_poses,
        build_reduced_coords_target_poses,
    )
    from protomotions.envs.context_sources import sensed
    from protomotions.envs.rewards.regularization import (
        compute_action_rate_l2,
        compute_self_contacts,
    )
    from protomotions.envs.rewards.tracking import (
        compute_global_anchor_ori_rew,
        compute_global_body_ang_vel_rew,
        compute_global_body_lin_vel_rew,
        compute_relative_body_ori_rew,
        compute_relative_body_pos_rew,
    )
    from protomotions.envs.terminations.tracking import anchor_height_error_value
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.rewards.regularization import compute_soft_pos_limit_rew
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.action import make_bm_pd_action_config

    # Control components configuration
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=[1, 2, 4, 8],
        )
    }

    # Observation components. L2C2: the actor's noisy sensor obs and their clean
    # (noise-free) twins are the same privileged-default kernels repointed by
    # sensed(); the critic reads privileged max coords; processed actions are
    # common to both and stay privileged.
    reduced_coords = MdpComponent.from_kernel(
        compute_humanoid_reduced_coords_observations,
        root_height_obs=False,
        root_vel_obs=False,
    )
    reduced_coords_target = MdpComponent.from_kernel(
        build_reduced_coords_target_poses,
        include_dof_vel=True,
        include_xy_offset=False,
    )
    actor = {
        "noisy_reduced_coords_obs": reduced_coords,
        "noisy_mimic_reduced_coords_target_poses": reduced_coords_target,
    }
    clean = {
        "clean_reduced_coords_obs": reduced_coords,
        "clean_mimic_reduced_coords_target_poses": reduced_coords_target,
    }
    privileged = {
        "max_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "mimic_max_coords_target_poses": MdpComponent.from_kernel(
            build_max_coords_target_poses,
            with_velocities=True,
            with_relative=True,
        ),
        "historical_previous_processed_actions": MdpComponent.from_kernel(
            compute_historical_actions_from_state,
            history_steps=1,
            historical_actions=EnvContext.processed_action_history,
        ),
    }
    observation_components = {
        **sensed(actor, "noisy"),
        **sensed(clean, "clean"),
        **privileged,
    }

    # Termination components
    termination_components = {
        "fall": TerminationRule(
            MdpComponent.from_kernel(anchor_height_error_value), 0.25
        ),
    }

    # Reward components
    self_contact_partners = [
        f"robot/{name}" for name in robot_cfg.kinematic_info.body_names
    ]
    self_contact_bodies = _self_contact_bodies(robot_cfg)
    reward_components = {
        # Global anchor (root) orientation
        "global_anchor_ori": MdpComponent.from_kernel(
            compute_global_anchor_ori_rew,
            meta=RewardMeta(weight=0.5),
            sigma=0.4,
        ),
        # Relative body position and orientation (region-weighted)
        "relative_body_pos": MdpComponent.from_kernel(
            compute_relative_body_pos_rew,
            meta=RewardMeta(weight=1.0),
            sigma=0.3,
        ),
        "relative_body_ori": MdpComponent.from_kernel(
            compute_relative_body_ori_rew,
            meta=RewardMeta(weight=1.0),
            sigma=0.4,
        ),
        # Global body velocities (region-weighted)
        "body_lin_vel": MdpComponent.from_kernel(
            compute_global_body_lin_vel_rew,
            meta=RewardMeta(weight=1.0),
            sigma=1.0,
        ),
        "body_ang_vel": MdpComponent.from_kernel(
            compute_global_body_ang_vel_rew,
            meta=RewardMeta(weight=1.0),
            sigma=3.14,
        ),
        "action_rate": MdpComponent.from_kernel(
            compute_action_rate_l2,
            meta=RewardMeta(weight=-0.1),
        ),
        "self_contact": MdpComponent.from_kernel(
            compute_self_contacts,
            meta=RewardMeta(weight=-0.1),
            required_keys={
                "contact_body_index": self_contact_bodies,
                "contact_partner_index": self_contact_partners,
            },
            body_names=self_contact_bodies,
            partner_names=self_contact_partners,
            force_threshold=1.0,
        ),
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
            },
            meta=RewardMeta(weight=-10.0),
            static_params={
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,
            },
        ),
    }

    # The robot's IMU and joint encoders, with the BeyondMimic uniform half-widths.
    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(pos_noise=0.01, vel_noise=0.5),  # rad, rad/s
        imu=IMUSensor(quat_noise=0.05, gyro_noise=0.2),  # xyzw component, rad/s
    )

    return EnvConfig(
        sensors=sensors,
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        # Divergence from the public recipe for this shipped checkpoint:
        # effort_fraction=1.0 restores the pre-a347b28b1 action scale the
        # 2026-08-10 G1 weights trained with (the public default is 0.25).
        action_config=make_bm_pd_action_config(
            robot_cfg, effort_fraction=1.0
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build agent configuration with L2C2 regularization."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdaptiveLRConfig,
        AdvantageNormalizationConfig,
        L2C2Config,
        StdMode,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig

    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations.tracking import (
        anchor_height_error_value,
        anchor_ori_error_value,
        max_body_pos_error,
        mean_body_pos_error,
        mean_body_rot_error,
        relative_body_pos_max_error,
    )
    from protomotions.envs.terminations.rule import TerminationRule

    # Actor configuration — obs normalization ON, learnable std from -2.9
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        std_mode=StdMode.GLOBAL,
        in_keys=[
            "noisy_reduced_coords_obs",
            "noisy_mimic_reduced_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    # Critic configuration — obs normalization ON
    critic_config = MLPWithConcatConfig(
        in_keys=[
            "max_coords_obs",
            "mimic_max_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=2e-5, betas=(0.95, 0.99)
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4, betas=(0.95, 0.99)
            ),
        ),
        normalize_rewards=False,
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        batch_size=args.batch_size,
        num_mini_epochs=2,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        l2c2=L2C2Config(
            enabled=True,
            lambda_l2c2=1.0,
            obs_pairs={
                "noisy_reduced_coords_obs": "clean_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses": "clean_mimic_reduced_coords_target_poses",
            },
        ),
        evaluator=MimicEvaluatorConfig(
            metric_components={
                "anchor_ori": MdpComponent.from_kernel(anchor_ori_error_value),
                "relative_body_pos": MdpComponent.from_kernel(
                    relative_body_pos_max_error,
                ),
                "anchor_height_error": MdpComponent.from_kernel(
                    anchor_height_error_value,
                ),
                "gt_error": MdpComponent.from_kernel(mean_body_pos_error),
                "gr_error": MdpComponent.from_kernel(mean_body_rot_error),
                "max_joint_error": MdpComponent.from_kernel(max_body_pos_error),
            },
            failure_components={
                "anchor_height_error": TerminationRule(
                    MdpComponent.from_kernel(anchor_height_error_value), 0.25
                ),
            },
            score_key="eval/success_rate",
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot and simulator for this experiment.

    Uses G1 default anchor (torso_link) — no pelvis override.
    """
    if not robot_cfg.asset.self_collisions:
        raise ValueError("The BM self-contact cost requires asset.self_collisions=True")
    # Match BM/Holosoma's scored bodies, but keep this cost robot-to-robot only.
    robot_cfg.update_fields(
        contact_bodies=_self_contact_bodies(robot_cfg),
        contact_partners=[
            f"robot/{name}" for name in robot_cfg.kinematic_info.body_names
        ],
    )

    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.1,
        root_pos_noise=[0.05, 0.05, 0.01],
        root_rot_noise=[0.1, 0.1, 0.2],
        root_vel_noise=[0.1, 0.1, 0.05],
        root_ang_vel_noise=[0.1, 0.1, 0.1],
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.025, 0.025), dof_names=[".*"], dof_indices=None
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(1.0, 3.0),
            max_linear_velocity=(0.5, 0.5, 0.2),
            max_angular_velocity=(0.52, 0.52, 0.78),
        ),
    )


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Evaluation view: play each clip to its end on clean sensors.

    The sensors read exactly (a framework inference default now), so the clean
    L2C2 observations are dropped (the actor's own sensors are already exact);
    the fixed terrain friction and no domain randomization keep playback
    deterministic.
    """
    tracking_playback(configs.env)
    configs.terrain.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
    configs.simulator.domain_randomization = None
    for key in [
        "clean_reduced_coords_obs",
        "clean_mimic_reduced_coords_target_poses",
    ]:
        configs.env.observation_components.pop(key, None)
