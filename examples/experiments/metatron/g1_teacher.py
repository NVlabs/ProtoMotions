# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metatron G1 expert: privileged FSQ/PPO tracking with five global targets.

Current state uses maximal and reduced coordinates plus one previous state and
raw action. Global targets stay in the current anchor-heading frame, retaining
world-space tracking error. This recipe has no AMP discriminator or reward.
Dataset composition is caller-owned; no dataset-specific shard IDs are assumed.
"""

import argparse
from copy import deepcopy

from protomotions.utils.experiment_configs import ExperimentConfigs
from examples.experiments.mimic import mlp, mlp_bm_l2c2
from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
from protomotions.agents.common.fsq_config import FSQAutoEncoderConfig

from protomotions.agents.ppo.config import PPOActorConfig
from protomotions.components.motion_lib import MotionFileSwitchMode, MotionLibConfig
from protomotions.components.terrains.config import (
    CombineMode,
    TerrainConfig,
    TerrainSimConfig,
)
from protomotions.envs.action import make_bm_pd_action_config
from protomotions.envs.obs.humanoid import (
    compute_humanoid_max_coords_observations,
    compute_humanoid_reduced_coords_observations,
)
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_actions_from_state,
    compute_historical_max_coords_from_state,
)
from protomotions.envs.rewards.regularization import (
    compute_action_rate_l2,
    compute_contact_match_rew,
    compute_self_contacts,
)
from protomotions.envs.rewards.tracking import (
    compute_global_body_ang_vel_rew,
    compute_global_body_lin_vel_rew,
    compute_relative_body_ori_rew,
    compute_relative_body_pos_rew,
)
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.mimic_control import MimicControlConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.obs.target_poses import (
    build_max_coords_target_poses,
    build_target_dof_pos,
)
from protomotions.envs.rewards.regularization import compute_soft_pos_limit_rew
from protomotions.envs.rewards.tracking import (
    compute_global_orientation_error_exp,
    compute_global_position_error_exp,
)
from protomotions.envs.terminations.low_pose import (
    tracking_failure_ratio,
    upright_foot_height_failed,
    upright_foot_position_failed,
)
from protomotions.robot_configs.base import abstract_names_to_body_names
from protomotions.simulator.base_simulator.config import (
    ActionNoiseDomainRandomizationConfig,
    BodyMassDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    LatencyDomainRandomizationConfig,
    PushDomainRandomizationConfig,
    RobotNoiseConfig,
)

ENCODER_WIDTHS = (2048, 1024, 512, 512)
DECODER_CRITIC_WIDTHS = (4096, 4096, 2048, 2048, 1024, 1024, 512, 512)
FSQ_SCALARS = 64
FSQ_LEVELS = 33
scene_lib_config = mlp.scene_lib_config


def _body_indices(robot, groups):
    return list(robot.body_indices(*groups))


def _future_steps(robot, args):
    # 0.5 s horizon at the policy control rate (fps / decimation == control_hz).
    template = (1, 2, 5, 7, 12, 18, 25)
    horizon = max(len(template), round(0.5 * robot.control_hz))
    return [
        min(max(round(step * horizon / 25), i + 1), horizon - (6 - i))
        for i, step in enumerate(template)
    ]


def terrain_config(args):
    return TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            combine_mode=CombineMode.MULTIPLY,
        )
    )


def motion_lib_config(args):
    # A concrete file works locally; a slurmrank pattern opts into live cycling.
    mode = (
        MotionFileSwitchMode.LIVE
        if "slurmrank" in str(args.motion_file)
        else MotionFileSwitchMode.FIXED
    )
    return MotionLibConfig(motion_file=args.motion_file, motion_file_switch_mode=mode)


def env_config(robot_cfg, args):
    cfg = mlp.env_config(robot_cfg, args)
    upper = [robot_cfg.anchor_body_index] + _body_indices(
        robot_cfg, ["all_left_hand_bodies", "all_right_hand_bodies"]
    )
    feet = _body_indices(robot_cfg, ["all_left_foot_bodies", "all_right_foot_bodies"])
    five = upper + feet
    cfg.sensors = None
    cfg.reset_grace_period = 5
    cfg.ref_respawn_offset = 0.0
    cfg.motion_manager.realign_motion_with_humanoid_on_each_step = False
    cfg.control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=_future_steps(robot_cfg, args),
            reference_height_smoothing_seconds=0.2,
        )
    }
    cfg.observation_components = {
        "max_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations,
            local_obs=True,
            root_height_obs=True,
        ),
        "historical_previous_max_coords_obs": MdpComponent.from_kernel(
            compute_historical_max_coords_from_state,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=1,
        ),
        "previous_raw_actions": MdpComponent.from_kernel(
            compute_historical_actions_from_state, history_steps=1
        ),
        "reduced_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_reduced_coords_observations
        ),
        "mimic_target_poses": MdpComponent.from_kernel(
            build_max_coords_target_poses, body_indices=five
        ),
        "target_dof_pos": MdpComponent(
            compute_func=build_target_dof_pos,
            dynamic_vars={"mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos},
        ),
    }
    # The shared helper already applies 0.25 * effort_limit / stiffness.
    cfg.action_config = make_bm_pd_action_config(robot_cfg)

    def position(indices, weight, sigma):
        return MdpComponent(
            compute_func=compute_global_position_error_exp,
            dynamic_vars={
                "x": EnvContext.current.rigid_body_pos,
                "ref_x": EnvContext.mimic.ref_state.rigid_body_pos,
            },
            static_params={"indices": indices, "sigma": sigma},
            meta=RewardMeta(weight=weight),
        )

    def orientation(indices, weight, sigma):
        return MdpComponent(
            compute_func=compute_global_orientation_error_exp,
            dynamic_vars={
                "q": EnvContext.current.rigid_body_rot,
                "ref_q": EnvContext.mimic.ref_state.rigid_body_rot,
            },
            static_params={"indices": indices, "sigma": sigma},
            meta=RewardMeta(weight=weight),
        )

    self_contact_partners = (
        [f"robot/{name}" for name in robot_cfg.kinematic_info.body_names]
    )
    self_contact_bodies = mlp_bm_l2c2._self_contact_bodies(robot_cfg)
    cfg.reward_components = {
        "gt_rew": position(five, 1.0, 0.3),
        "gr_rew": orientation(five, 1.0, 0.4),
        "hand_torso_pos": position(upper, 2.0, 0.1),
        "hand_torso_rot": orientation(upper, 0.4, 0.1),
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
        "gv_rew": MdpComponent.from_kernel(
            compute_global_body_lin_vel_rew,
            meta=RewardMeta(weight=1.0),
            sigma=1.0,
        ),
        "gav_rew": MdpComponent.from_kernel(
            compute_global_body_ang_vel_rew,
            meta=RewardMeta(weight=1.0),
            sigma=3.14,
        ),
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={"dof_pos": EnvContext.current.dof_pos},
            meta=RewardMeta(weight=-10.0),
            static_params={
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower
                * robot_cfg.control.soft_pos_limit,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper
                * robot_cfg.control.soft_pos_limit,
            },
        ),
        "action_smoothness": MdpComponent.from_kernel(
            compute_action_rate_l2,
            meta=RewardMeta(weight=-0.1),
        ),
        "contact_match_rew": MdpComponent.from_kernel(
            compute_contact_match_rew,
            meta=RewardMeta(weight=-0.1, zero_during_grace_period=True),
        ),
        "self_contacts": MdpComponent.from_kernel(
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
    }
    dynamic = {
        "current_position": EnvContext.current.rigid_body_pos,
        "reference_position": EnvContext.mimic.ref_state.rigid_body_pos,
        "smoothed_ref_anchor_height": EnvContext.mimic.smoothed_ref_anchor_height,
    }
    # The ratio is normalised (cut at 1); the foot checks are boolean kernels.
    cfg.termination_components = {
        "tracking_error": TerminationRule(
            MdpComponent(
                compute_func=tracking_failure_ratio,
                dynamic_vars=dynamic,
                static_params={
                    "upper_body_indices": upper,
                    "foot_indices": feet,
                    "upright_limit": 0.5,
                },
            ),
            1.0,
        ),
        "foot_position": TerminationRule(
            MdpComponent(
                compute_func=upright_foot_position_failed,
                dynamic_vars={
                    **dynamic,
                    "current_rotation": EnvContext.current.rigid_body_rot,
                    "reference_rotation": EnvContext.mimic.ref_state.rigid_body_rot,
                },
                static_params={
                    "body_indices": feet,
                    "anchor_index": robot_cfg.anchor_body_index,
                    "distance_limit": 0.2,
                },
            )
        ),
        "foot_height": TerminationRule(
            MdpComponent(
                compute_func=upright_foot_height_failed,
                dynamic_vars=dynamic,
                static_params={"body_indices": feet, "height_limit": 0.15},
            )
        ),
    }
    return cfg


def _mlp(keys, output, size, widths):
    return MLPWithConcatConfig(
        in_keys=keys,
        out_keys=[output],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=size,
        layers=[MLPLayerConfig(units=width, activation="silu") for width in widths],
    )


def agent_config(robot_cfg, env_cfg, args):
    cfg = mlp.agent_config(robot_cfg, env_cfg, args)
    keys = [
        "max_coords_obs",
        "mimic_target_poses",
        "historical_previous_max_coords_obs",
        "previous_raw_actions",
        "reduced_coords_obs",
        "target_dof_pos",
    ]
    cfg.model.actor = PPOActorConfig(
        num_out=robot_cfg.number_of_actions,
        actor_logstd=-2.9,
        in_keys=keys.copy(),
        mu_key="mu",
        mu_model=FSQAutoEncoderConfig(
            num_fsq_levels=FSQ_LEVELS,
            num_fsq_scalars=FSQ_SCALARS,
            encoder_out_keys=["latent"],
            decoder_out_keys=["mu"],
            encoder=_mlp(
                ["mimic_target_poses", "target_dof_pos"],
                "latent",
                FSQ_SCALARS,
                ENCODER_WIDTHS,
            ),
            decoder=_mlp(
                [
                    "max_coords_obs",
                    "historical_previous_max_coords_obs",
                    "previous_raw_actions",
                    "latent",
                    "reduced_coords_obs",
                ],
                "mu",
                robot_cfg.number_of_actions,
                DECODER_CRITIC_WIDTHS,
            ),
        ),
    )
    cfg.model.critic = _mlp(keys.copy(), "value", 1, DECODER_CRITIC_WIDTHS)
    # Re-derive the model's key unions after swapping the actor and critic in
    # place (the model derives them once, from its component configs).
    cfg.model.__post_init__()
    cfg.model.actor_optimizer = OptimizerConfig(
        _target_="torch.optim.Adam", lr=2e-5, betas=(0.95, 0.99)
    )
    cfg.model.critic_optimizer = OptimizerConfig(
        _target_="torch.optim.Adam", lr=1e-4, betas=(0.95, 0.99)
    )
    cfg.normalize_rewards = False
    cfg.task_reward_w = 0.8
    cfg.num_mini_epochs = 2
    cfg.gradient_clip_val = 1.0
    cfg.save_inference_checkpoint = True
    cfg.evaluator.max_eval_steps = 400
    cfg.evaluator.save_predicted_motion_lib_every = None
    # GT error remains diagnostic; the low-pose rules define success.
    cfg.evaluator.failure_components.clear()
    tracking = deepcopy(env_cfg.termination_components["tracking_error"].component)
    tracking.static_params.update(upright_mean=True)
    failures = cfg.evaluator.failure_components
    failures["tracking_failure_ratio"] = TerminationRule(tracking, 1.0)
    for name in ("foot_position", "foot_height"):
        failures[name] = deepcopy(env_cfg.termination_components[name])
    return cfg


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    mlp.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    robot_cfg.asset.self_collisions = True
    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.1,
        root_pos_noise=[0.05, 0.05, 0.01],
        root_rot_noise=[0.1, 0.1, 0.2],
    )
    # Full terrain labels feed contact matching; robot partners feed self-contact cost.
    robot_cfg.update_fields(
        contact_bodies="all",
        contact_partners=["terrain"]
        + [f"robot/{name}" for name in robot_cfg.kinematic_info.body_names],
    )
    simulator_cfg.default_robot_friction = 1.0
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.025, 0.025), dof_names=[".*"]
        ),
        latency=LatencyDomainRandomizationConfig(max_latency_ms=0.0),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            body_names=[".*"],
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
        ),
        body_mass=BodyMassDomainRandomizationConfig(
            num_buckets=64,
            mass_range=(0.001, 1.0),
            log_uniform=True,
            body_names=abstract_names_to_body_names(
                ["all_left_hand_bodies", "all_right_hand_bodies"], robot_cfg
            ),
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
    """Mimic playback (mlp's hook), then fixed friction and no randomization."""
    mlp.apply_inference_overrides(configs, args)
    configs.simulator.domain_randomization = None
    configs.terrain.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
