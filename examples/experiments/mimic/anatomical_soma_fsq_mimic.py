# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anatomical SOMA motion tracker: the SOMA23 FSQ/PPO tracker on G1-class actuators.

The network, observations, schedule and evaluation are those of the SOMA23 FSQ
tracker (``mimic/fsq.py``, one mini-epoch). The actuator and reward side follow
the G1 BeyondMimic tracker instead, because ``anatomic_soma`` carries G1-class
motors:

* BeyondMimic action scaling: a unit action moves the joint target far enough
  that the PD spring alone would push with a quarter of the joint's effort
  limit; no tanh projection onto the joint range.
* The G1 tracker's Gaussian body-tracking rewards, summed raw (no running
  reward normalization), plus SOMA's foot-contact match. The self-contact
  penalty is omitted; this recipe does not add its sensors.
* G1-style physical randomization (friction, torso centre of mass, hand mass,
  pushes, action noise, reset noise) and no observation noise. Hand mass
  varies by a fraction of its own mass rather than G1's rubber-payload range,
  since a human hand is not a payload slot.
* A wide fixed exploration std (log-std -1.51 instead of the trackers' -2.9).

Terminations keep SOMA's 0.5 m body-error rule. This is the configuration that
trained the released anatomical SOMA tracker, except that the run learned a
per-clip std (one row per training clip, started at -1.51); the shipped
checkpoint carries that table collapsed to its mean row.

Motions must be retargeted to ``anatomic_soma``
(``data/scripts/from_soma_to_anatomical_soma``). A concrete motion file works
locally; a ``slurmrank`` pattern cycles every shard across ranks after each
evaluation so a run on fewer GPUs than shards still sees the full corpus.
"""

from examples.experiments.mimic import fsq
from examples.experiments.mimic.fsq import (
    scene_lib_config as scene_lib_config,
    terrain_config as terrain_config,
)
from protomotions.components.motion_lib import MotionFileSwitchMode, MotionLibConfig
from protomotions.components.terrains.config import CombineMode, TerrainSimConfig
from protomotions.envs.action import make_bm_pd_action_config
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.presets import tracking_playback
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.rewards.regularization import (
    compute_action_rate_l2,
    compute_soft_pos_limit_rew,
)
from protomotions.envs.rewards.tracking import (
    compute_global_body_ang_vel_rew,
    compute_global_body_lin_vel_rew,
    compute_global_orientation_error_exp,
    compute_global_position_error_exp,
    compute_relative_body_ori_rew,
    compute_relative_body_pos_rew,
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

# Mass of one anatomic_soma hand body in the MJCF, in kilograms. Body-mass
# randomization takes absolute ranges, so +/-20% is expressed around it.
ANATOMIC_SOMA_HAND_MASS_KG = 0.780303732731
EFFORT_FRACTION = 0.25
LOGSTD = -1.51


def motion_lib_config(args):
    mode = (
        MotionFileSwitchMode.LIVE
        if "slurmrank" in str(args.motion_file)
        else MotionFileSwitchMode.FIXED
    )
    return MotionLibConfig(motion_file=args.motion_file, motion_file_switch_mode=mode)


def _tracking_rewards(robot_cfg, cfg):
    """The G1 BeyondMimic tracker's rewards on this robot's anchor, hands and feet."""
    upper = [robot_cfg.anchor_body_index] + list(
        robot_cfg.body_indices("all_left_hand_bodies", "all_right_hand_bodies")
    )
    feet = list(robot_cfg.body_indices("all_left_foot_bodies", "all_right_foot_bodies"))
    tracking_bodies = upper + feet

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

    soft = robot_cfg.control.soft_pos_limit
    return {
        "gt_rew": position(tracking_bodies, 1.0, 0.3),
        "gr_rew": orientation(tracking_bodies, 1.0, 0.4),
        "hand_torso_pos": position(upper, 2.0, 0.1),
        "hand_torso_rot": orientation(upper, 0.4, 0.1),
        "relative_body_pos": MdpComponent.from_kernel(
            compute_relative_body_pos_rew, meta=RewardMeta(weight=1.0), sigma=0.3
        ),
        "relative_body_ori": MdpComponent.from_kernel(
            compute_relative_body_ori_rew, meta=RewardMeta(weight=1.0), sigma=0.4
        ),
        "gv_rew": MdpComponent.from_kernel(
            compute_global_body_lin_vel_rew, meta=RewardMeta(weight=1.0), sigma=1.0
        ),
        "gav_rew": MdpComponent.from_kernel(
            compute_global_body_ang_vel_rew, meta=RewardMeta(weight=1.0), sigma=3.14
        ),
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={"dof_pos": EnvContext.current.dof_pos},
            meta=RewardMeta(weight=-10.0),
            static_params={
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower * soft,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper * soft,
            },
        ),
        "action_smoothness": MdpComponent.from_kernel(
            compute_action_rate_l2, meta=RewardMeta(weight=-0.1)
        ),
        "contact_match_rew": cfg.reward_components["contact_match_rew"],
    }


def env_config(robot_cfg, args):
    cfg = fsq.env_config(robot_cfg, args)
    cfg.action_config = make_bm_pd_action_config(
        robot_cfg, effort_fraction=EFFORT_FRACTION
    )
    cfg.reward_components = _tracking_rewards(robot_cfg, cfg)
    return cfg


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    if args.robot_name != "anatomic_soma":
        raise ValueError(
            "The anatomical SOMA tracker requires --robot-name anatomic_soma"
        )
    fsq.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    robot_cfg.asset.self_collisions = True
    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.1,
        root_pos_noise=[0.05, 0.05, 0.01],
        root_rot_noise=[0.1, 0.1, 0.2],
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
            mass_range=(
                0.8 * ANATOMIC_SOMA_HAND_MASS_KG,
                1.2 * ANATOMIC_SOMA_HAND_MASS_KG,
            ),
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


def agent_config(robot_cfg, env_cfg, args):
    cfg = fsq.agent_config(robot_cfg, env_cfg, args)
    cfg.num_mini_epochs = 1
    cfg.model.actor.actor_logstd = LOGSTD
    # The G1 terms are bounded Gaussian kernels the G1 tracker sums raw, so the
    # running reward normalization is off. It is one switch for the whole reward,
    # so SOMA's contact-match term is summed raw as well.
    cfg.normalize_rewards = False
    return cfg


def apply_inference_overrides(configs, args):
    """Play each clip through, with fixed friction and no randomization."""
    tracking_playback(configs.env)
    configs.simulator.domain_randomization = None
    configs.terrain.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
