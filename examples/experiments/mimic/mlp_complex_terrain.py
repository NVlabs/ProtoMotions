# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mimic Environment Configuration — Complex Terrain
===================================================

Same as mlp.py but with complex terrain and terrain observations in agent.
"""

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import ComplexTerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.envs.presets import tracking_playback
from protomotions.utils.experiment_configs import ExperimentConfigs
import argparse


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration — complex terrain."""
    return ComplexTerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    scene_file = args.scenes_file
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults)."""
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )
    from protomotions.envs.obs.target_poses import build_max_coords_target_poses
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import (
        compute_action_smoothness,
        compute_pow_rew,
        compute_terrain_foot_contact_match_rew,
    )
    from protomotions.envs.rewards.tracking import (
        compute_gav_rew,
        compute_gr_rew,
        compute_gt_rew,
        compute_gv_rew,
        compute_rh_rew,
    )
    from protomotions.envs.terminations.tracking import max_body_pos_error
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.body_selection import BodySelection
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.terrain import compute_terrain_height_map

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        "max_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations
        ),
        # The height map around the root, over the published terrain names.
        "terrain": MdpComponent.from_kernel(compute_terrain_height_map),
        "previous_actions": MdpComponent.from_kernel(
            compute_historical_actions_from_state, history_steps=1
        ),
        "mimic_target_poses": MdpComponent.from_kernel(
            build_max_coords_target_poses, with_velocities=True
        ),
    }

    termination_components = {
        "tracking_error": TerminationRule(
            MdpComponent.from_kernel(max_body_pos_error), 0.5
        ),
    }

    foot_contact_bodies = (
        robot_cfg.common_naming_to_robot_body_names["all_left_foot_bodies"]
        + robot_cfg.common_naming_to_robot_body_names["all_right_foot_bodies"]
    )
    reward_components = {
        "action_smoothness": MdpComponent.from_kernel(
            compute_action_smoothness,
            meta=RewardMeta(weight=-0.02),
        ),
        "gt_rew": MdpComponent.from_kernel(
            compute_gt_rew,
            meta=RewardMeta(weight=0.5),
            coefficient=-25.0,
        ),
        "gr_rew": MdpComponent.from_kernel(
            compute_gr_rew,
            meta=RewardMeta(weight=0.3),
            coefficient=-5.0,
        ),
        "gv_rew": MdpComponent.from_kernel(
            compute_gv_rew,
            meta=RewardMeta(weight=0.1),
            coefficient=-0.5,
        ),
        "gav_rew": MdpComponent.from_kernel(
            compute_gav_rew,
            meta=RewardMeta(weight=0.2),
            coefficient=-0.1,
        ),
        "rh_rew": MdpComponent.from_kernel(
            compute_rh_rew,
            meta=RewardMeta(weight=0.2),
            coefficient=-100.0,
        ),
        "pow_rew": MdpComponent.from_kernel(
            compute_pow_rew,
            meta=RewardMeta(weight=-1e-5, min_value=-0.5),
        ),
        "contact_match_rew": MdpComponent.from_kernel(
            compute_terrain_foot_contact_match_rew,
            meta=RewardMeta(weight=-0.1, zero_during_grace_period=True),
            required_keys={
                "contact_body_index": foot_contact_bodies,
                "contact_partner_index": ["terrain"],
            },
            contact_body_ids=BodySelection.named(*foot_contact_bodies),
            body_names=foot_contact_bodies,
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
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

    from protomotions.agents.evaluators.presets import mimic_tracking_benchmark

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "terrain", "mimic_target_poses", "previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "terrain", "mimic_target_poses", "previous_actions"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=mimic_tracking_benchmark(0.5),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors for foot contact tracking."""
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"],
        contact_partners=["terrain"],
    )


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Play each clip through to its end for evaluation."""
    tracking_playback(configs.env)
