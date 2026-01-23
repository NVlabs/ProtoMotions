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


"""
Mimic Environment Configuration
================================

Full-body motion tracking environment with pose and velocity tracking.
Uses early termination on tracking error and bootstrapping at episode end.
"""


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors for foot contact tracking."""
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
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
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
    )
    from protomotions.envs.terminations import tracking_error_factory

    # Control components configuration
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
    }

    termination_components = {
        "tracking_error": tracking_error_factory(threshold=0.5),
    }

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


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig, MotionWeightsRulesConfig

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
            output_activation="tanh",
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
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
                "previous_actions",
            ],
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
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
            eval_metric_keys=[
                "gt_err",
                "gr_err",
                "gr_err_degrees",
                "gt_rew",
                "gr_rew",
            ],
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
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
            env_cfg.termination_components = None
        if hasattr(env_cfg, "max_episode_length"):
            env_cfg.max_episode_length = 1000000
        if hasattr(env_cfg, "motion_manager"):
            if hasattr(env_cfg.motion_manager, "resample_on_reset"):
                env_cfg.motion_manager.resample_on_reset = True
            if hasattr(env_cfg.motion_manager, "init_start_prob"):
                env_cfg.motion_manager.init_start_prob = 1.0
