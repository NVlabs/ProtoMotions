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
from protomotions.robot_configs.base import RobotConfig, ControlType
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
)
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration with low friction settings."""
    from protomotions.components.terrains.config import (
        TerrainConfig,
        TerrainSimConfig,
        CombineMode,
    )

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
        MimicObsConfig,
        MimicMotionManagerConfig,
    )
    from protomotions.envs.base_env.config import RewardComponentConfig
    from protomotions.envs.obs.config import (
        HumanoidObsConfig,
        MaxCoordsSelfObsConfig,
        SelfObsConfig,
        MimicTargetPoseConfig,
        FuturePoseType,
    )
    from protomotions.envs.utils.rewards import (
        mean_squared_error_exp,
        rotation_error_exp,
        power_consumption_sum,
        norm,
        contact_mismatch_sum,
        impact_force_penalty,
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
            weight=-0.1,
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
                "coefficient": "-10.0",
            },
            weight=0.8,
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
                "x": "current_state.rigid_body_pos[:, 0, 2]",  # Root height (z-coord of body 0)
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
            weight=-2e-5,
            min_value=-0.5,
            zero_during_grace_period=True,
        ),
        "contact_match_rew": RewardComponentConfig(
            function=contact_mismatch_sum,
            variables={
                "sim_contacts": "current_state.rigid_body_contacts",
                "ref_contacts": "ref_state.rigid_body_contacts",
            },
            indices_subset=["all_left_foot_bodies", "all_right_foot_bodies"],
            weight=-0.1,
            zero_during_grace_period=True,
        ),
        "contact_force_change_rew": RewardComponentConfig(
            function=impact_force_penalty,
            variables={
                "current_forces": "current_contact_force_magnitudes",
                "previous_forces": "prev_contact_force_magnitudes",
            },
            indices_subset=["all_left_foot_bodies", "all_right_foot_bodies"],
            weight=-2e-4,
            min_value=-0.5,
            zero_during_grace_period=True,
        ),
    }

    env_config: MimicEnvConfig = MimicEnvConfig(
        ref_respawn_offset=0.01,
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        humanoid_obs=HumanoidObsConfig(
            max_coords_obs=MaxCoordsSelfObsConfig(
                enabled=True,
                num_historical_steps=3,
            ),
            reduced_coords_obs=SelfObsConfig(
                enabled=True,
                num_historical_steps=3,
            ),
        ),
        reward_config=reward_config,
        mimic_early_termination=mimic_early_termination,
        mimic_bootstrap_on_episode_end=True,
        mimic_obs=MimicObsConfig(
            enabled=True,
            mimic_target_pose=MimicTargetPoseConfig(
                enabled=True,
                type=FuturePoseType.MAX_COORDS_SIMPLE,
                with_velocities=True,
            ),
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.1,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=True,
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: MimicEnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build agent configuration (training defaults)."""
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=[
            "reduced_coords_obs",
            "mimic_target_poses",
            "historical_reduced_coords_obs",
        ],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "reduced_coords_obs",
                "mimic_target_poses",
                "historical_reduced_coords_obs",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
            output_activation="tanh",
        ),
    )

    # Critic configuration
    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "historical_max_coords_obs"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "max_coords_obs",
                "historical_max_coords_obs",
                "reduced_coords_obs",
                "historical_reduced_coords_obs",
                "mimic_target_poses",
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
        num_mini_epochs=2,  # seem to make training a bit faster
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
                "contact_match_rew",
                "contact_match_accuracy",
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
            static_friction_range=(0.6, 3.0),  # 0.3 ~ 1.5 since floor is 0.01
            dynamic_friction_range=(0.6, 3.0),  # 0.3 ~ 1.5 since floor is 0.01
            restitution_range=(0.0, 1.0),  # 0.0 ~ 0.5
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    # For mimic: disable early termination during evaluation
    env_cfg.mimic_early_termination = None
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    # Disable domain randomization during evaluation
    simulator_cfg.domain_randomization = None
