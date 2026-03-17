# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
import argparse
from protomotions.components.terrains.config import TerrainConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.amp.config import AMPAgentConfig, AMPParametersConfig


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
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
    """Build environment configuration (training defaults).
    
    Uses factory functions for standard observations/rewards/terminations.
    """
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    # Control components - MimicControl provides ref_state for mimic target poses
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    # Observation components configuration
    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "mimic_target_poses": mimic_target_poses_reduced_coords_factory(),
    }

    return EnvConfig(
        max_episode_length=1000,
        control_components=control_components,
        observation_components=observation_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.9,
            resample_on_reset=False,  # We resample on reset to ensure the discriminator and policy observe the same data distribution (random).
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> AMPAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.amp.config import AMPModelConfig, DiscriminatorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig
    from protomotions.envs.component_factories import gt_error_factory, gr_error_factory, max_joint_error_factory

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=["max_coords_obs", "mimic_target_poses"],
            out_keys=["actor_trunk_out"],
            normalize_obs=True,
            norm_clamp_value=5,
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    discriminator_config = DiscriminatorConfig(
        in_keys=["mimic_target_poses_diff"],
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=["mimic_target_poses_diff"],
                out_keys=["disc_logits"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            )
        ],
    )

    agent_config: AMPAgentConfig = AMPAgentConfig(
        _target_="protomotions.agents.mimic.agent_add.MimicADD",
        model=AMPModelConfig(
            in_keys=["max_coords_obs", "mimic_target_poses", "mimic_target_poses_diff"],
            out_keys=["action", "mean_action", "neglogp", "value", "disc_logits"],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        batch_size=args.batch_size,
        task_reward_w=0.0,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
        ),
        amp_parameters=AMPParametersConfig(
            discriminator_reward_threshold=0.03,
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
    # Reuse the mimic apply_inference_overrides function
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../mimic/mlp.py"
    )
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args)

    # Reuse the amp apply_inference_overrides function
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../amp/mlp.py"
    )
    apply_inference_overrides_fn(robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args)
