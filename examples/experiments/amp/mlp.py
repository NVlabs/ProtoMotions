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
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.amp.config import AMPAgentConfig
import argparse


# Dilated history steps for temporal context (used by actor and discriminator)
HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]


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
    """Build environment configuration (training defaults).
    
    Uses MdpComponent-based component configuration with explicit context bindings:
        MdpComponent(compute_func=compute_fn, dynamic_vars={...}, static_params={...})
    """
    from protomotions.envs.component_factories import max_coords_obs_factory, historical_max_coords_obs_factory
    from protomotions.envs.motion_manager.config import MotionManagerConfig
    from protomotions.envs.action import make_pd_action_config

    # Observation components configuration
    observation_components = {
        # Humanoid self-observations (current state)
        "max_coords_obs": max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        # Historical observations for AMP discriminator (from StateHistoryBuffer)
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=HISTORY_STEPS,
        ),
    }

    env_config: EnvConfig = EnvConfig(
        max_episode_length=300,  # Training default (eval override applied automatically)
        num_state_history_steps=max(HISTORY_STEPS),  # Store enough history for max dilation
        observation_components=observation_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MotionManagerConfig(
            init_start_prob=0.5  # Bias agent to start at the beginning of the motion to prevent getting stuck in a local-minima (standing still).
        ),
    )

    return env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> AMPAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig, ModuleContainerConfig
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.amp.config import (
        AMPModelConfig,
        DiscriminatorConfig,
        AMPParametersConfig,
    )
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs"],
            out_keys=["actor_trunk_out"],
            normalize_obs=True,
            norm_clamp_value=5,
            num_out=robot_config.number_of_actions,
            layers=[
                MLPLayerConfig(units=512, activation="relu"),
                MLPLayerConfig(units=256, activation="relu"),
            ],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
    )

    discriminator_config = DiscriminatorConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
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

    disc_critic_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "historical_max_coords_obs"],
                out_keys=["disc_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=256, activation="relu"),
                ],
            )
        ],
    )

    # Reference observation components for discriminator expert data
    # Agent injects motion_lib/motion_ids/motion_times/dt at runtime (not in EnvContext)
    from protomotions.envs.mdp_component import MdpComponent
    
    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},  # All parameters injected by agent
            static_params={"history_steps": HISTORY_STEPS},
        ),
    }

    agent_config: AMPAgentConfig = AMPAgentConfig(
        model=AMPModelConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs"],
            out_keys=["action", "mean_action", "neglogp", "value", "disc_logits", "disc_value"],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            disc_critic=disc_critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        reference_obs_components=reference_obs_components,
        batch_size=args.batch_size,
        task_reward_w=0.0,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        amp_parameters=AMPParametersConfig(
            discriminator_reward_threshold=0.02,  # Training default (eval override in apply_inference_overrides if needed)
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
    # For AMP: disable discriminator reward during evaluation
    if agent_cfg is not None and hasattr(agent_cfg, "amp_parameters"):
        agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0

    if env_cfg is not None:
        if hasattr(env_cfg, "max_episode_length"):
            env_cfg.max_episode_length = 1000000
        if hasattr(env_cfg, "motion_manager"):
            if hasattr(env_cfg.motion_manager, "init_start_prob"):
                env_cfg.motion_manager.init_start_prob = 1.0
