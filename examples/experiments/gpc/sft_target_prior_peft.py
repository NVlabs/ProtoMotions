# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT bootstrap for PEFT target-reaching on a fixed reference clip.

The frozen tracker rolls out the chosen reference motion, while the PEFT
prior is supervised by cross-entropy against the tracker's FSQ codes.
``task_obs`` comes from the same target observation factory used by RLFT, but
the target position is sourced from a future root XY point on the reference
clip plus jitter. After SFT, RLFT can swap the control source to
``TargetControl`` and keep the same task-conditioning language.
"""

import argparse

from examples.experiments.gpc.prior_context import nearest_surface_obs_params
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

TRACKER_CHECKPOINT = (
    "data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt"
)

PRIOR_TOP_P = 0.99
LOOKAHEAD_SECONDS_MIN = 1.0
LOOKAHEAD_SECONDS_MAX = 5.0
TARGET_JITTER_RADIUS = 0.5
TAR_PROXIMITY_THRESHOLD = 0.3


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--tracker-checkpoint", default=TRACKER_CHECKPOINT)
    parser.add_argument("--prior-checkpoint", required=True)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"],
    )


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def _tracker_future_steps(args: argparse.Namespace):
    from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint

    tracker_checkpoint = getattr(args, "tracker_checkpoint", TRACKER_CHECKPOINT)
    resolved = load_resolved_configs_from_checkpoint(tracker_checkpoint)
    mimic_config = resolved["env"].control_components.get("mimic")
    if mimic_config is None:
        raise ValueError(
            f"Tracker checkpoint '{tracker_checkpoint}' does not define a "
            "'mimic' control component; cannot infer future_steps."
        )
    return mimic_config.future_steps


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        nearest_surface_obs_factory,
        target_obs_factory,
        target_reward_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.control.gpc_sft_reference_target_control import (
        GPCSFTReferenceTargetControlConfig,
    )
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "task_obs": target_obs_factory(),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True
        ),
        "nearest_surface": nearest_surface_obs_factory(
            **nearest_surface_obs_params(robot_cfg),
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=256,
        reset_grace_period=2,
        ref_respawn_offset=0.0,
        control_components={
            "mimic": MimicControlConfig(
                bootstrap_on_episode_end=True,
                future_steps=_tracker_future_steps(args),
            ),
            "target": GPCSFTReferenceTargetControlConfig(
                lookahead_seconds_min=LOOKAHEAD_SECONDS_MIN,
                lookahead_seconds_max=LOOKAHEAD_SECONDS_MAX,
                target_jitter_radius=TARGET_JITTER_RADIUS,
                tar_proximity_threshold=TAR_PROXIMITY_THRESHOLD,
            ),
        },
        observation_components=observation_components,
        reward_components={
            "target": target_reward_factory(weight=1.0),
        },
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.5),
        },
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
):
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import PretrainedModelConfig
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.peft.prior_config import (
        DiscretePriorPEFTConfig,
        DiscretePriorPEFTActorConfig,
        DiscretePriorPEFTSFTAgentConfig,
        DiscretePriorPEFTSFTModelConfig,
    )

    prior_checkpoint = args.prior_checkpoint

    return DiscretePriorPEFTSFTAgentConfig(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=prior_checkpoint,
                module_path="",
            ),
        },
        model=DiscretePriorPEFTSFTModelConfig(
            actor=DiscretePriorPEFTActorConfig(
                in_keys=["task_obs"],
                peft=DiscretePriorPEFTConfig(
                    peft_type="dora",
                    rank=32,
                    alpha=64,
                    temperature=1.0,
                    top_p=0.9,
                    sampling_mode="prior_constraint",
                    prior_top_p=PRIOR_TOP_P,
                    m_clamp=1.6,
                    film_input_norm=True,
                ),
            ),
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.AdamW",
                lr=1e-4,
            ),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=32,
        num_mini_epochs=1,
        normalize_rewards=False,
        gradient_clip_val=50.0,
        save_last_checkpoint_every=10,
        evaluator=EvaluatorConfig(eval_metrics_every=100),
    )


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    from protomotions.envs.control.target_control import (
        KeyboardTargetCommandSourceConfig,
        TargetControlConfig,
    )

    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
    env_cfg.control_components["target"] = TargetControlConfig(
        tar_proximity_threshold=TAR_PROXIMITY_THRESHOLD,
        command_source=KeyboardTargetCommandSourceConfig(),
        enable_fall_termination=False,
    )
