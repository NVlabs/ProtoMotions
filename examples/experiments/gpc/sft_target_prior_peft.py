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
from typing import Tuple

from examples.experiments.gpc.prior_context import nearest_surface_obs_params
from protomotions.envs.presets import tracking_playback
from protomotions.envs.steps import normalize_steps
from protomotions.utils.experiment_configs import ExperimentConfigs
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

TRACKER_CHECKPOINT = "results/amass_tracker/score_based.ckpt"
PRIOR_CHECKPOINT = "results/fsq_prior_amass/last.ckpt"

PRIOR_TOP_P = 0.99
LOOKAHEAD_SECONDS_MIN = 1.0
LOOKAHEAD_SECONDS_MAX = 5.0
TARGET_JITTER_RADIUS = 0.5
TAR_PROXIMITY_THRESHOLD = 0.3


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--tracker-checkpoint", default=TRACKER_CHECKPOINT)
    parser.add_argument("--prior-checkpoint", default=PRIOR_CHECKPOINT)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    robot_cfg.update_fields(
        contact_bodies=None,
    )


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def _tracker(tracker_checkpoint: str) -> ExperimentConfigs:
    """The frozen tracker's resolved configs, loaded afresh on each call; the
    student's env and its agent config both read this one bundle."""
    from protomotions.utils.experiment_configs import (
        load_experiment_configs_from_checkpoint,
    )

    return load_experiment_configs_from_checkpoint(tracker_checkpoint)


def _tracker_future_steps(args: argparse.Namespace) -> Tuple[int, ...]:
    """The tracker's published future steps, as the tuple the student repeats
    (the saved value may still be the int an override wrote)."""
    tracker_checkpoint = args.tracker_checkpoint
    mimic_config = _tracker(tracker_checkpoint).env.control_components.get("mimic")
    if mimic_config is None:
        raise ValueError(
            f"Tracker checkpoint '{tracker_checkpoint}' does not define a "
            "'mimic' control component; cannot infer future_steps."
        )
    return normalize_steps(mimic_config.future_steps)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.obs.target import compute_target_obs
    from protomotions.envs.obs.nearest_surface_obs import (
        compute_nearest_surface_vectors,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )
    from protomotions.envs.obs.target_poses import (
        build_max_coords_target_poses,
    )
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.task import compute_target_rew
    from protomotions.envs.terminations.tracking import max_body_pos_error
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.control.gpc_sft_reference_target_control import (
        GPCSFTReferenceTargetControlConfig,
    )
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    observation_components = {
        "max_coords_obs": MdpComponent.from_kernel(
            compute_humanoid_max_coords_observations
        ),
        "task_obs": MdpComponent.from_kernel(compute_target_obs),
        "mimic_target_poses": MdpComponent.from_kernel(
            build_max_coords_target_poses, with_velocities=True
        ),
        "nearest_surface": MdpComponent.from_kernel(
            compute_nearest_surface_vectors,
            **nearest_surface_obs_params(
                robot_cfg, with_objects=args.scenes_file is not None
            ),
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
            "target": MdpComponent.from_kernel(
                compute_target_rew,
                meta=RewardMeta(weight=1.0),
            ),
        },
        termination_components={
            "tracking_error": TerminationRule(
                MdpComponent.from_kernel(max_body_pos_error), 0.5
            ),
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
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Play each clip to its end and drive the target from the keyboard."""
    from protomotions.envs.control.command_source import (
        KeyboardTargetCommandSourceConfig,
    )
    from protomotions.envs.control.target_control import (
        TargetControlConfig,
    )

    tracking_playback(configs.env)
    configs.env.control_components["target"] = TargetControlConfig(
        tar_proximity_threshold=TAR_PROXIMITY_THRESHOLD,
        command_source=KeyboardTargetCommandSourceConfig(),
        enable_fall_termination=False,
    )
