# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a GPC prior over tracker FSQ codes.

The frozen tracker supplies target FSQ codes and the latent decoder. The
supervised agent packs those FSQ codes into categorical prior tokens and trains
a transformer prior from ``max_coords_obs`` to that token sequence.
"""

import argparse
from copy import deepcopy
from typing import Tuple

from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.supervised.latent_prior_config import (
    DiscreteAutoregressiveLatentSupervisedAgentConfig,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.presets import tracking_playback
from protomotions.envs.steps import normalize_steps
from protomotions.utils.experiment_configs import ExperimentConfigs
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


TRACKER_MODULE_CONFIG_PATH = "agent.model.actor.mu_model"


def _tracker(tracker_checkpoint: str) -> ExperimentConfigs:
    """The frozen tracker's resolved configs, loaded afresh on each call.

    Both the student's env (for the mimic future steps) and its agent (for the
    embedded decoder config) read from this one bundle.
    """
    from protomotions.utils.experiment_configs import (
        load_experiment_configs_from_checkpoint,
    )

    return load_experiment_configs_from_checkpoint(tracker_checkpoint)


def _tracker_future_steps(tracker_checkpoint: str) -> Tuple[int, ...]:
    """The tracker's published future steps, as the tuple the student repeats
    (the saved value may still be the int an override wrote)."""
    mimic_config = _tracker(tracker_checkpoint).env.control_components.get("mimic")
    if mimic_config is None:
        raise ValueError(
            f"Tracker checkpoint '{tracker_checkpoint}' does not define a "
            "'mimic' control component."
        )
    return normalize_steps(mimic_config.future_steps)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tracker-checkpoint",
        required=True,
        help="Frozen FSQ tracker checkpoint used for target latents and decoding.",
    )


def terrain_config(args: argparse.Namespace):
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file
    return SceneLibConfig(
        scene_file=scene_file,
    )


def motion_lib_config(args: argparse.Namespace):
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )
    from protomotions.envs.obs.target_poses import (
        build_max_coords_target_poses,
    )
    from protomotions.envs.terminations.tracking import max_body_pos_error
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    future_steps = _tracker_future_steps(args.tracker_checkpoint)

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        reset_grace_period=5,
        ref_respawn_offset=0.0,
        control_components={
            "mimic": MimicControlConfig(
                bootstrap_on_episode_end=True,
                future_steps=future_steps,
            )
        },
        observation_components={
            "max_coords_obs": MdpComponent.from_kernel(
                compute_humanoid_max_coords_observations
            ),
            "mimic_target_poses": MdpComponent.from_kernel(
                build_max_coords_target_poses, with_velocities=True
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
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> DiscreteAutoregressiveLatentSupervisedAgentConfig:
    from protomotions.agents.common.config import (
        DiscreteAutoregressiveTransformerConfig,
        MLPLayerConfig,
        MLPWithConcatConfig,
        ModuleContainerConfig,
        PretrainedModelConfig,
    )
    from protomotions.agents.common.latent import (
        LATENT_KEY,
        LATENT_LOGITS_KEY,
        TARGET_LATENT_KEY,
    )
    from protomotions.agents.common.supervision import (
        SupervisionLossConfig,
        SupervisionLossType,
    )
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.agents.supervised.latent_prior_config import (
        DiscreteAutoregressiveLatentPriorModelConfig,
    )
    from protomotions.agents.supervised.config import RolloutActor
    from protomotions.agents.common.pretrained import get_path

    tracker_checkpoint = args.tracker_checkpoint

    fsq_scalars_per_prior_token = 5
    prior_hidden_dim = 1024
    context_in_keys = ["max_coords_obs"]
    prior_transformer_config = DiscreteAutoregressiveTransformerConfig(
        token_key="prior_tokens",
        logits_key=LATENT_LOGITS_KEY,
        generated_tokens_key=LATENT_KEY,
        context_encoder=ModuleContainerConfig(
            in_keys=context_in_keys,
            models=[
                MLPWithConcatConfig(
                    in_keys=context_in_keys,
                    out_keys=["context_embedding"],
                    num_out=prior_hidden_dim,
                    normalize_obs=True,
                    layers=[
                        MLPLayerConfig(
                            units=prior_hidden_dim,
                            activation="gelu",
                        )
                    ],
                )
            ],
        ),
        d_model=prior_hidden_dim,
        num_heads=4,
        num_layers=6,
        ff_size=4096,
        dropout=0.1,
        activation="gelu",
        num_tokens=0,
        vocab_size=0,
    )

    model_config = DiscreteAutoregressiveLatentPriorModelConfig(
        latent_decoder=PretrainedModelConfig(
            checkpoint_path=tracker_checkpoint,
            module_path="actor.mu",
            # Embed the decoder's config from the tracker bundle at build, so the
            # student bundle is self-contained and inference never opens the
            # tracker (the decoder's weights load with the student checkpoint).
            module_config=deepcopy(
                get_path(_tracker(tracker_checkpoint), TRACKER_MODULE_CONFIG_PATH)
            ),
        ),
        prior=prior_transformer_config,
        fsq_scalars_per_prior_token=fsq_scalars_per_prior_token,
        temperature=1.0,
        top_p=0.9,
        optimizer=OptimizerConfig(
            _target_="torch.optim.AdamW",
            lr=1e-4,
            weight_decay=0.01,
        ),
    )

    return DiscreteAutoregressiveLatentSupervisedAgentConfig(
        model=model_config,
        rollout_actor=RolloutActor.EXPERT,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=64,
        num_mini_epochs=1,
        gradient_clip_val=100.0,
        save_last_checkpoint_every=10,
        save_inference_checkpoint=True,
        evaluator=EvaluatorConfig(eval_metrics_every=200),
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.DISCRETE_CROSS_ENTROPY,
            prediction_key=LATENT_LOGITS_KEY,
            target_key=TARGET_LATENT_KEY,
            label_smoothing=0.01,
            log_prefix="prior",
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    # Match IsaacLab's implicit robot material when this policy was trained.
    simulator_cfg.default_robot_friction = 0.5


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Play each clip through to its end for evaluation."""
    tracking_playback(configs.env)
