# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-teacher Metatron G1 distillation with a spherical conditional VAE.

Training requires an immutable teacher checkpoint and its resolved configs.
Inference uses saved student configs and never opens that teacher checkpoint.
"""

import argparse
from copy import deepcopy

import torch

from examples.experiments.masked_mimic.deployable_g1_observations import (
    HISTORY_STEPS,
    configure_observations,
    configure_robot,
)
from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    MLPLayerConfig,
    MaskedMeanConfig,
    ModuleContainerConfig,
    ModuleOperationForwardConfig,
    ModuleOperationReshapeConfig,
    ObsProcessorConfig,
    TransformerConfig,
)
from protomotions.agents.evaluators.config import MaskedMimicActionSource, MaskedMimicEvaluatorConfig

from protomotions.agents.supervised.config import RolloutActor
from protomotions.utils.experiment_configs import ExperimentConfigs
from protomotions.agents.supervised.expert_utils import (
    get_expert_observation_components,
    validate_expert_sensor_contract,
)
from protomotions.agents.supervised.masked_mimic_config import (
    KLDScheduleConfig,
    LatentGeometry,
    MaskedMimicModelConfig,
    MaskedMimicSupervisedAgentConfig,
    MaskedMimicVAEConfig,
    TemporalLossConfig,
)
from protomotions.components.motion_lib import MotionFileSwitchMode, MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.obs.cast import to_float
from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
from protomotions.envs.obs.masked_mimic import compute_local_target_poses
from protomotions.envs.obs.target_poses import build_max_coords_target_poses
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.rewards.tracking import (
    compute_gav_rew,
    compute_gr_rew,
    compute_gt_rew,
    compute_gv_rew,
    compute_rh_rew,
)
from protomotions.envs.terminations.tracking import mean_body_pos_error
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControlConfig,
    MaskedMimicStyleConfig,
    MaskedMimicVelocityCommandConfig,
)
from protomotions.envs.control.mimic_control import MimicControlConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.obs.masked_mimic import (
    compute_target_masks_only,
    compute_target_time_offsets,
)
from protomotions.envs.obs.cast import to_float
from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    SensorConfig,
)
from protomotions.envs.terminations.low_pose import tracking_failure_ratio
from protomotions.utils.experiment_configs import load_experiment_configs_from_checkpoint

NUM_FUTURE_STEPS = 5
LATENT_DIM = 64
TOKEN_DIM = 1024
STATE_KEYS = ["fk_max_coords_obs", "reduced_coords_obs", "last_action"]


def additional_experiment_arguments(parser):
    parser.add_argument(
        "--expert-model-path",
        required=True,
        help="Frozen five-body G1 tracker checkpoint",
    )


def _teacher(args):
    if not args.expert_model_path:
        raise ValueError("Training requires --expert-model-path")
    return load_experiment_configs_from_checkpoint(args.expert_model_path)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    expert = _teacher(args)
    for field in ("dof_names", "body_names"):
        if getattr(robot_cfg.kinematic_info, field) != getattr(
            expert.robot.kinematic_info, field
        ):
            raise ValueError(f"Student and teacher {field} must match exactly")
    configure_robot(robot_cfg)
    robot_cfg.control = deepcopy(expert.robot.control)
    robot_cfg.reset_noise = deepcopy(expert.robot.reset_noise)
    robot_cfg.asset.self_collisions = expert.robot.asset.self_collisions
    teacher_sim = expert.simulator
    for field in (
        "domain_randomization",
        "default_robot_friction",
        "pd_target_max_accel",
    ):
        setattr(simulator_cfg, field, deepcopy(getattr(teacher_sim, field)))
    if type(simulator_cfg.sim) is not type(teacher_sim.sim):
        raise ValueError(
            "Train student and teacher with the same simulator; backend switching is inference-only"
        )
    simulator_cfg.sim = deepcopy(teacher_sim.sim)
    # Preserve teacher contact-body ordering, but omit robot partners: this
    # student has no self-contact cost. Physical self-collisions stay unchanged.
    robot_cfg.update_fields(
        contact_bodies=deepcopy(expert.robot.contact_bodies),
        contact_partners=["terrain"],
    )


def terrain_config(args):
    return deepcopy(_teacher(args).terrain)


def scene_lib_config(args):
    return SceneLibConfig(scene_file=args.scenes_file)


def motion_lib_config(args):
    mode = (
        MotionFileSwitchMode.LIVE
        if "slurmrank" in str(args.motion_file)
        else MotionFileSwitchMode.FIXED
    )
    return MotionLibConfig(motion_file=args.motion_file, motion_file_switch_mode=mode)


# The teacher's "clean" state namespaces, renamed to simulator truth (a
# sensor-free teacher's clean state IS simulator truth).
TEACHER_CLEAN_TO_PRIVILEGED = {"clean": "current", "clean_historical": "historical"}


def env_config(robot_cfg, args):
    teacher = _teacher(args)
    teacher_env = deepcopy(teacher.env)
    if teacher_env.sensors is not None:
        raise ValueError(
            "This recipe requires a privileged five-body teacher without sensors"
        )
    # 'clean' in a sensor-free teacher means simulator truth. Bind that source
    # explicitly before adding the student's physical sensor suite: the clean
    # namespaces are renamed to the privileged ones.
    for component in teacher_env.observation_components.values():
        for name, path in component.dynamic_vars.items():
            parts = str(path).split(".")
            if parts[0] in TEACHER_CLEAN_TO_PRIVILEGED:
                value = getattr(EnvContext, TEACHER_CLEAN_TO_PRIVILEGED[parts[0]])
                for part in parts[1:]:
                    value = getattr(value, part)
                component.dynamic_vars[name] = value
    controls = [
        value
        for value in teacher_env.control_components.values()
        if isinstance(value, MimicControlConfig)
    ]
    if len(controls) != 1:
        raise ValueError("Teacher must declare exactly one mimic reference stream")
    bodies = torch.tensor(robot_cfg.body_indices(*robot_cfg.trackable_bodies_subset))
    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(pos_noise=0.01, vel_noise=0.5),
        imu=IMUSensor(quat_noise=0.05, gyro_noise=0.2),
        localization=LocalizationSensor(
            body_name=robot_cfg.resolved_anchor_body_name, noise=0.01
        ),
    )
    control = MaskedMimicControlConfig(
        reference_height_smoothing_seconds=0.2,
        low_pose_grace_seconds=0.5,
        future_steps=deepcopy(controls[0].future_steps),
        num_masked_future_steps=NUM_FUTURE_STEPS,
        sparse_target_condition_probability=0.5,
        local_target_condition_probability=0.5,
        anchor_pose_submasking=True,
        style=MaskedMimicStyleConfig(
            visible_frame_count_probabilities=(0.50, 0.15, 0.15, 0.10, 0.07, 0.03)
        ),
        velocity_command=MaskedMimicVelocityCommandConfig(horizon_seconds=0.5),
    )
    cfg = EnvConfig(
        motion_manager=deepcopy(teacher_env.motion_manager),
        action_config=deepcopy(teacher_env.action_config),
        sensors=sensors,
        ref_respawn_offset=teacher_env.ref_respawn_offset,
        reset_grace_period=5,
        control_components={"masked_mimic": control},
        observation_components={
            "max_coords_obs": MdpComponent.from_kernel(
                compute_humanoid_max_coords_observations,
                local_obs=True,
                root_height_obs=True,
            ),
            "mimic_target_poses": MdpComponent.from_kernel(
                build_max_coords_target_poses,
                with_velocities=True,
                with_relative=True,
            ),
            "masked_mimic_local_target_poses": MdpComponent.from_kernel(
                compute_local_target_poses, conditionable_body_ids=bodies
            ),
            # Untimed style inputs, canonicalized and masked by the control; the
            # model encodes frames separately and pools only visible frames.
            "masked_mimic_style_poses": MdpComponent.from_kernel(
                to_float, x=EnvContext.masked_mimic.style_pose_samples
            ),
            "masked_mimic_style_velocities": MdpComponent.from_kernel(
                to_float, x=EnvContext.masked_mimic.style_velocity_samples
            ),
            "masked_mimic_style_masks": MdpComponent.from_kernel(
                to_float, x=EnvContext.masked_mimic.style_condition_masks
            ),
            "masked_mimic_style_frame_masks": MdpComponent.from_kernel(
                to_float, x=EnvContext.masked_mimic.style_frame_masks
            ),
        },
        reward_components={
            "gt_rew": MdpComponent.from_kernel(
                compute_gt_rew,
                meta=RewardMeta(weight=0.5),
            ),
            "gr_rew": MdpComponent.from_kernel(
                compute_gr_rew,
                meta=RewardMeta(weight=0.3),
            ),
            "gv_rew": MdpComponent.from_kernel(
                compute_gv_rew,
                meta=RewardMeta(weight=0.1),
            ),
            "gav_rew": MdpComponent.from_kernel(
                compute_gav_rew,
                meta=RewardMeta(weight=0.1),
            ),
            "rh_rew": MdpComponent.from_kernel(
                compute_rh_rew,
                meta=RewardMeta(weight=0.2),
            ),
        },
    )
    if not isinstance(cfg.motion_manager, MimicMotionManagerConfig):
        raise ValueError("Teacher requires a MimicMotionManagerConfig")
    cfg.motion_manager.realign_motion_with_humanoid_on_each_step = False
    configure_observations(robot_cfg, cfg)
    for prefix, mask_path, times_path, frames_path in (
        (
            "masked_mimic",
            EnvContext.masked_mimic.target_bodies_masks,
            EnvContext.masked_mimic.time_offsets,
            EnvContext.masked_mimic.target_poses_masks,
        ),
        (
            "masked_mimic_local",
            EnvContext.masked_mimic.local_target_bodies_masks,
            EnvContext.masked_mimic.local_time_offsets,
            EnvContext.masked_mimic.local_target_poses_masks,
        ),
    ):
        variables = {"masked_mimic_target_bodies_masks": mask_path}
        if prefix == "masked_mimic":
            variables.update(
                anchor_translation_mask=EnvContext.masked_mimic.anchor_translation_mask,
                anchor_rotation_mask=EnvContext.masked_mimic.anchor_rotation_mask,
            )
        cfg.observation_components[prefix + "_target_masks"] = MdpComponent(
            compute_func=compute_target_masks_only,
            dynamic_vars=variables,
            static_params={"conditionable_body_ids": bodies},
        )
        cfg.observation_components[prefix + "_target_times"] = MdpComponent(
            compute_func=compute_target_time_offsets,
            dynamic_vars={"masked_mimic_time_offsets": times_path},
        )
        cfg.observation_components[prefix + "_target_poses_masks"] = MdpComponent(
            compute_func=to_float,
            dynamic_vars={"x": frames_path},
        )
    validate_expert_sensor_contract(teacher_env, teacher.agent, sensors)
    expert_obs = get_expert_observation_components(
        teacher_env, teacher.agent, list(cfg.observation_components)
    )
    cfg.observation_components.update(expert_obs)
    feet = list(
        robot_cfg.body_indices("all_left_foot_bodies", "all_right_foot_bodies")
    )
    # The low-pose ratio is normalised: training and evaluation both cut it at 1.
    low_pose = {
        "current_position": EnvContext.current.rigid_body_pos,
        "reference_position": EnvContext.mimic.ref_state.rigid_body_pos,
        "smoothed_ref_anchor_height": EnvContext.mimic.smoothed_ref_anchor_height,
    }
    cfg.termination_components = {
        "tracking_error": TerminationRule(
            MdpComponent(
                compute_func=tracking_failure_ratio,
                dynamic_vars=low_pose,
                static_params={
                    "upper_body_indices": [
                        index for index in bodies.tolist() if index not in feet
                    ],
                    "foot_indices": feet,
                    "upright_limit": 0.25,
                },
            ),
            1.0,
        )
    }
    return cfg


def _mlp(keys, output, size, widths=(512, 256, 128), normalize=False):
    return MLPWithConcatConfig(
        in_keys=keys,
        out_keys=[output],
        num_out=size,
        normalize_obs=normalize,
        layers=[MLPLayerConfig(units=n, activation="silu") for n in widths],
    )


def _container(models, outputs):
    produced = {key for model in models for key in model.out_keys}
    inputs = list(
        dict.fromkeys(
            key for model in models for key in model.in_keys if key not in produced
        )
    )
    return ModuleContainerConfig(in_keys=inputs, out_keys=outputs, models=models)


def _goal_modules(keys, output, frames, normalized_keys):
    """Reshape masked observations, normalize values, then encode each frame.

    Observation kernels own body/property masking. Masks and time-to-go stay
    unnormalized; pooling/attention owns frame visibility. ForwardConfig shares
    value statistics across frames by flattening batch and time internally.
    """
    models = []
    for key in keys:
        operations = [ModuleOperationReshapeConfig(["batch_size", frames, -1])]
        if key in normalized_keys:
            operations.append(ModuleOperationForwardConfig())
        models.append(ObsProcessorConfig(
            in_keys=[key],
            out_keys=[key + "_sequence"],
            normalize_obs=key in normalized_keys,
            module_operations=operations,
        ))
    models.append(_mlp(
        [key + "_sequence" for key in keys], output, TOKEN_DIM, (1024,) * 3,
    ))
    return models


def agent_config(robot_cfg, env_cfg, args):
    prior_models = [
        _mlp(STATE_KEYS, "current_token", TOKEN_DIM, (1024,) * 3, normalize=True)
    ]
    prior_models.append(
        ObsProcessorConfig(
            in_keys=["historical_proprioception"],
            out_keys=["history_sequence"],
            normalize_obs=True,
            module_operations=[
                ModuleOperationReshapeConfig(["batch_size", len(HISTORY_STEPS), -1]),
                ModuleOperationForwardConfig(),
            ],
        )
    )
    prior_models.append(
        _mlp(["history_sequence"], "history_tokens", TOKEN_DIM, (1024,) * 3)
    )
    # Frame masks go directly to pooling/attention, not into the MLP.
    goals = (
        # (token, frames, value keys, frame-mask key, normalized keys)
        (
            "global_tokens",
            NUM_FUTURE_STEPS,
            [
                "fk_masked_mimic_target_poses",
                "masked_mimic_target_masks",
                "masked_mimic_target_times",
            ],
            "masked_mimic_target_poses_masks",
            ["fk_masked_mimic_target_poses"],
        ),
        (
            "local_tokens",
            NUM_FUTURE_STEPS,
            [
                "masked_mimic_local_target_poses",
                "masked_mimic_local_target_masks",
                "masked_mimic_local_target_times",
            ],
            "masked_mimic_local_target_poses_masks",
            ["masked_mimic_local_target_poses"],
        ),
        (
            "style_tokens",
            env_cfg.control_components["masked_mimic"].style.num_frames,
            [
                "masked_mimic_style_poses",
                "masked_mimic_style_masks",
                "masked_mimic_style_velocities",
            ],
            "masked_mimic_style_frame_masks",
            ["masked_mimic_style_poses", "masked_mimic_style_velocities"],
        ),
        (
            "velocity_tokens",
            1,
            ["masked_mimic_velocity_command", "masked_mimic_velocity_command_mask"],
            "masked_mimic_velocity_command_mask",
            [],
        ),
    )
    for token, frames, keys, _, normalized_keys in goals:
        prior_models.extend(_goal_modules(keys, token, frames, normalized_keys))
    prior_models.append(
        MaskedMeanConfig(
            in_keys=["style_tokens", "masked_mimic_style_frame_masks"],
            out_keys=["style_token"],
        )
    )
    masks = {
        "style_token" if token == "style_tokens" else token: frame_mask
        for token, _, _, frame_mask, _ in goals
    }
    prior_models.append(
        TransformerConfig(
            in_keys=["current_token", "history_tokens", *masks, *masks.values()],
            out_keys=["goal_features"],
            input_and_mask_mapping=masks,
            transformer_token_size=TOKEN_DIM,
            latent_dim=TOKEN_DIM,
            num_heads=4,
            ff_size=4096,
            num_layers=6,
            activation="silu",
            output_activation="silu",
            dropout=0.0,
        )
    )
    prior_models.extend(
        _mlp(["goal_features"], key, LATENT_DIM)
        for key in ("prior_mu", "prior_logvar")
    )
    encoder = _container(
        [
            ObsProcessorConfig(
                in_keys=["max_coords_obs", "mimic_target_poses"],
                out_keys=["privileged_normalized"],
                normalize_obs=True,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            _mlp(
                ["privileged_normalized"],
                "privileged_features",
                1024,
                (1024,) * 3,
            ),
            _mlp(["privileged_features"], "encoder_mu", LATENT_DIM),
            _mlp(["privileged_features"], "encoder_logvar", LATENT_DIM),
        ],
        ["encoder_mu", "encoder_logvar"],
    )
    trunk = _container(
        [
            # Decoder observations are supplied directly, not replaced by a
            # learned prior feature. Only the sampled latent comes from the VAE.
            ObsProcessorConfig(
                in_keys=STATE_KEYS,
                out_keys=["decoder_state"],
                normalize_obs=True,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            _mlp(["vae_latent"], "solution_features", LATENT_DIM),
            _mlp(
                ["decoder_state", "solution_features"],
                "actor_trunk_out",
                robot_cfg.number_of_actions,
                (1024,) * 3,
            ),
        ],
        ["actor_trunk_out", "solution_features"],
    )
    model = MaskedMimicModelConfig(
        encoder=encoder,
        prior=_container(prior_models, ["prior_mu", "prior_logvar"]),
        trunk=trunk,
        vae=MaskedMimicVAEConfig(
            vae_latent_dim=LATENT_DIM,
            latent_geometry=LatentGeometry.SPHERE,
            prior_mean_unit_norm_coeff=0.001,
            kld_schedule=KLDScheduleConfig(
                start_epoch=500, end_epoch=2000, cyclical=True
            ),
        ),
        temporal_losses=[
            TemporalLossConfig("privileged_action", 0.05),
            TemporalLossConfig("solution_features", 0.001),
        ],
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )
    tracking = deepcopy(env_cfg.termination_components["tracking_error"].component)
    return MaskedMimicSupervisedAgentConfig(
        model=model,
        expert_model_path=args.expert_model_path,
        rollout_actor=RolloutActor.STUDENT,
        expert_rollout_probability=0.1,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        num_mini_epochs=2,
        evaluator=MaskedMimicEvaluatorConfig(
            action_source=MaskedMimicActionSource.POSTERIOR,
            max_eval_steps=400,
            save_predicted_motion_lib_every=None,
            metric_components={
                "gt_error": MdpComponent.from_kernel(mean_body_pos_error),
                "tracking_failure_ratio": tracking,
            },
            failure_components={
                "tracking_failure_ratio": TerminationRule(tracking, 1.0),
            },
            score_key="eval/success_rate",
        ),
    )


def apply_inference_overrides(
    configs: ExperimentConfigs, args: argparse.Namespace
) -> None:
    """Drop the training-only goal observations and keep playback deterministic.

    Dropping the expert and its observations, the prior action path and the
    long episode are framework inference defaults now (the MaskedMimic
    supervised agent and EnvConfig apply them); the two goal-observation keys,
    clearing terminations, no domain randomization and no reset noise stay.
    """
    for key in ("max_coords_obs", "mimic_target_poses"):
        configs.env.observation_components.pop(key, None)
    configs.env.termination_components = {}
    configs.simulator.domain_randomization = None
    configs.robot.reset_noise = None
