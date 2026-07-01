# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer checks for example experiment config modules.

These keep FSQ/GPC example configs wired to the common model-building blocks
without starting training or loading real checkpoints.
"""

import argparse
import inspect
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from examples.experiments.gpc import prior as gpc_prior
from examples.experiments.gpc import task_steering_headvel_prior_peft
from examples.experiments.gpc import sft_target_prior_peft
from examples.experiments.gpc import task_target_prior_peft
from examples.experiments.gpc import task_steering_headvel_prior_peft_amp
from examples.experiments.gpc import task_target_prior_peft_amp
from examples.experiments.mimic import fsq as mimic_fsq
from protomotions.agents.common.latent import LATENT_KEY, LATENT_LOGITS_KEY
from protomotions.agents.supervised.config import RolloutActor


class _RobotConfig:
    number_of_actions = 1

    def __init__(self):
        self.updated = []
        self.kinematic_info = SimpleNamespace(
            hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
            dof_limits_lower=torch.tensor([-1.0]),
            dof_limits_upper=torch.tensor([1.0]),
            dof_names=["hinge"],
            body_names=["pelvis", "left_hand", "right_hand"],
        )
        self.trackable_bodies_subset = ["pelvis", "left_hand"]
        self.control = SimpleNamespace(
            control_info={
                "hinge": SimpleNamespace(stiffness=10.0, damping=1.0),
            }
        )

    def update_fields(self, **kwargs):
        self.updated.append(kwargs)


def _args(**overrides):
    values = {
        "motion_file": "motions.pt",
        "scenes_file": "scenes.pt",
        "batch_size": 32,
        "training_max_steps": 1024,
        "tracker_checkpoint": "tracker.ckpt",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_mimic_fsq_basic_config_factories_and_inference_overrides():
    args = _args()
    robot_cfg = _RobotConfig()
    simulator_cfg = SimpleNamespace()

    terrain_cfg = mimic_fsq.terrain_config(args)
    scene_cfg = mimic_fsq.scene_lib_config(args)
    motion_cfg = mimic_fsq.motion_lib_config(args)
    env_cfg = mimic_fsq.env_config(robot_cfg, args)
    agent_cfg = mimic_fsq.agent_config(robot_cfg, env_cfg, args)

    assert terrain_cfg.__class__.__name__ == "TerrainConfig"
    assert scene_cfg.scene_file == "scenes.pt"
    assert motion_cfg.motion_file == "motions.pt"
    assert env_cfg.control_components["mimic"].future_steps == [1, 2, 5, 7, 12, 18, 25]
    assert "max_coords_obs" in env_cfg.observation_components
    assert "nearest_surface" not in env_cfg.observation_components
    assert "mimic_target_poses" in env_cfg.observation_components
    assert "tracking_error" in env_cfg.termination_components
    assert "pow_rew" in env_cfg.reward_components
    assert "contact_match_rew" in env_cfg.reward_components
    assert env_cfg.observation_components["max_coords_obs"].static_params[
        "root_height_obs"
    ] is True
    assert env_cfg.reward_components["contact_match_rew"].static_params == {
        "weight": -0.1,
        "zero_during_grace_period": True,
    }

    assert agent_cfg.batch_size == 32
    assert agent_cfg.training_max_steps == 1024
    assert agent_cfg.model.actor.mu_model.num_fsq_levels == 9
    assert agent_cfg.model.actor.mu_model.num_fsq_scalars == 40
    assert agent_cfg.model.actor.mu_model.encoder.in_keys == ["mimic_target_poses"]
    assert agent_cfg.model.actor.mu_model.decoder.in_keys == [
        "max_coords_obs",
        "latent",
    ]
    assert agent_cfg.model.actor.in_keys == [
        "max_coords_obs",
        "mimic_target_poses",
    ]
    assert agent_cfg.model.critic.in_keys == [
        "max_coords_obs",
        "mimic_target_poses",
    ]
    assert agent_cfg.model.in_keys == [
        "max_coords_obs",
        "mimic_target_poses",
    ]
    assert agent_cfg.model.actor.mu_model.decoder.num_out == robot_cfg.number_of_actions
    assert agent_cfg.model.actor.actor_logstd == pytest.approx(-2.9)

    mimic_fsq.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    assert robot_cfg.updated == [
        {"contact_bodies": ["all_left_foot_bodies", "all_right_foot_bodies"]}
    ]

    mimic_fsq.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_cfg,
        scene_cfg,
        args,
    )

    assert env_cfg.termination_components == {}
    assert env_cfg.max_episode_length == 1000000
    assert env_cfg.motion_manager.resample_on_reset is True
    assert env_cfg.motion_manager.init_start_prob == 1.0


def test_mimic_fsq_scene_lib_config_handles_missing_scene_arg():
    scene_cfg = mimic_fsq.scene_lib_config(argparse.Namespace(motion_file="motions.pt"))

    assert scene_cfg.scene_file is None


def test_gpc_prior_parser_and_checkpoint_dependent_config(monkeypatch):
    parser = argparse.ArgumentParser()
    gpc_prior.additional_experiment_arguments(parser)
    parsed = parser.parse_args(["--tracker-checkpoint", "custom.ckpt"])
    assert parsed.tracker_checkpoint == "custom.ckpt"

    monkeypatch.setattr(gpc_prior, "_tracker_future_steps", lambda args: [1, 4, 8])

    args = _args()
    robot_cfg = _RobotConfig()
    simulator_cfg = SimpleNamespace()
    terrain_cfg = gpc_prior.terrain_config(args)
    scene_cfg = gpc_prior.scene_lib_config(args)
    motion_cfg = gpc_prior.motion_lib_config(args)
    env_cfg = gpc_prior.env_config(robot_cfg, args)
    agent_cfg = gpc_prior.agent_config(robot_cfg, env_cfg, args)

    assert terrain_cfg.__class__.__name__ == "TerrainConfig"
    assert scene_cfg.scene_file == "scenes.pt"
    assert motion_cfg.motion_file == "motions.pt"
    assert env_cfg.control_components["mimic"].future_steps == [1, 4, 8]
    assert "nearest_surface" not in env_cfg.observation_components
    assert "previous_actions" not in env_cfg.observation_components
    assert env_cfg.observation_components["max_coords_obs"].static_params[
        "root_height_obs"
    ] is True
    assert env_cfg.motion_manager.init_start_prob == 0.2

    assert agent_cfg.rollout_actor is RolloutActor.EXPERT
    assert agent_cfg.batch_size == 32
    assert agent_cfg.training_max_steps == 1024
    assert agent_cfg.model.latent_decoder.checkpoint_path == "tracker.ckpt"
    assert agent_cfg.model.latent_decoder.module_path == "actor.mu"
    assert agent_cfg.model.prior.token_key == "prior_tokens"
    assert agent_cfg.model.prior.in_keys == [
        "max_coords_obs",
        "prior_tokens",
    ]
    assert agent_cfg.model.prior.context_encoder.in_keys == [
        "max_coords_obs",
    ]
    assert agent_cfg.model.prior.context_encoder.models[0].in_keys == [
        "max_coords_obs",
    ]
    assert agent_cfg.model.prior.logits_key == LATENT_LOGITS_KEY
    assert agent_cfg.model.prior.generated_tokens_key == LATENT_KEY
    assert agent_cfg.model.fsq_scalars_per_prior_token == 5
    assert agent_cfg.loss.label_smoothing == pytest.approx(0.01)
    assert robot_cfg.updated == []


def test_gpc_prior_tracker_helpers_load_future_steps_and_embed_module_config(
    monkeypatch,
):
    mimic = SimpleNamespace(future_steps=[2, 5, 9])
    resolved = {
        "env": SimpleNamespace(control_components={"mimic": mimic}),
        "agent": SimpleNamespace(model=SimpleNamespace(actor=SimpleNamespace(mu_model="fsq"))),
    }

    def fake_load(path):
        assert path == "tracker.ckpt"
        return resolved

    monkeypatch.setattr(
        "protomotions.utils.config_utils.load_resolved_configs_from_checkpoint",
        fake_load,
    )

    assert gpc_prior._tracker_future_steps(_args()) == [2, 5, 9]

    agent_cfg = gpc_prior.agent_config(_RobotConfig(), SimpleNamespace(), _args())
    decoder_cfg = agent_cfg.model.latent_decoder
    assert decoder_cfg.module_config_path == gpc_prior.TRACKER_MODULE_CONFIG_PATH
    assert decoder_cfg.module_config is None

    agent_cfg.prepare_inference_config_for_save()
    assert decoder_cfg.module_config == "fsq"


def test_gpc_prior_tracker_future_steps_requires_mimic_component(monkeypatch):
    monkeypatch.setattr(
        "protomotions.utils.config_utils.load_resolved_configs_from_checkpoint",
        lambda path: {"env": SimpleNamespace(control_components={})},
    )

    with pytest.raises(ValueError, match="does not define.*mimic"):
        gpc_prior._tracker_future_steps(_args())


def _assert_default_actor_peft_model(actor_cfg, expected_in_keys):
    actor_peft_model_cfg = actor_cfg.peft.model
    assert actor_cfg.in_keys == expected_in_keys
    assert not hasattr(actor_cfg, "model")
    assert actor_peft_model_cfg.in_keys == expected_in_keys
    assert actor_peft_model_cfg.out_keys == ["task_cond"]
    assert len(actor_peft_model_cfg.models) == 1

    task_feature_cfg = actor_peft_model_cfg.models[0]
    assert task_feature_cfg.__class__.__name__ == "ObsProcessorConfig"
    assert task_feature_cfg.in_keys == expected_in_keys
    assert task_feature_cfg.out_keys == ["task_cond"]
    assert task_feature_cfg.normalize_obs is True

    peft_cfg = actor_cfg.peft
    assert peft_cfg.condition_key == "task_cond"
    assert not hasattr(peft_cfg, "prior_context_key")
    assert not hasattr(peft_cfg, "terrain_key")
    return task_feature_cfg


def _assert_custom_actor_peft_model(actor_cfg, expected_in_keys):
    actor_peft_model_cfg = actor_cfg.peft.model
    assert actor_cfg.in_keys == expected_in_keys
    assert not hasattr(actor_cfg, "model")
    assert actor_peft_model_cfg.in_keys == expected_in_keys
    assert actor_peft_model_cfg.out_keys == ["task_cond"]
    assert len(actor_peft_model_cfg.models) == 1

    peft_model_cfg = actor_peft_model_cfg.models[0]
    assert peft_model_cfg.in_keys == expected_in_keys
    assert peft_model_cfg.out_keys == ["task_cond"]

    peft_cfg = actor_cfg.peft
    assert peft_cfg.condition_key == "task_cond"
    assert not hasattr(peft_cfg, "prior_context_key")
    assert not hasattr(peft_cfg, "terrain_key")
    return peft_model_cfg


@pytest.mark.parametrize(
    "module",
    [
        sft_target_prior_peft,
        task_steering_headvel_prior_peft,
        task_target_prior_peft,
        task_target_prior_peft_amp,
        task_steering_headvel_prior_peft_amp,
    ],
)
def test_gpc_peft_inference_overrides_preserve_prior_only_pretrained_module(module):
    agent_cfg = SimpleNamespace(
        pretrained_modules={
            "prior": SimpleNamespace(checkpoint_path="prior.ckpt"),
        },
        amp_parameters=SimpleNamespace(discriminator_reward_threshold=1.0),
    )
    env_cfg = SimpleNamespace(
        termination_components={"fall": object()},
        control_components={
            "path": SimpleNamespace(enable_path_termination=True),
        },
        motion_manager=SimpleNamespace(
            resample_on_reset=False,
            init_start_prob=0.0,
        ),
    )

    module.apply_inference_overrides(
        _RobotConfig(),
        SimpleNamespace(),
        env_cfg,
        agent_cfg,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        _args(),
    )

    assert "prior" in agent_cfg.pretrained_modules
    assert "tracker" not in agent_cfg.pretrained_modules


def test_gpc_target_peft_config_uses_explicit_pretrained_modules(monkeypatch):
    monkeypatch.setattr(
        "protomotions.utils.config_utils.load_resolved_configs_from_checkpoint",
        lambda checkpoint, prefer_inference=False: {
            "env": SimpleNamespace(observation_components={})
        },
    )

    args = _args(prior_checkpoint="prior.ckpt")
    env_cfg = task_target_prior_peft.env_config(_RobotConfig(), args)
    agent_cfg = task_target_prior_peft.agent_config(
        _RobotConfig(),
        env_cfg,
        args,
    )

    field_names = {field.name for field in fields(type(agent_cfg))}
    assert "tracker_checkpoint" not in field_names
    assert "prior_checkpoint" not in field_names
    assert not hasattr(agent_cfg, "training_mode")
    assert set(agent_cfg.pretrained_modules) == {"prior"}
    assert agent_cfg.pretrained_modules["prior"].checkpoint_path == "prior.ckpt"
    assert agent_cfg.pretrained_modules["prior"].module_path == ""
    assert "PEFT_ADAPTER_CONFIG" not in inspect.getsource(task_target_prior_peft)
    assert "PEFT_CONDITIONING_OBS_KEYS" not in inspect.getsource(task_target_prior_peft)
    assert "PEFT_PRIOR_CONTEXT_OBS_KEY" not in inspect.getsource(task_target_prior_peft)
    assert "PriorPEFTConditioningConfig" not in inspect.getsource(task_target_prior_peft)
    actor_cfg = agent_cfg.model.actor
    assert actor_cfg.in_keys == ["task_obs"]
    assert actor_cfg.out_keys == [
        "action",
        "mean_action",
        "neglogp",
        "prior_tokens",
    ]
    peft_cfg = actor_cfg.peft
    _assert_default_actor_peft_model(actor_cfg, ["task_obs"])
    assert peft_cfg.peft_type == "dora"
    assert peft_cfg.rank == 32
    assert peft_cfg.alpha == 64
    assert peft_cfg.sampling_mode == "prior_constraint"
    assert peft_cfg.top_p == pytest.approx(1.0)
    assert peft_cfg.prior_top_p == pytest.approx(0.9)
    assert peft_cfg.kl_coeff == pytest.approx(0.0)
    assert peft_cfg.m_clamp is None
    assert peft_cfg.film_input_norm is True
    assert agent_cfg.target_kl is None
    assert agent_cfg.gradient_clip_val == pytest.approx(0.5)
    assert agent_cfg.model.critic.norm_clamp_value == pytest.approx(10)
    assert agent_cfg.entropy_coef == 0.0
    target_cfg = env_cfg.control_components["target"]
    assert target_cfg.command_source.tar_change_time_min == 7.0
    assert target_cfg.tar_proximity_threshold == 0.3
    assert env_cfg.reward_components["target"].static_params["pos_err_scale"] == pytest.approx(0.42)


def test_gpc_target_peft_config_declares_prior_nearest_surface_context():
    env_cfg = task_target_prior_peft.env_config(
        _RobotConfig(),
        _args(prior_checkpoint="prior.ckpt"),
    )

    assert "nearest_surface" in env_cfg.observation_components
    assert env_cfg.observation_components["nearest_surface"].static_params == {
        "body_ids": [0, 1],
        "terrain_horizontal_scale": 0.1,
    }


@pytest.mark.parametrize(
    "module",
    [
        task_steering_headvel_prior_peft,
        task_target_prior_peft,
    ],
)
def test_gpc_task_peft_configs_switch_sampling_mode(module):
    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)

    default_args = _args(
        **vars(parser.parse_args(["--prior-checkpoint", "prior.ckpt"]))
    )
    default_agent = module.agent_config(_RobotConfig(), SimpleNamespace(), default_args)
    default_peft = default_agent.model.actor.peft
    assert default_peft.sampling_mode == "prior_constraint"
    assert default_peft.top_p == pytest.approx(1.0)
    assert default_peft.prior_top_p == pytest.approx(0.9)
    assert default_peft.kl_coeff == pytest.approx(0.0)
    assert default_peft.film_input_norm is True

    nucleus_args = _args(
        **vars(
            parser.parse_args(
                [
                    "--prior-checkpoint",
                    "prior.ckpt",
                    "--peft-sampling-mode",
                    "nucleus",
                ]
            )
        )
    )
    nucleus_agent = module.agent_config(_RobotConfig(), SimpleNamespace(), nucleus_args)
    nucleus_peft = nucleus_agent.model.actor.peft
    assert nucleus_peft.sampling_mode == "nucleus"
    assert nucleus_peft.top_p == pytest.approx(0.9)
    assert nucleus_peft.prior_top_p == pytest.approx(1.0)
    assert nucleus_peft.kl_coeff == pytest.approx(0.1)
    assert nucleus_peft.film_input_norm is True


def test_gpc_steering_peft_config_declares_prior_nearest_surface_context():
    env_cfg = task_steering_headvel_prior_peft.env_config(
        _RobotConfig(),
        _args(prior_checkpoint="prior.ckpt"),
    )

    assert set(env_cfg.observation_components) == {
        "max_coords_obs",
        "previous_actions",
        "task_obs",
        "nearest_surface",
    }
    assert env_cfg.observation_components["nearest_surface"].static_params == {
        "body_ids": [0, 1],
        "terrain_horizontal_scale": 0.1,
    }


def test_gpc_sft_target_peft_config_uses_tracker_rollout_env(monkeypatch):
    def _resolved_configs(checkpoint, prefer_inference=False):
        assert checkpoint != "prior.ckpt"
        return {
            "env": SimpleNamespace(
                control_components={
                    "mimic": SimpleNamespace(future_steps=[1, 2, 4])
                },
                observation_components={},
            )
        }

    monkeypatch.setattr(
        "protomotions.utils.config_utils.load_resolved_configs_from_checkpoint",
        _resolved_configs,
    )
    args = _args(prior_checkpoint="prior.ckpt")
    robot_cfg = _RobotConfig()
    env_cfg = sft_target_prior_peft.env_config(robot_cfg, args)
    agent_cfg = sft_target_prior_peft.agent_config(robot_cfg, env_cfg, args)

    assert set(env_cfg.control_components) == {"mimic", "target"}
    assert env_cfg.control_components["mimic"].future_steps == [1, 2, 4]
    assert env_cfg.control_components["target"].lookahead_seconds_min == 1.0
    assert env_cfg.control_components["target"].lookahead_seconds_max == 5.0
    assert env_cfg.control_components["target"].target_jitter_radius == pytest.approx(0.5)
    assert env_cfg.control_components["target"].random_target_fraction == pytest.approx(0.0)
    assert env_cfg.control_components["target"].random_target_xy_radius == pytest.approx(6.0)
    assert env_cfg.motion_manager.init_start_prob == pytest.approx(0.2)
    assert "task_obs" in env_cfg.observation_components
    assert "mimic_target_poses" in env_cfg.observation_components
    assert "nearest_surface" in env_cfg.observation_components
    assert env_cfg.observation_components["nearest_surface"].static_params == {
        "body_ids": [0, 1],
        "terrain_horizontal_scale": 0.1,
    }
    assert "tracking_error" in env_cfg.termination_components
    assert not hasattr(agent_cfg, "training_mode")
    assert agent_cfg._target_ == "protomotions.agents.peft.sft_agent.DiscretePriorPEFTSFTAgent"
    assert not hasattr(agent_cfg.model, "critic")
    assert set(agent_cfg.pretrained_modules) == {"prior"}
    assert agent_cfg.pretrained_modules["prior"].checkpoint_path == "prior.ckpt"
    assert agent_cfg.model.actor.in_keys == ["task_obs"]
    assert agent_cfg.model.actor.peft.sampling_mode == "prior_constraint"
    assert agent_cfg.model.actor.peft.top_p == pytest.approx(0.9)
    assert agent_cfg.model.actor.peft.prior_top_p == pytest.approx(0.99)
    assert agent_cfg.model.actor.peft.m_clamp == pytest.approx(1.6)
    assert agent_cfg.model.actor.peft.film_input_norm is True
    assert agent_cfg.model.actor_optimizer.lr == pytest.approx(1e-4)
    assert agent_cfg.num_mini_epochs == 1
    assert agent_cfg.gradient_clip_val == pytest.approx(50.0)
    assert agent_cfg.normalize_rewards is False


@pytest.mark.parametrize(
    ("amp_module", "base_module"),
    [
        (task_target_prior_peft_amp, task_target_prior_peft),
        (task_steering_headvel_prior_peft_amp, task_steering_headvel_prior_peft),
    ],
)
def test_gpc_peft_amp_configs_add_discriminator_components(amp_module, base_module):
    args = _args(prior_checkpoint="prior.ckpt")
    robot_cfg = _RobotConfig()
    env_cfg = amp_module.env_config(robot_cfg, args)
    agent_cfg = amp_module.agent_config(robot_cfg, env_cfg, args)

    assert amp_module.DISC_HISTORY_STEPS == [1, 2, 4, 8, 16]
    assert (
        amp_module.additional_experiment_arguments
        is not base_module.additional_experiment_arguments
    )
    assert (
        amp_module.configure_robot_and_simulator
        is not base_module.configure_robot_and_simulator
    )
    assert amp_module.env_config is not base_module.env_config
    assert (
        amp_module.agent_config is not base_module.agent_config
    )
    assert (
        amp_module.apply_inference_overrides
        is not base_module.apply_inference_overrides
    )
    assert env_cfg.num_state_history_steps == 16
    assert "historical_max_coords_obs" in env_cfg.observation_components
    assert "nearest_surface" in env_cfg.observation_components
    assert env_cfg.observation_components["nearest_surface"].static_params == {
        "body_ids": [0, 1],
        "terrain_horizontal_scale": 0.1,
    }
    assert agent_cfg._target_ == (
        "protomotions.agents.peft.prior_amp_agent."
        "DiscretePriorPEFTRLFTAMPAgent"
    )
    assert agent_cfg.model._target_ == (
        "protomotions.agents.peft.prior_amp_model."
        "DiscretePriorPEFTRLFTAMPModel"
    )
    assert not hasattr(agent_cfg, "training_mode")
    assert agent_cfg.model.actor.in_keys == ["task_obs"]
    actor_cfg = agent_cfg.model.actor
    peft_cfg = actor_cfg.peft
    _assert_default_actor_peft_model(actor_cfg, ["task_obs"])
    assert peft_cfg.peft_type == "dora"
    assert peft_cfg.rank == 32
    assert peft_cfg.alpha == 64
    assert peft_cfg.sampling_mode == "prior_constraint"
    assert agent_cfg.model.discriminator.in_keys == ["historical_max_coords_obs"]
    assert agent_cfg.model.disc_critic.in_keys == [
        "max_coords_obs",
        "historical_max_coords_obs",
    ]
    assert agent_cfg.amp_parameters.discriminator_batch_size == (
        agent_cfg.batch_size // agent_cfg.num_mini_epochs
    )
    assert set(agent_cfg.reference_obs_components) == {"historical_max_coords_obs"}
    reference_obs_params = agent_cfg.reference_obs_components[
        "historical_max_coords_obs"
    ].get_params()
    assert reference_obs_params["history_steps"] == [1, 2, 4, 8, 16]
    assert reference_obs_params["num_state_history_steps"] == 16

    amp_module.apply_inference_overrides(
        robot_cfg,
        SimpleNamespace(),
        env_cfg,
        agent_cfg,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        args,
    )

    assert agent_cfg.amp_parameters.discriminator_reward_threshold == 0.0


@pytest.mark.parametrize(
    "experiment_module",
    [
        task_steering_headvel_prior_peft,
        task_target_prior_peft,
        task_target_prior_peft_amp,
        task_steering_headvel_prior_peft_amp,
        sft_target_prior_peft,
    ],
)
def test_gpc_peft_configs_define_task_feature_model_not_legacy_routing_fields(
    experiment_module,
):
    agent_cfg = experiment_module.agent_config(
        _RobotConfig(),
        SimpleNamespace(),
        _args(prior_checkpoint="prior.ckpt"),
    )
    source = inspect.getsource(experiment_module)
    for stale_name in (
        "PriorPEFTConditioningConfig",
        "conditioning_model",
        "conditioning_obs_keys",
        "prior_context_key",
        "prior_context_obs_key",
        "peft_condition",
        "task_conditioning_keys",
        "terrain_obs_key",
        "PEFT_ADAPTER_CONFIG",
        "PEFT_ALPHA",
        "PEFT_RANK",
        "PEFT_TERRAIN_ENCODER",
        "PEFT_TERRAIN_OBS_KEY",
        "PEFT_TYPE",
    ):
        assert stale_name not in source

    actor_cfg = agent_cfg.model.actor
    peft_cfg = actor_cfg.peft
    actor_peft_model_cfg = peft_cfg.model
    assert not hasattr(actor_cfg, "model")
    assert not hasattr(peft_cfg, "prior_context_key")
    assert peft_cfg.condition_key == "task_cond"
    assert set(actor_peft_model_cfg.in_keys).issubset(actor_cfg.in_keys)
    assert "max_coords_obs" not in actor_cfg.in_keys
    assert "max_coords_obs" not in actor_peft_model_cfg.in_keys
    assert "task_cond" in actor_peft_model_cfg.out_keys
    assert any("task_cond" in model.out_keys for model in actor_peft_model_cfg.models)
