# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer checks for example experiment config modules.

These keep FSQ/GPC example configs wired to the common model-building blocks
without starting training or loading real checkpoints.
"""

import argparse
import importlib
import inspect
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from protomotions.utils.experiment_configs import ExperimentConfigs
import torch
from protomotions.envs.body_selection import BodySelection
from protomotions.envs.context_requirements import bound_components
from protomotions.envs.obs.state_history_buffer import history_depth

from examples.experiments.gpc import prior as gpc_prior
from examples.experiments.gpc import steering_prior_peft
from examples.experiments.gpc import sft_target_prior_peft
from examples.experiments.gpc import target_prior_peft
from examples.experiments.gpc import task_target_prior_peft_amp
from examples.experiments.mimic import fsq as mimic_fsq
from protomotions.agents.common.config import validate_module_order
from protomotions.components.pose_lib import ControlInfo
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
            body_names=["pelvis", "left_hand", "right_hand", "left_foot", "right_foot"],
        )
        self.common_naming_to_robot_body_names = {
            "torso_body_name": ["pelvis"],
            "all_left_foot_bodies": ["left_foot"],
            "all_right_foot_bodies": ["right_foot"],
        }
        self.trackable_bodies_subset = ["pelvis", "left_hand"]
        self.default_dof_pos = torch.zeros(1)
        self.control = SimpleNamespace(
            control_info={
                "hinge": ControlInfo(stiffness=10.0, damping=1.0, effort_limit=20.0),
            }
        )

    def update_fields(self, **kwargs):
        self.updated.append(kwargs)

    def body_indices(self, *names):
        from protomotions.robot_configs.base import RobotConfig

        return RobotConfig.body_indices(self, *names)


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


@pytest.mark.parametrize(
    "module",
    [
        "gpc.sft_target_prior_peft",
        "gpc.steering_prior_peft",
        "gpc.target_prior_peft",
        "gpc.task_steering_headvel_prior_peft_amp",
        "gpc.task_target_prior_peft_amp",
    ],
)
def test_examples_without_contact_terms_disable_contact_reporting(module):
    recipe = importlib.import_module(f"examples.experiments.{module}")
    robot = _RobotConfig()
    recipe.configure_robot_and_simulator(robot, SimpleNamespace(), _args())
    assert robot.updated[-1]["contact_bodies"] is None


def test_bm_tracker_builds_with_beyondmimic_contact_selection():
    from examples.experiments.mimic import mlp_bm_l2c2 as recipe
    from protomotions.envs.component_manager import ComponentManager
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.base_simulator.contacts import ContactLayout

    robot = robot_config("g1")
    args = _args(scenes_file=None)
    simulator = SimpleNamespace()
    recipe.configure_robot_and_simulator(robot, simulator, args)
    env = recipe.env_config(robot, args)
    agent = recipe.agent_config(robot, env, args)
    assert agent.batch_size == args.batch_size
    assert agent.model.actor.std_mode.value == "global"
    assert robot.asset.self_collisions
    allowed_bodies = {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_rubber_hand",
        "right_rubber_hand",
    }
    expected_bodies = [
        name for name in robot.kinematic_info.body_names if name not in allowed_bodies
    ]
    assert robot.contact_bodies == expected_bodies
    assert "left_elbow_link" in expected_bodies
    assert "right_elbow_link" in expected_bodies
    expected_partners = [f"robot/{name}" for name in robot.kinematic_info.body_names]
    assert robot.contact_partners == expected_partners
    # G1 hand collision shapes belong to the wrist links, not rubber-hand markers.
    assert "robot/left_wrist_yaw_link" in expected_partners
    assert "robot/right_wrist_yaw_link" in expected_partners
    assert (
        env.observation_components["max_coords_obs"].static_params["observe_contacts"]
        is False
    )
    assert env.reward_components["action_rate"].meta.weight == -0.1
    assert set(env.termination_components) == {"fall"}

    layout = ContactLayout.build(
        robot.kinematic_info.body_names,
        100,
        2,
        contact_bodies=robot.contact_bodies,
        contact_partners=robot.contact_partners,
    )
    assert len(layout.body_index) == 27
    assert len(layout.partner_index) == len(robot.kinematic_info.body_names)
    reward = env.reward_components["self_contact"]
    assert reward.required_keys == {
        "contact_body_index": expected_bodies,
        "contact_partner_index": expected_partners,
    }
    assert reward.meta.weight == -0.1
    ctx = SimpleNamespace(
        contact_forces=torch.zeros(5, len(expected_bodies), len(expected_partners), 3),
        contact_body_index=layout.body_index,
        contact_partner_index=layout.partner_index,
    )
    left_hand = layout.partner_index["robot/left_wrist_yaw_link"]
    right_hand = layout.partner_index["robot/right_wrist_yaw_link"]
    torso = layout.body_index["torso_link"]
    ctx.contact_forces[1, torso, left_hand, 0] = 5
    ctx.contact_forces[2, torso, [left_hand, right_hand], 0] = 5
    # Knee-knee and elbow-torso collisions were missed by the torso-only example.
    for env_id, first, second in (
        (3, "left_knee_link", "right_knee_link"),
        (4, "left_elbow_link", "torso_link"),
    ):
        ctx.contact_forces[
            env_id, layout.body_index[first], layout.partner_index[f"robot/{second}"], 0
        ] = 5
        ctx.contact_forces[
            env_id, layout.body_index[second], layout.partner_index[f"robot/{first}"], 0
        ] = -5
    manager = ComponentManager(torch.device("cpu"))
    manager.validate_bindings({"self_contact": reward}, ctx, group="reward")
    torch.testing.assert_close(
        reward.compute(ctx), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0])
    )

    # Reporting overrides cannot silently change which bodies the reward needs.
    robot.update_fields(
        contact_bodies=["left_rubber_hand"], contact_partners=["terrain"]
    )
    changed_reward = recipe.env_config(robot, args).reward_components["self_contact"]
    assert changed_reward.required_keys == reward.required_keys
    ctx.contact_body_index = {"left_rubber_hand": 0}
    ctx.contact_partner_index = {"terrain": 0}
    with pytest.raises(ValueError, match="self_contact.*not registered"):
        manager.validate_bindings({"self_contact": changed_reward}, ctx, group="reward")


def test_bm_contact_selection_includes_physical_collision_links():
    from examples.experiments.mimic import mlp_bm_l2c2 as recipe
    from protomotions.assets import resolve_asset_root
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("g1")
    recipe.configure_robot_and_simulator(robot, SimpleNamespace(), _args())
    asset = ET.parse(
        Path(resolve_asset_root(robot.asset.asset_root)) / robot.asset.asset_file_name
    )
    colliding_bodies = {
        body.attrib["name"]
        for body in asset.findall(".//body")
        if any(int(geom.get("contype", "1")) for geom in body.findall("geom"))
    }
    # Marker bodies may be reported, but the real arm/leg collision links must
    # also be selected: a marker-only matrix makes the cost identically zero.
    for name in (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_elbow_link",
        "right_elbow_link",
        "torso_link",
    ):
        assert name in colliding_bodies
        assert f"robot/{name}" in robot.contact_partners
        if "wrist" not in name:
            assert name in robot.contact_bodies


def test_bm_contact_selection_uses_robot_foot_and_hand_groups():
    from examples.experiments.mimic import mlp_bm_l2c2 as recipe

    robot = SimpleNamespace(
        kinematic_info=SimpleNamespace(body_names=["chest", "elbow", "palm", "toe"]),
        common_naming_to_robot_body_names={
            "all_left_foot_bodies": ["toe"],
            "all_right_foot_bodies": [],
            "all_left_hand_bodies": ["palm"],
            "all_right_hand_bodies": [],
        },
    )
    assert recipe._self_contact_bodies(robot) == ["chest", "elbow"]


def test_bm_contact_cost_requires_physical_self_collisions():
    from examples.experiments.mimic import mlp_bm_l2c2 as recipe
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("g1")
    robot.asset.self_collisions = False
    with pytest.raises(ValueError, match="requires asset.self_collisions=True"):
        recipe.configure_robot_and_simulator(robot, SimpleNamespace(), _args())


def test_mimic_fsq_basic_config_factories_and_inference_overrides():
    args = _args()
    robot_cfg = _RobotConfig()
    simulator_cfg = SimpleNamespace()

    terrain_cfg = mimic_fsq.terrain_config(args)
    scene_cfg = mimic_fsq.scene_lib_config(args)
    motion_cfg = mimic_fsq.motion_lib_config(args)
    env_cfg = mimic_fsq.env_config(robot_cfg, args)
    agent_cfg = mimic_fsq.agent_config(robot_cfg, env_cfg, args)
    validate_module_order(agent_cfg)

    assert terrain_cfg.__class__.__name__ == "TerrainConfig"
    assert scene_cfg.scene_file == "scenes.pt"
    assert motion_cfg.motion_file == "motions.pt"
    assert env_cfg.control_components["mimic"].future_steps == (1, 2, 5, 7, 12, 18, 25)
    assert "max_coords_obs" in env_cfg.observation_components
    assert "nearest_surface" not in env_cfg.observation_components
    assert "mimic_target_poses" in env_cfg.observation_components
    assert "tracking_error" in env_cfg.termination_components
    assert "pow_rew" in env_cfg.reward_components
    assert "contact_match_rew" in env_cfg.reward_components
    assert (
        env_cfg.observation_components["max_coords_obs"].static_params[
            "root_height_obs"
        ]
        is True
    )
    foot_reward = env_cfg.reward_components["contact_match_rew"]
    assert foot_reward.meta.weight == -0.1
    assert foot_reward.meta.zero_during_grace_period is True
    assert foot_reward.static_params["body_names"] == ["left_foot", "right_foot"]
    # The saved configuration keeps the names; the build resolves them.
    assert foot_reward.static_params["contact_body_ids"] == BodySelection.named(
        "left_foot", "right_foot"
    )
    assert foot_reward.required_keys["contact_partner_index"] == ["terrain"]

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
    assert simulator_cfg.default_robot_friction == 0.5
    assert robot_cfg.updated == [
        {
            "contact_bodies": ["all_left_foot_bodies", "all_right_foot_bodies"],
            "contact_partners": ["terrain"],
        }
    ]

    env_cfg.apply_inference_defaults()  # the framework default for the episode length
    mimic_fsq.apply_inference_overrides(
        ExperimentConfigs(
            robot=robot_cfg,
            simulator=simulator_cfg,
            terrain=terrain_cfg,
            scene_lib=scene_cfg,
            motion_lib=motion_cfg,
            env=env_cfg,
            agent=agent_cfg,
        ),
        args,
    )

    assert env_cfg.termination_components == {}
    assert env_cfg.max_episode_length == 1000000
    assert env_cfg.motion_manager.resample_on_reset is True
    assert env_cfg.motion_manager.init_start_prob == 1.0


def test_mimic_fsq_scene_lib_config_defaults_scene_file_to_none():
    # --scenes-file is always declared (default None), so the recipe reads it
    # directly: no scene file means scene_file is None.
    scene_cfg = mimic_fsq.scene_lib_config(
        argparse.Namespace(motion_file="motions.pt", scenes_file=None)
    )

    assert scene_cfg.scene_file is None


@pytest.mark.parametrize("module", ["mlp", "fsq", "mlp_complex_terrain"])
def test_tracker_examples_only_measure_feet_against_terrain(module):
    from protomotions.envs.body_selection import resolve_body_selections
    from protomotions.envs.component_manager import ComponentManager
    from protomotions.envs.rewards.regularization import (
        compute_terrain_foot_contact_match_rew,
    )
    from protomotions.simulator.base_simulator.contacts import ContactLayout

    experiment = importlib.import_module(f"examples.experiments.mimic.{module}")
    robot = _RobotConfig()
    experiment.configure_robot_and_simulator(robot, SimpleNamespace(), _args())
    selection = robot.updated[-1]
    assert selection["contact_partners"] == ["terrain"]
    feet = [
        name
        for semantic in selection["contact_bodies"]
        for name in robot.common_naming_to_robot_body_names[semantic]
    ]
    layout = ContactLayout.build(
        robot.kinematic_info.body_names,
        100,
        4,
        contact_bodies=feet,
        contact_partners=selection["contact_partners"],
    )
    assert layout.body_index == {"left_foot": 0, "right_foot": 1}
    assert layout.partner_index == {"terrain": 0}
    env = experiment.env_config(robot, _args())
    # The recipe names the feet (a BodySelection); the build resolves them.
    reward = resolve_body_selections(env.reward_components, robot)["contact_match_rew"]
    assert reward.compute_func is compute_terrain_foot_contact_match_rew
    assert set(env.termination_components) == {"tracking_error"}
    forces = torch.zeros(2, 2, 1, 3)
    forces[0, 0, 0, 2] = 40
    labels = torch.zeros(2, 5)
    labels[0, 3] = 1  # full-body reference left foot, not compact row 0
    labels[1, 4] = 1  # right foot is missing in simulation
    ctx = SimpleNamespace(
        contact_forces=forces,
        contact_body_index=layout.body_index,
        contact_partner_index=layout.partner_index,
        mimic=SimpleNamespace(ref_state=SimpleNamespace(rigid_body_contacts=labels)),
    )
    ComponentManager(torch.device("cpu")).validate_bindings(
        {"contact_match": reward}, ctx, group="reward"
    )
    torch.testing.assert_close(reward.compute(ctx), torch.tensor([0.0, 1.0]))


def test_gpc_prior_parser_and_checkpoint_dependent_config(monkeypatch):
    parser = argparse.ArgumentParser()
    gpc_prior.additional_experiment_arguments(parser)
    parsed = parser.parse_args(["--tracker-checkpoint", "custom.ckpt"])
    assert parsed.tracker_checkpoint == "custom.ckpt"

    # Fake the one cached tracker read (bypasses its lru_cache): env_config reads
    # future_steps from it and agent_config embeds its mu_model config.
    monkeypatch.setattr(
        gpc_prior,
        "_tracker",
        lambda ckpt: SimpleNamespace(
            env=SimpleNamespace(
                control_components={"mimic": SimpleNamespace(future_steps=[1, 4, 8])}
            ),
            agent=SimpleNamespace(
                model=SimpleNamespace(
                    actor=SimpleNamespace(mu_model=SimpleNamespace(_target_="fsq"))
                )
            ),
        ),
    )

    args = _args()
    robot_cfg = _RobotConfig()
    simulator_cfg = SimpleNamespace()
    terrain_cfg = gpc_prior.terrain_config(args)
    scene_cfg = gpc_prior.scene_lib_config(args)
    motion_cfg = gpc_prior.motion_lib_config(args)
    env_cfg = gpc_prior.env_config(robot_cfg, args)
    agent_cfg = gpc_prior.agent_config(robot_cfg, env_cfg, args)
    validate_module_order(agent_cfg)

    assert terrain_cfg.__class__.__name__ == "TerrainConfig"
    assert scene_cfg.scene_file == "scenes.pt"
    assert motion_cfg.motion_file == "motions.pt"
    assert env_cfg.control_components["mimic"].future_steps == (1, 4, 8)
    assert "nearest_surface" not in env_cfg.observation_components
    assert "previous_actions" not in env_cfg.observation_components
    assert (
        env_cfg.observation_components["max_coords_obs"].static_params[
            "root_height_obs"
        ]
        is True
    )
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
    gpc_prior.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    assert simulator_cfg.default_robot_friction == 0.5
    assert robot_cfg.updated == []


def test_gpc_prior_tracker_helpers_load_future_steps_and_embed_module_config(
    monkeypatch,
):
    mimic = SimpleNamespace(future_steps=[2, 5, 9])
    resolved = SimpleNamespace(
        env=SimpleNamespace(control_components={"mimic": mimic}),
        agent=SimpleNamespace(
            model=SimpleNamespace(actor=SimpleNamespace(mu_model="fsq"))
        ),
    )
    # Fake the cached tracker read directly (bypasses its lru_cache).
    monkeypatch.setattr(gpc_prior, "_tracker", lambda ckpt: resolved)

    # The saved value is handed back as the normalised step tuple, whatever
    # form the bundle holds (a list here; an int after a CLI override).
    assert gpc_prior._tracker_future_steps("tracker.ckpt") == (2, 5, 9)

    # agent_config embeds the decoder's module config from the tracker bundle at
    # build, so the student bundle is self-contained.
    agent_cfg = gpc_prior.agent_config(_RobotConfig(), SimpleNamespace(), _args())
    validate_module_order(agent_cfg)
    decoder_cfg = agent_cfg.model.latent_decoder
    assert decoder_cfg.module_config == "fsq"
    assert decoder_cfg.checkpoint_path == "tracker.ckpt"


def test_gpc_prior_tracker_future_steps_requires_mimic_component(monkeypatch):
    monkeypatch.setattr(
        gpc_prior,
        "_tracker",
        lambda ckpt: SimpleNamespace(env=SimpleNamespace(control_components={})),
    )

    with pytest.raises(ValueError, match="does not define.*mimic"):
        gpc_prior._tracker_future_steps("tracker.ckpt")


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
        steering_prior_peft,
        target_prior_peft,
        task_target_prior_peft_amp,
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
        ExperimentConfigs(
            robot=_RobotConfig(),
            simulator=SimpleNamespace(),
            terrain=SimpleNamespace(),
            scene_lib=SimpleNamespace(),
            motion_lib=SimpleNamespace(),
            env=env_cfg,
            agent=agent_cfg,
        ),
        _args(),
    )

    assert "prior" in agent_cfg.pretrained_modules
    assert "tracker" not in agent_cfg.pretrained_modules


def test_gpc_target_peft_config_uses_explicit_pretrained_modules(monkeypatch):
    monkeypatch.setattr(
        "protomotions.utils.experiment_configs.load_experiment_configs_from_checkpoint",
        lambda checkpoint, prefer_inference=False: SimpleNamespace(
            env=SimpleNamespace(observation_components={})
        ),
    )

    args = _args(prior_checkpoint="prior.ckpt")
    env_cfg = target_prior_peft.env_config(_RobotConfig(), args)
    agent_cfg = target_prior_peft.agent_config(
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
    assert "PEFT_ADAPTER_CONFIG" not in inspect.getsource(target_prior_peft)
    assert "PEFT_CONDITIONING_OBS_KEYS" not in inspect.getsource(target_prior_peft)
    assert "PEFT_PRIOR_CONTEXT_OBS_KEY" not in inspect.getsource(target_prior_peft)
    assert "PriorPEFTConditioningConfig" not in inspect.getsource(target_prior_peft)
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
    assert env_cfg.reward_components["target"].static_params[
        "pos_err_scale"
    ] == pytest.approx(0.42)


def test_gpc_target_peft_config_declares_prior_nearest_surface_context():
    env_cfg = target_prior_peft.env_config(
        _RobotConfig(),
        _args(prior_checkpoint="prior.ckpt"),
    )

    assert "nearest_surface" in env_cfg.observation_components
    assert env_cfg.observation_components["nearest_surface"].static_params == {
        "body_ids": [0, 1],
        "terrain_horizontal_scale": 0.1,
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "examples.experiments.gpc.steering_prior_peft",
        "examples.experiments.gpc.target_prior_peft",
    ],
)
def test_gpc_task_peft_configs_switch_sampling_mode(module_name):
    module = importlib.import_module(module_name)
    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)

    default_args = _args(**vars(parser.parse_args([])))
    default_agent = module.agent_config(_RobotConfig(), SimpleNamespace(), default_args)
    default_peft = default_agent.model.actor.peft
    assert default_peft.sampling_mode == "prior_constraint"
    assert default_peft.top_p == pytest.approx(1.0)
    assert default_peft.prior_top_p == pytest.approx(0.9)
    assert default_peft.kl_coeff == pytest.approx(0.0)
    assert default_peft.film_input_norm is True

    nucleus_args = _args(**vars(parser.parse_args(["--peft-sampling-mode", "nucleus"])))
    nucleus_agent = module.agent_config(_RobotConfig(), SimpleNamespace(), nucleus_args)
    nucleus_peft = nucleus_agent.model.actor.peft
    assert nucleus_peft.sampling_mode == "nucleus"
    assert nucleus_peft.top_p == pytest.approx(0.9)
    assert nucleus_peft.prior_top_p == pytest.approx(1.0)
    assert nucleus_peft.kl_coeff == pytest.approx(0.1)
    assert nucleus_peft.film_input_norm is True


@pytest.mark.parametrize(
    ("alias_name", "clean_name"),
    [
        (
            "examples.experiments.gpc.task_steering_headvel_prior_peft",
            "examples.experiments.gpc.steering_prior_peft",
        ),
        (
            "examples.experiments.gpc.task_target_prior_peft",
            "examples.experiments.gpc.target_prior_peft",
        ),
    ],
)
def test_gpc_task_peft_compatibility_paths_alias_clean_configs(
    alias_name,
    clean_name,
):
    alias = importlib.import_module(alias_name)
    clean = importlib.import_module(clean_name)

    for name in (
        "PRIOR_CHECKPOINT",
        "additional_experiment_arguments",
        "configure_robot_and_simulator",
        "terrain_config",
        "scene_lib_config",
        "motion_lib_config",
        "env_config",
        "agent_config",
        "apply_inference_overrides",
    ):
        assert getattr(alias, name) is getattr(clean, name)


def test_gpc_steering_peft_config_declares_prior_nearest_surface_context():
    env_cfg = steering_prior_peft.env_config(
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
        return SimpleNamespace(
            env=SimpleNamespace(
                control_components={"mimic": SimpleNamespace(future_steps=[1, 2, 4])},
                observation_components={},
            )
        )

    monkeypatch.setattr(
        "protomotions.utils.experiment_configs.load_experiment_configs_from_checkpoint",
        _resolved_configs,
    )
    args = _args(prior_checkpoint="prior.ckpt")
    robot_cfg = _RobotConfig()
    env_cfg = sft_target_prior_peft.env_config(robot_cfg, args)
    agent_cfg = sft_target_prior_peft.agent_config(robot_cfg, env_cfg, args)

    assert set(env_cfg.control_components) == {"mimic", "target"}
    assert env_cfg.control_components["mimic"].future_steps == (1, 2, 4)
    assert env_cfg.control_components["target"].lookahead_seconds_min == 1.0
    assert env_cfg.control_components["target"].lookahead_seconds_max == 5.0
    assert env_cfg.control_components["target"].target_jitter_radius == pytest.approx(
        0.5
    )
    assert env_cfg.control_components["target"].random_target_fraction == pytest.approx(
        0.0
    )
    assert env_cfg.control_components[
        "target"
    ].random_target_xy_radius == pytest.approx(6.0)
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
    assert (
        agent_cfg._target_
        == "protomotions.agents.peft.sft_agent.DiscretePriorPEFTSFTAgent"
    )
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


def test_gpc_target_peft_amp_config_adds_discriminator_components():
    args = _args(prior_checkpoint="prior.ckpt")
    robot_cfg = _RobotConfig()
    env_cfg = task_target_prior_peft_amp.env_config(robot_cfg, args)
    agent_cfg = task_target_prior_peft_amp.agent_config(robot_cfg, env_cfg, args)

    assert task_target_prior_peft_amp.DISC_HISTORY_STEPS == [1, 2, 4, 8, 16]
    assert (
        task_target_prior_peft_amp.additional_experiment_arguments
        is not target_prior_peft.additional_experiment_arguments
    )
    assert (
        task_target_prior_peft_amp.configure_robot_and_simulator
        is not target_prior_peft.configure_robot_and_simulator
    )
    assert task_target_prior_peft_amp.env_config is not target_prior_peft.env_config
    assert task_target_prior_peft_amp.agent_config is not target_prior_peft.agent_config
    assert (
        task_target_prior_peft_amp.apply_inference_overrides
        is not target_prior_peft.apply_inference_overrides
    )
    # The discriminator's history observation reads 16 past slots; the ring
    # sizes itself from it.
    assert history_depth(bound_components(env_cfg, robot_cfg)) == 16
    assert "historical_max_coords_obs" in env_cfg.observation_components
    assert "nearest_surface" in env_cfg.observation_components
    assert env_cfg.observation_components["nearest_surface"].static_params == {
        "body_ids": [0, 1],
        "terrain_horizontal_scale": 0.1,
    }
    assert agent_cfg._target_ == (
        "protomotions.agents.peft.prior_amp_agent." "DiscretePriorPEFTRLFTAMPAgent"
    )
    assert agent_cfg.model._target_ == (
        "protomotions.agents.peft.prior_amp_model." "DiscretePriorPEFTRLFTAMPModel"
    )
    assert not hasattr(agent_cfg, "training_mode")
    assert agent_cfg.model.actor.in_keys == ["task_obs"]
    actor_cfg = agent_cfg.model.actor
    peft_cfg = actor_cfg.peft
    _assert_default_actor_peft_model(actor_cfg, ["task_obs"])
    assert peft_cfg.peft_type == "dora"
    assert peft_cfg.rank == 32
    assert peft_cfg.alpha == 64
    assert peft_cfg.sampling_mode == "nucleus"
    assert peft_cfg.top_p == pytest.approx(0.9)
    assert peft_cfg.prior_top_p == pytest.approx(1.0)
    assert peft_cfg.kl_coeff == pytest.approx(0.0)
    assert agent_cfg.model.discriminator.in_keys == ["historical_max_coords_obs"]
    assert agent_cfg.model.disc_critic.in_keys == [
        "max_coords_obs",
        "historical_max_coords_obs",
    ]
    assert agent_cfg.amp_parameters.discriminator_batch_size == (
        agent_cfg.batch_size // agent_cfg.num_mini_epochs
    )
    assert agent_cfg.amp_parameters.discriminator_reward_w == pytest.approx(0.05)
    assert agent_cfg.amp_parameters.discriminator_reward_threshold == pytest.approx(
        0.01
    )
    assert set(agent_cfg.reference_obs_components) == {"historical_max_coords_obs"}
    reference_obs_params = agent_cfg.reference_obs_components[
        "historical_max_coords_obs"
    ].get_params()
    assert reference_obs_params["history_steps"] == (1, 2, 4, 8, 16)

    task_target_prior_peft_amp.apply_inference_overrides(
        ExperimentConfigs(
            robot=robot_cfg,
            simulator=SimpleNamespace(),
            terrain=SimpleNamespace(),
            scene_lib=SimpleNamespace(),
            motion_lib=SimpleNamespace(),
            env=env_cfg,
            agent=agent_cfg,
        ),
        args,
    )

    assert agent_cfg.amp_parameters.discriminator_reward_threshold == 0.0


@pytest.mark.parametrize(
    "experiment_module",
    [
        steering_prior_peft,
        target_prior_peft,
        task_target_prior_peft_amp,
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
