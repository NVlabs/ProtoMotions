# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metatron release integration; fetch with scripts/download_metatron.py first.

--configs-only enables the config/contract tests without the model payload.
Tests never perform network downloads implicitly.
"""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.common import ObsProcessor
from protomotions.agents.common.config import (
    ModuleOperationReshapeConfig,
    ObsProcessorConfig,
)
from protomotions.utils.config_builder import load_experiment_module
from examples.inference.metatron import apply_inference_overrides
from protomotions.agents.evaluators.config import MaskedMimicActionSource
from protomotions.agents.supervised.masked_mimic_config import VAENoiseType
from protomotions.envs.context_requirements import bound_components
from protomotions.envs.obs.state_history_buffer import history_depth
from protomotions.envs.obs.masked_mimic import compute_fk_target_poses_only
from protomotions.envs.obs.scene_boxes import (
    compute_fk_scene_box_corners,
    compute_scene_box_mask,
)
from protomotions.robot_configs.g1 import G1RobotConfig
from protomotions.utils.baked_fk import BakedAnchorFK
from protomotions.simulator.factory import simulator_config
from protomotions.utils.experiment_configs import (
    ExperimentConfigs,
    is_tracking_metadata_key,
    load_experiment_configs,
    save_experiment_configs,
)
from protomotions.utils.inference_utils import derive_inference_configs
from protomotions.tests.support.recipe_stages import has_payload

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "data/pretrained_models/metatron/G1"
)
CHECKPOINT = ARTIFACT_DIR / "inference_last.ckpt"
RECIPE_PATH = ARTIFACT_DIR / "experiment_config.py"
if os.environ.get("PROTOMOTIONS_REQUIRE_METATRON") == "1" and not RECIPE_PATH.is_file():
    raise RuntimeError(
        "PROTOMOTIONS_REQUIRE_METATRON=1 but the Metatron G1 config sidecars are "
        "absent; run `python scripts/download_metatron.py --configs-only` first."
    )
pytestmark = pytest.mark.skipif(
    not RECIPE_PATH.is_file(),
    reason="fetch pinned G1 sidecars: python scripts/download_metatron.py --configs-only",
)
recipe = load_experiment_module(str(RECIPE_PATH)) if RECIPE_PATH.is_file() else None


def _configs():
    args = SimpleNamespace(
        robot_name="g1",
        scenes_file=None,
        motion_file="motions.pt",
        batch_size=4,
        training_max_steps=64,
    )
    robot = G1RobotConfig()
    sim = simulator_config("mujoco", robot, True, 1, "test")
    recipe.configure_robot_and_simulator(robot, sim, args)
    env = recipe.env_config(robot, args)
    return ExperimentConfigs(
        robot,
        sim,
        recipe.terrain_config(args),
        recipe.scene_lib_config(args),
        recipe.motion_lib_config(args),
        env,
        recipe.agent_config(robot, env, args),
    )


@pytest.mark.parametrize("headless", [False, True])
def test_demo_json_roundtrip_and_native_modes(tmp_path, headless):
    original = _configs()
    saved = save_experiment_configs(original, tmp_path)
    rebuilt = load_experiment_configs(saved)
    args = SimpleNamespace(headless=headless, full_eval=False)
    view = derive_inference_configs(
        rebuilt,
        args,
        {"simulator.headless": headless},
        [recipe.apply_inference_overrides, apply_inference_overrides],
    )
    task = view.env.control_components["masked_mimic"]
    assert task.command_source == ("script" if headless else "keyboard")
    assert task.reach_body_name == "right_rubber_hand"
    assert task.canvas.has_velocity_command
    assert task.canvas.conditionable_bodies == tuple(recipe.BODY_NAMES)
    assert view.env.termination_components == {}
    assert view.agent.evaluator.action_source is MaskedMimicActionSource.PRIOR
    assert view.agent.model.vae.vae_noise_type is VAENoiseType.ZEROS
    assert history_depth(bound_components(view.env, view.robot)) == 8
    assert view.env.observation_components["previous_actions"].get_bindings_dict() == {
        "x": "current_action",
    }
    assert view.robot.body_indices(*recipe.BODY_NAMES) == (16, 13, 7, 24, 32)
    model = view.agent.model
    assert [
        len(config.models) for config in (model.encoder, model.prior, model.trunk)
    ] == [14, 27, 6]
    assert model.prior.out_keys == ["prior_mu", "prior_logvar", "decoder_condition"]
    for container in (model.encoder, model.prior, model.trunk):
        for module in container.models:
            if isinstance(module, ObsProcessorConfig):
                assert len(module.out_keys) == 1


@pytest.mark.parametrize("container_name,index", [("encoder", 4), ("prior", 8)])
def test_split_time_normalizers_preserve_frozen_shared_statistics(
    container_name, index
):
    """Splitting streams is exact for frozen inference, not joint training updates."""
    configs = getattr(_configs().agent.model, container_name).models[index : index + 2]
    streams = [torch.arange(15).reshape(3, 5).float(), torch.full((3, 5), -100.0)]
    processors = [ObsProcessor(config).eval() for config in configs]
    for processor, stream, config in zip(processors, streams, configs):
        processor(TensorDict({config.in_keys[0]: stream}, batch_size=[3]))
        norm = processor.norm.running_obs_norm
        norm.mean.fill_(0.75)
        norm.var.fill_(2.0)
        norm.count.fill_(123)

    # Reference behavior: reshape, concatenate along the batch dimension,
    # normalize once, then split back into the two original streams.
    shaped = [
        stream.reshape(3, 5, -1)
        if isinstance(config.module_operations[0], ModuleOperationReshapeConfig)
        else stream
        for stream, config in zip(streams, configs)
    ]
    expected = processors[0].norm(torch.cat(shaped, dim=0)).chunk(2, dim=0)
    for processor, stream, config, reference in zip(
        processors, streams, configs, expected
    ):
        original_stats = {
            key: value.clone() for key, value in processor.state_dict().items()
        }
        result = processor(TensorDict({config.in_keys[0]: stream}, batch_size=[3]))
        torch.testing.assert_close(
            result[config.out_keys[0]], reference, rtol=0, atol=0
        )
        for key, value in processor.state_dict().items():
            torch.testing.assert_close(value, original_stats[key], rtol=0, atol=0)
    left_stats = processors[0].norm.running_obs_norm
    right_stats = processors[1].norm.running_obs_norm
    assert left_stats.mean.data_ptr() != right_stats.mean.data_ptr()


def test_demo_recipe_refuses_to_hide_real_furniture():
    with pytest.raises(ValueError, match="empty scenes"):
        recipe.scene_lib_config(SimpleNamespace(scenes_file="furniture.pt"))


def test_packaged_config_reads_the_scene_from_the_host():
    """Guard the published bundle: the scene stream is the control's, the corners
    come from the goal's FK kernel on the raw readings, and no body pose is bound."""
    configs = load_experiment_configs(ARTIFACT_DIR / "resolved_configs.json")
    scene = configs.env.control_components["scene_boxes"]
    assert (scene.max_objects, scene.max_boxes_per_object) == (1, 2)
    components = configs.env.observation_components
    corners = components["scene_box_corners"]
    assert corners.compute_func is compute_fk_scene_box_corners
    bindings = corners.get_bindings_dict()
    assert bindings["dof_pos"] == "noisy.dof_pos"
    assert bindings["anchor_pos"] == "noisy.anchor_pos"
    assert bindings["anchor_rot"] == "noisy.anchor_rot"
    assert {bindings[k] for k in ("centers", "extents", "valid", "object_pos", "object_rot")} == {
        "scene_boxes.centers",
        "scene_boxes.extents",
        "scene_boxes.valid",
        "scene_boxes.object_pos",
        "scene_boxes.object_rot",
    }
    assert not {"current_state_body_pos", "current_state_body_rot"} & bindings.keys()
    goal = components["fk_masked_mimic_target_poses"]
    assert corners.static_params["fk"].export_init() == goal.static_params["fk"].export_init()
    mask = components["scene_box_mask"]
    assert mask.compute_func is compute_scene_box_mask
    assert mask.get_bindings_dict() == {
        "visible": "scene_boxes.visible",
        "valid": "scene_boxes.valid",
    }
    for name in ("scene_box_corners", "scene_box_mask"):
        assert name in configs.agent.model.prior.in_keys


def test_packaged_config_binds_fk_goal_to_raw_sensor_readings():
    """Guard the published bundle, not a separate staging copy of its config."""
    configs = load_experiment_configs(ARTIFACT_DIR / "resolved_configs.json")
    components = configs.env.observation_components
    goal = components["fk_masked_mimic_target_poses"]
    assert goal.compute_func is compute_fk_target_poses_only
    bindings = goal.get_bindings_dict()
    assert bindings["dof_pos"] == "noisy.dof_pos"
    assert bindings["anchor_pos"] == "noisy.anchor_pos"
    assert bindings["anchor_rot"] == "noisy.anchor_rot"
    assert not {"current_state_body_pos", "current_state_body_rot"} & bindings.keys()
    fk = goal.static_params["fk"]
    assert isinstance(fk, BakedAnchorFK)
    assert fk.anchor_idx == configs.robot.anchor_body_index == 16
    assert fk.export_init() == {
        "anchor_idx": 16,
        "trackable_body_indices": [16, 13, 7, 24, 32],
    }
    assert (
        components["historical_proprioception"].static_params["fk"].export_init()
        == fk.export_init()
    )


def _check_trained_scene_response(policy_action, obs_td):
    """Probe the weights, not a native scene provider or furniture interaction."""
    hidden = obs_td.clone()
    hidden["scene_box_mask"].fill_(False)
    baseline = policy_action(hidden.clone())
    signs = torch.tensor(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        device=baseline.device,
        dtype=baseline.dtype,
    )
    # Two boxes in root-heading coordinates, with dimensions in metres.
    centers = signs.new_tensor([[0.7, 0.0, -0.4], [0.9, 0.0, -0.1]])
    extents = signs.new_tensor([[0.5, 0.6, 0.4], [0.1, 0.6, 0.6]])
    corners = centers[:, None] + 0.5 * extents[:, None] * signs
    hidden["scene_box_corners"].copy_(corners.flatten(1).unsqueeze(0))
    torch.testing.assert_close(policy_action(hidden.clone()), baseline, atol=0, rtol=0)
    visible = hidden.clone()
    visible["scene_box_mask"].fill_(True)
    action = policy_action(visible)
    assert torch.isfinite(action).all()
    assert (action - baseline).abs().max() > 1e-4


def test_release_configs_are_portable_and_have_no_tracking_metadata():
    def check(value):
        if isinstance(value, dict):
            assert not any(is_tracking_metadata_key(key) for key in value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            assert not Path(value).is_absolute(), value
            assert not value.startswith("~"), value
            assert not re.match(r"^[A-Za-z]:[\\/]", value), value
            assert "://" not in value, value

    for filename in ("config.yaml", "resolved_configs.json", "provenance.json"):
        check(json.loads((ARTIFACT_DIR / filename).read_text()))
    configs = load_experiment_configs(ARTIFACT_DIR / "resolved_configs.json")
    assert configs.agent.expert_model_path is None
    assert configs.motion_lib.motion_file == "examples/data/g1_crouch_to_run.motion"


def test_portable_config_reproduces_the_metatron_reference_layout():
    """The portable config.yaml is the launcher's full argument dump minus the
    tracking and launcher-run keys; stripping those from the parser's full key order
    reproduces this shipped config.yaml's keys in order. (training_max_* is stripped
    here because in config.yaml it is a run argument, though it is legitimate inside
    resolved_configs.json's agent root.) Writer and this check share the predicate, so
    the portable form of every pretrained directory matches this reference by
    construction."""
    from protomotions.launch.training_cli import create_training_parser
    from protomotions.utils.experiment_configs import LAUNCHER_RUN_KEYS

    reference = list(json.loads((ARTIFACT_DIR / "config.yaml").read_text()).keys())
    # The launcher's full config.yaml key order: the parser's dests in declaration
    # order, then wandb_id, which save_configs appends.
    parser = create_training_parser()
    full_order = [
        action.dest for action in parser._actions if action.dest != "help"
    ] + ["wandb_id"]
    portable_order = [
        key
        for key in full_order
        if not (is_tracking_metadata_key(key) or key in LAUNCHER_RUN_KEYS)
    ]
    assert portable_order == reference


@pytest.mark.skipif(
    not has_payload(CHECKPOINT),
    reason="fetch weights: python scripts/download_metatron.py",
)
def test_native_playback_runs_the_packaged_checkpoint(monkeypatch):
    """Load the packaged weights and run the card's MuJoCo playback for 64 steps
    with a reset in the middle; probe the scene response at the model boundary
    separately from the playback, whose scene stream is the control's empty
    scene. Every action and observation stays finite."""
    from lightning.fabric import Fabric

    from protomotions.simulator.factory import update_simulator_config_for_test
    from protomotions.tests.support.recipe_stages import eager_mujoco_build
    from protomotions.utils.component_builder import build_env_from_configs
    from protomotions.utils.experiment_configs import (
        load_experiment_configs_from_checkpoint,
    )
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import recipe_hooks

    eager_mujoco_build(monkeypatch)

    # The card's native-playback command: MuJoCo, one env, the interactive-goals
    # inference config. The view derives from the one bundle beside the
    # checkpoint (the recipe hook, then the --inference-config hook).
    training = load_experiment_configs_from_checkpoint(CHECKPOINT)
    training.simulator = update_simulator_config_for_test(
        current_simulator_config=training.simulator,
        new_simulator="mujoco",
        robot_config=training.robot,
    )
    args = SimpleNamespace(full_eval=False, headless=True)
    view = derive_inference_configs(
        training,
        args,
        {"simulator.num_envs": 1, "simulator.headless": True},
        recipe_hooks(
            CHECKPOINT.parent,
            ARTIFACT_DIR.parents[3] / "examples/inference/metatron.py",
        ),
    )

    fabric = Fabric(
        **FabricConfig(
            accelerator="cpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()
    env = build_env_from_configs(view, fabric.device)
    try:
        agent = get_class(view.agent._target_)(
            config=view.agent, env=env, fabric=fabric, root_dir=CHECKPOINT.parent
        )
        agent.setup(training=False)
        agent.load(
            str(CHECKPOINT),
            load_env=False,
            load_training_state=False,
            for_inference=True,
        )

        agent.eval()
        done_indices = None
        with torch.no_grad():  # inference, as the native playback runs it
            for step in range(64):
                obs, _ = env.reset(done_indices)
                agent.pre_collect_step(step)
                obs = agent.add_agent_info_to_obs(obs)
                obs_td = agent.obs_dict_to_tensordict(obs)
                if step == 0:
                    _check_trained_scene_response(
                        agent.evaluator._policy_action, obs_td
                    )
                for key, value in obs_td.items(include_nested=True, leaves_only=True):
                    if torch.is_floating_point(value):
                        assert torch.isfinite(
                            value
                        ).all(), f"non-finite obs {key} at step {step}"
                action = agent.evaluator._policy_action(obs_td)
                assert torch.isfinite(action).all(), f"non-finite action at {step}"
                _, _, dones, _, _ = env.step(action)
                # One full reset at the midpoint, on top of the natural dones.
                done_indices = (
                    torch.arange(dones.shape[0], device=dones.device)
                    if step == 31
                    else dones.nonzero(as_tuple=False).squeeze(-1)
                )
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()
