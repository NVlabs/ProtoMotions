# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metatron export uses the real publishers, prior and incoming-action timeline."""

import argparse
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from deployment.contract import DeploymentContract, InputSpec
from deployment.export_fk_to_onnx import FkSidecar, export_fk, fk_binding
from deployment.export_to_onnx import (
    _build_deployable_policy,
    _build_samples,
    export_tracker,
)
from deployment.input_history import InputHistory, history_specs
from deployment.sample_context import build_sample_context
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel
from protomotions.utils.export_utils import ObservationExportModule


@pytest.fixture
def recipe_configs(monkeypatch):
    from examples.experiments.metatron import g1_student as recipe
    from examples.experiments.metatron import g1_teacher as teacher_recipe
    from protomotions.robot_configs.factory import robot_config

    args = argparse.Namespace(
        motion_file="motion.pt",
        scenes_file=None,
        batch_size=4,
        training_max_steps=32,
        expert_model_path="teacher.ckpt",
    )
    robot = robot_config("g1")
    teacher_recipe.configure_robot_and_simulator(robot, SimpleNamespace(), args)
    teacher_env = teacher_recipe.env_config(robot, args)
    teacher_agent = teacher_recipe.agent_config(robot, teacher_env, args)
    monkeypatch.setattr(
        recipe,
        "_teacher",
        lambda args: SimpleNamespace(env=teacher_env, agent=teacher_agent),
    )
    recipe.configure_robot(robot)
    env = recipe.env_config(robot, args)

    # Same public architecture and observation wiring, smaller widths for CI.
    monkeypatch.setattr(recipe, "TOKEN_DIM", 16)
    monkeypatch.setattr(recipe, "LATENT_DIM", 4)
    original_mlp = recipe._mlp
    monkeypatch.setattr(
        recipe,
        "_mlp",
        lambda keys, output, size, widths=(), normalize=False: original_mlp(
            keys, output, 16 if size == 1024 else size, (16,), normalize
        ),
    )
    agent = recipe.agent_config(robot, env, args)
    from protomotions.agents.common.config import TransformerConfig

    for config in agent.model.prior.models:
        if isinstance(config, TransformerConfig):
            config.num_layers = 1
            config.ff_size = 32
    return robot, env, agent


def _observations(robot, env, model):
    components = {
        key: env.observation_components[key] for key in model.deployable_in_keys
    }
    context, future, history = build_sample_context(robot, env, components)
    samples = _build_samples(components, context, history)
    obs = ObservationExportModule(components, samples, "cpu")
    return context, future, history, obs


def test_export_build_preserves_training_noise_config(recipe_configs):
    from protomotions.agents.supervised.masked_mimic_config import VAENoiseType

    _, _, agent = recipe_configs
    config = replace(
        agent.model, vae=replace(agent.model.vae, vae_noise_type=VAENoiseType.NORMAL)
    )
    model = _build_deployable_policy(None, config)
    assert config.vae.vae_noise_type == VAENoiseType.NORMAL
    assert model.config.vae.vae_noise_type == VAENoiseType.NORMAL


def test_full_goal_context_and_disabled_modality(recipe_configs):
    robot, env, agent = recipe_configs
    model = MaskedMimicModel(agent.model)
    context, future, history, obs = _observations(robot, env, model)
    assert context["masked_mimic.ref_pos"].shape == (1, 5, 33, 3)
    assert context["mimic.future_pos"].shape[1] == 7
    assert context["masked_mimic.style_pose_samples"].shape == (1, 225)
    assert context["masked_mimic.style_velocity_samples"].shape == (1, 150)
    assert context["masked_mimic.target_bodies_masks"].dtype == torch.bool
    assert "masked_mimic.target_bodies_masks" in future
    assert "noisy_historical.raw_actions" in history
    assert "current_action" in obs.get_input_keys()
    assert not any(key.startswith("current.") for key in obs.get_input_keys())
    env.control_components["masked_mimic"] = replace(
        env.control_components["masked_mimic"], style=None
    )
    with pytest.raises(ValueError, match="no style stream"):
        _observations(robot, env, model)
    components = {
        key: value
        for key, value in env.observation_components.items()
        if "style" not in key
    }
    context, _, _ = build_sample_context(robot, env, components)
    assert not any(key.startswith("masked_mimic.style_") for key in context)


def _spec(key, shape):
    history, measurements = history_specs(key, shape)
    return InputSpec(
        key.replace(".", "_"),
        key,
        key.split(".")[0],
        "joint_pos",
        "joint_encoders",
        shape,
        "float32",
        (),
        None,
        history,
        observation_history=measurements,
    )


def test_host_history_matches_state_publisher_and_applied_action():
    from protomotions.envs.base_env.publishers import StatePublisher
    from protomotions.envs.obs.state_history_buffer import (
        HistorySample,
        StateHistoryBuffer,
    )
    from protomotions.envs.state import BodyState

    def sample(step, applied):
        pose = torch.full((1, 2), float(step))
        quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
        state = BodyState(
            dof_pos=pose,
            dof_vel=pose,
            rigid_body_rot=quat,
            rigid_body_ang_vel=torch.zeros(1, 1, 3),
        )
        action = torch.full((1, 2), float(applied))
        return HistorySample({"noisy": state}, action, action * 2)

    ring = StateHistoryBuffer(3, sample(0, 0))
    publisher = StatePublisher(ring, 0)
    keys = (
        "current_action",
        "raw_action_history",
        "noisy_historical.raw_actions",
        "noisy_historical.dof_pos",
    )
    context = publisher.publish(None)
    host = InputHistory([_spec(key, tuple(context[key].shape)) for key in keys])
    host.reset({"noisy.dof_pos": context["noisy.dof_pos"].numpy()})
    for step in range(10):
        for key, value in host.inputs().items():
            np.testing.assert_array_equal(value, context[key].numpy(), err_msg=key)
        # Deliberately different from the hypothetical network prediction.
        applied = np.full((1, 2), 100 + step, dtype=np.float32)
        host.commit(
            {"noisy.dof_pos": context["noisy.dof_pos"].numpy()}, {"actions": applied}
        )
        ring.push(sample(step + 1, 100 + step))
        context = publisher.publish(None)
    initial = sample(50, 0)
    ring.reset(torch.tensor([0]), initial)
    context = publisher.publish(None)
    host.reset({"noisy.dof_pos": context["noisy.dof_pos"].numpy()})
    for key, value in host.inputs().items():
        np.testing.assert_array_equal(value, context[key].numpy(), err_msg=key)


def test_host_history_honors_declared_command_reset_fill():
    spec = _spec("processed_action_history", (2, 3, 2))
    spec = replace(spec, history=replace(spec.history, reset_fill=[0.25, -0.5]))
    host = InputHistory([spec])
    host.reset({})
    expected = np.broadcast_to(np.array([0.25, -0.5]), (2, 3, 2))
    np.testing.assert_array_equal(host.inputs()[spec.key], expected)
    host.commit({}, {"joint_pos_targets": np.ones((2, 2))})
    np.testing.assert_array_equal(host.inputs()[spec.key][:, 0], np.ones((2, 2)))
    np.testing.assert_array_equal(host.inputs()[spec.key][:, 1:], expected[:, 1:])


def _save_recipe_checkpoint(recipe_configs, directory):
    """Save a full training checkpoint and its config bundle as last.ckpt."""
    from protomotions.utils.experiment_configs import (
        ExperimentConfigs,
        save_experiment_configs,
    )

    robot, env, agent = recipe_configs
    model = MaskedMimicModel(agent.model).eval()
    context, _, _, obs = _observations(robot, env, model)
    fk, fk_paths = fk_binding(
        {key: env.observation_components[key] for key in model.deployable_in_keys}
    )
    assert fk_paths == ("noisy.dof_pos", "noisy.anchor_pos", "noisy.anchor_rot")
    with torch.no_grad():
        values = obs(*(context[key] for key in obs.get_input_keys()))
        td = TensorDict(dict(zip(obs.get_output_keys(), values)), batch_size=[1])
        # Materialize the encoder too, to save a genuine full training checkpoint.
        td["max_coords_obs"] = torch.zeros(1, 12)
        td["mimic_target_poses"] = torch.zeros(1, 12)
        td["vae_noise"] = torch.zeros(1, agent.model.vae.vae_latent_dim)
        model(td)
    configs = ExperimentConfigs(
        robot=robot,
        simulator=None,
        terrain=None,
        scene_lib=None,
        motion_lib=None,
        env=env,
        agent=agent,
    )
    save_experiment_configs(configs, directory)
    torch.save({"model": model.state_dict()}, directory / "last.ckpt")
    return model, context, obs, fk, fk_paths


@pytest.mark.parametrize("leapp", [False, True])
def test_export_metatron_prior_onnx_roundtrip(
    recipe_configs, tmp_path, monkeypatch, leapp
):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if leapp:
        pytest.importorskip("leapp")
    robot, env, agent = recipe_configs
    model, context, obs, fk, fk_paths = _save_recipe_checkpoint(
        recipe_configs, tmp_path
    )

    def forbid_training_forward(*args, **kwargs):
        raise AssertionError("Export must use the prior, not the privileged encoder")

    monkeypatch.setattr(MaskedMimicModel, "forward", forbid_training_forward)
    path = export_tracker(
        str(tmp_path / "last.ckpt"), str(tmp_path / "out"), leapp=leapp
    )
    contract = DeploymentContract.from_yaml(path.with_suffix(".yaml"))
    assert contract.robot.conditionable_body_names == tuple(
        robot.trackable_bodies_subset
    )
    assert (
        contract.robot.conditionable_body_names
        == contract.masked_mimic.conditionable_bodies
    )
    # The policy graph keeps its four control outputs: the body poses it
    # computes stay inside it, and a host that needs them runs fk.onnx (exported
    # below from the same kernel), whose inputs are three of the policy's.
    assert [spec.onnx_name for spec in contract.outputs] == [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]
    fk_inputs = ("noisy_dof_pos", "noisy_anchor_pos", "noisy_anchor_rot")
    assert set(fk_inputs) <= {spec.onnx_name for spec in contract.inputs}
    if leapp:
        import yaml
        from leapp import InferenceManager
        from deployment.native.tools.leapp_export import load_pipeline_contract

        bundle = path.parent / "leapp" / "policy.yaml"
        assert load_pipeline_contract(bundle).to_dict() == contract.to_dict()
        assert (bundle.parent / "policy.onnx").read_bytes() == path.read_bytes()
        document = yaml.safe_load(bundle.read_text())
        assert not document["pipeline"]["feedback_flow"]
        described = document["models"]["policy"]["inputs"]
        assert [entry["protomotions"] for entry in described] == contract.to_dict()[
            "policy_inputs"
        ]
        outputs = {
            entry["name"]: entry for entry in document["models"]["policy"]["outputs"]
        }
        assert outputs["joint_pos_targets"]["kind"] == "target/joint/position"
        assert outputs["stiffness_targets"]["kind"] == "kp"
        assert "kind" not in outputs["actions"]
        assert {"body_pos", "body_rot"}.isdisjoint(outputs)
        manager = InferenceManager(str(bundle))
    assert contract.masked_mimic.num_masked_future_steps == 5
    assert contract.masked_mimic.style_frames == 5
    assert contract.motion.future_step_indices != tuple(range(5))
    inputs = {spec.key: spec for spec in contract.inputs}
    assert inputs["current_action"].history.steps_back == (1,)
    assert inputs["noisy_historical.raw_actions"].history.steps_back == tuple(
        range(2, 10)
    )
    assert inputs["noisy_historical.dof_pos"].observation_history.steps_back == tuple(
        range(1, 9)
    )
    assert inputs["masked_mimic.ref_pos"].future_steps == 5
    assert inputs["masked_mimic.local_ref_pos"].future_steps == 5
    assert inputs["masked_mimic.target_bodies_masks"].future_steps == 5
    assert inputs["masked_mimic.local_target_bodies_masks"].future_steps == 5
    assert all(spec.onnx_name == spec.key.replace(".", "_") for spec in contract.inputs)
    from deployment.native.examples.metatron_v1_contract import (
        BOOL_INPUTS,
        INPUT_SHAPES,
    )

    # Match the existing goal/history ABI where shared; explicitly name changes
    # instead of disguising the new recipe as a v1 scene bundle. No body pose is
    # an input: the goal computes them from noisy.dof_pos and the anchor pose.
    assert set(inputs) - INPUT_SHAPES.keys() == {"current_action"}
    assert INPUT_SHAPES.keys() - set(inputs) == {
        "current.root_pos",
        "current.root_rot",
        "raw_action_history",
        *(key for key in INPUT_SHAPES if key.startswith("scene_boxes.")),
    }
    assert not any("rigid_body" in key for key in inputs)
    for key in inputs.keys() & INPUT_SHAPES.keys():
        assert inputs[key].shape == (1, *INPUT_SHAPES[key])
        assert inputs[key].dtype == ("bool" if key in BOOL_INPUTS else "float32")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    from deployment.metatron_policy import MetatronPriorExport
    from protomotions.utils.export_utils import (
        ActionExportModule,
        UnifiedPipelineModule,
    )

    pipeline = UnifiedPipelineModule(
        obs,
        MetatronPriorExport(model),
        ActionExportModule(env.action_config, "cpu"),
        model.deployable_in_keys,
    ).eval()
    generator = torch.Generator().manual_seed(42)
    for batch_size in (1, 3):
        for visibility in ("none", "all", "mixed"):
            feed = {
                key: value.repeat(batch_size, *([1] * (value.ndim - 1)))
                for key, value in context.items()
                if torch.is_tensor(value)
            }
            for key, value in feed.items():
                if key.startswith("masked_mimic.") and value.dtype == torch.bool:
                    if visibility == "mixed":
                        value.copy_(torch.rand(value.shape, generator=generator) > 0.5)
                    else:
                        value.fill_(visibility == "all")
                elif value.is_floating_point() and "rot" not in key:
                    value.add_(torch.randn(value.shape, generator=generator) * 0.03)
            with torch.no_grad():
                expected = pipeline(*(feed[key] for key in obs.get_input_keys()))
            actual = session.run(
                None,
                {spec.onnx_name: feed[spec.key].numpy() for spec in contract.inputs},
            )
            assert len(expected) == len(actual) == len(contract.outputs)
            for truth, exported in zip(expected, actual):
                np.testing.assert_allclose(
                    exported, truth.numpy(), atol=1e-4, rtol=1e-4
                )
            if leapp:
                replay = manager.run_policy(
                    {
                        f"policy/{spec.onnx_name}": feed[spec.key]
                        for spec in contract.inputs
                    }
                )
                for spec, truth in zip(contract.outputs, expected):
                    np.testing.assert_allclose(
                        replay[f"policy/{spec.onnx_name}"],
                        truth.numpy(),
                        atol=1e-4,
                        rtol=1e-4,
                    )
    # fk.onnx is the same kernel traced on its own from the same bundle: it
    # equals the kernel to float rounding on the sample context, names its
    # inputs as the policy names the same paths, holds no data-dependent op (a
    # static TensorRT plan cannot), and its sidecar refuses a policy manifest
    # whose body order differs.
    import json
    import onnx

    fk_onnx = export_fk(str(tmp_path / "last.ckpt"), str(tmp_path / "fk"))
    sidecar = FkSidecar.from_yaml(fk_onnx.with_suffix(".yaml"))
    assert sidecar.input_names == fk_inputs
    assert tuple(path for _, path, _ in sidecar.inputs) == fk_paths
    assert sidecar.output_shapes == {"body_pos": (1, 33, 3), "body_rot": (1, 33, 4)}
    assert sidecar.body_names == tuple(contract.robot.body_names)
    assert sidecar.joint_names == tuple(contract.robot.joint_names)
    assert sidecar.anchor_body_index == contract.robot.anchor_body_index
    assert sidecar.checkpoint == "last.ckpt"
    fk_session = ort.InferenceSession(str(fk_onnx), providers=["CPUExecutionProvider"])
    feed = {spec.onnx_name: context[spec.key].numpy() for spec in contract.inputs}
    from_graph = fk_session.run(None, {name: feed[name] for name in fk_inputs})
    with torch.no_grad():
        own_pos, own_rot = fk.world_body_poses(*(context[key] for key in fk_paths))
    np.testing.assert_allclose(from_graph[0], own_pos.numpy(), atol=1e-5)
    np.testing.assert_allclose(from_graph[1], own_rot.numpy(), atol=1e-5)
    ops = {node.op_type for node in onnx.load(str(fk_onnx)).graph.node}
    assert not ops & {"NonZero", "If", "Loop", "Scan"}
    robot_contract = {
        "body_names": list(contract.robot.body_names),
        "joint_names": list(contract.robot.joint_names),
        "anchor_body_index": contract.robot.anchor_body_index,
    }
    manifest = tmp_path / "manifest.json"
    # The native manifest also lists the policy graph's inputs; the FK inputs
    # must be among them, since the host feeds one tensor under one name to
    # both graphs.
    input_names = [spec.onnx_name for spec in contract.inputs]
    manifest.write_text(
        json.dumps({"robot_contract": robot_contract, "input_names": input_names})
    )
    sidecar.check_manifest(manifest)
    renamed = [name for name in input_names if name != "noisy_anchor_pos"]
    manifest.write_text(
        json.dumps({"robot_contract": robot_contract, "input_names": renamed})
    )
    with pytest.raises(ValueError, match="not inputs of the policy manifest"):
        sidecar.check_manifest(manifest)
    swapped = dict(robot_contract, body_names=robot_contract["body_names"][::-1])
    manifest.write_text(
        json.dumps({"robot_contract": swapped, "input_names": input_names})
    )
    with pytest.raises(ValueError, match="body order differs"):
        sidecar.check_manifest(manifest)
    if leapp:
        import copy

        for change in (
            lambda d: d["models"]["policy"]["inputs"][0].update(
                protomotions_key="wrong.context"
            ),
            lambda d: d["models"]["policy"]["inputs"][0]["protomotions"].update(
                provider="wrong_provider"
            ),
            lambda d: d["models"]["policy"]["outputs"][0].update(
                kind="target/joint/position"
            ),
            lambda d: d["pipeline"].update(
                feedback_flow={"policy/actions": "policy/current_action"}
            ),
        ):
            altered = copy.deepcopy(document)
            change(altered)
            bundle.write_text(yaml.safe_dump(altered))
            with pytest.raises(ValueError, match="differs from deployment contract"):
                load_pipeline_contract(bundle)
    incomplete = model.state_dict()
    incomplete.pop(next(key for key in incomplete if key.endswith("weight")))
    torch.save({"model": incomplete}, tmp_path / "broken.ckpt")
    with pytest.raises(RuntimeError, match="Missing key"):
        export_tracker(str(tmp_path / "broken.ckpt"), str(tmp_path / "broken-export"))


def test_export_graph_history_bundle(recipe_configs, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("leapp")
    import os
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    _save_recipe_checkpoint(recipe_configs, tmp_path)
    repo = Path(__file__).resolve().parents[2]

    def export(jit):
        env = dict(os.environ, PYTORCH_JIT=jit, PYTHONPATH=str(repo))
        command = [
            sys.executable,
            str(repo / "deployment" / "export_to_onnx.py"),
            "--checkpoint",
            str(tmp_path / "last.ckpt"),
            "--output",
            str(tmp_path / f"out-jit{jit}"),
            "--leapp-graph-history",
        ]
        return subprocess.run(command, env=env, capture_output=True, text=True)

    scripted = export("1")
    assert scripted.returncode != 0
    assert "PYTORCH_JIT=0" in scripted.stderr

    result = export("0")
    assert result.returncode == 0, result.stderr
    bundle = tmp_path / "out-jit0" / "leapp_graph_history" / "policy"
    document = yaml.safe_load((bundle / "policy.yaml").read_text())
    inputs = {entry["name"] for entry in document["models"]["policy"]["inputs"]}
    assert {"applied_actions", "command_accepted"} <= inputs
    assert not inputs & {"current_action", "noisy_historical_raw_actions"}
    assert document["pipeline"]["feedback_flow"]
    assert not (bundle / "deployment_contract.yaml").exists()
