# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ONNX export utility plumbing."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

import protomotions.utils.export_utils as export_utils
from protomotions.envs.context_paths import FieldPath, NestedField
from protomotions.envs.mdp_component import MdpComponent


class _TinyTensorDictModule(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ["obs", "task"]
        self.out_keys = ["mu", "value"]

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict["mu"] = tensordict["obs"] + tensordict["task"]
        tensordict["value"] = tensordict["obs"].sum(dim=-1, keepdim=True)
        return tensordict


class _ObsView:
    obs: torch.Tensor = FieldPath()
    scale: float = FieldPath()


class _CurrentStateView:
    dof_pos: torch.Tensor = FieldPath()


class _ExportContext:
    current: _ObsView = NestedField(_ObsView)
    current_state: _CurrentStateView = NestedField(_CurrentStateView)


def _make_context() -> _ExportContext:
    current = _ObsView()
    current.obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    current.scale = 2.5

    current_state = _CurrentStateView()
    current_state.dof_pos = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    ctx = _ExportContext()
    ctx.current = current
    ctx.current_state = current_state
    return ctx


def _scaled_obs(
    obs: torch.Tensor,
    scale: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    return obs * scale + bias


def _state_obs(dof_pos: torch.Tensor) -> torch.Tensor:
    return dof_pos


def _process_action(
    action: torch.Tensor,
    offset: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    gain: float,
) -> dict[str, torch.Tensor]:
    return {
        "processed_action": action + offset * gain,
        "stiffness_targets": stiffness.expand_as(action),
        "damping_targets": damping.expand_as(action),
    }


class _PolicyModule(torch.nn.Module):
    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict["mean_action"] = (
            tensordict["current_state_dof_pos"] + tensordict["previous_actions"]
        )
        return tensordict


class _SingleOutputObsModule(torch.nn.Module):
    def __init__(self, input_keys: list[str], output_key: str | None):
        super().__init__()
        self._input_keys = input_keys
        self._output_key = output_key

    def get_input_keys(self) -> list[str]:
        return self._input_keys

    def get_output_keys(self) -> list[str]:
        return [] if self._output_key is None else [self._output_key]

    def forward(self, *args) -> tuple[torch.Tensor, ...]:
        if self._output_key is None:
            return ()
        return (args[0],)


class _RecordBatchPolicy(torch.nn.Module):
    def __init__(self, input_key: str | None):
        super().__init__()
        self.input_key = input_key
        self.batch_size = None

    def forward(self, tensordict: TensorDict) -> TensorDict:
        self.batch_size = tensordict.batch_size
        if self.input_key is None:
            tensordict["mean_action"] = torch.zeros(tensordict.batch_size[0], 2)
        else:
            tensordict["mean_action"] = tensordict[self.input_key]
        return tensordict


class _IdentityActionModule(torch.nn.Module):
    def forward(self, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return action, action, action


class _FakeOnnxValue:
    def __init__(self, name: str):
        self.name = name


def test_resolve_context_path_and_wrapper_forward_preserve_tensordict_order():
    ctx = SimpleNamespace(
        current=SimpleNamespace(obs=torch.tensor([[1.0, 2.0]])),
    )
    assert torch.equal(
        export_utils._resolve_context_path("current.obs", ctx),
        torch.tensor([[1.0, 2.0]]),
    )

    module = _TinyTensorDictModule()
    wrapper = export_utils.ONNXExportWrapper(module, ["obs", "task"], batch_size=2)
    obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    task = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    mu, value = wrapper(obs, task)

    assert torch.equal(mu, obs + task)
    assert torch.equal(value, torch.tensor([[3.0], [7.0]]))


def test_export_onnx_writes_metadata_and_invokes_torch_export(monkeypatch, tmp_path):
    module = _TinyTensorDictModule()
    td = TensorDict(
        {
            "obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "task": torch.ones(2, 2),
            "unused": torch.zeros(2, 1),
        },
        batch_size=[2],
    )
    path = tmp_path / "policy.onnx"
    captured = {}

    def fake_export(wrapper, input_tensors, export_path, **kwargs):
        captured["outputs"] = wrapper(*input_tensors)
        captured["path"] = export_path
        captured["input_shapes"] = [tuple(t.shape) for t in input_tensors]
        captured.update(kwargs)

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)

    meta = {"model_type": "tiny"}
    export_utils.export_onnx(
        module,
        td,
        str(path),
        meta=meta,
        validate=False,
        opset_version=18,
    )

    assert captured["path"] == str(path)
    assert captured["input_names"] == ["input_0", "input_1"]
    assert captured["output_names"] == ["output_0", "output_1"]
    assert captured["input_shapes"] == [(2, 2), (2, 2)]
    assert captured["opset_version"] == 18
    assert torch.equal(captured["outputs"][0], td["obs"] + td["task"])

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["model_type"] == "tiny"
    assert metadata["in_keys"] == ["obs", "task"]
    assert metadata["out_keys"] == ["mu", "value"]
    assert metadata["input_mapping"] == {"input_0": "obs", "input_1": "task"}
    assert metadata["output_mapping"] == {
        "output_0": "mu",
        "output_1": "value",
    }

    with pytest.raises(ValueError, match="must end with .onnx"):
        export_utils.export_onnx(module, td, str(tmp_path / "policy.pt"))


def test_export_onnx_validation_uses_onnxruntime(monkeypatch, tmp_path):
    module = _TinyTensorDictModule()
    td = TensorDict(
        {
            "obs": torch.tensor([[1.0, 2.0]], requires_grad=True),
            "task": torch.ones(1, 2),
        },
        batch_size=[1],
    )
    captured = {}

    def fake_export(wrapper, input_tensors, export_path, **kwargs):
        captured["outputs"] = wrapper(*input_tensors)

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def run(self, output_names, inputs):
            assert output_names is None
            assert set(inputs) == {"input_0", "input_1"}
            return [object(), object()]

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    export_utils.export_onnx(module, td, str(tmp_path / "policy.onnx"))

    assert torch.equal(captured["outputs"][0], td["obs"] + td["task"])


def test_export_onnx_validation_import_and_runtime_failures_warn(
    monkeypatch,
    tmp_path,
    capsys,
):
    module = _TinyTensorDictModule()
    td = TensorDict(
        {
            "obs": torch.tensor([[1.0, 2.0]]),
            "task": torch.ones(1, 2),
        },
        batch_size=[1],
    )

    def fake_export(wrapper, input_tensors, export_path, **kwargs):
        wrapper(*input_tensors)

    class FailingSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def run(self, output_names, inputs):
            raise RuntimeError("runtime validation exploded")

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    export_utils.export_onnx(module, td, str(tmp_path / "missing_ort.onnx"))
    assert "onnxruntime not installed" in capsys.readouterr().out

    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FailingSession),
    )
    export_utils.export_onnx(module, td, str(tmp_path / "failing_ort.onnx"))
    assert "ONNX validation failed: runtime validation exploded" in capsys.readouterr().out


def test_export_ppo_helpers_build_tensordicts_and_paths(monkeypatch, tmp_path):
    calls = []

    def fake_export_onnx(module, td, path, meta=None, validate=True):
        calls.append((module, td.clone(), path, meta, validate))

    monkeypatch.setattr(export_utils, "export_onnx", fake_export_onnx)

    actor = SimpleNamespace(name="actor")
    critic = SimpleNamespace(
        name="critic",
        config=SimpleNamespace(num_out=3),
    )
    model = SimpleNamespace(_actor=actor, _critic=critic)
    sample_obs = {
        "obs": torch.ones(4, 2),
        "task": torch.zeros(4, 1),
    }

    actor_path = tmp_path / "actor.onnx"
    critic_path = tmp_path / "critic.onnx"
    export_utils.export_ppo_actor(actor, sample_obs, str(actor_path), validate=False)
    export_utils.export_ppo_critic(critic, sample_obs, str(critic_path))
    result = export_utils.export_ppo_model(model, sample_obs, str(tmp_path), False)

    assert calls[0][0] is actor
    assert calls[0][1].batch_size == torch.Size([4])
    assert calls[0][2] == str(actor_path)
    assert calls[0][3]["model_type"] == "PPOActor"
    assert calls[0][4] is False

    assert calls[1][0] is critic
    assert calls[1][2] == str(critic_path)
    assert calls[1][3]["num_out"] == 3
    assert calls[1][4] is True

    assert result == {
        "actor": str(tmp_path / "actor.onnx"),
        "critic": str(tmp_path / "critic.onnx"),
        "metadata": {
            "actor_meta": str(tmp_path / "actor.json"),
            "critic_meta": str(tmp_path / "critic.json"),
        },
    }
    assert calls[2][0] is actor
    assert calls[3][0] is critic
    assert calls[2][4] is False
    assert calls[3][4] is False


def test_action_export_module_registers_tensor_constants_and_outputs_targets():
    action_module = export_utils.ActionExportModule(
        {
            "fn": _process_action,
            "offset": torch.tensor([1.0, -1.0]),
            "stiffness": torch.tensor([10.0, 20.0]),
            "damping": torch.tensor([1.0, 2.0]),
            "gain": 0.5,
        },
        torch.device("cpu"),
    )
    actions = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    processed, stiffness, damping = action_module(actions)

    assert action_module.get_output_keys() == [
        "processed_action",
        "stiffness_targets",
        "damping_targets",
    ]
    assert torch.equal(processed, torch.tensor([[0.5, 0.5], [2.5, 2.5]]))
    assert torch.equal(stiffness, torch.tensor([[10.0, 20.0], [10.0, 20.0]]))
    assert torch.equal(damping, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))

    assert export_utils.ActionExportModule(None, torch.device("cpu"))._action_function is None
    assert (
        export_utils.ActionExportModule({}, torch.device("cpu"))._action_function is None
    )


def test_unified_pipeline_uses_obs_passthrough_policy_and_action_modules():
    obs_module = export_utils.ObservationExportModule(
        {
            "current_state_dof_pos": MdpComponent(
                compute_func=_state_obs,
                dynamic_vars={"dof_pos": _ExportContext.current_state.dof_pos},
            ),
        },
        _make_context(),
        torch.device("cpu"),
    )
    action_module = export_utils.ActionExportModule(
        {
            "fn": _process_action,
            "offset": torch.tensor([0.25, 0.5]),
            "stiffness": torch.tensor([4.0, 5.0]),
            "damping": torch.tensor([0.4, 0.5]),
            "gain": 2.0,
        },
        torch.device("cpu"),
    )
    pipeline = export_utils.UnifiedPipelineModule(
        obs_module,
        _PolicyModule(),
        action_module,
        policy_in_keys=["current_state_dof_pos", "previous_actions"],
        policy_action_key="mean_action",
        passthrough_keys=["previous_actions"],
    )

    dof_pos = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    previous_actions = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    actions, processed, stiffness, damping = pipeline(dof_pos, previous_actions)

    assert pipeline.get_all_input_keys() == [
        "current_state.dof_pos",
        "previous_actions",
    ]
    assert torch.equal(actions, dof_pos + previous_actions)
    assert torch.equal(processed, actions + torch.tensor([0.5, 1.0]))
    assert torch.equal(stiffness, torch.tensor([[4.0, 5.0], [4.0, 5.0]]))
    assert torch.equal(damping, torch.tensor([[0.4, 0.5], [0.4, 0.5]]))


def test_unified_pipeline_infers_batch_size_without_passthrough_inputs():
    prefix_policy = _RecordBatchPolicy("current_state_features")
    prefix_pipeline = export_utils.UnifiedPipelineModule(
        _SingleOutputObsModule(["state"], "current_state_features"),
        prefix_policy,
        _IdentityActionModule(),
        policy_in_keys=["current_state_features"],
        policy_action_key="mean_action",
    )

    prefix_pipeline(torch.ones(3, 2))
    assert prefix_policy.batch_size == torch.Size([3])

    fallback_policy = _RecordBatchPolicy("features")
    fallback_pipeline = export_utils.UnifiedPipelineModule(
        _SingleOutputObsModule(["state"], "features"),
        fallback_policy,
        _IdentityActionModule(),
        policy_in_keys=["features"],
        policy_action_key="mean_action",
    )

    fallback_pipeline(torch.ones(4, 2))
    assert fallback_policy.batch_size == torch.Size([4])

    default_policy = _RecordBatchPolicy(None)
    default_pipeline = export_utils.UnifiedPipelineModule(
        _SingleOutputObsModule([], None),
        default_policy,
        _IdentityActionModule(),
        policy_in_keys=[],
        policy_action_key="mean_action",
    )

    default_pipeline()
    assert default_policy.batch_size == torch.Size([1])


def test_policy_yaml_builders_cover_history_names_and_constant_gain_outputs():
    joint_names = ["hip", "knee"]
    body_names = ["pelvis", "torso"]

    assert export_utils._build_policy_input(
        "unknown", {}, joint_names, body_names
    ) is None
    current = export_utils._build_policy_input(
        "current_dof_pos",
        {"current_dof_pos": [8, 2]},
        joint_names,
        body_names,
    )
    history = export_utils._build_policy_input(
        "historical_dof_vel",
        {"historical_dof_vel": [8, 4, 2]},
        joint_names,
        body_names,
    )
    root_ang_vel = export_utils._build_policy_input(
        "current_root_ang_vel",
        {"current_root_ang_vel": [8, 3]},
        joint_names,
        body_names,
    )
    processed_actions = export_utils._build_policy_input(
        "historical_processed_actions",
        {"historical_processed_actions": [8, 3, 2]},
        joint_names,
        body_names,
    )
    ref_joint_pos = export_utils._build_policy_input(
        "mimic_ref_dof_pos",
        {"mimic_ref_dof_pos": [8, 2]},
        joint_names,
        body_names,
    )
    future_rot = export_utils._build_policy_input(
        "mimic_future_rot",
        {"mimic_future_rot": [8, 5, 2, 4]},
        joint_names,
        body_names,
    )
    anchor_rot = export_utils._build_policy_input(
        "mimic_ref_anchor_rot",
        {"mimic_ref_anchor_rot": [8, 1, 4]},
        joint_names,
        body_names,
        anchor_body="torso",
    )

    assert current["shape"] == [1, 2]
    assert current["history"] == 0
    assert current["include_current_value_in_history"] is True
    assert current["element_names"] == [joint_names]
    assert history["history"] == 4
    assert history["include_current_value_in_history"] is False
    assert root_ang_vel["element_names"] == [["x", "y", "z"]]
    assert root_ang_vel["include_current_value_in_history"] is True
    assert processed_actions["output_key"] == "robot_action"
    assert ref_joint_pos["name"] == "reference_motion_joint_pos"
    assert ref_joint_pos["element_names"] == [joint_names]
    assert "history" not in ref_joint_pos
    assert future_rot["future_steps"] == 5
    assert future_rot["element_names"] == [body_names, list("xyzw")]
    assert "history" not in anchor_rot
    assert anchor_rot["element_names"] == [["torso"], list("xyzw")]

    assert export_utils._build_policy_output(
        "unknown", {}, joint_names, [1.0, 2.0], [0.1, 0.2]
    ) is None
    actions = export_utils._build_policy_output(
        "actions",
        {"actions": [8, 2]},
        joint_names,
        [1.0, 2.0],
        [0.1, 0.2],
    )
    stiffness = export_utils._build_policy_output(
        "stiffness_targets",
        {"stiffness_targets": [8, 2]},
        joint_names,
        [1.0, 2.0],
        [0.1, 0.2],
        use_onnx_for_gains=False,
    )
    damping = export_utils._build_policy_output(
        "damping_targets",
        {"damping_targets": [8, 2]},
        joint_names,
        [1.0, 2.0],
        [0.1, 0.2],
        use_onnx_for_gains=False,
    )

    assert actions["shape"] == [1, 2]
    assert "joint_names" not in actions
    assert stiffness["key"] is None
    assert stiffness["stiffness"] == [1.0, 2.0]
    assert damping["key"] is None
    assert damping["damping"] == [0.1, 0.2]

    content = export_utils._generate_yaml_content(
        input_shapes={
            "current_dof_pos": [8, 2],
            "ignored": [8, 7],
        },
        output_shapes={
            "actions": [8, 2],
            "joint_pos_targets": [8, 2],
        },
        onnx_in_names=["current_dof_pos", "ignored"],
        onnx_out_names=["actions", "joint_pos_targets", "ignored"],
        joint_names=joint_names,
        body_names=body_names,
        stiffness=[1.0, 2.0],
        damping=[0.1, 0.2],
        dt=1.0 / 30.0,
    )

    assert list(content.keys())[:2] == ["type", "dt"]
    assert [entry["key"] for entry in content["policy_inputs"]] == [
        "current_dof_pos"
    ]
    assert [entry["key"] for entry in content["policy_outputs"]] == [
        "actions",
        "joint_pos_targets",
    ]


def test_policy_yaml_generation_recovers_suffix_renamed_onnx_io_names():
    content = export_utils._generate_yaml_content(
        input_shapes={
            "current_state_dof_pos": [8, 2],
            "previous_actions": [8, 2],
        },
        output_shapes={
            "actions": [8, 2],
            "joint_pos_targets": [8, 2],
            "stiffness_targets": [8, 2],
            "damping_targets": [8, 2],
        },
        onnx_in_names=["current_state_dof_pos.1", "previous_actions_2"],
        onnx_out_names=[
            "actions.1",
            "joint_pos_targets_3",
            "stiffness_targets.2",
            "damping_targets_4",
        ],
        joint_names=["hip", "knee"],
        body_names=["pelvis"],
        stiffness=[1.0, 2.0],
        damping=[0.1, 0.2],
    )

    assert [entry["key"] for entry in content["policy_inputs"]] == [
        "current_state_dof_pos.1",
        "previous_actions_2",
    ]
    assert [entry["name"] for entry in content["policy_inputs"]] == [
        "joint_pos",
        "previous_actions",
    ]
    assert [entry["key"] for entry in content["policy_outputs"]] == [
        "actions.1",
        "joint_pos_targets_3",
        "stiffness_targets.2",
        "damping_targets_4",
    ]
    assert [entry["name"] for entry in content["policy_outputs"]] == [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]


def test_observation_export_module_resolves_tensor_inputs_and_context_constants():
    ctx = _make_context()
    module = export_utils.ObservationExportModule(
        {
            "scaled": MdpComponent(
                compute_func=_scaled_obs,
                dynamic_vars={
                    "obs": _ExportContext.current.obs,
                    "scale": _ExportContext.current.scale,
                },
                static_params={"bias": 1.0},
            ),
            "shifted": MdpComponent(
                compute_func=_scaled_obs,
                dynamic_vars={"obs": _ExportContext.current.obs},
                static_params={"scale": 1.0, "bias": -1.0},
            ),
        },
        ctx,
        torch.device("cpu"),
    )

    assert module.get_input_keys() == ["current.obs"]
    assert module.get_output_keys() == ["scaled", "shifted"]

    scaled, shifted = module(ctx.current.obs)

    assert torch.equal(scaled, ctx.current.obs * 2.5 + 1.0)
    assert torch.equal(shifted, ctx.current.obs - 1.0)

    with pytest.raises(AssertionError, match="MdpComponent"):
        export_utils.ObservationExportModule(
            {"bad": {"fn": _scaled_obs}},
            ctx,
            torch.device("cpu"),
        )


def test_export_observations_writes_metadata_and_validates_with_runtime(
    monkeypatch,
    tmp_path,
    caplog,
):
    ctx = _make_context()
    expected = ctx.current.obs + 3.0
    captured = {}

    def shifted(obs: torch.Tensor) -> torch.Tensor:
        return obs + 3.0

    def fake_export(module, input_tensors, path, **kwargs):
        captured["outputs"] = module(*input_tensors)
        captured["input_names"] = kwargs["input_names"]
        captured["output_names"] = kwargs["output_names"]

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [
                _FakeOnnxValue("current_obs"),
                _FakeOnnxValue("current_obs.1"),
                _FakeOnnxValue("orphan_input"),
            ]

        def get_outputs(self):
            return [_FakeOnnxValue("shifted")]

        def run(self, output_names, inputs):
            assert output_names == ["shifted"]
            assert set(inputs) == {"current_obs", "current_obs.1"}
            return [(expected + 0.01).numpy()]

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    path = tmp_path / "observations.onnx"
    result = export_utils.export_observations(
        {
            "shifted": MdpComponent(
                compute_func=shifted,
                dynamic_vars={"obs": _ExportContext.current.obs},
            ),
        },
        ctx,
        str(path),
        torch.device("cpu"),
        validate=True,
        meta={"experiment": "unit"},
    )

    assert "Could not match ONNX input 'orphan_input'" in caplog.text
    assert "No value for ONNX input 'orphan_input'" in caplog.text
    assert result == str(path)
    assert captured["input_names"] == ["current_obs"]
    assert captured["output_names"] == ["shifted"]
    assert torch.equal(captured["outputs"][0], expected)

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["type"] == "observations"
    assert metadata["experiment"] == "unit"
    assert metadata["onnx_in_names"] == [
        "current_obs",
        "current_obs.1",
        "orphan_input",
    ]
    assert metadata["onnx_name_to_in_key"] == {
        "current_obs": "current.obs",
        "current_obs.1": "current.obs",
    }
    assert metadata["input_shapes"] == {"current.obs": [2, 2]}
    assert metadata["output_shapes"] == {"shifted": [2, 2]}


def test_export_observations_rejects_non_tensor_context_inputs(tmp_path):
    class FlakyCurrent:
        def __init__(self):
            self.calls = 0

        @property
        def obs(self):
            self.calls += 1
            if self.calls == 1:
                raise TypeError("defer resolution until export")
            return 3.0

    ctx = SimpleNamespace(current=FlakyCurrent())

    with pytest.raises(ValueError, match="is not a tensor"):
        export_utils.export_observations(
            {
                "bad": MdpComponent(
                    compute_func=_scaled_obs,
                    dynamic_vars={"obs": _ExportContext.current.obs},
                ),
            },
            ctx,
            str(tmp_path / "bad.onnx"),
            torch.device("cpu"),
        )


def test_export_observations_validation_import_failure_warns(
    monkeypatch,
    tmp_path,
    caplog,
):
    ctx = _make_context()

    def shifted(obs: torch.Tensor) -> torch.Tensor:
        return obs + 1.0

    def fake_export(module, input_tensors, path, **kwargs):
        module(*input_tensors)

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [_FakeOnnxValue("current_obs")]

        def get_outputs(self):
            return [_FakeOnnxValue("shifted")]

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level("WARNING"):
        result = export_utils.export_observations(
            {
                "shifted": MdpComponent(
                    compute_func=shifted,
                    dynamic_vars={"obs": _ExportContext.current.obs},
                ),
            },
            ctx,
            str(tmp_path / "observations.onnx"),
            torch.device("cpu"),
            validate=True,
        )

    assert result == str(tmp_path / "observations.onnx")
    assert "onnxruntime not installed, skipping validation" in caplog.text


def test_export_observations_validation_exception_is_propagated(
    monkeypatch,
    tmp_path,
    caplog,
):
    ctx = _make_context()

    def shifted(obs: torch.Tensor) -> torch.Tensor:
        return obs + 1.0

    def fake_export(module, input_tensors, path, **kwargs):
        module(*input_tensors)

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [_FakeOnnxValue("current_obs")]

        def get_outputs(self):
            return [_FakeOnnxValue("shifted")]

        def run(self, output_names, inputs):
            raise RuntimeError("observation validation exploded")

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="observation validation exploded"):
            export_utils.export_observations(
                {
                    "shifted": MdpComponent(
                        compute_func=shifted,
                        dynamic_vars={"obs": _ExportContext.current.obs},
                    ),
                },
                ctx,
                str(tmp_path / "observations.onnx"),
                torch.device("cpu"),
                validate=True,
            )

    assert "Validation failed: observation validation exploded" in caplog.text


def test_export_unified_pipeline_exports_yaml_metadata_and_validates(
    monkeypatch,
    tmp_path,
):
    ctx = _make_context()
    captured = {}

    def fake_export(module, input_tensors, path, **kwargs):
        captured["outputs"] = module(*input_tensors)
        captured["input_names"] = kwargs["input_names"]
        captured["output_names"] = kwargs["output_names"]

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [_FakeOnnxValue(name) for name in captured["input_names"]]

        def get_outputs(self):
            return [_FakeOnnxValue(name) for name in captured["output_names"]]

        def run(self, output_names, inputs):
            assert output_names == captured["output_names"]
            assert set(inputs) == set(captured["input_names"])
            return [out.detach().cpu().numpy() for out in captured["outputs"]]

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(
            dof_names=["hip", "knee"],
            body_names=["pelvis", "torso"],
        ),
        control=SimpleNamespace(
            control_info={
                "hip": SimpleNamespace(stiffness=10.0, damping=1.0),
                "knee": SimpleNamespace(stiffness=20.0, damping=2.0),
            },
        ),
        anchor_body_name=None,
    )
    path = tmp_path / "pipeline.onnx"

    result = export_utils.export_unified_pipeline(
        observation_configs={
            "current_state_dof_pos": MdpComponent(
                compute_func=_state_obs,
                dynamic_vars={"dof_pos": _ExportContext.current_state.dof_pos},
            ),
        },
        action_config={
            "fn": _process_action,
            "offset": torch.tensor([0.25, 0.5]),
            "stiffness": torch.tensor([10.0, 20.0]),
            "damping": torch.tensor([1.0, 2.0]),
            "gain": 2.0,
        },
        sample_context=ctx,
        policy_module=_PolicyModule(),
        policy_in_keys=["current_state_dof_pos", "previous_actions"],
        policy_action_key="mean_action",
        path=str(path),
        device=torch.device("cpu"),
        robot_config=robot_config,
        passthrough_obs={"previous_actions": torch.ones(2, 2)},
        validate=True,
        meta={"checkpoint": "unit"},
        dt=0.02,
    )

    assert result == str(path)
    assert captured["input_names"] == ["current_state_dof_pos", "previous_actions"]
    assert captured["output_names"] == [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]

    import yaml

    content = yaml.safe_load(path.with_suffix(".yaml").read_text())
    assert content["type"] == "unified_pipeline"
    assert content["dt"] == 0.02
    assert content["joint_names"] == ["hip", "knee"]
    assert content["default_joint_stiffness"] == [10.0, 20.0]
    assert content["default_joint_damping"] == [1.0, 2.0]
    assert content["metadata"] == {"checkpoint": "unit"}
    assert content["_runtime"]["onnx_name_to_in_key"] == {
        "current_state_dof_pos": "current_state.dof_pos",
        "previous_actions": "previous_actions",
    }
    assert [entry["key"] for entry in content["policy_inputs"]] == [
        "current_state_dof_pos",
        "previous_actions",
    ]
    assert [entry["key"] for entry in content["policy_outputs"]] == [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]


def test_export_unified_pipeline_recovers_multi_digit_suffix_renamed_onnx_names(
    monkeypatch,
    tmp_path,
):
    ctx = _make_context()
    captured = {}
    actual_inputs = ["current_state_dof_pos.4", "previous_actions_12"]
    actual_outputs = [
        "actions.5",
        "joint_pos_targets_6",
        "stiffness_targets.7",
        "damping_targets_8",
    ]

    def fake_export(module, input_tensors, path, **kwargs):
        captured["outputs"] = module(*input_tensors)

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [_FakeOnnxValue(name) for name in actual_inputs]

        def get_outputs(self):
            return [_FakeOnnxValue(name) for name in actual_outputs]

        def run(self, output_names, inputs):
            assert output_names == actual_outputs
            assert set(inputs) == set(actual_inputs)
            return [out.detach().cpu().numpy() for out in captured["outputs"]]

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(
            dof_names=["hip", "knee"],
            body_names=["pelvis", "torso"],
        ),
        control=SimpleNamespace(
            control_info={
                "hip": SimpleNamespace(stiffness=10.0, damping=1.0),
                "knee": SimpleNamespace(stiffness=20.0, damping=2.0),
            },
        ),
        anchor_body_name="pelvis",
    )
    path = tmp_path / "pipeline.onnx"

    export_utils.export_unified_pipeline(
        observation_configs={
            "current_state_dof_pos": MdpComponent(
                compute_func=_state_obs,
                dynamic_vars={"dof_pos": _ExportContext.current_state.dof_pos},
            ),
        },
        action_config={
            "fn": _process_action,
            "offset": torch.tensor([0.25, 0.5]),
            "stiffness": torch.tensor([10.0, 20.0]),
            "damping": torch.tensor([1.0, 2.0]),
            "gain": 2.0,
        },
        sample_context=ctx,
        policy_module=_PolicyModule(),
        policy_in_keys=["current_state_dof_pos", "previous_actions"],
        policy_action_key="mean_action",
        path=str(path),
        device=torch.device("cpu"),
        robot_config=robot_config,
        passthrough_obs={"previous_actions": torch.ones(2, 2)},
        validate=True,
    )

    import yaml

    content = yaml.safe_load(path.with_suffix(".yaml").read_text())
    assert content["_runtime"]["onnx_name_to_in_key"] == {
        "current_state_dof_pos.4": "current_state.dof_pos",
        "previous_actions_12": "previous_actions",
    }
    assert [entry["key"] for entry in content["policy_inputs"]] == actual_inputs
    assert [entry["key"] for entry in content["policy_outputs"]] == actual_outputs


def test_export_unified_pipeline_warns_for_unmatched_inputs_and_validation_drift(
    monkeypatch,
    tmp_path,
    caplog,
):
    ctx = _make_context()
    captured = {}
    actual_inputs = ["current_state_dof_pos", "previous_actions", "mystery_input"]
    actual_outputs = [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]

    def fake_export(module, input_tensors, path, **kwargs):
        captured["outputs"] = module(*input_tensors)

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [_FakeOnnxValue(name) for name in actual_inputs]

        def get_outputs(self):
            return [_FakeOnnxValue(name) for name in actual_outputs]

        def run(self, output_names, inputs):
            assert output_names == actual_outputs
            assert set(inputs) == {"current_state_dof_pos", "previous_actions"}
            outputs = [out.detach().cpu().numpy() for out in captured["outputs"]]
            outputs[0] = outputs[0] + 1.0
            return outputs

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(
            dof_names=["hip", "knee"],
            body_names=["pelvis", "torso"],
        ),
        control=SimpleNamespace(
            control_info={
                "hip": SimpleNamespace(stiffness=10.0, damping=1.0),
                "knee": SimpleNamespace(stiffness=20.0, damping=2.0),
            },
        ),
        anchor_body_name="pelvis",
    )

    with caplog.at_level("WARNING"):
        export_utils.export_unified_pipeline(
            observation_configs={
                "current_state_dof_pos": MdpComponent(
                    compute_func=_state_obs,
                    dynamic_vars={"dof_pos": _ExportContext.current_state.dof_pos},
                ),
            },
            action_config={
                "fn": _process_action,
                "offset": torch.tensor([0.25, 0.5]),
                "stiffness": torch.tensor([10.0, 20.0]),
                "damping": torch.tensor([1.0, 2.0]),
                "gain": 2.0,
            },
            sample_context=ctx,
            policy_module=_PolicyModule(),
            policy_in_keys=["current_state_dof_pos", "previous_actions"],
            policy_action_key="mean_action",
            path=str(tmp_path / "pipeline.onnx"),
            device=torch.device("cpu"),
            robot_config=robot_config,
            passthrough_obs={"previous_actions": torch.ones(2, 2)},
            validate=True,
        )

    import yaml

    content = yaml.safe_load((tmp_path / "pipeline.yaml").read_text())
    assert content["_runtime"]["onnx_in_names"] == actual_inputs
    assert content["_runtime"]["onnx_name_to_in_key"] == {
        "current_state_dof_pos": "current_state.dof_pos",
        "previous_actions": "previous_actions",
    }
    assert "Could not match ONNX input 'mystery_input'" in caplog.text
    assert "Large difference detected for actions" in caplog.text


def test_export_unified_pipeline_validation_import_failure_warns_and_keeps_yaml(
    monkeypatch,
    tmp_path,
    caplog,
):
    ctx = _make_context()

    def fake_export(module, input_tensors, path, **kwargs):
        module(*input_tensors)

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [_FakeOnnxValue("current_state_dof_pos"), _FakeOnnxValue("previous_actions")]

        def get_outputs(self):
            return [
                _FakeOnnxValue("actions"),
                _FakeOnnxValue("joint_pos_targets"),
                _FakeOnnxValue("stiffness_targets"),
                _FakeOnnxValue("damping_targets"),
            ]

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(
            dof_names=["hip", "knee"],
            body_names=["pelvis", "torso"],
        ),
        control=SimpleNamespace(
            control_info={
                "hip": SimpleNamespace(stiffness=10.0, damping=1.0),
                "knee": SimpleNamespace(stiffness=20.0, damping=2.0),
            },
        ),
        anchor_body_name="pelvis",
    )
    path = tmp_path / "pipeline.onnx"

    with caplog.at_level("WARNING"):
        result = export_utils.export_unified_pipeline(
            observation_configs={
                "current_state_dof_pos": MdpComponent(
                    compute_func=_state_obs,
                    dynamic_vars={"dof_pos": _ExportContext.current_state.dof_pos},
                ),
            },
            action_config={
                "fn": _process_action,
                "offset": torch.tensor([0.25, 0.5]),
                "stiffness": torch.tensor([10.0, 20.0]),
                "damping": torch.tensor([1.0, 2.0]),
                "gain": 2.0,
            },
            sample_context=ctx,
            policy_module=_PolicyModule(),
            policy_in_keys=["current_state_dof_pos", "previous_actions"],
            policy_action_key="mean_action",
            path=str(path),
            device=torch.device("cpu"),
            robot_config=robot_config,
            passthrough_obs={"previous_actions": torch.ones(2, 2)},
            validate=True,
        )

    import yaml

    content = yaml.safe_load(path.with_suffix(".yaml").read_text())
    assert result == str(path)
    assert content["_runtime"]["onnx_name_to_in_key"] == {
        "current_state_dof_pos": "current_state.dof_pos",
        "previous_actions": "previous_actions",
    }
    assert "onnxruntime not installed, skipping validation" in caplog.text


def test_export_unified_pipeline_validation_exception_is_propagated(
    monkeypatch,
    tmp_path,
    caplog,
):
    ctx = _make_context()
    captured = {}

    def fake_export(module, input_tensors, path, **kwargs):
        captured["outputs"] = module(*input_tensors)

    class FakeSession:
        def __init__(self, path, providers):
            self.path = path
            self.providers = providers

        def get_inputs(self):
            return [_FakeOnnxValue("current_state_dof_pos"), _FakeOnnxValue("previous_actions")]

        def get_outputs(self):
            return [
                _FakeOnnxValue("actions"),
                _FakeOnnxValue("joint_pos_targets"),
                _FakeOnnxValue("stiffness_targets"),
                _FakeOnnxValue("damping_targets"),
            ]

        def run(self, output_names, inputs):
            assert set(inputs) == {"current_state_dof_pos", "previous_actions"}
            raise RuntimeError("unified validation exploded")

    monkeypatch.setattr(export_utils.torch.onnx, "export", fake_export)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )

    robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(
            dof_names=["hip", "knee"],
            body_names=["pelvis", "torso"],
        ),
        control=SimpleNamespace(
            control_info={
                "hip": SimpleNamespace(stiffness=10.0, damping=1.0),
                "knee": SimpleNamespace(stiffness=20.0, damping=2.0),
            },
        ),
        anchor_body_name="pelvis",
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="unified validation exploded"):
            export_utils.export_unified_pipeline(
                observation_configs={
                    "current_state_dof_pos": MdpComponent(
                        compute_func=_state_obs,
                        dynamic_vars={"dof_pos": _ExportContext.current_state.dof_pos},
                    ),
                },
                action_config={
                    "fn": _process_action,
                    "offset": torch.tensor([0.25, 0.5]),
                    "stiffness": torch.tensor([10.0, 20.0]),
                    "damping": torch.tensor([1.0, 2.0]),
                    "gain": 2.0,
                },
                sample_context=ctx,
                policy_module=_PolicyModule(),
                policy_in_keys=["current_state_dof_pos", "previous_actions"],
                policy_action_key="mean_action",
                path=str(tmp_path / "pipeline.onnx"),
                device=torch.device("cpu"),
                robot_config=robot_config,
                passthrough_obs={"previous_actions": torch.ones(2, 2)},
                validate=True,
            )

    assert captured["outputs"][0].shape == torch.Size([2, 2])
    assert "Validation failed: unified validation exploded" in caplog.text


def test_export_unified_pipeline_rejects_missing_policy_inputs(tmp_path):
    with pytest.raises(ValueError, match="Policy requires inputs"):
        export_utils.export_unified_pipeline(
            observation_configs={},
            action_config={"fn": _process_action},
            sample_context=_make_context(),
            policy_module=_PolicyModule(),
            policy_in_keys=["missing_obs"],
            policy_action_key="mean_action",
            path=str(tmp_path / "missing.onnx"),
            device=torch.device("cpu"),
            robot_config=SimpleNamespace(
                kinematic_info=SimpleNamespace(dof_names=[], body_names=["pelvis"]),
                control=SimpleNamespace(control_info={}),
                anchor_body_name="pelvis",
            ),
        )
