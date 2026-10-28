# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ONNX export tracing modules."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import protomotions.utils.export_utils as export_utils
from protomotions.envs.action.config import ActionConfig
from protomotions.envs.context import FrozenContext
from protomotions.envs.context_paths import FieldPath, NestedField
from protomotions.envs.mdp_component import MdpComponent


class _ObsView:
    obs: torch.Tensor = FieldPath()
    scale: float = FieldPath()


class _CurrentStateView:
    dof_pos: torch.Tensor = FieldPath()


class _ExportContext:
    current: _ObsView = NestedField(_ObsView)
    current_state: _CurrentStateView = NestedField(_CurrentStateView)


def _make_context() -> FrozenContext:
    """The sample context as the environment publishes it: a flat map."""
    return FrozenContext(
        {
            "current.obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "current.scale": 2.5,
            "current_state.dof_pos": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
        },
        env_ids=None,
        dt=0.02,
    )


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
    deployable_action_key = "mean_action"

    def forward_deployable(self, tensordict: TensorDict) -> TensorDict:
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
    deployable_action_key = "mean_action"

    def __init__(self, input_key: str | None):
        super().__init__()
        self.input_key = input_key
        self.batch_size = None

    def forward_deployable(self, tensordict: TensorDict) -> TensorDict:
        self.batch_size = tensordict.batch_size
        if self.input_key is None:
            tensordict["mean_action"] = torch.zeros(tensordict.batch_size[0], 2)
        else:
            tensordict["mean_action"] = tensordict[self.input_key]
        return tensordict


class _IdentityActionModule(torch.nn.Module):
    def forward(self, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return action, action, action


def test_action_export_module_registers_tensor_constants_and_outputs_targets():
    action_module = export_utils.ActionExportModule(
        ActionConfig(
            fn=_process_action,
            params={
                "offset": torch.tensor([1.0, -1.0]),
                "stiffness": torch.tensor([10.0, 20.0]),
                "damping": torch.tensor([1.0, 2.0]),
                "gain": 0.5,
            },
        ),
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

    assert (
        export_utils.ActionExportModule(None, torch.device("cpu"))._action_function
        is None
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
        ActionConfig(
            fn=_process_action,
            params={
                "offset": torch.tensor([0.25, 0.5]),
                "stiffness": torch.tensor([4.0, 5.0]),
                "damping": torch.tensor([0.4, 0.5]),
                "gain": 2.0,
            },
        ),
        torch.device("cpu"),
    )
    pipeline = export_utils.UnifiedPipelineModule(
        obs_module,
        _PolicyModule(),
        action_module,
        policy_in_keys=["current_state_dof_pos", "previous_actions"],
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


def test_unified_pipeline_refuses_a_policy_without_the_deployable_contract():
    class Bare(torch.nn.Module):
        def forward(self, tensordict: TensorDict) -> TensorDict:
            return tensordict

    with pytest.raises(TypeError, match="deployable contract"):
        export_utils.UnifiedPipelineModule(
            _SingleOutputObsModule([], None),
            Bare(),
            _IdentityActionModule(),
            policy_in_keys=[],
        )


def test_unified_pipeline_refuses_a_policy_without_batched_inputs():
    with pytest.raises(ValueError, match="at least one batched input"):
        export_utils.UnifiedPipelineModule(
            _SingleOutputObsModule([], None),
            _RecordBatchPolicy(None),
            _IdentityActionModule(),
            policy_in_keys=[],
        )


def test_unified_pipeline_passes_the_observation_batch_size_to_the_policy():
    """One pipeline follows the per-env observations as their batch size changes."""
    policy = _RecordBatchPolicy("obs")
    pipeline = export_utils.UnifiedPipelineModule(
        _SingleOutputObsModule(["dof_pos"], "obs"),
        policy,
        _IdentityActionModule(),
        policy_in_keys=["obs"],
    )
    for expected in (1, 3, 5):
        observations = torch.ones(expected, 2)
        actions, _, _, _ = pipeline(observations)
        assert policy.batch_size == torch.Size([expected])
        torch.testing.assert_close(actions, observations)


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

    scaled, shifted = module(ctx["current.obs"])

    assert torch.equal(scaled, ctx["current.obs"] * 2.5 + 1.0)
    assert torch.equal(shifted, ctx["current.obs"] - 1.0)

    with pytest.raises(AssertionError, match="MdpComponent"):
        export_utils.ObservationExportModule(
            {"bad": {"fn": _scaled_obs}},
            ctx,
            torch.device("cpu"),
        )


def test_observation_export_module_raises_for_a_binding_the_context_lacks():
    """A bound path absent from the sample context fails loud, not swallowed."""
    with pytest.raises(AttributeError, match="context has no"):
        export_utils.ObservationExportModule(
            {
                "missing": MdpComponent(
                    compute_func=_scaled_obs,
                    dynamic_vars={"obs": _ExportContext.current.obs},
                    static_params={"scale": 1.0, "bias": 0.0},
                ),
            },
            FrozenContext({}, env_ids=None, dt=0.02),
            torch.device("cpu"),
        )
