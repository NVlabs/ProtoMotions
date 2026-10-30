# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph-owned history update must reproduce InputHistory exactly."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.contract import HistorySlotSpec, InputSpec, ObservationHistorySpec
from deployment.input_history import InputHistory, history_specs
from deployment.native.tools.leapp_graph_history import (
    GraphHistory,
    _step,
    require_torchscript_disabled,
)


def _spec(
    key, shape, provider, history=None, observation_history=None, source_namespace=None
):
    return InputSpec(
        onnx_name=key.replace(".", "_"),
        key=key,
        source_namespace=source_namespace or key.split(".")[0],
        kind="test",
        provider=provider,
        shape=shape,
        dtype="float32",
        element_names=(),
        future_steps=None,
        history=history,
        observation_history=observation_history,
    )


def _metatron_like_contract():
    # Same layout as the Metatron export: one measurement with an 8-step history,
    # and a 1-step and an 8-step view of one raw-action history.
    current_action, _ = history_specs("current_action", (1, 29))
    historical_actions, _ = history_specs("noisy_historical.raw_actions", (1, 8, 29))
    return SimpleNamespace(
        inputs=[
            _spec("noisy.dof_pos", (1, 29), "joint_encoders"),
            _spec("current_action", (1, 29), "policy_history", history=current_action),
            _spec(
                "noisy_historical.dof_pos",
                (1, 8, 29),
                "joint_encoders",
                observation_history=ObservationHistorySpec(
                    current_key="noisy.dof_pos",
                    steps_back=tuple(range(1, 9)),
                    reset_fill="initial_measurement",
                ),
            ),
            _spec(
                "noisy_historical.raw_actions",
                (1, 8, 29),
                "policy_history",
                history=historical_actions,
            ),
        ]
    )


def test_graph_history_matches_input_history_through_resets_and_drops():
    contract = _metatron_like_contract()
    host = InputHistory(contract.inputs)
    graph = GraphHistory(contract)
    assert graph.applied_keys == ["actions"]
    rng = np.random.default_rng(3)
    state = graph.initial_state()
    applied, accepted = np.zeros((1, 29), np.float32), False
    for tick in range(120):
        measurements = {
            "noisy.dof_pos": rng.standard_normal((1, 29)).astype(np.float32)
        }
        if tick in (0, 50, 51, 90):
            host.reset(measurements)
            state = graph.initial_state()
        expected = host.inputs()
        actual, state = graph.advance(
            state,
            {key: torch.from_numpy(value) for key, value in measurements.items()},
            {"actions": torch.from_numpy(applied)},
            torch.tensor([[accepted]]),
        )
        for key, value in expected.items():
            np.testing.assert_allclose(
                actual[key].numpy(), value, atol=1e-7, err_msg=f"{key} @ {tick}"
            )
        # The controller edits some commands and drops others.
        applied = rng.standard_normal((1, 29)).astype(np.float32)
        accepted = bool(rng.random() > 0.2)
        if accepted:
            host.commit(measurements, {"actions": applied})


def test_measurement_history_requires_its_current_input():
    contract = _metatron_like_contract()
    contract.inputs = [spec for spec in contract.inputs if spec.key != "noisy.dof_pos"]
    with pytest.raises(ValueError, match="needs noisy.dof_pos"):
        GraphHistory(contract)


def test_measurement_history_follows_the_renamed_current_input():
    # The tracker export renames current sensor inputs to ``current.<leaf>``
    # (tracker_input_keys); the history still names its source, ``noisy.<leaf>``.
    _, dof_pos_history = history_specs("noisy_historical.dof_pos", (1, 4, 29))
    assert dof_pos_history.current_key == "noisy.dof_pos"
    contract = SimpleNamespace(
        inputs=[
            _spec(
                "current.dof_pos", (1, 29), "joint_encoders", source_namespace="noisy"
            ),
            _spec(
                "noisy_historical.dof_pos",
                (1, 4, 29),
                "joint_encoders",
                observation_history=dof_pos_history,
            ),
        ]
    )
    graph = GraphHistory(contract)
    assert [buffer.key for buffer in graph.buffers] == ["current.dof_pos"]
    host = InputHistory(contract.inputs)
    rng = np.random.default_rng(5)
    state = graph.initial_state()
    for tick in range(12):
        reading = rng.standard_normal((1, 29)).astype(np.float32)
        if tick in (0, 7):
            host.reset({"noisy.dof_pos": reading})
            state = graph.initial_state()
        expected = host.inputs()["noisy_historical.dof_pos"]
        actual, state = graph.advance(
            state,
            {"current.dof_pos": torch.from_numpy(reading)},
            {},
            torch.tensor([[True]]),
        )
        np.testing.assert_allclose(
            actual["noisy_historical.dof_pos"].numpy(),
            expected,
            atol=1e-7,
            err_msg=f"tick {tick}",
        )
        host.commit({"noisy.dof_pos": reading}, {})


def test_action_history_resets_to_the_contract_fill():
    # A processed-action history exported with ``fill_before_available`` set to
    # the default pose: after a reset every slot holds that pose, not zeros.
    pose = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
    targets = HistorySlotSpec(
        output_key="joint_pos_targets",
        slots=3,
        first_action_lag=2,
        reset_fill=[float(value) for value in pose],
        steps_back=(2, 3, 4),
        reset_value=None,
    )
    contract = SimpleNamespace(
        inputs=[
            _spec("noisy.dof_pos", (1, 29), "joint_encoders"),
            _spec(
                "noisy_historical.processed_actions",
                (1, 3, 29),
                "policy_history",
                history=targets,
            ),
        ]
    )
    host = InputHistory(contract.inputs)
    graph = GraphHistory(contract)
    assert graph.applied_keys == ["joint_pos_targets"]
    reading = {"noisy.dof_pos": np.zeros((1, 29), np.float32)}
    host.reset(reading)
    state = graph.initial_state()
    applied = np.zeros((1, 29), np.float32)
    for tick in range(6):
        expected = host.inputs()["noisy_historical.processed_actions"]
        if tick == 0:
            np.testing.assert_allclose(expected, np.broadcast_to(pose, (1, 3, 29)))
        actual, state = graph.advance(
            state,
            {"noisy.dof_pos": torch.zeros(1, 29)},
            {"joint_pos_targets": torch.from_numpy(applied)},
            torch.tensor([[True]]),
        )
        np.testing.assert_allclose(
            actual["noisy_historical.processed_actions"].numpy(),
            expected,
            atol=1e-7,
            err_msg=f"tick {tick}",
        )
        applied = np.full((1, 29), float(tick + 1), np.float32)
        host.commit(reading, {"joint_pos_targets": applied})
    # Four accepted commands later the reset fill has left every slot.
    assert not np.isclose(
        host.inputs()["noisy_historical.processed_actions"], pose
    ).any()


def test_step_names_a_pipeline_input_no_tick_supplies():
    contract = _metatron_like_contract()
    graph = GraphHistory(contract)
    tick = {spec.key: torch.zeros(spec.shape) for spec in graph.host_specs}
    tick["applied_actions"] = torch.zeros(1, 29)
    tick["command_accepted"] = torch.tensor([[True]])
    keys = [spec.key for spec in contract.inputs] + ["terrain.heights"]
    with pytest.raises(KeyError, match="terrain.heights"):
        _step(lambda *args: args, graph, keys, graph.initial_state(), tick)


def test_tracing_refuses_to_run_with_torchscript_enabled(monkeypatch):
    # Scripted helpers would be frozen at the first traced tick.
    monkeypatch.setattr(torch.jit._state._enabled, "enabled", True)
    with pytest.raises(RuntimeError, match="PYTORCH_JIT=0"):
        require_torchscript_disabled()
    monkeypatch.setattr(torch.jit._state._enabled, "enabled", False)
    require_torchscript_disabled()
