# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for eager multi-source state and action history."""

from dataclasses import replace

import pytest
import torch

from protomotions.envs.base_env.publishers import StatePublisher
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context import FrozenContext
from protomotions.envs.state_names import body_state_names
from protomotions.envs.context_paths import resolve_path
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.state_history_buffer import HistorySample, StateHistoryBuffer
from protomotions.envs.state import BodyState


def sample(step: float = 0, num_envs: int = 3) -> HistorySample:
    rot = torch.zeros(num_envs, 2, 4)
    rot[..., 3] = 1
    state = BodyState(
        rigid_body_rot=rot,
        rigid_body_ang_vel=torch.zeros(num_envs, 2, 3),
        dof_pos=torch.full((num_envs, 2), float(step)),
        dof_vel=torch.zeros(num_envs, 2),
        rigid_body_pos=torch.full((num_envs, 2, 3), float(step)),
    )
    return HistorySample(
        states={
            "privileged": state,
            "clean": state.map(torch.clone),
            "noisy": replace(state, dof_pos=state.dof_pos + 0.125),
        },
        raw_actions=torch.full((num_envs, 2), float(step * 10)),
        processed_actions=torch.full((num_envs, 2), float(step * 2)),
    )


@pytest.mark.parametrize("history_steps", [0, 1, 4])
def test_history_owns_values_and_keeps_sensor_reset_noise(history_steps):
    initial = sample()
    history = StateHistoryBuffer(history_steps, initial)
    initial.states["noisy"].dof_pos.fill_(9)
    captured = history.capture()
    assert captured.states["noisy"].dof_pos.shape == (3, history_steps + 1, 2)
    assert torch.all(captured.states["noisy"].dof_pos == 0.125)
    assert captured.states["noisy"].rigid_body_vel is None
    history.push(sample(1))
    assert torch.all(captured.states["noisy"].dof_pos == 0.125)


def test_partial_reset_seeds_all_sources_and_zeros_only_selected_commands():
    history = StateHistoryBuffer(3, sample())
    history.push(sample(1))
    old = history.capture()
    env_ids = torch.tensor([2, 0])
    history.reset(env_ids, sample(7, 2))
    current = history.capture()
    for source in current.states:
        torch.testing.assert_close(
            current.states[source].dof_pos[1], old.states[source].dof_pos[1]
        )
    assert torch.all(current.states["clean"].dof_pos[env_ids] == 7)
    assert torch.all(current.states["noisy"].dof_pos[env_ids] == 7.125)
    assert torch.count_nonzero(current.raw_actions[env_ids]) == 0
    assert torch.count_nonzero(current.processed_actions[env_ids]) == 0
    torch.testing.assert_close(current.raw_actions[1], old.raw_actions[1])


def test_action_lags_distinguish_policy_commands_from_state_aligned_history():
    history = StateHistoryBuffer(3, sample())
    for step in (1, 2, 3):
        history.push(sample(step))
    captured = history.capture()
    assert captured.raw_actions[0, :, 0].tolist() == [30, 20, 10, 0]
    assert captured.states["privileged"].dof_pos[0, :, 0].tolist() == [3, 2, 1, 0]
    published = StatePublisher(history, 1).publish(None)
    assert published["noisy_historical.raw_actions"][0, 0, 0] == 20
    assert published["noisy_historical.dof_pos"][0, 0, 0] == 2.125
    assert published["previous_action"][0, 0] == 20
    assert published["current_action"][0, 0] == 30
    assert (
        captured.processed_actions[0, 0, 0] - captured.processed_actions[0, 1, 0] == 2
    )


def test_snapshot_restore_is_exact_and_context_does_not_alias_future_writes():
    history = StateHistoryBuffer(2, sample())
    history.push(sample(1))
    snapshot = history.save_state()
    context = body_state_names(
        snapshot.states["noisy"].map(lambda value: value[:, 0]), 1
    )
    history.reset(torch.tensor([0]), sample(8, 1))
    history.push(sample(9))
    assert torch.all(context["dof_pos"] == 1.125)
    history.load_state(snapshot)
    for (name, actual), (_, expected) in zip(
        history.capture().tensors(), snapshot.tensors()
    ):
        torch.testing.assert_close(actual, expected, msg=name)


def test_selected_capture_owns_its_rows():
    history = StateHistoryBuffer(2, sample())
    selected = history.capture(torch.tensor([2, 0]))
    history.push(sample(1))
    assert torch.all(selected.states["noisy"].dof_pos == 0.125)
    selected.states["clean"].dof_pos.fill_(42)
    assert torch.all(history.capture().states["clean"].dof_pos[:, 0] == 1)


@pytest.mark.parametrize(
    "invalid", ["broadcast_shape", "missing_field", "missing_source"]
)
def test_invalid_push_does_not_broadcast_or_partially_update_history(invalid):
    history = StateHistoryBuffer(2, sample())
    before = dict(history.capture().tensors())
    new = sample(1)
    states = dict(new.states)
    if invalid == "broadcast_shape":
        states["noisy"] = replace(states["noisy"], dof_pos=torch.ones(3, 1))
    elif invalid == "missing_field":
        states["noisy"] = replace(states["noisy"], rigid_body_pos=None)
    else:
        del states["clean"]
    with pytest.raises(ValueError, match="History"):
        history.push(replace(new, states=states))
    for name, actual in history.capture().tensors():
        torch.testing.assert_close(actual, before[name], rtol=0, atol=0, msg=name)


def test_published_names_leave_unmeasured_fields_absent():
    state = replace(sample().states["clean"], rigid_body_pos=None)
    names = body_state_names(state, 1)
    assert "rigid_body_pos" not in names
    assert "root_pos" not in names
    assert "anchor_vel" not in names
    assert "dof_forces" not in names
    torch.testing.assert_close(names["anchor_rot"], state.rigid_body_rot[:, 1])


def test_finalizer_names_component_argument_and_missing_path():
    context = {
        f"clean.{name}": value
        for name, value in body_state_names(sample().states["clean"], 1).items()
    }
    component = MdpComponent(
        lambda velocity: velocity, {"velocity": EnvContext.clean.root_vel}
    )
    with pytest.raises(
        ValueError, match="observation component 'speed'.*'velocity'.*'clean.root_vel'"
    ):
        ComponentManager(torch.device("cpu")).validate_bindings(
            {"speed": component}, context, group="observation"
        )


def test_context_distinguishes_missing_from_explicit_none():
    context = FrozenContext(
        {"optional": None, "nested.dof_pos": torch.ones(2)}, env_ids=None, dt=0.1
    )
    assert "optional" in context and context["optional"] is None
    assert resolve_path(context, "optional") is None
    assert "absent" not in context
    assert EnvContext.noisy.dof_pos.path == "noisy.dof_pos"
    with pytest.raises(AttributeError, match="absent"):
        resolve_path(context, "absent")
    with pytest.raises(AttributeError):
        resolve_path(context, "nested")
