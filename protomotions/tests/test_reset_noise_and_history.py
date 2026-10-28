# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical reset perturbations and the unified state history."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.obs.state_history_buffer import HistorySample, StateHistoryBuffer
from protomotions.envs.reset_noise import apply_reset_noise
from protomotions.envs.state import BodyState


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def test_apply_reset_noise_clamps_dofs_and_updates_fields():
    reset_state = SimpleNamespace(
        dof_pos=torch.tensor([[-2.0, 0.5, 3.0]]),
        dof_vel=torch.ones(1, 3),
        root_pos=torch.zeros(1, 3),
        root_rot=_identity_quat(1),
        root_vel=torch.ones(1, 3),
        root_ang_vel=torch.ones(1, 3) * 2,
    )
    config = SimpleNamespace(
        dof_pos_noise=0.0,
        dof_vel_noise=0.0,
        root_pos_noise=0.0,
        root_rot_noise=0.0,
        root_vel_noise=0.0,
        root_ang_vel_noise=0.0,
    )

    apply_reset_noise(
        reset_state,
        config,
        dof_limits_lower=torch.tensor([-1.0, -1.0, -1.0]),
        dof_limits_upper=torch.tensor([1.0, 1.0, 1.0]),
        generator=None,
    )

    assert torch.equal(reset_state.dof_pos, torch.tensor([[-1.0, 0.5, 1.0]]))
    assert torch.equal(reset_state.root_rot, _identity_quat(1))


def test_apply_reset_noise_perturbs_within_the_declared_half_widths():
    torch.manual_seed(0)
    reset_state = SimpleNamespace(
        dof_pos=torch.zeros(64, 3),
        dof_vel=torch.zeros(64, 3),
        root_pos=torch.zeros(64, 3),
        root_rot=_identity_quat(64),
        root_vel=torch.zeros(64, 3),
        root_ang_vel=torch.zeros(64, 3),
    )
    config = SimpleNamespace(
        dof_pos_noise=0.1,
        dof_vel_noise=0.0,
        root_pos_noise=[0.05, 0.05, 0.0],
        root_rot_noise=0.2,
        root_vel_noise=0.0,
        root_ang_vel_noise=0.0,
    )

    apply_reset_noise(
        reset_state, config, torch.full((3,), -1.0), torch.full((3,), 1.0), None
    )

    assert reset_state.dof_pos.abs().max() <= 0.1
    assert reset_state.dof_pos.abs().max() > 0.0
    assert reset_state.root_pos[:, :2].abs().max() <= 0.05
    assert torch.equal(reset_state.root_pos[:, 2], torch.zeros(64))
    assert torch.allclose(reset_state.root_rot.norm(dim=-1), torch.ones(64))
    assert torch.equal(reset_state.dof_vel, torch.zeros(64, 3))


# =============================================================================
# State history
# =============================================================================

N, B, D, A = 3, 2, 3, 2


def _state(base: float, *, sensor: bool = False) -> BodyState:
    """Privileged state, or a sensor-like one without positions and contacts."""
    common = dict(
        rigid_body_rot=_identity_quat(N, B),
        rigid_body_ang_vel=torch.full((N, B, 3), base + 1),
        dof_pos=torch.full((N, D), base + 2),
        dof_vel=torch.full((N, D), base + 3),
    )
    if sensor:
        return BodyState(**common)
    return BodyState(
        **common,
        rigid_body_pos=torch.full((N, B, 3), base),
        rigid_body_vel=torch.full((N, B, 3), base + 4),
        ground_heights=torch.full((N,), base + 5),
        rigid_body_contacts=torch.tensor([[True, False], [False, True], [True, True]]),
        dof_forces=torch.full((N, D), base + 7),
    )


def _sample(base: float, *, sensors: bool = False) -> HistorySample:
    states = {"privileged": _state(base)}
    if sensors:
        states.update(
            clean=_state(base + 100, sensor=True), noisy=_state(base + 200, sensor=True)
        )
    return HistorySample(
        states, torch.full((N, A), base + 8), torch.full((N, A), base + 9)
    )


def test_construction_repeats_the_first_sample_and_zeroes_commands():
    buffer = StateHistoryBuffer(2, _sample(1.0))

    captured = buffer.capture()
    privileged = captured.states["privileged"]
    assert privileged.rigid_body_pos.shape == (N, 3, B, 3)
    for slot in range(3):
        assert torch.equal(
            privileged.rigid_body_pos[:, slot], torch.full((N, B, 3), 1.0)
        )
    assert torch.equal(captured.raw_actions, torch.zeros(N, 3, A))
    assert torch.equal(captured.processed_actions, torch.zeros(N, 3, A))


def test_push_shifts_every_source_and_stores_the_incoming_command():
    buffer = StateHistoryBuffer(2, _sample(1.0, sensors=True))

    buffer.push(_sample(10.0, sensors=True))
    buffer.push(_sample(20.0, sensors=True))

    captured = buffer.capture()
    assert torch.equal(
        captured.states["privileged"].dof_pos[:, 0], torch.full((N, D), 22.0)
    )
    assert torch.equal(
        captured.states["privileged"].dof_pos[:, 1], torch.full((N, D), 12.0)
    )
    assert torch.equal(
        captured.states["privileged"].dof_pos[:, 2], torch.full((N, D), 3.0)
    )
    assert torch.equal(
        captured.states["noisy"].dof_pos[:, 0], torch.full((N, D), 222.0)
    )
    assert torch.equal(captured.raw_actions[:, 0], torch.full((N, A), 28.0))
    assert torch.equal(captured.raw_actions[:, 1], torch.full((N, A), 18.0))
    assert torch.equal(captured.processed_actions[:, 0], torch.full((N, A), 29.0))
    # Sensor sources carry only what they measured.
    assert captured.states["noisy"].rigid_body_pos is None
    assert captured.states["noisy"].rigid_body_contacts is None


def test_capture_returns_owned_tensors_in_the_requested_order():
    buffer = StateHistoryBuffer(1, _sample(1.0))
    buffer.push(_sample(5.0))

    subset = buffer.capture(torch.tensor([2, 0]))
    subset.states["privileged"].dof_pos.fill_(-1.0)
    subset.raw_actions.fill_(-1.0)

    full = buffer.capture()
    assert torch.equal(full.states["privileged"].dof_pos[:, 0], torch.full((N, D), 7.0))
    assert torch.equal(full.raw_actions[:, 0], torch.full((N, A), 13.0))
    assert subset.states["privileged"].dof_pos.shape == (2, 2, D)


def test_partial_reset_repeats_the_new_state_and_leaves_other_rows_alone():
    buffer = StateHistoryBuffer(2, _sample(1.0))
    buffer.push(_sample(10.0))
    env_ids = torch.tensor([2, 0])
    reset_sample = _sample(50.0)
    reset_sample = reset_sample.map(lambda value: value[env_ids])

    buffer.reset(env_ids, reset_sample)

    captured = buffer.capture()
    privileged = captured.states["privileged"]
    for slot in range(3):
        assert torch.equal(privileged.dof_pos[env_ids, slot], torch.full((2, D), 52.0))
    assert torch.equal(privileged.dof_pos[1, 0], torch.full((D,), 12.0))
    assert torch.equal(privileged.dof_pos[1, 1], torch.full((D,), 3.0))
    assert torch.equal(captured.raw_actions[env_ids], torch.zeros(2, 3, A))
    assert torch.equal(captured.raw_actions[1, 0], torch.full((A,), 18.0))


def test_snapshot_round_trips_and_rejects_a_mismatched_schema():
    buffer = StateHistoryBuffer(1, _sample(1.0, sensors=True))
    buffer.push(_sample(10.0, sensors=True))
    snapshot = buffer.save_state()

    buffer.push(_sample(20.0, sensors=True))
    buffer.load_state(snapshot)

    restored = buffer.capture()
    assert torch.equal(
        restored.states["clean"].dof_pos[:, 0], torch.full((N, D), 112.0)
    )
    assert torch.equal(restored.raw_actions[:, 0], torch.full((N, A), 18.0))

    with pytest.raises(ValueError):
        buffer.push(_sample(30.0))  # the sensor sources are missing
    with pytest.raises(ValueError):
        buffer.load_state(
            StateHistoryBuffer(2, _sample(1.0, sensors=True)).save_state()
        )
    with pytest.raises(ValueError):
        StateHistoryBuffer(-1, _sample(1.0))


def test_history_depth_follows_the_ring_roots_lags():
    """``first_lag`` and ``history_depth``: the historical namespaces and the
    ``previous_*`` names start one step back, the two action rings at the
    current command, every other root is not the ring's; the depth is the
    deepest read, and a history window read without a step selection is
    refused by name."""
    import pytest

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
        compute_historical_reduced_coords_from_state,
    )
    from protomotions.envs.obs.state_history_buffer import first_lag, history_depth
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import compute_action_rate_l2

    rate = MdpComponent.from_kernel(
        compute_action_rate_l2, meta=RewardMeta(weight=-0.1)
    )
    assert first_lag("historical") == first_lag("noisy_historical") == 1
    assert first_lag("previous_action") == first_lag("previous_processed_action") == 1
    assert first_lag("raw_action_history") == first_lag("processed_action_history") == 0
    assert first_lag("current") is None and first_lag("mimic") is None

    assert history_depth({}) == 0
    # previous_action is one past slot; the action ring's slot 1 is the current
    # command (no past slot), its slot 2 one past slot.
    assert history_depth({"rate": rate}) == 1

    def actions(steps: int) -> MdpComponent:
        return MdpComponent.from_kernel(
            compute_historical_actions_from_state, history_steps=steps
        )

    assert history_depth({"a": actions(1)}) == 0
    assert history_depth({"a": actions(2)}) == 1
    # A historical read of steps 1 and 3 needs three past slots; the deepest wins.
    deep = MdpComponent.from_kernel(
        compute_historical_reduced_coords_from_state, history_steps=[1, 3]
    )
    assert history_depth({"h": deep, "rate": rate}) == 3

    unselected = MdpComponent(lambda x: x, {"x": EnvContext.historical.dof_pos})
    with pytest.raises(
        ValueError, match="observation/bare.*declares no step selection"
    ):
        history_depth({"observation/bare": unselected})
