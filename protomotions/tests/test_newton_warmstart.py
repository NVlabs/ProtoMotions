# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Newton reset/snapshot regressions, on CPU and captured CUDA stepping."""

import pytest
import torch

from protomotions.robot_configs.base import ControlType
from protomotions.tests.test_newton_explicit_control import _make_simulator

pytestmark = pytest.mark.needs_lfs


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def simulator(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    sim = _make_simulator(ControlType.PROPORTIONAL, device=request.param)
    if request.param == "cuda":
        assert sim.use_cuda_graph, "This test must exercise captured physics"
    yield sim
    sim.close()


def _warmstart(sim):
    import warp as wp

    return wp.to_torch(sim.solver.mjw_data.qacc_warmstart)


def _assert_same_robot(actual, expected):
    for field in (
        "dof_pos",
        "dof_vel",
        "rigid_body_pos",
        "rigid_body_rot",
        "rigid_body_vel",
        "rigid_body_ang_vel",
    ):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=1e-6
        )


@pytest.mark.parametrize("indices", [None, [1], []])
def test_reset_clears_only_selected_worlds_in_place(simulator, indices):
    sim = simulator
    warmstart = _warmstart(sim)
    warmstart.copy_(
        torch.arange(warmstart.numel(), device=sim.device).reshape_as(warmstart) + 1
    )
    before = warmstart.clone()
    pointer = warmstart.data_ptr()
    ids = (
        None
        if indices is None
        else torch.tensor(indices, device=sim.device, dtype=torch.long)
    )
    state = sim.get_default_robot_reset_state()
    sim.reset_envs(state if ids is None else state[ids], env_ids=ids)
    expected = before.clone()
    if ids is None:
        expected.zero_()
    else:
        expected[ids] = 0
    assert torch.equal(warmstart, expected)
    assert _warmstart(sim).data_ptr() == pointer


def test_backend_snapshot_is_independent_and_restores_in_place(simulator):
    import warp as wp

    sim = simulator
    warmstart = _warmstart(sim)
    warmstart.fill_(7.0)
    snapshot = sim.save_backend_state()
    pointer = warmstart.data_ptr()
    # Capture an actual Warp read of the solver buffer. A replacement array
    # would leave this graph reading old storage after a restore.
    if sim.device.type == "cuda":
        observed = wp.zeros_like(sim.solver.mjw_data.qacc_warmstart)
        with wp.ScopedCapture(device=str(sim.device)) as capture:
            wp.copy(observed, sim.solver.mjw_data.qacc_warmstart)
    warmstart.zero_()
    assert torch.all(snapshot["qacc_warmstart"] == 7.0)
    sim.restore_backend_state(snapshot)
    assert torch.equal(warmstart, snapshot["qacc_warmstart"])
    assert _warmstart(sim).data_ptr() == pointer
    if sim.device.type == "cuda":
        torch.cuda.synchronize(sim.device)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(str(sim.device))
        assert torch.equal(wp.to_torch(observed), snapshot["qacc_warmstart"])
    warmstart.fill_(9.0)
    assert torch.all(snapshot["qacc_warmstart"] == 7.0)
    with pytest.raises(ValueError, match="shape"):
        sim.restore_backend_state({"qacc_warmstart": snapshot["qacc_warmstart"][:1]})


def test_restore_replays_contact_step(simulator):
    sim = simulator
    if sim.device.type == "cuda":
        pytest.skip(
            "GPU trajectory determinism is not guaranteed; buffer/graph contracts are tested separately"
        )
    initial = sim.get_default_robot_reset_state()
    sim.reset_envs(initial)
    actions = initial.dof_pos.clone()
    for _ in range(10):
        sim.step(actions)
    robot = sim.get_robot_state()
    backend = sim.save_backend_state()
    assert torch.count_nonzero(backend["qacc_warmstart"]) > 0

    sim.step(actions)
    expected = sim.get_robot_state()
    expected_warmstart = _warmstart(sim).clone()
    # Stand in for an evaluation rollout that changes the solver history.
    for _ in range(4):
        sim.step(actions + 0.05)
    sim.reset_envs(robot)
    sim.restore_backend_state(backend)
    sim.step(actions)
    actual = sim.get_robot_state()
    _assert_same_robot(actual, expected)
    torch.testing.assert_close(
        _warmstart(sim),
        expected_warmstart,
        rtol=0,
        atol=1e-6,
    )


def test_fresh_episode_replays_after_other_episode(simulator):
    sim = simulator
    if sim.device.type == "cuda":
        pytest.skip(
            "GPU trajectory determinism is not guaranteed; buffer/graph contracts are tested separately"
        )
    initial = sim.get_default_robot_reset_state()
    actions = initial.dof_pos.clone()
    sim.reset_envs(initial)
    for _ in range(4):
        sim.step(actions)
    expected = sim.get_robot_state()
    for _ in range(4):
        sim.step(actions + 0.1)
    sim.reset_envs(initial)
    for _ in range(4):
        sim.step(actions)
    _assert_same_robot(sim.get_robot_state(), expected)
