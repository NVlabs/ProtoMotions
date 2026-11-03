# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small whole-trajectory regression in the PyRoki environment."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pytest
from types import SimpleNamespace

from solve_trajectory import (
    TrajectorySolver,
    build_robot,
    frame_masks,
    pad_trajectory,
)


@pytest.mark.parametrize("frames", [1, 2, 3, 7, 8, 9, 127, 128, 129, 255, 256, 257])
def test_padding_preserves_data_and_masks_temporal_boundaries(frames):
    values = np.arange(frames * 3).reshape(frames, 3)
    arrays = pad_trajectory(values, values, values, values, 8)
    padded = ((frames + 7) // 8) * 8
    for array in arrays:
        np.testing.assert_array_equal(array[:frames], values)
        assert len(array) == padded
    for arity, mask in enumerate(frame_masks(padded, frames), start=1):
        np.testing.assert_array_equal(
            mask, np.arange(padded - arity + 1) < max(0, frames - arity + 1)
        )


def test_empty_trajectory_rejected():
    with pytest.raises(ValueError, match="at least one"):
        pad_trajectory(*(np.empty((0, 3)) for _ in range(4)), 8)


def test_recovery_rate_exception_preserves_other_gates():
    solver = object.__new__(TrajectorySolver)
    solver.frame_bucket_size = 0
    solver.order = solver.inverse = np.array([0])
    solver.lower, solver.upper = np.array([-1.0]), np.array([1.0])
    solver.max_joint_velocity, solver.max_joint_acceleration = 20.0, 300.0
    summary = SimpleNamespace(
        iterations=1,
        termination_criteria=[True],
        termination_deltas=[0.0],
        cost_history=[2.0, 1.0],
    )
    q = np.array([[0.0], [0.9], [-0.9]])
    solver.optimize = lambda *args: (q, summary)
    args = (np.zeros((3, 1, 3)), np.zeros((3, 1, 4)), np.zeros((3, 1)), q, 1 / 30)
    with pytest.raises(ValueError, match="Joint motion budget exceeded"):
        solver(*args)
    np.testing.assert_array_equal(solver(*args, allow_motion_budget_exceedance=True), q)
    assert solver.diagnostics["initial_seed_smoothed"]
    assert solver.diagnostics["motion_budget_exceeded"]
    assert solver.diagnostics["motion_budget_warning_only"]
    captured = []
    solver.optimize = lambda *inputs: (captured.append(inputs) or q, summary)
    solver(*args, allow_motion_budget_exceedance=True)
    np.testing.assert_array_equal(captured[0][0], args[0])
    np.testing.assert_array_equal(captured[0][1], args[1])
    assert not np.array_equal(captured[0][3], args[3])
    summary.termination_criteria = [False]
    with pytest.raises(ValueError, match="did not converge"):
        solver(*args, allow_motion_budget_exceedance=True)
    summary.termination_criteria = [True]
    for invalid in (2.0, np.nan):
        q[0, 0] = invalid
        with pytest.raises(ValueError, match="non-finite/out-of-limit"):
            solver(*args, allow_motion_budget_exceedance=True)


def test_known_trajectory_is_fitted_without_clipping_or_timing_changes():
    asset = (
        Path(__file__).resolve().parents[2]
        / "protomotions/data/assets/mjcf/anatomic_soma.xml"
    )
    robot = build_robot(asset)
    tree = ET.parse(asset)
    names = tree.find("custom/text[@name='semantic_body_names']").get("data").split()
    dofs = [
        j.get("name")
        for j in tree.findall(".//worldbody//joint")
        if j.get("type") == "hinge"
    ]
    expected = np.zeros((6, len(dofs)))
    expected[:, dofs.index("Spine1_x")] = np.linspace(0.02, 0.12, 6)
    order = [dofs.index(n) for n in robot.joints.actuated_names]
    links = [robot.links.names.index(n) for n in names]
    pose = np.asarray(
        jax.vmap(robot.forward_kinematics)(jnp.array(expected[:, order]))
    )[:, links]
    positions = pose[:, :, 4:].copy()
    positions[:, :, 2] += 1.2
    rotations = pose[:, :, [1, 2, 3, 0]]
    solver = TrajectorySolver(robot, names, dofs, iterations=500)
    q = solver(
        positions, rotations, np.zeros((6, len(names))), np.zeros_like(expected), 1 / 30
    )
    assert q.shape == expected.shape
    assert np.any(solver.diagnostics["termination_criteria"])
    assert (
        solver.diagnostics["cost_history"][-1] < solver.diagnostics["cost_history"][0]
    )
    assert (q >= solver.lower - 1e-6).all() and (q <= solver.upper + 1e-6).all()
    fitted = np.asarray(jax.vmap(robot.forward_kinematics)(jnp.array(q[:, order])))[
        :, links, 4:
    ]
    target = pose[:, :, 4:]
    assert np.linalg.norm(fitted - target, axis=-1).max() < 0.002
    assert np.abs(np.diff(q, axis=0)).max() < 0.1

    padded_solver = TrajectorySolver(
        robot, names, dofs, iterations=500, frame_bucket_size=8
    )
    padded_q = padded_solver(
        positions, rotations, np.zeros((6, len(names))), np.zeros_like(expected), 1 / 30
    )
    assert padded_q.shape == q.shape
    assert padded_solver.diagnostics["solver_frames"] == 8
    np.testing.assert_allclose(padded_q, q, atol=5e-4, rtol=0)
    np.testing.assert_allclose(
        padded_solver.diagnostics["cost_history"][0],
        solver.diagnostics["cost_history"][0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        padded_solver.diagnostics["cost_history"][-1],
        solver.diagnostics["cost_history"][-1],
        rtol=1e-3,
        atol=1e-6,
    )

    # Fixed does not mean stationary: arbitrary source root movement must not
    # change the root-relative fitting problem when no contacts are active.
    root_r = jaxlie.SO3.from_z_radians(jnp.linspace(0.1, 0.5, len(q)))
    root_p = jnp.stack(
        (jnp.linspace(1.0, 1.2, len(q)), jnp.zeros(len(q)), jnp.ones(len(q))), axis=-1
    )
    moved_p = (
        root_r.apply(jnp.array(target).swapaxes(0, 1)).swapaxes(0, 1) + root_p[:, None]
    )
    moved_r = (
        root_r @ jaxlie.SO3(jnp.array(pose[:, :, :4]).swapaxes(0, 1))
    ).wxyz.swapaxes(0, 1)
    moved = solver(
        np.asarray(moved_p),
        np.asarray(moved_r[:, :, [1, 2, 3, 0]]),
        np.zeros((len(q), len(names))),
        q,
        1 / 30,
    )
    moved_fk = np.asarray(
        jax.vmap(robot.forward_kinematics)(jnp.array(moved[:, order]))
    )[:, links, 4:]
    np.testing.assert_allclose(moved_fk, fitted, atol=5e-4)


def test_goal_priorities_are_explicit_and_end_effector_oriented():
    from data.scripts.from_soma_to_anatomical_soma.soma23_to_anatomical_soma_retargeting import (
        SOMA_POSITION_WEIGHTS,
        SOMA_ORIENTATION_WEIGHTS,
    )

    for side in ("Left", "Right"):
        assert (
            SOMA_POSITION_WEIGHTS[f"{side}Hand"]
            == 4 * SOMA_POSITION_WEIGHTS[f"{side}ForeArm"]
        )
        assert (
            SOMA_ORIENTATION_WEIGHTS[f"{side}Hand"]
            > SOMA_ORIENTATION_WEIGHTS[f"{side}ForeArm"]
        )
    assert (
        SOMA_POSITION_WEIGHTS["Head"]
        == SOMA_POSITION_WEIGHTS["Chest"]
        == SOMA_POSITION_WEIGHTS["Hips"]
        == 1
    )


if __name__ == "__main__":
    test_known_trajectory_is_fitted_without_clipping_or_timing_changes("structured")
    print("Whole-clip fit, limits, continuity and convergence passed")
