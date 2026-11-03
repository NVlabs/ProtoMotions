# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense references for the structured trajectory algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from banded_trajectory_solver import (
    band_matvec,
    grouped_factor,
    normal_equations,
    solve,
    solve_tridiagonal,
)


@pytest.mark.parametrize("frames", [1, 2, 3, 5, 8, 9])
def test_normal_equations_and_cyclic_reduction_match_dense(frames):
    rng = np.random.default_rng(14)
    dofs = 3
    factors = []
    dense_j, dense_r = [], []
    for arity in range(1, min(frames, 3) + 1):
        count = frames - arity + 1
        residual = rng.normal(size=(count, 4)).astype(np.float32)
        jac = tuple(
            rng.normal(size=(count, 4, dofs)).astype(np.float32) for _ in range(arity)
        )
        factors.append((jnp.array(residual), tuple(map(jnp.array, jac))))
        rows = np.zeros((count, 4, frames, dofs), np.float32)
        for t in range(count):
            for i in range(arity):
                rows[t, :, t + i] = jac[i][t]
        dense_j.append(rows.reshape(-1, frames * dofs))
        dense_r.append(residual.ravel())
    jac = np.concatenate(dense_j)
    residual = np.concatenate(dense_r)
    expected = jac.T @ jac + np.eye(frames * dofs) * 0.5
    d, lower_band, s, g = normal_equations(factors, frames, dofs)
    d += 0.5 * jnp.eye(dofs)
    np.testing.assert_allclose(g.ravel(), jac.T @ residual, rtol=1e-5, atol=2e-6)
    x = rng.normal(size=(frames, dofs)).astype(np.float32)
    np.testing.assert_allclose(
        band_matvec(d, lower_band, s, x).ravel(),
        expected @ x.ravel(),
        rtol=1e-5,
        atol=1e-5,
    )
    factor, total = grouped_factor(d, lower_band, s)
    rhs = jnp.pad(g, ((0, total - frames), (0, 0))).reshape(-1, 2 * dofs, 1)
    actual = solve_tridiagonal(factor, rhs).reshape(total, dofs)[:frames]
    np.testing.assert_allclose(
        actual.ravel(),
        np.linalg.solve(expected, np.asarray(g).ravel()),
        rtol=3e-5,
        atol=3e-6,
    )


def test_masked_nonlinear_solve_preserves_padding_and_bounds():
    target = jnp.array([[0.2, -0.1], [0.3, 0.2], [-0.1, 0.4], [0.0, 0.0]])
    valid = jnp.array([1.0, 1.0, 1.0, 0.0])

    def residual(q, t, v):
        return (jnp.tanh(q) - t) * v

    def residuals(q):
        return (jax.vmap(residual)(q, target, valid),)

    def linearize(q):
        return ((residuals(q)[0], (jax.vmap(jax.jacfwd(residual))(q, target, valid),)),)

    q, summary = jax.jit(lambda q: solve(q, residuals, linearize, 3, 80))(
        jnp.zeros_like(target)
    )
    np.testing.assert_allclose(jnp.tanh(q), target, atol=1e-4)
    assert np.any(summary.termination_criteria)
    assert np.all(np.isfinite(q))
    np.testing.assert_array_equal(q[-1], 0)


def test_three_frame_jacobians_and_mask_match_dense_objective():
    q = jnp.arange(10, dtype=jnp.float32).reshape(5, 2) / 10
    valid = jnp.array([1.0, 0.0, 0.0])  # Only the first triple is real.

    def residual(a, b, c, mask):
        return (jnp.tanh(c) - 2 * jnp.tanh(b) + jnp.tanh(a)) * mask

    def all_residuals(q):
        return jax.vmap(residual)(q[:-2], q[1:-1], q[2:], valid).ravel()

    jacobians = jax.vmap(jax.jacfwd(residual, argnums=(0, 1, 2)))(
        q[:-2], q[1:-1], q[2:], valid
    )
    r = all_residuals(q)
    d, lower_band, s, g = normal_equations(((r.reshape(3, 2), jacobians),), 5, 2)
    jac = jax.jacfwd(all_residuals)(q).reshape(6, 10)
    x = jnp.arange(10, dtype=q.dtype).reshape(5, 2) / 7
    np.testing.assert_allclose(g.ravel(), jac.T @ r, atol=1e-6)
    np.testing.assert_allclose(
        band_matvec(d, lower_band, s, x).ravel(), jac.T @ jac @ x.ravel(), atol=1e-6
    )
    eps = 0.001
    finite_difference = (all_residuals(q + eps * x) - all_residuals(q - eps * x)) / (
        2 * eps
    )
    np.testing.assert_allclose(jac @ x.ravel(), finite_difference, atol=1e-4)
    np.testing.assert_array_equal(g[3:], 0)


def test_nonfinite_proposal_does_not_report_convergence():
    def residuals(q):
        return (q * jnp.nan,)

    def linearize(q):
        return ((residuals(q)[0], (jnp.ones((1, 1, 1)),)),)

    _, summary = solve(jnp.zeros((1, 1)), residuals, linearize, 1, 2)
    assert not np.any(summary.termination_criteria)
    assert int(summary.iterations) == 2
