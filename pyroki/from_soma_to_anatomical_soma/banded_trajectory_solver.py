# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory LM with banded normal equations and cyclic-reduction PCG.

The cyclic reduction and LM/PCG scheme follow the Agile retargeter. Grouping
two consecutive frames makes three-frame factors block tridiagonal, retaining
acceleration costs without constructing a trajectory-sized dense matrix.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def transpose(x):
    return jnp.swapaxes(x, -1, -2)


def inverse(x):
    # Keep even a single block on the batched inverse path, as in Agile.
    eye = jnp.eye(x.shape[-1], dtype=x.dtype)[None]
    return jnp.linalg.inv(jnp.concatenate((x, eye)))[:-1]


def previous(x):
    return jnp.concatenate((jnp.zeros_like(x[:1]), x[:-1]))


def following(x):
    return jnp.concatenate((x[1:], jnp.zeros_like(x[:1])))


def factor_tridiagonal(diagonal, lower):
    """Factor SPD blocks; count must be a power of two, lower[0] is unused."""
    if len(diagonal) < 1 or len(diagonal) & (len(diagonal) - 1):
        raise ValueError("Block count must be a positive power of two")
    levels = []
    lower = lower.at[0].set(0)
    while len(diagonal) > 1:
        odd_inv = inverse(diagonal[1::2])
        odd_lower, even_lower = lower[1::2], lower[::2]
        next_gain = transpose(odd_lower) @ odd_inv
        prev_gain = even_lower @ previous(odd_inv)
        levels.append((odd_inv, odd_lower, even_lower, next_gain, prev_gain))
        diagonal = (
            diagonal[::2] - next_gain @ odd_lower - prev_gain @ transpose(even_lower)
        )
        lower = -prev_gain @ previous(odd_lower)
    return levels, inverse(diagonal)


def solve_tridiagonal(factor, rhs):
    """Solve matrix right-hand sides shaped (blocks, block_size, columns)."""
    levels, last_inv = factor
    odds = []
    for _, _, _, next_gain, prev_gain in levels:
        odd = rhs[1::2]
        odds.append(odd)
        rhs = rhs[::2] - next_gain @ odd - prev_gain @ previous(odd)
    result = last_inv @ rhs
    for (odd_inv, odd_lower, even_lower, _, _), odd in zip(
        reversed(levels), reversed(odds)
    ):
        odd_result = odd_inv @ (
            odd
            - odd_lower @ result
            - transpose(following(even_lower)) @ following(result)
        )
        result = jnp.stack((result, odd_result), axis=1).reshape(
            (-1,) + result.shape[1:]
        )
    return result


def normal_equations(factors, frames, dofs):
    """Accumulate factors (residual, Jacobians) into D, A[i+1,i], A[i+2,i].

    Each Jacobian tuple corresponds to consecutive frame arguments. All rows
    are already masked by the residual functions, including their derivatives.
    """
    diagonal = jnp.zeros((frames, dofs, dofs))
    lower = jnp.zeros((max(0, frames - 1), dofs, dofs))
    second = jnp.zeros((max(0, frames - 2), dofs, dofs))
    gradient = jnp.zeros((frames, dofs))
    bands = [diagonal, lower, second]
    for residual, jacobians in factors:
        count = len(residual)
        for i, ji in enumerate(jacobians):
            gradient = gradient.at[i : i + count].add(
                jnp.einsum("trd,tr->td", ji, residual)
            )
            for j in range(i + 1):
                bands[i - j] = (
                    bands[i - j].at[j : j + count].add(transpose(ji) @ jacobians[j])
                )
    return (*bands, gradient)


def band_matvec(diagonal, lower, second, x):
    out = jnp.einsum("tij,tj->ti", diagonal, x)
    for lag, band in ((1, lower), (2, second)):
        if len(band):
            out = out.at[lag:].add(jnp.einsum("tij,tj->ti", band, x[:-lag]))
            out = out.at[:-lag].add(jnp.einsum("tji,tj->ti", band, x[lag:]))
    return out


def grouped_factor(diagonal, lower, second):
    frames, dofs, _ = diagonal.shape
    groups = (frames + 1) // 2
    blocks = 1 << (groups - 1).bit_length()
    total = 2 * blocks
    d = jnp.concatenate(
        (diagonal, jnp.broadcast_to(jnp.eye(dofs), (total - frames, dofs, dofs)))
    )
    lower_band = jnp.pad(lower, ((0, total - 1 - len(lower)), (0, 0), (0, 0)))
    s = jnp.pad(second, ((0, total - 2 - len(second)), (0, 0), (0, 0)))
    block_d = jnp.zeros((blocks, 2 * dofs, 2 * dofs))
    block_d = block_d.at[:, :dofs, :dofs].set(d[::2])
    block_d = block_d.at[:, dofs:, dofs:].set(d[1::2])
    block_d = block_d.at[:, dofs:, :dofs].set(lower_band[::2])
    block_d = block_d.at[:, :dofs, dofs:].set(transpose(lower_band[::2]))
    block_l = jnp.zeros_like(block_d)
    block_l = block_l.at[1:, :dofs, :dofs].set(s[::2])
    block_l = block_l.at[1:, dofs:, dofs:].set(s[1::2])
    block_l = block_l.at[1:, :dofs, dofs:].set(lower_band[1::2])
    return factor_tridiagonal(block_d, block_l), total


class Summary(NamedTuple):
    iterations: jax.Array
    termination_criteria: jax.Array
    termination_deltas: jax.Array
    cost_history: jax.Array


def solve(initial, residuals, linearize, valid_frames, iterations):
    """LM in bounded-coordinate space; full float32 is set by the caller.

    Retain SOMA's initial damping (1), max damping (1e12), and jaxls tolerances.
    Unlike Agile's signed gradient test, use the infinity norm (jaxls contract).
    A rejected nonzero-cost-change proposal cannot establish convergence.
    """
    frames, dofs = initial.shape
    valid = jnp.arange(frames) < valid_frames

    def cost(q):
        return sum(jnp.sum(r * r) for r in residuals(q))

    value = cost(initial)
    history = jnp.full((iterations + 1,), jnp.nan).at[0].set(value)
    # q, cost, damping, previous gradient norm, eta, iteration, flags, deltas, history
    state = (
        initial,
        value,
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.01),
        jnp.array(0),
        jnp.zeros(3, bool),
        jnp.full((3,), jnp.inf),
        history,
    )

    def step(state):
        q, value, damping, prev_norm, eta, iteration, _, _, history = state
        d, lower_band, s, g = normal_equations(linearize(q), frames, dofs)
        rhs = -g
        norm = jnp.linalg.norm(rhs)
        eta_new = jnp.clip(
            jnp.minimum(0.9 * (norm / (prev_norm + 1e-7)) ** 2, eta), 1e-7, 0.01
        )
        damped = d + damping * jnp.eye(dofs)
        factor, total = grouped_factor(damped, lower_band, s)

        def precondition(x):
            padded = jnp.pad(x, ((0, total - frames), (0, 0)))
            return solve_tridiagonal(factor, padded.reshape(-1, 2 * dofs, 1)).reshape(
                total, dofs
            )[:frames]

        delta, _ = cg(
            lambda x: band_matvec(damped, lower_band, s, x),
            rhs,
            x0=jnp.zeros_like(q),
            tol=eta_new,
            maxiter=frames * dofs,
            M=precondition,
        )
        proposal = q + delta
        proposed_cost = cost(proposal)
        model_change = 2 * jnp.sum(g * delta) + jnp.sum(
            delta * band_matvec(d, lower_band, s, delta)
        )
        improvement = value - proposed_cost
        accept = (
            (model_change < 0)
            & (improvement >= 0)
            & (improvement / -model_change >= 1e-3)
        )
        finite = jnp.isfinite(proposed_cost) & jnp.all(jnp.isfinite(delta))
        accept &= finite
        abs_change = jnp.abs(proposed_cost - value)
        relative = abs_change / jnp.maximum(value, jnp.finfo(q.dtype).tiny)
        gradient = jnp.max(jnp.abs(rhs))
        param = jnp.linalg.norm(jnp.where(valid[:, None], delta, 0)) / (
            jnp.sqrt(jnp.sum(jnp.where(valid[:, None], q * q, 0))) + 1e-6
        )
        flags = jnp.array(
            [relative < 1e-5, (iteration >= 10) & (gradient < 1e-4), param < 1e-6]
        )
        flags &= finite & (accept | (abs_change == 0))
        new_cost = jnp.where(accept, proposed_cost, value)
        return (
            jnp.where(accept, proposal, q),
            new_cost,
            jnp.clip(jnp.where(accept, damping / 2, damping * 2), 1e-5, 1e12),
            jnp.where(accept, norm, prev_norm),
            jnp.where(accept, eta_new, eta),
            iteration + 1,
            flags,
            jnp.array([relative, gradient, param]),
            history.at[iteration + 1].set(new_cost),
        )

    state = jax.lax.while_loop(
        lambda st: (st[5] < iterations) & ~jnp.any(st[6]), step, state
    )
    return state[0], Summary(state[5], state[6], state[7], state[8])
