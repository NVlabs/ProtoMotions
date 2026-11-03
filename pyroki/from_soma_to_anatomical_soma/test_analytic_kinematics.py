# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run in the PyRoki environment, independently of ProtoMotions/Torch."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from analytic_kinematics import make_forward_kinematics
from solve_trajectory import build_robot


def test_analytic_fk_matches_autodiff():
    jax.config.update("jax_enable_x64", True)
    asset = (
        Path(__file__).resolve().parents[2]
        / "protomotions/data/assets/mjcf/anatomic_soma.xml"
    )
    robot = build_robot(asset)
    ids = np.arange(robot.links.num_links)
    analytic = make_forward_kinematics(robot, ids)
    q = jnp.asarray(
        np.random.default_rng(1).uniform(-0.5, 0.5, robot.joints.num_actuated_joints)
    )

    def expected(cfg):
        return robot.forward_kinematics(cfg, unroll_fk=True)

    np.testing.assert_allclose(analytic(q), expected(q), atol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(analytic)(q), jax.jacfwd(expected)(q), atol=1e-10
    )
    direction = jnp.asarray(np.random.default_rng(2).normal(size=q.shape))
    _, tangent = jax.jvp(analytic, (q,), (direction,))
    finite_difference = (
        expected(q + 1e-6 * direction) - expected(q - 1e-6 * direction)
    ) / 2e-6
    np.testing.assert_allclose(tangent, finite_difference, atol=2e-8)


if __name__ == "__main__":
    test_analytic_fk_matches_autodiff()
    print("Analytic FK, autodiff and central differences agree")
