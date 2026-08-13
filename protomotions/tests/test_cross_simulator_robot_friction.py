# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the shared character friction configuration."""

from types import SimpleNamespace

import numpy as np
import pytest

from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
from protomotions.components.terrains.config import (
    CombineMode,
    TerrainConfig,
    TerrainSimConfig,
)
from protomotions.simulator.base_simulator.config import SimParams


def test_shared_simulator_config_defaults_character_friction_to_one():
    config = SimulatorConfig(
        _target_="unit.Target",
        w_last=True,
        headless=True,
        num_envs=1,
        sim=SimParams(),
        experiment_name="unit",
    )

    assert config.default_robot_friction == 1.0


def test_friction_conversion_uses_configured_robot_friction_without_dr():
    terrain = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1.0,
            dynamic_friction=1.0,
            combine_mode=CombineMode.AVERAGE,
        )
    )
    simulator = SimulatorConfig(
        _target_="protomotions.simulator.newton.simulator.NewtonSimulator",
        w_last=True,
        headless=True,
        num_envs=1,
        sim=SimParams(),
        experiment_name="unit",
        default_robot_friction=0.5,
    )

    adjusted_terrain, adjusted_simulator = convert_friction_for_simulator(
        terrain, simulator
    )

    assert adjusted_simulator.default_robot_friction == 0.5
    assert adjusted_terrain.sim_config.combine_mode is CombineMode.MAX
    assert adjusted_terrain.sim_config.static_friction == pytest.approx(0.75)


def test_mujoco_friction_targets_character_geoms_only():
    pytest.importorskip("mujoco")
    from protomotions.simulator.mujoco.simulator import MujocoSimulator

    class FakeModel:
        geom_bodyid = np.array([0, 1, 2, 3])
        geom_friction = np.array(
            [
                [0.2, 0.01, 0.02],
                [0.3, 0.03, 0.04],
                [0.4, 0.05, 0.06],
                [0.5, 0.07, 0.08],
            ],
            dtype=np.float64,
        )

        def body(self, name):
            return SimpleNamespace(id={"pelvis": 1, "left_foot": 2}[name])

    simulator = object.__new__(MujocoSimulator)
    simulator.model = FakeModel()
    simulator._body_names = ["pelvis", "left_foot"]
    simulator.config = SimpleNamespace(default_robot_friction=0.5)

    simulator._set_robot_geom_friction()

    np.testing.assert_allclose(
        simulator.model.geom_friction[:, 0], [0.2, 0.5, 0.5, 0.5]
    )
    np.testing.assert_allclose(
        simulator.model.geom_friction[:, 1:],
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]],
    )
