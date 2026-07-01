# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic tests for IsaacGym/PhysX friction combine semantics."""

import pytest

from protomotions.components.terrains.config import (
    CombineMode,
    TerrainConfig,
    TerrainSimConfig,
)
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.utils import (
    convert_friction_for_simulator,
    get_simulator_friction_combine_mode,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimulatorConfig
from protomotions.simulator.newton.config import NewtonSimulatorConfig


def _simulator_name(simulator_config):
    return simulator_config._target_.split(".")[-3]


def test_isaacgym_and_newton_configs_resolve_fixed_friction_modes():
    isaacgym_config = IsaacGymSimulatorConfig(
        headless=True,
        num_envs=1,
        experiment_name="unit",
    )
    newton_config = NewtonSimulatorConfig(
        headless=True,
        num_envs=1,
        experiment_name="unit",
    )

    assert _simulator_name(isaacgym_config) == "isaacgym"
    assert _simulator_name(newton_config) == "newton"
    assert (
        get_simulator_friction_combine_mode(_simulator_name(isaacgym_config))
        is CombineMode.AVERAGE
    )
    assert (
        get_simulator_friction_combine_mode(_simulator_name(newton_config))
        is CombineMode.MAX
    )


def test_physx_average_friction_config_converts_to_newton_max_config():
    terrain = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=0.4,
            dynamic_friction=0.2,
            restitution=0.1,
            combine_mode=CombineMode.AVERAGE,
        )
    )
    friction = FrictionDomainRandomizationConfig(
        body_indices=[0, 3],
        static_friction_range=(0.6, 1.6),
        dynamic_friction_range=(0.2, 1.0),
        restitution_range=(0.0, 0.4),
    )
    isaacgym_config = IsaacGymSimulatorConfig(
        headless=True,
        num_envs=2,
        experiment_name="unit",
        domain_randomization=DomainRandomizationConfig(friction=friction),
    )
    newton_config = NewtonSimulatorConfig(
        headless=True,
        num_envs=2,
        experiment_name="unit",
        domain_randomization=DomainRandomizationConfig(friction=friction),
    )

    isaacgym_terrain, isaacgym_simulator = convert_friction_for_simulator(
        terrain,
        isaacgym_config,
    )
    newton_terrain, newton_simulator = convert_friction_for_simulator(
        terrain,
        newton_config,
    )

    assert isaacgym_terrain is terrain
    assert isaacgym_simulator is isaacgym_config
    assert terrain.sim_config.combine_mode is CombineMode.AVERAGE
    assert friction.static_friction_range == (0.6, 1.6)

    assert newton_terrain is not terrain
    assert newton_terrain.sim_config.combine_mode is CombineMode.MAX
    assert newton_terrain.sim_config.static_friction == pytest.approx(0.5)
    assert newton_terrain.sim_config.dynamic_friction == pytest.approx(0.2)
    assert newton_terrain.sim_config.restitution == pytest.approx(0.05)

    assert newton_simulator is not newton_config
    assert newton_simulator.domain_randomization is not None
    newton_friction = newton_simulator.domain_randomization.friction
    assert newton_friction is not None
    assert newton_friction is not friction
    assert newton_friction.body_indices == [0, 3]
    assert newton_friction.static_friction_range == pytest.approx((0.5, 1.0))
    assert newton_friction.dynamic_friction_range == pytest.approx((0.2, 0.6))
    assert newton_friction.restitution_range == pytest.approx((0.05, 0.25))
