# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic tests for IsaacGym/PhysX friction combine semantics."""

from types import SimpleNamespace
import pytest
import torch

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


@pytest.mark.parametrize(
    ("friction_table", "expected_friction"),
    [
        (None, 0.75),
        (torch.tensor([[0.45]]), 0.45),
    ],
)
def test_isaacgym_variants_initialize_baseline_friction(
    friction_table, expected_friction
):
    pytest.importorskip("isaacgym")
    from protomotions.simulator.isaacgym.simulator import IsaacGymSimulator

    class FakeGym:
        def get_asset_rigid_shape_properties(self, asset):
            return asset.shape_props

        def set_asset_rigid_shape_properties(self, asset, shape_props):
            asset.shape_props = shape_props

    def fresh_asset():
        return SimpleNamespace(
            shape_props=[SimpleNamespace(friction=0.0, restitution=0.0)]
        )

    simulator = IsaacGymSimulator.__new__(IsaacGymSimulator)
    simulator.config = SimpleNamespace(default_robot_friction=0.75)
    simulator._gym = FakeGym()
    simulator._load_humanoid_asset = fresh_asset
    simulator._domain_randomization = {
        "friction": {
            "body_indices": [0],
            "static_friction": friction_table,
            "dynamic_friction": None,
            "restitution": torch.tensor([[0.2]]),
        }
    }

    assets = simulator._create_friction_randomized_assets(object())

    assert len(assets) == 1
    shape_prop = assets[0].shape_props[0]
    assert shape_prop.friction == pytest.approx(expected_friction)
    assert shape_prop.restitution == pytest.approx(0.2)
