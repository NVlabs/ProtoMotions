# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for simulator factory dispatch and material conversion utilities."""

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

import protomotions.simulator.factory as simulator_factory
import protomotions.simulator.base_simulator.utils as simulator_utils
from protomotions.components.terrains.config import (
    CombineMode,
    TerrainConfig,
    TerrainSimConfig,
)
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    SimParams,
    SimulatorConfig,
)
from protomotions.simulator.base_simulator.simulator_state import StateConversion
from protomotions.simulator.base_simulator.utils import (
    _compute_effective_friction_range,
    _convert_material_to_combine_mode,
    _friction_ranges_match,
    build_motion_data,
    convert_friction_for_combine_mode,
    convert_friction_for_simulator,
    get_simulator_friction_combine_mode,
)


@dataclass
class _FakeSimParams:
    fps: int = 120
    decimation: int = 2


@dataclass
class _FakeSimulatorConfig:
    sim: _FakeSimParams = field(default_factory=_FakeSimParams)
    headless: bool = False
    num_envs: int = 1
    experiment_name: str = "default"
    _target_: str = "fake.Target"
    w_last: bool = False
    copied_default: list = field(default_factory=lambda: ["fresh"])


def _install_fake_sim_config(monkeypatch, simulator_name, class_name):
    module = types.ModuleType(f"protomotions.simulator.{simulator_name}.config")
    setattr(module, class_name, _FakeSimulatorConfig)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_simulator_factory_dispatches_config_classes_and_builds_config(monkeypatch):
    for simulator_name, class_name in [
        ("isaacgym", "IsaacGymSimulatorConfig"),
        ("isaaclab", "IsaacLabSimulatorConfig"),
        ("newton", "NewtonSimulatorConfig"),
        ("genesis", "GenesisSimulatorConfig"),
        ("mujoco", "MujocoSimulatorConfig"),
    ]:
        _install_fake_sim_config(monkeypatch, simulator_name, class_name)
        assert simulator_factory.get_simulator_config_class(simulator_name) is _FakeSimulatorConfig

    robot_config = SimpleNamespace(
        simulation_params=SimpleNamespace(newton=_FakeSimParams(fps=240, decimation=4))
    )
    config = simulator_factory.simulator_config(
        "newton",
        robot_config=robot_config,
        headless=True,
        num_envs=8,
        experiment_name="unit",
    )

    assert isinstance(config, _FakeSimulatorConfig)
    assert config.sim.fps == 240
    assert config.headless is True
    assert config.num_envs == 8
    assert config.experiment_name == "unit"

    with pytest.raises(ValueError, match="Unsupported simulator"):
        simulator_factory.get_simulator_config_class("unknown")


def test_update_simulator_config_for_test_adds_missing_params_and_new_fields(monkeypatch):
    monkeypatch.setattr(
        simulator_factory,
        "get_simulator_config_class",
        lambda simulator_name: _FakeSimulatorConfig,
    )
    current = SimpleNamespace(
        _target_="old.Target",
        w_last=True,
        sim=_FakeSimParams(fps=30, decimation=1),
    )
    robot_config = SimpleNamespace(simulation_params=SimpleNamespace())

    updated = simulator_factory.update_simulator_config_for_test(
        current,
        "newton",
        robot_config,
    )

    assert updated is current
    assert updated._target_ == "fake.Target"
    assert updated.w_last is False
    assert updated.sim.fps == 120
    assert updated.copied_default == ["fresh"]
    assert robot_config.simulation_params.newton.fps == 120

    robot_config.simulation_params.mujoco = None
    robot_config.simulation_params.__dataclass_fields__ = {
        "newton": object(),
        "mujoco": object(),
    }
    with pytest.raises(ValueError, match="does not have simulation_params"):
        simulator_factory.update_simulator_config_for_test(current, "mujoco", robot_config)


def test_build_motion_data_maps_recorded_fields_and_fills_missing_values():
    recorded_motion = {
        "gts": [torch.ones(2, 3), torch.ones(2, 3) * 2.0],
        "grs": [torch.ones(2, 4), torch.ones(2, 4)],
        "contacts": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        "ignored": [torch.tensor([99.0])],
    }

    motion_data = build_motion_data(recorded_motion, fps=30, num_dof=3)

    assert motion_data["fps"] == 30
    assert motion_data["state_conversion"] is StateConversion.COMMON
    assert motion_data["rigid_body_pos"].shape == (2, 2, 3)
    assert motion_data["dof_pos"].shape == (2, 3)
    assert motion_data["dof_vel"].shape == (2, 3)
    assert torch.equal(motion_data["rigid_body_vel"], torch.zeros(2, 2, 3))
    assert motion_data["rigid_body_contacts"].dtype is torch.bool
    assert "ignored" not in motion_data

    without_contacts = build_motion_data(
        {
            "gts": [torch.zeros(1, 3)],
            "grs": [torch.zeros(1, 4)],
            "gvs": [torch.ones(1, 3)],
            "gavs": [torch.ones(1, 3)],
        },
        fps=60,
        num_dof=0,
    )
    assert torch.equal(without_contacts["rigid_body_contacts"], torch.zeros(1, 1, dtype=torch.bool))


def test_friction_range_helpers_cover_combine_modes_and_validation():
    assert CombineMode.from_str("AVERAGE") is CombineMode.AVERAGE
    with pytest.raises(ValueError, match="not a valid CombineMode"):
        CombineMode.from_str("invalid")

    assert _friction_ranges_match((1.0, 2.0), (1.0, 2.0 + 1e-7), 1e-6)
    assert not _friction_ranges_match((1.0, 2.0), (1.0, 2.1), 1e-6)
    assert _compute_effective_friction_range((0.5, 1.5), 1.0, CombineMode.AVERAGE) == (0.75, 1.25)
    assert _compute_effective_friction_range((0.5, 1.5), 1.0, CombineMode.MIN) == (0.5, 1.0)
    assert _compute_effective_friction_range((0.5, 1.5), 1.0, CombineMode.MAX) == (1.0, 1.5)
    assert _compute_effective_friction_range((0.5, 1.5), 2.0, CombineMode.MULTIPLY) == (1.0, 3.0)

    with pytest.raises(ValueError, match="Unknown combine mode"):
        _compute_effective_friction_range((0.5, 1.5), 1.0, object())

    with pytest.raises(ValueError, match="Unsupported target mode"):
        _convert_material_to_combine_mode(
            TerrainSimConfig(),
            None,
            (0.5, 1.0),
            (0.5, 1.0),
            (0.0, 0.1),
            SimpleNamespace(value="min"),
        )


def test_convert_friction_for_combine_mode_preserves_effective_ranges():
    terrain = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=0.8,
        restitution=0.2,
        combine_mode=CombineMode.AVERAGE,
    )
    friction = FrictionDomainRandomizationConfig(
        body_indices=[0],
        static_friction_range=(0.5, 1.5),
        dynamic_friction_range=(0.4, 1.2),
        restitution_range=(0.0, 0.4),
    )

    same_terrain, same_friction = convert_friction_for_combine_mode(
        terrain,
        friction,
        CombineMode.AVERAGE,
    )
    assert same_terrain is terrain
    assert same_friction is friction

    max_terrain, max_friction = convert_friction_for_combine_mode(
        terrain,
        friction,
        CombineMode.MAX,
    )
    assert max_terrain.combine_mode is CombineMode.MAX
    assert max_terrain.static_friction == pytest.approx(0.75)
    assert max_friction.static_friction_range == pytest.approx((0.75, 1.25))

    average_terrain, average_friction = convert_friction_for_combine_mode(
        max_terrain,
        max_friction,
        CombineMode.AVERAGE,
    )
    assert average_terrain.combine_mode is CombineMode.AVERAGE
    assert average_friction.static_friction_range == pytest.approx((0.75, 1.75))

    no_dr_terrain, no_dr_friction = convert_friction_for_combine_mode(
        terrain,
        None,
        CombineMode.MAX,
        default_robot_friction=1.2,
        default_robot_restitution=0.1,
    )
    assert no_dr_terrain.combine_mode is CombineMode.MAX
    assert no_dr_friction is None


def test_convert_friction_for_combine_mode_rejects_mismatched_conversion(monkeypatch):
    terrain = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=0.8,
        restitution=0.2,
        combine_mode=CombineMode.AVERAGE,
    )
    friction = FrictionDomainRandomizationConfig(
        body_indices=[0],
        static_friction_range=(0.5, 1.5),
        dynamic_friction_range=(0.4, 1.2),
        restitution_range=(0.0, 0.4),
    )

    def _bad_convert(
        terrain_sim_config,
        friction_dr_config,
        expected_static,
        expected_dynamic,
        expected_restitution,
        target_mode,
    ):
        return TerrainSimConfig(
            static_friction=9.0,
            dynamic_friction=9.0,
            restitution=9.0,
            combine_mode=target_mode,
        ), friction_dr_config

    monkeypatch.setattr(
        simulator_utils,
        "_convert_material_to_combine_mode",
        _bad_convert,
    )

    with pytest.raises(ValueError, match="effective range mismatch"):
        simulator_utils.convert_friction_for_combine_mode(
            terrain,
            friction,
            CombineMode.MAX,
        )


def test_convert_friction_for_simulator_handles_required_and_optional_modes():
    assert get_simulator_friction_combine_mode("newton") is CombineMode.MAX
    assert get_simulator_friction_combine_mode("isaacgym") is CombineMode.AVERAGE
    assert get_simulator_friction_combine_mode("mujoco") is None

    terrain = TerrainConfig(
        sim_config=TerrainSimConfig(static_friction=1.0, combine_mode=CombineMode.AVERAGE)
    )
    simulator = SimulatorConfig(
        _target_="protomotions.simulator.newton.simulator.NewtonSimulator",
        w_last=True,
        headless=True,
        num_envs=2,
        sim=SimParams(),
        experiment_name="unit",
        domain_randomization=DomainRandomizationConfig(
            friction=FrictionDomainRandomizationConfig(body_indices=[0])
        ),
    )

    adjusted_terrain, adjusted_simulator = convert_friction_for_simulator(
        terrain,
        simulator,
    )
    assert adjusted_terrain.sim_config.combine_mode is CombineMode.MAX
    assert adjusted_simulator.domain_randomization.friction.static_friction_range[0] == pytest.approx(0.75)

    unchanged_terrain, unchanged_simulator = convert_friction_for_simulator(
        terrain,
        SimulatorConfig(
            _target_="protomotions.simulator.mujoco.simulator.MujocoSimulator",
            w_last=True,
            headless=True,
            num_envs=2,
            sim=SimParams(),
            experiment_name="unit",
        ),
    )
    assert unchanged_terrain is terrain
    assert unchanged_simulator._target_.endswith("MujocoSimulator")

    terrain_without_sim = TerrainConfig(sim_config=None)
    assert convert_friction_for_simulator(terrain_without_sim, simulator)[0] is terrain_without_sim
    assert convert_friction_for_simulator(None, simulator)[0] is None
    already_max = TerrainConfig(sim_config=TerrainSimConfig(combine_mode=CombineMode.MAX))
    assert convert_friction_for_simulator(already_max, simulator)[0] is already_max
