# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from protomotions.components.scene_lib import ObjectOptions
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    ObjectAssetDomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.simulator import Simulator


def test_object_asset_dr_config_samples_absolute_ranges_per_asset():
    cfg = ObjectAssetDomainRandomizationConfig(
        num_buckets=4,
        static_friction_range=(0.2, 0.4),
        restitution_range=(0.0, 0.2),
        mass_range=(1.0, 2.0),
        center_of_mass_range={"x": (-0.05, 0.05), "z": (0.1, 0.2)},
    )

    samples = cfg.sample(num_samples=3, num_assets=2)

    assert samples["static_friction"].shape == (3, 2)
    assert samples["static_friction"].min() >= 0.2
    assert samples["static_friction"].max() <= 0.4
    assert samples["restitution"].shape == (3, 2)
    assert samples["mass"].shape == (3, 2)
    assert samples["center_of_mass"].shape == (3, 2, 3)
    assert samples["center_of_mass"][..., 0].min() >= -0.05
    assert samples["center_of_mass"][..., 0].max() <= 0.05
    assert torch.all(samples["center_of_mass"][..., 1] == 0.0)
    assert samples["center_of_mass"][..., 2].min() >= 0.1
    assert samples["center_of_mass"][..., 2].max() <= 0.2
    assert samples["dynamic_friction"] is None


def test_object_asset_dr_is_part_of_domain_randomization_config():
    cfg = ObjectAssetDomainRandomizationConfig(static_friction_range=(0.2, 0.4))

    assert DomainRandomizationConfig(object_assets=cfg).object_assets is cfg


def test_object_asset_dr_rejects_mass_and_density_together():
    try:
        ObjectAssetDomainRandomizationConfig(
            mass_range=(1.0, 2.0),
            density_range=(100.0, 200.0),
        )
    except ValueError as exc:
        assert "mass_range" in str(exc)
        assert "density_range" in str(exc)
    else:
        raise AssertionError("Expected mass_range and density_range to be exclusive")


def test_object_options_overrides_do_not_mutate_scene_defaults():
    base_options = ObjectOptions(
        density=1000.0,
        static_friction=0.5,
        restitution=0.0,
    )

    randomized = base_options.with_asset_property_overrides(
        {
            "mass": 2.0,
            "static_friction": 0.8,
            "dynamic_friction": 0.7,
            "restitution": 0.1,
        }
    )

    assert randomized.mass == 2.0
    assert randomized.density is None
    assert randomized.static_friction == 0.8
    assert randomized.dynamic_friction == 0.7
    assert randomized.restitution == 0.1
    assert base_options.mass is None
    assert base_options.density == 1000.0
    assert base_options.static_friction == 0.5


def test_simulator_object_asset_dr_overrides_by_bucket():
    obj = SimpleNamespace(
        first_instance_id=3,
        options=ObjectOptions(mass=1.0, static_friction=0.5, restitution=0.0),
    )
    simulator = SimpleNamespace(
        _domain_randomization={
            "object_assets": {
                "asset_id_to_column": {3: 0},
                "bucket_ids": torch.tensor([1]),
                "static_friction": torch.tensor([[0.2], [0.8]]),
                "dynamic_friction": None,
                "restitution": torch.tensor([[0.0], [0.1]]),
                "mass": torch.tensor([[1.0], [2.0]]),
                "density": None,
                "center_of_mass": torch.tensor(
                    [[[0.0, 0.0, 0.0]], [[0.1, 0.2, 0.3]]]
                ),
            }
        }
    )

    randomized = Simulator._get_object_options_for_randomized_asset(
        simulator, obj, env_id=0
    )

    assert randomized.mass == 2.0
    assert randomized.density is None
    assert abs(randomized.static_friction - 0.8) < 1e-6
    assert abs(randomized.restitution - 0.1) < 1e-6
    assert obj.options.mass == 1.0

    center_of_mass = Simulator._get_object_center_of_mass_for_randomized_asset(
        simulator, obj, env_id=0
    )
    assert torch.allclose(center_of_mass, torch.tensor([0.1, 0.2, 0.3]))
