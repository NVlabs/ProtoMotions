# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.simulator.base_simulator.config import (
    ProjectileConfig,
    SimParams,
    SimulatorConfig,
)


def test_projectiles_are_disabled_by_default_for_training():
    assert ProjectileConfig().num_projectiles == 0


def test_inference_defaults_enable_one_projectile_by_default():
    simulator_config = SimulatorConfig(
        _target_="test",
        w_last=True,
        headless=True,
        num_envs=1,
        sim=SimParams(),
        experiment_name="test",
        projectile=ProjectileConfig(num_projectiles=0),
    )

    simulator_config.apply_inference_defaults()

    assert simulator_config.projectile.num_projectiles == 1
