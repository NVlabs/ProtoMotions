# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the IsaacLab PhysX GPU buffer settings reach PhysxCfg."""

import ast
import pickle
from dataclasses import fields
from pathlib import Path

from protomotions.simulator.isaaclab.config import (
    IsaacLabPhysXParams,
    IsaacLabSimulatorConfig,
)
from protomotions.utils.config_utils import apply_overrides
from protomotions.utils.experiment_configs import ExperimentConfigs


def _physx_cfg_keywords():
    """Map each PhysxCfg keyword to the source text of the value passed."""
    source = (
        Path(__file__).parents[1] / "simulator" / "isaaclab" / "simulator.py"
    ).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PhysxCfg":
            return {
                keyword.arg: ast.get_source_segment(source, keyword.value)
                for keyword in node.keywords
            }
    raise AssertionError("PhysxCfg(...) call not found in the IsaacLab simulator")


def test_every_declared_gpu_buffer_setting_reaches_physx_cfg():
    declared = {
        f.name for f in fields(IsaacLabPhysXParams) if f.name.startswith("gpu_")
    }
    passed = _physx_cfg_keywords()

    assert "gpu_collision_stack_size" in declared
    # A setting that is declared but never passed on would silently do nothing,
    # and so would one whose keyword is fed from a different field.
    for name in declared:
        assert passed.get(name) == f"self.config.sim.physx.{name}", name


def test_collision_stack_defaults_to_isaaclab_value_and_old_pickles_load():
    assert IsaacLabPhysXParams().gpu_collision_stack_size == 2**26

    # A config pickled before the field existed has no instance value for it;
    # the class default has to fill in.
    old = IsaacLabPhysXParams()
    del old.__dict__["gpu_collision_stack_size"]
    assert pickle.loads(pickle.dumps(old)).gpu_collision_stack_size == 2**26


def test_collision_stack_can_be_overridden():
    config = IsaacLabSimulatorConfig(
        headless=True,
        num_envs=1,
        experiment_name="unit",
    )

    apply_overrides(
        ExperimentConfigs(None, config, None, None, None, None, None),
        {"simulator.sim.physx.gpu_collision_stack_size": 2**30},
    )

    assert config.sim.physx.gpu_collision_stack_size == 2**30
