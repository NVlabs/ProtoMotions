# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the IsaacLab robot contact-material baseline."""

import ast
from pathlib import Path

import torch

from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig


def test_isaaclab_default_robot_friction_is_one():
    config = IsaacLabSimulatorConfig(
        headless=True,
        num_envs=1,
        experiment_name="unit",
    )

    assert config.default_robot_friction == 1.0


def test_default_and_randomized_friction_share_material_tensor():
    from protomotions.simulator.isaaclab.utils.materials import (
        set_material_friction,
    )

    materials = torch.tensor(
        [
            [[0.5, 0.5, 0.1], [0.7, 0.6, 0.2]],
            [[0.4, 0.3, 0.3], [0.2, 0.1, 0.4]],
        ]
    )
    original_restitution = materials[..., 2].clone()

    set_material_friction(
        materials,
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    set_material_friction(
        materials[:, 1:2],
        static_friction=torch.tensor([[0.8], [0.9]]),
        dynamic_friction=torch.tensor([[0.6], [0.7]]),
    )

    torch.testing.assert_close(materials[:, 0, 0], torch.ones(2))
    torch.testing.assert_close(materials[:, 0, 1], torch.ones(2))
    torch.testing.assert_close(materials[:, 1, 0], torch.tensor([0.8, 0.9]))
    torch.testing.assert_close(materials[:, 1, 1], torch.tensor([0.6, 0.7]))
    torch.testing.assert_close(materials[..., 2], original_restitution)


def test_default_and_randomized_friction_use_one_material_write():
    simulator_path = (
        Path(__file__).parents[1] / "simulator" / "isaaclab" / "simulator.py"
    )
    tree = ast.parse(simulator_path.read_text())
    simulator_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IsaacLabSimulator"
    )
    apply_randomization = next(
        node
        for node in simulator_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_domain_randomization_if_needed"
    )
    friction_updates = sorted(
        node.lineno
        for node in ast.walk(apply_randomization)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set_material_friction"
    )
    material_writes = sorted(
        node.lineno
        for node in ast.walk(apply_randomization)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_material_properties"
    )

    assert len(friction_updates) == 2
    assert len(material_writes) == 1
    assert max(friction_updates) < material_writes[0]


def test_shape_discovery_uses_isaaclab3_physics_sim_view():
    simulator_path = (
        Path(__file__).parents[1] / "simulator" / "isaaclab" / "simulator.py"
    )
    tree = ast.parse(simulator_path.read_text())
    shape_view_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_rigid_body_view"
    ]

    assert len(shape_view_calls) == 1
    assert (
        ast.unparse(shape_view_calls[0].func.value)
        == "self._robot._physics_sim_view"
    )
