# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the asset-ordered property arrays passed to Isaac Gym."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import ControlType


class PropertyGym:
    """Record property writes at the Gym boundary without initializing physics."""

    def __init__(self):
        fields = (
            "driveMode",
            "stiffness",
            "damping",
            "armature",
            "friction",
            "effort",
            "velocity",
        )
        self.defaults = np.zeros(
            3,
            dtype=[
                (name, np.int32 if name == "driveMode" else np.float32)
                for name in fields
            ],
        )
        for index, name in enumerate(fields):
            self.defaults[name] = np.array([100, 200, 300]) + index
        self.writes = []

    def create_actor(self, *args):
        return "robot"

    def enable_actor_dof_force_sensors(self, env, actor):
        assert (env, actor) == ("env", "robot")

    def get_actor_dof_properties(self, env, actor):
        assert (env, actor) == ("env", "robot")
        return self.defaults.copy()

    def set_actor_dof_properties(self, env, actor, properties):
        assert (env, actor) == ("env", "robot")
        self.writes.append(properties.copy())


def build_env(backend):
    """Run the production build method without importing proprietary bindings."""
    path = Path(__file__).parents[1] / "simulator/isaacgym/simulator.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IsaacGymSimulator"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_env"
    )
    namespace = {
        "ControlType": ControlType,
        "gymapi": SimpleNamespace(
            Transform=SimpleNamespace,
            Vec3=lambda *values: values,
            Quat=lambda *values: values,
            DOF_MODE_POS=1,
            DOF_MODE_EFFORT=3,
        ),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["_build_env"](backend, 0, "env", "asset")


@pytest.mark.parametrize("control_type", list(ControlType))
@pytest.mark.parametrize("unset_optional", [False, True])
def test_dof_properties_follow_asset_order(control_type, unset_optional):
    """Every asset slot gets its joint's properties and retains unspecified limits."""
    common_names = ["hip", "knee", "ankle"]
    infos = {
        name: ControlInfo(
            stiffness=11.0 + index,
            damping=0.11 + index * 0.01,
            armature=0.01 + index * 0.01,
            friction=0.21 + index * 0.01,
            effort_limit=101.0 + index,
            velocity_limit=201.0 + index,
        )
        for index, name in enumerate(common_names)
    }
    # Dictionary order differs from both common and asset order.
    infos = {name: infos[name] for name in ("knee", "ankle", "hip")}
    if unset_optional:
        infos["knee"].friction = None
        infos["knee"].effort_limit = None
        infos["knee"].velocity_limit = None
    before = copy.deepcopy(infos)
    gym = PropertyGym()
    backend = SimpleNamespace(
        _gym=gym,
        _num_bodies=0,
        _num_dof=3,
        _dof_names=["ankle", "hip", "knee"],
        _humanoid_handles=[],
        robot_config=SimpleNamespace(
            asset=SimpleNamespace(self_collisions=True),
            kinematic_info=SimpleNamespace(dof_names=common_names),
            control=SimpleNamespace(control_info=infos),
        ),
        control_type=control_type,
        scene_lib=SimpleNamespace(num_objects_per_scene=0),
        _apply_com_domain_randomization_to_actor=lambda *args: None,
        _apply_body_mass_domain_randomization_to_actor=lambda *args: None,
        _build_projectile_actors=lambda *args: None,
        _build_markers=lambda *args: None,
    )
    build_env(backend)

    assert len(gym.writes) == 1
    actual = gym.writes[0]
    native = control_type == ControlType.BUILT_IN_PD
    np.testing.assert_array_equal(actual["driveMode"], 1 if native else 3)
    np.testing.assert_allclose(actual["stiffness"], [13, 11, 12] if native else 0)
    np.testing.assert_allclose(actual["damping"], [0.13, 0.11, 0.12] if native else 0)
    np.testing.assert_allclose(actual["armature"], [0.03, 0.01, 0.02])
    for name, expected in (
        ("effort", [103, 101, 102]),
        ("velocity", [203, 201, 202]),
        ("friction", [0.23, 0.21, 0.22]),
    ):
        if unset_optional:
            expected[2] = gym.defaults[name][2]
        np.testing.assert_allclose(actual[name], expected)
    assert infos == before
    assert common_names == ["hip", "knee", "ankle"]
    assert backend._humanoid_handles == ["robot"]
