# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import re

import pytest

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import ControlType
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.isaaclab.utils.actuator_groups import (
    build_isaaclab_joint_name_map,
    resolve_actuator_specs_for_control_type,
    single_actuator_params_by_joint,
)


@pytest.mark.parametrize("control_type", list(ControlType))
def test_resolver_uses_per_joint_specs_for_partially_specified_parameter(
    control_type,
):
    control_info = {
        "joint.1": ControlInfo(stiffness=10.0, damping=1.0, armature=0.1),
        "joint+2": ControlInfo(stiffness=5.0, damping=0.5),
    }

    specs = resolve_actuator_specs_for_control_type(control_info, control_type)

    assert len(specs) == 2
    assert specs[0].joint_names_expr == (r"joint\.1",)
    assert specs[0].params["armature"] == 0.1
    assert specs[1].joint_names_expr == (r"joint\+2",)
    assert "armature" not in specs[1].params
    expected_stiffness = 10.0 if control_type == ControlType.BUILT_IN_PD else 0.0
    expected_damping = 1.0 if control_type == ControlType.BUILT_IN_PD else 0.0
    assert specs[0].params["stiffness"] == expected_stiffness
    assert specs[0].params["damping"] == expected_damping


def test_single_actuator_params_are_per_joint_exact_regex_dicts():
    control_info = {
        "joint.1": ControlInfo(
            stiffness=10.0,
            damping=1.0,
            effort_limit=100.0,
            velocity_limit=20.0,
            armature=0.1,
        ),
        "joint+2": ControlInfo(
            stiffness=5.0,
            damping=0.5,
            effort_limit=50.0,
            velocity_limit=12.0,
            friction=0.2,
        ),
    }

    spec = single_actuator_params_by_joint(control_info)

    assert spec.name == "actuator_group_0"
    assert spec.joint_names_expr == (r"joint\.1", r"joint\+2")
    assert spec.params["stiffness"] == {r"joint\.1": 10.0, r"joint\+2": 5.0}
    assert spec.params["damping"] == {r"joint\.1": 1.0, r"joint\+2": 0.5}
    assert spec.params["armature"] == {r"joint\.1": 0.1}
    assert spec.params["friction"] == {r"joint\+2": 0.2}


def test_resolver_preserves_builtin_pd_and_zeroes_ideal_pd_gains():
    control_info = {
        "joint.1": ControlInfo(stiffness=10.0, damping=1.0),
        "joint+2": ControlInfo(stiffness=5.0, damping=0.5),
    }

    (builtin_spec,) = resolve_actuator_specs_for_control_type(
        control_info, ControlType.BUILT_IN_PD
    )
    (proportional_spec,) = resolve_actuator_specs_for_control_type(
        control_info, ControlType.PROPORTIONAL
    )
    (torque_spec,) = resolve_actuator_specs_for_control_type(
        control_info, ControlType.TORQUE
    )

    assert builtin_spec.params["stiffness"] == {
        r"joint\.1": 10.0,
        r"joint\+2": 5.0,
    }
    assert builtin_spec.params["damping"] == {
        r"joint\.1": 1.0,
        r"joint\+2": 0.5,
    }
    assert proportional_spec.params["stiffness"] == {
        r"joint\.1": 0.0,
        r"joint\+2": 0.0,
    }
    assert proportional_spec.params["damping"] == {
        r"joint\.1": 0.0,
        r"joint\+2": 0.0,
    }
    assert torque_spec.params["stiffness"] == proportional_spec.params["stiffness"]
    assert torque_spec.params["damping"] == proportional_spec.params["damping"]


@pytest.mark.parametrize("control_type", list(ControlType))
def test_resolver_returns_no_actuators_for_empty_control_info(control_type):
    assert resolve_actuator_specs_for_control_type({}, control_type) == ()


def test_single_actuator_specs_cover_public_robots():
    for robot_name in ("amp", "smpl", "smplx", "g1", "h1_2", "soma23"):
        control_info = robot_config(robot_name).control.control_info
        escaped_to_dof = {re.escape(name): name for name in control_info}

        (builtin,) = resolve_actuator_specs_for_control_type(
            control_info, ControlType.BUILT_IN_PD
        )
        (ideal,) = resolve_actuator_specs_for_control_type(
            control_info, ControlType.PROPORTIONAL
        )
        (torque,) = resolve_actuator_specs_for_control_type(
            control_info, ControlType.TORQUE
        )

        assert Counter(builtin.joint_names_expr) == Counter(escaped_to_dof.keys())
        assert ideal.params["stiffness"] == {
            expression: 0.0 for expression in escaped_to_dof
        }
        assert ideal.params["damping"] == {
            expression: 0.0 for expression in escaped_to_dof
        }
        assert resolve_actuator_specs_for_control_type(
            control_info, ControlType.BUILT_IN_PD
        ) == (builtin,)
        assert resolve_actuator_specs_for_control_type(
            control_info, ControlType.PROPORTIONAL
        ) == (ideal,)
        assert resolve_actuator_specs_for_control_type(
            control_info, ControlType.TORQUE
        ) == (torque,)

        for expression, dof_name in escaped_to_dof.items():
            matches = [name for name in control_info if re.fullmatch(expression, name)]
            assert matches == [dof_name]

        for config_key, control_key in _backend_parameter_map().items():
            if config_key in {"stiffness", "damping"}:
                continue
            expected = {
                re.escape(name): getattr(info, control_key)
                for name, info in control_info.items()
                if getattr(info, control_key) is not None
            }
            if expected:
                assert builtin.params[config_key] == expected
                assert ideal.params[config_key] == expected
            else:
                assert config_key not in builtin.params
                assert config_key not in ideal.params


def test_soma23_multiaxis_joints_use_physx_dof_names():
    config = robot_config("soma23")

    joint_names = build_isaaclab_joint_name_map(config.kinematic_info)

    assert joint_names.semantic_to_backend["Spine1_x"] == "Spine1_x:0"
    assert joint_names.semantic_to_backend["Spine1_y"] == "Spine1_x:1"
    assert joint_names.semantic_to_backend["Spine1_z"] == "Spine1_x:2"
    assert joint_names.backend_to_semantic["LeftHand_x:2"] == "LeftHand_z"


def test_single_axis_g1_joint_names_are_unchanged():
    config = robot_config("g1")

    joint_names = build_isaaclab_joint_name_map(config.kinematic_info)

    assert joint_names.semantic_to_backend == {
        name: name for name in config.kinematic_info.dof_names
    }
    assert joint_names.backend_to_semantic == {
        name: name for name in config.kinematic_info.dof_names
    }


def test_soma23_actuator_parameters_resolve_against_physx_names():
    isaaclab_string = pytest.importorskip("isaaclab.utils.string")
    config = robot_config("soma23")
    joint_names = build_isaaclab_joint_name_map(config.kinematic_info)
    control_info = {
        joint_names.semantic_to_backend[name]: info
        for name, info in config.control.control_info.items()
    }

    (spec,) = resolve_actuator_specs_for_control_type(
        control_info, ControlType.BUILT_IN_PD
    )
    _, resolved_names = isaaclab_string.resolve_matching_names(
        spec.joint_names_expr,
        list(joint_names.backend_to_semantic),
        preserve_order=True,
    )

    assert resolved_names == list(joint_names.backend_to_semantic)


@pytest.mark.parametrize(
    ("robot_name", "control_type"),
    [
        ("g1", ControlType.BUILT_IN_PD),
        ("soma23", ControlType.PROPORTIONAL),
    ],
)
def test_grouped_parameters_resolve_through_isaaclab(
    robot_name,
    control_type,
):
    isaaclab_string = pytest.importorskip("isaaclab.utils.string")
    control_info = robot_config(robot_name).control.control_info
    (spec,) = resolve_actuator_specs_for_control_type(control_info, control_type)
    joint_names = list(control_info)

    for config_key, control_key in _backend_parameter_map().items():
        if config_key not in spec.params:
            continue
        _, resolved_names, resolved_values = (
            isaaclab_string.resolve_matching_names_values(
                spec.params[config_key],
                joint_names,
                preserve_order=True,
            )
        )
        expected = {
            name: getattr(info, control_key)
            for name, info in control_info.items()
            if getattr(info, control_key) is not None
        }
        if (
            control_type != ControlType.BUILT_IN_PD
            and config_key in {"stiffness", "damping"}
        ):
            expected = {name: 0.0 for name in control_info}
        assert dict(zip(resolved_names, resolved_values)) == expected


def _backend_parameter_map():
    return {
        "stiffness": "stiffness",
        "damping": "damping",
        "armature": "armature",
        "effort_limit_sim": "effort_limit",
        "velocity_limit_sim": "velocity_limit",
        "friction": "friction",
    }
