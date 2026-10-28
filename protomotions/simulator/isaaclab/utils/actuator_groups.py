# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import re
from typing import Dict, Mapping, Tuple

from protomotions.components.pose_lib import ControlInfo, KinematicInfo
from protomotions.robot_configs.base import ControlType


_ACTUATOR_PARAM_FIELDS = (
    ("stiffness", "stiffness"),
    ("damping", "damping"),
    ("armature", "armature"),
    ("effort_limit_sim", "effort_limit"),
    ("velocity_limit_sim", "velocity_limit"),
    ("friction", "friction"),
)


@dataclass(frozen=True)
class SingleActuatorSpec:
    name: str
    joint_names_expr: Tuple[str, ...]
    params: dict


@dataclass(frozen=True)
class IsaacLabJointNameMap:
    semantic_to_backend: Mapping[str, str]
    backend_to_semantic: Mapping[str, str]


def build_isaaclab_joint_name_map(
    kinematic_info: KinematicInfo,
) -> IsaacLabJointNameMap:
    """Map MJCF joint names to IsaacLab 3 PhysX multi-axis DOF names."""
    semantic_to_backend = {}
    dof_offset = 0
    for body_index in range(kinematic_info.num_bodies):
        num_body_dofs = len(kinematic_info.hinge_axes_map.get(body_index, ()))
        semantic_names = kinematic_info.dof_names[
            dof_offset : dof_offset + num_body_dofs
        ]
        dof_offset += num_body_dofs

        if num_body_dofs <= 1:
            backend_names = semantic_names
        else:
            backend_names = [
                f"{semantic_names[0]}:{axis_index}"
                for axis_index in range(num_body_dofs)
            ]
        semantic_to_backend.update(zip(semantic_names, backend_names))

    if dof_offset != len(kinematic_info.dof_names):
        raise ValueError(
            "Kinematic body DOF counts do not cover all semantic joint names"
        )

    backend_to_semantic = {
        backend: semantic for semantic, backend in semantic_to_backend.items()
    }
    if len(backend_to_semantic) != len(semantic_to_backend):
        raise ValueError("IsaacLab joint-name mapping is not one-to-one")

    return IsaacLabJointNameMap(
        semantic_to_backend=semantic_to_backend,
        backend_to_semantic=backend_to_semantic,
    )


def single_actuator_params_by_joint(
    control_info_by_dof: Dict[str, ControlInfo],
    *,
    zero_stiffness_and_damping: bool = False,
) -> SingleActuatorSpec:
    joint_names_expr = tuple(re.escape(dof_name) for dof_name in control_info_by_dof)
    params = {}
    for config_key, control_key in _ACTUATOR_PARAM_FIELDS:
        if zero_stiffness_and_damping and config_key in {"stiffness", "damping"}:
            params[config_key] = {
                re.escape(dof_name): 0.0 for dof_name in control_info_by_dof
            }
            continue
        if control_key == "friction":
            # Friction is asset-owned: None means "use the engine's value", so a
            # DOF that leaves it unset is simply omitted (require() is a TypeError).
            values = {
                re.escape(dof_name): control_info.friction
                for dof_name, control_info in control_info_by_dof.items()
                if control_info.friction is not None
            }
            if values:
                params[config_key] = values
            continue
        params[config_key] = {
            re.escape(dof_name): control_info.require(control_key)
            for dof_name, control_info in control_info_by_dof.items()
        }
    return SingleActuatorSpec(
        name="actuator_group_0",
        joint_names_expr=joint_names_expr,
        params=params,
    )


def resolve_actuator_specs_for_control_type(
    control_info_by_dof: Dict[str, ControlInfo],
    control_type: ControlType,
) -> Tuple[SingleActuatorSpec, ...]:
    if not control_info_by_dof:
        return ()

    zero_stiffness_and_damping = control_type != ControlType.BUILT_IN_PD
    collapsed_spec = single_actuator_params_by_joint(
        control_info_by_dof,
        zero_stiffness_and_damping=zero_stiffness_and_damping,
    )
    if all(
        len(values) == len(control_info_by_dof)
        for values in collapsed_spec.params.values()
    ):
        return (collapsed_spec,)

    specs = []
    for index, (dof_name, control_info) in enumerate(control_info_by_dof.items()):
        params = {}
        for config_key, control_key in _ACTUATOR_PARAM_FIELDS:
            if config_key == "friction":
                # Asset-owned: omit a DOF that leaves friction unset.
                friction = control_info.friction
                if friction is not None:
                    params[config_key] = friction
                continue
            if zero_stiffness_and_damping and config_key in {"stiffness", "damping"}:
                params[config_key] = 0.0
            else:
                params[config_key] = control_info.require(control_key)
        specs.append(
            SingleActuatorSpec(
                name=f"actuator_group_{index}",
                joint_names_expr=(re.escape(dof_name),),
                params=params,
            )
        )
    return tuple(specs)
