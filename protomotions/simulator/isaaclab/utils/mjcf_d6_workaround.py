# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary IsaacLab MJCF D6 conversion compatibility fix.

IsaacLab's MJCF importer collapses multiple single-axis joints between the
same bodies into one PhysX D6 joint.  Until the upstream importer carries the
fix, ProtoMotions repairs only the two affected details after that collapse:
the shared joint frame and MuJoCo spring gains.  The importer remains the
owner of the conversion and all other MJCF behavior.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


_CONVERSION_MODULE = "isaacsim.asset.importer.utils.impl.mjc_to_physx_conversion_utils"
_PATCH_MARKER = "_protomotions_mjcf_d6_workaround"


@dataclass(frozen=True)
class _SourceAxisState:
    path: Any
    token: str
    local_rot0: Any
    local_rot1: Any
    drive_state: dict[str, Any]


def _authored_value(attribute: Any) -> Any:
    if not attribute or not attribute.IsValid() or not attribute.HasAuthoredValue():
        return None
    return attribute.Get()


def _axis_token(joint: Any, usd_physics: Any) -> str | None:
    axis = _authored_value(joint.GetAttribute("physics:axis"))
    if not axis:
        return None
    axis_tokens = {
        "X": "rotX" if joint.IsA(usd_physics.RevoluteJoint) else "transX",
        "Y": "rotY" if joint.IsA(usd_physics.RevoluteJoint) else "transY",
        "Z": "rotZ" if joint.IsA(usd_physics.RevoluteJoint) else "transZ",
    }
    if joint.IsA(usd_physics.RevoluteJoint) or joint.IsA(usd_physics.PrismaticJoint):
        return axis_tokens.get(str(axis).upper())
    return None


def _source_axis_states(joints: list[Any], usd_physics: Any) -> list[_SourceAxisState]:
    states: list[_SourceAxisState] = []
    for joint in joints:
        token = _axis_token(joint, usd_physics)
        if token is None:
            continue
        joint_api = usd_physics.Joint(joint)
        drive_state: dict[str, Any] = {}
        drive_instance = "angular" if joint.IsA(usd_physics.RevoluteJoint) else "linear"
        if joint.HasAPI(usd_physics.DriveAPI, drive_instance):
            source_drive = usd_physics.DriveAPI(joint, drive_instance)
            source_damping = _authored_value(source_drive.GetDampingAttr())
            source_stiffness = _authored_value(source_drive.GetStiffnessAttr())
            source_max_force = _authored_value(source_drive.GetMaxForceAttr())
            source_target_position = _authored_value(
                source_drive.GetTargetPositionAttr()
            )
            source_target_velocity = _authored_value(
                source_drive.GetTargetVelocityAttr()
            )
            source_type = _authored_value(source_drive.GetTypeAttr())
            if source_damping is not None:
                drive_state["damping"] = source_damping
            if source_stiffness is not None:
                drive_state["stiffness"] = source_stiffness
            if source_max_force is not None:
                drive_state["max_force"] = source_max_force
            if source_target_position is not None:
                drive_state["target_position"] = source_target_position
            if source_target_velocity is not None:
                drive_state["target_velocity"] = source_target_velocity
            if source_type is not None:
                drive_state["type"] = source_type

        mjc_stiffness = _authored_value(joint.GetAttribute("mjc:stiffness"))
        mjc_damping = _authored_value(joint.GetAttribute("mjc:damping"))
        if mjc_stiffness is not None:
            drive_state["stiffness"] = mjc_stiffness
        if mjc_damping is not None:
            drive_state["damping"] = mjc_damping
        states.append(
            _SourceAxisState(
                path=joint.GetPath(),
                token=token,
                local_rot0=_authored_value(joint_api.GetLocalRot0Attr()),
                local_rot1=_authored_value(joint_api.GetLocalRot1Attr()),
                drive_state=drive_state,
            )
        )
    return states


def _repair_d6_group(
    stage: Any,
    states: list[_SourceAxisState],
    source_joint_remap: dict[Any, Any],
    usd_physics: Any,
) -> None:
    # The importer omits duplicate-axis joints from source_joint_remap.  Do
    # not let metadata from one of those dropped joints overwrite the axis
    # that the importer retained on the D6.
    states = [state for state in states if state.path in source_joint_remap]
    if not states:
        return

    d6_path = source_joint_remap.get(states[0].path)
    if d6_path is None:
        return
    d6_prim = stage.GetPrimAtPath(d6_path)
    d6_joint = usd_physics.Joint(d6_prim)

    rotation_source = max(
        states,
        key=lambda candidate: sum(
            candidate.local_rot0 == other.local_rot0
            and candidate.local_rot1 == other.local_rot1
            for other in states
        ),
    )
    if rotation_source.local_rot0 is not None:
        d6_joint.CreateLocalRot0Attr().Set(rotation_source.local_rot0)
    if rotation_source.local_rot1 is not None:
        d6_joint.CreateLocalRot1Attr().Set(rotation_source.local_rot1)

    used_tokens = {state.token for state in states}
    for state in states:
        if not state.drive_state:
            continue
        drive = usd_physics.DriveAPI.Apply(d6_prim, state.token)
        if "damping" in state.drive_state:
            drive.CreateDampingAttr().Set(state.drive_state["damping"])
        if "stiffness" in state.drive_state:
            drive.CreateStiffnessAttr().Set(state.drive_state["stiffness"])
        if "max_force" in state.drive_state:
            drive.CreateMaxForceAttr().Set(state.drive_state["max_force"])
        if "target_position" in state.drive_state:
            drive.CreateTargetPositionAttr().Set(state.drive_state["target_position"])
        if "target_velocity" in state.drive_state:
            drive.CreateTargetVelocityAttr().Set(state.drive_state["target_velocity"])
        if "type" in state.drive_state:
            drive.CreateTypeAttr().Set(state.drive_state["type"])

    for drive_instance in ("angular", "linear"):
        if d6_prim.HasAPI(usd_physics.DriveAPI, drive_instance):
            d6_prim.RemoveAPI(usd_physics.DriveAPI, drive_instance)

    edit_layer = stage.GetEditTarget().GetLayer()
    edit_prim_spec = edit_layer.GetPrimAtPath(d6_path)
    if edit_prim_spec is not None:
        for property_name in (
            "physics:axis",
            "physics:lowerLimit",
            "physics:upperLimit",
        ):
            property_spec = edit_prim_spec.attributes.get(property_name)
            if property_spec is not None:
                edit_prim_spec.RemoveProperty(property_spec)
            attribute = d6_prim.GetAttribute(property_name)
            if attribute and attribute.IsValid() and attribute.HasAuthoredValueOpinion():
                attribute.Block()

    d6_tokens = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")
    for token in d6_tokens:
        if token in used_tokens:
            continue
        limit = usd_physics.LimitAPI.Apply(d6_prim, token)
        limit.CreateLowAttr().Set(1.0)
        limit.CreateHighAttr().Set(-1.0)


def _has_patch_marker(function: Any) -> bool:
    return callable(function) and vars(function).get(_PATCH_MARKER) is True


def install_isaaclab_mjcf_d6_workaround() -> None:
    """Install the narrow D6 repair before an IsaacLab MJCF conversion."""
    conversion_module = importlib.import_module(_CONVERSION_MODULE)
    if _has_patch_marker(
        conversion_module._convert_overconstrained_group_to_d6
    ) and _has_patch_marker(
        conversion_module.combine_overconstrained_joints_in_physx_layer
    ):
        return

    original = conversion_module._convert_overconstrained_group_to_d6
    original_combine = conversion_module.combine_overconstrained_joints_in_physx_layer

    def convert_with_d6_repair(
        stage: Any,
        joints: list[Any],
        body0: tuple,
        body1: tuple,
        source_joint_remap: dict[Any, Any],
    ) -> bool:
        from pxr import UsdPhysics

        states = _source_axis_states(joints, UsdPhysics)
        converted = original(stage, joints, body0, body1, source_joint_remap)
        if converted:
            _repair_d6_group(stage, states, source_joint_remap, UsdPhysics)
        return converted

    vars(convert_with_d6_repair)[_PATCH_MARKER] = True
    conversion_module._convert_overconstrained_group_to_d6 = convert_with_d6_repair

    def combine_with_mjcf_metadata(physx_layer_path: str) -> int:
        import os

        if not os.path.exists(physx_layer_path):
            return original_combine(physx_layer_path)

        from pxr import Sdf, Usd

        physx_layer = Sdf.Layer.FindOrOpen(physx_layer_path)
        if physx_layer is None:
            return original_combine(physx_layer_path)

        mujoco_layer_path = os.path.join(
            os.path.dirname(physx_layer_path), "mujoco.usda"
        )
        if not os.path.exists(mujoco_layer_path):
            return original_combine(physx_layer_path)

        composition_layer = Sdf.Layer.CreateAnonymous(
            "mjc_to_physx_conversion.usda"
        )
        composition_layer.subLayerPaths = [
            physx_layer.identifier,
            mujoco_layer_path,
        ]
        stage = Usd.Stage.Open(composition_layer)
        if stage is None:
            return original_combine(physx_layer_path)

        previous_target = stage.GetEditTarget()
        try:
            stage.SetEditTarget(stage.GetEditTargetForLocalLayer(physx_layer))
            conversion_module.convert_mjc_to_physx(stage)
            converted = conversion_module.combine_overconstrained_joints_to_d6(stage)
        finally:
            stage.SetEditTarget(previous_target)

        if converted:
            physx_layer.Save()
        return converted

    vars(combine_with_mjcf_metadata)[_PATCH_MARKER] = True
    conversion_module.combine_overconstrained_joints_in_physx_layer = (
        combine_with_mjcf_metadata
    )
