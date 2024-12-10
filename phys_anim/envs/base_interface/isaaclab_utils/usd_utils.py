# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
from pathlib import Path

import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
import carb
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics, UsdShade, UsdGeom


def set_drive_type(prim_path, drive_type):
    joint_prim = get_prim_at_path(prim_path)

    # set drive type ("angular" or "linear")
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)
    return drive


def set_drive_target_position(drive, target_value):
    if not drive.GetTargetPositionAttr():
        drive.CreateTargetPositionAttr(target_value)
    else:
        drive.GetTargetPositionAttr().Set(target_value)


def set_drive_target_velocity(drive, target_value):
    if not drive.GetTargetVelocityAttr():
        drive.CreateTargetVelocityAttr(target_value)
    else:
        drive.GetTargetVelocityAttr().Set(target_value)


def set_drive_stiffness(drive, stiffness):
    if not drive.GetStiffnessAttr():
        drive.CreateStiffnessAttr(stiffness)
    else:
        drive.GetStiffnessAttr().Set(stiffness)


def set_drive_damping(drive, damping):
    if not drive.GetDampingAttr():
        drive.CreateDampingAttr(damping)
    else:
        drive.GetDampingAttr().Set(damping)


def set_drive_max_force(drive, max_force):
    if not drive.GetMaxForceAttr():
        drive.CreateMaxForceAttr(max_force)
    else:
        drive.GetMaxForceAttr().Set(max_force)


def set_drive(
    prim_path, drive_type, target_type, target_value, stiffness, damping, max_force
) -> None:
    drive = set_drive_type(prim_path, drive_type)

    # set target type ("position" or "velocity")
    if target_type == "position":
        set_drive_target_position(drive, target_value)
    elif target_type == "velocity":
        set_drive_target_velocity(drive, target_value)

    set_drive_stiffness(drive, stiffness)
    set_drive_damping(drive, damping)
    set_drive_max_force(drive, max_force)


def create_sphere_light(prim_path="/World/sphereLight", intensity=6000, radius=20):
    stage = get_current_stage()
    light = UsdLux.SphereLight.Define(stage, prim_path)
    light.CreateIntensityAttr(intensity)
    light.CreateRadiusAttr(radius)

    prim_at_path = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim_at_path)
    xformable.AddTranslateOp().Set((20, 20, 25))


def create_distant_light(prim_path="/World/defaultDistantLight", intensity=3000):
    stage = get_current_stage()
    light = UsdLux.DistantLight.Define(stage, prim_path)
    light.CreateIntensityAttr(intensity)


def add_terrain_to_stage(stage, vertices, triangles, position=None, orientation=None):
    num_faces = triangles.shape[0]
    terrain_mesh = stage.DefinePrim("/World/terrain", "Mesh")
    terrain_mesh.GetAttribute("points").Set(vertices)
    terrain_mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten())
    terrain_mesh.GetAttribute("faceVertexCounts").Set(np.asarray([3] * num_faces))

    # Create a reference to your material USD
    material_prim = stage.DefinePrim("/World/Materials/terrain_material", "Material")

    path = Path(os.path.dirname(os.path.abspath(__file__))).parents[2]
    material_usd = path / "data" / "assets" / "usd" / "terrain_material.usda"

    material_prim.GetReferences().AddReference(str(material_usd))
    # Get the existing material
    material_prim = stage.GetPrimAtPath(
        "/World/Materials/terrain_material/solid_white_grid"
    )
    # Get the material
    material = UsdShade.Material(material_prim)

    # Assign material to mesh
    UsdShade.MaterialBindingAPI(terrain_mesh).Bind(material)

    terrain = XFormPrim(
        prim_path="/World/terrain",
        name="terrain",
        position=position,
        orientation=orientation,
    )

    UsdPhysics.CollisionAPI.Apply(terrain.prim)
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
    physx_collision_api.GetContactOffsetAttr().Set(0.02)
    physx_collision_api.GetRestOffsetAttr().Set(0.00)

    # # create parent prim
    # import omni.isaac.core.utils.prims as prim_utils
    # import omni.isaac.lab.sim as sim_utils
    #
    # prim_path = "/World/terrain"
    # prim_utils.create_prim(prim_path, "Xform")
    # # create mesh prim
    # prim = prim_utils.create_prim(
    #     f"{prim_path}/mesh",
    #     "Mesh",
    #     translation=position,
    #     orientation=orientation,
    #     attributes={
    #         "points": vertices,
    #         "faceVertexIndices": triangles.flatten(),
    #         "faceVertexCounts": np.asarray([3] * len(triangles)),
    #         "subdivisionScheme": "bilinear",
    #     },
    # )
    # # apply collider properties
    # collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    # sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)
    #
    # physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
    #     friction_combine_mode="average",
    #     restitution_combine_mode="average",
    #     static_friction=1.0,
    #     dynamic_friction=1.0,
    #     restitution=0.0,
    # )
    #
    # # spawn the material
    # physics_material_cfg.func(f"{prim_path}/physicsMaterial", physics_material_cfg)
    # sim_utils.bind_physics_material(prim.GetPrimPath(), f"{prim_path}/physicsMaterial")


def createJoint(stage, joint_type, from_prim, to_prim):
    # for single selection use to_prim
    if to_prim is None:
        to_prim = from_prim
        from_prim = None

    from_path = (
        from_prim.GetPath().pathString
        if from_prim is not None and from_prim.IsValid()
        else ""
    )
    to_path = (
        to_prim.GetPath().pathString
        if to_prim is not None and to_prim.IsValid()
        else ""
    )
    single_selection = from_path == "" or to_path == ""

    # to_path can be not writable as in case of instancing, find first writable path
    joint_base_path = to_path
    base_prim = stage.GetPrimAtPath(joint_base_path)
    while base_prim != stage.GetPseudoRoot():
        if base_prim.IsInstance():
            base_prim = base_prim.GetParent()
        elif base_prim.IsInstanceProxy():
            base_prim = base_prim.GetParent()
        elif base_prim.IsInstanceable():
            base_prim = base_prim.GetParent()
        else:
            break
    joint_base_path = str(base_prim.GetPrimPath())
    if joint_base_path == "/":
        joint_base_path = ""

    joint_name = "/" + joint_type + "Joint"
    joint_path = joint_base_path + joint_name
    print("Joint path:", joint_path)

    if joint_type == "Fixed":
        component = UsdPhysics.FixedJoint.Define(stage, joint_path)

        # Ensure the object (to_prim) is not static
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(to_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr().Set(True)

    elif joint_type == "Revolute":
        component = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        component.CreateAxisAttr("X")
    elif joint_type == "Prismatic":
        component = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        component.CreateAxisAttr("X")
    elif joint_type == "Spherical":
        component = UsdPhysics.SphericalJoint.Define(stage, joint_path)
        component.CreateAxisAttr("X")
    elif joint_type == "Distance":
        component = UsdPhysics.DistanceJoint.Define(stage, joint_path)
        component.CreateMinDistanceAttr(0.0)
        component.CreateMaxDistanceAttr(0.0)
    elif joint_type == "Gear":
        component = PhysxSchema.PhysxPhysicsGearJoint.Define(stage, joint_path)
    elif joint_type == "RackAndPinion":
        component = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(stage, joint_path)
    else:
        component = UsdPhysics.Joint.Define(stage, joint_path)
        prim = component.GetPrim()
        for limit_name in ["transX", "transY", "transZ", "rotX", "rotY", "rotZ"]:
            limit_api = UsdPhysics.LimitAPI.Apply(prim, limit_name)
            limit_api.CreateLowAttr(1.0)
            limit_api.CreateHighAttr(-1.0)

    xfCache = UsdGeom.XformCache()

    if not single_selection:
        to_pose = xfCache.GetLocalToWorldTransform(to_prim)
        from_pose = xfCache.GetLocalToWorldTransform(from_prim)
        rel_pose = to_pose * from_pose.GetInverse()
        rel_pose = rel_pose.RemoveScaleShear()
        pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
        rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())

        component.CreateBody0Rel().SetTargets([Sdf.Path(from_path)])
        component.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
        component.CreateLocalPos0Attr().Set(pos1)
        component.CreateLocalRot0Attr().Set(rot1)
        component.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    else:
        to_pose = xfCache.GetLocalToWorldTransform(to_prim)
        to_pose = to_pose.RemoveScaleShear()
        pos1 = Gf.Vec3f(to_pose.ExtractTranslation())
        rot1 = Gf.Quatf(to_pose.ExtractRotationQuat())

        # For fixed joints, we don't set Body0 (which would be the world)
        if joint_type != "Fixed":
            component.CreateBody0Rel().SetTargets([Sdf.Path("/World")])
        component.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
        component.CreateLocalPos0Attr().Set(pos1)
        component.CreateLocalRot0Attr().Set(rot1)
        component.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # component.CreateBreakForceAttr().Set(100000.0)
    # component.CreateBreakTorqueAttr().Set(100000.0)

    return stage.GetPrimAtPath(joint_base_path + joint_name)
