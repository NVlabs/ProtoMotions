# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Metatron v1 simulation asset without changing its source XML."""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

_G1_PD_PROFILES = {
    "hip_pitch": (40.17923847137318, 2.5578897650279457, 88.0, 0.010177520),
    "hip_yaw": (40.17923847137318, 2.5578897650279457, 88.0, 0.010177520),
    "hip_roll": (99.09842777666113, 6.3088018534966395, 139.0, 0.025101925),
    "knee": (99.09842777666113, 6.3088018534966395, 139.0, 0.025101925),
    "ankle": (28.50124619574858, 1.814445686584846, 50.0, 0.007219450),
    "waist_yaw": (40.17923847137318, 2.5578897650279457, 88.0, 0.010177520),
    "waist_roll": (28.50124619574858, 1.814445686584846, 50.0, 0.007219450),
    "waist_pitch": (28.50124619574858, 1.814445686584846, 50.0, 0.007219450),
    "shoulder": (14.25062309787429, 0.907222843292423, 25.0, 0.003609725),
    "elbow": (14.25062309787429, 0.907222843292423, 25.0, 0.003609725),
    "wrist_roll": (14.25062309787429, 0.907222843292423, 25.0, 0.003609725),
    "wrist_pitch": (16.77832748089279, 1.06814150219, 5.0, 0.004250000),
    "wrist_yaw": (16.77832748089279, 1.06814150219, 5.0, 0.004250000),
}


def _add_semantic_fk_bodies(root: ET.Element) -> None:
    """Add the non-dynamic G1 body points required by the policy FK contract.

    Some mesh-bearing G1 MJCFs expose the hand and head meshes on an existing
    link but omit the named ``head``/``rubber_hand`` bodies used by the
    ProtoMotions motion representation.  Those points have no independent
    joints or mass: they are pure kinematic markers.  Adding empty bodies
    keeps the source asset's visual geometry and dynamics unchanged while
    making the MuJoCo and ProtoMotions body order explicit and identical.
    """

    worldbodies = root.findall("worldbody")
    worldbody = worldbodies[0] if worldbodies else None
    if worldbody is None:
        raise ValueError("MuJoCo model has no worldbody")
    pelvis = worldbody.find("body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("G1 model has no pelvis body")

    head = pelvis.find("body[@name='head']")
    if head is None:
        pelvis.insert(0, ET.Element("body", {"name": "head", "pos": "0 0 0.4"}))

    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        if wrist is None:
            raise ValueError(f"G1 model has no {side}_wrist_yaw_link body")
        hand_name = f"{side}_rubber_hand"
        if wrist.find(f"body[@name='{hand_name}']") is None:
            wrist.append(ET.Element("body", {"name": hand_name, "pos": "0.12 0 0"}))

    required = (
        "head",
        "left_rubber_hand",
        "right_rubber_hand",
    )
    missing = [
        name for name in required if root.find(f".//body[@name='{name}']") is None
    ]
    if missing:
        raise ValueError(f"failed to add semantic G1 FK bodies: {missing}")


def _set_rubber_hand_mass(root: ET.Element, mass: float) -> None:
    """Give each semantic rubber-hand body an explicit test inertial.

    The mesh-bearing G1 asset normally uses these bodies as massless FK
    markers.  This opt-in override is for quick MuJoCo dynamics experiments;
    it never changes the source XML and does not add contact geometry.  The
    small isotropic inertia keeps the inertial definition well-conditioned
    while the requested mass is applied at the marker origin.
    """

    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("rubber hand mass must be finite and positive")
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_rubber_hand']")
        if body is None:
            raise ValueError(f"missing {side}_rubber_hand semantic body")
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.Element("inertial")
            body.insert(0, inertial)
        inertial.attrib.update(
            {
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "mass": str(mass),
                "diaginertia": "1e-4 1e-4 1e-4",
            }
        )


def patch_model(
    source: Path,
    pd_scale: float | None = None,
    pd_mode: str = "implicit",
    zero_frictionloss: bool = False,
    rubber_hand_mass: float | None = None,
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Prepare a visual G1 model with the ProtoMotions position contract.

    The patch is deliberately applied to a temporary copy.  It may add the
    massless semantic FK markers missing from mesh-only G1 assets, but it does
    not modify the source asset on disk or add dynamic bodies.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    if pd_mode not in ("implicit", "explicit"):
        raise ValueError("pd_mode must be 'implicit' or 'explicit'")
    compiler = root.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        # The patched XML lives in a temporary directory; keep asset lookup
        # rooted at the source tree instead of resolving ../mesh from /tmp.
        meshdir = Path(compiler.attrib["meshdir"])
        compiler.attrib["meshdir"] = str((source.parent / meshdir).resolve())
    worldbodies = root.findall("worldbody")
    worldbody = worldbodies[0] if worldbodies else None
    if worldbody is None:
        raise ValueError("MuJoCo model has no worldbody")
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    if not asset.findall("mesh"):
        raise ValueError(
            "the viewer requires a mesh-bearing G1 XML; "
            "use a mesh-bearing G1 asset instead of a reduced no-mesh model"
        )
    _add_semantic_fk_bodies(root)
    if rubber_hand_mass is not None:
        _set_rubber_hand_mass(root, rubber_hand_mass)
    if asset.find("texture[@name='deployment_floor_texture']") is None:
        ET.SubElement(
            asset,
            "texture",
            name="deployment_floor_texture",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1="0.16 0.19 0.22",
            rgb2="0.07 0.09 0.11",
        )
    if asset.find("material[@name='deployment_floor_material']") is None:
        ET.SubElement(
            asset,
            "material",
            name="deployment_floor_material",
            texture="deployment_floor_texture",
            texrepeat="12 12",
            reflectance="0.12",
        )
    has_floor = any(
        node.find("geom[@name='floor']") is not None for node in worldbodies
    )
    if not has_floor:
        ET.SubElement(
            worldbody,
            "geom",
            name="floor",
            type="plane",
            size="20 20 .1",
            pos="0 0 0",
            contype="1",
            conaffinity="1",
            material="deployment_floor_material",
        )
    # Keep collision geoms active for dynamics but place them in a viewer-only
    # group that is disabled below. In the G1 asset, most capsule collision
    # geoms inherit ``contype`` from a default class, so checking only the
    # explicit attribute would miss them. Mesh visuals remain in their
    # original visual group, and the floor remains visible in group zero.
    for geom in root.iter("geom"):
        if geom.attrib.get("name") == "floor":
            continue
        if geom.attrib.get("type", "") == "mesh":
            # Meshes are visual geometry in the deployment model.  A few
            # third-party G1 assets duplicate collision meshes; retaining
            # those contacts makes this path dynamically different from the
            # trained capsule-collision model.
            geom.attrib["contype"] = "0"
            geom.attrib["conaffinity"] = "0"
        else:
            geom.attrib["group"] = "5"
    lights = {node.attrib.get("name") for node in worldbody.findall("light")}
    if "deployment_key_light" not in lights:
        ET.SubElement(
            worldbody,
            "light",
            name="deployment_key_light",
            pos="2 -3 5",
            dir="-0.25 0.35 -1",
            directional="true",
            diffuse="0.9 0.9 0.9",
            specular="0.35 0.35 0.35",
            castshadow="true",
        )
    if "deployment_fill_light" not in lights:
        ET.SubElement(
            worldbody,
            "light",
            name="deployment_fill_light",
            pos="-3 1 2.5",
            dir="0.45 -0.2 -1",
            directional="true",
            diffuse="0.45 0.52 0.62",
            specular="0.12 0.12 0.12",
        )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.attrib.update({"offwidth": "1600", "offheight": "900"})
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(
        {
            "diffuse": "0.35 0.4 0.5",
            "ambient": "0.25 0.25 0.25",
            "specular": "0.2 0.2 0.2",
        }
    )
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("MuJoCo model has no actuators")
    # ``explicit`` uses the hardware torque-PD contract: the policy emits a
    # joint-position target,
    # then the bridge computes a torque from the current DOF state at every
    # policy callback and holds it for 20 physics substeps. ``implicit`` keeps
    # the position-actuator path used by the exported ProtoMotions policy.
    # A multiplier is available for deliberate visual experiments, but the
    # default is exactly the ProtoMotions/Unitree gain contract.
    gain_scale = 1.0 if pd_scale is None else pd_scale
    if not math.isfinite(gain_scale) or gain_scale <= 0.0:
        raise ValueError("--pd-scale must be finite and positive")
    gains = {
        name: (kp * gain_scale, kv * gain_scale, effort, armature)
        for name, (kp, kv, effort, armature) in _G1_PD_PROFILES.items()
    }
    original = list(actuator)
    actuator.clear()
    for motor in original:
        if motor.tag != "motor":
            actuator.append(motor)
            continue
        joint = motor.attrib["joint"]
        joint_node = root.find(f".//joint[@name='{joint}']")
        if joint_node is None or "range" not in joint_node.attrib:
            raise ValueError(f"actuated joint {joint!r} has no finite range")
        key = next((name for name in gains if name in joint), None)
        if key is None:
            raise ValueError(f"no PD gains configured for joint {joint!r}")
        kp, kv, effort, armature = gains[key]
        # Explicit PD owns the complete joint spring/damper law.  Clear any
        # passive MJCF spring/damper so a source asset with incidental defaults
        # (for example g1_bm.xml) cannot silently double-count the gains.
        # This is also the canonical ProtoMotions MuJoCo setup.
        joint_node.attrib["stiffness"] = "0"
        joint_node.attrib["damping"] = "0"
        if zero_frictionloss:
            joint_node.attrib["frictionloss"] = "0.0"
        joint_node.attrib["armature"] = str(armature)
        if pd_mode == "explicit":
            # Keep the source asset's friction, solver, and integrator
            # settings. The hardware adapter applies this force limit in its
            # explicit-PD step; the equivalent MJCF limit also protects this bridge from
            # an excessive torque.
            motor.attrib["ctrllimited"] = "false"
            motor.attrib["forcelimited"] = "true"
            motor.attrib["forcerange"] = f"{-effort} {effort}"
            actuator.append(motor)
            continue
        # The implicit position path owns the PD terms through the position
        # actuator and uses the same canonical armature/friction contract.
        ET.SubElement(
            actuator,
            "position",
            name=joint,
            joint=joint,
            kp=str(kp),
            kv=str(kv),
            ctrllimited="false",
            forcerange=f"{-effort} {effort}",
            forcelimited="true",
        )
    # Public BM no-mesh assets contain optional IMU references to a site that
    # is absent in that reduced model. The deployment state comes from qpos /
    # qvel here, so drop only those optional sensors rather than inventing data.
    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)
    directory = tempfile.TemporaryDirectory(prefix="protomotions-g1-")
    output = Path(directory.name) / source.name
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output, directory
