# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MJCF inertias must not depend on a simulator's default link mass."""

from pathlib import Path
import xml.etree.ElementTree as ET

from dm_control import mjcf
import mujoco
import numpy as np
import pytest

from protomotions.simulator.base_simulator.robot_mass_properties import (
    prepare_robot_mjcf,
    _read_mjcf,
)


ASSET_DIR = Path(__file__).resolve().parents[1] / "data" / "assets" / "mjcf"
MARKER_ASSETS = {
    "g1_bm.xml": ("head", "left_rubber_hand", "right_rubber_hand"),
    "g1_bm_box_feet.xml": ("head", "left_rubber_hand", "right_rubber_hand"),
    "g1_bm_no_mesh_box_feet.xml": ("head", "left_rubber_hand", "right_rubber_hand"),
    "g1_holo_compat.xml": ("head", "left_rubber_hand", "right_rubber_hand"),
    "g1_holo_compat_box_feet.xml": ("head", "left_rubber_hand", "right_rubber_hand"),
    "h1_2.xml": ("head_aux",),
    "h1_2_box_feet.xml": ("head_aux",),
    "h1_2_no_mesh_box_feet.xml": ("head_aux",),
}


def asset_case(path):
    """Keep the XML checks in public CI and mesh compilation in LFS CI."""
    root = ET.parse(path).getroot()
    marks = pytest.mark.needs_lfs if any(e.get("file") for e in root.iter()) else ()
    return pytest.param(path, marks=marks, id=path.name)


def compile_body_inertias(path):
    """Compile body properties without unrelated sensor/contact references.

    Some robot-only assets refer to an IMU or ground injected by the runtime.
    Remove only non-inertial sections in this test; keep bodies, joints, geoms,
    compiler settings and defaults so mass inference is exercised normally.
    dm_control resolves includes and packages mesh dependencies for MuJoCo.
    """
    root = _read_mjcf(path)
    for tag in ("sensor", "contact", "actuator", "tendon", "equality", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    source = mjcf.from_xml_string(ET.tostring(root, encoding="unicode"))
    return mujoco.MjModel.from_xml_string(source.to_xml_string(), source.get_assets())


@pytest.mark.parametrize(
    "path", [asset_case(ASSET_DIR / name) for name in sorted(MARKER_ASSETS)]
)
def test_markers_have_explicit_one_gram_inertias(path, tmp_path):
    original = path.read_bytes()
    prepared = prepare_robot_mjcf(path, tmp_path)
    root = ET.parse(prepared).getroot()
    assert path.read_bytes() == original
    for name in MARKER_ASSETS[path.name]:
        body = root.find(f".//body[@name='{name}']")
        assert body is not None
        assert body.find("joint") is None and body.find("freejoint") is None
        inertial = body.find("inertial")
        assert inertial is not None
        assert float(inertial.get("mass")) == 0.001
        np.testing.assert_array_equal(
            np.fromstring(inertial.get("diaginertia"), sep=" "), [4e-8] * 3
        )
        np.testing.assert_array_equal(
            np.fromstring(inertial.get("pos"), sep=" "), [0.0] * 3
        )


@pytest.mark.parametrize(
    "path", [asset_case(p) for p in sorted(ASSET_DIR.glob("*.xml"))]
)
def test_robot_links_have_positive_mass_and_inertia(path, tmp_path):
    original = compile_body_inertias(path)
    model = compile_body_inertias(prepare_robot_mjcf(path, tmp_path))
    # The world body is intentionally static and has no inertial properties.
    assert np.all(model.body_mass[1:] > 0), path.name
    assert np.all(model.body_inertia[1:] > 0), path.name
    for name in MARKER_ASSETS.get(path.name, ()):
        np.testing.assert_array_equal(model.body(name).mass, [0.001])
        np.testing.assert_allclose(model.body(name).inertia, [4e-8] * 3, rtol=1e-12)
        np.testing.assert_array_equal(model.body(name).ipos, [0.0] * 3)
    physical = original.body_mass > 0
    np.testing.assert_allclose(model.body_mass[physical], original.body_mass[physical])
    np.testing.assert_allclose(
        model.body_inertia[physical], original.body_inertia[physical]
    )


@pytest.mark.parametrize("inference", ["auto", "true"])
def test_normalization_resolves_defaults_includes_and_geometry(tmp_path, inference):
    source = tmp_path / "source"
    source.mkdir()
    (source / "child.xml").write_text(
        '<mujocoinclude><body name="marker" childclass="massless">'
        '<geom type="sphere" size="0.0001"/></body></mujocoinclude>'
    )
    path = source / "robot.xml"
    path.write_text(
        f'<mujoco><compiler inertiafromgeom="{inference}"/>'
        '<default><default class="massless"><geom density="0"/></default></default>'
        '<worldbody><body name="physical"><freejoint/>'
        '<geom type="sphere" size="0.1" density="1000"/>'
        '<include file="child.xml"/></body></worldbody></mujoco>'
    )
    original = compile_body_inertias(path)
    prepared = prepare_robot_mjcf(path, tmp_path / "prepared")
    model = compile_body_inertias(prepared)
    np.testing.assert_allclose(
        model.body("physical").mass, original.body("physical").mass
    )
    np.testing.assert_allclose(
        model.body("physical").inertia, original.body("physical").inertia
    )
    np.testing.assert_array_equal(model.body("marker").mass, [0.001])
    np.testing.assert_array_equal(model.body("marker").inertia, [4e-8] * 3)
    # Preparing an already normalized asset is a no-op.
    assert prepare_robot_mjcf(prepared, tmp_path / "second") == prepared
