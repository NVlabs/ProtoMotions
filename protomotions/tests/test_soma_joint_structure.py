# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from data.scripts.from_soma_to_anatomical_soma.anatomical_soma_joint_structure import (
    joint_order,
    reorder_soma_tree,
)
from data.scripts.from_soma_to_anatomical_soma.soma23_to_anatomical_soma_retargeting import (
    hinge_seed,
    semantic_names,
)

ASSETS = Path(__file__).resolve().parents[1] / "data/assets/mjcf"


def test_orders_and_axis_limits_remain_associated():
    tree = ET.parse(ASSETS / "soma23_humanoid.xml")
    original = {
        j.get("name"): dict(j.attrib) for j in tree.iter("joint") if j.get("name")
    }
    reorder_soma_tree(tree, serial=False)
    for body in tree.find("worldbody").iter("body"):
        joints = body.findall("joint")
        if len(joints) == 3:
            assert [j.get("name").rsplit("_", 1)[1] for j in joints] == list(
                joint_order(body.get("name"))
            )
            assert all(j.attrib == original[j.get("name")] for j in joints)


@pytest.fixture
def ordered_soma(tmp_path):
    """Exercise grouped hinges using only the public SOMA23 source asset."""
    from data.scripts.from_soma_to_anatomical_soma.build_soma_joint_structure import (
        build,
    )

    output = tmp_path / "ordered_soma.xml"
    build(ASSETS / "soma23_humanoid.xml", output, serial=False)
    return output


def test_reordering_preserves_rest_geometry_and_dynamics(ordered_soma):
    models = [
        mujoco.MjModel.from_xml_path(str(path))
        for path in (ASSETS / "soma23_humanoid.xml", ordered_soma)
    ]
    before, after = models
    assert after.nv == before.nv == 72
    for field in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "geom_pos",
        "geom_size",
    ):
        np.testing.assert_allclose(
            getattr(after, field), getattr(before, field), atol=1e-12
        )
    data = [mujoco.MjData(m) for m in models]
    for m, d in zip(models, data):
        mujoco.mj_forward(m, d)
    np.testing.assert_allclose(data[0].xpos, data[1].xpos, atol=1e-12)
    np.testing.assert_allclose(data[0].xquat, data[1].xquat, atol=1e-12)


def test_anatomical_orders_actuators_and_idempotence():
    tree = ET.parse(ASSETS / "anatomic_soma.xml")
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "anatomic_soma.xml"))
    # The joint order is the asset itself: hinges of one semantic part in document order.
    metadata = {}
    for body in tree.getroot().find("worldbody").iter("body"):
        for joint in body.findall("joint"):
            if joint.get("type") == "hinge":
                metadata.setdefault(joint.get("name").rsplit("_", 1)[0], []).append(
                    joint.get("name")
                )
    for name, joints in metadata.items():
        if len(joints) == 3:
            assert [j.rsplit("_", 1)[1] for j in joints] == list(joint_order(name))
    assert metadata["LeftArm"] == ["LeftArm_x", "LeftArm_z", "LeftArm_y"]
    assert metadata["Spine1"] == ["Spine1_z", "Spine1_y", "Spine1_x"]
    assert metadata["LeftHand"] == ["LeftHand_y", "LeftHand_x", "LeftHand_z"]
    for dof, joint in enumerate(range(1, model.njnt)):
        assert model.actuator_trnid[dof, 0] == joint
    expected = ET.tostring(tree.getroot())
    reorder_soma_tree(tree, serial=True)
    assert ET.tostring(tree.getroot()) == expected


def test_seed_handles_fixed_rest_frame(tmp_path):
    tree = ET.parse(ASSETS / "anatomic_soma.xml")
    # A right-angle fixed mount exercises a full signed axis triple.
    head = tree.find(".//body[@name='Head_x_link']")
    head.set("quat", "0.7071067811865476 0 0 0.7071067811865476")
    path = tmp_path / "mounted.xml"
    tree.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(31)
    q = rng.uniform(-0.03, 0.03, (7, 52))
    q = np.clip(q, model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    ids = [model.body(n).id for n in semantic_names(path)]
    rotations = []
    for row in q:
        data.qpos[7:] = row
        mujoco.mj_forward(model, data)
        rotations.append(data.xquat[ids][:, [1, 2, 3, 0]].copy())
    seed, names = hinge_seed(path, np.asarray(rotations))
    assert names == [model.joint(i).name for i in range(1, model.njnt)]
    np.testing.assert_allclose(seed, q, atol=1e-6)


def test_grouped_orientation_adapter_uses_new_joint_order(ordered_soma):
    path = ordered_soma
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    ids = [model.body(n).id for n in semantic_names(path)]
    q = np.random.default_rng(42).uniform(-0.2, 0.2, (8, 66))
    rotations = []
    for row in q:
        data.qpos[7:] = row
        mujoco.mj_forward(model, data)
        rotations.append(data.xquat[ids][:, [1, 2, 3, 0]].copy())
    seed, names = hinge_seed(path, np.asarray(rotations))
    assert names == [model.joint(i).name for i in range(1, model.njnt)]
    np.testing.assert_allclose(seed, q, atol=1e-6)
