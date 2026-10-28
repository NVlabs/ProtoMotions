# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained G1 simulation preparation invariants."""

import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

from tools.metatron_v1_mujoco_model import (
    patch_model,
    _G1_PD_PROFILES,
    _add_semantic_fk_bodies,
)

G1_XML = (
    Path(__file__).resolve().parents[3]
    / "protomotions/data/assets/mjcf/g1_holo_compat.xml"
)
_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class ModelPreparationTest(unittest.TestCase):
    def test_g1_pd_profile_matches_robot_contract(self) -> None:
        # These are the exact G1RobotConfig armature-derived values, not the
        # rounded source-MJCF defaults.  The model patch applies them to both
        # the canonical mesh asset and reduced visual variants.
        self.assertEqual(_G1_PD_PROFILES["hip_pitch"][2], 88.0)
        self.assertEqual(_G1_PD_PROFILES["hip_roll"][2], 139.0)
        self.assertEqual(_G1_PD_PROFILES["ankle"][2], 50.0)
        self.assertAlmostEqual(_G1_PD_PROFILES["hip_pitch"][0], 40.17923847137318)
        self.assertAlmostEqual(_G1_PD_PROFILES["hip_roll"][1], 6.3088018534966395)
        self.assertAlmostEqual(_G1_PD_PROFILES["wrist_yaw"][3], 0.00425)

    def test_explicit_patch_owns_passive_pd_and_preserves_force_limits(self) -> None:
        source = G1_XML
        source_root = ET.parse(source).getroot()
        source_frictionloss = {
            node.attrib["name"]: float(node.attrib.get("frictionloss", "0"))
            for node in source_root.iter("joint")
            if "name" in node.attrib
        }
        patched, temporary = patch_model(source, pd_mode="explicit")
        try:
            model = mujoco.MjModel.from_xml_path(str(patched))
            self.assertEqual(model.nu, 29)
            for joint_id in range(model.njnt):
                if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                self.assertEqual(float(model.jnt_stiffness[joint_id]), 0.0)
                self.assertEqual(
                    float(model.dof_damping[model.jnt_dofadr[joint_id]]), 0.0
                )
            for actuator_id in range(model.nu):
                joint_id = int(model.actuator_trnid[actuator_id, 0])
                joint_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                profile = next(
                    values
                    for name, values in _G1_PD_PROFILES.items()
                    if name in joint_name
                )
                self.assertAlmostEqual(
                    float(model.dof_armature[model.jnt_dofadr[joint_id]]),
                    profile[3],
                )
                self.assertAlmostEqual(
                    float(model.actuator_forcerange[actuator_id, 1]), profile[2]
                )
            # Do not silently change Coulomb friction while adapting the
            # source XML.  This is part of the simulator asset contract.
            for actuator_id in range(model.nu):
                joint_id = int(model.actuator_trnid[actuator_id, 0])
                joint_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                self.assertAlmostEqual(
                    float(model.dof_frictionloss[model.jnt_dofadr[joint_id]]),
                    source_frictionloss[joint_name],
                )
        finally:
            temporary.cleanup()

    def test_holo_actuator_order_matches_policy_contract(self) -> None:
        source = G1_XML
        patched, temporary = patch_model(source, pd_mode="explicit")
        try:
            model = mujoco.MjModel.from_xml_path(str(patched))
            names = tuple(
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    int(model.actuator_trnid[index, 0]),
                )
                for index in range(model.nu)
            )
            self.assertEqual(names, _G1_JOINT_NAMES)
        finally:
            temporary.cleanup()

    def test_missing_semantic_fk_bodies_are_massless_markers(self) -> None:
        root = ET.fromstring(
            """
            <mujoco>
              <worldbody>
                <body name="pelvis">
                  <body name="left_wrist_yaw_link" />
                  <body name="right_wrist_yaw_link" />
                </body>
              </worldbody>
            </mujoco>
            """
        )

        _add_semantic_fk_bodies(root)

        pelvis = root.find(".//body[@name='pelvis']")
        self.assertIsNotNone(pelvis)
        self.assertEqual(pelvis[0].attrib, {"name": "head", "pos": "0 0 0.4"})
        for side in ("left", "right"):
            hand = root.find(f".//body[@name='{side}_rubber_hand']")
            self.assertIsNotNone(hand)
            self.assertEqual(
                hand.attrib, {"name": f"{side}_rubber_hand", "pos": "0.12 0 0"}
            )
            self.assertEqual(list(hand), [])

    def test_rubber_hand_mass_override_adds_one_kg_each(self) -> None:
        source = G1_XML
        patched, temporary = patch_model(
            source, pd_mode="explicit", rubber_hand_mass=1.0
        )
        try:
            model = mujoco.MjModel.from_xml_path(str(patched))
            for side in ("left", "right"):
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rubber_hand"
                )
                self.assertGreaterEqual(body_id, 0)
                self.assertAlmostEqual(float(model.body_mass[body_id]), 1.0)
        finally:
            temporary.cleanup()

    def test_rubber_hand_mass_override_rejects_nonpositive_mass(self) -> None:
        source = G1_XML
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            patch_model(source, pd_mode="explicit", rubber_hand_mass=0.0)
