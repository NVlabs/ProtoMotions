# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np

from examples.deployment_context import ProtoMotionsGlobalContext
from examples.metatron_recovery import (
    RecoveryPose,
    StagedRecovery,
    contains_inversion,
    slerp,
)
from examples.metatron_v1_contract import BOOL_INPUTS, INPUT_SHAPES


def pose(height: float, pitch: float = 0) -> RecoveryPose:
    positions = np.zeros((33, 3), np.float32)
    positions[16, 2] = height
    positions[7, 1], positions[13, 1] = 0.1, -0.1
    positions[24, 1], positions[32, 1] = 0.25, -0.25
    rotations = np.tile([0, np.sin(pitch / 2), 0, np.cos(pitch / 2)], (33, 1)).astype(
        np.float32
    )
    return RecoveryPose(positions, rotations, np.zeros(29, np.float32))


def inputs() -> dict[str, np.ndarray]:
    return {
        key: np.zeros(shape, bool if key in BOOL_INPUTS else np.float32)
        for key, shape in INPUT_SHAPES.items()
    }


class StagedRecoveryTest(unittest.TestCase):
    def test_roll_disables_recovery_for_the_entire_requested_clip(self) -> None:
        upright = np.array([[0, 0, 0, 1]], np.float32)
        self.assertFalse(contains_inversion(np.repeat(upright, 3, axis=0)))
        roll = np.concatenate((upright, [[0, 1, 0, 0]], upright))
        self.assertTrue(contains_inversion(roll))

    def setUp(self) -> None:
        start, finish = pose(0.1, np.pi / 2), pose(0.79)
        names = [str(index) for index in range(33)]
        names[24], names[32] = "left_rubber_hand", "right_rubber_hand"
        plan = {
            "body_names": names,
            "anchor_body_index": 16,
            "quaternion_order": "xyzw",
            "frames": 2,
            "dt_seconds": 1,
            "positions": [start.positions.tolist(), finish.positions.tolist()],
            "rotations": [start.rotations.tolist(), finish.rotations.tolist()],
        }
        self.recovery = StagedRecovery(plan, names)

    def test_prone_is_exclusive_local_prompt(self) -> None:
        feed = inputs()
        for name in BOOL_INPUTS:
            feed[name].fill(True)
        self.assertTrue(self.recovery.step(feed, pose(0.1, np.pi / 2), 0.02))
        self.assertEqual(self.recovery.phase, "prone_plan")
        for name in (
            "target_bodies_masks",
            "target_poses_masks",
            "style_frame_masks",
            "velocity_command_mask",
        ):
            self.assertFalse(feed[f"masked_mimic.{name}"].any())
        self.assertFalse(feed["scene_boxes.visible"].any())
        masks = feed["masked_mimic.local_target_bodies_masks"].reshape(5, 5, 2)
        self.assertTrue(masks[:, :, 0].all())
        self.assertTrue(masks[:, 0, 1].all())
        self.assertFalse(masks[:, 1:, 1].any())

    def test_recovery_updates_the_actual_onnx_feed(self) -> None:
        keys = tuple(INPUT_SHAPES)
        names = tuple(key.replace(".", "_") for key in keys)
        context = ProtoMotionsGlobalContext(
            keys,
            names,
            INPUT_SHAPES.values(),
            ("bool" if key in BOOL_INPUTS else "float32" for key in keys),
        )
        context.set_many(inputs())
        feed = context.feed()
        views = {key: feed[name][0] for key, name in zip(keys, names)}
        self.recovery.step(views, pose(0.1, np.pi / 2), 0.02)
        # Test the batched, sanitized tensors handed to ORT, not just the
        # semantic dictionary mutated by the prompt owner.
        np.testing.assert_allclose(
            feed["masked_mimic_local_ref_pos"][0, :, 16, 2],
            0.1 + np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * 0.69,
            atol=1e-7,
        )
        self.assertTrue(feed["masked_mimic_local_target_poses_masks"].all())
        self.assertFalse(feed["scene_boxes_visible"].any())
        np.testing.assert_array_equal(
            context.feed()["masked_mimic_local_ref_pos"],
            feed["masked_mimic_local_ref_pos"],
        )

    def test_ground_proxy_is_translation_invariant(self) -> None:
        first = pose(0.1, np.pi / 2)
        translated = RecoveryPose(
            first.positions + [40, -10, 3],
            first.rotations,
            first.joints,
        )
        np.testing.assert_allclose(first.sensed(), translated.sensed(), atol=1e-6)

    def test_prone_clock_slows_without_changing_future_horizons(self) -> None:
        feed = inputs()
        fallen = pose(0.1, np.pi / 2)
        self.recovery.step(feed, fallen, 0.02)
        for _ in range(30):
            self.recovery.step(feed, fallen, 0.02)
        offsets = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_allclose(feed["masked_mimic.local_time_offsets"], offsets)
        np.testing.assert_allclose(
            feed["masked_mimic.local_ref_pos"][:, 16, 2],
            0.1 + np.minimum(0.6 / 3 + offsets, 1) * 0.69,
            atol=1e-7,
        )

    def test_mirroring_swaps_bodies_and_latches_from_measured_geometry(self) -> None:
        np.testing.assert_array_equal(
            self.recovery.mirrored_positions[:, 24],
            self.recovery.positions[:, 32] * [1, -1, 1],
        )
        np.testing.assert_array_equal(
            self.recovery.mirrored_rotations[:, 32],
            self.recovery.rotations[:, 24] * [-1, 1, -1, 1],
        )
        tucked = pose(0.1, np.pi / 2)
        tucked.positions[32, 1] = 0.02
        feed = inputs()
        self.recovery.step(feed, tucked, 0.02)
        self.assertTrue(self.recovery.mirror_prone)
        expected = np.empty((5, 33, 3), np.float32)
        rotations = np.empty((5, 33, 4), np.float32)
        self.recovery._sample_plan(expected, rotations)
        np.testing.assert_array_equal(feed["masked_mimic.local_ref_pos"], expected)
        self.recovery.step(feed, pose(0.1, np.pi / 2), 0.02)
        self.assertTrue(self.recovery.mirror_prone)
        self.recovery.reset()
        self.recovery.step(feed, pose(0.1, np.pi / 2), 0.02)
        self.assertFalse(self.recovery.mirror_prone)

    def test_mirror_selection_is_invariant_to_world_yaw_and_translation(self) -> None:
        from scipy.spatial.transform import Rotation

        original = pose(0.1, np.pi / 2)
        original.positions[32, 1] = 0.02
        for yaw in (-2.7, -0.8, 1.2, 2.4):
            rotation = Rotation.from_euler("z", yaw)
            moved = RecoveryPose(
                rotation.apply(original.positions) + [30, -5, 2],
                (rotation * Rotation.from_quat(original.rotations)).as_quat(),
                original.joints,
            )
            self.assertAlmostEqual(
                original.right_hand_lateral(), moved.right_hand_lateral()
            )
            self.recovery.reset()
            self.recovery.step(inputs(), moved, 0.02)
            self.assertTrue(self.recovery.mirror_prone)

    def test_partial_rise_requires_hold_then_settle(self) -> None:
        self.recovery.step(inputs(), pose(0.1, -np.pi / 2), 0.02)
        self.assertEqual(self.recovery.phase, "ramp")
        for _ in range(10):
            self.recovery.step(inputs(), pose(0.6), 0.02)
        self.assertEqual(self.recovery.phase, "ramp")
        self.recovery.step(inputs(), pose(0.6), 0.02)
        self.assertEqual(self.recovery.phase, "settle")
        for _ in range(25):
            self.recovery.step(inputs(), pose(0.79), 0.02)
        self.assertEqual(self.recovery.phase, "inactive")
        self.assertTrue(self.recovery.just_finished)

    def test_explicit_low_torso_goal_is_not_a_fall(self) -> None:
        feed = inputs()
        feed["masked_mimic.local_target_bodies_masks"].reshape(5, 5, 2)[:, 0, 0] = True
        feed["masked_mimic.local_target_poses_masks"].fill(True)
        feed["masked_mimic.local_ref_pos"][:, 16, 2] = 0.4
        self.assertFalse(self.recovery.step(feed, pose(0.4), 0.02))
        # Turning the torso onto its belly overrides the intentional-low guard.
        self.assertTrue(self.recovery.step(feed, pose(0.4, np.pi / 2), 0.02))

    def test_reset_and_unit_quaternion_interpolation(self) -> None:
        a = np.array([[0, 0, 0, 1]], np.float32)
        np.testing.assert_allclose(slerp(a, -a, 0.5), a)
        self.recovery.step(inputs(), pose(0.1, np.pi / 2), 0.02)
        self.recovery.reset()
        self.assertEqual(self.recovery.phase, "inactive")
        self.assertEqual(self.recovery.elapsed, 0)
        self.assertEqual(self.recovery.transitions, [])

    def test_crouch_release_raises_torso_instead_of_requesting_floor_pose(self) -> None:
        for pitch in (0, 0.75):
            with self.subTest(pitch=pitch):
                self.recovery.reset()
                crouching = pose(0.48, pitch)
                crouching.joints[[3, 9]] = 1.9
                feed = inputs()
                feed["masked_mimic.style_pose_samples"].reshape(5, 5, 9)[:, 0, 2] = 0.48
                feed["masked_mimic.style_condition_masks"][:, 0] = True
                feed["masked_mimic.style_frame_masks"].fill(True)
                self.assertFalse(self.recovery.step(feed, crouching, 0.02))
                feed["masked_mimic.style_frame_masks"].fill(False)
                self.assertTrue(self.recovery.step(feed, crouching, 0.02))
                self.assertEqual(self.recovery.phase, "ramp")
                masks = feed["masked_mimic.local_target_bodies_masks"].reshape(5, 5, 2)
                self.assertEqual(masks.sum(), 5)
                self.assertTrue(masks[:, 0, 0].all())
                np.testing.assert_allclose(
                    feed["masked_mimic.local_ref_pos"][:, 16, 2],
                    [0.64, 0.79, 0.79, 0.79, 0.79],
                    atol=1e-7,
                )

    def test_floor_templates_require_low_pose_and_prone_takes_precedence(self) -> None:
        for pitch, floor_phase in ((0, "seated_plan"), (0.75, "prone_plan")):
            for height, expected in ((0.349, floor_phase), (0.351, "ramp")):
                with self.subTest(pitch=pitch, height=height):
                    self.recovery.reset()
                    sitting = pose(height, pitch)
                    sitting.joints[[3, 9]] = 1.9
                    self.recovery.step(inputs(), sitting, 0.02)
                    self.assertEqual(self.recovery.phase, expected)

    def test_seated_plan_offsets_source_not_clock_and_requires_measured_rise(
        self,
    ) -> None:
        self.recovery.positions = np.zeros((8, 33, 3), np.float32)
        self.recovery.positions[:, 16, 2] = 0.1 + 0.1 * np.arange(8)
        self.recovery.rotations = np.tile([0, 0, 0, 1], (8, 33, 1)).astype(np.float32)
        sitting = pose(0.3)
        sitting.joints[[3, 9]] = 1.9
        feed = inputs()
        self.recovery.step(feed, sitting, 0.02)
        self.assertEqual(self.recovery.phase, "seated_plan")
        np.testing.assert_allclose(
            feed["masked_mimic.local_time_offsets"], [0.2, 0.4, 0.6, 0.8, 1]
        )
        self.assertAlmostEqual(
            float(feed["masked_mimic.local_ref_pos"][0, 16, 2]), 0.57
        )
        for _ in range(150):
            self.recovery.step(feed, sitting, 0.02)
        self.assertEqual(self.recovery.phase, "seated_plan")
        self.assertAlmostEqual(
            float(feed["masked_mimic.local_ref_pos"][0, 16, 2]), 0.67
        )
        for _ in range(1000):
            self.recovery.step(feed, sitting, 0.02)
        self.assertEqual(self.recovery.phase, "seated_plan")
        self.assertAlmostEqual(float(feed["masked_mimic.local_ref_pos"][0, 16, 2]), 0.8)
        for _ in range(11):
            self.recovery.step(feed, pose(0.6), 0.02)
        self.assertEqual(self.recovery.phase, "settle")


if __name__ == "__main__":
    unittest.main()
