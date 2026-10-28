# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from examples.metatron_goals import MetatronGoals
from examples.metatron_v1_contract import BOOL_INPUTS, INPUT_SHAPES


class MetatronGoalsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = {
            key: np.zeros(shape, bool if key in BOOL_INPUTS else np.float32)
            for key, shape in INPUT_SHAPES.items()
        }
        for name in (
            "target_bodies_masks",
            "target_poses_masks",
            "anchor_translation_mask",
            "anchor_rotation_mask",
        ):
            self.inputs["masked_mimic." + name].fill(True)
        self.inputs["masked_mimic.time_offsets"][:] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.goals = MetatronGoals(
            np.full((33, 3), 2, np.float32),
            np.tile([0, 0, 0, 1], (33, 1)),
            np.ones(225, np.float32),
        )

    def test_tracking_has_no_world_translation_goal(self) -> None:
        self.goals.apply(self.inputs, "tracking", False)
        p = "masked_mimic."
        local = self.inputs[p + "local_target_bodies_masks"].reshape(5, 5, 2)
        self.assertTrue(local.all())
        global_masks = self.inputs[p + "target_bodies_masks"].reshape(5, 5, 2)
        self.assertEqual(global_masks.sum(), 5)
        self.assertTrue(global_masks[:, 0, 1].all())
        self.assertFalse(self.inputs[p + "anchor_translation_mask"].any())
        self.assertFalse(self.inputs[p + "anchor_rotation_mask"][:, 0].any())

    def test_moving_hands_stay_sparse(self) -> None:
        p = "masked_mimic."
        masks = self.inputs[p + "target_bodies_masks"].reshape(5, 5, 2)
        masks[:, 3:, :] = False
        masks[-1, 3:, 0] = True
        self.goals.apply(self.inputs, "steering", False)
        local = self.inputs[p + "local_target_bodies_masks"].reshape(5, 5, 2)
        self.assertEqual(local[:, 3:].sum(), 2)
        self.assertTrue(local[-1, 3:, 0].all())
        self.assertFalse(self.inputs[p + "style_frame_masks"].any())

    def test_idle_uses_fixed_template_and_zero_velocity_style(self) -> None:
        self.goals.apply(self.inputs, "steering", True)
        p = "masked_mimic."
        np.testing.assert_array_equal(
            self.inputs[p + "local_ref_pos"], np.full((5, 33, 3), 2)
        )
        self.assertTrue(self.inputs[p + "style_condition_masks"].all())
        self.assertFalse(self.inputs[p + "style_velocity_samples"].any())
        self.assertFalse(self.inputs[p + "average_velocity"].any())
        self.assertFalse(self.inputs[p + "anchor_translation_mask"].any())
        self.assertTrue(self.inputs[p + "velocity_command_mask"].all())

    def test_odometry_remains_explicit_global_position_task(self) -> None:
        before = {key: value.copy() for key, value in self.inputs.items()}
        self.goals.apply(self.inputs, "odometry", False)
        for key, value in before.items():
            np.testing.assert_array_equal(self.inputs[key], value)


if __name__ == "__main__":
    unittest.main()
