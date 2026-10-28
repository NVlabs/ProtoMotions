# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin state/action timing, including edited actions, reset and buffer reuse."""

import unittest

import numpy as np

from examples.metatron_v1_contract import (
    BOOL_INPUTS,
    CONTRACT,
    INPUT_SHAPES,
    IncomingActionHistory,
    SensorReading,
    hidden_scene_inputs,
    validate_manifest,
)


def reading(t: float) -> SensorReading:
    return SensorReading(
        dof_pos=np.full(29, t, np.float32),
        dof_vel=np.full(29, t + 100, np.float32),
        root_local_ang_vel=np.full(3, t + 200, np.float32),
        anchor_rot=np.array([0, 0, 0, 1], np.float32),
        anchor_pos=np.array([0, 0, t], np.float32),
        root_pos=np.array([0, 0, t], np.float32),
        root_rot=np.array([0, 0, 0, 1], np.float32),
    )


class SceneHistoryTest(unittest.TestCase):
    def test_manifest_rejects_reordered_recovery_bodies_and_knees(self) -> None:
        bodies = [f"body_{i}" for i in range(33)]
        for index, name in (
            (7, "left_ankle_roll_link"),
            (13, "right_ankle_roll_link"),
            (16, "torso_link"),
            (24, "left_rubber_hand"),
            (32, "right_rubber_hand"),
        ):
            bodies[index] = name
        joints = [f"joint_{i}" for i in range(29)]
        joints[3], joints[9] = "left_knee_joint", "right_knee_joint"
        manifest = {
            "contract": CONTRACT,
            "input_keys": list(INPUT_SHAPES),
            "input_names": [key.replace(".", "_") for key in INPUT_SHAPES],
            "input_shapes": list(INPUT_SHAPES.values()),
            "input_dtypes": [
                "bool" if key in BOOL_INPUTS else "float32" for key in INPUT_SHAPES
            ],
            "robot_contract": {
                "quaternion_order": "xyzw",
                "anchor_body_index": 16,
                "conditionable_body_ids": [16, 13, 7, 24, 32],
                "body_names": bodies,
                "joint_names": joints,
            },
            "previous_action_domain": "raw_policy_action",
            "action_map": {"kind": "affine", "scale": [0.25] * 29, "offset": [0] * 29},
        }
        validate_manifest(manifest)
        bodies[7], bodies[13] = bodies[13], bodies[7]
        with self.assertRaisesRegex(ValueError, "body ordering"):
            validate_manifest(manifest)
        bodies[7], bodies[13] = bodies[13], bodies[7]
        joints[3], joints[9] = joints[9], joints[3]
        with self.assertRaisesRegex(ValueError, "knee ordering"):
            validate_manifest(manifest)

    def test_current_incoming_action_and_previous_state_pairs(self) -> None:
        history = IncomingActionHistory()
        for t in range(12):
            state = reading(t)
            feed = history.inputs(state)
            expected_actions = np.array([max(t - k, 0) for k in range(9)])
            np.testing.assert_array_equal(
                feed["raw_action_history"][:, 0], expected_actions
            )
            np.testing.assert_array_equal(
                feed["noisy_historical.raw_actions"][:, 0], expected_actions[1:]
            )
            np.testing.assert_array_equal(
                feed["noisy_historical.dof_pos"][:, 0],
                [max(t - k - 1, 0) for k in range(8)],
            )
            history.commit(state, np.full(29, t + 1, np.float32))
            state.dof_pos.fill(-999)  # Simulate a bridge reusing its sensor buffers.

    def test_repeated_queries_do_not_advance_and_reset_has_zero_actions(self) -> None:
        history = IncomingActionHistory()
        state = reading(4)
        for _ in range(3):
            feed = history.inputs(state)
            np.testing.assert_array_equal(feed["raw_action_history"], np.zeros((9, 29)))
            np.testing.assert_array_equal(
                feed["noisy_historical.dof_pos"], np.full((8, 29), 4)
            )
        history.commit(state, np.ones(29))
        history.reset()
        feed = history.inputs(reading(9))
        np.testing.assert_array_equal(
            feed["noisy_historical.dof_pos"], np.full((8, 29), 9)
        )
        self.assertFalse(feed["raw_action_history"].any())

    def test_no_object_means_masked_and_identity_orientation(self) -> None:
        scene = hidden_scene_inputs()
        self.assertFalse(scene["scene_boxes.visible"].any())
        self.assertFalse(scene["scene_boxes.valid"].any())
        np.testing.assert_array_equal(scene["scene_boxes.object_rot"], [[0, 0, 0, 1]])

    def test_transport_ack_does_not_rewrite_recorded_inference_inputs(self) -> None:
        history = IncomingActionHistory()
        state = reading(0)
        recorded = history.inputs(state)
        history.commit(state, np.ones(29))
        self.assertFalse(recorded["raw_action_history"].any())
        self.assertFalse(recorded["noisy_historical.raw_actions"].any())
        self.assertTrue(history.inputs(reading(1))["raw_action_history"][0].all())


if __name__ == "__main__":
    unittest.main()
