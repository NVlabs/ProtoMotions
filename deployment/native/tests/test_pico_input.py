# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure tests for the optional PICO/XRoboToolkit input contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.pico_input import (
    PicoInput,
    body_poses_to_local_points,
    controller_poses_to_frame_data,
    controller_poses_to_frame_points,
    controller_poses_to_local_points,
    headset_pose_to_robot_yaw,
    normalize_body_poses,
)


class FakeXrt:
    def __init__(self) -> None:
        self.state_count = 0
        self.closed = False

    def init(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def get_state_update_count(self) -> int:
        return self.state_count

    def get_time_stamp_ns(self) -> int:
        return self.state_count * 1_000_000

    def get_headset_pose(self) -> list[float]:
        return [0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0]

    def get_left_controller_pose(self) -> list[float]:
        return [-0.25, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0]

    def get_right_controller_pose(self) -> list[float]:
        return [0.25, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0]

    def is_body_data_available(self) -> bool:
        return False

    def get_left_trigger(self) -> float:
        return 0.0

    def get_right_trigger(self) -> float:
        return 0.0

    def get_left_grip(self) -> float:
        return 0.0

    def get_right_grip(self) -> float:
        return 0.0


class PicoInputContractTest(unittest.TestCase):
    def test_unity_positions_are_converted_to_root_local_robot_axes(self) -> None:
        poses = np.zeros((24, 7), dtype=np.float32)
        poses[0, 6] = 1.0
        poses[22, :3] = [1.0, 2.0, 3.0]
        poses[23, :3] = [-1.0, 2.0, 3.0]
        poses[12, :3] = [0.0, 1.0, 2.0]
        hands, neck = body_poses_to_local_points(poses)

        # Unity [x, y, z] maps to robot [-x, z, y], then the reference root
        # correction rotates the local XY frame by -90 degrees.
        np.testing.assert_allclose(
            hands, [[-3.0, -1.0, 2.0], [-3.0, 1.0, 2.0]], atol=1e-6
        )
        np.testing.assert_allclose(neck, [-2.0, 0.0, 1.0], atol=1e-6)

    def test_controller_fallback_uses_headset_local_robot_axes(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        left = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        right = [-1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        hands, neck = controller_poses_to_local_points(left, right, identity)
        np.testing.assert_allclose(
            hands, [[-3.0, -1.0, 2.0], [-3.0, 1.0, 2.0]], atol=1e-6
        )
        np.testing.assert_allclose(neck, 0.0)

    def test_controller_frame_maps_physical_front_and_left_right_axes(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # XRoboToolkit uses its headset-space axes. In the calibrated Pico
        # basis, Unity -Z is robot forward and Unity -X/+X are robot
        # left/right respectively.
        left = [-0.2, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        right = [0.2, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        hands_local, hands_world = controller_poses_to_frame_points(
            left, right, identity
        )
        np.testing.assert_allclose(hands_local, [[1.0, 0.2, 1.0], [1.0, -0.2, 1.0]])
        np.testing.assert_allclose(hands_world, hands_local)

    def test_controller_frame_preserves_left_right_and_floor_height(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        left = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        right = [-1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        hands_local, hands_world = controller_poses_to_frame_points(
            left, right, identity
        )
        # The first row is always the left controller and the second the
        # right controller. Unity Y is the floor-frame robot Z coordinate.
        np.testing.assert_allclose(
            hands_local, [[-3.0, -1.0, 2.0], [-3.0, 1.0, 2.0]], atol=1e-6
        )
        np.testing.assert_allclose(
            hands_world, [[-3.0, -1.0, 2.0], [-3.0, 1.0, 2.0]], atol=1e-6
        )

    def test_controller_frame_returns_left_right_yaw_local_rotations(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        hands, _, rotations, yaw = controller_poses_to_frame_data(
            identity, identity, identity
        )
        self.assertEqual(hands.shape, (2, 3))
        self.assertEqual(rotations.shape, (2, 4))
        self.assertAlmostEqual(yaw, 0.0)
        np.testing.assert_allclose(
            rotations,
            np.broadcast_to([0.0, 0.0, 0.0, 1.0], (2, 4)),
            atol=1e-6,
        )

    def test_controller_rotation_removes_shared_headset_yaw(self) -> None:
        angle = np.deg2rad(30.0)
        shared_yaw = [
            0.0,
            0.0,
            0.0,
            0.0,
            np.sin(angle / 2.0),
            0.0,
            np.cos(angle / 2.0),
        ]
        _, _, rotations, _ = controller_poses_to_frame_data(
            shared_yaw, shared_yaw, shared_yaw
        )
        np.testing.assert_allclose(
            rotations,
            np.broadcast_to([0.0, 0.0, 0.0, 1.0], (2, 4)),
            atol=1e-6,
        )

    def test_controller_xy_frame_ignores_headset_pitch_and_roll(self) -> None:
        pitched_headset = [0.0, 0.0, 0.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
        left = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        right = [-1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        hands_local, _ = controller_poses_to_frame_points(left, right, pitched_headset)
        # Headset pitch/roll is not allowed to tilt the planar torso frame.
        np.testing.assert_allclose(
            hands_local, [[-3.0, -1.0, 2.0], [-3.0, 1.0, 2.0]], atol=1e-6
        )

    def test_headset_yaw_is_reported_in_robot_heading_convention(self) -> None:
        angle = np.deg2rad(30.0)
        headset = [
            0.0,
            0.0,
            0.0,
            0.0,
            np.sin(angle / 2.0),
            0.0,
            np.cos(angle / 2.0),
        ]
        self.assertAlmostEqual(headset_pose_to_robot_yaw(headset), angle, places=6)

    def test_missing_required_pose_is_rejected(self) -> None:
        poses = np.zeros((24, 7), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "root"):
            normalize_body_poses(poses)

        poses[0, 6] = 1.0
        with self.assertRaisesRegex(ValueError, "left wrist"):
            normalize_body_poses(poses)

    def test_feet_are_not_required(self) -> None:
        poses = np.zeros((24, 7), dtype=np.float32)
        poses[0, 6] = 1.0
        poses[22, :3] = [0.2, 1.0, 0.5]
        poses[23, :3] = [-0.2, 1.0, 0.5]
        # Indices 7, 8, 10, and 11 are deliberately left untracked/zero.
        hands, _ = body_poses_to_local_points(poses)
        self.assertTrue(np.all(np.isfinite(hands)))

    def test_startup_distinguishes_local_sdk_from_headset_packets(self) -> None:
        xrt = FakeXrt()
        with tempfile.TemporaryDirectory() as directory:
            pico = PicoInput(
                xrt_module=xrt,
                client_lock_path=Path(directory) / "pico.lock",
            )
            try:
                self.assertIsNone(pico.wait_for_sample(timeout_seconds=0.03))
                self.assertIn("no headset state packet", pico.status_message())
                xrt.state_count = 1
                sample = pico.wait_for_sample(timeout_seconds=0.5)
                self.assertIsNotNone(sample)
                assert sample is not None
                self.assertTrue(sample.teleop_pose_valid)
                self.assertIn("source=controllers", pico.status_message())
            finally:
                pico.close()
        self.assertTrue(xrt.closed)

    def test_only_one_process_client_lease_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "pico.lock"
            first = PicoInput(xrt_module=FakeXrt(), client_lock_path=lock_path)
            try:
                with self.assertRaisesRegex(RuntimeError, "another ProtoMotions"):
                    PicoInput(xrt_module=FakeXrt(), client_lock_path=lock_path)
            finally:
                first.close()
            replacement = PicoInput(xrt_module=FakeXrt(), client_lock_path=lock_path)
            replacement.close()


if __name__ == "__main__":
    unittest.main()
