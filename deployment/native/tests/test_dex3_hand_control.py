# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the Pico-to-Dex-3 safety boundary."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from examples.dex3_hand_control import (
    DEX3_CLOSE_LEFT,
    DEX3_CLOSE_RIGHT,
    DEX3_OPEN,
    PicoDex3HandController,
    dex3_targets,
)
from examples.pico_input import _read_finger_controls


class _FakeController:
    def __init__(self, connected: bool = True) -> None:
        self.connected = connected
        self.commands = []

    def hand_self_check(self) -> bool:
        return self.connected

    def step_hands(self, left, right) -> None:
        self.commands.append((tuple(left), tuple(right)))


class Dex3HandControlTest(unittest.TestCase):
    def test_targets_match_calibrated_open_and_midpoint_close(self) -> None:
        left, right, left_closed, right_closed = dex3_targets(1.0, 0.0, enabled=True)
        self.assertEqual(left, DEX3_CLOSE_LEFT)
        self.assertEqual(right, DEX3_OPEN)
        self.assertTrue(left_closed)
        self.assertFalse(right_closed)

        left, right, left_closed, right_closed = dex3_targets(0.0, 0.5, enabled=True)
        self.assertEqual(left, DEX3_OPEN)
        self.assertEqual(right, DEX3_CLOSE_RIGHT)
        self.assertFalse(left_closed)
        self.assertTrue(right_closed)

    def test_commands_are_suppressed_until_both_hand_states_are_verified(self) -> None:
        controller = _FakeController(connected=False)
        adapter = PicoDex3HandController(controller)
        status = adapter.update(
            SimpleNamespace(left_trigger=1.0, right_trigger=1.0), enabled=True
        )
        self.assertFalse(status.connected)
        self.assertEqual(controller.commands, [])

        controller.connected = True
        status = adapter.update(
            SimpleNamespace(left_trigger=1.0, right_trigger=0.0),
            enabled=True,
            allow_commands=False,
        )
        self.assertTrue(status.connected)
        self.assertFalse(status.sent)
        self.assertEqual(controller.commands, [])

        status = adapter.update(
            SimpleNamespace(
                left_trigger=1.0,
                right_trigger=0.0,
                left_grip=1.0,
                right_grip=0.0,
            ),
            enabled=True,
        )
        self.assertTrue(status.connected)
        self.assertTrue(status.sent)
        self.assertEqual(controller.commands[-1], (DEX3_CLOSE_LEFT, DEX3_OPEN))

        # The same command is not reallocated/re-sent every 50 Hz policy step.
        status = adapter.update(
            SimpleNamespace(
                left_trigger=1.0,
                right_trigger=0.0,
                left_grip=1.0,
                right_grip=0.0,
            ),
            enabled=True,
        )
        self.assertFalse(status.sent)
        self.assertEqual(len(controller.commands), 1)

        # Releasing the pose clutch freezes both the arm target and the last
        # open/closed finger state.
        status = adapter.update(
            SimpleNamespace(
                left_trigger=0.0,
                right_trigger=0.0,
                left_grip=0.0,
                right_grip=0.0,
            ),
            enabled=True,
        )
        self.assertTrue(status.left_closed)
        self.assertFalse(status.sent)
        self.assertEqual(controller.commands[-1], (DEX3_CLOSE_LEFT, DEX3_OPEN))

        # Re-clutching with the trigger released explicitly opens the hand.
        status = adapter.update(
            SimpleNamespace(
                left_trigger=0.0,
                right_trigger=0.0,
                left_grip=1.0,
                right_grip=0.0,
            ),
            enabled=True,
        )
        self.assertFalse(status.left_closed)
        self.assertTrue(status.sent)
        self.assertEqual(controller.commands[-1], (DEX3_OPEN, DEX3_OPEN))

        # Put the hand back in a closed state so disabling control must issue
        # an explicit open target.
        status = adapter.update(
            SimpleNamespace(
                left_trigger=1.0,
                right_trigger=0.0,
                left_grip=1.0,
                right_grip=0.0,
            ),
            enabled=True,
        )
        self.assertTrue(status.left_closed)
        self.assertTrue(status.sent)

        # Disabling Pico hand control explicitly opens both hands.
        status = adapter.update(
            SimpleNamespace(
                left_trigger=1.0,
                right_trigger=1.0,
                left_grip=1.0,
                right_grip=1.0,
            ),
            enabled=False,
        )
        self.assertTrue(status.sent)
        self.assertEqual(controller.commands[-1], (DEX3_OPEN, DEX3_OPEN))

    def test_xrt_finger_controls_are_read_in_left_right_order(self) -> None:
        xrt = SimpleNamespace(
            get_left_trigger=lambda: 0.7,
            get_right_trigger=lambda: 0.2,
            get_left_grip=lambda: 0.4,
            get_right_grip=lambda: 0.9,
        )
        self.assertEqual(_read_finger_controls(xrt), (0.7, 0.2, 0.4, 0.9))


if __name__ == "__main__":
    unittest.main()
