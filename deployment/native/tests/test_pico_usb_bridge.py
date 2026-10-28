# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Pico USB/ADB transport helper."""

from __future__ import annotations

import unittest

from tools.pico_usb_bridge import (
    AdbDevice,
    parse_adb_devices,
    reverse_is_configured,
    select_adb_device,
)


class PicoUsbBridgeTest(unittest.TestCase):
    def test_parse_and_select_one_authorized_device(self) -> None:
        devices = parse_adb_devices(
            "List of devices attached\n"
            "PICO123 device product:PICO model:PICO_4 transport_id:1\n"
        )
        self.assertEqual(
            select_adb_device(devices),
            AdbDevice(
                "PICO123",
                "device",
                "product:PICO model:PICO_4 transport_id:1",
            ),
        )

    def test_unauthorized_device_has_actionable_error(self) -> None:
        devices = parse_adb_devices(
            "List of devices attached\nPICO unauthorized usb:1-2\n"
        )
        with self.assertRaisesRegex(RuntimeError, "accept the USB debugging"):
            select_adb_device(devices)

    def test_no_permissions_device_has_udev_action(self) -> None:
        devices = parse_adb_devices(
            "List of devices attached\n"
            "PICO no permissions (missing udev rules?); see help\n"
        )
        self.assertEqual(devices[0].state, "no permissions")
        with self.assertRaisesRegex(RuntimeError, "51-pico-adb.rules"):
            select_adb_device(devices)

    def test_multiple_devices_require_a_serial(self) -> None:
        devices = [AdbDevice("one", "device"), AdbDevice("two", "device")]
        with self.assertRaisesRegex(RuntimeError, "pass --serial"):
            select_adb_device(devices)
        self.assertEqual(select_adb_device(devices, "two").serial, "two")

    def test_reverse_listing_accepts_global_and_serial_formats(self) -> None:
        self.assertTrue(reverse_is_configured("PICO tcp:63901 tcp:63901\n", "PICO"))
        self.assertTrue(reverse_is_configured("UsbFfs tcp:63901 tcp:63901\n", "PICO"))
        self.assertTrue(reverse_is_configured("tcp:63901 tcp:63901\n", "PICO"))
        self.assertFalse(reverse_is_configured("PICO tcp:60061 tcp:60061\n", "PICO"))


if __name__ == "__main__":
    unittest.main()
