# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Print raw key events from an evdev device with timestamps.

A second, independent reader of the same device: if a bumper press shows up
here but not in the launcher's trace, the launcher lost it; if it is missing
here too, the controller never sent it.
"""
import struct
import sys
import time

DEVICE = sys.argv[1] if len(sys.argv) > 1 else (
    "/dev/input/by-id/usb-PowerA_Xbox_Series_X_Adv_HE_wired_Controller_Fortnite_Glow_000001014AD9654E-event-joystick")
EV_KEY, EV_SYN, SYN_DROPPED = 0x01, 0x00, 0x03
NAMES = {0x130: "A", 0x131: "B", 0x133: "X", 0x134: "Y", 0x136: "LB", 0x137: "RB", 0x13a: "View", 0x13b: "Menu"}
FORMAT = "llHHi"
SIZE = struct.calcsize(FORMAT)

start = time.monotonic()
with open(DEVICE, "rb", buffering=0) as device:
    print(f"reading {DEVICE}; press buttons, Ctrl-C to stop")
    while True:
        data = device.read(SIZE)
        if len(data) < SIZE:
            continue
        _, _, kind, code, value = struct.unpack(FORMAT, data)
        stamp = time.monotonic() - start
        if kind == EV_KEY:
            print(f"{stamp:8.3f}s  {NAMES.get(code, hex(code)):>5}  {'down' if value == 1 else 'up' if value == 0 else 'repeat'}")
        elif kind == EV_SYN and code == SYN_DROPPED:
            print(f"{stamp:8.3f}s  *** kernel dropped events ***")
