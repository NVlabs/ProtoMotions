# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Pico XRoboToolkit headset transport available over USB/ADB."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


HEADSET_SERVICE_PORT = 63901
LOCAL_SDK_PORT = 60061


@dataclass(frozen=True)
class AdbDevice:
    serial: str
    state: str
    details: str = ""


def parse_adb_devices(output: str) -> list[AdbDevice]:
    """Parse ``adb devices -l`` without depending on its optional columns."""

    devices: list[AdbDevice] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices") or line.startswith("*"):
            continue
        columns = line.split(maxsplit=2)
        if len(columns) < 2:
            continue
        state = columns[1]
        details = columns[2] if len(columns) > 2 else ""
        if state == "no" and details.startswith("permissions"):
            state = "no permissions"
        devices.append(
            AdbDevice(
                serial=columns[0],
                state=state,
                details=details,
            )
        )
    return devices


def select_adb_device(
    devices: list[AdbDevice], requested_serial: str | None = None
) -> AdbDevice:
    """Select exactly one authorized USB device and explain unsafe ambiguity."""

    if requested_serial is not None:
        matches = [device for device in devices if device.serial == requested_serial]
        if not matches:
            raise RuntimeError(f"ADB device {requested_serial!r} was not found")
        selected = matches[0]
        if selected.state != "device":
            if selected.state == "no permissions":
                raise RuntimeError(
                    f"ADB device {selected.serial} has no USB permissions; "
                    "install tools/51-pico-adb.rules under /etc/udev/rules.d, "
                    "reload udev, then reconnect the headset"
                )
            raise RuntimeError(
                f"ADB device {selected.serial} is {selected.state!r}; unlock the "
                "headset and accept the USB debugging prompt"
            )
        return selected

    authorized = [device for device in devices if device.state == "device"]
    if len(authorized) == 1:
        return authorized[0]
    if len(authorized) > 1:
        serials = ", ".join(device.serial for device in authorized)
        raise RuntimeError(
            f"multiple authorized ADB devices are attached ({serials}); pass --serial"
        )
    if devices:
        if any(device.state == "no permissions" for device in devices):
            raise RuntimeError(
                "Pico ADB USB permissions are missing; install "
                "tools/51-pico-adb.rules under /etc/udev/rules.d, reload "
                "udev, then reconnect the headset"
            )
        states = ", ".join(f"{device.serial}:{device.state}" for device in devices)
        raise RuntimeError(
            f"no authorized ADB device ({states}); unlock the headset and accept "
            "the USB debugging prompt"
        )
    raise RuntimeError(
        "no ADB device detected; connect the Pico over USB with Developer Mode "
        "and USB debugging enabled"
    )


def reverse_is_configured(output: str, serial: str) -> bool:
    """Return whether an ADB reverse listing contains the headset TCP tunnel."""

    port = f"tcp:{HEADSET_SERVICE_PORT}"
    for line in output.splitlines():
        columns = line.split()
        # Depending on the ADB transport, the first column may be the device
        # serial or a transport label such as ``UsbFfs``. The local/remote port
        # pair is the stable part of the reverse-list contract.
        if len(columns) >= 2 and columns[-2:] == [port, port]:
            return True
    return False


def linux_tcp_port_is_listening(port: int) -> bool:
    """Check Linux listen tables without opening a protocol connection."""

    target = f"{int(port):04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            columns = line.split()
            if len(columns) > 3 and columns[1].rsplit(":", 1)[-1] == target:
                if columns[3] == "0A":
                    return True
    return False


class PicoUsbBridge:
    """Configure and monitor one USB reverse tunnel without using the XRT SDK."""

    def __init__(self, adb: str, serial: str | None = None) -> None:
        resolved = shutil.which(adb) if "/" not in adb else adb
        if not resolved or not Path(resolved).is_file():
            raise RuntimeError(
                f"ADB executable {adb!r} was not found; install android-tools-adb"
            )
        self.adb = resolved
        self.serial = serial

    def _run(self, arguments: list[str]) -> str:
        result = subprocess.run(
            [self.adb, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"{' '.join(arguments)} failed" + (f": {detail}" if detail else "")
            )
        return result.stdout

    def ensure(self) -> AdbDevice:
        if not linux_tcp_port_is_listening(
            LOCAL_SDK_PORT
        ) or not linux_tcp_port_is_listening(HEADSET_SERVICE_PORT):
            raise RuntimeError(
                "XRoboToolkit PC service is not listening on ports 60061 and "
                "63901; start/restart RoboticsServiceProcess"
            )
        devices = parse_adb_devices(self._run(["devices", "-l"]))
        device = select_adb_device(devices, self.serial)
        prefix = ["-s", device.serial]
        listing = self._run(prefix + ["reverse", "--list"])
        if not reverse_is_configured(listing, device.serial):
            self._run(
                prefix
                + [
                    "reverse",
                    f"tcp:{HEADSET_SERVICE_PORT}",
                    f"tcp:{HEADSET_SERVICE_PORT}",
                ]
            )
            listing = self._run(prefix + ["reverse", "--list"])
            if not reverse_is_configured(listing, device.serial):
                raise RuntimeError(
                    "ADB accepted the command but did not list the reverse tunnel"
                )
        return device


def print_service_status() -> None:
    sdk = linux_tcp_port_is_listening(LOCAL_SDK_PORT)
    headset = linux_tcp_port_is_listening(HEADSET_SERVICE_PORT)
    print(
        "XRoboToolkit PC service: "
        f"SDK 127.0.0.1:{LOCAL_SDK_PORT}={'ready' if sdk else 'missing'}, "
        f"headset TCP :{HEADSET_SERVICE_PORT}={'ready' if headset else 'missing'}",
        flush=True,
    )
    if not sdk or not headset:
        raise RuntimeError(
            "XRoboToolkit PC service is not listening on both required ports; "
            "start/restart RoboticsServiceProcess before the deployment"
        )


def run_watch(bridge: PicoUsbBridge, interval_seconds: float) -> None:
    last_status: str | None = None
    while True:
        try:
            device = bridge.ensure()
            status = f"ready:{device.serial}"
            if status != last_status:
                print(
                    f"Pico USB bridge ready for {device.serial}: device "
                    f"tcp:{HEADSET_SERVICE_PORT} -> host tcp:{HEADSET_SERVICE_PORT}",
                    flush=True,
                )
                print(
                    "In the Pico app use PC Service IP 127.0.0.1 and press "
                    "Reconnect. If USB network sharing is enabled and the app "
                    "offers a discovered 192.168.255.x PC address, selecting "
                    "that address is also valid. Enable Head Tracking, "
                    "Controller Tracking, and Send; disable Switch w/ A Button.",
                    flush=True,
                )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            status = f"waiting:{exc}"
            if status != last_status:
                print(f"Pico USB bridge waiting: {exc}", flush=True)
        last_status = status
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expose the XRoboToolkit PC service to a USB-connected Pico"
    )
    parser.add_argument("--adb", default="adb", help="ADB executable")
    parser.add_argument(
        "--serial", help="ADB serial when multiple devices are attached"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep the tunnel configured across USB disconnects",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="watch interval in seconds (default: 1)",
    )
    args = parser.parse_args()
    if args.interval <= 0.0:
        parser.error("--interval must be positive")

    bridge = PicoUsbBridge(args.adb, args.serial)
    if args.watch:
        try:
            run_watch(bridge, args.interval)
        except KeyboardInterrupt:
            print("Pico USB bridge stopped", flush=True)
        return
    print_service_status()
    try:
        device = bridge.ensure()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Pico USB bridge error: {exc}\n")
    print(
        f"Pico USB bridge ready for {device.serial}: use 127.0.0.1 and press "
        "Reconnect, or select the 192.168.255.x address advertised by USB "
        "network sharing; then enable Head + Controller + Send.",
        flush=True,
    )


if __name__ == "__main__":
    main()
