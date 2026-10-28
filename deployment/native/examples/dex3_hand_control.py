# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe Pico trigger to Unitree Dex-3 hand command adapter.

The target poses use the calibrated Dex-3 hand command contract. This module
contains no SDK imports, so the mapping is reusable in MuJoCo tests and with a
different robot transport.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


DEX3_MOTOR_COUNT = 7
PICO_TRIGGER_THRESHOLD = 0.5

DEX3_OPEN = (0.0,) * DEX3_MOTOR_COUNT
DEX3_CLOSE_LEFT = (0.0, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875)
DEX3_CLOSE_RIGHT = (0.0, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875)


def _trigger_is_pressed(value: Any, threshold: float) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= threshold
    except (TypeError, ValueError):
        return False


def _finger_closed_after_sample(
    previous: bool,
    trigger: Any,
    grip: Any,
    *,
    enabled: bool,
    threshold: float,
) -> bool:
    """Update a finger latch only while that hand's pose clutch is held."""

    if not enabled:
        return False
    if not _trigger_is_pressed(grip, threshold):
        return previous
    return _trigger_is_pressed(trigger, threshold)


def dex3_targets(
    left_trigger: Any,
    right_trigger: Any,
    *,
    enabled: bool,
    threshold: float = PICO_TRIGGER_THRESHOLD,
) -> tuple[tuple[float, ...], tuple[float, ...], bool, bool]:
    """Build deterministic open/closed targets for one Pico sample."""

    left_closed = bool(enabled and _trigger_is_pressed(left_trigger, threshold))
    right_closed = bool(enabled and _trigger_is_pressed(right_trigger, threshold))
    return (
        DEX3_CLOSE_LEFT if left_closed else DEX3_OPEN,
        DEX3_CLOSE_RIGHT if right_closed else DEX3_OPEN,
        left_closed,
        right_closed,
    )


@dataclass(frozen=True)
class Dex3HandStatus:
    """Observable hand transport state for logs and time-series recording."""

    connected: bool
    enabled: bool
    left_trigger: float
    right_trigger: float
    left_grip: float
    right_grip: float
    left_closed: bool
    right_closed: bool
    left_target: tuple[float, ...]
    right_target: tuple[float, ...]
    sent: bool
    error: str | None = None


class PicoDex3HandController:
    """Gate Dex-3 commands on live hand state and the Pico pose clutch.

    A trigger updates its matching hand only while that controller's grip is
    held. Releasing the grip preserves the last open/closed state until the
    hand is clutched again or Pico hand control is disabled.
    With ``latch_on_grip=False`` the reference profile uses the triggers
    directly, without a grip requirement. Transport and safety gates are shared.

    The SDK repeats the last accepted target in its low-rate hand writer. We
    therefore only send when the open/closed state changes, while checking the
    connection on every control cycle. A disconnected or disabled transport
    cannot publish a new hand command.
    """

    def __init__(
        self,
        sdk_controller: Any,
        *,
        trigger_threshold: float = PICO_TRIGGER_THRESHOLD,
        latch_on_grip: bool = True,
    ) -> None:
        self._controller = sdk_controller
        self._threshold = float(trigger_threshold)
        self._latch_on_grip = bool(latch_on_grip)
        if not math.isfinite(self._threshold) or not 0.0 < self._threshold < 1.0:
            raise ValueError("Dex-3 trigger threshold must be in (0, 1)")
        self._last_targets: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        self._closed = [False, False]

    def update(
        self,
        snapshot: Any | None,
        *,
        enabled: bool,
        allow_commands: bool = True,
    ) -> Dex3HandStatus:
        left_trigger = (
            float(getattr(snapshot, "left_trigger", 0.0))
            if snapshot is not None
            else 0.0
        )
        right_trigger = (
            float(getattr(snapshot, "right_trigger", 0.0))
            if snapshot is not None
            else 0.0
        )
        left_grip = (
            float(getattr(snapshot, "left_grip", 0.0)) if snapshot is not None else 0.0
        )
        right_grip = (
            float(getattr(snapshot, "right_grip", 0.0)) if snapshot is not None else 0.0
        )
        try:
            connected = bool(self._controller.hand_self_check())
            error = None
        except Exception as exc:
            connected = False
            error = f"{type(exc).__name__}: {exc}"

        if not connected:
            # Force a resend after reconnection instead of assuming the SDK
            # still has a valid target from before the disconnect.
            self._last_targets = None
            self._closed = [False, False]
            return Dex3HandStatus(
                connected=False,
                enabled=False,
                left_trigger=left_trigger,
                right_trigger=right_trigger,
                left_grip=left_grip,
                right_grip=right_grip,
                left_closed=False,
                right_closed=False,
                left_target=DEX3_OPEN,
                right_target=DEX3_OPEN,
                sent=False,
                error=error,
            )

        if not allow_commands:
            self._last_targets = None
            self._closed = [False, False]
            left, right, left_closed, right_closed = (
                DEX3_OPEN,
                DEX3_OPEN,
                False,
                False,
            )
            return Dex3HandStatus(
                connected=True,
                enabled=False,
                left_trigger=left_trigger,
                right_trigger=right_trigger,
                left_grip=left_grip,
                right_grip=right_grip,
                left_closed=left_closed,
                right_closed=right_closed,
                left_target=left,
                right_target=right,
                sent=False,
            )

        self._closed[0] = _finger_closed_after_sample(
            self._closed[0],
            left_trigger,
            left_grip if self._latch_on_grip else 1.0,
            enabled=enabled,
            threshold=self._threshold,
        )
        self._closed[1] = _finger_closed_after_sample(
            self._closed[1],
            right_trigger,
            right_grip if self._latch_on_grip else 1.0,
            enabled=enabled,
            threshold=self._threshold,
        )
        left_closed, right_closed = self._closed
        left = DEX3_CLOSE_LEFT if left_closed else DEX3_OPEN
        right = DEX3_CLOSE_RIGHT if right_closed else DEX3_OPEN
        targets = (left, right)
        sent = False
        if targets != self._last_targets:
            self._controller.step_hands(list(left), list(right))
            self._last_targets = targets
            sent = True
        return Dex3HandStatus(
            connected=True,
            enabled=bool(enabled),
            left_trigger=left_trigger,
            right_trigger=right_trigger,
            left_grip=left_grip,
            right_grip=right_grip,
            left_closed=left_closed,
            right_closed=right_closed,
            left_target=left,
            right_target=right,
            sent=sent,
        )
