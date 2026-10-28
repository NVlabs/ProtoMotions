# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, transport-independent relative-goal navigation helpers.

The Xbox command is intentionally not a velocity latch.  A directional press
creates one world-frame goal from the robot pose at the instant of the press;
the policy is then asked to walk toward that goal until the odometer reports
that it is inside the arrival radius.
"""

from __future__ import annotations

import math

import numpy as np


class RelativeGoalNavigator:
    """Create fixed-distance goals in the robot's heading frame."""

    def __init__(self, distance_m: float = 2.0, arrival_radius_m: float = 0.2):
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            raise ValueError("goal distance must be finite and positive")
        if (
            not math.isfinite(arrival_radius_m)
            or not 0.0 < arrival_radius_m < distance_m
        ):
            raise ValueError(
                "arrival radius must be finite and smaller than the goal distance"
            )
        self.distance_m = float(distance_m)
        self.arrival_radius_m = float(arrival_radius_m)
        self.goal_position = np.zeros(3, dtype=np.float64)
        self.goal_heading = 0.0
        self.local_direction = 0.0
        self.active = False
        self.sequence = 0

    @staticmethod
    def _pose(position: np.ndarray, heading: float) -> tuple[np.ndarray, float]:
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("navigation position must be a finite xyz vector")
        if not math.isfinite(heading):
            raise ValueError("navigation heading must be finite")
        return position.copy(), float(heading)

    def start(
        self, position: np.ndarray, heading: float, local_direction: float
    ) -> np.ndarray:
        """Start a goal and return its world-frame position."""
        position, heading = self._pose(position, heading)
        if not math.isfinite(local_direction):
            raise ValueError("local direction must be finite")
        self.local_direction = float(local_direction)
        self.goal_heading = heading + self.local_direction
        self.goal_position = position
        self.goal_position[:2] += self.distance_m * np.asarray(
            [math.cos(self.goal_heading), math.sin(self.goal_heading)],
            dtype=np.float64,
        )
        self.active = True
        self.sequence += 1
        return self.goal_position.copy()

    def update(self, position: np.ndarray) -> bool:
        """Return ``True`` exactly once when the active goal is reached."""
        if not self.active:
            return False
        position, _ = self._pose(position, 0.0)
        if (
            float(np.linalg.norm(self.goal_position[:2] - position[:2]))
            > self.arrival_radius_m
        ):
            return False
        self.active = False
        return True

    def distance_remaining(self, position: np.ndarray) -> float | None:
        if not self.active:
            return None
        position, _ = self._pose(position, 0.0)
        return float(np.linalg.norm(self.goal_position[:2] - position[:2]))

    def snapshot(self, position: np.ndarray | None = None) -> dict:
        remaining = None
        if position is not None:
            remaining = self.distance_remaining(position)
        return {
            "active": bool(self.active),
            "sequence": int(self.sequence),
            "goal_position": self.goal_position.copy(),
            "goal_heading": float(self.goal_heading),
            "local_direction": float(self.local_direction),
            "distance_m": float(self.distance_m),
            "arrival_radius_m": float(self.arrival_radius_m),
            "distance_remaining_m": remaining,
        }
